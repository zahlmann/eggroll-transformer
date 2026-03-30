"""
Batch verification kernel for speculative decoding.

Processes K draft tokens through the target model in ONE parallel forward pass.
Each token attends to the full KV cache prefix (P previous tokens) plus
causally to the other draft tokens. This replaces K sequential decode calls.

Architecture:
- Internally pads to PAD_K=16 tokens for valid tl.dot shapes (inner dim >= 16)
- For each layer: LN -> Q/K/V projection -> attention -> O -> FFN
- Attention: each draft token i attends to all P cached tokens + draft tokens 0..i
- KV cache is updated with K new entries (masked writes for padding)

This is a "mini-prefill" starting from position P with existing KV cache.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)
KV_TILE    = tl.constexpr(64)
PAD_K      = tl.constexpr(16)   # internal padding for tl.dot compatibility


@triton.jit
def _verify_decode_kernel(
    # Embedding weights
    token_emb_ptr, pos_emb_ptr,
    # Packed per-layer weights (all layers concatenated, bf16)
    packed_w_ptr,
    # Final LN + output
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    # Verification inputs
    token_ids_ptr,   # (PAD_K,) int32 — draft token IDs, padded with 0
    start_pos_ptr,   # scalar int32 — position of first draft token
    real_k_ptr,      # scalar int32 — actual number of draft tokens (2-8)
    # Packed KV caches input (bf16)
    kv_in_ptr,
    # Outputs
    logits_ptr,      # (PAD_K, VOCAB_PAD) float32
    kv_out_ptr,      # updated packed KV caches (bf16)
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    d = tl.arange(0, D_MODEL)
    ki = tl.arange(0, PAD_K)
    start_pos = tl.load(start_pos_ptr)
    real_k = tl.load(real_k_ptr)
    real_mask = ki < real_k  # (PAD_K,) — True for real tokens

    # ── Embedding: (PAD_K, D_MODEL) ──
    # Padded tokens (ki >= real_k) get token_id=0 and arbitrary position — doesn't matter
    token_ids = tl.load(token_ids_ptr + ki)
    h = (tl.load(token_emb_ptr + token_ids[:, None] * D_MODEL + d[None, :]).to(tl.float32)
       + tl.load(pos_emb_ptr + (start_pos + ki[:, None]) * D_MODEL + d[None, :]).to(tl.float32))
    # Zero out padded token activations to prevent them from affecting real tokens
    h = tl.where(real_mask[:, None], h, 0.0)

    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +
        4 * D_MODEL * D_MODEL +
        D_MODEL + D_MODEL +
        D_MODEL * D_FF + D_FF +
        D_FF * D_MODEL + D_MODEL
    )
    LAYER_KV_SIZE: tl.constexpr = 2 * N_HEADS * MAX_SEQ * D_HEAD

    scale = 0.17677669529663689  # 1/sqrt(32)
    dh = tl.arange(0, D_HEAD)

    for layer in tl.static_range(N_LAYERS):
        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_HEADS * MAX_SEQ * D_HEAD

        off = w_base
        ln1_s_off = off;    off += D_MODEL
        ln1_b_off = off;    off += D_MODEL
        wq_off = off;       off += D_MODEL * D_MODEL
        wk_off = off;       off += D_MODEL * D_MODEL
        wv_off = off;       off += D_MODEL * D_MODEL
        wo_off = off;       off += D_MODEL * D_MODEL
        ln2_s_off = off;    off += D_MODEL
        ln2_b_off = off;    off += D_MODEL
        up_off = off;       off += D_MODEL * D_FF
        up_b_off = off;     off += D_FF
        down_off = off;     off += D_FF * D_MODEL
        down_b_off = off

        # ── LN1 ──
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln1_b_off + d).to(tl.float32)
        mean = tl.sum(h, axis=1)[:, None] / D_MODEL
        hc = h - mean
        var = tl.sum(hc * hc, axis=1)[:, None] / D_MODEL
        h_norm = ln_s[None, :] * hc * tl.math.rsqrt(var + 1e-5) + ln_b[None, :]
        h_norm = tl.where(real_mask[:, None], h_norm, 0.0)

        # ── Attention ──
        attn_accum = tl.zeros((PAD_K, D_MODEL), dtype=tl.float32)

        for head in tl.range(N_HEADS):
            hd = head * D_HEAD + dh
            cache_off = head * MAX_SEQ * D_HEAD

            # Q/K/V projections: (PAD_K, D_MODEL) @ (D_MODEL, D_HEAD) → (PAD_K, D_HEAD)
            wq_slice = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
            Q = tl.dot(h_norm.to(tl.bfloat16), wq_slice).to(tl.float32)

            wk_slice = tl.load(packed_w_ptr + wk_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
            K_new = tl.dot(h_norm.to(tl.bfloat16), wk_slice).to(tl.float32)

            wv_slice = tl.load(packed_w_ptr + wv_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
            V_new = tl.dot(h_norm.to(tl.bfloat16), wv_slice).to(tl.float32)

            # Write new K/V to output cache (only real tokens)
            new_pos = start_pos + ki
            tl.store(kv_out_ptr + kc_base + cache_off + new_pos[:, None] * D_HEAD + dh[None, :],
                     K_new.to(tl.bfloat16), mask=real_mask[:, None])
            tl.store(kv_out_ptr + vc_base + cache_off + new_pos[:, None] * D_HEAD + dh[None, :],
                     V_new.to(tl.bfloat16), mask=real_mask[:, None])

            # Copy existing cache (positions 0..start_pos-1)
            for t in tl.range(0, MAX_SEQ, KV_TILE):
                tile_pos = t + tl.arange(0, KV_TILE)
                tile_mask = tile_pos < start_pos
                k_tile = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                                mask=tile_mask[:, None], other=0.0)
                tl.store(kv_out_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                        k_tile, mask=tile_mask[:, None])
                v_tile = tl.load(kv_in_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                                mask=tile_mask[:, None], other=0.0)
                tl.store(kv_out_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                        v_tile, mask=tile_mask[:, None])

            # ── Attention with online softmax ──
            m_i = tl.full((PAD_K,), value=-1e9, dtype=tl.float32)
            l_i = tl.zeros((PAD_K,), dtype=tl.float32)
            o_i = tl.zeros((PAD_K, D_HEAD), dtype=tl.float32)

            # Phase 1: attend to cached tokens [0, start_pos) in tiles
            for t in tl.range(0, MAX_SEQ, KV_TILE):
                tile_pos = t + tl.arange(0, KV_TILE)
                K_tile = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                                mask=(tile_pos < start_pos)[:, None], other=0.0).to(tl.float32)
                V_tile = tl.load(kv_in_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                                mask=(tile_pos < start_pos)[:, None], other=0.0).to(tl.float32)

                s = tl.dot(Q.to(tl.bfloat16), tl.trans(K_tile.to(tl.bfloat16))).to(tl.float32) * scale
                s = tl.where(tile_pos[None, :] < start_pos, s, -1e9)

                m_ij = tl.max(s, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, axis=1)
                o_i = o_i * alpha[:, None] + tl.dot(p.to(tl.bfloat16), V_tile.to(tl.bfloat16)).to(tl.float32)
                m_i = m_new

            # Phase 2: attend to draft tokens with causal mask
            # (PAD_K, D_HEAD) @ (D_HEAD, PAD_K) → (PAD_K, PAD_K) — inner=D_HEAD=32 >= 16 ✓
            s_draft = tl.dot(Q.to(tl.bfloat16), tl.trans(K_new.to(tl.bfloat16))).to(tl.float32) * scale
            # Causal among real drafts only: token i sees drafts 0..i, not padding
            draft_mask = (ki[:, None] >= ki[None, :]) & (ki[None, :] < real_k)
            s_draft = tl.where(draft_mask, s_draft, -1e9)

            m_ij = tl.max(s_draft, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p_draft = tl.exp(s_draft - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p_draft, axis=1)
            # (PAD_K, PAD_K) @ (PAD_K, D_HEAD) → inner=PAD_K=16 >= 16 ✓
            o_i = o_i * alpha[:, None] + tl.dot(p_draft.to(tl.bfloat16), V_new.to(tl.bfloat16)).to(tl.float32)
            m_i = m_new

            # Normalize
            attn_out = o_i / tl.maximum(l_i[:, None], 1e-9)

            # O projection: (PAD_K, D_HEAD) @ (D_HEAD, D_MODEL)
            wo_slice = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            attn_accum += tl.dot(attn_out.to(tl.bfloat16), wo_slice).to(tl.float32)

        h = h + attn_accum

        # ── LN2 + FFN ──
        ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln2_b_off + d).to(tl.float32)
        mean = tl.sum(h, axis=1)[:, None] / D_MODEL
        hc = h - mean
        var = tl.sum(hc * hc, axis=1)[:, None] / D_MODEL
        h_norm = ln_s[None, :] * hc * tl.math.rsqrt(var + 1e-5) + ln_b[None, :]
        h_norm = tl.where(real_mask[:, None], h_norm, 0.0)

        ffn_accum = tl.zeros((PAD_K, D_MODEL), dtype=tl.float32)
        for k in tl.range(0, D_FF, BLOCK_K):
            kk = k + tl.arange(0, BLOCK_K)
            up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
            up = tl.dot(h_norm.to(tl.bfloat16), up_w).to(tl.float32)
            up += tl.load(packed_w_ptr + up_b_off + kk).to(tl.float32)[None, :]
            act = up * tl.sigmoid(1.702 * up)
            down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            ffn_accum += tl.dot(act.to(tl.bfloat16), down_w).to(tl.float32)

        h = h + ffn_accum + tl.load(packed_w_ptr + down_b_off + d).to(tl.float32)[None, :]
        h = tl.where(real_mask[:, None], h, 0.0)

    # ── Output ──
    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = h - mean
    var = tl.sum(hc * hc, axis=1)[:, None] / D_MODEL
    h_final = ln_s[None, :] * hc * tl.math.rsqrt(var + 1e-5) + ln_b[None, :]

    for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
        vv = v_start + tl.arange(0, VOCAB_TILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final.to(tl.bfloat16), out_w).to(tl.float32)
        tile_logits = tl.where(vv[None, :] < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + ki[:, None] * VOCAB_PAD + vv[None, :], tile_logits)


def verify_decode(w, config, token_ids, start_pos, kv_packed, vocab_size, real_k):
    """Verify K draft tokens through the target model in one parallel kernel call.

    Internally pads to 16 tokens for Triton tl.dot compatibility.
    Only the first real_k output logits are meaningful.

    Args:
        w: precomputed weights from prepare_decode_weights_nlayer()
        config: model config
        token_ids: (real_k,) int32 — draft token IDs
        start_pos: scalar int32 — position of first draft token
        kv_packed: flat bf16 buffer (existing KV cache)
        vocab_size: actual vocabulary size
        real_k: actual number of draft tokens (2-8)

    Returns:
        logits: (real_k, vocab_size) float32
        kv_out: flat bf16 buffer — updated KV cache
    """
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_heads * max_seq * d_head

    # Pad token_ids to 16
    padded_ids = jnp.zeros(16, dtype=jnp.int32)
    padded_ids = padded_ids.at[:real_k].set(token_ids[:real_k])

    logits_pad, kv_out = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
        padded_ids,
        jnp.int32(start_pos),
        jnp.int32(real_k),
        kv_packed,
        kernel=_verify_decode_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((16, vocab_pad), jnp.float32),
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
        ],
        grid=(1,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_LAYERS=n_layers, MAX_SEQ=max_seq,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
    )

    return logits_pad[:real_k, :vocab_size], kv_out
