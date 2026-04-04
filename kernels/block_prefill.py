"""
Multi-block prefill kernels for d_model >= 128.

Phase C architecture: RMSNorm, RoPE, SwiGLU, no biases, tied embeddings, GQA.

Three Triton kernels per layer (grid=num_blocks, each block handles BLOCK_SEQ rows):
  1. _proj_kernel:  RMSNorm1 + Q/K/V projections + RoPE on Q,K → writes Q,K,V to HBM
  2. _attn_kernel:  causal attention + O projection + residual → updates h
  3. _ffn_kernel:   RMSNorm2 + SwiGLU FFN + residual → updates h

Plus _output_kernel for final RMSNorm + tiled output projection (tied embeddings).
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

from model import precompute_rope_table

BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)
PROJ_TILE  = tl.constexpr(512)


@triton.jit
def _proj_kernel(
    h_ptr,
    ln_scale_ptr,
    wq_ptr, wk_ptr, wv_ptr,
    cos_ptr, sin_ptr,
    h_norm_buf_ptr, q_buf_ptr, k_cache_ptr, v_cache_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_HALF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_KV: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    TILE_PROJ: tl.constexpr,
):
    """RMSNorm1 + Q/K/V projections + RoPE. Supports GQA + non-power-of-2 D_MODEL."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL
    dh = tl.arange(0, D_HEAD)
    dh_lo = tl.arange(0, D_HALF)
    dh_hi = D_HALF + tl.arange(0, D_HALF)

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    # RMSNorm
    ln_s = tl.load(ln_scale_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    h_sq = tl.where(d_mask[None, :], h * h, 0.0)
    h_norm = tl.where(d_mask[None, :],
                      ln_s[None, :] * h * tl.math.rsqrt(tl.sum(h_sq, axis=1)[:, None] / D_MODEL + 1e-5),
                      0.0)

    # Store h_norm for tiled projection reload
    tl.store(h_norm_buf_ptr + rows[:, None] * D_MODEL + d[None, :],
             h_norm.to(tl.bfloat16), mask=d_mask[None, :])

    # Load RoPE cos/sin for these rows: (BLOCK_SEQ, D_HALF)
    cos = tl.load(cos_ptr + rows[:, None] * D_HALF + dh_lo[None, :]).to(tl.float32)
    sin = tl.load(sin_ptr + rows[:, None] * D_HALF + dh_lo[None, :]).to(tl.float32)

    # Q projections: N_HEADS heads, split into halves for RoPE
    for head in tl.range(N_HEADS):
        hd_lo = head * D_HEAD + dh_lo
        hd_hi = head * D_HEAD + dh_hi
        head_off = head * SEQ * D_HEAD
        Q_lo = tl.zeros((BLOCK_SEQ, D_HALF), dtype=tl.float32)
        Q_hi = tl.zeros((BLOCK_SEQ, D_HALF), dtype=tl.float32)
        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            h_tile = tl.load(h_norm_buf_ptr + rows[:, None] * D_MODEL + dt[None, :],
                             mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            wq_lo = tl.load(wq_ptr + dt[:, None] * D_MODEL + hd_lo[None, :],
                            mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            Q_lo += tl.dot(h_tile, wq_lo).to(tl.float32)
            wq_hi = tl.load(wq_ptr + dt[:, None] * D_MODEL + hd_hi[None, :],
                            mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            Q_hi += tl.dot(h_tile, wq_hi).to(tl.float32)
        # Apply RoPE to Q
        Q_rot_lo = Q_lo * cos - Q_hi * sin
        Q_rot_hi = Q_lo * sin + Q_hi * cos
        tl.store(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh_lo[None, :], Q_rot_lo)
        tl.store(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh_hi[None, :], Q_rot_hi)

    # K/V projections: N_KV_HEADS heads, RoPE on K only
    for kv_head in tl.range(N_KV_HEADS):
        kv_hd_lo = kv_head * D_HEAD + dh_lo
        kv_hd_hi = kv_head * D_HEAD + dh_hi
        kv_hd = kv_head * D_HEAD + dh
        kv_head_off = kv_head * SEQ * D_HEAD

        K_lo = tl.zeros((BLOCK_SEQ, D_HALF), dtype=tl.float32)
        K_hi = tl.zeros((BLOCK_SEQ, D_HALF), dtype=tl.float32)
        V = tl.zeros((BLOCK_SEQ, D_HEAD), dtype=tl.float32)
        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            h_tile = tl.load(h_norm_buf_ptr + rows[:, None] * D_MODEL + dt[None, :],
                             mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            wk_lo = tl.load(wk_ptr + dt[:, None] * D_KV + kv_hd_lo[None, :],
                            mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            K_lo += tl.dot(h_tile, wk_lo).to(tl.float32)
            wk_hi = tl.load(wk_ptr + dt[:, None] * D_KV + kv_hd_hi[None, :],
                            mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            K_hi += tl.dot(h_tile, wk_hi).to(tl.float32)
            wv_tile = tl.load(wv_ptr + dt[:, None] * D_KV + kv_hd[None, :],
                              mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            V += tl.dot(h_tile, wv_tile).to(tl.float32)
        # Apply RoPE to K
        K_rot_lo = K_lo * cos - K_hi * sin
        K_rot_hi = K_lo * sin + K_hi * cos
        tl.store(k_cache_ptr + kv_head_off + rows[:, None] * D_HEAD + dh_lo[None, :], K_rot_lo.to(tl.bfloat16))
        tl.store(k_cache_ptr + kv_head_off + rows[:, None] * D_HEAD + dh_hi[None, :], K_rot_hi.to(tl.bfloat16))
        tl.store(v_cache_ptr + kv_head_off + rows[:, None] * D_HEAD + dh[None, :], V.to(tl.bfloat16))


@triton.jit
def _attn_kernel(
    h_ptr,
    q_buf_ptr, k_cache_ptr, v_cache_ptr,
    wo_ptr,
    attn_scratch_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    TILE_PROJ: tl.constexpr,
):
    """Causal attention + O projection + residual. Q,K already have RoPE applied."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    all_pos = tl.arange(0, SEQ)
    dh = tl.arange(0, D_HEAD)

    scale = 1.0 / (D_HEAD ** 0.5)

    # Initialize h_out with residual h (in tiles)
    for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
        dt = dd + tl.arange(0, TILE_PROJ)
        dt_mask = dt < D_MODEL
        h_tile = tl.load(h_ptr + rows[:, None] * D_MODEL + dt[None, :],
                         mask=dt_mask[None, :], other=0.0).to(tl.float32)
        tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :], h_tile, mask=dt_mask[None, :])

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD
        kv_head = head // GQA_GROUP
        kv_head_off = kv_head * SEQ * D_HEAD

        Q = tl.load(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
        K = tl.load(k_cache_ptr + kv_head_off + all_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
        V = tl.load(v_cache_ptr + kv_head_off + all_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        mask = rows[:, None] >= all_pos[None, :]
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        attn = exp_s / tl.sum(exp_s, axis=1)[:, None]

        attn_out = tl.dot(attn.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)

        tl.store(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :],
                 attn_out.to(tl.bfloat16))

        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            ao = tl.load(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :]).to(tl.bfloat16)
            wo_tile = tl.load(wo_ptr + hd[:, None] * D_MODEL + dt[None, :],
                              mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            o_tile = tl.dot(ao, wo_tile).to(tl.float32)
            prev = tl.load(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                           mask=dt_mask[None, :], other=0.0).to(tl.float32)
            tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                     prev + o_tile, mask=dt_mask[None, :])


@triton.jit
def _flash_attn_kernel(
    h_ptr,
    q_buf_ptr, k_cache_ptr, v_cache_ptr,
    wo_ptr,
    attn_scratch_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    KV_TILE: tl.constexpr,
    TILE_PROJ: tl.constexpr,
):
    """FlashAttention: tiled KV with online softmax. Q,K already have RoPE applied."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dh = tl.arange(0, D_HEAD)

    scale = 1.0 / (D_HEAD ** 0.5)

    # Initialize h_out with residual h (in tiles)
    for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
        dt = dd + tl.arange(0, TILE_PROJ)
        dt_mask = dt < D_MODEL
        h_tile = tl.load(h_ptr + rows[:, None] * D_MODEL + dt[None, :],
                         mask=dt_mask[None, :], other=0.0).to(tl.float32)
        tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :], h_tile, mask=dt_mask[None, :])

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD
        kv_head = head // GQA_GROUP
        kv_head_off = kv_head * SEQ * D_HEAD

        Q = tl.load(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

        m_i = tl.full((BLOCK_SEQ,), value=-1e9, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
        o_i = tl.zeros((BLOCK_SEQ, D_HEAD), dtype=tl.float32)

        for kv_start in tl.range(0, SEQ, KV_TILE):
            kv_pos = kv_start + tl.arange(0, KV_TILE)

            K_tile = tl.load(k_cache_ptr + kv_head_off + kv_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
            V_tile = tl.load(v_cache_ptr + kv_head_off + kv_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

            s = tl.dot(Q.to(tl.bfloat16), tl.trans(K_tile.to(tl.bfloat16))).to(tl.float32) * scale

            causal_mask = rows[:, None] >= kv_pos[None, :]
            s = tl.where(causal_mask, s, -1e9)

            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            o_i = o_i * alpha[:, None] + tl.dot(p.to(tl.bfloat16), V_tile.to(tl.bfloat16)).to(tl.float32)
            m_i = m_new

        attn_out = o_i / l_i[:, None]

        tl.store(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :],
                 attn_out.to(tl.bfloat16))

        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            ao = tl.load(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :]).to(tl.bfloat16)
            wo_tile = tl.load(wo_ptr + hd[:, None] * D_MODEL + dt[None, :],
                              mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            o_tile = tl.dot(ao, wo_tile).to(tl.float32)
            prev = tl.load(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                           mask=dt_mask[None, :], other=0.0).to(tl.float32)
            tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                     prev + o_tile, mask=dt_mask[None, :])


@triton.jit
def _ffn_kernel(
    h_ptr,
    ln_scale_ptr,
    ffn_gate_ptr, ffn_up_ptr, ffn_down_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_FF: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """RMSNorm2 + SwiGLU FFN + residual. Supports D_BLOCK padding."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    # RMSNorm
    ln_s = tl.load(ln_scale_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    h_sq = tl.where(d_mask[None, :], h * h, 0.0)
    h_norm = tl.where(d_mask[None, :],
                      ln_s[None, :] * h * tl.math.rsqrt(tl.sum(h_sq, axis=1)[:, None] / D_MODEL + 1e-5),
                      0.0)

    # SwiGLU FFN (tiled over D_FF)
    ffn_out = tl.zeros((BLOCK_SEQ, D_BLOCK), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        # Gate projection
        gate = tl.dot(h_norm.to(tl.bfloat16),
                      tl.load(ffn_gate_ptr + d[:, None] * D_FF + kk[None, :], mask=d_mask[:, None], other=0.0).to(tl.bfloat16)).to(tl.float32)
        # Up projection
        up = tl.dot(h_norm.to(tl.bfloat16),
                    tl.load(ffn_up_ptr + d[:, None] * D_FF + kk[None, :], mask=d_mask[:, None], other=0.0).to(tl.bfloat16)).to(tl.float32)
        # SwiGLU: SiLU(gate) * up
        act = (gate * tl.sigmoid(gate)) * up
        # Down projection
        ffn_out += tl.dot(act.to(tl.bfloat16),
                          tl.load(ffn_down_ptr + kk[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.bfloat16)).to(tl.float32)

    # Residual
    h = h + ffn_out
    tl.store(h_out_ptr + rows[:, None] * D_MODEL + d[None, :], h, mask=d_mask[None, :])


@triton.jit
def _output_kernel(
    h_ptr,
    ln_scale_ptr,
    output_proj_ptr,
    logits_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    VTILE: tl.constexpr,
):
    """Final RMSNorm + tiled output projection (tied embeddings)."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    # RMSNorm
    ln_s = tl.load(ln_scale_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    h_sq = tl.where(d_mask[None, :], h * h, 0.0)
    h_final = tl.where(d_mask[None, :],
                       ln_s[None, :] * h * tl.math.rsqrt(tl.sum(h_sq, axis=1)[:, None] / D_MODEL + 1e-5),
                       0.0)

    for v_start in tl.range(0, VOCAB_PAD, VTILE):
        vv = v_start + tl.arange(0, VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :], mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
        tile_logits = tl.dot(h_final.to(tl.bfloat16), out_w).to(tl.float32)
        tile_logits = tl.where(vv[None, :] < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + rows[:, None] * VOCAB_PAD + vv[None, :], tile_logits)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# ──────────────────────────────────────────────────────────────────────
# Python orchestrator
# ──────────────────────────────────────────────────────────────────────

def block_prefill(params, config, x, vocab_size):
    """Multi-block prefill for d_model >= 128.

    Args:
        params: weight dict from model.init_transformer
        config: model config dict
        x: (seq_len,) int32 token IDs
        vocab_size: actual vocabulary size

    Returns:
        logits: (seq_len, vocab_size) float32
        k_caches: list of (n_kv_heads, seq_len, d_head) bf16 per layer
        v_caches: list of (n_kv_heads, seq_len, d_head) bf16 per layer
    """
    seq_len = x.shape[0]
    d_model = config["d_model"]
    d_head = config["d_head"]
    d_half = d_head // 2
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_kv = n_kv_heads * d_head
    gqa_group = n_heads // n_kv_heads
    d_block = _next_power_of_2(d_model)
    n_layers = config["n_layers"]
    d_ff = config["d_ff"]
    block_seq = 8 if d_model >= 512 else (16 if d_model >= 256 else 32)
    num_blocks = seq_len // block_seq
    vocab_pad = ((vocab_size + 127) // 128) * 128

    proj_tile = min(d_block, 512)

    # Precompute bf16 weights
    w = {k: v.astype(jnp.bfloat16) for k, v in params.items()}

    # Precompute RoPE tables
    cos, sin = precompute_rope_table(config["context_len"], d_head)
    cos_bf16 = cos[:seq_len].astype(jnp.bfloat16)
    sin_bf16 = sin[:seq_len].astype(jnp.bfloat16)

    # Embedding (no positional — RoPE applied in attention)
    h = params["token_emb"][x].astype(jnp.float32)

    all_k_caches = []
    all_v_caches = []

    for layer in range(n_layers):
        p = f"layer{layer}"

        # Kernel 1: RMSNorm + Q/K/V projections + RoPE
        _h_norm_buf, q_buf, k_cache, v_cache = jt.triton_call(
            h,
            w[f"{p}.ln1.scale"],
            w[f"{p}.attn.q"], w[f"{p}.attn.k"], w[f"{p}.attn.v"],
            cos_bf16, sin_bf16,
            kernel=_proj_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((seq_len, d_model), jnp.bfloat16),
                jax.ShapeDtypeStruct((n_heads, seq_len, d_head), jnp.float32),
                jax.ShapeDtypeStruct((n_kv_heads, seq_len, d_head), jnp.bfloat16),
                jax.ShapeDtypeStruct((n_kv_heads, seq_len, d_head), jnp.bfloat16),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, D_HALF=d_half,
            N_HEADS=n_heads, N_KV_HEADS=n_kv_heads, D_KV=d_kv,
            SEQ=seq_len, BLOCK_SEQ=block_seq, TILE_PROJ=proj_tile,
        )
        all_k_caches.append(k_cache)
        all_v_caches.append(v_cache)

        # Kernel 2: Attention + O projection + residual
        use_flash = seq_len > 256
        if use_flash:
            kv_tile = 64
            _attn_scratch, h = jt.triton_call(
                h, q_buf, k_cache, v_cache,
                w[f"{p}.attn.o"],
                kernel=_flash_attn_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((seq_len, d_head), jnp.bfloat16),
                    jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
                ],
                grid=(num_blocks,),
                num_warps=4, num_stages=1,
                D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, N_HEADS=n_heads,
                GQA_GROUP=gqa_group,
                SEQ=seq_len, BLOCK_SEQ=block_seq, KV_TILE=kv_tile,
                TILE_PROJ=proj_tile,
            )
        else:
            _attn_scratch, h = jt.triton_call(
                h, q_buf, k_cache, v_cache,
                w[f"{p}.attn.o"],
                kernel=_attn_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((seq_len, d_head), jnp.bfloat16),
                    jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
                ],
                grid=(num_blocks,),
                num_warps=4, num_stages=1,
                D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, N_HEADS=n_heads,
                GQA_GROUP=gqa_group,
                SEQ=seq_len, BLOCK_SEQ=block_seq,
                TILE_PROJ=proj_tile,
            )

        # Kernel 3: RMSNorm + SwiGLU FFN + residual
        (h,) = jt.triton_call(
            h,
            w[f"{p}.ln2.scale"],
            w[f"{p}.ffn.gate"], w[f"{p}.ffn.up"], w[f"{p}.ffn.down"],
            kernel=_ffn_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_BLOCK=d_block, D_FF=d_ff, SEQ=seq_len,
            BLOCK_SEQ=block_seq,
        )

    # Kernel 4: Final RMSNorm + output projection (tied embeddings)
    pad_v = vocab_pad - vocab_size
    output_proj_padded = jnp.pad(params["token_emb"].T, [(0, 0), (0, pad_v)]).astype(jnp.bfloat16)

    (logits_pad,) = jt.triton_call(
        h,
        w["ln_final.scale"],
        output_proj_padded,
        kernel=_output_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((seq_len, vocab_pad), jnp.float32),
        ],
        grid=(num_blocks,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_BLOCK=d_block, SEQ=seq_len,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        BLOCK_SEQ=block_seq,
        VTILE=32 if d_model >= 512 else 128,
    )

    return logits_pad[:, :vocab_size], all_k_caches, all_v_caches
