"""Speculative decoding: draft model proposes K tokens, target model verifies in parallel.

Uses the small (d=128, 1L) model as draft and the large (d=256, 4L) model as target.
Both share the same BPE vocabulary (vocab=4096, TinyStories).

Algorithm (greedy):
  1. Draft model generates K tokens autoregressively (fast, ~2500+ tok/s)
  2. Target model verifies all K tokens in ONE parallel kernel call
  3. Find first disagreement; accept all tokens before it + target's correction
  4. Repeat from step 1

The parallel verification kernel processes K tokens through the full target model
in a single launch, attending to the existing KV cache + causally to each other.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import prepare_data, load_bpe_vocab
from model import count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches)
from kernels.verify_decode import verify_decode


def load_model(path):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def generate_standard(params, config, prompt, n_tokens, vocab_size):
    """Standard autoregressive decode with fused N-layer kernel."""
    x = jnp.pad(prompt, (0, config["context_len"] - len(prompt))).astype(jnp.int32)
    logits, k_caches, v_caches = block_prefill(params, config, x, vocab_size)
    kv_packed = pack_kv_caches(k_caches, v_caches)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)

    tokens = []
    tok = jnp.argmax(logits[len(prompt) - 1])
    tokens.append(int(tok))

    for i in range(n_tokens - 1):
        logits, kv_packed = fused_decode_nlayer(
            w, config, tok, len(prompt) + i, kv_packed, vocab_size)
        tok = jnp.argmax(logits)
        tokens.append(int(tok))
    return tokens


def generate_speculative_sequential(draft_params, draft_config, target_params, target_config,
                                    prompt, n_tokens, vocab_size, K=4):
    """Speculative decoding with SEQUENTIAL target verification (baseline)."""
    ctx_t = target_config["context_len"]
    ctx_d = draft_config["context_len"]

    x_t = jnp.pad(prompt, (0, ctx_t - len(prompt))).astype(jnp.int32)
    x_d = jnp.pad(prompt, (0, ctx_d - len(prompt))).astype(jnp.int32)

    t_logits, t_kc, t_vc = block_prefill(target_params, target_config, x_t, vocab_size)
    d_logits, d_kc, d_vc = block_prefill(draft_params, draft_config, x_d, vocab_size)

    t_kv = pack_kv_caches(t_kc, t_vc)
    d_kv = pack_kv_caches(d_kc, d_vc)
    t_w = prepare_decode_weights_nlayer(target_params, target_config, vocab_size)
    d_w = prepare_decode_weights_nlayer(draft_params, draft_config, vocab_size)

    tok = int(jnp.argmax(t_logits[len(prompt) - 1]))
    tokens = [tok]
    pos = len(prompt)
    total_draft = 0
    total_accepted = 0

    while len(tokens) < n_tokens:
        k = min(K, n_tokens - len(tokens))

        # Draft K tokens
        draft_tokens = []
        draft_tok = tok
        draft_kv_tmp = d_kv
        for j in range(k):
            dl, draft_kv_tmp = fused_decode_nlayer(d_w, draft_config, draft_tok, pos + j, draft_kv_tmp, vocab_size)
            draft_tok = int(jnp.argmax(dl))
            draft_tokens.append(draft_tok)
        total_draft += k

        # Verify K tokens SEQUENTIALLY through target
        target_toks = []
        verify_kv = t_kv
        verify_tok = tok
        for j in range(k):
            tl_j, verify_kv = fused_decode_nlayer(t_w, target_config, verify_tok, pos + j, verify_kv, vocab_size)
            target_toks.append(int(jnp.argmax(tl_j)))
            verify_tok = draft_tokens[j]

        # Find first disagreement
        n_acc = 0
        for j in range(k):
            if target_toks[j] == draft_tokens[j]:
                n_acc += 1
            else:
                break

        total_accepted += n_acc

        if n_acc == k:
            # All accepted — add draft tokens + bonus from target
            for dt in draft_tokens:
                tokens.append(dt)
            pos += k
            tl_bonus, verify_kv = fused_decode_nlayer(t_w, target_config, draft_tokens[-1], pos, verify_kv, vocab_size)
            bonus = int(jnp.argmax(tl_bonus))
            tokens.append(bonus)
            pos += 1
            _, draft_kv_tmp = fused_decode_nlayer(d_w, draft_config, draft_tokens[-1], pos - 1, draft_kv_tmp, vocab_size)
            t_kv = verify_kv
            d_kv = draft_kv_tmp
            tok = bonus
        else:
            # Partial accept: take accepted drafts + target's correction
            accepted = draft_tokens[:n_acc]
            correction = target_toks[n_acc]
            for dt in accepted:
                tokens.append(dt)
            tokens.append(correction)
            pos += n_acc + 1

            # Re-advance draft through accepted + correction
            d_kv_rb = d_kv
            rb_tok = tok
            for jj in range(n_acc + 1):
                actual = draft_tokens[jj] if jj < n_acc else correction
                _, d_kv_rb = fused_decode_nlayer(d_w, draft_config, rb_tok, pos - n_acc - 1 + jj, d_kv_rb, vocab_size)
                rb_tok = actual

            t_kv = verify_kv
            d_kv = d_kv_rb
            tok = correction

    return tokens[:n_tokens], total_accepted / max(total_draft, 1), total_accepted, total_draft


def generate_speculative_parallel(draft_params, draft_config, target_params, target_config,
                                  prompt, n_tokens, vocab_size, K=4):
    """Speculative decoding with PARALLEL target verification kernel."""
    ctx_t = target_config["context_len"]
    ctx_d = draft_config["context_len"]

    x_t = jnp.pad(prompt, (0, ctx_t - len(prompt))).astype(jnp.int32)
    x_d = jnp.pad(prompt, (0, ctx_d - len(prompt))).astype(jnp.int32)

    t_logits, t_kc, t_vc = block_prefill(target_params, target_config, x_t, vocab_size)
    d_logits, d_kc, d_vc = block_prefill(draft_params, draft_config, x_d, vocab_size)

    t_kv = pack_kv_caches(t_kc, t_vc)
    d_kv = pack_kv_caches(d_kc, d_vc)
    t_w = prepare_decode_weights_nlayer(target_params, target_config, vocab_size)
    d_w = prepare_decode_weights_nlayer(draft_params, draft_config, vocab_size)

    tok = int(jnp.argmax(t_logits[len(prompt) - 1]))
    tokens = [tok]
    pos = len(prompt)
    total_draft = 0
    total_accepted = 0

    while len(tokens) < n_tokens:
        k = min(K, n_tokens - len(tokens))

        # Draft K tokens
        draft_tokens = []
        draft_tok = tok
        draft_kv_tmp = d_kv
        for j in range(k):
            dl, draft_kv_tmp = fused_decode_nlayer(d_w, draft_config, draft_tok, pos + j, draft_kv_tmp, vocab_size)
            draft_tok = int(jnp.argmax(dl))
            draft_tokens.append(draft_tok)
        total_draft += k

        # Verify K tokens in ONE parallel kernel call
        # Input to verify: the tokens that were fed as input to each position
        # Position 0 gets `tok`, position 1 gets draft_tokens[0], etc.
        verify_input = jnp.array([tok] + draft_tokens[:k-1], dtype=jnp.int32)
        ver_logits, verify_kv = verify_decode(t_w, target_config, verify_input, pos, t_kv, vocab_size, k)

        # Find first disagreement
        n_acc = 0
        target_toks = [int(jnp.argmax(ver_logits[j])) for j in range(k)]
        for j in range(k):
            if target_toks[j] == draft_tokens[j]:
                n_acc += 1
            else:
                break

        total_accepted += n_acc

        if n_acc == k:
            # All accepted — add draft tokens + bonus from target
            for dt in draft_tokens:
                tokens.append(dt)
            pos += k
            # Get bonus token: target processes the last draft token
            tl_bonus, verify_kv = fused_decode_nlayer(t_w, target_config, draft_tokens[-1], pos, verify_kv, vocab_size)
            bonus = int(jnp.argmax(tl_bonus))
            tokens.append(bonus)
            pos += 1
            _, draft_kv_tmp = fused_decode_nlayer(d_w, draft_config, draft_tokens[-1], pos - 1, draft_kv_tmp, vocab_size)
            t_kv = verify_kv
            d_kv = draft_kv_tmp
            tok = bonus
        else:
            # Partial accept
            accepted = draft_tokens[:n_acc]
            correction = target_toks[n_acc]
            for dt in accepted:
                tokens.append(dt)
            tokens.append(correction)
            pos += n_acc + 1

            # The verify kernel wrote KV for all K positions, but we only accepted n_acc+1.
            # We need to truncate the KV cache. Since fused_decode_nlayer reads kv_in and
            # writes kv_out, the extra entries at positions > pos-1 are just stale data
            # that will be overwritten in future steps. So verify_kv is usable as-is.
            t_kv = verify_kv

            # Re-advance draft
            d_kv_rb = d_kv
            rb_tok = tok
            for jj in range(n_acc + 1):
                actual = draft_tokens[jj] if jj < n_acc else correction
                _, d_kv_rb = fused_decode_nlayer(d_w, draft_config, rb_tok, pos - n_acc - 1 + jj, d_kv_rb, vocab_size)
                rb_tok = actual
            d_kv = d_kv_rb
            tok = correction

    return tokens[:n_tokens], total_accepted / max(total_draft, 1), total_accepted, total_draft


def bench(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, tuple):
            _ = int(result[0][-1])
        else:
            _ = int(result[-1])
        times.append(time.perf_counter() - t0)
    return times, result


def main():
    draft_params, draft_config = load_model("draft_weights.pkl")
    target_params, target_config = load_model("target_weights.pkl")
    vocab_size = target_config["vocab_size"]

    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]
    data = prepare_data(tokenizer="trained_bpe", bpe_vocab_size=vocab_size, dataset="tinystories")

    PROMPT_LEN = 128
    GEN_LEN = 128
    WARMUP = 3
    BENCH_ITERS = 10

    prompt = jnp.array(data["train_x"][0][:PROMPT_LEN], dtype=jnp.int32)
    print(f"Draft:  d={draft_config['d_model']} h={draft_config['n_heads']} l={draft_config['n_layers']} params={count_params(draft_params):,}")
    print(f"Target: d={target_config['d_model']} h={target_config['n_heads']} l={target_config['n_layers']} params={count_params(target_params):,}")
    print(f"Prompt: {decode_fn(prompt)[:200]}...\n")

    # Standard decode
    print("=== Standard Target Decode ===")
    std_times, std_tokens = bench(
        lambda: generate_standard(target_params, target_config, prompt, GEN_LEN, vocab_size),
        WARMUP, BENCH_ITERS)
    std_ms = np.mean(std_times) * 1000
    std_tps = GEN_LEN / np.mean(std_times)
    print(f"  {std_tps:.0f} tok/s  ({std_ms:.1f}ms)")
    print(f"  text: {decode_fn(std_tokens)[:200]}...\n")

    results = [f"Standard: {std_tps:.0f} tok/s ({std_ms:.1f}ms)"]

    for K in [2, 4]:
        # Sequential verification
        print(f"=== Speculative K={K} (sequential verify) ===")
        seq_times, seq_result = bench(
            lambda K=K: generate_speculative_sequential(
                draft_params, draft_config, target_params, target_config,
                prompt, GEN_LEN, vocab_size, K=K),
            WARMUP, BENCH_ITERS)
        seq_tokens, seq_acc, seq_nacc, seq_ndraft = seq_result
        seq_ms = np.mean(seq_times) * 1000
        seq_tps = GEN_LEN / np.mean(seq_times)
        print(f"  {seq_tps:.0f} tok/s  ({seq_ms:.1f}ms)  speedup={seq_tps/std_tps:.2f}x")
        print(f"  acceptance={seq_acc:.1%}  accepted={seq_nacc}/{seq_ndraft}")

        # Parallel verification
        print(f"=== Speculative K={K} (parallel verify) ===")
        par_times, par_result = bench(
            lambda K=K: generate_speculative_parallel(
                draft_params, draft_config, target_params, target_config,
                prompt, GEN_LEN, vocab_size, K=K),
            WARMUP, BENCH_ITERS)
        par_tokens, par_acc, par_nacc, par_ndraft = par_result
        par_ms = np.mean(par_times) * 1000
        par_tps = GEN_LEN / np.mean(par_times)
        print(f"  {par_tps:.0f} tok/s  ({par_ms:.1f}ms)  speedup={par_tps/std_tps:.2f}x")
        print(f"  acceptance={par_acc:.1%}  accepted={par_nacc}/{par_ndraft}")

        # Verify text matches
        text_match = "MATCH" if par_tokens == seq_tokens else "DIFFER"
        print(f"  text {text_match}: {decode_fn(par_tokens)[:150]}...")

        kernel_speedup = seq_ms / par_ms if par_ms > 0 else 0
        results.append(f"K={K} seq: {seq_tps:.0f} tok/s, par: {par_tps:.0f} tok/s, kernel_speedup={kernel_speedup:.2f}x, acc={par_acc:.1%}")
        print()

    # Save
    with open("speculative_results.txt", "w") as f:
        f.write(f"Draft:  d={draft_config['d_model']} l={draft_config['n_layers']} params={count_params(draft_params):,}\n")
        f.write(f"Target: d={target_config['d_model']} l={target_config['n_layers']} params={count_params(target_params):,}\n")
        for r in results:
            f.write(r + "\n")
    print("Results saved to speculative_results.txt")


if __name__ == "__main__":
    main()
