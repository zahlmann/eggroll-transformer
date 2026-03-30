"""Quick test: compare kv_splits=1 vs kv_splits=2 for correctness and performance."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import load_bpe_vocab
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_params():
    with open("weights.pkl", "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def decode_n_tokens(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens, kv_splits):
    kv = kv_packed
    t = tok
    tokens = []
    for i in range(n_tokens):
        t, logits, kv = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv, vocab_size,
                                                kv_splits=kv_splits)
        tokens.append(int(t))
    return tokens, logits, kv


def benchmark(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens, kv_splits, n_runs=10):
    # Warmup
    kv = kv_packed
    t = tok
    for i in range(3):
        t, _, kv = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv, vocab_size,
                                           kv_splits=kv_splits)
        _ = int(t)

    times = []
    for _ in range(n_runs):
        kv = kv_packed
        t = tok
        t0 = time.perf_counter()
        for i in range(n_tokens):
            t, _, kv = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv, vocab_size,
                                               kv_splits=kv_splits)
            _ = int(t)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    n_heads = config["n_heads"]
    prompt_len = 128
    gen_len = 128

    prompt = jnp.arange(prompt_len, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - prompt_len)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[prompt_len - 1])

    # Correctness: compare tokens from kv_splits=1 vs kv_splits=2
    print("=== Correctness Test ===")
    tokens_1, logits_1, _ = decode_n_tokens(w, config, tok, prompt_len, kv_packed, vocab_size,
                                             n_tokens=32, kv_splits=1)
    tokens_2, logits_2, _ = decode_n_tokens(w, config, tok, prompt_len, kv_packed, vocab_size,
                                             n_tokens=32, kv_splits=2)
    match = tokens_1 == tokens_2
    logit_diff = float(jnp.max(jnp.abs(logits_1 - logits_2)))
    print(f"kv_splits=1 tokens: {tokens_1[:10]}...")
    print(f"kv_splits=2 tokens: {tokens_2[:10]}...")
    print(f"Token match: {match} ({sum(a==b for a,b in zip(tokens_1, tokens_2))}/32)")
    print(f"Max logit diff (last step): {logit_diff:.6f}")
    print()

    # Performance
    print("=== Performance Test ===")
    for kv_splits in [1, 2, 4]:
        total_blocks = n_heads * kv_splits
        try:
            ms = benchmark(w, config, tok, prompt_len, kv_packed, vocab_size,
                          n_tokens=gen_len, kv_splits=kv_splits, n_runs=10)
            tok_s = gen_len / ms * 1000
            print(f"kv_splits={kv_splits} (grid={total_blocks:2d}): {tok_s:.0f} tok/s  ({ms/gen_len:.3f} ms/tok)")
        except Exception as e:
            print(f"kv_splits={kv_splits} (grid={total_blocks:2d}): FAILED — {e}")


if __name__ == "__main__":
    main()
