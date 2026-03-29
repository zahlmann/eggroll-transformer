"""Inference benchmark: fused Triton kernels vs JAX/XLA baseline."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import prepare_data
from model import transformer_forward
from kernels.fused_prefill import fused_prefill
from kernels.fused_decode import fused_decode

PROMPT_LEN = 64
GEN_LEN = 64
WARMUP = 5
BENCH_ITERS = 20


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    return {k: jnp.array(v) for k, v in saved["params"].items()}, saved["config"]


def generate_triton(params, prompt, n_tokens):
    x = jnp.pad(prompt, (0, 128 - len(prompt))).astype(jnp.int32)
    logits, kc, vc = fused_prefill(params, x)

    tokens = []
    tok = jnp.argmax(logits[len(prompt) - 1])
    tokens.append(int(tok))

    for i in range(n_tokens - 1):
        logits, kc, vc = fused_decode(params, tok, len(prompt) + i, kc, vc)
        tok = jnp.argmax(logits)
        tokens.append(int(tok))

    return tokens


def generate_jax(fwd, prompt, n_tokens):
    seq = list(prompt)
    for _ in range(n_tokens):
        logits = fwd(jnp.array(seq, dtype=jnp.int32))
        seq.append(int(jnp.argmax(logits[-1])))
    return seq[len(prompt):]


def bench(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        _ = int(result[-1])  # force sync
        times.append(time.perf_counter() - t0)
    return times, result


def main():
    params, config = load_params()
    data = prepare_data()
    prompt = jnp.array(data["train_x"][0][:PROMPT_LEN], dtype=jnp.int32)
    chars = data["chars"]
    decode = lambda ids: ''.join(chars[i] for i in ids)

    print(f"Prompt: {decode(prompt)}\n")

    # Triton
    tri_times, tri_tokens = bench(
        lambda: generate_triton(params, prompt, GEN_LEN), WARMUP, BENCH_ITERS)
    tri_ms = np.mean(tri_times) * 1000
    tri_tps = GEN_LEN / np.mean(tri_times)
    print(f"Triton:  {tri_tps:>6.0f} tok/s  ({tri_ms:.1f}ms)  text: {decode(tri_tokens)}")

    # JAX baseline
    fwd = jax.jit(lambda x: transformer_forward(params, config, x))
    jax_times, jax_tokens = bench(
        lambda: generate_jax(fwd, prompt, GEN_LEN), WARMUP, BENCH_ITERS)
    jax_ms = np.mean(jax_times) * 1000
    jax_tps = GEN_LEN / np.mean(jax_times)
    print(f"JAX:     {jax_tps:>6.0f} tok/s  ({jax_ms:.1f}ms)  text: {decode(jax_tokens)}")

    speedup = jax_ms / tri_ms
    print(f"\nSpeedup: {speedup:.2f}x")

    # Save
    with open(os.path.join(os.path.dirname(__file__), "inference_results.txt"), "w") as f:
        f.write(f"Triton: {tri_tps:.0f} tok/s ({tri_ms:.1f}ms)\n")
        f.write(f"JAX:    {jax_tps:.0f} tok/s ({jax_ms:.1f}ms)\n")
        f.write(f"Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
