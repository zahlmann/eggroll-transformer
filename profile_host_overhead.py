"""Profile host overhead vs GPU kernel time.

Measures:
  - Total per-step time (wall clock)
  - GPU kernel time (CUDA events via JAX)
  - Host overhead = total - GPU time
  - argmax sync cost (GPU→CPU transfer)
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import load_bpe_vocab
from model import count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches)


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    d = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    ctx = config["context_len"]
    PROMPT_LEN = 128
    GEN_LEN = 64

    print(f"Model: d={d} h={n_heads} l={n_layers} ctx={ctx}")
    print(f"Params: {count_params(params):,}")
    print()

    # Prefill
    prompt = jnp.arange(PROMPT_LEN, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, ctx - PROMPT_LEN)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    # Warmup decode
    kv_tmp = kv_packed
    t = tok
    for i in range(10):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)

    # ── Measurement 1: Total wall-clock per step ──
    print("=" * 60)
    print("Measurement 1: Total wall-clock per decode step")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    step_times = []
    for i in range(GEN_LEN):
        t0 = time.perf_counter()
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)  # force GPU→CPU sync
        step_times.append(time.perf_counter() - t0)

    step_times = np.array(step_times)
    print(f"  Per step (median):  {np.median(step_times)*1000:.3f} ms")
    print(f"  Per step (mean):    {np.mean(step_times)*1000:.3f} ms")
    print(f"  Per step (min):     {np.min(step_times)*1000:.3f} ms")
    print(f"  Per step (max):     {np.max(step_times)*1000:.3f} ms")
    print(f"  Per step (p5):      {np.percentile(step_times, 5)*1000:.3f} ms")
    print(f"  Per step (p95):     {np.percentile(step_times, 95)*1000:.3f} ms")
    print()

    # ── Measurement 2: Kernel launch + compute (no argmax sync) ──
    print("=" * 60)
    print("Measurement 2: Kernel call only (no argmax, no sync)")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    kernel_times = []
    for i in range(GEN_LEN):
        t0 = time.perf_counter()
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t1 = time.perf_counter()
        # Now do sync separately
        t = jnp.argmax(logits_d)
        _ = int(t)
        kernel_times.append(t1 - t0)

    kernel_times = np.array(kernel_times)
    print(f"  Kernel call (median): {np.median(kernel_times)*1000:.3f} ms")
    print(f"  Kernel call (mean):   {np.mean(kernel_times)*1000:.3f} ms")
    print(f"  Kernel call (min):    {np.min(kernel_times)*1000:.3f} ms")
    print()

    # ── Measurement 3: argmax + sync cost ──
    print("=" * 60)
    print("Measurement 3: argmax + GPU→CPU sync cost")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    sync_times = []
    for i in range(GEN_LEN):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = logits_d.block_until_ready()  # ensure kernel done
        t0 = time.perf_counter()
        t = jnp.argmax(logits_d)
        _ = int(t)
        sync_times.append(time.perf_counter() - t0)

    sync_times = np.array(sync_times)
    print(f"  argmax+sync (median): {np.median(sync_times)*1000:.3f} ms")
    print(f"  argmax+sync (mean):   {np.mean(sync_times)*1000:.3f} ms")
    print()

    # ── Measurement 4: Pure GPU time via block_until_ready ──
    print("=" * 60)
    print("Measurement 4: GPU kernel time (block_until_ready)")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    gpu_times = []
    for i in range(GEN_LEN):
        # Make sure previous work is done
        _ = kv_tmp.block_until_ready()
        t0 = time.perf_counter()
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = logits_d.block_until_ready()
        t1 = time.perf_counter()
        # Now sync the token
        t = jnp.argmax(logits_d)
        _ = int(t)
        gpu_times.append(t1 - t0)

    gpu_times = np.array(gpu_times)
    print(f"  GPU time (median):  {np.median(gpu_times)*1000:.3f} ms")
    print(f"  GPU time (mean):    {np.mean(gpu_times)*1000:.3f} ms")
    print(f"  GPU time (min):     {np.min(gpu_times)*1000:.3f} ms")
    print()

    # ── Measurement 5: Batch of N steps with single sync at end ──
    print("=" * 60)
    print("Measurement 5: Amortized (batch N steps, sync once)")
    print("=" * 60)
    # This tells us how much time is pure GPU work (if we could pipeline)
    kv_tmp = kv_packed
    t = tok
    # Pre-sync
    _ = kv_tmp.block_until_ready()
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        # Don't sync to CPU — let JAX keep it on device
    _ = int(t)  # single sync at end
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    print(f"  Total ({GEN_LEN} steps): {total_ms:.1f} ms")
    print(f"  Per step:           {total_ms/GEN_LEN:.3f} ms")
    print(f"  Throughput:         {GEN_LEN/total_ms*1000:.0f} tok/s")
    print()

    # ── Measurement 6: Cost of int() sync vs staying on device ──
    print("=" * 60)
    print("Measurement 6: int() sync cost per step")
    print("=" * 60)
    # With int() sync every step
    kv_tmp = kv_packed
    t = tok
    _ = kv_tmp.block_until_ready()
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)
    t1 = time.perf_counter()
    with_sync_ms = (t1 - t0) * 1000

    # Without int() sync (keep on device)
    kv_tmp = kv_packed
    t = tok
    _ = kv_tmp.block_until_ready()
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        # No int() — keep as JAX scalar
    _ = int(t)
    t1 = time.perf_counter()
    no_sync_ms = (t1 - t0) * 1000

    print(f"  With int() sync:    {with_sync_ms:.1f} ms  ({with_sync_ms/GEN_LEN:.3f} ms/step)")
    print(f"  Without int() sync: {no_sync_ms:.1f} ms  ({no_sync_ms/GEN_LEN:.3f} ms/step)")
    print(f"  Overhead per step:  {(with_sync_ms - no_sync_ms)/GEN_LEN:.3f} ms/step")
    print(f"  Throughput w/sync:  {GEN_LEN/with_sync_ms*1000:.0f} tok/s")
    print(f"  Throughput no sync: {GEN_LEN/no_sync_ms*1000:.0f} tok/s")
    print()

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_per_step = np.median(step_times) * 1000
    gpu_per_step = np.median(gpu_times) * 1000
    argmax_per_step = np.median(sync_times) * 1000
    host_overhead = total_per_step - gpu_per_step - argmax_per_step
    print(f"  Total per step:     {total_per_step:.3f} ms")
    print(f"  GPU kernel time:    {gpu_per_step:.3f} ms")
    print(f"  argmax + sync:      {argmax_per_step:.3f} ms")
    print(f"  Host overhead:      {host_overhead:.3f} ms (dispatch, JAX, Python)")
    print(f"  Theoretical min:    0.081 ms (836 GB/s bandwidth)")
    print()
    print(f"  GPU kernel is {gpu_per_step/total_per_step*100:.0f}% of total step time")
    print(f"  argmax+sync is {argmax_per_step/total_per_step*100:.0f}% of total step time")
    print(f"  Host overhead is {host_overhead/total_per_step*100:.0f}% of total step time")

    # Save results
    with open(os.path.join(os.path.dirname(__file__), "profile_host_results.txt"), "w") as f:
        f.write(f"# Host overhead profiling — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"total_per_step_ms: {total_per_step:.3f}\n")
        f.write(f"gpu_kernel_ms: {gpu_per_step:.3f}\n")
        f.write(f"argmax_sync_ms: {argmax_per_step:.3f}\n")
        f.write(f"host_overhead_ms: {host_overhead:.3f}\n")
        f.write(f"no_sync_per_step_ms: {no_sync_ms/GEN_LEN:.3f}\n")
        f.write(f"with_sync_per_step_ms: {with_sync_ms/GEN_LEN:.3f}\n")


if __name__ == "__main__":
    main()
