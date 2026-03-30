"""Profile multi-SM decode kernel: breakdown of where time is spent."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from model import count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches)
from kernels.multi_sm_decode import multi_sm_decode_nlayer


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
    print()

    # Setup
    prompt = jnp.arange(PROMPT_LEN, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, ctx - PROMPT_LEN)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()
    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    # Warmup
    kv_tmp = kv_packed
    t = tok
    for i in range(10):
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)

    # ── 1. Per-step with int() sync ──
    print("=" * 60)
    print("1. Per-step: int() sync every step")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    step_times = []
    for i in range(GEN_LEN):
        t0 = time.perf_counter()
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)
        step_times.append(time.perf_counter() - t0)
    step_times = np.array(step_times)
    print(f"  Median:  {np.median(step_times)*1000:.3f} ms  ({1000/np.median(step_times):.0f} tok/s)")

    # ── 2. Without int() sync ──
    print("=" * 60)
    print("2. Without int() sync")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
    _ = int(t)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Total:   {total_ms:.1f} ms  ({total_ms/GEN_LEN:.3f} ms/step, {GEN_LEN/total_ms*1000:.0f} tok/s)")

    # ── 3. GPU kernel time ──
    print("=" * 60)
    print("3. GPU kernel time (block_until_ready)")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    gpu_times = []
    for i in range(GEN_LEN):
        _ = kv_tmp.block_until_ready()
        t0 = time.perf_counter()
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = logits_d.block_until_ready()
        gpu_times.append(time.perf_counter() - t0)
        t = jnp.argmax(logits_d)
        _ = int(t)
    gpu_times = np.array(gpu_times)
    print(f"  Median:  {np.median(gpu_times)*1000:.3f} ms")
    print(f"  Min:     {np.min(gpu_times)*1000:.3f} ms")

    # ── 4. argmax + sync cost ──
    print("=" * 60)
    print("4. argmax + int() sync cost")
    print("=" * 60)
    kv_tmp = kv_packed
    t = tok
    sync_times = []
    for i in range(GEN_LEN):
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = logits_d.block_until_ready()
        t0 = time.perf_counter()
        t = jnp.argmax(logits_d)
        _ = int(t)
        sync_times.append(time.perf_counter() - t0)
    sync_times = np.array(sync_times)
    print(f"  Median:  {np.median(sync_times)*1000:.3f} ms")

    # ── Summary ──
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = np.median(step_times) * 1000
    gpu = np.median(gpu_times) * 1000
    sync = np.median(sync_times) * 1000
    host = total - gpu - sync
    print(f"  Total per step:     {total:.3f} ms  ({1000/total:.0f} tok/s)")
    print(f"  GPU kernel:         {gpu:.3f} ms ({gpu/total*100:.0f}%)")
    print(f"  argmax + sync:      {sync:.3f} ms ({sync/total*100:.0f}%)")
    print(f"  Host overhead:      {host:.3f} ms ({host/total*100:.0f}%)")
    print(f"  Theoretical min:    0.081 ms")
    print(f"  BW util (kernel):   {0.081/gpu*100:.0f}%")
    print(f"  No-sync throughput: {GEN_LEN/total_ms*1000:.0f} tok/s")

    with open(os.path.join(os.path.dirname(__file__), "profile_multi_sm_results.txt"), "w") as f:
        f.write(f"# Multi-SM profiling — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"total_per_step_ms: {total:.3f}\n")
        f.write(f"gpu_kernel_ms: {gpu:.3f}\n")
        f.write(f"argmax_sync_ms: {sync:.3f}\n")
        f.write(f"host_overhead_ms: {host:.3f}\n")
        f.write(f"no_sync_per_step_ms: {total_ms/GEN_LEN:.3f}\n")


if __name__ == "__main__":
    main()
