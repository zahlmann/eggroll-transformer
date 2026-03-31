"""Test parallel residual model: correctness and performance vs sequential."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from model import prefill_with_kv
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_weights(path):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def benchmark(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens=128, n_runs=10):
    kv, t = kv_packed, tok
    for i in range(3):
        t, _, kv = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv, vocab_size)
        _ = int(t)

    # With sync
    times_sync = []
    for _ in range(n_runs):
        kv, t = kv_packed, tok
        t0 = time.perf_counter()
        for i in range(n_tokens):
            t, _, kv = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv, vocab_size)
            _ = int(t)
        times_sync.append(time.perf_counter() - t0)

    # No sync
    times_nosync = []
    for _ in range(n_runs):
        kv, t = kv_packed, tok
        t0 = time.perf_counter()
        for i in range(n_tokens):
            t, _, kv = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv, vocab_size)
        _ = kv.block_until_ready()
        times_nosync.append(time.perf_counter() - t0)

    return np.median(times_sync) * 1000, np.median(times_nosync) * 1000


def test_model(path, label):
    params, config = load_weights(path)
    vocab_size = config["vocab_size"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    parallel = config.get("parallel_residual", False)
    prompt_len, gen_len = 128, 128

    print(f"=== {label} ===")
    print(f"  d={config['d_model']} h={n_heads} kv_h={n_kv_heads} l={config['n_layers']}")
    print(f"  parallel_residual={parallel}")

    prompt = jnp.arange(prompt_len, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - prompt_len)).astype(jnp.int32)
    logits, kc, vc = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[prompt_len - 1])

    # Correctness: generate tokens
    tokens = []
    kv, t = kv_packed, tok
    for i in range(20):
        t, _, kv = multi_sm_decode_nlayer(w, config, t, prompt_len + i, kv, vocab_size)
        tokens.append(int(t))
    print(f"  tokens: {tokens[:10]}...")

    # Performance
    ms_sync, ms_nosync = benchmark(w, config, tok, prompt_len, kv_packed, vocab_size, gen_len)
    print(f"  sync:   {gen_len/ms_sync*1000:.0f} tok/s ({ms_sync/gen_len:.3f} ms/tok)")
    print(f"  nosync: {gen_len/ms_nosync*1000:.0f} tok/s ({ms_nosync/gen_len:.3f} ms/tok)")
    return gen_len/ms_sync*1000, gen_len/ms_nosync*1000


def main():
    results = {}

    # Test MHA (sequential)
    if os.path.exists("weights_mha.pkl"):
        sync, nosync = test_model("weights_mha.pkl", "MHA Sequential")
        results["mha_seq"] = (sync, nosync)
        print()

    # Test parallel residual model
    if os.path.exists("weights.pkl"):
        sync, nosync = test_model("weights.pkl", "Current Model")
        results["current"] = (sync, nosync)
        print()

    # Summary
    if len(results) >= 2:
        print("=== COMPARISON ===")
        for name, (s, ns) in results.items():
            print(f"  {name}: {s:.0f} tok/s (sync), {ns:.0f} tok/s (nosync)")
        if "mha_seq" in results and "current" in results:
            s_mha, ns_mha = results["mha_seq"]
            s_cur, ns_cur = results["current"]
            print(f"  Speedup: {s_cur/s_mha:.2f}x (sync), {ns_cur/ns_mha:.2f}x (nosync)")


if __name__ == "__main__":
    main()
