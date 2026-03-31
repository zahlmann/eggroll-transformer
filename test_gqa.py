"""Test GQA model: correctness (JAX vs kernel) and performance comparison with MHA."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from model import prefill_with_kv, transformer_forward
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_weights(path):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def benchmark_decode(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens=128, n_runs=10):
    # Warmup
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

    ms_sync = np.median(times_sync) * 1000
    ms_nosync = np.median(times_nosync) * 1000
    return ms_sync, ms_nosync


def main():
    # Check if GQA weights exist
    gqa_path = "weights.pkl"
    mha_path = "weights_mha.pkl"

    params, config = load_weights(gqa_path)
    vocab_size = config["vocab_size"]
    n_kv_heads = config.get("n_kv_heads", config["n_heads"])
    n_heads = config["n_heads"]
    d_model = config["d_model"]
    prompt_len = 128
    gen_len = 128

    print(f"Model: d={d_model} h={n_heads} kv_h={n_kv_heads} l={config['n_layers']} ctx={config['context_len']}")
    print(f"GQA: {'yes' if n_kv_heads != n_heads else 'no'} ({n_heads}Q/{n_kv_heads}KV)")
    print()

    # Prefill with JAX
    prompt = jnp.arange(prompt_len, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - prompt_len)).astype(jnp.int32)

    logits_jax, kc, vc = prefill_with_kv(params, config, x)
    _ = logits_jax.block_until_ready()

    # Correctness: compare JAX forward with kernel decode for first token
    tok_jax = int(jnp.argmax(logits_jax[prompt_len - 1]))
    print(f"First token (JAX prefill): {tok_jax}")

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)

    # Decode one token with kernel
    tok_kernel, logits_kernel, _ = multi_sm_decode_nlayer(
        w, config, tok_jax, prompt_len, kv_packed, vocab_size)
    tok_kernel = int(tok_kernel)

    # Compare with JAX decode
    # JAX: run full forward with one more token
    x2 = x.at[prompt_len].set(tok_jax)
    logits_jax2, _, _ = prefill_with_kv(params, config, x2)
    tok_jax2 = int(jnp.argmax(logits_jax2[prompt_len]))

    print(f"Second token - JAX: {tok_jax2}, Kernel: {tok_kernel}, Match: {tok_jax2 == tok_kernel}")
    print()

    # Performance
    print(f"=== Performance ({gen_len} tokens) ===")
    ms_sync, ms_nosync = benchmark_decode(w, config, tok_jax, prompt_len, kv_packed, vocab_size,
                                           n_tokens=gen_len)
    tok_s_sync = gen_len / ms_sync * 1000
    tok_s_nosync = gen_len / ms_nosync * 1000

    kv_cache_mb = config["n_layers"] * 2 * n_kv_heads * config["context_len"] * config["d_head"] * 2 / 1e6
    # Weight buffer size (approximate)
    d_kv = n_kv_heads * config["d_head"]
    per_layer_w = (2*d_model + d_model*d_model + 2*d_model*d_kv + d_model*d_model +
                   2*d_model + d_model*4*d_model + 4*d_model + 4*d_model*d_model + d_model) * 2
    weight_mb = (config["n_layers"] * per_layer_w + vocab_size*d_model*2 + config["context_len"]*d_model*2 +
                 2*d_model*2 + d_model*vocab_size*2) / 1e6
    total_mb = weight_mb + kv_cache_mb

    print(f"With sync:    {tok_s_sync:.0f} tok/s  ({ms_sync/gen_len:.3f} ms/tok)")
    print(f"No sync:      {tok_s_nosync:.0f} tok/s  ({ms_nosync/gen_len:.3f} ms/tok)")
    print(f"Weight buf:   ~{weight_mb:.1f} MB")
    print(f"KV cache:     {kv_cache_mb:.1f} MB")
    print(f"Total data:   ~{total_mb:.1f} MB (L2: {'fits' if total_mb < 64 else 'overflow'})")
    print()

    # Compare with MHA if available
    if os.path.exists(mha_path):
        print(f"=== MHA Comparison ===")
        params_mha, config_mha = load_weights(mha_path)
        logits_mha, kc_mha, vc_mha = prefill_with_kv(params_mha, config_mha,
            jnp.pad(prompt, (0, config_mha["context_len"] - prompt_len)).astype(jnp.int32))
        _ = logits_mha.block_until_ready()
        kv_mha = pack_kv_caches(kc_mha, vc_mha)
        w_mha = prepare_decode_weights_nlayer(params_mha, config_mha, config_mha["vocab_size"])
        tok_mha = jnp.argmax(logits_mha[prompt_len - 1])

        ms_sync_mha, ms_nosync_mha = benchmark_decode(
            w_mha, config_mha, tok_mha, prompt_len, kv_mha, config_mha["vocab_size"], n_tokens=gen_len)
        tok_s_sync_mha = gen_len / ms_sync_mha * 1000
        tok_s_nosync_mha = gen_len / ms_nosync_mha * 1000
        print(f"MHA with sync:  {tok_s_sync_mha:.0f} tok/s  ({ms_sync_mha/gen_len:.3f} ms/tok)")
        print(f"MHA no sync:    {tok_s_nosync_mha:.0f} tok/s  ({ms_nosync_mha/gen_len:.3f} ms/tok)")
        print(f"GQA speedup:    {tok_s_sync/tok_s_sync_mha:.2f}x (sync), {tok_s_nosync/tok_s_nosync_mha:.2f}x (no sync)")

    # Save results
    with open("gqa_results.txt", "w") as f:
        f.write(f"model: d={d_model} h={n_heads} kv_h={n_kv_heads}\n")
        f.write(f"sync_tok_s: {tok_s_sync:.0f}\n")
        f.write(f"nosync_tok_s: {tok_s_nosync:.0f}\n")
        f.write(f"kv_cache_mb: {kv_cache_mb:.1f}\n")
        f.write(f"total_data_mb: {total_mb:.1f}\n")
    print("\nSaved to gqa_results.txt")


if __name__ == "__main__":
    main()
