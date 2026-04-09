"""Profile the multi-SM decode kernel.

Usage:
  uv run profile_kernels.py              # profile current model
  uv run profile_kernels.py --gen-len 256
"""

import os
import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import load_bpe_vocab
from model import count_params, prefill_with_kv
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    return params, saved["config"]


def measure_prefill(params, config, prompt, n_runs=20):
    """Measure JAX prefill latency."""
    x = jnp.pad(prompt, (0, config["context_len"] - len(prompt))).astype(jnp.int32)
    for _ in range(3):
        logits, kc, vc = prefill_with_kv(params, config, x)
        _ = logits.block_until_ready()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        logits, kc, vc = prefill_with_kv(params, config, x)
        _ = logits.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000, logits, kc, vc


def measure_decode(w, config, tok, start_pos, kv_packed, n_tokens=128, n_runs=10):
    """Measure multi-SM decode throughput."""
    kv_tmp = kv_packed
    t = tok
    for i in range(5):
        t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv_tmp)
        _ = int(t)

    times = []
    all_tokens = []
    for _ in range(n_runs):
        kv_tmp = kv_packed
        t = tok
        tokens = []
        t0 = time.perf_counter()
        for i in range(n_tokens):
            t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv_tmp)
            tokens.append(int(t))
        times.append(time.perf_counter() - t0)
        all_tokens = tokens
    return np.median(times) * 1000, all_tokens


def compute_memory_stats(config):
    """Compute theoretical memory usage."""
    d = config["d_model"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    n_layers = config["n_layers"]
    d_head = config["d_head"]
    d_kv = n_kv_heads * d_head
    d_ff = config.get("d_ff", 4 * d)
    ctx = config["context_len"]

    per_layer = (d + d*d + d*d_kv + d*d_kv + d*d + d + d*d_ff + d*d_ff + d_ff*d)
    total_weights = config["vocab_size"] * d + n_layers * per_layer + d
    weight_bytes = total_weights * 2  # bf16
    kv_total = n_layers * 2 * n_kv_heads * ctx * d_head * 2  # bf16

    return {
        "weight_buffer_mb": weight_bytes / 1e6,
        "kv_cache_mb": kv_total / 1e6,
        "total_inference_mb": (weight_bytes + kv_total) / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    params, config = load_params()
    d = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    ctx = config["context_len"]
    GEN_LEN = args.gen_len
    PROMPT_LEN = min(128, ctx)

    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]

    # Real prompt for meaningful generated text
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(bpe_vocab["tokenizer_path"])
    _prompt_text = ("The cat sat on the mat and looked at the sky. Once upon a time, "
                    "in a land far away, there lived a young wizard who dreamed of "
                    "adventure and discovery. Every morning he would wake before dawn, "
                    "climb to the top of the highest tower, and gaze at the stars fading "
                    "into the pale blue light. He wondered what secrets the universe held, "
                    "and whether he would ever find the courage to leave his small village "
                    "and explore the great unknown world beyond the distant mountains and "
                    "the vast dark forests that stretched endlessly toward the horizon. "
                    "The wizard had read every book in the village library, memorized "
                    "every spell and incantation, and practiced until his fingers ached.")
    _prompt_ids = _tok.encode(_prompt_text).ids[:PROMPT_LEN]
    assert len(_prompt_ids) >= PROMPT_LEN, f"prompt too short: {len(_prompt_ids)} < {PROMPT_LEN}"
    prompt = jnp.array(_prompt_ids, dtype=jnp.int32)

    mem = compute_memory_stats(config)

    print(f"{'='*60}")
    print(f"KERNEL PROFILING")
    print(f"{'='*60}")
    print(f"Model:    d={d} h={n_heads} l={n_layers} ctx={ctx}")
    print(f"Params:   {count_params(params):,}")
    print(f"Generate: {GEN_LEN} tokens from {PROMPT_LEN}-token prompt")
    print(f"GPU:      {jax.devices()[0].device_kind}")
    print(f"Weights:  {mem['weight_buffer_mb']:.1f} MB, KV cache: {mem['kv_cache_mb']:.1f} MB")
    print()

    # Prefill (JAX)
    print(f"--- Prefill ({PROMPT_LEN} tokens, JAX) ---")
    prefill_ms, logits, kc, vc = measure_prefill(params, config, prompt)
    print(f"  {prefill_ms:.1f} ms ({PROMPT_LEN / prefill_ms * 1000:.0f} tok/s)")
    print()

    # Decode (Triton multi-SM)
    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    print(f"--- Decode ({GEN_LEN} tokens, Triton multi-SM, grid={n_heads}) ---")
    decode_ms, tokens = measure_decode(w, config, tok, PROMPT_LEN, kv_packed,
                                       n_tokens=GEN_LEN, n_runs=args.n_runs)
    tok_per_s = GEN_LEN / decode_ms * 1000
    ms_per_tok = decode_ms / GEN_LEN
    print(f"  {decode_ms:.1f} ms total, {ms_per_tok:.3f} ms/tok, {tok_per_s:.0f} tok/s")
    print()

    # End-to-end
    total_ms = prefill_ms + decode_ms
    print(f"--- End-to-End ---")
    print(f"  {total_ms:.1f} ms for {PROMPT_LEN + GEN_LEN} tokens")
    print(f"  Text: {decode_fn(tokens)[:200]}...")
    print()

    # Roofline
    bytes_per_step = (mem["weight_buffer_mb"] + mem["kv_cache_mb"]) * 1e6
    theoretical_min_ms = bytes_per_step / (836e9) * 1000  # 836 GB/s RTX 4080 Super
    bandwidth_util = theoretical_min_ms / ms_per_tok * 100
    print(f"--- Roofline ---")
    print(f"  {bytes_per_step / 1e6:.1f} MB per step, theoretical min {theoretical_min_ms:.3f} ms/tok")
    print(f"  Achieved {ms_per_tok:.3f} ms/tok = {bandwidth_util:.0f}% bandwidth utilization")


if __name__ == "__main__":
    main()
