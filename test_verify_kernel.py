"""Test that verify_decode kernel matches sequential decode."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import numpy as np
import jax
import jax.numpy as jnp

from data import prepare_data, load_bpe_vocab
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


def main():
    params, config = load_model("target_weights.pkl")
    vocab_size = config["vocab_size"]
    data = prepare_data(tokenizer="trained_bpe", bpe_vocab_size=vocab_size, dataset="tinystories")

    prompt = jnp.array(data["train_x"][0][:128], dtype=jnp.int32)
    ctx_len = config["context_len"]

    # Prefill
    x = jnp.pad(prompt, (0, ctx_len - len(prompt))).astype(jnp.int32)
    logits, k_caches, v_caches = block_prefill(params, config, x, vocab_size)
    kv_packed = pack_kv_caches(k_caches, v_caches)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)

    # Get first token
    tok = int(jnp.argmax(logits[len(prompt) - 1]))
    print(f"First token: {tok}")

    # Generate 4 tokens sequentially for ground truth
    K = 4
    seq_tokens = []
    seq_logits = []
    kv_seq = kv_packed
    cur_tok = tok
    for i in range(K):
        logits_i, kv_seq = fused_decode_nlayer(w, config, cur_tok, len(prompt) + i, kv_seq, vocab_size)
        seq_tokens.append(int(jnp.argmax(logits_i)))
        seq_logits.append(np.array(logits_i))
        cur_tok = seq_tokens[-1]

    print(f"Sequential tokens: {seq_tokens}")
    print(f"Sequential logits[0] top5: {np.argsort(seq_logits[0])[-5:][::-1]}")

    # Now verify the same tokens with the parallel kernel
    # The verify kernel takes the tokens that were FED to the model (not the outputs)
    # Token 0 at position len(prompt) is `tok`, token 1 is `seq_tokens[0]`, etc.
    verify_input = jnp.array([tok] + seq_tokens[:K-1], dtype=jnp.int32)
    print(f"\nVerify input tokens: {list(verify_input)}")

    # Need NUM_TOKENS to be a power of 2 for Triton
    num_tokens = K  # 4 is already power of 2
    ver_logits, kv_ver = verify_decode(w, config, verify_input, len(prompt), kv_packed, vocab_size, num_tokens)

    print(f"\nVerify logits shape: {ver_logits.shape}")
    for i in range(K):
        ver_tok = int(jnp.argmax(ver_logits[i]))
        seq_tok = seq_tokens[i]
        match = "OK" if ver_tok == seq_tok else "MISMATCH"
        max_diff = float(jnp.max(jnp.abs(ver_logits[i] - seq_logits[i])))
        print(f"  Position {i}: verify={ver_tok} sequential={seq_tok} {match}  max_logit_diff={max_diff:.4f}")

    # Check overall match
    all_match = all(int(jnp.argmax(ver_logits[i])) == seq_tokens[i] for i in range(K))
    print(f"\nAll tokens match: {all_match}")


if __name__ == "__main__":
    main()
