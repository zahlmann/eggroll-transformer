"""Debug: test if asymmetric barrier produces correct results vs original barrier."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import jax
import jax.numpy as jnp

from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_params():
    with open("weights.pkl", "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    prompt_len = 128

    prompt = jnp.arange(prompt_len, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - prompt_len)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[prompt_len - 1])

    # Compare step-by-step: kv_splits=1 vs kv_splits=2
    print("Step-by-step comparison (kv_splits=1 vs kv_splits=2):")
    kv1 = kv_packed
    kv2 = kv_packed
    t1 = tok
    t2 = tok
    for i in range(20):
        t1, logits1, kv1 = multi_sm_decode_nlayer(w, config, t1, prompt_len + i, kv1, vocab_size, kv_splits=1)
        t2, logits2, kv2 = multi_sm_decode_nlayer(w, config, t2, prompt_len + i, kv2, vocab_size, kv_splits=2)
        tok1 = int(t1)
        tok2 = int(t2)
        ldiff = float(jnp.max(jnp.abs(logits1 - logits2)))
        kv_diff = float(jnp.max(jnp.abs(kv1.astype(jnp.float32) - kv2.astype(jnp.float32))))
        match = "Y" if tok1 == tok2 else "N"
        print(f"  step {i:2d}: tok={tok1:4d}/{tok2:4d} {match}  logit_diff={ldiff:.4f}  kv_diff={kv_diff:.4f}")
        if tok1 != tok2:
            break


if __name__ == "__main__":
    main()
