import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
import pickle, time, numpy as np, jax, jax.numpy as jnp
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer

with open("weights.pkl", "rb") as f:
    saved = pickle.load(f)
params = {k: jnp.array(v) for k, v in saved["params"].items()}
config = saved["config"]
vocab_size = config["vocab_size"]
prompt = jnp.arange(128, dtype=jnp.int32) % vocab_size
x = jnp.pad(prompt, (0, config["context_len"] - 128)).astype(jnp.int32)
logits, kc, vc = block_prefill(params, config, x, vocab_size)
_ = logits.block_until_ready()
kv_packed = pack_kv_caches(kc, vc)
w = prepare_decode_weights_nlayer(params, config, vocab_size)
tok = jnp.argmax(logits[127])

for kv in [2, 4]:
    # warmup
    kv_tmp, t = kv_packed, tok
    for i in range(5):
        t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, 128+i, kv_tmp, vocab_size, kv_splits=kv)
        _ = int(t)
    # no-sync benchmark
    times = []
    for _ in range(10):
        kv_tmp, t = kv_packed, tok
        t0 = time.perf_counter()
        for i in range(128):
            t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, 128+i, kv_tmp, vocab_size, kv_splits=kv)
        _ = kv_tmp.block_until_ready()
        times.append(time.perf_counter() - t0)
    ms = np.median(times) * 1000
    print(f"kv_splits={kv} (grid={config['n_heads']*kv}): {128/ms*1000:.0f} tok/s no-sync ({ms/128:.3f} ms/tok)")
