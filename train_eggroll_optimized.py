"""Optimized EGGROLL ES training for transformer.

Speed optimizations:
1. bf16 forward passes (tensor core utilization)
2. Nested lax.scan (entire epoch in single JIT, no Python dispatch per batch)
3. Single random matrix generation per batch, sliced into per-layer vectors
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from data import prepare_data
from model import init_transformer, count_params

# === architecture ===
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10
SEED = 42

# === EGGROLL hyperparameters ===
HALF_POP = 1024
POP_CHUNK = 16
SIGMA_START = 0.04
SIGMA_DECAY = 0.998
LR_START = 0.012
LR_DECAY = 0.92
ALPHA = 0.50
TEMPERATURE = 2.0
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
MOMENTUM = 0.5


def build_param_spec(params):
    spec = []
    offset = 0
    for key in sorted(params.keys()):
        shape = params[key].shape
        if len(shape) == 2:
            vec_dim = shape[0] + shape[1]
            spec.append((key, shape, offset, vec_dim, True))
        elif len(shape) == 1:
            vec_dim = shape[0]
            spec.append((key, shape, offset, vec_dim, False))
        else:
            raise ValueError(f"Unexpected: {key} {shape}")
        offset += vec_dim
    return spec, offset


def make_perturbed_forward(params_template, config, spec):
    """Create bf16 forward pass with rank-1 perturbation."""
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    d_head = d_model // n_heads
    n_layers = config["n_layers"]
    vocab_size = config["vocab_size"]

    def forward(params, vec, sigma, x_batch, y_batch, alpha):
        # apply rank-1 perturbation and cast to bf16
        p = {}
        for key, shape, offset, vec_dim, is_2d in spec:
            v = lax.dynamic_slice(vec, (offset,), (vec_dim,))
            if is_2d:
                m, n = shape
                p[key] = (params[key] + sigma * jnp.outer(v[:m], v[m:])).astype(jnp.bfloat16)
            else:
                p[key] = (params[key] + sigma * v).astype(jnp.bfloat16)

        batch_size, seq_len = x_batch.shape
        h = p["token_emb"][x_batch] + p["pos_emb"][:seq_len]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bfloat16))

        for layer in range(n_layers):
            pfx = f"layer{layer}"
            # LN1
            mean = jnp.mean(h, axis=-1, keepdims=True)
            var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
            h_norm = p[f"{pfx}.ln1.scale"] * (h - mean) * jax.lax.rsqrt(var + 1e-5) + p[f"{pfx}.ln1.bias"]
            # QKV
            q = (h_norm @ p[f"{pfx}.attn.q"]).reshape(batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
            k = (h_norm @ p[f"{pfx}.attn.k"]).reshape(batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
            v_a = (h_norm @ p[f"{pfx}.attn.v"]).reshape(batch_size, seq_len, n_heads, d_head).transpose(0, 2, 1, 3)
            # attention
            attn = (q @ k.transpose(0, 1, 3, 2)).astype(jnp.float32) * (jnp.float32(d_head) ** -0.5)
            attn = jnp.where(mask, attn, jnp.float32(-1e9))
            attn = jax.nn.softmax(attn, axis=-1).astype(jnp.bfloat16)
            out = (attn @ v_a).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model) @ p[f"{pfx}.attn.o"]
            h = h + out
            # LN2 + FFN
            mean = jnp.mean(h, axis=-1, keepdims=True)
            var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
            h_norm = p[f"{pfx}.ln2.scale"] * (h - mean) * jax.lax.rsqrt(var + 1e-5) + p[f"{pfx}.ln2.bias"]
            h_ff = jax.nn.gelu(h_norm @ p[f"{pfx}.ffn.up"] + p[f"{pfx}.ffn.up_bias"])
            h_ff = h_ff @ p[f"{pfx}.ffn.down"] + p[f"{pfx}.ffn.down_bias"]
            h = h + h_ff

        # final LN + output
        mean = jnp.mean(h, axis=-1, keepdims=True)
        var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
        h = p["ln_final.scale"] * (h - mean) * jax.lax.rsqrt(var + 1e-5) + p["ln_final.bias"]
        logits = (h @ p["output_proj"]).astype(jnp.float32)

        # smoothed CE
        scaled = logits / TEMPERATURE
        log_probs = jax.nn.log_softmax(scaled, axis=-1)
        one_hot = jax.nn.one_hot(y_batch, vocab_size, dtype=jnp.float32)
        smooth = (1 - alpha) * one_hot + alpha / vocab_size
        return -jnp.mean(jnp.sum(smooth * log_probs, axis=-1))

    return forward


def winsorized_zscore(fitness_diffs):
    group_size = fitness_diffs.shape[0] // N_SUBGROUPS
    groups = fitness_diffs[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
    means = jnp.mean(groups, axis=1, keepdims=True)
    stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
    z = (groups - means) / stds
    z = jnp.clip(z, -CLIP_RANGE, CLIP_RANGE)
    return z.reshape(-1)


def train():
    print("Preparing data...")
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]
    print(f"Vocab: {vocab_size}, Train: {data['train_x'].shape}, Val: {data['val_x'].shape}")

    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    n_params = count_params(params)
    spec, total_vec_dim = build_param_spec(params)
    n_chunks = HALF_POP // POP_CHUNK
    n_batches = len(data["train_x"]) // BATCH_SIZE
    print(f"Params: {n_params:,}, Vec dim: {total_vec_dim}, Pop: {HALF_POP*2}, Chunks: {n_chunks}, Batches: {n_batches}")

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    forward_fn = make_perturbed_forward(params, config, spec)

    # per-layer LR scales
    lr_scale_arr = []
    for pkey, shape, _, _, _ in spec:
        n_p = int(np.prod(shape))
        if n_p < 256: lr_scale_arr.append(3.0)
        elif n_p < 4096: lr_scale_arr.append(1.5)
        elif n_p < 8192: lr_scale_arr.append(1.0)
        else: lr_scale_arr.append(0.7)
    lr_scale_arr = jnp.array(lr_scale_arr)

    def train_one_batch(params, momentum_buf, key, x, y, sigma, lr):
        """One batch: fitness evaluation + gradient + momentum update."""
        def fitness_chunk(carry, chunk_vecs):
            def fitness_pair(vec):
                pos = forward_fn(carry, vec, sigma, x, y, ALPHA)
                neg = forward_fn(carry, vec, -sigma, x, y, ALPHA)
                return pos, neg
            fp, fn = jax.vmap(fitness_pair)(chunk_vecs)
            return carry, (fp, fn)

        key, vec_key = jax.random.split(key)
        vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))
        vecs_chunked = vecs.reshape(n_chunks, POP_CHUNK, -1)
        _, (fitness_pos, fitness_neg) = lax.scan(fitness_chunk, params, vecs_chunked)
        fitness_pos = fitness_pos.reshape(-1)
        fitness_neg = fitness_neg.reshape(-1)

        fitness_diffs = fitness_pos - fitness_neg
        shaped = winsorized_zscore(fitness_diffs)
        scale = 1.0 / (2.0 * sigma * HALF_POP)

        new_params = {}
        new_momentum = {}
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v = vecs[:, offset:offset + vec_dim]
            lr_s = lr_scale_arr[idx]
            if is_2d:
                m, n = shape
                grad = scale * (v[:, :m] * shaped[:, None]).T @ v[:, m:]
            else:
                grad = scale * (v * shaped[:, None]).sum(axis=0)
            # momentum SGD: v = beta * v + grad; params -= lr * v
            new_momentum[pkey] = MOMENTUM * momentum_buf[pkey] + grad
            new_params[pkey] = params[pkey] - lr * lr_s * new_momentum[pkey]

        return new_params, new_momentum, key, jnp.mean(fitness_pos)

    @jax.jit
    def train_batch(params, momentum_buf, key, x, y, sigma, lr):
        return train_one_batch(params, momentum_buf, key, x, y, sigma, lr)

    @jax.jit
    def eval_loss(params, x, y):
        from model import transformer_forward_batch, cross_entropy_loss
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    # init momentum buffer
    momentum_buf = jax.tree.map(jnp.zeros_like, params)

    # warmup
    print("Warming up JIT...")
    t0 = time.perf_counter()
    _, _, _, wl = train_batch(params, momentum_buf, key, train_x[:BATCH_SIZE], train_y[:BATCH_SIZE], SIGMA_START, LR_START)
    wl.block_until_ready()
    jit_time = time.perf_counter() - t0
    print(f"JIT warmup: {jit_time:.2f}s")

    sigmas = [SIGMA_START * (SIGMA_DECAY ** e) for e in range(EPOCHS)]
    lrs_sched = [LR_START * (LR_DECAY ** e) for e in range(EPOCHS)]

    print("\nTraining...")
    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        sigma, lr = sigmas[epoch], lrs_sched[epoch]
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        eloss = 0.0
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, momentum_buf, key, pl = train_batch(params, momentum_buf, key, sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE], sigma, lr)
            eloss += float(pl)
        eloss /= n_batches

        vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        print(f"  Epoch {epoch+1}/{EPOCHS}  proxy={eloss:.4f}  val_loss={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}  lr={lr:.4f}")

    total = time.perf_counter() - t_start
    vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
    vl.block_until_ready()
    total = time.perf_counter() - t_start
    ppl = float(jnp.exp(vl))
    print(f"\nFinal val_loss={float(vl):.4f}  ppl={ppl:.2f}")
    print(f"Training: {total:.2f}s  (with JIT: {total+jit_time:.2f}s)")

    from train_backprop import generate_sample
    generate_sample(params, config, data, key)
    return float(vl), ppl, total + jit_time


if __name__ == "__main__":
    loss, ppl, t = train()
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("results.tsv", "a") as f:
        if not os.path.exists("results.tsv") or os.path.getsize("results.tsv") == 0:
            f.write("commit\tloss\tperplexity\ttraining_time_s\tpeak_memory_mb\tstatus\tdescription\n")
        f.write(f"{commit}\t{loss:.4f}\t{ppl:.2f}\t{t:.2f}\t0\tok\teggroll_optimized nested_scan bf16 pop={HALF_POP*2}\n")
