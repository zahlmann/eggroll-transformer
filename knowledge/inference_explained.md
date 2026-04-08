# Custom GPU Kernels for Transformer Inference

How to write Triton kernels that fuse an entire transformer decode step into a single
GPU launch — from register-level data flow to multi-SM parallelism with atomic barriers.
No GPU experience required — just Python and a rough idea of what neural nets do.

---

## Table of Contents

**Part I: The Fundamentals (d=64, 1 layer, 66K params)**

1. [The Problem](#1-the-problem)
2. [How GPUs Actually Work](#2-how-gpus-actually-work)
3. [The Memory Wall](#3-the-memory-wall)
4. [Our Transformer (the Tiny One)](#4-our-transformer-the-tiny-one)
5. [What Happens When JAX Runs Your Model](#5-what-happens-when-jax-runs-your-model)
6. [The Big Idea: One Kernel to Rule Them All](#6-the-big-idea-one-kernel-to-rule-them-all)
7. [Triton: GPU Programming for Humans](#7-triton-gpu-programming-for-humans)
8. [The Prefill Kernel, Line by Line](#8-the-prefill-kernel-line-by-line)
9. [KV Cache: Why We Don't Recompute Everything](#9-kv-cache-why-we-dont-recompute-everything)
10. [The Decode Kernel, Line by Line](#10-the-decode-kernel-line-by-line)
11. [The Generation Loop](#11-the-generation-loop)
12. [Results and Why It's Faster](#12-results-and-why-its-faster)

**Part II: Scaling Up (d=512, 8 layers, 30M params)**

13. [Scaling the Model](#13-scaling-the-model)
14. [Multi-SM Decode: Using the Whole GPU](#14-multi-sm-decode-using-the-whole-gpu)
15. [What Didn't Work](#15-what-didnt-work)
16. [Key Lessons](#16-key-lessons)

**Part III: Production Scale (d=768+, 12-24 layers)**

17. [GQA: Grouped Query Attention](#17-gqa-grouped-query-attention)
18. [Non-Power-of-2 D_MODEL (D_BLOCK Padding)](#18-non-power-of-2-d_model-d_block-padding)
19. [L2 Cache Hints and Bandwidth Optimization](#19-l2-cache-hints-and-bandwidth-optimization)
20. [Tensor Core Batched Projections](#20-tensor-core-batched-projections)
21. [Sampling and Decoding Strategies](#21-sampling-and-decoding-strategies)

---

## 1. The Problem

You have a trained transformer model. You want to generate text with it — feed in a prompt,
get tokens out, one at a time. The standard approach uses JAX (or PyTorch), which compiles
your Python code into GPU operations via XLA. This works, but it's slow.

**How slow?** For our tiny model (66K parameters, character-level Shakespeare), JAX generates
at 185 tokens/second. Our custom Triton kernel does the same job at 740 tokens/second.
**4x faster**, same outputs.

The speed difference isn't about better math. Both do identical matrix multiplications.
The difference is about **where the data lives** during computation.

---

## 2. How GPUs Actually Work

A GPU is not one fast processor — it's thousands of slow processors that all run the same
code simultaneously. An NVIDIA GPU has about 10,000 "CUDA cores" organized into groups.

### The three things that matter

**Threads.** The GPU runs your code on thousands of threads at once. You don't control
individual threads. Instead, you write a "kernel" — a function that says what ONE thread
block should do — and the GPU runs many copies of it in parallel.

**Warps.** Threads are grouped into "warps" of 32 threads that execute in lockstep. If one
thread in a warp takes a branch and the others don't, all 32 wait. A "thread block" typically
has 4-8 warps (128-256 threads).

**Tensor cores.** Special hardware units that multiply small matrices (like 16×16) in a single
clock cycle. They're the reason modern GPUs are so fast for deep learning. To use them, your
data must be in bfloat16 or float16 format.

### The memory hierarchy

This is the single most important concept for GPU performance:

```
┌─────────────────────────────┐
│        Registers            │   ← Fastest. Private to each thread.
│   Speed: ~1 cycle           │     128-256 registers per thread.
│   Size:  ~256 KB per block  │     Data here costs almost nothing to read.
├─────────────────────────────┤
│      Shared Memory          │   ← Fast. Shared within a thread block.
│   Speed: ~5 cycles          │     Up to 164 KB per block (on modern GPUs).
│   Size:  ~164 KB per block  │     Useful for communication between threads.
├─────────────────────────────┤
│     HBM (Global Memory)     │   ← Slow. The GPU's main memory.
│   Speed: ~200-400 cycles    │     Where your tensors normally live.
│   Size:  16-80 GB           │     Every jnp.array sits here.
└─────────────────────────────┘
```

**The speed gap is enormous.** Reading from registers is roughly 200x faster than reading
from HBM. Most GPU programs are "memory-bound" — the math units sit idle, waiting for
data to arrive from HBM.

For our model, every weight matrix fits in registers. The entire model (66K parameters ×
2 bytes = 132 KB) fits in the register file of a single thread block (128 threads × 255
registers × 4 bytes = 130 KB). This is the foundation of everything that follows.

---

## 3. The Memory Wall

When JAX (or PyTorch) runs a transformer, XLA compiles each operation into a separate GPU
kernel:

```
Operation              What happens in memory
─────────────────────  ─────────────────────────────────
token_emb[x]           Load tokens from HBM, load embedding table from HBM,
                       write result to HBM

+ pos_emb[:seq_len]    Load pos_emb from HBM, load previous result from HBM,
                       write sum to HBM

layer_norm(...)        Load from HBM, compute mean+var, write to HBM

x @ Wq                 Load x from HBM, load Wq from HBM, write Q to HBM
x @ Wk                 Load x from HBM, load Wk from HBM, write K to HBM
x @ Wv                 Load x from HBM, load Wv from HBM, write V to HBM

Q @ K^T                Load Q from HBM, load K from HBM, write scores to HBM
softmax(scores)        Load scores from HBM, write attention to HBM
attn @ V               Load attn from HBM, load V from HBM, write out to HBM

... (FFN, final LN, output projection — same pattern)
```

Count the HBM round-trips. **Every intermediate result** gets written to slow global memory
and then read back for the next operation. For a single forward pass with our model, that's
roughly 15-20 separate kernels, each doing a load-compute-store cycle.

The math takes nanoseconds. The memory transfers take microseconds. The kernel launch overhead
(just starting each kernel) takes microseconds. **Most of the GPU's time is spent waiting.**

---

## 4. Our Transformer

Before diving into the kernel code, let's understand exactly what our model computes.

### Architecture

```
vocab_size:    32000  (BPE tokenizer trained on corpus)
d_model:        1024  (dimension of token representations)
n_heads:          16  (number of query attention heads)
n_kv_heads:        4  (number of key/value heads — GQA)
d_head:           64  (= d_model / n_heads)
d_ff:           2816  (SwiGLU FFN hidden dimension)
n_layers:         24  (transformer layers)
context_len:     512  (maximum sequence length)
parameters: 306M
```

### The forward pass

Given a sequence of token IDs like `[14, 27, 51, 3, ...]`, the model:

**Step 1: Embedding.** Look up each token in a (32000 × 1024) table. Token 14 becomes a
1024-dimensional vector. No positional embedding — position is encoded via RoPE in step 3.
Result: `h` with shape (512, 1024).

**Step 2: RMSNorm.** For each position independently, normalize the 1024 values by their
root-mean-square, then scale by learned parameters. Simpler than LayerNorm (no mean
subtraction, no bias):

```
rms = sqrt(mean(h²) + 1e-5)
h_norm = scale * h / rms
```

**Step 3: Multi-Head Attention with GQA and RoPE.** This is where tokens "talk to each
other." 16 query heads, but only 4 KV heads (each shared by 4 query heads — GQA).

```
Q = h_norm @ Wq    # (512, 1024) — 16 heads × 64 dims
K = h_norm @ Wk    # (512, 256)  — 4 KV heads × 64 dims
V = h_norm @ Wv    # (512, 256)  — 4 KV heads × 64 dims

Q, K = apply_rope(Q, K, position)  # rotate by position-dependent angles

scores = Q @ K^T / sqrt(64)  # attention scores
scores = mask_future(scores)  # causal: position i only sees 0..i
attn = softmax(scores)
out = attn @ V
```

RoPE (Rotary Position Embeddings) encodes position by rotating Q and K vectors. The dot
product Q·K then depends on relative position, not absolute — better generalization.

Project back: `h = h + out @ Wo`

**Step 4: RMSNorm + SwiGLU FFN.** Three weight matrices with a gating mechanism:

```
h_norm = rms_norm(h)
gate = h_norm @ W_gate          # (512, 2816)
up   = h_norm @ W_up            # (512, 2816)
act  = silu(gate) * up          # gated activation
down = act @ W_down             # (2816, 1024) → back to model dim
h = h + down                    # residual connection
```

SiLU: `silu(x) = x * sigmoid(x)`. SwiGLU uses the gate output to control how much of
each hidden dimension passes through — consistently better quality than standard ReLU FFN.

**Repeat** steps 2-4 for all 24 layers.

**Step 5: Final RMSNorm + Output Projection.**

```
h = rms_norm(h)
logits = h @ token_emb.T    # (512, 32000) — tied weights (reuse embedding table)
```

The logits at position `i` are scores for what token should come at position `i+1`.
To generate text, look at the last position's logits and pick the highest (greedy)
or sample from the distribution.

---

## 5. What Happens When JAX Runs Your Model

JAX's `transformer_forward` in `model.py` is clean, readable Python:

```python
def transformer_forward(params, config, x):
    h = params["token_emb"][x] + params["pos_emb"][:seq_len]
    h_norm = layer_norm(h, params["layer0.ln1.scale"], params["layer0.ln1.bias"])
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads)
    h = h + attn_out
    h_norm = layer_norm(h, params["layer0.ln2.scale"], params["layer0.ln2.bias"])
    h_ff = jax.nn.gelu(h_norm @ params["layer0.ffn.up"])
    h_ff = h_ff @ params["layer0.ffn.down"]
    h = h + h_ff
    h = layer_norm(h, params["ln_final.scale"], params["ln_final.bias"])
    return h @ params["output_proj"]
```

When you call this with `jax.jit`, XLA compiles it into a sequence of GPU kernels.
Each line becomes one or more kernel launches. Each kernel reads its inputs from HBM,
computes, writes results to HBM, and the next kernel picks up from there.

For **one forward pass**, there are roughly 15-20 HBM round-trips. For **autoregressive
generation** of 64 tokens, JAX calls this function 64 times (once per token, with growing
input length), so that's **~1000 kernel launches** with HBM round-trips each.

This is what we're going to fix.

---

## 6. The Big Idea: One Kernel to Rule Them All

What if we wrote a single GPU kernel that does the **entire forward pass** — embedding,
layer norm, attention, FFN, output projection — without ever writing intermediate results
to HBM?

```
Normal (JAX/XLA):
  HBM → Embedding → HBM → LN → HBM → QKV → HBM → Attention → HBM → FFN → HBM → Logits

Fused kernel:
  HBM → [Embedding → LN → QKV → Attention → FFN → Logits] → HBM
         └──────────── all in registers ────────────────┘
```

The data enters registers once (loading weights + input tokens from HBM), flows through
every operation in registers, and writes the final logits back to HBM once. **One HBM
round-trip instead of fifteen.**

This is possible because our model is small enough to fit in registers:

```
Total parameters: 66,368 × 2 bytes (bf16) = ~132 KB
Register file per block: 128 threads × 255 regs × 4 bytes = ~130 KB
```

It's tight, but it works. The weights aren't all live simultaneously — each phase loads
the weights it needs, uses them, then those registers get reused for the next phase.

For the attention scores matrix (128 × 128 = 16K values at 4 bytes each = 64 KB), this
is the largest live tensor and the binding constraint. It fits because we use 4 warps
(128 threads), giving each thread enough registers.

---

## 7. Triton: GPU Programming for Humans

[Triton](https://openai.com/index/triton/) is a GPU programming language by OpenAI.
It compiles Python-like code to the same PTX machine code that CUDA produces, but it's
much easier to write because:

1. **You think in blocks, not threads.** Instead of "what does thread #47 do?", you think
   "what does this block of 128 values do?"
2. **Memory coalescing is automatic.** Triton figures out how to load data efficiently.
3. **Register allocation is automatic.** You don't manually manage which values go in
   which registers.

### Triton basics

A Triton kernel looks like a Python function with special types:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr):
    # tl.arange creates a vector of indices: [0, 1, 2, ..., 127]
    idx = tl.arange(0, 128)

    # tl.load reads from GPU memory into registers
    x = tl.load(input_ptr + idx)

    # Math happens in registers (no memory access)
    y = x * 2.0 + 1.0

    # tl.store writes from registers back to GPU memory
    tl.store(output_ptr + idx, y)
```

Key operations we use:

| Operation | What it does | Example |
|-----------|-------------|---------|
| `tl.load(ptr + offsets)` | Read from HBM into registers | Loading a weight matrix |
| `tl.store(ptr + offsets, val)` | Write from registers to HBM | Storing output logits |
| `tl.dot(A, B)` | Matrix multiply using tensor cores | Q @ K^T, h @ W_q |
| `tl.sum(x, axis=)` | Reduce along an axis | Computing mean for layer norm |
| `tl.exp(x)` | Element-wise exponential | Softmax numerator |
| `tl.where(cond, a, b)` | Conditional select | Causal mask |
| `tl.arange(0, N)` | Index vector [0, 1, ..., N-1] | Position offsets |
| `.to(tl.bfloat16)` | Cast to bfloat16 | Preparing for tensor core matmul |

### The bf16 + f32 pattern

Tensor cores require bfloat16 inputs but produce float32 outputs. So the standard pattern is:

```python
# Cast inputs to bf16 for tensor cores, accumulate in f32 for precision
result = tl.dot(A.to(tl.bfloat16), B.to(tl.bfloat16)).to(tl.float32)
```

This gives you the speed of tensor cores (8-16× faster than scalar math) with the precision
of float32 accumulation.

### Calling Triton kernels from JAX

We use `jax_triton`, a bridge library. It passes JAX arrays as pointers to the Triton kernel
with zero-copy sharing (same GPU memory):

```python
import jax_triton as jt

result = jt.triton_call(
    input_array,          # inputs (JAX arrays → pointers)
    kernel=my_kernel,     # the @triton.jit function
    out_shape=[           # shape/dtype of outputs (jax_triton allocates them)
        jax.ShapeDtypeStruct((128,), jnp.float32),
    ],
    grid=(1,),            # how many thread blocks to launch
    num_warps=4,          # threads per block = num_warps × 32
)
```

---

## 8. Prefill: Processing the Prompt

"Prefill" processes the entire input prompt at once — all positions computed in parallel.
It produces the KV cache that the decode kernel reads from.

In our system, prefill runs in **JAX** (`model.py: prefill_with_kv`), not in a custom
Triton kernel. This is because:

1. Prefill only runs **once** per generation (processing the prompt). Decode runs hundreds
   of times (one call per generated token). Optimizing decode has 100x more impact.
2. Prefill is **compute-bound** (large matrix multiplications), not memory-bound. JAX/XLA
   already handles compute-bound operations efficiently via cuDNN.
3. The prefill code in `model.py` is straightforward JAX: run all layers, save K/V from
   each layer into cache arrays, return logits + caches.

After prefill, the KV caches are packed into a flat buffer with `pack_kv_caches()`
(`kernels/fused_decode_nlayer.py`) for the Triton decode kernel to read.

---

## 9. KV Cache: Why We Don't Recompute Everything

When generating text, we produce tokens one at a time:

```
Prompt:    "The cat sat on the"
Generate:   → "m"     (position 5)
            → "a"     (position 6)
            → "t"     (position 7)
            ...
```

At position 6, the model needs to attend to positions 0-6. The Q/K/V projections for
positions 0-5 are **exactly the same** as they were when we processed the prompt. Only
position 6 is new.

Without caching, we'd recompute Q, K, V for ALL previous positions every time we generate
a token. For 64 generated tokens, that's 64 + 63 + 62 + ... + 1 = 2,080 position
computations instead of just 64.

**The KV cache** stores K and V for all previous positions. When generating position 6,
we only compute K and V for position 6, look up K and V for positions 0-5 from the cache,
and do the attention.

```
Cache layout: (n_heads, max_seq, d_head) = (2, 128, 32) in bf16
              = 2 × 128 × 32 × 2 bytes = 16 KB

After prefill:  positions 0-63 filled (from prompt)
After decode 1: position 64 filled
After decode 2: position 65 filled
...
```

The prefill kernel outputs the KV cache as a side effect. The decode kernel reads
from it and outputs new K/V vectors, which the JAX wrapper inserts at the right position.

---

## 10. The Decode Kernel: Processing One Token

The decode kernel (`kernels/fused_decode_nlayer.py: _fused_decode_nlayer`) processes
**one token** through all 24 layers in a single kernel launch. This is the simpler
single-SM version; section 14 covers the multi-SM version that we actually use.

Decode is fundamentally different from prefill:

| | Prefill | Decode |
|---|---|---|
| Tokens processed | 128 | 1 |
| Matmul shape | (128, 1024) @ (1024, 64) | (1, 1024) @ (1024, 64) |
| Attention | (128, 128) — compute-bound | (1, pos) — memory-bound |
| Bottleneck | Compute (matmuls) | Memory (loading 607MB of weights) |

### The layer loop

The kernel processes all 24 layers in a `tl.range` loop. All weights for all layers
are packed into one flat buffer, with offsets computed per layer:

```python
for layer in tl.range(N_LAYERS):
    w_base = layer * LAYER_W_SIZE   # offset into packed weight buffer
    kv_base = layer * LAYER_KV_SIZE # offset into packed KV cache
```

`tl.range` (not `tl.static_range`) means Triton compiles the loop body **once** and
reuses it. With 24 layers, `tl.static_range` would unroll into 24 copies, causing
10+ minute compilation and register spilling.

### Projections via tl.dot

At d=1024, the representation vector `h` has 1024 elements. Projecting it to a
64-dimensional head via element-wise `h[:, None] * W` would need a `(1024, 64)` tensor
in registers — 65,536 values, far exceeding the 255-register limit. Instead we use
`tl.dot`:

```python
Q = tl.dot(h_norm_2d, wq_tile).to(tl.float32).sum(axis=0)
```

`h_norm_2d` is `h_norm[None, :]` reshaped to `(1, 1024)`. `tl.dot` tiles the reduction
internally, never materializing the full outer product. This uses tensor cores despite
M=1 because we reshape h to be a 2D matrix.

### Online softmax attention over the KV cache

The KV cache can hold up to 512 positions. We can't load all 512 K vectors at once
(too much memory), so we tile in chunks of `KV_TILE=64` positions and use **online
softmax** to accumulate the result:

```python
m_i = tl.full((1,), value=-1e9, dtype=tl.float32)  # running max
l_i = tl.zeros((1,), dtype=tl.float32)              # running sum of exp
o_i = tl.zeros((D_HEAD,), dtype=tl.float32)         # running weighted output

for t in tl.range(0, MAX_SEQ, KV_TILE):
    tile_pos = t + tl.arange(0, KV_TILE)
    tile_mask = tile_pos <= pos    # only attend to positions 0..pos

    K_tile = tl.load(kv_in_ptr + ..., mask=tile_mask[:, None], other=0.0)
    V_tile = tl.load(kv_in_ptr + ..., mask=tile_mask[:, None], other=0.0)

    # Insert new K/V at current position
    K_tile = tl.where(tile_pos[:, None] == pos, K_new[None, :], K_tile)
    V_tile = tl.where(tile_pos[:, None] == pos, V_new[None, :], V_tile)

    s = tl.sum(Q[None, :] * K_tile, axis=1) * scale
    s = tl.where(tile_mask, s, -1e9)

    # Online softmax update
    m_ij = tl.max(s)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_new)     # rescale previous accumulation
    p = tl.exp(s - m_new)           # current tile's softmax weights
    l_i = l_i * alpha + tl.sum(p)
    o_i = o_i * alpha + tl.sum(p[:, None] * V_tile, axis=0)
    m_i = m_new

attn_out = o_i / l_i
```

The key insight: standard softmax needs the max across ALL positions before computing
any exponentials. Online softmax tracks a running max and rescales previous results
when a new max is found. This is the same trick FlashAttention uses — process tiles
sequentially without ever materializing the full attention matrix.

### SwiGLU FFN with tiled weights

The FFN has three weight matrices (gate, up, down) with d_ff=2816 hidden units.
We tile over the hidden dimension in chunks of `BLOCK_K=32`:

```python
for k in tl.range(0, D_FF, BLOCK_K):
    kk = k + tl.arange(0, BLOCK_K)
    gate = tl.dot(h_norm_2d, gate_w).to(tl.float32).sum(axis=0)
    up   = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
    act  = (gate * tl.sigmoid(gate)) * up    # SwiGLU activation
    ffn_accum += tl.dot(act_2d, down_w).to(tl.float32).sum(axis=0)
```

Each iteration processes 32 hidden units. The activation `gate * sigmoid(gate) * up`
is SwiGLU — a gated variant that consistently outperforms standard ReLU.

---

## 11. The Generation Loop

The complete generation pipeline (simplified from `generate.py`):

```python
def generate(params, config, prompt_ids, n_tokens, vocab_size):
    # 1. Prefill: JAX processes entire prompt, produces KV caches
    x = jnp.pad(prompt_ids, (0, ctx_len - len(prompt_ids)))
    logits, k_caches, v_caches = prefill_with_kv(params, config, x)

    # 2. Pack weights and KV caches for the Triton decode kernel
    w = prepare_decode_weights_nlayer(params, config, vocab_size, kv_splits=1)
    kv_packed = pack_kv_caches(k_caches, v_caches)

    # 3. First token from prefill logits
    token = jnp.argmax(logits[len(prompt_ids) - 1])

    # 4. Decode loop: one Triton kernel call per token
    for i in range(n_tokens):
        token, _, kv_packed = multi_sm_decode_nlayer(
            w, config, token, len(prompt_ids) + i, kv_packed, vocab_size, kv_splits=1)
        print(decode_fn([int(token)]), end='', flush=True)
```

**Step 1:** JAX prefill runs the prompt through all 24 layers, saving K/V at each layer.

**Step 2:** `prepare_decode_weights_nlayer` packs all weights into a single flat bf16
buffer. `pack_kv_caches` does the same for KV caches. This is done once before decoding.

**Step 3:** `argmax` picks the most probable next token (greedy decoding).

**Step 4:** Each iteration calls the Triton multi-SM decode kernel. The kernel returns
the next token (via in-kernel argmax) and the updated KV cache. The KV cache grows by
one entry per step — the kernel writes the new K/V at the current position.

---

## 12. Results and Why It's Faster

### The numbers (d=64, 1 layer)

```
Prompt: 64 tokens, Generate: 64 tokens

Triton fused:   740 tok/s  (86.5 ms total)
JAX baseline:   185 tok/s  (362.6 ms total)
Speedup:        4.0x
```

Both produce identical text, confirming the kernels are correct.

### Where does the speedup come from?

**1. Fewer HBM round-trips.** The prefill does one round-trip instead of ~15. The decode
kernel does one kernel launch instead of ~15. Over 64 decode steps, that's ~900 fewer
kernel launches.

**2. No kernel launch overhead.** Each GPU kernel launch has 5-10 µs of overhead. For JAX's
~15 kernels per forward pass × 64 decode steps = ~960 launches × ~7 µs = ~6.7 ms of pure
overhead. The Triton version has 65 launches (1 prefill + 64 decode) × ~7 µs = ~0.5 ms.

**3. Data stays in registers.** Intermediate activations (the `h` matrix, attention scores,
FFN hidden states) never leave registers. In JAX, each intermediate is written to HBM
(~400 cycles per read/write) and read back. In the fused kernel, it's ~1 cycle per access.

**4. No redundant computation.** The KV cache means the decode kernel only processes 1 token
instead of recomputing all previous tokens. (JAX doesn't implement KV caching in our
baseline, so it processes the full growing sequence each time.)

### What it's NOT

- **Not better math.** Both do the same multiplications.
- **Not parallel tokens.** Both process one token at a time during decode.
- **Not quantization.** Both use bf16 for matmuls.

The speedup is purely from keeping data close to the compute units.

This was the starting point. Everything that follows is about scaling this approach to a
real model and dealing with the problems that emerge at larger scale.

---

## 13. Scaling the Model

The fused-kernel approach works brilliantly at d=64 because the entire model fits in
registers. But what happens when we scale up?

### The scaling journey

```
d=64,  1L,   66K params:    740 tok/s  (4.0x vs JAX)
d=128, 2L,  674K params:   2504 tok/s  (14.0x — in-kernel KV cache updates)
d=256, 4L, 5.3M params:   1396 tok/s  (15.1x — fused N-layer decode)
d=512, 8L,  30M params:    287 tok/s  (single SM, 2% bandwidth utilization)
                          4777 tok/s  (persistent kernel, final)
```

Each scale-up introduced new problems. Here's what happened and how we solved them.

### d=128: The model no longer fits in one block

At d=128, the hidden state `h` is (128, 128) = 64 KB — it can't coexist with the attention
scores matrix (also 128 × 128 = 64 KB) in the 130 KB register file. Solution: **multi-block
prefill**. Tile the sequence dimension into BLOCK_SEQ=32 row blocks. Each of 4 thread blocks
handles 32 positions. h_block is (32, 128) = 16 KB. Scores are (32, 128) = 16 KB. Fits.

The cost is that blocks can't communicate within a single kernel. So projections (which write
K/V to HBM) must be a separate kernel launch from attention (which reads K/V from HBM).
Three kernels per layer instead of one.

The decode kernel stays fused — with M=1, all tensors are 1D and tiny.

The three biggest optimizations at this scale:

1. **In-kernel KV cache updates (3.6x win).** Instead of outputting K_new/V_new and having
   JAX do `.at[pos].set()` (which copies the entire cache per step), the kernel writes the
   updated cache directly to an output buffer. This eliminated the biggest bottleneck.

2. **Precomputed bf16 weights (57% win).** `.astype(bf16)` inside the decode loop created
   a new JAX array per call — 28 weights × 63 steps = 1764 allocations. Converting once
   before the loop: free.

3. **Fused multi-layer decode (35% win).** Instead of one kernel call per layer, a single
   kernel processes all layers with h staying in registers between them.

### d=256: Fused N-layer decode with packed buffers

At 4 layers, passing individual weight pointers becomes unwieldy (50+ arguments). Solution:
**packed weight buffer** — concatenate all per-layer weights into one bf16 buffer. The kernel
computes offsets from the layer index. Same for KV caches: all layers packed into one flat
buffer.

FlashAttention becomes necessary at context > 256. The full scores matrix would be (16, 512)
= 32 KB per head, too large alongside K and V. Tiled KV with online softmax (KV_TILE=64)
keeps peak memory at ~20 KB per tile regardless of context length.

### d=512: Register pressure and the single-SM bottleneck

At d=512, two critical problems emerge:

**Register overflow.** Element-wise projections `tl.sum(h[:, None] * W, axis=0)` create a
(512, 32) intermediate = 128 registers per thread. The product with h needs another 128.
Total: 260 > 255 limit. Fix: `tl.dot` tiles the reduction internally, never materializing
the full intermediate.

**Single-SM bottleneck.** The fused kernel runs on grid=(1,) — one thread block on one SM.
The RTX 4080 Super has 80 SMs. At d=512 with 8 layers, the kernel takes 3.48 ms per token.
Profiling showed: kernel=93% of time, host=3%. The GPU is 98% idle — not because it's
waiting for data, but because 79 of 80 SMs have no work.

This is the fundamental challenge that the next three sections address.

---

## 14. Multi-SM Decode: Using the Whole GPU

### The problem

At d=512, a single thread block runs the entire decode step. It processes 16 attention heads
sequentially, then the full FFN, then layer norm, for each of 8 layers. 79 out of 80 SMs
sit idle. Bandwidth utilization: 2%.

### The solution: one block per attention head

Launch grid=(N_HEADS,) = grid=(16,). Each of the 16 blocks handles one attention head's
Q/K/V projections, attention, and O-projection. Then all 16 blocks split the FFN work
(each handles D_FF/16 = 128 columns of the up-projection).

```
Block 0:  head 0 attention + FFN columns 0-127
Block 1:  head 1 attention + FFN columns 128-255
...
Block 15: head 15 attention + FFN columns 1920-2047
```

### Cross-block synchronization with atomic barriers

The problem: blocks can't communicate within a kernel. After attention, all blocks must
wait for all partial results before anyone can compute the next layer's input.

Solution: **atomic barriers** using GPU-scope atomics in L2 cache.

```python
# Barrier implementation (simplified):
# Each block arrives by atomically incrementing a counter
old = tl.atomic_add(barrier_ptr, 1, sem='release', scope='gpu')
if old == N_BLOCKS - 1:
    # Last block to arrive: set done flag
    tl.atomic_xchg(done_ptr, 1, sem='release', scope='gpu')
# All blocks wait for done flag
while tl.atomic_add(done_ptr, 0, sem='acquire', scope='gpu') == 0:
    pass  # spin-wait
```

Two barriers per layer: one after attention (before FFN), one after FFN (before next layer).
With 8 layers + 1 final: 17 barriers total.

**Key insight: redundant computation is cheaper than synchronization.** All 16 blocks
independently compute LayerNorm and reduce the 16 partial FFN results. This is redundant
(each block reads 32 KB from L2) but costs only ~1 µs. The alternative — one block computes
and broadcasts — would need an additional barrier (~5 µs) plus a serial bottleneck.

### KV-split parallelism (FlashDecoding)

With 16 blocks on 80 SMs, utilization is only 20%. **KV-split parallelism** doubles the
grid to 32 by having 2 blocks per attention head, each handling half the KV cache tiles.

Each block computes a partial online softmax (partial O-projection + running max + running
sum). After the barrier, all blocks merge the partials using the log-sum-exp trick:

$$h_{head} = \frac{\sum_s o_s \cdot l_s \cdot e^{m_s - m_{max}}}{\sum_s l_s \cdot e^{m_s - m_{max}}}$$

where $o_s$ is the normalized partial output, $l_s$ is the partial sum of exponentials, and
$m_s$ is the partial maximum. This is mathematically exact — no approximation.

### Split barrier optimization

The standard barrier has all blocks polling the same counter address while others are still
arriving. This causes **L2 cache line thrashing** — arrivals invalidate the line that pollers
are reading.

Fix: separate the arrival counter and done-flag on different cache lines. Arrivals write to
`counter[]`, the last-arriving block sets `done[]`, all blocks poll `done[]`. Result: +5%
kernel speedup.

### Results

```
Single-SM (grid=1):        287 tok/s  (3.48 ms/tok, 2% BW utilization)
Multi-SM (grid=16):       1734 tok/s  (0.58 ms/tok, 14% BW, 6.0x)
+ KV splits (grid=32):   1851 tok/s  (0.54 ms/tok, 15% BW, +7%)
+ Split barrier:          1937 tok/s  (0.52 ms/tok, 16% BW, +5%)
Without GPU→CPU sync:     3108 tok/s  (0.32 ms/tok — pure GPU speed)
```

The 6.0x speedup from multi-SM is the single largest improvement at d=512.
GPU→CPU sync (`int()` per token) costs 34% of total time — addressed in section 16.

---

## 15. What Didn't Work

Not everything we tried improved performance. Documenting failures is as important as
documenting successes — they reveal what the actual bottlenecks are.

**GQA (Grouped Query Attention).** 4 KV heads instead of 16 Q heads. KV cache shrinks from
8.4 MB to 2.1 MB per sequence. But decode speed was unchanged — the kernel is
barrier-limited, not memory-limited. The data already fits in L2 at high bandwidth. GQA
helps batched inference (where KV cache scales with batch size) but not single-sequence.

**Parallel residual (attn || FFN).** Compute attention and FFN in parallel, reducing
barriers from 17 to 9. Result: +1.3% speedup. Why? Barrier cost is dominated by straggler
*variance* (proportional to total work per step), not fixed overhead. Halving barriers
saves ~8 µs of fixed overhead, but the ~80 µs of straggler time stays constant.

**num_warps sweep.** 2, 4, 8 warps all within 5%. The kernel is not warp-limited.

**Speculative decoding.** With fused kernels, both draft (~2500 tok/s) and target (~1400
tok/s) are fast. The speed ratio is only ~2x, not the ~10x needed. Acceptance rate was
36-51% with a 1-layer draft — you need 80%+ to break even at this ratio.

---

## 16. Key Lessons

### Fundamentals

**1. Memory bandwidth is the bottleneck, not compute.** Modern GPUs do trillions of ops/s
but read ~2 TB/s from HBM. For small models, math finishes instantly — the GPU waits for data.

**2. Kernel fusion is the highest-leverage optimization.** Eliminating HBM round-trips
gives the biggest speedup. In-kernel KV cache updates alone gave 3.6x.

**3. bf16 + f32 accumulation is the sweet spot.** Cast to bf16 for tensor cores, accumulate
in f32. FP8 was slower (register pressure from casts outweighed gains).

**4. Dynamic loops prevent register spilling.** `tl.range` emits a real loop; `tl.static_range`
unrolls, requiring registers for all tiles simultaneously.

### Scaling

**5. BLOCK_SEQ scales inversely with d_model.** Register file is fixed. d=128: BLOCK_SEQ=32.
d=256: BLOCK_SEQ=16. d=512: BLOCK_SEQ=8. The h_block stays at ~16 KB.

**6. Use tl.dot for projections at d >= 512.** Element-wise `h[:, None] * W` materializes the
full (d, d_head) intermediate. tl.dot tiles internally, avoiding register overflow.

**7. The bottleneck shifts with model size.** At d=64, host dispatch dominates (Python overhead).
At d=512, the GPU kernel dominates (93% of step time). Optimizations that help at one scale
may be irrelevant at another. Always profile first.

### Multi-SM and parallelism

**8. Multi-SM with atomic barriers unlocks the full GPU.** Going from grid=(1,) to grid=(32,)
gave 6.8x speedup at d=512. The technique: one block per attention head, all blocks split
the FFN, atomic barriers for cross-block sync.

**9. Redundant computation beats synchronization.** All blocks independently compute LayerNorm
(~1 µs from L2) rather than one block computing and broadcasting (needs a ~5 µs barrier).

**10. GPU→CPU sync is the #1 bottleneck at 30M params.** `int()` per token costs 30-60% of
wall time. Persistent kernels and pipelining eliminate this entirely.

### Batched inference

**11. Weight amortization requires careful loop structure.** Outer k-loop / inner b-loop loads
weights once per tile. The naive fused approach loads them B times.

**12. Shared buffers need double-buffering or phase separation.** Any buffer that is both read
and written by all blocks within a barrier-delimited phase needs either double-buffering or a
separate buffer for the write phase.

### What doesn't help

**13. Reducing barrier count gives minimal speedup.** Straggler variance (proportional to total
work) dominates fixed barrier overhead. Halving barriers from 17 to 9 gave only 1.3%.

**14. GQA doesn't help when barrier-limited.** If data already fits in L2, reducing it further
doesn't matter. GQA helps when KV cache is the memory bottleneck (batched, long-context).

---

## Running It Yourself

```bash
# generate text
uv run generate.py --prompt "Once upon a time" --temp 0.7 --top-p 0.95

# profile decode kernel
uv run profile_kernels.py
```

---

## Part III: Production Scale (d=768+, 12-24 layers)

---

## 17. GQA: Grouped Query Attention

Standard multi-head attention uses N_HEADS separate K/V heads. GQA (Grouped Query
Attention) shares K/V across groups of Q heads, reducing KV cache size.

With d=768, 24 Q heads, 6 KV heads: each KV head is shared by 4 Q heads (GQA_GROUP=4).
KV cache shrinks 4x: from 24 × ctx × d_head to 6 × ctx × d_head per layer.

**Kernel change:** In the attention loop, map Q head to KV head via `kv_head = head_id // GQA_GROUP`.
Load K/V from `kv_head`'s cache, not `head_id`'s. Weights follow the same pattern:
Q/O weight matrices are (D_MODEL, N_HEADS × D_HEAD), K/V are (D_MODEL, N_KV_HEADS × D_HEAD).

**Prefill kernel change:** The projection kernel has two loops — Q over N_HEADS, K/V over
N_KV_HEADS. The attention kernel maps each Q head to its KV group. KV cache output shape
is (N_KV_HEADS, SEQ, D_HEAD) instead of (N_HEADS, SEQ, D_HEAD).

**Why GQA matters:** At d=768 with B=8 batched inference, GQA reduces KV cache from
37.6 MB to 9.4 MB per step — fits entirely in L2 cache (~64 MB on RTX 4080 Super).

---

## 18. Non-Power-of-2 D_MODEL (D_BLOCK Padding)

Triton's `tl.arange` requires power-of-2 dimensions. For d_model=768, we use D_BLOCK=1024
(next power of 2) with masking:

```python
d = tl.arange(0, D_BLOCK)      # 0..1023
d_mask = d < D_MODEL            # True for 0..767, False for 768..1023

# All loads use the mask:
h = tl.load(ptr + d, mask=d_mask, other=0.0)

# All stores use the mask:
tl.store(ptr + d, h, mask=d_mask)
```

The padding (positions 768-1023) is always zero. `tl.dot` with zero-padded operands
produces correct results: $0 \times \text{anything} = 0$.

**Cost:** D_BLOCK=1024 wastes 33% of register/shared memory capacity compared to
D_BLOCK=768 (if it were possible). This is why `num_stages=2` (software pipelining)
fails at d=768 — the double-buffered tiles at (1024, 32) exceed shared memory.

---

## 19. L2 Cache Hints and Bandwidth Optimization

At d=768, the weight buffer (162 MB) exceeds the L2 cache (~64 MB). Every decode step
re-fetches all weights from HBM — the "terminal bandwidth bottleneck."

**Eviction policies** tell the GPU which data to keep and which to evict first:

```python
# KV cache (4.7 MB, fits in L2, reused across steps):
K_tile = tl.load(kv_ptr + ..., eviction_policy='evict_last')

# Output projection (6.3 MB, single-use per step):
out_w = tl.load(out_ptr + ..., eviction_policy='evict_first')
```

`evict_last` keeps KV cache hot in L2 between decode steps. `evict_first` on output
projection prevents it from evicting KV data. Combined effect: ~3% throughput improvement.

**Barrier merging:** The persistent kernel had 2 barriers per step for output (one for
output projection sync, one for step-sync). By having the last-arriving block do the
argmax reduction inline before signaling, we merge them into 1 barrier. Saves ~2%.

**What was tried but didn't help:**
- KV_SPLITS=1 (24 blocks): 15% slower — insufficient parallelism
- KV_SPLITS=4 (96 blocks): compilation too slow, likely worse from contention
- num_stages=2: shared memory overflow at D_BLOCK=1024

---

## 20. Tensor Core Batched Projections

With batched inference, each block processes B sequences. Previously, Q/K/V projections
looped over sequences:

```python
# OLD: B sequential (1, D_BLOCK) × (D_BLOCK, D_HEAD) dots
wq = tl.load(...)  # weight loaded once
for b in range(BATCH_SIZE):
    h_norm = tl.load(h_norm_ptr + b * D_BLOCK + d)
    Q = tl.dot(h_norm[None, :], wq).sum(axis=0)   # element-wise, M=1
```

The optimization: stack all B hidden states into a 2D matrix and do one dot product:

```python
# NEW: single (B, D_BLOCK) × (D_BLOCK, D_HEAD) matmul
b_range = tl.arange(0, BATCH_SIZE)
h_batch = tl.load(h_norm_ptr + b_range[:, None] * D_BLOCK + d[None, :])
Q_batch = tl.dot(h_batch, wq)  # (B, D_HEAD) — uses tensor cores when B >= 16
```

At B>=16, the (16, 1024) × (1024, 32) matmul activates tensor cores (Ada Lovelace FP16
tensor cores need M >= 16). At B=4, tensor cores don't activate but the 2D load/store
pattern still reduces loop overhead.

**Result:** +8-12% across all batch sizes. O projection not batched because the output
(B, D_BLOCK) = (16, 1024) × 4B = 64KB overflows shared memory alongside the weight matrix.

---

## 21. Sampling and Decoding Strategies

Greedy decoding (always pick the highest-probability token) is simple but causes
**repetition collapse**: once the model assigns slightly higher probability to a phrase
it just produced, it locks into a loop. With our 242M param model, greedy decoding gives
a 4-gram diversity score of just 0.22 — meaning 78% of 4-grams are repeated.

Three techniques fix this, each attacking a different part of the problem:

### Temperature

Temperature scales the logits before softmax. Lower temperature sharpens the distribution
(more deterministic), higher temperature flattens it (more random):

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

At $T=0$ this is greedy (argmax). At $T=1$ this samples from the model's actual
distribution. At $T>1$ the distribution flattens and rare tokens become more likely.

Concretely: if the model outputs logits $[5.0, 4.8, 2.0, 1.0]$ for four tokens:
- $T=0.5$: divides by 0.5 → $[10.0, 9.6, 4.0, 2.0]$ → probabilities $[0.60, 0.40, 0.003, 0.0001]$ (almost always picks token 0 or 1)
- $T=1.0$: unchanged → $[0.44, 0.36, 0.022, 0.008]$ (token 2 has a 2% chance)
- $T=1.5$: divides by 1.5 → $[3.33, 3.20, 1.33, 0.67]$ → probabilities $[0.35, 0.31, 0.05, 0.02]$ (more spread out)

The sweet spot for our model is $T \approx 0.7$ — enough randomness to avoid loops,
not so much that it produces gibberish.

### Top-p (nucleus sampling)

Instead of sampling from the full vocabulary, sort tokens by probability and only keep
the smallest set whose cumulative probability exceeds a threshold $p$:

1. Sort tokens by descending probability
2. Compute cumulative sum
3. Keep tokens until cumsum $\ge p$
4. Renormalize and sample from this subset

With $p=0.95$, this dynamically adjusts the number of candidates: when the model is
confident (one token has 90% probability), only 1-2 tokens are considered. When unsure,
hundreds might be included. This is better than a fixed top-k because it adapts to the
model's certainty at each position.

```python
sorted_idx = np.argsort(-probs)
sorted_probs = probs[sorted_idx]
cumsum = np.cumsum(sorted_probs)
cutoff = np.searchsorted(cumsum, top_p) + 1
keep_idx = sorted_idx[:cutoff]
```

### Repetition penalty

Repetition penalty directly discourages the model from producing tokens it already
generated. Before computing softmax, we modify the logits of previously-seen tokens:

$$z_i' = \begin{cases} z_i / \alpha & \text{if } z_i > 0 \text{ and token } i \text{ was already generated} \\ z_i \times \alpha & \text{if } z_i < 0 \text{ and token } i \text{ was already generated} \end{cases}$$

With $\alpha = 1.2$: a positive logit of 5.0 becomes 4.17 (less likely), and a negative
logit of -2.0 becomes -2.4 (even less likely). This pushes the model away from repeating
itself without completely banning any token.

### What we found

We swept 18 parameter combinations across 6 prompt types (story, knowledge, code,
creative, explanation, dialogue). The metric is 4-gram diversity: fraction of unique
4-grams out of total 4-grams (1.0 = no repetition).

```
Setting                         Avg diversity
greedy                          0.221  (78% repetition)
temp=0.7                        0.825
temp=0.7 top_p=0.95            0.534
temp=0.7 top_p=0.95 rep=1.2    0.987  ← best balance
temp=0.8 top_p=0.95 rep=1.2    0.994
temp=0.8 top_p=0.95 rep=1.3    0.999  (too aggressive — forces odd word choices)
```

Key findings:
- **Repetition penalty is the biggest single improvement.** Going from rep=1.0 to 1.2
  with temp=0.7/top_p=0.95 takes diversity from 0.53 to 0.99.
- **Top-p alone hurts** at moderate temperatures. With temp=0.7, adding top_p=0.95
  actually reduced diversity from 0.83 to 0.53 because it cuts the tail that would have
  introduced variety. Top-p works best combined with repetition penalty.
- **temp=0.7 is more coherent than 0.8+.** Higher temperatures occasionally produce
  gibberish (hallucinated citations, garbled text). The model at 242M params doesn't
  have enough capacity to stay coherent at high temperature.
- **Code is the hardest domain.** Even with optimal settings, code output has correct
  Python structure but incorrect logic. This is expected at 242M params with only 18%
  code in the training mix.

Usage:
```bash
uv run generate.py --prompt "your text" --temp 0.7 --top-p 0.95 --rep-penalty 1.2
```

---

---

### Files

```
Inference:
  kernels/multi_sm_decode.py         — fused multi-SM decode (all 24 layers in one kernel)
  kernels/fused_decode_nlayer.py     — weight/KV packing utilities
  generate.py                        — streaming text generation CLI
  profile_kernels.py                 — decode kernel profiling
```

For the training pipeline (data preparation, model architecture, training loop),
see [`training_explained.md`](training_explained.md).
