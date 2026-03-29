# EGGROLL Transformer — Agent Program

*You are a GPU kernel engineer. Your job: build the fastest possible inference for this
small transformer using custom Triton kernels. The model is trained with standard backprop.
The focus is on learning to write high-performance GPU kernels for transformer inference.*

**Background:** This project started as an ES training experiment (EGGROLL). That work
produced a fused Triton kernel that runs the entire transformer forward pass in one kernel
call — a technique that transfers directly to inference optimization. The ES training code
and history are preserved but the focus has shifted to inference kernel engineering.

---

## Your Mission

1. Train the model with backprop (already done: `uv run train_backprop.py`, val_loss=1.84)
2. Write a fused Triton inference kernel that generates text token-by-token
3. Benchmark against JAX/XLA baseline inference
4. Optimize: fuse operations, minimize memory bandwidth, maximize throughput

The goal is to learn kernel-writing skills on a small model where iteration is fast,
then apply those skills to larger models.

---

## Technology Stack

- **Triton** (OpenAI): Python-like GPU kernel language that compiles to PTX. Handles tiling,
  memory coalescing, and register allocation automatically. Used for all production kernels.
- **JAX**: ML framework with XLA compiler. Used for model definition, training, and baseline.
- **jax-triton**: Bridge that calls Triton kernels from JAX with zero-copy tensor sharing.
- **CUDA C++ kernel** in `kernels/cuda/`: experimental, working but 14x slower than Triton
  (no tensor cores). Kept as infrastructure for learning, not production.

---

## Model Architecture

```
Decoder-only transformer (character-level Shakespeare)
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128, vocab_size: 65
Parameters: 66,368
```

Weights: token_emb (65,64), pos_emb (128,64), Q/K/V/O (64,64 each), FFN up (64,256),
FFN down (256,64), LN scales/biases, output_proj (64,65).

---

## Existing Kernel Infrastructure

### `kernels/fused_transformer_ce.py` — ES Training Kernel (reference)

Fused Triton kernel that processes the ENTIRE forward pass in one kernel call:
Embedding → LayerNorm → Multi-Head Attention → FFN → LayerNorm → Output → CE Loss

Key techniques (transfer to inference):
- **All weights in registers.** Each weight matrix (max 64×256 = 16K bf16 values = 32KB)
  fits in register file. No HBM round-trips between layers.
- **bf16 tensor core matmuls with f32 accumulation.** `tl.dot(A.bf16, B.bf16).f32` uses
  hardware tensor cores for 8-16× throughput vs scalar fp32.
- **K-tiled FFN loop.** The (128,256) FFN is tiled into 8 blocks of (128,32) using
  `tl.range(0, 256, 32)` with dynamic range to prevent register pressure from unrolling.
- **Causal attention in-register.** Full (128,128) attention matrix computed and softmaxed
  without going through HBM. Works because d_head=32 is small.
- **num_warps=4 (128 threads/block).** Optimal for this model — fewer warps = more
  registers/thread. 255 regs/thread, 0 bytes spill.

### `kernels/fused_transformer_ce_dual.py` — Dual-Number Kernel (reference)

Same architecture but tracks both primal values AND tangent (forward-mode AD) through
every operation. Shows how to propagate auxiliary data through the fused kernel.

### `kernels/cuda/` — CUDA C++ Kernel (reference)

Raw CUDA implementation with shared memory staging. Correct but 14× slower than Triton
(scalar FP32 without WMMA tensor cores). JAX custom_call binding works.

---

## Inference Kernel Plan

### Phase 1: Prefill Kernel

The "prefill" phase processes the full input prompt in parallel (all positions at once).
This is structurally identical to the ES kernel's forward pass, minus the perturbation
and CE loss.

**Start from `fused_transformer_ce.py` and:**
1. Remove all perturbation code (sigma, vecs, rank-1 corrections)
2. Remove CE loss computation
3. Output the logits for the LAST position only (or all positions for flexibility)
4. Output the KV cache for use in the decode phase

**Input:** token IDs (seq_len,), all weight matrices
**Output:** logits (vocab_size,) for next-token prediction, KV cache

**Target:** Single kernel call for full prefill. Benchmark against JAX baseline
(`transformer_forward` in model.py).

### Phase 2: Decode Kernel (single-token step)

The "decode" phase processes ONE new token at a time, using cached K/V from previous steps.

**Key differences from prefill:**
- Input is 1 position, not 128
- Attention uses KV cache: Q is (1, d_head), K/V are (seq_so_far, d_head)
- The (1, seq) attention scores → softmax → weighted sum of V
- Much lower compute intensity (dominated by memory reads of KV cache)

**Design:**
1. Load the new token's embedding
2. LayerNorm, Q/K/V projections for the new position
3. Append new K/V to cache
4. Attention: Q (1, d_head) @ K_cache^T (d_head, seq) → scores (1, seq)
5. Softmax, weighted V sum
6. O projection, residual, LayerNorm, FFN, output projection
7. Output: logits (vocab,), updated KV cache

### Phase 3: Autoregressive Generation Loop

Combine prefill + decode into a complete text generation function:
1. Prefill the prompt
2. Sample next token from logits
3. Decode step with the new token
4. Repeat until done

Benchmark: tokens/second for generation of 128 tokens with 64-token prompt.
Compare against JAX baseline (model.py `transformer_forward` called in a loop).

### Phase 4: Optimizations

Once the basic kernels work, optimize:
- **Quantization**: INT8 or FP8 weights for faster matmuls and lower memory
- **FlashAttention-style tiling** for decode (if sequence gets long)
- **Speculative decoding**: generate multiple candidate tokens in parallel
- **Persistent kernel**: keep the kernel running and feed tokens via shared memory
- **Batched decode**: process multiple sequences simultaneously

---

## Profiling Commands

```bash
uv run train_backprop.py             # train the model (4s)
uv run profile_triton.py             # profile existing kernel
uv run benchmark.py                  # current EGGROLL vs backprop comparison
# After writing inference kernel:
uv run inference_benchmark.py        # inference speed comparison (you'll write this)
```

---

## Files

```
program.md                          — this file (read first)
README.md                           — project overview
model.py                            — JAX transformer model (inference baseline)
train_backprop.py                   — standard backprop training (use this to get weights)
train_eggroll.py                    — EGGROLL ES training (reference, not the focus now)
data.py                             — char-level Shakespeare dataset
kernels/fused_transformer_ce.py     — fused Triton forward kernel (START HERE for prefill)
kernels/fused_transformer_ce_dual.py — dual-number kernel (reference for auxiliary data propagation)
kernels/cuda/                       — CUDA C++ kernel (reference for raw CUDA patterns)
validate_kernel.py                  — kernel correctness validation pattern
profile_triton.py                   — kernel time profiler
benchmark.py                        — speed comparison benchmark
```

---

## Architecture Details

```
Model: decoder-only transformer
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128, vocab_size: 65 (character-level Shakespeare)
Parameters: 66,368

Weight shapes:
  token_emb:     (65, 64)    — 4,160 params
  pos_emb:       (128, 64)   — 8,192 params
  layer0.attn.q: (64, 64)    — 4,096 params
  layer0.attn.k: (64, 64)    — 4,096 params
  layer0.attn.v: (64, 64)    — 4,096 params
  layer0.attn.o: (64, 64)    — 4,096 params
  layer0.ln1:    (64,) + (64,) — 128 params
  layer0.ffn.up: (64, 256)   — 16,384 params
  layer0.ffn.up_bias: (256,) — 256 params
  layer0.ffn.down: (256, 64) — 16,384 params
  layer0.ffn.down_bias: (64,) — 64 params
  layer0.ln2:    (64,) + (64,) — 128 params
  ln_final:      (64,) + (64,) — 128 params
  output_proj:   (64, 65)    — 4,160 params
```

---

## Kernel Engineering Lessons (from EGGROLL sessions)

These lessons were learned the hard way during kernel optimization. Apply them:

1. **num_warps=4 is optimal for this model.** Fewer warps = more registers per thread.
   num_warps=2 is slower (poor occupancy), num_warps=8 is slower (thread overhead).

2. **Dynamic loops (`tl.range`) prevent register-pressure blowup from unrolling.**
   The FFN K-tiling loop MUST use `tl.range`, not `tl.static_range`. Static unrolling
   of 8 iterations causes 340 bytes of register spilling.

3. **bf16 matmuls with f32 accumulation are the sweet spot.** FP8 was tried and was slower
   (register pressure from casts outweighed tensor core gains). fp16 out_dtype didn't help.

4. **The entire model fits in registers.** Total weight storage: ~66K params × 2 bytes =
   132KB. With 128 threads/block and 255 regs/thread, we have 128 × 255 × 4 = 131KB of
   register file. Tight but fits (weights are loaded on-demand, not all live simultaneously).

5. **Attention scores (128×128) in registers are the bottleneck.** This is 16K f32 values
   = 64KB. It works at num_warps=4 but is the largest live tensor.

6. **Shared memory is NOT used in the current kernel.** All data flows through registers.
   For the decode kernel with KV cache, shared memory will be needed.

7. **maxnreg doesn't help** (tried via jax_triton monkey-patch). The kernel is compute-bound,
   not occupancy-bound. Forcing fewer registers just causes spilling.

8. **Triton compile timeout at BATCH_TILE=2** — don't try to batch multiple sequences
   per block. One sequence per block is optimal for this model size.

---

## EGGROLL Training History (archived)

The ES training experiments are preserved for reference. Key results:
- **Best EGGROLL**: val_loss=2.38 (3-seed avg 2.376), 272s, HALF_POP=7168
- **Backprop+Adam**: val_loss=1.84, 4.1s
- 30+ experiments confirmed population size is the only quality lever
- Dual-number kernel (fused forward-mode AD): technically correct, loses to finite
  differences at bf16 precision due to accumulated tangent rounding errors

See git history for full experiment logs.
