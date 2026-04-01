# Fused Inference

Custom Triton kernels for transformer inference on a single **RTX 4080 Super (16GB VRAM, 836 GB/s bandwidth)**. The entire decode step — embedding, attention, FFN, output projection — runs in a **single GPU kernel call** across all layers.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — a coding agent is pointed at `program.md` repeatedly to make progress autonomously. The human only steers which direction to work on next; the agent handles implementation, debugging, benchmarking, and documentation.

See [`repo_explained_from_zero.md`](repo_explained_from_zero.md) for a ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all the techniques used in this project.

## Performance

```
RTX 4080 Super | 3 model scales tested

XXL (d=768, h=24, l=12, ctx=512, GQA 6 KV heads, 81.1M params, ppl=2.60):
  Multi-SM sync:       1006 tok/s  (0.99 ms/tok, 20% BW util)
  Pipelined:           1463 tok/s  (0.68 ms/tok)
  Persistent B=4:      2225 tok/s  (1.80 ms/step)
  Persistent B=8:      2460 tok/s  (3.25 ms/step)
  Weight buffer:       162 MB (exceeds L2 cache — truly HBM-bound)

XL (d=512, h=16, l=8, ctx=512, GQA 4 KV heads, 26.5M params, ppl=2.96):
  Persistent kernel:   5129 tok/s  (0.195 ms/tok, single launch)
  Persistent B=4:      7351 tok/s  (0.544 ms/step)
  Persistent B=8:      7862 tok/s  (1.018 ms/step)
  Weight buffer:       53 MB, KV cache: 2.1 MB/seq

XL-ctx (d=512, h=16, l=8, ctx=2048, GQA 4 KV heads, 27.3M params, ppl=2.84):
  Persistent kernel:   3133 tok/s  (0.319 ms/tok)
  Persistent B=4:      4560 tok/s  (0.877 ms/step)
  KV cache:            8.4 MB/seq (4x vs ctx=512)
```

### Optimization history

```
Phase A1: d=64, 1L  — fused prefill+decode              740 tok/s
Phase A2: d=128, 2L — in-kernel KV updates (3.6x win)  2504 tok/s
Phase A3: d=256, 4L — fused N-layer decode              1396 tok/s
Phase A4: d=512, 8L — tl.dot projections, tiled KV      287 tok/s (2% BW util)
Phase A5: multi-SM   — grid=(32,) + atomic barriers     1937 tok/s (6.8x, 16% BW)
Phase A6: batched     — shared weights, B sequences     3821 tok/s at B=4 (1.97x)
Phase A7: persistent  — single launch, no host sync     4777 tok/s (2.56x vs sync)
Phase A8: persist+KV  — in-place KV, pos-only store     5129 tok/s (+7.1%)
Phase A9: persist-B   — persistent batched decode       7351 tok/s at B=4 (+10%)
Phase B1: d=768, 12L — D_BLOCK padding, BLOCK_K=16     1006 tok/s (20% BW, 81M params)
Phase B2: ctx=2048   — 4x context, 32 KV tiles         3133 tok/s persistent
```

## Quick Start

```bash
# train the model (d=512, 8L, ~26M params, ~4 hours)
uv run train.py --d-model 512 --n-heads 16 --n-kv-heads 4 --n-layers 8 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16

# train larger model (d=768, 12L, ~81M params, ~11 hours)
uv run train.py --d-model 768 --n-heads 24 --n-kv-heads 6 --n-layers 12 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16

# profile kernels (primary benchmark — run after any kernel change)
uv run profile_kernels.py

# quick inference demo
uv run inference_benchmark.py
```

## How It Works

### Prefill (process full prompt)

Multi-block approach with 3 Triton kernels per layer (projections, attention, FFN). FlashAttention (tiled KV + online softmax) for context > 256. BLOCK_SEQ scales inversely with d_model to fit in registers.

### Decode (generate tokens one at a time)

**Multi-SM kernel** distributes work across all GPU SMs:
- **grid=(N_HEADS × KV_SPLITS,)** — each block handles one attention head (or KV split), all blocks split the FFN
- **Atomic barriers** — release/acquire semantics for cross-block synchronization
- **KV-split parallelism** (FlashDecoding-style) — splits KV cache across 2 blocks per head, merges with online softmax correction
- **Split barrier** — separate counter/done-flag cache lines to reduce L2 contention
- **D_BLOCK padding** — non-power-of-2 d_model (e.g., 768) supported via masking to next power of 2

**Persistent kernel** runs all decode steps in a single kernel launch:
- Tokens stay on GPU — no per-step host sync
- Fresh barrier slots per step
- In-kernel next-token feedback (block 0 writes, step-sync barrier broadcasts)
- In-place KV with pos-only store (avoids 4MB tile copy per step)

**Batched kernel** processes B sequences per launch:
- Weight loads amortized across batch (weight-amortized FFN: outer k-loop / inner b-loop)
- Double-buffered h_buf to avoid cross-block read-write races
- Separate ffn_buf to avoid merge/FFN phase conflicts
- Persistent batched variant: single launch for B × N_STEPS

### Key techniques

- **Packed weight buffer** — all per-layer weights in one bf16 buffer, kernel computes offsets
- **Packed KV caches** — all layers' caches in one flat buffer, no per-step pack/unpack
- **In-kernel KV cache updates** — kernel writes updated caches directly (biggest single win: 3.6x)
- **tl.dot projections** — avoids register overflow at d=512+ by tiling internally
- **Tiled KV decode** — online softmax with KV_TILE=64 for large context
- **Gradient checkpointing** — per-layer remat for training memory efficiency

### What was tried but didn't help

- **GQA** — no single-sequence speedup (barrier-limited, not memory-limited). Helps batched inference.
- **Parallel residual** — 9 barriers instead of 17, but only 1.3% speedup (straggler variance dominates)
- **num_warps sweep** — 2/4/8 all within 5%
- **Speculative decoding** — acceptance rate too low at this scale (36-51%), draft/target speed ratio only 2x
- **Shared memory workspace** — all hot buffers already in registers; cross-block buffers can't use shared mem

## Architecture

```
Decoder-only transformer (d_head=32 for all sizes)

Small:  d=64,  h=2,  l=1, vocab=1024,  189K params  (Shakespeare)
Medium: d=128, h=4,  l=2, vocab=1024,  674K params  (Shakespeare)
Large:  d=256, h=8,  l=4, vocab=4096, 5.3M params   (TinyStories)
XL:     d=512, h=16, l=8, vocab=4096, 26.5M params  (TinyStories, GQA 4 KV heads)
XXL:    d=768, h=24, l=12, vocab=4096, 81.1M params (TinyStories, GQA 6 KV heads)
```

## Files

```
Core:
  model.py                          JAX transformer (with gradient checkpointing)
  data.py                           Shakespeare + TinyStories + BPE + tokenized data cache
  train.py                          AdamW training with LR schedule + XLA cache

Kernels:
  kernels/fused_prefill.py          fused prefill (d_model <= 64)
  kernels/fused_decode.py           fused decode (d_model <= 64)
  kernels/block_prefill.py          multi-block prefill + FlashAttention (d_model >= 128)
  kernels/block_decode.py           per-layer decode orchestrator (d_model >= 128)
  kernels/fused_decode_nlayer.py    fused N-layer decode (packed weights/caches)
  kernels/multi_sm_decode.py        multi-SM decode with atomic barriers + KV-split
  kernels/batched_decode.py         batched multi-SM decode (B sequences)
  kernels/persistent_decode.py      persistent decode (single launch, all steps)
  kernels/persistent_batched_decode.py  persistent batched (single launch, B × N steps)

Benchmarking:
  profile_kernels.py                primary profiling tool (run after every change)
  inference_benchmark.py            quick throughput + text generation demo
  baseline_metrics.txt              current performance numbers

Documentation:
  program.md                        agent program / development log
  repo_explained_from_zero.md       ground-up GPU kernel explanation
  cuda_kernels_docs/                Triton, CUDA, jax-triton reference docs
```
