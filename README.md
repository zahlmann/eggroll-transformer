# Single-GPU Transformer

Training and inference for a decoder-only transformer, entirely from scratch on one **RTX 4080 Super (16GB VRAM)**. Custom Triton kernels for decode. JAX for training. No frameworks, no shortcuts.

Built using [karpathy/autoresearch](https://github.com/karpathy/autoresearch)-style autonomous development — a coding agent is pointed at `program.md` repeatedly. The human steers direction; the agent handles implementation, debugging, benchmarking, and documentation.

See [`repo_explained_from_zero.md`](repo_explained_from_zero.md) for a ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all techniques used.

## Current Model

```
XXL (d=768, h=24, l=12, ctx=512, GQA 6 KV heads, 81.1M params)
  Trained on TinyStories (487M tokens, 3 epochs = 1.46B tokens seen)
  val_loss=1.068, ppl=2.60
  Training: ~11 hours, bf16 forward (53K tok/s throughput)
```

## Inference Performance

```
RTX 4080 Super | 81.1M param model (d=768)

  Multi-SM sync:       1006 tok/s  (0.99 ms/tok, 20% BW util)
  Pipelined:           1463 tok/s  (0.68 ms/tok)
  Persistent B=4:      2225 tok/s  (1.80 ms/step)
  Persistent B=8:      2460 tok/s  (3.25 ms/step)
  Prefill (128 tok):   12.9 ms    (2.4x faster than JAX baseline)
  Weight buffer:       162 MB (exceeds L2 — HBM-bound)
```

The entire decode step — embedding, attention, FFN, output projection — runs in a single GPU kernel call across all 12 layers.

## Quick Start

```bash
# generate text (streaming)
uv run generate.py --prompt "Once upon a time"

# batched inference with paged KV cache
uv run serve.py --paged --prompts "Once upon a time" "The cat sat"

# continuous batching (more prompts than batch slots)
uv run serve.py --continuous --batch-size 2 --prompts "A" "B" "C" "D"

# profile decode kernels
uv run profile_kernels.py

# train (d=768, ~11 hours on RTX 4080 Super)
uv run train.py --d-model 768 --n-heads 24 --n-kv-heads 6 --n-layers 12 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16
```

## What's In Here

**Training**: JAX model with gradient checkpointing, bf16 forward pass for tensor core utilization, AdamW with cosine LR schedule. Data pipeline handles TinyStories (487M tokens) with trained BPE tokenizer.

**Inference kernels**: 9 Triton kernel files covering single-sequence, batched, persistent, and pipelined decode. Multi-SM parallelism with atomic barriers, KV-split attention (FlashDecoding-style), weight-amortized FFN.

**Serving**: Variable-length batched inference server with GPU-accelerated paged KV cache (JIT gather/scatter) and continuous batching.

## Files

```
Training:
  model.py                          JAX transformer + gradient checkpointing
  data.py                           TinyStories + Shakespeare + BPE tokenizer
  train.py                          AdamW training with bf16 forward + LR schedule

Inference kernels:
  kernels/multi_sm_decode.py        multi-SM decode with atomic barriers + KV-split
  kernels/batched_decode.py         batched multi-SM decode (B sequences)
  kernels/persistent_decode.py      persistent decode (single launch, all steps)
  kernels/persistent_batched_decode.py  persistent batched (B × N steps)
  kernels/block_prefill.py          multi-block prefill + FlashAttention + GQA
  kernels/paged_kv.py               paged KV cache (GPU gather/scatter)

Serving:
  generate.py                       streaming text generation CLI
  serve.py                          batched server + continuous batching
  speculative_decode.py             speculative decoding benchmark

Benchmarking:
  profile_kernels.py                primary profiling tool
  test_paged_kernel.py              paged KV correctness + benchmark

Documentation:
  program.md                        agent program / development log
  repo_explained_from_zero.md       ground-up GPU kernel explanation
```
