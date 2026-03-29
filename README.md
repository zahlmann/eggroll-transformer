# Fused Inference

Custom Triton kernels for transformer inference — the entire forward pass runs in a **single GPU kernel call** with all weights resident in registers. **4x faster** than JAX/XLA baseline.

## Quick Start

```bash
uv run train_backprop.py         # train model, save weights (4s)
uv run inference_benchmark.py    # benchmark: Triton vs JAX (740 vs 185 tok/s)
```

## Results

```
Prompt: 64 tokens, Generate: 64 tokens

Triton fused:   740 tok/s  (86.5 ms)
JAX baseline:   185 tok/s  (362.6 ms)
Speedup:        4.0x
```

Both produce identical text.

## How It Works

Two fused Triton kernels replace the ~15 separate XLA kernels that JAX generates:

**Prefill kernel** — processes the full prompt (128 tokens) in one kernel call. All weights stay in the GPU register file (132 KB model fits in 130 KB register budget). Outputs logits + KV cache.

**Decode kernel** — generates one token at a time using the KV cache. Uses element-wise ops instead of tensor cores (M=1 is too small for tensor cores). Outputs logits + new K/V vectors for cache update.

The speedup comes from eliminating HBM round-trips between operations. In JAX, every intermediate result (embeddings, attention scores, FFN activations) gets written to HBM and read back. In the fused kernel, data flows through registers end-to-end.

See `inference_guide.md` for a detailed walkthrough (no GPU experience required).

## Architecture

- Decoder-only transformer: d_model=64, 2 heads, 1 layer, d_ff=256
- Character-level Shakespeare, vocab=65, context=128
- 66,368 parameters

## Files

```
inference_guide.md                      ground-up explanation of GPU kernels + this project
model.py                                JAX transformer (inference baseline)
train_backprop.py                       backprop+Adam training, saves weights
kernels/fused_prefill.py                fused prefill kernel (full sequence)
kernels/fused_decode.py                 fused decode kernel (one token + KV cache)
inference_benchmark.py                  benchmark + text generation
data.py                                 Shakespeare dataset
```

