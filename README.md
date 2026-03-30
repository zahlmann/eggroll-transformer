# Fused Inference

Custom Triton kernels for transformer inference — up to **15x faster** than JAX/XLA baseline. The entire decode step (embedding, attention, FFN, output projection) runs in a **single GPU kernel call**.

## Quick Start

```bash
# Train the scaled model on TinyStories (5.3M params, ~10 min)
uv run train_backprop.py

# Benchmark Triton vs JAX
uv run inference_benchmark.py
```

## Results

```
Small model (d=64, 1 layer, 189K params, Shakespeare):
  Triton:    3056 tok/s
  JAX:        181 tok/s
  Speedup:   16.9x

Medium model (d=128, 2 layers, 674K params, Shakespeare):
  Triton:    2589 tok/s
  JAX:        187 tok/s
  Speedup:   13.9x

Large model (d=256, 4 layers, 5.3M params, TinyStories):
  Triton:    1396 tok/s
  JAX:         92 tok/s
  Speedup:   15.1x
```

Text quality at 5.3M params (ppl=5.4) produces coherent multi-paragraph stories.

## How It Works

### Prefill (process full prompt)

**Small model (d_model=64):** Single fused kernel — all weights stay in registers (132KB model fits in 130KB register budget). Zero HBM round-trips between operations.

**Larger models (d_model=128-256):** Multi-block approach with configurable BLOCK_SEQ (32 for d=128, 16 for d=256). Three Triton kernels per layer (projections, attention, FFN). FlashAttention (tiled KV + online softmax) for context>256.

### Decode (generate tokens one at a time)

Fully fused N-layer kernel processes all layers in one launch using packed weight buffers and packed KV caches. Key optimizations:
- **In-kernel KV cache updates** — the kernel writes full updated caches directly, avoiding expensive `.at[].set()` scatter ops in Python (3.6x speedup alone)
- **Packed weight buffer** — all per-layer weights in one bf16 buffer, offsets computed from layer index
- **Packed KV caches** — all layers' caches in one flat buffer, stays packed between decode steps
- **Precomputed bf16 weights** — dtype conversions done once before the decode loop
- **Multi-layer fusion** — h stays in registers between layers (no HBM round-trip)

### Training

AdamW with linear warmup + cosine decay on TinyStories (14M BPE tokens, vocab=4096). Trained ByteLevel BPE tokenizer on the corpus (0% UNK).

### Key Insight

For small-to-medium models, **host-side overhead dominates GPU kernel time**. The GPU kernel takes <1ms per decode step, but Python/JAX dispatch, dtype conversions, and array scatters add several ms. Eliminating all host overhead through kernel fusion gives 15x speedup consistently across model scales.

## Architecture

```
Decoder-only transformer

Small:  d=64,  h=2, l=1, vocab=1024, 189K params  (Shakespeare)
Medium: d=128, h=4, l=2, vocab=1024, 674K params  (Shakespeare)
Large:  d=256, h=8, l=4, vocab=4096, 5.3M params  (TinyStories)
```

## Files

```
program.md                              agent program (read first for context)
inference_guide.md                      ground-up explanation of GPU kernels
model.py                                JAX transformer (inference baseline)
train_backprop.py                       AdamW training with LR schedule
data.py                                 Shakespeare + TinyStories + BPE tokenizer
kernels/fused_prefill.py                fused prefill kernel (d_model<=64)
kernels/fused_decode.py                 fused decode kernel (d_model<=64)
kernels/block_prefill.py                multi-block prefill + FlashAttention (d_model>=128)
kernels/block_decode.py                 per-layer decode (d_model>=128)
kernels/fused_decode_2layer.py          fully fused 2-layer decode
kernels/fused_decode_nlayer.py          fully fused N-layer decode (packed weights/caches)
inference_benchmark.py                  speed comparison benchmark
```
