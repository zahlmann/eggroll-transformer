# EGGROLL Transformer

Training a small transformer with **EGGROLL** (Evolution Strategies with low-rank perturbations) instead of backpropagation.

## Quick Start

```bash
uv run train_eggroll.py          # EGGROLL training (175s, val_loss ~2.44)
uv run train_backprop.py         # Backprop+Adam baseline (4s, val_loss ~1.84)
uv run benchmark.py              # Side-by-side comparison
uv run validate.py               # 3-seed quality validation
```

## Architecture

- Decoder-only transformer: d_model=64, 2 heads, 1 layer, d_ff=256
- Character-level Shakespeare, vocab=65, context=128
- 66,368 parameters

## How EGGROLL Works

1. Generate 4096 random perturbation vectors (rank-1 compressed)
2. Evaluate perturbed model on each batch (+sigma and -sigma)
3. Estimate gradient from fitness differences (Winsorized z-score)
4. Update with Adam optimizer

All 8192 forward passes (4096 pairs) run in a single fused Triton kernel.

## Files

```
train_eggroll.py              main EGGROLL training script
train_backprop.py             backprop+Adam baseline
benchmark.py                  comparison benchmark
validate.py                   3-seed quality validation
data.py                       Shakespeare dataset
model.py                      transformer model
kernels/fused_transformer_ce.py  fused Triton kernel
profile_triton.py             kernel profiler
validate_kernel.py            kernel correctness test
results.tsv                   experiment log
program.md                    agent instructions
```

## Results (10 epochs)

| Method | val_loss | Time | Memory |
|--------|----------|------|--------|
| EGGROLL (Triton kernel) | 2.44 | 175s | 70MB |
| Backprop+Adam | 1.84 | 4.1s | 160MB |
| Backprop+SGD | 2.45 | 1.3s | 300MB |
