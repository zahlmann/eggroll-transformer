# EGGROLL Transformer

Training a small transformer with **EGGROLL** (Evolution Strategies with low-rank perturbations) instead of backpropagation.

## Quick Start

```bash
uv run train_eggroll.py          # EGGROLL training (153s, val_loss ~2.49)
uv run train_backprop.py         # Backprop+Adam baseline (4s, val_loss ~1.84)
uv run benchmark.py              # Side-by-side comparison
uv run validate.py               # 3-seed quality validation
```

## Architecture

- Decoder-only transformer: d_model=64, 2 heads, 1 layer, d_ff=256
- Character-level Shakespeare, vocab=65, context=128
- 66,368 parameters

## How EGGROLL Works

1. Generate 4096 random perturbation vectors (rank-1 compressed, 28.8x compression)
2. Evaluate perturbed model on each batch (+sigma and -sigma)
3. Estimate gradient from fitness differences (Winsorized z-score, per-subgroup)
4. Update with Adam optimizer

All 8192 forward passes (4096 pairs) run in a single fused Triton kernel.

## Results (10 epochs)

| Method | val_loss | ppl | Time | Memory |
|--------|----------|-----|------|--------|
| EGGROLL (Triton kernel) | 2.49 | 12.1 | 153s | 70MB |
| Backprop+Adam | 1.84 | 6.3 | 4.1s | 160MB |
| Backprop+SGD | 2.45 | 11.6 | 1.3s | 300MB |

## Current Status

**Speed is near its limit** (153s, Triton kernel at 255 regs/thread, 47% utilization).
Further speed improvements require CUDA PTX. See program.md for full history.

**Quality gap is the main challenge**: val_loss 2.49 vs backprop's 1.84. The next agent
should focus on closing this gap through better gradient estimation, learning rate
schedules, perturbation strategies, or algorithmic improvements. See program.md
"What to Try Next — Quality Improvement" for detailed approaches.

## Generated Text Examples (10 epochs, seed=42)

**EGGROLL** (val_loss 2.49 — repetitive, lacks structure):
```
ROMEO:
O, the what the sin ay youre mive the the ther forer, wherou hathe he
the beporeant thin wind the forin me the maron thinge rard s and meeps
thingin a$e oue thand sot, thin that the o the houngor...
```

**Backprop+Adam** (val_loss 1.84 — learns formatting, punctuation, word boundaries):
```
ROMEO:
O, live the the sighalt nower is come countizen:
I pay herow have Are they their me.
AN Rome, groow! them thy nearws thing I brook and meepss I wains:
I not what thei, being bady; you mee hou gre...
```

Both are gibberish at this model size/training budget, but backprop picks up Shakespeare's
formatting conventions (newlines, colons, "I" as a word) while EGGROLL mostly learns
common character n-gram patterns.

## Files

```
train_eggroll.py              main EGGROLL training script
train_backprop.py             backprop+Adam baseline
benchmark.py                  comparison benchmark
validate.py                   3-seed quality validation
data.py                       Shakespeare dataset
model.py                      transformer model
kernels/fused_transformer_ce.py  fused Triton kernel (heavily optimized)
kernels/cuda/                 CUDA kernel infrastructure (working, needs PTX optimization)
profile_triton.py             kernel profiler
validate_kernel.py            kernel correctness test
results.tsv                   experiment log
program.md                    agent instructions (READ THIS FIRST)
```

## Optimization History

Total speedup: **444s → 153s (2.9x)** across two sessions.

Key milestones:
1. HALF_POP 8192→4096 (2x speedup)
2. num_warps 8→4 (18% speedup)
3. FFN dynamic loop (10.6% speedup)
4. Dynamic head loop + sigma compensation (2% speedup)
5. Per-batch sync removal (1.3% speedup)

See program.md for the complete experiment history and what was tried.
