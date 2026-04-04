# Remote Training Instructions — RTX 4090 (24GB)

## Your mission

Train this transformer model as fast as possible. Minimize wall-clock training time.
Everything is implemented and tested. Your job is to set up the environment, tune batch
size for 24GB VRAM, and run the training.

## What this project is

A 306M parameter decoder-only transformer (JAX + Triton), trained from scratch on 7.85B
tokens (web + code + math). The codebase includes custom Triton inference kernels, but
training uses JAX/XLA with cuDNN FlashAttention.

## Architecture

- d_model=1024, n_heads=16, n_kv_heads=4 (GQA), n_layers=24, ctx=512
- RMSNorm, RoPE, SwiGLU FFN (d_ff=2816), no biases, tied embeddings
- Vocab: 32K BPE tokenizer
- 306M params total

## Data (already prepared, 31.4GB)

7.85B unique tokens with EOS between documents:
- 34% FineWeb-Edu (quality-filtered web, score >= 3)
- 30% StarCoder code (13 languages)
- 19% OpenWebMath (math with LaTeX)
-  9% Wikipedia
-  8% Cosmopedia (synthetic textbooks)

Stored as `data/tokens_v2/train.bin` (flat int32 binary, memory-mapped) + `data/tokens_v2/val.npy`.

## Setup

```bash
# 1. Clone the repo
git clone git@github.com:zahlmann/transformer.git
cd transformer

# 2. Install dependencies (uses uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. Copy the data directory from the source machine
# The data/ directory is ~50GB total. You need:
#   data/tokens_v2/train.bin        (31.4 GB — the training data)
#   data/tokens_v2/val.npy          (150 MB — validation data)
#   data/tokens_v2/metadata.json    (metadata)
#   data/tokenizer_32000.json       (2.2 MB — BPE tokenizer)
#   data/bpe_vocab.pkl              (tokenizer config for inference)
#
# Use rsync or scp:
#   rsync -avP source_machine:transformer/data/tokens_v2/ data/tokens_v2/
#   rsync -avP source_machine:transformer/data/tokenizer_32000.json data/
#   rsync -avP source_machine:transformer/data/bpe_vocab.pkl data/
```

## Training command

```bash
# RECOMMENDED: No MTP, maximize batch size for 24GB
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 32 --epochs 3 \
  --dataset combined_v2 --curriculum --lr 3e-4 \
  2>&1 | tee training.log
```

## Tuning for maximum speed

### Batch size
On 4080 Super (16GB): bs=16 used 12.3GB VRAM.
On 4090 (24GB): **start with bs=32** (should use ~20GB). If it fits, try bs=48.
If OOM, fall back to bs=24.

The curriculum flag multiplies batch size in early phases:
- Phase 1 (10%): ctx=128, bs=bs×4 (e.g., 128 at bs=32)
- Phase 2 (20%): ctx=256, bs=bs×2 (e.g., 64 at bs=32)
- Phase 3 (70%): ctx=512, bs=bs (e.g., 32)

So with bs=32, phase 1 uses bs=128 at ctx=128. Check that this fits in VRAM.
If phase 1 OOMs, reduce base bs until all phases fit.

### MTP (multi-token prediction)
Adding `--mtp-heads 3` improves model quality but adds ~56% training time
(3 extra passes over 32K vocab per step). Skip it for speed. Add it later
if you want better quality.

### JAX cache
Create the cache dir to avoid compilation warnings:
```bash
mkdir -p .jax_cache
```

### Disable XLA debug output
```bash
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export TF_CPP_MIN_LOG_LEVEL=3
```

### Monitor
```bash
# In another terminal:
tail -f training.log | grep "step "
nvidia-smi -l 5  # watch GPU utilization
```

## Expected performance

On RTX 4090 with bs=32, no MTP:
- Estimated: 60-70K tok/s
- 23.5B tokens (3 epochs) → ~100 hours (4.2 days)
- Loss should reach ~3.0-3.5 by end of training

On RTX 4080 Super with bs=16, no MTP:
- Measured: 28.4K tok/s → ~230 hours (9.6 days)

## Key files

```
train.py                    — training script (AdamW, cosine LR, curriculum)
model.py                    — JAX model (forward, fused CE, MTP)
data.py                     — data loading (streaming v2 support)
prepare_data_v2.py          — data pipeline (already run, don't need to re-run)
program.md                  — full project history and architecture docs
```

## What NOT to change

- Don't change the model architecture (it's been carefully designed)
- Don't change the data pipeline (already optimized)
- Don't change the tokenizer (data is pre-tokenized with it)
- Don't add quantization (project goal is learning GPU kernels, not shrinking)
- Focus ONLY on training speed: batch size, XLA flags, compilation, etc.

## After training completes

The model saves to `weights.pkl`. Copy it back:
```bash
scp weights.pkl source_machine:transformer/
```

## Checkpointing

Currently the script only saves at the end. For a multi-day run, you may want to
add periodic checkpoint saves. The save code (from train.py end):
```python
import pickle
save_path = "weights.pkl"
with open(save_path, "wb") as f:
    pickle.dump({"params": jax.tree.map(np.asarray, params), "config": config}, f)
```

Add this inside the epoch loop after the `eval_loss` line for per-epoch checkpoints.
