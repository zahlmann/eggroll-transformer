# 303M JAX Transformer

This repository contains a pretrained decoder-only transformer and the code used
to train it. The current focus is understanding the trained base model clearly
and using it as the starting point for small, controlled supervised fine-tuning
experiments.

The model is not instruction-tuned yet. It has been pretrained with next-token
prediction on a mix of web text, code, math, Wikipedia, and synthetic textbook
data.

## Current Model

```text
303,350,784 parameters
d_model:      1024
layers:       24
query heads:  16
kv heads:     4
head dim:     64
ffn dim:      2816
context:      512 tokens
vocab:        32,000 BPE tokens
```

Architecture:

```text
decoder-only transformer
RMSNorm
RoPE
grouped-query attention
SwiGLU feed-forward layers
no biases
tied input/output embeddings
fused chunked cross-entropy for large-vocab training
```

The latest training log in this checkout shows:

```text
dataset: data/tokens_v3
train tokens: 42.24B
final shown validation loss: 2.6865
final shown perplexity: 14.68
```

## Start Here

- [`knowledge/pretrained_transformer_guide.md`](knowledge/pretrained_transformer_guide.md) explains the current model and training loop from first principles.
- [`model.py`](model.py) defines the transformer architecture and loss.
- [`train.py`](train.py) contains the pretraining loop.
- [`data.py`](data.py) loads tokenized training and validation data.

The guide is the main document. It explains the math, the shapes, the training
logic, and the simplest next step.

## Local Model Artifacts

Large model files are intentionally not tracked by git, but this checkout may
contain:

```text
weights.pkl
weights_v2.pkl
checkpoint.pkl
checkpoint_v2.pkl
```

Keep these files. The `weights*.pkl` files are weights-only exports. The
`checkpoint*.pkl` files also include optimizer state and training progress.

## Training

The pretraining entry point is:

```bash
uv run python -u train.py \
  --d-model 1024 \
  --n-heads 16 \
  --n-kv-heads 4 \
  --n-layers 24 \
  --context-len 512 \
  --batch-size 256 \
  --epochs 2 \
  --curriculum \
  --lr 3e-4 \
  --no-checkpoint \
  --data-dir data/tokens_v3
```

The large batch size above is for a large GPU. Smaller GPUs need a smaller batch
size and may need gradient checkpointing enabled.

## Data

The data preparation scripts are:

```text
prepare_data_v2.py
prepare_data_v3.py
```

The later training run used `tokens_v3`, which expands the dataset beyond the
earlier `tokens_v2` run. Tokenized data is too large for git and is ignored.

## Next Step

The simplest useful next step is supervised fine-tuning with a tiny bash-agent
trace dataset:

```text
1. Keep the pretrained checkpoint fixed as the base model.
2. Generate or hand-write 10 to 50 short, verified bash-agent traces.
3. Build `train_sft.py`.
4. Tokenize traces with plain text markers like `User:`, `Assistant:`, `[BASH]`,
   and `[BASH_RESULT]`.
5. Use a loss mask so the model trains only on assistant/tool-call tokens.
6. First prove the code can overfit the tiny dataset.
7. Only then scale trace generation with DeepSeek.
```

Do not start with reinforcement learning. Do not start with a huge synthetic
dataset. The first milestone is a boring masked-SFT loop that works.

## Optional Background

The repo also contains earlier inference optimization experiments:

```text
generate.py
profile_kernels.py
kernels/
knowledge/inference_explained.md
```

Those files are useful background, but they are not the main path right now. The
main path is understanding the pretrained model and adding a small SFT stage.

## File Map

```text
model.py
  JAX transformer architecture, forward pass, and fused cross-entropy loss.

train.py
  Pretraining loop with AdamW, curriculum sequence lengths, checkpointing, and
  bfloat16 forward/loss computation.

data.py
  Tokenized dataset loader using memory-mapped training tokens.

prepare_data_v2.py, prepare_data_v3.py
  Dataset download, tokenizer, and tokenization scripts.

knowledge/pretrained_transformer_guide.md
  Main learning guide for the current project state.

knowledge/training_explained.md
  Older broader training explanation.

knowledge/agentic_sft_plan.md
  Earlier SFT planning notes.

knowledge/inference_explained.md
  Inference-kernel explanation kept as optional background.
```
