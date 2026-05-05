# The Pretrained Transformer, Explained From First Principles

This is the main learning document for the current project state.

The project currently has a pretrained decoder-only transformer. The useful next
step is not to add more cleverness. The useful next step is to understand the
base model clearly, keep inference experiments out of the main path for now, and
only then decide whether to do supervised fine-tuning.

This guide assumes you know some Python and basic machine learning. It does not
assume that transformer math is already comfortable.

## What This Guide Covers

- What was trained.
- What the model architecture is.
- How the training data becomes input and target arrays.
- What next-token prediction means.
- How the loss is computed.
- How the optimizer updates the weights.
- How to read `data.py`, `model.py`, and `train.py`.
- What the simplest next step should be.

It does not explain the custom inference kernels. Those files can stay in the
repo, but they are not the main project path right now.

## Current State In One Page

The trained model is a decoder-only transformer with these core settings:

```text
vocab_size   = 32000
d_model      = 1024
n_layers     = 24
n_heads      = 16
n_kv_heads   = 4
d_head       = 64
d_ff         = 2816
context_len  = 512
parameters   = 303,350,784
```

The architecture uses:

- token embeddings
- RMSNorm
- causal self-attention
- grouped-query attention, or GQA
- RoPE positional encoding
- SwiGLU feed-forward layers
- residual connections
- tied input/output embeddings
- next-token cross-entropy loss

The repo also contains local model artifacts:

```text
weights.pkl        weights-only export from an earlier run
weights_v2.pkl     newer weights-only export
checkpoint.pkl     resumable checkpoint from an earlier run
checkpoint_v2.pkl  newer resumable checkpoint
```

Do not delete those. They are not just generated junk. They are the trained model.

The training logs show two main phases:

```text
training.log
  dataset: v2, about 7.64B train tokens
  final shown result: epoch 3/3, train=2.2528, val=2.8577, ppl=17.42

training_v3.log
  dataset: v3, 42.24B train tokens
  final shown result: epoch 2/2, train=2.0922, val=2.6865, ppl=14.68
```

The v3 run is the better checkpoint by validation loss and perplexity.

## The Smallest Mental Model

A language model is a function.

It receives token IDs:

```text
[15496, 995, 318, 257]
```

It predicts the next token at every position:

```text
input:  [15496, 995, 318]
target: [995,   318, 257]
```

In plain English:

- Given token 15496, predict token 995.
- Given tokens 15496 and 995, predict token 318.
- Given tokens 15496, 995, and 318, predict token 257.

The model does not learn by being told "this sentence is good" or "this answer is
correct." During pretraining, it only learns this:

```text
Given the text so far, make the next token more likely.
```

That is why this model is pretrained but not instruction-tuned. It has learned a
lot of language, code, and math patterns, but it has not yet been specifically
trained to follow an assistant conversation format.

## Tokens

Computers do not directly process text like `"hello"`. They process numbers.

A tokenizer maps text to token IDs:

```text
"hello world" -> [some_id_1, some_id_2]
```

The model's tokenizer has 32,000 possible tokens. That means every position in
the sequence is an integer from 0 to 31,999.

So the model sees data shaped like this:

```text
batch_size = 256
context_len = 512

x shape = (256, 512)
y shape = (256, 512)
```

Each row is one training example. Each column is one token position.

For a tiny example, imagine one sequence has 6 tokens:

```text
raw tokens = [10, 20, 30, 40, 50, 60]
```

If the context length is 5, training creates:

```text
x = [10, 20, 30, 40, 50]
y = [20, 30, 40, 50, 60]
```

The target is just the input shifted left by one position.

## Parameter Count

This is a good place to understand the model size. The model has 303,350,784
learned numbers.

We can derive that from the architecture.

### Embedding Parameters

The embedding table has one vector per token.

```text
vocab_size = 32000
d_model = 1024
```

So:

$$
32000 \times 1024 = 32768000
$$

The embedding table has 32,768,000 parameters.

### Per-Layer Attention Parameters

Each transformer layer has attention matrices:

```text
W_q: (1024, 1024)
W_k: (1024, 256)
W_v: (1024, 256)
W_o: (1024, 1024)
```

Why is `W_k` only 256 wide?

Because this model uses grouped-query attention:

```text
n_heads = 16
n_kv_heads = 4
d_head = 64
d_kv = n_kv_heads * d_head = 4 * 64 = 256
```

Now count the attention parameters:

$$
W_q = 1024 \times 1024 = 1048576
$$

$$
W_k = 1024 \times 256 = 262144
$$

$$
W_v = 1024 \times 256 = 262144
$$

$$
W_o = 1024 \times 1024 = 1048576
$$

Add them:

$$
1048576 + 262144 + 262144 + 1048576 = 2621440
$$

Attention contributes 2,621,440 parameters per layer.

### Per-Layer RMSNorm Parameters

Each layer has two RMSNorm scale vectors:

```text
ln1.scale: 1024
ln2.scale: 1024
```

So:

$$
1024 + 1024 = 2048
$$

RMSNorm contributes 2,048 parameters per layer.

### Per-Layer Feed-Forward Parameters

The SwiGLU feed-forward block has three matrices:

```text
gate: (1024, 2816)
up:   (1024, 2816)
down: (2816, 1024)
```

Each matrix has:

$$
1024 \times 2816 = 2883584
$$

There are three of them:

$$
2883584 + 2883584 + 2883584 = 8650752
$$

The feed-forward block contributes 8,650,752 parameters per layer.

### Total Per Layer

Now add attention, RMSNorm, and feed-forward parameters:

$$
2621440 + 2048 + 8650752 = 11274240
$$

Each transformer layer has 11,274,240 parameters.

There are 24 layers:

$$
11274240 \times 24 = 270581760
$$

The stack of layers has 270,581,760 parameters.

### Final Total

Now add:

```text
token embedding = 32,768,000
all layers      = 270,581,760
final RMSNorm   = 1,024
```

Step by step:

$$
32768000 + 270581760 = 303349760
$$

$$
303349760 + 1024 = 303350784
$$

That matches the training log:

```text
303,350,784 params
```

## The Architecture As A Data Flow

For one sequence, the model roughly does this:

```text
token IDs
  -> token embeddings
  -> layer 0
  -> layer 1
  -> ...
  -> layer 23
  -> final RMSNorm
  -> output logits
  -> cross-entropy loss
```

Each layer does this:

```text
h
  -> RMSNorm
  -> causal self-attention with RoPE and GQA
  -> add back to h
  -> RMSNorm
  -> SwiGLU feed-forward block
  -> add back to h
```

The repeated "add back to h" steps are residual connections. They let every layer
modify the representation without having to rebuild it from scratch.

## Embeddings

The first line of the model's real computation is:

```python
h = params["token_emb"][x]
```

If `x` is token IDs, then `params["token_emb"][x]` looks up one vector per token.

Tiny example:

```text
token_emb[10] = [0.2, -0.1, 0.7]
token_emb[20] = [0.0,  0.5, 0.3]
token_emb[30] = [0.4,  0.4, 0.1]
```

If:

```text
x = [10, 20, 30]
```

Then:

```text
h = [
  [0.2, -0.1, 0.7],
  [0.0,  0.5, 0.3],
  [0.4,  0.4, 0.1],
]
```

In the real model, each vector has 1024 numbers, not 3.

## RMSNorm

RMSNorm keeps the size of vectors stable.

The code is:

```python
def rms_norm(x, scale, eps=1e-5):
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps).astype(x.dtype)
    return scale * (x / rms)
```

The formula is:

$$
\text{rms}(x) = \sqrt{\text{mean}(x^2) + \epsilon}
$$

Then:

$$
\text{output} = \text{scale} \times {x \over \text{rms}(x)}
$$

Numerical example:

```text
x = [3, 4]
scale = [1, 1]
eps = 0 for the simple example
```

Square each value:

$$
3^2 = 9
$$

$$
4^2 = 16
$$

Take the mean:

$$
{9 + 16 \over 2} = {25 \over 2} = 12.5
$$

Take the square root:

$$
\sqrt{12.5} = 3.5355
$$

Divide each value by the RMS:

$$
{3 \over 3.5355} = 0.8485
$$

$$
{4 \over 3.5355} = 1.1314
$$

So the normalized vector is approximately:

```text
[0.8485, 1.1314]
```

The `scale` parameter lets the model learn how large each dimension should be
after normalization.

## Attention

Attention lets each token look back at previous tokens.

For each token vector `x`, the model computes:

```text
query = x @ W_q
key   = x @ W_k
value = x @ W_v
```

The query means:

```text
What am I looking for?
```

The key means:

```text
What information do I contain?
```

The value means:

```text
What should I pass along if someone attends to me?
```

The attention score between one query and one key is:

$$
\text{score} = {q \cdot k \over \sqrt{d_\text{head}}}
$$

The dot product is high when two vectors point in a similar direction.

### Tiny Attention Example

Use very small vectors:

```text
q = [1, 2]
k_1 = [1, 0]
k_2 = [0, 2]
d_head = 2
```

First dot product:

$$
q \cdot k_1 = 1 \times 1 + 2 \times 0
$$

$$
q \cdot k_1 = 1 + 0 = 1
$$

Second dot product:

$$
q \cdot k_2 = 1 \times 0 + 2 \times 2
$$

$$
q \cdot k_2 = 0 + 4 = 4
$$

Now divide by $\sqrt{d_\text{head}}$:

$$
\sqrt{2} = 1.4142
$$

$$
{1 \over 1.4142} = 0.7071
$$

$$
{4 \over 1.4142} = 2.8284
$$

So the scores are:

```text
[0.7071, 2.8284]
```

Softmax turns scores into probabilities.

First exponentiate:

$$
\exp(0.7071) = 2.028
$$

$$
\exp(2.8284) = 16.918
$$

Add them:

$$
2.028 + 16.918 = 18.946
$$

Divide each by the sum:

$$
{2.028 \over 18.946} = 0.107
$$

$$
{16.918 \over 18.946} = 0.893
$$

So the query pays about 10.7% attention to token 1 and 89.3% attention to token 2.

In a decoder-only language model, attention is causal. Position 10 can look at
positions 0 through 10, but not position 11 or later. This prevents the model from
cheating during training.

## RoPE

RoPE means rotary positional embeddings.

Attention by itself compares token content, but it does not know where tokens are
in the sequence. RoPE injects position by rotating pairs of query/key dimensions.

For one pair of dimensions:

```text
before = [x_even, x_odd]
```

RoPE computes:

$$
\text{new_even} = x_\text{even} \times \cos(\theta) - x_\text{odd} \times \sin(\theta)
$$

$$
\text{new_odd} = x_\text{even} \times \sin(\theta) + x_\text{odd} \times \cos(\theta)
$$

The angle $\theta$ depends on the token position and the feature dimension.

Tiny example:

```text
x_even = 2
x_odd = 1
cos(theta) = 0.8
sin(theta) = 0.6
```

Compute the new even value:

$$
2 \times 0.8 - 1 \times 0.6 = 1.6 - 0.6 = 1.0
$$

Compute the new odd value:

$$
2 \times 0.6 + 1 \times 0.8 = 1.2 + 0.8 = 2.0
$$

So:

```text
[2, 1] -> [1, 2]
```

The vector changed because its position changed.

## Grouped-Query Attention

Normal multi-head attention would use 16 query heads, 16 key heads, and 16 value
heads.

This model uses:

```text
query heads = 16
key heads   = 4
value heads = 4
```

That is grouped-query attention.

Since:

$$
16 \over 4 = 4
$$

each key/value head is shared by 4 query heads.

Why do this?

- It reduces parameters.
- It reduces memory used by key/value caches during generation.
- It usually keeps most of the quality of full multi-head attention.

For training, it is just part of the architecture. The model learns with this
constraint from the start.

## SwiGLU Feed-Forward Block

After attention, each token goes through a feed-forward network.

The code is:

```python
h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
```

Break it into smaller steps:

```python
gate_pre = h_norm2 @ ffn_gate
up = h_norm2 @ ffn_up
gate = jax.nn.silu(gate_pre)
mixed = gate * up
h_ff = mixed @ ffn_down
```

The SiLU function is:

$$
\text{silu}(x) = x \times \text{sigmoid}(x)
$$

and:

$$
\text{sigmoid}(x) = {1 \over 1 + \exp(-x)}
$$

Numerical example:

```text
gate_pre = 2
up = 3
```

Compute sigmoid:

$$
\text{sigmoid}(2) = {1 \over 1 + \exp(-2)}
$$

$$
\exp(-2) = 0.1353
$$

$$
1 + 0.1353 = 1.1353
$$

$$
{1 \over 1.1353} = 0.8808
$$

Compute SiLU:

$$
\text{silu}(2) = 2 \times 0.8808 = 1.7616
$$

Multiply by `up`:

$$
1.7616 \times 3 = 5.2848
$$

That multiplication is the "GLU" part. The gate decides how much of the `up`
branch should pass through.

## Residual Connections

The layer does:

```python
h = h + attn_out
...
return h + h_ff
```

That means attention and feed-forward blocks learn changes to the vector, not the
whole vector from scratch.

If:

```text
h = [10, 20]
attn_out = [1, -3]
```

Then:

```text
h + attn_out = [11, 17]
```

The original information is still there. The layer only adjusted it.

## Output Logits

At the end, the model has one hidden vector per token position.

The code computes:

```python
logits = h @ params["token_emb"].T
```

This compares each hidden vector against every token embedding.

If the hidden vector is similar to the embedding for token 123, the logit for
token 123 becomes large.

The model uses tied embeddings:

```text
same table for input token vectors and output token prediction
```

This saves parameters and is a standard language-model design.

## Cross-Entropy Loss

The model produces logits. Logits are raw scores, not probabilities.

For one position, imagine the model outputs three logits:

```text
logits = [2, 1, 0]
target token = 2
```

Softmax turns logits into probabilities.

First exponentiate:

$$
\exp(2) = 7.389
$$

$$
\exp(1) = 2.718
$$

$$
\exp(0) = 1
$$

Add them:

$$
7.389 + 2.718 + 1 = 11.107
$$

Divide each by the sum:

$$
{7.389 \over 11.107} = 0.665
$$

$$
{2.718 \over 11.107} = 0.245
$$

$$
{1 \over 11.107} = 0.090
$$

So the model assigns probability 0.090 to the correct token.

Cross-entropy loss is:

$$
\text{loss} = -\log(p_\text{correct})
$$

Here:

$$
\text{loss} = -\log(0.090)
$$

$$
\log(0.090) = -2.408
$$

$$
-\log(0.090) = 2.408
$$

So this one prediction has loss 2.408.

Lower is better.

## Perplexity

Perplexity is:

$$
\text{perplexity} = \exp(\text{loss})
$$

The final v3 validation loss in the log is:

```text
val = 2.6865
```

So:

$$
\text{perplexity} = \exp(2.6865)
$$

$$
\exp(2.6865) = 14.68
$$

This matches the log:

```text
ppl=14.68
```

Intuition:

```text
perplexity 14.68 means the model is, roughly, as uncertain as choosing among
about 14.68 plausible next tokens on average.
```

That is not instruction-following quality. It is base-model next-token quality.

## Why The Loss Is Fused And Chunked

The naive output logits tensor would be huge.

For the v3 run:

```text
batch_size = 256
context_len = 512
vocab_size = 32000
```

Number of token positions:

$$
256 \times 512 = 131072
$$

Number of logits:

$$
131072 \times 32000 = 4194304000
$$

That is 4,194,304,000 logits.

If each logit were stored as `float32`, it would use 4 bytes:

$$
4194304000 \times 4 = 16777216000
$$

That is about 16.8 GB just for logits.

If each logit were stored as `bfloat16`, it would use 2 bytes:

$$
4194304000 \times 2 = 8388608000
$$

That is about 8.4 GB just for logits.

That would waste too much memory.

So `model.py` computes cross-entropy in chunks of the vocabulary. With a chunk
size of 4096:

$$
{32000 \over 4096} = 7.8125
$$

So it processes 8 chunks instead of creating the full logits tensor at once.

The important idea:

```text
We only need the final loss and gradients. We do not need to keep the full
(batch, sequence, vocab) logits tensor in memory.
```

## The Training Loop

Training repeats this process:

```text
1. Load a batch of token IDs.
2. Split it into x and y.
3. Run the model on x.
4. Compute cross-entropy against y.
5. Use automatic differentiation to compute gradients.
6. Use AdamW to update parameters.
7. Repeat.
```

## Learning Rate Schedule

The code uses:

```python
optax.linear_schedule(0.0, args.lr, args.warmup_steps)
optax.cosine_decay_schedule(args.lr, total_steps - args.warmup_steps, alpha=args.lr * 0.01)
```

During warmup, the learning rate rises from 0 to the maximum.

If:

```text
max lr = 0.0003
warmup steps = 200
current step = 100
```

Then:

$$
\text{lr} = 0.0003 \times {100 \over 200}
$$

$$
{100 \over 200} = 0.5
$$

$$
\text{lr} = 0.0003 \times 0.5 = 0.00015
$$

After warmup, cosine decay gradually lowers the learning rate. The reason is
simple: early in training, large updates help the model learn quickly; later in
training, smaller updates avoid disturbing what the model has already learned.

## AdamW

The code uses:

```python
optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
```

The gradient tells us:

```text
If this parameter goes up a little, does the loss go up or down?
```

Plain gradient descent would do:

$$
w_\text{new} = w_\text{old} - \text{lr} \times g
$$

Example:

```text
w_old = 1.0
lr = 0.1
g = 0.3
```

Then:

$$
w_\text{new} = 1.0 - 0.1 \times 0.3
$$

$$
0.1 \times 0.3 = 0.03
$$

$$
w_\text{new} = 1.0 - 0.03 = 0.97
$$

AdamW is more sophisticated. It keeps two moving averages:

```text
m = average gradient direction
v = average squared gradient size
```

So it can move consistently in useful directions and reduce step sizes for
parameters whose gradients are noisy or large.

Weight decay adds a small pull toward zero. It helps prevent parameters from
growing unnecessarily large.

## Curriculum Training

The code can train on shorter contexts first:

```text
first 10% of steps: context 128, batch size x4
next 20% of steps:  context 256, batch size x2
last 70% of steps:  context 512, batch size x1
```

Why this helps:

- Shorter sequences are cheaper.
- The model learns local patterns first.
- Later it learns longer-range patterns.

The target model still has context length 512. The curriculum just changes how
much of that context is used early in training.

## Mixed Precision

The code keeps master parameters in higher precision, then casts them to
`bfloat16` for the forward/loss computation:

```python
params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
```

The intuition:

- `float32` is safer for optimizer state and accumulated updates.
- `bfloat16` is faster and uses less memory for big matrix multiplications.

This is a normal modern training pattern.

## Reading `data.py`

`data.py` is small and important.

### Imports

```python
import json
import os
import pickle
import numpy as np
```

- `json` reads metadata like vocabulary size and tokenizer path.
- `os` builds file paths.
- `pickle` saves a small tokenizer reference for generation.
- `numpy` loads arrays and memory-maps the training token file.

### Data Directory

```python
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
```

This means:

```text
the data folder next to data.py
```

So if the repo is:

```text
/path/to/transformer
```

then:

```text
DATA_DIR = /path/to/transformer/data
```

### `load_data`

```python
def load_data(context_len, data_dir=None):
```

This function loads tokenized training and validation data.

```python
from tokenizers import Tokenizer
```

This import checks that the tokenizer dependency is available. The function does
not actually need the `Tokenizer` name later, so this line is more like an early
dependency check.

```python
token_dir = data_dir or os.path.join(DATA_DIR, "tokens_v2")
```

If the caller passes `--data-dir`, use that. Otherwise use `data/tokens_v2`.

For the later run, the command used `data/tokens_v3`.

```python
train_bin = os.path.join(token_dir, "train.bin")
val_npy = os.path.join(token_dir, "val.npy")
meta_path = os.path.join(token_dir, "metadata.json")
```

The tokenized dataset has:

- `train.bin`: raw training token IDs
- `val.npy`: validation token IDs
- `metadata.json`: vocabulary size and tokenizer path

```python
assert os.path.exists(train_bin), f"Missing {train_bin}. Run: uv run prepare_data_v3.py"
assert os.path.exists(meta_path), f"Missing {meta_path}"
```

These fail early if the tokenized data is missing.

```python
with open(meta_path) as f:
    meta = json.load(f)
```

Read metadata from JSON.

```python
vocab_size = meta["vocab_size"]
```

The model needs this to size the embedding table.

```python
train_mmap = np.memmap(train_bin, dtype=np.int32, mode="r")
```

This is important. The full token file is too large to load into RAM. `memmap`
lets NumPy treat the file like an array and read pieces on demand.

```python
val_data = np.load(val_npy)
```

Validation is smaller, so it is loaded normally.

```python
n_val = (len(val_data) - 1) // context_len
```

This computes how many full validation sequences fit.

Example:

```text
len(val_data) = 1001
context_len = 100
```

Then:

$$
{1001 - 1 \over 100} = {1000 \over 100} = 10
$$

So 10 full validation sequences fit.

```python
val_data = val_data[: n_val * context_len + 1]
```

Trim validation data so it has exactly enough tokens for `x` and shifted `y`.

```python
val_x = val_data[:-1].reshape(n_val, context_len)
val_y = val_data[1:].reshape(n_val, context_len)
```

This creates input and target arrays.

Tiny example:

```text
val_data = [10, 20, 30, 40, 50, 60]
context_len = 5

val_x = [10, 20, 30, 40, 50]
val_y = [20, 30, 40, 50, 60]
```

```python
n_train = (len(train_mmap) - 1) // context_len
```

Same idea for training data, but it does not reshape the whole training file
because that file is streamed by batches later.

```python
print(...)
```

The print statements tell you which dataset was loaded and how many sequences it
contains.

```python
tok_path = meta["tokenizer_path"]
if not os.path.isabs(tok_path):
    tok_path = os.path.join(os.path.dirname(__file__), tok_path)
```

Metadata may store a relative tokenizer path. This turns it into an absolute path.

```python
assert os.path.exists(tok_path), f"Missing tokenizer at {tok_path}"
```

Fail early if the tokenizer is missing.

```python
with open(os.path.join(DATA_DIR, "bpe_vocab.pkl"), "wb") as f:
    pickle.dump({"tokenizer_path": tok_path, "vocab_size": vocab_size}, f)
```

This saves a tiny reference used by generation code. It does not save the full
training dataset.

```python
return {
    "train_tokens": train_mmap,
    "val_x": val_x,
    "val_y": val_y,
    "vocab_size": vocab_size,
}
```

The training script receives exactly what it needs:

- streaming training tokens
- validation inputs
- validation targets
- vocabulary size

## Reading `model.py`

`model.py` defines the architecture and the loss.

The training path through the file is:

```text
train.py
  -> transformer_loss_fused
  -> _transformer_trunk
  -> _attn_layer
  -> causal_attention
  -> fused_output_and_loss
  -> fused_cross_entropy
```

The file also contains inference helpers like `prefill_with_kv`. They matter for
generation, but they are not needed to understand how the model was pretrained.

### Imports

```python
import functools

import jax
import jax.numpy as jnp
```

- `functools` is used for the `jax.custom_vjp` decorator on the fused
  cross-entropy function.
- `jax` provides random initialization, automatic differentiation support,
  checkpointing, and neural-network helpers.
- `jax.numpy` is imported as `jnp`. It looks like NumPy, but operations can run
  on the GPU and participate in automatic differentiation.

### `_swiglu_d_ff`

```python
def _swiglu_d_ff(d_model):
    return ((8 * d_model // 3 + 127) // 128) * 128
```

The comment in the file explains why:

```text
standard FFN params: 2 * d * 4d = 8d^2
SwiGLU params:       3 * d * d_ff
```

Set them roughly equal:

$$
3 \times d \times d_\text{ff} = 8 \times d^2
$$

Divide both sides by $3d$:

$$
d_\text{ff} = {8d^2 \over 3d}
$$

Cancel one $d$:

$$
d_\text{ff} = {8d \over 3}
$$

For this model:

$$
{8 \times 1024 \over 3} = {8192 \over 3} = 2730.67
$$

The code rounds up to a multiple of 128:

```text
2730.67 -> 2816
```

GPU matrix kernels often prefer dimensions with friendly multiples like 128.

### `init_transformer`

This function creates all parameters and a config dictionary.

```python
assert d_model % n_heads == 0
```

The model dimension must split evenly across attention heads.

```text
1024 / 16 = 64
```

So each head gets 64 dimensions.

```python
d_head = d_model // n_heads
```

For this model:

```text
d_head = 64
```

```python
assert n_heads % n_kv_heads == 0
```

The query heads must split evenly across key/value heads.

```text
16 / 4 = 4
```

So each KV head serves 4 query heads.

```python
params = {}
config = {...}
```

`params` stores learned arrays. `config` stores shape settings.

```python
params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02
```

This initializes embeddings with small random numbers.

Why small?

If initial weights are too large, activations can explode. If they are all zero,
every feature starts identical. Small random values break symmetry without making
the network unstable.

Inside the layer loop, each layer gets:

```text
ln1.scale
attn.q
attn.k
attn.v
attn.o
ln2.scale
ffn.gate
ffn.up
ffn.down
```

There are no bias vectors. This is common in modern transformer implementations.

### `precompute_rope_table`

```python
half = d_head // 2
```

RoPE rotates pairs of dimensions, so it needs half as many angles as dimensions.

For `d_head = 64`:

```text
half = 32
```

```python
freqs = base ** (-jnp.arange(0, half, dtype=jnp.float32) * 2.0 / d_head)
```

This creates different rotation speeds for different dimensions. Some dimensions
rotate quickly, some slowly.

```python
positions = jnp.arange(context_len, dtype=jnp.float32)
```

This creates positions:

```text
[0, 1, 2, ..., 511]
```

```python
angles = positions[:, None] * freqs[None, :]
```

This creates one angle per position and per frequency.

```python
return jnp.cos(angles), jnp.sin(angles)
```

The model later uses these tables to rotate query and key vectors.

### `apply_rope`

```python
half = x.shape[-1] // 2
x_even, x_odd = x[..., :half], x[..., half:]
```

Split the vector into two halves.

```python
return jnp.concatenate([
    x_even * cos - x_odd * sin,
    x_even * sin + x_odd * cos,
], axis=-1)
```

Apply the rotation formula and join the two halves back together.

### `causal_attention`

This function receives one sequence of hidden vectors:

```text
x shape = (seq_len, d_model)
```

It computes:

```python
q = (x @ wq).reshape(seq_len, n_heads, d_head)
k = (x @ wk).reshape(seq_len, n_kv_heads, d_head)
v = (x @ wv).reshape(seq_len, n_kv_heads, d_head)
```

For the real model:

```text
q shape = (seq_len, 16, 64)
k shape = (seq_len, 4, 64)
v shape = (seq_len, 4, 64)
```

Then it applies RoPE to `q` and `k`, but not `v`. Positions matter for matching
queries to keys. Values are the content that gets mixed after attention weights
are known.

```python
jax.nn.dot_product_attention(..., is_causal=True)
```

This computes causal attention. On GPU with `bfloat16`, the code asks JAX to use
cuDNN FlashAttention.

Finally:

```python
return out.reshape(seq_len, d_model) @ wo
```

This joins all heads back into one vector per token and applies the output matrix.

### `_attn_layer`

This is one transformer block:

```python
h_norm = rms_norm(h, ln1_s)
attn_out = causal_attention(...)
h = h + attn_out
h_norm2 = rms_norm(h, ln2_s)
h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
return h + h_ff
```

In plain English:

```text
normalize
run attention
add attention result back
normalize again
run feed-forward network
add feed-forward result back
```

### `_transformer_trunk`

```python
h = params["token_emb"][x]
```

Turn token IDs into vectors.

```python
use_checkpoint = config.get("gradient_checkpoint", True)
maybe_checkpoint = jax.checkpoint if use_checkpoint else lambda f, **kw: f
```

Gradient checkpointing saves memory by recomputing some values during the
backward pass instead of storing them all.

Tradeoff:

```text
less memory, more compute
```

Then:

```python
for layer in range(config["n_layers"]):
    ...
```

Run the same kind of block 24 times, with different parameters for each layer.

```python
return rms_norm(h, params["ln_final.scale"])
```

Apply final normalization.

### `transformer_forward`

```python
h = _transformer_trunk(params, config, x)
return h @ params["token_emb"].T
```

This returns logits for every token position.

Shape:

```text
input x: (seq_len,)
hidden h: (seq_len, 1024)
token_emb.T: (1024, 32000)
logits: (seq_len, 32000)
```

### Inference Helpers In `model.py`

`prefill_with_kv` is used by generation code. It runs the model over a prompt and
returns key/value caches so decoding can continue one token at a time.

For pretraining, you can ignore it.

`transformer_forward_batch` is a convenience wrapper:

```python
return jax.vmap(lambda x: transformer_forward(params, config, x))(x_batch)
```

It takes the one-sequence forward pass and maps it over a batch.

### `transformer_loss_fused`

```python
h_batch = jax.vmap(lambda x: _transformer_trunk(params, config, x))(x_batch)
```

`_transformer_trunk` handles one sequence. `vmap` makes it handle a batch of
sequences.

```python
loss = fused_output_and_loss(h_batch, token_emb, targets, chunk_size)
```

This computes next-token cross-entropy without materializing all logits.

The optional MTP code is present, but the trained config here uses:

```text
n_mtp_heads = 0
```

So the extra MTP loop does not matter for the current model.

### `cross_entropy_loss`

```python
def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)
```

This is the simple version of cross-entropy. It is easier to read, but it creates
the full logits tensor first. The training code uses the fused version because
the full tensor is too large.

Line by line:

- `jax.nn.log_softmax` converts logits into log-probabilities.
- `take_along_axis` selects the log-probability of the correct target token at
  each position.
- `squeeze(-1)` removes the extra final dimension added for indexing.
- `-jnp.mean(...)` turns correct-token log-probabilities into average loss.

### `fused_cross_entropy`

This is the memory-saving loss path.

The forward pass:

```text
for each vocab chunk:
  compute logits for that chunk
  update the running max logit
  update the running exp sum
```

The backward pass recomputes chunks and accumulates gradients.

This is more complex than normal cross-entropy, but the reason is simple:

```text
the full logits tensor is too large to keep in memory.
```

### `count_params`

```python
def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))
```

`params` is a dictionary of arrays. `jax.tree.leaves(params)` returns all arrays
inside it. Each array has a `.size`, meaning number of scalar values. The function
adds those sizes to get the total parameter count.

## Reading `train.py`

`train.py` is the script that actually trained the model.

### Cache Setup

```python
_jax_cache = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(_jax_cache, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _jax_cache
os.environ.setdefault("JAX_COMPILATION_CACHE_MAX_SIZE", str(2 * 1024**3))
```

JAX compiles Python functions into optimized GPU programs. Compilation can be
slow. These lines cache compiled programs in `.jax_cache`.

The max cache size is:

$$
2 \times 1024^3
$$

Since:

$$
1024^3 = 1073741824
$$

Then:

$$
2 \times 1073741824 = 2147483648
$$

So the cache limit is about 2 GB.

### Imports

```python
import argparse
import pickle
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from data import load_data
from model import init_transformer, transformer_loss_fused, count_params
```

- `argparse` reads command-line flags like `--d-model`.
- `pickle` loads and saves checkpoints.
- `tempfile` creates temporary checkpoint files for atomic saves.
- `time` measures elapsed time and ETA.
- `numpy` handles CPU arrays, shuffling, memory maps, and batch assembly.
- `jax` moves data to GPU and compiles training steps.
- `jax.numpy` is the GPU/differentiable NumPy-like API.
- `optax` provides AdamW and learning-rate schedules.
- `load_data` loads tokenized data from `data.py`.
- `init_transformer` initializes a new model when not resuming.
- `transformer_loss_fused` computes the pretraining loss.
- `count_params` prints the model size.

### Arguments

The script requires model size settings:

```python
--d-model
--n-heads
--n-kv-heads
--n-layers
--context-len
--batch-size
--epochs
```

The important command for this project shape is:

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

The exact batch size depends on GPU memory. The large run used a big GPU. A local
16 GB card cannot use the same batch size.

### Loading Data

```python
data = load_data(context_len=args.context_len, data_dir=args.data_dir)
vocab_size = data["vocab_size"]
train_tokens = data["train_tokens"]
```

This loads metadata, validation arrays, and a memory-mapped training token stream.

```python
n_train_seqs = (len(train_tokens) - 1) // args.context_len
n_batches = n_train_seqs // args.batch_size
```

Example:

```text
n_train_seqs = 82493477
batch_size = 256
```

Then:

$$
{82493477 \over 256} = 322240.14
$$

Integer division gives:

```text
n_batches = 322240
```

That matches `training_v3.log`.

### Validation Batch

```python
val_x = jnp.array(data["val_x"][:args.batch_size])
val_y = jnp.array(data["val_y"][:args.batch_size])
```

This takes one validation batch and moves it into JAX arrays.

Important limitation:

```text
Validation loss is computed on this batch slice, not necessarily the full
validation set.
```

That is okay for quick tracking, but a serious final eval should average over
more validation batches.

### Total Steps

```python
total_steps = n_batches * args.epochs
```

For the v3 log:

```text
n_batches = 322240
epochs = 2
```

So:

$$
322240 \times 2 = 644480
$$

That matches:

```text
644480 total
```

### Resume Logic

If `--resume` is passed, the script loads an existing pickle:

```python
with open(args.resume, "rb") as f:
    ckpt = pickle.load(f)
```

Then:

```python
params = jax.tree.map(jnp.array, ckpt["params"])
config = ckpt["config"]
```

This loads model weights and config.

If optimizer state exists:

```python
resumed_opt_state = jax.tree.map(jnp.array, ckpt["opt_state"])
resume_step = ckpt["global_step"]
resume_epoch = ckpt["epoch"]
resume_bi = ckpt["batch_index"]
```

Then training can continue from the same optimizer momentum and same batch index.

If optimizer state does not exist, the script can still load weights, but AdamW
starts fresh.

### New Model Logic

If no resume path is passed:

```python
key, init_key = jax.random.split(jax.random.key(args.seed))
params, config = init_transformer(...)
```

This creates a new random model.

That is not what you want for SFT. For SFT, you want to resume from pretrained
weights.

### Optimizer

```python
schedule = optax.join_schedules(...)
optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
```

This builds the learning-rate schedule and AdamW optimizer.

```python
opt_state = resumed_opt_state if resumed_opt_state is not None else optimizer.init(params)
```

If the checkpoint has optimizer state, keep using it. Otherwise initialize a new
optimizer state.

### Train Step

```python
def make_train_step(phase_config):
```

This creates a compiled step for a specific context length.

```python
pc = {**config, "context_len": phase_config["ctx"]}
```

Copy config, but override context length for this curriculum phase.

```python
@jax.jit
def step(params, opt_state, x, y):
```

Compile the step function.

Inside:

```python
def loss_fn(params):
    params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
    return transformer_loss_fused(params_bf16, pc, x, y, ce_chunk)
```

This says:

```text
cast parameters to bfloat16 for model/loss computation
compute fused cross-entropy
```

Then:

```python
loss, grads = jax.value_and_grad(loss_fn)(params)
```

This computes both the loss and the gradient of the loss with respect to every
parameter.

Then:

```python
updates, opt_state = optimizer.update(grads, opt_state, params)
return optax.apply_updates(params, updates), opt_state, loss
```

AdamW converts gradients into updates. Then the updates are applied to parameters.

### Eval Loss

```python
@jax.jit
def eval_loss(params, x, y):
    params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
    return transformer_loss_fused(params_bf16, config, x, y, ce_chunk)
```

This is like training loss, but without gradients or optimizer updates.

### Batch Creation

```python
def _get_batch_streaming(seq_indices, ctx, bs, offset):
    indices = seq_indices[offset:offset + bs]
    batch = np.stack([train_tokens[i * args.context_len:i * args.context_len + ctx + 1]
                      for i in indices])
    return batch[:, :ctx], batch[:, 1:ctx + 1]
```

This is one of the most important pieces.

Suppose:

```text
ctx = 5
i = 10
args.context_len = 512
```

The script reads tokens starting at:

$$
i \times 512 = 10 \times 512 = 5120
$$

It reads `ctx + 1` tokens:

```text
tokens 5120 through 5125
```

If those tokens are:

```text
[10, 20, 30, 40, 50, 60]
```

Then:

```text
x = [10, 20, 30, 40, 50]
y = [20, 30, 40, 50, 60]
```

The extra token is needed because targets are shifted by one.

### Checkpoint Saving

```python
fd, tmp = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
...
os.replace(tmp, ckpt_path)
```

This writes to a temporary file first, then atomically replaces the checkpoint.

Why this matters:

```text
If the process crashes halfway through writing, it should not leave a corrupted
checkpoint.pkl.
```

### Main Epoch Loop

```python
for epoch in range(args.epochs):
```

Run each epoch.

```python
rng = np.random.default_rng(args.seed + epoch)
seq_perm = rng.permutation(n_train_seqs)
```

Shuffle sequence order differently each epoch.

```python
while bi < n_batches:
```

Loop through batches.

```python
cur_phase = next(p for p in phases if global_step < p["end"])
ctx = cur_phase["ctx"]
bs = args.batch_size * cur_phase["bs_mult"]
```

Pick the current curriculum phase.

If early phase says:

```text
ctx = 128
bs_mult = 4
base batch size = 256
```

Then:

$$
256 \times 4 = 1024
$$

So the model trains with shorter sequences but more examples per step.

Token count per step stays similar:

$$
1024 \times 128 = 131072
$$

For the full phase:

$$
256 \times 512 = 131072
$$

Same tokens per step, different sequence length.

### Prefetching

```python
next_bx = jax.device_put(jnp.array(bx_np))
next_by = jax.device_put(jnp.array(by_np))
```

This moves the next batch to the GPU.

The loop tries to prepare the next batch while the current batch is training.
That reduces waiting.

### One Training Step

```python
params, opt_state, loss = train_step(params, opt_state, bx, by)
```

This is the actual learning step.

Everything else in the script exists to feed this line and save the result.

### End Of Epoch

```python
vl = float(eval_loss(params, val_x, val_y))
avg_train = eloss / steps_this_epoch
print(...)
save_checkpoint(...)
```

Compute validation loss, print training stats, and save a checkpoint.

### Final Weights

```python
with open(save_path, "wb") as f:
    pickle.dump({"params": jax.tree.map(np.asarray, params), "config": config}, f)
```

At the end, the script saves a weights-only file.

Difference:

```text
checkpoint.pkl = weights + optimizer state + progress info
weights.pkl    = weights + config only
```

For continuing pretraining, use a checkpoint if you have it.

For generation or SFT initialization, weights-only can be enough, but SFT code
must initialize a new optimizer state.

## What The Model Has Not Learned Yet

Pretraining teaches:

```text
predict the next token in raw text
```

It does not specifically teach:

- follow a user/assistant chat format
- call tools in a strict format
- wait for tool results
- keep answers concise
- optimize for correctness under tests
- prefer verified answers over plausible text

That is why SFT can make sense.

But SFT should be small and controlled first.

## The Simplest Next Step

The clean path is:

```text
1. Freeze the current base model as the pretrained checkpoint.
2. Keep inference-kernel work out of the main README path.
3. Create a tiny SFT dataset of bash-agent traces.
4. Build `train_sft.py` with loss masking.
5. Run a tiny overfit test on 10 to 50 traces.
6. Only then generate a larger DeepSeek dataset.
```

Do not start with RL.

Do not start with 100,000 traces.

Do not change the tokenizer yet.

Do not add special tokens yet unless the plain-text markers fail.

Use plain markers that the tokenizer already knows:

```text
User:
Assistant:
[BASH]
[/BASH]
[BASH_RESULT]
[/BASH_RESULT]
```

That avoids resizing the embedding table.

## What Loss Masking Means For SFT

In SFT, the model should learn to produce assistant text and tool calls. It should
not learn to produce the user's prompt or the tool result.

Example training text:

```text
User:
Compute 2 + 2 with bash.

Assistant:
[BASH]
python - <<'PY'
print(2 + 2)
PY
[/BASH]

[BASH_RESULT]
4
[/BASH_RESULT]

Assistant:
4
```

The model should be trained on:

```text
assistant bash call
assistant final answer
```

The model should not be trained on:

```text
user prompt
tool result
```

So SFT needs a `loss_mask`.

For each token:

```text
loss_mask = 1 means train on this token
loss_mask = 0 means use this token as context only
```

The masked loss is:

$$
\text{masked loss} = {\sum \text{loss}_i \times \text{mask}_i \over \sum \text{mask}_i}
$$

Tiny example:

```text
token losses = [2.0, 1.0, 3.0, 4.0]
mask         = [0,   1,   0,   1]
```

Multiply loss by mask:

$$
2.0 \times 0 = 0
$$

$$
1.0 \times 1 = 1.0
$$

$$
3.0 \times 0 = 0
$$

$$
4.0 \times 1 = 4.0
$$

Sum masked losses:

$$
0 + 1.0 + 0 + 4.0 = 5.0
$$

Sum mask:

$$
0 + 1 + 0 + 1 = 2
$$

Average:

$$
{5.0 \over 2} = 2.5
$$

The model learns only from the two unmasked tokens.

## Is DeepSeek A Good Next Step?

Yes, but only as a controlled data generator after the local SFT path is clear.

As of May 5, 2026, the DeepSeek API docs list `deepseek-v4-pro` and
`deepseek-v4-flash` as current models. The older `deepseek-chat` and
`deepseek-reasoner` names are compatibility aliases and are documented as
deprecated on July 24, 2026.

Useful current docs:

- DeepSeek models and pricing: https://api-docs.deepseek.com/quick_start/pricing
- DeepSeek first API call: https://api-docs.deepseek.com/
- DeepSeek function calling: https://api-docs.deepseek.com/guides/function_calling/
- DeepSeek thinking mode: https://api-docs.deepseek.com/guides/thinking_mode
- DeepSeek changelog: https://api-docs.deepseek.com/updates

For this project, use DeepSeek like this:

```text
teacher model: deepseek-v4-pro or deepseek-v4-flash
tool setup: one bash tool
thinking: disabled at first
output: short traces with command, result, final answer
filtering: keep only traces that pass a verifier or judge
```

Why disable thinking at first?

Because the small model should learn a visible, simple action format. Hidden
reasoning fields make the data pipeline harder and are not the first thing to
teach.

## A Practical SFT Pilot

Start with a tiny dataset:

```text
10 traces    prove the training code can overfit
100 traces   prove formatting can be learned
1000 traces  prove the idea has some generalization
```

Good first tasks:

- compute a hash
- write a tiny Python file
- parse JSON
- run a simple unit test
- inspect a text file
- fix a one-line bug

Bad first tasks:

- full repo bug fixes
- long multi-file edits
- RL
- hidden chain-of-thought distillation
- huge synthetic datasets

The goal is not to make the 303M model suddenly become a strong coding agent.
The goal is to answer this narrow question:

```text
Can this pretrained base model learn the bash-agent format from clean SFT data?
```

If yes, scale carefully.

If no, the next fix is probably data format, context length, or model size, not
RL.

## What I Would Ignore For Now

Ignore these until the training/SFT path is understandable:

- custom Triton decode kernels
- profiling scripts
- speculative decoding ideas
- RL or GRPO
- large-scale DeepSeek generation
- tokenizer surgery
- adding new architecture features

They are interesting, but they are not the current bottleneck.

The current bottleneck is project clarity.

## Suggested Repo Focus

The README should present the repo as:

```text
A small pretrained JAX decoder-only transformer, with documentation for the
architecture and training loop, plus early experiments toward bash-agent SFT.
```

Inference kernels can be described as historical or optional experiments, not the
main first page.

The next useful file to create after this guide is:

```text
train_sft.py
```

It should be boring:

- load pretrained weights
- load JSONL traces
- tokenize plain-text markers
- create `x`, `y`, and `loss_mask`
- compute masked cross-entropy
- fine-tune with a small learning rate
- save `sft_weights.pkl`
- run a tiny overfit test first

## Definition Of Progress

This project is moving again when these are true:

```text
1. You can explain what every major part of `model.py` does.
2. You can explain why `x` and `y` are shifted by one token.
3. You can explain why cross-entropy is the right pretraining loss.
4. You can explain why the full logits tensor is too large.
5. You can run or inspect generation from the base model.
6. You can overfit a tiny SFT dataset.
7. You can compare base vs SFT outputs on a fixed prompt set.
```

That is enough structure. Everything else can wait.
