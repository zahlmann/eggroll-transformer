"""Download and prepare character-level Shakespeare dataset."""

import os
import urllib.request
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print("Downloading tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r") as f:
        text = f.read()
    return text


def prepare_data(context_len=128, val_fraction=0.1):
    text = download_shakespeare()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}

    data = np.array([char_to_idx[c] for c in text], dtype=np.int32)

    n = len(data)
    split = int(n * (1 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]

    # create sequences: input[i] -> target[i] = input[i+1]
    def make_sequences(arr):
        n_seq = (len(arr) - 1) // context_len
        arr = arr[: n_seq * context_len + 1]
        x = arr[:-1].reshape(n_seq, context_len)
        y = arr[1:].reshape(n_seq, context_len)
        return x, y

    train_x, train_y = make_sequences(train_data)
    val_x, val_y = make_sequences(val_data)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": vocab_size,
        "chars": chars,
        "char_to_idx": char_to_idx,
    }


if __name__ == "__main__":
    d = prepare_data()
    print(f"Vocab size: {d['vocab_size']}")
    print(f"Train sequences: {d['train_x'].shape}")
    print(f"Val sequences: {d['val_x'].shape}")
    print(f"Characters: {''.join(d['chars'][:20])}...")
