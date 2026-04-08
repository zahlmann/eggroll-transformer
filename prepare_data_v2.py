"""Prepare high-quality training data v2.

Data mix:
  34% FineWeb-Edu (score>=3) — quality-filtered web
  30% StarCoder code         — deduplicated, 13 languages
  19% OpenWebMath            — math with LaTeX
   9% Wikipedia              — encyclopedia
   8% Cosmopedia v2          — synthetic textbooks

Usage:
  uv run prepare_data_v2.py                    # download + tokenize everything
  uv run prepare_data_v2.py --tokenize-only    # just retokenize existing raw data
  uv run prepare_data_v2.py --stats            # show data statistics
"""

import json
import os
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
TOKEN_DIR = DATA_DIR / "tokens_v2"
VOCAB_SIZE = 32000
VAL_FRACTION = 0.005
EOS_TOKEN_ID = 1

SOURCES = {
    "fineweb_edu":  {"tokens": 3_500_000_000, "chars_per_tok": 3.5, "min_len": 100},
    "wikipedia":    {"tokens": 800_000_000,   "chars_per_tok": 3.5, "min_len": 200},
    "cosmopedia":   {"tokens": 900_000_000,   "chars_per_tok": 3.5, "min_len": 100},
    "starcoderdata":{"tokens": 2_500_000_000, "chars_per_tok": 3.0, "min_len": 50},
    "openwebmath":  {"tokens": 1_500_000_000, "chars_per_tok": 3.5, "min_len": 100},
}

# existing raw files to dedup from (per source)
EXISTING_FILES = {
    "fineweb_edu":   ["fineweb_edu.jsonl", "fineweb_edu_e2.jsonl"],
    "wikipedia":     ["wikipedia.jsonl", "wikipedia_e2.jsonl"],
    "cosmopedia":    ["cosmopedia.jsonl", "cosmopedia_e2.jsonl"],
    "starcoderdata": ["code.jsonl", "code_e2.jsonl"],
    "openwebmath":   [],
}


def _dedup_key(doc, source):
    """Extract dedup key from doc."""
    if source == "wikipedia":
        return doc.get("title") or doc.get("text", "")[:200]
    return doc.get("text", doc.get("content", ""))[:200]


def _read_existing(source, seen, fout):
    """Read existing raw files, dedup, write to output. Returns (n_docs, n_chars)."""
    n_docs = n_chars = 0
    cfg = SOURCES[source]
    for fname in EXISTING_FILES[source]:
        p = RAW_DIR / fname
        if not p.exists():
            continue
        print(f"  Reading {fname}...")
        with open(p) as fin:
            for line in fin:
                doc = json.loads(line)
                text = doc.get("text", doc.get("content", ""))
                if len(text) < cfg["min_len"]:
                    continue
                key = _dedup_key(doc, source)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "source": source}) + "\n")
                n_docs += 1
                n_chars += len(text)
    return n_docs, n_chars


def _hf_stream(source):
    """Return HuggingFace streaming dataset iterator for a source."""
    from datasets import load_dataset
    if source == "fineweb_edu":
        return load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                            split="train", streaming=True)
    if source == "wikipedia":
        return load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
    if source == "cosmopedia":
        return load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                            split="train", streaming=True)
    if source == "openwebmath":
        return load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    assert False, f"unknown source: {source}"


def _download_source(source):
    """Download a single source, deduplicating against existing raw files."""
    cfg = SOURCES[source]
    out_path = RAW_DIR / f"{source}_all.jsonl"
    max_chars = cfg["tokens"] * cfg["chars_per_tok"]

    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * (max_chars / cfg["tokens"]) / cfg["chars_per_tok"]
        print(f"{source}: already have {n:,} docs")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()

    with open(out_path, "w") as fout:
        n_docs, n_chars = _read_existing(source, seen, fout)
        if n_docs:
            print(f"  Existing: {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens")

        if n_chars < max_chars:
            print(f"  Downloading {source}...")
            t0 = time.time()
            ds = _hf_stream(source)
            for doc in ds:
                if source == "fineweb_edu" and doc.get("score", 0) < 3.0:
                    continue
                text = doc.get("text", doc.get("content", ""))
                if len(text) < cfg["min_len"]:
                    continue
                key = _dedup_key(doc, source)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "source": source}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 50000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens, "
                          f"{time.time()-t0:.0f}s")
                if n_chars >= max_chars:
                    break

    del seen
    print(f"  {source}: {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens")
    return out_path


def _download_starcoderdata():
    """Download StarCoder — needs special handling for multi-language iteration."""
    from datasets import load_dataset

    cfg = SOURCES["starcoderdata"]
    out_path = RAW_DIR / "starcoderdata_all.jsonl"
    max_chars = cfg["tokens"] * cfg["chars_per_tok"]

    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"StarCoder: already have {n:,} docs")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    langs = ["python", "javascript", "typescript", "java", "c", "cpp",
             "rust", "go", "shell", "sql", "html", "css", "markdown"]

    with open(out_path, "w") as fout:
        n_docs, n_chars = _read_existing("starcoderdata", seen, fout)
        if n_docs:
            print(f"  Existing code: {n_docs:,} docs, ~{n_chars/3.0/1e9:.2f}B tokens")

        print(f"  Downloading StarCoder ({len(langs)} languages)...")
        t0 = time.time()
        for lang in langs:
            if n_chars >= max_chars:
                break
            try:
                ds = load_dataset("bigcode/starcoderdata", data_dir=lang,
                                  split="train", streaming=True)
                lang_docs = 0
                for doc in ds:
                    text = doc.get("content", "")
                    if len(text) < 50 or len(text) > 100000:
                        continue
                    key = text[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps({"text": text, "source": "starcoderdata"}) + "\n")
                    n_docs += 1
                    n_chars += len(text)
                    lang_docs += 1
                    if n_docs % 100000 == 0:
                        print(f"    {n_docs:,} docs ({lang}: {lang_docs:,}), "
                              f"~{n_chars/3.0/1e9:.2f}B tokens, {time.time()-t0:.0f}s")
                    if n_chars >= max_chars:
                        break
                print(f"    {lang}: {lang_docs:,} docs")
            except Exception as e:
                print(f"    {lang}: FAILED ({e})")

    del seen
    print(f"  StarCoder total: {n_docs:,} docs, ~{n_chars/3.0/1e9:.2f}B tokens")
    return out_path


def download_all():
    """Download all data sources."""
    print("=" * 60)
    print("Downloading all data sources")
    print("=" * 60)
    paths = {}
    for source in ["fineweb_edu", "wikipedia", "cosmopedia", "openwebmath"]:
        paths[source] = _download_source(source)
    paths["starcoderdata"] = _download_starcoderdata()
    return paths


def train_tokenizer():
    """Train BPE tokenizer on combined corpus sample."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok_path = DATA_DIR / f"tokenizer_{VOCAB_SIZE}.json"
    if tok_path.exists():
        print(f"Tokenizer already exists: {tok_path}")
        return tok_path

    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    sample_chars = 100_000_000
    texts = []
    for f in RAW_DIR.glob("*_all.jsonl"):
        chars = 0
        with open(f) as fh:
            for line in fh:
                doc = json.loads(line)
                texts.append(doc.get("text", ""))
                chars += len(texts[-1])
                if chars >= sample_chars:
                    break
        print(f"  Sampled {len(texts)} docs from {f.name} ({chars/1e6:.0f}M chars)")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(tok_path))
    print(f"Tokenizer saved: {tok_path} (vocab={tokenizer.get_vocab_size()})")
    return tok_path


def _tokenize_source(source_name, raw_path, tok_path, target_tokens):
    """Tokenize a source with EOS tokens between documents."""
    from tokenizers import Tokenizer

    cache_path = TOKEN_DIR / f"{source_name}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        n = len(data["train"]) + len(data["val"])
        print(f"  {source_name}: cached, {n/1e9:.2f}B tokens")
        return source_name, n

    tok = Tokenizer.from_file(str(tok_path))
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Tokenizing {source_name} (target: {target_tokens/1e9:.1f}B tokens)...")
    t0 = time.time()

    token_chunks = []
    total_tokens = 0
    batch_texts = []
    batch_chars = 0

    with open(raw_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", doc.get("content", ""))
            if len(text) < 50:
                continue
            batch_texts.append(text)
            batch_chars += len(text)

            if batch_chars >= 50_000_000:
                encodings = tok.encode_batch(batch_texts)
                for enc in encodings:
                    chunk_arr = np.array(enc.ids + [EOS_TOKEN_ID], dtype=np.int32)
                    token_chunks.append(chunk_arr)
                    total_tokens += len(chunk_arr)
                batch_texts = []
                batch_chars = 0
                print(f"    {source_name}: {total_tokens/1e6:.0f}M tokens, {time.time()-t0:.0f}s")
                if total_tokens >= target_tokens:
                    break

    if batch_texts:
        encodings = tok.encode_batch(batch_texts)
        for enc in encodings:
            chunk_arr = np.array(enc.ids + [EOS_TOKEN_ID], dtype=np.int32)
            token_chunks.append(chunk_arr)
            total_tokens += len(chunk_arr)

    tokens = np.concatenate(token_chunks)
    del token_chunks

    n_val = max(int(len(tokens) * VAL_FRACTION), 10000)
    np.savez(cache_path, train=tokens[n_val:], val=tokens[:n_val])
    print(f"    {source_name}: {(len(tokens)-n_val)/1e6:.1f}M train + {n_val/1e6:.1f}M val "
          f"({time.time()-t0:.0f}s)")
    return source_name, len(tokens)


def tokenize_all():
    """Tokenize all sources and combine into shuffled flat binary."""
    tok_path = train_tokenizer()

    source_raw = {
        "fineweb_edu":   RAW_DIR / "fineweb_edu_all.jsonl",
        "wikipedia":     RAW_DIR / "wikipedia_all.jsonl",
        "cosmopedia":    RAW_DIR / "cosmopedia_all.jsonl",
        "starcoderdata": RAW_DIR / "starcoderdata_all.jsonl",
        "openwebmath":   RAW_DIR / "openwebmath.jsonl",
    }

    print("\n--- Tokenizing ---")
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    source_stats = {}

    for name, raw_path in source_raw.items():
        if not raw_path.exists():
            print(f"  {name}: SKIPPED (no raw data)")
            continue
        _, n_tokens = _tokenize_source(name, raw_path, str(tok_path), SOURCES[name]["tokens"])
        source_stats[name] = n_tokens

    # combine val (small, fits in memory)
    print("\n--- Combining ---")
    val_parts = []
    for name in source_raw:
        cache = TOKEN_DIR / f"{name}.npz"
        if cache.exists():
            data = np.load(cache)
            print(f"  {name}: {len(data['train'])/1e9:.2f}B train, {len(data['val'])/1e6:.1f}M val")
            val_parts.append(data["val"])
            del data
    np.save(TOKEN_DIR / "val.npy", np.concatenate(val_parts))
    del val_parts

    # write train to flat binary
    train_bin = TOKEN_DIR / "train.bin"
    print("Writing train data to flat binary...")
    with open(train_bin, "wb") as f:
        for name in source_raw:
            cache = TOKEN_DIR / f"{name}.npz"
            if cache.exists():
                arr = np.load(cache)["train"]
                arr.tofile(f)
                print(f"  wrote {name}: {len(arr)/1e9:.2f}B tokens")
                del arr

    # shuffle via memmap
    print("Shuffling (memory-mapped)...")
    src = np.memmap(train_bin, dtype=np.int32, mode="r")
    chunk_size = 512
    n_chunks = len(src) // chunk_size
    usable = n_chunks * chunk_size

    perm = np.random.default_rng(42).permutation(n_chunks)
    shuffled_bin = TOKEN_DIR / "train_shuffled.bin"
    dst = np.memmap(shuffled_bin, dtype=np.int32, mode="w+", shape=(usable,))
    batch = 100000
    for i in range(0, n_chunks, batch):
        end = min(i + batch, n_chunks)
        for j, ci in enumerate(perm[i:end]):
            dst[(i+j)*chunk_size:(i+j+1)*chunk_size] = src[ci*chunk_size:(ci+1)*chunk_size]
        if (i // batch) % 10 == 0:
            print(f"  shuffled {end}/{n_chunks} chunks ({end*chunk_size/1e9:.2f}B tokens)")
    dst.flush()
    del src, dst

    train_bin.unlink()
    shuffled_bin.rename(train_bin)

    total_train = usable
    total_val = len(np.load(TOKEN_DIR / "val.npy"))

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer_path": str(tok_path.relative_to(Path(__file__).parent)),
        "sources": source_stats,
        "total_train_tokens": total_train,
        "total_val_tokens": total_val,
        "val_fraction": VAL_FRACTION,
        "eos_token_id": EOS_TOKEN_ID,
        "has_eos_between_docs": True,
        "format": "flat_binary",
    }
    with open(TOKEN_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL DATASET:")
    print(f"  Train: {total_train:,} tokens ({total_train/1e9:.2f}B)")
    print(f"  Val:   {total_val:,} tokens ({total_val/1e6:.1f}M)")
    for name, n in source_stats.items():
        pct = n / sum(source_stats.values()) * 100
        print(f"  {name}: {n/1e9:.2f}B ({pct:.0f}%)")
    print(f"{'='*60}")


def show_stats():
    """Show current data statistics."""
    print("=== Raw data ===")
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        n = sum(1 for _ in open(f))
        size = f.stat().st_size / 1e9
        print(f"  {f.name}: {n:,} docs, {size:.2f} GB")

    meta_path = TOKEN_DIR / "metadata.json"
    if not meta_path.exists():
        return
    print("\n=== Tokenized (v2) ===")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  Total train: {meta['total_train_tokens']/1e9:.2f}B tokens")
    print(f"  Total val: {meta['total_val_tokens']/1e6:.1f}M tokens")
    print(f"  Vocab: {meta['vocab_size']}")
    for name, n in meta["sources"].items():
        pct = n / sum(meta["sources"].values()) * 100
        print(f"    {name}: {n/1e9:.2f}B ({pct:.0f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.tokenize_only:
        tokenize_all()
    else:
        download_all()
        tokenize_all()
