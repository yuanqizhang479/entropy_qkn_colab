"""Pinned public corpora, explicit provenance, and reproducible token batches.

WikiText rows are lines, NOT documents: reconstruct articles before inserting EOS.
No executable remote dataset scripts and no credential are required.
"""
from __future__ import annotations

import argparse
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import urllib.parse
import urllib.request

import numpy as np

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_ROOT = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
                 f"{WIKITEXT_REVISION}/wikitext-103-raw-v1/")
WIKITEXT_FILES = {
    "train": [
        ("train-00000-of-00002.parquet", "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
        ("train-00001-of-00002.parquet", "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"),
    ],
    "validation": [("validation-00000-of-00001.parquet", "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c")],
    "test": [("test-00000-of-00001.parquet", "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91")],
}
TOKENIZER_ASSETS = {
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe":
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json":
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, dest, expected_sha256=None, expected_md5=None):
    """Atomic verified download; cached files are verified on every reuse."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def valid(path):
        if not path.is_file():
            return False
        if expected_sha256 and sha256_file(path) != expected_sha256:
            return False
        if expected_md5:
            h = hashlib.md5(usedforsecurity=False)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if base64.b64encode(h.digest()).decode() != expected_md5:
                return False
        return True

    if valid(dest):
        return dest
    error = None
    for attempt in range(3):
        partial = dest.with_name(dest.name + ".partial")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "entropy-qkn-research/1.0"})
            with urllib.request.urlopen(request, timeout=90) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if not valid(partial):
                raise ValueError(f"Checksum mismatch for {url}; refusing changed source.")
            os.replace(partial, dest)
            return dest
        except Exception as exc:
            error = exc
            partial.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Download failed: {url}. See docs/DATA.md for manual download.") from error


def _source_file(filename, cache_dir, offline_dir, url, sha256=None, md5=None):
    if offline_dir:
        source = Path(offline_dir) / filename
        if not source.is_file():
            # Allow original nested dataset layout as well as flat files.
            source = Path(offline_dir) / "wikitext-103-raw-v1" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required manual source missing: {filename}")
        if sha256 and sha256_file(source) != sha256:
            raise ValueError(f"Manual file checksum mismatch: {source}")
        if md5:
            digest = hashlib.md5(source.read_bytes(), usedforsecurity=False).digest()
            if base64.b64encode(digest).decode() != md5:
                raise ValueError(f"Manual file MD5 mismatch: {source}")
        return source
    return _download(url, Path(cache_dir) / filename, sha256, md5)


def _top_heading(line):
    line = line.strip()
    if not (line.startswith("= ") and line.endswith(" =")):
        return False
    title = line[2:-2].strip()
    return bool(title) and not title.startswith("=") and not title.endswith("=")


def wikitext_articles(rows):
    """Join original rows within a full article, preserving their text exactly."""
    # Math continuation lines can also look like '= something ='. Actual article
    # titles in the pinned corpus are isolated by empty rows on both sides.
    def text_rows():
        for row in rows:
            yield row["text"] if isinstance(row, dict) else str(row)

    iterator = iter(text_rows())
    previous = ""
    line = next(iterator, None)
    parts = []
    while line is not None:
        following = next(iterator, None)
        is_boundary = _top_heading(line) and not previous.strip() and (following is None or not following.strip())
        if is_boundary and parts:
            text = "".join(parts)
            if text.strip():
                yield text
            parts = []
        parts.append(line)
        previous, line = line, following
    text = "".join(parts)
    if text.strip():
        yield text


def _parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
            yield from batch.column(0).to_pylist()


def _encoding(offline_dir=None):
    import tiktoken
    if offline_dir:
        # tiktoken cache names are SHA1(URL), while content is checked with SHA256.
        cache = Path(offline_dir) / "tiktoken_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for url, expected in TOKENIZER_ASSETS.items():
            source = Path(offline_dir) / url.rsplit("/", 1)[-1]
            target = cache / hashlib.sha1(url.encode()).hexdigest()
            if source.is_file():
                if sha256_file(source) != expected:
                    raise ValueError(f"Tokenizer checksum mismatch: {source}")
                shutil.copyfile(source, target)
            elif not target.is_file():
                raise FileNotFoundError(f"Manual tokenizer source missing: {source}")
            elif sha256_file(target) != expected:
                raise ValueError(f"Tokenizer cache checksum mismatch: {target}")
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    else:
        cache = Path(os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache"))))
        for url, expected in TOKENIZER_ASSETS.items():
            _download(url, cache / hashlib.sha1(url.encode()).hexdigest(), expected_sha256=expected)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    enc = tiktoken.get_encoding("gpt2")
    if enc.n_vocab != 50257 or enc.eot_token != 50256:
        raise ValueError("Unexpected GPT-2 encoding.")
    return enc


def encode_documents(documents, encoder, out_dir, split, max_tokens=0, seen=None, excluded_hashes=None):
    """Hash ALL source docs; encode only the deterministic prefix requested.

    Full-document overlap is audited even after a token budget is exhausted.
    This catches exact duplicates only, never claims to exclude near duplicates.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be nonnegative (0 means all).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = {} if seen is None else seen
    count = 0
    source_docs = 0
    encoded_docs = 0
    within_duplicates = 0
    cross_duplicates = []
    excluded_documents = []
    excluded_hashes = set() if excluded_hashes is None else excluded_hashes
    with (out_dir / f"{split}.bin").open("wb") as binary, (out_dir / f"{split}.documents.jsonl").open("w", encoding="utf-8") as ledger:
        for index, item in enumerate(documents):
            source_id, text = item if isinstance(item, tuple) else (str(index), item)
            if not text.strip():
                continue
            source_docs += 1
            digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            if digest in excluded_hashes:
                excluded_documents.append({"source_id": source_id, "sha256": digest, "characters": len(text)})
                ledger.write(json.dumps({"source_id": source_id, "sha256_text_stripped": digest,
                                         "characters": len(text), "token_start": count,
                                         "token_end": count, "excluded_heldout_exact_duplicate": True}) + "\n")
                continue
            previous = seen.get(digest)
            if previous and previous != split:
                cross_duplicates.append({"sha256": digest, "other_split": previous, "source_id": source_id})
            elif previous == split:
                within_duplicates += 1
            seen.setdefault(digest, split)
            start = count
            truncated = False
            if not max_tokens or count < max_tokens:
                tokens = encoder.encode_ordinary(text) + [encoder.eot_token]
                if tokens and (min(tokens) < 0 or max(tokens) >= encoder.n_vocab or max(tokens) > 65535):
                    raise ValueError("Token IDs outside declared uint16 vocabulary.")
                if max_tokens and len(tokens) > max_tokens - count:
                    tokens = tokens[:max_tokens - count]
                    truncated = True
                np.asarray(tokens, dtype="<u2").tofile(binary)
                count += len(tokens)
                encoded_docs += 1
            ledger.write(json.dumps({"source_id": source_id, "sha256_text_stripped": digest,
                                     "characters": len(text), "token_start": start,
                                     "token_end": count, "prefix_truncated": truncated}) + "\n")
    if count == 0:
        raise ValueError(f"No tokens produced for {split}.")
    return {"file": f"{split}.bin", "tokens": count, "bytes": count * 2,
            "sha256": sha256_file(out_dir / f"{split}.bin"),
            "documents_file": f"{split}.documents.jsonl",
            "documents_sha256": sha256_file(out_dir / f"{split}.documents.jsonl"),
            "source_documents": source_docs, "encoded_documents": encoded_docs,
            "within_split_exact_duplicates": within_duplicates,
            "cross_split_exact_duplicates": cross_duplicates, "max_tokens": max_tokens,
            "excluded_heldout_exact_duplicates": excluded_documents}


def _versions():
    result = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ["tiktoken", "pyarrow"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def prepare(args):
    out = Path(args.out).resolve()
    if out.exists() and not args.force:
        raise FileExistsError(f"Output exists: {out}; use verify or --force to deliberately replace it.")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=out.name + ".preparing-", dir=out.parent))
    try:
        manifest = {"schema_version": 1, "dataset": "synthetic" if args.synthetic else args.dataset,
                    "synthetic": bool(args.synthetic), "dtype": "<u2", "splits": {},
                    "versions": _versions(), "split_policy": "canonical source split; no random resplit",
                    "token_budget_policy": "deterministic source-order prefix; never shuffle before selection",
                    "text_policy": "source text preserved; GPT-2 EOS between full articles/books; final prefix may truncate a document",
                    "exact_overlap_audit": "full document SHA256 of UTF-8 text.strip(); near-duplicate/semantic overlap not tested",
                    "sources": []}
        if args.synthetic:
            if not 2 <= args.synthetic_vocab_size <= 65536:
                raise ValueError("Synthetic vocab must be between 2 and 65536.")
            manifest.update(vocab_size=args.synthetic_vocab_size, tokenizer="synthetic integer tokens", paper_eligible=False)
            for offset, split in enumerate(["train", "validation", "test"]):
                n = args.max_train_tokens if split == "train" else args.max_eval_tokens
                n = n or (8192 if split == "train" else 2048)
                tokens = np.random.default_rng(1000 + offset).integers(0, args.synthetic_vocab_size, n, dtype=np.uint16)
                tokens.astype("<u2").tofile(temp / f"{split}.bin")
                manifest["splits"][split] = {"file": f"{split}.bin", "tokens": n, "bytes": 2 * n,
                                              "sha256": sha256_file(temp / f"{split}.bin")}
        else:
            encoder = _encoding(args.offline_dir)
            manifest.update(vocab_size=encoder.n_vocab, eos_token=encoder.eot_token,
                            tokenizer={"name": "gpt2", "implementation": "tiktoken", "assets_sha256": TOKENIZER_ASSETS},
                            paper_eligible=True)
            seen = {}
            if args.dataset == "wikitext103":
                manifest.update(dataset_repo="Salesforce/wikitext", revision=WIKITEXT_REVISION,
                                config="wikitext-103-raw-v1", license="CC BY-SA 3.0 / GFDL (source card)",
                                split_policy="official validation/test unchanged; remove full-article exact heldout duplicates from official train before prefix selection",
                                benchmark_variant="WikiText-103 raw, GPT-2 BPE, training exact-deduplicated against canonical heldouts")
                # Audit both full held-out splits before any training prefix is chosen.
                for split in ["validation", "test", "train"]:
                    print(f"Preparing WikiText-103 {split}: verify source files, audit full articles, encode prefix.", flush=True)
                    paths = []
                    for filename, digest in WIKITEXT_FILES[split]:
                        url = WIKITEXT_ROOT + filename
                        path = _source_file(filename, args.cache_dir, args.offline_dir, url, sha256=digest)
                        paths.append(path)
                        manifest["sources"].append({"url": url, "sha256": digest, "bytes": path.stat().st_size})
                    budget = args.max_train_tokens if split == "train" else args.max_eval_tokens
                    excluded = set(seen) if split == "train" else set()
                    manifest["splits"][split] = encode_documents(wikitext_articles(_parquet_rows(paths)), encoder, temp, split, budget, seen, excluded)
            else:
                source_list = Path(__file__).resolve().parents[1] / "configs" / "pg19_test_sources.json"
                snapshot = json.loads(source_list.read_text())
                manifest.update(dataset_repo="google-deepmind/pg19", revision="GCS per-object generation IDs in sources",
                                split_policy="official PG-19 test only; never used for training or tuning",
                                source_list_sha256=sha256_file(source_list), license="Apache-2.0 (benchmark repository); retain source notices")
                def load_book(item):
                    name = item["name"]
                    url = f"https://storage.googleapis.com/deepmind-gutenberg/{name}?generation={item['generation']}"
                    path = _source_file(name, Path(args.cache_dir) / "pg19", args.offline_dir, url, md5=item["md5_base64"])
                    if path.stat().st_size != item["bytes"]:
                        raise ValueError(f"Source size mismatch: {name}")
                    return {**item, "url": url, "sha256": sha256_file(path)}, path

                def books():
                    # executor.map preserves the pinned input order, unlike as_completed.
                    # Only small text files are downloaded concurrently; tokenization stays serial.
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        for book_index, (source, path) in enumerate(pool.map(load_book, snapshot["objects"])):
                            if book_index % 20 == 0:
                                print(f"Preparing PG-19 test: book {book_index + 1}/{len(snapshot['objects'])}.", flush=True)
                            manifest["sources"].append(source)
                            yield source["name"], path.read_text(encoding="utf-8")
                manifest["splits"]["test"] = encode_documents(books(), encoder, temp, "test", args.max_eval_tokens, seen)
            overlaps = [entry for info in manifest["splits"].values() for entry in info["cross_split_exact_duplicates"]]
            if overlaps:
                raise ValueError(f"Found {len(overlaps)} exact documents crossing official splits; inspect corpus before training.")
            manifest["exact_cross_split_document_duplicates"] = 0
            manifest["source_training_articles_excluded_as_exact_heldout_duplicates"] = len(manifest["splits"].get("train", {}).get("excluded_heldout_exact_duplicates", []))
            manifest["full_source_train_heldout_exact_duplicate_articles"] = manifest["source_training_articles_excluded_as_exact_heldout_duplicates"]
            manifest["after_filter_cross_split_exact_document_duplicates"] = 0
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        verify(temp)
        if out.exists():
            shutil.rmtree(out)
        os.replace(temp, out)
        print(json.dumps({"prepared": str(out), "dataset": manifest["dataset"],
                          "synthetic": manifest["synthetic"], "tokens": {k: v["tokens"] for k, v in manifest["splits"].items()}}, indent=2))
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def verify(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["dtype"] != "<u2":
        raise ValueError("Unsupported token dtype.")
    if not isinstance(manifest.get("synthetic"), bool):
        raise ValueError("Manifest must explicitly declare synthetic=true/false.")
    if not manifest["synthetic"]:
        tokenizer = manifest.get("tokenizer", {})
        if not isinstance(tokenizer, dict) or tokenizer.get("name") != "gpt2" or tokenizer.get("assets_sha256") != TOKENIZER_ASSETS:
            raise ValueError("Real corpus must declare the pinned GPT-2 tokenizer assets.")
        if manifest.get("vocab_size") != 50257 or manifest.get("eos_token") != 50256:
            raise ValueError("Real corpus vocabulary/EOS does not match pinned GPT-2.")
    for split, info in manifest["splits"].items():
        path = data_dir / info["file"]
        if path.stat().st_size != info["tokens"] * 2 or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Token byte count mismatch: {split}")
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"Token checksum mismatch: {split}")
        tokens = np.memmap(path, dtype="<u2", mode="r")
        if len(tokens) == 0 or int(tokens.max()) >= manifest["vocab_size"]:
            raise ValueError(f"Invalid token range: {split}")
        if info.get("documents_file") and sha256_file(data_dir / info["documents_file"]) != info["documents_sha256"]:
            raise ValueError(f"Document ledger checksum mismatch: {split}")
    return manifest


class TokenBatcher:
    """Random contiguous token windows with serializable independent NumPy RNG.

    Windows may cross EOS, as in packed causal LM training. x/y are shifted by
    exactly one token. RNG is never taken from model/global RNG.
    """
    def __init__(self, data_dir, split, batch_size, seq_len, seed, device="cpu"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = device
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        self.info = self.manifest["splits"][split]
        self.data = np.memmap(self.data_dir / self.info["file"], dtype="<u2", mode="r")
        if self.batch_size < 1 or self.seq_len < 1 or len(self.data) <= self.seq_len:
            raise ValueError("Positive batch/sequence lengths and at least seq_len+1 tokens required.")
        self.rng = np.random.default_rng(seed)

    def next(self):
        import torch
        starts = self.rng.integers(0, len(self.data) - self.seq_len, size=self.batch_size)
        block = np.stack([self.data[i:i + self.seq_len + 1] for i in starts]).astype(np.int64)
        x = torch.from_numpy(block[:, :-1].copy()).to(self.device)
        y = torch.from_numpy(block[:, 1:].copy()).to(self.device)
        return x, y

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "split": self.split,
                "data_sha256": self.info["sha256"], "batch_size": self.batch_size, "seq_len": self.seq_len}

    def load_state_dict(self, state):
        for name in ["split", "batch_size", "seq_len"]:
            if state[name] != getattr(self, name):
                raise ValueError(f"Sampler state mismatch for {name}.")
        if state["data_sha256"] != self.info["sha256"]:
            raise ValueError("Sampler checkpoint belongs to different token data.")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dataset", choices=["wikitext103", "pg19"], default="wikitext103")
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=".cache/qknlab-sources")
    p.add_argument("--offline-dir", help="Manual source folder; no downloads. Include tokenizer files.")
    p.add_argument("--max-train-tokens", type=int, default=20000000)
    p.add_argument("--max-eval-tokens", type=int, default=0, help="0 means entire official eval split")
    p.add_argument("--synthetic", action="store_true", help="SMOKE TEST ONLY; never paper evidence")
    p.add_argument("--synthetic-vocab-size", type=int, default=128)
    p.add_argument("--force", action="store_true")
    v = commands.add_parser("verify")
    v.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.max_train_tokens < 0 or args.max_eval_tokens < 0:
            parser.error("Token budgets must be >=0.")
        prepare(args)
    else:
        manifest = verify(args.data_dir)
        print(json.dumps({"verified": str(Path(args.data_dir).resolve()), "synthetic": manifest["synthetic"]}))


if __name__ == "__main__":
    main()
