import json
from pathlib import Path

import numpy as np
import pytest
import torch

from qknlab.data import TokenBatcher, encode_documents, sha256_file, verify, wikitext_articles


class FakeEncoder:
    n_vocab = 257
    eot_token = 256

    @staticmethod
    def encode_ordinary(text):
        return list(text.encode("utf-8"))


def fixture_data(tmp_path, n=128):
    splits = {}
    for i, split in enumerate(["train", "validation", "test"]):
        path = tmp_path / f"{split}.bin"
        (np.arange(n, dtype=np.uint16) + i).astype("<u2").tofile(path)
        splits[split] = {"file": path.name, "tokens": n, "bytes": 2*n, "sha256": sha256_file(path)}
    manifest = {"dtype": "<u2", "synthetic": True, "vocab_size": 257, "splits": splits}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_wikitext_article_boundaries_are_not_line_boundaries():
    rows = ["", " = Article one = \n", "", "A paragraph.\n", "", " = = Section = = \n", "", "Another paragraph.\n", "", " = Article two = \n", "", "Second article.\n"]
    docs = list(wikitext_articles(rows))
    assert len(docs) == 2
    assert "Section" in docs[0]
    assert "Another paragraph" in docs[0]
    assert docs[1].startswith(" = Article two")


def test_formula_continuation_is_not_an_article_title():
    rows = ["", " = Constant k filter = \n", "", "The frequency is omega\n", " = 1 rad / s and a nominal impedance k = \n", "1 ohm.\n", "", " = Next article = \n", "", "Body\n"]
    docs = list(wikitext_articles(rows))
    assert len(docs) == 2
    assert "nominal impedance" in docs[0]


def test_document_eos_and_prefix_budget(tmp_path):
    info = encode_documents(["abc", "def"], FakeEncoder(), tmp_path, "train", max_tokens=6)
    assert np.fromfile(tmp_path / "train.bin", dtype="<u2").tolist() == [97, 98, 99, 256, 100, 101]
    assert info["source_documents"] == 2
    ledger = [json.loads(x) for x in (tmp_path / "train.documents.jsonl").read_text().splitlines()]
    assert ledger[1]["prefix_truncated"] is True


def test_overlap_audit_continues_after_token_cap(tmp_path):
    seen = {}
    encode_documents(["first", "overlap"], FakeEncoder(), tmp_path, "train", max_tokens=2, seen=seen)
    info = encode_documents(["overlap"], FakeEncoder(), tmp_path, "test", seen=seen)
    assert len(info["cross_split_exact_duplicates"]) == 1


def test_heldout_duplicate_removed_before_prefix_selection(tmp_path):
    seen = {}
    encode_documents(["heldout"], FakeEncoder(), tmp_path, "test", seen=seen)
    info = encode_documents(["heldout", "train"], FakeEncoder(), tmp_path, "train", max_tokens=3,
                            seen=seen, excluded_hashes=set(seen))
    assert np.fromfile(tmp_path / "train.bin", dtype="<u2").tolist() == list(b"tra")
    assert len(info["excluded_heldout_exact_duplicates"]) == 1
    assert info["cross_split_exact_duplicates"] == []


def test_nonascii_and_literal_special_token_text(tmp_path):
    # Byte-valued fixture proves non-ASCII source is UTF-8, not silently dropped.
    info = encode_documents(["中文 <|endoftext|>"], FakeEncoder(), tmp_path, "train")
    ids = np.fromfile(tmp_path / "train.bin", dtype="<u2").tolist()
    assert bytes(ids[:-1]).decode() == "中文 <|endoftext|>"
    assert info["tokens"] == len(ids)


def test_batcher_shift_dtype_and_resume(tmp_path):
    fixture_data(tmp_path)
    batcher = TokenBatcher(tmp_path, "train", 3, 8, 42, "cpu")
    x, y = batcher.next()
    assert x.dtype == torch.long and x.shape == (3, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
    state = json.loads(json.dumps(batcher.state_dict()))
    expected = batcher.next()
    resumed = TokenBatcher(tmp_path, "train", 3, 8, 999, "cpu")
    resumed.load_state_dict(state)
    actual = resumed.next()
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_batcher_rejects_different_data_or_shape(tmp_path):
    fixture_data(tmp_path)
    batcher = TokenBatcher(tmp_path, "train", 2, 8, 42, "cpu")
    state = batcher.state_dict()
    with pytest.raises(ValueError, match="seq_len"):
        TokenBatcher(tmp_path, "train", 2, 9, 42).load_state_dict(state)
    state["data_sha256"] = "changed"
    with pytest.raises(ValueError, match="different token"):
        batcher.load_state_dict(state)


def test_manifest_verification_detects_corruption(tmp_path):
    fixture_data(tmp_path)
    verify(tmp_path)
    with (tmp_path / "train.bin").open("r+b") as f:
        f.write(b"\xff\xff")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)


def test_single_available_window_is_valid(tmp_path):
    fixture_data(tmp_path, n=9)
    x, y = TokenBatcher(tmp_path, "train", 1, 8, 123).next()
    assert x.tolist() == [list(range(8))]
    assert y.tolist() == [list(range(1, 9))]
