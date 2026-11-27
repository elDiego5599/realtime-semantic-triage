from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import write_embeddings
from semtriage.models import Event, Source
from semtriage.sparse_cache import (
    lexical_weights_to_sparse,
    main,
    read_sparse,
    write_sparse,
)
from semtriage.text_mode import build_text, parse_text_mode_label


def make_event(source: Source, cve: str, day: int, text: str = "alpha advisory") -> Event:
    when = datetime(2024, 1, day, tzinfo=UTC)
    return Event(
        event_id=f"{source.value}:{cve}",
        source=source,
        source_id=cve,
        published=when,
        modified=when,
        text=text,
        cve_ids=(cve,),
        cwe_ids=(),
        ecosystems=(),
        packages=(),
        products=(),
        cna=None,
        severity=None,
    )


def test_lexical_weights_to_sparse_sorted_ints():
    assert lexical_weights_to_sparse({"10": 0.5, "9": 0.25}) == ([9, 10], [0.25, 0.5])


def test_sparse_roundtrip(tmp_path):
    path = tmp_path / "sparse.npz"
    ids = ["nvd:1", "ghsa:2"]
    texts = ["alpha", "beta"]
    vectors = [([3, 7], [0.25, 0.5]), ([11], [1.0])]
    write_sparse(path, ids, texts, vectors, embedder="bge-m3", text_mode="full")
    loaded = read_sparse(path, ids, texts, text_mode="full")
    assert [row[0] for row in loaded] == [[3, 7], [11]]
    np.testing.assert_allclose(loaded[0][1], [0.25, 0.5], atol=1e-6)
    np.testing.assert_allclose(loaded[1][1], [1.0], atol=1e-6)


def test_empty_weights_roundtrip(tmp_path):
    path = tmp_path / "sparse.npz"
    write_sparse(path, ["nvd:1"], [""], [([], [])], embedder="bge-m3", text_mode="full")
    assert read_sparse(path, ["nvd:1"], [""], text_mode="full") == [([], [])]


@pytest.mark.parametrize("kind", ["texts", "ids", "mode"])
def test_read_sparse_rejects_mismatch(tmp_path, kind):
    path = tmp_path / "sparse.npz"
    ids = ["nvd:1", "ghsa:2"]
    texts = ["alpha", "beta"]
    write_sparse(
        path, ids, texts, [([1], [1.0]), ([], [])], embedder="bge-m3", text_mode="truncate:40"
    )
    other_ids = ["nvd:1"] if kind == "ids" else ids
    other_texts = ["changed", "beta"] if kind == "texts" else texts
    other_mode = "full" if kind == "mode" else "truncate:40"
    match = (
        "sanitized texts differ"
        if kind == "texts"
        else ("event ids" if kind == "ids" else "text mode differs")
    )
    with pytest.raises(ValueError, match=match):
        read_sparse(path, other_ids, other_texts, text_mode=other_mode)


def test_write_sparse_rejects_bad_rows(tmp_path):
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="length"):
        write_sparse(path, ["nvd:1"], ["alpha"], [([1], [])], embedder="bge-m3")
    with pytest.raises(ValueError, match="strictly increasing"):
        write_sparse(path, ["nvd:1"], ["alpha"], [([2, 1], [1.0, 1.0])], embedder="bge-m3")
    with pytest.raises(ValueError, match="finite"):
        write_sparse(path, ["nvd:1"], ["alpha"], [([1], [float("nan")])], embedder="bge-m3")
    with pytest.raises(ValueError, match="finite"):
        write_sparse(path, ["nvd:1"], ["alpha"], [([1], [1e39])], embedder="bge-m3")
    with pytest.raises(ValueError, match="range"):
        write_sparse(path, ["nvd:1"], ["alpha"], [([-5], [1.0])], embedder="bge-m3")
    with pytest.raises(ValueError, match="expected 2 texts"):
        write_sparse(path, ["nvd:1", "nvd:2"], ["alpha"], [([1], [1.0])], embedder="bge-m3")
    with pytest.raises(ValueError, match="expected 1 vectors"):
        write_sparse(path, ["nvd:1"], ["alpha"], [], embedder="bge-m3")


def test_empty_row_in_middle_and_zero_rows_roundtrip(tmp_path):
    path = tmp_path / "sparse.npz"
    vectors = [([1], [1.0]), ([], []), ([2, 5], [0.5, 0.25])]
    ids, texts = ["a:1", "a:2", "a:3"], ["x", "y", "z"]
    write_sparse(path, ids, texts, vectors, embedder="bge-m3")
    assert read_sparse(path, ids, texts) == vectors
    write_sparse(path, [], [], [], embedder="bge-m3")
    assert read_sparse(path, [], []) == []


def corrupt(path, **changes):
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(changes)
    for key in [key for key, value in changes.items() if value is None]:
        del arrays[key]
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"indptr": np.array([1, 2], dtype=np.int64)}, "bad CSR indptr"),
        ({"indptr": np.array([0, 3], dtype=np.int64)}, "bad CSR indptr"),
        ({"indices": np.array([3, 1], dtype=np.int32)}, "strictly increasing"),
        ({"indices": np.array([-1, 1], dtype=np.int32)}, "range"),
        ({"values": np.array([np.inf, 1.0], dtype=np.float32)}, "finite"),
        ({"embedder": None}, "missing"),
    ],
)
def test_read_sparse_rejects_corrupt_file(tmp_path, changes, match):
    path = tmp_path / "sparse.npz"
    write_sparse(path, ["a:1"], ["x"], [([1, 3], [1.0, 1.0])], embedder="bge-m3")
    corrupt(path, **changes)
    with pytest.raises(ValueError, match=match):
        read_sparse(path, ["a:1"], ["x"])


def test_read_sparse_rejects_non_npz(tmp_path):
    path = tmp_path / "sparse.npz"
    path.write_text("not a zip")
    with pytest.raises(ValueError, match="cannot read sparse file"):
        read_sparse(path, [], [])


class FakeSparseEncoder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[str]):
        batch = list(texts)
        self.calls.append(batch)
        rows = []
        for text in batch:
            words = sorted(set(text.lower().split()))
            ids = [sum(ord(char) for char in word) % 1000 for word in words]
            rows.append((ids, [1.0] * len(ids)))
        return rows


def test_cli_writes_cache_aligned_with_dense(tmp_path, monkeypatch, capsys):
    events = [
        make_event(Source.NVD, "CVE-1", 1, "Alpha " * 30),
        make_event(Source.GHSA, "CVE-1", 2, "Beta " * 30),
    ]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    ids = [event.event_id for event in events]
    mode = "truncate:40"
    text_mode, max_chars = parse_text_mode_label(mode)
    texts = [build_text(event, text_mode, max_chars=max_chars) for event in events]
    dense_path = tmp_path / "dense.npz"
    write_embeddings(
        dense_path,
        ids,
        texts,
        np.eye(2, dtype=np.float32),
        embedder="test",
        text_mode=mode,
    )
    out = tmp_path / "sparse.npz"
    fake = FakeSparseEncoder()
    monkeypatch.setattr(
        "semtriage.sparse_cache.load_sparse_encoder", lambda use_fp16, batch_size: fake
    )

    assert (
        main(["--events", str(events_path), "--embeddings", str(dense_path), "--out", str(out)])
        == 0
    )
    assert fake.calls == [texts]
    assert read_sparse(out, ids, texts, text_mode=mode) == FakeSparseEncoder()(texts)
    with np.load(out) as data:
        assert str(data["text_mode"]) == mode
    assert capsys.readouterr().out.strip() == f"Wrote 2 sparse vectors to {out}"


def test_cli_rejects_mismatched_dense_cache(tmp_path, monkeypatch):
    events = [make_event(Source.NVD, "CVE-1", 1, "alpha")]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(events[0].model_dump_json() + "\n")
    dense_path = tmp_path / "dense.npz"
    write_embeddings(dense_path, ["wrong:id"], ["wrong text"], np.zeros((1, 2)), embedder="test")
    monkeypatch.setattr(
        "semtriage.sparse_cache.load_sparse_encoder",
        lambda use_fp16, batch_size: pytest.fail("encoder must not load"),
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(dense_path),
                "--out",
                str(tmp_path / "s.npz"),
            ]
        )
    assert exc.value.code == 2
