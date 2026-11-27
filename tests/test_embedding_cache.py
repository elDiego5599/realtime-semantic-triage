import sys
import types
from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import (
    main,
    read_embeddings,
    run_baseline,
    write_embeddings,
)
from semtriage.embed import HashingEmbedder
from semtriage.models import Event, Source

WHEN = datetime(2024, 5, 1, tzinfo=UTC)


def make_event(
    source_id: str, text: str, cve_ids: tuple[str, ...], source: Source = Source.NVD
) -> Event:
    return Event(
        event_id=f"{source.value}:{source_id}",
        source=source,
        source_id=source_id,
        published=WHEN,
        modified=WHEN,
        text=text,
        cve_ids=cve_ids,
        cwe_ids=(),
        ecosystems=(),
        packages=(),
        cna=None,
        severity=None,
    )


def sample_events() -> list[Event]:
    return [
        make_event("1", "heap overflow in libfoo", ("CVE-2024-0001",)),
        make_event("2", "heap overflow in libfoo parser", ("CVE-2024-0001",), Source.GHSA),
        make_event("3", "XSS in admin page", ("CVE-2024-0002",)),
        make_event("4", "Bundle of fixes", ("CVE-2024-0003", "CVE-2024-0004")),
    ]


def write_jsonl(path, events: list[Event]) -> None:
    path.write_text("\n".join(event.model_dump_json() for event in events) + "\n")


KEPT_IDS = ["nvd:1", "ghsa:2", "nvd:3"]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "vectors.npz"
    vectors = np.arange(6, dtype=np.float64).reshape(3, 2)
    write_embeddings(path, KEPT_IDS, vectors, embedder="e1")
    assert path.exists()
    loaded, name = read_embeddings(path, KEPT_IDS)
    assert name == "e1"
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, vectors.astype(np.float32))


def test_save_keeps_exact_path_without_npz_suffix(tmp_path):
    path = tmp_path / "vectors.bin"
    write_embeddings(path, KEPT_IDS, np.zeros((3, 2)), embedder="e1")
    assert path.exists()
    assert not (tmp_path / "vectors.bin.npz").exists()


def test_save_rejects_row_mismatch(tmp_path):
    with pytest.raises(ValueError, match="rows"):
        write_embeddings(tmp_path / "v.npz", KEPT_IDS, np.zeros((2, 2)), embedder="e1")


@pytest.mark.parametrize(
    "ids",
    [
        ["ghsa:2", "nvd:1", "nvd:3"],
        ["nvd:1", "ghsa:2"],
        ["nvd:1", "ghsa:2", "nvd:3", "nvd:9"],
        ["nvd:1", "ghsa:2", "nvd:4"],
    ],
)
def test_load_rejects_id_mismatch(tmp_path, ids):
    path = tmp_path / "vectors.npz"
    write_embeddings(path, KEPT_IDS, np.zeros((3, 2)), embedder="e1")
    with pytest.raises(ValueError, match="event ids"):
        read_embeddings(path, ids)


@pytest.mark.parametrize(
    ("event_ids", "vectors"),
    [
        (np.array(KEPT_IDS), np.zeros((2, 2), dtype=np.float32)),
        (np.array(KEPT_IDS), np.zeros(3, dtype=np.float32)),
        (np.array([KEPT_IDS]), np.zeros((3, 2), dtype=np.float32)),
    ],
)
def test_load_rejects_bad_shapes(tmp_path, event_ids, vectors):
    path = tmp_path / "vectors.npz"
    np.savez(path, event_ids=event_ids, vectors=vectors, embedder=np.array("e1"))
    with pytest.raises(ValueError, match="shape"):
        read_embeddings(path, KEPT_IDS)


def test_load_rejects_missing_keys_and_garbage(tmp_path):
    path = tmp_path / "vectors.npz"
    np.savez(path, vectors=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="missing"):
        read_embeddings(path, KEPT_IDS)
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="garbage.npz"):
        read_embeddings(garbage, KEPT_IDS)
    with pytest.raises(ValueError, match="nope.npz"):
        read_embeddings(tmp_path / "nope.npz", KEPT_IDS)


class FailingEmbedder:
    name = "failing"
    dim = 2

    def embed(self, texts):
        raise AssertionError("embedder must not run")


def test_run_baseline_save_then_load_gives_same_report(tmp_path):
    path = tmp_path / "vectors.npz"
    events = sample_events()
    first = run_baseline(events, HashingEmbedder(), [0.5, 0.9], save_embeddings=path)
    second = run_baseline(events, None, [0.5, 0.9], load_embeddings=path)
    assert second.rows == first.rows
    assert second.embedder == "hashing-256"
    assert second.n_events == 3
    assert second.n_excluded == 1


def test_run_baseline_requires_exactly_one_vector_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        run_baseline(sample_events(), None, [0.5])
    with pytest.raises(ValueError, match="exactly one"):
        run_baseline(sample_events(), FailingEmbedder(), [0.5], load_embeddings=tmp_path / "v.npz")


def test_run_baseline_load_rejects_other_dataset(tmp_path):
    path = tmp_path / "vectors.npz"
    run_baseline(sample_events(), HashingEmbedder(), [0.5], save_embeddings=path)
    other = sample_events()[:2] + [make_event("7", "other", ("CVE-2024-0009",))]
    with pytest.raises(ValueError, match="event ids"):
        run_baseline(other, None, [0.5], load_embeddings=path)


def test_main_save_then_load(tmp_path, capsys, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    write_jsonl(events_path, sample_events())
    cache = tmp_path / "vectors.npz"
    assert main(["--events", str(events_path), "--save-embeddings", str(cache)]) == 0
    saved_out = capsys.readouterr().out
    # FlagEmbedding and torch must never be imported when vectors are loaded.
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert main(["--events", str(events_path), "--load-embeddings", str(cache)]) == 0
    assert capsys.readouterr().out == saved_out


def test_main_load_rejects_mismatch_with_exit_2(tmp_path, capsys):
    events_path = tmp_path / "events.jsonl"
    write_jsonl(events_path, sample_events())
    cache = tmp_path / "vectors.npz"
    write_embeddings(cache, ["nvd:1", "nvd:3", "ghsa:2"], np.zeros((3, 2)), embedder="e1")
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events_path), "--load-embeddings", str(cache)])
    assert excinfo.value.code == 2
    assert "event ids" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--load-embeddings", "a.npz", "--save-embeddings", "b.npz"],
        ["--load-embeddings", "a.npz", "--embedder", "hashing"],
        ["--load-embeddings", "a.npz", "--fp16"],
        ["--fp16"],
        ["--fp16", "--embedder", "hashing"],
    ],
)
def test_main_rejects_flag_conflicts(tmp_path, extra):
    events_path = tmp_path / "events.jsonl"
    write_jsonl(events_path, sample_events())
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events_path), *extra])
    assert excinfo.value.code == 2


def test_main_fp16_passes_use_fp16(tmp_path, capsys, monkeypatch):
    seen = {}

    class FakeModel:
        def __init__(self, model_name, use_fp16=False, **kwargs):
            seen["use_fp16"] = use_fp16

        def encode(self, sentences, **kwargs):
            return {"dense_vecs": np.ones((len(sentences), 1024))}

    module = types.ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = FakeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)
    events_path = tmp_path / "events.jsonl"
    write_jsonl(events_path, sample_events())
    argv = ["--events", str(events_path), "--embedder", "bge-m3", "--thresholds", "0.5"]
    assert main(argv) == 0
    assert seen["use_fp16"] is False
    assert main([*argv, "--fp16"]) == 0
    assert seen["use_fp16"] is True


def test_help_lists_new_flags(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for flag in ("--save-embeddings", "--load-embeddings", "--fp16"):
        assert flag in out
