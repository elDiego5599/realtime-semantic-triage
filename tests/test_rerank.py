from __future__ import annotations

import json
import sys
import types
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from semtriage import rerank
from semtriage.baseline import write_embeddings
from semtriage.models import Event, Source


def make_event(event_id: str, day: int, text: str) -> Event:
    when = datetime(2024, 1, day, tzinfo=UTC)
    return Event(
        event_id=event_id,
        source=Source.NVD,
        source_id=event_id,
        published=when,
        modified=when,
        text=text,
        cve_ids=(f"CVE-{day}",),
        cwe_ids=(),
        ecosystems=(),
        packages=(),
        products=(),
        cna=None,
        severity=None,
    )


def overlap_scorer() -> tuple[rerank.Scorer, list[int]]:
    call_sizes: list[int] = []

    def score(pairs: Sequence[tuple[str, str]]) -> list[float]:
        call_sizes.append(len(pairs))
        return [
            float(len(set(query.lower().split()) & set(doc.lower().split())))
            for query, doc in pairs
        ]

    return score, call_sizes


def test_rerank_orders_top_by_score() -> None:
    texts = {
        "q": "alpha beta",
        "d1": "none",
        "d2": "alpha only",
        "d3": "alpha beta exact",
    }
    scorer, _ = overlap_scorer()
    assert rerank.rerank_runs({"q": ["d1", "d2", "d3"]}, texts, scorer, depth=3) == {
        "q": ["d3", "d2", "d1"]
    }


def test_rerank_keeps_tail_after_depth() -> None:
    texts = {
        "q": "alpha",
        "d1": "none",
        "d2": "alpha",
        "d3": "alpha",
        "d4": "tail",
    }
    scorer, _ = overlap_scorer()
    assert rerank.rerank_runs({"q": ["d1", "d2", "d3", "d4"]}, texts, scorer, depth=2) == {
        "q": ["d2", "d1", "d3", "d4"]
    }


def test_rerank_short_run_keeps_all_ids() -> None:
    texts = {"q": "alpha", "d1": "none", "d2": "alpha"}
    scorer, _ = overlap_scorer()
    assert rerank.rerank_runs({"q": ["d1", "d2"]}, texts, scorer, depth=50) == {"q": ["d2", "d1"]}


def test_rerank_ties_keep_original_order() -> None:
    texts = {"q": "alpha", "d1": "beta", "d2": "gamma", "d3": "delta"}
    scorer, _ = overlap_scorer()
    assert rerank.rerank_runs({"q": ["d1", "d2", "d3"]}, texts, scorer) == {"q": ["d1", "d2", "d3"]}


def test_rerank_batches_pairs(monkeypatch) -> None:
    monkeypatch.setattr(rerank, "PAIR_BATCH", 64)
    texts = {"q1": "alpha", "q2": "alpha", "q3": "alpha"}
    docs = [f"d{index}" for index in range(50)]
    texts.update({doc: "none" for doc in docs})
    scorer, call_sizes = overlap_scorer()
    rerank.rerank_runs(
        {"q1": docs, "q2": docs, "q3": docs},
        texts,
        scorer,
        depth=50,
    )
    assert sum(call_sizes) == 150
    assert call_sizes == [64, 64, 22]
    assert max(call_sizes) <= 64


def test_rerank_wrong_score_count_is_model_error() -> None:
    def bad_scorer(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [1.0]

    with pytest.raises(RuntimeError, match="returned 1 scores for 2 pairs"):
        rerank.rerank_runs(
            {"q": ["d1", "d2"]},
            {"q": "alpha", "d1": "alpha", "d2": "beta"},
            bad_scorer,
        )


def test_rerank_empty_runs_does_not_call_scorer() -> None:
    def fail_scorer(pairs: Sequence[tuple[str, str]]) -> list[float]:
        raise AssertionError("scorer should not be called")

    assert rerank.rerank_runs({}, {}, fail_scorer) == {}


def write_inputs(tmp_path: Path) -> tuple[Path, Path, list[Event]]:
    events = [
        make_event("q1", 1, "alpha beta"),
        make_event("d1", 2, "none"),
        make_event("d2", 3, "alpha beta"),
    ]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    ids = [event.event_id for event in events]
    texts = [event.text for event in events]
    embeddings_path = tmp_path / "vectors.npz"
    write_embeddings(
        embeddings_path,
        ids,
        texts,
        np.eye(3, dtype=np.float32),
        embedder="test",
    )
    return events_path, embeddings_path, events


def test_cli_writes_reranked_runs(tmp_path: Path, monkeypatch, capsys) -> None:
    events_path, embeddings_path, _ = write_inputs(tmp_path)
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"hybrid": {"q1": ["d1", "d2"]}}))
    out_path = tmp_path / "reranked.json"
    call = {}
    scorer, _ = overlap_scorer()

    def fake_reranker(*, use_fp16: bool = True, batch_size: int = 64, max_length: int = 1024):
        call.update(use_fp16=use_fp16, batch_size=batch_size, max_length=max_length)
        return scorer

    monkeypatch.setattr(rerank, "bge_reranker", fake_reranker)
    assert (
        rerank.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--runs",
                str(runs_path),
                "--config",
                "hybrid",
                "--out",
                str(out_path),
                "--depth",
                "2",
                "--batch-size",
                "7",
                "--no-fp16",
            ]
        )
        == 0
    )
    assert call["use_fp16"] is False
    assert call["batch_size"] == 7
    assert call["max_length"] == 1024
    assert json.loads(out_path.read_text()) == {"hybrid+rerank": {"q1": ["d2", "d1"]}}
    assert "Wrote 1 reranked runs" in capsys.readouterr().out


def test_cli_rejects_missing_config(tmp_path: Path) -> None:
    events_path, embeddings_path, _ = write_inputs(tmp_path)
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"dense": {"q1": ["d1"]}}))
    with pytest.raises(SystemExit) as exc:
        rerank.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--runs",
                str(runs_path),
                "--config",
                "hybrid",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "runs",
    [
        {"hybrid": {"zzz": ["d1"]}},
        {"hybrid": {"q1": ["zzz"]}},
    ],
)
def test_cli_rejects_unknown_event_id(tmp_path: Path, runs: dict) -> None:
    events_path, embeddings_path, _ = write_inputs(tmp_path)
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps(runs))
    with pytest.raises(SystemExit) as exc:
        rerank.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--runs",
                str(runs_path),
                "--config",
                "hybrid",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
    assert exc.value.code == 2


def test_cli_rejects_bad_runs_json(tmp_path: Path) -> None:
    events_path, embeddings_path, _ = write_inputs(tmp_path)
    runs_path = tmp_path / "runs.json"
    runs_path.write_text("{bad json")
    with pytest.raises(SystemExit) as exc:
        rerank.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--runs",
                str(runs_path),
                "--config",
                "hybrid",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
    assert exc.value.code == 2


def test_cli_rejects_missing_runs_file(tmp_path: Path) -> None:
    events_path, embeddings_path, _ = write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc:
        rerank.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--runs",
                str(tmp_path / "missing.json"),
                "--config",
                "hybrid",
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
    assert exc.value.code == 2


def test_cli_rejects_negative_depth(tmp_path: Path) -> None:
    events_path, embeddings_path, _ = write_inputs(tmp_path)
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps({"hybrid": {"q1": ["d1"]}}))
    with pytest.raises(SystemExit) as exc:
        rerank.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--runs",
                str(runs_path),
                "--config",
                "hybrid",
                "--out",
                str(tmp_path / "out.json"),
                "--depth",
                "-1",
            ]
        )
    assert exc.value.code == 2


def test_bge_reranker(monkeypatch) -> None:
    class FakeFlagReranker:
        def __init__(self, model_name: str, *, use_fp16: bool) -> None:
            assert model_name == "BAAI/bge-reranker-v2-m3"
            assert use_fp16 is False

        def compute_score(
            self,
            pairs,
            *,
            batch_size: int,
            max_length: int,
        ):
            assert batch_size == 3
            assert max_length == 512
            if len(pairs) == 1:
                return 0.25
            return [0.25, 0.75]

    module = types.ModuleType("FlagEmbedding")
    module.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)

    scorer = rerank.bge_reranker(use_fp16=False, batch_size=3, max_length=512)
    assert scorer([("alpha", "beta")]) == [0.25]
    assert scorer([("alpha", "beta"), ("gamma", "delta")]) == [0.25, 0.75]
