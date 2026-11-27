from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from semtriage import ablation  # noqa: E402
from semtriage.baseline import write_embeddings  # noqa: E402
from semtriage.models import Event, Source  # noqa: E402
from semtriage.text_mode import build_text, parse_text_mode_label  # noqa: E402


def make_event(
    source: Source,
    cve: str,
    day: int,
    *,
    cwe_ids: tuple[str, ...] = (),
    products: tuple[str, ...] = (),
    text: str = "alpha advisory",
) -> Event:
    when = datetime(2024, 1, day, tzinfo=UTC)
    return Event(
        event_id=f"{source.value}:{cve}",
        source=source,
        source_id=cve,
        published=when,
        modified=when,
        text=text,
        cve_ids=(cve,),
        cwe_ids=cwe_ids,
        ecosystems=(),
        packages=(),
        products=products,
        cna=None,
        severity=None,
    )


def fixture() -> tuple[list[Event], np.ndarray]:
    events = [
        make_event(Source.NVD, "CVE-1", 1, cwe_ids=("CWE-79",), products=("one",)),
        make_event(Source.GHSA, "CVE-1", 2, cwe_ids=("CWE-79",), products=("one",)),
        make_event(Source.NVD, "CVE-2", 3, cwe_ids=("CWE-89",), products=("two",)),
        make_event(Source.GHSA, "CVE-2", 4, cwe_ids=("CWE-89",), products=("two",)),
        make_event(Source.NVD, "CVE-3", 5, cwe_ids=("CWE-79",), products=("three",)),
        make_event(Source.GHSA, "CVE-3", 6, cwe_ids=("CWE-79",), products=("three",)),
        make_event(Source.NVD, "CVE-4", 7, cwe_ids=("CWE-79",), products=("four",)),
        make_event(Source.GHSA, "CVE-4", 8, cwe_ids=("CWE-79",), products=("four",)),
        make_event(Source.NVD, "CVE-5", 9, cwe_ids=("CWE-89",), products=("five",)),
        make_event(Source.GHSA, "CVE-5", 10, cwe_ids=("CWE-89",), products=("five",)),
        make_event(Source.NVD, "CVE-6", 11, cwe_ids=("CWE-79",), products=("six",)),
        make_event(Source.GHSA, "CVE-6", 12, cwe_ids=("CWE-79",), products=("six",)),
    ]
    dense = np.eye(len(events), dtype=np.float32)
    for index in range(0, len(events), 2):
        dense[index + 1] = dense[index]
    return events, dense


def write_mode_inputs(tmp_path: Path, label: str) -> tuple[str, str, list[Event], np.ndarray]:
    events, vectors = fixture()
    long_text = "First line of the advisory text.\n" + "word " * 60
    events = [event.model_copy(update={"text": long_text}) for event in events]
    mode, max_chars = parse_text_mode_label(label)
    texts = [build_text(event, mode, max_chars=max_chars) for event in events]
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    embeddings_path = tmp_path / "vectors.npz"
    ids = [event.event_id for event in events]
    write_embeddings(
        embeddings_path,
        ids,
        texts,
        vectors,
        embedder="test-embedder",
        text_mode=label,
    )
    return str(events_path), str(embeddings_path), events, vectors


class FakeSparse:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
        batch = list(texts)
        self.calls.append(batch)
        encoded = []
        for text in batch:
            words = sorted(set(text.lower().split()))
            indices = [sum(ord(char) for char in word) % 1000 for word in words]
            encoded.append((indices, [1.0] * len(indices)))
        return encoded


def test_table_has_one_row_per_config(tmp_path: Path, monkeypatch, capsys) -> None:
    events_path, embeddings_path, _, _ = write_mode_inputs(tmp_path, "truncate:40")
    fake = FakeSparse()
    monkeypatch.setattr(ablation, "bm25_encoder", lambda: fake)
    assert ablation.main(["--events", events_path, "--embeddings", embeddings_path]) == 0
    out = capsys.readouterr().out
    assert "Text: truncate:40" in out
    assert "| dense |" in out
    assert "| bm25 |" in out
    assert "| hybrid |" in out
    assert "| hybrid+cwe |" in out


def test_perfect_dense_scores_one(monkeypatch) -> None:
    events, dense = fixture()
    fake = FakeSparse()
    monkeypatch.setattr(ablation, "bm25_encoder", lambda: fake)
    result = ablation.run_ablation(
        events,
        dense,
        [event.text for event in events],
        fake,
        ("dense",),
    )
    report = result["dense"]["cve"]
    assert report.means["recall@1"] == pytest.approx(1.0)


def test_save_runs_json_roundtrip(tmp_path: Path, monkeypatch) -> None:
    events, dense = fixture()
    fake = FakeSparse()
    monkeypatch.setattr(ablation, "bm25_encoder", lambda: fake)
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    texts = [event.text for event in events]
    embeddings_path = tmp_path / "vectors.npz"
    write_embeddings(
        embeddings_path,
        [event.event_id for event in events],
        texts,
        dense,
        embedder="test",
    )
    saved = tmp_path / "runs.json"
    assert (
        ablation.main(
            [
                "--events",
                str(events_path),
                "--embeddings",
                str(embeddings_path),
                "--configs",
                "dense",
                "--save-runs",
                str(saved),
            ]
        )
        == 0
    )
    import json

    data = json.loads(saved.read_text())
    assert set(data) == {"dense"}
    assert all(isinstance(ids, list) for ids in data["dense"].values())


def test_cli_rejects_unknown_config(tmp_path: Path) -> None:
    events_path, embeddings_path, _, _ = write_mode_inputs(tmp_path, "full")
    with pytest.raises(SystemExit) as exc:
        ablation.main(
            ["--events", events_path, "--embeddings", embeddings_path, "--configs", "bogus"]
        )
    assert exc.value.code == 2


def test_cli_rejects_mismatched_cache(tmp_path: Path) -> None:
    events_path, embeddings_path, events, vectors = write_mode_inputs(tmp_path, "full")
    write_embeddings(
        embeddings_path,
        [event.event_id for event in events[:-1]],
        [event.text for event in events[:-1]],
        vectors[:-1],
        embedder="test",
    )
    with pytest.raises(SystemExit) as exc:
        ablation.main(["--events", events_path, "--embeddings", embeddings_path])
    assert exc.value.code == 2


def test_cli_uses_cached_text_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    events_path, embeddings_path, _, _ = write_mode_inputs(tmp_path, "truncate:40")
    fake = FakeSparse()
    monkeypatch.setattr(ablation, "bm25_encoder", lambda: fake)
    assert ablation.main(["--events", events_path, "--embeddings", embeddings_path]) == 0
    out = capsys.readouterr().out
    assert "Text: truncate:40" in out
    assert all(len(text) <= 40 for call in fake.calls for text in call)


def test_run_ablation_rejects_unknown_config(monkeypatch) -> None:
    events, dense = fixture()
    fake = FakeSparse()
    monkeypatch.setattr(ablation, "bm25_encoder", lambda: fake)
    with pytest.raises(ValueError, match="unknown"):
        ablation.run_ablation(events, dense, [event.text for event in events], fake, ("bogus",))
