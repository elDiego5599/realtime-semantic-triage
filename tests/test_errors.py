import sys
from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import write_embeddings
from semtriage.errors import analyze_errors, format_error_report, main
from semtriage.models import Event, Source
from semtriage.sanitize import sanitize
from semtriage.text_mode import build_text

WHEN = datetime(2024, 5, 1, tzinfo=UTC)


def make_event(
    source_id: str,
    text: str,
    cve_ids: tuple[str, ...],
    source: Source = Source.NVD,
    *,
    packages: tuple[str, ...] = (),
    ecosystems: tuple[str, ...] = (),
    cwe_ids: tuple[str, ...] = (),
    cna: str | None = None,
) -> Event:
    return Event(
        event_id=f"{source.value}:{source_id}",
        source=source,
        source_id=source_id,
        published=WHEN,
        modified=WHEN,
        text=text,
        cve_ids=cve_ids,
        cwe_ids=cwe_ids,
        ecosystems=ecosystems,
        packages=packages,
        cna=cna,
        severity=None,
    )


def unit(angle: float) -> list[float]:
    return [float(np.cos(angle)), float(np.sin(angle))]


# Rows 0..4 are labeled; row 5 has two CVEs and is excluded.
# 0 (nvd, A) and 1 (ghsa, A): sim 0.99, true pair.
# 2 (nvd, B) and 3 (ghsa, C): sim 0.95, false merge sharing package, ecosystem and CWE.
# 4 (ghsa, A): far from 0 and 1, so two missed duplicates (0-4 and 1-4).
EVENTS = [
    make_event("1", "heap overflow in libfoo " * 30, ("CVE-1",), packages=("foo",), cna="mitre"),
    make_event("2", "libfoo overflow", ("CVE-1",), Source.GHSA, packages=("foo",)),
    make_event(
        "3",
        "sql injection in bar",
        ("CVE-2",),
        packages=("bar",),
        ecosystems=("npm",),
        cwe_ids=("CWE-89",),
        cna="github",
    ),
    make_event(
        "4",
        "sql injection in bar admin",
        ("CVE-3",),
        Source.GHSA,
        packages=("bar", "baz"),
        ecosystems=("npm",),
        cwe_ids=("CWE-89",),
        cna="mitre",
    ),
    make_event("5", "unrelated text", ("CVE-1",), Source.GHSA),
    make_event("6", "bundle", ("CVE-8", "CVE-9")),
]
VECTORS = np.array(
    [
        unit(0.0),
        unit(np.arccos(0.99)),
        unit(1.2),
        unit(1.2 + np.arccos(0.95)),
        unit(-1.5),
    ]
)


KEPT_TEXTS = [sanitize(event.text) for event in EVENTS[:5]]


def test_counts_and_shares():
    report = analyze_errors(EVENTS, VECTORS, 0.9, samples=5)
    assert report.n_events == 5
    assert report.n_excluded == 1
    assert report.pairs == "cross-source"
    assert report.n_false_merges == 1
    assert report.n_true_pairs == 1
    assert report.n_missed == 2
    assert report.false_merge_shares == {
        "package": 1.0,
        "ecosystem": 1.0,
        "cwe": 1.0,
        "cna": 0.0,
        "any": 1.0,
    }
    assert report.false_merge_quantiles[0.5] == pytest.approx(0.95)
    assert report.true_pair_quantiles[0.5] == pytest.approx(0.99)


def test_missed_duplicates_lengths_and_similarity():
    report = analyze_errors(EVENTS, VECTORS, 0.9, samples=5)
    sims = sorted(sample.similarity for sample in report.missed_samples)
    assert sims == pytest.approx(sorted([float(VECTORS[0] @ VECTORS[4]), VECTORS[1] @ VECTORS[4]]))
    ids = {(sample.left_id, sample.right_id) for sample in report.missed_samples}
    assert ids == {("nvd:1", "ghsa:5"), ("ghsa:2", "ghsa:5")}
    lengths = {len(sanitize(EVENTS[i].text)) for i in (0, 1, 4)}
    for side in ("shorter", "longer"):
        values = report.missed_length_quantiles[side].values()
        assert all(min(lengths) <= value <= max(lengths) for value in values)
    assert report.missed_length_quantiles["longer"][1.0] == max(lengths)
    assert report.missed_same_source == 1


def test_direct_false_edges_and_cluster_sizes():
    report = analyze_errors(EVENTS, VECTORS, 0.9, samples=5)
    assert report.n_false_edges == 1
    assert report.false_edge_shares == {
        "package": 1.0,
        "ecosystem": 1.0,
        "cwe": 1.0,
        "cna": 0.0,
        "any": 1.0,
    }
    assert report.false_edge_quantiles[0.5] == pytest.approx(0.95)
    [sample] = report.false_edge_samples
    assert (sample.left_id, sample.right_id) == ("nvd:3", "ghsa:4")
    assert sample.similarity == pytest.approx(0.95)
    assert report.cluster_sizes == {"1": 1, "2": 2, "3": 0, "4-10": 0, ">10": 0}
    assert report.largest_cluster == 2


def test_direct_edges_follow_pairs_mode():
    # Two NVD events with different CVEs and identical vectors.
    events = [make_event("1", "a", ("CVE-1",)), make_event("2", "b", ("CVE-2",))]
    vectors = np.array([unit(0.0), unit(0.0)])
    assert analyze_errors(events, vectors, 0.9).n_false_edges == 0
    assert analyze_errors(events, vectors, 0.9, pairs="all").n_false_edges == 1


def test_cluster_size_buckets():
    # One cluster of 12, one of 5, one singleton.
    events = [make_event(str(i), "x", (f"CVE-{i}",)) for i in range(18)]
    vectors = np.array([unit(0.0)] * 12 + [unit(1.5)] * 5 + [unit(3.0)])
    report = analyze_errors(events, vectors, 0.9, pairs="all", samples=0)
    assert report.cluster_sizes == {"1": 1, "2": 0, "3": 0, "4-10": 1, ">10": 1}
    assert report.largest_cluster == 12


def test_samples_truncate_text_and_respect_count():
    report = analyze_errors(EVENTS, VECTORS, 0.9, samples=1)
    assert len(report.missed_samples) == 1
    assert len(report.false_merge_samples) == 1
    sample = report.false_merge_samples[0]
    assert (sample.left_id, sample.right_id) == ("nvd:3", "ghsa:4")
    assert sample.similarity == pytest.approx(0.95)
    assert sample.left_text == sanitize(EVENTS[2].text)[:300]
    long_report = analyze_errors(EVENTS, VECTORS, 0.0, samples=50, pairs="all")
    assert all(len(s.left_text) <= 300 for s in long_report.false_merge_samples)


def test_transitive_cluster_pairs_count_as_predicted():
    # 0-1 and 1-2 linked; 0-2 is a predicted pair only through 1.
    events = [
        make_event("1", "a", ("CVE-1",)),
        make_event("2", "b", ("CVE-1",), Source.GHSA),
        make_event("3", "c", ("CVE-2",)),
    ]
    vectors = np.array([unit(0.0), unit(0.3), unit(0.6)])
    report = analyze_errors(events, vectors, 0.9, samples=0)
    assert report.n_true_pairs == 1
    assert report.n_false_merges == 2
    assert report.n_missed == 0
    assert report.false_merge_samples == []
    # Only the 1-2 edge is a direct model error; 0-2 comes from chaining.
    assert report.n_false_edges == 1
    assert report.false_edge_quantiles[0.0] == pytest.approx(np.cos(0.3))
    assert report.false_merge_quantiles[0.0] == pytest.approx(np.cos(0.6))
    assert report.cluster_sizes == {"1": 0, "2": 0, "3": 1, "4-10": 0, ">10": 0}
    assert report.largest_cluster == 3


def test_pairs_all_links_same_source():
    events = [make_event("1", "a", ("CVE-1",)), make_event("2", "b", ("CVE-1",))]
    vectors = np.array([unit(0.0), unit(0.0)])
    assert analyze_errors(events, vectors, 0.9).n_missed == 1
    assert analyze_errors(events, vectors, 0.9, pairs="all").n_true_pairs == 1


def test_no_false_merges_has_empty_quantiles_and_zero_shares():
    report = analyze_errors(EVENTS, VECTORS, 0.999)
    assert report.n_false_merges == 0
    assert report.false_merge_quantiles == {}
    assert report.false_merge_shares["any"] == 0.0
    assert report.n_false_edges == 0
    assert report.false_edge_quantiles == {}
    text = format_error_report(report)
    assert "False merges (pairs inside predicted clusters): 0" in text
    assert "Direct false edges: 0" in text


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="rows"):
        analyze_errors(EVENTS, VECTORS[:3], 0.9)
    with pytest.raises(ValueError, match="pairs"):
        analyze_errors(EVENTS, VECTORS, 0.9, pairs="same")
    with pytest.raises(ValueError, match="samples"):
        analyze_errors(EVENTS, VECTORS, 0.9, samples=-1)


def test_format_error_report_sections():
    text = format_error_report(analyze_errors(EVENTS, VECTORS, 0.9, samples=2))
    assert "Threshold: 0.90" in text
    assert "False merges (pairs inside predicted clusters): 1" in text
    assert "Direct false edges: 1" in text
    assert "Missed duplicates: 2" in text
    assert "| 2 | 2 |" in text
    assert "Largest cluster: 2" in text
    assert "| package | 1.000 |" in text
    assert "nvd:3" in text and "ghsa:4" in text
    assert "0.950" in text


def write_inputs(tmp_path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("\n".join(e.model_dump_json() for e in EVENTS) + "\n")
    cache = tmp_path / "vectors.npz"
    ids = [e.event_id for e in EVENTS[:5]]
    write_embeddings(cache, ids, KEPT_TEXTS, VECTORS, embedder="synthetic")
    return events_path, cache


def test_main_runs_without_embedding_libraries(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    events_path, cache = write_inputs(tmp_path)
    argv = ["--events", str(events_path), "--embeddings", str(cache), "--threshold", "0.9"]
    assert main([*argv, "--samples", "1"]) == 0
    out = capsys.readouterr().out
    assert "False merges (pairs inside predicted clusters): 1" in out
    assert "Direct false edges: 1" in out
    assert "Embedder: synthetic" in out


def test_main_rejects_mismatched_embeddings(tmp_path, capsys):
    events_path, cache = write_inputs(tmp_path)
    write_embeddings(cache, ["a", "b", "c", "d", "e"], KEPT_TEXTS, VECTORS, embedder="synthetic")
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events_path), "--embeddings", str(cache), "--threshold", "0.9"])
    assert excinfo.value.code == 2
    assert "event ids" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["nan", "inf", "x"])
def test_main_rejects_bad_threshold(tmp_path, value):
    events_path, cache = write_inputs(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events_path), "--embeddings", str(cache), "--threshold", value])
    assert excinfo.value.code == 2


def test_main_help_lists_flags(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for flag in ("--events", "--embeddings", "--threshold", "--pairs", "--samples"):
        assert flag in out


def test_main_rejects_stale_texts(tmp_path, capsys):
    events_path, cache = write_inputs(tmp_path)
    changed = [*EVENTS]
    changed[0] = changed[0].model_copy(update={"text": "edited text"})
    events_path.write_text("\n".join(e.model_dump_json() for e in changed) + "\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events_path), "--embeddings", str(cache), "--threshold", "0.9"])
    assert excinfo.value.code == 2
    assert "sanitized texts" in capsys.readouterr().err


def test_main_runs_on_lead_cache(tmp_path, capsys):
    events_path, cache = write_inputs(tmp_path)
    lead_texts = [build_text(event, "lead", max_chars=20) for event in EVENTS[:5]]
    ids = [event.event_id for event in EVENTS[:5]]
    assert lead_texts != KEPT_TEXTS
    write_embeddings(cache, ids, lead_texts, VECTORS, embedder="synthetic", text_mode="lead:20")
    argv = ["--events", str(events_path), "--embeddings", str(cache), "--threshold", "0.9"]
    assert main(argv) == 0
    assert "Direct false edges: 1" in capsys.readouterr().out


def test_main_rejects_lead_cache_with_wrong_label(tmp_path, capsys):
    events_path, cache = write_inputs(tmp_path)
    lead_texts = [build_text(event, "lead", max_chars=20) for event in EVENTS[:5]]
    ids = [event.event_id for event in EVENTS[:5]]
    write_embeddings(cache, ids, lead_texts, VECTORS, embedder="synthetic", text_mode="lead:30")
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(events_path), "--embeddings", str(cache), "--threshold", "0.9"])
    assert excinfo.value.code == 2
