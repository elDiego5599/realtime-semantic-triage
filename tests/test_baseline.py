import sys
import types
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import (
    DEFAULT_THRESHOLDS,
    PAIR_MODES,
    BaselineReport,
    SweepRow,
    candidate_edges,
    edge_clusters,
    format_report,
    gold_label,
    load_events,
    main,
    run_baseline,
    singletons_reference,
    sweep,
    threshold_clusters,
)
from semtriage.metrics import Scores, bcubed_scores, pairwise_scores
from semtriage.models import Event, Source
from semtriage.sanitize import sanitize

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


# Exact dot products: a.b = 0.6, b.c = 0.8, a.c = 0.0
A = [1.0, 0.0]
B = [0.6, 0.8]
C = [0.0, 1.0]
ABC = np.array([A, B, C])


def test_gold_label_single_cve():
    assert gold_label(make_event("1", "x", ("CVE-2024-0001",))) == "CVE-2024-0001"


@pytest.mark.parametrize("cves", [(), ("CVE-2024-0001", "CVE-2024-0002")])
def test_gold_label_none_without_exactly_one_cve(cves):
    assert gold_label(make_event("1", "x", cves)) is None


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.9, [0, 1, 2]), (0.8, [0, 1, 1]), (0.7, [0, 1, 1]), (0.6, [0, 0, 0]), (-1.0, [0, 0, 0])],
)
def test_threshold_clusters_single_link(threshold, expected):
    assert threshold_clusters(ABC, threshold) == expected


def test_threshold_clusters_label_is_smallest_index():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert threshold_clusters(vectors, 0.9) == [0, 1, 0]


def test_threshold_clusters_empty_and_single():
    assert threshold_clusters(np.zeros((0, 4)), 0.5) == []
    assert threshold_clusters(np.array([[1.0, 0.0]]), 0.5) == [0]


def test_threshold_clusters_returns_python_ints():
    labels = threshold_clusters(ABC, 0.8)
    assert all(type(label) is int for label in labels)


def test_threshold_clusters_zero_rows_stay_separate():
    # Zero rows have similarity 0 with everything, including themselves.
    vectors = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert threshold_clusters(vectors, 0.5) == [0, 1]


def test_sweep_exact_scores_on_small_example():
    gold = {"a": "X", "b": "Y", "c": "Y"}
    rows = sweep(["a", "b", "c"], ABC, gold, [0.8, 0.6])
    assert rows[0] == SweepRow(
        threshold=0.8,
        n_clusters=2,
        pairwise=Scores(precision=1.0, recall=1.0, f1=1.0),
        bcubed=Scores(precision=1.0, recall=1.0, f1=1.0),
    )
    assert rows[1].threshold == 0.6
    assert rows[1].n_clusters == 1
    assert rows[1].pairwise.precision == pytest.approx(1 / 3)
    assert rows[1].pairwise.recall == 1.0


def test_sweep_keeps_input_order_and_repeats():
    gold = {"a": "X", "b": "Y", "c": "Y"}
    rows = sweep(["a", "b", "c"], ABC, gold, [0.6, 0.9, 0.6])
    assert [row.threshold for row in rows] == [0.6, 0.9, 0.6]
    assert rows[0] == rows[2]
    assert rows[1].n_clusters == 3


def test_sweep_matches_threshold_clusters_on_random_vectors():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(40, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"e{i}" for i in range(40)]
    gold = {item_id: f"CVE-{i % 7}" for i, item_id in enumerate(ids)}
    thresholds = [0.95, 0.1, 0.5, 0.3, 0.7, -0.2]
    rows = sweep(ids, vectors, gold, thresholds)
    assert len(rows) == len(thresholds)
    for row, threshold in zip(rows, thresholds, strict=True):
        labels = threshold_clusters(vectors, threshold)
        pred = dict(zip(ids, labels, strict=True))
        assert row.threshold == threshold
        assert row.n_clusters == len(set(labels))
        assert row.pairwise == pairwise_scores(pred, gold)
        assert row.bcubed == bcubed_scores(pred, gold)


def test_sweep_rejects_length_mismatch():
    with pytest.raises(ValueError):
        sweep(["a", "b"], ABC, {"a": 1, "b": 1}, [0.5])


def test_sweep_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        sweep(["a", "a", "c"], ABC, {"a": 1, "c": 1}, [0.5])


def test_sweep_rejects_empty_thresholds():
    with pytest.raises(ValueError):
        sweep(["a", "b", "c"], ABC, {"a": 1, "b": 1, "c": 1}, [])


def test_sweep_rejects_gold_mismatch():
    with pytest.raises(ValueError):
        sweep(["a", "b", "c"], ABC, {"a": 1, "b": 1}, [0.5])


class RecordingEmbedder:
    name = "recording"
    dim = 2

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.texts.extend(texts)
        rows = [A if "overflow" in text else C for text in texts]
        return np.array(rows, dtype=np.float32).reshape(len(texts), 2)


def sample_events() -> list[Event]:
    return [
        make_event("1", "CVE-2024-0001: heap overflow in libfoo", ("CVE-2024-0001",)),
        make_event(
            "2",
            "Heap overflow in libfoo (GHSA-abcd-efgh-ijkl)",
            ("CVE-2024-0001",),
            Source.GHSA,
        ),
        make_event("3", "XSS in admin page", ("CVE-2024-0002",)),
        make_event("4", "Bundle of fixes", ("CVE-2024-0003", "CVE-2024-0004")),
        make_event("5", "No identifier at all", ()),
    ]


def test_run_baseline_sanitizes_filters_and_sweeps():
    embedder = RecordingEmbedder()
    events = sample_events()
    report = run_baseline(events, embedder, [0.9, 0.5])
    assert report.embedder == "recording"
    assert report.n_events == 3
    assert report.n_excluded == 2
    assert embedder.texts == [sanitize(event.text) for event in events[:3]]
    assert all("CVE-" not in text and "GHSA-" not in text for text in embedder.texts)
    assert [row.threshold for row in report.rows] == [0.9, 0.5]
    assert report.rows[0].n_clusters == 2
    assert report.rows[0].pairwise == Scores(precision=1.0, recall=1.0, f1=1.0)


def test_run_baseline_accepts_iterator():
    report = run_baseline(iter(sample_events()), RecordingEmbedder(), [0.5])
    assert report.n_events == 3


def test_run_baseline_without_labeled_events_raises():
    events = [make_event("5", "No identifier at all", ())]
    with pytest.raises(ValueError, match="labeled"):
        run_baseline(events, RecordingEmbedder(), [0.5])


def test_default_thresholds():
    assert DEFAULT_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def test_format_report_layout():
    report = BaselineReport(
        embedder="hashing-256",
        n_events=3,
        n_excluded=2,
        rows=[
            SweepRow(
                threshold=0.8,
                n_clusters=2,
                pairwise=Scores(precision=1.0, recall=0.5, f1=2 / 3),
                bcubed=Scores(precision=0.75, recall=0.25, f1=0.375),
            )
        ],
    )
    lines = format_report(report).splitlines()
    assert "Embedder: hashing-256" in lines
    assert "Events: 3 (excluded: 2)" in lines
    header = (
        "| threshold | clusters | pairwise P | pairwise R | pairwise F1 | B3 P | B3 R | B3 F1 |"
    )
    assert header in lines
    assert lines[lines.index(header) + 1].startswith("|---")
    assert "| 0.80 | 2 | 1.000 | 0.500 | 0.667 | 0.750 | 0.250 | 0.375 |" in lines


def write_jsonl(path, events: list[Event]) -> None:
    lines = [event.model_dump_json() for event in events]
    path.write_text(lines[0] + "\n\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")


def test_load_events_round_trip_and_skips_blank_lines(tmp_path):
    events = sample_events()
    path = tmp_path / "events.jsonl"
    write_jsonl(path, events)
    assert load_events(path) == events
    assert load_events(str(path)) == events


def test_main_default_hashing_embedder(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, sample_events())
    assert main(["--events", str(path), "--thresholds", "0.5,0.9"]) == 0
    out = capsys.readouterr().out
    assert "Embedder: hashing-256" in out
    assert "Events: 3 (excluded: 2)" in out
    assert "| 0.50 |" in out
    assert "| 0.90 |" in out
    assert "| 0.55 |" not in out


def test_main_default_thresholds(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, sample_events())
    assert main(["--events", str(path)]) == 0
    out = capsys.readouterr().out
    for threshold in DEFAULT_THRESHOLDS:
        assert f"| {threshold:.2f} |" in out


def test_main_bge_m3_embedder_uses_flag_embedding(tmp_path, capsys, monkeypatch):
    class FakeModel:
        def __init__(self, model_name, use_fp16=True, **kwargs):
            pass

        def encode(self, sentences, **kwargs):
            return {"dense_vecs": np.ones((len(sentences), 1024))}

    module = types.ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = FakeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)
    path = tmp_path / "events.jsonl"
    write_jsonl(path, sample_events())
    assert main(["--events", str(path), "--embedder", "bge-m3", "--thresholds", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "Embedder: BAAI/bge-m3" in out
    assert "| 0.50 | 1 |" in out


def test_main_rejects_unknown_embedder(tmp_path):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, sample_events())
    with pytest.raises(SystemExit):
        main(["--events", str(path), "--embedder", "word2vec"])


# Review fixes (P1-T7c)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sweep_rejects_non_finite_thresholds(bad):
    gold = {"a": "X", "b": "Y", "c": "Y"}
    with pytest.raises(ValueError, match="finite"):
        sweep(["a", "b", "c"], ABC, gold, [bad, 0.5])
    with pytest.raises(ValueError, match="finite"):
        sweep(["a", "b", "c"], ABC, gold, [0.5, bad])


def test_threshold_clusters_rejects_non_finite_threshold():
    with pytest.raises(ValueError, match="finite"):
        threshold_clusters(ABC, float("nan"))


def test_sweep_rejects_non_finite_vectors():
    vectors = np.array([[1.0, 0.0], [np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        sweep(["a", "b", "c"], vectors, {"a": 1, "b": 1, "c": 1}, [0.5])


def test_sweep_thresholds_above_one_give_singletons():
    gold = {"a": "X", "b": "Y", "c": "Y"}
    rows = sweep(["a", "b", "c"], ABC, gold, [1.5, 2.0])
    assert [row.n_clusters for row in rows] == [3, 3]


def test_sweep_zero_and_negative_zero_are_equal_rows():
    gold = {"a": "X", "b": "Y", "c": "Y"}
    rows = sweep(["a", "b", "c"], ABC, gold, [0.0, -0.0])
    assert rows[0] == rows[1]
    assert rows[0].n_clusters == 1


def test_sweep_matches_threshold_clusters_at_exact_observed_similarities():
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(300, 16))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = vectors @ vectors.T
    observed = [float(sims[0, 1]), float(sims[5, 9]), float(sims[100, 200])]
    ids = [f"e{i}" for i in range(300)]
    gold = {item_id: f"CVE-{i % 40}" for i, item_id in enumerate(ids)}
    thresholds = [*observed, 0.6, 0.2]
    rows = sweep(ids, vectors, gold, thresholds)
    for row, threshold in zip(rows, thresholds, strict=True):
        labels = threshold_clusters(vectors, threshold)
        assert row.n_clusters == len(set(labels))
        assert row.pairwise == pairwise_scores(dict(zip(ids, labels, strict=True)), gold)


def test_sweep_scales_to_thousands_of_events():
    import time

    rng = np.random.default_rng(3)
    n = 2000
    vectors = rng.normal(size=(n, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"e{i}" for i in range(n)]
    gold = {item_id: f"CVE-{i % 500}" for i, item_id in enumerate(ids)}
    start = time.perf_counter()
    rows = sweep(ids, vectors, gold, [0.99, 0.9, 0.5, -1.0])
    elapsed = time.perf_counter() - start
    assert rows[-1].n_clusters == 1
    assert elapsed < 5.0


def test_load_events_reports_line_number_of_malformed_line(tmp_path):
    path = tmp_path / "events.jsonl"
    good = sample_events()[0].model_dump_json()
    path.write_text(good + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"events\.jsonl:2"):
        load_events(path)


@pytest.mark.parametrize("value", ["", ",", "nan", "0.5,inf", "abc"])
def test_main_rejects_bad_thresholds(tmp_path, value):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, sample_events())
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(path), "--thresholds", value])
    assert excinfo.value.code == 2


def test_main_missing_events_file_is_a_usage_error(tmp_path, capsys):
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(missing)])
    assert excinfo.value.code == 2
    assert "nope.jsonl" in capsys.readouterr().err


def test_main_events_directory_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(tmp_path)])
    assert excinfo.value.code == 2


def test_main_malformed_events_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(path)])
    assert excinfo.value.code == 2
    assert "events.jsonl:1" in capsys.readouterr().err


# Review minors (P1-T7e)


def _brute_force_labels(vectors, threshold):
    sims = vectors.astype(np.float64) @ vectors.astype(np.float64).T
    n = len(vectors)
    labels = list(range(n))
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in np.nonzero(sims[i] >= threshold)[0]:
                low = min(labels[i], labels[j])
                if labels[i] != low or labels[j] != low:
                    labels[i] = labels[j] = low
                    changed = True
    return labels


def test_threshold_clusters_across_row_blocks_matches_brute_force():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(60, 6))
    picks = rng.integers(0, 60, size=1100)
    vectors = base[picks] + rng.normal(scale=0.05, size=(1100, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for threshold in (0.995, 0.97):
        assert threshold_clusters(vectors, threshold) == _brute_force_labels(vectors, threshold)


def test_load_events_non_utf8_reports_path(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"\xff\xfe not utf-8\n")
    with pytest.raises(ValueError, match=r"events\.jsonl"):
        load_events(path)


def test_main_non_utf8_events_is_a_usage_error(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"\xff\xfe not utf-8\n")
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(path)])
    assert excinfo.value.code == 2


# Cross-source pairs and singleton reference


def _brute_force_cross_labels(vectors, threshold, groups):
    sims = vectors.astype(np.float64) @ vectors.astype(np.float64).T
    groups = np.asarray(groups, dtype=object)
    n = len(vectors)
    labels = list(range(n))
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in np.nonzero((sims[i] >= threshold) & (groups != groups[i]))[0]:
                low = min(labels[i], labels[j])
                if labels[i] != low or labels[j] != low:
                    labels[i] = labels[j] = low
                    changed = True
    return labels


def test_cross_source_never_joins_identical_same_source_rows():
    vectors = np.array([A, A, A, C])
    groups = ["nvd", "nvd", "nvd", "ghsa"]
    assert threshold_clusters(vectors, 0.9, groups=groups) == [0, 1, 2, 3]
    assert threshold_clusters(vectors, -1.0, groups=groups) == [0, 0, 0, 0]
    assert threshold_clusters(np.array([A, A]), 0.9, groups=["nvd", "nvd"]) == [0, 1]


def test_cross_source_joins_through_other_source():
    vectors = np.array([A, A, A])
    assert threshold_clusters(vectors, 0.9, groups=["nvd", "ghsa", "nvd"]) == [0, 0, 0]


def test_groups_none_keeps_all_pairs_behavior():
    vectors = np.array([A, A, C])
    assert threshold_clusters(vectors, 0.9) == [0, 0, 2]
    assert threshold_clusters(vectors, 0.9, groups=None) == [0, 0, 2]


def test_single_group_gives_singletons():
    vectors = np.array([A, A, B])
    assert threshold_clusters(vectors, -1.0, groups=["x", "x", "x"]) == [0, 1, 2]
    assert threshold_clusters(np.array([A]), 0.5, groups=["x"]) == [0]
    assert threshold_clusters(np.zeros((0, 2)), 0.5, groups=[]) == []


@pytest.mark.parametrize("n_groups", [2, 3])
def test_cross_source_matches_brute_force_across_row_blocks(n_groups):
    rng = np.random.default_rng(9)
    base = rng.normal(size=(50, 6))
    picks = rng.integers(0, 50, size=1300)
    vectors = base[picks] + rng.normal(scale=0.05, size=(1300, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # Unbalanced groups, interleaved in row order.
    groups = [Source.GHSA if i % 7 == 0 else Source.NVD for i in range(1300)]
    if n_groups == 3:
        groups = [g if i % 11 else "osv" for i, g in enumerate(groups)]
    for threshold in (0.995, 0.97):
        expected = _brute_force_cross_labels(vectors, threshold, groups)
        assert threshold_clusters(vectors, threshold, groups=groups) == expected


def test_sweep_with_groups_matches_threshold_clusters():
    rng = np.random.default_rng(13)
    vectors = rng.normal(size=(60, 5))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"e{i}" for i in range(60)]
    groups = ["nvd" if i % 3 else "ghsa" for i in range(60)]
    gold = {item_id: f"CVE-{i % 9}" for i, item_id in enumerate(ids)}
    thresholds = [0.9, 0.4, 0.0, -1.0]
    rows = sweep(ids, vectors, gold, thresholds, groups=groups)
    for row, threshold in zip(rows, thresholds, strict=True):
        labels = threshold_clusters(vectors, threshold, groups=groups)
        pred = dict(zip(ids, labels, strict=True))
        assert row.n_clusters == len(set(labels))
        assert row.pairwise == pairwise_scores(pred, gold)
        assert row.bcubed == bcubed_scores(pred, gold)


def test_groups_length_mismatch_raises():
    with pytest.raises(ValueError, match="groups"):
        threshold_clusters(ABC, 0.5, groups=["nvd", "ghsa"])
    with pytest.raises(ValueError, match="groups"):
        sweep(["a", "b", "c"], ABC, {"a": 1, "b": 1, "c": 1}, [0.5], groups=["nvd"])


def test_singletons_reference_hand_computed():
    gold = {"a": "X", "b": "X", "c": "Y"}
    reference = singletons_reference(["a", "b", "c"], gold)
    assert reference.name == "singletons"
    assert reference.n_clusters == 3
    # Pairwise precision is undefined (no predicted pairs); metrics report 1.0 by convention.
    assert reference.pairwise == Scores(precision=1.0, recall=0.0, f1=0.0)
    assert reference.bcubed.precision == 1.0
    assert reference.bcubed.recall == pytest.approx(2 / 3)
    assert reference.bcubed.f1 == pytest.approx(0.8)


def test_pair_modes():
    assert PAIR_MODES == ("cross-source", "all")


def test_run_baseline_default_is_cross_source_with_reference():
    events = [
        make_event("1", "heap overflow in libfoo", ("CVE-2024-0001",)),
        make_event("2", "heap overflow in libfoo again", ("CVE-2024-0001",)),
        make_event("3", "XSS in admin page", ("CVE-2024-0002",)),
    ]
    report = run_baseline(events, RecordingEmbedder(), [0.9])
    assert report.pairs == "cross-source"
    assert report.rows[0].n_clusters == 3
    assert report.reference == singletons_reference(
        ["nvd:1", "nvd:2", "nvd:3"],
        {"nvd:1": "CVE-2024-0001", "nvd:2": "CVE-2024-0001", "nvd:3": "CVE-2024-0002"},
    )
    all_report = run_baseline(events, RecordingEmbedder(), [0.9], pairs="all")
    assert all_report.pairs == "all"
    assert all_report.rows[0].n_clusters == 2
    assert all_report.rows[0].pairwise == Scores(precision=1.0, recall=1.0, f1=1.0)
    assert all_report.reference == report.reference


def test_run_baseline_rejects_unknown_pairs():
    with pytest.raises(ValueError, match="pairs"):
        run_baseline(sample_events(), RecordingEmbedder(), [0.5], pairs="same-source")


def test_format_report_shows_pairs_reference_and_na():
    reference = singletons_reference(["a", "b", "c"], {"a": "X", "b": "X", "c": "Y"})
    report = BaselineReport(
        embedder="hashing-256",
        n_events=3,
        n_excluded=0,
        rows=[
            SweepRow(
                threshold=0.95,
                n_clusters=3,
                pairwise=Scores(precision=1.0, recall=0.0, f1=0.0),
                bcubed=reference.bcubed,
            ),
            SweepRow(
                threshold=0.5,
                n_clusters=2,
                pairwise=Scores(precision=1.0, recall=1.0, f1=1.0),
                bcubed=Scores(precision=1.0, recall=1.0, f1=1.0),
            ),
        ],
        pairs="cross-source",
        reference=reference,
    )
    lines = format_report(report).splitlines()
    assert "Pairs: cross-source" in lines
    ref_line = "| singletons | 3 | n/a | 0.000 | 0.000 | 1.000 | 0.667 | 0.800 |"
    assert ref_line in lines
    assert lines.index(ref_line) < lines.index(
        "| 0.50 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |"
    )
    assert lines[lines.index(ref_line) - 1].startswith("|---")
    assert "| 0.95 | 3 | n/a | 0.000 | 0.000 | 1.000 | 0.667 | 0.800 |" in lines


def test_format_report_without_pairs_or_reference_has_no_extra_lines():
    report = BaselineReport(embedder="e", n_events=1, n_excluded=0, rows=[])
    text = format_report(report)
    assert "Pairs:" not in text
    assert "singletons" not in text


def test_main_pairs_option(tmp_path, capsys):
    events = [
        make_event("1", "heap overflow in libfoo", ("CVE-2024-0001",)),
        make_event("2", "heap overflow in libfoo again", ("CVE-2024-0001",)),
        make_event("3", "XSS in admin page", ("CVE-2024-0002",)),
    ]
    path = tmp_path / "events.jsonl"
    write_jsonl(path, events)
    assert main(["--events", str(path), "--thresholds", "-1"]) == 0
    out = capsys.readouterr().out
    assert "Pairs: cross-source" in out
    assert "| -1.00 | 3 |" in out
    assert "| singletons | 3 |" in out
    assert main(["--events", str(path), "--thresholds", "-1", "--pairs", "all"]) == 0
    out = capsys.readouterr().out
    assert "Pairs: all" in out
    assert "| -1.00 | 1 |" in out


def test_main_rejects_unknown_pairs(tmp_path):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, sample_events())
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(path), "--pairs", "same-source"])
    assert excinfo.value.code == 2


def test_cross_source_joins_pair_at_exact_threshold():
    # 1 * 0.5 + 0 * sqrt(0.75) is exactly 0.5 in binary floating point.
    vectors = np.array([[1.0, 0.0], [0.5, np.sqrt(0.75)]])
    assert threshold_clusters(vectors, 0.5, groups=["nvd", "ghsa"]) == [0, 0]


def test_format_report_numeric_precision_when_pairs_predicted_with_zero_recall():
    report = BaselineReport(
        embedder="e",
        n_events=3,
        n_excluded=0,
        rows=[
            SweepRow(
                threshold=0.5,
                n_clusters=2,
                pairwise=Scores(precision=0.0, recall=0.0, f1=0.0),
                bcubed=Scores(precision=1.0, recall=1.0, f1=1.0),
            )
        ],
    )
    lines = format_report(report).splitlines()
    assert "| 0.50 | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |" in lines


def test_edge_indices_are_int32():
    from semtriage.baseline import _collect_cross_edges, _collect_edges

    vectors = np.array([A, A, C])
    for _, lefts, rights in (
        _collect_edges(vectors, -1.0),
        _collect_cross_edges(vectors, ["nvd", "ghsa", "nvd"], -1.0),
        _collect_edges(vectors, 2.0),
        _collect_cross_edges(vectors, ["nvd", "nvd", "nvd"], -1.0),
    ):
        assert lefts.dtype == np.int32
        assert rights.dtype == np.int32


@pytest.mark.parametrize("threshold", [0.9, 0.8, 0.6, -1.0])
@pytest.mark.parametrize("groups", [None, ["nvd", "ghsa", "nvd"]])
def test_edge_clusters_match_threshold_clusters(threshold, groups):
    similarities, lefts, rights = candidate_edges(ABC, threshold, groups=groups)
    assert np.all(similarities >= threshold)
    assert np.all(lefts < rights)
    expected = threshold_clusters(ABC, threshold, groups=groups)
    assert edge_clusters(len(ABC), lefts, rights) == expected
