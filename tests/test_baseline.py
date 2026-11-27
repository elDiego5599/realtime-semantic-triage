import sys
import types
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import (
    DEFAULT_THRESHOLDS,
    PAIR_MODES,
    RULES,
    BaselineReport,
    SweepRow,
    candidate_edges,
    edge_clusters,
    format_report,
    gold_label,
    load_events,
    main,
    mutual_best_clusters,
    mutual_best_pairs,
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


# Row 0 (nvd) and row 1 (ghsa) are the true pair (dot 0.9). Row 2 (nvd) has its best
# match in row 1 (dot 0.8), but row 1 prefers row 0, so only single link chains row 2 in.
CHAIN = np.array([[0.9, np.sqrt(0.19)], [1.0, 0.0], [0.8, -0.6]])
CHAIN_GROUPS = ["nvd", "ghsa", "nvd"]


def brute_force_mutual(vectors, groups, threshold):
    """Reference rule from the full cross matrix (lowest index wins ties)."""
    groups = np.asarray(groups)
    first, second = np.unique(groups)[:2]
    rows_a = np.flatnonzero(groups == first)
    rows_b = np.flatnonzero(groups == second)
    full = vectors[rows_a] @ vectors[rows_b].T
    best_b = full.argmax(axis=1)
    best_a = full.argmax(axis=0)
    labels = list(range(len(vectors)))
    for i, j in enumerate(best_b):
        if best_a[j] == i and full[i, j] >= threshold:
            left, right = sorted((int(rows_a[i]), int(rows_b[j])))
            labels[right] = left
    return labels


def test_rules():
    assert RULES == ("single-link", "mutual-best")


def test_mutual_best_links_only_true_pair_where_single_link_chains():
    single = sweep(
        ["a", "b", "c"],
        CHAIN,
        {"a": "X", "b": "X", "c": "Y"},
        [0.7],
        groups=CHAIN_GROUPS,
    )[0]
    mutual = sweep(
        ["a", "b", "c"],
        CHAIN,
        {"a": "X", "b": "X", "c": "Y"},
        [0.7],
        groups=CHAIN_GROUPS,
        rule="mutual-best",
    )[0]
    assert single.n_clusters == 1
    assert mutual.n_clusters == 2
    assert mutual.pairwise.f1 == 1.0
    assert mutual_best_clusters(CHAIN, 0.7, CHAIN_GROUPS) == [0, 0, 2]


def test_mutual_best_pairs_output():
    similarities, lefts, rights = mutual_best_pairs(CHAIN, CHAIN_GROUPS)
    assert lefts.tolist() == [0]
    assert rights.tolist() == [1]
    assert similarities.tolist() == pytest.approx([0.9])
    assert lefts.dtype == np.int32
    assert rights.dtype == np.int32


def test_threshold_cuts_mutual_pair_and_keeps_exact_match():
    # 1 * 0.5 + 0 * sqrt(0.75) is exactly 0.5 in binary floating point.
    vectors = np.array([[1.0, 0.0], [0.5, np.sqrt(0.75)]])
    groups = ["nvd", "ghsa"]
    assert mutual_best_clusters(vectors, 0.5, groups) == [0, 0]
    assert mutual_best_clusters(vectors, np.nextafter(0.5, 1.0), groups) == [0, 1]


@pytest.mark.parametrize("block_rows", [1, 2, 1024])
def test_ties_pick_lowest_index(block_rows):
    # Row 0 (ghsa) ties between nvd rows 1 and 2; nvd row 3 ties between ghsa rows 4 and 5.
    vectors = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    groups = ["ghsa", "nvd", "nvd", "nvd", "ghsa", "ghsa"]
    _, lefts, rights = mutual_best_pairs(vectors, groups, block_rows=block_rows)
    assert list(zip(lefts.tolist(), rights.tolist(), strict=True)) == [(0, 1), (3, 4)]


@pytest.mark.parametrize("seed", range(5))
def test_block_size_independence_and_brute_force(seed):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(41, 6))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    groups = rng.choice(["nvd", "ghsa"], size=len(vectors)).tolist()
    small = mutual_best_pairs(vectors, groups, block_rows=1)
    large = mutual_best_pairs(vectors, groups, block_rows=10_000)
    # Pairs must match exactly; BLAS may round a block's dot products in the last bit.
    np.testing.assert_array_equal(small[1], large[1])
    np.testing.assert_array_equal(small[2], large[2])
    np.testing.assert_allclose(small[0], large[0], rtol=0, atol=1e-12)
    assert len(small[1]) > 0
    for threshold in (-1.0, 0.0, 0.3, 0.6):
        assert mutual_best_clusters(vectors, threshold, groups, block_rows=1) == (
            brute_force_mutual(vectors, groups, threshold)
        )


def test_sweep_mutual_best_matches_per_threshold_clusters():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(30, 4))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    groups = ["nvd" if i % 3 else "ghsa" for i in range(30)]
    ids = [str(i) for i in range(30)]
    gold = {item: int(item) // 2 for item in ids}
    thresholds = [0.9, 0.2, 0.5, 0.2]
    rows = sweep(ids, vectors, gold, thresholds, groups=groups, rule="mutual-best")
    for row, threshold in zip(rows, thresholds, strict=True):
        labels = mutual_best_clusters(vectors, threshold, groups)
        pred = dict(zip(ids, labels, strict=True))
        assert row.threshold == threshold
        assert row.n_clusters == len(set(labels))
        assert row.pairwise == pairwise_scores(pred, gold)
        assert row.bcubed == bcubed_scores(pred, gold)


def test_sweep_mutual_best_computes_best_matches_once(monkeypatch):
    import semtriage.baseline as baseline

    calls = []
    original = baseline.mutual_best_pairs

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(baseline, "mutual_best_pairs", counting)
    ids = ["a", "b", "c"]
    sweep(
        ids,
        CHAIN,
        {"a": 1, "b": 1, "c": 2},
        [0.9, 0.7, 0.5],
        groups=CHAIN_GROUPS,
        rule="mutual-best",
    )
    assert len(calls) == 1


def test_single_group_or_empty_gives_no_pairs():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert mutual_best_clusters(vectors, -1.0, ["nvd", "nvd"]) == [0, 1]
    assert mutual_best_clusters(np.empty((0, 2)), 0.5, []) == []


def test_more_than_two_groups_rejected():
    vectors = np.eye(3)
    with pytest.raises(ValueError, match="two"):
        mutual_best_pairs(vectors, ["a", "b", "c"])


def test_mutual_best_requires_groups():
    with pytest.raises(ValueError, match="groups"):
        mutual_best_pairs(CHAIN, None)
    with pytest.raises(ValueError, match="cross-source"):
        sweep(["a", "b", "c"], CHAIN, {"a": 1, "b": 1, "c": 2}, [0.5], rule="mutual-best")


def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="rule"):
        sweep(["a"], np.ones((1, 2)), {"a": 1}, [0.5], groups=["nvd"], rule="average")


def cross_events() -> list[Event]:
    return [
        make_event("1", "heap overflow in libfoo", ("CVE-2024-0001",)),
        make_event("2", "heap overflow in libfoo", ("CVE-2024-0001",), Source.GHSA),
        make_event("3", "heap overflow in libbar", ("CVE-2024-0002",)),
    ]


def test_run_baseline_rule():
    single = run_baseline(cross_events(), RecordingEmbedder(), [0.9])
    assert single.rule == "single-link"
    assert single.rows[0].n_clusters == 1
    mutual = run_baseline(cross_events(), RecordingEmbedder(), [0.9], rule="mutual-best")
    assert mutual.rule == "mutual-best"
    assert mutual.rows[0].n_clusters == 2


def test_run_baseline_rejects_mutual_best_with_all_pairs_before_embedding():
    embedder = RecordingEmbedder()
    with pytest.raises(ValueError, match="cross-source"):
        run_baseline(cross_events(), embedder, [0.9], pairs="all", rule="mutual-best")
    assert embedder.texts == []


def test_mutual_best_rejects_bad_block_rows():
    with pytest.raises(ValueError, match="block_rows"):
        mutual_best_pairs(CHAIN, CHAIN_GROUPS, block_rows=0)


def test_main_rejects_mutual_best_with_all_pairs_before_reading_events(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            ["--events", str(tmp_path / "missing.jsonl"), "--rule", "mutual-best", "--pairs", "all"]
        )
    assert excinfo.value.code == 2
    assert "--rule mutual-best" in capsys.readouterr().err


def test_run_baseline_rejects_unknown_rule():
    with pytest.raises(ValueError, match="rule"):
        run_baseline(cross_events(), RecordingEmbedder(), [0.9], rule="average")


def test_main_rule_flag(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, cross_events())
    assert main(["--events", str(path), "--thresholds", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "Rule:" not in out
    assert main(["--events", str(path), "--thresholds", "0.5", "--rule", "mutual-best"]) == 0
    out = capsys.readouterr().out
    assert "Rule: mutual-best" in out.splitlines()
    assert "| 0.50 | 2 |" in out


@pytest.mark.parametrize(
    "extra",
    [["--rule", "average"], ["--rule", "mutual-best", "--pairs", "all"]],
)
def test_main_rule_errors_are_usage_errors(tmp_path, extra):
    path = tmp_path / "events.jsonl"
    write_jsonl(path, cross_events())
    with pytest.raises(SystemExit) as excinfo:
        main(["--events", str(path), *extra])
    assert excinfo.value.code == 2


def test_parse_thresholds_is_shared_with_holdout() -> None:
    from semtriage import baseline, holdout

    assert holdout.parse_thresholds is baseline.parse_thresholds
    assert baseline.parse_thresholds(" 0.7, 0.8 ,") == (0.7, 0.8)


@pytest.mark.parametrize("value", ["", " , ", "a,b", "0.7,nan", "inf"])
def test_parse_thresholds_rejects_bad_values(value) -> None:
    import argparse

    from semtriage.baseline import parse_thresholds

    with pytest.raises(argparse.ArgumentTypeError):
        parse_thresholds(value)


def test_precision_cell_is_na_without_predicted_pairs() -> None:
    from semtriage.baseline import precision_cell
    from semtriage.metrics import Scores

    scores = Scores(precision=0.25, recall=0.5, f1=1 / 3)
    assert precision_cell(scores, n_clusters=4, n_events=4) == "n/a"
    assert precision_cell(scores, n_clusters=3, n_events=4) == "0.250"
