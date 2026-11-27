import math
from datetime import UTC, datetime

import numpy as np
import pytest

from semtriage.baseline import RULES, SweepRow, sweep, write_embeddings
from semtriage.holdout import (
    DEFAULT_THRESHOLDS,
    evaluate_holdout,
    format_holdout,
    main,
    pick_threshold,
    split_by_cve,
)
from semtriage.metrics import Scores
from semtriage.models import Event, Source
from semtriage.sanitize import sanitize


def make_event(
    source: Source, cve: str | None, day: int, text: str = "text", month: int = 1
) -> Event:
    source_id = f"{cve or 'none'}-{day}-{month}"
    when = datetime(2024, month, day, tzinfo=UTC)
    return Event(
        event_id=f"{source.value}:{source_id}",
        source=source,
        source_id=source_id,
        published=when,
        modified=when,
        text=text,
        cve_ids=(cve,) if cve else (),
        cwe_ids=(),
        ecosystems=(),
        packages=(),
        cna=None,
        severity=None,
    )


def pair_vectors(similarities: list[float], singles: int) -> np.ndarray:
    """Rows 2k and 2k+1 have dot product similarities[k]; every other dot product is 0.

    After the pairs come `singles` rows, each orthogonal to everything else.
    """
    dims = 2 * len(similarities) + singles
    rows = []
    for index, similarity in enumerate(similarities):
        first = np.zeros(dims)
        first[2 * index] = 1.0
        second = np.zeros(dims)
        second[2 * index] = similarity
        second[2 * index + 1] = math.sqrt(1.0 - similarity**2)
        rows += [first, second]
    for index in range(singles):
        row = np.zeros(dims)
        row[2 * len(similarities) + index] = 1.0
        rows.append(row)
    return np.array(rows)


def fixture(test_similarity: float = 0.65) -> tuple[list[Event], np.ndarray]:
    """Six CVEs; with train fraction 0.7 the oldest four are the tune split.

    Tune: CVE-1 duplicate at 0.75, CVE-2 duplicate at 0.85 (its GHSA copy is published
    after every test CVE), and a false cross-source pair CVE-3 (NVD) x CVE-4 (GHSA) at
    0.72. Test: CVE-5 duplicate at `test_similarity`, CVE-6 a lone NVD event.
    """
    events = [
        make_event(Source.NVD, "CVE-1", 1),
        make_event(Source.GHSA, "CVE-1", 2),
        make_event(Source.NVD, "CVE-2", 3),
        make_event(Source.GHSA, "CVE-2", 1, month=3),
        make_event(Source.NVD, "CVE-3", 4),
        make_event(Source.GHSA, "CVE-4", 5),
        make_event(Source.NVD, "CVE-5", 1, month=2),
        make_event(Source.GHSA, "CVE-5", 2, month=2),
        make_event(Source.NVD, "CVE-6", 3, month=2),
    ]
    return events, pair_vectors([0.75, 0.85, 0.72, test_similarity], singles=1)


THRESHOLDS = (0.6, 0.65, 0.7, 0.73, 0.74, 0.75, 0.8, 0.9)


def test_default_thresholds_are_a_fine_grid() -> None:
    assert DEFAULT_THRESHOLDS[0] == 0.6
    assert DEFAULT_THRESHOLDS[-1] == 0.9
    assert len(DEFAULT_THRESHOLDS) == 31
    assert DEFAULT_THRESHOLDS[1] == 0.61


def test_split_keeps_both_advisories_of_a_cve_on_one_side() -> None:
    events, _ = fixture()
    tune, test = split_by_cve(events, 0.7)
    assert sorted(tune + test) == list(range(len(events)))
    tune_cves = {events[index].cve_ids[0] for index in tune}
    test_cves = {events[index].cve_ids[0] for index in test}
    assert tune_cves == {"CVE-1", "CVE-2", "CVE-3", "CVE-4"}
    assert test_cves == {"CVE-5", "CVE-6"}
    # CVE-2's GHSA copy is published after the test CVEs but follows its CVE.
    assert 3 in tune


def test_split_orders_cves_by_earliest_published_not_input_order() -> None:
    events = [
        make_event(Source.NVD, "CVE-LATE", 9),
        make_event(Source.NVD, "CVE-EARLY", 1),
        make_event(Source.GHSA, "CVE-MID", 5),
        make_event(Source.NVD, "CVE-MID", 20),
    ]
    tune, test = split_by_cve(events, 0.7)
    assert sorted(tune) == [1, 2, 3]
    assert test == [0]


def test_split_ties_on_date_break_by_cve_id() -> None:
    events = [
        make_event(Source.NVD, "CVE-B", 1),
        make_event(Source.NVD, "CVE-A", 1),
    ]
    tune, test = split_by_cve(events, 0.5)
    assert tune == [1]
    assert test == [0]


@pytest.mark.parametrize(("fraction", "n_tune"), [(0.7, 3), (0.625, 3), (0.6, 2)])
def test_split_count_rounds_half_up(fraction: float, n_tune: int) -> None:
    events = [make_event(Source.NVD, f"CVE-{day}", day) for day in range(1, 5)]
    tune, _ = split_by_cve(events, fraction)
    assert tune == list(range(n_tune))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_split_rejects_fraction_outside_open_interval(fraction: float) -> None:
    events, _ = fixture()
    with pytest.raises(ValueError, match="train fraction"):
        split_by_cve(events, fraction)


def test_split_rejects_an_empty_side() -> None:
    events = [make_event(Source.NVD, "CVE-1", 1), make_event(Source.NVD, "CVE-2", 2)]
    with pytest.raises(ValueError, match="empty"):
        split_by_cve(events, 0.1)


def test_split_rejects_unlabeled_events() -> None:
    events = [make_event(Source.NVD, "CVE-1", 1), make_event(Source.NVD, None, 2)]
    with pytest.raises(ValueError, match="CVE"):
        split_by_cve(events, 0.5)


def row(threshold: float, f1: float) -> SweepRow:
    scores = Scores(precision=f1, recall=f1, f1=f1)
    return SweepRow(threshold=threshold, n_clusters=1, pairwise=scores, bcubed=scores)


def test_pick_threshold_takes_best_pairwise_f1() -> None:
    rows = [row(0.6, 0.2), row(0.7, 0.9), row(0.8, 0.5)]
    assert pick_threshold(rows).threshold == 0.7


def test_pick_threshold_ties_go_to_lowest_threshold() -> None:
    rows = [row(0.8, 0.9), row(0.6, 0.5), row(0.75, 0.9), row(0.9, 0.9)]
    assert pick_threshold(rows).threshold == 0.75


def test_pick_threshold_ranks_by_pairwise_f1_not_bcubed() -> None:
    pairwise_best = SweepRow(
        threshold=0.8,
        n_clusters=1,
        pairwise=Scores(precision=0.9, recall=0.9, f1=0.9),
        bcubed=Scores(precision=0.5, recall=0.5, f1=0.5),
    )
    bcubed_best = SweepRow(
        threshold=0.7,
        n_clusters=1,
        pairwise=Scores(precision=0.4, recall=0.4, f1=0.4),
        bcubed=Scores(precision=0.99, recall=0.99, f1=0.99),
    )
    assert pick_threshold([bcubed_best, pairwise_best]) is pairwise_best


def test_pick_threshold_rejects_no_rows() -> None:
    with pytest.raises(ValueError):
        pick_threshold([])


@pytest.mark.parametrize("rule", RULES)
def test_selected_threshold_and_scores_for_each_rule(rule: str) -> None:
    events, vectors = fixture()
    report = evaluate_holdout(events, vectors, THRESHOLDS, train_fraction=0.7, rules=(rule,))
    assert [result.rule for result in report.results] == [rule]
    result = report.results[0]
    # 0.73, 0.74 and 0.75 all give tune F1 1.0; the lowest wins.
    assert result.tune.threshold == 0.73
    assert result.tune.pairwise.f1 == pytest.approx(1.0)
    assert result.test.threshold == 0.73
    assert result.test.pairwise.recall == pytest.approx(0.0)
    assert result.oracle.threshold == 0.6
    assert result.oracle.pairwise.f1 == pytest.approx(1.0)
    assert report.n_tune_events == 6
    assert report.n_test_events == 3
    assert report.n_tune_cves == 4
    assert report.n_test_cves == 2


def test_default_rules_are_both_rules() -> None:
    events, vectors = fixture()
    report = evaluate_holdout(events, vectors, THRESHOLDS, train_fraction=0.7)
    assert [result.rule for result in report.results] == list(RULES)


@pytest.mark.parametrize("rule", RULES)
def test_test_split_signal_does_not_change_the_selected_threshold(rule: str) -> None:
    low_events, low_vectors = fixture(test_similarity=0.65)
    high_events, high_vectors = fixture(test_similarity=0.95)
    low = evaluate_holdout(low_events, low_vectors, THRESHOLDS, train_fraction=0.7, rules=(rule,))
    high = evaluate_holdout(
        high_events, high_vectors, THRESHOLDS, train_fraction=0.7, rules=(rule,)
    )
    assert low.results[0].tune == high.results[0].tune
    assert low.results[0].test.threshold == high.results[0].test.threshold == 0.73
    assert low.results[0].test.pairwise.f1 == pytest.approx(0.0)
    assert high.results[0].test.pairwise.f1 == pytest.approx(1.0)


def test_clustering_runs_within_each_split_only() -> None:
    # A tune NVD event and a test GHSA event are near-identical but belong to
    # different CVEs; if splits were clustered together the mutual best match of
    # the tune pair would move to the test event.
    events = [
        make_event(Source.NVD, "CVE-1", 1),
        make_event(Source.GHSA, "CVE-1", 2),
        make_event(Source.GHSA, "CVE-2", 1, month=2),
        make_event(Source.NVD, "CVE-2", 2, month=2),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.99, 0.0, math.sqrt(1 - 0.99**2)],
            [0.0, 0.0, 1.0],
        ]
    )
    report = evaluate_holdout(
        events, vectors, (0.1, 0.5, 0.79), train_fraction=0.5, rules=("mutual-best",)
    )
    result = report.results[0]
    assert result.tune.pairwise.f1 == pytest.approx(1.0)
    assert result.tune.threshold == 0.1


def test_single_link_does_not_bridge_tune_events_through_a_test_event() -> None:
    # The test GHSA event is close to two tune NVD events of different CVEs; clustered
    # together, single link would merge them into a false tune pair.
    events = [
        make_event(Source.NVD, "CVE-1", 1),
        make_event(Source.GHSA, "CVE-1", 2),
        make_event(Source.NVD, "CVE-3", 3),
        make_event(Source.GHSA, "CVE-2", 1, month=2),
    ]
    half = math.sqrt(0.5)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
            [half, 0.0, half],
        ]
    )
    report = evaluate_holdout(
        events, vectors, (0.5, 0.7), train_fraction=0.67, rules=("single-link",)
    )
    result = report.results[0]
    assert report.n_test_events == 1
    assert result.tune.pairwise.f1 == pytest.approx(1.0)
    assert result.tune.pairwise.precision == pytest.approx(1.0)


def test_evaluate_rejects_mismatched_vectors() -> None:
    events, vectors = fixture()
    with pytest.raises(ValueError, match="vectors"):
        evaluate_holdout(events, vectors[:-1], THRESHOLDS, train_fraction=0.7)


def test_format_holdout_prints_a_markdown_table() -> None:
    events, vectors = fixture()
    report = evaluate_holdout(events, vectors, THRESHOLDS, train_fraction=0.7)
    text = format_holdout(report, embedder="test-embedder")
    assert "Embedder: test-embedder" in text
    assert "Tune: 6 events, 4 CVEs" in text
    assert "Test: 3 events, 2 CVEs" in text
    assert "| rule | split | threshold |" in text
    assert "| single-link | tune | 0.73 |" in text
    assert "| single-link | test | 0.73 |" in text
    assert "| single-link | test oracle | 0.60 |" in text
    assert "| mutual-best | tune | 0.73 |" in text
    assert "| single-link | tune (size-matched) | 0.73 |" in text
    assert "Tune: 6 events, 4 CVEs (last CVE published 2024-01-05T00:00:00+00:00)" in text
    assert "Test: 3 events, 2 CVEs (first CVE published 2024-02-01T00:00:00+00:00)" in text
    assert "Size-matched tune: 2 events, 2 CVEs (latest tune CVEs)" in text
    assert "own events only" in text
    assert "share the boundary date" not in text


def test_format_holdout_notes_a_shared_boundary_date() -> None:
    events = [
        make_event(Source.NVD, "CVE-A", 1),
        make_event(Source.GHSA, "CVE-A", 2),
        make_event(Source.NVD, "CVE-B", 1, text="other"),
        make_event(Source.GHSA, "CVE-B", 3),
    ]
    vectors = pair_vectors([0.9, 0.9], singles=0)
    report = evaluate_holdout(events, vectors, (0.5,), train_fraction=0.5)
    assert report.tune_end == report.test_start
    assert "share the boundary date" in format_holdout(report, embedder="e")


@pytest.mark.parametrize("rule", RULES)
def test_size_matched_tune_row_scores_the_latest_tune_cves(rule: str) -> None:
    events, vectors = fixture()
    report = evaluate_holdout(events, vectors, THRESHOLDS, train_fraction=0.7, rules=(rule,))
    result = report.results[0]
    # Two test CVEs, so the two latest tune CVEs: CVE-3 and CVE-4 (rows 4 and 5),
    # whose cross-source pair at 0.72 stays apart at the chosen 0.73.
    assert report.n_matched_cves == 2
    assert report.n_matched_events == 2
    subset = events[4:6]
    expected = sweep(
        [event.event_id for event in subset],
        vectors[4:6],
        {event.event_id: event.cve_ids[0] for event in subset},
        [0.73],
        groups=[event.source for event in subset],
        rule=rule,
    )[0]
    assert result.tune_matched == expected


def test_size_matched_tune_row_uses_whole_tune_split_when_test_is_larger() -> None:
    events = [
        make_event(Source.NVD, "CVE-1", 1),
        make_event(Source.GHSA, "CVE-1", 2),
        make_event(Source.NVD, "CVE-2", 3),
        make_event(Source.NVD, "CVE-3", 4),
        make_event(Source.NVD, "CVE-4", 5),
    ]
    vectors = pair_vectors([0.9], singles=3)
    report = evaluate_holdout(events, vectors, (0.5, 0.95), train_fraction=0.25)
    assert report.n_matched_cves == report.n_tune_cves == 1
    assert report.results[0].tune_matched == report.results[0].tune


def test_format_holdout_shows_na_precision_without_predicted_pairs() -> None:
    events, vectors = fixture()
    report = evaluate_holdout(events, vectors, THRESHOLDS, train_fraction=0.7)
    line = next(
        line
        for line in format_holdout(report, embedder="e").splitlines()
        if line.startswith("| single-link | test | ")
    )
    assert line.split("|")[5].strip() == "n/a"


def write_inputs(tmp_path, events: list[Event], vectors: np.ndarray) -> tuple[str, str]:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(event.model_dump_json() + "\n" for event in events))
    embeddings_path = tmp_path / "vectors.npz"
    write_embeddings(
        embeddings_path,
        [event.event_id for event in events],
        [sanitize(event.text) for event in events],
        vectors,
        embedder="test-embedder",
    )
    return str(events_path), str(embeddings_path)


def test_cli_runs_on_cached_vectors(tmp_path, capsys) -> None:
    events, vectors = fixture()
    events_path, embeddings_path = write_inputs(tmp_path, events, vectors)
    # An unlabeled event in the file is dropped, as in the baseline.
    with open(events_path, "a") as target:
        target.write(make_event(Source.GHSA, None, 9).model_dump_json() + "\n")
    code = main(
        [
            "--events",
            events_path,
            "--embeddings",
            embeddings_path,
            "--thresholds",
            ",".join(str(threshold) for threshold in THRESHOLDS),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Embedder: test-embedder" in out
    assert "excluded: 1" in out
    assert "| mutual-best | tune | 0.73 |" in out


def test_cli_rejects_bad_train_fraction(tmp_path) -> None:
    events, vectors = fixture()
    events_path, embeddings_path = write_inputs(tmp_path, events, vectors)
    with pytest.raises(SystemExit) as exc:
        main(["--events", events_path, "--embeddings", embeddings_path, "--train-fraction", "1"])
    assert exc.value.code == 2


def test_cli_rejects_stale_embeddings(tmp_path) -> None:
    events, vectors = fixture()
    events_path, embeddings_path = write_inputs(tmp_path, events[:-1], vectors[:-1])
    with open(events_path, "a") as target:
        target.write(events[-1].model_dump_json() + "\n")
    with pytest.raises(SystemExit) as exc:
        main(["--events", events_path, "--embeddings", embeddings_path])
    assert exc.value.code == 2


def test_cli_rejects_malformed_thresholds(tmp_path) -> None:
    events, vectors = fixture()
    events_path, embeddings_path = write_inputs(tmp_path, events, vectors)
    with pytest.raises(SystemExit) as exc:
        main(["--events", events_path, "--embeddings", embeddings_path, "--thresholds", "a,b"])
    assert exc.value.code == 2
