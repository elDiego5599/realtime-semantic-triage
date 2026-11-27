"""Held-out check of the linking threshold on a later time split.

CVEs are ordered by the earliest `published` date of their advisories. The oldest
fraction is the tune split and the rest the test split, so both advisories of a CVE
stay on the same side. For each linking rule the threshold with the best pairwise F1
on the tune split (ties: lowest threshold) is then scored on the test split, next to
the test split's own best (oracle) threshold. Clustering runs within each split only.

The cut is by CVE, not by date: a tune CVE may have a later advisory published after
the first test CVE, and CVEs that share the boundary date can land on both sides.

Each split is clustered and scored against its own events only, so a smaller split
has fewer wrong candidates per event and its precision is not directly comparable
with a larger one. The report therefore adds a size-matched tune row: the latest tune
CVEs, as many as the test split has, scored at the chosen threshold.
"""

import argparse
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from semtriage.baseline import (
    RULES,
    SweepRow,
    gold_label,
    labeled_events,
    load_events,
    read_embeddings,
    read_text_mode,
    sweep,
)
from semtriage.models import Event
from semtriage.text_mode import build_text, parse_text_mode_label

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.60 + 0.01 * step, 2) for step in range(31))

DEFAULT_TRAIN_FRACTION = 0.7


def split_by_cve(events: Sequence[Event], train_fraction: float) -> tuple[list[int], list[int]]:
    """Split event indices into (tune, test) by CVE, oldest CVEs first.

    CVEs are ordered by the earliest `published` date of their events (ties by CVE
    id). The first train_fraction * n_cves CVEs, rounded half up, form the tune split. Every
    event must carry exactly one CVE id, and both splits must be non-empty.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train fraction must be between 0 and 1, got {train_fraction}")
    labels, ordered, _ = _order_cves(events)
    n_tune = math.floor(train_fraction * len(ordered) + 0.5)
    if n_tune == 0 or n_tune == len(ordered):
        raise ValueError(
            f"train fraction {train_fraction} leaves a split empty ({len(ordered)} CVEs)"
        )
    tune_cves = set(ordered[:n_tune])
    tune = [index for index, label in enumerate(labels) if label in tune_cves]
    test = [index for index, label in enumerate(labels) if label not in tune_cves]
    return tune, test


def _order_cves(events: Sequence[Event]) -> tuple[list[str], list[str], dict[str, datetime]]:
    """Return each event's CVE, the CVEs oldest first and each CVE's earliest date."""
    earliest: dict[str, datetime] = {}
    labels: list[str] = []
    for event in events:
        label = gold_label(event)
        if label is None:
            raise ValueError(f"event {event.event_id} does not have exactly one CVE id")
        labels.append(label)
        if label not in earliest or event.published < earliest[label]:
            earliest[label] = event.published
    return labels, sorted(earliest, key=lambda cve: (earliest[cve], cve)), earliest


def pick_threshold(rows: Sequence[SweepRow]) -> SweepRow:
    """Return the row with the best pairwise F1; ties go to the lowest threshold."""
    if not rows:
        raise ValueError("no sweep rows to pick from")
    return min(rows, key=lambda row: (-row.pairwise.f1, row.threshold))


@dataclass(frozen=True, slots=True)
class RuleResult:
    """One rule's tune pick, its test score and the test split's own best.

    `tune_matched` scores the chosen threshold on the size-matched tune subset.
    """

    rule: str
    tune: SweepRow
    tune_matched: SweepRow
    test: SweepRow
    oracle: SweepRow


@dataclass(frozen=True, slots=True)
class HoldoutReport:
    """Results of a held-out threshold check."""

    train_fraction: float
    n_tune_events: int
    n_test_events: int
    n_tune_cves: int
    n_test_cves: int
    n_matched_events: int
    n_matched_cves: int
    tune_end: datetime
    test_start: datetime
    results: list[RuleResult]


def _split_sweep(
    events: Sequence[Event],
    vectors: np.ndarray,
    indices: Sequence[int],
    thresholds: Sequence[float],
    rule: str,
) -> list[SweepRow]:
    """Sweep thresholds over one split, with cross-source pairs inside that split."""
    subset = [events[index] for index in indices]
    ids = [event.event_id for event in subset]
    gold = {event.event_id: gold_label(event) for event in subset}
    return sweep(
        ids,
        vectors[list(indices)],
        gold,
        thresholds,
        groups=[event.source for event in subset],
        rule=rule,
    )


def evaluate_holdout(
    events: Sequence[Event],
    vectors: np.ndarray,
    thresholds: Sequence[float],
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    rules: Iterable[str] = RULES,
) -> HoldoutReport:
    """Pick each rule's threshold on the tune split and score it on the test split.

    `events` must all carry exactly one CVE id, with `vectors` row i for event i.
    """
    matrix = np.asarray(vectors)
    if matrix.ndim != 2 or len(matrix) != len(events):
        raise ValueError(f"vectors must be 2-D with {len(events)} rows, got shape {matrix.shape}")
    tune, test = split_by_cve(events, train_fraction)
    labels, ordered, earliest = _order_cves(events)
    n_tune_cves = len({labels[index] for index in tune})
    n_test_cves = len(ordered) - n_tune_cves
    matched_cves = set(ordered[max(0, n_tune_cves - n_test_cves) : n_tune_cves])
    matched = [index for index in tune if labels[index] in matched_cves]
    results: list[RuleResult] = []
    for rule in rules:
        tune_rows = _split_sweep(events, matrix, tune, thresholds, rule)
        test_rows = _split_sweep(events, matrix, test, thresholds, rule)
        picked = pick_threshold(tune_rows)
        test_row = next(row for row in test_rows if row.threshold == picked.threshold)
        (matched_row,) = _split_sweep(events, matrix, matched, [picked.threshold], rule)
        results.append(
            RuleResult(
                rule=rule,
                tune=picked,
                tune_matched=matched_row,
                test=test_row,
                oracle=pick_threshold(test_rows),
            )
        )
    return HoldoutReport(
        train_fraction=train_fraction,
        n_tune_events=len(tune),
        n_test_events=len(test),
        n_tune_cves=n_tune_cves,
        n_test_cves=n_test_cves,
        n_matched_events=len(matched),
        n_matched_cves=len(matched_cves),
        tune_end=earliest[ordered[n_tune_cves - 1]],
        test_start=earliest[ordered[n_tune_cves]],
        results=results,
    )


def _result_row(rule: str, split: str, row: SweepRow, n_events: int) -> str:
    """Format one table row; pairwise precision is n/a when no pair is predicted."""
    precision = "n/a" if row.n_clusters == n_events else f"{row.pairwise.precision:.3f}"
    return (
        f"| {rule} | {split} | {row.threshold:.2f} | {row.n_clusters} | {precision} "
        f"| {row.pairwise.recall:.3f} | {row.pairwise.f1:.3f} | {row.bcubed.f1:.3f} |"
    )


def format_holdout(
    report: HoldoutReport, *, embedder: str, n_excluded: int = 0, text_mode: str = "full"
) -> str:
    """Format a held-out report as markdown lines."""
    lines = [
        f"Embedder: {embedder}",
        f"Events: {report.n_tune_events + report.n_test_events} (excluded: {n_excluded})",
        "Pairs: cross-source",
    ]
    if text_mode != "full":
        mode, max_chars = parse_text_mode_label(text_mode)
        lines.append(f"Text: {mode} (max {max_chars} chars)")
    lines += [
        f"Train fraction: {report.train_fraction}",
        f"Tune: {report.n_tune_events} events, {report.n_tune_cves} CVEs "
        f"(last CVE published {report.tune_end.isoformat()})",
        f"Test: {report.n_test_events} events, {report.n_test_cves} CVEs "
        f"(first CVE published {report.test_start.isoformat()})",
        f"Size-matched tune: {report.n_matched_events} events, {report.n_matched_cves} CVEs "
        "(latest tune CVEs)",
        "CVE dates are each CVE's earliest advisory.",
    ]
    if report.tune_end == report.test_start:
        lines.append("Note: tune and test CVEs share the boundary date.")
    lines += [
        "Note: each split is clustered and scored against its own events only; a smaller "
        "pool has fewer wrong candidates, so compare test with the size-matched tune row.",
        "",
        "| rule | split | threshold | clusters | pairwise P | pairwise R | pairwise F1 | B3 F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report.results:
        lines.append(_result_row(result.rule, "tune", result.tune, report.n_tune_events))
        lines.append(
            _result_row(
                result.rule,
                "tune (size-matched)",
                result.tune_matched,
                report.n_matched_events,
            )
        )
        lines.append(_result_row(result.rule, "test", result.test, report.n_test_events))
        lines.append(_result_row(result.rule, "test oracle", result.oracle, report.n_test_events))
    return "\n".join(lines)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse a comma-separated threshold sequence."""
    try:
        thresholds = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated floats") from exc
    if not thresholds or any(not math.isfinite(threshold) for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated finite floats")
    return thresholds


def main(argv: Sequence[str] | None = None) -> int:
    """Run the held-out threshold check on cached vectors."""
    parser = argparse.ArgumentParser(
        description="Check whether the tuned threshold holds on a later time split"
    )
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Vectors saved by semtriage.baseline --save-embeddings",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated similarity thresholds (default 0.60 to 0.90 step 0.01)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Share of CVEs, oldest first, in the tune split (default 0.7)",
    )
    args = parser.parse_args(argv)
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")
    try:
        kept, _, n_excluded = labeled_events(load_events(args.events))
        label = read_text_mode(args.embeddings)
        text_mode, max_chars = parse_text_mode_label(label)
        ids = [event.event_id for event in kept]
        texts = [build_text(event, text_mode, max_chars=max_chars) for event in kept]
        vectors, embedder = read_embeddings(args.embeddings, ids, texts, text_mode=label)
        report = evaluate_holdout(
            kept, vectors, args.thresholds, train_fraction=args.train_fraction
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(format_holdout(report, embedder=embedder, n_excluded=n_excluded, text_mode=label))
    return 0


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_TRAIN_FRACTION",
    "HoldoutReport",
    "RuleResult",
    "evaluate_holdout",
    "format_holdout",
    "main",
    "pick_threshold",
    "split_by_cve",
]


if __name__ == "__main__":
    raise SystemExit(main())
