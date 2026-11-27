"""Offline advisory deduplication baseline and reporting."""

import argparse
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from semtriage.embed import BgeM3Embedder, Embedder, HashingEmbedder
from semtriage.metrics import Scores, bcubed_scores, pairwise_scores
from semtriage.models import Event
from semtriage.sanitize import sanitize

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
)

PAIR_MODES: tuple[str, ...] = ("cross-source", "all")


def gold_label(event: Event) -> str | None:
    """Return the event's sole CVE label, if it has exactly one."""
    if len(event.cve_ids) == 1:
        return event.cve_ids[0]
    return None


class _UnionFind:
    """Union-find with deterministic minimum-index roots."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_count = size

    def find(self, item: int) -> int:
        """Return the root for an item."""
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        """Join two components."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent_root, child_root = left_root, right_root
        else:
            parent_root, child_root = right_root, left_root
        self.parent[child_root] = parent_root
        self.component_count -= 1

    def labels(self) -> list[int]:
        """Label every item with the smallest index in its component (always its root)."""
        return [self.find(item) for item in range(len(self.parent))]


def _finite_matrix(vectors: np.ndarray) -> np.ndarray:
    """Convert vectors to a finite float64 matrix."""
    matrix = np.asarray(vectors)
    try:
        matrix = matrix.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must contain finite numbers") from exc
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors must contain finite numbers")
    return matrix


_BLOCK_ROWS = 1024

_Edges = tuple[np.ndarray, np.ndarray, np.ndarray]


def _sorted_edges(
    similarity_parts: list[np.ndarray],
    left_parts: list[np.ndarray],
    right_parts: list[np.ndarray],
) -> _Edges:
    """Join edge blocks and order them by descending similarity (stable)."""
    if not similarity_parts:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    similarities = np.concatenate(similarity_parts)
    lefts = np.concatenate(left_parts)
    rights = np.concatenate(right_parts)
    order = np.argsort(-similarities, kind="stable")
    return similarities[order], lefts[order], rights[order]


def _collect_edges(matrix: np.ndarray, minimum_threshold: float) -> _Edges:
    """Collect upper-triangular candidate edges in NumPy row blocks."""
    similarity_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for start in range(0, len(matrix), _BLOCK_ROWS):
        block = matrix[start : start + _BLOCK_ROWS] @ matrix[start:].T
        local_rows, local_cols = np.nonzero(block >= minimum_threshold)
        keep = local_rows < local_cols
        if np.any(keep):
            similarity_parts.append(block[local_rows[keep], local_cols[keep]])
            left_parts.append((local_rows[keep] + start).astype(np.int32))
            right_parts.append((local_cols[keep] + start).astype(np.int32))
    return _sorted_edges(similarity_parts, left_parts, right_parts)


def _collect_cross_edges(
    matrix: np.ndarray,
    groups: Sequence[Hashable],
    minimum_threshold: float,
) -> _Edges:
    """Collect edges only between rows of different groups, in NumPy row blocks.

    Only cross-group dot products are computed. The larger group of each pair is
    walked in row blocks against all rows of the smaller one, so a block holds at
    most 1024 rows times the smaller group's size. That bound is for the block
    only: the returned edge list grows with the number of edges at or above
    `minimum_threshold`. Indices are int32 and similarities stay float64, so no
    edge moves across a threshold.
    """
    rows_by_group: dict[Hashable, list[int]] = {}
    for row, group in enumerate(groups):
        rows_by_group.setdefault(group, []).append(row)
    group_rows = [np.asarray(rows, dtype=np.int32) for rows in rows_by_group.values()]

    similarity_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for index, first in enumerate(group_rows):
        for second in group_rows[index + 1 :]:
            outer, inner = (first, second) if len(first) >= len(second) else (second, first)
            inner_matrix = matrix[inner]
            for start in range(0, len(outer), _BLOCK_ROWS):
                outer_rows = outer[start : start + _BLOCK_ROWS]
                block = matrix[outer_rows] @ inner_matrix.T
                local_rows, local_cols = np.nonzero(block >= minimum_threshold)
                if len(local_rows):
                    rows = outer_rows[local_rows]
                    cols = inner[local_cols]
                    similarity_parts.append(block[local_rows, local_cols])
                    left_parts.append(np.minimum(rows, cols))
                    right_parts.append(np.maximum(rows, cols))
    return _sorted_edges(similarity_parts, left_parts, right_parts)


def _check_thresholds(thresholds: Sequence[float]) -> None:
    """Reject an empty or non-finite threshold sequence."""
    if len(thresholds) == 0:
        raise ValueError("thresholds must not be empty")
    if any(not math.isfinite(threshold) for threshold in thresholds):
        raise ValueError("thresholds must be finite")


def _labels_by_threshold(
    vectors: np.ndarray,
    thresholds: Sequence[float],
    *,
    groups: Sequence[Hashable] | None = None,
) -> dict[float, list[int]]:
    """Label rows for each distinct threshold, walking thresholds from high to low."""
    _check_thresholds(thresholds)
    matrix = _finite_matrix(vectors)
    if groups is not None and len(groups) != len(matrix):
        raise ValueError("groups must have the same length as vectors")
    distinct = sorted(set(thresholds), reverse=True)
    if groups is None:
        similarities, lefts, rights = _collect_edges(matrix, distinct[-1])
    else:
        similarities, lefts, rights = _collect_cross_edges(matrix, groups, distinct[-1])
    union_find = _UnionFind(len(matrix))
    edge_index = 0
    labels: dict[float, list[int]] = {}
    for threshold in distinct:
        while (
            union_find.component_count > 1
            and edge_index < len(similarities)
            and similarities[edge_index] >= threshold
        ):
            union_find.union(int(lefts[edge_index]), int(rights[edge_index]))
            edge_index += 1
        labels[threshold] = union_find.labels()
    return labels


def threshold_clusters(
    vectors: np.ndarray,
    threshold: float,
    *,
    groups: Sequence[Hashable] | None = None,
) -> list[int]:
    """Cluster rows with single-link dot-product similarity."""
    return _labels_by_threshold(vectors, [threshold], groups=groups)[threshold]


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One threshold's clustering results."""

    threshold: float
    n_clusters: int
    pairwise: Scores
    bcubed: Scores


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    """One reference clustering result."""

    name: str
    n_clusters: int
    pairwise: Scores
    bcubed: Scores


def singletons_reference(
    ids: Sequence[str],
    gold: Mapping[str, Hashable],
) -> ReferenceRow:
    """Score a reference prediction where every item is a singleton."""
    prediction = {item_id: index for index, item_id in enumerate(ids)}
    return ReferenceRow(
        name="singletons",
        n_clusters=len(ids),
        pairwise=pairwise_scores(prediction, gold),
        bcubed=bcubed_scores(prediction, gold),
    )


def sweep(
    ids: Sequence[str],
    vectors: np.ndarray,
    gold: Mapping[str, Hashable],
    thresholds: Sequence[float],
    *,
    groups: Sequence[Hashable] | None = None,
) -> list[SweepRow]:
    """Sweep thresholds efficiently using a shared similarity matrix."""
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must have the same length")
    if groups is not None and len(groups) != len(vectors):
        raise ValueError("groups must have the same length as vectors")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    if set(ids) != set(gold):
        raise ValueError("ids and gold must contain the same item IDs")
    labels_by_threshold = _labels_by_threshold(vectors, thresholds, groups=groups)

    rows: list[SweepRow] = []
    for threshold in thresholds:
        labels = labels_by_threshold[threshold]
        pred = dict(zip(ids, labels, strict=True))
        pairwise = pairwise_scores(pred, gold)
        bcubed = bcubed_scores(pred, gold)
        rows.append(
            SweepRow(
                threshold=threshold,
                n_clusters=len(set(labels)),
                pairwise=pairwise,
                bcubed=bcubed,
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class BaselineReport:
    """Results from one baseline evaluation run."""

    embedder: str
    n_events: int
    n_excluded: int
    rows: list[SweepRow]
    pairs: str | None = None
    reference: ReferenceRow | None = None


def run_baseline(
    events: Iterable[Event],
    embedder: Embedder,
    thresholds: Sequence[float],
    *,
    pairs: str = "cross-source",
) -> BaselineReport:
    """Embed and evaluate the labeled subset of advisory events."""
    if pairs not in PAIR_MODES:
        raise ValueError(f"pairs must be one of {PAIR_MODES}, got {pairs!r}")
    materialized = list(events)
    kept_with_labels = [
        (event, label) for event in materialized if (label := gold_label(event)) is not None
    ]
    kept = [event for event, _ in kept_with_labels]
    n_excluded = len(materialized) - len(kept)
    if not kept:
        raise ValueError("No labeled events to evaluate")

    vectors = embedder.embed([sanitize(event.text) for event in kept])
    gold = {event.event_id: label for event, label in kept_with_labels}
    rows = sweep(
        [event.event_id for event in kept],
        vectors,
        gold,
        thresholds,
        groups=[event.source for event in kept] if pairs == "cross-source" else None,
    )
    return BaselineReport(
        embedder=embedder.name,
        n_events=len(kept),
        n_excluded=n_excluded,
        rows=rows,
        pairs=pairs,
        reference=singletons_reference([event.event_id for event in kept], gold),
    )


def _table_row(label: str, n_clusters: int, pairwise: Scores, bcubed: Scores, n_events: int) -> str:
    """Format one table row; pairwise precision is n/a when no pair is predicted."""
    precision = "n/a" if n_clusters == n_events else f"{pairwise.precision:.3f}"
    return (
        f"| {label} | {n_clusters} | {precision} | {pairwise.recall:.3f} "
        f"| {pairwise.f1:.3f} | {bcubed.precision:.3f} | {bcubed.recall:.3f} "
        f"| {bcubed.f1:.3f} |"
    )


def format_report(report: BaselineReport) -> str:
    """Format a baseline report as markdown lines.

    Every cluster a singleton means no predicted pair, so pairwise precision is
    undefined and shown as n/a (the metrics module stores 1.0 by convention).
    """
    lines = [
        f"Embedder: {report.embedder}",
        f"Events: {report.n_events} (excluded: {report.n_excluded})",
    ]
    if report.pairs is not None:
        lines.append(f"Pairs: {report.pairs}")
    lines += [
        "",
        "| threshold | clusters | pairwise P | pairwise R | pairwise F1 | B3 P | B3 R | B3 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    reference = report.reference
    if reference is not None:
        lines.append(
            _table_row(
                reference.name,
                reference.n_clusters,
                reference.pairwise,
                reference.bcubed,
                report.n_events,
            )
        )
    for row in report.rows:
        lines.append(
            _table_row(
                f"{row.threshold:.2f}",
                row.n_clusters,
                row.pairwise,
                row.bcubed,
                report.n_events,
            )
        )
    return "\n".join(lines)


def load_events(path: str | Path) -> list[Event]:
    """Load non-blank JSONL lines as advisory events."""
    events: list[Event] = []
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            for lineno, line in enumerate(source, start=1):
                if line.strip():
                    try:
                        events.append(Event.model_validate_json(line))
                    except ValidationError as exc:
                        raise ValueError(f"{path}:{lineno}: invalid event") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not valid UTF-8") from exc
    return events


def _parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse a comma-separated threshold sequence."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated floats")
    try:
        thresholds = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated floats") from exc
    if any(not math.isfinite(threshold) for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be finite")
    return thresholds


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline baseline command-line interface."""
    parser = argparse.ArgumentParser(description="Run the offline deduplication baseline")
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embedder",
        choices=("hashing", "bge-m3"),
        default="hashing",
        help="Embedding implementation",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated similarity thresholds",
    )
    parser.add_argument(
        "--pairs",
        choices=PAIR_MODES,
        default="cross-source",
        help="Candidate pair mode: cross-source joins only events from different sources",
    )
    args = parser.parse_args(argv)
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")

    embedder: Embedder
    if args.embedder == "hashing":
        embedder = HashingEmbedder()
    else:
        embedder = BgeM3Embedder()

    try:
        report = run_baseline(
            load_events(args.events),
            embedder,
            args.thresholds,
            pairs=args.pairs,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(format_report(report))
    return 0


__all__ = [
    "DEFAULT_THRESHOLDS",
    "PAIR_MODES",
    "BaselineReport",
    "ReferenceRow",
    "SweepRow",
    "format_report",
    "gold_label",
    "load_events",
    "main",
    "run_baseline",
    "singletons_reference",
    "sweep",
    "threshold_clusters",
]


if __name__ == "__main__":
    raise SystemExit(main())
