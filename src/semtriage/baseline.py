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


def _collect_edges(
    matrix: np.ndarray,
    minimum_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect upper-triangular candidate edges in NumPy row blocks."""
    row_count = len(matrix)
    if row_count < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    similarity_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for start in range(0, row_count, 1024):
        stop = min(start + 1024, row_count)
        block = matrix[start:stop] @ matrix.T
        local_rows, cols = np.nonzero(block >= minimum_threshold)
        rows = local_rows + start
        keep = rows < cols
        if np.any(keep):
            rows = rows[keep]
            cols = cols[keep]
            similarity_parts.append(block[local_rows[keep], cols])
            left_parts.append(rows)
            right_parts.append(cols)

    if not similarity_parts:
        similarities = np.empty(0, dtype=np.float64)
        lefts = np.empty(0, dtype=np.int64)
        rights = np.empty(0, dtype=np.int64)
    else:
        similarities = np.concatenate(similarity_parts)
        lefts = np.concatenate(left_parts)
        rights = np.concatenate(right_parts)

    order = np.argsort(-similarities, kind="stable")
    return similarities[order], lefts[order], rights[order]


def _check_thresholds(thresholds: Sequence[float]) -> None:
    """Reject an empty or non-finite threshold sequence."""
    if len(thresholds) == 0:
        raise ValueError("thresholds must not be empty")
    if any(not math.isfinite(threshold) for threshold in thresholds):
        raise ValueError("thresholds must be finite")


def _labels_by_threshold(
    vectors: np.ndarray, thresholds: Sequence[float]
) -> dict[float, list[int]]:
    """Label rows for each distinct threshold, walking thresholds from high to low."""
    _check_thresholds(thresholds)
    matrix = _finite_matrix(vectors)
    distinct = sorted(set(thresholds), reverse=True)
    similarities, lefts, rights = _collect_edges(matrix, distinct[-1])
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


def threshold_clusters(vectors: np.ndarray, threshold: float) -> list[int]:
    """Cluster rows with single-link dot-product similarity."""
    return _labels_by_threshold(vectors, [threshold])[threshold]


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One threshold's clustering results."""

    threshold: float
    n_clusters: int
    pairwise: Scores
    bcubed: Scores


def sweep(
    ids: Sequence[str],
    vectors: np.ndarray,
    gold: Mapping[str, Hashable],
    thresholds: Sequence[float],
) -> list[SweepRow]:
    """Sweep thresholds efficiently using a shared similarity matrix."""
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must have the same length")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    if set(ids) != set(gold):
        raise ValueError("ids and gold must contain the same item IDs")
    labels_by_threshold = _labels_by_threshold(vectors, thresholds)

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


def run_baseline(
    events: Iterable[Event],
    embedder: Embedder,
    thresholds: Sequence[float],
) -> BaselineReport:
    """Embed and evaluate the labeled subset of advisory events."""
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
    )
    return BaselineReport(
        embedder=embedder.name,
        n_events=len(kept),
        n_excluded=n_excluded,
        rows=rows,
    )


def format_report(report: BaselineReport) -> str:
    """Format a baseline report as markdown lines."""
    lines = [
        f"Embedder: {report.embedder}",
        f"Events: {report.n_events} (excluded: {report.n_excluded})",
        "",
        "| threshold | clusters | pairwise P | pairwise R | pairwise F1 | B3 P | B3 R | B3 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.threshold:.2f} | {row.n_clusters} "
            f"| {row.pairwise.precision:.3f} | {row.pairwise.recall:.3f} "
            f"| {row.pairwise.f1:.3f} | {row.bcubed.precision:.3f} "
            f"| {row.bcubed.recall:.3f} | {row.bcubed.f1:.3f} |"
        )
    return "\n".join(lines)


def load_events(path: str | Path) -> list[Event]:
    """Load non-blank JSONL lines as advisory events."""
    events: list[Event] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if line.strip():
                try:
                    events.append(Event.model_validate_json(line))
                except ValidationError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid event") from exc
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
    args = parser.parse_args(argv)
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")

    embedder: Embedder
    if args.embedder == "hashing":
        embedder = HashingEmbedder()
    else:
        embedder = BgeM3Embedder()

    try:
        report = run_baseline(load_events(args.events), embedder, args.thresholds)
    except ValueError as exc:
        parser.error(str(exc))
    print(format_report(report))
    return 0


__all__ = [
    "DEFAULT_THRESHOLDS",
    "BaselineReport",
    "SweepRow",
    "format_report",
    "gold_label",
    "load_events",
    "main",
    "run_baseline",
    "sweep",
    "threshold_clusters",
]


if __name__ == "__main__":
    raise SystemExit(main())
