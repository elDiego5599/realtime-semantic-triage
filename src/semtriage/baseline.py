"""Offline advisory deduplication baseline and reporting."""

import argparse
import hashlib
import math
import zipfile
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from semtriage.embed import BgeM3Embedder, Embedder, HashingEmbedder
from semtriage.metrics import Scores, bcubed_scores, pairwise_scores
from semtriage.models import Event
from semtriage.text_mode import DEFAULT_MAX_CHARS, TEXT_MODES, build_text, text_mode_label

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

RULES: tuple[str, ...] = ("single-link", "mutual-best")


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


def candidate_edges(
    vectors: np.ndarray,
    minimum_threshold: float,
    *,
    groups: Sequence[Hashable] | None = None,
) -> _Edges:
    """Return (similarities, lefts, rights) of candidate edges at or above a threshold.

    Edges are ordered by descending similarity, with left < right. With `groups`,
    only rows from different groups are paired.
    """
    matrix = _finite_matrix(vectors)
    if groups is not None and len(groups) != len(matrix):
        raise ValueError("groups must have the same length as vectors")
    if groups is None:
        return _collect_edges(matrix, minimum_threshold)
    return _collect_cross_edges(matrix, groups, minimum_threshold)


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
    distinct = sorted(set(thresholds), reverse=True)
    similarities, lefts, rights = candidate_edges(vectors, distinct[-1], groups=groups)
    union_find = _UnionFind(len(vectors))
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


def edge_clusters(n_rows: int, lefts: np.ndarray, rights: np.ndarray) -> list[int]:
    """Label rows by the connected components of the given edges (smallest index wins)."""
    union_find = _UnionFind(n_rows)
    for left, right in zip(lefts.tolist(), rights.tolist(), strict=True):
        union_find.union(left, right)
    return union_find.labels()


def _check_rule(rule: str) -> None:
    """Reject an unknown linking rule."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")


def mutual_best_pairs(
    vectors: np.ndarray,
    groups: Sequence[Hashable] | None,
    *,
    block_rows: int = _BLOCK_ROWS,
) -> _Edges:
    """Return (similarities, lefts, rights) of mutual best matches between two groups.

    Each row's best match is its most similar row (dot product) in the other group.
    A pair is returned when each row is the other's best match, at any similarity, so
    a threshold can be applied afterwards. Ties go to the lowest row index: argmax
    picks the first maximum and the running column maximum only moves on a strictly
    greater value. The larger group is walked in blocks of `block_rows` rows against
    the whole smaller group, so the full matrix is never built. Pairs come with
    left < right (int32), ordered by left. With fewer than two groups there are no
    pairs; more than two groups raise ValueError.
    """
    if groups is None:
        raise ValueError("the mutual-best rule needs groups (cross-source pairs)")
    if block_rows < 1:
        raise ValueError("block_rows must be at least 1")
    matrix = _finite_matrix(vectors)
    if len(groups) != len(matrix):
        raise ValueError("groups must have the same length as vectors")
    rows_by_group: dict[Hashable, list[int]] = {}
    for row, group in enumerate(groups):
        rows_by_group.setdefault(group, []).append(row)
    if len(rows_by_group) > 2:
        raise ValueError("the mutual-best rule needs at most two groups")
    if len(rows_by_group) < 2:
        return _sorted_edges([], [], [])
    first, second = (np.asarray(rows, dtype=np.int32) for rows in rows_by_group.values())
    outer, inner = (first, second) if len(first) >= len(second) else (second, first)

    inner_matrix = matrix[inner]
    outer_best = np.empty(len(outer), dtype=np.int64)
    outer_similarity = np.empty(len(outer), dtype=np.float64)
    inner_best = np.zeros(len(inner), dtype=np.int64)
    inner_similarity = np.full(len(inner), -np.inf)
    columns = np.arange(len(inner))
    for start in range(0, len(outer), block_rows):
        block = matrix[outer[start : start + block_rows]] @ inner_matrix.T
        stop = start + len(block)
        best = block.argmax(axis=1)
        outer_best[start:stop] = best
        outer_similarity[start:stop] = block[np.arange(len(block)), best]
        column_best = block.argmax(axis=0)
        column_similarity = block[column_best, columns]
        better = column_similarity > inner_similarity
        inner_similarity[better] = column_similarity[better]
        inner_best[better] = column_best[better] + start

    mutual = inner_best[outer_best] == np.arange(len(outer))
    rows = outer[mutual]
    cols = inner[outer_best[mutual]]
    lefts = np.minimum(rows, cols)
    order = np.argsort(lefts, kind="stable")
    return (
        outer_similarity[mutual][order],
        lefts[order],
        np.maximum(rows, cols)[order],
    )


def _mutual_labels_by_threshold(
    vectors: np.ndarray,
    thresholds: Sequence[float],
    groups: Sequence[Hashable] | None,
    *,
    block_rows: int = _BLOCK_ROWS,
) -> dict[float, list[int]]:
    """Label rows for each distinct threshold from one pass of mutual best matches."""
    _check_thresholds(thresholds)
    similarities, lefts, rights = mutual_best_pairs(vectors, groups, block_rows=block_rows)
    labels: dict[float, list[int]] = {}
    for threshold in set(thresholds):
        keep = similarities >= threshold
        labels[threshold] = edge_clusters(len(vectors), lefts[keep], rights[keep])
    return labels


def mutual_best_clusters(
    vectors: np.ndarray,
    threshold: float,
    groups: Sequence[Hashable] | None,
    *,
    block_rows: int = _BLOCK_ROWS,
) -> list[int]:
    """Cluster rows by mutual best cross-group matches at or above a threshold.

    Clusters are the connected components of the kept pairs (at most two rows each),
    labeled by their smallest row index, as in `edge_clusters`.
    """
    return _mutual_labels_by_threshold(vectors, [threshold], groups, block_rows=block_rows)[
        threshold
    ]


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
    rule: str = "single-link",
) -> list[SweepRow]:
    """Sweep thresholds, computing similarities (or mutual best matches) only once."""
    _check_rule(rule)
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must have the same length")
    if groups is not None and len(groups) != len(vectors):
        raise ValueError("groups must have the same length as vectors")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    if set(ids) != set(gold):
        raise ValueError("ids and gold must contain the same item IDs")
    if rule == "mutual-best":
        labels_by_threshold = _mutual_labels_by_threshold(vectors, thresholds, groups)
    else:
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
    rule: str = "single-link"
    text_mode: str = "full"
    max_chars: int = DEFAULT_MAX_CHARS


def labeled_events(events: Iterable[Event]) -> tuple[list[Event], dict[str, str], int]:
    """Keep events with a single CVE label; return them, their gold map and the excluded count."""
    materialized = list(events)
    kept: list[Event] = []
    gold: dict[str, str] = {}
    for event in materialized:
        label = gold_label(event)
        if label is not None:
            kept.append(event)
            gold[event.event_id] = label
    return kept, gold, len(materialized) - len(kept)


def texts_sha256(texts: Sequence[str]) -> str:
    """Hex sha256 over the texts in order, each prefixed by its UTF-8 byte length."""
    digest = hashlib.sha256(len(texts).to_bytes(8, "big"))
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_embeddings(
    path: str | Path,
    ids: Sequence[str],
    texts: Sequence[str],
    vectors: np.ndarray,
    *,
    embedder: str,
    text_mode: str = "full",
) -> None:
    """Save event ids, a hash of the embedded texts, float32 vectors and the embedder name.

    `texts` are the exact (sanitized, possibly shortened) texts that were embedded,
    in the order of `ids`; `text_mode` is the `text_mode_label` that built them.
    Everything goes to one .npz file at `path`.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(ids):
        raise ValueError(f"vectors must be 2-D with {len(ids)} rows, got shape {matrix.shape}")
    if len(texts) != len(ids):
        raise ValueError(f"expected {len(ids)} texts, got {len(texts)}")
    with Path(path).open("wb") as target:
        np.savez(
            target,
            event_ids=np.asarray(list(ids), dtype=str),
            vectors=matrix,
            embedder=np.asarray(embedder, dtype=str),
            text_sha256=np.asarray(texts_sha256(texts), dtype=str),
            text_mode=np.asarray(text_mode, dtype=str),
        )


def read_text_mode(path: str | Path) -> str:
    """Return the text mode label stored in an embeddings file; files without one are full."""
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            return str(data["text_mode"]) if "text_mode" in data.files else "full"
    except (OSError, ValueError, AttributeError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"{path}: cannot read embeddings file ({exc})") from exc


def read_embeddings(
    path: str | Path,
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    text_mode: str | None = None,
) -> tuple[np.ndarray, str]:
    """Load cached vectors, refusing them unless both ids and texts match what was saved.

    `ids` must equal the stored event ids in order, and `texts` (the embedded texts
    of those events) must hash to the stored digest, so stale vectors are never reused.
    When `text_mode` is given and the file records one (files written before text
    modes existed do not), a different label is refused with a clearer message;
    the hash check alone already refuses texts built with another mode.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError, AttributeError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"{path}: cannot read embeddings file ({exc})") from exc
    missing = {"event_ids", "vectors", "embedder", "text_sha256"} - set(arrays)
    if missing:
        raise ValueError(f"{path}: embeddings file is missing {sorted(missing)}")
    stored_ids, vectors = arrays["event_ids"], arrays["vectors"]
    embedder = str(arrays["embedder"])
    if stored_ids.ndim != 1 or vectors.ndim != 2 or len(vectors) != len(stored_ids):
        raise ValueError(
            f"{path}: bad shape, event_ids {stored_ids.shape} and vectors {vectors.shape}"
        )
    if stored_ids.tolist() != list(ids):
        raise ValueError(
            f"{path}: event ids do not match the labeled events "
            f"({len(stored_ids)} stored, {len(ids)} expected, same order required)"
        )
    stored_mode = str(arrays["text_mode"]) if "text_mode" in arrays else None
    if text_mode is not None and stored_mode is not None and stored_mode != text_mode:
        raise ValueError(
            f"{path}: text mode differs: cache built with {stored_mode}, "
            f"this run uses {text_mode}; re-embed or use the same mode"
        )
    if str(arrays["text_sha256"]) != texts_sha256(texts):
        raise ValueError(
            f"{path}: sanitized texts differ from the ones embedded "
            "(stale cache, for example after a sanitize change); re-embed"
        )
    return vectors.astype(np.float32, copy=False), embedder


def run_baseline(
    events: Iterable[Event],
    embedder: Embedder | None,
    thresholds: Sequence[float],
    *,
    pairs: str = "cross-source",
    rule: str = "single-link",
    save_embeddings: str | Path | None = None,
    load_embeddings: str | Path | None = None,
    text_mode: str = "full",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> BaselineReport:
    """Embed (or load cached vectors for) the labeled subset of events and evaluate it.

    `text_mode` and `max_chars` choose the embedded text (see `build_text`).
    """
    if pairs not in PAIR_MODES:
        raise ValueError(f"pairs must be one of {PAIR_MODES}, got {pairs!r}")
    _check_rule(rule)
    mode_label = text_mode_label(text_mode, max_chars)
    if rule == "mutual-best" and pairs != "cross-source":
        raise ValueError("the mutual-best rule needs cross-source pairs")
    if (embedder is None) == (load_embeddings is None):
        raise ValueError("pass exactly one of embedder and load_embeddings")
    kept, gold, n_excluded = labeled_events(events)
    if not kept:
        raise ValueError("No labeled events to evaluate")
    ids = [event.event_id for event in kept]
    texts = [build_text(event, text_mode, max_chars=max_chars) for event in kept]

    if embedder is None:
        vectors, embedder_name = read_embeddings(load_embeddings, ids, texts, text_mode=mode_label)
    else:
        vectors = embedder.embed(texts)
        embedder_name = embedder.name
    if save_embeddings is not None:
        write_embeddings(
            save_embeddings, ids, texts, vectors, embedder=embedder_name, text_mode=mode_label
        )
    rows = sweep(
        ids,
        vectors,
        gold,
        thresholds,
        groups=[event.source for event in kept] if pairs == "cross-source" else None,
        rule=rule,
    )
    return BaselineReport(
        embedder=embedder_name,
        n_events=len(kept),
        n_excluded=n_excluded,
        rows=rows,
        pairs=pairs,
        reference=singletons_reference(ids, gold),
        rule=rule,
        text_mode=text_mode,
        max_chars=max_chars,
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
    if report.rule != "single-link":
        lines.append(f"Rule: {report.rule}")
    if report.text_mode != "full":
        lines.append(f"Text: {report.text_mode} (max {report.max_chars} chars)")
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


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline baseline command-line interface."""
    parser = argparse.ArgumentParser(description="Run the offline deduplication baseline")
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embedder",
        choices=("hashing", "bge-m3"),
        default=None,
        help="Embedding implementation (default: hashing)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run bge-m3 in half precision (only with --embedder bge-m3)",
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
    parser.add_argument(
        "--rule",
        choices=RULES,
        default="single-link",
        help="Linking rule: mutual-best joins two events only if each is the other's "
        "most similar cross-source event (needs --pairs cross-source)",
    )
    parser.add_argument(
        "--text-mode",
        choices=TEXT_MODES,
        default="full",
        help="Text to embed: full sanitized text (default), truncate to --max-chars, "
        "or lead (GHSA title plus Summary section or first paragraph, no code blocks; "
        "NVD description), capped at --max-chars",
    )
    parser.add_argument(
        "--max-chars",
        type=_positive_int,
        default=None,
        metavar="N",
        help=f"Character cap for --text-mode truncate or lead (default {DEFAULT_MAX_CHARS})",
    )
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--save-embeddings",
        type=Path,
        metavar="PATH",
        help="Save event ids and float32 vectors of the labeled events to this .npz file",
    )
    cache.add_argument(
        "--load-embeddings",
        type=Path,
        metavar="PATH",
        help="Load vectors from a --save-embeddings file instead of embedding",
    )
    args = parser.parse_args(argv)
    if args.load_embeddings is not None and (args.embedder is not None or args.fp16):
        parser.error("--load-embeddings cannot be combined with --embedder or --fp16")
    if args.fp16 and args.embedder != "bge-m3":
        parser.error("--fp16 requires --embedder bge-m3")
    if args.rule == "mutual-best" and args.pairs != "cross-source":
        parser.error("--rule mutual-best requires --pairs cross-source")
    if args.max_chars is not None and args.text_mode == "full":
        parser.error("--max-chars requires --text-mode truncate or lead")
    max_chars = DEFAULT_MAX_CHARS if args.max_chars is None else args.max_chars
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")

    embedder: Embedder | None = None
    if args.load_embeddings is None:
        if args.embedder == "bge-m3":
            embedder = BgeM3Embedder(use_fp16=args.fp16)
        else:
            embedder = HashingEmbedder()

    try:
        report = run_baseline(
            load_events(args.events),
            embedder,
            args.thresholds,
            pairs=args.pairs,
            rule=args.rule,
            save_embeddings=args.save_embeddings,
            load_embeddings=args.load_embeddings,
            text_mode=args.text_mode,
            max_chars=max_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(format_report(report))
    return 0


__all__ = [
    "DEFAULT_THRESHOLDS",
    "PAIR_MODES",
    "RULES",
    "BaselineReport",
    "ReferenceRow",
    "SweepRow",
    "candidate_edges",
    "edge_clusters",
    "format_report",
    "gold_label",
    "labeled_events",
    "load_events",
    "main",
    "mutual_best_clusters",
    "mutual_best_pairs",
    "read_embeddings",
    "read_text_mode",
    "run_baseline",
    "singletons_reference",
    "sweep",
    "texts_sha256",
    "threshold_clusters",
    "write_embeddings",
]


if __name__ == "__main__":
    raise SystemExit(main())
