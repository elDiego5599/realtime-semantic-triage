"""Error analysis of the threshold baseline at one threshold over cached vectors."""

import argparse
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semtriage.baseline import (
    PAIR_MODES,
    candidate_edges,
    edge_clusters,
    labeled_events,
    load_events,
    read_embeddings,
)
from semtriage.models import Event
from semtriage.sanitize import sanitize

QUANTILES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
SHARED_FIELDS: tuple[str, ...] = ("package", "ecosystem", "cwe", "cna", "any")
SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-10", 4, 10),
    (">10", 11, 2**62),
)
SAMPLE_CHARS = 300
MAX_PAIRS = 50_000_000
_DOT_CHUNK = 8192

_Pairs = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class PairSample:
    """One example pair with its similarity and the start of both texts."""

    left_id: str
    right_id: str
    similarity: float
    left_text: str
    right_text: str


@dataclass(frozen=True, slots=True)
class ErrorReport:
    """False merges and missed duplicates of the baseline at one threshold.

    `*_false_merge*` and `n_true_pairs` count pairs inside predicted clusters, so
    they include pairs joined only by single-link chaining. `*_false_edge*` count
    candidate edges at or above the threshold whose events have different gold
    labels: the direct model errors.
    """

    threshold: float
    pairs: str
    n_events: int
    n_excluded: int
    n_true_pairs: int
    n_false_merges: int
    n_false_edges: int
    n_missed: int
    missed_same_source: int
    false_merge_shares: dict[str, float]
    false_edge_shares: dict[str, float]
    false_merge_quantiles: dict[float, float]
    false_edge_quantiles: dict[float, float]
    true_pair_quantiles: dict[float, float]
    missed_quantiles: dict[float, float]
    missed_length_quantiles: dict[str, dict[float, float]]
    cluster_sizes: dict[str, int]
    largest_cluster: int
    false_merge_samples: list[PairSample]
    false_edge_samples: list[PairSample]
    missed_samples: list[PairSample]
    embedder: str | None = None


def _pairs_within(labels: np.ndarray) -> _Pairs:
    """Return every (left, right) row pair, left < right, that shares a label."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    sizes = np.diff(np.r_[starts, len(labels)])
    total = int(np.sum(sizes * (sizes - 1) // 2))
    if total > MAX_PAIRS:
        raise ValueError(f"{total} pairs to analyze exceeds {MAX_PAIRS}; use a higher threshold")
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    triangles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for start, size in zip(starts[sizes > 1], sizes[sizes > 1], strict=True):
        rows = order[start : start + size]
        if size not in triangles:
            triangles[size] = np.triu_indices(size, 1)
        upper, lower = triangles[size]
        lefts.append(rows[upper])
        rights.append(rows[lower])
    if not lefts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(lefts), np.concatenate(rights)


def _row_dots(matrix: np.ndarray, pairs: _Pairs) -> np.ndarray:
    """Dot product of each row pair, in chunks to bound memory."""
    left, right = pairs
    out = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), _DOT_CHUNK):
        stop = start + _DOT_CHUNK
        out[start:stop] = np.einsum("ij,ij->i", matrix[left[start:stop]], matrix[right[start:stop]])
    return out


def _quantiles(values: np.ndarray) -> dict[float, float]:
    """Return the fixed quantiles of values, or an empty map for no values."""
    if len(values) == 0:
        return {}
    return dict(zip(QUANTILES, np.quantile(values, QUANTILES).tolist(), strict=True))


def _shares(events: Sequence[Event], pairs: _Pairs) -> dict[str, float]:
    """Share of pairs whose events have a package, ecosystem, CWE or CNA in common."""
    counts = dict.fromkeys(SHARED_FIELDS, 0)
    for left, right in zip(pairs[0].tolist(), pairs[1].tolist(), strict=True):
        a, b = events[left], events[right]
        hits = {
            "package": not set(a.packages).isdisjoint(b.packages),
            "ecosystem": not set(a.ecosystems).isdisjoint(b.ecosystems),
            "cwe": not set(a.cwe_ids).isdisjoint(b.cwe_ids),
            "cna": a.cna is not None and a.cna == b.cna,
        }
        hits["any"] = any(hits.values())
        for field, hit in hits.items():
            counts[field] += hit
    total = len(pairs[0])
    return {field: (count / total if total else 0.0) for field, count in counts.items()}


def _samples(
    events: Sequence[Event],
    texts: Sequence[str],
    pairs: _Pairs,
    similarities: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[PairSample]:
    """Draw up to count random pairs, listed by descending similarity."""
    if count == 0 or len(similarities) == 0:
        return []
    chosen = rng.choice(len(similarities), size=min(count, len(similarities)), replace=False)
    chosen = chosen[np.argsort(-similarities[chosen], kind="stable")]
    return [
        PairSample(
            left_id=events[pairs[0][i]].event_id,
            right_id=events[pairs[1][i]].event_id,
            similarity=float(similarities[i]),
            left_text=texts[pairs[0][i]][:SAMPLE_CHARS],
            right_text=texts[pairs[1][i]][:SAMPLE_CHARS],
        )
        for i in chosen.tolist()
    ]


def _select(pairs: _Pairs, mask: np.ndarray) -> _Pairs:
    return pairs[0][mask], pairs[1][mask]


def _cluster_sizes(labels: np.ndarray) -> tuple[dict[str, int], int]:
    """Count clusters per size bucket and return the largest cluster size."""
    sizes = np.bincount(labels) if len(labels) else np.zeros(0, dtype=np.int64)
    sizes = sizes[sizes > 0]
    buckets = {
        name: int(np.sum((sizes >= low) & (sizes <= high))) for name, low, high in SIZE_BUCKETS
    }
    return buckets, int(sizes.max()) if len(sizes) else 0


def analyze_errors(
    events: Iterable[Event],
    vectors: np.ndarray,
    threshold: float,
    *,
    pairs: str = "cross-source",
    samples: int = 5,
    seed: int = 0,
    embedder: str | None = None,
) -> ErrorReport:
    """Cluster the labeled events at threshold and describe where it disagrees with gold.

    `vectors` holds one row per labeled event, in the order `labeled_events` keeps them.
    Predicted pairs are all pairs in one predicted cluster, as in the pairwise metric;
    direct false edges are the candidate edges themselves that join different labels.
    """
    if pairs not in PAIR_MODES:
        raise ValueError(f"pairs must be one of {PAIR_MODES}, got {pairs!r}")
    if samples < 0:
        raise ValueError("samples must not be negative")
    kept, gold, n_excluded = labeled_events(events)
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(kept):
        raise ValueError(f"vectors must be 2-D with {len(kept)} rows, got shape {matrix.shape}")
    groups = [event.source for event in kept] if pairs == "cross-source" else None
    edge_sims, edge_lefts, edge_rights = candidate_edges(matrix, threshold, groups=groups)
    predicted = np.asarray(edge_clusters(len(kept), edge_lefts, edge_rights), dtype=np.int64)
    _, gold_codes = np.unique([gold[event.event_id] for event in kept], return_inverse=True)
    cluster_sizes, largest_cluster = _cluster_sizes(predicted)

    false_edge_mask = gold_codes[edge_lefts] != gold_codes[edge_rights]
    false_edges = (edge_lefts[false_edge_mask], edge_rights[false_edge_mask])
    false_edge_sims = edge_sims[false_edge_mask]

    predicted_pairs = _pairs_within(predicted)
    same_gold = gold_codes[predicted_pairs[0]] == gold_codes[predicted_pairs[1]]
    false_pairs = _select(predicted_pairs, ~same_gold)
    true_pairs = _select(predicted_pairs, same_gold)
    gold_pairs = _pairs_within(gold_codes)
    missed_pairs = _select(gold_pairs, predicted[gold_pairs[0]] != predicted[gold_pairs[1]])

    false_sims = _row_dots(matrix, false_pairs)
    true_sims = _row_dots(matrix, true_pairs)
    missed_sims = _row_dots(matrix, missed_pairs)
    texts = [sanitize(event.text) for event in kept]
    lengths = np.array([len(text) for text in texts], dtype=np.float64)
    missed_left, missed_right = lengths[missed_pairs[0]], lengths[missed_pairs[1]]
    sources = np.array([event.source.value for event in kept])
    rng = np.random.default_rng(seed)
    return ErrorReport(
        threshold=threshold,
        pairs=pairs,
        n_events=len(kept),
        n_excluded=n_excluded,
        n_true_pairs=len(true_sims),
        n_false_merges=len(false_sims),
        n_false_edges=len(false_edge_sims),
        n_missed=len(missed_sims),
        missed_same_source=int(np.sum(sources[missed_pairs[0]] == sources[missed_pairs[1]])),
        false_merge_shares=_shares(kept, false_pairs),
        false_edge_shares=_shares(kept, false_edges),
        false_merge_quantiles=_quantiles(false_sims),
        false_edge_quantiles=_quantiles(false_edge_sims),
        true_pair_quantiles=_quantiles(true_sims),
        missed_quantiles=_quantiles(missed_sims),
        missed_length_quantiles={
            "shorter": _quantiles(np.minimum(missed_left, missed_right)),
            "longer": _quantiles(np.maximum(missed_left, missed_right)),
        },
        cluster_sizes=cluster_sizes,
        largest_cluster=largest_cluster,
        false_merge_samples=_samples(kept, texts, false_pairs, false_sims, samples, rng),
        false_edge_samples=_samples(kept, texts, false_edges, false_edge_sims, samples, rng),
        missed_samples=_samples(kept, texts, missed_pairs, missed_sims, samples, rng),
        embedder=embedder,
    )


def _quantile_table(rows: Sequence[tuple[str, dict[float, float]]], fmt: str) -> list[str]:
    header = "| series | " + " | ".join(f"q{q:g}" for q in QUANTILES) + " |"
    lines = [header, "|---|" + "---:|" * len(QUANTILES)]
    for name, values in rows:
        cells = [format(values[q], fmt) if values else "n/a" for q in QUANTILES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _sample_lines(title: str, items: Sequence[PairSample]) -> list[str]:
    lines = ["", f"## {title}"]
    if not items:
        lines.append("(none)")
    for item in items:
        lines += [
            "",
            f"- {item.left_id} / {item.right_id} similarity {item.similarity:.3f}",
            f"  left: {' '.join(item.left_text.split())}",
            f"  right: {' '.join(item.right_text.split())}",
        ]
    return lines


def format_error_report(report: ErrorReport) -> str:
    """Format an error report as markdown lines."""
    lines = []
    if report.embedder is not None:
        lines.append(f"Embedder: {report.embedder}")
    lines += [
        f"Events: {report.n_events} (excluded: {report.n_excluded})",
        f"Pairs: {report.pairs}",
        f"Threshold: {report.threshold:.2f}",
        "",
        f"Direct false edges: {report.n_false_edges}",
        f"True pairs (pairs inside predicted clusters): {report.n_true_pairs}",
        f"False merges (pairs inside predicted clusters): {report.n_false_merges}",
        f"Missed duplicates: {report.n_missed} (same source: {report.missed_same_source})",
        "",
        "## Predicted cluster sizes",
        "",
        "| size | clusters |",
        "|---|---:|",
    ]
    lines += [f"| {size} | {count} |" for size, count in report.cluster_sizes.items()]
    lines += [
        "",
        f"Largest cluster: {report.largest_cluster}",
        "",
        "## Share of false pairs sharing a field",
        "",
        "| field | direct false edges | false merges (pairs inside predicted clusters) |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {field} | {report.false_edge_shares[field]:.3f} | {share:.3f} |"
        for field, share in report.false_merge_shares.items()
    ]
    lines += ["", "## Similarity quantiles", ""]
    lines += _quantile_table(
        [
            ("direct false edges", report.false_edge_quantiles),
            ("false merges (pairs inside predicted clusters)", report.false_merge_quantiles),
            ("true pairs (pairs inside predicted clusters)", report.true_pair_quantiles),
            ("missed duplicates", report.missed_quantiles),
        ],
        ".3f",
    )
    lines += ["", "## Missed duplicates text length quantiles (characters)", ""]
    lines += _quantile_table(
        [
            ("shorter text", report.missed_length_quantiles["shorter"]),
            ("longer text", report.missed_length_quantiles["longer"]),
        ],
        ".0f",
    )
    lines += _sample_lines("Sample direct false edges", report.false_edge_samples)
    lines += _sample_lines(
        "Sample false merges (pairs inside predicted clusters)", report.false_merge_samples
    )
    lines += _sample_lines("Sample missed duplicates", report.missed_samples)
    return "\n".join(lines)


def _parse_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a float") from exc
    if not math.isfinite(threshold):
        raise argparse.ArgumentTypeError("threshold must be finite")
    return threshold


def _parse_samples(value: str) -> int:
    try:
        samples = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("samples must be an integer") from exc
    if samples < 0:
        raise argparse.ArgumentTypeError("samples must not be negative")
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    """Run the error analysis command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze baseline false merges and missed duplicates at one threshold"
    )
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        metavar="PATH",
        help="Vectors saved by semtriage.baseline --save-embeddings",
    )
    parser.add_argument(
        "--threshold", required=True, type=_parse_threshold, help="Similarity threshold"
    )
    parser.add_argument(
        "--pairs",
        choices=PAIR_MODES,
        default="cross-source",
        help="Candidate pair mode: cross-source joins only events from different sources",
    )
    parser.add_argument(
        "--samples",
        type=_parse_samples,
        default=5,
        help="Sample pairs to show of each kind (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sample selection")
    args = parser.parse_args(argv)
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")

    try:
        events = load_events(args.events)
        kept, _, _ = labeled_events(events)
        vectors, embedder = read_embeddings(
            args.embeddings,
            [event.event_id for event in kept],
            [sanitize(event.text) for event in kept],
        )
        report = analyze_errors(
            events,
            vectors,
            args.threshold,
            pairs=args.pairs,
            samples=args.samples,
            seed=args.seed,
            embedder=embedder,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(format_error_report(report))
    return 0


__all__ = [
    "ErrorReport",
    "PairSample",
    "analyze_errors",
    "format_error_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
