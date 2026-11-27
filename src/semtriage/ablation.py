"""Run retrieval ablations over cached dense vectors."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np

from semtriage.baseline import (
    labeled_events,
    load_events,
    read_embeddings,
    read_text_mode,
)
from semtriage.holdout import split_by_cve
from semtriage.index import SparseEncoder, bm25_encoder, build_index
from semtriage.models import Event
from semtriage.qrels import build_qrels
from semtriage.retrieval_metrics import RetrievalReport, evaluate
from semtriage.retrieve import CONFIGS, retrieve
from semtriage.text_mode import build_text, parse_text_mode_label

SERVER_TIMEOUT = 300


def _validate_configs(configs: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(configs)
    for config in requested:
        if config not in CONFIGS:
            raise ValueError(f"unknown retrieval config: {config}")
    if not requested:
        raise ValueError("at least one retrieval config is required")
    return requested


def collect_runs(
    events: Sequence[Event],
    dense: np.ndarray,
    texts: Sequence[str],
    sparse: SparseEncoder,
    configs: Sequence[str],
    *,
    train_fraction: float = 0.7,
    client=None,
) -> dict[str, dict[str, list[str]]]:
    """Build the local index once and collect every requested ranking."""
    requested = _validate_configs(configs)
    _, test_indices = split_by_cve(events, train_fraction)
    query_indices = list(test_indices)
    queries = [events[index] for index in query_indices]

    if client is None:
        from qdrant_client import QdrantClient

        client = QdrantClient(":memory:")
    build_index(client, events, dense, texts, sparse)

    query_texts = [texts[index] for index in query_indices]
    sparse_vectors: list[tuple[list[int], list[float]]] = []
    for start in range(0, len(query_texts), 1024):
        sparse_vectors.extend(sparse(query_texts[start : start + 1024]))
    if len(sparse_vectors) != len(queries):
        raise ValueError("sparse encoder returned a different number of vectors")

    runs: dict[str, dict[str, list[str]]] = {}
    for config in requested:
        query_runs: dict[str, list[str]] = {}
        for index, query, sparse_vector in zip(query_indices, queries, sparse_vectors, strict=True):
            query_runs[query.event_id] = retrieve(
                client,
                query,
                dense[index],
                sparse_vector,
                config,
                limit=100,
            )
        runs[config] = query_runs
    return runs


def _safe_evaluate(
    runs: dict[str, list[str]],
    qrels: dict[str, set[str]] | dict[str, dict[str, float]],
) -> RetrievalReport:
    try:
        return evaluate(runs, qrels, ks=(1, 5, 10))
    except ValueError as exc:
        if str(exc) != "no query has a relevant id":
            raise
        names = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@1", "ndcg@5", "ndcg@10"]
        means = MappingProxyType(dict.fromkeys(names, 0.0))
        no_relevant = sum(1 for query_id in qrels if not qrels[query_id])
        skipped = sum(1 for query_id in runs if query_id not in qrels)
        return RetrievalReport(
            means=means,
            evaluated=0,
            skipped=skipped,
            no_relevant=no_relevant,
        )


def run_ablation(
    events: Sequence[Event],
    dense: np.ndarray,
    texts: Sequence[str],
    sparse: SparseEncoder,
    configs: Sequence[str],
    *,
    train_fraction: float = 0.7,
    client=None,
) -> dict[str, dict[str, RetrievalReport]]:
    """Collect rankings once and evaluate each config against CVE and graded qrels."""
    runs = collect_runs(
        events, dense, texts, sparse, configs, train_fraction=train_fraction, client=client
    )
    _, test_indices = split_by_cve(events, train_fraction)
    return _reports_from_runs(events, runs, [events[index].event_id for index in test_indices])


def _reports_from_runs(
    events: Sequence[Event],
    runs_by_config: dict[str, dict[str, list[str]]],
    query_ids: Sequence[str],
) -> dict[str, dict[str, RetrievalReport]]:
    cve_qrels, graded_qrels = build_qrels(events, query_ids)
    result: dict[str, dict[str, RetrievalReport]] = {}
    for config, runs in runs_by_config.items():
        result[config] = {
            "cve": _safe_evaluate(runs, cve_qrels),
            "graded": _safe_evaluate(runs, graded_qrels),
        }
    return result


def make_client(index_path: Path | None = None, url: str | None = None):
    """Return a Qdrant client for a server URL or a local path, or None for in-memory."""
    if index_path is None and url is None:
        return None
    from qdrant_client import QdrantClient

    if url is not None:
        return QdrantClient(url=url, timeout=SERVER_TIMEOUT)
    return QdrantClient(path=str(index_path))


def format_ablation(
    reports: dict[str, dict[str, RetrievalReport]],
    *,
    text_label: str,
    n_queries: int,
) -> str:
    """Format ablation header and markdown table."""
    cve_report = next(iter(reports.values()))["cve"]
    graded_report = next(iter(reports.values()))["graded"]
    lines = [
        f"Text: {text_label}",
        f"Queries: {n_queries}",
        f"No relevant (cve): {cve_report.no_relevant}",
        f"No relevant (graded): {graded_report.no_relevant}",
        "",
        "| Config | R@1 | R@5 | R@10 | MRR | nDCG@10 (graded) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config, report in reports.items():
        cve = report["cve"].means
        graded = report["graded"].means
        lines.append(
            f"| {config} | {cve['recall@1']:.3f} | {cve['recall@5']:.3f} "
            f"| {cve['recall@10']:.3f} | {cve['mrr']:.3f} "
            f"| {graded['ndcg@10']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run retrieval ablation on cached dense vectors."""
    parser = argparse.ArgumentParser(description="Compare offline retrieval configurations")
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Vectors saved by semtriage.baseline --save-embeddings",
    )
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help="Comma-separated retrieval configs",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Share of CVEs, oldest first, in the train split (default 0.7)",
    )
    parser.add_argument("--save-runs", type=Path, help="Write retrieved ids as JSON")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--index-path", type=Path, help="Persist the local Qdrant index")
    location.add_argument(
        "--qdrant-url", help="Use a Qdrant server, e.g. http://127.0.0.1:6333 (recreates 'events')"
    )
    args = parser.parse_args(argv)

    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")
    if not args.embeddings.is_file():
        parser.error(f"embeddings file not found: {args.embeddings}")

    configs = tuple(part.strip() for part in args.configs.split(",") if part.strip())
    try:
        configs = _validate_configs(configs)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        kept, _, _ = labeled_events(load_events(args.events))
        text_label = read_text_mode(args.embeddings)
        text_mode, max_chars = parse_text_mode_label(text_label)
        ids = [event.event_id for event in kept]
        texts = [build_text(event, text_mode, max_chars=max_chars) for event in kept]
        dense, _ = read_embeddings(args.embeddings, ids, texts, text_mode=text_label)
        _, test_indices = split_by_cve(kept, args.train_fraction)

        sparse = bm25_encoder()
        client = make_client(args.index_path, args.qdrant_url)
        query_ids = [kept[index].event_id for index in test_indices]
        runs = collect_runs(
            kept,
            dense,
            texts,
            sparse,
            configs,
            train_fraction=args.train_fraction,
            client=client,
        )
        reports = _reports_from_runs(kept, runs, query_ids)
    except ValueError as exc:
        parser.error(str(exc))

    print(format_ablation(reports, text_label=text_label, n_queries=len(test_indices)))
    if args.save_runs is not None:
        args.save_runs.write_text(json.dumps(runs, indent=2) + "\n")
    return 0


__all__ = [
    "collect_runs",
    "format_ablation",
    "make_client",
    "main",
    "run_ablation",
]


if __name__ == "__main__":
    raise SystemExit(main())
