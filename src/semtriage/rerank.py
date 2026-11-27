"""Rerank saved retrieval runs with the BGE-M3 cross encoder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from semtriage.baseline import labeled_events, load_events, read_embeddings, read_text_mode
from semtriage.text_mode import build_text, parse_text_mode_label

PAIR_BATCH = 4096
Scorer = Callable[[Sequence[tuple[str, str]]], list[float]]


def bge_reranker(
    *,
    use_fp16: bool = True,
    batch_size: int = 64,
    max_length: int = 1024,
) -> Scorer:
    """Load BGE reranker support lazily and return a pair scoring callable."""
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise ImportError(
            "BGE reranker support is missing; install it with uv sync --group embed"
        ) from exc

    model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=use_fp16)

    def score(pairs: Sequence[tuple[str, str]]) -> list[float]:
        pair_list = list(pairs)
        if not pair_list:
            return []
        raw = model.compute_score(
            pair_list,
            batch_size=batch_size,
            max_length=max_length,
        )
        if isinstance(raw, (float, int)):
            return [float(raw)]
        return [float(value) for value in raw]

    return score


def rerank_runs(
    runs: Mapping[str, Sequence[str]],
    texts: Mapping[str, str],
    scorer: Scorer,
    *,
    depth: int = 50,
) -> dict[str, list[str]]:
    """Rerank the first depth ids of every query across shared pair batches."""
    if depth < 0:
        raise ValueError(f"rerank depth must be non-negative, got {depth}")

    pair_counts: list[int] = []
    pairs: list[tuple[str, str]] = []
    for query_id, doc_ids in runs.items():
        top = list(doc_ids[:depth])
        pair_counts.append(len(top))
        if query_id not in texts:
            raise ValueError(f"text missing for query id: {query_id}")
        for doc_id in top:
            if doc_id not in texts:
                raise ValueError(f"text missing for document id: {doc_id}")
            pairs.append((texts[query_id], texts[doc_id]))

    all_scores: list[float] = []
    for start in range(0, len(pairs), PAIR_BATCH):
        chunk = pairs[start : start + PAIR_BATCH]
        scores = scorer(chunk)
        if len(scores) != len(chunk):
            raise RuntimeError(f"reranker returned {len(scores)} scores for {len(chunk)} pairs")
        all_scores.extend(float(score) for score in scores)

    reranked: dict[str, list[str]] = {}
    offset = 0
    for (query_id, doc_ids), top_count in zip(runs.items(), pair_counts, strict=True):
        original = list(doc_ids)
        top = original[:top_count]
        query_scores = all_scores[offset : offset + top_count]
        offset += top_count
        ranked = sorted(
            zip(top, query_scores, strict=True),
            key=lambda item: -item[1],
        )
        reranked[query_id] = [doc_id for doc_id, _ in ranked] + original[top_count:]
    return reranked


def _read_runs(path: Path) -> dict[str, dict[str, list[str]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: bad JSON ({exc.msg})") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: cannot read runs file ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: runs JSON must be an object")
    for config, runs in data.items():
        if not isinstance(config, str) or not isinstance(runs, dict):
            raise ValueError(f"{path}: runs JSON must map config names to query rankings")
        for query_id, doc_ids in runs.items():
            if not isinstance(query_id, str) or not isinstance(doc_ids, list):
                raise ValueError(f"{path}: bad run shape for config {config}")
            if not all(isinstance(doc_id, str) for doc_id in doc_ids):
                raise ValueError(f"{path}: bad run shape for config {config}")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """Rerank one saved retrieval configuration."""
    parser = argparse.ArgumentParser(description="Rerank saved retrieval runs")
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Vectors saved by semtriage.baseline --save-embeddings",
    )
    parser.add_argument("--runs", required=True, type=Path, help="Saved runs JSON")
    parser.add_argument("--config", required=True, help="Run configuration to rerank")
    parser.add_argument("--out", required=True, type=Path, help="Output reranked runs JSON")
    parser.add_argument("--depth", type=int, default=50, help="Rerank depth (default 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Reranker batch size")
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        default=True,
        help="Disable FP16 model loading",
    )
    args = parser.parse_args(argv)

    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")
    if not args.embeddings.is_file():
        parser.error(f"embeddings file not found: {args.embeddings}")
    if not args.runs.is_file():
        parser.error(f"runs file not found: {args.runs}")
    if args.depth < 0:
        parser.error("--depth must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        kept, _, _ = labeled_events(load_events(args.events))
        text_label = read_text_mode(args.embeddings)
        text_mode, max_chars = parse_text_mode_label(text_label)
        ids = [event.event_id for event in kept]
        texts = [build_text(event, text_mode, max_chars=max_chars) for event in kept]
        read_embeddings(args.embeddings, ids, texts, text_mode=text_label)

        all_runs = _read_runs(args.runs)
        if args.config not in all_runs:
            parser.error(f"config not found in runs: {args.config}")
        config_runs = all_runs[args.config]

        known_ids = set(ids)
        for query_id, doc_ids in config_runs.items():
            if query_id not in known_ids:
                parser.error(f"unknown event id in runs: {query_id}")
            for doc_id in doc_ids:
                if doc_id not in known_ids:
                    parser.error(f"unknown event id in runs: {doc_id}")

        text_by_id = dict(zip(ids, texts, strict=True))
        scorer = bge_reranker(
            use_fp16=args.fp16,
            batch_size=args.batch_size,
        )
        reranked = rerank_runs(
            config_runs,
            text_by_id,
            scorer,
            depth=args.depth,
        )
        args.out.write_text(
            json.dumps({f"{args.config}+rerank": reranked}, indent=2) + "\n",
            encoding="utf-8",
        )
    except ImportError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {len(reranked)} reranked runs to {args.out}")
    return 0


__all__ = [
    "PAIR_BATCH",
    "Scorer",
    "bge_reranker",
    "main",
    "rerank_runs",
]


if __name__ == "__main__":
    raise SystemExit(main())
