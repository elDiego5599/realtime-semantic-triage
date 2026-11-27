"""Ranking metrics for similar-incident retrieval: Recall@k, MRR and nDCG@k.

A ranking is a sequence of document ids, best first. Relevance judgments are
either a collection of relevant ids (binary gain 1) or a mapping from id to a
graded gain. Ids with gain 0 count as not relevant. Ids must be strings and a
ranking must not repeat an id (ValueError), since a repeated hit would be
counted twice.
"""

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

type Relevance = Collection[str] | Mapping[str, float]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Mean metrics over the evaluated queries.

    ``means`` is keyed ``recall@k``, ``mrr`` and ``ndcg@k`` and is read-only.
    ``evaluated`` counts queries with at least one relevant id, ``skipped``
    counts queries that appear in the runs but not in the judgments, and
    ``no_relevant`` counts judged queries without a relevant id.
    """

    means: Mapping[str, float]
    evaluated: int
    skipped: int
    no_relevant: int


def _check_k(k: int) -> None:
    """Require an integer cutoff of at least 1."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {k!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def _check_ranking(ranked: Sequence[str]) -> None:
    """Require a sequence of distinct string ids."""
    if isinstance(ranked, str):
        raise TypeError("ranked must be a sequence of ids, not a string")
    seen: set[str] = set()
    for doc_id in ranked:
        if not isinstance(doc_id, str):
            raise TypeError(f"ranked ids must be strings, got {doc_id!r}")
        if doc_id in seen:
            raise ValueError(f"duplicate id in ranking: {doc_id!r}")
        seen.add(doc_id)


def _gains(relevant: Relevance) -> dict[str, float]:
    """Return the positive gain of each relevant id."""
    if isinstance(relevant, str):
        raise TypeError("relevant must be a collection of ids, not a string")
    if isinstance(relevant, Mapping):
        items = list(relevant.items())
    else:
        items = [(doc_id, 1.0) for doc_id in relevant]
    gains: dict[str, float] = {}
    for doc_id, gain in items:
        if not isinstance(doc_id, str):
            raise TypeError(f"relevant ids must be strings, got {doc_id!r}")
        if isinstance(gain, bool) or not isinstance(gain, int | float):
            raise TypeError(f"gain must be a number, got {gain!r} for {doc_id!r}")
        if not math.isfinite(gain) or gain < 0:
            raise ValueError(f"gain must be finite and non-negative, got {gain!r} for {doc_id!r}")
        if gain > 0:
            gains[doc_id] = float(gain)
    return gains


def _recall(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    return sum(1 for doc_id in ranked[:k] if doc_id in gains) / len(gains)


def _reciprocal_rank(ranked: Sequence[str], gains: Mapping[str, float]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gains:
            return 1.0 / rank
    return 0.0


def _dcg(gains_in_order: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains_in_order, start=1))


def _ndcg(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    ideal = _dcg(sorted(gains.values(), reverse=True)[:k])
    if ideal == 0.0:
        return 0.0
    return _dcg([gains.get(doc_id, 0.0) for doc_id in ranked[:k]]) / ideal


def recall_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Share of relevant ids that appear in the first ``k`` results.

    Raises ValueError when there is no relevant id (empty collection, or only
    zero gains), because recall is undefined then.
    """
    _check_k(k)
    _check_ranking(ranked)
    gains = _gains(relevant)
    if not gains:
        raise ValueError("recall is undefined without relevant ids")
    return _recall(ranked, gains, k)


def reciprocal_rank(ranked: Sequence[str], relevant: Relevance) -> float:
    """Return 1 / rank of the first relevant hit (ranks start at 1), or 0.0 if none."""
    _check_ranking(ranked)
    return _reciprocal_rank(ranked, _gains(relevant))


def ndcg_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Normalized discounted cumulative gain of the first ``k`` results.

    The gain is 1 for ids in a collection and the mapped value for a mapping
    (linear gain). The result at rank r is discounted by log2(r + 1). The ideal
    DCG sorts all relevant gains in descending order and keeps the top ``k``.
    Returns 0.0 when the ideal DCG is 0.
    """
    _check_k(k)
    _check_ranking(ranked)
    return _ndcg(ranked, _gains(relevant), k)


def evaluate(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Relevance],
    ks: Sequence[int] = (1, 5, 10),
) -> RetrievalReport:
    """Average Recall@k, MRR and nDCG@k over queries with at least one relevant id.

    A judged query missing from ``runs`` counts as an empty ranking (score 0).
    Queries in ``runs`` without judgments are ignored and counted as skipped;
    judged queries without a relevant id are counted as ``no_relevant``.
    Raises ValueError when no query has a relevant id.
    """
    if not ks:
        raise ValueError("ks must not be empty")
    for k in ks:
        _check_k(k)
    if len(set(ks)) != len(ks):
        raise ValueError(f"ks must not repeat a value, got {tuple(ks)}")
    for query_id in (*runs, *qrels):
        if not isinstance(query_id, str):
            raise TypeError(f"query ids must be strings, got {query_id!r}")
    for ranked in runs.values():
        _check_ranking(ranked)

    names = [f"recall@{k}" for k in ks] + ["mrr"] + [f"ndcg@{k}" for k in ks]
    totals = dict.fromkeys(names, 0.0)
    evaluated = no_relevant = 0
    for query_id, relevant in qrels.items():
        gains = _gains(relevant)
        if not gains:
            no_relevant += 1
            continue
        evaluated += 1
        ranked = runs.get(query_id, ())
        for k in ks:
            totals[f"recall@{k}"] += _recall(ranked, gains, k)
            totals[f"ndcg@{k}"] += _ndcg(ranked, gains, k)
        totals["mrr"] += _reciprocal_rank(ranked, gains)
    if evaluated == 0:
        raise ValueError("no query has a relevant id")

    means = {name: totals[name] / evaluated for name in names}
    skipped = sum(1 for query_id in runs if query_id not in qrels)
    return RetrievalReport(
        means=MappingProxyType(means),
        evaluated=evaluated,
        skipped=skipped,
        no_relevant=no_relevant,
    )
