from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402

from semtriage.index import build_index  # noqa: E402
from semtriage.models import Event, Source  # noqa: E402
from semtriage.retrieve import CONFIGS, retrieve  # noqa: E402


def make_event(
    event_id: str,
    day: int,
    *,
    cwe_ids: tuple[str, ...] = (),
    hour: int = 0,
) -> Event:
    when = datetime(2024, 1, day, hour, tzinfo=UTC)
    return Event(
        event_id=event_id,
        source=Source.NVD,
        source_id=event_id,
        published=when,
        modified=when,
        text="text",
        cve_ids=(),
        cwe_ids=cwe_ids,
        ecosystems=(),
        packages=(),
        products=(),
        cna=None,
        severity=None,
    )


def fake_sparse(texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
    vocabulary = {"alpha": 1, "beta": 2, "gamma": 3}
    encoded = []
    for text in texts:
        indices = sorted({vocabulary[word] for word in text.lower().split() if word in vocabulary})
        encoded.append((indices, [1.0] * len(indices)))
    return encoded


@pytest.fixture
def retrieval_setup() -> tuple[QdrantClient, list[Event], np.ndarray]:
    events = [
        make_event("nvd:self", 3, cwe_ids=("CWE-79",)),
        make_event("nvd:dense", 2, cwe_ids=("CWE-79",)),
        make_event("nvd:sparse", 1, cwe_ids=("CWE-89",)),
        make_event("nvd:future", 4, cwe_ids=("CWE-79",)),
    ]
    dense = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    client = QdrantClient(":memory:")
    build_index(
        client,
        events,
        dense,
        ["gamma", "beta", "alpha", "gamma"],
        fake_sparse,
    )
    return client, events, dense


def test_dense_ranks_nearest_first(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    hits = retrieve(client, events[0], dense[0], ([1], [1.0]), "dense", limit=10)
    assert hits == ["nvd:dense", "nvd:sparse"]


def test_bm25_ranks_token_overlap_first(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    hits = retrieve(client, events[0], dense[0], ([1], [1.0]), "bm25", limit=10)
    assert hits == ["nvd:sparse"]


def test_hybrid_contains_both_top_hits(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    hits = retrieve(client, events[0], dense[0], ([1], [1.0]), "hybrid", limit=10)
    assert "nvd:dense" in hits
    assert "nvd:sparse" in hits


def test_date_rule_allows_same_time_excludes_self(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    query = make_event("nvd:self", 3, hour=12)
    dense_query = np.array([1.0, 0.0], dtype=np.float32)
    hits = retrieve(client, query, dense_query, ([1], [1.0]), "dense", limit=10)
    assert "nvd:self" not in hits
    assert "nvd:dense" in hits
    assert "nvd:future" not in hits


def test_cwe_filter_keeps_only_shared_cwe(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    query = make_event("nvd:query", 3, cwe_ids=("CWE-89",))
    hits = retrieve(client, query, dense[0], ([1], [1.0]), "hybrid+cwe", limit=10)
    assert hits == ["nvd:sparse"]


def test_cwe_filter_skips_queries_without_cwe(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    query = make_event("nvd:query", 3)
    plain = retrieve(client, query, dense[0], ([1], [1.0]), "hybrid", limit=10)
    filtered = retrieve(client, query, dense[0], ([1], [1.0]), "hybrid+cwe", limit=10)
    assert filtered == plain


def test_unknown_config_raises(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    assert CONFIGS == ("dense", "bm25", "hybrid", "hybrid+cwe")
    with pytest.raises(ValueError):
        retrieve(client, events[0], dense[0], ([1], [1.0]), "unknown")


def test_bm25_empty_query_returns_empty(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    assert retrieve(client, events[0], dense[0], ([], []), "bm25", limit=10) == []


def test_hybrid_with_empty_sparse_equals_dense_order(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    dense_hits = retrieve(client, events[0], dense[0], ([], []), "dense", limit=10)
    hybrid_hits = retrieve(client, events[0], dense[0], ([], []), "hybrid", limit=10)
    assert hybrid_hits == dense_hits


def test_limit_caps_results(
    retrieval_setup: tuple[QdrantClient, list[Event], np.ndarray],
) -> None:
    client, events, dense = retrieval_setup
    hits = retrieve(client, events[0], dense[0], ([1], [1.0]), "dense", limit=1)
    assert len(hits) == 1
