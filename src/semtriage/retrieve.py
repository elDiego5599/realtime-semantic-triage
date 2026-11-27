"""Retrieval configurations for the local advisory index."""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from semtriage.models import Event

CONFIGS = ("dense", "bm25", "hybrid", "hybrid+cwe")


def retrieve(
    client: QdrantClient,
    query: Event,
    dense_vector: np.ndarray,
    sparse_vector: tuple[list[int], list[float]],
    config: str,
    *,
    limit: int = 100,
    collection: str = "events",
) -> list[str]:
    """Retrieve event ids best first using the requested configuration."""
    if config not in CONFIGS:
        raise ValueError(f"unknown retrieval config: {config}")

    epoch = int(query.published.timestamp())
    must = [FieldCondition(key="published", range=Range(lte=epoch))]
    must_not = [FieldCondition(key="event_id", match=MatchValue(value=query.event_id))]

    if config == "hybrid+cwe" and query.cwe_ids:
        must.append(
            FieldCondition(
                key="cwe_ids",
                match=MatchAny(any=list(query.cwe_ids)),
            )
        )

    query_filter = Filter(must=must, must_not=must_not)

    if config == "dense":
        result = client.query_points(
            collection,
            query=dense_vector.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=["event_id"],
        )
        return [point.payload["event_id"] for point in result.points]

    sparse_indices, sparse_values = sparse_vector
    if not sparse_indices:
        if config == "bm25":
            return []
        if config in ("hybrid", "hybrid+cwe"):
            result = client.query_points(
                collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector.tolist(),
                        using="dense",
                        filter=query_filter,
                        limit=limit,
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                with_payload=["event_id"],
            )
            return [point.payload["event_id"] for point in result.points]

    sparse = SparseVector(indices=sparse_indices, values=sparse_values)

    if config == "bm25":
        result = client.query_points(
            collection,
            query=sparse,
            using="bm25",
            query_filter=query_filter,
            limit=limit,
            with_payload=["event_id"],
        )
        return [point.payload["event_id"] for point in result.points]

    result = client.query_points(
        collection,
        prefetch=[
            Prefetch(
                query=dense_vector.tolist(),
                using="dense",
                filter=query_filter,
                limit=limit,
            ),
            Prefetch(
                query=sparse,
                using="bm25",
                filter=query_filter,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=["event_id"],
    )
    return [point.payload["event_id"] for point in result.points]
