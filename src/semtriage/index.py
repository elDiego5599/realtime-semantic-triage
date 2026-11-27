"""Local Qdrant index over normalized advisory events."""

from collections.abc import Callable, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from semtriage.models import Event

SparseEncoder = Callable[[Sequence[str]], list[tuple[list[int], list[float]]]]

BATCH_SIZE = 1024


def bm25_encoder() -> SparseEncoder:
    """Return an encoder that maps texts to fastembed BM25 sparse vectors."""
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding("Qdrant/bm25")

    def encode(texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
        return [(result.indices.tolist(), result.values.tolist()) for result in model.embed(texts)]

    return encode


def build_index(
    client: QdrantClient,
    events: Sequence[Event],
    dense: np.ndarray,
    texts: Sequence[str],
    sparse: SparseEncoder,
    *,
    collection: str = "events",
) -> None:
    """Recreate the collection and upsert one point per event."""
    if not len(events) == dense.shape[0] == len(texts):
        raise ValueError(
            f"length mismatch: {len(events)} events, {dense.shape[0]} dense rows, "
            f"{len(texts)} texts"
        )

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=dense.shape[1], distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )
    client.create_payload_index(collection, "published", PayloadSchemaType.INTEGER)
    client.create_payload_index(collection, "cwe_ids", PayloadSchemaType.KEYWORD)

    for start in range(0, len(events), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(events))
        sparse_vectors = sparse(texts[start:stop])
        if len(sparse_vectors) != stop - start:
            raise ValueError("sparse encoder returned a different number of vectors")
        points = [
            PointStruct(
                id=row,
                vector={
                    "dense": dense[row].tolist(),
                    "bm25": SparseVector(indices=indices, values=values),
                },
                payload={
                    "event_id": events[row].event_id,
                    "source": events[row].source.value,
                    "published": int(events[row].published.timestamp()),
                    "cwe_ids": list(events[row].cwe_ids),
                },
            )
            for row, (indices, values) in zip(range(start, stop), sparse_vectors, strict=True)
        ]
        client.upsert(collection, points=points)
