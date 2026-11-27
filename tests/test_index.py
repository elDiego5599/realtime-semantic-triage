from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402

from semtriage.index import build_index  # noqa: E402
from semtriage.models import Event, Source  # noqa: E402


def make_event(
    source: Source,
    event_id: str,
    day: int,
    *,
    cwe: tuple[str, ...] = (),
) -> Event:
    when = datetime(2024, 1, day, tzinfo=UTC)
    return Event(
        event_id=f"{source.value}:{event_id}",
        source=source,
        source_id=event_id,
        published=when,
        modified=when,
        text="text",
        cve_ids=(),
        cwe_ids=cwe,
        ecosystems=(),
        packages=(),
        products=(),
        cna=None,
        severity=None,
    )


def fake_sparse(texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
    encoded = []
    for text in texts:
        indices = sorted({hash(token) % 1000 for token in text.split()})
        encoded.append((indices, [1.0] * len(indices)))
    return encoded


def make_events() -> list[Event]:
    return [
        make_event(Source.NVD, "one", 1, cwe=("CWE-79",)),
        make_event(Source.GHSA, "two", 2, cwe=("CWE-89", "CWE-79")),
        make_event(Source.NVD, "three", 3),
    ]


def make_dense(rows: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(rows, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_build_index_counts_points() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    build_index(client, events, make_dense(len(events)), ["a b", "b c", "d"], fake_sparse)
    assert client.count("events", exact=True).count == len(events)


def test_rebuild_recreates_collection() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    build_index(client, events, make_dense(len(events)), ["a b", "b c", "d"], fake_sparse)
    build_index(client, events[:1], make_dense(1), ["a"], fake_sparse)
    assert client.count("events", exact=True).count == 1


def test_payload_has_epoch_published() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    build_index(client, events, make_dense(len(events)), ["a b", "b c", "d"], fake_sparse)
    payload = client.retrieve("events", ids=[1])[0].payload
    assert payload == {
        "event_id": "ghsa:two",
        "source": "ghsa",
        "published": int(events[1].published.timestamp()),
        "cwe_ids": ["CWE-89", "CWE-79"],
    }


def test_stored_vectors_match_input() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    texts = ["a b", "b c", "d"]
    dense = make_dense(len(events))
    build_index(client, events, dense, texts, fake_sparse)
    vectors = client.retrieve("events", ids=[0], with_vectors=True)[0].vector
    np.testing.assert_allclose(vectors["dense"], dense[0], atol=1e-6)
    expected_indices, expected_values = fake_sparse([texts[0]])[0]
    assert vectors["bm25"].indices == expected_indices
    assert vectors["bm25"].values == expected_values


def test_length_mismatch_raises() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    with pytest.raises(ValueError):
        build_index(client, events, make_dense(2), ["a b", "b c", "d"], fake_sparse)
    with pytest.raises(ValueError):
        build_index(client, events, make_dense(3), ["a b", "b c"], fake_sparse)


def test_index_with_m3_stores_both_sparse_fields() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    m3 = [([1], [0.25]), ([2], [0.5]), ([1, 2], [0.3, 0.7])]
    build_index(
        client,
        events,
        make_dense(len(events)),
        ["a b", "b c", "d"],
        fake_sparse,
        m3=m3,
    )
    vectors = client.retrieve("events", ids=[0], with_vectors=True)[0].vector
    assert "bm25" in vectors
    assert vectors["m3"].indices == [1]
    assert vectors["m3"].values == [0.25]


def test_index_rejects_m3_length_mismatch() -> None:
    client = QdrantClient(":memory:")
    events = make_events()
    with pytest.raises(ValueError):
        build_index(
            client,
            events,
            make_dense(len(events)),
            ["a b", "b c", "d"],
            fake_sparse,
            m3=[([1], [0.25])],
        )
