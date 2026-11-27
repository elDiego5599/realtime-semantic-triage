"""BGE-M3 lexical-weight cache aligned with the dense embedding cache."""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from semtriage.baseline import (
    labeled_events,
    load_events,
    read_embeddings,
    read_text_mode,
    texts_sha256,
)
from semtriage.embed import BgeM3Embedder
from semtriage.text_mode import build_text, parse_text_mode_label

Sparse = tuple[list[int], list[float]]
SparseEncoder = Callable[[Sequence[str]], list[Sparse]]
INDEX_LIMIT = 2**31


def lexical_weights_to_sparse(weights: Mapping[str | int, float]) -> Sparse:
    """Convert and sort lexical weights into integer-index sparse rows."""
    by_index: dict[int, float] = {}
    for key, value in weights.items():
        by_index[int(key)] = float(value)
    indices = sorted(by_index)
    return indices, [by_index[index] for index in indices]


def write_sparse(
    path: str | Path,
    ids: Sequence[str],
    texts: Sequence[str],
    vectors: Sequence[Sparse],
    *,
    embedder: str,
    text_mode: str = "full",
) -> None:
    """Write aligned sparse lexical-weight vectors in CSR layout."""
    if len(texts) != len(ids):
        raise ValueError(f"expected {len(ids)} texts, got {len(texts)}")
    if len(vectors) != len(ids):
        raise ValueError(f"expected {len(ids)} vectors, got {len(vectors)}")

    indptr = [0]
    all_indices: list[int] = []
    all_values: list[float] = []
    for indices, values in vectors:
        if len(indices) != len(values):
            raise ValueError("sparse indices and values lengths differ")
        if any(left >= right for left, right in zip(indices, indices[1:], strict=False)):
            raise ValueError("sparse indices must be strictly increasing within a row")
        if indices and not (0 <= indices[0] and indices[-1] < INDEX_LIMIT):
            raise ValueError(f"sparse indices out of range [0, {INDEX_LIMIT})")
        with np.errstate(over="ignore"):
            row_values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(row_values).all():
            raise ValueError("sparse values must contain finite numbers")
        all_indices.extend(indices)
        all_values.extend(values)
        indptr.append(len(all_indices))

    with Path(path).open("wb") as target:
        np.savez(
            target,
            event_ids=np.asarray(list(ids), dtype=str),
            indptr=np.asarray(indptr, dtype=np.int64),
            indices=np.asarray(all_indices, dtype=np.int32),
            values=np.asarray(all_values, dtype=np.float32),
            text_sha256=np.asarray(texts_sha256(texts), dtype=str),
            text_mode=np.asarray(text_mode, dtype=str),
            embedder=np.asarray(embedder, dtype=str),
        )


def read_sparse(
    path: str | Path,
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    text_mode: str | None = None,
) -> list[Sparse]:
    """Load CSR sparse vectors, refusing stale ids, modes or texts."""
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError, AttributeError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"{path}: cannot read sparse file ({exc})") from exc

    required = {
        "event_ids",
        "indptr",
        "indices",
        "values",
        "text_sha256",
        "text_mode",
        "embedder",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"{path}: sparse file is missing {sorted(missing)}")

    stored_ids = arrays["event_ids"]
    indptr = arrays["indptr"]
    indices = arrays["indices"]
    values = arrays["values"]
    if (
        stored_ids.ndim != 1
        or indptr.ndim != 1
        or indices.ndim != 1
        or values.ndim != 1
        or len(indptr) != len(stored_ids) + 1
    ):
        raise ValueError(
            f"{path}: bad shape, event_ids {stored_ids.shape}, indptr {indptr.shape}, "
            f"indices {indices.shape}, values {values.shape}"
        )
    if len(indices) != len(values):
        raise ValueError(f"{path}: sparse indices and values lengths differ")
    if indptr.dtype != np.int64 or indptr.size == 0 or indptr[0] != 0:
        raise ValueError(f"{path}: bad CSR indptr")
    if not np.all(np.diff(indptr) >= 0):
        raise ValueError(f"{path}: bad CSR indptr")
    if indptr[-1] != len(indices):
        raise ValueError(f"{path}: bad CSR indptr")

    if stored_ids.tolist() != list(ids):
        raise ValueError(
            f"{path}: event ids do not match the labeled events "
            f"({len(stored_ids)} stored, {len(ids)} expected, same order required)"
        )
    stored_mode = str(arrays["text_mode"])
    if text_mode is not None and stored_mode != text_mode:
        raise ValueError(
            f"{path}: text mode differs: cache built with {stored_mode}, "
            f"this run uses {text_mode}; re-embed or use the same mode"
        )
    if str(arrays["text_sha256"]) != texts_sha256(texts):
        raise ValueError(
            f"{path}: sanitized texts differ from the ones embedded "
            "(stale cache, for example after a sanitize change); re-embed"
        )

    vectors: list[Sparse] = []
    for start, end in zip(indptr[:-1], indptr[1:], strict=True):
        row_indices = [int(index) for index in indices[start:end]]
        row_values = [float(value) for value in values[start:end]]
        if any(left >= right for left, right in zip(row_indices, row_indices[1:], strict=False)):
            raise ValueError(f"{path}: sparse indices must be strictly increasing within a row")
        if row_indices and row_indices[0] < 0:
            raise ValueError(f"{path}: sparse indices out of range")
        if not all(np.isfinite(value) for value in row_values):
            raise ValueError(f"{path}: sparse values must contain finite numbers")
        vectors.append((row_indices, row_values))
    return vectors


def load_sparse_encoder(use_fp16: bool, batch_size: int) -> SparseEncoder:
    """Load BGE-M3 and return its sparse encoding callable."""
    return BgeM3Embedder(use_fp16=use_fp16, batch_size=batch_size).embed_sparse


def main(argv: Sequence[str] | None = None) -> int:
    """Build a sparse cache aligned with an existing dense cache."""
    parser = argparse.ArgumentParser(description="Build a BGE-M3 sparse vector cache")
    parser.add_argument("--events", required=True, type=Path, help="Path to Event JSONL")
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Dense vectors saved by semtriage.baseline --save-embeddings",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output sparse .npz cache")
    parser.add_argument("--fp16", action="store_true", help="Load BGE-M3 with FP16")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    args = parser.parse_args(argv)
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")
    if not args.embeddings.is_file():
        parser.error(f"embeddings file not found: {args.embeddings}")
    try:
        kept, _, _ = labeled_events(load_events(args.events))
        label = read_text_mode(args.embeddings)
        text_mode, max_chars = parse_text_mode_label(label)
        ids = [event.event_id for event in kept]
        texts = [build_text(event, text_mode, max_chars=max_chars) for event in kept]
        read_embeddings(args.embeddings, ids, texts, text_mode=label)
        encoder = load_sparse_encoder(args.fp16, args.batch_size)
        vectors = encoder(texts)
        if len(vectors) != len(texts):
            raise ValueError(
                f"sparse encoder returned {len(vectors)} vectors for {len(texts)} texts"
            )
        write_sparse(
            args.out,
            ids,
            texts,
            vectors,
            embedder="bge-m3",
            text_mode=label,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {len(ids)} sparse vectors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
