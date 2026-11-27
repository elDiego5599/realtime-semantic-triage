"""Embedding interfaces and implementations."""

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Protocol for text embedding implementations."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of texts into normalized vectors."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a float32 copy of matrix with each row L2-normalized; zero rows stay zero."""
    values = np.array(matrix, dtype=np.float64, copy=True)
    if values.ndim != 2:
        raise ValueError("matrix must be 2-D")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, norms, out=values, where=norms != 0)
    return values.astype(np.float32)


def _check_texts(texts: Sequence[str]) -> None:
    """Reject a bare string, which would otherwise be embedded one character per row."""
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single str")


class HashingEmbedder:
    """Deterministic token-hashing text embedder."""

    def __init__(self, dim: int = 256) -> None:
        """Initialize the embedder with a positive vector dimension."""
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts using deterministic count-based hashing."""
        _check_texts(texts)
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % self.dim
                vectors[row_index, bucket] += 1.0
        return l2_normalize(vectors)


class BgeM3Embedder:
    """BGE-M3 dense text embedder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        use_fp16: bool = False,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        """Load a BGE-M3 model and store its encoding settings."""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "BGE-M3 embedding support is missing; install it with uv sync --group embed"
            ) from exc

        self.name = model_name
        self.dim = 1024
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts with BGE-M3 dense vectors."""
        _check_texts(texts)
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        encoded = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = np.asarray(encoded["dense_vecs"])
        if dense.shape != (len(texts), self.dim):
            raise ValueError(
                f"expected dense vectors of shape ({len(texts)}, {self.dim}), got {dense.shape}"
            )
        return l2_normalize(dense)

    def embed_sparse(self, texts: Sequence[str]):
        """Embed texts into sorted sparse lexical-weight vectors."""
        from semtriage.sparse_cache import lexical_weights_to_sparse

        _check_texts(texts)
        if len(texts) == 0:
            return []
        encoded = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [lexical_weights_to_sparse(weights) for weights in encoded["lexical_weights"]]
