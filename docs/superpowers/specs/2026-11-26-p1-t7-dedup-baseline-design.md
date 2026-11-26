# P1-T7 design: offline dedup baseline

Status: approved by Diego on 2026-11-26.

## Intent

Produce the first deduplication number for the project: embed labeled advisories, group them by a cosine similarity threshold, and report pairwise and B-cubed scores (P1-T2 metrics) for a sweep of thresholds. Phases 2 and 3 (hybrid retrieval, incremental clustering) must beat this baseline.

## Goals

- `Embedder` protocol so the clustering code never depends on a specific model.
- A deterministic, dependency-free `HashingEmbedder` used as the test fake and as a lexical baseline.
- A dense `BgeM3Embedder` (bge-m3 through FlagEmbedding), the model named in the stack.
- Single-link threshold clustering and an efficient threshold sweep.
- A markdown report and a minimal CLI over a JSONL file of `Event` objects.

## Non-goals

- Sparse vectors, Qdrant, reranking (phase 2).
- Order-dependent or incremental clustering (phase 3).
- Building the dataset from bronze (P1-T6). The CLI reads `Event` JSONL, one `Event.model_dump_json()` per line.

## Anti-goals

- Leaking the label: texts are passed through `sanitize()` before embedding.
- Tests that download a model or touch the network.

## Dependencies (decided by Diego)

- `numpy` becomes a base dependency (similarity matrix; pure Python is far too slow for thousands of events).
- `FlagEmbedding` goes in an optional dependency group `embed` (`uv sync --group embed`). It pulls torch and downloads bge-m3 (about 2.3 GB) on first real use. The base install stays light.

## Approaches considered

1. Single-link connected components over edges with similarity >= threshold (chosen). One parameter to sweep, order independent, sweep can reuse one similarity matrix with union-find.
2. Leader clustering (assign to the most similar existing cluster). Order dependent; belongs to the streaming work in phase 3.
3. HDBSCAN or agglomerative with average linkage. Extra dependency and no single threshold that maps to the live system.

## Design

### `semtriage/embed.py`

- `Embedder` protocol: attributes `name: str`, `dim: int`; method `embed(texts: Sequence[str]) -> np.ndarray` returning float32 shape `(len(texts), dim)` with L2-normalized rows (an all-zero row stays zero).
- `l2_normalize(matrix)`: row normalization that leaves zero rows untouched.
- `HashingEmbedder(dim=256)`: lowercase alphanumeric tokens, each hashed with `blake2b` (digest size 8, little-endian int) modulo `dim`, counts accumulated, rows normalized. Deterministic across processes (no Python `hash`).
- `BgeM3Embedder(model_name="BAAI/bge-m3", *, use_fp16=False, batch_size=32, max_length=512)`: imports `FlagEmbedding.BGEM3FlagModel` inside `__init__`; a missing package raises `ImportError` telling the user to run `uv sync --group embed`. `embed` calls `encode(..., return_dense=True, return_sparse=False, return_colbert_vecs=False)` and normalizes `dense_vecs`. `dim = 1024`.

### `semtriage/baseline.py`

- `gold_label(event)`: the CVE ID when the event has exactly one, else `None`. Unlabeled events are excluded from the evaluation and counted.
- `threshold_clusters(vectors, threshold) -> list[int]`: single-link components over pairs with dot product >= threshold (vectors assumed normalized, computed in float64). Each label is the smallest row index in its component.
- `SweepRow(threshold, n_clusters, pairwise, bcubed)` and `sweep(ids, vectors, gold, thresholds)`: one row per threshold, in input order. Results must equal `threshold_clusters` for every threshold; the implementation computes the similarity matrix once and adds edges with union-find while walking thresholds from high to low.
- `BaselineReport(embedder, n_events, n_excluded, rows)` and `run_baseline(events, embedder, thresholds)`: filter labeled events, sanitize, embed, sweep.
- `format_report(report) -> str`: markdown table with threshold, clusters, pairwise P/R/F1, B-cubed P/R/F1.
- `load_events(path)` reads Event JSONL; `main(argv)` is the CLI: `python -m semtriage.baseline --events PATH [--embedder hashing|bge-m3] [--thresholds 0.5,0.6]`, default thresholds 0.50 to 0.95 in steps of 0.05.

### Tests

- No network and no model: `BgeM3Embedder` is tested by inserting a fake `FlagEmbedding` module into `sys.modules` with `monkeypatch`.
- Clustering tests use hand-built unit vectors with exact dot products; the sweep is checked against `threshold_clusters` on seeded random vectors.
- End-to-end tests run `run_baseline` and `main` with `HashingEmbedder` on in-memory events and a `tmp_path` JSONL file.

## Left to the builder

The internal sweep algorithm, provided results match `threshold_clusters`; formatting details of the table beyond its columns.
