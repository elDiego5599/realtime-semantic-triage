# Phase 2 Retrieval, Tower Slice, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the bge-m3 sparse rows and the bge-reranker-v2-m3 row to the retrieval ablation table.

**Architecture:**
- A tower CLI writes a bge-m3 lexical-weight cache that is aligned with an existing dense cache.
- The index gains an optional second sparse vector `m3`, and retrieve gains two configs that use it.
- A rerank CLI reorders the hybrid top 50 of saved runs with bge-reranker-v2-m3 and writes new runs.
- The ablation loads saved runs as extra rows, so tower and Mac rows land in one table.

**Tech Stack:** Python 3.12, numpy, FlagEmbedding (`BGEM3FlagModel`, `FlagReranker`, in the `embed` group), qdrant-client, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-27-phase2-retrieval-design.md`. This plan builds on the Mac slice, `docs/superpowers/plans/2026-11-27-phase2-retrieval.md`.

## Global Constraints

**Dependencies and models**
- No new dependency group. The model code uses the existing `embed` group, and the index code uses the `retrieval` group.
- Import FlagEmbedding only inside the functions that load a model.

**Tests**
- No network, no model download and no GPU.
- Every model sits behind an injectable callable, so tests use fakes.

**Retrieval rules**
- The Mac slice rules hold: the date rule, depth 100, k 1, 5 and 10, and test-split queries.
- Reranker depth is 50.
- The reranker reads the same texts as the dense cache, built with the cache's stored text mode.

**Repo conventions**
- Repo conventions as in the Mac slice: English, no emojis, short imperative commits with no Co-Authored-By line, line length 100.
- `uv run pytest` and ruff are clean after every task.

## Review Focus

1. **A sparse cache built from other texts than the dense cache.** It is refused with ValueError, and the CLI exits 2 (Task 1 test `test_read_sparse_rejects_other_texts`; Task 2 test `test_cli_rejects_stale_sparse_cache`).
2. **An empty lexical-weight dict.** A very short text can come out empty. It is stored as an empty row, and m3 queries with no indices behave like `bm25` with no indices (Task 1 test `test_empty_weights_roundtrip`).
3. **Saved runs whose query ids are not the test split.** Ids outside the split are rejected with exit 2, and missing ids are counted as skipped by `evaluate` (Task 4 test `test_load_runs_rejects_unknown_query`).
4. **A run shorter than the rerank depth.** Every id is still reranked, and none is dropped or duplicated (Task 3 test `test_rerank_short_run_keeps_all_ids`).
5. **A token id key from FlagEmbedding is a string.** It is converted to int and sorted, and duplicate ids cannot occur (Task 1 test `test_lexical_weights_to_sparse_sorted_ints`).

---

### Task 1: bge-m3 sparse cache

**Files**
- Create: `src/semtriage/sparse_cache.py`
- Modify: `src/semtriage/embed.py`, to add the sparse method
- Test: `tests/test_sparse_cache.py`

**Interfaces**
- Produces: `Sparse = tuple[list[int], list[float]]`.
- Produces: `lexical_weights_to_sparse(weights: Mapping[str | int, float]) -> Sparse`. Indices are sorted ascending ints and values are float.
- Produces: `BgeM3Embedder.embed_sparse(texts) -> list[Sparse]`. It calls `encode(..., return_dense=False, return_sparse=True)` and converts each `lexical_weights` entry.
- Produces: `write_sparse(path, ids, texts, vectors: Sequence[Sparse], *, embedder: str, text_mode: str) -> None`.
  - One `.npz` with `event_ids`, `text_sha256` (from `baseline.texts_sha256`), `text_mode` and `embedder`.
  - The vectors are stored as CSR: `indptr` int64, `indices` int32, `values` float32.
- Produces: `read_sparse(path, ids, texts, *, text_mode: str) -> list[Sparse]`.
  - It refuses a file whose ids, text mode or text hash differ, using the same checks and messages as `baseline.read_embeddings`.
- Produces: CLI `python -m semtriage.sparse_cache --events PATH --embeddings DENSE.npz --out PATH [--fp16] [--batch-size 32]`.
  - It loads the labeled events and reads the text mode from the dense cache.
  - It builds the texts exactly as holdout does, checks them against the dense cache with `read_embeddings`, embeds them with `BgeM3Embedder(use_fp16=...)` and writes the sparse cache.
  - Bad input exits 2.

- [ ] **Step 1: Write failing tests**
  - `test_lexical_weights_to_sparse_sorted_ints`: `{"7": 0.5, "3": 0.25}` gives `([3, 7], [0.25, 0.5])`.
  - `test_sparse_roundtrip`: write, then read with the same ids and texts, gives equal lists (values within 1e-6).
  - `test_empty_weights_roundtrip`: a row `([], [])` survives the roundtrip.
  - `test_read_sparse_rejects_other_texts`, `test_read_sparse_rejects_other_ids`, `test_read_sparse_rejects_other_mode`: each raises ValueError.
  - `test_cli_writes_cache_aligned_with_dense`:
    - Write tmp events and a `truncate:40` dense cache, as in `tests/test_ablation.py`.
    - Monkeypatch `sparse_cache.load_sparse_encoder` (a module function returning the embedder's `embed_sparse`) with a fake.
    - Run `main`, then `read_sparse` with the same texts.
  - `test_cli_rejects_mismatched_dense_cache`: exits 2.
- [ ] **Step 2: Run** `uv run pytest tests/test_sparse_cache.py`. Expected: FAIL.
- [ ] **Step 3: Implement** as in Interfaces. `load_sparse_encoder(use_fp16: bool, batch_size: int) -> Callable[[Sequence[str]], list[Sparse]]` builds the `BgeM3Embedder`.
- [ ] **Step 4: Run** the full suite and ruff. Expected: clean.
- [ ] **Step 5: Commit** "Add bge-m3 sparse cache".

### Task 2: m3 sparse configs in index, retrieve and ablation

**Files**
- Modify: `src/semtriage/index.py`, `src/semtriage/retrieve.py`, `src/semtriage/ablation.py`
- Test: `tests/test_index.py`, `tests/test_retrieve.py`, `tests/test_ablation.py`

**Interfaces**
- Consumes: `read_sparse` and `Sparse` from Task 1.
- Produces: `build_index(..., m3: Sequence[Sparse] | None = None)`.
  - When it is given, the collection also gets the sparse vector `"m3"`: `SparseVectorParams()`, with no IDF modifier, because lexical weights are already weighted.
  - It needs the same length as `events`, else ValueError.
- Produces: `CONFIGS = ("dense", "bm25", "hybrid", "hybrid+cwe", "m3-sparse", "m3-hybrid")` and `M3_CONFIGS = ("m3-sparse", "m3-hybrid")`.
  - `m3-sparse` behaves like `bm25`, and `m3-hybrid` like `hybrid`, but both use the sparse field `"m3"`.
  - The `retrieve` signature is unchanged. The caller passes the m3 query vector as `sparse_vector` for m3 configs.
- Produces: ablation `--sparse-cache PATH`.
  - Default configs stay the four Mac configs. The m3 configs need `--sparse-cache`; asking for one without it exits 2.
  - The query m3 vector is the query's own row of the sparse cache.
  - `run_ablation` and `collect_runs` gain `m3: Sequence[Sparse] | None = None`.

- [ ] **Step 1: Write failing tests**
  - `test_index_with_m3_stores_both_sparse_fields`.
  - `test_index_rejects_m3_length_mismatch`.
  - `test_m3_sparse_ranks_weight_overlap_first`.
  - `test_m3_hybrid_contains_both_top_hits`.
  - `test_m3_config_without_m3_field_raises`: on an index built without m3, `retrieve(..., "m3-sparse")` raises (`pytest.raises(Exception)`). The CLI never gets there, because it requires `--sparse-cache`.
  - `test_cli_m3_config_requires_sparse_cache`: exits 2.
  - `test_cli_rejects_stale_sparse_cache`: exits 2.
  - `test_table_has_m3_rows_with_sparse_cache`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** In `retrieve`, map each config to its sparse field name once, so the bm25 and m3 paths share code.
- [ ] **Step 4: Run** the full suite and ruff. Expected: clean.
- [ ] **Step 5: Commit** "Add bge-m3 sparse retrieval configs".

### Task 3: Reranker

**Files**
- Create: `src/semtriage/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces**
- Produces: `Scorer = Callable[[Sequence[tuple[str, str]]], list[float]]`.
- Produces: `bge_reranker(*, use_fp16: bool = True, batch_size: int = 64, max_length: int = 1024) -> Scorer`.
  - It wraps `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=...)` and calls `compute_score(pairs, batch_size=..., max_length=...)`.
  - It always returns a list, also for a single pair.
- Produces: `rerank_runs(runs: Mapping[str, Sequence[str]], texts: Mapping[str, str], scorer: Scorer, *, depth: int = 50) -> dict[str, list[str]]`.
  - For each query, it scores `(texts[query_id], texts[doc_id])` for the first `depth` ids and sorts them by score, descending. Ties keep the original order (stable sort).
  - The ids after `depth` follow unchanged.
  - Pairs are sent to the scorer in one call per 4096 pairs, across queries.
- Produces: CLI `python -m semtriage.rerank --events PATH --embeddings DENSE.npz --runs RUNS.json --config hybrid --out PATH [--depth 50] [--batch-size 64] [--no-fp16]`.
  - It builds the texts from the dense cache's text mode, as holdout does, and verifies them with `read_embeddings`.
  - It writes JSON `{"<config>+rerank": {query_id: [ids]}}`.
  - Bad input (missing file, config not in the runs, unknown event id in the runs) exits 2.

- [ ] **Step 1: Write failing tests** with a fake scorer that scores word overlap.
  - `test_rerank_orders_top_by_score`.
  - `test_rerank_keeps_tail_after_depth`.
  - `test_rerank_short_run_keeps_all_ids`.
  - `test_rerank_ties_keep_original_order`.
  - `test_rerank_batches_pairs`: a scorer that records its call sizes, 3 queries times 50 ids, with a batch of 64 monkeypatched in. No call is larger than the batch, and the total is 150.
  - `test_cli_writes_reranked_runs`: monkeypatch `rerank.bge_reranker`.
  - `test_cli_rejects_missing_config`.
  - `test_cli_rejects_unknown_event_id`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** The pair batch size is a module constant `PAIR_BATCH = 4096`.
- [ ] **Step 4: Run** the full suite and ruff. Expected: clean.
- [ ] **Step 5: Commit** "Add bge-reranker-v2-m3 rerank of saved runs".

### Task 4: Ablation `--load-runs`

**Files**
- Modify: `src/semtriage/ablation.py`
- Test: `tests/test_ablation.py`

**Interfaces**
- Produces: `--load-runs PATH`, which can be repeated. Each file is JSON `{config: {query_id: [ids]}}`.
  - Every config in it becomes an extra table row, after the computed rows.
  - The rows are scored with the same qrels.
- Produces: `--configs ""` (empty), which runs no retrieval and builds no index. It is allowed only with `--load-runs`.
- A loaded config name that clashes with a computed one, or with an earlier loaded file, exits 2.
- A query id outside the test split exits 2.
- Produces: `load_runs(path) -> dict[str, dict[str, list[str]]]`. It raises ValueError on bad JSON or a bad shape.

- [ ] **Step 1: Write failing tests**
  - `test_load_runs_adds_rows`.
  - `test_load_runs_only_skips_index`: with `--configs ""`, monkeypatch `build_index` to fail if it is called.
  - `test_load_runs_rejects_unknown_query`.
  - `test_load_runs_rejects_name_clash`.
  - `test_load_runs_rejects_bad_json`.
  - `test_empty_configs_without_load_runs_rejected`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the full suite and ruff. Expected: clean.
- [ ] **Step 5: Commit** "Score saved runs in the ablation".

### Task 5 (tower run, coordinator and Diego)

Run on the tower, with Qdrant on localhost:

- [ ] `python -m semtriage.sparse_cache --events data/dataset/events.jsonl --embeddings data/dataset/bge-m3-truncate.npz --out data/dataset/bge-m3-sparse-truncate.npz --fp16`
- [ ] Ablation over truncate with `--qdrant-url`, `--sparse-cache` and all six configs, using `--save-runs data/dataset/ablation-runs-truncate.json`.
- [ ] `python -m semtriage.rerank ... --runs data/dataset/ablation-runs-truncate.json --config hybrid --out data/dataset/rerank-runs-truncate.json`. If m3-hybrid beats hybrid, rerank m3-hybrid as well.
- [ ] Ablation `--configs "" --load-runs` with both run files, to get the final table.
- [ ] Record the time, max RSS, peak VRAM and the table in MASTER_PLAN, and bring back `data/dataset/ablation-*`, `rerank-*` and the sparse cache.
- [ ] Repeat the dense-only ablation with `full` for its extra row.
