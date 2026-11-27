# Phase 2 Retrieval, Mac Slice, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offline ablation of dense, BM25 sparse, hybrid RRF and hybrid plus CWE filter retrieval over the advisory events, scored with Recall@k, MRR and nDCG@k.

**Architecture:** Product keys are added to `Event`. Qrels come from shared CVE ids and from shared CWE plus product. A local in-process Qdrant collection holds the cached bge-m3 dense vectors and fastembed BM25 sparse vectors. One retrieve function serves every configuration, and an ablation CLI prints a markdown table.

**Tech Stack:** Python 3.12, numpy, pydantic, qdrant-client (local mode), fastembed (`Qdrant/bm25`), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-27-phase2-retrieval-design.md`

## Global Constraints

**Dependencies**
- New dependency group `retrieval = ["qdrant-client", "fastembed"]` in `pyproject.toml`.
- The base package gets no new runtime dependency.
- Import qdrant_client and fastembed only inside `semtriage.index`, `semtriage.retrieve` and `semtriage.ablation`.

**Tests**
- No network and no model download.
- Build sparse vectors through an injectable encoder so tests use a fake one.
- Use the in-memory Qdrant client: `QdrantClient(":memory:")`.

**Retrieval rules**
- Date rule, always on: a candidate has `published <= query.published` and is never the query itself.
- Retrieval depth is 100.
- Metrics run at k 1, 5 and 10.

**Corpus and queries**
- The corpus is the events kept by `baseline.labeled_events`, in the order the embeddings cache stores them.
- Queries are the test split of `holdout.split_by_cve(events, 0.7)`.

**Repo conventions**
- Everything in English, with no emojis.
- Commits are imperative and short, with no Co-Authored-By line.
- Line length is 100.
- `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` must be clean after every task.

## Review Focus

1. **Query without a CWE under the CWE filter.** The query must not be filtered. It gets the same hits as plain hybrid (Task 4 test `test_cwe_filter_skips_queries_without_cwe`).
2. **Ties on publish time.** A candidate published at the same second as the query is allowed. The query itself is not (Task 4 test `test_date_rule_allows_same_time_excludes_self`).
3. **A CPE product with escaped characters.** `data\:\:entropy` becomes `data::entropy` (Task 1 test `test_cpe_product_unescaped`).
4. **An old `events.jsonl` without `products`.** It still loads, with `products == ()` (Task 1 test `test_event_without_products_loads`).
5. **A cache whose ids do not match the kept events.** The index builder raises ValueError and the CLI exits 2. This goes through `read_embeddings` (Task 5 test `test_cli_rejects_mismatched_cache`).

---

### Task 1: Product keys on Event

**Files**
- Modify: `src/semtriage/models.py`, `src/semtriage/normalize.py`
- Test: `tests/test_normalize.py`, `tests/test_models.py` (create if absent)

**Interfaces**
- Produces: `Event.products: tuple[str, ...] = ()`, holding sorted, unique, lowercased product keys.
- Produces: `normalize.product_key_from_package(name: str) -> str`.
- Produces: `normalize.product_keys_from_cpes(configurations: list[dict]) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing tests**

```python
def test_package_key_takes_last_segment():
    assert product_key_from_package("org.apache.logging.log4j:log4j-core") == "log4j-core"
    assert product_key_from_package("@babel/Core") == "core"
    assert product_key_from_package("lodash") == "lodash"

def test_cpe_product_unescaped():
    conf = [{"nodes": [{"cpeMatch": [
        {"criteria": "cpe:2.3:a:rrwo:data\\:\\:entropy:*:*:*:*:*:perl:*:*"},
        {"criteria": "cpe:2.3:a:Apache:Log4j:2.0:*:*:*:*:*:*:*"},
        {"criteria": "cpe:2.3:a:apache:log4j:2.1:*:*:*:*:*:*:*"}]}]}]
    assert product_keys_from_cpes(conf) == ("data::entropy", "log4j")

def test_event_without_products_loads():
    # model_validate on a dict without "products" gives products == ()
```

Also add a test for each source:
- `normalize_nvd` fills `products` from `cve["configurations"]`.
- `normalize_ghsa` fills `products` from its package names.

Build the fixtures from the existing ones in `tests/test_normalize.py`.

- [ ] **Step 2: Run** `uv run pytest tests/test_normalize.py tests/test_models.py`. Expected: FAIL, with an ImportError or AttributeError.

- [ ] **Step 3: Implement**
- Add the field with default `()`.
- CPE parsing: split the criteria string on `:` only where the colon is not preceded by a backslash, take index 4 (the product), drop the backslash escapes and lowercase it.
- Walk nested `nodes` and `children` if present.
- Package key: take the text after the last `/` or `:`, lowercased.

- [ ] **Step 4: Run** `uv run pytest`. Expected: all pass.

- [ ] **Step 5: Commit** "Add product keys to events".

- [ ] **Step 6 (coordinator, not implementer):** rebuild `data/dataset/events.jsonl` from bronze with `semtriage-dataset`. Check that the text sha in the caches still matches by loading `bge-m3-truncate.npz` with holdout.

### Task 2: Qrels

**Files**
- Create: `src/semtriage/qrels.py`
- Test: `tests/test_qrels.py`

**Interfaces**
- Consumes: `Event` (Task 1).
- Produces: `build_qrels(events: Sequence[Event], query_ids: Sequence[str]) -> tuple[dict[str, set[str]], dict[str, dict[str, float]]]`.
  - It returns `(cve_qrels, graded_qrels)`, keyed by query `event_id`.
  - `cve_qrels[q]` holds the event ids of the other source that share a CVE id with q and have `published <= q.published`.
  - `graded_qrels[q]` gives gain 2.0 to each id in `cve_qrels[q]`, and gain 1.0 to every other eligible event (any source, not q, `published <= q.published`) that shares at least one CWE and at least one product key with q.
  - Queries with no positives still appear, with empty values.

- [ ] **Step 1: Write failing tests.** Use synthetic events.
  - `test_cve_qrels_other_source_only`: a same-source event with the same CVE is not relevant.
  - `test_future_events_excluded`.
  - `test_graded_needs_cwe_and_product`: a shared CWE alone gives 0, a shared product alone gives 0, both give 1.0.
  - `test_cve_match_gain_two_wins_over_graded`.
  - `test_query_without_positives_present_empty`.
- [ ] **Step 2: Run.** Expected: FAIL, with an ImportError.
- [ ] **Step 3: Implement** with inverted indexes, CVE to events and (CWE, product) to events, not with an O(n²) scan. The real run has about 14k queries over 47k events.
- [ ] **Step 4: Run** `uv run pytest`. Expected: pass.
- [ ] **Step 5: Commit** "Build CVE and graded qrels".

### Task 3: Local Qdrant index

**Files**
- Modify: `pyproject.toml` (the `retrieval` group), `uv.lock` (via `uv lock`)
- Create: `src/semtriage/index.py`
- Test: `tests/test_index.py`

**Interfaces**
- Consumes: `Event`, plus a dense matrix aligned with the events.
- Produces:
  - `SparseEncoder = Callable[[Sequence[str]], list[tuple[list[int], list[float]]]]`
  - `bm25_encoder() -> SparseEncoder`, which wraps `fastembed.SparseTextEmbedding("Qdrant/bm25")` and is never called in tests.
  - `build_index(client: QdrantClient, events: Sequence[Event], dense: np.ndarray, texts: Sequence[str], sparse: SparseEncoder, *, collection: str = "events") -> None`
- Collection layout:
  - Named dense vector `"dense"` (cosine, dim from the matrix).
  - Named sparse vector `"bm25"` with `modifier=Modifier.IDF`.
  - Point id is the row index. Payload is `event_id`, `source`, `published` (int epoch seconds) and `cwe_ids` (list).
  - Payload indexes on `published` (integer) and `cwe_ids` (keyword).
  - Raise ValueError when `len(events)`, `dense.shape[0]` and `len(texts)` differ.

- [ ] **Step 1: Write failing tests** with `QdrantClient(":memory:")`, the hashing embedder or random unit vectors, and a fake encoder that maps each token's hash mod 1000 to weight 1.0.
  - `test_build_index_counts_points`.
  - `test_payload_has_epoch_published`.
  - `test_length_mismatch_raises`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Add the group with `uv add --group retrieval qdrant-client fastembed`. Upsert in batches of 1024.
- [ ] **Step 4: Run** `uv run --group retrieval pytest`. Expected: pass. Confirm that `uv run pytest` without the group skips `tests/test_index.py` through `pytest.importorskip("qdrant_client")`.
- [ ] **Step 5: Commit** "Index events in local Qdrant".

### Task 4: Retrieve configurations

**Files**
- Create: `src/semtriage/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces**
- Consumes: the Task 3 collection layout and `SparseEncoder`.
- Produces:
  - `CONFIGS = ("dense", "bm25", "hybrid", "hybrid+cwe")`
  - `retrieve(client, query: Event, dense_vector: np.ndarray, sparse_vector: tuple[list[int], list[float]], config: str, *, limit: int = 100, collection: str = "events") -> list[str]`, which returns event ids best first. An unknown config raises ValueError.
- Every config filters with `published <= query epoch` and `must_not` on the query's `event_id`.
- `hybrid` uses `query_points` with two `Prefetch` entries (dense and bm25, each with `limit`) and `FusionQuery(fusion=Fusion.RRF)`.
- `hybrid+cwe` adds a `cwe_ids` MatchAny filter when the query has CWEs, and to both prefetches.

- [ ] **Step 1: Write failing tests** on a small hand-built index where the expected order is obvious.
  - `test_dense_ranks_nearest_first`.
  - `test_bm25_ranks_token_overlap_first`.
  - `test_hybrid_contains_both_top_hits`.
  - `test_date_rule_allows_same_time_excludes_self`.
  - `test_cwe_filter_keeps_only_shared_cwe`.
  - `test_cwe_filter_skips_queries_without_cwe`.
  - `test_unknown_config_raises`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run --group retrieval pytest`. Expected: pass.
- [ ] **Step 5: Commit** "Retrieve with dense, BM25, hybrid and CWE filter".

### Task 5: Ablation CLI

**Files**
- Create: `src/semtriage/ablation.py`
- Test: `tests/test_ablation.py`

**Interfaces**
- Consumes: `load_events`, `labeled_events`, `read_text_mode`, `read_embeddings` (baseline); `parse_text_mode_label`, `build_text` (text_mode); `split_by_cve` (holdout); `build_qrels` (Task 2); `build_index`, `bm25_encoder` (Task 3); `retrieve`, `CONFIGS` (Task 4); `evaluate` (retrieval_metrics).
- Produces: `main(argv) -> int`, and `run_ablation(events, dense, texts, sparse, configs, *, train_fraction=0.7) -> dict[str, dict[str, RetrievalReport]]`. The result maps config to a report per qrel set, under the keys `"cve"` and `"graded"`.

**CLI**

```
python -m semtriage.ablation --events PATH --embeddings PATH [--configs dense,bm25,hybrid,hybrid+cwe] [--train-fraction 0.7] [--save-runs PATH] [--index-path DIR]
```

- The index uses `QdrantClient(path=index_path)` when the flag is given, and `":memory:"` otherwise.
- `--save-runs` writes JSON: `{config: {query_id: [ids]}}`.
- Header lines: text mode, number of queries, and `no_relevant` for each qrel set.

**Output table** (markdown)

```
| Config | R@1 | R@5 | R@10 | MRR | nDCG@10 (graded) |
```

- R@k and MRR come from the cve qrels. nDCG@10 comes from the graded qrels.
- Values have 3 decimals.
- Bad input (missing file, mismatched cache, unknown config) exits 2 with a message on stderr.

- [ ] **Step 1: Write failing tests**, with tmp files written as in `tests/test_holdout.py` (`write_mode_inputs`) and the fake sparse encoder injected by monkeypatching `ablation.bm25_encoder`.
  - `test_table_has_one_row_per_config`.
  - `test_perfect_dense_scores_one`: a synthetic set where the CVE twin is the nearest neighbour gives R@1 1.000.
  - `test_save_runs_json_roundtrip`.
  - `test_cli_rejects_unknown_config`.
  - `test_cli_rejects_mismatched_cache`.
  - `test_cli_uses_cached_text_mode` (a truncate:40 cache builds truncated BM25 texts).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Dense query vectors are the query's own row of the cache. Encode sparse query vectors with the same encoder over the same texts.
- [ ] **Step 4: Run** `uv run --group retrieval pytest` and ruff. Expected: clean.
- [ ] **Step 5: Commit** "Add retrieval ablation CLI".
- [ ] **Step 6 (coordinator):** run the ablation on `data/dataset/bge-m3-truncate.npz` and `bge-m3.npz`. Record the time, the RSS and the table in MASTER_PLAN, and save the output under `data/dataset/ablation-*.txt`.

## Later (tower tasks, separate plans)

- The bge-m3 sparse cache and a `bge-m3-sparse` config.
- `semtriage.rerank`: bge-reranker-v2-m3 over the hybrid top 50, written as saved runs that `ablation --load-runs` scores.
- A README ablation section, once the reranker row exists.
