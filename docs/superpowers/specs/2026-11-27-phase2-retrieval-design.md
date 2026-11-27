# Phase 2: similar-incident retrieval (design)

## Intent

Given a new security advisory, return the past advisories an analyst should read next:
first the same vulnerability reported by the other feed, then advisories of the same weakness
in the same product. Phase 1 answered "is this a duplicate?" with a pair rule. Phase 2 answers
"what is related?" as a ranked list, and measures which retrieval stack does it best with an
ablation table for the README.

## Goals

- One offline evaluation CLI that indexes the events, runs every query, and prints Recall@k,
  MRR and nDCG@k per retriever configuration (`semtriage.retrieval_metrics.evaluate`).
- Configurations: dense, sparse, hybrid (RRF), hybrid plus CWE filter, hybrid plus reranker.
- Everything except the reranker and bge-m3 sparse runs on the Mac.

## Non-goals

- A live service, a streaming index, or an API. The index is rebuilt per run.
- The reranker as a dedup pair verifier. It is a follow-up once the reranker exists.
- Tuning RRF weights or k on the test queries.

## Anti-goals

- Leaking the future: a query may only retrieve advisories published at or before it.
- Reporting a number picked on the same queries it is scored on.

## Task definition

- Corpus: every event in `events.jsonl`.
- Queries: every event in the test split of `semtriage.holdout` (CVEs ordered by earliest
  published, latest 30 percent). Same split, so phase 1 and phase 2 numbers describe the same
  period.
- Date filter, always on: candidates have `published <= query.published`, the query itself is
  excluded. This is part of the task, not an ablation row.
- Relevance, option (c), two qrel sets:
  - `cve` (binary): events of the other source that share a CVE id with the query. Primary
    metric set for Recall@k and MRR.
  - `graded`: gain 2 for the `cve` set; gain 1 for any other event that shares at least one CWE
    id and one product key with the query. Used for nDCG@10.
- Product key: lowercased product name. GHSA: package name after its last `/` or `:`
  (`org.apache.logging.log4j:log4j-core` gives `log4j-core`). NVD: the CPE 2.3 product field of
  the `configurations` criteria (part 5 of the CPE string, unescaped). Events without a key get
  no graded gain. About 37 percent of NVD bronze lines carry CPEs, so graded gain is sparse for
  NVD; the report prints how many queries have any graded positive.
- Queries with no positive in a qrel set are counted as `no_relevant` and reported, as
  `evaluate` already does.

## Approaches considered

1. Qdrant in-process local mode (`QdrantClient(path=...)`), named dense and sparse vectors,
   Query API with prefetch and `FusionQuery(RRF)`, payload filters on `published` and
   `cwe_ids`. Same client code as a Qdrant server later. Chosen.
2. Pure numpy and a hand-written BM25. Fastest to test, but it is not the stack the project
   wants to show and fusion and filters would be reimplemented.
3. Qdrant server in Docker on the Mac. Same code as 1 with an extra moving part; local mode is
   enough for 47k points.

Sparse vectors: BM25 through fastembed (`Qdrant/bm25`, collection sparse vector with
`Modifier.IDF`) runs on the Mac now. bge-m3 lexical weights need the GPU, so they become a
tower run that writes a sparse cache; the CLI gains that row when the cache exists.

## Pieces

1. `Event` gains `products: tuple[str, ...]` (product keys as above), filled by `normalize`
   for both sources; rebuild `events.jsonl` from bronze. Text and text hashes do not change.
2. `semtriage.qrels`: build the `cve` and `graded` qrel sets for a list of query ids from the
   events, honoring the date rule.
3. `semtriage.index`: create a local Qdrant collection from events plus a dense cache (`.npz`,
   any text mode; reads the label like holdout does) and BM25 sparse vectors; payload
   `published` (epoch seconds), `cwe_ids`, `source`.
4. `semtriage.retrieve`: one function per configuration returning ranked event ids for a query,
   top 100, with the date filter and optional CWE filter (any shared CWE; queries without CWE
   are not filtered).
5. `semtriage.rerank`: bge-reranker-v2-m3 (FlagEmbedding `FlagReranker`, fp16) over the hybrid
   top 50; tower only, writes the reranked runs to a file the Mac CLI can score.
6. `semtriage.ablation` CLI: runs configurations, scores them with `evaluate` at k 1, 5, 10,
   prints a markdown table, and can save and load runs as JSON so tower and Mac rows join.

## Constraints

- Dependencies in a new `retrieval` group: `qdrant-client`, `fastembed`. Reranker uses the
  existing `embed` group. No new runtime dependency for the base package.
- Index directory under `data/` (git-ignored).
- Tests use small synthetic events and the hashing embedder; no network, no model download.
  fastembed BM25 is wrapped so tests can inject a fake sparse encoder.
- Dense query vectors come from the same cache as the corpus (queries are corpus events), so
  the Mac needs no bge-m3 model.

## Calls I made (correct any)

- Dense text mode for the table: `truncate:1000`, since it matches `full` on dedup and is
  cheaper; `full` gets one extra row.
- RRF with Qdrant defaults, no weight tuning.
- Reranker depth 50, final list 10 for metrics, top 100 for retrieval.
- Product key from CPE product and package last segment, no vendor matching.

## Left to the builder

- Exact payload index setup and batch sizes for upserts.
- JSON layout for saved runs.
