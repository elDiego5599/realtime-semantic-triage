# MASTER_PLAN: Streaming Semantic Engine (realtime-semantic-triage)

This file is the authority for the project. Every session reads it first and updates it before finishing. Project state lives in files (this plan, specs, ADRs, git), never only in chat.

## Goal

Ingest a continuous stream of security advisories and use Qdrant for:

1. Live semantic deduplication: group events that describe the same incident even when worded differently.
2. Historical memory: retrieve similar past incidents with hybrid search (dense plus sparse), metadata filters and reranking.

A local LLM combines the live event with the retrieved context and writes a diagnosis with citations. Everything runs locally, with no API costs.

## Infrastructure

- Windows tower, RTX 5070 12 GB (Blackwell sm_120, CUDA 12.8 or newer, PyTorch cu128, Python in WSL2): Ollama, Qdrant in Docker (named volume), FastAPI backend.
- MacBook Air M4 over LAN: Streamlit frontend, evaluation, development.
- Ollama and Qdrant bind to 127.0.0.1. Only FastAPI, behind an API key, is exposed to the LAN. Image versions are pinned.

## Stack

bge-m3 (dense and sparse) through FlagEmbedding, bge-reranker-v2-m3, an 8B instruct LLM through Ollama (configurable), Qdrant, FastAPI with Pydantic, Streamlit. Qdrant BM25 through fastembed is benchmarked as the sparse alternative.

## Evaluation principles

- Deduplication is clustering: pairwise precision, recall and F1 plus B-cubed, on a stream labeled with the CVE ID.
- Recall@5, MRR and nDCG apply only to similar-incident retrieval.
- Deterministic evaluation first. An LLM judge is a secondary metric.
- No Kafka. A poller writes raw events to disk (bronze), one consumer processes them in order, and a deterministic replay of the recorded stream drives evaluation.

## Decisions

- [ADR 0001](docs/adr/0001-data-source.md): data source is NVD plus reviewed GHSA, CVE ID as label.

## Conventions

- Python 3.12 or newer, `uv`, src layout, package `semtriage`, `pytest`, `ruff`.
- Raw and derived data live under `data/` (git-ignored).
- Work happens on branches, never directly on `main`.
- Tests are written first. Implementation tasks are delegated to DeepSeek v4.1 (variant max) through `opencode run`, one fresh run per task, with a brief file in `.briefs/` (kept local, not committed). Claude reviews the diff, runs tests and lint, and commits. P1-T1 and P1-T2 were done with Copilot CLI; their briefs remain in `docs/briefs/`.
- Briefs and every file the implementer can read must hold no private data or credentials.

## Phase 1: recording, labeled dataset, offline dedup

| ID | Task | Status | Brief |
|---|---|---|---|
| P1-T1 | Project scaffold and text sanitizer that strips CVE and GHSA identifiers and advisory URLs | done | [brief](docs/briefs/P1-T1-sanitize.md) |
| P1-T2 | Clustering metrics: pairwise P/R/F1 and B-cubed | done | [brief](docs/briefs/P1-T2-metrics.md) |
| P1-T3 | `Event` model and normalizers from NVD and GHSA raw JSON (fixtures from real responses) | done | local `.briefs/P1-T3-events.md` |
| P1-T4 | HTTP clients for NVD and GHSA with pagination and rate limiting (mocked transport in tests) | done | local `.briefs/P1-T4-clients.md` |
| P1-T5 | Bronze writer (raw JSONL per source and day, content hash dedup) and backfill CLI by date range | todo | |
| P1-T6 | Dataset builder: bronze to sanitized events ordered by `published`, with dataset statistics and GHSA vs NVD lag | todo | |
| P1-T7 | Offline dedup baseline: `Embedder` protocol, dense embedding plus threshold clustering, threshold sweep reported with P1-T2 metrics | todo | |

## Phase 2: hybrid search, filters, reranker, ablation table

Not started.

## Phase 3: live consumer, incremental clustering, replay, latency p50 and p95 per stage

Not started.

## Phase 4: LLM diagnosis with citations

Not started.

## Open items

- Request a free NVD API key (Diego).
- Copilot credit usage for P1-T1 and P1-T2 is logged in `docs/briefs/USAGE.md`.
- DeepSeek runs: P1-T3 right first time, 1 run, 61 s wall time, no fixes needed. P1-T4 right first time, 1 run, 26 s, no fixes needed.
- Coordinator mode: opencode output always goes to a log file; independent tasks get their own session and worktree, in parallel only when files are disjoint. After P1-T4, P1-T5 and P1-T7 can run in parallel; P1-T6 needs the P1-T5 bronze format.
- Briefs must say that `uv.lock` is kept and that tool configuration in `pyproject.toml` must not be loosened to pass checks.

## Session log

- 2026-11-25: validated data sources (ADR 0001), created this plan, split phase 1 into tasks, completed P1-T1 with Copilot (branch `p1-t1-sanitize`).
- 2026-11-25: completed P1-T2 with Copilot (branch `p1-t2-metrics`).
- 2026-11-25: published the repository on GitHub and merged P1-T1 (PR 1) and P1-T2 (PR 2) into `main`. Next task: P1-T3.
- 2026-11-25: switched the implementer to DeepSeek v4.1 through opencode. Completed P1-T3 (branch `p1-t3-events`, not pushed). Next task: P1-T4.
- 2026-11-25: completed P1-T4 with DeepSeek (branch `p1-t4-clients`, stacked on `p1-t3-events`, not pushed). Next: P1-T5 and P1-T7 in parallel sessions.
- 2026-11-25: P1-T4b client robustness fixes (403 rate limits, Retry-After and x-ratelimit-reset, transport retries, NVD page advance by records returned, UTC window split, limiter loop, GHSA token host check). 14 new failing tests written first; DeepSeek right first time, 1 run, 55 s, no fixes; 105 tests pass, lint clean. Fresh opus review pending. Left uncommitted (no commits on 2026-11-25). Sanitizer boundary fix proposed as its own session.
- 2026-11-25: P1-T4b fresh opus review verdict "With fixes". Important, to do next: (1) GHSA token must require https, host api.github.com and port 443 (Link to http://api.github.com still gets the token); (2) ignore non-finite header waits and cap waits with a max_wait (Retry-After inf crashes sleep, 1e9 stalls). Minor: stricter integer Retry-After, tests for past reset, remaining != "0" and NVD clock, module-level GHSA 403 predicate. Then commit on or after 2026-11-26.
- 2026-11-25: P1-T4c review fixes (https, api.github.com and default port for the GHSA token; integer-only Retry-After; non-finite reset ignored; max_wait cap, default 3600 s, on all waits; module-level 403 predicates). DeepSeek not right first time: first run hung 28 min with no output and was stopped with SIGTERM; second run 78 s, no manual fixes. Fresh opus re-review: no Critical, code correct, but tests did not pin isfinite, the transport-error cap or the strip/isascii check; added 9 tests (mutation-checked), 127 tests pass, lint clean. Deferred minors: validate max_wait (0 <= max_wait < inf), comment on `_nvd_rate_limited`. Sanitizer boundary fix done in worktree branch `claude/brave-wilson-05b039` (64 tests pass). All uncommitted until 2026-11-26.
- 2026-11-26: P1-T4d minors: `max_wait` validated in both clients (finite and >= 0, ValueError otherwise) and a comment on `_nvd_rate_limited`. 8 failing tests written first. DeepSeek failed twice (both runs hung with an empty log until the 600 s alarm, the second with stdin closed), so finished by hand per the two-failure rule. 135 tests pass, lint clean. Still uncommitted; ready to commit.
