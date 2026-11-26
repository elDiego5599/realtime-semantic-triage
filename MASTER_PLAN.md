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
- Implementation tasks are delegated to Copilot CLI through brief files in `docs/briefs/`. Tests are written first.

## Phase 1: recording, labeled dataset, offline dedup

| ID | Task | Status | Brief |
|---|---|---|---|
| P1-T1 | Project scaffold and text sanitizer that strips CVE and GHSA identifiers and advisory URLs | done | [brief](docs/briefs/P1-T1-sanitize.md) |
| P1-T2 | Clustering metrics: pairwise P/R/F1 and B-cubed | done | [brief](docs/briefs/P1-T2-metrics.md) |
| P1-T3 | `Event` model and normalizers from NVD and GHSA raw JSON (fixtures from real responses) | todo | |
| P1-T4 | HTTP clients for NVD and GHSA with pagination and rate limiting (mocked transport in tests) | todo | |
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
- Copilot credit usage per task is logged in `docs/briefs/USAGE.md`. P1-T1 cost 1.03 AI credits, P1-T2 cost 0.89.
- Briefs must say that `uv.lock` is kept and that tool configuration in `pyproject.toml` must not be loosened to pass checks.

## Session log

- 2026-11-25: validated data sources (ADR 0001), created this plan, split phase 1 into tasks, completed P1-T1 with Copilot (branch `p1-t1-sanitize`).
- 2026-11-25: completed P1-T2 with Copilot (branch `p1-t2-metrics`).
- 2026-11-25: published the repository on GitHub and merged P1-T1 (PR 1) and P1-T2 (PR 2) into `main`. Next task: P1-T3.
- 2026-11-25: fixed label-leak gaps in `sanitize.py` found by the post-merge review (ID boundaries next to `_` and letters, Unicode dashes, URL boundaries, repeated empty-markup cleanup, rescan until stable). Implemented by DeepSeek through opencode in two runs (103 s and 86 s), reviewed by a fresh reviewer. Branch `claude/brave-wilson-05b039`, uncommitted. Open: decide whether U+2212, U+FF0D, zero-width spaces or plain spaces as ID separators need handling.
