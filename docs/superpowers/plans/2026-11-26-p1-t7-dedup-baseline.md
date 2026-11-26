# P1-T7 Offline Dedup Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed labeled advisories, cluster them by cosine threshold, and report pairwise and B-cubed scores over a threshold sweep.

**Architecture:** `semtriage/embed.py` holds the `Embedder` protocol and two embedders (hashing, bge-m3). `semtriage/baseline.py` holds gold labels, single-link threshold clustering, the sweep, the report and the CLI. The clustering code only sees numpy arrays.

**Tech Stack:** Python 3.12, numpy, FlagEmbedding (optional group `embed`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-26-p1-t7-dedup-baseline-design.md`

## Global Constraints

- No network and no model download in tests.
- `numpy` is a base dependency; `FlagEmbedding` only in dependency group `embed` (both already added, `uv.lock` updated; keep it).
- Do not loosen ruff or pytest configuration in `pyproject.toml`.
- Everything in English, no emojis. No private data in any file.
- Implementer (DeepSeek through `ocrun`) never runs git; Claude writes the failing tests, checks them with a sha file, reviews and commits.

## Review Focus

- Empty input (no texts, no labeled events): `embed([])` returns shape `(0, dim)`; `run_baseline` with no labeled events raises `ValueError` with a clear message instead of a metrics crash.
- Empty or punctuation-only text: hashing embedding is an all-zero row, not NaN.
- Duplicate event IDs or length mismatch between ids and vectors in `sweep`: `ValueError`.
- Thresholds out of order or repeated: rows come back in input order and repeated thresholds give identical rows.
- Missing `FlagEmbedding`: `ImportError` naming `uv sync --group embed`.

Each line has a test in the owning task's test file.

---

### Task 1: Embedders

**Files:**
- Create: `src/semtriage/embed.py`
- Test: `tests/test_embed.py` (written by Claude)

**Interfaces:**
- Produces:
  - `class Embedder(Protocol)`: `name: str`, `dim: int`, `embed(texts: Sequence[str]) -> np.ndarray` (float32, `(n, dim)`, L2-normalized rows, zero rows stay zero).
  - `l2_normalize(matrix: np.ndarray) -> np.ndarray` (float32).
  - `HashingEmbedder(dim: int = 256)`, `name == f"hashing-{dim}"`; `dim < 1` raises `ValueError`.
  - `BgeM3Embedder(model_name: str = "BAAI/bge-m3", *, use_fp16: bool = False, batch_size: int = 32, max_length: int = 512)`, `dim == 1024`, `name == model_name`.

- [ ] Step 1: write `tests/test_embed.py`, record its sha256 in `.briefs/P1-T7a.sha`.
- [ ] Step 2: run it, expect collection failure (`ModuleNotFoundError: semtriage.embed`).
- [ ] Step 3: brief `.briefs/P1-T7a-embed.md`; tokens are `re.findall(r"[a-z0-9]+", text.lower())`, bucket is `int.from_bytes(blake2b(token.encode(), digest_size=8).digest(), "little") % dim`; bge-m3 import inside `__init__`, `encode(list(texts), batch_size=..., max_length=..., return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]`.
- [ ] Step 4: run DeepSeek; verify sha unchanged, `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.
- [ ] Step 5: fresh opus reviewer; apply fixes; commit "Add Embedder protocol with hashing and bge-m3 embedders".

### Task 2: Threshold clustering, sweep, report and CLI

**Files:**
- Create: `src/semtriage/baseline.py`
- Test: `tests/test_baseline.py` (written by Claude)

**Interfaces:**
- Consumes: `Embedder`, `HashingEmbedder`, `BgeM3Embedder` from Task 1; `Scores`, `pairwise_scores`, `bcubed_scores` from `semtriage.metrics`; `sanitize` from `semtriage.sanitize`; `Event` from `semtriage.models`.
- Produces:
  - `gold_label(event: Event) -> str | None`
  - `threshold_clusters(vectors: np.ndarray, threshold: float) -> list[int]` (label = smallest row index in the component; pairs with float64 dot product >= threshold are linked)
  - `@dataclass(frozen=True, slots=True) SweepRow(threshold: float, n_clusters: int, pairwise: Scores, bcubed: Scores)`
  - `sweep(ids: Sequence[str], vectors: np.ndarray, gold: Mapping[str, Hashable], thresholds: Sequence[float]) -> list[SweepRow]`
  - `@dataclass(frozen=True, slots=True) BaselineReport(embedder: str, n_events: int, n_excluded: int, rows: list[SweepRow])`
  - `run_baseline(events: Iterable[Event], embedder: Embedder, thresholds: Sequence[float]) -> BaselineReport`
  - `format_report(report: BaselineReport) -> str`
  - `load_events(path: str | Path) -> list[Event]`
  - `DEFAULT_THRESHOLDS: tuple[float, ...]` = 0.50, 0.55, ..., 0.95
  - `main(argv: Sequence[str] | None = None) -> int`, plus `if __name__ == "__main__": raise SystemExit(main())`

- [ ] Step 1: write `tests/test_baseline.py`, record sha in `.briefs/P1-T7b.sha`.
- [ ] Step 2: run it, expect collection failure.
- [ ] Step 3: brief `.briefs/P1-T7b-baseline.md`. Sweep algorithm: similarity matrix once (float64), upper-triangle pairs with sim >= min threshold sorted by sim descending, walk the thresholds from high to low with a union-find, snapshot labels per threshold (relabel each component by its smallest index), return rows in input order.
- [ ] Step 4: run DeepSeek; verify sha, tests, lint, format.
- [ ] Step 5: fresh opus reviewer; apply fixes; commit "Add offline dedup baseline with threshold sweep".

### Task 3: Close out

- [ ] Append the P1-T7 status and a session log line to `MASTER_PLAN.md` (append only), commit "Update master plan for P1-T7".
