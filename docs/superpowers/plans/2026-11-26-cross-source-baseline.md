# Cross-source baseline Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict baseline candidate edges to NVD x GHSA pairs by default and add the all-singletons reference row to the report.

**Architecture:** An optional `groups` keyword on the clustering core selects cross-group edge collection, computed per pair of groups in row blocks. A `ReferenceRow` scored with the existing metrics is attached to `BaselineReport` and printed first.

**Tech Stack:** Python 3.12, NumPy, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-11-26-cross-source-baseline-design.md`

## Global Constraints

- Keep `uv.lock`; do not loosen tool configuration in `pyproject.toml`.
- English, no emojis; `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` pass.
- Edge collection stays in row blocks of 1024 rows.

## Review Focus

- A group with no rows on the other side (all events from one source): no edges, all singletons, no crash.
- Identical vectors from the same source in cross-source mode: never joined.
- `groups` length differs from the row count: `ValueError`.
- An unknown `pairs` value in `run_baseline`: `ValueError`.
- Threshold rows with no predicted pairs: pairwise P shown as `n/a`.

---

### Task 1: Cross-source edges, `pairs` option, singleton reference

**Files:**
- Modify: `src/semtriage/baseline.py`
- Test: `tests/test_baseline.py` (tests written first by the coordinator, section "Cross-source pairs and singleton reference")

**Interfaces:**
- Produces: `threshold_clusters(vectors, threshold, *, groups=None) -> list[int]`; `sweep(ids, vectors, gold, thresholds, *, groups=None) -> list[SweepRow]`; `ReferenceRow(name: str, n_clusters: int, pairwise: Scores, bcubed: Scores)`; `singletons_reference(ids, gold) -> ReferenceRow` with `name="singletons"`; `BaselineReport(..., pairs: str | None = None, reference: ReferenceRow | None = None)`; `run_baseline(events, embedder, thresholds, *, pairs="cross-source")`; `PAIR_MODES = ("cross-source", "all")`; CLI `--pairs`.

- [ ] Step 1: Coordinator writes the failing tests; run `uv run pytest tests/test_baseline.py` and see them fail on import or assertions.
- [ ] Step 2: Implementer changes `baseline.py` per the spec until all tests pass.
- [ ] Step 3: Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`; all pass.
- [ ] Step 4: Coordinator reviews, then commits.
