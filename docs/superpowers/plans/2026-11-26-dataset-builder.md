# Dataset builder Implementation Plan

**Goal:** Build the sanitized, `published`-ordered event dataset and its statistics from bronze.

**Spec:** `docs/superpowers/specs/2026-11-26-dataset-builder-design.md`

**Execution:** the coordinator writes `tests/test_dataset.py` first (sha256 pinned locally); one
implementer run from a local brief; diff review by the coordinator and a fresh reviewer.

## Global Constraints

- Tests never touch the network; data only under git-ignored `data/`.
- Do not edit tests, `uv.lock`, or loosen tool configuration in `pyproject.toml`.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` must be clean.
- English only, no emojis.

## Review Focus

1. A later rejected or withdrawn version removes the event (`test_later_rejected_version_drops_the_event`).
2. Version choice by UTC `modified`, then `fetched_at`, then read position, across files.
3. Corrupt bronze fails loudly with `<path>:<line>`, including normalizer errors on the chosen line.
4. Leak guard on event text (`test_leaked_identifier_after_normalization_raises`).
5. Byte-identical output for the same bronze.

### Task 1: dataset builder and CLI

**Files:** Create `src/semtriage/dataset.py`; modify `pyproject.toml` (`[project.scripts]`
`semtriage-dataset = "semtriage.dataset:cli"`). Test: `tests/test_dataset.py`.

**Interfaces:**
- Consumes: `normalize_nvd`, `normalize_ghsa` (imported by name into `semtriage.dataset`, tests
  patch `semtriage.dataset.normalize_nvd`), `find_identifiers`, `Event`, `Source`.
- Produces: `Dataset(events, stats)` frozen dataclass, `build_dataset(bronze_dir)`,
  `write_dataset(dataset, out_dir)`, `read_events(path)`, `main(argv=None) -> int`, `cli()`.

- [ ] `uv run pytest tests/test_dataset.py` fails on import.
- [ ] Implement per the spec; `if __name__ == "__main__"` guard for `python -m semtriage.dataset`.
- [ ] Full suite and ruff clean; commit `Add dataset builder from bronze with statistics`.
