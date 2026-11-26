# Bronze writer and backfill CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store raw NVD and GHSA records as JSONL per source and published day with content-hash
dedup, and fill that store for a date range from a CLI.

**Architecture:** `semtriage.bronze` owns the on-disk format (hash, day, append). `semtriage.backfill`
wires the existing `NvdClient` and `GhsaClient` to a `BronzeWriter` and exposes `main()`.

**Tech Stack:** Python 3.12, stdlib `json`/`hashlib`/`argparse`/`dataclasses`, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-11-26-bronze-backfill-design.md`

**Execution:** tests are written by the coordinator first (`tests/test_bronze.py`,
`tests/test_backfill.py`, sha256 pinned in `.briefs/P1-T5.sha256`); each task is implemented by
DeepSeek through `ocrun` from a brief in `.briefs/`, then checked by a fresh reviewer.

## Global Constraints

- Tests never touch the network; raw data only under git-ignored `data/`.
- Do not edit tests, `uv.lock`, or loosen tool configuration in `pyproject.toml`.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` must be clean.
- English only, no emojis.

## Review Focus

1. Re-running an overlapping range: no new lines (`test_main_rerun_writes_nothing_new`).
2. Restart with existing files: hashes reloaded from every day file of the source
   (`test_new_writer_loads_existing_hashes_from_every_day_file`).
3. Truncated or foreign line in bronze: loud `ValueError` with file and line
   (`test_corrupt_existing_line_raises_with_path_and_line`).
4. GHSA EPSS rescoring alone must not add a line (`test_write_skips_ghsa_record_that_only_changed_epss`).
5. Empty token env vars must not send empty credentials (`test_main_empty_token_is_treated_as_missing`).

---

### Task 1: bronze writer

**Files:** Create `src/semtriage/bronze.py`. Test: `tests/test_bronze.py`.

**Interfaces:**
- Produces:
  - `content_hash(source: Source, record: dict) -> str`
  - `published_day(source: Source, record: dict) -> datetime.date`
  - `BronzeWriter(root: Path | str, *, clock: Callable[[], datetime] = <utc now>)`,
    method `write(source: Source, record: dict) -> bool`
  - module constant `VOLATILE_KEYS: dict[Source, frozenset[str]]` = `{NVD: {}, GHSA: {"epss"}}`

- [ ] Run `uv run pytest tests/test_bronze.py`: fails with `ModuleNotFoundError`.
- [ ] Implement per the spec's "Bronze layout" section. Day = UTC date of `record["cve"]["published"]`
  (NVD) or `record["published_at"]` (GHSA); naive timestamps are UTC. Hashes are loaded lazily per
  source from `root/<source>/*.jsonl` on the first `write` for that source. Compute the day before
  touching disk, so a bad record writes nothing. One `write()` call per line in append mode.
- [ ] `uv run pytest tests/test_bronze.py` passes; full suite and ruff clean.
- [ ] Commit `Add bronze JSONL writer with content-hash dedup`.

### Task 2: backfill CLI

**Files:** Create `src/semtriage/backfill.py`; modify `pyproject.toml` (`[project.scripts]`
`semtriage-backfill = "semtriage.backfill:cli"`, where `cli()` calls `sys.exit(main())`).
Test: `tests/test_backfill.py`.

**Interfaces:**
- Consumes: Task 1 `BronzeWriter`; `NvdClient(http, api_key=...)`, `GhsaClient(http, token=...)`
  from `semtriage.clients`.
- Produces:
  - `@dataclass BackfillStats(fetched: int = 0, written: int = 0, duplicates: int = 0)`
  - `run_backfill(writer, start: datetime, end: datetime, *, nvd=None, ghsa=None) -> dict[Source, BackfillStats]`
    (calls `iter_cves(start, end, field="published")` and `iter_advisories(start, end, field="published")`;
    NVD first; keys only for clients given)
  - `main(argv: list[str] | None = None, *, http: httpx.Client | None = None) -> int`

- [ ] Run `uv run pytest tests/test_backfill.py`: fails with `ModuleNotFoundError`.
- [ ] Implement. Arguments: `--start`, `--end` (required, `date.fromisoformat`, inclusive UTC days;
  query `[start 00:00Z, end + 1 day 00:00Z)`; end before start -> `parser.error`), `--source`
  choices `nvd`, `ghsa`, `all` (default `all`), `--data-dir` default `data/bronze`. Tokens from
  `NVD_API_KEY` / `GITHUB_TOKEN`, empty string means none. When `http` is None, create
  `httpx.Client(timeout=30.0)` and close it. Print `"<source>: fetched=N written=N duplicates=N"`
  per source. `if __name__ == "__main__"` guard for `python -m semtriage.backfill`.
- [ ] `uv run pytest` passes; ruff check and format clean.
- [ ] Commit `Add backfill CLI that fills bronze from NVD and GHSA`.
