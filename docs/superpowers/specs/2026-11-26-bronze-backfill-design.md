# P1-T5: bronze writer and backfill CLI (short design)

## Intent

Record raw NVD and reviewed GHSA records on disk exactly as the APIs return them, so that the
labeled dataset (P1-T6) and the deterministic replay (phase 3) are built from files, never from
live API calls. A backfill CLI fills bronze for a date range through the existing clients
(`NvdClient.iter_cves`, `GhsaClient.iter_advisories`).

## Goals, non-goals, anti-goals

- Goals: raw records are kept byte-for-byte as JSON values; re-running a backfill over the same or
  an overlapping range adds nothing when the records did not change; the layout is documented for
  P1-T6.
- Non-goals: normalization, sanitization, ordering (P1-T6); a live poller (phase 3); parallel
  writers to the same bronze directory.
- Anti-goals: tests that touch the network; raw data committed to git; a dedup that silently drops
  a record whose content really changed.

## Approaches

1. Content-hash dedup per source, partition by published day (chosen). The partition key never
   changes for a record, so every version of one advisory lands in the same file, and re-runs are
   idempotent.
2. Partition by fetch day. Simple, but the same unchanged record fetched on two days needs a
   cross-file index anyway, and files do not line up with the dataset's `published` order.
3. Dedup by `(source_id, modified)` instead of a content hash. Cheaper, but trusts the API to bump
   the timestamp on every change; the plan asks for a content hash.

## Bronze layout (contract for P1-T6)

```
<data_dir>/                     default: data/bronze (git-ignored through data/)
  nvd/YYYY-MM-DD.jsonl          one file per UTC day of cve.published
  ghsa/YYYY-MM-DD.jsonl         one file per UTC day of published_at
```

Each line is one JSON object, UTF-8, terminated by `\n`, keys in this order:

```json
{"sha256": "<64 hex>", "fetched_at": "2026-11-26T12:00:00Z", "record": { ...raw API item... }}
```

- `record` is the raw item: for NVD one element of `vulnerabilities` (`{"cve": {...}}`), for GHSA
  one advisory object.
- `sha256` is the hex SHA-256 of the canonical JSON of `record` (`sort_keys=True`,
  `separators=(",", ":")`, `ensure_ascii=False`, UTF-8) after dropping the volatile top-level keys
  listed per source. GHSA volatile keys: `epss` (rescored daily, it would add a new line for an
  unchanged advisory on every run). NVD has none. The stored `record` keeps those keys.
- `fetched_at` is the UTC time the line was written, ISO 8601 with `Z`, second precision.
- A file is append-only. Lines appear in fetch order, not sorted. The same `source_id` may appear
  on several lines when its content changed; P1-T6 keeps the version it needs (for example the
  latest `lastModified` / `updated_at`).
- Dedup scope is the whole source directory: a hash already present in any file of that source is
  not written again.
- A day is the UTC calendar date of the published timestamp (NVD timestamps without an offset are
  UTC).

## Components

- `semtriage.bronze`:
  - `content_hash(source, record) -> str`
  - `published_day(source, record) -> date`
  - `BronzeWriter(root, *, clock=...)` with `write(source, record) -> bool` (True when a new line
    was written). It loads the existing hashes of a source lazily on the first write for that
    source. A line in an existing file that is not valid JSON, or lacks `sha256`, raises
    `ValueError` naming the file and line number (no silent repair).
- `semtriage.backfill`:
  - `run_backfill(writer, start, end, *, nvd=None, ghsa=None) -> dict[Source, BackfillStats]`
    where `BackfillStats` counts `fetched`, `written`, `duplicates`. Both clients query by
    published date.
  - `main(argv=None, *, http=None) -> int`: `--start YYYY-MM-DD --end YYYY-MM-DD` (both inclusive
    UTC days, so the query is `[start 00:00, end + 1 day 00:00)`), `--source nvd|ghsa|all`
    (default all), `--data-dir` (default `data/bronze`). Reads `NVD_API_KEY` and `GITHUB_TOKEN`
    from the environment, optional. Prints one summary line per source. Exit code 2 on bad
    arguments (argparse), 0 on success.
  - `python -m semtriage.backfill` and a `semtriage-backfill` console script.

## Left to the builder

Internal helpers, file handle management (open in append mode per write is fine at this volume),
exact wording of the summary line.
