# P1-T6: dataset builder (short design)

## Intent

Turn bronze (raw NVD and GHSA JSONL, see `2026-11-26-bronze-backfill-design.md`) into the labeled,
sanitized event stream used by offline dedup (P1-T7) and replay (phase 3), plus statistics that
describe it: counts per source, events per CVE cluster, and GHSA vs NVD publication lag.

## Goals, non-goals, anti-goals

- Goals: deterministic output (same bronze, same bytes); one event per advisory, built from its
  latest bronze version; no CVE or GHSA identifier or advisory URL in any event text.
- Non-goals: embeddings, clustering, train/test splits, incremental builds.
- Anti-goals: silently skipping a corrupt bronze line; keeping an old version of an advisory that
  was later rejected or withdrawn.

## Pipeline

1. Read `<bronze_dir>/nvd/*.jsonl` and `<bronze_dir>/ghsa/*.jsonl` (files in name order, blank lines
   skipped). A missing source directory means no records for that source; a missing `bronze_dir`
   raises `ValueError`. A line that is not JSON, not an object, has no object `record`, or has no
   string `fetched_at`, or is not valid UTF-8 raises `ValueError` naming the file and line number
   (`<path>:<line>`).
2. Latest version per `(source, source_id)`: the line with the greatest `(modified, fetched_at,
   read position)`, where `modified` is `cve.lastModified` (NVD) or `updated_at` (GHSA) parsed to
   UTC, and read position is (file name, line number). A missing or invalid id or modified value
   raises `ValueError` with file and line. A valid NVD id matches `CVE-\d{4}-\d{4,}`; a valid GHSA
   id matches `GHSA(-[0-9a-z]{4}){3}`, both case-sensitive.
3. Normalize the latest version with `normalize_nvd` / `normalize_ghsa` (they already sanitize the
   text with `semtriage.sanitize.sanitize`). `None` (rejected, withdrawn, empty text) counts as
   dropped. An exception from the normalizer becomes `ValueError` with file and line.
4. Leak guard: if `find_identifiers(event.text)` is not empty, raise `ValueError` naming the event.
5. Order events by `(published, event_id)`.

## Model input and labels

Only `text` (and the structured `cwe_ids`, `ecosystems`, `packages`, `severity`) may be fed to a
dedup model. `event_id`, `source_id` and `cve_ids` carry the label and are metadata for evaluation
only; P1-T7 and later must never embed them or the whole event JSON.

Known gap, owned by `semtriage.sanitize` and not by this builder: the sanitizer and the leak guard
do not catch links without a scheme (`www.…`, `github.com/…/advisories/…`), CVE ids written with
spaces or U+2212, or other databases' ids (`PYSEC-…`, `GO-…`, `RUSTSEC-…`, `OSV-…`) that map to a
CVE. Measure their frequency on real bronze before deciding the fix.

## Output

`<out_dir>/events.jsonl`: one `Event.model_dump_json()` per line, in order. `read_events(path)`
reads it back. `<out_dir>/stats.json`: `json.dumps(stats, indent=2, sort_keys=True)` plus a newline (histogram
keys therefore sort as strings).

```
{
  "events": <total events>,
  "sources": {"nvd" | "ghsa": {"lines": n, "records": distinct ids, "events": n, "dropped": n}},
  "clusters": {
    "cves": <distinct CVE ids over all events>,
    "size_histogram": {"<events per CVE>": <number of CVEs>},
    "multi_event_cves": <CVEs with 2 or more events>,
    "events_without_cve": n,
    "events_with_multiple_cves": n
  },
  "lag_hours": {
    "pairs": <CVEs with an NVD event and at least one GHSA event>,
    "ghsa_first": n, "nvd_first": n, "same_time": n,
    "min": x | null, "median": x | null, "max": x | null
  }
}
```

An event counts once for each CVE id in its `cve_ids`. Lag for a pair is the earliest GHSA
`published` for that CVE minus the NVD `published`, in hours (positive: GHSA published later).

## Components (`semtriage.dataset`)

- `Dataset` (frozen dataclass): `events: list[Event]`, `stats: dict`.
- `build_dataset(bronze_dir) -> Dataset`
- `write_dataset(dataset, out_dir) -> None` (creates `out_dir`, overwrites both files)
- `read_events(path) -> list[Event]`
- `main(argv=None) -> int`: `--bronze-dir` (default `data/bronze`), `--out-dir` (default
  `data/dataset`); prints `<source>: lines=N records=N events=N dropped=N` per source and
  `total: N events`; on `ValueError` or `OSError` prints the message to stderr and returns 1
  without creating the output directory. `cli()` and console script `semtriage-dataset`.

## Left to the builder

Internal helpers, the exact wording of the summary lines.
