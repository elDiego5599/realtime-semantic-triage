"""Build deterministic labeled datasets from bronze advisory records."""

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from semtriage.models import Event, Source
from semtriage.normalize import normalize_ghsa, normalize_nvd
from semtriage.sanitize import find_identifiers

NVD_ID_RE = re.compile(r"CVE-\d{4}-\d{4,}")
GHSA_ID_RE = re.compile(r"GHSA(-[0-9a-z]{4}){3}")


@dataclass(frozen=True)
class Dataset:
    """A normalized event list and its aggregate statistics."""

    events: list[Event]
    stats: dict


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be valid ISO 8601") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _line_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {message}")


def _source_id_and_modified(source: Source, record: dict) -> tuple[str, datetime]:
    if source == Source.NVD:
        cve = record.get("cve")
        if not isinstance(cve, dict):
            raise ValueError("NVD record has no cve object")
        source_id = cve.get("id")
        modified = cve.get("lastModified")
        id_pattern = NVD_ID_RE
    else:
        source_id = record.get("ghsa_id")
        modified = record.get("updated_at")
        id_pattern = GHSA_ID_RE

    if not isinstance(source_id, str) or not source_id:
        raise ValueError("missing or invalid source id")
    if id_pattern.fullmatch(source_id) is None:
        raise ValueError("missing or invalid source id")
    if not isinstance(modified, str):
        raise ValueError("missing or invalid modified timestamp")
    return source_id, _parse_utc(modified)


def _read_source(bronze_dir: Path, source: Source) -> tuple[list[dict], int]:
    source_dir = bronze_dir / source.value
    if not source_dir.is_dir():
        return [], 0

    records: list[dict] = []
    line_count = 0
    for path in sorted(source_dir.glob("*.jsonl"), key=lambda item: item.name):
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise _line_error(path, line_number, "invalid UTF-8") from exc
                if not line.strip():
                    continue
                line_count += 1
                try:
                    envelope = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _line_error(path, line_number, "invalid JSON") from exc
                if not isinstance(envelope, dict):
                    raise _line_error(path, line_number, "bronze line must be an object")
                record = envelope.get("record")
                if not isinstance(record, dict):
                    raise _line_error(path, line_number, "record must be an object")
                fetched_at = envelope.get("fetched_at")
                if not isinstance(fetched_at, str):
                    raise _line_error(path, line_number, "fetched_at must be a string")
                try:
                    parsed_fetched_at = _parse_utc(fetched_at)
                except ValueError as exc:
                    raise _line_error(path, line_number, f"invalid fetched_at: {exc}") from exc
                records.append(
                    {
                        "source": source,
                        "record": record,
                        "path": path,
                        "line": line_number,
                        "fetched_at": parsed_fetched_at,
                    }
                )
    return records, line_count


def _latest_versions(bronze_dir: Path, source: Source) -> tuple[dict[str, dict], int]:
    raw_records, line_count = _read_source(bronze_dir, source)
    latest: dict[str, dict] = {}
    for item in raw_records:
        path = item["path"]
        line_number = item["line"]
        record = item["record"]
        try:
            source_id, modified = _source_id_and_modified(source, record)
        except ValueError as exc:
            raise _line_error(path, line_number, str(exc)) from exc
        key = (modified, item["fetched_at"], path.name, line_number)
        previous = latest.get(source_id)
        if previous is None or key > previous["version_key"]:
            latest[source_id] = {
                "record": record,
                "path": path,
                "line": line_number,
                "version_key": key,
            }
    return latest, line_count


def _build_stats(
    events: list[Event],
    source_stats: dict,
) -> dict:
    cve_event_counts: dict[str, int] = {}
    events_without_cve = 0
    events_with_multiple_cves = 0

    nvd_published: dict[str, datetime] = {}
    ghsa_published: dict[str, datetime] = {}

    for event in events:
        if not event.cve_ids:
            events_without_cve += 1
        if len(event.cve_ids) > 1:
            events_with_multiple_cves += 1
        for cve_id in event.cve_ids:
            cve_event_counts[cve_id] = cve_event_counts.get(cve_id, 0) + 1

            target = nvd_published if event.source == Source.NVD else ghsa_published
            previous = target.get(cve_id)
            if previous is None or event.published < previous:
                target[cve_id] = event.published

    histogram: dict[str, int] = {}
    for count in cve_event_counts.values():
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1

    lags: list[float] = []
    for cve_id in nvd_published.keys() & ghsa_published.keys():
        delta = ghsa_published[cve_id] - nvd_published[cve_id]
        lags.append(delta / timedelta(hours=1))

    # Lag is GHSA minus NVD: a negative lag means GHSA published first.
    ghsa_first = sum(lag < 0 for lag in lags)
    nvd_first = sum(lag > 0 for lag in lags)
    same_time = sum(lag == 0 for lag in lags)

    lag_hours: dict[str, int | float | None]
    lag_hours = {
        "pairs": len(lags),
        "ghsa_first": ghsa_first,
        "nvd_first": nvd_first,
        "same_time": same_time,
        "min": min(lags) if lags else None,
        "median": statistics.median(lags) if lags else None,
        "max": max(lags) if lags else None,
    }

    return {
        "events": len(events),
        "sources": source_stats,
        "clusters": {
            "cves": len(cve_event_counts),
            "size_histogram": histogram,
            "multi_event_cves": sum(count >= 2 for count in cve_event_counts.values()),
            "events_without_cve": events_without_cve,
            "events_with_multiple_cves": events_with_multiple_cves,
        },
        "lag_hours": lag_hours,
    }


def build_dataset(bronze_dir: Path | str) -> Dataset:
    """Build a labeled dataset from the bronze directory."""
    root = Path(bronze_dir)
    if not root.is_dir():
        raise ValueError(f"bronze directory missing: {root}")

    events: list[Event] = []
    source_stats: dict = {}

    for source in (Source.NVD, Source.GHSA):
        latest, line_count = _latest_versions(root, source)
        dropped = 0
        normalizer = normalize_nvd if source == Source.NVD else normalize_ghsa
        for item in latest.values():
            record = item["record"]
            path = item["path"]
            line_number = item["line"]
            try:
                event = normalizer(record)
            except Exception as exc:
                raise _line_error(path, line_number, f"normalizer failed: {exc}") from exc
            if event is None:
                dropped += 1
                continue
            identifiers = find_identifiers(event.text)
            if identifiers:
                raise _line_error(
                    path,
                    line_number,
                    f"identifier leak in event {event.event_id}: {', '.join(sorted(identifiers))}",
                )
            events.append(event)

        source_stats[source.value] = {
            "lines": line_count,
            "records": len(latest),
            "events": sum(event.source == source for event in events),
            "dropped": dropped,
        }

    events.sort(key=lambda event: (event.published, event.event_id))
    return Dataset(events=events, stats=_build_stats(events, source_stats))


def write_dataset(dataset: Dataset, out_dir: Path | str) -> None:
    """Write dataset events and statistics under an output directory."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    events_text = "".join(event.model_dump_json() + "\n" for event in dataset.events)
    (output / "events.jsonl").write_text(events_text, encoding="utf-8")
    stats_text = json.dumps(dataset.stats, indent=2, sort_keys=True) + "\n"
    (output / "stats.json").write_text(stats_text, encoding="utf-8")


def read_events(path: Path | str) -> list[Event]:
    """Read an events JSONL file, skipping blank lines."""
    events = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(Event.model_validate_json(line))
    return events


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the dataset, and print source summaries."""
    parser = argparse.ArgumentParser(description="Build a labeled dataset from bronze JSONL")
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--out-dir", default="data/dataset")
    args = parser.parse_args(argv)

    try:
        dataset = build_dataset(args.bronze_dir)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        write_dataset(dataset, args.out_dir)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for source in ("nvd", "ghsa"):
        counts = dataset.stats["sources"][source]
        print(
            f"{source}: lines={counts['lines']} records={counts['records']} "
            f"events={counts['events']} dropped={counts['dropped']}"
        )
    print(f"total: {dataset.stats['events']} events")
    return 0


def cli() -> None:
    """Run the console-script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
