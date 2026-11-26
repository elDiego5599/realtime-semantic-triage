"""CLI for backfilling raw NVD and GHSA records into bronze storage."""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from semtriage.bronze import BronzeWriter
from semtriage.clients import GhsaClient, NvdClient
from semtriage.models import Source


@dataclass
class BackfillStats:
    """Counts for one backfill source."""

    fetched: int = 0
    written: int = 0
    duplicates: int = 0


def run_backfill(
    writer: BronzeWriter,
    start: datetime,
    end: datetime,
    *,
    nvd: NvdClient | None = None,
    ghsa: GhsaClient | None = None,
) -> dict[Source, BackfillStats]:
    """Fetch and write published records from the selected clients."""
    stats: dict[Source, BackfillStats] = {}

    if nvd is not None:
        source_stats = BackfillStats()
        _drain(writer, Source.NVD, nvd.iter_cves(start, end, field="published"), source_stats)
        stats[Source.NVD] = source_stats

    if ghsa is not None:
        source_stats = BackfillStats()
        _drain(
            writer,
            Source.GHSA,
            ghsa.iter_advisories(start, end, field="published"),
            source_stats,
        )
        stats[Source.GHSA] = source_stats

    return stats


def _drain(
    writer: BronzeWriter,
    source: Source,
    records,
    stats: BackfillStats,
    *,
    progress=None,
) -> None:
    """Drain records into bronze storage while updating counts."""
    for record in records:
        stats.fetched += 1
        if writer.write(source, record):
            stats.written += 1
        else:
            stats.duplicates += 1
        if progress is not None and stats.fetched % 1000 == 0:
            progress(source, stats.fetched)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value} is not a valid date; expected YYYY-MM-DD"
        ) from exc


def main(argv: list[str] | None = None, *, http: httpx.Client | None = None) -> int:
    """Parse CLI arguments and backfill bronze records."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill NVD and GHSA bronze records. Uses NVD_API_KEY and GITHUB_TOKEN "
            "environment variables when set. Re-running the same range is safe; "
            "duplicates are skipped."
        )
    )
    date_help = "first/last UTC day to fetch, inclusive, YYYY-MM-DD"
    parser.add_argument("--start", required=True, type=_parse_date, help=date_help)
    parser.add_argument("--end", required=True, type=_parse_date, help=date_help)
    parser.add_argument(
        "--source",
        choices=("nvd", "ghsa", "all"),
        default="all",
        help="source to backfill",
    )
    parser.add_argument(
        "--data-dir",
        default="data/bronze",
        help="bronze data directory",
    )
    args = parser.parse_args(argv)

    if args.end < args.start:
        parser.error("--end must not be before --start")

    start = datetime.combine(args.start, datetime.min.time(), tzinfo=UTC)
    end = datetime.combine(args.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)

    nvd_key = os.environ.get("NVD_API_KEY") or None
    ghsa_token = os.environ.get("GITHUB_TOKEN") or None
    if args.source in ("nvd", "all") and nvd_key is None:
        print("NVD_API_KEY is not set", file=sys.stderr)
    if args.source in ("ghsa", "all") and ghsa_token is None:
        print("GITHUB_TOKEN is not set", file=sys.stderr)

    writer = BronzeWriter(Path(args.data_dir))
    close_http = http is None
    client = http if http is not None else httpx.Client(timeout=30.0)

    try:

        def progress(source: Source, fetched: int) -> None:
            print(f"{source.value}: {fetched} records so far", file=sys.stderr)

        for source in (Source.NVD, Source.GHSA):
            if args.source != "all" and args.source != source.value:
                continue

            stats = BackfillStats()
            try:
                if source == Source.NVD:
                    nvd = NvdClient(client, api_key=nvd_key)
                    records = nvd.iter_cves(start, end, field="published")
                else:
                    ghsa = GhsaClient(client, token=ghsa_token)
                    records = ghsa.iter_advisories(start, end, field="published")
                _drain(writer, source, records, stats, progress=progress)
            except httpx.HTTPError as exc:
                print(
                    f"{source.value}: fetched={stats.fetched} "
                    f"written={stats.written} duplicates={stats.duplicates} (partial)"
                )
                print(str(exc), file=sys.stderr)
                print("re-run the same range to resume", file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                print(
                    f"{source.value}: fetched={stats.fetched} "
                    f"written={stats.written} duplicates={stats.duplicates} (partial)"
                )
                print("KeyboardInterrupt", file=sys.stderr)
                print("re-run the same range to resume", file=sys.stderr)
                return 130

            print(
                f"{source.value}: fetched={stats.fetched} "
                f"written={stats.written} duplicates={stats.duplicates}"
            )
    finally:
        if close_http:
            client.close()
    return 0


def cli() -> None:
    """Run the console-script entry point."""
    sys.exit(main())


if __name__ == "__main__":
    cli()
