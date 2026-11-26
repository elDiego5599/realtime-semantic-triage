"""Append-only bronze JSONL storage for raw source records."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from semtriage.models import Source

VOLATILE_KEYS: dict[Source, frozenset[str]] = {
    Source.NVD: frozenset(),
    Source.GHSA: frozenset({"epss"}),
}


def _parse_utc(value: str) -> datetime:
    """Parse an ISO 8601 timestamp and normalize it to UTC."""
    if "T" not in value and " " not in value:
        raise ValueError("timestamp must include a time")
    parsed = datetime.fromisoformat(value)
    return _to_utc(parsed)


def _to_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def content_hash(source: Source, record: dict) -> str:
    """Return the canonical SHA-256 hash of a raw record."""
    canonical = {key: value for key, value in record.items() if key not in VOLATILE_KEYS[source]}
    text = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def published_day(source: Source, record: dict) -> date:
    """Return the UTC published calendar day for a raw record."""
    if source == Source.NVD:
        cve = record.get("cve") if isinstance(record, dict) else None
        value = cve.get("published") if isinstance(cve, dict) else None
    elif source == Source.GHSA:
        value = record.get("published_at") if isinstance(record, dict) else None
    else:
        raise ValueError(f"unsupported source: {source}")
    if not isinstance(value, str):
        raise ValueError("published timestamp must be a string")
    try:
        return _parse_utc(value).date()
    except (OverflowError, ValueError) as exc:
        raise ValueError("published timestamp must be a valid ISO 8601 timestamp") from exc


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BronzeWriter:
    """Write deduplicated raw records to partitioned bronze JSONL files."""

    def __init__(
        self,
        root: Path | str,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.root = Path(root)
        self._clock = clock
        self._known_hashes: dict[Source, set[str]] = {}

    def _load_hashes(self, source: Source) -> set[str]:
        if source in self._known_hashes:
            return self._known_hashes[source]

        hashes: set[str] = set()
        source_dir = self.root / source.value
        if source_dir.is_dir():
            for path in sorted(source_dir.glob("*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            envelope = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                        if not isinstance(envelope, dict):
                            raise ValueError(f"{path}:{line_number}: invalid sha256")
                        sha256 = envelope.get("sha256")
                        if (
                            not isinstance(sha256, str)
                            or len(sha256) != 64
                            or any(character not in "0123456789abcdef" for character in sha256)
                        ):
                            raise ValueError(f"{path}:{line_number}: invalid sha256")
                        hashes.add(sha256)

        self._known_hashes[source] = hashes
        return hashes

    def write(self, source: Source, record: dict) -> bool:
        """Append a raw record once unless its source-wide content hash exists."""
        day = published_day(source, record)
        digest = content_hash(source, record)
        known = self._load_hashes(source)
        if digest in known:
            return False

        source_dir = self.root / source.value
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{day.isoformat()}.jsonl"

        now = self._clock()
        now = _to_utc(now)
        fetched_at = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        envelope = {
            "sha256": digest,
            "fetched_at": fetched_at,
            "record": record,
        }
        text = json.dumps(
            envelope,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

        known.add(digest)
        return True
