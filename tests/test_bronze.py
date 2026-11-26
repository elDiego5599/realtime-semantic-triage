import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from semtriage.bronze import BronzeWriter, content_hash, published_day
from semtriage.models import Source

FIXTURES = Path(__file__).parent / "fixtures"
FETCHED = datetime(2026, 11, 26, 12, 30, 45, 123456, tzinfo=UTC)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def nvd_record(cve_id: str = "CVE-2026-00001", published: str = "2026-11-11T05:19:51.527") -> dict:
    return {"cve": {"id": cve_id, "published": published, "descriptions": []}}


def ghsa_record(
    ghsa_id: str = "GHSA-aaaa-bbbb-cccc", published: str = "2026-11-25T15:26:23Z", **extra
) -> dict:
    return {"ghsa_id": ghsa_id, "published_at": published, **extra}


def writer(root: Path) -> BronzeWriter:
    return BronzeWriter(root, clock=lambda: FETCHED)


def read_lines(path: Path) -> list[dict]:
    raw = path.read_bytes().decode("utf-8")
    assert raw.endswith("\n")
    return [json.loads(line) for line in raw.splitlines()]


def canonical_sha(record: dict) -> str:
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# content_hash


def test_content_hash_is_sha256_of_canonical_json() -> None:
    record = {"b": 1, "a": {"y": "é", "x": [1, 2]}}
    assert content_hash(Source.NVD, record) == canonical_sha(record)


def test_content_hash_ignores_key_order() -> None:
    first = {"a": 1, "b": {"c": 2, "d": 3}}
    second = {"b": {"d": 3, "c": 2}, "a": 1}
    assert content_hash(Source.NVD, first) == content_hash(Source.NVD, second)


def test_content_hash_changes_with_content() -> None:
    assert content_hash(Source.NVD, nvd_record()) != content_hash(
        Source.NVD, nvd_record(cve_id="CVE-2026-00002")
    )


def test_ghsa_hash_ignores_epss_only() -> None:
    base = ghsa_record(severity="high")
    rescored = ghsa_record(severity="high", epss={"percentage": 0.5, "percentile": 0.9})
    assert content_hash(Source.GHSA, rescored) == content_hash(Source.GHSA, base)
    assert content_hash(Source.GHSA, rescored) == canonical_sha(base)
    changed = ghsa_record(severity="critical", epss={"percentage": 0.5, "percentile": 0.9})
    assert content_hash(Source.GHSA, changed) != content_hash(Source.GHSA, base)


def test_nvd_hash_keeps_top_level_epss_key() -> None:
    record = {**nvd_record(), "epss": 1}
    assert content_hash(Source.NVD, record) == canonical_sha(record)


def test_content_hash_does_not_mutate_record() -> None:
    record = ghsa_record(epss={"percentage": 0.1})
    content_hash(Source.GHSA, record)
    assert record["epss"] == {"percentage": 0.1}


# published_day


def test_published_day_nvd_naive_timestamp_is_utc() -> None:
    assert published_day(Source.NVD, nvd_record(published="2026-11-11T23:59:59.999")) == date(
        2026, 11, 11
    )


def test_published_day_nvd_offset_is_converted_to_utc() -> None:
    record = nvd_record(published="2026-11-11T21:00:00.000-05:00")
    assert published_day(Source.NVD, record) == date(2026, 11, 12)


def test_published_day_ghsa_uses_published_at() -> None:
    assert published_day(Source.GHSA, ghsa_record(published="2026-11-25T00:00:00Z")) == date(
        2026, 11, 25
    )


def test_published_day_from_real_fixtures() -> None:
    assert published_day(Source.NVD, load("nvd_cve.json")) == date(2026, 11, 11)
    assert published_day(Source.GHSA, load("ghsa_advisory.json")) == date(2026, 11, 25)


# BronzeWriter


def test_write_creates_day_file_with_envelope(tmp_path: Path) -> None:
    record = load("nvd_cve.json")
    assert writer(tmp_path).write(Source.NVD, record) is True

    path = tmp_path / "nvd" / "2026-11-11.jsonl"
    lines = read_lines(path)
    assert lines == [
        {
            "sha256": content_hash(Source.NVD, record),
            "fetched_at": "2026-11-26T12:30:45Z",
            "record": record,
        }
    ]
    assert list(json.loads(path.read_text(encoding="utf-8").splitlines()[0])) == [
        "sha256",
        "fetched_at",
        "record",
    ]


def test_write_keeps_raw_record_including_volatile_keys(tmp_path: Path) -> None:
    record = load("ghsa_advisory.json")
    writer(tmp_path).write(Source.GHSA, record)
    [line] = read_lines(tmp_path / "ghsa" / "2026-11-25.jsonl")
    assert line["record"] == record
    assert "epss" in line["record"]


def test_write_stores_utf8_not_ascii_escapes(tmp_path: Path) -> None:
    record = nvd_record()
    record["cve"]["descriptions"] = [{"lang": "es", "value": "inyección de código"}]
    writer(tmp_path).write(Source.NVD, record)
    raw = (tmp_path / "nvd" / "2026-11-11.jsonl").read_bytes()
    assert "inyección".encode() in raw
    assert b"\\u00f3" not in raw


def test_write_skips_duplicate_content(tmp_path: Path) -> None:
    bronze = writer(tmp_path)
    assert bronze.write(Source.NVD, nvd_record()) is True
    assert bronze.write(Source.NVD, nvd_record()) is False
    assert len(read_lines(tmp_path / "nvd" / "2026-11-11.jsonl")) == 1


def test_write_skips_ghsa_record_that_only_changed_epss(tmp_path: Path) -> None:
    bronze = writer(tmp_path)
    assert bronze.write(Source.GHSA, ghsa_record(epss={"percentage": 0.1})) is True
    assert bronze.write(Source.GHSA, ghsa_record(epss={"percentage": 0.2})) is False
    assert len(read_lines(tmp_path / "ghsa" / "2026-11-25.jsonl")) == 1


def test_write_appends_new_version_of_same_record(tmp_path: Path) -> None:
    bronze = writer(tmp_path)
    first = nvd_record()
    second = nvd_record()
    second["cve"]["lastModified"] = "2026-11-20T00:00:00.000"
    assert bronze.write(Source.NVD, first) is True
    assert bronze.write(Source.NVD, second) is True
    lines = read_lines(tmp_path / "nvd" / "2026-11-11.jsonl")
    assert [line["record"] for line in lines] == [first, second]


def test_write_partitions_by_source_and_day(tmp_path: Path) -> None:
    bronze = writer(tmp_path)
    bronze.write(Source.NVD, nvd_record("CVE-2026-00001", "2026-11-11T01:00:00.000"))
    bronze.write(Source.NVD, nvd_record("CVE-2026-00002", "2026-11-12T01:00:00.000"))
    bronze.write(Source.GHSA, ghsa_record(published="2026-11-11T01:00:00Z"))
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.jsonl")) == [
        "ghsa/2026-11-11.jsonl",
        "nvd/2026-11-11.jsonl",
        "nvd/2026-11-12.jsonl",
    ]


def test_same_content_in_other_source_is_not_a_duplicate(tmp_path: Path) -> None:
    record = {
        "cve": {"id": "X", "published": "2026-11-11T00:00:00"},
        "published_at": "2026-11-11T00:00:00Z",
    }
    bronze = writer(tmp_path)
    assert bronze.write(Source.NVD, record) is True
    assert bronze.write(Source.GHSA, record) is True


def test_new_writer_loads_existing_hashes_from_every_day_file(tmp_path: Path) -> None:
    first = writer(tmp_path)
    first.write(Source.NVD, nvd_record("CVE-2026-00001", "2026-11-11T01:00:00.000"))
    first.write(Source.NVD, nvd_record("CVE-2026-00002", "2026-11-12T01:00:00.000"))

    second = writer(tmp_path)
    assert (
        second.write(Source.NVD, nvd_record("CVE-2026-00001", "2026-11-11T01:00:00.000")) is False
    )
    assert (
        second.write(Source.NVD, nvd_record("CVE-2026-00002", "2026-11-12T01:00:00.000")) is False
    )
    assert second.write(Source.NVD, nvd_record("CVE-2026-00003", "2026-11-12T01:00:00.000")) is True
    assert len(read_lines(tmp_path / "nvd" / "2026-11-11.jsonl")) == 1
    assert len(read_lines(tmp_path / "nvd" / "2026-11-12.jsonl")) == 2


def test_writer_does_not_create_files_until_written(tmp_path: Path) -> None:
    root = tmp_path / "bronze"
    BronzeWriter(root)
    assert not root.exists()


def test_corrupt_existing_line_raises_with_path_and_line(tmp_path: Path) -> None:
    writer(tmp_path).write(Source.NVD, nvd_record())
    path = tmp_path / "nvd" / "2026-11-11.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sha256": "abc", "record": {\n')

    with pytest.raises(ValueError) as excinfo:
        writer(tmp_path).write(Source.NVD, nvd_record("CVE-2026-00009"))
    message = str(excinfo.value)
    assert "2026-11-11.jsonl" in message
    assert ":2" in message or "line 2" in message


def test_existing_line_without_sha256_raises(tmp_path: Path) -> None:
    path = tmp_path / "ghsa" / "2026-11-25.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"record": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2026-11-25.jsonl"):
        writer(tmp_path).write(Source.GHSA, ghsa_record())


def test_record_without_published_timestamp_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises((KeyError, ValueError)):
        writer(tmp_path).write(Source.GHSA, {"ghsa_id": "GHSA-xxxx-xxxx-xxxx"})
    assert not list(tmp_path.rglob("*.jsonl"))


def test_default_clock_is_utc_now(tmp_path: Path) -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    BronzeWriter(tmp_path).write(Source.NVD, nvd_record())
    after = datetime.now(UTC)
    [line] = read_lines(tmp_path / "nvd" / "2026-11-11.jsonl")
    stamp = line["fetched_at"]
    assert stamp.endswith("Z") and len(stamp) == 20
    parsed = datetime.fromisoformat(stamp)
    assert before <= parsed <= after


def test_blank_lines_in_existing_file_are_skipped(tmp_path: Path) -> None:
    writer(tmp_path).write(Source.NVD, nvd_record())
    path = tmp_path / "nvd" / "2026-11-11.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    second = writer(tmp_path)
    assert second.write(Source.NVD, nvd_record()) is False
    assert second.write(Source.NVD, nvd_record("CVE-2026-00002")) is True


@pytest.mark.parametrize("bad", ['{"sha256": 5}', '{"sha256": "abc"}', "[1, 2]", '"text"'])
def test_existing_line_with_invalid_sha256_raises(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "nvd" / "2026-11-11.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2026-11-11.jsonl"):
        writer(tmp_path).write(Source.NVD, nvd_record())


@pytest.mark.parametrize(
    ("source", "record"),
    [
        (Source.NVD, {}),
        (Source.NVD, {"cve": {}}),
        (Source.NVD, {"cve": {"published": None}}),
        (Source.GHSA, {"published_at": None}),
        (Source.GHSA, {"published_at": "not a date"}),
    ],
)
def test_bad_published_value_raises_value_error(source: Source, record: dict) -> None:
    with pytest.raises(ValueError):
        published_day(source, record)


def test_non_finite_number_is_rejected_and_nothing_written(tmp_path: Path) -> None:
    record = nvd_record()
    record["cve"]["score"] = float("nan")
    with pytest.raises(ValueError):
        writer(tmp_path).write(Source.NVD, record)
    assert not list(tmp_path.rglob("*.jsonl"))


def test_each_line_is_written_with_a_single_write_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original_open = Path.open

    class Spy:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self.handle.__exit__(*exc)

        def __iter__(self):
            return iter(self.handle)

        def write(self, text: str) -> int:
            calls.append(text)
            return self.handle.write(text)

    def spy_open(self, *args, **kwargs):
        return Spy(original_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", spy_open)
    record = nvd_record()
    record["cve"]["descriptions"] = [{"lang": "en", "value": "x" * 20000}]
    writer(tmp_path).write(Source.NVD, record)
    assert len(calls) == 1
    assert calls[0].endswith("\n") and calls[0].count("\n") == 1
