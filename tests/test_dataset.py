import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from semtriage import dataset
from semtriage.bronze import BronzeWriter
from semtriage.dataset import Dataset, build_dataset, main, read_events, write_dataset
from semtriage.models import Event, Source
from semtriage.sanitize import find_identifiers

FETCHED = "2026-11-26T12:00:00Z"


def nvd(
    cve_id: str,
    published: str,
    *,
    modified: str | None = None,
    text: str = "Buffer overflow in the request parser.",
    status: str = "Analyzed",
) -> dict:
    return {
        "cve": {
            "id": cve_id,
            "sourceIdentifier": "cna@example.org",
            "published": published,
            "lastModified": modified or published,
            "vulnStatus": status,
            "descriptions": [{"lang": "en", "value": text}],
        }
    }


def ghsa(
    ghsa_id: str,
    published: str,
    cves: tuple[str, ...] = (),
    *,
    updated: str | None = None,
    text: str = "Overflow in the request parser",
    withdrawn: str | None = None,
) -> dict:
    identifiers = [{"type": "GHSA", "value": ghsa_id}]
    identifiers += [{"type": "CVE", "value": cve} for cve in cves]
    return {
        "ghsa_id": ghsa_id,
        "cve_id": cves[0] if cves else None,
        "identifiers": identifiers,
        "summary": text,
        "description": "",
        "published_at": published,
        "updated_at": updated or published,
        "withdrawn_at": withdrawn,
        "severity": "high",
        "cwes": [],
        "vulnerabilities": [],
    }


def put(root: Path, source: str, day: str, *items: tuple[dict, str]) -> Path:
    """Append (record, fetched_at) pairs as bronze envelope lines."""
    path = root / source / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record, fetched_at in items:
            envelope = {"sha256": "0" * 64, "fetched_at": fetched_at, "record": record}
            handle.write(json.dumps(envelope) + "\n")
    return path


def put_raw(root: Path, source: str, day: str, *lines: str) -> Path:
    path = root / source / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    return path


def texts(result: Dataset) -> dict[str, str]:
    return {event.event_id: event.text for event in result.events}


# Ordering and versions


def test_events_are_ordered_by_published_then_event_id(tmp_path: Path) -> None:
    put(tmp_path, "nvd", "2026-01-03", (nvd("CVE-2026-0003", "2026-01-03T00:00:00.000"), FETCHED))
    put(tmp_path, "nvd", "2026-01-01", (nvd("CVE-2026-0001", "2026-01-01T06:00:00.000"), FETCHED))
    put(
        tmp_path,
        "ghsa",
        "2026-01-01",
        (ghsa("GHSA-bbbb-bbbb-bbbb", "2026-01-01T06:00:00Z"), FETCHED),
        (ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T01:00:00Z"), FETCHED),
    )

    result = build_dataset(tmp_path)

    assert [event.event_id for event in result.events] == [
        "ghsa:GHSA-aaaa-aaaa-aaaa",
        "ghsa:GHSA-bbbb-bbbb-bbbb",
        "nvd:CVE-2026-0001",
        "nvd:CVE-2026-0003",
    ]
    assert all(isinstance(event, Event) for event in result.events)


def test_latest_version_is_chosen_by_modified_not_by_line_order(tmp_path: Path) -> None:
    newer = nvd("CVE-2026-0001", "2026-01-01T00:00:00", modified="2026-02-01T00:00:00", text="New")
    older = nvd("CVE-2026-0001", "2026-01-01T00:00:00", modified="2026-01-15T00:00:00", text="Old")
    put(tmp_path, "nvd", "2026-01-01", (newer, FETCHED), (older, "2026-11-27T00:00:00Z"))

    result = build_dataset(tmp_path)

    assert texts(result) == {"nvd:CVE-2026-0001": "New"}


def test_modified_is_compared_in_utc(tmp_path: Path) -> None:
    # 2026-01-02T01:00+05:00 is 2026-01-01T20:00Z, earlier than 2026-01-01T21:00Z.
    first = ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z", updated="2026-01-01T21:00:00Z")
    second = ghsa(
        "GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z", updated="2026-01-02T01:00:00+05:00"
    )
    put(
        tmp_path,
        "ghsa",
        "2026-01-01",
        ({**first, "summary": "Later in UTC"}, FETCHED),
        ({**second, "summary": "Earlier in UTC"}, FETCHED),
    )

    result = build_dataset(tmp_path)

    assert texts(result) == {"ghsa:GHSA-aaaa-aaaa-aaaa": "Later in UTC"}


def test_fetched_at_breaks_a_modified_tie(tmp_path: Path) -> None:
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text="Fetched later"), "2026-11-27T00:00:00Z"),
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text="Fetched earlier"), FETCHED),
    )

    result = build_dataset(tmp_path)

    assert texts(result) == {"nvd:CVE-2026-0001": "Fetched later"}


def test_last_line_breaks_a_full_tie(tmp_path: Path) -> None:
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text="First line"), FETCHED),
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text="Second line"), FETCHED),
    )

    result = build_dataset(tmp_path)

    assert texts(result) == {"nvd:CVE-2026-0001": "Second line"}


def test_versions_in_different_files_are_merged(tmp_path: Path) -> None:
    old = nvd("CVE-2026-0001", "2026-01-01T00:00:00", modified="2026-01-02T00:00:00", text="Old")
    new = nvd("CVE-2026-0001", "2026-01-02T00:00:00", modified="2026-01-03T00:00:00", text="New")
    put(tmp_path, "nvd", "2026-01-02", (new, FETCHED))
    put(tmp_path, "nvd", "2026-01-01", (old, FETCHED))

    result = build_dataset(tmp_path)

    assert texts(result) == {"nvd:CVE-2026-0001": "New"}
    assert result.stats["sources"]["nvd"] == {"lines": 2, "records": 1, "events": 1, "dropped": 0}


def test_later_rejected_version_drops_the_event(tmp_path: Path) -> None:
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00"), FETCHED),
        (
            nvd(
                "CVE-2026-0001",
                "2026-01-01T00:00:00",
                modified="2026-01-05T00:00:00",
                status="Rejected",
            ),
            FETCHED,
        ),
    )

    result = build_dataset(tmp_path)

    assert result.events == []
    assert result.stats["sources"]["nvd"] == {"lines": 2, "records": 1, "events": 0, "dropped": 1}


def test_withdrawn_ghsa_is_dropped(tmp_path: Path) -> None:
    withdrawn = ghsa(
        "GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z", withdrawn="2026-01-02T00:00:00Z"
    )
    put(tmp_path, "ghsa", "2026-01-01", (withdrawn, FETCHED))

    result = build_dataset(tmp_path)

    assert result.events == []
    assert result.stats["sources"]["ghsa"] == {"lines": 1, "records": 1, "events": 0, "dropped": 1}


# Sanitization


def test_event_text_has_no_identifiers_or_urls(tmp_path: Path) -> None:
    leaky = (
        "Overflow tracked as CVE-2026-0001 (see GHSA-abcd-efgh-ijkl) "
        "and https://example.com/advisories/CVE-2026-0001 for details."
    )
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text=leaky), FETCHED),
    )
    put(
        tmp_path,
        "ghsa",
        "2026-01-01",
        (
            ghsa("GHSA-abcd-efgh-ijkl", "2026-01-01T00:00:00Z", ("CVE-2026-0001",), text=leaky),
            FETCHED,
        ),
    )

    result = build_dataset(tmp_path)

    assert len(result.events) == 2
    for event in result.events:
        assert find_identifiers(event.text) == set()
        assert "http" not in event.text
        assert "Overflow tracked as" in event.text
        assert event.cve_ids == ("CVE-2026-0001",)


def test_leaked_identifier_after_normalization_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(tmp_path, "nvd", "2026-01-01", (nvd("CVE-2026-0001", "2026-01-01T00:00:00"), FETCHED))
    real = dataset.normalize_nvd

    def leaky(item: dict) -> Event | None:
        event = real(item)
        assert event is not None
        return event.model_copy(update={"text": "Overflow, see CVE-2026-0001."})

    monkeypatch.setattr(dataset, "normalize_nvd", leaky)

    with pytest.raises(ValueError, match="nvd:CVE-2026-0001"):
        build_dataset(tmp_path)


# Statistics


def _stats_bronze(root: Path) -> None:
    writer = BronzeWriter(root, clock=lambda: datetime(2026, 11, 26, 12, tzinfo=UTC))
    for record in (
        nvd("CVE-2026-1001", "2026-01-01T00:00:00.000"),
        nvd("CVE-2026-1001", "2026-01-01T00:00:00.000", modified="2026-01-04T00:00:00.000"),
        nvd("CVE-2026-1002", "2026-01-05T12:00:00.000"),
        nvd("CVE-2026-1003", "2026-01-06T00:00:00.000"),
        nvd("CVE-2026-1005", "2026-01-08T00:00:00.000", status="Rejected"),
    ):
        assert writer.write(Source.NVD, record)
    for record in (
        ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-02T00:00:00Z", ("CVE-2026-1001",)),
        ghsa("GHSA-bbbb-bbbb-bbbb", "2026-01-03T00:00:00Z", ("CVE-2026-1001",)),
        ghsa("GHSA-cccc-cccc-cccc", "2026-01-05T09:30:00Z", ("CVE-2026-1002",)),
        ghsa("GHSA-dddd-dddd-dddd", "2026-01-06T00:00:00Z", ("CVE-2026-1003", "CVE-2026-1004")),
        ghsa("GHSA-eeee-eeee-eeee", "2026-01-07T00:00:00Z"),
    ):
        assert writer.write(Source.GHSA, record)


def test_stats_from_bronze_written_by_the_bronze_writer(tmp_path: Path) -> None:
    _stats_bronze(tmp_path)

    result = build_dataset(tmp_path)

    assert result.stats == {
        "events": 8,
        "sources": {
            "nvd": {"lines": 5, "records": 4, "events": 3, "dropped": 1},
            "ghsa": {"lines": 5, "records": 5, "events": 5, "dropped": 0},
        },
        "clusters": {
            "cves": 4,
            "size_histogram": {"1": 1, "2": 2, "3": 1},
            "multi_event_cves": 3,
            "events_without_cve": 1,
            "events_with_multiple_cves": 1,
        },
        "lag_hours": {
            "pairs": 3,
            "ghsa_first": 1,
            "nvd_first": 1,
            "same_time": 1,
            "min": -2.5,
            "median": 0.0,
            "max": 24.0,
        },
    }


def test_lag_stats_are_null_without_pairs(tmp_path: Path) -> None:
    put(tmp_path, "nvd", "2026-01-01", (nvd("CVE-2026-0001", "2026-01-01T00:00:00"), FETCHED))
    put(
        tmp_path,
        "ghsa",
        "2026-01-01",
        (ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z", ("CVE-2026-0002",)), FETCHED),
    )

    result = build_dataset(tmp_path)

    assert result.stats["lag_hours"] == {
        "pairs": 0,
        "ghsa_first": 0,
        "nvd_first": 0,
        "same_time": 0,
        "min": None,
        "median": None,
        "max": None,
    }


def test_median_of_an_even_number_of_lags(tmp_path: Path) -> None:
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00"), FETCHED),
        (nvd("CVE-2026-0002", "2026-01-01T00:00:00"), FETCHED),
    )
    put(
        tmp_path,
        "ghsa",
        "2026-01-01",
        (ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T01:00:00Z", ("CVE-2026-0001",)), FETCHED),
        (ghsa("GHSA-bbbb-bbbb-bbbb", "2026-01-01T04:00:00Z", ("CVE-2026-0002",)), FETCHED),
    )

    result = build_dataset(tmp_path)

    lag = result.stats["lag_hours"]
    assert (lag["pairs"], lag["min"], lag["median"], lag["max"]) == (2, 1.0, 2.5, 4.0)


# Reading bronze


def test_missing_source_directory_means_no_records(tmp_path: Path) -> None:
    put(tmp_path, "nvd", "2026-01-01", (nvd("CVE-2026-0001", "2026-01-01T00:00:00"), FETCHED))

    result = build_dataset(tmp_path)

    assert len(result.events) == 1
    assert result.stats["sources"]["ghsa"] == {"lines": 0, "records": 0, "events": 0, "dropped": 0}


def test_missing_bronze_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        build_dataset(tmp_path / "missing")


def test_blank_lines_and_other_files_are_ignored(tmp_path: Path) -> None:
    line = json.dumps(
        {
            "sha256": "0" * 64,
            "fetched_at": FETCHED,
            "record": nvd("CVE-2026-0001", "2026-01-01T00:00:00"),
        }
    )
    put_raw(tmp_path, "nvd", "2026-01-01", "", line, "   ")
    (tmp_path / "nvd" / "notes.txt").write_text("not bronze\n", encoding="utf-8")

    result = build_dataset(tmp_path)

    assert len(result.events) == 1
    assert result.stats["sources"]["nvd"]["lines"] == 1


GOOD_NVD = json.dumps(
    {
        "sha256": "0" * 64,
        "fetched_at": FETCHED,
        "record": nvd("CVE-2026-0001", "2026-01-01T00:00:00"),
    }
)


def _without(record: dict, *path: str) -> dict:
    copy = json.loads(json.dumps(record))
    target = copy
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return copy


def _envelope(record: object, fetched_at: object = FETCHED) -> str:
    return json.dumps({"sha256": "0" * 64, "fetched_at": fetched_at, "record": record})


BAD_NVD_LINES = {
    "not_json": '{"sha256": "0',
    "not_object": "[1, 2]",
    "record_missing": json.dumps({"sha256": "0" * 64, "fetched_at": FETCHED}),
    "record_not_object": _envelope([1]),
    "fetched_at_missing": json.dumps(
        {"sha256": "0" * 64, "record": nvd("CVE-2026-0002", "2026-01-01T00:00:00")}
    ),
    "fetched_at_not_string": _envelope(nvd("CVE-2026-0002", "2026-01-01T00:00:00"), 5),
    "cve_missing": _envelope({"id": "CVE-2026-0002"}),
    "id_missing": _envelope(_without(nvd("CVE-2026-0002", "2026-01-01T00:00:00"), "cve", "id")),
    "modified_missing": _envelope(
        _without(nvd("CVE-2026-0002", "2026-01-01T00:00:00"), "cve", "lastModified")
    ),
    "modified_invalid": _envelope(
        nvd("CVE-2026-0002", "2026-01-01T00:00:00", modified="yesterday")
    ),
    "id_invalid": _envelope(nvd("not-a-cve", "2026-01-01T00:00:00")),
    "normalizer_fails": _envelope(
        _without(nvd("CVE-2026-0002", "2026-01-01T00:00:00"), "cve", "published")
    ),
}


@pytest.mark.parametrize("bad", BAD_NVD_LINES.values(), ids=BAD_NVD_LINES.keys())
def test_bad_nvd_line_raises_with_path_and_line(tmp_path: Path, bad: str) -> None:
    path = put_raw(tmp_path, "nvd", "2026-01-01", GOOD_NVD, bad)

    with pytest.raises(ValueError, match=re.escape(f"{path}:2")):
        build_dataset(tmp_path)


BAD_GHSA_RECORDS = {
    "id_missing": _without(ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z"), "ghsa_id"),
    "id_invalid": ghsa("not-a-ghsa", "2026-01-01T00:00:00Z"),
    "modified_missing": _without(ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z"), "updated_at"),
    "modified_not_string": {
        **ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z"),
        "updated_at": 12,
    },
    "normalizer_fails": _without(
        ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T00:00:00Z"), "published_at"
    ),
}


@pytest.mark.parametrize("record", BAD_GHSA_RECORDS.values(), ids=BAD_GHSA_RECORDS.keys())
def test_bad_ghsa_record_raises_with_path_and_line(tmp_path: Path, record: dict) -> None:
    path = put_raw(tmp_path, "ghsa", "2026-01-01", _envelope(record))

    with pytest.raises(ValueError, match=re.escape(f"{path}:1")):
        build_dataset(tmp_path)


# Output


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    _stats_bronze(tmp_path / "bronze")
    result = build_dataset(tmp_path / "bronze")
    out = tmp_path / "out" / "nested"

    write_dataset(result, out)

    events_text = (out / "events.jsonl").read_text(encoding="utf-8")
    assert events_text == "".join(event.model_dump_json() + "\n" for event in result.events)
    stats_text = (out / "stats.json").read_text(encoding="utf-8")
    assert stats_text == json.dumps(result.stats, indent=2, sort_keys=True) + "\n"
    assert read_events(out / "events.jsonl") == result.events


def test_output_is_deterministic_and_overwritten(tmp_path: Path) -> None:
    _stats_bronze(tmp_path / "bronze")
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    out_a.mkdir()
    (out_a / "events.jsonl").write_text("stale\n" * 50, encoding="utf-8")
    (out_a / "stats.json").write_text("stale\n" * 50, encoding="utf-8")

    write_dataset(build_dataset(tmp_path / "bronze"), out_a)
    write_dataset(build_dataset(tmp_path / "bronze"), out_b)

    for name in ("events.jsonl", "stats.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
        assert b"stale" not in (out_a / name).read_bytes()


def test_read_events_skips_blank_lines(tmp_path: Path) -> None:
    _stats_bronze(tmp_path / "bronze")
    result = build_dataset(tmp_path / "bronze")
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(event.model_dump_json() for event in result.events) + "\n\n",
        encoding="utf-8",
    )

    assert read_events(path) == result.events


# CLI


def test_main_builds_dataset_and_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stats_bronze(tmp_path / "bronze")
    out = tmp_path / "dataset"

    code = main(["--bronze-dir", str(tmp_path / "bronze"), "--out-dir", str(out)])

    assert code == 0
    assert len(read_events(out / "events.jsonl")) == 8
    assert json.loads((out / "stats.json").read_text(encoding="utf-8"))["events"] == 8
    stdout = capsys.readouterr().out
    assert stdout.splitlines() == [
        "nvd: lines=5 records=4 events=3 dropped=1",
        "ghsa: lines=5 records=5 events=5 dropped=0",
        "total: 8 events",
    ]


def test_main_uses_default_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stats_bronze(tmp_path / "data" / "bronze")
    monkeypatch.chdir(tmp_path)

    assert main([]) == 0
    assert (tmp_path / "data" / "dataset" / "events.jsonl").is_file()
    assert (tmp_path / "data" / "dataset" / "stats.json").is_file()


def test_main_reports_bad_bronze_and_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = put_raw(tmp_path / "bronze", "nvd", "2026-01-01", "not json")
    out = tmp_path / "dataset"

    code = main(["--bronze-dir", str(tmp_path / "bronze"), "--out-dir", str(out)])

    assert code == 1
    assert f"{path}:1" in capsys.readouterr().err
    assert not out.exists()


def test_lag_direction_counts_which_source_published_first(tmp_path: Path) -> None:
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00"), FETCHED),
        (nvd("CVE-2026-0002", "2026-01-01T00:00:00"), FETCHED),
        (nvd("CVE-2026-0003", "2026-01-01T12:00:00"), FETCHED),
    )
    put(
        tmp_path,
        "ghsa",
        "2026-01-01",
        (ghsa("GHSA-aaaa-aaaa-aaaa", "2026-01-01T06:00:00Z", ("CVE-2026-0001",)), FETCHED),
        (ghsa("GHSA-bbbb-bbbb-bbbb", "2026-01-01T03:00:00Z", ("CVE-2026-0002",)), FETCHED),
        (ghsa("GHSA-cccc-cccc-cccc", "2026-01-01T00:00:00Z", ("CVE-2026-0003",)), FETCHED),
    )

    lag = build_dataset(tmp_path).stats["lag_hours"]

    assert (lag["nvd_first"], lag["ghsa_first"], lag["same_time"]) == (2, 1, 0)
    assert (lag["min"], lag["median"], lag["max"]) == (-12.0, 3.0, 6.0)


def test_last_file_breaks_a_full_tie_across_files(tmp_path: Path) -> None:
    put(
        tmp_path,
        "nvd",
        "2026-01-01",
        (nvd("CVE-2026-0009", "2026-01-01T00:00:00"), FETCHED),
        (nvd("CVE-2026-0008", "2026-01-01T00:00:00"), FETCHED),
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text="Earlier file"), FETCHED),
    )
    put(
        tmp_path,
        "nvd",
        "2026-01-02",
        (nvd("CVE-2026-0001", "2026-01-01T00:00:00", text="Later file"), FETCHED),
    )

    result = build_dataset(tmp_path)

    assert texts(result)["nvd:CVE-2026-0001"] == "Later file"


def test_invalid_utf8_line_raises_with_path_and_line(tmp_path: Path) -> None:
    path = put_raw(tmp_path, "nvd", "2026-01-01", GOOD_NVD)
    with path.open("ab") as handle:
        handle.write(b'{"record": "\xff"}\n')

    with pytest.raises(ValueError, match=re.escape(f"{path}:2")):
        build_dataset(tmp_path)


def test_main_reports_unreadable_bronze_and_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bronze" / "nvd" / "2026-01-01.jsonl"
    bad.mkdir(parents=True)

    code = main(["--bronze-dir", str(tmp_path / "bronze"), "--out-dir", str(tmp_path / "out")])

    assert code == 1
    assert str(bad) in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
