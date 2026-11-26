import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from semtriage.backfill import BackfillStats, main, run_backfill
from semtriage.bronze import BronzeWriter
from semtriage.clients import GHSA_URL, NVD_URL
from semtriage.models import Source

START = datetime(2026, 11, 20, tzinfo=UTC)
END = datetime(2026, 11, 22, tzinfo=UTC)


def nvd_item(cve_id: str, published: str = "2026-11-20T10:00:00.000") -> dict:
    return {"cve": {"id": cve_id, "published": published}}


def ghsa_item(ghsa_id: str, published: str = "2026-11-21T10:00:00Z") -> dict:
    return {"ghsa_id": ghsa_id, "published_at": published, "epss": {"percentage": 0.1}}


class FakeNvd:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.calls: list[tuple] = []

    def iter_cves(self, start, end, *, field="modified"):
        self.calls.append((start, end, field))
        yield from self.items


class FakeGhsa:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.calls: list[tuple] = []

    def iter_advisories(self, start, end, *, field="updated"):
        self.calls.append((start, end, field))
        yield from self.items


def all_lines(root: Path) -> list[dict]:
    lines = []
    for path in sorted(root.rglob("*.jsonl")):
        lines.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return lines


# run_backfill


def test_run_backfill_queries_by_published_and_writes_both_sources(tmp_path: Path) -> None:
    nvd = FakeNvd([nvd_item("CVE-2026-00001"), nvd_item("CVE-2026-00002")])
    ghsa = FakeGhsa([ghsa_item("GHSA-aaaa-bbbb-cccc")])

    stats = run_backfill(BronzeWriter(tmp_path), START, END, nvd=nvd, ghsa=ghsa)

    assert nvd.calls == [(START, END, "published")]
    assert ghsa.calls == [(START, END, "published")]
    assert stats == {
        Source.NVD: BackfillStats(fetched=2, written=2, duplicates=0),
        Source.GHSA: BackfillStats(fetched=1, written=1, duplicates=0),
    }
    assert (tmp_path / "nvd" / "2026-11-20.jsonl").exists()
    assert (tmp_path / "ghsa" / "2026-11-21.jsonl").exists()


def test_run_backfill_counts_duplicates_within_and_across_runs(tmp_path: Path) -> None:
    items = [nvd_item("CVE-2026-00001"), nvd_item("CVE-2026-00001")]
    first = run_backfill(BronzeWriter(tmp_path), START, END, nvd=FakeNvd(items))
    assert first == {Source.NVD: BackfillStats(fetched=2, written=1, duplicates=1)}

    second = run_backfill(BronzeWriter(tmp_path), START, END, nvd=FakeNvd(items))
    assert second == {Source.NVD: BackfillStats(fetched=2, written=0, duplicates=2)}
    assert len(all_lines(tmp_path)) == 1


def test_run_backfill_skips_sources_without_client(tmp_path: Path) -> None:
    ghsa = FakeGhsa([ghsa_item("GHSA-aaaa-bbbb-cccc")])
    stats = run_backfill(BronzeWriter(tmp_path), START, END, ghsa=ghsa)
    assert list(stats) == [Source.GHSA]
    assert not (tmp_path / "nvd").exists()


def test_backfill_stats_defaults_to_zero() -> None:
    assert BackfillStats() == BackfillStats(fetched=0, written=0, duplicates=0)


# main


def api_handler(nvd_items: list[dict], ghsa_items: list[dict], seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url.copy_with(query=None))
        if url == NVD_URL:
            body = {
                "resultsPerPage": len(nvd_items),
                "startIndex": 0,
                "totalResults": len(nvd_items),
                "vulnerabilities": nvd_items,
            }
            return httpx.Response(200, json=body)
        if url == GHSA_URL:
            return httpx.Response(200, json=ghsa_items)
        return httpx.Response(404)

    return handler


@pytest.fixture
def no_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVD_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def run_main(args: list[str], handler) -> int:
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        return main(args, http=http)


def test_main_backfills_inclusive_day_range(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[httpx.Request] = []
    handler = api_handler([nvd_item("CVE-2026-00001")], [ghsa_item("GHSA-aaaa-bbbb-cccc")], seen)
    args = ["--start", "2026-11-20", "--end", "2026-11-21", "--data-dir", str(tmp_path)]

    assert run_main(args, handler) == 0

    nvd_request = next(r for r in seen if r.url.host == "services.nvd.nist.gov")
    assert nvd_request.url.params["pubStartDate"] == "2026-11-20T00:00:00.000"
    assert nvd_request.url.params["pubEndDate"] == "2026-11-22T00:00:00.000"
    assert "lastModStartDate" not in nvd_request.url.params
    ghsa_request = next(r for r in seen if r.url.host == "api.github.com")
    assert ghsa_request.url.params["published"] == "2026-11-20T00:00:00Z..2026-11-22T00:00:00Z"
    assert "updated" not in ghsa_request.url.params

    assert (tmp_path / "nvd" / "2026-11-20.jsonl").exists()
    assert (tmp_path / "ghsa" / "2026-11-21.jsonl").exists()
    out = capsys.readouterr().out
    assert "nvd: fetched=1 written=1 duplicates=0" in out
    assert "ghsa: fetched=1 written=1 duplicates=0" in out


def test_main_rerun_writes_nothing_new(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[httpx.Request] = []
    handler = api_handler([nvd_item("CVE-2026-00001")], [ghsa_item("GHSA-aaaa-bbbb-cccc")], seen)
    args = ["--start", "2026-11-20", "--end", "2026-11-21", "--data-dir", str(tmp_path)]
    assert run_main(args, handler) == 0
    capsys.readouterr()

    assert run_main(args, handler) == 0
    out = capsys.readouterr().out
    assert "nvd: fetched=1 written=0 duplicates=1" in out
    assert "ghsa: fetched=1 written=0 duplicates=1" in out
    assert len(all_lines(tmp_path)) == 2


def test_main_single_source(tmp_path: Path, no_tokens: None) -> None:
    seen: list[httpx.Request] = []
    handler = api_handler([nvd_item("CVE-2026-00001")], [ghsa_item("GHSA-aaaa-bbbb-cccc")], seen)
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--source", "ghsa"]
    assert run_main([*args, "--data-dir", str(tmp_path)], handler) == 0
    assert {r.url.host for r in seen} == {"api.github.com"}
    assert not (tmp_path / "nvd").exists()


def test_main_reads_tokens_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NVD_API_KEY", "test-nvd-key")
    monkeypatch.setenv("GITHUB_TOKEN", "test-gh-token")
    seen: list[httpx.Request] = []
    handler = api_handler([], [], seen)
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--data-dir", str(tmp_path)]
    assert run_main(args, handler) == 0
    nvd_request = next(r for r in seen if r.url.host == "services.nvd.nist.gov")
    ghsa_request = next(r for r in seen if r.url.host == "api.github.com")
    assert nvd_request.headers["apiKey"] == "test-nvd-key"
    assert ghsa_request.headers["Authorization"] == "Bearer test-gh-token"


def test_main_without_tokens_sends_no_credentials(tmp_path: Path, no_tokens: None) -> None:
    seen: list[httpx.Request] = []
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--data-dir", str(tmp_path)]
    assert run_main(args, api_handler([], [], seen)) == 0
    assert seen
    for request in seen:
        assert "apiKey" not in request.headers
        assert "Authorization" not in request.headers


def test_main_empty_token_is_treated_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NVD_API_KEY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    seen: list[httpx.Request] = []
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--data-dir", str(tmp_path)]
    assert run_main(args, api_handler([], [], seen)) == 0
    for request in seen:
        assert "apiKey" not in request.headers
        assert "Authorization" not in request.headers


@pytest.mark.parametrize(
    "args",
    [
        ["--start", "2026-11-21", "--end", "2026-11-20"],
        ["--start", "2026-11-20"],
        ["--start", "20-11-2026", "--end", "2026-11-21"],
        ["--start", "2026-11-20", "--end", "2026-11-21", "--source", "osv"],
    ],
)
def test_main_rejects_bad_arguments(tmp_path: Path, no_tokens: None, args: list[str]) -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(SystemExit) as excinfo:
        run_main([*args, "--data-dir", str(tmp_path)], api_handler([], [], seen))
    assert excinfo.value.code == 2
    assert seen == []
    assert not list(tmp_path.rglob("*.jsonl"))


def test_main_default_data_dir_is_data_bronze(
    tmp_path: Path, no_tokens: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seen: list[httpx.Request] = []
    handler = api_handler([nvd_item("CVE-2026-00001")], [], seen)
    assert run_main(["--start", "2026-11-20", "--end", "2026-11-20"], handler) == 0
    assert (tmp_path / "data" / "bronze" / "nvd" / "2026-11-20.jsonl").exists()


def test_main_http_failure_keeps_finished_source_and_returns_1(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "services.nvd.nist.gov":
            return api_handler([nvd_item("CVE-2026-00001")], [], [])(request)
        return httpx.Response(404)

    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--data-dir", str(tmp_path)]
    assert run_main(args, handler) == 1
    captured = capsys.readouterr()
    assert "nvd: fetched=1 written=1 duplicates=0\n" in captured.out
    assert "ghsa: fetched=0 written=0 duplicates=0 (partial)" in captured.out
    assert "re-run the same range to resume" in captured.err
    assert "404" in captured.err
    assert (tmp_path / "nvd" / "2026-11-20.jsonl").exists()


def test_main_interrupt_returns_130_with_partial_counts(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--data-dir", str(tmp_path)]
    assert run_main(args, handler) == 130
    captured = capsys.readouterr()
    assert "nvd: fetched=0 written=0 duplicates=0 (partial)" in captured.out
    assert "ghsa" not in captured.out
    assert "re-run the same range to resume" in captured.err


def test_main_prints_progress_every_1000_records(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    items = [nvd_item(f"CVE-2026-{i:05d}") for i in range(2500)]
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--source", "nvd"]
    assert run_main([*args, "--data-dir", str(tmp_path)], api_handler(items, [], [])) == 0
    err = capsys.readouterr().err
    assert "nvd: 1000 records so far" in err
    assert "nvd: 2000 records so far" in err
    assert "nvd: 3000 records so far" not in err


def test_main_warns_about_missing_tokens_on_stderr(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--source", "nvd"]
    assert run_main([*args, "--data-dir", str(tmp_path)], api_handler([], [], [])) == 0
    captured = capsys.readouterr()
    assert "NVD_API_KEY is not set" in captured.err
    assert "GITHUB_TOKEN" not in captured.err
    assert "not set" not in captured.out


def test_main_no_token_warning_when_tokens_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NVD_API_KEY", "test-nvd-key")
    monkeypatch.setenv("GITHUB_TOKEN", "test-gh-token")
    args = ["--start", "2026-11-20", "--end", "2026-11-20", "--data-dir", str(tmp_path)]
    assert run_main(args, api_handler([], [], [])) == 0
    captured = capsys.readouterr()
    assert "not set" not in captured.err
    assert "test-nvd-key" not in captured.out + captured.err
    assert "test-gh-token" not in captured.out + captured.err


def test_main_bad_date_message_names_expected_format(
    tmp_path: Path, no_tokens: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        run_main(["--start", "2026/11/20", "--end", "2026-11-21"], api_handler([], [], []))
    assert "YYYY-MM-DD" in capsys.readouterr().err
