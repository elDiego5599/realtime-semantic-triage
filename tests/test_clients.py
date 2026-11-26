from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from semtriage.clients import GHSA_URL, NVD_URL, GhsaClient, NvdClient, RateLimiter

START = datetime(2026, 11, 20, tzinfo=UTC)
END = datetime(2026, 11, 21, tzinfo=UTC)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class CountingLimiter:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self) -> None:
        self.calls += 1


def make_http(handler) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(wrapped)), seen


def nvd_page(request: httpx.Request, total: int) -> httpx.Response:
    start = int(request.url.params["startIndex"])
    size = int(request.url.params["resultsPerPage"])
    ids = range(start, min(start + size, total))
    body = {
        "resultsPerPage": len(ids),
        "startIndex": start,
        "totalResults": total,
        "vulnerabilities": [{"cve": {"id": f"CVE-2026-{i:05d}"}} for i in ids],
    }
    return httpx.Response(200, json=body)


def nvd_client(handler, **kwargs) -> tuple[NvdClient, list[httpx.Request]]:
    http, seen = make_http(handler)
    kwargs.setdefault("limiter", CountingLimiter())
    kwargs.setdefault("sleep", lambda _: None)
    return NvdClient(http, **kwargs), seen


def ghsa_client(handler, **kwargs) -> tuple[GhsaClient, list[httpx.Request]]:
    http, seen = make_http(handler)
    kwargs.setdefault("limiter", CountingLimiter())
    kwargs.setdefault("sleep", lambda _: None)
    return GhsaClient(http, **kwargs), seen


# RateLimiter


def test_limiter_allows_burst_then_waits_for_window():
    clock = FakeClock()
    limiter = RateLimiter(3, 30.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    assert clock.sleeps == []
    limiter.acquire()
    assert clock.sleeps == [pytest.approx(30.0)]


def test_limiter_window_slides():
    clock = FakeClock()
    limiter = RateLimiter(2, 10.0, clock=clock, sleep=clock.sleep)
    limiter.acquire()  # t=1000
    clock.now += 6
    limiter.acquire()  # t=1006
    clock.now += 5  # t=1011: first call left the window
    limiter.acquire()
    assert clock.sleeps == []
    limiter.acquire()  # must wait until 1016, when the call at 1006 expires
    assert clock.sleeps == [pytest.approx(5.0)]


def test_limiter_exposes_settings_and_validates():
    limiter = RateLimiter(5, 30.0)
    assert (limiter.max_calls, limiter.window) == (5, 30.0)
    with pytest.raises(ValueError):
        RateLimiter(0, 30.0)
    with pytest.raises(ValueError):
        RateLimiter(5, 0.0)


def test_limiter_sleeps_again_after_early_wake():
    clock = FakeClock()
    woke_early = []

    def short_sleep(seconds: float) -> None:
        if not woke_early:
            woke_early.append(True)
            seconds -= 1.0
        clock.sleep(seconds)

    limiter = RateLimiter(1, 10.0, clock=clock, sleep=short_sleep)
    limiter.acquire()  # t=1000
    limiter.acquire()  # first sleep wakes at 1009, must sleep 1 more second
    assert clock.sleeps == [pytest.approx(9.0), pytest.approx(1.0)]
    assert clock.now == pytest.approx(1010.0)


# NVD


def test_nvd_single_page_request_shape():
    client, seen = nvd_client(lambda r: nvd_page(r, 2))
    items = list(client.iter_cves(START, END))
    assert [i["cve"]["id"] for i in items] == ["CVE-2026-00000", "CVE-2026-00001"]
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url).startswith(NVD_URL + "?")
    assert dict(request.url.params) == {
        "lastModStartDate": "2026-11-20T00:00:00.000",
        "lastModEndDate": "2026-11-21T00:00:00.000",
        "resultsPerPage": "2000",
        "startIndex": "0",
    }
    assert "apiKey" not in request.headers


def test_nvd_paginates_by_start_index():
    client, seen = nvd_client(lambda r: nvd_page(r, 5), page_size=2)
    items = list(client.iter_cves(START, END))
    assert len(items) == 5
    assert [r.url.params["startIndex"] for r in seen] == ["0", "2", "4"]


def test_nvd_empty_result_makes_one_request():
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    assert list(client.iter_cves(START, END)) == []
    assert len(seen) == 1


def test_nvd_published_field_uses_pub_dates():
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    list(client.iter_cves(START, END, field="published"))
    params = seen[0].url.params
    assert params["pubStartDate"] == "2026-11-20T00:00:00.000"
    assert params["pubEndDate"] == "2026-11-21T00:00:00.000"
    assert "lastModStartDate" not in params


def test_nvd_splits_ranges_longer_than_120_days():
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 6, 1, tzinfo=UTC)
    list(client.iter_cves(start, end))
    windows = [(r.url.params["lastModStartDate"], r.url.params["lastModEndDate"]) for r in seen]
    assert windows == [
        ("2026-01-01T00:00:00.000", "2026-05-01T00:00:00.000"),
        ("2026-05-01T00:00:00.000", "2026-06-01T00:00:00.000"),
    ]


def test_nvd_exact_120_day_range_is_one_window():
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    list(client.iter_cves(start, start + timedelta(days=120)))
    assert len(seen) == 1


def test_nvd_splits_windows_in_utc_across_dst():
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    new_york = ZoneInfo("America/New_York")
    start = datetime(2026, 3, 1, tzinfo=new_york)  # EST, 05:00 UTC
    end = datetime(2026, 7, 15, tzinfo=new_york)  # EDT, 04:00 UTC
    list(client.iter_cves(start, end))
    windows = [(r.url.params["lastModStartDate"], r.url.params["lastModEndDate"]) for r in seen]
    assert windows == [
        ("2026-03-01T05:00:00.000", "2026-06-29T05:00:00.000"),
        ("2026-06-29T05:00:00.000", "2026-07-15T04:00:00.000"),
    ]


def test_nvd_converts_aware_datetimes_to_utc():
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    bogota = timezone(timedelta(hours=-5))
    list(client.iter_cves(datetime(2026, 11, 19, 19, 0, tzinfo=bogota), END))
    assert seen[0].url.params["lastModStartDate"] == "2026-11-20T00:00:00.000"


def test_nvd_sends_api_key_and_picks_limiter():
    http, seen = make_http(lambda r: nvd_page(r, 0))
    keyed = NvdClient(http, api_key="test-key")
    anonymous = NvdClient(http)
    assert (keyed.limiter.max_calls, keyed.limiter.window) == (50, 30.0)
    assert (anonymous.limiter.max_calls, anonymous.limiter.window) == (5, 30.0)
    list(keyed.iter_cves(START, END))
    assert seen[0].headers["apiKey"] == "test-key"


@pytest.mark.parametrize(
    "start, end",
    [
        (datetime(2026, 11, 20), END),
        (START, datetime(2026, 11, 21)),
        (END, START),
        (START, START),
    ],
)
def test_nvd_rejects_bad_ranges(start, end):
    client, seen = nvd_client(lambda r: nvd_page(r, 0))
    with pytest.raises(ValueError):
        list(client.iter_cves(start, end))
    assert seen == []


def test_nvd_retries_transient_errors_with_backoff():
    statuses = iter([503, 429])
    sleeps: list[float] = []

    def handler(request):
        status = next(statuses, None)
        return httpx.Response(status) if status else nvd_page(request, 1)

    limiter = CountingLimiter()
    client, seen = nvd_client(handler, limiter=limiter, sleep=sleeps.append, backoff=2.0)
    assert len(list(client.iter_cves(START, END))) == 1
    assert len(seen) == 3
    assert limiter.calls == 3
    assert sleeps == [2.0, 4.0]


def test_nvd_gives_up_after_max_retries():
    client, seen = nvd_client(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        list(client.iter_cves(START, END))
    assert len(seen) == 3


def test_nvd_does_not_retry_client_errors():
    client, seen = nvd_client(lambda r: httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        list(client.iter_cves(START, END))
    assert len(seen) == 1


def test_nvd_retries_forbidden_rate_limit():
    statuses = iter([403])
    sleeps: list[float] = []

    def handler(request):
        status = next(statuses, None)
        return httpx.Response(status) if status else nvd_page(request, 1)

    client, seen = nvd_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_cves(START, END))) == 1
    assert len(seen) == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "retry_after, backoff, expected",
    [("7", 1.0, 7.0), ("1", 2.0, 2.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0, 1.0)],
)
def test_nvd_waits_at_least_retry_after(retry_after, backoff, expected):
    responses = iter([httpx.Response(429, headers={"Retry-After": retry_after})])
    sleeps: list[float] = []

    def handler(request):
        return next(responses, None) or nvd_page(request, 1)

    client, _ = nvd_client(handler, sleep=sleeps.append, backoff=backoff)
    assert len(list(client.iter_cves(START, END))) == 1
    assert sleeps == [expected]


def test_nvd_retries_transport_errors():
    failures = iter([httpx.ReadTimeout("slow")])
    sleeps: list[float] = []

    def handler(request):
        error = next(failures, None)
        if error:
            raise error
        return nvd_page(request, 1)

    client, seen = nvd_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_cves(START, END))) == 1
    assert len(seen) == 2
    assert sleeps == [1.0]


def test_nvd_reraises_transport_error_after_max_retries():
    def handler(request):
        raise httpx.ConnectError("down")

    client, seen = nvd_client(handler, max_retries=1)
    with pytest.raises(httpx.ConnectError):
        list(client.iter_cves(START, END))
    assert len(seen) == 2


def test_nvd_advances_by_records_returned():
    def capped(request):  # server caps pages at 2 even though 3 were asked for
        start = int(request.url.params["startIndex"])
        ids = range(start, min(start + 2, 5))
        body = {
            "resultsPerPage": len(ids),
            "startIndex": start,
            "totalResults": 5,
            "vulnerabilities": [{"cve": {"id": f"CVE-2026-{i:05d}"}} for i in ids],
        }
        return httpx.Response(200, json=body)

    client, seen = nvd_client(capped, page_size=3)
    ids = [i["cve"]["id"] for i in client.iter_cves(START, END)]
    assert ids == [f"CVE-2026-{i:05d}" for i in range(5)]
    assert [r.url.params["startIndex"] for r in seen] == ["0", "2", "4"]


def test_nvd_stops_on_empty_page():
    def empty(request):
        body = {"resultsPerPage": 0, "startIndex": 0, "totalResults": 10, "vulnerabilities": []}
        return httpx.Response(200, json=body)

    client, seen = nvd_client(empty, page_size=2)
    assert list(client.iter_cves(START, END)) == []
    assert len(seen) == 1


@pytest.mark.parametrize("page_size", [0, -1, 2001])
def test_nvd_rejects_bad_page_size(page_size):
    http, _ = make_http(lambda r: nvd_page(r, 0))
    with pytest.raises(ValueError):
        NvdClient(http, page_size=page_size)


def test_nvd_is_lazy():
    client, seen = nvd_client(lambda r: nvd_page(r, 4), page_size=2)
    iterator = client.iter_cves(START, END)
    assert seen == []
    next(iterator)
    assert len(seen) == 1


# GHSA


def ghsa_pages(pages: list[list[dict]]):
    def handler(request: httpx.Request) -> httpx.Response:
        index = int(request.url.params.get("after", "0"))
        headers = {}
        if index + 1 < len(pages):
            next_url = f"{GHSA_URL}?cursor_page=1&after={index + 1}"
            headers["Link"] = f'<{next_url}>; rel="next"'
        return httpx.Response(200, json=pages[index], headers=headers)

    return handler


def test_ghsa_single_page_request_shape():
    client, seen = ghsa_client(ghsa_pages([[{"ghsa_id": "GHSA-aaaa-bbbb-cccc"}]]))
    items = list(client.iter_advisories(START, END))
    assert items == [{"ghsa_id": "GHSA-aaaa-bbbb-cccc"}]
    request = seen[0]
    assert str(request.url).startswith(GHSA_URL + "?")
    assert dict(request.url.params) == {
        "type": "reviewed",
        "updated": "2026-11-20T00:00:00Z..2026-11-21T00:00:00Z",
        "per_page": "100",
    }
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert "Authorization" not in request.headers


def test_ghsa_follows_link_header_exactly():
    pages = [[{"n": 1}, {"n": 2}], [{"n": 3}], [{"n": 4}]]
    client, seen = ghsa_client(ghsa_pages(pages))
    assert [i["n"] for i in client.iter_advisories(START, END)] == [1, 2, 3, 4]
    assert len(seen) == 3
    assert str(seen[1].url) == f"{GHSA_URL}?cursor_page=1&after=1"
    assert str(seen[2].url) == f"{GHSA_URL}?cursor_page=1&after=2"


def test_ghsa_published_field():
    client, seen = ghsa_client(ghsa_pages([[]]))
    list(client.iter_advisories(START, END, field="published"))
    params = seen[0].url.params
    assert params["published"] == "2026-11-20T00:00:00Z..2026-11-21T00:00:00Z"
    assert "updated" not in params


def test_ghsa_sends_token_and_picks_limiter():
    http, seen = make_http(ghsa_pages([[]]))
    authed = GhsaClient(http, token="test-token")
    anonymous = GhsaClient(http)
    assert (authed.limiter.max_calls, authed.limiter.window) == (5000, 3600.0)
    assert (anonymous.limiter.max_calls, anonymous.limiter.window) == (60, 3600.0)
    list(authed.iter_advisories(START, END))
    assert seen[0].headers["Authorization"] == "Bearer test-token"


def test_ghsa_rejects_naive_datetimes():
    client, seen = ghsa_client(ghsa_pages([[]]))
    with pytest.raises(ValueError):
        list(client.iter_advisories(datetime(2026, 11, 20), END))
    assert seen == []


def test_ghsa_retries_transient_errors():
    statuses = iter([502])
    sleeps: list[float] = []
    ok = ghsa_pages([[{"n": 1}]])

    def handler(request):
        status = next(statuses, None)
        return httpx.Response(status) if status else ok(request)

    client, seen = ghsa_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_advisories(START, END))) == 1
    assert len(seen) == 2
    assert sleeps == [1.0]


def test_ghsa_is_lazy_and_acquires_per_request():
    limiter = CountingLimiter()
    client, seen = ghsa_client(ghsa_pages([[{"n": 1}], [{"n": 2}]]), limiter=limiter)
    iterator = client.iter_advisories(START, END)
    assert seen == []
    next(iterator)
    assert len(seen) == 1
    assert list(iterator) == [{"n": 2}]
    assert limiter.calls == 2


def test_ghsa_retries_primary_rate_limit_until_reset():
    clock = FakeClock()  # now = 1000.0
    limited = httpx.Response(
        403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1060"}
    )
    responses = iter([limited])
    ok = ghsa_pages([[{"n": 1}]])

    def handler(request):
        return next(responses, None) or ok(request)

    client, seen = ghsa_client(handler, sleep=clock.sleep, clock=clock)
    assert len(list(client.iter_advisories(START, END))) == 1
    assert len(seen) == 2
    assert clock.sleeps == [pytest.approx(60.0)]


def test_ghsa_retries_secondary_rate_limit_with_retry_after():
    responses = iter([httpx.Response(403, headers={"Retry-After": "30"})])
    sleeps: list[float] = []
    ok = ghsa_pages([[{"n": 1}]])

    def handler(request):
        return next(responses, None) or ok(request)

    client, seen = ghsa_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_advisories(START, END))) == 1
    assert sleeps == [30.0]


def test_ghsa_does_not_retry_plain_forbidden():
    client, seen = ghsa_client(lambda r: httpx.Response(403))
    with pytest.raises(httpx.HTTPStatusError):
        list(client.iter_advisories(START, END))
    assert len(seen) == 1


def test_ghsa_token_is_not_sent_to_other_hosts():
    def handler(request):
        if request.url.host == "api.github.com":
            link = '<https://mirror.example.test/advisories?after=1>; rel="next"'
            return httpx.Response(200, json=[{"n": 1}], headers={"Link": link})
        return httpx.Response(200, json=[{"n": 2}])

    http, seen = make_http(handler)
    client = GhsaClient(http, token="test-token", limiter=CountingLimiter())
    assert [i["n"] for i in client.iter_advisories(START, END)] == [1, 2]
    assert seen[0].headers["Authorization"] == "Bearer test-token"
    assert "Authorization" not in seen[1].headers


@pytest.mark.parametrize(
    "next_url",
    [
        "http://api.github.com/advisories?after=1",
        "https://api.github.com:8443/advisories?after=1",
    ],
)
def test_ghsa_token_requires_https_default_port(next_url):
    def handler(request):
        if request.url.params.get("after") is None:
            return httpx.Response(
                200, json=[{"n": 1}], headers={"Link": f'<{next_url}>; rel="next"'}
            )
        return httpx.Response(200, json=[{"n": 2}])

    http, seen = make_http(handler)
    client = GhsaClient(http, token="test-token", limiter=CountingLimiter())
    assert [i["n"] for i in client.iter_advisories(START, END)] == [1, 2]
    assert "Authorization" in seen[0].headers
    assert "Authorization" not in seen[1].headers


@pytest.mark.parametrize(
    "retry_after", ["inf", "nan", "1e3", "-5", "1_0", "2.5", "\u0663".encode()]
)
def test_retry_after_must_be_whole_seconds(retry_after):
    responses = iter([httpx.Response(429, headers={"Retry-After": retry_after})])
    sleeps: list[float] = []

    def handler(request):
        return next(responses, None) or nvd_page(request, 1)

    client, _ = nvd_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_cves(START, END))) == 1
    assert sleeps == [1.0]


def test_header_wait_is_capped_by_max_wait():
    responses = iter([httpx.Response(429, headers={"Retry-After": "1000000000"})])
    sleeps: list[float] = []

    def handler(request):
        return next(responses, None) or nvd_page(request, 1)

    client, _ = nvd_client(handler, sleep=sleeps.append, max_wait=60.0)
    assert len(list(client.iter_cves(START, END))) == 1
    assert sleeps == [60.0]


def test_default_max_wait_is_one_hour():
    clock = FakeClock()
    far = httpx.Response(
        403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(1000 + 10**9)}
    )
    responses = iter([far])
    ok = ghsa_pages([[{"n": 1}]])

    def handler(request):
        return next(responses, None) or ok(request)

    client, _ = ghsa_client(handler, sleep=clock.sleep, clock=clock)
    assert len(list(client.iter_advisories(START, END))) == 1
    assert clock.sleeps == [pytest.approx(3600.0)]


@pytest.mark.parametrize(
    "remaining, reset, expected",
    [("0", "900", 1.0), ("5", "1060", 1.0), ("0", "1060", 60.0)],
)
def test_ratelimit_reset_only_when_exhausted_and_in_future(remaining, reset, expected):
    clock = FakeClock()  # now = 1000.0
    limited = httpx.Response(
        429, headers={"x-ratelimit-remaining": remaining, "x-ratelimit-reset": reset}
    )
    responses = iter([limited])

    def handler(request):
        return next(responses, None) or nvd_page(request, 1)

    client, _ = nvd_client(handler, sleep=clock.sleep, clock=clock)
    assert len(list(client.iter_cves(START, END))) == 1
    assert clock.sleeps == [pytest.approx(expected)]


@pytest.mark.parametrize(
    "next_url",
    [
        "https://API.GITHUB.COM/advisories?after=1",
        "https://api.github.com:443/advisories?after=1",
    ],
)
def test_ghsa_token_sent_to_equivalent_api_urls(next_url):
    def handler(request):
        if request.url.params.get("after") is None:
            return httpx.Response(
                200, json=[{"n": 1}], headers={"Link": f'<{next_url}>; rel="next"'}
            )
        return httpx.Response(200, json=[{"n": 2}])

    http, seen = make_http(handler)
    client = GhsaClient(http, token="test-token", limiter=CountingLimiter())
    assert [i["n"] for i in client.iter_advisories(START, END)] == [1, 2]
    assert seen[1].headers["Authorization"] == "Bearer test-token"


def test_retry_after_allows_surrounding_whitespace():
    responses = iter([httpx.Response(429, headers={"Retry-After": " 7 "})])
    sleeps: list[float] = []

    def handler(request):
        return next(responses, None) or nvd_page(request, 1)

    client, _ = nvd_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_cves(START, END))) == 1
    assert sleeps == [7.0]


@pytest.mark.parametrize("reset", ["inf", "-inf", "nan"])
def test_ratelimit_reset_ignores_non_finite(reset):
    limited = httpx.Response(
        429, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset}
    )
    responses = iter([limited])
    sleeps: list[float] = []

    def handler(request):
        return next(responses, None) or nvd_page(request, 1)

    client, _ = nvd_client(handler, sleep=sleeps.append)
    assert len(list(client.iter_cves(START, END))) == 1
    assert sleeps == [1.0]


def test_transport_error_wait_is_capped_by_max_wait():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return nvd_page(request, 1)

    sleeps: list[float] = []
    client, _ = nvd_client(handler, sleep=sleeps.append, backoff=100.0, max_wait=5.0)
    assert len(list(client.iter_cves(START, END))) == 1
    assert sleeps == [5.0]


def test_ghsa_custom_max_wait():
    responses = iter([httpx.Response(429, headers={"Retry-After": "1000"})])
    ok = ghsa_pages([[{"n": 1}]])
    sleeps: list[float] = []

    def handler(request):
        return next(responses, None) or ok(request)

    client, _ = ghsa_client(handler, sleep=sleeps.append, max_wait=10.0)
    assert len(list(client.iter_advisories(START, END))) == 1
    assert sleeps == [10.0]


@pytest.mark.parametrize("cls", [NvdClient, GhsaClient])
@pytest.mark.parametrize("max_wait", [-1.0, float("inf"), float("nan")])
def test_max_wait_must_be_finite_and_non_negative(cls, max_wait):
    http, _ = make_http(lambda r: httpx.Response(200))
    with pytest.raises(ValueError):
        cls(http, max_wait=max_wait)


@pytest.mark.parametrize("cls", [NvdClient, GhsaClient])
def test_max_wait_zero_is_allowed(cls):
    http, _ = make_http(lambda r: httpx.Response(200))
    cls(http, max_wait=0.0)
