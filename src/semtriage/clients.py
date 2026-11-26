"""HTTP clients for the NVD CVE API and GitHub global advisories."""

import math
import time
from collections import deque
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import httpx

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GHSA_URL = "https://api.github.com/advisories"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_NVD_MAX_WINDOW = timedelta(days=120)
_NVD_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.000"
_GHSA_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Limiter(Protocol):
    """Something that blocks until another request is allowed."""

    def acquire(self) -> None: ...


class RateLimiter:
    """Sliding-window limiter allowing at most max_calls per window seconds."""

    def __init__(
        self,
        max_calls: int,
        window: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if window <= 0:
            raise ValueError("window must be positive")
        self.max_calls = max_calls
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        """Block until a call is allowed, then record it."""
        now = self._clock()
        self._drop_expired(now)
        while len(self._calls) >= self.max_calls:
            self._sleep(self._calls[0] + self.window - now)
            now = self._clock()
            self._drop_expired(now)
        self._calls.append(now)

    def _drop_expired(self, now: float) -> None:
        cutoff = now - self.window
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()


def _format_utc(value: datetime, fmt: str) -> str:
    return value.astimezone(UTC).strftime(fmt)


def _validate_range(start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if start >= end:
        raise ValueError("start must be before end")


def _header_wait(response: httpx.Response, wall_clock: Callable[[], float]) -> float:
    wait = 0.0
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        value = retry_after.strip()
        if value.isascii() and value.isdigit():
            wait = max(wait, float(value))
    remaining = response.headers.get("x-ratelimit-remaining")
    reset = response.headers.get("x-ratelimit-reset")
    if remaining == "0" and reset is not None:
        try:
            seconds = float(reset)
        except ValueError:
            pass
        else:
            if math.isfinite(seconds):
                wait = max(wait, seconds - wall_clock())
    return max(wait, 0.0)


def _validate_max_wait(max_wait: float) -> None:
    if not (math.isfinite(max_wait) and max_wait >= 0):
        raise ValueError("max_wait must be finite and non-negative")


def _ghsa_rate_limited(response: httpx.Response) -> bool:
    return response.headers.get("x-ratelimit-remaining") == "0" or "Retry-After" in response.headers


# NVD answers 403 when throttled, so every 403 is retried.
def _nvd_rate_limited(response: httpx.Response) -> bool:
    return True


def _request(
    http: httpx.Client,
    limiter: Limiter,
    url: str,
    *,
    params: dict[str, str] | None,
    headers: dict[str, str],
    max_retries: int,
    backoff: float,
    max_wait: float,
    sleep: Callable[[float], None],
    retry_forbidden: Callable[[httpx.Response], bool],
    wall_clock: Callable[[], float],
) -> httpx.Response:
    retries = 0
    while True:
        limiter.acquire()
        try:
            response = http.get(url, params=params, headers=headers)
        except httpx.TransportError:
            if retries >= max_retries:
                raise
            sleep(min(backoff * 2**retries, max_wait))
            retries += 1
            continue
        retryable = response.status_code in RETRY_STATUSES or (
            response.status_code == 403 and retry_forbidden(response)
        )
        if retryable and retries < max_retries:
            wait = max(backoff * 2**retries, _header_wait(response, wall_clock))
            sleep(min(wait, max_wait))
            retries += 1
            continue
        response.raise_for_status()
        return response


class NvdClient:
    """Paged reader for the NVD CVE API 2.0."""

    def __init__(
        self,
        http: httpx.Client,
        *,
        api_key: str | None = None,
        limiter: Limiter | None = None,
        page_size: int = 2000,
        max_retries: int = 3,
        backoff: float = 1.0,
        max_wait: float = 3600.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 1 <= page_size <= 2000:
            raise ValueError("page_size must be between 1 and 2000")
        _validate_max_wait(max_wait)
        self._http = http
        self._headers = {"apiKey": api_key} if api_key else {}
        self.limiter = limiter or RateLimiter(50 if api_key else 5, 30.0)
        self._page_size = page_size
        self._max_retries = max_retries
        self._backoff = backoff
        self._max_wait = max_wait
        self._sleep = sleep
        self._clock = clock

    def iter_cves(
        self,
        start: datetime,
        end: datetime,
        *,
        field: Literal["modified", "published"] = "modified",
    ) -> Iterator[dict]:
        """Yield every CVE modified or published within the range."""
        _validate_range(start, end)
        if field == "modified":
            start_key, end_key = "lastModStartDate", "lastModEndDate"
        else:
            start_key, end_key = "pubStartDate", "pubEndDate"

        window_start = start.astimezone(UTC)
        end = end.astimezone(UTC)
        while window_start < end:
            window_end = min(window_start + _NVD_MAX_WINDOW, end)
            yield from self._iter_window(start_key, end_key, window_start, window_end)
            window_start = window_end

    def _iter_window(
        self,
        start_key: str,
        end_key: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Iterator[dict]:
        start_index = 0
        while True:
            params = {
                start_key: _format_utc(window_start, _NVD_DATE_FORMAT),
                end_key: _format_utc(window_end, _NVD_DATE_FORMAT),
                "resultsPerPage": str(self._page_size),
                "startIndex": str(start_index),
            }
            response = _request(
                self._http,
                self.limiter,
                NVD_URL,
                params=params,
                headers=self._headers,
                max_retries=self._max_retries,
                backoff=self._backoff,
                max_wait=self._max_wait,
                sleep=self._sleep,
                retry_forbidden=_nvd_rate_limited,
                wall_clock=self._clock,
            )
            body = response.json()
            vulnerabilities = body["vulnerabilities"]
            yield from vulnerabilities
            if not vulnerabilities:
                return
            start_index += len(vulnerabilities)
            if start_index >= body["totalResults"]:
                return


class GhsaClient:
    """Cursor-paged reader for the GitHub global advisories API."""

    def __init__(
        self,
        http: httpx.Client,
        *,
        token: str | None = None,
        limiter: Limiter | None = None,
        per_page: int = 100,
        max_retries: int = 3,
        backoff: float = 1.0,
        max_wait: float = 3600.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        _validate_max_wait(max_wait)
        self._http = http
        self._token = token
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.limiter = limiter or RateLimiter(5000 if token else 60, 3600.0)
        self._per_page = per_page
        self._max_retries = max_retries
        self._backoff = backoff
        self._max_wait = max_wait
        self._sleep = sleep
        self._clock = clock

    def _headers_for(self, url: str) -> dict[str, str]:
        headers = dict(self._headers)
        target = httpx.URL(url)
        if (
            self._token
            and target.scheme == "https"
            and target.host == "api.github.com"
            and target.port in (None, 443)
        ):
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def iter_advisories(
        self,
        start: datetime,
        end: datetime,
        *,
        field: Literal["updated", "published"] = "updated",
    ) -> Iterator[dict]:
        """Yield every advisory updated or published within the range."""
        _validate_range(start, end)
        params: dict[str, str] | None = {
            "type": "reviewed",
            field: (
                f"{_format_utc(start, _GHSA_DATE_FORMAT)}..{_format_utc(end, _GHSA_DATE_FORMAT)}"
            ),
            "per_page": str(self._per_page),
        }
        url = GHSA_URL
        while True:
            response = _request(
                self._http,
                self.limiter,
                url,
                params=params,
                headers=self._headers_for(url),
                max_retries=self._max_retries,
                backoff=self._backoff,
                max_wait=self._max_wait,
                sleep=self._sleep,
                retry_forbidden=_ghsa_rate_limited,
                wall_clock=self._clock,
            )
            yield from response.json()
            next_link = response.links.get("next")
            if not next_link:
                return
            url = next_link["url"]
            params = None
