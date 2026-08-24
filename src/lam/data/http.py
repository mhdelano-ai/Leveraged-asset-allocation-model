"""Shared HTTP client: browser UA, session reuse, polite pacing, backoff.

Both Yahoo and FRED throttle bursts. Yahoo returns HTTP 429 on bare requests and
on rapid sequences; FRED simply stops responding (connection timeouts) after a
few dozen fast requests. Neither is a permanent block -- both clear after a
cooldown -- so the correct handling is paced requests with exponential backoff,
not parallelism.
"""

from __future__ import annotations

import time

import requests

# A desktop browser UA is required: bare/absent UA gets HTTP 429 from Yahoo
# immediately, regardless of rate.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_MIN_INTERVAL = 1.1  # seconds between requests to the same host
_last_request: dict[str, float] = {}
_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json,text/csv,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        _session = s
    return _session


def _pace(host: str) -> None:
    now = time.monotonic()
    prev = _last_request.get(host)
    if prev is not None:
        wait = _MIN_INTERVAL - (now - prev)
        if wait > 0:
            time.sleep(wait)
    _last_request[host] = time.monotonic()


def get(
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 60,
    tries: int = 5,
    headers: dict | None = None,
) -> bytes:
    """GET with pacing and exponential backoff. Returns the raw body.

    ``headers`` overrides the session defaults for this request only. FRED needs
    it: the desktop UA that Yahoo *requires* is the one FRED's bot filter stalls
    on -- the connection is accepted and then never answered, so it surfaces as a
    read timeout and five rounds of backoff rather than a 403.
    """
    host = url.split("/")[2]
    last_err: Exception | None = None
    for attempt in range(tries):
        _pace(host)
        try:
            resp = session().get(url, params=params, timeout=timeout, headers=headers)
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - retry on any transport failure
            last_err = exc
            if attempt == tries - 1:
                break
            # 4s, 9s, 16s, 25s - Yahoo's burst limiter needs seconds, not ms.
            time.sleep((attempt + 2) ** 2)
    raise RuntimeError(f"GET failed after {tries} attempts: {url}") from last_err
