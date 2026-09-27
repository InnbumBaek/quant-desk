"""Read one sector per symbol from Yahoo, a bounded number of symbols per run.

`scripts/fetch_listings.py` owns the universe; this owns the one field the
universe cannot get anywhere else. It is separate because its shape is
different from every other fetcher here: one request per symbol rather than one
file, a session to establish first, and a budget that stops it well short of
finishing.

**Why one symbol at a time, and why a budget.** The bulk sources are gone --
every one of them refused this runner in a measured probe (ADR-0019) -- so what
is left is an endpoint that answers per symbol. Thirteen thousand of those at a
polite pace is hours, which is not a weekly job. But a sector is a slow-moving
fact and the committed listings file already carries last week's answers, so
each run only needs the symbols that have none. The coverage climbs over
several weeks and then stays flat, and the file is the resume point: there is
no state to keep anywhere else.

**Why a session.** Yahoo's profile endpoint wants a crumb, and a crumb wants a
cookie. That is the ordinary flow of its own clients, not a lock being picked:
the two-step is documented behaviour and the requests still say who we are. A
bare crumb request with no cookie is what returns 429, which is what the probe
measured before this existed.

**What it will not do.** It does not retry a symbol that answered with no
sector -- an empty sector is an answer, and a fund or a shell genuinely has
none. It does not fabricate a bucket for an unmapped label. And it never
touches the order path: a symbol with no bucket simply cannot be ordered
(ADR-0017), which is the failure closing where it belongs.
"""

from __future__ import annotations

import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from core.config import USER_AGENT
from core.data.yahoo_sectors import bucket_for_yahoo_sector, unmapped_labels

#: The page that sets the cookie the crumb endpoint checks.
COOKIE_URL = "https://fc.yahoo.com/"
CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
PROFILE_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"

#: Seconds between symbol requests. A weekly job reading a few hundred symbols
#: at this pace is a rounding error to Yahoo and takes minutes here.
PAUSE_SECONDS = 0.4
#: How many symbols one run will read. Sized so the step stays well inside a
#: normal job rather than to finish quickly: the file is the resume point, so
#: finishing sooner is worth less than never being the reason a run is killed.
DEFAULT_BUDGET = 1200
#: Consecutive failures that mean the session died rather than a symbol being
#: odd. Reading a thousand more symbols after that is a thousand more refusals.
GIVE_UP_AFTER = 25


class ProfileError(RuntimeError):
    """Yahoo did not give us a usable session. Never swallowed into zero sectors."""


@dataclass
class Harvest:
    """What one run read, and what it could not."""

    buckets: dict[str, str] = field(default_factory=dict)
    #: Symbols Yahoo answered for with no sector: funds, trusts, shells. Worth
    #: recording so the next run does not spend its budget asking again.
    no_sector: list[str] = field(default_factory=list)
    #: Symbols that failed. These are worth asking about again.
    failed: list[str] = field(default_factory=list)
    unmapped: tuple[str, ...] = ()
    asked: int = 0
    stopped_early: str = ""


def _request(url: str, opener: urllib.request.OpenerDirector, timeout: float = 20.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}
    )
    with opener.open(request, timeout=timeout) as response:  # noqa: S310 - fixed host
        return response.read()


def open_session(timeout: float = 20.0) -> tuple[urllib.request.OpenerDirector, str]:
    """A cookie jar with Yahoo's cookie in it, and the crumb that goes with it."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    try:
        _request(COOKIE_URL, opener, timeout)
    except urllib.error.HTTPError as error:
        # fc.yahoo.com answers 404 while still setting the cookie, which is the
        # whole reason we call it. Only a missing cookie is a failure.
        if not len(jar):
            raise ProfileError(f"no cookie from {COOKIE_URL}: HTTP {error.code}") from error
    except OSError as error:
        raise ProfileError(f"could not reach {COOKIE_URL}: {error}") from error

    if not len(jar):
        raise ProfileError(f"{COOKIE_URL} set no cookie, so the crumb will be refused")
    try:
        crumb = _request(CRUMB_URL, opener, timeout).decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as error:
        raise ProfileError(f"crumb refused: HTTP {error.code} {error.reason}") from error
    except OSError as error:
        raise ProfileError(f"could not reach the crumb endpoint: {error}") from error
    # A crumb is a short opaque token. An HTML page here means a challenge, and
    # sending it as a query parameter would make every symbol fail obscurely.
    if not crumb or len(crumb) > 64 or "<" in crumb:
        raise ProfileError(f"the crumb endpoint returned something that is not a crumb: {crumb[:60]!r}")
    return opener, crumb


def profile_url(symbol: str, crumb: str) -> str:
    query = urllib.parse.urlencode({"modules": "assetProfile", "crumb": crumb})
    return f"{PROFILE_URL.format(symbol=urllib.parse.quote(symbol))}?{query}"


def read_sector(raw: bytes) -> str | None:
    """The `sector` string out of a quoteSummary reply, or None if it carries none."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"the profile reply is not JSON: {error}") from error
    finance = payload.get("finance") or {}
    if isinstance(finance, dict) and finance.get("error"):
        raise ProfileError(f"the profile reply carries an error: {finance['error']}")
    summary = payload.get("quoteSummary") or {}
    if summary.get("error"):
        raise ProfileError(f"the profile reply carries an error: {summary['error']}")
    results = summary.get("result") or []
    if not results:
        return None
    profile = (results[0] or {}).get("assetProfile") or {}
    sector = profile.get("sector")
    return str(sector).strip() if sector else None


def harvest(
    symbols: Iterable[str],
    budget: int = DEFAULT_BUDGET,
    pause: Callable[[float], None] = time.sleep,
    session: tuple[urllib.request.OpenerDirector, str] | None = None,
) -> Harvest:
    """Read up to `budget` symbols. Raises only if the session itself failed."""
    opener, crumb = session or open_session()
    out = Harvest()
    labels: list[str] = []
    consecutive = 0

    for symbol in list(symbols)[: max(budget, 0)]:
        if out.asked:
            pause(PAUSE_SECONDS)
        out.asked += 1
        try:
            sector = read_sector(_request(profile_url(symbol, crumb), opener))
            consecutive = 0
        except (ProfileError, urllib.error.HTTPError, OSError) as error:
            out.failed.append(symbol)
            consecutive += 1
            if consecutive >= GIVE_UP_AFTER:
                out.stopped_early = f"{consecutive} symbols in a row failed; last: {error}"
                break
            continue
        if sector is None:
            out.no_sector.append(symbol)
            continue
        labels.append(sector)
        bucket = bucket_for_yahoo_sector(sector)
        if bucket is None:
            continue
        out.buckets[symbol] = bucket

    out.unmapped = unmapped_labels(labels)
    return out
