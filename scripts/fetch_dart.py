"""Fetch Korean disclosure metadata from DART, paged, for the event pod.

Run on the GitHub Actions runner, never here: this container reaches no
Korean host, and the key lives only in Actions secrets (ADR-0012, ADR-0023).

**What this fetches and what it does not.** The disclosure *list* -- who filed
what, when, and under which receipt number -- and nothing else. Not the filing
bodies, not the financial statements. The list is what an event study needs to
build a timeline, and it is small enough to commit, which matters because this
container can never fetch it itself.

**Why the status code is the whole design.** DART answers a rejected key, a
maintenance window and a genuinely quiet day all with HTTP 200. A fetcher that
checks only the transport writes an empty day in all three cases, and an empty
day on disk is indistinguishable from a holiday. `core/data/dart.py` sorts the
codes; this script acts on the sorting -- an answer is written, a retryable
refusal is waited out, and a rejected key stops the run loudly.

**The window is closed, never open-ended.** DART's list is paged and a wide
window is thousands of rows. One day per request keeps each answer small,
makes a partial run resumable by date, and means a re-run of one bad day costs
one request.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from core.config import USER_AGENT, require_env
from core.data.dart import DartRefused, DartShapeError, Disclosure, read_page

ENDPOINT = "https://opendart.fss.or.kr/api/list.json"
KEY_ENV = "DART_API_KEY"

#: Rows per page. DART allows 100 and a Korean trading day runs to a few
#: hundred filings, so a day is two or three requests.
PAGE_SIZE = 100
#: Pages per day before we stop believing the paging. A day has never held
#: 5,000 filings; twenty pages means `total_page` is not what we think it is.
MAX_PAGES = 50

PAUSE_SECONDS = 1.0
RETRYABLE_HTTP = (429, 500, 502, 503, 504)
BACKOFF_SECONDS = (10.0, 30.0, 90.0)


class FetchError(RuntimeError):
    """DART did not give us a day. Never swallowed into an empty one."""


def _explain(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read()
    except OSError:
        return "no response body"
    return " ".join(body.decode("utf-8", errors="replace").split())[:300] or "empty response body"


def page_url(key: str, day: date, page: int = 1, page_size: int = PAGE_SIZE) -> str:
    """One day of the list.

    The key is a query parameter because that is the only shape DART accepts.
    It therefore must never be printed, logged or committed: `redact` below is
    what every message in this module goes through, and the tests hold it to
    that.
    """
    params = {
        "crtfc_key": key,
        "bgn_de": day.strftime("%Y%m%d"),
        "end_de": day.strftime("%Y%m%d"),
        "page_no": str(page),
        "page_count": str(page_size),
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def redact(text: str, key: str) -> str:
    """The key out of any string that might be printed.

    DART puts the key in the URL, so every error message that quotes a URL is
    a credential leak waiting for a public job log. This is the one place that
    is allowed to know the key is a secret, and everything printed goes
    through it.
    """
    return text.replace(key, "<DART_API_KEY>") if key else text


def _get(url: str, key: str, timeout: float = 30.0, sleep: Callable[[float], None] = time.sleep) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last = ""
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
                return response.read()
        except urllib.error.HTTPError as error:
            last = redact(f"HTTP {error.code}: {_explain(error)}", key)
            if error.code not in RETRYABLE_HTTP:
                raise FetchError(last) from error
        except OSError as error:
            last = redact(f"{type(error).__name__}: {error}", key)
        if attempt < len(BACKOFF_SECONDS):
            sleep(BACKOFF_SECONDS[attempt])
    raise FetchError(f"DART did not answer after {len(BACKOFF_SECONDS) + 1} attempts; last: {last}")


def fetch_day(
    day: date,
    key: str,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Disclosure, ...]:
    """Every filing of one day, following DART's paging."""
    out: list[Disclosure] = []
    for page in range(1, max_pages + 1):
        if page > 1:
            sleep(PAUSE_SECONDS)
        raw = _get(page_url(key, day, page, page_size), key, sleep=sleep)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            head = redact(" ".join(raw.decode("utf-8", errors="replace").split())[:200], key)
            raise FetchError(f"{day}: the reply is not JSON: {head!r}") from error
        try:
            got = read_page(payload)
        except DartRefused as refused:
            if refused.retryable:
                # A rate limit or a maintenance window. `_get` has already
                # spent its HTTP retries on transport errors, which this is
                # not, so the wait belongs here.
                sleep(BACKOFF_SECONDS[-1])
                raw = _get(page_url(key, day, page, page_size), key, sleep=sleep)
                got = read_page(json.loads(raw.decode("utf-8")))
            else:
                raise FetchError(f"{day}: {refused}") from refused
        except DartShapeError as error:
            raise FetchError(f"{day}: {error}") from error
        out.extend(got.disclosures)
        if not got.has_more:
            break
    else:
        raise FetchError(f"{day}: DART kept reporting more pages past {max_pages}")
    return tuple(out)


def weekdays(start: date, end: date) -> list[date]:
    """Every Monday-to-Friday in the window. DART itself says which were quiet."""
    day, out = start, []
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def write_day(day: date, disclosures: Iterable[Disclosure], directory: Path) -> Path:
    """One file per day, so a re-run replaces exactly one day and nothing else."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(disclosures, key=lambda d: d.receipt_no)
    path = directory / f"{day.isoformat()}.json"
    path.write_text(
        json.dumps(
            {
                "day": day.isoformat(),
                "source": "dart-open-api:list.json",
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "count": len(rows),
                "listed_filers": sum(1 for row in rows if row.listed),
                "disclosures": [
                    {
                        "receipt_no": row.receipt_no,
                        "filed_on": row.filed_on.isoformat(),
                        "corp_code": row.corp_code,
                        "corp_name": row.corp_name,
                        "stock_code": row.stock_code,
                        "corp_class": row.corp_class,
                        "report": row.report,
                        "filer": row.filer,
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def fetch(
    directory: Path,
    days: int = 7,
    as_of: date | None = None,
    key: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    end = as_of or datetime.now(UTC).date()
    start = end - timedelta(days=max(days, 1))
    secret = key or require_env(KEY_ENV)

    written: list[str] = []
    quiet: list[str] = []
    total = 0
    for index, day in enumerate(weekdays(start, end)):
        if index:
            sleep(PAUSE_SECONDS)
        disclosures = fetch_day(day, secret, sleep=sleep)
        if not disclosures:
            # DART said 013 for this day. That is an answer, and writing an
            # empty file records it as one rather than leaving a hole that
            # looks the same as a day we never asked about.
            quiet.append(day.isoformat())
        write_day(day, disclosures, directory)
        written.append(day.isoformat())
        total += len(disclosures)

    if not written:
        raise FetchError(f"no weekdays between {start} and {end}; the window is wrong")
    return {
        "days": len(written),
        "quiet_days": quiet,
        "disclosures": total,
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "directory": str(directory),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", default="registry/filings/kr")
    args = parser.parse_args(argv)

    try:
        report = fetch(Path(args.out), days=args.days)
    except (FetchError, RuntimeError) as error:
        print(f"DART fetch failed: {error}", file=sys.stderr)
        return 1

    print(f"## DART {report['disclosures']} disclosures over {report['days']} weekdays")
    quiet = report["quiet_days"]
    print(f"- days DART reported nothing for: {len(quiet)}{' ' + ', '.join(quiet) if quiet else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
