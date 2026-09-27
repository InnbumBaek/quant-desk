"""Fetch macro series from FRED (US) and ECOS (Bank of Korea).

Runs on the GitHub Actions runner, never here: this container reaches neither
host, and both keys live only in Actions secrets (ADR-0012, ADR-0024).

**The initial release, not today's number.** FRED is asked with
`output_type=4`, which returns each observation as it was *first published*,
and its `realtime_start` is that publication day. Both dates land in the file.
Asking for the default (`output_type=1`) would return today's revised values
stamped with old period dates -- look-ahead that nothing about the file would
reveal.

**Both keys must be scrubbed from every printed string.** FRED takes its key
as a query parameter; ECOS takes it as a *path segment*, which is worse because
it survives any naive query-stripping. `redact` knows about both and every
message in this module goes through it.

**A refused series is not an empty series.** ECOS answers a bad stat code, a
bad key and a genuinely empty window all with HTTP 200 (ADR-0023's lesson,
same shape). A refusal is recorded per series and makes the job exit non-zero;
the series that did answer are still written, so a fixed code costs one series
and not a whole backfill.

**Why the ECOS table list is dumped too.** Nobody here has held a keyed ECOS
response, so this catalogue's stat codes come from documentation. The job
writes `StatisticTableList` to the registry, which turns the next revision of
the catalogue into a lookup against measured fact instead of another guess.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from core.config import USER_AGENT, require_env
from core.data.macro import (
    VINTAGE_INITIAL,
    MacroRefused,
    MacroSeries,
    MacroShapeError,
    read_ecos,
    read_fred,
    to_rows,
)

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
ECOS_BASE = "https://ecos.bok.or.kr/api"
FRED_KEY_ENV = "FRED_API_KEY"
ECOS_KEY_ENV = "ECOS_API_KEY"

#: FRED's "observations, initial release only". The whole point of the fetch.
FRED_INITIAL_RELEASE = "4"

#: ECOS caps a page; a daily series since 1990 is under ten thousand rows.
ECOS_PAGE_ROWS = 100_000

PAUSE_SECONDS = 1.0
RETRYABLE_HTTP = (429, 500, 502, 503, 504)
BACKOFF_SECONDS = (10.0, 30.0, 90.0)

DEFAULT_START = date(1990, 1, 1)


@dataclass(frozen=True)
class FredSeries:
    """One FRED series and why this desk reads it."""

    series_id: str
    note: str
    units: str = ""


@dataclass(frozen=True)
class EcosSeries:
    """One ECOS series. `stat` is the table, `item` the row inside it."""

    stat: str
    item: str
    cycle: str
    note: str

    @property
    def series_id(self) -> str:
        return f"{self.stat}.{self.item}.{self.cycle}"


#: US series. Every one of these is published by a federal agency or the
#: Federal Reserve Board, which is why the files can be committed.
FRED_CATALOGUE: tuple[FredSeries, ...] = (
    FredSeries("DFF", "정책금리 — 레짐 분류의 기준선", "percent"),
    FredSeries("DGS3MO", "3개월 국채 — 커브 단기단", "percent"),
    FredSeries("DGS2", "2년 국채", "percent"),
    FredSeries("DGS10", "10년 국채 — 커브 기울기는 여기서 유도한다", "percent"),
    FredSeries("T10YIE", "10년 기대인플레이션(브레이크이븐)", "percent"),
    FredSeries("DTWEXBGS", "광범위 달러지수 — 크로스에셋 포드의 통화 팩터", "index"),
    FredSeries("CPIAUCSL", "소비자물가 — 발표지연이 5주라 as-of가 반드시 필요한 계열", "index"),
    FredSeries("PAYEMS", "비농업 고용 — 개정폭이 큰 대표적 계열", "thousands"),
    FredSeries("UNRATE", "실업률", "percent"),
    FredSeries("INDPRO", "산업생산", "index"),
)

#: Series this desk wants and deliberately does not fetch here. FRED carries
#: them, but their publishers do not permit redistribution, and a fetched file
#: that cannot be committed is a file no research run can read anyway
#: (ADR-0007's reasoning, applied to macro).
NOT_REDISTRIBUTABLE: dict[str, str] = {
    "VIXCLS": "CBOE 저작물 — 재배포 불가. 변동성 레짐은 자체 계산으로 대체한다",
    "BAMLH0A0HYM2": "ICE BofA 지수 — 라이선스 확인 전까지 받지 않는다",
}

#: Korean series. The codes come from ECOS documentation and nothing here has
#: verified them against a keyed response: the first run either writes the file
#: or names the refusal, and the table-list dump is how the next revision of
#: this list stops being a guess.
ECOS_CATALOGUE: tuple[EcosSeries, ...] = (
    EcosSeries("722Y001", "0101000", "M", "한국은행 기준금리"),
    EcosSeries("817Y002", "010190000", "D", "국고채 3년 수익률"),
    EcosSeries("731Y001", "0000001", "D", "원/달러 매매기준율"),
    EcosSeries("901Y009", "0", "M", "소비자물가지수"),
)


class FetchError(RuntimeError):
    """A source did not give us a series. Never swallowed into an empty one."""


def redact(text: str, *keys: str) -> str:
    """Both keys out of any string that might be printed.

    FRED's key rides in the query and ECOS's in the path, so an unredacted URL
    in a public job log is a disclosed credential either way. This is the only
    place allowed to know the keys are secret.
    """
    for key in keys:
        if key:
            text = text.replace(key, "<REDACTED_KEY>")
    return text


def fred_url(series_id: str, key: str, start: date) -> str:
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        # Each observation as first published, with realtime_start saying when.
        "output_type": FRED_INITIAL_RELEASE,
        "observation_start": start.isoformat(),
    }
    return f"{FRED_ENDPOINT}?{urllib.parse.urlencode(params)}"


def ecos_url(series: EcosSeries, key: str, start: date, end: date) -> str:
    """ECOS's path-shaped request. The key is the third segment, hence `redact`."""
    stamp = _ecos_stamp(series.cycle)
    parts = [
        ECOS_BASE,
        "StatisticSearch",
        key,
        "json",
        "kr",
        "1",
        str(ECOS_PAGE_ROWS),
        series.stat,
        series.cycle,
        start.strftime(stamp),
        end.strftime(stamp),
        series.item,
    ]
    return "/".join(parts)


def ecos_tables_url(key: str) -> str:
    return f"{ECOS_BASE}/StatisticTableList/{key}/json/kr/1/{ECOS_PAGE_ROWS}"


def _ecos_stamp(cycle: str) -> str:
    """How ECOS wants a window boundary written, per cycle."""
    stamps = {"D": "%Y%m%d", "M": "%Y%m", "Q": "%Y%m", "A": "%Y"}
    if cycle not in stamps:
        raise FetchError(f"cycle {cycle!r} is not one ECOS documents (one of {', '.join(sorted(stamps))})")
    return stamps[cycle]


def _explain(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read()
    except OSError:
        return "no response body"
    return " ".join(body.decode("utf-8", errors="replace").split())[:300] or "empty response body"


def _get(
    url: str,
    *keys: str,
    timeout: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last = ""
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed hosts
                return response.read()
        except urllib.error.HTTPError as error:
            last = redact(f"HTTP {error.code}: {_explain(error)}", *keys)
            if error.code not in RETRYABLE_HTTP:
                raise FetchError(last) from error
        except OSError as error:
            last = redact(f"{type(error).__name__}: {error}", *keys)
        if attempt < len(BACKOFF_SECONDS):
            sleep(BACKOFF_SECONDS[attempt])
    raise FetchError(f"no answer after {len(BACKOFF_SECONDS) + 1} attempts; last: {last}")


def _payload(raw: bytes, label: str, *keys: str) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        head = redact(" ".join(raw.decode("utf-8", errors="replace").split())[:200], *keys)
        raise FetchError(f"{label}: the reply is not JSON: {head!r}") from error
    if not isinstance(payload, dict):
        raise FetchError(f"{label}: the reply is a {type(payload).__name__}, not an object")
    return payload


def schema_of(payload: dict, root: str, row_key: str) -> dict[str, object]:
    """Field names and a row count. Never values -- a key could be echoed back.

    The catalogue's field names came from documentation, so a successful run
    has to report the shape it actually got: that is how the guess becomes a
    measurement (the discipline `scripts/fetch_krx.py` set, ADR-0022).
    """
    if root:
        block = payload.get(root)
        rows = block.get(row_key) if isinstance(block, dict) else None
    else:
        rows = payload.get(row_key)
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return {"rows": 0, "fields": []}
    return {"rows": len(rows), "fields": sorted(str(key) for key in rows[0])}


def fetch_fred(
    series: FredSeries,
    key: str,
    start: date,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[MacroSeries, dict[str, object]]:
    raw = _get(fred_url(series.series_id, key, start), key, sleep=sleep)
    payload = _payload(raw, series.series_id, key)
    shape = schema_of(payload, "", "observations")
    try:
        got = read_fred(payload, series.series_id, vintage=VINTAGE_INITIAL, units=series.units)
    except MacroRefused as refused:
        raise FetchError(f"{series.series_id}: {redact(str(refused), key)}") from refused
    except MacroShapeError as error:
        raise FetchError(f"{series.series_id}: {redact(str(error), key)}") from error
    return got, shape


def fetch_ecos(
    series: EcosSeries,
    key: str,
    start: date,
    end: date,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[MacroSeries, dict[str, object]]:
    raw = _get(ecos_url(series, key, start, end), key, sleep=sleep)
    payload = _payload(raw, series.series_id, key)
    shape = schema_of(payload, "StatisticSearch", "row")
    try:
        got = read_ecos(payload, series.series_id)
    except MacroRefused as refused:
        raise FetchError(f"{series.series_id}: {redact(str(refused), key)}") from refused
    except MacroShapeError as error:
        raise FetchError(f"{series.series_id}: {redact(str(error), key)}") from error
    return got, shape


def write_series(series: MacroSeries, directory: Path, name: str | None = None) -> Path:
    """One file per series, header first, so a re-run replaces exactly that series."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name or series.series_id}.csv"
    rows: list[Sequence[str]] = to_rows(series)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_meta(entries: list[dict[str, object]], path: Path) -> Path:
    """What was fetched, from where, in which vintage, and what shape came back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sources": {
                    "fred": "api.stlouisfed.org/fred/series/observations?output_type=4 (initial release)",
                    "ecos": "ecos.bok.or.kr/api/StatisticSearch (revisions in place; no vintage parameter)",
                },
                "not_redistributable": NOT_REDISTRIBUTABLE,
                "series": sorted(entries, key=lambda entry: str(entry["series_id"])),
            },
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def fetch_ecos_tables(key: str, directory: Path, sleep: Callable[[float], None] = time.sleep) -> int:
    """ECOS's own table list, so the next catalogue revision is a lookup."""
    raw = _get(ecos_tables_url(key), key, sleep=sleep)
    payload = _payload(raw, "StatisticTableList", key)
    block = payload.get("StatisticTableList")
    rows = block.get("row") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        result = payload.get("RESULT")
        detail = f": {redact(json.dumps(result, ensure_ascii=False), key)}" if result else ""
        raise FetchError(f"ECOS returned no table list{detail}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ecos_tables.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(rows)


def fetch(
    directory: Path,
    start: date = DEFAULT_START,
    end: date | None = None,
    fred_key: str | None = None,
    ecos_key: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    fred_catalogue: tuple[FredSeries, ...] | None = None,
    ecos_catalogue: tuple[EcosSeries, ...] | None = None,
) -> dict[str, object]:
    """Every catalogued series. A refusal is recorded, not swallowed, and not fatal.

    One bad stat code must not cost the other nine series, and it must not look
    like a quiet window either. So each failure is collected with its reason and
    `main` exits non-zero on any of them.
    """
    last = end or datetime.now(UTC).date()
    us_key = fred_key or require_env(FRED_KEY_ENV)
    kr_key = ecos_key or require_env(ECOS_KEY_ENV)
    # Read at call time, not bound as a default: a default argument freezes the
    # catalogue at import and a test that swaps it then silently fetches the
    # real list instead of the one it asked for.
    fred_catalogue = FRED_CATALOGUE if fred_catalogue is None else fred_catalogue
    ecos_catalogue = ECOS_CATALOGUE if ecos_catalogue is None else ecos_catalogue

    entries: list[dict[str, object]] = []
    refused: list[str] = []
    first = True

    for series in fred_catalogue:
        if not first:
            sleep(PAUSE_SECONDS)
        first = False
        try:
            got, shape = fetch_fred(series, us_key, start, sleep=sleep)
        except FetchError as error:
            refused.append(str(error))
            continue
        write_series(got, directory / "us")
        entries.append(
            {
                "series_id": got.series_id,
                "source": "fred",
                "vintage": got.vintage,
                "point_in_time": got.point_in_time,
                "observations": len(got.observations),
                "span": [day.isoformat() for day in got.span] if got.span else None,
                "note": series.note,
                "shape": shape,
            }
        )

    for series in ecos_catalogue:
        sleep(PAUSE_SECONDS)
        try:
            got, shape = fetch_ecos(series, kr_key, start, last, sleep=sleep)
        except FetchError as error:
            refused.append(str(error))
            continue
        write_series(got, directory / "kr", name=f"{series.stat}_{series.item}_{series.cycle}")
        entries.append(
            {
                "series_id": got.series_id,
                "source": "ecos",
                "vintage": got.vintage,
                "point_in_time": got.point_in_time,
                "observations": len(got.observations),
                "span": [day.isoformat() for day in got.span] if got.span else None,
                "note": series.note,
                "shape": shape,
            }
        )

    tables = 0
    try:
        tables = fetch_ecos_tables(kr_key, directory)
    except FetchError as error:
        refused.append(str(error))

    write_meta(entries, directory / "manifest.json")
    return {
        "series": len(entries),
        "asked": len(fred_catalogue) + len(ecos_catalogue),
        "point_in_time": sum(1 for entry in entries if entry["point_in_time"]),
        "observations": sum(int(entry["observations"]) for entry in entries),
        "ecos_tables": tables,
        "refused": refused,
        "window": {"from": start.isoformat(), "to": last.isoformat()},
        "directory": str(directory),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch FRED and ECOS macro series")
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--out", default="registry/macro")
    args = parser.parse_args(argv)

    try:
        report = fetch(Path(args.out), start=date.fromisoformat(args.start))
    except (FetchError, RuntimeError, ValueError) as error:
        print(f"macro fetch failed: {error}", file=sys.stderr)
        return 1

    print(f"## Macro {report['series']}/{report['asked']} series, {report['observations']} observations")
    print(f"- point-in-time (initial release): {report['point_in_time']}")
    print(f"- ECOS table list rows: {report['ecos_tables']}")
    refused = report["refused"]
    if isinstance(refused, list) and refused:
        print(f"- refused ({len(refused)}):")
        for reason in refused:
            print(f"  - {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
