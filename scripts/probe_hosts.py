"""Ask, in one runner minute, which data hosts answer from GitHub Actions.

Two sources have now refused this desk from the Actions address range, and each
one cost several runs to establish: `www.sec.gov` returned "Request Rate
Threshold Exceeded" fourteen times across an hour (ADR-0019), and
`export.arxiv.org` returns 406 with an empty body to every request shape we can
honestly send. Both refusals look identical from inside a fetcher -- a status
code and nothing to read -- and both were diagnosed one weekly run at a time.

That is the wrong loop. A weekly job that learns one bit per week cannot keep up
with a desk that still has KRX, DART, ECOS, FRED and KIS to connect. This script
asks every host at once and writes down what each said, so the next adapter
starts from a measurement instead of an attempt.

**It is a report, not a gate.** Nothing here is on the order path, nothing
retries, and it exits zero even when every host refuses -- a prober that fails
the job cannot commit the reason it failed. It sends one request per target and
reads at most a few kilobytes, which is politer than the fetchers it advises.

**Every request identifies us honestly.** No target is probed with a browser
user-agent. A 200 obtained by lying about who we are would send the next adapter
down a path that breaks the moment it is used in earnest, and would make the
refusal after that undiagnosable.
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.config import USER_AGENT

#: The same contact string the fetchers use. A probe that declares itself
#: differently is not probing the thing we are about to do.

#: How much of a successful body to read. Enough to tell a feed from an error
#: page, not enough to be a download.
SNIFF_BYTES = 2048
#: Characters of a response body kept in the report, success or refusal alike.
BODY_CHARS = 300


@dataclass(frozen=True)
class Target:
    """One host we either depend on or are considering depending on."""

    label: str
    url: str
    #: What a working answer starts with. A host that returns 200 and an error
    #: page is refusing us too, just less honestly, and this is what catches it.
    expect: str = ""
    #: Why this target is in the table, for whoever reads the report later.
    note: str = ""
    accept: str = "*/*"
    #: A target whose failure means the probe itself is broken, not the host.
    control: bool = False
    #: Extra request headers this host documents as required. Only for a header
    #: a host genuinely asks for -- never one chosen to look like somebody else.
    headers: tuple[tuple[str, str], ...] = ()


#: Ordered so the control comes first: if `nasdaqtrader` fails, nothing else in
#: the report means anything and the run was a network problem of our own.
ATOM_FIRST = "application/atom+xml, application/xml;q=0.9, */*;q=0.8"
XML_FIRST = "application/xml, text/xml;q=0.9, */*;q=0.8"

TARGETS: tuple[Target, ...] = (
    Target(
        label="nasdaqtrader-symbols",
        url="https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        expect="Symbol|",
        note="the membership source in use today; the control for this probe",
        control=True,
    ),
    # --- arXiv: the refusal tracks how much work the query is ---------------
    #
    # Eight readings are out now (ADR-0020). Ordering and rate died when the
    # list was reversed; cache and URL length died in the run after that:
    #
    #   cat:q-fin.ST&max_results=1   200   never asked before -> not a cache
    #   cat:q-fin&max_results=1      406   shorter than both  -> not length
    #
    # What the served requests have in common is that they are cheap for the
    # server: one entry from one narrow category, or `Identify`, which is a
    # constant. What the refused ones share is work -- a whole archive, a sort,
    # 25 entries, a set harvest. So the reading now is that an expensive query
    # is refused, and Fastly reports it as 406.
    #
    # These five say where the line is, one parameter at a time from a request
    # that is served. Two of them matter most: `start-only` adds a parameter
    # without asking for more entries, and `max-2` asks for one more entry than
    # the request that works. If both are served, the threshold is a count and
    # the sweep can page just under it. If `max-2` refuses, one entry per
    # category is all this interface will ever give us, and the weekly sweep
    # has to move to the RSS host, which answers every time.
    Target(
        label="arxiv-api-max-2",
        url="https://export.arxiv.org/api/query?search_query=cat:q-fin.PM&max_results=2",
        expect="<?xml",
        note="one entry more than the request that is always served",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-api-max-10",
        url="https://export.arxiv.org/api/query?search_query=cat:q-fin.PM&max_results=10",
        expect="<?xml",
        note="between the served 1 and the refused 25, to find the step",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-api-start-only",
        url="https://export.arxiv.org/api/query?search_query=cat:q-fin.PM&max_results=1&start=0",
        expect="<?xml",
        note="the served request plus `start`, which asks for no more work. 406 means the rule reads keys",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-api-sort-only",
        url=(
            "https://export.arxiv.org/api/query?search_query=cat:q-fin.PM&max_results=1&sortBy=submittedDate"
        ),
        expect="<?xml",
        note="sort without sortOrder: one entry back, but the server must order the category first",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-api-narrow-novel",
        url="https://export.arxiv.org/api/query?search_query=cat:q-fin.TR&max_results=1",
        expect="<?xml",
        note="a third narrow category never asked for, as the control that narrow-and-small still works",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-api",
        url="https://export.arxiv.org/api/query?search_query=cat:q-fin.PM&max_results=1",
        expect="<?xml",
        note="the weekly literature sweep (ADR-0011); served in every position, every run",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-api-archive-wide",
        url="https://export.arxiv.org/api/query?search_query=cat:q-fin&max_results=1",
        expect="<?xml",
        note="the same request over a whole archive rather than one category; refused so far",
        accept=ATOM_FIRST,
    ),
    Target(
        label="arxiv-oaipmh",
        url="https://oaipmh.arxiv.org/oai?verb=Identify",
        expect="<?xml",
        note="a constant answer from the harvest host; served every run",
        accept=XML_FIRST,
    ),
    Target(
        label="arxiv-oai-listrecords",
        url=("https://oaipmh.arxiv.org/oai?verb=ListRecords&set=q-fin&metadataPrefix=arXiv&from=2026-09-20"),
        expect="<?xml",
        note="the URL the sweep asks for; refused first, last and in the middle",
        accept=XML_FIRST,
    ),
    Target(
        label="arxiv-rss",
        url="https://rss.arxiv.org/rss/q-fin.PM",
        expect="<?xml",
        note="arXiv's RSS host. Served every run so far, and the fallback if the API stays closed",
        accept="application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    ),
    # --- the keyed sources, asked before a single adapter is written --------
    #
    # Six keys are waiting to be registered as Actions secrets (ADR-0012), and
    # writing an adapter for a host that refuses this runner would waste the
    # same weekly cycles SEC and arXiv already cost. None of these needs a key
    # to answer: a "key missing" reply is an answer, and it proves the host is
    # reachable. What we are measuring here is the network, not the account.
    Target(
        label="dart-list",
        url="https://opendart.fss.or.kr/api/list.json",
        expect="{",
        note="DART filings (ADR-0012). Without a key it answers JSON with a status code, which is enough",
        accept="application/json",
    ),
    Target(
        label="ecos-tablelist",
        url="https://ecos.bok.or.kr/api/StatisticTableList/sample/json/kr/1/10/",
        expect="{",
        note="Bank of Korea statistics. `sample` is the vendor's own documented no-key path",
        accept="application/json",
    ),
    Target(
        label="fred-series",
        url="https://api.stlouisfed.org/fred/series?series_id=GDP",
        note="FRED macro series. Without a key it answers 400 with a message, which proves reachability",
        accept="application/json",
    ),
    Target(
        label="krx-openapi",
        url="https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd",
        note="KRX's keyed Open API, a different host from the website endpoint probed below",
        accept="application/json",
    ),
    Target(
        label="kis-paper",
        url="https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-price",
        note=(
            "Korea Investment paper trading, on its own port. A GET here is unauthenticated and will "
            "be refused; the point is whether the port answers at all from Actions"
        ),
        accept="application/json",
    ),
    Target(
        label="kis-real",
        url="https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/quotations/inquire-price",
        note="the live host, probed only for reachability. Nothing here ever reaches the order path",
        accept="application/json",
    ),
    Target(
        label="stooq-prices",
        url="https://stooq.com/q/d/l/?s=aapl.us&i=d",
        expect="Date,",
        note="a price fallback candidate; no key, and it serves CSV directly",
        accept="text/csv, */*;q=0.8",
    ),
    Target(
        label="frankfurter-fx",
        url="https://api.frankfurter.app/latest?from=USD&to=KRW",
        expect="{",
        note="FX for the Korean book, if ECOS stays behind a key",
        accept="application/json",
    ),
)


@dataclass
class Result:
    """What one host said, in the terms the next adapter needs."""

    label: str
    url: str
    note: str
    control: bool
    status: int | None = None
    reason: str = ""
    elapsed_ms: int = 0
    bytes_read: int = 0
    content_type: str = ""
    body_head: str = ""
    looks_right: bool | None = None
    verdict: str = ""
    redirected_to: str = ""
    headers_of_interest: dict[str, str] = field(default_factory=dict)


#: Response headers worth keeping. A WAF usually signs its own refusals, and
#: knowing which one is in front of a host is most of knowing what to try next.
INTERESTING = ("server", "via", "x-served-by", "cf-ray", "retry-after", "x-cache")


def _head(payload: bytes, encoding: str = "") -> str:
    """The readable start of a body, whatever it was compressed with.

    The first run of this script read 2 KB of a gzip stream and tried to
    `gzip.decompress` it, which cannot work: a truncated member has no trailer.
    It reported the control host as broken and the SEC's refusal as line noise
    -- two false readings out of eight, in the one tool whose whole job is not
    to produce false readings. A streaming decompressor takes what it is given
    and returns what it could read, which is what a sniff needs.
    """
    label = (encoding or "").strip().lower()
    if label in ("gzip", "x-gzip", "deflate"):
        wbits = zlib.MAX_WBITS | 16 if label != "deflate" else zlib.MAX_WBITS
        try:
            payload = zlib.decompressobj(wbits).decompress(payload)
        except zlib.error:
            try:  # a "deflate" body is raw deflate about as often as it is zlib
                payload = zlib.decompressobj(-zlib.MAX_WBITS).decompress(payload)
            except zlib.error as error:
                return f"[{label} body that would not decompress: {error}]"
    return " ".join(payload.decode("utf-8", errors="replace").split())[:BODY_CHARS]


def _encoding(headers: object) -> str:
    get = getattr(headers, "get", None)
    return get("Content-Encoding", "") if callable(get) else ""


def probe(target: Target, timeout: float = 30.0, opener: Callable | None = None) -> Result:
    """One request, one verdict. Never raises: a refusal is the measurement."""
    result = Result(label=target.label, url=target.url, note=target.note, control=target.control)
    request = urllib.request.Request(
        target.url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": target.accept,
            "Accept-Encoding": "gzip, deflate",
            "Host": urllib.parse.urlsplit(target.url).netloc,
            **dict(target.headers),
        },
    )
    open_url = opener or urllib.request.urlopen
    started = time.monotonic()
    try:
        with open_url(request, timeout=timeout) as response:  # noqa: S310 - fixed, declared hosts
            payload = response.read(SNIFF_BYTES)
            result.status = getattr(response, "status", 200)
            result.content_type = response.headers.get("Content-Type", "")
            result.bytes_read = len(payload)
            result.body_head = _head(payload, _encoding(response.headers))
            result.headers_of_interest = {
                name: response.headers.get(name, "") for name in INTERESTING if response.headers.get(name)
            }
            final = getattr(response, "url", target.url)
            if final and final != target.url:
                result.redirected_to = final
    except urllib.error.HTTPError as error:
        result.status = error.code
        result.reason = str(error.reason)
        result.content_type = error.headers.get("Content-Type", "") if error.headers else ""
        result.headers_of_interest = {
            name: error.headers.get(name, "")
            for name in INTERESTING
            if error.headers and error.headers.get(name)
        }
        try:
            body = error.read()
        except OSError:
            body = b""
        result.bytes_read = len(body)
        result.body_head = _head(body, _encoding(error.headers))
    except (TimeoutError, urllib.error.URLError, ssl.SSLError, OSError) as error:
        result.reason = f"{type(error).__name__}: {error}"
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    result.looks_right, result.verdict = _verdict(target, result)
    return result


def _verdict(target: Target, result: Result) -> tuple[bool | None, str]:
    """Name what happened, so the report can be skimmed instead of parsed.

    A 200 carrying an error page is the case worth naming separately: it is the
    one an adapter silently accepts and turns into an empty universe.
    """
    if result.status is None:
        return False, "unreachable"
    if result.status != 200:
        blank = " (empty body, so the host explained nothing)" if not result.bytes_read else ""
        return False, f"refused HTTP {result.status}{blank}"
    if target.expect and not result.body_head.lstrip().startswith(target.expect):
        return False, "answered 200 with something that is not what we asked for"
    return True, "ok"


#: Gap before any request, and before another request to a host already asked.
#: The first run of this prober taught it the hard way: it fired three requests
#: at arXiv inside a third of a second and every request after that was
#: refused, across two of arXiv's hosts (ADR-0020). That measured our own burst
#: rather than arXiv's reachability, which is the one thing a prober must not
#: do. The same-host gap is what keeps the next run's answer about the host.
GAP_SECONDS = 1.0
SAME_HOST_GAP_SECONDS = 5.0


def run(
    targets: Iterable[Target] = TARGETS,
    timeout: float = 30.0,
    pause: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    results = []
    asked: set[str] = set()
    for index, target in enumerate(targets):
        host = urllib.parse.urlsplit(target.url).netloc
        if index:
            pause(SAME_HOST_GAP_SECONDS if host in asked else GAP_SECONDS)
        asked.add(host)
        results.append(asdict(probe(target, timeout=timeout)))
    control = [r for r in results if r["control"]]
    return {
        "probed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "user_agent": USER_AGENT,
        # If the control did not answer, the run measured our own network and
        # every other line below is worthless. Say so in the record itself.
        "control_ok": all(r["looks_right"] for r in control) if control else None,
        "reachable": sorted(r["label"] for r in results if r["looks_right"]),
        "refused": sorted(r["label"] for r in results if not r["looks_right"]),
        "results": results,
    }


def as_markdown(report: dict[str, object]) -> str:
    lines = ["| host | verdict | status | ms | what it said |", "| --- | --- | --- | --- | --- |"]
    for r in report["results"]:  # type: ignore[union-attr]
        said = (r["body_head"] or r["reason"] or "")[:90].replace("|", "\\|")
        lines.append(
            f"| {r['label']} | {r['verdict']} | {r['status'] if r['status'] is not None else '-'} "
            f"| {r['elapsed_ms']} | {said} |"
        )
    if report["control_ok"] is False:
        lines.append("")
        lines.append("**The control host refused, so this run measured our own network, not theirs.**")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="registry/probes", help="directory for the probe record")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    report = run(timeout=args.timeout)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(UTC).date().isoformat()}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(as_markdown(report))
    print(f"\nwritten to {path}")
    # Always zero: a prober that fails the job cannot commit the reason.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
