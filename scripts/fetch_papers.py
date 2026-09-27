"""Fetch the week's quantitative-finance papers from arXiv and screen them deterministically.

Two halves, kept apart on purpose:

- **This script decides nothing.** It fetches, deduplicates, and scores each
  paper by counting matches against a fixed term list that mirrors the strategy
  registry and the gate vocabulary. The score is arithmetic anybody can redo
  from the committed record; it is a sort order, not a verdict.
- **The `literature-review` agent decides.** It reads the screened list and
  writes what each relevant paper would change here, which of our data we would
  need, and whether it is testable at all. That judgement is the point of having
  an agent, and it is the half a keyword count cannot do.

This runs on the GitHub Actions runner. Claude's research container has no route
to `export.arxiv.org` (measured: refused, like every vendor host), so the fetch
lives where the network is open -- the same split as market data in ADR-0007.

**What gets committed.** Title, authors, arXiv id, categories, dates, link and
abstract. arXiv states its *metadata* is available under CC0, and the abstract is
part of that metadata; the record has to carry the abstract or no later session
can review it, since the container cannot re-fetch. Full texts are a different
matter -- each paper's PDF carries its own licence -- so no PDF is ever
downloaded or committed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

from core.config import USER_AGENT

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ENDPOINT = "https://export.arxiv.org/api/query"

#: q-fin subclasses worth reading weekly, plus the two neighbours that carry
#: most of the method papers we actually use.
DEFAULT_CATEGORIES = (
    "q-fin.PM",  # portfolio management
    "q-fin.TR",  # trading and market microstructure
    "q-fin.ST",  # statistical finance
    "q-fin.RM",  # risk management
    "q-fin.CP",  # computational finance
    "econ.EM",  # econometrics
)

#: Terms and their weight. The high weights are the things that would change how
#: this repository decides something; the low ones are merely adjacent. Grouped
#: so the committed record can say which bucket a paper matched.
TERM_WEIGHTS: dict[str, tuple[tuple[str, ...], int]] = {
    "validation": (
        (
            "deflated sharpe",
            "probability of backtest overfitting",
            "backtest overfitting",
            "multiple testing",
            "multiple hypothesis",
            "data snooping",
            "p-hacking",
            "purged",
            "combinatorial cross-validation",
            "walk-forward",
            "look-ahead",
            "survivorship",
            "false discovery",
            "replication crisis",
        ),
        3,
    ),
    "strategy": (
        (
            "time series momentum",
            "time-series momentum",
            "cross-sectional momentum",
            "trend following",
            "trend-following",
            "short-term reversal",
            "betting against beta",
            "low volatility anomaly",
            "carry trade",
            "value premium",
            "quality minus junk",
            "factor momentum",
            "statistical arbitrage",
            "pairs trading",
        ),
        3,
    ),
    "risk": (
        (
            "volatility targeting",
            "risk parity",
            "drawdown control",
            "kelly",
            "position sizing",
            "covariance shrinkage",
            "ledoit",
            "tail risk",
            "expected shortfall",
        ),
        2,
    ),
    "execution": (
        (
            "market impact",
            "transaction cost",
            "implementation shortfall",
            "optimal execution",
            "capacity",
            "liquidity provision",
            "slippage",
        ),
        2,
    ),
    "korea": (("korea", "kospi", "kosdaq", "korean stock"), 2),
    "method": (
        (
            "machine learning",
            "deep learning",
            "reinforcement learning",
            "large language model",
            "regime",
            "nowcasting",
            "alternative data",
        ),
        1,
    ),
}


class FetchError(RuntimeError):
    """arXiv did not return a usable feed. Never swallowed into an empty week."""


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    categories: list[str]
    link: str
    abstract: str
    score: int = 0
    matched: dict[str, list[str]] = field(default_factory=dict)


ACCEPT = "application/atom+xml, application/xml;q=0.9, */*;q=0.8"

#: **Why arXiv's 406 is waited out and never worked around** (ADR-0020).
#:
#: The refusal is decided by the request, not by how many we have made. That
#: was settled by reordering the probe's target list: asked first,
#: `max_results=25` and `ListRecords` were still refused; asked last, after
#: five refusals, the baseline was still served. Seven hypotheses fell before
#: that one stood -- the Accept header, four request shapes, the Host header,
#: the User-Agent, the request size, this code bursting, and the rate limit
#: itself.
#:
#: What is served is what has been asked for before; what is refused is every
#: URL that is new, however small. So the sweep cannot vary its way out: a
#: different shape is a new URL, which is the one thing that reliably fails.
#:
#: Two things follow. A 406 is waited out, not varied. And **nothing here may
#: burst**: the shape ladder this file used to carry fired four requests back
#: to back, and every one of them was a new URL. It is gone, and `_wait_turn`
#: makes bursting impossible rather than merely discouraged.
RETRYABLE = (403, 406, 429, 500, 502, 503, 504)
BACKOFF_SECONDS = (30.0, 60.0, 180.0)

#: arXiv asks for no more than one request every three seconds. This is that,
#: enforced in the one place every request passes through.
MIN_INTERVAL_SECONDS = 3.0
_last_call = 0.0


def _wait_turn(sleep: Callable[[float], None], clock: Callable[[], float] = time.monotonic) -> None:
    """Hold until three seconds after the last request, whoever made it."""
    global _last_call  # noqa: PLW0603 - one process, one rate budget
    gap = MIN_INTERVAL_SECONDS - (clock() - _last_call)
    if gap > 0:
        sleep(gap)
    _last_call = clock()


def _get(
    url: str,
    timeout: float = 45.0,
    attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """GET, one request at a time, waiting out a refusal rather than varying it.

    There is no ladder of request shapes any more. Four of them were tried and
    all four were red herrings, and firing them back to back was itself part of
    what triggered the refusal.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": ACCEPT,
        "Accept-Encoding": "gzip, deflate",
        "Host": urllib.parse.urlsplit(url).netloc,
    }
    last = ""
    for attempt in range(attempts):
        _wait_turn(sleep, clock)
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed host
                urllib.request.Request(url, headers=headers), timeout=timeout
            ) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    payload = gzip.decompress(payload)
                return payload
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code} from arXiv: {_explain(error)}"
            if error.code not in RETRYABLE:
                raise FetchError(last) from error
        except OSError as error:  # timeout, DNS, refused proxy CONNECT
            last = f"{type(error).__name__} reaching arXiv: {error}"
        if attempt < attempts - 1:
            delay = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
            print(f"retrying in {delay:.0f}s -- {last}", file=sys.stderr)
            sleep(delay)
    raise FetchError(f"{attempts} attempts failed; last: {last}")


def _explain(error: urllib.error.HTTPError) -> str:
    """What the server actually said, trimmed.

    arXiv has answered 406 twice now (runs 36269801287 and 36270027635) and the
    Accept header did not change it, so the next run has to say more than the
    status code. Guessing at a third party's WAF one runner cycle at a time is
    not diagnosis; carrying its own words back is. This is the same fix
    `scripts/fetch_listings.py` needed, and it paid for itself there in one run.
    """
    try:
        body = error.read()
    except OSError:
        return "no response body"
    text = " ".join(body.decode("utf-8", errors="replace").split())
    return text[:300] or "empty response body"


def _text(node, path: str) -> str:
    found = node.find(path)
    return " ".join(found.text.split()) if found is not None and found.text else ""


# --- the harvest interface, which is the one that answers -------------------

#: `export.arxiv.org` refuses this runner with 406 and an empty body, and every
#: honest request shape gets the same. `oaipmh.arxiv.org` answers 200 from the
#: same runner in the same minute (registry/probes/2026-09-26.json), so the
#: sweep moves there. It is also the interface arXiv built for this: we are
#: harvesting a date range, not searching.
OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"
OAI = "{http://www.openarchives.org/OAI/2.0/}"
OAI_ARXIV = "{http://arxiv.org/OAI/arXiv/}"

#: OAI takes one set per request, and our categories span two archives.
#: Harvesting the parent set and filtering by category afterwards is one
#: request per archive instead of one per subclass, which is politer and gives
#: the same answer -- the category filter is `DEFAULT_CATEGORIES` either way.
OAI_SETS = ("q-fin", "econ")
#: A resumption loop with no bound is a way to hang a runner on somebody else's
#: bug. One week of q-fin is a page or two; twenty is already absurd.
OAI_MAX_PAGES = 20


def oai_url(oai_set: str, since: date, until: date | None = None) -> str:
    params = {"verb": "ListRecords", "set": oai_set, "metadataPrefix": "arXiv", "from": since.isoformat()}
    if until is not None:
        params["until"] = until.isoformat()
    return f"{OAI_ENDPOINT}?{urllib.parse.urlencode(params)}"


def oai_resume_url(token: str) -> str:
    return f"{OAI_ENDPOINT}?{urllib.parse.urlencode({'verb': 'ListRecords', 'resumptionToken': token})}"


def parse_oai(body: bytes) -> tuple[list[Paper], str]:
    """Parse one `ListRecords` page into papers and the token for the next.

    `noRecordsMatch` is the one OAI error that is not a failure: it means the
    window really was empty, which a quiet week can be. Every other error code
    is arXiv telling us the request was wrong, and a sweep that swallows it
    reports an empty week that never happened.
    """
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise FetchError(f"arXiv returned {body[:120]!r}, which is not OAI-PMH: {error}") from error
    if not root.tag.endswith("OAI-PMH"):
        raise FetchError(f"arXiv returned a <{root.tag}> document, not an OAI-PMH envelope")

    failure = root.find(f"{OAI}error")
    if failure is not None:
        code = failure.get("code", "")
        if code == "noRecordsMatch":
            return [], ""
        raise FetchError(f"arXiv OAI error {code!r}: {(failure.text or '').strip()}")

    listing = root.find(f"{OAI}ListRecords")
    if listing is None:
        raise FetchError("the OAI response carries neither an error nor a ListRecords element")

    papers: list[Paper] = []
    for record in listing.findall(f"{OAI}record"):
        header = record.find(f"{OAI}header")
        # A deleted record has no metadata. Skipping it is right: it is a paper
        # arXiv withdrew, and reviewing a withdrawal wastes the agent's turn.
        if header is not None and header.get("status") == "deleted":
            continue
        meta = record.find(f"{OAI}metadata/{OAI_ARXIV}arXiv")
        if meta is None:
            continue
        arxiv_id = _text(meta, f"{OAI_ARXIV}id")
        if not arxiv_id:
            continue
        categories = _text(meta, f"{OAI_ARXIV}categories").split()
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_text(meta, f"{OAI_ARXIV}title"),
                authors=_oai_authors(meta),
                published=_text(meta, f"{OAI_ARXIV}created"),
                updated=_text(meta, f"{OAI_ARXIV}updated") or _text(meta, f"{OAI_ARXIV}created"),
                # arXiv lists the primary category first in this field.
                primary_category=categories[0] if categories else "",
                categories=categories,
                link=f"https://arxiv.org/abs/{arxiv_id}",
                abstract=_text(meta, f"{OAI_ARXIV}abstract"),
            )
        )

    token = listing.find(f"{OAI}resumptionToken")
    return papers, (token.text or "").strip() if token is not None else ""


def _oai_authors(meta) -> list[str]:
    """`<author><keyname>…</keyname><forenames>…</forenames></author>` as one string.

    Forenames first, so the name reads the way the paper prints it. A record
    with only a keyname is normal (collaborations, single-name authors) and is
    kept rather than dropped.
    """
    out: list[str] = []
    for author in meta.findall(f"{OAI_ARXIV}authors/{OAI_ARXIV}author"):
        keyname = _text(author, f"{OAI_ARXIV}keyname")
        forenames = _text(author, f"{OAI_ARXIV}forenames")
        name = " ".join(part for part in (forenames, keyname) if part)
        if name:
            out.append(name)
    return out


def harvest(
    since: date,
    sets: tuple[str, ...] = OAI_SETS,
    pause: Callable[[float], None] = time.sleep,
    max_pages: int = OAI_MAX_PAGES,
) -> list[Paper]:
    """Every record each archive stamped on or after `since`, following tokens.

    `pause` is handed to `_get` rather than called here: the three-second gap
    is one gate for the whole process, so counting it in two places would
    either double the wait or, worse, let one caller skip it.
    """
    papers: list[Paper] = []
    for oai_set in sets:
        url = oai_url(oai_set, since)
        for _page in range(max_pages):
            batch, token = parse_oai(_get(url, sleep=pause))
            papers.extend(batch)
            if not token:
                break
            url = oai_resume_url(token)
        else:
            raise FetchError(
                f"the {oai_set} harvest did not finish in {max_pages} pages; "
                "either the window is far wider than a week or the token is looping"
            )
    return papers


#: Papers per search request. `max_results=1` is served and `max_results=25` is
#: refused, every run and in any position, but that is not about the number:
#: what the two differ in is whether the URL has been asked for before
#: (ADR-0020). The paging stays on its own merits -- a page that fails costs
#: one retry window rather than the week -- and the pacing gate in `_get`
#: makes the extra requests cost time rather than goodwill.
PAGE_SIZE = 25
#: Pages per category before giving up on it. A week of one q-fin subclass is a
#: page or two; ten is already a sign the window or the sort is wrong.
MAX_PAGES = 10


def query_url(categories: tuple[str, ...], max_results: int, start: int = 0) -> str:
    search = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": search,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def search(
    categories: tuple[str, ...],
    since: datetime,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    pause: Callable[[float], None] = time.sleep,
) -> list[Paper]:
    """One category at a time, one page at a time, stopping at the window edge.

    The results are sorted newest first, so a page whose entries are all older
    than the window means this category is done -- there is nothing further
    back that we want. That, and not the page count, is what normally ends the
    loop; `max_pages` is only there so a sort that is not what we think it is
    cannot spin.
    """
    papers: list[Paper] = []
    for category in categories:
        for page in range(max_pages):
            url = query_url((category,), page_size, start=page * page_size)
            batch = parse_feed(_get(url, sleep=pause))
            papers.extend(batch)
            if len(batch) < page_size or not any(within(paper, since) for paper in batch):
                break
    if not papers:
        raise FetchError(f"the search API returned no entries at all for {len(categories)} categories")
    return papers


def parse_feed(body: bytes) -> list[Paper]:
    """Parse the Atom feed. Anything that is not a feed with entries is an error."""
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise FetchError(f"arXiv returned {body[:120]!r}, which is not Atom: {error}") from error
    if not root.tag.endswith("feed"):
        raise FetchError(f"arXiv returned a <{root.tag}> document, not a feed")

    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = _text(entry, f"{ATOM}id")
        if not raw_id:
            continue
        # http://arxiv.org/abs/2509.01234v2 -> 2509.01234
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
        primary = entry.find(f"{ARXIV}primary_category")
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_text(entry, f"{ATOM}title"),
                authors=[
                    " ".join(name.text.split())
                    for name in entry.findall(f"{ATOM}author/{ATOM}name")
                    if name.text
                ],
                published=_text(entry, f"{ATOM}published"),
                updated=_text(entry, f"{ATOM}updated"),
                primary_category=primary.get("term", "") if primary is not None else "",
                categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category")],
                link=f"https://arxiv.org/abs/{arxiv_id}",
                abstract=_text(entry, f"{ATOM}summary"),
            )
        )
    if not papers:
        raise FetchError("arXiv returned a feed with no entries; a silent empty week is not a week")
    return papers


def score(paper: Paper) -> Paper:
    """Count term matches in title and abstract. Arithmetic, reproducible, not a verdict."""
    haystack = f"{paper.title} {paper.abstract}".lower()
    total = 0
    matched: dict[str, list[str]] = {}
    for bucket, (terms, weight) in TERM_WEIGHTS.items():
        hits = sorted({term for term in terms if term in haystack})
        if hits:
            matched[bucket] = hits
            total += weight * len(hits)
    paper.score = total
    paper.matched = matched
    return paper


def within(paper: Paper, since: datetime) -> bool:
    stamp = paper.published or paper.updated
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        # The search API dates carry an offset and the harvest dates are bare
        # days. Reading a bare day as UTC midnight is the generous reading: it
        # keeps a paper stamped on the boundary rather than dropping it.
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return when >= since
    except ValueError:
        # An unparseable date is kept rather than dropped: losing a paper silently
        # is worse than reviewing one that turns out to be a week old.
        return True


def gather(
    categories: tuple[str, ...],
    days: int,
    max_results: int,
    today: date | None = None,
    pause: Callable[[float], None] = time.sleep,
) -> tuple[list[Paper], str, str]:
    """The week's papers, from whichever arXiv interface answers.

    The harvest interface is tried first because it is the one that answers and
    the one built for this. The search API stays as the fallback rather than
    being deleted: it is the same publisher, its refusal may be temporary, and
    a fallback that was never exercised is not a fallback. Which one answered
    goes in the report, so a silent switch is not possible.
    """
    day = today or datetime.now(UTC).date()
    since = day - timedelta(days=days)
    window = datetime.combine(since, datetime.min.time(), tzinfo=UTC)
    try:
        return harvest(since, pause=pause), "oai-pmh", ""
    except FetchError as harvest_error:
        try:
            page_size = min(max_results, PAGE_SIZE)
            return (
                search(categories, window, page_size=page_size, pause=pause),
                "search-api",
                str(harvest_error),
            )
        except FetchError as api_error:
            raise FetchError(
                f"neither arXiv interface answered. harvest: {harvest_error} -- search api: {api_error}"
            ) from api_error


def collect(
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    days: int = 7,
    max_results: int = 120,
    min_score: int = 3,
    pause: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    papers, interface, degraded = gather(categories, days, max_results, pause=pause)

    since = datetime.now(UTC) - timedelta(days=days)
    wanted = set(categories)
    seen: set[str] = set()
    fresh: list[Paper] = []
    for paper in papers:
        # The harvest returns whole archives, so the category filter that the
        # search query used to carry has to be applied here instead.
        if not wanted.intersection(paper.categories or [paper.primary_category]):
            continue
        if paper.arxiv_id in seen or not within(paper, since):
            continue
        seen.add(paper.arxiv_id)
        fresh.append(score(paper))

    shortlist = sorted(
        (p for p in fresh if p.score >= min_score),
        key=lambda p: (-p.score, p.arxiv_id),
    )
    return {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "week": datetime.now(UTC).strftime("%G-W%V"),
        "query": {
            "categories": list(categories),
            "days": days,
            "max_results": max_results,
            "min_score": min_score,
        },
        "interface": interface,
        "degraded": degraded,
        "returned": len(papers),
        "in_window": len(fresh),
        "shortlisted": len(shortlist),
        "papers": [asdict(p) for p in shortlist],
        "screened_out": [
            {"arxiv_id": p.arxiv_id, "title": p.title, "score": p.score}
            for p in sorted(fresh, key=lambda p: p.arxiv_id)
            if p.score < min_score
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-results", type=int, default=120)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--out", default="registry/literature")
    args = parser.parse_args(argv)

    categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    report = collect(categories, args.days, args.max_results, args.min_score)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['week']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")

    print(
        f"{report['shortlisted']} of {report['in_window']} papers shortlisted -> {path} "
        f"(via {report['interface']}, {report['returned']} records read)"
    )
    if report["degraded"]:
        print(f"the harvest interface was unavailable: {report['degraded']}")
    for paper in report["papers"][:10]:
        print(f"  [{paper['score']:>2}] {paper['arxiv_id']}  {paper['title'][:88]}")
    if not report["shortlisted"]:
        # Not an error: a quiet week is a real outcome. The agent still reports it.
        print("no paper cleared the screen this week", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
