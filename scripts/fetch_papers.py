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
#: A shared runner address gets rate-limited on somebody else's traffic.
RETRYABLE = (403, 429, 500, 502, 503, 504)
BACKOFF_SECONDS = (5.0, 20.0, 60.0)

#: arXiv answers **406 Not Acceptable with an empty body** (runs 36269801287,
#: 36270027635, 36270144373). An empty body is the problem: there is nothing to
#: read, so the usual "carry the server's own words back" does not apply and
#: three runs were spent guessing at a header one at a time. Adding an `Accept`
#: header -- the first guess -- did not change it.
#:
#: So stop guessing serially. The request shapes below are tried in order within
#: a single run and the one that answered is recorded in the report, which turns
#: a week of one-bit replies into one measurement.
#:
#: The order is a hypothesis, not a preference. `Accept-Encoding: identity` is
#: the one thing this client did that no browser does, and a WAF reading it as a
#: bot signature would produce exactly this: a refusal with no explanation. Every
#: shape still identifies us honestly -- none of them pretends to be a browser,
#: which arXiv asks of automated clients and which would make the next failure
#: undiagnosable again.
REQUEST_SHAPES: tuple[tuple[str, dict[str, str]], ...] = (
    ("gzip", {"User-Agent": USER_AGENT, "Accept": ACCEPT, "Accept-Encoding": "gzip, deflate"}),
    ("identity", {"User-Agent": USER_AGENT, "Accept": ACCEPT, "Accept-Encoding": "identity"}),
    ("bare", {"User-Agent": USER_AGENT}),
)
#: Which shape last worked, for the report. Set by `_get`.
LAST_SHAPE = ""


def _get(
    url: str,
    timeout: float = 45.0,
    attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET the Atom feed: every request shape, then the backoff, then give up.

    A 406 is not retried by waiting -- the server will say the same thing in a
    minute -- so it moves to the next shape immediately. A rate limit is the
    opposite, and waits.
    """
    global LAST_SHAPE  # noqa: PLW0603 - one process, one fetch, and the report needs it
    last = ""
    for attempt in range(attempts):
        worth_waiting = False
        for label, headers in REQUEST_SHAPES:
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
                    payload = response.read()
                    if response.headers.get("Content-Encoding", "").lower() == "gzip":
                        payload = gzip.decompress(payload)
                    LAST_SHAPE = label
                    return payload
            except urllib.error.HTTPError as error:
                last = f"HTTP {error.code} from arXiv ({label}): {_explain(error)}"
                if error.code in RETRYABLE:
                    worth_waiting = True
                    break  # a rate limit is not a shape problem; wait instead
                print(f"shape {label} refused -- {last}", file=sys.stderr)
            except OSError as error:  # timeout, DNS, refused proxy CONNECT
                last = f"{type(error).__name__} reaching arXiv ({label}): {error}"
                worth_waiting = True
                break
        if not worth_waiting:
            # Every shape was refused outright. The server will say the same
            # thing in a minute, and 85 seconds of waiting to hear it is 85
            # seconds of a weekly job pretending to be resilient.
            raise FetchError(f"every request shape was refused; last: {last}")
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
#: arXiv asks harvesters to leave a gap between requests. A weekly job paging
#: through one week of one archive makes a handful of them, so this costs
#: nothing and is the difference between a guest and a scraper.
OAI_PAUSE_SECONDS = 3.0
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
    """Every record each archive stamped on or after `since`, following tokens."""
    papers: list[Paper] = []
    for index, oai_set in enumerate(sets):
        if index:
            pause(OAI_PAUSE_SECONDS)
        url = oai_url(oai_set, since)
        for page in range(max_pages):
            if page:
                pause(OAI_PAUSE_SECONDS)
            batch, token = parse_oai(_get(url))
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


def query_url(categories: tuple[str, ...], max_results: int) -> str:
    search = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": search,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


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
    since = (today or datetime.now(UTC).date()) - timedelta(days=days)
    try:
        return harvest(since, pause=pause), "oai-pmh", ""
    except FetchError as harvest_error:
        try:
            return parse_feed(_get(query_url(categories, max_results))), "search-api", str(harvest_error)
        except FetchError as api_error:
            raise FetchError(
                f"neither arXiv interface answered. harvest: {harvest_error} -- search api: {api_error}"
            ) from api_error


def collect(
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    days: int = 7,
    max_results: int = 120,
    min_score: int = 3,
) -> dict[str, object]:
    papers, interface, degraded = gather(categories, days, max_results)

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
        "request_shape": LAST_SHAPE,
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
