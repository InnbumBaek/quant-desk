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
from email.utils import parsedate_to_datetime
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
    #: new / cross / replace / replace-cross, or "unknown" when the feed did not
    #: say. Only the RSS interface reports it (ADR-0021).
    announce_type: str = ""
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


# --- the daily announcement feed, the one arXiv host that answers ------------

#: **Why the sweep moved here** (ADR-0021). `export.arxiv.org` and
#: `oaipmh.arxiv.org` refuse this runner's address range at origin: every 200 we
#: ever saw from them was a Fastly edge hit on a URL somebody else had already
#: asked for (ADR-0020). `rss.arxiv.org` has answered every probe, every run.
#:
#: The cost is that RSS is not a query. It is **today's announcements**, with no
#: date range and no paging, so the sweep runs daily and accumulates a week
#: instead of asking for one. That is why the store below exists.
DEFAULT_OUT = Path("registry/literature")

RSS_BASE = "https://rss.arxiv.org/rss"
RSS_DC = "{http://purl.org/dc/elements/1.1/}"

#: One feed per archive, not per subclass: `q-fin` carries every q-fin.* paper
#: and `DEFAULT_CATEGORIES` filters afterwards, which is one request instead of
#: six for the same answer.
RSS_FEEDS = ("q-fin", "econ.EM")

#: arXiv marks each item as new, cross, replace or replace-cross. A replacement
#: is a paper this desk has already seen and reviewed, so only the first two are
#: kept -- but an announce type we cannot read is kept, because dropping a paper
#: on a field we failed to parse is the expensive direction.
ANNOUNCE_KEEP = ("new", "cross")
ANNOUNCE_UNKNOWN = "unknown"

#: How many stored-and-empty days in a row stop being a weekend and start being
#: a feed we can no longer read. arXiv's longest normal gap is a weekend plus a
#: holiday Monday, so four consecutive empty days is not a calendar fact.
#: Without this, a feed that silently changed shape looks exactly like a quiet
#: stretch -- which is the one failure ADR-0021 exists to prevent.
EMPTY_DAY_ALARM = 4


def _today() -> date:
    """The run date, in one place so `main` can be tested on a fixed day."""
    return datetime.now(UTC).date()


def rss_url(feed: str) -> str:
    return f"{RSS_BASE}/{feed}"


def _announce_type(item, description: str) -> tuple[str, str]:
    """The announce type and where it was found.

    Two readings, because nobody here has held this feed: the documented
    `arxiv:announce_type` element, then the prefix arXiv writes into the
    description. Both absent is recorded rather than guessed, and the report
    carries which reading worked so the first run settles the shape.
    """
    node = item.find(f"{ARXIV}announce_type")
    if node is not None and node.text and node.text.strip():
        return node.text.strip().lower(), "element"
    match = re.search(r"Announce\s+Type:\s*([a-z][a-z-]*)", description, re.IGNORECASE)
    if match:
        return match.group(1).lower(), "description"
    return ANNOUNCE_UNKNOWN, "absent"


def _rss_abstract(description: str) -> str:
    """The abstract out of the description, which arXiv prefixes with id and type."""
    text = " ".join(description.split())
    marker = re.search(r"Abstract:\s*", text, re.IGNORECASE)
    return text[marker.end() :].strip() if marker else text


def _bare_id(raw: str) -> str:
    """`2509.01234v2` or `oai:arXiv.org:2509.01234v2` -> `2509.01234`.

    The version is stripped so a replacement announcement cannot enter the store
    as a second paper.
    """
    tail = raw.strip().rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def _channel_day(channel, fallback: date) -> tuple[str, str]:
    """The feed's own announcement day, and how it was obtained."""
    for field_name in ("pubDate", "lastBuildDate"):
        stamp = _text(channel, field_name)
        if not stamp:
            continue
        try:
            return parsedate_to_datetime(stamp).astimezone(UTC).date().isoformat(), field_name
        except (TypeError, ValueError):
            continue
    # No readable date on the feed. The run date is the honest substitute --
    # this feed is today's announcements -- and the report says so.
    return fallback.isoformat(), "run_date"


def _skip_days(channel) -> list[str]:
    """The days the feed itself says it does not publish on.

    Recorded rather than acted on: it is the feed's own explanation for an empty
    day, and having it in the report is what separates "arXiv does not announce
    on Sundays" from "we stopped being able to read this feed".
    """
    node = channel.find("skipDays")
    if node is None:
        return []
    return [day.text.strip() for day in node.findall("day") if day.text and day.text.strip()]


def parse_rss(body: bytes, feed: str = "", today: date | None = None) -> tuple[list[Paper], dict]:
    """One RSS feed into papers, plus what shape it actually had.

    The shape report is the point of the first run: the field names came from
    arXiv's documentation and nothing here has parsed this feed before, so a
    successful run has to say what it saw (the discipline `core/data/krx.py`
    set, ADR-0022).
    """
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise FetchError(f"{feed}: arXiv returned {body[:120]!r}, which is not XML: {error}") from error
    channel = root.find("channel")
    if channel is None:
        raise FetchError(
            f"{feed}: no <channel> in a <{root.tag}> document. Its children: "
            f"{', '.join(child.tag for child in root) or '(none)'}"
        )
    channel_fields = sorted({child.tag for child in channel})
    skip_days = _skip_days(channel)
    day, day_from = _channel_day(channel, today or _today())

    items = channel.findall("item")
    if not items:
        # Measured 2026-09-27, a Sunday, on the first live run: the channel came
        # back whole (pubDate, lastBuildDate, skipDays, title, ...) and carried no
        # item of any namespace. That is arXiv answering "nothing was announced",
        # not arXiv refusing, and the difference is the whole point of ADR-0021 --
        # so it is an empty day in the store, not a fetch failure that sends the
        # run down to two origin-blocked fallbacks. A feed that goes quiet for
        # good is caught by `empty_days_in_a_row`, not by failing one Sunday.
        return [], {
            "feed": feed,
            "items": 0,
            "kept": 0,
            "announce_day": day,
            "announce_day_from": day_from,
            "announce_type_from": {},
            "announce_types": {},
            "item_fields": [],
            "channel_fields": channel_fields,
            "skip_days": skip_days,
        }

    papers: list[Paper] = []
    announce_from: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for item in items:
        description = _text(item, "description")
        link = _text(item, "link")
        raw = _text(item, "guid") or link
        if not raw:
            raise FetchError(
                f"{feed}: an item carries neither guid nor link, so it cannot be identified. "
                f"It does carry: {', '.join(sorted({child.tag for child in item}))}"
            )
        kind, source = _announce_type(item, description)
        announce_from[source] = announce_from.get(source, 0) + 1
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind not in ANNOUNCE_KEEP and kind != ANNOUNCE_UNKNOWN:
            continue
        arxiv_id = _bare_id(raw)
        creator = _text(item, f"{RSS_DC}creator")
        categories = [
            " ".join(node.text.split())
            for node in item.findall("category")
            if node.text and node.text.strip()
        ]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_text(item, "title"),
                authors=[name.strip() for name in creator.split(",") if name.strip()],
                published=day,
                updated=day,
                primary_category=categories[0] if categories else "",
                categories=categories,
                link=link or f"https://arxiv.org/abs/{arxiv_id}",
                abstract=_rss_abstract(description),
                announce_type=kind,
            )
        )

    shape = {
        "feed": feed,
        "items": len(items),
        "kept": len(papers),
        "announce_day": day,
        "announce_day_from": day_from,
        "announce_type_from": announce_from,
        "announce_types": kinds,
        "item_fields": sorted({child.tag for child in items[0]}),
        "channel_fields": channel_fields,
        "skip_days": skip_days,
    }
    return papers, shape


def fetch_rss(
    feeds: tuple[str, ...] = RSS_FEEDS,
    pause: Callable[[float], None] = time.sleep,
    today: date | None = None,
) -> tuple[list[Paper], list[dict]]:
    """Every archive's feed. One refusal is fatal: a partial day looks like a quiet one."""
    papers: list[Paper] = []
    shapes: list[dict] = []
    for feed in feeds:
        batch, shape = parse_rss(_get(rss_url(feed), sleep=pause), feed=feed, today=today)
        papers.extend(batch)
        shapes.append(shape)
    return papers, shapes


# --- the store, because a daily feed cannot answer a weekly question ---------

#: One file per announcement day under here. A day that was never fetched has no
#: file, and a day that was fetched and held nothing has an empty one: those are
#: different facts, and the weekly report reports both (ADR-0021).
STORE = "announced"


def store_dir(directory: Path) -> Path:
    return directory / STORE


def _iso_day(stamp: str, fallback: str) -> str:
    """The day part of whatever date shape an interface used."""
    head = (stamp or "").strip()[:10]
    try:
        return date.fromisoformat(head).isoformat()
    except ValueError:
        return fallback


def group_by_day(papers: list[Paper], fallback: str) -> dict[str, list[Paper]]:
    groups: dict[str, list[Paper]] = {}
    for paper in papers:
        day = _iso_day(paper.published or paper.updated, fallback)
        groups.setdefault(day, []).append(paper)
    return groups


def write_days(groups: dict[str, list[Paper]], directory: Path, days_asked: list[str]) -> list[str]:
    """Merge each day's papers into its file, and write an empty file for a
    quiet day that was asked for.

    Merging rather than replacing: a later run must not delete what an earlier
    one recorded, and a fallback interface backfilling three days at once is a
    recovery, not an overwrite.
    """
    out = store_dir(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for day in sorted(set(groups) | set(days_asked)):
        path = out / f"{day}.json"
        existing: dict[str, dict] = {}
        if path.exists():
            try:
                for row in json.loads(path.read_text(encoding="utf-8")).get("papers", []):
                    existing[str(row.get("arxiv_id"))] = row
            except (OSError, json.JSONDecodeError, AttributeError):
                # A corrupt day file is replaced rather than allowed to stop the
                # run, and the replacement is what this run measured.
                existing = {}
        for paper in groups.get(day, []):
            existing.setdefault(paper.arxiv_id, asdict(paper))
        path.write_text(
            json.dumps(
                {
                    "day": day,
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "count": len(existing),
                    "papers": [existing[key] for key in sorted(existing)],
                },
                indent=1,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(day)
    return written


def empty_days_in_a_row(directory: Path, today: date, limit: int = 30) -> int:
    """Consecutive stored-and-empty days ending today.

    A *missing* day breaks the count rather than extending it: we did not ask, so
    we did not measure quiet. Only a day we fetched and found nothing in counts,
    which is why `write_days` writes the empty file at all.
    """
    out = store_dir(directory)
    streak = 0
    for offset in range(limit + 1):
        path = out / f"{(today - timedelta(days=offset)).isoformat()}.json"
        if not path.exists():
            break
        try:
            papers = json.loads(path.read_text(encoding="utf-8")).get("papers", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            break
        if papers:
            break
        streak += 1
    return streak


def read_window(directory: Path, days: int, today: date) -> tuple[list[Paper], list[str], list[str]]:
    """Every paper stored in the last `days` days, and which days are missing.

    The missing list is the whole reason this returns three things. A weekly
    report built from four stored days is not a week, and one that does not say
    so reads exactly like a quiet week.
    """
    out = store_dir(directory)
    papers: list[Paper] = []
    present: list[str] = []
    missing: list[str] = []
    for offset in range(days, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        path = out / f"{day}.json"
        if not path.exists():
            missing.append(day)
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8")).get("papers", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            missing.append(day)
            continue
        present.append(day)
        for row in rows:
            fields = {key: row.get(key) for key in Paper.__dataclass_fields__ if key in row}
            papers.append(Paper(**fields))
    return papers, present, missing


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


def seed_from_weekly(directory: Path = DEFAULT_OUT) -> list[str]:
    """Put the papers already committed in weekly files into the day store.

    Run once, when the sweep moved to the daily feed (ADR-0021): the store starts
    empty and the first weekly rebuild would otherwise be poorer than the file it
    replaces. Only the full records in `papers` can be seeded -- a `screened_out`
    entry carries an id, a title and a score but no abstract, and a stub cannot be
    re-scored. Writing one anyway would put a fabricated score in the store, which
    is worse than the gap; the pre-migration files are kept beside the new ones
    instead.
    """
    groups: dict[str, list[Paper]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = report.get("papers") if isinstance(report, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not row.get("arxiv_id"):
                continue
            fields = {key: row.get(key) for key in Paper.__dataclass_fields__ if key in row}
            paper = Paper(**fields)
            day = _iso_day(paper.published or paper.updated, "")
            if day:
                groups.setdefault(day, []).append(paper)
    return write_days(groups, directory, [])


def within(paper: Paper, since: datetime) -> bool:
    """Whether a paper is inside the window. The search API's paging stop.

    Only the search fallback needs this now: the store answers the window
    question by filename, which is why a day that was never fetched is
    distinguishable from a day that held nothing (ADR-0021).
    """
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


def fetch_today(
    feeds: tuple[str, ...] = RSS_FEEDS,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    days: int = 7,
    max_results: int = 120,
    today: date | None = None,
    pause: Callable[[float], None] = time.sleep,
) -> tuple[list[Paper], str, str, list[dict]]:
    """Today's announcements, from whichever arXiv interface answers.

    RSS first, because it is the only arXiv host that has ever answered this
    runner (ADR-0020, ADR-0021). The other two stay as fallbacks rather than
    being deleted: they are the same publisher, an origin block can lift, and a
    fallback that was never exercised is not a fallback. Which one answered goes
    in the report, so a silent switch is not possible.

    The fallbacks return a window rather than a day. Their papers are grouped by
    their own dates on the way into the store, so a recovery run after three
    dark days fills three day files instead of piling a week onto one.
    """
    day = today or _today()
    try:
        papers, shapes = fetch_rss(feeds, pause=pause, today=day)
        return papers, "rss", "", shapes
    except FetchError as rss_error:
        since = day - timedelta(days=days)
        try:
            return harvest(since, pause=pause), "oai-pmh", str(rss_error), []
        except FetchError as harvest_error:
            window = datetime.combine(since, datetime.min.time(), tzinfo=UTC)
            try:
                page_size = min(max_results, PAGE_SIZE)
                return (
                    search(categories, window, page_size=page_size, pause=pause),
                    "search-api",
                    f"rss: {rss_error} -- harvest: {harvest_error}",
                    [],
                )
            except FetchError as api_error:
                raise FetchError(
                    f"no arXiv interface answered. rss: {rss_error} -- "
                    f"harvest: {harvest_error} -- search api: {api_error}"
                ) from api_error


def collect(
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    days: int = 7,
    max_results: int = 120,
    min_score: int = 3,
    pause: Callable[[float], None] = time.sleep,
    directory: Path = DEFAULT_OUT,
    today: date | None = None,
    fetch: bool = True,
) -> dict[str, object]:
    """Fetch today into the store, then build the week's shortlist from the store.

    Both halves run every day. The day file is the irreplaceable part -- a daily
    feed cannot be asked for yesterday -- and rebuilding the weekly file each
    time costs nothing and means a missed day loses one day rather than a week.
    """
    day = today or _today()
    interface, degraded, shapes, stored_today = "store-only", "", [], []
    if fetch:
        stored_today, interface, degraded, shapes = fetch_today(
            categories=categories, days=days, max_results=max_results, today=day, pause=pause
        )
        write_days(group_by_day(stored_today, day.isoformat()), directory, [day.isoformat()])

    papers, present, missing = read_window(directory, days, day)

    wanted = set(categories)
    seen: set[str] = set()
    fresh: list[Paper] = []
    for paper in papers:
        # The feeds carry whole archives, so the category filter that the search
        # query used to carry is applied here instead.
        if not wanted.intersection(paper.categories or [paper.primary_category]):
            continue
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        fresh.append(score(paper))

    shortlist = sorted(
        (p for p in fresh if p.score >= min_score),
        key=lambda p: (-p.score, p.arxiv_id),
    )
    return {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "week": day.strftime("%G-W%V"),
        "query": {
            "categories": list(categories),
            "days": days,
            "max_results": max_results,
            "min_score": min_score,
        },
        "interface": interface,
        "degraded": degraded,
        "feed_shape": shapes,
        "announced_today": len(stored_today),
        # A week built from four stored days is not a week, and one that does not
        # say so reads exactly like a quiet week (ADR-0021).
        "days_present": present,
        "days_missing": missing,
        # An empty day is a real outcome; a run of them is a broken reader.
        "empty_days_in_a_row": empty_days_in_a_row(directory, day),
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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="rebuild the weekly list from the store without asking arXiv",
    )
    parser.add_argument(
        "--seed-from-weekly",
        action="store_true",
        help="one-off: seed the day store from the weekly files already committed",
    )
    args = parser.parse_args(argv)

    categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    out = Path(args.out)
    if args.seed_from_weekly:
        seeded = seed_from_weekly(out)
        print(f"seeded {len(seeded)} day(s) into the store: {', '.join(seeded) or '(none)'}")
        return 0
    report = collect(
        categories,
        args.days,
        args.max_results,
        args.min_score,
        directory=out,
        fetch=not args.no_fetch,
    )

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['week']}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")

    print(
        f"{report['shortlisted']} of {report['in_window']} papers shortlisted -> {path} "
        f"(via {report['interface']}, {report['announced_today']} announced today)"
    )
    missing = report["days_missing"]
    if isinstance(missing, list) and missing:
        # Said out loud every run: a short window is the one failure that looks
        # exactly like a quiet week.
        print(f"- the store is missing {len(missing)} day(s) of the window: {', '.join(missing)}")
    for shape in report["feed_shape"] or []:
        fields = shape["item_fields"] or (
            f"(no items; channel carried {shape['channel_fields']}, "
            f"skipDays {shape['skip_days'] or '(none)'})"
        )
        print(f"- {shape['feed']}: {shape['kept']}/{shape['items']} items kept, fields {fields}")
    if report["degraded"]:
        print(f"- the RSS feed was unavailable: {report['degraded']}")
    for paper in report["papers"][:10]:
        print(f"  [{paper['score']:>2}] {paper['arxiv_id']}  {paper['title'][:88]}")
    if not report["shortlisted"]:
        # Not an error: a quiet week is a real outcome. The agent still reports it.
        print("no paper cleared the screen this week", file=sys.stderr)
    streak = report["empty_days_in_a_row"]
    if isinstance(streak, int) and streak >= EMPTY_DAY_ALARM:
        # The day files are already written, so this fails after the irreplaceable
        # part is on disk ("commit first, red afterwards").
        print(
            f"{streak} stored days in a row held no paper, which is longer than any "
            f"arXiv weekend. The feed is no longer being read: check the channel "
            f"fields printed above against a browser.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
