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
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ENDPOINT = "https://export.arxiv.org/api/query"
USER_AGENT = "quant-desk/0.1 (research; +https://github.com/InnbumBaek/quant-desk)"

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


def _get(url: str, timeout: float = 45.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
            return response.read()
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code} from arXiv") from error
    except OSError as error:  # timeout, DNS, refused proxy CONNECT
        raise FetchError(f"{type(error).__name__} reaching arXiv: {error}") from error


def _text(node, path: str) -> str:
    found = node.find(path)
    return " ".join(found.text.split()) if found is not None and found.text else ""


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
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")) >= since
    except ValueError:
        # An unparseable date is kept rather than dropped: losing a paper silently
        # is worse than reviewing one that turns out to be a week old.
        return True


def collect(
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    days: int = 7,
    max_results: int = 120,
    min_score: int = 3,
) -> dict[str, object]:
    body = _get(query_url(categories, max_results))
    papers = parse_feed(body)

    since = datetime.now(UTC) - timedelta(days=days)
    seen: set[str] = set()
    fresh: list[Paper] = []
    for paper in papers:
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

    print(f"{report['shortlisted']} of {report['in_window']} papers shortlisted -> {path}")
    for paper in report["papers"][:10]:
        print(f"  [{paper['score']:>2}] {paper['arxiv_id']}  {paper['title'][:88]}")
    if not report["shortlisted"]:
        # Not an error: a quiet week is a real outcome. The agent still reports it.
        print("no paper cleared the screen this week", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
