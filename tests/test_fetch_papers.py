"""The paper fetcher is tested for what it refuses and for how it scores.

No test touches the network: the container has no route to export.arxiv.org, so
a test that needed one would fail for the wrong reason. Every case runs against
a recorded Atom body.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from scripts import fetch_papers as fp

#: Tests never wait on arXiv's politeness gap; the gap itself is tested directly.
NO_PAUSE = lambda _seconds: None  # noqa: E731


def entry(
    arxiv_id: str = "2509.01234v1",
    title: str = "A note on something",
    abstract: str = "Nothing of interest here.",
    published: str | None = None,
    category: str = "q-fin.PM",
) -> str:
    published = published or datetime.now(UTC).isoformat()
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <published>{published}</published>
    <updated>{published}</updated>
    <title>{title}</title>
    <summary>{abstract}</summary>
    <author><name>Jane Q. Researcher</name></author>
    <author><name>Kim Minjun</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="{category}"/>
    <category term="{category}"/>
  </entry>"""


def feed(*entries: str) -> bytes:
    body = "".join(entries)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Query</title>{body}
</feed>""".encode()


def rss_item(
    arxiv_id: str = "2509.01234v1",
    title: str = "A note on something",
    abstract: str = "Nothing of interest here.",
    category: str = "q-fin.PM",
    announce: str | None = "new",
    creator: str = "Jane Q. Researcher, Kim Minjun",
) -> str:
    """One item in arXiv's RSS shape. `announce=None` leaves the field out entirely."""
    element = f"\n    <arxiv:announce_type>{announce}</arxiv:announce_type>" if announce else ""
    prefix = f"arXiv:{arxiv_id} Announce Type: {announce} " if announce else f"arXiv:{arxiv_id} "
    return f"""
  <item>
    <title>{title}</title>
    <link>https://arxiv.org/abs/{arxiv_id}</link>
    <description>{prefix}Abstract: {abstract}</description>
    <dc:creator>{creator}</dc:creator>
    <category>{category}</category>
    <guid isPermaLink="false">oai:arXiv.org:{arxiv_id}</guid>{element}
  </item>"""


def rss_feed(
    *items: str,
    pub_date: str = "Fri, 25 Sep 2026 00:00:00 -0400",
    skip_days: tuple[str, ...] = (),
) -> bytes:
    body = "".join(items)
    stamp = f"\n  <pubDate>{pub_date}</pubDate>" if pub_date else ""
    skipped = (
        "\n  <skipDays>" + "".join(f"<day>{day}</day>" for day in skip_days) + "</skipDays>"
        if skip_days
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:arxiv="http://arxiv.org/schemas/atom">
 <channel>
  <title>q-fin updates on arXiv.org</title>
  <link>http://arxiv.org/</link>{stamp}{skipped}{body}
 </channel>
</rss>""".encode()


@pytest.fixture
def store(tmp_path):
    """Every collect writes to a temporary store. A test must never touch the repo's."""
    return tmp_path / "literature"


@pytest.fixture
def no_network(monkeypatch):
    def refuse(url, timeout=45.0, **_):
        raise AssertionError(f"unexpected network call to {url}")

    monkeypatch.setattr(fp, "_get", refuse)
    return monkeypatch


def serve(monkeypatch, body: bytes):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: body)


def oai_record(
    arxiv_id: str = "2509.01234",
    title: str = "A note on something",
    abstract: str = "Nothing of interest here.",
    categories: str = "q-fin.PM q-fin.ST",
    created: str | None = None,
    status: str = "",
) -> str:
    created = created or datetime.now(UTC).date().isoformat()
    flag = f' status="{status}"' if status else ""
    body = (
        ""
        if status == "deleted"
        else f"""
    <metadata>
      <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
        <id>{arxiv_id}</id>
        <created>{created}</created>
        <authors>
          <author><keyname>Kim</keyname><forenames>Ji-woo</forenames></author>
          <author><keyname>CMS Collaboration</keyname></author>
        </authors>
        <title>{title}</title>
        <categories>{categories}</categories>
        <abstract>{abstract}</abstract>
      </arXiv>
    </metadata>"""
    )
    return f"""
  <record>
    <header{flag}><identifier>oai:arXiv.org:{arxiv_id}</identifier><datestamp>{created}</datestamp></header>{body}
  </record>"""


def oai_page(*records: str, token: str = "") -> bytes:
    resumption = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-09-27T00:00:00Z</responseDate>
  <ListRecords>{"".join(records)}
    {resumption}
  </ListRecords>
</OAI-PMH>""".encode()


def oai_error(code: str, text: str = "") -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <error code="{code}">{text}</error>
</OAI-PMH>""".encode()


# --- the request itself -----------------------------------------------------


class Response:
    """The little of `urlopen`'s result that `_get` reads."""

    def __init__(self, body: bytes = b"<feed/>", headers: dict[str, str] | None = None):
        self._body, self.headers = body, headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def refusal(code: int, url: str = "https://export.arxiv.org/api/query"):
    import email.message
    import urllib.error

    return urllib.error.HTTPError(url, code, "refused", email.message.Message(), None)


class Clock:
    """A clock that only the test's own `sleep` moves.

    `_get` paces itself against the wall clock, so a test that did not own the
    clock would either wait out the real three seconds or assert on a float it
    cannot predict. Here every wait is recorded and advances time by exactly
    what was asked, which is also the only honest model of a sleep.
    """

    def __init__(self, now: float = 1_000.0):
        self.now, self.slept = now, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def _fresh_rate_budget():
    """The pacing gate is process-wide state; no test may inherit another's."""
    fp._last_call = 0.0
    yield
    fp._last_call = 0.0


def serve_requests(monkeypatch, answer):
    """Route `_get` through `answer(request, n)`, recording every request made."""
    import urllib.request

    seen: list = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        result = answer(request, len(seen))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def fetch(clock: Clock, url: str = "https://export.arxiv.org/api/query") -> bytes:
    return fp._get(url, sleep=clock.sleep, clock=clock)


def test_the_request_identifies_us_and_says_what_it_accepts(monkeypatch):
    seen = serve_requests(monkeypatch, lambda request, n: Response())
    fetch(Clock())

    headers = {key.lower(): value for key, value in seen[0].headers.items()}
    assert "atom+xml" in headers["accept"]
    assert "quant-desk" in headers["user-agent"]


def test_the_request_asks_for_gzip(monkeypatch):
    seen = serve_requests(monkeypatch, lambda request, n: Response())
    fetch(Clock())

    headers = {key.lower(): value for key, value in seen[0].headers.items()}
    assert "gzip" in headers["accept-encoding"]


def test_a_gzipped_feed_is_decompressed(monkeypatch):
    import gzip as gziplib

    packed = gziplib.compress(b"<feed/>")
    serve_requests(monkeypatch, lambda request, n: Response(packed, {"Content-Encoding": "gzip"}))
    assert fetch(Clock()) == b"<feed/>"


def test_we_never_pretend_to_be_a_browser(monkeypatch):
    """A refusal we caused by lying about who we are is a refusal we cannot diagnose."""
    seen = serve_requests(monkeypatch, lambda request, n: Response())
    fetch(Clock())

    agent = seen[0].headers["User-agent"]
    assert "quant-desk" in agent and "Mozilla" not in agent


def test_a_406_is_waited_out_rather_than_varied(monkeypatch):
    """The measured cause is a rate limit, so a second shape is a second offence."""
    seen = serve_requests(monkeypatch, lambda request, n: Response() if n == 2 else refusal(406))
    clock = Clock()
    assert fetch(clock) == b"<feed/>"

    assert len(seen) == 2, "the retry is the same request, not a different one"
    assert seen[0].headers == seen[1].headers
    assert fp.BACKOFF_SECONDS[0] in clock.slept


def test_the_backoff_lengthens_rather_than_hammering(monkeypatch):
    serve_requests(monkeypatch, lambda request, n: refusal(406))
    clock = Clock()
    with pytest.raises(fp.FetchError, match="attempts failed"):
        fetch(clock)

    waits = [wait for wait in clock.slept if wait in fp.BACKOFF_SECONDS]
    assert waits == sorted(waits) and waits == list(fp.BACKOFF_SECONDS)


def test_no_two_requests_leave_inside_the_rate_limit(monkeypatch):
    """Bursting is what caused the 406 in the first place; it has to be impossible."""
    clock = Clock()
    departures: list[float] = []

    def answer(request, n):
        departures.append(clock())
        return Response() if n == 4 else refusal(429)

    serve_requests(monkeypatch, answer)
    fetch(clock)

    gaps = [later - earlier for earlier, later in zip(departures, departures[1:], strict=False)]
    assert len(gaps) == 3
    assert all(gap >= fp.MIN_INTERVAL_SECONDS for gap in gaps)


def test_the_gate_holds_even_when_nothing_else_waited(monkeypatch):
    """Two clean requests in a row still leave three seconds apart."""
    serve_requests(monkeypatch, lambda request, n: Response())
    clock = Clock()
    fetch(clock)
    before = clock()
    fetch(clock)
    assert clock() - before >= fp.MIN_INTERVAL_SECONDS


def test_a_refusal_that_is_not_a_rate_limit_fails_at_once(monkeypatch):
    """404 means the URL is wrong. Waiting three minutes will not make it right."""
    seen = serve_requests(monkeypatch, lambda request, n: refusal(404))
    clock = Clock()
    with pytest.raises(fp.FetchError, match="HTTP 404"):
        fetch(clock)
    assert len(seen) == 1
    assert not [wait for wait in clock.slept if wait in fp.BACKOFF_SECONDS]


# --- parsing ----------------------------------------------------------------


def test_an_entry_becomes_a_paper(no_network):
    serve(no_network, feed(entry()))
    papers = fp.parse_feed(feed(entry()))
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2509.01234", "the version suffix must be stripped for dedupe"
    assert paper.link == "https://arxiv.org/abs/2509.01234"
    assert paper.authors == ["Jane Q. Researcher", "Kim Minjun"]
    assert paper.primary_category == "q-fin.PM"


def test_the_same_paper_twice_is_counted_once(no_network, store):
    serve(no_network, feed(entry("2509.01234v1"), entry("2509.01234v3")))
    report = fp.collect(pause=NO_PAUSE, directory=store)
    assert report["in_window"] == 1


def test_a_title_spanning_lines_is_collapsed():
    papers = fp.parse_feed(feed(entry(title="A very\n   long   title")))
    assert papers[0].title == "A very long title"


# --- refusals ---------------------------------------------------------------


def test_an_html_error_page_is_an_error():
    """Well-formed HTML parses as XML, so it is the root tag that catches it."""
    with pytest.raises(fp.FetchError, match="not a feed"):
        fp.parse_feed(b"<!DOCTYPE html><html><body>503</body></html>")


def test_a_truncated_response_is_an_error():
    with pytest.raises(fp.FetchError, match="not Atom"):
        fp.parse_feed(b'<?xml version="1.0"?><feed><entry><id>htt')


def test_a_non_feed_document_is_an_error():
    with pytest.raises(fp.FetchError, match="not a feed"):
        fp.parse_feed(b'<?xml version="1.0"?><error>rate limited</error>')


def test_an_empty_feed_is_an_error_not_a_quiet_week():
    """A parse that yields nothing means the query broke, not that nobody published."""
    with pytest.raises(fp.FetchError, match="no entries"):
        fp.parse_feed(feed())


# --- scoring ----------------------------------------------------------------


def test_a_validation_paper_outscores_a_method_paper():
    validation = fp.score(
        fp.Paper(
            "1",
            "Deflated Sharpe and the probability of backtest overfitting",
            [],
            "",
            "",
            "",
            [],
            "",
            "We revisit multiple testing in finance.",
        )
    )
    method = fp.score(fp.Paper("2", "Deep learning for something", [], "", "", "", [], "", "A regime model."))
    assert validation.score > method.score
    assert "validation" in validation.matched
    assert set(method.matched) == {"method"}


def test_the_matched_terms_are_recorded_so_the_score_can_be_redone():
    paper = fp.score(
        fp.Paper("3", "Time series momentum and volatility targeting", [], "", "", "", [], "", "")
    )
    assert paper.matched["strategy"] == ["time series momentum"]
    assert paper.matched["risk"] == ["volatility targeting"]
    assert paper.score == 3 + 2


def test_a_korean_market_paper_is_picked_up():
    paper = fp.score(fp.Paper("4", "Momentum in the KOSPI cross-section", [], "", "", "", [], "", ""))
    assert "korea" in paper.matched


def test_scoring_is_case_insensitive():
    paper = fp.score(fp.Paper("5", "TREND FOLLOWING works", [], "", "", "", [], "", ""))
    assert "strategy" in paper.matched


# --- the window and the shortlist -------------------------------------------


def test_a_paper_older_than_the_window_is_dropped(no_network, store):
    old = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    serve(no_network, feed(entry("2508.00001v1", published=old), entry("2509.00002v1")))
    report = fp.collect(days=7, pause=NO_PAUSE, directory=store)
    assert report["in_window"] == 1
    assert report["papers"] == [] or report["papers"][0]["arxiv_id"] != "2508.00001"


def test_an_unparseable_date_is_kept_rather_than_lost():
    """Losing a paper silently is worse than reviewing a stale one."""
    paper = fp.Paper("6", "t", [], "not-a-date", "", "", [], "", "")
    assert fp.within(paper, datetime.now(UTC))


def test_a_low_scoring_paper_is_recorded_as_screened_out_not_deleted(no_network, store):
    """The screen has to be auditable, so what it rejected stays visible."""
    serve(no_network, feed(entry("2509.00003v1", title="On the weather")))
    report = fp.collect(min_score=3, pause=NO_PAUSE, directory=store)
    assert report["shortlisted"] == 0
    assert [p["arxiv_id"] for p in report["screened_out"]] == ["2509.00003"]


def test_the_shortlist_is_ordered_by_score(no_network, store):
    serve(
        no_network,
        feed(
            entry("2509.00010v1", title="Trend following", abstract="x"),
            entry(
                "2509.00011v1",
                title="Backtest overfitting and multiple testing",
                abstract="purged cross-validation and look-ahead",
            ),
        ),
    )
    report = fp.collect(min_score=1, pause=NO_PAUSE, directory=store)
    scores = [p["score"] for p in report["papers"]]
    assert scores == sorted(scores, reverse=True)
    assert report["papers"][0]["arxiv_id"] == "2509.00011"


def test_the_query_asks_arxiv_for_the_declared_categories():
    url = fp.query_url(("q-fin.PM", "q-fin.TR"), 50)
    assert "cat%3Aq-fin.PM+OR+cat%3Aq-fin.TR" in url
    assert "sortBy=submittedDate" in url
    assert "max_results=50" in url


def test_no_pdf_is_ever_requested():
    """Metadata is CC0; a paper's full text carries its own licence."""
    url = fp.query_url(fp.DEFAULT_CATEGORIES, 10)
    assert "/pdf/" not in url


# --- the harvest interface, which is the one that answers -------------------


def test_a_record_becomes_a_paper():
    papers, token = fp.parse_oai(oai_page(oai_record()))
    assert token == ""
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2509.01234"
    assert paper.link == "https://arxiv.org/abs/2509.01234"
    assert paper.categories == ["q-fin.PM", "q-fin.ST"]
    assert paper.primary_category == "q-fin.PM"
    assert paper.abstract == "Nothing of interest here."


def test_an_author_reads_the_way_the_paper_prints_it():
    papers, _ = fp.parse_oai(oai_page(oai_record()))
    assert papers[0].authors == ["Ji-woo Kim", "CMS Collaboration"]


def test_a_withdrawn_record_is_skipped_rather_than_reviewed():
    papers, _ = fp.parse_oai(oai_page(oai_record(status="deleted"), oai_record("2509.00002")))
    assert [p.arxiv_id for p in papers] == ["2509.00002"]


def test_an_empty_window_is_not_an_error():
    """A quiet week is a real answer; only a wrong request is a failure."""
    papers, token = fp.parse_oai(oai_error("noRecordsMatch"))
    assert papers == [] and token == ""


def test_any_other_oai_error_is_an_error():
    with pytest.raises(fp.FetchError, match="badArgument"):
        fp.parse_oai(oai_error("badArgument", "from is not a date"))


def test_the_atom_feed_is_not_mistaken_for_a_harvest():
    with pytest.raises(fp.FetchError, match="not an OAI-PMH envelope"):
        fp.parse_oai(feed(entry()))


def test_a_refusal_page_is_an_error_not_an_empty_week():
    with pytest.raises(fp.FetchError, match="not OAI-PMH"):
        fp.parse_oai(b"<html>Access Denied")


def test_a_well_formed_page_that_is_not_oai_is_an_error_too():
    """A WAF's error page can be valid XML; only the envelope decides."""
    with pytest.raises(fp.FetchError, match="not an OAI-PMH envelope"):
        fp.parse_oai(b"<html>Access Denied</html>")


def test_a_response_with_neither_error_nor_records_is_an_error():
    body = b'<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"/>'
    with pytest.raises(fp.FetchError, match="neither an error nor a ListRecords"):
        fp.parse_oai(body)


# --- paging ------------------------------------------------------------------


def test_the_harvest_follows_the_resumption_token(monkeypatch):
    pages = [
        oai_page(oai_record("2509.00001"), token="tok-1"),
        oai_page(oai_record("2509.00002")),
        oai_page(oai_record("2509.00003")),
    ]
    urls: list[str] = []

    def fake_get(url, timeout=45.0, **_):
        urls.append(url)
        return pages[min(len(urls) - 1, len(pages) - 1)]

    monkeypatch.setattr(fp, "_get", fake_get)
    papers = fp.harvest(date(2026, 9, 20), sets=("q-fin",), pause=lambda _: None)

    assert [p.arxiv_id for p in papers] == ["2509.00001", "2509.00002"]
    assert "resumptionToken=tok-1" in urls[1]


def test_each_archive_is_harvested(monkeypatch):
    seen: list[str] = []

    def fake_get(url, timeout=45.0, **_):
        seen.append(url)
        return oai_page(oai_record(f"2509.0000{len(seen)}"))

    monkeypatch.setattr(fp, "_get", fake_get)
    fp.harvest(date(2026, 9, 20), pause=lambda _: None)
    assert [s for s in fp.OAI_SETS if any(f"set={s}" in url for url in seen)] == list(fp.OAI_SETS)


def test_a_token_that_never_ends_is_refused(monkeypatch):
    """An unbounded resumption loop is a way to hang a runner on somebody else's bug."""
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: oai_page(oai_record(), token="same"))
    with pytest.raises(fp.FetchError, match="did not finish"):
        fp.harvest(date(2026, 9, 20), sets=("q-fin",), pause=lambda _: None, max_pages=3)


def test_the_harvest_paces_through_the_one_gate_rather_than_its_own(monkeypatch):
    """Counting the gap in two places would double the wait or let a caller skip it."""
    sleepers: list[object] = []

    def fake_get(url, timeout=45.0, sleep=None, **_):
        sleepers.append(sleep)
        return oai_page(oai_record(), token="t")

    monkeypatch.setattr(fp, "_get", fake_get)
    waits: list[float] = []

    def pause(seconds: float) -> None:
        waits.append(seconds)

    with pytest.raises(fp.FetchError):
        fp.harvest(date(2026, 9, 20), sets=("q-fin",), pause=pause, max_pages=3)

    assert sleepers and all(sleeper is pause for sleeper in sleepers)
    assert waits == [], "the gap belongs to `_get`, not to the loop around it"


def test_the_window_starts_days_before_today():
    url = fp.oai_url("q-fin", date(2026, 9, 20))
    assert "from=2026-09-20" in url and "set=q-fin" in url and "metadataPrefix=arXiv" in url


# --- which interface answered ------------------------------------------------


def test_rss_is_preferred(monkeypatch):
    """The one arXiv host that answers this runner goes first (ADR-0020, ADR-0021)."""
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: rss_feed(rss_item()))
    papers, interface, degraded, shapes = fp.fetch_today(pause=NO_PAUSE)
    assert interface == "rss" and degraded == ""
    assert len(papers) == len(fp.RSS_FEEDS)  # one item per archive feed
    assert [shape["feed"] for shape in shapes] == list(fp.RSS_FEEDS)


def test_the_harvest_is_the_first_fallback_and_says_why(monkeypatch):
    def fake_get(url, timeout=45.0, **_):
        if "rss" in url:
            raise fp.FetchError("HTTP 503 from rss.arxiv.org")
        return oai_page(oai_record())

    monkeypatch.setattr(fp, "_get", fake_get)
    _papers, interface, degraded, _shapes = fp.fetch_today(pause=NO_PAUSE)
    assert interface == "oai-pmh" and "503" in degraded


def test_the_search_api_is_the_last_fallback(monkeypatch):
    def fake_get(url, timeout=45.0, **_):
        if "rss" in url:
            raise fp.FetchError("406 from rss")
        if "oaipmh" in url:
            raise fp.FetchError("HTTP 503 from arXiv (gzip): empty response body")
        return feed(entry())

    monkeypatch.setattr(fp, "_get", fake_get)
    _papers, interface, degraded, _shapes = fp.fetch_today(pause=NO_PAUSE)
    assert interface == "search-api"
    assert "503" in degraded and "406" in degraded


def test_every_interface_failing_reports_all_three(monkeypatch):
    def fake_get(url, timeout=45.0, **_):
        raise fp.FetchError("406 here" if "export" in url else "503 there")

    monkeypatch.setattr(fp, "_get", fake_get)
    with pytest.raises(fp.FetchError, match="no arXiv interface answered"):
        fp.fetch_today(pause=NO_PAUSE)


def test_the_report_names_the_interface_it_used(monkeypatch, store):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: oai_page(oai_record()))
    report = fp.collect(min_score=0, pause=NO_PAUSE, directory=store)
    assert report["interface"] == "oai-pmh"
    # One record per archive harvested, the same paper in both: the store keys by
    # arXiv id, so it lands once.
    assert report["returned"] == 1


def test_a_paper_outside_our_categories_is_filtered_out(monkeypatch, store):
    """The harvest returns whole archives, so the category filter moves here."""
    monkeypatch.setattr(
        fp,
        "_get",
        lambda url, timeout=45.0, **_: oai_page(
            oai_record("2509.00100", categories="q-fin.GN"),
            oai_record("2509.00101", categories="q-fin.PM"),
        ),
    )
    report = fp.collect(min_score=0, pause=NO_PAUSE, directory=store)
    assert {p["arxiv_id"] for p in report["papers"]} == {"2509.00101"}


# --- the search API asks small, because that is what it answers --------------


def test_the_search_asks_one_category_at_a_time(monkeypatch):
    """A page of one category is served; the week of six OR-ed ones is refused."""
    urls: list[str] = []

    def fake_get(url, timeout=45.0, **_):
        urls.append(url)
        return feed(entry())

    monkeypatch.setattr(fp, "_get", fake_get)
    fp.search(("q-fin.PM", "q-fin.ST"), datetime.now(UTC) - timedelta(days=7), pause=NO_PAUSE)

    assert len(urls) == 2
    assert all(url.count("cat%3A") == 1 for url in urls)
    assert all(f"max_results={fp.PAGE_SIZE}" in url for url in urls)


def test_a_short_page_ends_the_category(monkeypatch):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: feed(entry()))
    papers = fp.search(("q-fin.PM",), datetime.now(UTC) - timedelta(days=7), pause=NO_PAUSE)
    assert len(papers) == 1


def test_a_full_page_is_followed_by_the_next_one(monkeypatch):
    urls: list[str] = []

    def fake_get(url, timeout=45.0, **_):
        urls.append(url)
        size = 2 if len(urls) == 1 else 1
        return feed(*[entry(f"2509.0000{n}v1") for n in range(size)])

    monkeypatch.setattr(fp, "_get", fake_get)
    fp.search(("q-fin.PM",), datetime.now(UTC) - timedelta(days=7), page_size=2, pause=NO_PAUSE)
    assert len(urls) == 2
    assert "start=2" in urls[1]


def test_paging_stops_once_the_page_is_older_than_the_window(monkeypatch):
    """Sorted newest first, so a page entirely outside the window ends the category."""
    old = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    urls: list[str] = []

    def fake_get(url, timeout=45.0, **_):
        urls.append(url)
        return feed(entry("2506.00001v1", published=old), entry("2506.00002v1", published=old))

    monkeypatch.setattr(fp, "_get", fake_get)
    fp.search(("q-fin.PM",), datetime.now(UTC) - timedelta(days=7), page_size=2, pause=NO_PAUSE)
    assert len(urls) == 1


def test_a_search_that_returns_nothing_at_all_is_an_error(monkeypatch):
    """An empty week is possible; an empty feed means the query is wrong."""
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: feed())
    with pytest.raises(fp.FetchError, match="no entries"):
        fp.search(("q-fin.PM",), datetime.now(UTC) - timedelta(days=7), pause=NO_PAUSE)


def test_the_page_never_asks_for_more_than_the_measured_ceiling(monkeypatch):
    """`max_results=120` was refused; the caller's number cannot raise the page size."""
    urls: list[str] = []
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: urls.append(url) or feed(entry()))

    _papers, interface, _degraded, _shapes = fp.fetch_today(max_results=500, pause=NO_PAUSE)

    assert interface == "search-api"  # an Atom body is neither RSS nor a harvest, so it falls through
    searches = [url for url in urls if "search_query" in url]
    assert searches and all(f"max_results={fp.PAGE_SIZE}" in url for url in searches)


# --- the RSS feed, which is the only arXiv host that answers ------------------


def test_the_feed_url_is_one_per_archive():
    assert fp.rss_url("q-fin") == "https://rss.arxiv.org/rss/q-fin"


def test_an_item_becomes_a_paper_with_the_version_stripped():
    papers, shape = fp.parse_rss(rss_feed(rss_item("2509.01234v2")), feed="q-fin")
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2509.01234", "a replacement must not enter the store as a second paper"
    assert paper.title == "A note on something"
    assert paper.authors == ["Jane Q. Researcher", "Kim Minjun"]
    assert paper.categories == ["q-fin.PM"] and paper.primary_category == "q-fin.PM"
    assert shape["items"] == 1 and shape["kept"] == 1


def test_the_abstract_is_taken_out_of_the_description_prefix():
    """arXiv writes the id and announce type in front of the abstract."""
    papers, _ = fp.parse_rss(rss_feed(rss_item(abstract="Momentum decays after 2003.")))
    assert papers[0].abstract == "Momentum decays after 2003."


def test_a_replacement_announcement_is_not_reviewed_again():
    papers, shape = fp.parse_rss(rss_feed(rss_item(announce="replace")))
    assert papers == [] and shape["announce_types"] == {"replace": 1}


def test_a_cross_listing_is_kept():
    papers, _ = fp.parse_rss(rss_feed(rss_item(announce="cross")))
    assert [p.announce_type for p in papers] == ["cross"]


def test_an_announce_type_only_in_the_description_is_still_read():
    """Two readings, because nobody here has held this feed with a real response."""
    item = rss_item(announce="new").replace("\n    <arxiv:announce_type>new</arxiv:announce_type>", "")
    papers, shape = fp.parse_rss(rss_feed(item))
    assert [p.announce_type for p in papers] == ["new"]
    assert shape["announce_type_from"] == {"description": 1}


def test_an_unreadable_announce_type_keeps_the_paper():
    """Dropping a paper on a field we failed to parse is the expensive direction."""
    papers, shape = fp.parse_rss(rss_feed(rss_item(announce=None)))
    assert [p.announce_type for p in papers] == [fp.ANNOUNCE_UNKNOWN]
    assert shape["announce_type_from"] == {"absent": 1}


def test_the_feeds_own_date_is_the_announcement_day():
    _papers, shape = fp.parse_rss(rss_feed(rss_item(), pub_date="Fri, 25 Sep 2026 00:00:00 -0400"))
    assert shape["announce_day"] == "2026-09-25" and shape["announce_day_from"] == "pubDate"


def test_a_feed_with_no_readable_date_falls_back_to_the_run_day_and_says_so():
    _papers, shape = fp.parse_rss(rss_feed(rss_item(), pub_date="whenever"), today=date(2026, 9, 27))
    assert shape["announce_day"] == "2026-09-27" and shape["announce_day_from"] == "run_date"


def test_the_shape_report_carries_the_field_names_the_feed_actually_had():
    """The field names came from documentation; the first real run settles them."""
    _papers, shape = fp.parse_rss(rss_feed(rss_item()))
    assert "description" in shape["item_fields"] and "guid" in shape["item_fields"]


def test_an_empty_feed_is_an_empty_day_not_a_failure():
    """Measured on the first live run: a Sunday channel comes back whole and itemless.

    arXiv answering "nothing was announced" must not send the run down to two
    origin-blocked fallbacks and fail the job (ADR-0021).
    """
    papers, shape = fp.parse_rss(rss_feed(), feed="q-fin")
    assert papers == []
    assert shape["items"] == 0 and shape["kept"] == 0 and shape["item_fields"] == []


def test_an_empty_feed_still_reports_what_the_channel_carried():
    """The only evidence that it was arXiv's silence and not a changed feed shape."""
    _papers, shape = fp.parse_rss(rss_feed())
    assert "title" in shape["channel_fields"] and "pubDate" in shape["channel_fields"]


def test_the_shape_report_carries_the_feeds_own_skipdays():
    _papers, shape = fp.parse_rss(rss_feed(skip_days=("Saturday", "Sunday")))
    assert shape["skip_days"] == ["Saturday", "Sunday"]


def test_a_feed_with_items_reports_the_channel_fields_too():
    """Same keys either way, so a report reader never has to branch on emptiness."""
    _papers, shape = fp.parse_rss(rss_feed(rss_item()))
    assert "title" in shape["channel_fields"] and shape["skip_days"] == []


def test_an_empty_feed_does_not_reach_for_the_fallbacks(monkeypatch):
    """The fallbacks are origin-blocked; asking them on a Sunday turns quiet into red."""
    calls: list[str] = []

    def fake_get(url, timeout=45.0, **_):
        calls.append(url)
        return rss_feed()

    monkeypatch.setattr(fp, "_get", fake_get)
    papers, interface, degraded, shapes = fp.fetch_today(pause=NO_PAUSE, today=date(2026, 9, 27))
    assert papers == [] and interface == "rss" and degraded == ""
    assert len(shapes) == len(fp.RSS_FEEDS)
    assert all("rss.arxiv.org" in url for url in calls)


def test_a_document_that_is_not_rss_names_its_root():
    with pytest.raises(fp.FetchError, match="no <channel>"):
        fp.parse_rss(b'<?xml version="1.0"?><feed><entry/></feed>')


def test_an_html_error_page_is_an_error_naming_its_root():
    """An error page parses as XML often enough that "it parsed" proves nothing."""
    with pytest.raises(fp.FetchError, match="html"):
        fp.parse_rss(b"<!DOCTYPE html><html><body>503</body></html>")


def test_a_body_that_is_not_markup_at_all_is_an_error():
    with pytest.raises(fp.FetchError, match="not XML"):
        fp.parse_rss(b"503 Service Unavailable")


def test_an_item_with_neither_guid_nor_link_cannot_be_identified():
    broken = rss_item().replace('<guid isPermaLink="false">oai:arXiv.org:2509.01234v1</guid>', "")
    broken = broken.replace("<link>https://arxiv.org/abs/2509.01234v1</link>", "")
    with pytest.raises(fp.FetchError, match="neither guid nor link"):
        fp.parse_rss(rss_feed(broken))


def test_one_refused_feed_fails_the_day(monkeypatch):
    """A partial day looks exactly like a quiet one, so half an answer is no answer."""

    def fake_get(url, timeout=45.0, **_):
        if url.endswith("econ.EM"):
            raise fp.FetchError("HTTP 503")
        return rss_feed(rss_item())

    monkeypatch.setattr(fp, "_get", fake_get)
    with pytest.raises(fp.FetchError, match="503"):
        fp.fetch_rss(pause=NO_PAUSE)


# --- the store, because a daily feed cannot answer a weekly question ---------


def paper(arxiv_id: str = "2509.00001", day: str = "2026-09-25", **overrides) -> fp.Paper:
    fields = {
        "arxiv_id": arxiv_id,
        "title": "t",
        "authors": ["A"],
        "published": day,
        "updated": day,
        "primary_category": "q-fin.PM",
        "categories": ["q-fin.PM"],
        "link": f"https://arxiv.org/abs/{arxiv_id}",
        "abstract": "a",
    }
    fields.update(overrides)
    return fp.Paper(**fields)


def test_a_day_is_one_file_named_by_the_announcement_day(store):
    fp.write_days(fp.group_by_day([paper()], "2026-09-27"), store, ["2026-09-25"])
    assert (fp.store_dir(store) / "2026-09-25.json").exists()


def test_a_fallback_that_returns_a_week_fills_a_file_per_day(store):
    """A recovery run after three dark days is a recovery, not a pile-up."""
    papers = [paper("2509.00001", "2026-09-23"), paper("2509.00002", "2026-09-24")]
    fp.write_days(fp.group_by_day(papers, "2026-09-27"), store, [])
    assert sorted(path.name for path in fp.store_dir(store).glob("*.json")) == [
        "2026-09-23.json",
        "2026-09-24.json",
    ]


def test_a_second_run_does_not_delete_what_the_first_recorded(store):
    fp.write_days(fp.group_by_day([paper("2509.00001")], "2026-09-25"), store, [])
    fp.write_days(fp.group_by_day([paper("2509.00002")], "2026-09-25"), store, [])
    papers, present, missing = fp.read_window(store, 1, date(2026, 9, 25))
    assert {p.arxiv_id for p in papers} == {"2509.00001", "2509.00002"}
    assert present == ["2026-09-25"] and missing == ["2026-09-24"]


def test_a_quiet_day_is_written_as_an_empty_file(store):
    """No file means the day was never asked for; an empty file means it held nothing."""
    fp.write_days({}, store, ["2026-09-26"])
    path = fp.store_dir(store) / "2026-09-26.json"
    assert json.loads(path.read_text(encoding="utf-8"))["count"] == 0


def test_a_missing_day_is_reported_rather_than_read_as_quiet(store):
    fp.write_days(fp.group_by_day([paper(day="2026-09-25")], "2026-09-25"), store, [])
    _papers, present, missing = fp.read_window(store, 3, date(2026, 9, 27))
    assert present == ["2026-09-25"]
    assert missing == ["2026-09-24", "2026-09-26", "2026-09-27"]


def test_a_corrupt_day_file_counts_as_missing(store):
    fp.store_dir(store).mkdir(parents=True)
    (fp.store_dir(store) / "2026-09-25.json").write_text("{not json", encoding="utf-8")
    _papers, present, missing = fp.read_window(store, 0, date(2026, 9, 25))
    assert present == [] and missing == ["2026-09-25"]


def test_the_weekly_report_says_which_days_it_is_missing(no_network, store):
    serve(no_network, rss_feed(rss_item()))
    report = fp.collect(days=7, min_score=0, pause=NO_PAUSE, directory=store, today=date(2026, 9, 27))
    # Two files exist: the day the feed was about, and an empty file for the day
    # we asked. Six of the eight days in the window were never fetched, and the
    # report has to say so or a short window reads as a quiet week.
    assert report["days_present"] == ["2026-09-25", "2026-09-27"]
    assert len(report["days_missing"]) == 6


def test_the_weekly_list_can_be_rebuilt_without_asking_arxiv(no_network, store):
    """`--no-fetch`: the store is the record, so a rebuild needs no network."""
    fp.write_days(fp.group_by_day([paper(day="2026-09-26")], "2026-09-26"), store, [])
    report = fp.collect(min_score=0, pause=NO_PAUSE, directory=store, today=date(2026, 9, 26), fetch=False)
    assert report["interface"] == "store-only" and report["in_window"] == 1


def test_todays_papers_land_in_the_store_and_in_the_report(no_network, store):
    serve(no_network, rss_feed(rss_item("2509.09999v1", title="Trend following")))
    report = fp.collect(min_score=0, pause=NO_PAUSE, directory=store, today=date(2026, 9, 25))
    assert report["announced_today"] == len(fp.RSS_FEEDS)
    assert [p["arxiv_id"] for p in report["papers"]] == ["2509.09999"]
    stored = json.loads((fp.store_dir(store) / "2026-09-25.json").read_text(encoding="utf-8"))
    assert stored["count"] == 1


def test_a_quiet_day_the_feed_itself_reported_is_stored_as_an_empty_file(no_network, store):
    serve(no_network, rss_feed(skip_days=("Saturday", "Sunday")))
    report = fp.collect(min_score=0, pause=NO_PAUSE, directory=store, today=date(2026, 9, 27))
    stored = json.loads((fp.store_dir(store) / "2026-09-27.json").read_text(encoding="utf-8"))
    assert stored["count"] == 0 and report["announced_today"] == 0
    assert report["empty_days_in_a_row"] == 1


def test_stored_empty_days_in_a_row_are_counted(store):
    for day in ("2026-09-25", "2026-09-26", "2026-09-27"):
        fp.write_days({}, store, [day])
    assert fp.empty_days_in_a_row(store, date(2026, 9, 27)) == 3


def test_a_missing_day_breaks_the_empty_streak(store):
    """We did not ask, so we did not measure quiet: a gap is not evidence of silence."""
    for day in ("2026-09-25", "2026-09-27"):
        fp.write_days({}, store, [day])
    assert fp.empty_days_in_a_row(store, date(2026, 9, 27)) == 1


def test_a_day_that_held_a_paper_ends_the_streak(store):
    fp.write_days({}, store, ["2026-09-27"])
    fp.write_days(fp.group_by_day([paper(day="2026-09-26")], "2026-09-26"), store, [])
    fp.write_days({}, store, ["2026-09-25"])
    assert fp.empty_days_in_a_row(store, date(2026, 9, 27)) == 1


def test_four_empty_days_in_a_row_fails_the_job(no_network, store, capsys):
    """Longer than any arXiv weekend, so it is a reader that stopped reading."""
    for day in ("2026-09-24", "2026-09-25", "2026-09-26"):
        fp.write_days({}, store, [day])
    serve(no_network, rss_feed(skip_days=("Sunday",)))
    no_network.setattr(fp, "_today", lambda: date(2026, 9, 27))
    code = fp.main(["--out", str(store), "--min-score", "0", "--days", "3"])
    captured = capsys.readouterr()
    assert code == 1
    assert "no longer being read" in captured.err


def test_three_empty_days_is_still_a_weekend(no_network, store):
    for day in ("2026-09-25", "2026-09-26"):
        fp.write_days({}, store, [day])
    serve(no_network, rss_feed(skip_days=("Sunday",)))
    no_network.setattr(fp, "_today", lambda: date(2026, 9, 27))
    assert fp.main(["--out", str(store), "--min-score", "0", "--days", "3"]) == 0


def test_the_run_prints_the_channel_fields_when_no_items_came(no_network, store, capsys):
    serve(no_network, rss_feed(skip_days=("Sunday",)))
    no_network.setattr(fp, "_today", lambda: date(2026, 9, 27))
    fp.main(["--out", str(store), "--min-score", "0", "--days", "3"])
    out = capsys.readouterr().out
    assert "no items; channel carried" in out and "skipDays ['Sunday']" in out


# --- the one-off migration into the store ------------------------------------


def test_the_store_is_seeded_from_the_weekly_files_already_committed(store):
    """The store starts empty, so the first rebuild would be poorer than what it replaces."""
    store.mkdir(parents=True)
    (store / "2026-W39.json").write_text(
        json.dumps(
            {
                "week": "2026-W39",
                "papers": [
                    {
                        "arxiv_id": "2609.18019",
                        "title": "Model-Free Passive Execution",
                        "authors": ["A"],
                        "published": "2026-09-16T02:16:13Z",
                        "updated": "2026-09-16T02:16:13Z",
                        "primary_category": "q-fin.TR",
                        "categories": ["q-fin.TR"],
                        "link": "https://arxiv.org/abs/2609.18019",
                        "abstract": "execution and slippage",
                        "score": 7,
                    }
                ],
                "screened_out": [{"arxiv_id": "2609.00001", "title": "On the weather", "score": 1}],
            }
        ),
        encoding="utf-8",
    )
    assert fp.seed_from_weekly(store) == ["2026-09-16"]
    stored = json.loads((fp.store_dir(store) / "2026-09-16.json").read_text(encoding="utf-8"))
    assert [row["arxiv_id"] for row in stored["papers"]] == ["2609.18019"]


def test_a_screened_out_stub_is_not_seeded_as_a_paper(store):
    """It carries no abstract, so a seeded stub would score 0 -- a fabricated number."""
    store.mkdir(parents=True)
    (store / "2026-W38.json").write_text(
        json.dumps({"papers": [], "screened_out": [{"arxiv_id": "2609.00002", "score": 2}]}),
        encoding="utf-8",
    )
    assert fp.seed_from_weekly(store) == []


def test_seeding_ignores_a_file_that_is_not_a_report(store):
    store.mkdir(parents=True)
    (store / "notes.json").write_text("[1, 2, 3]", encoding="utf-8")
    (store / "broken.json").write_text("{not json", encoding="utf-8")
    assert fp.seed_from_weekly(store) == []
