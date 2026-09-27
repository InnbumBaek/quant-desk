"""The paper fetcher is tested for what it refuses and for how it scores.

No test touches the network: the container has no route to export.arxiv.org, so
a test that needed one would fail for the wrong reason. Every case runs against
a recorded Atom body.
"""

from __future__ import annotations

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


def test_the_same_paper_twice_is_counted_once(no_network):
    serve(no_network, feed(entry("2509.01234v1"), entry("2509.01234v3")))
    report = fp.collect(pause=NO_PAUSE)
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


def test_a_paper_older_than_the_window_is_dropped(no_network):
    old = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    serve(no_network, feed(entry("2508.00001v1", published=old), entry("2509.00002v1")))
    report = fp.collect(days=7, pause=NO_PAUSE)
    assert report["in_window"] == 1
    assert report["papers"] == [] or report["papers"][0]["arxiv_id"] != "2508.00001"


def test_an_unparseable_date_is_kept_rather_than_lost():
    """Losing a paper silently is worse than reviewing a stale one."""
    paper = fp.Paper("6", "t", [], "not-a-date", "", "", [], "", "")
    assert fp.within(paper, datetime.now(UTC))


def test_a_low_scoring_paper_is_recorded_as_screened_out_not_deleted(no_network):
    """The screen has to be auditable, so what it rejected stays visible."""
    serve(no_network, feed(entry("2509.00003v1", title="On the weather")))
    report = fp.collect(min_score=3, pause=NO_PAUSE)
    assert report["shortlisted"] == 0
    assert [p["arxiv_id"] for p in report["screened_out"]] == ["2509.00003"]


def test_the_shortlist_is_ordered_by_score(no_network):
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
    report = fp.collect(min_score=1, pause=NO_PAUSE)
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


def test_the_harvest_is_preferred(monkeypatch):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: oai_page(oai_record()))
    papers, interface, degraded = fp.gather(fp.DEFAULT_CATEGORIES, 7, 120, pause=lambda _: None)
    assert interface == "oai-pmh" and degraded == ""
    assert len(papers) == len(fp.OAI_SETS)  # one record per archive harvested


def test_the_search_api_is_the_fallback_and_says_why(monkeypatch):
    def fake_get(url, timeout=45.0, **_):
        if "oaipmh" in url:
            raise fp.FetchError("HTTP 503 from arXiv (gzip): empty response body")
        return feed(entry())

    monkeypatch.setattr(fp, "_get", fake_get)
    _papers, interface, degraded = fp.gather(fp.DEFAULT_CATEGORIES, 7, 120, pause=lambda _: None)
    assert interface == "search-api"
    assert "503" in degraded


def test_both_interfaces_failing_reports_both(monkeypatch):
    def fake_get(url, timeout=45.0, **_):
        raise fp.FetchError("406 here" if "export" in url else "503 there")

    monkeypatch.setattr(fp, "_get", fake_get)
    with pytest.raises(fp.FetchError, match="neither arXiv interface answered"):
        fp.gather(fp.DEFAULT_CATEGORIES, 7, 120, pause=lambda _: None)


def test_the_report_names_the_interface_it_used(monkeypatch):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0, **_: oai_page(oai_record()))
    report = fp.collect(min_score=0, pause=NO_PAUSE)
    assert report["interface"] == "oai-pmh"
    assert report["returned"] == 2  # one record per archive harvested


def test_a_paper_outside_our_categories_is_filtered_out(monkeypatch):
    """The harvest returns whole archives, so the category filter moves here."""
    monkeypatch.setattr(
        fp,
        "_get",
        lambda url, timeout=45.0, **_: oai_page(
            oai_record("2509.00100", categories="q-fin.GN"),
            oai_record("2509.00101", categories="q-fin.PM"),
        ),
    )
    report = fp.collect(min_score=0, pause=NO_PAUSE)
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

    _papers, interface, _degraded = fp.gather(fp.DEFAULT_CATEGORIES, 7, 500, pause=NO_PAUSE)

    assert interface == "search-api"  # an Atom body is not a harvest, so it falls through
    searches = [url for url in urls if "search_query" in url]
    assert searches and all(f"max_results={fp.PAGE_SIZE}" in url for url in searches)
