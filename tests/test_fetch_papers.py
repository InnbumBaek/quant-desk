"""The paper fetcher is tested for what it refuses and for how it scores.

No test touches the network: the container has no route to export.arxiv.org, so
a test that needed one would fail for the wrong reason. Every case runs against
a recorded Atom body.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts import fetch_papers as fp


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
    def refuse(url, timeout=45.0):
        raise AssertionError(f"unexpected network call to {url}")

    monkeypatch.setattr(fp, "_get", refuse)
    return monkeypatch


def serve(monkeypatch, body: bytes):
    monkeypatch.setattr(fp, "_get", lambda url, timeout=45.0: body)


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
    report = fp.collect()
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
    report = fp.collect(days=7)
    assert report["in_window"] == 1
    assert report["papers"] == [] or report["papers"][0]["arxiv_id"] != "2508.00001"


def test_an_unparseable_date_is_kept_rather_than_lost():
    """Losing a paper silently is worse than reviewing a stale one."""
    paper = fp.Paper("6", "t", [], "not-a-date", "", "", [], "", "")
    assert fp.within(paper, datetime.now(UTC))


def test_a_low_scoring_paper_is_recorded_as_screened_out_not_deleted(no_network):
    """The screen has to be auditable, so what it rejected stays visible."""
    serve(no_network, feed(entry("2509.00003v1", title="On the weather")))
    report = fp.collect(min_score=3)
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
    report = fp.collect(min_score=1)
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
