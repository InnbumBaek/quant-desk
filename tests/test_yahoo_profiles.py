"""The per-symbol sector reader: what it records, and what it refuses to invent.

Whether Yahoo answers at all is settled on the runner (this container reaches
no vendor host). These fix what the code does with each kind of answer.
"""

from __future__ import annotations

import http.cookiejar
import io
import json
import urllib.error

import pytest

from core.data.sic import BUCKETS
from core.data.yahoo_sectors import YAHOO_SECTORS, bucket_for_yahoo_sector, unmapped_labels
from scripts import yahoo_profiles as yp

VENDOR_LABELS = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)


# --- the mapping ------------------------------------------------------------


@pytest.mark.parametrize("label", VENDOR_LABELS)
def test_every_vendor_label_reaches_a_bucket(label):
    assert bucket_for_yahoo_sector(label) in BUCKETS


def test_every_bucket_is_reachable_from_some_vendor_label():
    """A bucket the vendor cannot express is a bucket that empties when SIC is out."""
    assert {bucket_for_yahoo_sector(label) for label in VENDOR_LABELS} == set(BUCKETS)


def test_the_table_never_names_a_bucket_no_limit_knows():
    assert set(YAHOO_SECTORS.values()) <= set(BUCKETS)


def test_the_vendors_vocabulary_is_translated_not_adopted():
    """Cyclical/Defensive and Financial Services are the three that differ from ours."""
    assert bucket_for_yahoo_sector("Consumer Cyclical") == "consumer_discretionary"
    assert bucket_for_yahoo_sector("Consumer Defensive") == "consumer_staples"
    assert bucket_for_yahoo_sector("Financial Services") == "financials"


def test_a_fund_with_no_sector_gets_no_bucket():
    assert bucket_for_yahoo_sector("") is None
    assert bucket_for_yahoo_sector(None) is None


def test_a_label_we_have_never_seen_is_reported_not_guessed():
    assert bucket_for_yahoo_sector("Conglomerates") is None
    assert unmapped_labels(["Conglomerates", "Energy", ""]) == ("Conglomerates",)


# --- reading one reply -------------------------------------------------------


def summary(sector: str | None = "Technology", result: bool = True) -> bytes:
    profile = {} if sector is None else {"sector": sector}
    body = {"quoteSummary": {"result": [{"assetProfile": profile}] if result else [], "error": None}}
    return json.dumps(body).encode("utf-8")


def test_the_sector_comes_out_of_the_profile():
    assert yp.read_sector(summary("Financial Services")) == "Financial Services"


def test_a_profile_without_a_sector_is_an_answer_not_a_failure():
    """A fund, a trust or a shell genuinely has none, and asking again next week wastes the budget."""
    assert yp.read_sector(summary(None)) is None
    assert yp.read_sector(summary(result=False)) is None


def test_an_error_in_the_reply_is_an_error():
    body = json.dumps({"quoteSummary": {"result": None, "error": {"code": "Not Found"}}}).encode()
    with pytest.raises(yp.ProfileError, match="carries an error"):
        yp.read_sector(body)


def test_a_challenge_page_is_an_error_not_a_missing_sector():
    with pytest.raises(yp.ProfileError, match="not JSON"):
        yp.read_sector(b"<html>Verify you are human</html>")


def test_the_crumb_rides_in_the_query():
    url = yp.profile_url("BRK.B", "abc123")
    assert "crumb=abc123" in url and "BRK.B" in url and "assetProfile" in url


# --- the session -------------------------------------------------------------


class Reply:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def http_error(code: int, url: str = "https://fc.yahoo.com/"):
    import email.message

    return urllib.error.HTTPError(url, code, "refused", email.message.Message(), io.BytesIO(b""))


class FakeOpener:
    """Stands in for the cookie-processing opener, answering by URL."""

    def __init__(self, answers, jar=None):
        self.answers, self.seen = answers, []
        self.jar = jar

    def open(self, request, timeout=None):
        self.seen.append(request.full_url)
        answer = self.answers(request.full_url, len(self.seen))
        if isinstance(answer, Exception):
            raise answer
        return Reply(answer)


def session_with(monkeypatch, answers, cookies: int = 1):
    jar = http.cookiejar.CookieJar()
    for index in range(cookies):
        jar.set_cookie(
            http.cookiejar.Cookie(
                0,
                f"A{index}",
                "v",
                None,
                False,
                ".yahoo.com",
                True,
                False,
                "/",
                True,
                False,
                None,
                True,
                None,
                None,
                {},
            )
        )
    opener = FakeOpener(answers, jar)
    monkeypatch.setattr(yp.urllib.request, "build_opener", lambda *_: opener)
    monkeypatch.setattr(yp.http.cookiejar, "CookieJar", lambda: jar)
    return opener


def test_the_session_takes_the_cookie_then_the_crumb(monkeypatch):
    opener = session_with(monkeypatch, lambda url, n: b"crumb-token")
    _opener, crumb = yp.open_session()
    assert crumb == "crumb-token"
    assert opener.seen == [yp.COOKIE_URL, yp.CRUMB_URL]


def test_the_cookie_page_answering_404_is_fine_if_it_set_the_cookie(monkeypatch):
    """fc.yahoo.com answers 404 while setting the cookie, which is why we call it."""
    session_with(monkeypatch, lambda url, n: http_error(404) if n == 1 else b"crumb-token")
    _opener, crumb = yp.open_session()
    assert crumb == "crumb-token"


def test_no_cookie_means_no_session(monkeypatch):
    session_with(monkeypatch, lambda url, n: http_error(403), cookies=0)
    with pytest.raises(yp.ProfileError, match="cookie"):
        yp.open_session()


def test_a_refused_crumb_says_so(monkeypatch):
    """A bare crumb request with no cookie is what the probe measured as 429."""
    session_with(monkeypatch, lambda url, n: b"" if n == 1 else http_error(429, yp.CRUMB_URL))
    with pytest.raises(yp.ProfileError, match="HTTP 429"):
        yp.open_session()


def test_a_challenge_instead_of_a_crumb_is_caught_before_it_poisons_every_symbol(monkeypatch):
    session_with(monkeypatch, lambda url, n: b"" if n == 1 else b"<html>are you a robot</html>")
    with pytest.raises(yp.ProfileError, match="not a crumb"):
        yp.open_session()


# --- the harvest -------------------------------------------------------------


def harvesting(answers):
    return FakeOpener(answers), "crumb"


def test_each_symbol_is_asked_once_and_sorted_into_one_place():
    def answers(url, n):
        if "AAPL" in url:
            return summary("Technology")
        if "TRST" in url:
            return summary(None)
        return http_error(404, url)

    got = yp.harvest(["AAPL", "TRST", "ZZZZ"], pause=lambda _: None, session=harvesting(answers))
    assert got.buckets == {"AAPL": "technology"}
    assert got.no_sector == ["TRST"]
    assert got.failed == ["ZZZZ"]
    assert got.asked == 3


def test_the_budget_stops_the_run():
    got = yp.harvest(
        [f"S{i}" for i in range(100)],
        budget=5,
        pause=lambda _: None,
        session=harvesting(lambda u, n: summary()),
    )
    assert got.asked == 5


def test_there_is_a_gap_between_symbols():
    waits: list[float] = []
    yp.harvest(["A", "B", "C"], pause=waits.append, session=harvesting(lambda u, n: summary()))
    assert len(waits) == 2 and all(wait > 0 for wait in waits)


def test_a_dead_session_stops_rather_than_burning_the_budget():
    """A thousand more symbols after the session died is a thousand more refusals."""
    got = yp.harvest(
        [f"S{i}" for i in range(500)],
        pause=lambda _: None,
        session=harvesting(lambda u, n: http_error(401, u)),
    )
    assert got.asked == yp.GIVE_UP_AFTER
    assert "in a row failed" in got.stopped_early


def test_an_unmapped_label_is_reported_and_leaves_the_symbol_unclassified():
    got = yp.harvest(
        ["WEIRD"], pause=lambda _: None, session=harvesting(lambda u, n: summary("Conglomerates"))
    )
    assert got.buckets == {}
    assert got.unmapped == ("Conglomerates",)
    assert got.no_sector == []  # it answered with a sector; we just have no bucket for it
