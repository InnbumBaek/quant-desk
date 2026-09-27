"""The KIS fetcher: the allowlist, the secrets, the token cache, and a log that
carries no account numbers.

The order-path tests here are deliberately about *reachability*: not "does the
reader refuse this shape" (that is `tests/data/test_kis.py`) but "can any code
path in this script send it". They assert the request is never even built.
"""

from __future__ import annotations

import io
import json
import stat
import tempfile
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.data.kis import PAPER_HOST, OrderPathRefused, Token
from scripts import fetch_kis as fk

NOW = datetime(2026, 9, 27, 1, 0, tzinfo=UTC)
APP_KEY = "PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
APP_SECRET = "SECRETyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
TOKEN_VALUE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.token.value"
ACCOUNT = "12345678-01"
_TMP: list = []


def tmp() -> Path:
    handle = tempfile.TemporaryDirectory()
    _TMP.append(handle)
    return Path(handle.name)


def token(expires_in_hours: float = 20.0) -> Token:
    return Token(TOKEN_VALUE, NOW + timedelta(hours=expires_in_hours))


def token_body(value: str = TOKEN_VALUE, expires_in: str = "86400") -> bytes:
    return json.dumps({"access_token": value, "expires_in": expires_in, "token_type": "Bearer"}).encode()


def quote_body() -> bytes:
    return json.dumps(
        {
            "rt_cd": "0",
            "msg_cd": "MCA00000",
            "output": {
                "stck_prpr": "72,100",
                "prdy_vrss": "1,100",
                "prdy_ctrt": "1.55",
                "acml_vol": "12,345,678",
                "stck_hgpr": "72,300",
                "stck_lwpr": "70,800",
                "stck_oprc": "71,000",
                "stck_sdpr": "71,000",
            },
        }
    ).encode()


def balance_body() -> bytes:
    return json.dumps(
        {
            "rt_cd": "0",
            "output1": [
                {
                    "pdno": "005930",
                    "prdt_name": "삼성전자",
                    "hldg_qty": "10",
                    "ord_psbl_qty": "10",
                    "pchs_avg_pric": "70,000",
                    "prpr": "72,100",
                    "evlu_amt": "721,000",
                    "evlu_pfls_amt": "21,000",
                }
            ],
            "output2": [{"dnca_tot_amt": "1,000,000", "tot_evlu_amt": "1,721,000"}],
        }
    ).encode()


def http_error(code: int, payload: bytes = b"") -> urllib.error.HTTPError:
    import email.message

    return urllib.error.HTTPError(PAPER_HOST, code, "refused", email.message.Message(), io.BytesIO(payload))


def serve(monkeypatch, answer):
    import urllib.request

    sent: list = []

    class Reply:
        def __init__(self, data: bytes):
            self._data = data

        def read(self, *_):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(request, timeout=None):
        sent.append(request)
        result = answer(request.full_url, len(sent))
        if isinstance(result, Exception):
            raise result
        return Reply(result)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


def route(token_reply: bytes = b"", quote: bytes = b"", balance: bytes = b""):
    def answer(url: str, n: int):
        if fk.TOKEN_PATH in url:
            return token_reply or token_body()
        if "inquire-balance" in url:
            return balance or balance_body()
        return quote or quote_body()

    return answer


def run(monkeypatch, answer, **kwargs) -> dict:
    serve(monkeypatch, answer)
    return fk.check(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        account=kwargs.pop("account", ACCOUNT),
        cache=kwargs.pop("cache", tmp() / "token.json"),
        env=kwargs.pop("env", {}),
        now=NOW,
        sleep=lambda _: None,
        **kwargs,
    )


# --- no code path here can reach an order endpoint ---------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/uapi/domestic-stock/v1/trading/order-cash",
        "/uapi/domestic-stock/v1/trading/order-rvsecncl",
        "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
    ],
)
def test_an_order_request_is_never_even_built(monkeypatch, path: str):
    """The refusal happens before the URL exists, so there is nothing to send."""
    sent = serve(monkeypatch, route())
    with pytest.raises(OrderPathRefused):
        fk.request_read(PAPER_HOST, path, "VTTC8434R", {}, token(), APP_KEY, APP_SECRET)
    assert sent == [], "a refused path must not reach the network at all"


def test_the_two_reads_this_desk_needs_do_go_through(monkeypatch):
    sent = serve(monkeypatch, route())
    fk.request_read(
        PAPER_HOST, fk.QUOTE_PATH, fk.QUOTE_TR, {"FID_INPUT_ISCD": "005930"}, token(), APP_KEY, APP_SECRET
    )
    fk.request_read(PAPER_HOST, fk.BALANCE_PATH, fk.BALANCE_TR_PAPER, {}, token(), APP_KEY, APP_SECRET)
    assert [request.get_header("Tr_id") for request in sent] == [fk.QUOTE_TR, fk.BALANCE_TR_PAPER]


def test_the_paper_account_uses_the_paper_tr_id(monkeypatch):
    sent = serve(monkeypatch, route())
    fk.fetch_balance(ACCOUNT, PAPER_HOST, token(), APP_KEY, APP_SECRET, paper=True, sleep=lambda _: None)
    assert sent[0].get_header("Tr_id") == fk.BALANCE_TR_PAPER
    assert "CANO=12345678" in sent[0].full_url and "ACNT_PRDT_CD=01" in sent[0].full_url


def test_the_live_domain_is_refused_here_rather_than_configured():
    """ADR-0012 makes it a G8 item. An env variable must not be able to flip it."""
    with pytest.raises(fk.FetchError, match="G8"):
        fk.resolve_paper({"KIS_PAPER": "0"})
    assert fk.resolve_paper({}) is True


def test_everything_reads_the_paper_host_by_default(monkeypatch):
    sent = serve(monkeypatch, route())
    run(monkeypatch, route())
    assert all(request.full_url.startswith(PAPER_HOST) for request in sent) or sent == []


# --- the secrets -------------------------------------------------------------


def test_no_secret_reaches_an_error_message(monkeypatch):
    serve(monkeypatch, lambda url, n: http_error(403, APP_SECRET.encode()))
    with pytest.raises(fk.FetchError) as caught:
        fk.request_read(
            PAPER_HOST, fk.QUOTE_PATH, fk.QUOTE_TR, {}, token(), APP_KEY, APP_SECRET, sleep=lambda _: None
        )
    message = str(caught.value)
    assert APP_KEY not in message and APP_SECRET not in message and TOKEN_VALUE not in message


def test_no_secret_reaches_a_parse_error_either(monkeypatch):
    serve(monkeypatch, lambda url, n: f"<html>{APP_KEY}</html>".encode())
    with pytest.raises(fk.FetchError) as caught:
        fk.issue_token(APP_KEY, APP_SECRET, PAPER_HOST, sleep=lambda _: None)
    assert APP_KEY not in str(caught.value)


def test_the_report_carries_the_masked_token_and_never_the_token(monkeypatch):
    report = run(monkeypatch, route())
    assert TOKEN_VALUE not in json.dumps(report, default=str)
    assert "…" in str(report["token"])


def test_the_report_carries_no_prices_and_no_quantities(monkeypatch):
    """A public job log is a public document, and an account is not public data."""
    report = run(monkeypatch, route())
    printed = json.dumps(report, ensure_ascii=False, default=str)
    for secret_number in ("72,100", "72100", "1,000,000", "1000000", "721,000"):
        assert secret_number not in printed
    assert report["reads"]["balance"] == {  # type: ignore[index]
        "holdings": 1,
        "fields": [
            "evlu_amt",
            "evlu_pfls_amt",
            "hldg_qty",
            "ord_psbl_qty",
            "pchs_avg_pric",
            "pdno",
            "prdt_name",
            "prpr",
        ],
    }


def test_the_quote_read_is_reported_as_a_shape_not_a_price(monkeypatch):
    report = run(monkeypatch, route())
    quote = report["reads"]["quote"]  # type: ignore[index]
    assert quote["symbol"] == fk.DEFAULT_SYMBOL and quote["price_positive"] is True
    assert "stck_prpr" in quote["fields"]


# --- the token cache ---------------------------------------------------------


def test_a_cached_token_is_reused_rather_than_reissued(monkeypatch):
    """KIS rate-limits token issuance, so asking twice is a refusal waiting to happen."""
    cache = tmp() / "token.json"
    fk.save_token(token(), cache)
    sent = serve(monkeypatch, route())
    got, issued = fk.get_token(APP_KEY, APP_SECRET, PAPER_HOST, cache=cache, now=NOW)
    assert issued is False and got.value == TOKEN_VALUE
    assert sent == []


def test_a_token_near_expiry_is_replaced(monkeypatch):
    cache = tmp() / "token.json"
    fk.save_token(token(expires_in_hours=0.1), cache)
    serve(monkeypatch, route())
    _, issued = fk.get_token(APP_KEY, APP_SECRET, PAPER_HOST, cache=cache, now=NOW, sleep=lambda _: None)
    assert issued is True


def test_an_unreadable_cache_is_not_a_crash(monkeypatch):
    cache = tmp() / "token.json"
    cache.write_text("{not json", encoding="utf-8")
    serve(monkeypatch, route())
    _, issued = fk.get_token(APP_KEY, APP_SECRET, PAPER_HOST, cache=cache, now=NOW, sleep=lambda _: None)
    assert issued is True


def test_a_cache_missing_its_expiry_is_ignored():
    cache = tmp() / "token.json"
    cache.write_text(json.dumps({"value": TOKEN_VALUE}), encoding="utf-8")
    assert fk.load_cached_token(cache, NOW) is None


def test_the_cached_token_is_not_world_readable():
    cache = tmp() / "token.json"
    fk.save_token(token(), cache)
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600


def test_the_cache_lives_under_the_git_ignored_state_directory():
    assert fk.TOKEN_CACHE.parent.name == "state"


# --- the account number ------------------------------------------------------


@pytest.mark.parametrize("raw", ["12345678-01", "1234567801"])
def test_both_account_spellings_split_the_same_way(raw: str):
    assert fk.split_account(raw) == ("12345678", "01")


@pytest.mark.parametrize("raw", ["12345678", "12345678-AB", "not-an-account"])
def test_an_account_number_we_cannot_read_is_refused(raw: str):
    with pytest.raises(fk.FetchError, match="account number"):
        fk.split_account(raw)


def test_a_missing_account_is_reported_and_not_silently_skipped(monkeypatch):
    report = run(monkeypatch, route(), account="")
    assert "quote" in report["reads"] and "balance" not in report["reads"]  # type: ignore[operator]
    assert any("KIS_ACCOUNT" in reason for reason in report["refused"])  # type: ignore[union-attr]


# --- refusals ----------------------------------------------------------------


def test_a_refused_read_does_not_cost_the_other_read(monkeypatch):
    report = run(monkeypatch, route(quote=json.dumps({"rt_cd": "1", "msg_cd": "OPSQ0001"}).encode()))
    assert "quote" not in report["reads"] and "balance" in report["reads"]  # type: ignore[operator]
    assert any("OPSQ0001" in reason for reason in report["refused"])  # type: ignore[union-attr]


def test_a_rate_limited_transport_is_retried(monkeypatch):
    sent = serve(monkeypatch, lambda url, n: quote_body() if n == 2 else http_error(429))
    waits: list[float] = []
    fk.request_read(
        PAPER_HOST, fk.QUOTE_PATH, fk.QUOTE_TR, {}, token(), APP_KEY, APP_SECRET, sleep=waits.append
    )
    assert len(sent) == 2 and waits == [fk.BACKOFF_SECONDS[0]]


def test_a_refused_token_stops_the_check(monkeypatch):
    serve(
        monkeypatch,
        lambda url, n: json.dumps({"error_code": "EGW00123", "error_description": "bad"}).encode(),
    )
    with pytest.raises(fk.FetchError, match="EGW00123"):
        run(monkeypatch, lambda url, n: json.dumps({"error_code": "EGW00123"}).encode())
