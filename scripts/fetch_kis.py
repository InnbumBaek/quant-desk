"""Korea Investment Open API: prove the read path works, and only the read path.

Runs on the GitHub Actions runner. This container never reached either KIS host
(the paper port answered a bare GET with 500, the live host closed the
connection), and the credential lives only in Actions secrets (ADR-0012).

**What this script is for.** Not research data -- KRX already gives this desk
Korean prices (ADR-0022). What KIS gives is the *account*: the broker's own view
of positions and cash, which is the other side of every reconciliation, and a
live quote path for when execution becomes real. So this job's job is to prove
the credential works, the token cache behaves, and the account reads parse --
and to stop there.

**Why it prints no numbers.** A public job log is a public document, and an
account's holdings and cash are not public data (the same reason `state/` is
never committed). The report carries field names, counts and the outcome. Never
a price, never a quantity, never the token.

**Every request goes through the allowlist.** `request_read` calls
`check_read_only` before it builds anything, so there is no code path in this
repository that can reach an order endpoint with this credential. Orders, if
they ever exist, go through the center book and the pre-trade gate
(CLAUDE.md 1항·5항, ADR-0025).

**The live domain is not this script's decision.** `KIS_PAPER=0` is refused
here with the reason; switching domains is a G8 approval item.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from core.config import STATE, USER_AGENT, require_env
from core.data.kis import (
    TOKEN_PATH,
    Balance,
    KisRefused,
    KisShapeError,
    Quote,
    Token,
    check_read_only,
    host,
    read_balance,
    read_quote,
    read_token,
)

APP_KEY_ENV = "KIS_APP_KEY"
APP_SECRET_ENV = "KIS_APP_SECRET"
ACCOUNT_ENV = "KIS_ACCOUNT"
PAPER_ENV = "KIS_PAPER"

#: The token lasts a day and KIS rate-limits issuing it, so it is cached. Under
#: `state/`, which is git-ignored, because a cached token is a credential.
TOKEN_CACHE = STATE / "kis_token.json"

QUOTE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"
QUOTE_TR = "FHKST01010100"
BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
BALANCE_TR_PAPER = "VTTC8434R"
BALANCE_TR_LIVE = "TTTC8434R"

#: Samsung Electronics: the most liquid Korean name, so a quote that fails here
#: is the API and not the symbol.
DEFAULT_SYMBOL = "005930"

PAUSE_SECONDS = 1.0
RETRYABLE_HTTP = (429, 500, 502, 503, 504)
BACKOFF_SECONDS = (10.0, 30.0, 60.0)


class FetchError(RuntimeError):
    """KIS did not answer, or answered no. Never swallowed into an empty read."""


def redact(text: str, *secrets: str) -> str:
    """Every secret out of any string that might be printed.

    The app key, the app secret and the access token all travel in headers, and
    a header dump in an exception is a disclosed credential. Nothing prints
    without passing through here.
    """
    for secret in secrets:
        if secret and len(secret) > 6:
            text = text.replace(secret, "<REDACTED>")
    return text


def resolve_paper(env: Mapping[str, str] | None = None) -> bool:
    """Paper unless somebody explicitly says otherwise -- and then refuse anyway.

    ADR-0012 makes the live domain a G8 approval item. A script that could flip
    it from an environment variable would make that approval decorative.
    """
    values = env if env is not None else os.environ
    if str(values.get(PAPER_ENV, "1")).strip() == "0":
        raise FetchError(
            f"{PAPER_ENV}=0 asks for the live domain, which is a G8 approval item (ADR-0012). "
            "This script reads the paper domain only"
        )
    return True


def split_account(raw: str) -> tuple[str, str]:
    """KIS wants the account as CANO plus a two-digit product code."""
    text = raw.strip()
    if "-" in text:
        cano, _, product = text.partition("-")
    else:
        # Ten digits is the only unambiguous undashed form: eight for the
        # account, two for the product. Splitting an eight-digit string would
        # silently read 12345678 as account 123456 product 78 -- a real account
        # number, and the wrong one.
        cano, product = text[:8], text[8:]
    if not (cano.isdigit() and len(cano) == 8 and product.isdigit() and len(product) == 2):
        raise FetchError(
            f"{ACCOUNT_ENV} is not an account number like 12345678-01 "
            f"(eight digits, then a two-digit product code; got {len(text)} characters)"
        )
    return cano, product


def _explain(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read()
    except OSError:
        return "no response body"
    return " ".join(body.decode("utf-8", errors="replace").split())[:300] or "empty response body"


def _send(
    request: urllib.request.Request,
    *secrets: str,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    last = ""
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed hosts
                return response.read()
        except urllib.error.HTTPError as error:
            last = redact(f"HTTP {error.code}: {_explain(error)}", *secrets)
            if error.code not in RETRYABLE_HTTP:
                raise FetchError(last) from error
        except OSError as error:
            last = redact(f"{type(error).__name__}: {error}", *secrets)
        if attempt < len(BACKOFF_SECONDS):
            sleep(BACKOFF_SECONDS[attempt])
    raise FetchError(f"KIS did not answer after {len(BACKOFF_SECONDS) + 1} attempts; last: {last}")


def _payload(raw: bytes, label: str, *secrets: str) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        head = redact(" ".join(raw.decode("utf-8", errors="replace").split())[:200], *secrets)
        raise FetchError(f"{label}: the reply is not JSON: {head!r}") from error
    if not isinstance(payload, dict):
        raise FetchError(f"{label}: the reply is a {type(payload).__name__}, not an object")
    return payload


def load_cached_token(path: Path = TOKEN_CACHE, now: datetime | None = None) -> Token | None:
    """A cached token if it is still good. Any doubt returns None and we ask again."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        token = Token(
            value=str(cached["value"]),
            expires_at=datetime.fromisoformat(str(cached["expires_at"])),
            token_type=str(cached.get("token_type") or "Bearer"),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return token if token.usable_at(now or datetime.now(UTC)) else None


def save_token(token: Token, path: Path = TOKEN_CACHE) -> Path:
    """The token to a git-ignored file, readable only by its owner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "value": token.value,
                "expires_at": token.expires_at.isoformat(),
                "token_type": token.token_type,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def issue_token(
    app_key: str,
    app_secret: str,
    base: str,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Token:
    body = json.dumps(
        {"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}{TOKEN_PATH}",
        data=body,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    raw = _send(request, app_key, app_secret, sleep=sleep)
    payload = _payload(raw, "token", app_key, app_secret)
    try:
        return read_token(payload, now=now)
    except KisRefused as refused:
        raise FetchError(redact(str(refused), app_key, app_secret)) from refused
    except KisShapeError as error:
        raise FetchError(redact(str(error), app_key, app_secret)) from error


def get_token(
    app_key: str,
    app_secret: str,
    base: str,
    cache: Path = TOKEN_CACHE,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Token, bool]:
    """A usable token, reusing the cache when it has one. True means newly issued."""
    cached = load_cached_token(cache, now)
    if cached is not None:
        return cached, False
    token = issue_token(app_key, app_secret, base, now=now, sleep=sleep)
    save_token(token, cache)
    return token, True


def request_read(
    base: str,
    path: str,
    tr_id: str,
    params: dict[str, str],
    token: Token,
    app_key: str,
    app_secret: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """The only request builder in this module, and it starts with the allowlist.

    `check_read_only` raises rather than returning a verdict, so there is no
    version of this function that quietly sends an order.
    """
    check_read_only(path, tr_id)
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json; charset=utf-8",
            "authorization": token.header,
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": tr_id,
        },
    )
    raw = _send(request, app_key, app_secret, token.value, sleep=sleep)
    return _payload(raw, path, app_key, app_secret, token.value)


def fetch_quote(
    symbol: str,
    base: str,
    token: Token,
    app_key: str,
    app_secret: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Quote, list[str]]:
    payload = request_read(
        base,
        QUOTE_PATH,
        QUOTE_TR,
        {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        token,
        app_key,
        app_secret,
        sleep=sleep,
    )
    row = payload.get("output")
    fields = sorted(str(key) for key in row) if isinstance(row, Mapping) else []
    try:
        return read_quote(payload, symbol), fields
    except KisRefused as refused:
        raise FetchError(f"{symbol}: {redact(str(refused), app_key, app_secret, token.value)}") from refused
    except KisShapeError as error:
        raise FetchError(f"{symbol}: {redact(str(error), app_key, app_secret, token.value)}") from error


def fetch_balance(
    account: str,
    base: str,
    token: Token,
    app_key: str,
    app_secret: str,
    paper: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Balance, list[str]]:
    cano, product = split_account(account)
    payload = request_read(
        base,
        BALANCE_PATH,
        BALANCE_TR_PAPER if paper else BALANCE_TR_LIVE,
        {
            "CANO": cano,
            "ACNT_PRDT_CD": product,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
        token,
        app_key,
        app_secret,
        sleep=sleep,
    )
    rows = payload.get("output1")
    first = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], Mapping) else {}
    fields = sorted(str(key) for key in first)
    try:
        return read_balance(payload), fields
    except KisRefused as refused:
        raise FetchError(f"balance: {redact(str(refused), app_key, app_secret, token.value)}") from refused
    except KisShapeError as error:
        raise FetchError(f"balance: {redact(str(error), app_key, app_secret, token.value)}") from error


def check(
    symbol: str = DEFAULT_SYMBOL,
    app_key: str | None = None,
    app_secret: str | None = None,
    account: str | None = None,
    cache: Path = TOKEN_CACHE,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Token, one quote, and the account -- reported as shapes and counts only.

    A public log gets field names and whether each read worked. Prices and
    quantities stay out: an account is not public data.
    """
    paper = resolve_paper(env)
    base = host(paper=paper)
    key = app_key or require_env(APP_KEY_ENV)
    secret = app_secret or require_env(APP_SECRET_ENV)
    values = env if env is not None else os.environ
    number = account if account is not None else str(values.get(ACCOUNT_ENV, "")).strip()

    token, issued = get_token(key, secret, base, cache=cache, now=now, sleep=sleep)
    report: dict[str, object] = {
        "domain": "paper" if paper else "live",
        "token": token.masked,
        "token_issued": issued,
        "token_expires_at": token.expires_at.isoformat(timespec="seconds"),
        "reads": {},
        "refused": [],
    }
    reads: dict[str, object] = report["reads"]  # type: ignore[assignment]
    refused: list[str] = report["refused"]  # type: ignore[assignment]

    sleep(PAUSE_SECONDS)
    try:
        quote, fields = fetch_quote(symbol, base, token, key, secret, sleep=sleep)
    except FetchError as error:
        refused.append(f"quote {symbol}: {error}")
    else:
        # Field names and whether the price was positive. Not the price.
        reads["quote"] = {"symbol": quote.symbol, "fields": fields, "price_positive": quote.price > 0}

    if not number:
        refused.append(f"{ACCOUNT_ENV} is not set, so the account read was skipped")
        return report

    sleep(PAUSE_SECONDS)
    try:
        balance, fields = fetch_balance(number, base, token, key, secret, paper=paper, sleep=sleep)
    except FetchError as error:
        refused.append(f"balance: {error}")
    else:
        reads["balance"] = {"holdings": len(balance.holdings), "fields": fields}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the KIS read path")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args(argv)

    try:
        report = check(symbol=args.symbol)
    except (FetchError, RuntimeError) as error:
        print(f"KIS check failed: {error}", file=sys.stderr)
        return 1

    reads = report["reads"]
    print(f"## KIS {report['domain']} domain, {len(reads)} read(s) working")  # type: ignore[arg-type]
    print(f"- token: {report['token']}, expires {report['token_expires_at']}")
    for name, detail in reads.items():  # type: ignore[union-attr]
        print(f"- {name}: {detail}")
    refused = report["refused"]
    if isinstance(refused, list) and refused:
        print(f"- refused ({len(refused)}):")
        for reason in refused:
            print(f"  - {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
