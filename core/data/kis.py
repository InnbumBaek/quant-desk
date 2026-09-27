"""Reading Korea Investment's Open API, with the order path refused by construction.

This is the one credential on the desk that **can place an order**. Every other
key can only ever return data; this one, pointed at one more endpoint, moves
money. So the interesting part of this module is not what it reads -- it is what
it refuses.

**The refusal is an allowlist, not a convention.** A comment saying "we only
read here" is worth nothing the first time somebody adds an endpoint. So the
readable paths and the readable `tr_id` values are enumerated, `check_read_only`
is the only way to build a request, and anything not on the list is refused with
the reason. The order `tr_id` values are listed too, by name, so that a reader
sees they were considered and excluded rather than forgotten
(CLAUDE.md 1항·5항, ADR-0012, ADR-0025).

**The account read is not an order.** Reading a balance is a read, and the
reconciliation work (`pnl-recon`) needs it. It goes through the same allowlist.

**KIS hides its outcome behind HTTP 200.** `rt_cd` is "0" for success and
anything else is a refusal carrying `msg_cd`/`msg1`. Third source in a row with
this shape (ADR-0023, ADR-0024), and the same treatment: the outcome is read
before the payload.

**The token is a secret with a clock.** It lasts a day and KIS rate-limits its
issuance, so it is cached -- but never in the repository, never printed, and
never logged. `Token.masked` is what a log line is allowed to say about it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

#: The two hosts. Paper is the default everywhere; the live host is a separate
#: decision that a human makes (ADR-0012).
PAPER_HOST = "https://openapivts.koreainvestment.com:29443"
LIVE_HOST = "https://openapi.koreainvestment.com:9443"

#: Path prefixes this desk may ask for. Quotations are read-only by nature;
#: trading paths are allowed only for their `inquire-` endpoints, which return
#: state and submit nothing.
READ_ONLY_PREFIXES: tuple[str, ...] = (
    "/uapi/domestic-stock/v1/quotations/",
    "/uapi/domestic-stock/v1/trading/inquire-",
    "/uapi/overseas-price/v1/quotations/",
)

#: Any path holding one of these is refused whatever else matches. Belt and
#: braces: a prefix list can be widened by accident, and this cannot be
#: widened without deleting a line that says why it is here.
FORBIDDEN_FRAGMENTS: tuple[str, ...] = ("order", "revise", "cancel", "credit", "sell", "buy")

#: `tr_id` values this desk may send, and what each reads.
READ_ONLY_TR: dict[str, str] = {
    "FHKST01010100": "국내주식 현재가 시세",
    "FHKST01010400": "국내주식 기간별 시세",
    "VTTC8434R": "모의투자 주식잔고조회",
    "TTTC8434R": "실계좌 주식잔고조회",
}
#: Deliberately absent: 매수가능조회 (`VTTC8908R`/`TTTC8908R`). It only reads, but
#: it lives at `/uapi/domestic-stock/v1/trading/inquire-psbl-order`, and a path
#: holding "order" gets no exception here. The gate is worth more than the
#: convenience of one endpoint nothing needs yet.

#: Order `tr_id` values, listed so that excluding them is a recorded decision
#: and not an omission. Nothing in this repository may send one of these.
ORDER_TR: dict[str, str] = {
    "TTTC0802U": "실계좌 현금매수 주문",
    "TTTC0801U": "실계좌 현금매도 주문",
    "VTTC0802U": "모의투자 현금매수 주문",
    "VTTC0801U": "모의투자 현금매도 주문",
    "TTTC0803U": "실계좌 정정취소 주문",
    "VTTC0803U": "모의투자 정정취소 주문",
}

TOKEN_PATH = "/oauth2/tokenP"

OK = "0"
#: Refusals worth waiting out. A rate limit and a maintenance window pass; a
#: rejected app key does not, because waiting will not approve it.
RETRYABLE_MSG: frozenset[str] = frozenset({"EGW00201", "EGW00002", "EGW00133"})

#: How long before expiry a cached token is treated as spent. A token that
#: expires mid-request is a failure that looks like a refusal.
TOKEN_MARGIN = timedelta(minutes=10)


class KisShapeError(ValueError):
    """The payload is not a KIS response we can read."""


class KisRefused(RuntimeError):
    """KIS answered, and the answer is no."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"KIS {code}: {message or 'no message'}")

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_MSG


class OrderPathRefused(PermissionError):
    """Something asked for an endpoint that can move money. Never a warning."""


def host(paper: bool = True) -> str:
    return PAPER_HOST if paper else LIVE_HOST


def check_read_only(path: str, tr_id: str) -> None:
    """The only way to build a KIS request. Refuses anything that is not a read.

    Raises rather than returning a boolean: a caller that forgets to check a
    boolean gets an order; a caller that forgets to call this gets nothing at
    all, because every request helper here goes through it.
    """
    lowered = path.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in lowered:
            raise OrderPathRefused(
                f"{path} contains {fragment!r}. This desk's KIS credential is read-only: "
                "orders go through core/execution/orders.py and the pre-trade gate, never here"
            )
    if not any(lowered.startswith(prefix) for prefix in READ_ONLY_PREFIXES):
        raise OrderPathRefused(
            f"{path} is not one of the read-only prefixes ({', '.join(READ_ONLY_PREFIXES)})"
        )
    if tr_id in ORDER_TR:
        raise OrderPathRefused(f"tr_id {tr_id} is {ORDER_TR[tr_id]}, which this desk never sends")
    if tr_id not in READ_ONLY_TR:
        raise OrderPathRefused(
            f"tr_id {tr_id} is not on the read-only list ({', '.join(sorted(READ_ONLY_TR))}). "
            "Add it with a reason, or it is not sent"
        )


@dataclass(frozen=True)
class Token:
    """An access token and when it stops working. Never printed in full."""

    value: str
    expires_at: datetime
    token_type: str = "Bearer"

    @property
    def masked(self) -> str:
        """What a log line may say. Enough to tell two tokens apart, not to use one."""
        return f"{self.value[:4]}…{self.value[-2:]} ({len(self.value)} chars)" if self.value else "(empty)"

    def usable_at(self, now: datetime) -> bool:
        return now + TOKEN_MARGIN < self.expires_at

    @property
    def header(self) -> str:
        return f"{self.token_type} {self.value}"


@dataclass(frozen=True)
class Quote:
    """One symbol's current price, as KIS reports it."""

    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    high: float
    low: float
    open: float
    previous_close: float


@dataclass(frozen=True)
class Holding:
    """One position in the account. Quantities as the broker has them."""

    symbol: str
    name: str
    quantity: float
    available: float
    average_cost: float
    price: float
    market_value: float
    unrealised: float


@dataclass(frozen=True)
class Balance:
    """The account as the broker sees it, which is what reconciliation compares to."""

    holdings: tuple[Holding, ...]
    cash: float
    total_value: float

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(holding.symbol for holding in self.holdings)


def _number(row: Mapping[str, object], field: str, label: str) -> float:
    if field not in row:
        raise KisShapeError(
            f"{label}: no {field}. What the row carries: "
            f"{', '.join(sorted(str(key) for key in row)) or '(nothing)'}"
        )
    text = str(row[field] or "").strip().replace(",", "")
    if not text:
        # KIS sends an empty string for a field it has no value for. Absent,
        # never zero: a zero position and an unreported one are different facts.
        raise KisShapeError(f"{label}: {field} is empty, which is not a number")
    try:
        return float(text)
    except ValueError as error:
        raise KisShapeError(f"{label}: {field} is {text!r}, which is not a number") from error


def outcome_of(payload: Mapping[str, object]) -> str:
    """`rt_cd`, read before anything else. KIS puts refusals behind HTTP 200."""
    if not isinstance(payload, Mapping):
        raise KisShapeError(f"the response is a {type(payload).__name__}, not an object")
    if "rt_cd" not in payload:
        raise KisShapeError(
            "the response carries no rt_cd, so it is not a KIS reply. Its keys: "
            f"{', '.join(sorted(str(key) for key in payload)) or '(none)'}"
        )
    return str(payload["rt_cd"]).strip()


def _checked(payload: Mapping[str, object]) -> Mapping[str, object]:
    code = outcome_of(payload)
    if code != OK:
        raise KisRefused(
            str(payload.get("msg_cd") or f"rt_cd={code}"), str(payload.get("msg1") or "").strip()
        )
    return payload


def read_token(payload: Mapping[str, object], now: datetime | None = None) -> Token:
    """The OAuth reply. An expiry we cannot read is an error, not a guess."""
    if not isinstance(payload, Mapping):
        raise KisShapeError(f"the token response is a {type(payload).__name__}, not an object")
    value = str(payload.get("access_token") or "").strip()
    if not value:
        code = str(payload.get("error_code") or payload.get("msg_cd") or "token")
        raise KisRefused(code, str(payload.get("error_description") or payload.get("msg1") or "").strip())
    seconds = payload.get("expires_in")
    try:
        lifetime = int(str(seconds).strip())
    except (TypeError, ValueError) as error:
        raise KisShapeError(f"expires_in is {seconds!r}, which is not a number of seconds") from error
    if lifetime <= 0:
        raise KisShapeError(f"expires_in is {lifetime}, so the token is already spent")
    return Token(
        value=value,
        expires_at=(now or datetime.now(UTC)) + timedelta(seconds=lifetime),
        token_type=str(payload.get("token_type") or "Bearer").strip(),
    )


def read_quote(payload: Mapping[str, object], symbol: str) -> Quote:
    """`inquire-price`'s single-row output."""
    row = _checked(payload).get("output")
    if not isinstance(row, Mapping):
        raise KisShapeError(f"{symbol}: output is a {type(row).__name__}, not an object")
    label = f"{symbol} quote"
    return Quote(
        symbol=symbol,
        price=_number(row, "stck_prpr", label),
        change=_number(row, "prdy_vrss", label),
        change_pct=_number(row, "prdy_ctrt", label),
        volume=int(_number(row, "acml_vol", label)),
        high=_number(row, "stck_hgpr", label),
        low=_number(row, "stck_lwpr", label),
        open=_number(row, "stck_oprc", label),
        previous_close=_number(row, "stck_sdpr", label),
    )


def read_balance(payload: Mapping[str, object]) -> Balance:
    """`inquire-balance`'s two outputs: positions in output1, totals in output2."""
    checked = _checked(payload)
    rows = checked.get("output1")
    if not isinstance(rows, list):
        raise KisShapeError(f"output1 is a {type(rows).__name__}, not a list of holdings")
    totals = checked.get("output2")
    if isinstance(totals, list):
        totals = totals[0] if totals and isinstance(totals[0], Mapping) else None
    if not isinstance(totals, Mapping):
        raise KisShapeError("output2 carries no totals row, so cash and account value are unknown")

    holdings: list[Holding] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise KisShapeError(f"a {type(row).__name__} where a holding was expected")
        symbol = str(row.get("pdno") or "").strip()
        if not symbol:
            raise KisShapeError("a holding carries no pdno, so it cannot be matched to a position")
        label = f"{symbol} holding"
        quantity = _number(row, "hldg_qty", label)
        if quantity == 0:
            # KIS returns closed positions with zero quantity. Not a holding.
            continue
        holdings.append(
            Holding(
                symbol=symbol,
                name=str(row.get("prdt_name") or "").strip(),
                quantity=quantity,
                available=_number(row, "ord_psbl_qty", label),
                average_cost=_number(row, "pchs_avg_pric", label),
                price=_number(row, "prpr", label),
                market_value=_number(row, "evlu_amt", label),
                unrealised=_number(row, "evlu_pfls_amt", label),
            )
        )

    return Balance(
        holdings=tuple(sorted(holdings, key=lambda holding: holding.symbol)),
        cash=_number(totals, "dnca_tot_amt", "account totals"),
        total_value=_number(totals, "tot_evlu_amt", "account totals"),
    )
