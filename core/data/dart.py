"""Reading DART's disclosure list, where the status code is the first fact.

DART answers almost everything with HTTP 200 and puts the real outcome in a
`status` field. A rejected key, a system maintenance window and a genuinely
quiet day all arrive as 200, and a reader that only checks the transport sees
three empty lists. That is the failure this module exists to prevent: the
status is read before anything else, and each code is sorted into one of three
kinds -- an answer, a refusal, or nothing found.

**`013` is an answer, not an error.** "조회된 데이타가 없습니다" means the
window really held no filings, which happens on a holiday. Treating it as a
failure would make every Korean holiday a red job; treating a *refusal* as an
empty day would make a rejected key look like a quiet market. Both mistakes
have been made in this repository already, on other sources, so they are
separated by type here.

**The shape is asserted, not assumed.** Nobody here has held a keyed DART
response: this container cannot reach the host and the probe only got the
key-less error page. So a missing field raises with the field names that were
actually present, and the first keyed run on the runner teaches us the schema
in one line of a job log (the same discipline as `core/data/krx.py`, ADR-0022).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

#: DART's status codes, as its documentation lists them. The text is DART's;
#: the grouping below is ours and is what the caller acts on.
STATUS_TEXT: dict[str, str] = {
    "000": "정상",
    "010": "등록되지 않은 키입니다",
    "011": "사용할 수 없는 키입니다",
    "012": "접근할 수 없는 IP입니다",
    "013": "조회된 데이타가 없습니다",
    "014": "파일이 존재하지 않습니다",
    "020": "요청 제한을 초과하였습니다",
    "021": "조회 가능한 회사 개수가 초과하였습니다",
    "100": "필드의 부적절한 값입니다",
    "101": "부적절한 접근입니다",
    "800": "시스템 점검으로 인한 서비스가 중지 중입니다",
    "900": "정의되지 않은 오류가 발생하였습니다",
    "901": "사용자 계정의 개인정보보호가 요청되었습니다",
}

OK = "000"
NO_DATA = "013"
#: Codes worth trying again later. A rate limit and a maintenance window pass;
#: a bad key does not, because waiting will not register it.
RETRYABLE_STATUS = frozenset({"020", "800", "900"})

#: The fields a disclosure row must carry, and what each becomes.
FIELDS: dict[str, str] = {
    "corp_code": "corp_code",
    "corp_name": "corp_name",
    "stock_code": "stock_code",
    "corp_cls": "corp_class",
    "report_nm": "report",
    "rcept_no": "receipt_no",
    "flr_nm": "filer",
    "rcept_dt": "filed_on",
}


class DartShapeError(ValueError):
    """The payload is not a DART response we can read."""


class DartRefused(RuntimeError):
    """DART answered, and the answer is no. Carries the code so a caller can retry."""

    def __init__(self, status: str, message: str = ""):
        self.status = status
        self.message = message or STATUS_TEXT.get(status, "")
        super().__init__(f"DART status {status}: {self.message or 'no message'}")

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUS


@dataclass(frozen=True)
class Disclosure:
    """One filing's metadata. Never the filing itself."""

    corp_code: str
    corp_name: str
    stock_code: str
    corp_class: str
    report: str
    receipt_no: str
    filer: str
    filed_on: date

    @property
    def listed(self) -> bool:
        """Whether this filer has a ticker. Unlisted filers file too."""
        return bool(self.stock_code.strip())

    @property
    def url(self) -> str:
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.receipt_no}"


@dataclass(frozen=True)
class Page:
    """One page of the list, and enough to know whether another follows."""

    disclosures: tuple[Disclosure, ...]
    page_no: int
    total_pages: int
    total_count: int

    @property
    def has_more(self) -> bool:
        return self.page_no < self.total_pages


def _day(raw: object) -> date:
    text = str(raw or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise DartShapeError(f"rcept_dt is {text!r}, expected eight digits like 20260925")
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def read_row(row: Mapping[str, object]) -> Disclosure:
    missing = [key for key in FIELDS if key not in row]
    if missing:
        raise DartShapeError(
            f"the row is missing {', '.join(missing)}. What it does carry: "
            f"{', '.join(sorted(str(key) for key in row)) or '(nothing)'}"
        )
    receipt = str(row["rcept_no"] or "").strip()
    if not receipt:
        raise DartShapeError("a row carries no rcept_no, so the filing cannot be identified or deduped")
    return Disclosure(
        corp_code=str(row["corp_code"] or "").strip(),
        corp_name=str(row["corp_name"] or "").strip(),
        stock_code=str(row["stock_code"] or "").strip(),
        corp_class=str(row["corp_cls"] or "").strip(),
        report=" ".join(str(row["report_nm"] or "").split()),
        receipt_no=receipt,
        filer=str(row["flr_nm"] or "").strip(),
        filed_on=_day(row["rcept_dt"]),
    )


def status_of(payload: Mapping[str, object]) -> str:
    """The status code, or an error if there is not one. Read before anything else."""
    if not isinstance(payload, Mapping):
        raise DartShapeError(f"the response is a {type(payload).__name__}, not an object")
    if "status" not in payload:
        raise DartShapeError(
            "the response carries no status field, so it is not a DART reply. Its keys: "
            f"{', '.join(sorted(str(key) for key in payload)) or '(none)'}"
        )
    return str(payload["status"]).strip()


def read_page(payload: Mapping[str, object]) -> Page:
    """One list page. `013` comes back as an empty page, every other refusal raises."""
    status = status_of(payload)
    if status == NO_DATA:
        # A real answer: the window held nothing. An empty page rather than an
        # exception, because a holiday must not turn a weekly job red.
        return Page((), page_no=1, total_pages=1, total_count=0)
    if status != OK:
        raise DartRefused(status, str(payload.get("message") or ""))

    rows = payload.get("list")
    if not isinstance(rows, list):
        raise DartShapeError(
            f"status was 000 but `list` is a {type(rows).__name__}. "
            "A success with no rows is what status 013 is for"
        )
    disclosures = tuple(read_row(row) if isinstance(row, Mapping) else _not_a_row(row) for row in rows)
    return Page(
        disclosures=disclosures,
        page_no=_count(payload, "page_no", default=1),
        total_pages=_count(payload, "total_page", default=1),
        total_count=_count(payload, "total_count", default=len(disclosures)),
    )


def _not_a_row(row: object) -> Disclosure:
    raise DartShapeError(f"a {type(row).__name__} where a disclosure row was expected")


def _count(payload: Mapping[str, object], key: str, default: int) -> int:
    raw = payload.get(key, default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as error:
        raise DartShapeError(f"{key} is {raw!r}, which is not a number") from error
