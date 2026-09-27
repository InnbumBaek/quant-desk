"""KSIC industry label -> concentration bucket, for the Korean listings file.

`core/data/sic.py` does this for the US side from the SEC's own four-digit code.
Korea has no equivalent code in anything this desk can reach: KIND publishes a
**Korean-language KSIC sub-class name** and nothing numeric
(`registry/universe/kr_industry.source.json`, measured 2026-09-27). So the key
here is the label text itself, matched exactly.

**Exact match, never substring.** `기타 금융업` and `기타 금속 가공제품 제조업`
share a prefix, `전기업` is a substring of nothing but sits one character from
`전기 통신업`, and a substring rule would put a steelmaker in financials the
first time the vendor reworded a line. A label this table has not seen returns
`None` and shows up in the coverage report, which is the failure this desk can
act on; a wrong bucket is the one it cannot see.

**How each line was decided.** In order:

1. *The label's own words decide.* These are KSIC sub-class names and they
   describe the business plainly. `통신 및 방송 장비 제조업` is communications
   equipment, which is technology hardware, whatever is actually listed under it.
2. *Where the words leave it open, match `core/data/sic.py`.* The limit
   `sector_max` is fund-level: a business that counts as industrials in New York
   and materials in Seoul makes the cap under-measure real concentration. Five
   places where the two tables disagreed were fixed in `sic.py` rather than
   copied here; ADR-0029 lists them.
3. *Never a default bucket.* Rule 1 answers or the line is not written.

**The two tables are coarse for the same reason.** A classification fine enough
to never bind is the same as no limit at all, so 158 labels collapse into the
eleven buckets `sic.py` already produces. Fund-level buckets in
`classification.SECTORS` (`us_equity_broad` and the rest) describe ETFs and
cannot come out of an operating company's industry, so they are not here.

**Where this table is knowingly coarse**, because the label is coarser than the
bucket and a finer source would be needed to do better (ADR-0029):

- `특수 목적용 기계 제조업` (169 names, the second largest label) is
  special-purpose machinery, which reads as industrials and is where Korea's
  semiconductor **equipment** makers sit. GICS would call those technology. This
  is the single largest misclassification risk in the table.
- `기타 금융업` (106) holds most Korean holding companies. GICS looks through a
  holdco to what it owns; a label cannot.
- `자연과학 및 공학 연구개발업` (80) is research and development of every kind,
  mapped to health_care because that is where `sic.py` sends SIC 8731 for the
  same reason -- it is where biotech files.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from core.data.sic import BUCKETS

#: Every distinct label in `registry/universe/kr_industry.source.json` as of the
#: 2026-09-27 census, with that run's count beside it. The count is there so a
#: reviewer reads the table by weight: a line carrying 192 names moves the limit
#: and a line carrying 1 does not. Counts are a snapshot and are not read by any
#: code -- `industry_counts` in the sidecar is the live number.
#:
#: Ordered as the census orders it (by label), so a diff against a later census
#: is readable.
LABELS: dict[str, str] = {
    # --- metals, chemicals, paper, building materials ------------------------
    "1차 비철금속 제조업": "materials",  # 23
    "1차 철강 제조업": "materials",  # 60
    "고무제품 제조업": "materials",  # 8
    "골판지, 종이 상자 및 종이용기 제조업": "materials",  # 9
    "금속 주조업": "materials",  # 3
    "기초 화학물질 제조업": "materials",  # 50
    "기타 비금속 광물제품 제조업": "materials",  # 9
    "기타 종이 및 판지 제품 제조업": "materials",  # 3
    "기타 화학제품 제조업": "materials",  # 102
    "나무제품 제조업": "materials",  # 2
    "내화, 비내화 요업제품 제조업": "materials",  # 5
    "비료, 농약 및 살균, 살충제 제조업": "materials",  # 8
    "시멘트, 석회, 플라스터 및 그 제품 제조업": "materials",  # 17
    "유리 및 유리제품 제조업": "materials",  # 5
    "제재 및 목재 가공업": "materials",  # 4
    "펄프, 종이 및 판지 제조업": "materials",  # 10
    "플라스틱제품 제조업": "materials",  # 45
    "합성고무 및 플라스틱 물질 제조업": "materials",  # 8
    "화학섬유 제조업": "materials",  # 10
    # --- energy ---------------------------------------------------------------
    "석유 정제품 제조업": "energy",  # 5
    # 연료 소매업 is LPG and fuel distribution, not filling stations; `sic.py`
    # sends SIC 5541 (gasoline service stations) to consumer_discretionary
    # because that code really is a retail forecourt. Different businesses, one
    # similar-looking label, so rule 1 wins over rule 2 here.
    "연료 소매업": "energy",  # 3
    # --- utilities ------------------------------------------------------------
    "연료용 가스 제조 및 배관공급업": "utilities",  # 9
    "전기업": "utilities",  # 3
    "증기, 냉·온수 및 공기조절 공급업": "utilities",  # 1
    # --- industrials: construction and engineering ----------------------------
    "건물 건설업": "industrials",  # 24
    "건물설비 설치 공사업": "industrials",  # 2
    "건축기술, 엔지니어링 및 관련 기술 서비스업": "industrials",  # 18
    "기반조성 및 시설물 축조관련 전문공사업": "industrials",  # 4
    "실내건축 및 건축마무리 공사업": "industrials",  # 5
    "전기 및 통신 공사업": "industrials",  # 8
    "토목 건설업": "industrials",  # 17
    "해체, 선별 및 원료 재생업": "industrials",  # 1
    "폐기물 처리업": "industrials",  # 2
    # --- industrials: machinery, electrical equipment, transport equipment ----
    "구조용 금속제품, 탱크 및 증기발생기 제조업": "industrials",  # 21
    "그외 기타 운송장비 제조업": "industrials",  # 5
    "기타 금속 가공제품 제조업": "industrials",  # 28
    "기타 전기장비 제조업": "industrials",  # 15
    "무기 및 총포탄 제조업": "industrials",  # 3
    "선박 및 보트 건조업": "industrials",  # 13
    "일반 목적용 기계 제조업": "industrials",  # 51
    # Batteries read as electrical equipment, which is how GICS classifies the
    # cell makers. `sic.py` gained a carve-out for SIC 3691 to agree (ADR-0029).
    "일차전지 및 이차전지 제조업": "industrials",  # 21
    "전구 및 조명장치 제조업": "industrials",  # 5
    "전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업": "industrials",  # 41
    "절연선 및 케이블 제조업": "industrials",  # 8
    "측정, 시험, 항해, 제어 및 기타 정밀기기 제조업; 광학기기 제외": "industrials",  # 30
    # Reads as machinery. It is also where semiconductor equipment sits, which
    # GICS calls technology -- the table's largest known coarseness.
    "특수 목적용 기계 제조업": "industrials",  # 169
    "항공기,우주선 및 부품 제조업": "industrials",  # 11
    # --- industrials: transport, wholesale, business services -----------------
    "건축자재, 철물 및 난방장치 도매업": "industrials",  # 1
    "경비, 경호 및 탐정업": "industrials",  # 1
    "그외 기타 전문, 과학 및 기술 서비스업": "industrials",  # 17
    "기계장비 및 관련 물품 도매업": "industrials",  # 24
    "기타 과학기술 서비스업": "industrials",  # 5
    "기타 사업지원 서비스업": "industrials",  # 8
    "기타 운송관련 서비스업": "industrials",  # 7
    "기타 전문 도매업": "industrials",  # 36
    "기타 전문 서비스업": "industrials",  # 2
    "도로 화물 운송업": "industrials",  # 7
    "사업시설 유지·관리 서비스업": "industrials",  # 1
    "산업용 기계 및 장비 임대업": "industrials",  # 2
    "상품 종합 도매업": "industrials",  # 21
    "상품 중개업": "industrials",  # 8
    "생활용품 도매업": "industrials",  # 18
    "시장조사 및 여론조사업": "industrials",  # 1
    "운송장비 임대업": "industrials",  # 2
    "육상 여객 운송업": "industrials",  # 3
    "전문디자인업": "industrials",  # 2
    "항공 여객 운송업": "industrials",  # 6
    "해상 운송업": "industrials",  # 6
    # Head offices and management consulting: most Korean holding companies that
    # carry an operating label at all are here or in 기타 금융업.
    "회사 본부 및 경영 컨설팅 서비스업": "industrials",  # 11
    # --- consumer staples -----------------------------------------------------
    "곡물가공품, 전분 및 전분제품 제조업": "consumer_staples",  # 8
    "과실, 채소 가공 및 저장 처리업": "consumer_staples",  # 3
    "기타 식품 제조업": "consumer_staples",  # 37
    "담배 제조업": "consumer_staples",  # 1
    "도시락 및 식사용 조리식품 제조업": "consumer_staples",  # 1
    "도축, 육류 가공 및 저장 처리업": "consumer_staples",  # 7
    "동·식물성 유지 및 낙농제품 제조업": "consumer_staples",  # 4
    "동물용 사료 및 조제식품 제조업": "consumer_staples",  # 10
    "떡, 빵 및 과자류 제조업": "consumer_staples",  # 1
    "비알코올음료 및 얼음 제조업": "consumer_staples",  # 2
    "수산물 가공 및 저장 처리업": "consumer_staples",  # 6
    "알코올음료 제조업": "consumer_staples",  # 10
    "어로 어업": "consumer_staples",  # 2
    # Food, drink and tobacco distribution. `sic.py` gained carve-outs for SIC
    # 5140-5159 and 5180-5182 so the two markets agree (ADR-0029).
    "음·식료품 및 담배 도매업": "consumer_staples",  # 9
    "음·식료품 및 담배 소매업": "consumer_staples",  # 1
    "산업용 농·축산물 및 동·식물 도매업": "consumer_staples",  # 3
    "작물 재배업": "consumer_staples",  # 2
    # --- consumer discretionary ----------------------------------------------
    "가구 제조업": "consumer_discretionary",  # 9
    "가전제품 및 정보통신장비 소매업": "consumer_discretionary",  # 2
    "가정용 기기 제조업": "consumer_discretionary",  # 9
    "가죽, 가방 및 유사제품 제조업": "consumer_discretionary",  # 5
    "개인 및 가정용품 수리업": "consumer_discretionary",  # 1
    "개인 및 가정용품 임대업": "consumer_discretionary",  # 1
    "교육지원 서비스업": "consumer_discretionary",  # 2
    "귀금속 및 장신용품 제조업": "consumer_discretionary",  # 2
    "그외 기타 개인 서비스업": "consumer_discretionary",  # 1
    "그외 기타 제품 제조업": "consumer_discretionary",  # 10
    "기타 교육기관": "consumer_discretionary",  # 2
    "기타 상품 전문 소매업": "consumer_discretionary",  # 3
    "기타 생활용품 소매업": "consumer_discretionary",  # 3
    "기타 섬유제품 제조업": "consumer_discretionary",  # 4
    "무점포 소매업": "consumer_discretionary",  # 5
    "방적 및 가공사 제조업": "consumer_discretionary",  # 2
    "봉제의복 제조업": "consumer_discretionary",  # 29
    "섬유, 의복, 신발 및 가죽제품 소매업": "consumer_discretionary",  # 11
    "섬유제품 염색, 정리 및 마무리 가공업": "consumer_discretionary",  # 1
    "신발 및 신발 부분품 제조업": "consumer_discretionary",  # 2
    "악기 제조업": "consumer_discretionary",  # 1
    # Household audio and video equipment: consumer electronics, which GICS puts
    # in consumer discretionary rather than with the component makers.
    "영상 및 음향기기 제조업": "consumer_discretionary",  # 18
    "운동 및 경기용구 제조업": "consumer_discretionary",  # 2
    "유원지 및 기타 오락관련 서비스업": "consumer_discretionary",  # 3
    "여행사 및 기타 여행보조 서비스업": "consumer_discretionary",  # 6
    "의복 액세서리 제조업": "consumer_discretionary",  # 2
    "일반 교습 학원": "consumer_discretionary",  # 6
    "일반 및 생활 숙박시설 운영업": "consumer_discretionary",  # 3
    "자동차 부품 및 내장품 판매업": "consumer_discretionary",  # 5
    "자동차 신품 부품 제조업": "consumer_discretionary",  # 104
    "자동차 재제조 부품 제조업": "consumer_discretionary",  # 1
    "자동차 차체나 트레일러 제조업": "consumer_discretionary",  # 1
    "자동차 판매업": "consumer_discretionary",  # 3
    "자동차용 엔진 및 자동차 제조업": "consumer_discretionary",  # 4
    "종합 소매업": "consumer_discretionary",  # 15
    "직물직조 및 직물제품 제조업": "consumer_discretionary",  # 6
    "초등 교육기관": "consumer_discretionary",  # 2
    "편조의복 제조업": "consumer_discretionary",  # 1
    # --- health care ----------------------------------------------------------
    "기초 의약물질 제조업": "health_care",  # 46
    "의료용 기기 제조업": "health_care",  # 76
    "의료용품 및 기타 의약 관련제품 제조업": "health_care",  # 34
    "의약품 제조업": "health_care",  # 107
    # R&D of every kind, mapped where `sic.py` sends SIC 8731: it is where
    # biotech files, and on the 2026-09-27 census the recent listings under this
    # label were biotech.
    "자연과학 및 공학 연구개발업": "health_care",  # 80
    # --- technology -----------------------------------------------------------
    "마그네틱 및 광학 매체 제조업": "technology",  # 1
    "반도체 제조업": "technology",  # 74
    # Optical components and photographic equipment: electronic equipment and
    # instruments in GICS terms.
    "사진장비 및 광학기기 제조업": "technology",  # 12
    "소프트웨어 개발 및 공급업": "technology",  # 192
    "전자부품 제조업": "technology",  # 136
    "컴퓨터 및 주변장치 제조업": "technology",  # 11
    "컴퓨터 프로그래밍, 시스템 통합 및 관리업": "technology",  # 40
    # Communications and broadcast equipment. `sic.py` gained a carve-out for
    # SIC 3661-3669 so the same business does not split across the two markets
    # (ADR-0029).
    "통신 및 방송 장비 제조업": "technology",  # 66
    # --- communication services -----------------------------------------------
    "광고업": "communication_services",  # 19
    "기록매체 복제업": "communication_services",  # 1
    "기타 정보 서비스업": "communication_services",  # 19
    "서적, 잡지 및 기타 인쇄물 출판업": "communication_services",  # 10
    "스포츠 서비스업": "communication_services",  # 2
    "영상·오디오물 제공 서비스업": "communication_services",  # 3
    "영화, 비디오물, 방송프로그램 제작 및 배급업": "communication_services",  # 32
    "오디오물 출판 및 원판 녹음업": "communication_services",  # 6
    "인쇄 및 인쇄관련 산업": "communication_services",  # 1
    # The label names 포털 and 인터넷 정보매개, which is interactive media, and
    # this is where the two largest Korean internet names sit. `sic.py` cannot
    # follow: SIC 7370 puts Alphabet and IBM on the same code, so the US side
    # keeps them in technology. Rule 1 beats rule 2 -- see ADR-0029.
    "자료처리, 호스팅, 포털 및 기타 인터넷 정보매개 서비스업": "communication_services",  # 15
    "전기 통신업": "communication_services",  # 14
    "창작 및 예술관련 서비스업": "communication_services",  # 7
    "텔레비전 방송업": "communication_services",  # 9
    # --- financials -----------------------------------------------------------
    # Where the SPACs are: 66 of the 71 names with 스팩 in them on the
    # 2026-09-27 census.
    "금융 지원 서비스업": "financials",  # 90
    # And where most holding companies are.
    "기타 금융업": "financials",  # 106
    "보험 및 연금관련 서비스업": "financials",  # 2
    "보험업": "financials",  # 10
    "신탁업 및 집합투자업": "financials",  # 15
    "은행 및 저축기관": "financials",  # 5
    "재 보험업": "financials",  # 1
    # --- real estate ----------------------------------------------------------
    # Where 21 of the 27 REITs are.
    "부동산 임대 및 공급업": "real_estate",  # 27
}


def bucket_for_label(label: str | None) -> str | None:
    """The concentration bucket a KIND industry label counts against, or None.

    None is the answer for a blank label, a label this table has not seen, and
    anything that is not a string. `core/data/universe.py` already treats a name
    with no bucket as loaded but not orderable, which is where an unknown label
    belongs until somebody has looked at it.
    """
    if not isinstance(label, str):
        return None
    return LABELS.get(label.strip()) or None


def unmapped(labels: Iterable[str]) -> tuple[str, ...]:
    """The labels no line of this table covers, sorted, for a coverage report."""
    return tuple(sorted({label for label in labels if bucket_for_label(label) is None}))


def bucket_counts(by_symbol: Mapping[str, str]) -> dict[str, int]:
    """How many symbols land in each bucket, plus `None` for the unmapped ones.

    A table that quietly sends half a market into one bucket is worse than no
    table, and the only way anybody sees that is if the run prints this.
    """
    counts: dict[str, int] = {}
    for label in by_symbol.values():
        bucket = bucket_for_label(label) or "(unmapped)"
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def coverage(by_symbol: Mapping[str, str]) -> float:
    """The share of symbols this table can place, 0.0 to 1.0."""
    if not by_symbol:
        return 0.0
    placed = sum(1 for label in by_symbol.values() if bucket_for_label(label) is not None)
    return placed / len(by_symbol)


#: Every bucket this table produces is one `sic.py` produces. Asserted at import
#: rather than in a test: a typo in a bucket name would otherwise sit in the file
#: until a limit failed to bind.
_UNKNOWN = sorted(set(LABELS.values()) - set(BUCKETS))
if _UNKNOWN:  # pragma: no cover - a typo in the table, caught at import
    raise ValueError(f"{_UNKNOWN} are not buckets `core/data/sic.py` produces: {list(BUCKETS)}")
