# registry/universe

주간 러너가 쓰는 상장목록이다. 여기 있는 파일은 데이터가 아니라 **기록**이라서
`data/`가 아니라 `registry/`에 있고, git에 커밋된다.

- `us.csv` — 미국 거래소 상장 전 종목. SEC `company_tickers_exchange.json`(멤버십)과
  DERA 재무제표 데이터셋 `sub.txt`(SIC → 섹터 버킷)를 CIK로 조인한 결과.
- `us.source.json` — `as_of`, `point_in_time`, 읽은 분기, 커버리지(거래소 밖 제외 수,
  최근 신고 없는 종목 수, 분류율). 이 사이드카가 없으면 `load_universe`가 파일을
  읽지 않는다(ADR-0017).
- `kr_industry.csv` — 한국 상장 전 종목. KRX KIND 상장법인목록 한 요청에서 온다
  (키 없음). `market`·`sector`가 붙어 있어 **`load_universe`가 그대로 읽는다**:
  멤버십과 분류가 둘 다 여기서 나오고, 기다리는 KRX 키는 시세용이다. 버킷은
  `core/data/ksic.py`의 라벨 표가 붙이고 벤더의 원래 업종 라벨이 옆 컬럼에 남아
  있어서, 매핑을 산출물에서 바로 검토할 수 있다(ADR-0028, ADR-0029).
  이름은 `kr_industry`로 남긴다 — 내용은 유니버스지만 이름을 바꾸면 아래의 폐지
  이력이 끊긴다.
- `kr_industry.source.json` — 위와 같은 세 키에 더해 전수 라벨과 개수, 시장구분별
  개수, 버킷 분포와 커버리지, 표가 모르는 라벨, 그리고 버려진 행마다의 사유.
  **`kr.csv`는 여기 없다** — `scripts/fetch_krx.py`의 것이고 KRX 키를 기다린다.

**이 파일의 커밋 이력이 상장폐지 이력이다.** SEC는 지금 상장된 회사만 발표하므로,
지난 커밋에 있고 이번 커밋에 없는 종목은 떠난 것이다. 그래서 매 실행을 커밋하고,
실행 결과가 같아도 이력을 끊지 않는다. 근거는 ADR-0018.

생성: `uv run python -m scripts.fetch_listings` (미국, 주간 `listings` 워크플로),
`uv run python -m scripts.fetch_kind` (한국, 주간 `korea-industry` 워크플로).
둘 다 러너에서 돈다 — 연구 컨테이너는 sec.gov도 kind.krx.co.kr도 닿지 않는다.
비용이 다른 두 페치라서 트리거를 나눠 뒀다(ADR-0028).
