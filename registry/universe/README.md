# registry/universe

주간 러너가 쓰는 상장목록이다. 여기 있는 파일은 데이터가 아니라 **기록**이라서
`data/`가 아니라 `registry/`에 있고, git에 커밋된다.

- `us.csv` — 미국 거래소 상장 전 종목. SEC `company_tickers_exchange.json`(멤버십)과
  DERA 재무제표 데이터셋 `sub.txt`(SIC → 섹터 버킷)를 CIK로 조인한 결과.
- `us.source.json` — `as_of`, `point_in_time`, 읽은 분기, 커버리지(거래소 밖 제외 수,
  최근 신고 없는 종목 수, 분류율). 이 사이드카가 없으면 `load_universe`가 파일을
  읽지 않는다(ADR-0017).

**이 파일의 커밋 이력이 상장폐지 이력이다.** SEC는 지금 상장된 회사만 발표하므로,
지난 커밋에 있고 이번 커밋에 없는 종목은 떠난 것이다. 그래서 매 실행을 커밋하고,
실행 결과가 같아도 이력을 끊지 않는다. 근거는 ADR-0018.

생성: `uv run python -m scripts.fetch_listings` (러너에서. 연구 컨테이너는 sec.gov에
닿지 않는다.)
