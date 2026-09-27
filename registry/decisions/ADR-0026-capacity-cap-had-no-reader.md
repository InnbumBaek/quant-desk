# ADR-0026 — 캐패시티 상한에 읽는 코드가 없었다

- 상태: 채택
- 날짜: 2026-09-27
- 관련: ADR-0002(게이트 기준), ADR-0015(포드 한도 부재 차단),
  ADR-0016(펀드 층 집행), ADR-0004(백테스트 실행기)
- 코드: `core/risk/capacity.py`, `core/backtest/gates.py`(G6),
  `core/backtest/engine.py`, `core/pipeline.py`
- 규약: CLAUDE.md 7항(캐패시티 상한 80%), 4항(fail-closed)

## 문제

`limits.yaml`에 `capacity:` 블록이 처음부터 있었다 — `respect_hard_cap: true`,
`capacity_utilisation_max: 0.80`. **읽는 코드가 하나도 없었다.**

CLAUDE.md 7항은 "추정 캐패시티의 80%가 상한이고, 성과가 좋다는 이유로 올리지
않는다"고 규정한다. 집행자가 없었다. G6은 `gates:` 블록의 ADV 참여율과 북 상관을
보는데, 그건 다른 질문이다 — **이 전략이 거래될 수 있는가**(G6)와 **자본을 얼마까지
들 수 있는가**(7항)는 다르다.

같은 종류를 이 데스크에서 세 번째로 찾았다: 포드 한도 6개가 측정값 없이 통과하던
것(ADR-0015), `fund:` 블록 전체가 안 읽히던 것(ADR-0016), 그리고 이것.
**한도표에 적혀 있다는 사실은 한도가 있다는 뜻이 아니다.**

## 더 나쁜 것 — 0이 무한 캐패시티로 환산된다

캐패시티 추정은 참여율 스케일링이다:
`capacity = tested_capital × participation_cap / observed_participation`.

`core/backtest/engine.py`의 `adv_participation()`은 **거래대금 패널이 없으면 0.0을
돌려준다.** 0으로 나누면 캐패시티는 무한이다 — 측정을 안 해서 도달한, 가능한 가장
관대한 숫자다. 실행기는 이걸 알고 있었고 RunReport에 노트를 남겼다("ADV
participation is 0.0 by absence, not by measurement"). **노트는 게이트가 아니다.**
G6은 그 제출을 통과시켰다.

## 결정 1 — 참여율이 측정값인지를 제출이 들고 다닌다

`Submission.adv_measured: bool = False`. 기본값이 False이고, 실행기가
`panel.dollar_volume is not None`로 채운다. G6은 그것이 True가 아니면 **임계값 비교를
하지 않고 기각한다.** 임계값이 아니라 부재이므로 `gates:` 블록에서 오지 않는다 —
ADR-0015의 `*_UNMEASURED`와 같은 성격이다.

## 결정 2 — 0은 측정됐다고 선언해도 나눌 수 없다

`estimate_capacity()`는 참여율이 0 이하·NaN·문자열·bool이면 `None`을 돌려준다.
자본도 같다. **추정 불가는 추정치가 아니고**, 부재를 숫자로 바꾸는 순간 이 모듈이
막으려는 것이 통과한다.

## 결정 3 — 상한을 끄는 것 자체가 위반이다

`respect_hard_cap: false`는 "추정치를 넘겨 배분해도 된다"로 읽힐 수 있다. 그럴 수
없다 — CLAUDE.md 7항은 절대 조항이고 `limits.yaml` 변경은 소유자 승인 사항이다.
이 플래그는 **끄는 것을 보이게 하려고** 있는 것이지 구멍을 여는 스위치가 아니다.
그래서 `false`도 부재도 `CAPACITY_CAP_DISABLED` 차단이다.

`capacity_utilisation_max`가 1.0을 넘으면 **클램프하지 않고 거부한다.** 조용히
1.0으로 깎으면 승인된 파일의 오타가 숨는다.

## 결정 4 — 파이프라인에 붙인다. 안 붙이면 또 하나의 안 읽히는 모듈이다

`run_day(capacity_snapshot=...)`이고, **스냅샷이 없으면 차단**이다. 펀드·파이낸싱
스냅샷과 같은 처리다(ADR-0016): 아무도 캐패시티를 측정하지 않는 날이 바로 어떤 포드가
이미 상한을 넘어 있는 날이다. 기존 `run_day` 호출 전부가 스냅샷을 넘기도록 고쳐야
했고, 그것이 이 결정의 비용이다.

배분기(`core/portfolio/allocate.py`, ADR-0010, PR #1)가 병합되면 `max_allocatable()`이
그쪽의 천장이 된다. 지금은 일간 책에서 집행된다.

## 결정 5 — 캐너리는 자기 결함으로만 기각돼야 한다

G6이 엄격해지자 캐너리 4종 전부가 미측정 참여율로도 기각됐다. 기각 자체는
`failed & EXPECTED_GATE`가 교집합으로 보기 때문에 테스트가 **초록으로 남았다** —
`canary_lookahead`가 "G0만이 잡는다"는 성질을 조용히 잃으면서도 통과했다.

그래서 두 가지를 했다: 캐너리가 측정된 참여율 0.005를 선언하고,
`canary_lookahead`의 기각 게이트 집합을 **동일성으로** 못박는 테스트를 추가했다
(`== {"G0_data"}`). 누출 스캔이 유일한 방어선이라는 주장은 그 성질에 달려 있으므로,
다음에 누가 게이트를 건드려 그 성질을 깨면 테스트가 말해야 한다.

## 이것은 기준 완화가 아니다

G6은 **더 엄격해졌다**(통과했던 제출이 기각된다). 캐너리 4종은 여전히 각자의 사유로
기각된다. `limits.yaml`은 건드리지 않았다.

## 검증

`tests/limits/test_capacity.py` 31개, `tests/test_pipeline_failclosed.py`에 4개,
`tests/canaries/test_canaries.py`에 1개. 고정하는 것: 캐패시티가 참여율 상한에
도달하는 자본이다, 참여율 0·NaN·문자열·bool·음수는 추정치를 만들지 않는다, 0.80
천장이 정확히 계산되고 경계값은 허용된다, `respect_hard_cap`을 끄면 차단이다,
`capacity_utilisation_max` > 1.0은 거부된다, `capacity:` 블록 자체가 없으면 "제약
없음"이 아니라 차단이다, 빈 스냅샷은 깨끗한 책이 아니다, 포드마다 판정되고 위반에
포드 이름이 담긴다, 파이프라인이 스냅샷 부재·상한 초과·미측정 참여율에 주문을 막는다.

전체 974 passed.

## 열려 있는 것

- **이것은 임팩트 모델이 아니다.** 참여율 스케일링은 선형 프록시이고, 실제 캐패시티는
  임팩트가 비선형이므로 이보다 작을 것이다. 프록시가 캐패시티를 **과대추정하는**
  방향이라는 뜻이고, 그래서 80% 상한이 여유가 아니라 최소한이다.
  `core/execution/impact.py`가 생기면 추정 근거가 교체되고 `basis` 문자열이 바뀐다.
- **한도표에 아직 읽는 코드가 없는 키가 남았다**: `horizon.risk_budget_share_max`,
  `cost_attribution.charge_pods`, `cost_attribution.min_net_of_cost_ir`. 셋 다 포드별
  리스크 기여도와 비용 귀속을 아는 배분기가 필요하므로 **PR #1 병합에 달려 있다.**
  ADR-0015가 ADR-0016에 넘긴 것과 같은 방식으로 여기 적어둔다.
- **`scripts/check_limits_change.py`는 `limits.yaml`만 감시한다.** 이번 변경처럼
  임계값이 아닌 새 실패 조건을 게이트에 넣는 것은 가드가 못 잡는다. `gates.py`를
  감시 목록에 넣으면 잡히지만 기계적 리팩터에도 ADR을 요구하게 된다. **프로세스
  변경이므로 소유자 판단으로 남긴다** — 이 ADR은 그 제안만 기록한다.
- 캐패시티 스냅샷을 누가 만드는가는 아직 수동이다. 백테스트 RunReport에서
  자동 생성하는 것이 다음 단계다.
