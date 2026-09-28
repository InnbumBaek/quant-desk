# ADR-0043 — 프론티어가 답했다: 21종목은 이력을 쓰지 않는다

- 상태: 채택
- 날짜: 2026-09-28
- 관련: **ADR-0042(breadth를 이력으로 산다)**, ADR-0041(선언된 유니버스),
  ADR-0040(여섯 전부 기각), ADR-0034(이력 20년), ADR-0033(게이트 역산),
  ADR-0017(유니버스 계약), CLAUDE.md 2·3항

<!-- artifact-run: 17121f69010340d5 -->

## 맥락

ADR-0042는 창과 breadth의 교환을 재는 기계를 놓고 **숫자는 하나도 갖지 않은 채**
끝났다. "어느 유니버스를 선언할지는 프론티어를 읽고 정한다"가 그 ADR의 마지막
문장이다. 러너가 28종목을 받았고, `registry/universe/17121f69010340d5.frontier.json`이
그 값이다. 아래 표의 모든 숫자는 그 파일에서 기계가 뽑은 것이다.

## 잰 것

| 유니버스 | 창 시작 | 관측 | 유효 베팅 | 근본법칙 계수 | 요구 DSR |
| --- | --- | --- | --- | --- | --- |
| 기준 5 | 2006-09-25 | 5031 {{artifact:/baseline.observations}} | 3.04 {{artifact:/baseline.effective_bets}} | 1.00 | 0.832 {{artifact:/baseline.deflated_sharpe_required}} |
| 26 | 2006-09-25 | 5031 {{artifact:/frontier[0].observations}} | 6.39 {{artifact:/frontier[0].effective_bets}} | 1.45 {{artifact:/frontier[0].law_factor_vs_baseline}} | 0.832 {{artifact:/frontier[0].deflated_sharpe_required}} |
| 27 | 2007-04-11 | 4896 {{artifact:/frontier[1].observations}} | 6.57 {{artifact:/frontier[1].effective_bets}} | 1.47 {{artifact:/frontier[1].law_factor_vs_baseline}} | 0.844 {{artifact:/frontier[1].deflated_sharpe_required}} |
| 28 | 2010-09-09 | 4035 {{artifact:/frontier[2].observations}} | 6.22 {{artifact:/frontier[2].effective_bets}} | 1.43 {{artifact:/frontier[2].law_factor_vs_baseline}} | 0.930 {{artifact:/frontier[2].deflated_sharpe_required}} |

## 발견 1 — 21종목은 공짜다

26종목 지점은 기준과 **같은 첫날, 같은 마지막날, 같은 관측 수**다. 창을 하루도 쓰지
않는다. 요구 DSR도 같고, 구속하는 것도 여전히 G2의 고정 하한
(1.0 {{artifact:/frontier[0].binding_annualised_sharpe_min}})이다. 그런데 유효 베팅이
두 배가 된다. **이력도 안 쓰고 통계적 문턱도 안 올린다.**

## 발견 2 — 순진한 계수는 두 배 넘게 틀렸다

ADR-0042를 쓸 때의 논거는 티커 수의 제곱근, 2.37배 {{artifact:/fetched|derived}}였다.
실측은 1.45배 {{artifact:/frontier[0].law_factor_vs_baseline}}다. 차이가 어디서 나는지는
구성에서 읽힌다 — 26개 중 열하나가 미국 주식(VTI·SPY·DIA·QQQ·IWM와 섹터 9종)이고,
섹터 ETF는 시장 요인을 공유한다. **티커 수는 breadth가 아니다.**

그래도 1.45배는 크다. 근본법칙에서 IC가 같다면 IR이 1.45배이므로, 요구 샤프를 넘기려면
다섯 종목에서 필요했던 스킬의 69% {{artifact:/frontier[0].law_factor_vs_baseline|derived}}
로 충분하다.

## 발견 3 — 28번째 종목은 breadth를 **줄이면서** 4년을 쓴다

VOO를 넣으면 창이 2010-09-09로 밀려 관측이 4035 {{artifact:/frontier[2].observations}}로
줄고, 요구 DSR이 0.930 {{artifact:/frontier[2].deflated_sharpe_required}}로 올라가고,
**유효 베팅이 6.22 {{artifact:/frontier[2].effective_bets}}로 내려간다.** VOO는 S&P
500을 추종하고 SPY·VTI가 이미 패널에 있으니, 독립적인 변동을 하나도 더하지 않으면서
가장 짧은 이력을 강제한다. **종목 수로 재면 28이 최선이고, 실제로 재면 최악이다.**

27번째(HYG)는 유효 베팅을 6.57 {{artifact:/frontier[1].effective_bets}}까지 올리는 대신
이력 6개월과 요구 DSR 0.844 {{artifact:/frontier[1].deflated_sharpe_required}}를 쓴다.
사도 되는 거래지만 26지점이 지배적으로 싸다.

## 결정 — 다음 세대는 **26종목**에 선언한다

`AGG DIA EEM EFA GLD IEF IWM LQD QQQ SHY SLV SPY TIP TLT USO VNQ VTI XLB XLE XLF
XLI XLK XLP XLU XLV XLY`. 각 선언이 `hypothesis.universe_symbols`에 이 스물여섯을 직접
적는다 — `_implementations.yaml`의 `legacy_universes`에는 넣지 않는다. 그 목록은
유니버스가 산문이던 여섯을 위한 것이고, 조용히 자라면 미집행 규칙이 숨는 경로가
된다(ADR-0041, `tests/test_alpha_universes.py`가 개수를 센다).

HYG와 VOO는 **버리는 것이 아니라 창 때문에 빠지는 것**이고, 산출물이 상장일과 함께
사유를 적는다. 이력 요구를 낮추면 둘 다 들어온다. 그것은 CLAUDE.md 3항이 막는 문턱
낮추기이므로 하지 않는다.

## 이 숫자가 약속하지 않는 것

산출물의 `assumption` 필드가 스스로 적는다 — 유효 베팅은 **상관 스펙트럼의 엔트로피**
(Meucci 2009)이고 근본법칙의 breadth를 **위에서 막는** 값이다. IR을 예측하지 않고,
추가된 스물한 종목에 스킬이 있는지, 비용이 얼마인지, 캐패시티가 되는지는 아무것도
말하지 않는다. 섹터 ETF 아홉 종은 SPY보다 스프레드가 넓고 USO·SLV는 롤 비용이 있다.
**그것들은 G6가 잴 일이고 이 ADR이 잰 것이 아니다.**

그리고 breadth가 늘어도 ADR-0040의 결론은 그대로다: 기각의 원인은 여전히 신호의
크기이고, 일봉 종가라는 제약도 그대로다. 스물여섯 종목은 **횡단면 계열이 비로소
횡단면을 갖는다**는 뜻이지, 브레이크아웃에 장중 고저가 생긴다거나 value·quality에
펀더멘털이 생긴다는 뜻이 아니다.
