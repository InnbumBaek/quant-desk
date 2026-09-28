<!-- artifact-run: 6eb8c7a706839245.tsmom-002 -->

# ADR-0049 — breadth를 샀고 아무것도 사지 못했다

- 상태: 채택
- 날짜: 2026-09-28
- 관련: ADR-0039·0040(데스크 N), ADR-0041(선언된 유니버스), ADR-0042·0043(프론티어),
  ADR-0046(같은 여섯을 스물여섯에), ADR-0048(핀)

## 실험

ADR-0046이 설계한 한 변수 실험의 결과다. 같은 코드, 같은 격자, 같은 보고 지점,
같은 스냅샷 날짜, 하나의 `git_sha`. 다른 것은 유니버스뿐이다 — `-001` 여섯은
미국 ETF 다섯에서, `-002` 여섯은 프론티어가 고른 스물여섯에서 판정됐다.
표본 길이도 같다: 양쪽 다 `{{artifact:/is_rows}}` + `{{artifact:/oos_rows}}` 행.

ADR-0043이 실측한 것은 스물여섯이 **이력을 하루도 쓰지 않고** 유효 베팅을
`{{artifact:66fc72a0b2671a3a.frontier/baseline.effective_bets}}`에서
`{{artifact:66fc72a0b2671a3a.frontier/frontier[0].effective_bets}}`로,
근본 법칙의 계수로는 `{{artifact:66fc72a0b2671a3a.frontier/frontier[0].law_factor_vs_baseline}}`배
올린다는 것이었다. 통계 문턱도 오르지 않았다 — 요구 DSR은 양쪽이 같다.
**공짜로 breadth를 산 것이다.**

## 결과

| 계열 | 표본 내 샤프 (5) | 표본 내 샤프 (26) | OOS 샤프 (5) | OOS 샤프 (26) |
| --- | --- | --- | --- | --- |
| `tsmom` | 0.038 {{artifact:26a1ef6c1dc7d0d9.tsmom-001/verdicts[1].metrics.is_sharpe}} | -0.010 {{artifact:6eb8c7a706839245.tsmom-002/verdicts[1].metrics.is_sharpe}} | 0.260 {{artifact:26a1ef6c1dc7d0d9.tsmom-001/verdicts[2].metrics.oos_sharpe}} | -0.454 {{artifact:6eb8c7a706839245.tsmom-002/verdicts[2].metrics.oos_sharpe}} |
| `xsmom` | -0.157 {{artifact:26a1ef6c1dc7d0d9.xsmom-001/verdicts[1].metrics.is_sharpe}} | 0.001 {{artifact:6eb8c7a706839245.xsmom-002/verdicts[1].metrics.is_sharpe}} | 0.369 {{artifact:26a1ef6c1dc7d0d9.xsmom-001/verdicts[2].metrics.oos_sharpe}} | 0.283 {{artifact:6eb8c7a706839245.xsmom-002/verdicts[2].metrics.oos_sharpe}} |
| `strev` | -0.037 {{artifact:26a1ef6c1dc7d0d9.strev-001/verdicts[1].metrics.is_sharpe}} | -0.053 {{artifact:6eb8c7a706839245.strev-002/verdicts[1].metrics.is_sharpe}} | -0.166 {{artifact:26a1ef6c1dc7d0d9.strev-001/verdicts[2].metrics.oos_sharpe}} | -0.737 {{artifact:6eb8c7a706839245.strev-002/verdicts[2].metrics.oos_sharpe}} |
| `bab` | -0.126 {{artifact:26a1ef6c1dc7d0d9.bab-001/verdicts[1].metrics.is_sharpe}} | -0.133 {{artifact:6eb8c7a706839245.bab-002/verdicts[1].metrics.is_sharpe}} | -0.444 {{artifact:26a1ef6c1dc7d0d9.bab-001/verdicts[2].metrics.oos_sharpe}} | -0.987 {{artifact:6eb8c7a706839245.bab-002/verdicts[2].metrics.oos_sharpe}} |
| `breakout` | 0.156 {{artifact:26a1ef6c1dc7d0d9.breakout-001/verdicts[1].metrics.is_sharpe}} | 0.049 {{artifact:6eb8c7a706839245.breakout-002/verdicts[1].metrics.is_sharpe}} | 0.316 {{artifact:26a1ef6c1dc7d0d9.breakout-001/verdicts[2].metrics.oos_sharpe}} | 0.072 {{artifact:6eb8c7a706839245.breakout-002/verdicts[2].metrics.oos_sharpe}} |
| `blend` | -0.153 {{artifact:26a1ef6c1dc7d0d9.blend-001/verdicts[1].metrics.is_sharpe}} | -0.130 {{artifact:6eb8c7a706839245.blend-002/verdicts[1].metrics.is_sharpe}} | -0.076 {{artifact:26a1ef6c1dc7d0d9.blend-001/verdicts[2].metrics.oos_sharpe}} | -0.976 {{artifact:6eb8c7a706839245.blend-002/verdicts[2].metrics.oos_sharpe}} |

**표본 내 샤프가 오른 것은 여섯 중 둘, OOS 샤프가 오른 것은 여섯 중 영이다.**
DSR이 오른 것은 하나(`xsmom`). 열두 건 전부 기각이고, 넓힌 쪽은 좁은 쪽보다
더 나쁘다.

## 무엇을 배웠나

근본 법칙은 IR ≈ IC·√breadth다. breadth는 실측으로
`{{artifact:66fc72a0b2671a3a.frontier/frontier[0].law_factor_vs_baseline}}`배
올랐는데 실현된 샤프는 내려갔다. 그러면 남는 해석은 하나다: **이 여섯의 IC가
0 근처이거나 음수다.** 0에 √breadth를 곱해도 0이고, 음수에 곱하면 더 나빠진다.

그래서 이 데스크의 구속 조건은 **breadth가 아니었다.** ADR-0043은 "다음에 살 것은
데이터다"라고 적었고 그중 횡단면 breadth는 26종목으로 해결됐다고 봤다. 이번
측정은 그 해결이 **성과와 무관했다**는 것을 보여준다. 넓힐 곳이 남아 있어도
넓히는 것은 답이 아니다. 답은 **다른 신호**이거나 **다른 데이터**다 —
장중 고저, 펀더멘털, 이벤트처럼 종가 시계열에 들어 있지 않은 정보.

`breakout`은 특히 선명하다. 다섯에서는 PBO가
0.114 {{artifact:26a1ef6c1dc7d0d9.breakout-001/verdicts[3].metrics.pbo}}였고 G3·G5를
통과한 적이 있는 유일한 알파였다. 스물여섯에서 PBO는
0.857 {{artifact:6eb8c7a706839245.breakout-002/verdicts[3].metrics.pbo}}로 올라 G5까지
잃었다. **종목을 늘리자 고른 채널이 우연이었다는 것이 더 분명해졌다.** 넓은
유니버스는 성과를 올리지 않았지만 과최적화를 더 잘 드러냈다. 그것이 이 실험이
실제로 산 것이다.

## 결정

1. **다음 세대를 또 넓히지 않는다.** 스물일곱·스물여덟은
   `registry/universe/*.frontier.json`에 이미 재어 두었고, 지금 측정은 그 방향의
   기대 이득이 0임을 말한다. 유니버스는 스물여섯에서 멈춘다.
2. **다음 선언은 종가 이외의 정보를 쓰는 것으로 한다.** 선언을 하나 더 하면
   데스크 N이 올라 기존 알파의 문턱이 올라간다(ADR-0039). 현재 N은
   60 {{artifact:/desk_trials.total}}이고 요구 연율 샤프는
   0.895 {{artifact:/sharpe_required.at_desk_trials}}로 아직 G2의 고정 하한 1.0보다
   낮다. 예산은 ADR-0040이 잰 237까지 남아 있으므로 **새 데이터로만** 쓴다.
3. **여섯 계열은 기각 상태를 유지한다.** `_implementations.yaml`을 고쳐 코드를
   손보는 것은 같은 선언에 새 시행을 더하는 것이고, 그것은 세어지지 않는 시행이다.
   고치려면 새 id로 다시 선언한다.

## 반증 조건

- 스물여섯에서 어떤 계열이든 표본 내 샤프가 1.0을 넘으면 이 결정은 재검토
  대상이다. 지금은 가장 높은 것이 `breakout-002`의
  0.049 {{artifact:6eb8c7a706839245.breakout-002/verdicts[1].metrics.is_sharpe}}다.
- 두 판정의 표본 길이가 다르면 이 비교는 성립하지 않는다. `is_rows`와 `oos_rows`가
  양쪽에서 같은 것이 그 조건이고, 제출 기록에 남아 있다.
- 두 판정이 서로 다른 `git_sha`로 나왔다면 한 변수 실험이 아니다. ADR-0048의
  핀 규약이 그것을 보장한다.

## 대안과 기각 이유

- **격자를 다시 돌려 더 나은 점을 고른다.** 기각. 사전등록한 격자 안에서 보고
  지점을 바꾸는 것은 새 시행이고, `grid_within_declaration`이 막는다. 막지
  않았더라도 그것이 PBO가 재는 바로 그 행동이다.
- **스물일곱으로 한 칸 더 넓힌다.** 기각. 유효 베팅은 더 오르지만
  (`{{artifact:66fc72a0b2671a3a.frontier/frontier[1].effective_bets}}`) 창이
  135행 짧아지고, 이번 측정은 breadth의 한계 이득이 성과로 나타나지 않음을
  보였다. 이력을 대가로 지불할 이유가 없다.
- **여섯 중 가장 나은 것에 자본을 조금 준다.** 기각. G8은 사람의 승인이고,
  `live_blockers`는 절대 빈 목록을 돌려주지 않는다. 기각된 알파에 소액을
  주는 경로는 만들지 않는다.
