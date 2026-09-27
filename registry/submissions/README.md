# registry/submissions

한 알파를 게이트 배터리에 통과시킨 기록. 파일 하나가 실행 하나다:
`<run_id>.<alpha_id>.json`.

`run_id`는 코드·데이터·시드 세 가지를 묶은 해시라(`core/repro.py`) 같은 이름이
두 번 나오면 같은 실행이고, 다르면 셋 중 하나가 달라진 것이다.

## 왜 판정이 `registry/alphas/<id>.yaml`에 들어가지 않는가

사전등록 파일은 **git에 커밋되고 그 뒤로 수정되지 않아야** G1을 통과한다
(`core/backtest/prereg.py`). 결과를 그 파일에 써 넣으면 파일이 수정되고, 다음
실행에서 G1이 "결과를 보고 쓴 선언일 수 있다"는 이유로 기각한다. 그래서 선언과
판정은 다른 파일에 산다. `_template.yaml`의 `gates:` 칸이 비어 있는 것은 누락이
아니라 이 제약의 결과다.

## 이 파일을 읽는 법

- `approved`는 **G0~G6**, 즉 연구가 성립하는지에 대한 답이다. 실전 자본과는
  다른 질문이고, 그 답은 `live_blockers`에 있다. 이 목록은 절대 비지 않는다 —
  G8은 사람의 승인이고 코드가 주지 않는다(ADR-0032).
- `declaration.committed_and_unmodified`가 `false`면 나머지 숫자는 증거가 아니다.
- `declaration.declared_trials`가 G4의 편향 샤프를 디플레이트하는 N이다. 실행이
  선언보다 **적게** 검색한 것은 통과한다(더 큰 N으로 깎이므로 전략에 불리하다).
- `desk_trials`는 **데스크 전체가 선언한 시행 수**다. 자본은 통과한 알파 하나에
  가므로 선택은 모든 알파의 모든 설정에 대해 일어난다(ADR-0039).
  `sharpe_required`가 자기 N과 데스크 N에서의 요구 샤프를 나란히 남긴다 — 그 차이가
  넓게 찾는 값이다. `desk_trials.measured`가 false면 총계는 쓰이지 않는다.
- `grid_crowding`은 게이트가 아니라 측정값이다. 격자 다섯 점이 서로 거의 같은
  신호면 N=5로 깎이는 것이 실제보다 후한 디플레이션이라는 뜻이다.
- `catalogue`는 CLAUDE.md 6항의 집행 기록이다. 제출 신호가 다른 패밀리와 같은
  카탈로그에서 중복 검사를 받는다.
- 숫자를 인용할 때는 `{{artifact:<run_id>/<metric>}}` 형식을 쓴다(CLAUDE.md 2항).
