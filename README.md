# quant-desk

월가 대형 퀀트 헤지펀드의 조직을 에이전트로 옮긴 운용 데스크. 23개 에이전트가
5개 본부로 나뉘어 리서치, 포트폴리오 구성, 실행, 리스크, 운영을 맡는다.
설계안 전문은 프로젝트 파일 `quant-desk-plan/PLAN.md`에 있다.

## 설계 원칙

1. **LLM은 핫패스에 없다.** 주문 생성·한도 체크·손절은 결정론적 코드가 한다
   (`core/risk/limits.py`, `core/execution/orders.py`, `core/pipeline.py`).
   에이전트는 전략을 쓰고, 검증하고, 해석하고, 예외를 올린다.
2. **제안자와 검증자가 다르다.** 포드는 자기 알파를 승인할 수 없고,
   `adversarial-validator`는 포드의 리포트를 보지 않은 채 통계량을 다시 계산하며
   거부권만 갖는다.
3. **모든 결과는 재현 가능하다.** (git SHA, 데이터 스냅샷 ID, 시드) 3중 핀.
4. **게이트를 통과하지 않은 알파는 자본을 받지 못한다.** G0–G8, 기준값은
   `core/risk/limits.yaml`이 단일 출처다.
5. **실자본 전환과 한도 상향은 사람이 승인한다.**
6. **통제를 테스트한다.** 가짜 알파 4종(`tests/canaries/`)이 CI에서 상시 돌고,
   하나라도 게이트를 통과하면 빌드가 깨진다.

## 구조

```
.claude/agents/      23개 에이전트 정의
.claude/skills/      조직의 사규 — alpha-gate, backtest-protocol, risk-limits, loss-attribution
.claude/hooks/       pretrade_gate.py (주문 전 강제 검사), numbers_from_artifacts.py
core/                데이터·피처·백테스트·포트폴리오·실행·리스크·운영
registry/alphas/     알파 등록부(게이트 이력, 시행횟수 N, DD 분포는 봉인)
registry/decisions/  ADR
tests/               한도·복구·캐너리 시나리오
runbooks/            fail-closed, 크래시 복구
```

## 개발

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

## 상태

P0 (골격) 완료. 로드맵은 P1 데이터·백테스트 코어 → P2 알파 팩토리 →
P3 포트폴리오·리스크 → P4 페이퍼 풀루프 → P5 거버넌스 순이다. 실자본은 G7을
통과하고 사람이 승인한 뒤의 선택 단계다.

## 주의

- **이 리포지토리는 public이다.** 자격증명, 라이선스 데이터, 포지션·PnL 파일은
  절대 커밋하지 않는다. 키는 환경변수로만 다룬다(`.env.example`).
- 자기자본 운용 전제다. 타인 자금을 운용·자문하면 인가·등록 대상이다.
