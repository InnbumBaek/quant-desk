# 에이전트 역할 ↔ 코드 연결 현황 (2026-09-22)

> **v3 갱신 (2026-09-22).** 이 문서는 23개 에이전트 시점에 작성되었다. v3에서 조직이
> **25개**가 되었고(`financing-treasury`, `internal-audit` 신설), 코드가 두 개 늘었다.
> 따라서 아래 A군은 실제로 7개다:
> - `financing-treasury` → `core/risk/financing.py` (증거금·브로커 집중도·조달비용·현금 버퍼 판정, 값이 비면 위반)
> - `portfolio-construction`의 센터북 몫 → `core/portfolio/center_book.py` (포드 간 넷팅 + 축소 방향 오버레이, 파이프라인에 연결됨)
>
> 그래서 C군의 "`core/portfolio/`는 `__init__.py`만 있다"는 더 이상 사실이 아니다.
> `internal-audit`은 D군(판단·검증이 본업)이지만, 검증 대상은 전부 코드 산출물이다.
> B군(백테스트 엔진·통계 계산기)은 그대로 가장 큰 빈틈이고, 권고한 수직 슬라이스가
> 여전히 다음 단계다.

설계(`quant-desk-plan/PLAN.md` §1)와 에이전트 정의(`.claude/agents/`, 23개)는 완성돼
있습니다. 이 문서는 그 다음 질문에 답합니다 — **각 역할 뒤에 실제로 돌아가는 코드가
있는가.** P0 트리를 파일 단위로 확인한 결과이고, 추정치는 없습니다.

요약: **23개 역할 중 5개만 결정론적 코드가 받치고 있고, 나머지 18개는 프롬프트와
문서만 있습니다.** 이것은 결함이 아니라 P0의 정의입니다(ADR-0001). 다만 "에이전트가
있다"와 "역할이 작동한다"는 다른 상태이므로, 어디가 어느 쪽인지 여기서 고정합니다.

---

## A. 코드가 이미 판정하는 역할 (5) — 에이전트는 읽고 보고만 한다

| 에이전트 | 입력 | 산출물 | 실제 코드 | 상태 |
|---|---|---|---|---|
| `data-quality` | 일간 데이터 스냅샷 | `HealthReport` (게이트 0) | `core/data/quality.py` (42줄) | 판정 로직 동작, 체크 항목은 호출자가 주입 |
| `risk-officer` | 목표 포지션, 한도표 | 위반 목록·감축 집행 | `core/risk/limits.py` (86줄) + `limits.yaml` | 그로스/넷/집중/DD 이원조건 판정 동작 |
| `execution-trader` | 주문 파일, 브로커 상태 | 체결·주문 상태머신 | `core/execution/orders.py` (74줄) | 멱등 ID·상태머신·`blocking_orders` 동작. 브로커 연동은 없음 |
| `compliance-surveillance` | 전 산출물 | 감사로그 무결성 | `core/audit.py` (71줄, 해시체인) | 체인 검증 동작. 금지종목·이상패턴은 미구현 |
| (전 에이전트 공통) | 리포트 파일 | 수치 인용 강제 | `.claude/hooks/numbers_from_artifacts.py` | 동작. 환각 수치 차단 |

여기에 더해 `core/pipeline.py`(66줄)와 `.claude/hooks/pretrade_gate.py`가 세 가지
차단 조건(데이터 헬스 실패 / 미해결 주문 / 한도 위반)을 fail-closed로 강제합니다.
**이 경로에 LLM은 없습니다.** 파이프라인의 각 *스테이지*는 아직 no-op이고, 실제인
것은 제어 흐름과 감사 기록입니다.

---

## B. 기준은 있으나 계산 코드가 없는 역할 (6) — 가장 큰 빈틈

알파 게이트 G0~G8의 **임계값**은 `core/risk/limits.yaml`의 `gates:` 블록에 숫자로
박혀 있고, 절차는 `.claude/skills/alpha-gate/SKILL.md`에 있습니다. 그러나 그 숫자를
**계산하는 코드가 없습니다** — `core/backtest/`는 빈 디렉터리입니다.

| 에이전트 | 무엇이 없나 | 필요한 것 |
|---|---|---|
| `backtest-engineer` | 백테스트 엔진, purged/embargo CV, 비용모델, 재현성 3중 핀 | `core/backtest/engine.py`, `cv.py`, `costs.py` |
| `adversarial-validator` | DSR·PBO·팩터잔차 t 계산기 | `core/backtest/stats.py` (거부권 판정은 코드 반환값이어야 함) |
| `alpha-pod-*` (4개) | 실행할 엔진이 없어 알파를 제출할 수 없음 | B의 위 두 항목에 종속 |
| `feature-factory` | 피처 카탈로그, \|ρ\|>0.9 중복 거부 | `core/features/catalog.py` |

`tests/canaries/`의 가짜 알파 4종은 현재 strict-xfail입니다. 즉 **게이트가 가짜를
걸러내는지는 아직 증명되지 않았고**, B가 채워지는 순간 이 테스트가 진짜 통제가 됩니다.

---

## C. 모듈이 비어 있는 역할 (7)

`core/portfolio/`, `core/ops/`는 `__init__.py`만 있습니다.

| 에이전트 | 대응 모듈 | 우선순위 근거 |
|---|---|---|
| `portfolio-construction` | `core/portfolio/optimize.py` (Ledoit-Wolf + 제약 최적화) | 알파가 1개라도 통과해야 의미가 생김 |
| `capital-allocator` | `core/portfolio/allocate.py` (리스크패리티 × 하프켈리 × 캐패시티) | 포드 2개 이상부터 |
| `microstructure-research` / `tca-analyst` | `core/execution/impact.py`, `tca.py` | G6(캐패시티) 판정에 필요 |
| `stress-testing` / `model-risk` | `core/risk/stress.py`, `tracking.py` | 페이퍼(G7) 시작 시점부터 |
| `pnl-recon` / `platform-sre` / `ir-reporting` | `core/ops/*` | 페이퍼 운영 시작 시점부터 |

---

## D. 코드 연결이 필요 없는 역할 (3)

`cio`, `chief-of-staff`, 그리고 `data-acquisition`은 판단·기록·온보딩이 본업이라
결정론 코드로 대체할 대상이 아닙니다. 다만 `cio`의 G7·G8 판정은 **B가 만들어낸
숫자를 읽는 것**이어야 하고, 스스로 근거를 만들면 안 됩니다(`.claude/agents/cio.md`의
"하지 않는 일"에 이미 명시).

---

## 권고 — 23개를 다 붙이지 말고 수직 슬라이스 하나를 관통시킨다

현재 구조의 위험은 역할이 모자란 것이 아니라 **역할이 너무 많고 그 중 대부분이
실행 불가능**하다는 점입니다. 다음 단계(P1)는 폭이 아니라 깊이여야 합니다.

**최소 관통 경로 6개 에이전트**:
`data-quality` → `feature-factory` → `alpha-pod-equity-statarb` →
`backtest-engineer` → `adversarial-validator` → `risk-officer`

이 여섯이 알파 하나를 G0에서 G5까지 실제로 통과시키거나 **떨어뜨리면**, 나머지 17개는
같은 패턴의 복제입니다. 반대로 이 경로가 막히면 17개를 더 만들어도 한 걸음도 나아가지
않습니다. 성공 판정 기준은 "알파가 통과했다"가 아니라 **캐너리 4종이 strict-xfail을
벗고 실제로 기각되는 것**입니다.

---

## 설계안에서 정정한 것

이 스레드를 연 카드는 "Stock_Agent의 현재 코드와 어떻게 연결되는지"를 요구했습니다.
2026-09-21에 사용자가 이 프로젝트는 Stock_monitoring / Stock_Agent와 **무관한 신규
구축**이라고 명시했으므로, 그 항목은 설계에서 제외했습니다. 위 표의 "코드"는 모두
`quant-desk` 트리 내부를 가리킵니다.
