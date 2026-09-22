# 에이전트 역할 ↔ 코드 연결 현황 (2026-09-22)

설계(`docs/PLAN.md` §1)와 에이전트 정의(`.claude/agents/`, **26개**)는 완성돼
있습니다. 이 문서는 그 다음 질문에 답합니다 — **각 역할 뒤에 실제로 돌아가는 코드가
있는가.** P0 트리를 파일 단위로 확인한 결과이고, 추정치는 없습니다.

요약: P0 시점에는 **25개 역할 중 5개만** 결정론적 코드가 받치고 있었습니다.
2026-09-22 P1으로 게이트 엔진·백테스트 실행기·피처 카탈로그가 들어가면서 **10개**가 됐습니다 (`feature-factory` 추가; 실행기는 이미 세던 `backtest-engineer`를 더 단단하게 만든 것이라 새 역할은 아닙니다).
"에이전트가 있다"와 "역할이 작동한다"는 다른 상태이므로, 어디가 어느 쪽인지 여기서
고정합니다.

> **갱신 (2026-09-22)** — 아래 B절의 가장 큰 빈틈이 메워졌습니다. `backtest-engineer`와
> `adversarial-validator`는 이제 실제 코드로 판정합니다. 캐너리 4종은 strict-xfail을
> 벗었고, 게이트가 가짜 알파를 실제로 기각하는 것이 CI에서 증명됩니다
> (전체 스위트 258 passed). 상세는 `registry/decisions/ADR-0002-gate-engine.md`,
> `ADR-0004-backtest-runner.md`, `ADR-0005-feature-catalogue.md`,
> `ADR-0006-data-snapshots-and-the-repro-pin.md`.
>
> **v3 조직 반영 (2026-09-22).** 조직이 23 → **25개**가 되면서(`financing-treasury`,
> `internal-audit` 신설) 코드가 받치는 역할도 두 개 늘었습니다. 아래 A군은 실제로 7개입니다:
>
> - `financing-treasury` → `core/risk/financing.py` — 증거금 사용률·프라임브로커 집중도·
>   조달비용·익일 현금 버퍼를 판정하고, 값이 비어 있으면 위반으로 셉니다.
> - `portfolio-construction`의 센터북 몫 → `core/portfolio/center_book.py` — 포드 간 넷팅과
>   축소 방향 오버레이. `core/pipeline.py`에서 한도 검사 **앞에** 호출됩니다.
>
> 따라서 아래 C군의 "`core/portfolio/`는 `__init__.py`만 있다"는 더 이상 사실이 아닙니다.
> `internal-audit`은 판단이 본업이므로 D군이지만, 검증 대상은 전부 코드 산출물입니다.

---

## A. 코드가 이미 판정하는 역할 (5) — 에이전트는 읽고 보고만 한다

| 에이전트 | 입력 | 산출물 | 실제 코드 | 상태 |
|---|---|---|---|---|
| `data-quality` | 일간 데이터 스냅샷 | `HealthReport` (게이트 0) | `core/data/quality.py` | 판정 로직 동작, 체크 항목은 호출자가 주입 |
| `data-quality` (적재) | 심볼당 일간 CSV | 검증된 `PricePanel` + 스냅샷 매니페스트 | `core/data/sources.py` | 동작. 날짜 교집합·구멍 거부·바이트 지문 (ADR-0006) |
| (전 산출물 공통) | git SHA·스냅샷 ID·시드 | `ReproPin`·`run_id` | `core/repro.py` | 동작. 더티 트리 핀 거부, 스크래치 핀은 게이트 입력 불가 |
| `literature-review` | arXiv q-fin 주간 피드 | 선별 목록 + 논문별 리뷰 | `scripts/fetch_papers.py` | 수집·중복제거·가중 선별 동작. 판정은 에이전트 몫 (ADR-0011) |
| (전 포드 공통) | 가격 패널 + 파라미터 | 가중치 행렬 | `core/strategies/` | 전략 6종 등록, 파라미터 수정 가능, 전부 G0 스캔 통과 |
| `ir-reporting` | 순수익 시계열·회전율·벤치마크 | CAGR·변동성·Sharpe·Sortino·MDD·Calmar·회전율·베타·알파 | `core/report/metrics.py` | 동작. 측정 불가는 `None`과 이유로 기록, 0.0으로 쓰지 않음 (ADR-0008) |
| `risk-officer` | 목표 포지션, 한도표 | 위반 목록·감축 집행 | `core/risk/limits.py` (86줄) + `limits.yaml` | 그로스/넷/집중/DD 이원조건 판정 동작 |
| `execution-trader` | 주문 파일, 브로커 상태 | 체결·주문 상태머신 | `core/execution/orders.py` (74줄) | 멱등 ID·상태머신·`blocking_orders` 동작. 브로커 연동은 없음 |
| `compliance-surveillance` | 전 산출물 | 감사로그 무결성 | `core/audit.py` (71줄, 해시체인) | 체인 검증 동작. 금지종목·이상패턴은 미구현 |
| (전 에이전트 공통) | 리포트 파일 | 수치 인용 강제 | `.claude/hooks/numbers_from_artifacts.py` | 동작. 환각 수치 차단 |

여기에 더해 `core/pipeline.py`(66줄)와 `.claude/hooks/pretrade_gate.py`가 세 가지
차단 조건(데이터 헬스 실패 / 미해결 주문 / 한도 위반)을 fail-closed로 강제합니다.
**이 경로에 LLM은 없습니다.** 파이프라인의 각 *스테이지*는 아직 no-op이고, 실제인
것은 제어 흐름과 감사 기록입니다.

---

## B. 게이트 엔진 — 2026-09-22 구현 완료

알파 게이트 G0~G8의 **임계값**은 `core/risk/limits.yaml`의 `gates:` 블록에 있고,
이제 그 숫자를 **계산하는 코드**가 있습니다.

| 에이전트 | 입력 | 산출물 | 실제 코드 | 상태 |
|---|---|---|---|---|
| `backtest-engineer` | 수익률 시계열, 시행 격자 | purged CV 분할, 폴드 부호 안정성 | `core/backtest/cv.py` | 동작 |
| `data-quality` (G0) | 시그널 함수, 데이터 | 룩어헤드 누출 리포트 | `core/backtest/leakage.py` | 동작 |
| `adversarial-validator` | 제출 패킷 | DSR·PBO·잔차 t·부트스트랩 p | `core/backtest/stats.py` | 동작 |
| (판정) | 위 전부 | G0·G2~G6 Verdict, 감사로그 기록 | `core/backtest/gates.py` | 동작 |
| `feature-factory` | 피처 후보 | 카탈로그 등록·거부, 순위상관 0.9 중복 거부 | `core/features/catalog.py` | 동작. 등록 시 누출 스캔까지 돌린다 (ADR-0005) |
| `backtest-engineer` (실행) | 가격 패널, 전략, 그리드 | `Submission` + `RunReport` | `core/backtest/engine.py` | 동작. 시점 정합·시행횟수·그로스 한도가 코드 성질 (ADR-0004) |
| `alpha-pod-*` (4개) | 피처, 데이터 | 제출 패킷 | — | 실행기는 생겼으나 실데이터 소스 미연결 |

`tests/canaries/`의 가짜 알파 4종은 더 이상 xfail이 아닙니다. 특히
`canary_lookahead`는 실제로 수익이 나고 G2~G6를 모두 통과하며 **G0만이 잡아냅니다** —
누출 스캔이 조용해지면 하류의 어떤 게이트도 이걸 못 잡는다는 뜻이고, 그것이 테스트로
고정돼 있습니다.

남은 것은 **실데이터 그 자체**입니다. 실행기(`core/backtest/engine.py`)는 검증된
가격 패널에서 제출 패킷을 만들고, 카탈로그(`core/features/catalog.py`)는 피처를 받아
누출·중복·NaN을 거르고, 적재기(`core/data/sources.py`)는 CSV에서 패널과 스냅샷
지문을 만듭니다. 이 환경은 시세 호스트 외부 접속이 정책상 차단되므로 벤더
클라이언트는 없습니다 — 파일을 `data/`에 넣으면 나머지는 돕니다.

## C. 모듈이 비어 있는 역할 (7)

`core/ops/`는 `__init__.py`만 있습니다. `core/portfolio/`에는 센터북 넷팅(`center_book.py`)만 있고, 최적화·자본배분 모듈은 아직 없습니다.

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

이 중 `backtest-engineer`·`adversarial-validator`·`risk-officer`의 판정 코드와
`data-quality`의 G0 누출 스캔은 완료됐습니다. 남은 것은 `feature-factory`(피처
카탈로그)와 실데이터에서 제출 패킷을 만드는 실행기입니다.

이 여섯이 알파 하나를 G0에서 G5까지 실제로 통과시키거나 **떨어뜨리면**, 나머지 17개는
같은 패턴의 복제입니다. 반대로 이 경로가 막히면 17개를 더 만들어도 한 걸음도 나아가지
않습니다. 성공 판정 기준은 "알파가 통과했다"가 아니라 **캐너리 4종이 strict-xfail을
벗고 실제로 기각되는 것**이었고, 2026-09-22에 달성했습니다.

---

## 설계안에서 정정한 것

이 스레드를 연 카드는 "Stock_Agent의 현재 코드와 어떻게 연결되는지"를 요구했습니다.
2026-09-21에 사용자가 이 프로젝트는 Stock_monitoring / Stock_Agent와 **무관한 신규
구축**이라고 명시했으므로, 그 항목은 설계에서 제외했습니다. 위 표의 "코드"는 모두
`quant-desk` 트리 내부를 가리킵니다.
