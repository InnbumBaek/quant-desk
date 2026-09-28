<!-- 생성 파일. `uv run python -m scripts.alpha_status`가 덮어쓴다. 직접 고치지 말 것. -->
# 알파 현황

상태는 **선언이 아니라 기록에서 파생된다.** 선언 파일(`registry/alphas/<id>.yaml`)의
`status:`는 선언 시점의 값이고 갱신되지 않는다 — 고치면 G1이 기각한다(ADR-0036).

**측정은 올리고, 원장은 내린다.** `registry/submissions/`의 판정이 `proposed`를
`gated`까지 올릴 수 있고, 그 위는 G7(페이퍼 기록)이 올린다.
`registry/alphas/lifecycle.yaml`에 적은 사람의 결정은 `stopped`·`retired`로
**내리기만** 한다(ADR-0037).

**`live`는 여기 나오지 않는다.** 실전 자본은 G8이고 사람의 승인이다.
`gates.live_blockers`가 절대 빈 목록을 돌려주지 않으므로, 이 표의 어떤 값도
자본을 받아도 된다는 뜻이 아니다. 주문·배분 경로는 이 표를 읽지 않는다.


| 알파 | 현재 | 선언 시점 | 근거 | 제출 |
| --- | --- | --- | --- | --- |
| `bab-001` | **proposed** | proposed | declared, never submitted | 0 |
| `blend-001` | **proposed** | proposed | declared, never submitted | 0 |
| `breakout-001` | **proposed** | proposed | declared, never submitted | 0 |
| `strev-001` | **proposed** | proposed | declared, never submitted | 0 |
| `tsmom-001` | **proposed** | proposed | 3 submission(s), none cleared the research gates; latest failed G2_in_sample, G3_oos, G4_statistics, G5_robustness | 3 |
| `xsmom-001` | **proposed** | proposed | declared, never submitted | 0 |

---

어느 줄도 자본 허가가 아니다. 허가는 `gates.live_blockers`가 비는 것이고, 비지 않는다.
