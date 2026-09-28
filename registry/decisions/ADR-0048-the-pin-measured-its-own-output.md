<!-- artifact-run: ea466c53689374f9.tsmom-001 -->

# ADR-0048 — 핀이 자기 자신의 출력을 재고 있었다

- 상태: 채택
- 날짜: 2026-09-28
- 관련: ADR-0012(재현 3중 핀), ADR-0035, ADR-0040(한 스냅샷·한 핀·한 데스크 N),
  ADR-0041(선언된 유니버스), ADR-0046(같은 여섯을 스물여섯에서), ADR-0047

## 무엇이 일어났나

ADR-0046이 선언한 대로 러너가 `submit_alpha --all`을 두 유니버스에 돌렸다.
결과는 절반이었다. `-001` 여섯은 자기 다섯 종목에서 판정이 나왔고
(`universe.source` = `legacy`, `universe.declared` 다섯), `-002` 여섯은 하나도
평가되지 않았다. 로그의 마지막 줄:

```
core.repro.NotReproducible: the working tree has uncommitted changes, so HEAD
does not describe the code that would run.
  File "scripts/submit_alpha.py", line 705, in main
    pin = pin_current(manifest.snapshot_id, args.seed, allow_dirty=args.allow_dirty)
```

트리를 더럽힌 것은 사람도 아니고 다른 잡도 아니었다. **첫 유니버스가 방금 쓴
판정 기록 여섯 개다.** 루프가 유니버스마다 `pin_current`를 불렀고, 두 번째
호출 시점에 `registry/submissions/`에는 첫 번째 유니버스의 출력이 커밋되지 않은
채 놓여 있었다. 핀은 코드를 재는 검사인데, 실제로 잰 것은 그 실행 자신의
출력이었다.

## 왜 이 검사는 옳고 호출 위치는 틀렸나

`pin_current`가 더러운 트리를 거부하는 것은 ADR-0012의 핵심이고 완화 대상이
아니다. `allow_dirty`로 우회하면 run id가 그 숫자를 만든 코드를 더 이상
기술하지 않는다. 문제는 기준이 아니라 **언제 묻는가**였다.

코드 상태는 실행 전체에 대해 하나다. 유니버스는 여럿이다. 그래서 순서를
뒤집는다: 코드는 **아무것도 쓰기 전에 한 번** 확인하고, 유니버스마다 그
동일한 코드 상태를 자기 스냅샷 id로 봉인한다. `ReproPin.for_snapshot()`이
그 봉인이고, 새 핀도 감사로그에 남으므로 스냅샷마다 하나씩 기록된다.
`dirty` 플래그는 그대로 따라간다 — 코드에 관한 사실이고 패널 사이에 코드가
바뀌지 않았기 때문이다. 이 메서드로 `dirty`를 지울 수는 없다. 스크래치 실행은
그 안의 모든 유니버스에 대해 스크래치 실행이다.

부수적으로 패널 로딩도 앞으로 당겼다. 유니버스 전부의 패널을 먼저 읽고,
그 다음에 핀을 잡고, 그 다음에 쓴다. 읽기가 실패하는 유니버스는 어떤 기록도
남기기 전에 걸러진다.

## 두 번째 결함: 한 사실을 두 번 조회했다

이 수정의 회귀 테스트가 원래 찾던 것과 다른 것을 찾아냈다. `main()`은
`universes.for_alpha(alpha_id, alphas, mapping)`로 유니버스를 풀어 그룹을
만드는데, `submit()`은 자기 안에서 **기본 레지스트리 경로로** 같은 조회를 다시
했다. `--alphas`로 다른 디렉터리를 가리킨 배치는 그 디렉터리의 선언으로
묶이고 라이브 디렉터리의 선언으로 판정된다. 운영에서는 두 경로가 일치하므로
이 불일치는 보이지 않았다 — 보이지 않았던 이유가 바로 그것이다. 이제
`main()`이 푼 `(symbols, source)`를 `submit(universe=...)`로 넘기고, 조회는
한 곳에만 있다.

**한 사실을 두 번 조회하면 두 개의 사실이 된다.** 적혀 있는 것이 집행되는
것과 다른 열두 번째 사례이고, 이번에는 불일치가 값이 아니라 *출처*였다.

## 예산은 그대로 구속되지 않았다

실패한 실행이 이미 답한 것 하나. 선언이 여섯에서 열둘로 늘어 데스크 N이
`{{artifact:/sharpe_required.desk_trials}}`가 되었고, 그 N에서 요구되는 연율
샤프는 `{{artifact:/sharpe_required.at_desk_trials}}`다. G2의 고정 하한
`1.0`보다 낮으므로 **구속하는 기준은 여전히 표본 내 샤프 하한이다.**
ADR-0046이 예산으로 주장한 것이 실측으로 확인됐다. 다중 검정 문턱이
G2를 넘어서는 N은 ADR-0040이 잰 237이고, 지금은 열둘이다.

## 반증 조건

- `for_snapshot`이 `dirty=True`를 `False`로 바꿀 수 있게 되면 이 결정은
  무효다. `tests/test_repro.py::test_resealing_does_not_launder_a_scratch_run`이
  막는다.
- 두 유니버스가 서로 다른 `git_sha`로 판정되면 무효다.
  `test_the_two_universes_are_pinned_to_different_snapshots`가 같은 `git_sha`와
  다른 `snapshot_id`·`run_id`를 요구한다.
- `submit()`이 다시 유니버스를 자기 힘으로 조회하기 시작하면 무효다.
  `--alphas`를 임시 디렉터리로 가리키는 엔드투엔드 테스트가 그때 깨진다.

## 대안과 기각 이유

- **`--allow-dirty`로 러너를 돌린다.** 기각. run id가 코드를 기술하지 않게
  되고, 그러면 재현 핀이 장식이 된다.
- **유니버스마다 중간 커밋을 한다.** 기각. 러너가 판정 사이에 커밋하면
  실패한 실행이 절반의 기록을 main에 남기고, 어느 절반인지는 실패 지점에
  달린다.
- **유니버스를 한 번에 하나만 돌린다.** 기각. ADR-0040의 "한 데스크 N"이
  깨진다. 두 유니버스의 판정이 서로 다른 N으로 나오면 비교가 성립하지 않는다.
