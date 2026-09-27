# registry/probes

어느 데이터 호스트가 GitHub Actions 러너에서 **실제로 응답하는지**의 기록.

이 디렉터리는 데이터가 아니라 **측정 기록**이다. 클로드의 리서치 컨테이너는
`pypi.org`와 `raw.githubusercontent.com` 말고는 거의 아무 데도 못 간다. 그래서
"이 벤더를 붙일 수 있나"라는 질문은 전부 러너에서만 답할 수 있고, 한 번에 하나씩
물으면 답 하나에 주간 잡 한 사이클이 든다. `www.sec.gov`가 그렇게 네 런을 먹었고
`export.arxiv.org`가 세 런을 먹었다.

`scripts/probe_hosts.py`가 표에 적힌 호스트 전부를 한 번씩 치고 각자가 뭐라고
했는지를 그대로 적는다. **판정이 아니라 기록이다** — 전부 거부당해도 종료코드 0이다.
실패하는 프로버는 실패한 이유를 커밋하지 못한다.

## 읽는 법

- `control_ok`가 `false`면 **그 런은 남의 네트워크가 아니라 우리 네트워크를 쟀다.**
  나머지 줄은 전부 무의미하니 다시 돌린다.
- `verdict`가 `answered 200 with something that is not what we asked for`인 줄이
  가장 위험한 경우다. 어댑터가 조용히 받아들여 빈 유니버스를 만드는 종류다.
- `body_head`는 거부당했을 때도 남는다. 상태 코드만으로는 레이트리밋과 차단과
  포맷 변경이 구분되지 않는다.
- **같은 조직의 두 호스트가 같은 상태 코드로 서로 다른 것을 말할 수 있다.**
  2026-09-26 런에서 `www.sec.gov`는 `Request Rate Threshold Exceeded`,
  `data.sec.gov`는 `Your Request Originates from an Undeclared Automated Tool`을
  줬다. 둘 다 403이다. 하나는 우리가 고칠 수 없고 하나는 고칠 수 있는데, **첫
  줄만 읽고 8일을 보냈다**(ADR-0030). 이 파일의 모든 `body_head`를 읽는다 —
  기록하는 것과 읽는 것은 다른 일이다.
- `verdict`가 `not measured`인 줄은 **통과가 아니라 질문을 안 한 것이다.** 사유가
  `reason`에 있다(예: `SEC_CONTACT_EMAIL` 미설정).
- `headers_of_interest`에 WAF가 자기 이름을 남기는 경우가 많다. 앞에 뭐가 있는지
  아는 것이 다음에 뭘 시도할지의 대부분이다.

## 규칙

- **URL에 키를 넣지 않는다.** 이 기록은 커밋되고 리포는 공개다. 키가 든 프로브
  URL은 공개된 키다(테스트가 막는다).
- **브라우저인 척하지 않는다.** 거짓말로 얻은 200은 정작 쓸 때 깨지고, 그다음
  거부는 진단 불가가 된다.
- 주기는 월 1회다. 도달 가능성은 남의 WAF 규칙 따라 바뀌고, 오래된 답은 답이
  없는 것보다 나쁘다.
