# data/

여기에 들어오는 파일은 **커밋되지 않습니다**(`.gitignore`). 라이선스가 걸려 있고
크기도 크기 때문입니다. 커밋되는 것은 이 README와, 무엇을 읽었는지 지문으로
남기는 `registry/snapshots/<snapshot_id>.json` 뿐입니다(ADR-0006).

## 기대하는 형식

심볼당 일간 CSV 한 개. 컬럼 이름은 대부분의 벤더 내보내기와 같습니다.

```
Date,Open,High,Low,Close,Volume
2024-01-02,186.06,186.74,185.19,185.64,82488700
```

- `Date`와 `Close`는 필수, `Volume`은 있으면 달러 거래대금으로 환산해 G6 ADV
  참여율 판정에 씁니다. 없으면 참여율은 0이 되고 "측정값이 아님"이 기록됩니다.
- 날짜가 중복되면 읽기를 거부합니다. 조정 가격을 쓰려면 `close_col`을 바꿔
  넘기세요. 같은 패널 안에서 조정/비조정을 섞지 마십시오.

## 읽는 방법

```python
from pathlib import Path
from core.data.sources import load_csv_panel, write_manifest

files = {p.stem.upper(): p for p in Path("data").glob("*.csv")}
panel, manifest = load_csv_panel(files)
write_manifest(manifest, Path("registry/snapshots"))
```

날짜는 **교집합**으로 맞춥니다. 한 심볼에만 있는 날짜는 채우지 않고 버리며,
버린 수를 `manifest.dates_dropped`에 남깁니다. 공통 날짜 비율이 98% 아래면
패널을 아예 반환하지 않습니다 — 구멍은 패널이 아니라 소스에서 고칩니다.

## 네트워크

현재 실행 환경은 시세 호스트로의 외부 접속을 차단합니다(egress 프록시 거부).
그래서 벤더 클라이언트는 없고, 데이터는 파일로 들어옵니다. 환경의 네트워크
정책을 바꾸면 그때 어댑터를 붙입니다.
