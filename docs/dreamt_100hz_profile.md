# DreamT `data_100Hz` 구조 확인 결과

사용자가 Colab에서 실행한 `inspect_dreamt` 요약 기준입니다.

## 파일 구조

- root: `/content/drive/MyDrive/data_100Hz`
- file count: 100
- inaccessible file count: 0
- skipped copy file count: 0
- 파일명 패턴: `S002_PSG_df_updated.csv`, `S003_PSG_df_updated.csv`, ...
- 파일 크기: subject당 약 1.44-1.49GB
- delimiter: comma
- sampling assumption: 100Hz
- 30초 epoch row count: 3,000 rows

## 확인된 컬럼

공통 컬럼:

- `TIMESTAMP`
- `C4-M1`
- `F4-M1`
- `O2-M1`
- `Fp1-O2`
- `T3 - CZ`
- `CZ - T4`
- `CHIN`
- `E1`
- `E2`
- `ECG`
- `LAT`
- `RAT`
- `SNORE`
- `PTAF`
- `FLOW`
- `THORAX`
- `ABDOMEN`
- `SAO2`
- `BVP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`
- `EDA`
- `HR`
- `IBI`
- `Sleep_Stage`
- `Obstructive_Apnea`
- `Central_Apnea`
- `Hypopnea`
- `Multiple_Events`

## 앱 입력과의 대응

최종 앱 raw 입력은 `IR PPG`, `RED PPG`, `ACC_X`, `ACC_Y`, `ACC_Z`, `TEMP`입니다.

DreamT `data_100Hz`에는 `IR PPG`, `RED PPG` raw column이 없습니다. 대신 `BVP`, `HR`, `IBI`, `SAO2`가 있습니다.

따라서 1차 학습에서는 다음 정책을 사용합니다.

- 앱 raw와 직접 일치: `ACC_X`, `ACC_Y`, `ACC_Z`, `TEMP`
- 앱 PPG에서 계산 가능한 proxy/derived: `BVP`, `HR`, `IBI`
- calibration 확인 후 optional: `SAO2`
- 앱 serving 모델에서 제외: EEG/EOG/EMG/ECG/호흡/이벤트/EDA 계열

## 1차 모델 입력 권장안

기본 feature model:

- `BVP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`
- `HR`
- `IBI`

`SAO2`는 DreamT에는 있지만 실제 앱에서 RED/IR calibration이 확보되어야 안정적으로 계산할 수 있으므로 기본 입력에서는 제외하고 ablation으로 비교합니다.

PSG-rich upper-bound model:

- EEG: `C4-M1`, `F4-M1`, `O2-M1`, `Fp1-O2`, `T3 - CZ`, `CZ - T4`
- EMG: `CHIN`
- EOG 후보: `E1`, `E2`
- ECG/호흡/산소포화도 등

이 모델은 앱 탑재용이 아니라 성능 상한 비교용입니다.

## 아직 확인해야 할 것

`Sleep_Stage` 컬럼의 실제 값이 첫 1,000 row sample에서는 stage alias로 감지되지 않았습니다. 시작 구간이 비어 있거나 stage encoding이 다른 숫자/문자일 수 있습니다.

Colab에서 아래 요약을 한 번 더 실행해 stage encoding을 확인합니다.

```python
import csv
from collections import Counter
from pathlib import Path

root = Path("/content/drive/MyDrive/data_100Hz")
files = sorted(root.glob("S*_PSG_df_updated.csv"))[:3]

for path in files:
    counts = Counter()
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = (row.get("Sleep_Stage") or "").strip()
            if value:
                counts[value] += 1
            rows += 1
            if rows >= 500000:
                break
    print(path.name, "rows_checked=", rows, counts.most_common(20))
```

결과에 따라 `labels.py`의 alias 또는 label map을 확정합니다.

