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

`Sleep_Stage` 컬럼의 실제 값이 첫 1,000 row sample에서는 stage alias로 감지되지 않았습니다. 추가로 처음 500,000 rows를 확인했을 때 `S002`, `S003`, `S004` 모두 `P`만 나왔습니다.

100Hz에서 500,000 rows는 약 83분입니다. 이 구간이 모두 `P`라면 `P`는 5-class 수면 단계가 아니라 pre-recording, placeholder, preparation period 같은 별도 marker일 가능성이 있습니다. 의미가 확정되기 전까지 `P`를 Wake/N1/N2/N3/REM 중 하나로 매핑하지 않습니다.

Colab에서 전체 파일 또는 더 긴 구간을 스캔해 stage 값의 종류와 전환 시점을 확인합니다.

```python
%cd /content/SSE
!git pull
!PYTHONPATH=src python -m sse_sleep.probe_stage_values \
  --root "/content/drive/MyDrive/data_100Hz" \
  --limit-files 3 \
  --out "/content/drive/MyDrive/dreamt_stage_probe.json"
```

빠른 확인만 하고 싶으면 `--max-rows`를 둡니다.

```python
!PYTHONPATH=src python -m sse_sleep.probe_stage_values \
  --root "/content/drive/MyDrive/data_100Hz" \
  --limit-files 3 \
  --max-rows 2000000
```

생성된 JSON이 너무 길면 아래 명령으로 요약합니다.

```python
!PYTHONPATH=src python -m sse_sleep.summarize_stage_probe \
  --input "/content/drive/MyDrive/dreamt_stage_probe.json" \
  --out "/content/drive/MyDrive/dreamt_stage_probe_summary.txt"
```

출력에서 `total_stage_counts`, 파일별 `stage_counts`, `first_seen`, `transitions`만 공유하면 됩니다.

결과에 따라 `labels.py`의 alias 또는 ignore label 정책을 확정합니다.
