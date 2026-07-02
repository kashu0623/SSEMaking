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

## Sleep_Stage 확인 결과

Stage probe 결과 `Sleep_Stage`는 row-level로 저장되어 있지만 실제로는 30초 epoch label이 100Hz row마다 반복된 형태입니다.

확인된 raw label:

- `P`: ignore. 파일 앞부분의 pre-recording/placeholder 구간으로 보고 학습에서 제외
- `W`: Wake
- `N1`: N1
- `N2`: N2
- `N3`: N3
- `R`: REM

첫 3개 파일 합산:

- `N2`: 3,558,000 rows
- `P`: 2,225,909 rows
- `W`: 1,880,991 rows
- `N1`: 777,000 rows
- `R`: 597,000 rows
- `N3`: 411,000 rows

`P` 이후 첫 실제 label 전환:

- `S002`: row 917,403, epoch 305, `P -> W`
- `S003`: row 719,403, epoch 239, `P -> W`
- `S004`: row 589,103, epoch 196, `P -> W`

주의할 점은 첫 `P -> W` 전환이 항상 3,000-row 경계에 맞지 않는다는 것입니다. 또한 subject마다 이후 label boundary의 row offset도 다를 수 있습니다. 따라서 loader는 파일 row 0부터 무조건 3,000행씩 자르면 안 됩니다.

권장 epoching 정책:

1. `Sleep_Stage` run-length를 먼저 만든다.
2. raw label `P` run은 버린다.
3. `W/N1/N2/N3/R` run 중 길이가 3,000 rows인 full run을 하나의 30초 epoch로 사용한다.
4. 길이가 3,000 rows보다 짧은 partial run은 기본적으로 제외한다.
5. 길이가 3,000 rows의 배수인 run은 3,000 rows씩 나눠 같은 label epoch로 사용한다.

이 정책을 쓰면 subject별 stage boundary offset이 달라도 label과 feature window가 어긋나는 문제를 피할 수 있습니다.

## Stage probe 재실행 방법

필요하면 Colab에서 전체 파일 또는 더 긴 구간을 스캔해 stage 값의 종류와 전환 시점을 다시 확인합니다.

Colab 명령:

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

## 전처리 실행

대용량 CSV를 통째로 메모리에 올리지 않고 streaming 방식으로 30초 epoch feature CSV를 생성합니다.

먼저 1개 파일 일부 row로 smoke test를 실행합니다.

```python
%cd /content/SSE
!git pull
!PYTHONPATH=src python -m sse_sleep.preprocess_dreamt_100hz \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs" \
  --limit-files 1 \
  --max-rows 1200000
```

문제가 없으면 전체 파일을 처리합니다.

```python
!PYTHONPATH=src python -m sse_sleep.preprocess_dreamt_100hz \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs"
```

출력 파일:

- `dreamt_100hz_epoch_features.csv`: 모델 학습용 epoch feature table
- `dreamt_100hz_preprocess_summary.json`: subject별 alignment offset, skip count, label count 요약

기본 입력 feature:

- `BVP`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`
- `HR`
- `IBI`

`SAO2`를 ablation에 포함하려면 `--include-sao2`를 추가합니다.

## 학습용 NPZ 생성

전처리가 완료되면 epoch feature CSV를 subject-wise train/validation/test split으로 나누고, causal sequence window를 만들어 `.npz`로 저장합니다.

기본 설정:

- subject-wise split: train 70%, validation 15%, test 15%
- context: 과거 10개 epoch, 즉 5분
- target: context의 마지막/latest epoch label
- normalization: train split mean/std로 모든 split 표준화
- non-contiguous epoch gap이 있는 sequence는 제외

Colab 명령:

```python
!PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv" \
  --out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_lstm_context10.npz" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_lstm_context10_summary.json" \
  --context-epochs 10
```

출력:

- `dreamt_100hz_lstm_context10.npz`
- `dreamt_100hz_lstm_context10_summary.json`

NPZ 주요 배열:

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`
- `feature_names`
- `stage5_names`
- `train_feature_mean`, `train_feature_std`
