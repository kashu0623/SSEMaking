# SSE Sleep Stage Estimation 다음 채팅방 Handoff

이 문서는 DreamT 기반 수면 단계 예측 모델 개발을 다음 채팅방에서 이어가기 위한 요약이다.

## 최신 상태 요약: 2026-07-14

아래 2026-07-08 섹션은 과거 진행 로그로 보존한다. 다음 채팅방은 이 2026-07-14 섹션을 우선 기준으로 이어가면 된다.

현재 앱/4-class 관점 best 후보는 계속 아래 모델이다.

```text
full w20 = temporal_w20 context20 h64 inverse 1.0x
```

구성:

- epoch feature CSV: `dreamt_100hz_epoch_features_temporal_w20.csv`
- temporal transform:
  - delta lags: `1, 3, 20`
  - rolling windows: `3, 5, 20`
- feature count: `157`
- sequence context: `20` epochs
- model: 1-layer LSTM
- hidden size: `64`
- dropout: `0.4`
- class weight: `inverse`
- N3 weight multiplier: `1.0`
- loss: cross entropy

3-seed 평균 기준 baseline:

```text
                 original temporal   full w20
5-class Macro F1 0.3224              0.3266
5-class Kappa    0.2086              0.2022
4-class Macro F1 0.3881              0.4001
4-class Kappa    0.2273              0.2365
Wake F1          0.4966              0.5011
N3 F1            0.0833              0.1234
REM F1           0.3650              0.3433
```

판단:

```text
유지 best: full w20 = temporal_w20 context20 h64 inverse 1.0x
장점: original temporal 대비 4-class Macro F1/Kappa와 N3 F1 개선
약점: REM F1 하락, 5-class Kappa 소폭 하락
```

### 완료된 targeted/long-window ablation 결론

w20의 N3/4-class 이득을 유지하면서 REM 하락을 줄이기 위해 long-window 대상 feature를 줄이는 실험을 진행했다. 결론적으로 모두 full w20을 넘지 못했다.

3-seed 평균:

```text
variant                          5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
full w20                         0.3266   0.2022   0.4001   0.2365   0.5011  0.1234  0.3433
targeted_w20                     0.3256   0.2082   0.3955   0.2390   0.5110  0.0864  0.3238
cardio_temp_w20                  0.3115   0.1862   0.3895   0.2273   0.5011  0.0809  0.3465
cardio_temp_acc_activity_w20     0.3147   0.1948   0.3795   0.2117   0.5091  0.0738  0.3251
```

Seed42에서만 확인 후 탈락:

```text
movement_only_w20: N3/REM/4-class 모두 낮아 탈락
cardio_temp_acc_mean_w20: REM, 4-class, Kappa 모두 낮아 탈락
```

요약:

```text
targeted_w20은 seed42에서는 유망했지만 seed7/123에서 N3/REM 안정성이 낮았다.
cardio_temp_w20은 REM을 조금 회복했지만 N3/4-class 이득을 잃었다.
ACC activity를 하나만 되돌린 조합도 3-seed에서 실패했다.
따라서 full w20을 계속 best로 유지한다.
```

### 완료된 causal per-night baseline 실험 결론

long-window ablation 이후, original short temporal CSV 위에 prior-only expanding baseline feature를 추가하는 실험을 시작했다. 이 실험은 후처리가 아니라 feature engineering이다.

구현/스크립트:

```text
src/sse_sleep/add_causal_baseline_features.py
scripts/run_causal_baseline_colab.sh
```

feature 생성 방식:

```text
feature_expanding_mean
feature_expanding_std
feature_causal_zscore
```

중요 조건:

```text
현재 epoch 이전 history만 사용한다.
subject_id + source_file 단위로 history를 분리한다.
aligned_epoch_index가 끊기면 history를 reset한다.
미래 epoch나 test subject 통계를 쓰지 않는다.
앱에서도 online update 가능해야 한다.
```

Seed42 결과:

```text
variant                          5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
temporal_baseline                0.3189   0.1737   0.3972   0.2195   0.4411  0.0925  0.4057
cardio_baseline                  0.3227   0.1790   0.3964   0.2094   0.4263  0.1721  0.3089
cardio_temp_baseline             0.3070   0.1863   0.3813   0.2338   0.4603  0.0420  0.3518
movement_cardio_temp_baseline    0.3219   0.2105   0.3918   0.2498   0.4926  0.0275  0.3824
bvp_baseline                     0.2955   0.1733   0.3599   0.1924   0.4900  0.0563  0.2893
bvp_cardio_baseline              0.2684   0.1187   0.3511   0.1565   0.4080  0.0929  0.3154
bvp_temp_baseline                0.3122   0.1798   0.3739   0.1922   0.4160  0.0939  0.3322
```

해석:

```text
temporal_baseline: REM은 매우 좋지만 Wake/Kappa/N3 약함
cardio_baseline: N3는 좋지만 REM/Wake/Kappa 약함
cardio_temp_baseline: REM/Wake/Kappa 일부 회복, N3 붕괴
movement_cardio_temp_baseline: REM/Wake/Kappa 좋지만 N3가 거의 붕괴
bvp_baseline: bvp_std 단독은 test 기준 N3/REM/4-class가 모두 낮아 seed42에서 탈락
bvp_cardio_baseline: N3는 original temporal 이상으로 회복하지만 Wake/Kappa/4-class와 REM이 낮아 seed42에서 탈락
bvp_temp_baseline: bvp 분리 3개 중 가장 균형은 낫지만 Wake/Kappa/4-class가 낮고 full w20을 넘지 못해 seed42에서 탈락
```

현재 causal baseline 후보 중 3-seed 확장할 모델은 없다.

### 다음 성능 향상 테스트 방향

`bvp_std` causal baseline 분리까지 모두 seed42에서 탈락했으므로, 다음은 full w20과 original temporal의 장단점을 직접 결합/보존하는 방향으로 진행한다.

우선순위:

```text
1. original temporal + full w20 prediction late fusion
2. class-wise fusion: REM은 original temporal 쪽, N3/4-class는 full w20 쪽 비중을 높임
3. full w20 label smoothing 0.05 또는 REM weight 1.1
4. full w20 long-feature dropout: *_20 feature train-time dropout
5. w15 중간 long-window 확인
```

구현/스크립트:

```text
src/sse_sleep/evaluate_prediction_fusion.py
scripts/run_prediction_fusion_colab.sh
scripts/run_full_w20_next_training_colab.sh
scripts/run_temporal_long_window_colab.sh  # VARIANTS="15"로 w15 실행
```

권장 실행 순서:

```bash
%cd /content/SSE
!git pull

# 1-2. 학습 없이 prediction fusion 진단
!bash scripts/run_prediction_fusion_colab.sh

# 3-4. seed42 full w20 후속 학습 후보
!bash scripts/run_full_w20_next_training_colab.sh

# 5. w15 중간 window
!VARIANTS="15" bash scripts/run_temporal_long_window_colab.sh
```

판단 기준:

```text
full w20 3-seed 기준:
4-class Macro F1 0.4001
4-class Kappa    0.2365
N3 F1            0.1234
REM F1           0.3433

original temporal 3-seed 기준:
REM F1           0.3650
N3 F1            0.0833
```

목표:

```text
REM F1은 full w20보다 회복하고, 가능하면 original temporal 0.3650 근처로 접근
N3 F1은 original temporal 0.0833 이상, 가능하면 full w20 0.1234 근처 유지
4-class Macro F1/Kappa는 full w20 근처를 유지
Wake F1이 temporal_baseline처럼 크게 무너지면 채택하지 않음
```

## 최신 상태 요약: 2026-07-08

현재 새 best 후보는 기존 `lstm_temporal_context20_h64_inverse`가 아니라 아래 모델이다.

```text
temporal w20 context20 h64 inverse
```

구성:

- epoch feature CSV: `dreamt_100hz_epoch_features_temporal_w20.csv`
- temporal transform:
  - delta lags: `1, 3, 20`
  - rolling windows: `3, 5, 20`
- feature count: `157`
- sequence context: `20` epochs
- model: 1-layer LSTM
- hidden size: `64`
- dropout: `0.4`
- class weight: `inverse`
- loss: cross entropy

3-seed 평균:

```text
                 original temporal   temporal w20   변화
5-class Macro F1 0.3224              0.3266         +0.0042
5-class Kappa    0.2086              0.2022         -0.0064
4-class Macro F1 0.3881              0.4001         +0.0120
4-class Kappa    0.2273              0.2365         +0.0092
Wake F1          0.4966              0.5011         +0.0045
N3 F1            0.0833              0.1234         +0.0401
REM F1           0.3650              0.3433         -0.0217
```

판단:

```text
앱/4-class 관점 현재 best 후보: temporal w20 context20 h64 inverse
5-class staging 관점: original temporal과 trade-off 있음
남은 약점: REM F1 하락, 5-class Kappa 소폭 하락
```

### 최근 실험 결론

아래 실험은 모두 w20 개선 후보로 확인했지만, w20 단독보다 최종적으로 낫지 않았다.

```text
w20 + focal_g2: seed42에서 바로 탈락
w20 + focal_g15: 3-seed에서 N3/REM 하락, 탈락
w20 + selection 4combo: seed42에서 w20과 동일 epoch 선택, 추가 이득 없음
w20 + n3_weight 1.2: 3-seed에서 w20보다 하락, 탈락
w10: seed42 탈락
w30: seed42 탈락
Deep/N3 auxiliary head: seed42 sweep에서 전체 균형 하락, 보류
```

따라서 다음 채팅방은 **w20 단독을 기준선으로 고정**하고, feature 설계를 더 정교하게 조정하는 방향으로 진행한다.

### 다음 채팅방의 우선 작업

목표는 w20의 장점인 N3/4-class 개선을 유지하면서 REM 하락을 줄이는 것이다. 가장 먼저 아래 feature ablation을 한다.

#### 1. Targeted slow w20 feature set

현재 full w20은 10개 base feature 전체에 `delta_20`, `roll_mean_20`, `roll_std_20`을 추가한다. 이 방식은 N3에는 도움이 됐지만 REM을 깎았다. 다음 실험은 long-window를 모든 feature에 붙이지 않고, N3와 관련이 큰 slow physiology/movement feature에만 붙인다.

추천 feature set:

```text
acc_vm_activity
acc_vm_mean
hr_mean
hr_std
ibi_mean
ibi_std
temp_mean
temp_slope
```

의도:

- BVP long-window는 REM 구분을 흐릴 수 있으므로 우선 제외한다.
- 움직임/HR/IBI/TEMP의 긴 안정성만 N3 cue로 추가한다.
- 기존 short temporal feature는 유지한다.

Colab 실행 예시:

```python
%cd /content/SSE
!git pull

# 1) 기존 short temporal feature 생성 또는 기존 파일 사용
!PYTHONPATH=src python -m sse_sleep.add_temporal_features \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv" \
  --out-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features_temporal.csv" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_features_summary.json"

# 2) short temporal CSV 위에 targeted w20만 추가
!PYTHONPATH=src python -m sse_sleep.add_temporal_features \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features_temporal.csv" \
  --out-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features_temporal_targeted_w20.csv" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_targeted_w20_features_summary.json" \
  --base-features "acc_vm_activity,acc_vm_mean,hr_mean,hr_std,ibi_mean,ibi_std,temp_mean,temp_slope" \
  --delta-windows 20 \
  --rolling-window-list 20

# 3) context20 NPZ
!PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features_temporal_targeted_w20.csv" \
  --out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_targeted_w20_lstm_context20.npz" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_targeted_w20_lstm_context20_summary.json" \
  --context-epochs 20

# 4) seed42 학습
!PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_targeted_w20_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_targeted_w20_context20_h64_inverse" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

판단 기준:

```text
targeted_w20이 full w20 대비:
- 4-class Macro F1/Kappa를 유지 또는 개선하는지
- REM F1이 회복되는지
- N3 F1이 full w20의 0.1234 근처를 유지하는지
```

seed42에서 좋으면 seed7/123으로 확장한다. seed42에서 full w20보다 명확히 나쁘면 중단한다.

#### 2. REM-preserving w20 ablation

targeted slow w20 다음 후보는 full w20에서 REM을 해칠 수 있는 long-window feature만 제외하는 ablation이다.

우선 제외 후보:

```text
bvp_mean
bvp_std
```

즉, long-window 대상은 다음 8개로 제한한다.

```text
acc_vm_mean
acc_vm_activity
temp_mean
temp_slope
hr_mean
hr_std
ibi_mean
ibi_std
```

이것은 targeted slow w20과 거의 같은 방향이다. seed42 결과가 애매하면 base feature 조합을 조금 좁혀 다음처럼 비교한다.

```text
movement_only_w20: acc_vm_mean, acc_vm_activity
cardio_temp_w20: hr_mean, hr_std, ibi_mean, ibi_std, temp_mean, temp_slope
```

#### 3. Causal per-night baseline feature

long-window ablation이 한계에 닿으면, 다음은 개인/밤별 기준선 feature를 추가한다. 앱에서도 계산 가능한 방식이어야 하므로 미래를 쓰지 않는 expanding history만 사용한다.

후보:

```text
feature_expanding_mean
feature_expanding_std
feature_causal_zscore
```

대상 feature:

```text
acc_vm_activity
hr_mean
ibi_mean
temp_mean
bvp_std
```

의도:

- subject마다 HR/IBI/TEMP baseline이 다르므로 global normalized raw value만으로는 수면단계 분리가 어려울 수 있다.
- 현재 epoch가 “그 밤의 이전 흐름 대비 낮은 움직임/안정 HR/안정 IBI인지”를 표현한다.

주의:

- 반드시 현재 epoch 이전 history만 쓰거나, 현재 epoch 포함 causal expanding만 쓴다.
- subject/test leakage 금지.
- serving에서도 같은 방식으로 online 업데이트 가능해야 한다.

#### 4. 모델 구조 실험은 뒤로 미룬다

지금까지의 실험상 loss/auxiliary head로 N3를 억지로 올리면 REM/Wake/Kappa 손실이 반복됐다. 따라서 다음 채팅방의 1순위는 구조 변경이 아니라 feature ablation이다.

보류:

```text
focal 추가 탐색
Deep auxiliary head 추가 탐색
weighted sampler
hidden128
GRU 기본 채택
```

## 프로젝트 목표

자체 제작 웨어러블 기기에서 실시간으로 들어오는 raw sensor:

- `GREEN PPG`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`

를 이용해 30초 epoch 단위 수면 단계를 추론하고, 스마트 알람 앱에서 깨우기 좋은 수면 단계에 알람을 울리는 알고리즘을 만드는 것이 목표다.

모델은 5-class로 학습한다.

- Wake
- N1
- N2
- N3
- REM

평가는 5-class 원본 성능과 함께 4-class 병합 성능도 출력한다.

- Wake = Wake
- Light = N1 + N2
- Deep = N3
- REM = REM

평가 지표는 Accuracy만 보지 않고 Macro F1, Cohen's Kappa, confusion matrix, class-wise precision/recall/F1을 본다.

## 저장소와 실행 환경

GitHub repo:

```text
https://github.com/kashu0623/SSEMaking.git
```

로컬 작업 폴더:

```text
/Users/chan/Documents/SSE
```

Colab 기준 데이터 경로:

```text
/content/drive/MyDrive/data_100Hz
```

Colab output 경로:

```text
/content/drive/MyDrive/SSE_outputs
```

현재 repo는 `main` branch에 push되어 있다. 마지막 주요 commit:

```text
0eb3516 Add temporal rolling feature builder
```

다음 채팅방 시작 시 Colab에서는 먼저:

```python
%cd /content/SSE
!git pull
```

## DreamT 데이터 구조 확인 결과

DreamT Drive 폴더:

```text
/content/drive/MyDrive/data_100Hz
```

구조:

- CSV 100개
- subject별 파일명: `S002_PSG_df_updated.csv`, `S003_PSG_df_updated.csv`, ...
- 각 파일 약 1.4GB
- 100Hz row-level CSV
- 30초 epoch = 3000 rows

확인된 주요 컬럼:

- `TIMESTAMP`
- PSG 계열: `C4-M1`, `F4-M1`, `O2-M1`, `Fp1-O2`, `T3 - CZ`, `CZ - T4`, `CHIN`, `E1`, `E2`, `ECG`, ...
- 앱 후보/파생 feature 계열: `BVP`, `ACC_X`, `ACC_Y`, `ACC_Z`, `TEMP`, `HR`, `IBI`
- DreamT-only optional ablation: `SAO2`
- label: `Sleep_Stage`

중요한 설계 판단:

- DreamT에는 앱 raw 입력인 `GREEN_PPG`가 직접 없다.
- 대신 `BVP`, `HR`, `IBI`, `SAO2`가 있다.
- 1차 앱 후보 모델에서는 `BVP`, `ACC_X/Y/Z`, `TEMP`, `HR`, `IBI`를 사용한다.
- `SAO2`는 GREEN 단일 PPG 앱 입력에서 안정적으로 계산할 수 없으므로 core feature가 아니라 DreamT-only optional ablation으로 둔다.
- EEG/EOG/EMG/ECG/호흡/이벤트/EDA 계열은 앱 serving model 입력에서 제외한다.

## Sleep_Stage 라벨 구조

Stage probe 결과:

- `P`: pre-recording/placeholder로 보고 학습 제외
- `W`: Wake
- `N1`: N1
- `N2`: N2
- `N3`: N3
- `R`: REM

중요:

- 첫 `P -> W` 전환 row가 subject마다 3000-row boundary와 딱 맞지 않는다.
- 따라서 file row 0부터 무조건 3000행씩 자르면 label window가 틀어질 수 있다.
- preprocessing에서는 subject/file별 dominant stage-transition offset을 찾아 그 offset 기준의 3000-row window만 사용한다.
- window 내부 label이 하나로 고정된 경우만 epoch로 저장한다.
- `P` window와 mixed-label window는 제외한다.

관련 파일:

- `src/sse_sleep/probe_stage_values.py`
- `src/sse_sleep/summarize_stage_probe.py`
- `src/sse_sleep/preprocess_dreamt_100hz.py`
- `configs/dreamt_100hz_column_map.json`
- `docs/dreamt_100hz_profile.md`

## 전처리 결과

전체 100개 파일 전처리 완료.

출력:

```text
/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv
/content/drive/MyDrive/SSE_outputs/dreamt_100hz_preprocess_summary.json
```

전체 전처리 summary:

```text
files_processed: 100
total_epochs_written: 79,865

Wake: 20,034  25.08%
N1:    8,806  11.03%
N2:   39,938  50.01%
N3:    2,703   3.38%
REM:   8,384  10.50%
```

subject당 epoch 수:

```text
min: 623
median: 806
max: 954
mean: 약 798.6
```

mixed label skip:

```text
total mixed skipped: 141
total ignored P windows: 23,716
partial total: 2
```

전처리 명령:

```python
!PYTHONPATH=src python -m sse_sleep.preprocess_dreamt_100hz \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs"
```

## NPZ 생성

기본 epoch feature CSV에서 LSTM용 context10/20/30 NPZ를 만들었다.

기본 context10 summary:

```text
subject_count: 100
feature_count: 67

X_train: [55133, 10, 67]
X_val:   [11787, 10, 67]
X_test:  [11844, 10, 67]

subject-wise split:
train 70명 / val 15명 / test 15명
```

context20:

```text
X_train: [54314, 20, 67]
X_val:   [11602, 20, 67]
X_test:  [11694, 20, 67]
```

context30:

```text
X_train: [53521, 30, 67]
X_val:   [11422, 30, 67]
X_test:  [11544, 30, 67]
```

NPZ 생성 명령 예시:

```python
!PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv" \
  --out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_lstm_context20.npz" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_lstm_context20_summary.json" \
  --context-epochs 20
```

## 현재까지 LSTM 실험 결과

모델:

- `src/sse_sleep/train_lstm.py`
- 1-layer LSTM
- causal context
- target은 latest epoch label
- output은 5-class
- evaluation은 5-class와 4-class 둘 다 저장

### Baseline 1: context10, hidden128, dropout0.2, inverse class weight

```text
5-class test Macro F1: 0.2972
4-class test Macro F1: 0.3675
5-class test Kappa:    0.1369
```

### Baseline 2: context10, hidden64, dropout0.4, inverse class weight

```text
5-class test Macro F1: 0.3036
4-class test Macro F1: 0.3783
5-class test Kappa:    0.1435
```

### Class weight 실험

`--class-weight-mode` 옵션 추가됨:

- `inverse`: 기존 방식. minority class 보정 강함.
- `sqrt`: 보정 완화.
- `none`: unweighted cross entropy.

결론:

- Accuracy는 `sqrt`/`none`에서 올라가지만 N1/N3가 거의 무너진다.
- 5-class/4-class Macro F1 기준으로는 `inverse`가 가장 낫다.

`sqrt h64 dropout0.4`:

```text
5-class test Macro F1: 0.2922
4-class test Macro F1: 0.3520
```

`none h64 dropout0.4`:

```text
5-class test Macro F1: 0.2707
4-class test Macro F1: 0.3447
```

### Context 길이 실험

context10 -> context20 -> context30 비교.

`context20 h64 dropout0.4 inverse`:

```text
5-class test Macro F1: 0.3252
4-class test Macro F1: 0.3916
5-class test Kappa:    0.1682
4-class test Kappa:    0.1793

Wake F1: 0.453
N1 F1:   0.172
N2 F1:   0.469
N3 F1:   0.244
REM F1:  0.288
```

`context30 h64 dropout0.4 inverse`:

```text
5-class test Macro F1: 0.3160
4-class test Macro F1: 0.3805
5-class test Kappa:    0.1655

Wake F1: 0.419
N1 F1:   0.204
N2 F1:   0.515
N3 F1:   0.186
REM F1:  0.255
```

결론:

- context20가 현재 기본 feature 기준 best.
- context30은 N1/N2는 좋아지지만 Wake/N3/REM이 떨어진다.

## 방금 진행하던 실험: Rolling/Delta Temporal Feature

문제의식:

- 기존 feature는 각 30초 epoch 안의 통계값만 있다.
- 수면 단계는 흐름이 중요하다.
- HR/IBI/움직임/체온/BVP가 이전 epoch 대비 어떻게 변했는지 feature로 명시하면 LSTM이 덜 힘들 수 있다.

추가된 스크립트:

```text
src/sse_sleep/add_temporal_features.py
```

기능:

- 기존 `dreamt_100hz_epoch_features.csv`를 읽는다.
- subject별, aligned_epoch_index 순서로 처리한다.
- subject 경계를 넘지 않는다.
- epoch index gap이 있으면 history를 reset한다.
- 현재 epoch보다 과거 epoch만 사용하므로 실시간 앱 조건과 맞다.

기본 대상 base feature 10개:

```text
bvp_mean
bvp_std
acc_vm_mean
acc_vm_activity
temp_mean
temp_slope
hr_mean
hr_std
ibi_mean
ibi_std
```

추가 feature:

- `delta_1`
- `delta_3`
- `roll_mean_3`
- `roll_std_3`
- `roll_mean_5`
- `roll_std_5`

즉 10개 base feature * 6개 temporal transform = 60개 추가.

기존 feature:

```text
67개
```

temporal feature 추가 후:

```text
127개
```

실행 명령:

```python
%cd /content/SSE
!git pull

!PYTHONPATH=src python -m sse_sleep.add_temporal_features \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv" \
  --out-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features_temporal.csv" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_features_summary.json"
```

그 다음 temporal CSV로 context20 NPZ 생성:

```python
!PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features_temporal.csv" \
  --out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20_summary.json" \
  --context-epochs 20
```

학습:

```python
!PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

### Temporal feature 실험 결과

파일:

```text
/Users/chan/Downloads/lstm_metrics-7.json
```

설정:

```text
context: 20
features: 127
hidden_size: 64
dropout: 0.4
class_weight_mode: inverse
```

결과:

```text
5-class test Macro F1: 0.3326
4-class test Macro F1: 0.4036
5-class test Kappa:    0.2258
4-class test Kappa:    0.2633
```

기존 best `context20 base` 대비:

```text
5-class Macro F1: +0.0074
4-class Macro F1: +0.0120
5-class Kappa:    +0.0576
4-class Kappa:    +0.0840
```

클래스별 변화:

```text
Wake F1: 0.453 -> 0.490
N1 F1:   0.172 -> 0.177
N2 F1:   0.469 -> 0.529
N3 F1:   0.244 -> 0.051  크게 나빠짐
REM F1:  0.288 -> 0.416  크게 좋아짐
```

해석:

- rolling/delta feature는 전체적으로 효과가 있다.
- 특히 Wake/N2/REM과 Kappa가 좋아졌다.
- 그러나 N3가 크게 무너졌다.
- 스마트 알람 앱 관점에서는 4-class Macro F1과 Kappa가 좋아진 점이 의미 있다.
- 하지만 5-class 모델로는 N3 보존이 중요하므로 다음 실험은 N3를 살리는 방향이 좋다.

현재 best:

```text
lstm_temporal_context20_h64_inverse
5-class Macro F1: 0.3326
4-class Macro F1: 0.4036
```

## 다음 채팅방에서 이어서 할 일

바로 이어서 할 가장 좋은 다음 실험:

### 1. Temporal context20에서 hidden size 128 실험

의도:

- temporal feature 추가로 feature 수가 67 -> 127로 증가했다.
- hidden 64가 표현력이 부족해 N3를 못 살렸을 가능성이 있다.
- hidden 128로 올리고 dropout 0.4를 유지해서 N3/REM/Wake 균형을 확인한다.

실행:

```python
!PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h128_inverse" \
  --hidden-size 128 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

결과 파일:

```text
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h128_inverse/lstm_metrics.json
```

이 결과를 현재 best인 `lstm_metrics-7.json`과 비교한다.

### 2. N3 보존 개선

hidden128 결과:

```text
lstm_temporal_context20_h128_inverse
5-class Macro F1: 0.3192
4-class Macro F1: 0.3861
5-class Kappa:    0.1979
4-class Kappa:    0.2193

Wake F1: 0.479
N1 F1:   0.166
N2 F1:   0.518
N3 F1:   0.085
REM F1:  0.347
```

해석:

- hidden128은 N3를 h64의 0.051 -> 0.085로 조금 회복했지만 여전히 낮다.
- 대신 REM, 4-class Macro F1, Kappa가 모두 떨어졌다.
- 따라서 현재 best는 계속 `lstm_temporal_context20_h64_inverse`.
- 다음 실험은 hidden size 증가보다 N3 class weight 직접 강화가 우선이다.

추가된 옵션:

```text
--n3-weight-multiplier
```

기본값은 `1.0`이며, 기존 결과 재현에는 영향이 없다. `inverse`/`sqrt` weight 계산 후 N3 weight만 지정 배율만큼 키우고 다시 평균 1로 normalize한다.

바로 이어서 실행할 실험:

```python
!PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_n3x15" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse \
  --n3-weight-multiplier 1.5
```

```python
!PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_n3x20" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse \
  --n3-weight-multiplier 2.0
```

판단 기준:

- N3/Deep F1이 최소 0.12~0.15 이상 회복되는지 확인한다.
- 동시에 REM F1과 4-class Macro F1/Kappa가 크게 무너지면 채택하지 않는다.
- N3가 좋아져도 4-class Macro F1이 h64 best `0.4036`에서 너무 떨어지면 스마트 알람 관점의 overall best는 아니다.

그 다음 후보:

- temporal feature set 조정
- N3에 도움이 될 수 있는 feature 추가/선별
- Deep/N3 binary auxiliary head 또는 multi-task 구조

### 3. 반복 split 확인 결과

N3 multiplier는 `1.0x`와 `1.2x`를 seed 42/7/123에서 반복 검증했다.

```text
3-seed mean:

                 1.0x      1.2x
5-class Macro F1 0.3224    0.3226
5-class Kappa    0.2086    0.1908
4-class Macro F1 0.3881    0.3883
4-class Kappa    0.2273    0.2020
N3 F1            0.0833    0.1477
REM F1           0.3650    0.3339
Wake F1          0.4966    0.4768
```

결론:

```text
기본 모델: temporal context20 h64 inverse 1.0x
1.2x: Deep-aware ablation으로만 보관
```

1.2x는 seed42에서는 N3를 크게 살렸지만 seed7에서는 실패했고 seed123에서는 거의 차이가 없었다. 평균 Macro F1은 거의 같지만 Kappa, Wake, REM이 낮아지므로 기본 모델로는 1.0x가 더 안정적이다.

### 4. GRU h64 inverse 결과

hidden size 증가와 N3 weight 조정이 안정적 개선을 만들지 못했으므로, 같은 temporal context20 입력과 inverse weight에서 recurrent cell만 LSTM에서 GRU로 바꿔 비교한다.

추가된 옵션:

```text
--model-type lstm|gru
```

기존 LSTM 실험 재현에는 영향이 없도록 기본값은 `lstm`이다.

실행:

```python
%cd /content/SSE
!git pull
!bash scripts/run_gru_seed_validation_colab.sh
```

결과 파일:

```text
/content/drive/MyDrive/SSE_outputs/gru_temporal_context20_h64_inverse_seed42/lstm_metrics.json
/content/drive/MyDrive/SSE_outputs/gru_temporal_context20_h64_inverse_seed7/lstm_metrics.json
/content/drive/MyDrive/SSE_outputs/gru_temporal_context20_h64_inverse_seed123/lstm_metrics.json
```

3-seed 평균 비교:

```text
                 LSTM 1.0x   GRU
5-class Macro F1 0.3224     0.3234
5-class Kappa    0.2086     0.1939
4-class Macro F1 0.3881     0.3948
4-class Kappa    0.2273     0.2233
N3 F1            0.0833     0.0950
REM F1           0.3650     0.3479
Wake F1          0.4966     0.4934
```

seed별 변화:

```text
seed42: GRU가 LSTM보다 확실히 나쁨
  5-class Macro F1: -0.0320
  4-class Macro F1: -0.0265
  4-class Kappa:    -0.0503
  REM F1:           -0.0567
  N2 F1:            -0.1037

seed7: GRU가 약간 좋음
  5-class Macro F1: +0.0080
  4-class Macro F1: +0.0122
  4-class Kappa:    +0.0033
  N3 F1:            +0.0524
  REM F1:           -0.0369

seed123: GRU가 꽤 좋음
  5-class Macro F1: +0.0268
  4-class Macro F1: +0.0345
  4-class Kappa:    +0.0351
  REM F1:           +0.0424
  N3 F1:            -0.0325
```

결론:

```text
기본 모델 유지: LSTM temporal context20 h64 inverse 1.0x
GRU: ablation 후보로 보관, 기본 채택은 보류
```

GRU는 4-class Macro F1과 N3 F1 평균을 조금 올렸지만, 5-class/4-class Kappa가 낮고 seed42에서 크게 무너졌다. 스마트 알람 앱 기본 모델로는 아직 LSTM이 더 안전하다.

### 5. 다음 실험 후보였던 causal smoothing 평가

업데이트: causal smoothing은 후처리 기법이므로 지금은 우선순위를 낮춘다. 현재 모델의 학습 성능, 특히 N3/REM/Wake 균형과 4-class Kappa가 아직 충분히 안정적이지 않으므로, smoothing은 나중에 앱 정책 안정화/진단용으로 다시 평가한다.

원래 계획은 재학습이 아니라 post-processing 평가였다. 실제 앱에서는 30초마다 수면 단계가 크게 튀는 raw prediction을 그대로 쓰지 않고, 최근 몇 epoch의 예측을 causal하게 smoothing하는 것이 자연스럽다.

평가 목적:

- raw epoch prediction 대비 4-class Macro F1, Kappa가 좋아지는지 확인
- Wake/REM/N3의 과도한 튐을 줄이는지 확인
- 스마트 알람 정책에서 사용할 안정적인 stage probability/state를 만들 수 있는지 확인

우선 대상 모델:

```text
LSTM temporal context20 h64 inverse 1.0x
seed42, seed7, seed123
```

필요한 파일:

```text
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse/lstm_predictions.npz
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_seed7/lstm_predictions.npz
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_seed123/lstm_predictions.npz
```

현재 `train_lstm.py`의 `lstm_predictions.npz`에는 `y_true`, `y_pred`만 저장된다. 확률 smoothing을 하려면 logits/probabilities가 필요하므로 다음 채팅방에서 우선 `train_lstm.py`를 확장해 `val_logits`, `test_logits` 또는 `val_probs`, `test_probs`를 저장하도록 수정한다.

그 다음 smoothing 평가 스크립트를 추가한다.

후보 smoothing:

```text
1. 최근 3 epoch majority vote
2. 최근 5 epoch majority vote
3. 최근 3 epoch probability moving average
4. 최근 5 epoch probability moving average
5. transition guard: N3/REM 전환은 2 epoch 이상 지속될 때 확정
```

주의:

- 모든 smoothing은 미래 epoch를 쓰지 않는 causal 방식이어야 한다.
- subject 경계를 넘으면 안 된다.
- 가능하면 `test_subject_ids`, `test_epoch_indices`를 같이 사용해 subject별 sequence 순서를 보존한다.

### 6. 현재 우선순위: 학습 성능 개선

현재 방향은 smoothing보다 학습 성능 개선이다.

추가된 문서와 스크립트:

```text
docs/training_improvement_plan.md
scripts/run_learning_improvement_colab.sh
```

`train_lstm.py`에 추가된 실험 옵션:

```text
--selection-metric
--loss-type cross_entropy|focal
--focal-gamma
--label-smoothing
--train-sampler none|weighted
```

1차 후보는 seed42에서 빠르게 필터링한 뒤 유망한 후보만 seed42/7/123으로 반복 검증한다.

## 주요 코드 파일

- `src/sse_sleep/preprocess_dreamt_100hz.py`: DreamT 100Hz raw CSV -> 30초 epoch feature CSV
- `src/sse_sleep/add_temporal_features.py`: epoch feature CSV -> rolling/delta feature CSV
- `src/sse_sleep/build_npz_dataset.py`: feature CSV -> subject-wise split NPZ
- `src/sse_sleep/train_lstm.py`: LSTM/GRU 학습 및 5-class/4-class 평가
- `src/sse_sleep/metrics.py`: Accuracy, Macro F1, Cohen's Kappa, confusion matrix, class-wise metrics
- `src/sse_sleep/alarm.py`: 스마트 알람 정책 초안
- `docs/dreamt_100hz_profile.md`: DreamT 데이터 구조 및 실행 명령 정리
- `docs/dreamt_pipeline_design.md`: 전체 설계 문서

## 다음 채팅방 시작 메시지 예시

다음 채팅방에는 이 파일을 올리고 이렇게 시작하면 된다.

```text
이전 채팅방에서 DreamT data_100Hz 기반 수면 단계 예측 파이프라인을 여기까지 진행했다.
docs/next_chat_handoff.md 내용을 읽고 이어서 진행해줘.
현재 앱/4-class 관점 best 후보는 full w20 = temporal_w20 context20 h64 inverse 1.0x다.
targeted/long-window ablation과 causal baseline 일부 실험은 완료했고 모두 full w20을 넘지 못했다.
다음 작업은 causal per-night baseline에서 bvp_std 역할을 분리하기 위해 bvp_baseline, bvp_cardio_baseline, bvp_temp_baseline variant를 추가하고 seed42부터 진행해줘.
```
