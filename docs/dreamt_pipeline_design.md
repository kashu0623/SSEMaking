# DreamT 수면 단계 예측 데이터 파이프라인 설계

## 현재 확인 상태

작업 폴더(`/Users/chan/Documents/SSE`)에는 DreamT 원본을 로컬에 두지 않습니다. 원본은 Google Drive의 `/content/drive/MyDrive/data_100Hz`에서 Colab으로 읽습니다.

Colab inspection 결과 `data_100Hz`는 100개 CSV 파일로 구성되어 있고, 각 파일은 subject별 100Hz 통합 테이블입니다. 확인된 자세한 구조는 [dreamt_100hz_profile.md](/Users/chan/Documents/SSE/docs/dreamt_100hz_profile.md)에 정리했습니다.

고정 매핑은 [dreamt_100hz_column_map.json](/Users/chan/Documents/SSE/configs/dreamt_100hz_column_map.json)을 기준으로 합니다.

## 앱 입력과 feature provenance

최종 웨어러블 앱 raw 입력:

- `IR PPG`
- `RED PPG`
- `ACC_X`
- `ACC_Y`
- `ACC_Z`
- `TEMP`

학습 단계에서 DreamT가 더 많은 PSG 신호를 제공하더라도, feature는 다음 세 범주로 나눕니다.

- `app_raw`: 앱 raw 입력에서 직접 계산 가능
- `app_derived`: 앱 raw 입력에서 품질 조건이 맞으면 계산 가능
- `dreamt_only`: DreamT/PSG에만 있어 최종 앱 입력으로는 계산 불가

1차 모델에는 `app_raw`와 `app_derived` feature만 넣는 것을 권장합니다. `dreamt_only`는 교사용 분석, ablation, upper-bound 비교에는 쓸 수 있지만 앱 탑재 모델의 입력으로 사용하면 train-serving skew가 생깁니다.

## 권장 채널 사용 정책

DreamT에 PPG, ACC, TEMP가 있으면 앱과 같은 입력군으로 맞춥니다.

- PPG: `data_100Hz`에는 `IR_PPG`/`RED_PPG` raw가 없고 `BVP`, `HR`, `IBI`, `SAO2`가 있음
- ACC: 3축이 있으면 vector magnitude와 움직임 feature 생성
- TEMP: 저주파 추세와 안정도 feature 생성

따라서 `data_100Hz` 1차 앱 후보 모델에서는 `BVP`, `HR`, `IBI`를 PPG-derived proxy로 사용합니다. 실제 앱에서는 `IR_PPG`/`RED_PPG` raw에서 동일 계열 feature를 계산해야 하므로 train-serving 차이를 별도 검증합니다.

DreamT에 ECG, EEG, EOG, EMG, airflow, SpO2 등이 있으면 기본 앱 모델 입력에서는 제외합니다. 단, 다음 용도는 허용합니다.

- 라벨 alignment 검증
- PPG 품질/심박 feature 계산의 참조 비교
- 앱 입력 모델과 PSG-rich 모델 성능 차이 측정

## 30초 epoch 구성

모든 신호는 subject/session 단위로 정렬합니다.

1. 원본 timestamp 또는 sample index를 기준으로 recording start를 확정합니다.
2. label annotation을 30초 epoch 기준으로 canonical stage에 매핑합니다.
3. 각 epoch 구간 `[start, start + 30s)`에 들어오는 raw sample을 잘라냅니다.
4. epoch별 신호 누락률과 품질 지표를 계산합니다.
5. 누락/품질 기준을 통과한 epoch만 학습에 사용하거나, mask feature를 추가합니다.

`data_100Hz`에서는 `Sleep_Stage`가 100Hz row마다 반복되어 있습니다. 다만 파일 row 0이 실제 30초 sleep-stage boundary와 맞지 않습니다. 따라서 `data_100Hz` loader는 row 0부터 3,000행씩 자르지 않고, `Sleep_Stage` 값이 바뀌는 run-length segment를 기준으로 epoch를 만듭니다.

- `P` run은 pre-recording/placeholder로 보고 제외
- `W`, `N1`, `N2`, `N3`, `R`만 5-class label로 사용
- `R`은 `REM`으로 매핑
- 3,000 rows보다 짧은 partial run은 기본 제외
- 3,000 rows의 배수인 run은 3,000 rows씩 나눠 같은 label epoch로 사용

라벨은 AASM 기준 5-class로 통일합니다.

| Canonical | 허용 alias 예시 |
| --- | --- |
| Wake | `W`, `Wake`, `WAKE`, `0` |
| N1 | `N1`, `S1`, `Stage 1`, `1` |
| N2 | `N2`, `S2`, `Stage 2`, `2` |
| N3 | `N3`, `N4`, `S3`, `S4`, `Deep`, `3`, `4` |
| REM | `R`, `REM`, `Stage R`, `5` |

N4가 남아 있는 데이터는 N3로 병합합니다. Movement/Unknown/Artifact는 학습에서 제외하거나 별도 ignore label로 둡니다.

## 5-class와 4-class 평가

모델은 항상 5-class logit을 출력합니다.

5-class:

- Wake
- N1
- N2
- N3
- REM

4-class 평가는 예측과 정답을 모두 아래처럼 병합해 계산합니다.

- Wake = Wake
- Light = N1 + N2
- Deep = N3
- REM = REM

보고 지표:

- Accuracy
- Macro F1
- Cohen's Kappa
- Confusion matrix
- Class-wise precision/recall/F1

## Feature 설계 초안

### PPG feature

앱 계산 가능:

- PPG amplitude statistics: mean, std, median, IQR, min/max, slope
- PPG quality: clipping ratio, missing ratio, flatline ratio, high-frequency noise proxy
- BVP: IR/RED PPG에서 band-pass/filtering 후 pulse waveform proxy로 사용
- Heart rate feature: peak 기반 HR mean/std/min/max, valid beat ratio
- IBI feature: beat-to-beat interval mean/std, SDNN, RMSSD, pNN50
- HRV frequency feature: LF/HF는 30초 단독 epoch에서는 불안정하므로 multi-epoch window에서만 사용
- Dual PPG feature: IR/RED correlation, normalized AC/DC ratio

주의:

- SpO2는 RED/IR와 calibration이 있어야 신뢰할 수 있으므로 `app_derived`로 두고 별도 검증 전에는 core feature에서 제외합니다.
- HRV frequency-domain feature는 30초 단독 epoch에서는 불안정합니다. 주변 context window 또는 sequence model 입력에서 보조 feature로 사용합니다.

### ACC feature

앱 계산 가능:

- 축별 mean/std/energy
- vector magnitude mean/std
- activity count
- posture proxy: 축별 gravity component, angle summary
- movement burst count

### TEMP feature

앱 계산 가능:

- mean/std/min/max
- local slope
- deviation from session baseline
- short-term stability

### DreamT 전용 feature

EEG/EOG/EMG/ECG 등 앱 raw 입력에 없는 채널에서 직접 계산되는 feature는 앱 탑재 모델의 입력에서 제외합니다.

## 모델 입력 형태

초기 구현은 feature sequence 기반으로 시작합니다.

- epoch feature vector: `[num_epochs, num_features]`
- model input: 주변 context를 포함한 `[batch, context_epochs, num_features]`
- target: center epoch의 5-class label

권장 context:

- 실시간 알람 앱 기준: 과거 context only
- 초기 실험: 5분/10분/30분 과거 context ablation
- offline upper-bound: 양방향 context를 별도 실험으로만 사용

초기 모델:

- CNN1D over epoch feature sequence
- LSTM/GRU over epoch feature sequence
- 이후 raw PPG/ACC branch + feature branch hybrid로 확장

## SSE 알고리즘 런타임 구조

앱 탑재용 Sleep State Estimation(SSE)은 두 단계로 분리합니다.

1. `SleepStageModel`: 30초 epoch마다 5-class 확률 `[W, N1, N2, N3, REM]` 출력
2. `AlarmDecisionPolicy`: 최근 확률 흐름을 보고 스마트 알람 트리거 여부 결정

이렇게 분리하는 이유는 모델 학습/평가와 제품 정책을 독립적으로 조정하기 위해서입니다. 예를 들어 모델은 5-class Macro F1과 Kappa 기준으로 선택하고, 알람 정책은 사용자 경험 기준으로 threshold를 별도 튜닝합니다.

### Hybrid CNN/LSTM 후보 구조

초기 앱 후보 모델은 다음 구조를 권장합니다.

- raw branch: PPG/ACC/TEMP epoch waveform 또는 downsampled segment를 CNN으로 인코딩
- feature branch: BVP, HR, SpO2 proxy, IBI std, LF/HF, 움직임, 체온 feature sequence 입력
- temporal branch: CNN/feature embedding을 causal LSTM/GRU에 입력
- output head: center/latest epoch의 5-class logit 출력

실시간 앱에서는 미래 epoch를 볼 수 없으므로 temporal branch는 causal 설정을 기본값으로 둡니다. 양방향 LSTM은 offline upper-bound 실험으로만 사용합니다.

### 스마트 알람 정책 초안

모델 출력은 최근 3 epoch 이동평균으로 smoothing합니다.

- Wake smoothing 확률이 임계치 이상이면 즉시 알람 트리거
- N3/Deep smoothing 확률이 높으면 알람 회피
- N1 또는 N2가 우세한 Light sleep 구간이면 알람 후보
- N1 확률이 N2 확률에 가중치를 더한 값보다 커지는 교차 시점을 좋은 기상 후보로 본다
- N1 gradient가 양수이고 N2 gradient가 음수이면 Light sleep의 얕아지는 흐름으로 가산한다

정책 파라미터는 처음에는 보수적으로 둡니다.

- smoothing window: 3 epoch, 90초
- Wake trigger threshold: validation에서 false wake trigger를 보며 조정
- Deep avoid threshold: validation에서 N3 recall 저하를 막는 방향으로 조정
- N1/N2 crossing weight: 사용자별 민감도와 알람 허용 window에 따라 조정

알람 정책 평가는 수면 단계 분류 성능과 별도로 봅니다.

- Deep 상태 알람 회피율
- 허용 알람 window 내 트리거 성공률
- Wake/N1/N2에서의 트리거 비율
- 트리거 지연 시간
- 사용자별 threshold sensitivity

## 데이터 분할

반드시 subject-wise split을 사용합니다.

- train/validation/test subject가 겹치면 안 됩니다.
- class imbalance가 크므로 subject-level stratification을 시도합니다.
- 같은 subject의 epoch가 train과 test에 동시에 들어가는 leakage를 금지합니다.

## 구현 순서

1. `inspect_dreamt`로 파일 구조, 컬럼명, 라벨 alias, sampling rate 후보 확인
2. 실제 DreamT 구조에 맞는 `column_map.json` 작성
3. subject/session loader 구현
4. 30초 epoch slicer 구현
5. 앱 계산 가능 feature extractor 구현
6. feature matrix와 label 저장
7. subject-wise split
8. 5-class 학습
9. 5-class 및 4-class 평가 리포트 출력
