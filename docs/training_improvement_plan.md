# Training Improvement Plan

현재 앱/4-class 관점의 새 best 후보는 `temporal w20 context20 h64 inverse`다. Causal smoothing은 후처리 진단으로 미루고, 우선 모델 자체의 5-class/4-class 성능을 올리는 실험을 진행한다.

최신 3-seed 평균:

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

현재 결론:

```text
기본 후보: temporal w20 context20 h64 inverse
개선 목표: N3/4-class 이득을 유지하면서 REM F1 하락을 줄이기
```

## 2026-07-15 방향 전환

팀 회의 결과, 연산량에 상관하지 않고 fusion을 계속 가져가도 되며 성능 향상에 집중하기로 했다. 따라서 single-model 개선 실험은 후순위로 두고, fixed fusion을 기준으로 더 공격적인 fusion 탐색을 진행한다.

## 2026-07-15 추가: 5-stage one-vs-rest specialist bank

REM/N3 specialist만 붙이면 Wake/N1/N2의 낮은 정답률은 그대로 남을 수 있으므로, 5개 stage 전체에 대해 one-vs-rest specialist를 학습하고 validation 기반 fusion으로 최종 stage를 결정하는 실험을 추가한다.

구현:

```text
src/sse_sleep/train_binary_specialist.py
src/sse_sleep/evaluate_specialist_fusion.py
scripts/run_ovr_specialist_fusion_colab.sh
```

실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_ovr_specialist_fusion_colab.sh
```

3-seed:

```bash
!SEEDS="42 7 123" bash scripts/run_ovr_specialist_fusion_colab.sh
```

평가 후보:

```text
specialist_raw_prob_argmax
specialist_raw_logit_argmax
specialist_platt_prob_argmax
meta_specialists_lr_none / balanced
meta_specialists_plus_base_lr_none / balanced
```

`meta_specialists_plus_base`는 original temporal, full w20, fixed fusion 확률까지 같이 넣는 logistic meta-fusion이다. raw argmax는 calibration 위험을 보는 baseline이고, 실제 후보는 calibrated/meta-fusion이 fixed fusion보다 4-class Macro/Kappa 및 REM/N3 균형을 개선하는지로 판단한다.

주의:

```text
Platt calibration과 logistic meta-fusion은 train split에 fit하고 validation split으로 선택한다.
validation에 fit하고 같은 validation으로 선택하면 점수가 과대평가되므로 사용하지 않는다.
```

다음 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_performance_fusion_colab.sh
```

실행 내용:

```text
1. original temporal + full w20 dense 2-model class-wise fusion
2. original temporal + full w20 + remaux_w05 3-model class-wise fusion
```

다른 세 번째 모델을 넣는 예:

```bash
!THIRD_VARIANT="remaux_w05_sel4combo" bash scripts/run_performance_fusion_colab.sh
!THIRD_MODEL_PREFIX="lstm_temporal_w20_context20_inverse_capacity_h96" bash scripts/run_performance_fusion_colab.sh
```

## 이전 기준 모델

```bash
PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

3-seed 기준으로 original temporal LSTM 1.0x는 안정성이 GRU보다 낫고, N3 1.2x는 N3 F1을 올리지만 Wake/REM/Kappa 손실이 있었다. 이후 temporal long-window 실험에서 w20이 앱/4-class 관점 새 best 후보가 되었다.

## 현재 기준 모델

```bash
PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_w20_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_w20_context20_h64_inverse" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

이 기준 위에서 이미 확인한 결과:

```text
w20 + focal_g2: 탈락
w20 + focal_g15: 3-seed에서 w20 단독보다 N3/REM 하락
w20 + selection 4combo: seed42에서 w20과 동일
w20 + n3_weight 1.2: 3-seed에서 w20 단독보다 하락
```

## 새 학습 옵션

`train_lstm.py`에 아래 옵션을 추가했다. 기본값은 기존 실험 재현과 같다.

- `--selection-metric`: best checkpoint와 early stopping 기준
  - `5_macro_f1` 기본값
  - `4_macro_f1`
  - `5_kappa`
  - `4_kappa`
  - `5_macro_f1_plus_4_kappa`
  - `4_macro_f1_plus_4_kappa`
- `--loss-type cross_entropy|focal`
- `--focal-gamma`
- `--label-smoothing`
- `--train-sampler none|weighted`
- `--aux-head none|deep`
- `--aux-weight`
- `--aux-deep-pos-weight-mode balanced|none`

## 1차 후보 실험

빠른 후보 필터링:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_learning_improvement_colab.sh
```

처음에는 seed42만 돌린다. 유망한 후보가 나오면:

```bash
!SEEDS="42 7 123" bash scripts/run_learning_improvement_colab.sh
```

스크립트가 돌리는 후보:

- `inverse_select4combo`: 학습은 그대로 두고 4-class Macro F1 + 4-class Kappa 기준으로 checkpoint 선택
- `inverse_focal_g1`: inverse weight + focal loss gamma 1.0
- `inverse_focal_g2`: inverse weight + focal loss gamma 2.0
- `inverse_ls005`: inverse weight + label smoothing 0.05
- `weighted_sampler_none_weight`: class weight 없이 minority class oversampling
- `weighted_sampler_sqrt_weight`: sqrt weight + minority class oversampling

## Focal gamma 1.5 후속 실험

`gamma=1.0`과 `gamma=2.0` 사이의 균형점을 확인하기 위해 `gamma=1.5`를 3-seed로 따로 검증한다.

```bash
%cd /content/SSE
!git pull
!bash scripts/run_focal_g15_colab.sh
```

출력 폴더:

```text
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_focal_g15_seed42
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_focal_g15_seed7
/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_focal_g15_seed123
```

## Deep/N3 auxiliary head 실험

5-class stage head와 별도로 `N3 vs non-N3` binary head를 붙여 encoder가 Deep/N3 구분을 더 명시적으로 배우게 한다.

구조:

```text
shared LSTM encoder
  -> 5-class stage head
  -> Deep/N3 binary auxiliary head
```

loss:

```text
total_loss = stage_loss + aux_weight * deep_binary_loss
```

빠른 seed42 sweep:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_deep_aux_colab.sh
```

기본 aux weight 후보:

```text
0.2
0.5
1.0
```

유망한 weight만 3-seed로 확장:

```bash
!SEEDS="42 7 123" AUX_WEIGHTS="0.5" bash scripts/run_deep_aux_colab.sh
```

`lstm_metrics.json`에는 기존 5-class/4-class 지표와 함께 아래 Deep binary 지표가 저장된다.

```text
final_test.deep_binary_from_stage_metrics
final_test.deep_binary_aux_metrics
```

## Temporal long-window feature 실험

새 feature 종류를 만들기 전에, 기존 rolling/delta feature의 window만 길게 늘려 N3에 필요한 느린 흐름을 더 잘 잡는지 확인한다.

기본 비교:

```text
w10: delta 1/3/10, rolling 3/5/10
w20: delta 1/3/20, rolling 3/5/20
```

30초 epoch 기준:

```text
10 epoch = 5분
20 epoch = 10분
```

seed42에서 w10/w20 빠른 비교:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_temporal_long_window_colab.sh
```

출력 폴더:

```text
/content/drive/MyDrive/SSE_outputs/lstm_temporal_w10_context20_h64_inverse
/content/drive/MyDrive/SSE_outputs/lstm_temporal_w20_context20_h64_inverse
```

유망한 window만 3-seed로 확장:

```bash
!SEEDS="42 7 123" VARIANTS="10" bash scripts/run_temporal_long_window_colab.sh
```

`add_temporal_features.py`는 기존 comma 형식과 Colab에서 쓰기 편한 space-separated alias를 모두 지원한다.

```bash
--delta-lags 1,3,10
--rolling-windows 3,5,10

--delta-windows 1 3 10
--rolling-window-list 3 5 10
```

결론:

```text
w10: 탈락
w20: 새 best 후보
w30: 탈락
```

## 다음 feature 개선 계획

다음 목표는 full w20의 장점인 4-class/N3 개선을 유지하면서 REM F1 하락을 줄이는 것이다.

### 0. Prediction fusion 진단

새 학습 전에 original temporal과 full w20의 prediction probability를 섞어 두 모델이 상보적인지 확인한다.

```text
alpha = 0.0: original temporal only
alpha = 1.0: full w20 only
```

Colab seed42 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_prediction_fusion_colab.sh
```

class-wise fusion도 같은 스크립트에서 함께 평가한다. 기본 설정은 non-REM class에는 full w20 비중을 높이고, REM에는 original temporal 비중을 더 줄 수 있는 grid를 돈다.

유망하면 3-seed로 확장:

```bash
!SEEDS="42 7 123" bash scripts/run_prediction_fusion_colab.sh
```

판단 기준:

```text
validation 기준으로 alpha/class-wise weight를 고른다.
test는 선택된 weight의 일반화 확인용으로만 본다.
full w20 대비 4-class Macro/Kappa와 N3를 유지하면서 REM이 회복되면 다음은 single-model distillation 또는 REM-preserving 학습으로 간다.
```

Seed42 결과:

```text
best by validation: classwise_nonrem1.00_rem0.60
test: 4 Macro 0.4106, 4 Kappa 0.2515, Wake 0.5025, N3 0.1040, REM 0.3846
```

판단:

```text
REM과 4-class는 full w20보다 개선되고, N3는 full w20보다는 낮지만 original temporal보다는 높다.
3-seed 확장 가치가 있다.
```

3-seed 결과, 각 seed별 validation-selected top1 기준:

```text
seed  selected                         4 Macro  4 Kappa  Wake    N3      REM
42    classwise_nonrem1.00_rem0.60     0.4106   0.2515   0.5025  0.1040  0.3846
7     classwise_nonrem0.80_rem0.20     0.4081   0.2290   0.4606  0.1647  0.3641
123   classwise_nonrem1.00_rem0.30     0.3928   0.2377   0.5463  0.0722  0.3476
mean                                  0.4038   0.2394   0.5031  0.1136  0.3654
```

판단:

```text
앱/4-class 관점에서는 full w20보다 낫다.
REM은 original temporal 평균 0.3650 수준까지 회복된다.
N3는 full w20보다 낮아지지만 original temporal보다 높게 유지된다.
`scripts/run_prediction_fusion_colab.sh`는 여러 seed 실행 시 고정 weight별 평균 요약을 자동 출력하고 `/content/drive/MyDrive/SSE_outputs/fusion_original_temporal_full_w20_context20_h64_summary.json`에 저장한다.
```

고정 weight 3-seed 평균 기준 best:

```text
classwise_nonrem0.90_rem0.20

4 Macro 0.4074
4 Kappa 0.2458
Wake    0.5034
N3      0.1220
REM     0.3722
```

full w20 대비:

```text
4 Macro +0.0073
4 Kappa +0.0093
Wake    +0.0023
N3      -0.0014
REM     +0.0289
```

결론:

```text
앱/4-class 관점 새 best 후보는 fixed fusion classwise_nonrem0.90_rem0.20이다.
다만 2-model ensemble이므로 비용/지연이 있다. 다음 full w20 후속 학습 후보와 w15에서 single-model 대안을 찾는다.
```

### 0.25. Fusion distillation

full w20 후속 학습 후보와 w15가 seed42에서 탈락했으므로, fixed fusion teacher를 full w20 단일 student로 distill한다.

teacher:

```text
fixed fusion classwise_nonrem0.90_rem0.20
Wake/N1/N2/N3: 90% full w20 + 10% original temporal
REM:           20% full w20 + 80% original temporal
```

학습:

```text
loss = hard cross entropy + distill_weight * KL(teacher_probs || student_probs)
```

seed42 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_distillation_colab.sh
```

후보:

```text
distill_w02
distill_w05
distill_w10
```

유망한 후보만 3-seed 확장:

```bash
!SEEDS="42 7 123" VARIANTS="distill_w05" bash scripts/run_distillation_colab.sh
```

판단 기준:

```text
full w20 대비 REM 또는 4-class Macro/Kappa를 개선하면서 N3를 유지하면 확장한다.
fixed fusion에 가까워질수록 좋지만, single-model이므로 fixed fusion보다 약간 낮아도 앱 비용 측면에서 후보가 될 수 있다.
```

Seed42 결과:

```text
variant       5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
distill_w02   0.3188   0.1933   0.3886   0.2187   0.4608  0.0919  0.3619
distill_w05   0.2995   0.1829   0.3657   0.2105   0.4447  0.0000  0.3664
distill_w10   0.3105   0.1916   0.3780   0.2225   0.4459  0.0166  0.3841
```

판단:

```text
distill_w02가 가장 균형은 낫지만 full w20/fixed fusion을 넘지 못한다.
distill_w05/w10은 REM은 회복하지만 N3가 붕괴한다.
distillation 1차 후보는 3-seed 확장하지 않는다.
```

다음 방향:

```text
1. fixed fusion classwise_nonrem0.90_rem0.20의 앱 배포 비용/지연 검토
2. validation 기반 REM calibration / threshold
3. soft KL distillation 대신 teacher hard-label pseudo-label 학습
4. REM binary auxiliary head
5. N3 보존 + REM 보정 multi-task loss
6. 그래도 single-model 대안이 없으면 h96/h128 또는 2-layer 등 모델 capacity 실험
```

fusion을 제외한 full w20 단일 모델 개선 후보:

```text
1. validation 기반 REM calibration / threshold
2. teacher hard-label pseudo-label 학습
3. REM binary auxiliary head
4. N3 보존 + REM 보정 multi-task loss
5. 모델 capacity 조정: h96/h128, 2-layer
```

추가 평가 후보:

```text
4. fixed fusion 및 full w20에 causal post-processing 재평가
```

우선 구현 후보:

```text
pseudo_w02: ground truth CE + 0.2 * teacher hard CE
pseudo_w05: ground truth CE + 0.5 * teacher hard CE
pseudo_rem_only: REM target에만 teacher hard label 보조 loss
```

구현:

```text
src/sse_sleep/train_lstm.py
  --teacher-hard-weight
  --teacher-hard-mode all|rem_only

scripts/run_pseudo_label_colab.sh
```

seed42 결과:

```text
variant          5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
pseudo_w02       0.3091   0.1900   0.3796   0.2224   0.4802  0.0331  0.3556
pseudo_w05       0.3060   0.1878   0.3797   0.2302   0.4781  0.0127  0.3639
pseudo_rem_only  0.2894   0.1556   0.3619   0.1897   0.4281  0.0449  0.3429
```

결론:

```text
REM을 조금 회복하는 후보도 있지만 N3가 크게 붕괴한다.
3-seed 확장하지 않고 중단한다.
다음은 validation 기반 REM threshold/calibration을 학습 없이 평가한다.
```

구현:

```text
src/sse_sleep/evaluate_rem_threshold.py
scripts/run_rem_threshold_colab.sh
```

full w20 seed42 결과:

```text
validation-selected: argmax_baseline

variant             val score  4 Macro  4 Kappa  Wake    N3      REM
argmax_baseline     0.6988     0.4036   0.2401   0.5011  0.1043  0.3646
rem_threshold0.40   0.6985     0.4031   0.2393   0.5011  0.1043  0.3638
rem_threshold0.35   0.6959     0.4018   0.2379   0.5011  0.1043  0.3618
rem_threshold0.30   0.6885     0.4010   0.2377   0.5014  0.1044  0.3627
rem_threshold0.25   0.6776     0.3988   0.2347   0.5017  0.1047  0.3629
rem_threshold0.20   0.6628     0.3924   0.2272   0.5012  0.1014  0.3558
```

결론:

```text
full w20에서는 threshold lowering이 REM을 개선하지 못하고 4-class/Kappa만 낮춘다.
validation도 argmax baseline을 선택하므로 full w20 REM threshold는 중단한다.
다음은 REM binary auxiliary head 또는 N3 보존 + REM 보정 multi-task loss를 확인한다.
```

### 0.45. REM auxiliary / Deep+REM multi-task

REM threshold가 실패했으므로 학습 단계에서 REM-vs-rest auxiliary head를 붙인다. N3 보존을 같이 보기 위해 N3-vs-rest와 REM-vs-rest head를 동시에 두는 `deep_rem` multi-task도 확인한다.

구현:

```text
src/sse_sleep/train_lstm.py
  --aux-head rem
  --aux-head deep_rem

scripts/run_rem_aux_colab.sh
```

seed42 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_rem_aux_colab.sh
```

기본 후보:

```text
remaux_w02
remaux_w05
deeprem_w02
deeprem_w05
```

판정:

```text
full w20 seed42 baseline:
4 Macro 0.4036 / 4 Kappa 0.2401 / Wake 0.5011 / N3 0.1043 / REM 0.3646

REM이 baseline 이상으로 오르고 N3가 original temporal 평균 0.0833 이상이면 3-seed 확장 후보.
REM만 오르고 N3가 0.05 이하로 무너지면 중단.
4-class Macro/Kappa가 baseline보다 크게 낮으면 앱 후보로는 탈락.
```

seed42 일부 결과:

```text
variant      5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
deeprem_w05  0.3069   0.1877   0.3781   0.2338   0.4525  0.0102  0.3529
deeprem_w02  0.3128   0.1788   0.3864   0.2146   0.4745  0.0932  0.3162
remaux_w05   0.3286   0.2227   0.4008   0.2582   0.5398  0.0678  0.3568
```

결론:

```text
deeprem_w05는 N3 붕괴로 탈락.
deeprem_w02는 N3는 보존하지만 REM/4-class가 낮아 탈락.
remaux_w05는 4 Kappa/Wake가 좋지만 N3가 original temporal 평균 0.0833 아래이고 REM도 full w20 seed42 0.3646보다 낮아 3-seed 확장 보류.
```

다음 실행:

```bash
!VARIANTS="remaux_w005 remaux_w01 remaux_w02 deeprem_w005 deeprem_w01" RUN_TAG="low_aux" bash scripts/run_rem_aux_colab.sh
!VARIANTS="remaux_w005 remaux_w01 remaux_w02 remaux_w05" SELECTION_METRIC="4_macro_f1_plus_4_kappa" RUN_TAG="sel4combo" bash scripts/run_rem_aux_colab.sh
```

low_aux/sel4combo 추가 seed42 결과:

```text
variant                 5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
remaux_w05              0.3286   0.2227   0.4008   0.2582   0.5398  0.0678  0.3568
remaux_w05_sel4combo    0.3286   0.2227   0.4008   0.2582   0.5398  0.0678  0.3568
remaux_w02_sel4combo    0.2860   0.1558   0.3645   0.2102   0.4630  0.0121  0.2953
remaux_w01_sel4combo    0.2782   0.1395   0.3518   0.1837   0.4359  0.0330  0.2531
remaux_w005_sel4combo   0.3088   0.1583   0.3959   0.2181   0.4732  0.1329  0.3046
deeprem_w01_low_aux     0.2856   0.1534   0.3568   0.1904   0.4458  0.0203  0.2919
remaux_w02_low_aux      0.3147   0.2177   0.3816   0.2442   0.5299  0.0283  0.3493
remaux_w01_low_aux      0.2987   0.1422   0.3837   0.2068   0.4404  0.1095  0.2843
remaux_w005_low_aux     0.2991   0.1807   0.3829   0.2400   0.4893  0.0295  0.3506
```

결론:

```text
REM auxiliary / deep_rem multi-task 계열은 3-seed 확장하지 않는다.
remaux_w05는 4 Kappa/Wake만 개선되고 N3/REM이 baseline보다 낮다.
remaux_w005_sel4combo와 remaux_w01_low_aux는 N3를 살리지만 REM/4-class가 낮다.
다음은 single-model capacity 조정을 seed42로 확인한다.
```

### 0.5. Full w20 후속 학습 후보

Fusion에서 상보성이 보이면 full w20 단일 모델 쪽 후속 후보를 seed42로 확인한다.

```bash
!bash scripts/run_full_w20_next_training_colab.sh
```

기본 후보:

```text
ls005: label smoothing 0.05
remx11: REM class weight multiplier 1.1
longdrop_p10: *_20 long-window feature train-time dropout p=0.10
```

유망한 후보만 3-seed 확장:

```bash
!SEEDS="42 7 123" VARIANTS="ls005" bash scripts/run_full_w20_next_training_colab.sh
```

`train_lstm.py`에 추가된 옵션:

```text
--rem-weight-multiplier
--feature-dropout-pattern
--feature-dropout-prob
```

Seed42 결과:

```text
variant       5 Macro  5 Kappa  4 Macro  4 Kappa  Wake    N3      REM
ls005         0.3026   0.1659   0.3765   0.1985   0.4276  0.0736  0.3269
remx11        0.3175   0.2037   0.3877   0.2364   0.4933  0.0417  0.3720
longdrop_p10  0.2945   0.1809   0.3687   0.2251   0.4854  0.0056  0.3332
```

판단:

```text
세 후보 모두 seed42 탈락.
remx11은 REM을 회복하지만 N3가 크게 무너진다.
ls005와 longdrop_p10은 4-class/REM/N3 균형이 모두 fixed fusion 및 full w20보다 약하다.
```

### 0.55. Full w20 capacity 후보

feature/loss/threshold/distillation/pseudo-label/auxiliary 계열이 single-model 대안을 만들지 못했으므로, 계산 비용을 감수하고 capacity 조정을 seed42로 확인한다.

구현:

```text
scripts/run_full_w20_capacity_colab.sh
```

seed42 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_full_w20_capacity_colab.sh
```

기본 후보:

```text
h96:        hidden 96,  1 layer, dropout 0.4
h128:       hidden 128, 1 layer, dropout 0.4
layers2_h64 hidden 64,  2 layers, dropout 0.3
```

### 0.75. w15 중간 long-window

w10은 탈락, w20은 현재 best, w30은 탈락했으므로 중간점인 w15를 seed42로 확인한다.

```bash
!VARIANTS="15" bash scripts/run_temporal_long_window_colab.sh
```

Seed42 결과:

```text
w15: 5 Macro 0.2979, 5 Kappa 0.1689, 4 Macro 0.3652, 4 Kappa 0.1923, Wake 0.4511, N3 0.0458, REM 0.2999
```

판단:

```text
w15도 seed42 탈락.
```

### 1. Targeted slow w20

기존 short temporal feature는 유지하고, long-window 20 feature는 N3와 관련성이 큰 slow physiology/movement feature에만 추가한다.

추천 대상:

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

Colab seed42 실행:

```bash
bash scripts/run_targeted_slow_w20_colab.sh
```

seed42에서 좋으면 3-seed로 확장:

```bash
SEEDS="42 7 123" bash scripts/run_targeted_slow_w20_colab.sh
```

narrow ablation까지 함께 비교:

```bash
VARIANTS="targeted_w20 movement_only_w20 cardio_temp_w20" bash scripts/run_targeted_slow_w20_colab.sh
```

판단 기준:

```text
full w20 대비 REM F1 회복
full w20 대비 4-class Macro F1/Kappa 유지 또는 개선
N3 F1은 full w20 평균 0.1234 근처 유지
```

결과 JSON만 따로 비교할 때:

```bash
PYTHONPATH=src python -m sse_sleep.summarize_lstm_metrics \
  --metrics full_w20=/content/drive/MyDrive/SSE_outputs/lstm_temporal_w20_context20_h64_inverse/lstm_metrics.json \
  --metrics targeted_w20=/content/drive/MyDrive/SSE_outputs/lstm_temporal_targeted_w20_context20_h64_inverse/lstm_metrics.json \
  --baseline-label full_w20
```

### 2. Narrow ablation

targeted slow w20이 애매하면 long-window target을 더 좁혀 비교한다.

```text
movement_only_w20: acc_vm_mean, acc_vm_activity
cardio_temp_w20: hr_mean, hr_std, ibi_mean, ibi_std, temp_mean, temp_slope
cardio_temp_acc_activity_w20: cardio_temp_w20 + acc_vm_activity
cardio_temp_acc_mean_w20: cardio_temp_w20 + acc_vm_mean
```

`cardio_temp_w20` 3-seed에서 REM은 조금 회복됐지만 N3/4-class가 하락하면, ACC long-window를 하나씩만 다시 더해 N3를 살리면서 REM 손실을 줄일 수 있는지 확인한다.

```bash
VARIANTS="cardio_temp_acc_activity_w20 cardio_temp_acc_mean_w20" bash scripts/run_targeted_slow_w20_colab.sh
```

### 3. Causal per-night baseline feature

long-window ablation 이후에는 subject/night baseline 차이를 causal하게 보정하는 feature를 검토한다.

```text
feature_expanding_mean
feature_expanding_std
feature_causal_zscore
```

대상:

```text
acc_vm_activity
hr_mean
ibi_mean
temp_mean
bvp_std
```

이 feature는 앱에서도 online으로 계산 가능해야 하며, 미래 epoch나 test subject 통계 leakage를 쓰면 안 된다.

1차 실험은 original short temporal CSV 위에 prior-only expanding baseline feature를 추가한다.

```bash
bash scripts/run_causal_baseline_colab.sh
```

seed42에서 original temporal 대비 4-class Macro F1/Kappa가 유지 또는 개선되고, N3 F1이 full w20 쪽으로 회복되면서 REM F1이 크게 무너지지 않으면 3-seed로 확장한다.

```bash
SEEDS="42 7 123" bash scripts/run_causal_baseline_colab.sh
```

REM은 좋아졌지만 Wake/Kappa가 약하면 cardio-only baseline을 먼저 확인한다.

```bash
VARIANTS="cardio_baseline" bash scripts/run_causal_baseline_colab.sh
```

cardio-only가 N3는 살리지만 REM/Wake/Kappa를 깎으면 temp baseline을 추가해 균형 회복을 확인한다.

```bash
VARIANTS="cardio_temp_baseline" bash scripts/run_causal_baseline_colab.sh
```

`temporal_baseline`의 REM 개선이 `bvp_std` baseline 때문인지 보기 전에, `bvp_std`만 제외한 movement/cardio/temp 조합도 비교한다.

```bash
VARIANTS="movement_cardio_temp_baseline" bash scripts/run_causal_baseline_colab.sh
```

## 판단 기준

1. 3-seed 평균에서 5-class Macro F1 또는 4-class Macro F1이 baseline보다 올라야 한다.
2. 4-class Kappa가 baseline보다 크게 낮아지면 앱 기본 모델로 채택하지 않는다.
3. N3 F1이 좋아져도 Wake/REM F1이 크게 무너지면 Deep-aware ablation으로만 둔다.
4. seed42 하나에서만 좋아진 후보는 채택하지 않는다.

우선은 `inverse_focal_g1`, `inverse_ls005`, `weighted_sampler_sqrt_weight`가 가장 확인 가치가 높다. `weighted_sampler_none_weight`는 N3/N1 recall 진단용 성격이 강하다.
