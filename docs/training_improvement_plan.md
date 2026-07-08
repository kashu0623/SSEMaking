# Training Improvement Plan

현재 기본 모델은 `LSTM temporal context20 h64 inverse 1.0x`다. Causal smoothing은 후처리 진단으로 미루고, 우선 모델 자체의 5-class/4-class 성능을 올리는 실험을 진행한다.

## 기준 모델

```bash
PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

3-seed 기준으로 LSTM 1.0x는 안정성이 GRU보다 낫고, N3 1.2x는 N3 F1을 올리지만 Wake/REM/Kappa 손실이 있었다. 다음 실험은 구조를 크게 바꾸기 전에 loss, sampler, checkpoint selection을 점검한다.

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

## 판단 기준

1. 3-seed 평균에서 5-class Macro F1 또는 4-class Macro F1이 baseline보다 올라야 한다.
2. 4-class Kappa가 baseline보다 크게 낮아지면 앱 기본 모델로 채택하지 않는다.
3. N3 F1이 좋아져도 Wake/REM F1이 크게 무너지면 Deep-aware ablation으로만 둔다.
4. seed42 하나에서만 좋아진 후보는 채택하지 않는다.

우선은 `inverse_focal_g1`, `inverse_ls005`, `weighted_sampler_sqrt_weight`가 가장 확인 가치가 높다. `weighted_sampler_none_weight`는 N3/N1 recall 진단용 성격이 강하다.
