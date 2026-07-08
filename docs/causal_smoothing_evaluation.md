# Causal Smoothing Evaluation

DreamT temporal LSTM/GRU 모델의 raw epoch prediction을 앱 런타임 조건에 맞게 causal post-processing으로 평가한다.

## 목적

- raw 30초 epoch prediction 대비 5-class/4-class Macro F1, Kappa 변화를 확인한다.
- subject 경계와 epoch gap을 넘지 않고 최근 과거 epoch만 사용한다.
- 스마트 알람 정책에서 쓸 안정적인 state/probability 후보를 고른다.

## 필요한 prediction 파일

`train_lstm.py`는 재학습 후 `lstm_predictions.npz`에 다음을 저장한다.

- `val_y_true`, `val_y_pred`, `val_logits`, `val_probs`
- `test_y_true`, `test_y_pred`, `test_logits`, `test_probs`
- `val_subject_ids`, `val_epoch_indices`
- `test_subject_ids`, `test_epoch_indices`

기존 prediction 파일에 확률이나 subject/epoch 메타데이터가 없으면, 같은 설정으로 모델을 다시 학습해 prediction 파일을 재생성한다.

기본 모델 재학습 예시:

```bash
PYTHONPATH=src python -m sse_sleep.train_lstm \
  --npz "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_temporal_lstm_context20.npz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse" \
  --hidden-size 64 \
  --dropout 0.4 \
  --class-weight-mode inverse
```

## 평가 방법

`evaluate_causal_smoothing.py`는 아래 후보를 한 번에 평가한다.

- `raw`
- `majority_vote_3`
- `majority_vote_5`
- `probability_ma_3`
- `probability_ma_5`
- `transition_guard_n3_rem_2`

규칙:

- majority vote tie는 가장 최근 tied class로 결정한다.
- probability moving average는 현재 epoch와 과거 window 안의 확률 평균을 argmax한다.
- transition guard는 N3/REM 진입을 raw prediction이 2 epoch 이상 지속될 때까지 지연한다.
- subject가 바뀌거나 `epoch_index`가 연속되지 않으면 smoothing history를 reset한다.

단일 seed 평가:

```bash
PYTHONPATH=src python -m sse_sleep.evaluate_causal_smoothing \
  --predictions "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse/lstm_predictions.npz" \
  --out-json "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse/causal_smoothing_metrics.json" \
  --splits val test
```

3-seed 평균 평가:

```bash
PYTHONPATH=src python -m sse_sleep.evaluate_causal_smoothing \
  --predictions \
    "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse/lstm_predictions.npz" \
    "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_seed7/lstm_predictions.npz" \
    "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_seed123/lstm_predictions.npz" \
  --out-json "/content/drive/MyDrive/SSE_outputs/lstm_temporal_context20_h64_inverse_causal_smoothing_3seed.json" \
  --splits test
```

## 채택 기준

우선순위는 다음 순서로 본다.

1. 4-class Macro F1과 4-class Kappa가 raw보다 개선되는지
2. Wake/REM/N3 class F1이 과도하게 무너지지 않는지
3. `prediction_change_rate_vs_raw`가 너무 높아 모델 출력을 과하게 덮어쓰지 않는지
4. seed별 결과가 한 seed에만 의존하지 않는지
