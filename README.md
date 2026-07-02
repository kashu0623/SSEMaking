# SSE Sleep Stage Pipeline

DreamT 기반 수면 단계 예측 모델을 개발하기 위한 초기 데이터 로딩/전처리 설계입니다.

현재 목표:

- 30초 epoch 단위 입력 구성
- 5-class sleep staging: Wake, N1, N2, N3, REM
- 평가 시 4-class 병합 결과도 산출: Wake, Light(N1+N2), Deep(N3), REM
- 실제 웨어러블 앱 입력(`IR PPG`, `RED PPG`, `ACC_X`, `ACC_Y`, `ACC_Z`, `TEMP`)으로 계산 가능한 feature와 DreamT 전용 feature를 명시적으로 분리

현재 확인된 DreamT Drive 폴더는 `/content/drive/MyDrive/data_100Hz`이며, 구조 요약은 [docs/dreamt_100hz_profile.md](/Users/chan/Documents/SSE/docs/dreamt_100hz_profile.md)에 있습니다.

## 빠른 시작

DreamT 원본 데이터를 Google Drive에서 `data/dreamt/` 아래로 내려받거나, Google Drive Desktop 동기화 폴더를 `data/dreamt/`로 복사/심볼릭 링크한 뒤 구조를 먼저 확인합니다.

로컬 저장공간이 부족하면 원본 데이터를 내려받지 말고 [docs/cloud_data_workflow.md](/Users/chan/Documents/SSE/docs/cloud_data_workflow.md)의 Colab + Google Drive 방식을 사용합니다.

```bash
PYTHONPATH=src python3 -m sse_sleep.inspect_dreamt --root data/dreamt --out reports/dreamt_schema.json
```

Google Drive의 DreamT 폴더명이 `data_100Hz`라면 Colab에서는 보통 다음처럼 실행합니다. 같은 CSV의 사본 파일은 기본적으로 건너뜁니다.

```bash
PYTHONPATH=src python -m sse_sleep.inspect_dreamt \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out "/content/drive/MyDrive/dreamt_schema.json"
```

Stage probe JSON 요약:

```bash
PYTHONPATH=src python -m sse_sleep.summarize_stage_probe \
  --input "/content/drive/MyDrive/dreamt_stage_probe.json"
```

DreamT 100Hz 전처리 smoke test:

```bash
PYTHONPATH=src python -m sse_sleep.preprocess_dreamt_100hz \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out-dir "/content/drive/MyDrive/SSE_outputs" \
  --limit-files 1 \
  --max-rows 1200000
```

출력:

- `/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv`
- `/content/drive/MyDrive/SSE_outputs/dreamt_100hz_preprocess_summary.json`

학습용 NPZ 생성:

```bash
PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
  --input-csv "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_epoch_features.csv" \
  --out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_lstm_context10.npz" \
  --summary-out "/content/drive/MyDrive/SSE_outputs/dreamt_100hz_lstm_context10_summary.json" \
  --context-epochs 10
```

그 다음 [docs/dreamt_pipeline_design.md](/Users/chan/Documents/SSE/docs/dreamt_pipeline_design.md)의 컬럼 매핑 기준에 따라 실제 파일 구조에 맞는 loader를 확정합니다.

예시:

```bash
mkdir -p data
ln -s "/path/to/Google Drive/DreamT" data/dreamt
PYTHONPATH=src python3 -m sse_sleep.inspect_dreamt --root data/dreamt --out reports/dreamt_schema.json
```
