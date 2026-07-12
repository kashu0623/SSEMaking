# 로컬 저장공간 없이 DreamT 처리하기

로컬 저장공간이 부족하면 DreamT 원본을 `/Users/chan/Documents/SSE/data`에 내려받지 않고, Google Drive 또는 클라우드 스토리지에 둔 채로 학습/전처리를 실행합니다.

## 권장안: Google Colab + Google Drive

DreamT가 이미 Google Drive에 있다면 가장 간단한 방식입니다.

장점:

- 로컬 디스크를 거의 쓰지 않음
- Drive에 있는 원본 데이터를 바로 읽음
- GPU 런타임으로 학습 가능
- 지금 저장소의 `src/sse_sleep` 코드를 그대로 재사용 가능

Colab 셀 예시:

```python
from google.colab import drive
drive.mount("/content/drive")
```

저장소를 Colab 런타임에 가져옵니다.

```bash
cd /content
git clone <YOUR_REPO_URL> SSE
cd SSE
pip install -r requirements.txt
```

아직 원격 Git 저장소가 없다면, `src`, `docs`, `configs`, `requirements.txt`만 Drive나 Colab 파일 업로드로 복사해도 됩니다.

DreamT 경로를 지정해서 구조를 확인합니다. 사용자의 Drive에 `data_100Hz`라는 이름으로 저장되어 있다면 root는 보통 아래와 같습니다.

```bash
cd /content/SSE
PYTHONPATH=src python -m sse_sleep.inspect_dreamt \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out "/content/drive/MyDrive/dreamt_schema.json"
```

이 방식에서는 inspection 결과만 Drive에 저장하고, 원본 DreamT는 로컬 Mac에 복사하지 않습니다.

## CSV 사본 파일 처리

`data_100Hz` 안에 CSV마다 사본이 하나씩 있으면 같은 subject/session이 두 번 들어가서 데이터 누수와 class count 왜곡이 생길 수 있습니다.

이 저장소의 `inspect_dreamt`는 기본적으로 원본과 같은 위치에 있는 사본 후보를 건너뜁니다.

사본으로 간주하는 파일명 예시:

- `Copy of sample.csv`
- `sample - Copy.csv`
- `sample - 사본.csv`
- `sample의 사본.csv`
- `sample (1).csv`

원본이 있는 경우에만 사본 후보를 제외하고, 사본만 존재하는 파일은 버리지 않습니다. 정말 사본까지 포함해서 확인해야 하면 `--include-copy-files`를 붙입니다.

```bash
PYTHONPATH=src python -m sse_sleep.inspect_dreamt \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out "/content/drive/MyDrive/dreamt_schema_with_copies.json" \
  --include-copy-files
```

## Colab 오류 대응

### `Transport endpoint is not connected`

Google Drive mount가 끊겼을 때 나는 오류입니다. 런타임에서 Drive를 다시 mount한 뒤 재실행합니다.

```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount("/content/drive", force_remount=True)
```

경로가 다시 보이는지 확인합니다.

```python
!ls "/content/drive/MyDrive/data_100Hz" | head
```

그 다음 inspection을 다시 실행합니다.

```python
!PYTHONPATH=src python -m sse_sleep.inspect_dreamt \
  --root "/content/drive/MyDrive/data_100Hz" \
  --out "/content/drive/MyDrive/dreamt_schema.json"
```

최신 코드에서는 일부 파일만 일시적으로 접근 불가한 경우 전체 실행을 중단하지 않고 `inaccessible_files`에 기록합니다. `inaccessible_file_count`가 0이 아니면 Drive remount 후 다시 실행하는 편이 좋습니다.

### Colab 셀에서 `SyntaxError: invalid syntax`

`git clone`, `pip install`, `PYTHONPATH=...` 같은 명령은 Python 문법이 아니라 shell 명령입니다. Colab/Jupyter 셀에서는 앞에 `!`를 붙입니다.

```python
!git clone https://github.com/kashu0623/SSEMaking.git SSE
%cd SSE
!pip install -r requirements.txt
```

## 대안 1: Google Drive Desktop 스트리밍

Google Drive Desktop의 “스트리밍” 모드는 파일을 전부 로컬에 내려받지 않고 필요할 때 읽습니다.

주의:

- 실제 학습처럼 전체 데이터를 반복해서 읽으면 로컬 캐시가 커질 수 있습니다.
- 대용량 EDF/CSV를 랜덤 접근하면 속도가 느릴 수 있습니다.
- 저장공간이 아주 부족하면 Colab 방식이 더 안전합니다.

## 대안 2: 외장 SSD

DreamT 원본만 외장 SSD에 두고, 코드와 결과 요약만 로컬 저장소에 둡니다.

예시:

```bash
PYTHONPATH=src python3 -m sse_sleep.inspect_dreamt \
  --root "/Volumes/ExternalSSD/DreamT" \
  --out reports/dreamt_schema.json
```

## 대안 3: 클라우드 VM

Google Cloud, AWS, Azure 같은 VM에 데이터를 올리고 학습까지 수행합니다.

권장 상황:

- Colab 세션 시간이 부족함
- 대량 실험을 반복해야 함
- 학습 결과와 로그를 장기간 보관해야 함

주의:

- 비용 관리가 필요합니다.
- 원본 데이터 접근 권한과 개인정보/IRB 조건을 반드시 확인해야 합니다.

## 이 프로젝트의 권장 운영 방식

초기 단계:

1. Colab에서 Drive mount
2. `inspect_dreamt` 실행
3. `dreamt_schema.json`만 공유
4. 로컬에서는 schema를 보고 loader/feature 코드를 개발

학습 단계:

1. Colab 또는 클라우드 VM에서 전처리와 학습 실행
2. 모델 checkpoint, metrics report, confusion matrix만 Drive에 저장
3. 로컬 저장소에는 코드와 작은 설정 파일만 유지

앱 탑재 전 단계:

1. 최종 feature list 고정
2. 앱 raw 입력 5종으로 재현 가능한 feature만 사용했는지 검증
3. 모델을 경량화하거나 ONNX/TFLite/Core ML 변환 경로 검토
