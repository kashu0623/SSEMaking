# SSE 현재 진행 요약

이 문서는 DreamT 수면 단계 예측 실험의 최신 진행 상황만 정리하는 rolling summary다.
기존 `docs/next_chat_handoff.md`는 과거 채팅방 로그 보존용으로 두고, 앞으로 다음 채팅방 인계와 현재 best/next experiment는 이 파일을 기준으로 갱신한다.

## 현재 목표

비용, 모델 수, 추론량은 무시하고 성능만 최우선으로 본다.

기본 선택 기준:

```text
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.
```

## 현재 Best

```text
4-model grouped flexible fusion
classwise4_w_p0.77_c0.10_l0.00_ld_p0.76_c0.02_l0.17_rem_p0.00_c0.34_l0.04
```

사용 모델:

```text
1. original temporal = lstm_temporal_context20_h64_inverse
2. full w20 = lstm_temporal_w20_context20_h64_inverse
3. capacity_h128 = lstm_temporal_w20_context20_inverse_capacity_h128
4. h128_ls003 = lstm_temporal_w20_context20_inverse_h128_ls003
```

현재 best weight:

```text
Wake:
  original 0.13 / full_w20 0.77 / capacity_h128 0.10 / h128_ls003 0.00

Light+Deep grouped:
  original 0.05 / full_w20 0.76 / capacity_h128 0.02 / h128_ls003 0.17

REM:
  original 0.62 / full_w20 0.00 / capacity_h128 0.34 / h128_ls003 0.04
```

3-seed 평균:

```text
4M 0.4128 / 4K 0.2543
Wake 0.5083 / Light 0.6372 / Deep 0.1252 / REM 0.3805
```

## 이전 기준 대비 향상

이전 fixed 2-model 기준:

```text
classwise_nonrem0.90_rem0.20
4M 0.4074 / 4K 0.2458
Wake 0.5034 / Light 0.6321 / Deep 0.1220 / REM 0.3722
```

현재 best 대비:

```text
4 Macro +0.0054 (+1.32%)
4 Kappa +0.0085 (+3.45%)
Wake    +0.0049 (+0.97%)
Light   +0.0051 (+0.81%)
Deep    +0.0032 (+2.65%)
REM     +0.0083 (+2.23%)
```

## 최근 실험 흐름

```text
1. 2-model fixed fusion
   classwise_nonrem0.90_rem0.20
   4M 0.4074 / 4K 0.2458

2. 3-model ultra refine
   original + full_w20 + capacity_h128 근방 탐색
   fixed 2-model 대비 성능 개선 확인

3. 4-model flex4
   original + full_w20 + capacity_h128 + h128_ls003
   Wake / Light+Deep / REM classwise weight 분리

4. 4-model flex4 refine
   현재 best 도출
   4M 0.4128 / 4K 0.2543
```

flex4_refine에서 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.08_l0.00_ld_p0.76_c0.02_l0.18_rem_p0.00_c0.34_l0.04
4M 0.4130 / 4K 0.2543 / Wake 0.5081 / Light 0.6372 / Deep 0.1266 / REM 0.3802
```

하지만 tie band 내에서 Wake+REM이 높은 현재 best를 선택했다.

## 현재 코드 상태

최근 추가된 핵심 스크립트:

```text
scripts/run_four_model_flex4_stage_refinement_colab.sh
```

기능:

```text
Light(N1/N2)와 Deep(N3)을 분리해서 4-model flexible fusion weight를 탐색한다.
```

평가기는 아래 옵션을 지원하도록 확장되어 있다.

```text
src/sse_sleep/evaluate_four_model_fusion.py

--deep-primary-alphas
--deep-secondary-alphas
--deep-tertiary-alphas
```

`DEEP_*` 환경변수를 주지 않으면 기존 grouped Light+Deep 동작을 유지한다.

## 다음 실험

우선순위 1:

```text
4-model flex4 근방에서 Light(N1/N2)와 Deep(N3)을 분리하는 stage-split refinement
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_colab.sh
```

현재 default grid:

```text
Wake:
  full_w20 0.77,0.78
  capacity_h128 0.08,0.10
  h128_ls003 0

Light(N1/N2):
  full_w20 0.75,0.76,0.77
  capacity_h128 0,0.02
  h128_ls003 0.15,0.17

Deep(N3):
  full_w20 0.74,0.76
  capacity_h128 0,0.02
  h128_ls003 0.18,0.20

REM:
  full_w20 0
  capacity_h128 0.34,0.36
  h128_ls003 0.04,0.05
```

결과 summary JSON을 받으면 아래를 비교한다.

```text
1. 현재 best 대비 4M+4K 변화
2. tie band 안에서는 Wake+REM 변화
3. Light/Deep 분리로 Deep이 개선됐는지
4. REM이 희생됐는지
5. 새 best 채택 여부
```

## 다음 채팅방 시작 프롬프트

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.
현재 목표는 비용 무시, 성능-only fixed/flexible fusion 개선이야.
현재 best는 4-model grouped flexible fusion:
classwise4_w_p0.77_c0.10_l0.00_ld_p0.76_c0.02_l0.17_rem_p0.00_c0.34_l0.04
3-seed 평균은 4M 0.4128 / 4K 0.2543 / Wake 0.5083 / Light 0.6372 / Deep 0.1252 / REM 0.3805.
다음 실험은 Light(N1/N2)와 Deep(N3)을 분리하는 flex4_stage_refine부터 진행해줘.
Colab에서는 git pull 후 scripts/run_four_model_flex4_stage_refinement_colab.sh를 실행하면 돼.
결과 summary JSON을 받으면 current best 대비 4M+4K, Wake+REM, Light/Deep/REM 변화를 비교하고 이 current_progress_summary.md를 갱신해줘.
```

