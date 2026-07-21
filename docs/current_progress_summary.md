# SSE 현재 진행 요약

이 문서는 DreamT 수면 단계 예측 실험의 최신 진행 상황만 정리하는 rolling summary다.
`docs/next_chat_handoff.md`는 다음 채팅방에 그대로 전달할 최소 프롬프트만 담고, 현재 best/next experiment/결과 비교 이력은 이 파일을 기준으로 갱신한다.

## 현재 목표

비용, 모델 수, 추론량은 무시하고 성능만 최우선으로 본다.

기본 선택 기준:

```text
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.
```

## 현재 Best

```text
4-model stage-split flexible fusion
classwise4_w_p0.78_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.75_c0.01_l0.20_rem_p0.00_c0.42_l0.12
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
  original 0.18 / full_w20 0.78 / capacity_h128 0.04 / h128_ls003 0.00

Light(N1/N2):
  original 0.04 / full_w20 0.79 / capacity_h128 0.02 / h128_ls003 0.15

Deep(N3):
  original 0.04 / full_w20 0.75 / capacity_h128 0.01 / h128_ls003 0.20

REM:
  original 0.46 / full_w20 0.00 / capacity_h128 0.42 / h128_ls003 0.12
```

3-seed 평균:

```text
4M 0.4143 / 4K 0.2571
Wake 0.5084 / Light 0.6414 / Deep 0.1243 / REM 0.3829
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
4 Macro +0.0068 (+1.68%)
4 Kappa +0.0113 (+4.58%)
Wake    +0.0050 (+1.00%)
Light   +0.0093 (+1.47%)
Deep    +0.0024 (+1.93%)
REM     +0.0107 (+2.87%)
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
   grouped flexible best 도출
   4M 0.4128 / 4K 0.2543

5. 4-model flex4 stage-split refine
   Light(N1/N2)와 Deep(N3) weight 분리
   stage-split best 도출
   4M 0.4133 / 4K 0.2546

6. 4-model flex4 stage-split refine round2
   새 stage-split best 주변 조밀 탐색
   stage-split best 도출
   4M 0.4135 / 4K 0.2555

7. 4-model flex4 stage-split refine round3
   round2 best가 grid edge에 걸린 축 확장
   stage-split best 도출
   4M 0.4138 / 4K 0.2559

8. 4-model flex4 stage-split refine round4
   round3 best가 grid edge에 걸린 축 재확장
   stage-split best 도출
   4M 0.4139 / 4K 0.2564

9. 4-model flex4 stage-split refine round5
   round4 best에서 edge에 닿은 Light/REM 축 확장
   stage-split best 도출
   4M 0.4142 / 4K 0.2571

10. 4-model flex4 stage-split refine round6
    round5 best와 pure top 사이 동시 탐색
    현재 best 도출
    4M 0.4143 / 4K 0.2571
```

flex4_refine에서 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.08_l0.00_ld_p0.76_c0.02_l0.18_rem_p0.00_c0.34_l0.04
4M 0.4130 / 4K 0.2543 / Wake 0.5081 / Light 0.6372 / Deep 0.1266 / REM 0.3802
```

하지만 tie band 내에서 Wake+REM이 높은 현재 best를 선택했다.

flex4_stage_refine 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.08_l0.00_li_p0.77_c0.02_l0.17_d_p0.76_c0.00_l0.18_rem_p0.00_c0.36_l0.05
4M 0.4133 / 4K 0.2549 / Wake 0.5078 / Light 0.6387 / Deep 0.1262 / REM 0.3804
4M+4K 0.6682 / Wake+REM 0.8882
```

현재 best는 pure top 대비 4M+4K가 0.0003 낮아 tie band 안에 있고, Wake+REM이 더 높아서 선택 기준상 우선된다.

```text
classwise4_w_p0.77_c0.08_l0.00_li_p0.77_c0.02_l0.15_d_p0.76_c0.00_l0.20_rem_p0.00_c0.34_l0.05
4M 0.4133 / 4K 0.2546 / Wake 0.5083 / Light 0.6376 / Deep 0.1263 / REM 0.3809
4M+4K 0.6679 / Wake+REM 0.8892
```

이전 grouped best 대비:

```text
4M+4K +0.0008
Wake+REM +0.0004
4 Macro +0.0005 / 4 Kappa +0.0003
Wake +0.0000 / Light +0.0004 / Deep +0.0011 / REM +0.0004
```

flex4_stage_refine_round2 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.06_l0.00_li_p0.77_c0.00_l0.17_d_p0.77_c0.02_l0.18_rem_p0.00_c0.36_l0.06
4M 0.4136 / 4K 0.2557 / Wake 0.5092 / Light 0.6385 / Deep 0.1260 / REM 0.3806
4M+4K 0.6693 / Wake+REM 0.8898
```

현재 best는 pure top 대비 4M+4K가 0.0003 낮아 tie band 안에 있고, Wake+REM이 더 높아서 선택 기준상 우선된다.

```text
classwise4_w_p0.77_c0.06_l0.00_li_p0.77_c0.00_l0.17_d_p0.77_c0.02_l0.20_rem_p0.00_c0.36_l0.06
4M 0.4135 / 4K 0.2555 / Wake 0.5092 / Light 0.6383 / Deep 0.1257 / REM 0.3807
4M+4K 0.6689 / Wake+REM 0.8899
```

이전 stage-split best 대비:

```text
4M+4K +0.0010
Wake+REM +0.0007
4 Macro +0.0002 / 4 Kappa +0.0008
Wake +0.0009 / Light +0.0007 / Deep -0.0006 / REM -0.0001
```

flex4_stage_refine_round3 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.77_c0.02_l0.17_d_p0.77_c0.01_l0.18_rem_p0.00_c0.40_l0.08
4M 0.4139 / 4K 0.2560 / Wake 0.5091 / Light 0.6392 / Deep 0.1256 / REM 0.3816
4M+4K 0.6699 / Wake+REM 0.8907
```

현재 best는 pure top 대비 4M+4K가 0.0002 낮아 tie band 안에 있고, Wake+REM이 더 높아서 선택 기준상 우선된다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.77_c0.02_l0.17_d_p0.76_c0.03_l0.20_rem_p0.00_c0.40_l0.08
4M 0.4138 / 4K 0.2559 / Wake 0.5091 / Light 0.6390 / Deep 0.1254 / REM 0.3818
4M+4K 0.6697 / Wake+REM 0.8909
```

이전 round2 best 대비:

```text
4M+4K +0.0007
Wake+REM +0.0010
4 Macro +0.0003 / 4 Kappa +0.0004
Wake -0.0001 / Light +0.0007 / Deep -0.0003 / REM +0.0010
```

flex4_stage_refine_round4 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.04_l0.17_d_p0.76_c0.03_l0.20_rem_p0.00_c0.42_l0.10
4M 0.4141 / 4K 0.2566 / Wake 0.5088 / Light 0.6410 / Deep 0.1254 / REM 0.3814
4M+4K 0.6707 / Wake+REM 0.8901
```

현재 best는 pure top 대비 4M+4K가 0.0004 낮아 tie band 안에 있고, Wake+REM이 더 높아서 선택 기준상 우선된다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.02_l0.19_d_p0.77_c0.02_l0.20_rem_p0.00_c0.42_l0.10
4M 0.4139 / 4K 0.2564 / Wake 0.5090 / Light 0.6405 / Deep 0.1251 / REM 0.3812
4M+4K 0.6703 / Wake+REM 0.8902
```

이전 round3 best 대비:

```text
4M+4K +0.0007
Wake+REM -0.0007
4 Macro +0.0001 / 4 Kappa +0.0005
Wake -0.0001 / Light +0.0015 / Deep -0.0003 / REM -0.0006
```

flex4_stage_refine_round5 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.04_l0.17_d_p0.76_c0.02_l0.20_rem_p0.00_c0.42_l0.12
4M 0.4145 / 4K 0.2573 / Wake 0.5088 / Light 0.6418 / Deep 0.1254 / REM 0.3819
4M+4K 0.6718 / Wake+REM 0.8907
```

현재 best는 pure top 대비 4M+4K가 0.0005 낮아 tie band 안에 있고, Wake+REM이 더 높아서 선택 기준상 우선된다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.02_l0.17_d_p0.76_c0.01_l0.18_rem_p0.00_c0.44_l0.11
4M 0.4142 / 4K 0.2571 / Wake 0.5086 / Light 0.6414 / Deep 0.1244 / REM 0.3822
4M+4K 0.6713 / Wake+REM 0.8908
```

이전 round4 best 대비:

```text
4M+4K +0.0009
Wake+REM +0.0006
4 Macro +0.0002 / 4 Kappa +0.0007
Wake -0.0004 / Light +0.0009 / Deep -0.0007 / REM +0.0010
```

flex4_stage_refine_round6 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.04_l0.17_d_p0.77_c0.00_l0.16_rem_p0.00_c0.42_l0.12
4M 0.4144 / 4K 0.2574 / Wake 0.5087 / Light 0.6420 / Deep 0.1253 / REM 0.3818
4M+4K 0.6718 / Wake+REM 0.8905
```

현재 best는 pure top 대비 4M+4K가 0.0005 낮아 tie band 안에 있고, Wake+REM이 더 높아서 선택 기준상 우선된다.

```text
classwise4_w_p0.78_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.75_c0.01_l0.20_rem_p0.00_c0.42_l0.12
4M 0.4143 / 4K 0.2571 / Wake 0.5084 / Light 0.6414 / Deep 0.1243 / REM 0.3829
4M+4K 0.6714 / Wake+REM 0.8913
```

이전 round5 best 대비:

```text
4M+4K +0.0001
Wake+REM +0.0005
4 Macro +0.0001 / 4 Kappa -0.0000
Wake -0.0002 / Light +0.0000 / Deep -0.0001 / REM +0.0007
```

## 현재 코드 상태

최근 추가된 핵심 스크립트:

```text
scripts/run_four_model_flex4_stage_refinement_colab.sh
scripts/run_four_model_flex4_stage_refinement_round2_colab.sh
scripts/run_four_model_flex4_stage_refinement_round3_colab.sh
scripts/run_four_model_flex4_stage_refinement_round4_colab.sh
scripts/run_four_model_flex4_stage_refinement_round5_colab.sh
scripts/run_four_model_flex4_stage_refinement_round6_colab.sh
scripts/run_four_model_flex4_stage_refinement_round7_colab.sh
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

## 최근 완료 실험

완료:

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

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_stage_refine_summary.json
```

완료:

```text
새 stage-split best 주변을 더 조밀하게 보는 flex4_stage_refine_round2
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round2_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_stage_refine_round2_summary.json
```

완료:

```text
round2 best가 grid edge에 걸린 축을 확장하는 flex4_stage_refine_round3
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round3_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_stage_refine_round3_summary.json
```

완료:

```text
round3 best가 grid edge에 걸린 축을 한 번 더 확장하는 flex4_stage_refine_round4
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round4_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_stage_refine_round4_summary.json
```

완료:

```text
round4 best에서 edge에 닿은 Light/REM 축을 확장하는 flex4_stage_refine_round5
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round5_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_stage_refine_round5_summary.json
```

완료:

```text
round5 best와 pure top 사이를 같이 덮는 flex4_stage_refine_round6
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round6_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_stage_refine_round6_summary.json
```

## 다음 실험

우선순위 1:

```text
round6 best의 Wake+REM 우세 후보와 pure top 후보를 같이 덮는 flex4_stage_refine_round7
```

권장 grid:

```text
Wake:
  full_w20 0.78,0.79
  capacity_h128 0.02,0.04,0.06
  h128_ls003 0

Light(N1/N2):
  full_w20 0.78,0.79
  capacity_h128 0.02,0.04
  h128_ls003 0.13,0.15,0.17

Deep(N3):
  full_w20 0.74,0.75,0.76,0.77
  capacity_h128 0,0.01,0.02
  h128_ls003 0.18,0.20,0.22

REM:
  full_w20 0
  capacity_h128 0.42,0.44
  h128_ls003 0.12,0.13
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round7_colab.sh
```

비교 포인트:

```text
1. 현재 stage-split best 대비 4M+4K 변화
2. tie band 안에서는 Wake+REM 변화
3. REM 0.3828+를 유지하면서 Wake를 0.5085 이상으로 회복할 수 있는지
4. pure top 쪽 Light 0.642와 Deep 0.125+ 회복이 tie band 안에 남는지
5. 4M+4K 개선이 0.0005를 넘는 새 후보가 나오는지
```

## 다음 채팅방 시작 프롬프트

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.
현재 목표는 비용 무시, 성능-only fixed/flexible fusion 개선이야.
현재 best는 4-model stage-split flexible fusion:
classwise4_w_p0.78_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.75_c0.01_l0.20_rem_p0.00_c0.42_l0.12
3-seed 평균은 4M 0.4143 / 4K 0.2571 / Wake 0.5084 / Light 0.6414 / Deep 0.1243 / REM 0.3829.
다음 실험은 round6 best의 Wake+REM 우세 후보와 pure top 후보를 같이 덮는 flex4_stage_refine_round7야.
Colab에서는 git pull 후 scripts/run_four_model_flex4_stage_refinement_round7_colab.sh를 실행하면 돼.
결과 summary JSON을 받으면 current best 대비 4M+4K, Wake+REM, Light/Deep/REM 변화를 비교하고 이 current_progress_summary.md를 갱신해줘.
```
