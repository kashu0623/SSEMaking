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
classwise4_w_p0.74_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.44_l0.12
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
  original 0.20 / full_w20 0.74 / capacity_h128 0.06 / h128_ls003 0.00

Light(N1/N2):
  original 0.05 / full_w20 0.80 / capacity_h128 0.02 / h128_ls003 0.13

Deep(N3):
  original 0.01 / full_w20 0.81 / capacity_h128 0.00 / h128_ls003 0.18

REM:
  original 0.44 / full_w20 0.00 / capacity_h128 0.44 / h128_ls003 0.12
```

3-seed 평균:

```text
4M 0.4150 / 4K 0.2578
Wake 0.5095 / Light 0.6419 / Deep 0.1266 / REM 0.3821
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
4 Macro +0.0076 (+1.87%)
4 Kappa +0.0120 (+4.90%)
Wake    +0.0061 (+1.21%)
Light   +0.0098 (+1.55%)
Deep    +0.0046 (+3.77%)
REM     +0.0099 (+2.66%)
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

11. 4-model flex4 kappa refine
    4K 0.2575~0.2580 근방을 직접 겨냥한 compact grid
    current best 유지
    best_by_4K 4M 0.4144 / 4K 0.2574

12. 4-model flex4 kappa refine round2
    kappa ridge edge 축 확장
    현재 best 도출
    4M 0.4143 / 4K 0.2574

13. 4-model flex4 kappa refine round3
    round2 pure top과 current best 사이 동시 탐색
    현재 best 도출
    4M 0.4145 / 4K 0.2576

14. 4-model flex4 kappa refine round4
    round3 pure top과 current best 사이 동시 탐색
    현재 best 도출
    4M 0.4149 / 4K 0.2575

15. 4-model flex4 kappa refine round5
    round4 pure top과 current best 주변 edge 축 확장
    현재 best 도출
    4M 0.4150 / 4K 0.2578
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

flex4_kappa_refine 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.04_l0.17_d_p0.77_c0.00_l0.16_rem_p0.00_c0.42_l0.12
4M 0.4144 / 4K 0.2574 / Wake 0.5087 / Light 0.6420 / Deep 0.1253 / REM 0.3818
4M+4K 0.6718 / Wake+REM 0.8905
```

best_by_4K는 아래 후보였다.

```text
classwise4_w_p0.77_c0.02_l0.00_li_p0.79_c0.02_l0.17_d_p0.77_c0.00_l0.16_rem_p0.00_c0.44_l0.12
4M 0.4144 / 4K 0.2574 / Wake 0.5084 / Light 0.6422 / Deep 0.1249 / REM 0.3819
4M+4K 0.6718 / Wake+REM 0.8903
```

기존 선택 기준을 적용하면 현재 best가 pure top 대비 4M+4K가 0.0005 낮아 tie band 안에 있고, Wake+REM이 더 높아서 계속 우선된다.

```text
classwise4_w_p0.78_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.75_c0.01_l0.20_rem_p0.00_c0.42_l0.12
4M 0.4143 / 4K 0.2571 / Wake 0.5084 / Light 0.6414 / Deep 0.1243 / REM 0.3829
4M+4K 0.6714 / Wake+REM 0.8913
```

current best 대비 best_by_4K 변화:

```text
4M+4K +0.0004
Wake+REM -0.0010
4 Macro +0.0001 / 4 Kappa +0.0003
Wake -0.0001 / Light +0.0008 / Deep +0.0006 / REM -0.0010
```

flex4_kappa_refine_round2 결과 pure 4M+4K top과 best_by_4K는 같은 후보였다.

```text
classwise4_w_p0.76_c0.02_l0.00_li_p0.79_c0.02_l0.17_d_p0.77_c0.00_l0.16_rem_p0.00_c0.44_l0.12
4M 0.4145 / 4K 0.2577 / Wake 0.5088 / Light 0.6423 / Deep 0.1249 / REM 0.3818
4M+4K 0.6721 / Wake+REM 0.8906
```

선택 기준상 채택한 새 current best는 pure top 대비 4M+4K가 0.0004 낮아 tie band 안에 있고, Wake+REM이 더 높다. 이전 current best는 pure top 대비 4M+4K가 0.0008 낮아 tie band 밖으로 밀렸으므로 새 best를 채택한다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.80_c0.00_l0.17_d_p0.78_c0.00_l0.14_rem_p0.00_c0.44_l0.12
4M 0.4143 / 4K 0.2574 / Wake 0.5089 / Light 0.6420 / Deep 0.1242 / REM 0.3821
4M+4K 0.6717 / Wake+REM 0.8910
```

이전 current best 대비:

```text
4M+4K +0.0003
Wake+REM -0.0003
4 Macro +0.0000 / 4 Kappa +0.0003
Wake +0.0004 / Light +0.0005 / Deep -0.0002 / REM -0.0007
```

flex4_kappa_refine_round3 결과 pure 4M+4K top과 best_by_4K는 같은 후보였다.

```text
classwise4_w_p0.77_c0.04_l0.00_li_p0.80_c0.02_l0.15_d_p0.77_c0.00_l0.16_rem_p0.00_c0.44_l0.12
4M 0.4146 / 4K 0.2577 / Wake 0.5087 / Light 0.6424 / Deep 0.1250 / REM 0.3824
4M+4K 0.6724 / Wake+REM 0.8912
```

선택 기준상 채택한 새 current best는 pure top 대비 4M+4K가 0.0003 낮아 tie band 안에 있고, Wake+REM이 더 높다. 이전 current best는 새 pure top 대비 4M+4K가 0.0007 낮아 tie band 밖으로 밀렸으므로 새 best를 채택한다.

```text
classwise4_w_p0.76_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.79_c0.00_l0.16_rem_p0.00_c0.42_l0.12
4M 0.4145 / 4K 0.2576 / Wake 0.5089 / Light 0.6418 / Deep 0.1249 / REM 0.3825
4M+4K 0.6721 / Wake+REM 0.8914
```

이전 current best 대비:

```text
4M+4K +0.0004 (+0.0526%)
Wake+REM +0.0004 (+0.0425%)
4 Macro +0.0002 (+0.0527%)
4 Kappa +0.0001 (+0.0525%)
Wake -0.0000 (-0.0026%)
Light -0.0002 (-0.0345%)
Deep +0.0007 (+0.5767%)
REM +0.0004 (+0.1024%)
```

flex4_kappa_refine_round4 결과 pure 4M+4K top과 best_by_4K는 같은 후보였다.

```text
classwise4_w_p0.75_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.80_c0.00_l0.18_rem_p0.00_c0.44_l0.12
4M 0.4150 / 4K 0.2577 / Wake 0.5089 / Light 0.6421 / Deep 0.1268 / REM 0.3823
4M+4K 0.6728 / Wake+REM 0.8912
```

선택 기준상 채택한 새 current best는 pure top 대비 4M+4K가 0.0003 낮아 tie band 안에 있고, Wake+REM이 더 높다. 이전 current best는 새 pure top 대비 4M+4K가 0.0007 낮아 tie band 밖으로 밀렸으므로 새 best를 채택한다.

```text
classwise4_w_p0.75_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.80_c0.00_l0.18_rem_p0.00_c0.44_l0.11
4M 0.4149 / 4K 0.2575 / Wake 0.5089 / Light 0.6415 / Deep 0.1268 / REM 0.3825
4M+4K 0.6724 / Wake+REM 0.8914
```

이전 current best 대비:

```text
4M+4K +0.0004 (+0.0539%)
Wake+REM +0.0000 (+0.0003%)
4 Macro +0.0004 (+0.1038%)
4 Kappa -0.0001 (-0.0265%)
Wake +0.0000 (+0.0049%)
Light -0.0002 (-0.0339%)
Deep +0.0019 (+1.5505%)
REM -0.0000 (-0.0057%)
```

flex4_kappa_refine_round5 결과 pure 4M+4K top은 아래 후보였다.

```text
classwise4_w_p0.74_c0.06_l0.00_li_p0.81_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.42_l0.12
4M 0.4152 / 4K 0.2578 / Wake 0.5093 / Light 0.6417 / Deep 0.1278 / REM 0.3820
4M+4K 0.6730 / Wake+REM 0.8912
```

best_by_4K는 아래 후보였다.

```text
classwise4_w_p0.74_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.80_c0.00_l0.18_rem_p0.00_c0.44_l0.12
4M 0.4151 / 4K 0.2579 / Wake 0.5095 / Light 0.6420 / Deep 0.1268 / REM 0.3820
4M+4K 0.6730 / Wake+REM 0.8915
```

선택 기준상 채택한 새 current best는 pure top 대비 4M+4K가 0.0001 낮아 tie band 안에 있고, Wake+REM이 더 높다. 이전 current best는 새 pure top 대비 4M+4K가 0.0006 낮아 tie band 밖으로 밀렸으므로 새 best를 채택한다.

```text
classwise4_w_p0.74_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.44_l0.12
4M 0.4150 / 4K 0.2578 / Wake 0.5095 / Light 0.6419 / Deep 0.1266 / REM 0.3821
4M+4K 0.6729 / Wake+REM 0.8916
```

이전 current best 대비:

```text
4M+4K +0.0004 (+0.0657%)
Wake+REM +0.0002 (+0.0210%)
4 Macro +0.0001 (+0.0195%)
4 Kappa +0.0004 (+0.1403%)
Wake +0.0006 (+0.1150%)
Light +0.0003 (+0.0525%)
Deep -0.0002 (-0.1580%)
REM -0.0004 (-0.1040%)
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
scripts/run_four_model_flex4_kappa_refinement_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round2_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round3_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round4_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round5_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round6_colab.sh
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

완료:

```text
Kappa를 직접 겨냥해서 4K 0.2575~0.2580 근방을 노리는 flex4_kappa_refine
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_summary.json
```

완료:

```text
flex4_kappa_refine best_by_4K 주변의 edge 축을 확장하는 flex4_kappa_refine_round2
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round2_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_round2_summary.json
```

완료:

```text
flex4_kappa_refine_round2의 pure top과 새 current best 사이를 같이 덮는 flex4_kappa_refine_round3
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round3_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_round3_summary.json
```

완료:

```text
flex4_kappa_refine_round3의 pure top과 새 current best 사이를 같이 덮는 flex4_kappa_refine_round4
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round4_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_round4_summary.json
```

완료:

```text
flex4_kappa_refine_round4의 pure top과 새 current best 주변 edge 축을 확장하는 flex4_kappa_refine_round5
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round5_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_round5_summary.json
```

## 다음 실험

우선순위 1:

```text
flex4_kappa_refine_round5의 4K ridge와 새 current best 주변을 확장하는 flex4_kappa_refine_round6
```

권장 grid:

```text
Wake:
  full_w20 0.73,0.74,0.75
  capacity_h128 0.04,0.06
  h128_ls003 0

Light(N1/N2):
  full_w20 0.80,0.81,0.82
  capacity_h128 0.02,0.04
  h128_ls003 0.13,0.15

Deep(N3):
  full_w20 0.81,0.82
  capacity_h128 0
  h128_ls003 0.18,0.20

REM:
  full_w20 0
  capacity_h128 0.42,0.44,0.46
  h128_ls003 0.12,0.13
```

후보 수를 약 1.7k로 제한해서 summary JSON이 너무 커지지 않도록 한다.

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round6_colab.sh
```

비교 포인트:

```text
1. best_by_4K가 0.2575~0.2580에 도달하는지
2. best_by_4K의 4M+4K가 current best와 얼마나 차이 나는지
3. 새 best가 나오면 이전 current best 대비 절대 변화와 % 변화를 같이 기록
4. 기존 선택 기준상 overall best도 갱신되는지
5. 4K 0.2580 돌파 가능성과 Deep 0.127대 유지 여부
```

## 다음 채팅방 시작 프롬프트

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.
현재 목표는 비용 무시, 성능-only fixed/flexible fusion 개선이야.
현재 best는 4-model stage-split flexible fusion:
classwise4_w_p0.74_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.44_l0.12
3-seed 평균은 4M 0.4150 / 4K 0.2578 / Wake 0.5095 / Light 0.6419 / Deep 0.1266 / REM 0.3821.
다음 실험은 flex4_kappa_refine_round5의 4K ridge와 새 current best 주변을 확장하는 flex4_kappa_refine_round6이야.
Colab에서는 git pull 후 scripts/run_four_model_flex4_kappa_refinement_round6_colab.sh를 실행하면 돼.
결과 summary JSON을 받으면 best_by_4K와 기존 선택 기준 overall best를 둘 다 current best 대비 비교하고 이 current_progress_summary.md를 갱신해줘.
```
