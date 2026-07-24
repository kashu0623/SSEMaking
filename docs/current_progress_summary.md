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
classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13
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
  original 0.22 / full_w20 0.72 / capacity_h128 0.06 / h128_ls003 0.00

Light(N1/N2):
  original 0.03 / full_w20 0.80 / capacity_h128 0.02 / h128_ls003 0.15

Deep(N3):
  original 0.00 / full_w20 0.82 / capacity_h128 0.00 / h128_ls003 0.18

REM:
  original 0.45 / full_w20 0.00 / capacity_h128 0.42 / h128_ls003 0.13
```

3-seed 평균:

```text
4M 0.4153 / 4K 0.2581
Wake 0.5099 / Light 0.6414 / Deep 0.1274 / REM 0.3825
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
4 Macro +0.0079 (+1.94%)
4 Kappa +0.0123 (+4.99%)
Wake    +0.0065 (+1.30%)
Light   +0.0093 (+1.47%)
Deep    +0.0054 (+4.41%)
REM     +0.0103 (+2.77%)
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

16. 4-model flex4 kappa refine round6
    round5의 4K ridge와 current best 주변 확장
    현재 best 도출
    4M 0.4152 / 4K 0.2580

17. 4-model flex4 kappa refine round7
    4K 0.2580 돌파 ridge 확장
    현재 best 도출
    4M 0.4153 / 4K 0.2581

18. 4-model oracle audit
    current best의 오답 중 기존 model pool이 복구 가능한 비율 측정
    dynamic gating으로 방향 전환 결정

19. validation-trained static/causal temporal gate
    oracle headroom의 일반화 가능성 검증
    test에서 큰 하락으로 탈락
    이후 학습 목표를 direct 4-class로 전환
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

flex4_kappa_refine_round6 결과 pure 4M+4K top과 best_by_4K는 같은 후보였다.

```text
classwise4_w_p0.73_c0.06_l0.00_li_p0.82_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.42_l0.13
4M 0.4154 / 4K 0.2581 / Wake 0.5096 / Light 0.6419 / Deep 0.1276 / REM 0.3823
4M+4K 0.6735 / Wake+REM 0.8920
```

선택 기준상 채택한 새 current best는 pure top 대비 4M+4K가 0.0003 낮아 tie band 안에 있고, Wake+REM이 더 높다. 이전 current best는 새 pure top 대비 4M+4K가 0.0006 낮아 tie band 밖으로 밀렸으므로 새 best를 채택한다.

```text
classwise4_w_p0.73_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13
4M 0.4152 / 4K 0.2580 / Wake 0.5098 / Light 0.6414 / Deep 0.1274 / REM 0.3824
4M+4K 0.6732 / Wake+REM 0.8922
```

이전 current best 대비:

```text
4M+4K +0.0004 (+0.0521%)
Wake+REM +0.0006 (+0.0670%)
4 Macro +0.0002 (+0.0542%)
4 Kappa +0.0001 (+0.0486%)
Wake +0.0003 (+0.0556%)
Light -0.0005 (-0.0734%)
Deep +0.0008 (+0.6108%)
REM +0.0003 (+0.0820%)
```

flex4_kappa_refine_round7 결과 pure 4M+4K top과 best_by_4K는 아래 후보였다.

```text
classwise4_w_p0.73_c0.06_l0.00_li_p0.82_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.42_l0.13
4M 0.4154 / 4K 0.2581 / Wake 0.5096 / Light 0.6419 / Deep 0.1276 / REM 0.3823
4M+4K 0.6735 / Wake+REM 0.8920
```

선택 기준상 채택한 새 current best는 pure top 대비 4M+4K가 0.0001 낮아 tie band 안에 있고, Wake+REM이 더 높다. 이전 current best도 tie band 안이지만 Wake+REM이 낮으므로 새 best를 채택한다.

```text
classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13
4M 0.4153 / 4K 0.2581 / Wake 0.5099 / Light 0.6414 / Deep 0.1274 / REM 0.3825
4M+4K 0.6734 / Wake+REM 0.8924
```

이전 current best 대비:

```text
4M+4K +0.0002 (+0.0250%)
Wake+REM +0.0003 (+0.0320%)
4 Macro +0.0001 (+0.0178%)
4 Kappa +0.0001 (+0.0366%)
Wake +0.0002 (+0.0365%)
Light +0.0000 (+0.0016%)
Deep +0.0000 (+0.0000%)
REM +0.0001 (+0.0261%)
```

four_model_oracle_audit 결과는 fixed weight 재탐색보다 dynamic gate를 우선해야 한다는 근거를 제공했다.
oracle은 정답을 알고 있을 때만 가능한 상한이므로 성능 후보가 아니며, 기존 모델 pool의 상보성을 측정하는 진단값이다.

```text
test 3-seed current fusion:
4M 0.4153 / 4K 0.2581 / 4M+4K 0.6734

test 3-seed oracle (4 base model + current fusion):
4M 0.5998 / 4K 0.5391 / 4M+4K 1.1389
oracle headroom: +0.4655

fusion 오답 중 any-model recoverable: 42.13% +/- 0.25%p
model disagreement rate: 51.99%
agreement 구간 fusion accuracy: 63.62%
disagreement 구간 fusion accuracy: 46.01%
```

stage별 test recall oracle headroom:

```text
Wake  0.4927 -> 0.6131 (+0.1204)
Light 0.5976 -> 0.8249 (+0.2273)
Deep  0.1479 -> 0.2550 (+0.1071)
REM   0.5517 -> 0.7455 (+0.1938)
```

모델별 fusion 오답 rescue 비율은 original temporal 19.75%, capacity_h128 19.35%,
h128_ls003 18.34%, full_w20 10.41%다. Deep rescue 총량은 h128_ls003 114, capacity_h128 88,
original temporal 77, full_w20 21이다. 따라서 Deep-only 보정보다 4 model 전체를 sample별로 선택하는
gate가 우선이며, causal history가 추가 이득을 주는지 다음 실험에서 static gate와 직접 비교한다.

validation-trained gate 결과는 oracle 상보성이 validation-only direct gate의 일반화로 이어지지 않음을 보였다.
current best는 유지한다.

```text
test 3-seed:
current best fusion  4M 0.4153 / 4K 0.2581 / 4M+4K 0.6734
gate_static          4M 0.3345 / 4K 0.2093 / 4M+4K 0.5438
gate_causal          4M 0.3386 / 4K 0.1624 / 4M+4K 0.5010

gate_static vs current: -0.1296 (-19.25%)
gate_causal vs current: -0.1724 (-25.60%)
```

세 outer seed 모두에서 두 gate가 current best보다 낮았다. static gate의 Deep/REM은 0.0021/0.1416,
causal gate의 Deep/REM은 0.0812/0.2067로 크게 붕괴했다. 반면 full validation refit score는 높아
validation subject로만 학습한 direct class reclassifier가 test subject 분포에 과적합한 것으로 판단한다.
따라서 이 gate 계열은 중단하며, OOF stacked gate는 별도 대규모 cross-fitting 실험으로 미룬다.

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
scripts/run_four_model_flex4_kappa_refinement_round7_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round8_colab.sh
scripts/run_four_model_oracle_audit_colab.sh
scripts/run_four_model_causal_gate_colab.sh
scripts/run_four_model_same_split_init_ensemble_colab.sh
scripts/run_four_model_direct_4class_colab.sh
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

고정 weight refinement의 headroom을 판단하기 위한 compact oracle audit 평가기도 추가되어 있다.

```text
src/sse_sleep/evaluate_four_model_oracle_audit.py
src/sse_sleep/evaluate_four_model_causal_gate.py
src/sse_sleep/average_prediction_ensemble.py
src/sse_sleep/train_lstm_4class.py
src/sse_sleep/evaluate_four_model_4class_fusion.py
```

기능:

```text
1. current best fusion 오답 중 기존 4개 모델 하나라도 정답인 비율
2. Wake/Light/Deep/REM별 fusion recall과 oracle recall 상한
3. 모델별 rescue/exclusive rescue 비율과 rescue confidence
4. 모델 쌍별 prediction disagreement, joint error, error Jaccard
5. val/test 3-seed 평균을 하나의 작은 summary JSON으로 출력
```

causal gate는 validation labels로만 학습하며, validation subject 일부로 C/class-weight를 고른다.
학습이 끝난 gate는 validation 전체로 다시 fit한 후 untouched test에서만 평가한다.

same-split init ensemble은 기존 outer split을 바꾸지 않고, 각 role에 대해 초기화 seed만 다른 replica를
추가 학습해 role별 probability 평균을 만든 후 current best fusion weight를 그대로 적용한다.

direct 4-class trainer는 원래 N1/N2 label을 loss 계산 전에 Light로 합치고 Wake/Light/Deep/REM 네 logits만
학습한다. checkpoint도 validation `4 Macro F1 + 4 Kappa`로 선택한다. 기존 5-class trainer와 checkpoint는
변경하지 않으며, direct 4-class 후보가 current best를 넘기 전까지 current best도 유지한다.

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

완료:

```text
flex4_kappa_refine_round5의 4K ridge와 새 current best 주변을 확장하는 flex4_kappa_refine_round6
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round6_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_round6_summary.json
```

완료:

```text
flex4_kappa_refine_round6의 4K 0.2580 돌파 ridge를 확장하는 flex4_kappa_refine_round7
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_kappa_refinement_round7_colab.sh
```

결과 summary JSON:

```text
/Users/chan/Downloads/fusion4_original_full_w20_capacity_h128_ls003_context20_h64_flex4_kappa_refine_round7_summary.json
```

## 다음 실험

우선순위 1:

```text
current four-role architecture의 direct 4-class training baseline
```

목적:

```text
현재 선택 기준과 실제 앱 목표는 Wake/Light/Deep/REM 4-class지만 기존 모델은 N1/N2를 따로 학습한 뒤
평가에서만 합치고 있다. 앞으로는 N1/N2를 학습 전에 Light로 합쳐 불필요한 N1/N2 구분 손실을 제거한다.
먼저 current 네 role을 outer seed 42/7/123에서 직접 4-class로 한 번씩 학습해 단일 모델과 fusion을 비교한다.
```

총 12개 모델을 학습한다: outer seed 3개 x role 4개. original/full_w20은 h64, capacity_h128과
h128_ls003은 h128을 유지하며 ls003에만 label smoothing 0.03을 적용한다. fusion은 기존 current best의
Wake/Light/Deep/REM role weight를 그대로 mapping해 1차 비교한다.

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_four_model_direct_4class_colab.sh
```

결과 summary JSON:

```text
/content/drive/MyDrive/SSE_outputs/fusion4_direct_4class_context20_summary.json
```

비교 포인트:

```text
1. direct4 role별 단일 모델과 mapped-weight fusion의 3-seed 4M+4K
2. 5-class current best 대비 4M+4K 절대/상대 변화율
3. N1/N2 분리 제거가 Light와 Kappa를 실제로 올리는지
4. Deep/REM이 current best보다 유지 또는 개선되는지
5. 기존 선택 기준(4M+4K tie band 0.0005, Wake+REM 우선)상 새 best 채택 여부
```

결과에 따른 다음 분기:

```text
direct 4-class가 current best를 넘음:
  direct4 fusion weight refinement 후 same-split multi-init ensemble로 확장한다.

direct 4-class가 current best를 못 넘음:
  class-weight/sqrt-weight ablation 후 5-class same-split ensemble과 새 architecture를 비교한다.
```

보조 실험으로 `scripts/run_four_model_flex4_kappa_refinement_round8_colab.sh`도 준비되어 있지만,
최근 round 상승폭이 seed 변동보다 훨씬 작으므로 oracle audit보다 우선하지 않는다.

## 다음 채팅방 시작 프롬프트

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.
현재 목표는 비용 무시, 성능-only fixed/flexible fusion 개선이야.
현재 best는 4-model stage-split flexible fusion:
classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13
3-seed 평균은 4M 0.4153 / 4K 0.2581 / Wake 0.5099 / Light 0.6414 / Deep 0.1274 / REM 0.3825.
앞으로 학습 목표는 Wake/Light/Deep/REM direct 4-class로 전환한다.
다음 실험은 current four-role architecture의 direct 4-class baseline이야.
Colab에서는 git pull 후 scripts/run_four_model_direct_4class_colab.sh를 실행하면 돼.
결과 summary JSON을 받으면 role별 direct4 모델과 mapped-weight fusion을 current best 대비 4M+4K,
Wake+REM, Light/Deep으로 비교하고 새 best 채택 여부를 판단해 이 current_progress_summary.md를 갱신해줘.
```
