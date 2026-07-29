# SSE 현재 진행 요약

이 문서는 DreamT 수면 단계 예측 실험의 최신 진행 상황만 정리하는 rolling summary다.
`docs/next_chat_handoff.md`는 다음 채팅방에 그대로 전달할 최소 프롬프트만 담고, 현재 best/next experiment/결과 비교 이력은 이 파일을 기준으로 갱신한다.

## 현재 목표

비용, 모델 수, 추론량은 무시하고 성능만 최우선으로 본다.

4-class fusion refinement는 잠시 멈춘다. 현재 앱 목적에 직접 맞춘
`Light(N1/N2) vs Other(Wake/Deep/REM)` 분류를 새 주 실험 트랙으로 둔다.
Deep은 제거하지 않고 Light로 오인하면 안 되는 hard negative로 사용한다.

기본 선택 기준:

```text
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.
```

위 기준은 기존 4-class 트랙에만 적용한다. 새 app Light-vs-rest 트랙은
validation 3-seed 평균으로만 model/threshold를 선택하고 test는 보고에만 사용한다.
현재 Deep-veto audit은 validation Deep→Light가 기존 baseline 이하인 후보 중
`Light F1 + binary Kappa`가 가장 높은 후보를 우선한다.

## 최종 알람 판단 맥락

수면 단계 AI 하나로 최종 알람 여부를 결정하지 않는다. 별도로 개발 중인
`PotchArousalCalculator`와 결합해 판단한다.

```text
PotchArousalCalculator:
  미세 움직임, 호흡수(RR), 호흡변이도(RRV), 심박수(HR), 심박변이도(HRV),
  피부온도 변화를 각각 0~1 점수로 정규화한다.
  정규화 점수를 가중합해 최종 각성 점수를 계산하고,
  임계값을 넘으면 기상 타이밍 후보로 판정한다.
```

수면 단계 AI와 각성 점수의 구체적인 최종 결합 규칙은 아직 확정하지 않았다.

## 현재 Best

```text
current 4-role same-split ensemble + classwise-blended direct4 hybrid
+ h128/h256 Light-vs-Deep pair-blend specialist conditional replacement

static base:
  source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.19_li0.54_d0.82_rem0.00_dg1.15
specialist:
  blend_a0600__beta0.97_scale0.75_bias0.25
  exact beta: 0.975
  exact blend: 0.40 original-h128-CE + 0.60 Light-h256-2layer-CE
```

사용 모델:

```text
기존 current ensemble:
  role별 기존 checkpoint 1개 + 같은 outer split의 initialization replica 5개를 probability-average한다.

1. original temporal ensemble: 6 checkpoints
2. full w20 ensemble: 6 checkpoints
3. capacity_h128 ensemble: 6 checkpoints
4. h128_ls003 ensemble: 6 checkpoints
5. original direct4 source: 기존 single + 6-checkpoint ensemble의 classwise blend
6. original temporal Light-vs-Deep h128 1-layer CE specialist: 1 checkpoint
7. original temporal Light-vs-Deep h256 2-layer CE specialist: 1 checkpoint

outer split 하나의 실제 hybrid는 current 24 + direct4 6 + specialists 2로
총 32 checkpoints다.
outer seed 42/7/123은 3-seed 평가용이며 동시에 합쳐 배포하는 모델은 아니다.
```

현재 best 계산:

```text
1. 기존 24-checkpoint current ensemble을 아래 stage-split weight로 fusion한다.
   classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13

2. 5-class score를 Wake/max(N1,N2)/N3/REM 4개 score로 바꾼다.

3. 기존 single direct4와 6-checkpoint direct4 ensemble을 stage별로 섞고 정규화한다.
   Wake: 1.00 single + 0.00 ensemble
   Light: 1.00 single + 0.00 ensemble
   Deep: 0.75 single + 0.25 ensemble
   REM: 0.50 single + 0.50 ensemble

4. classwise-blended direct4 probability를 stage별로 hybrid한다.
   Wake: 0.8125 current + 0.1875 direct4 source
   Light: 0.45625 current + 0.54375 direct4 source
   Deep: 0.18125 current + 0.81875 direct4 source, 이후 Deep score x 1.15
   REM: 1.00 current + 0.00 direct4 source

5. true Light/Deep epoch만 사용해 original temporal specialist 2개를 학습한다.
   member 1: h128 1-layer LSTM / inverse class-weighted CE
   member 2: h256 2-layer LSTM / inverse class-weighted CE
   member probability: 0.40 h128 + 0.60 h256
   current hybrid의 Light+Deep 총질량은 보존한다.
   P(Deep | Light,Deep)는 current score와 blended specialist probability를
   beta 0.975로 결합한다.
   blended Deep logit calibration: scale 0.75 / bias +0.25
   Wake/REM score는 직접 변경하지 않는다.
```

3-seed 평균:

```text
4M 0.4572 / 4K 0.2917 / 4M+4K 0.7489
Wake 0.5356 / Light 0.6719 / Deep 0.2463 / REM 0.3751
Deep precision 0.2531 / Deep recall 0.2526
Wake+REM 0.9107
```

## 이전 기준 대비 향상

직전 current best:

```text
blend_a0600__beta1.00_scale0.75_bias0.25
4M 0.457428 / 4K 0.291478 / 4M+4K 0.748906
Wake 0.535473 / Light 0.671380 / Deep 0.247912 / REM 0.374946
Deep precision 0.252908 / Deep recall 0.255870 / Wake+REM 0.910419
```

새 current best의 직전 best 대비 변화:

```text
4M+4K -0.000028 (-0.0037%)
4 Macro -0.000206 (-0.0451%)
4 Kappa +0.000178 (+0.0611%)
Wake +0.000086 (+0.0160%)
Light +0.000529 (+0.0789%)
Deep -0.001597 (-0.6443%)
Deep precision +0.000160 (+0.0632%)
Deep recall -0.003221 (-1.2587%)
REM +0.000158 (+0.0421%)
Wake+REM +0.000243 (+0.0267%)
```

refinement pure top과 tie-band Deep top은
`blend_a0600__beta1.00_scale0.78_bias0.25`(exact scale 0.775)이며,
4M+4K `0.749230`이다. tie-rule selected는 pure top보다 총점이
`0.000352` 낮지만 Wake+REM이 `0.000580` 높다.

selected는 직전 best보다도 총점이 `0.000028` 낮아 tie band 안이고
Wake+REM이 `0.000243` 높으므로 프로젝트 규칙에 따라 새 current best로 채택한다.
pooled Deep 정답은 `433 -> 427`(-6), Deep→Light는 `1,018 -> 1,024`(+6),
Light→Deep은 `1,094 -> 1,081`(-13), Deep false positive는
`1,393 -> 1,378`(-15)다. validation 4M+4K는
`0.686747 -> 0.686888`(+0.000141, +0.0205%)이다.

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
    single-checkpoint best 도출
    4M 0.4153 / 4K 0.2581

18. 4-model oracle audit
    current best의 오답 중 기존 model pool이 복구 가능한 비율 측정
    dynamic gating으로 방향 전환 결정

19. validation-trained static/causal temporal gate
    oracle headroom의 일반화 가능성 검증
    test에서 큰 하락으로 탈락
    이후 학습 목표를 direct 4-class로 전환

20. same-split multi-init ensemble
    role별 기존 checkpoint + initialization replica 5개를 probability-average
    새 benchmark best 도출
    4M 0.4151 / 4K 0.2649
    Light/Kappa는 개선됐지만 Deep이 0.1000으로 하락

21. Deep probability/threshold audit
    current ensemble N3 probability의 ranking 및 validation-selected threshold 일반화 측정
    test ROC-AUC 0.7362 / AP 0.1055로 신호는 남아 있음
    raw threshold만으로는 recall과 false-positive tradeoff가 alarm veto에 부족

22. Causal temporal Deep probability audit
    raw/causal mean 3,5,10/EMA 0.20,0.40,0.60,0.80 비교
    모든 temporal variant의 test ROC-AUC/AP가 raw보다 낮음
    EMA 0.80의 threshold F1 개선도 0.2% 안팎으로 실질 개선 아님
    후처리 방향을 중단하고 direct 4-class 학습으로 전환

23. current four-role direct 4-class baseline
    original direct4가 Deep F1 0.1767로 benchmark의 0.1000보다 크게 개선
    하지만 4M+4K 0.6641로 benchmark 0.6799보다 낮아 benchmark best는 유지
    기존 5-class weight를 direct4 role에 그대로 mapping한 fusion은 Deep 0.0418로 실패
    current benchmark + original direct4 Deep specialist hybrid refinement로 전환

24. current ensemble + original direct4 hybrid Deep refinement
    alpha=0에서 이전 best를 정확히 재현
    새 best hybrid_w0.20_li0.10_d1.00_rem0.00 도출
    4M 0.4331 / 4K 0.2751 / 4M+4K 0.7082
    Deep 0.1592와 Wake+REM 0.8984도 동시에 상승
    Wake/Light/Deep grid 상단을 확장하는 round2로 전환

25. current + original direct4 hybrid Deep refinement round2
    round1 best를 정확히 재현
    새 best hybrid_w0.30_li0.24_d0.95_rem0.00_dg1.30 도출
    4M 0.4382 / 4K 0.2787 / 4M+4K 0.7169
    Deep 0.1776 / Wake+REM 0.9052로 동시 개선
    Light 상단/Deep alpha 하단의 두 near-tied ridge를 덮는 round3로 전환

26. current + original direct4 hybrid Deep refinement round3
    round2 best를 정확히 재현
    pure top은 hybrid_w0.32_li0.24_d1.00_rem0.00_dg1.25
    tie rule로 hybrid_w0.30_li0.34_d0.98_rem0.00_dg1.25를 새 best로 선택
    4M 0.4383 / 4K 0.2794 / 4M+4K 0.7177 / Wake+REM 0.9063
    Light 0.34 상단을 확장하는 마지막 static-grid round4로 전환

27. current + original direct4 hybrid Deep refinement round4
    pure top은 hybrid_w0.31_li0.34_d0.98_rem0.00_dg1.25
    tie rule로 hybrid_w0.31_li0.34_d0.85_rem0.00_dg1.20을 새 best로 선택
    4M 0.4378 / 4K 0.2799 / 4M+4K 0.7177 / Wake+REM 0.9072
    직전 best 대비 4M+4K +0.0012%, Wake+REM +0.0986%, Deep -1.8444%
    static grid를 중단하고 original direct4 same-split multi-init ensemble로 전환

28. original direct4 same-split multi-init ensemble + hybrid recalibration
    기존 direct4 1개와 새 initialization replica 5개를 동일 가중 평균
    direct4 단독 4M+4K는 +1.4886%, Kappa는 +5.0822% 개선
    하지만 direct4 단독 Deep F1이 -33.4829%로 붕괴
    hybrid pure top/tie-rule selected는 hybrid_w0.15_li0.55_d0.85_rem0.00_dg1.20
    4M 0.4264 / 4K 0.2748 / 4M+4K 0.7012 / Deep 0.1316 / Wake+REM 0.8996
    current best 대비 4M+4K -2.2964%, Deep -24.0839%이므로 채택하지 않음
    single/ensemble direct4 score를 stage별로 섞는 source blend audit로 전환

29. single/6-checkpoint direct4 classwise source blend + hybrid recalibration
    current best reference 4M+4K 0.717680을 정확히 재현
    새 best source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.15_li0.55_d0.85_rem0.00_dg1.20
    4M 0.4386 / 4K 0.2797 / 4M+4K 0.7182 / Deep 0.1770 / Wake+REM 0.9077
    직전 best 대비 4M+4K +0.0786%, Deep +2.1092%, Wake+REM +0.0486%
    validation 4M+4K도 +0.9245% 상승해 새 best로 채택
    source beta와 Wake/Light hybrid alpha 주변 joint refinement round2로 전환

30. direct4 classwise source blend + hybrid joint refinement round2
    current best reference 4M+4K 0.718244를 정확히 재현
    pure top은 w0.175/li0.55/d0.90/gain1.15, 4M+4K 0.718325
    tie-rule selected는 w0.15/li0.55/d0.80/gain1.25, 4M+4K 0.717992
    selected는 직전 best 대비 4M+4K -0.0351%, Deep -0.4572%, Wake+REM +0.0290%
    pooled Deep 정답은 3개 늘고 Deep→Light는 3개 감소
    tie rule에 따라 selected를 새 best로 채택
    동일 source를 고정하고 pure top/selected/직전 best 사이 hybrid ridge round3로 전환

31. direct4 classwise source blend + hybrid ridge refinement round3
    current best reference 4M+4K 0.717992를 정확히 재현
    pure top은 w0.1625/li0.55/d0.825/gain1.15, 4M+4K 0.718620
    tie-rule selected는 w0.20/li0.525/d0.775/gain1.15, 4M+4K 0.718172
    selected는 직전 best 대비 4M+4K +0.0251%, Wake+REM +0.0090%
    Deep F1 -1.8659%, pooled Deep 정답 -13, false positive -117
    총점과 Wake+REM이 모두 상승해 selected를 새 best로 채택
    selected/pure top/tie-band Deep top 사이 ridge round4로 전환

32. direct4 classwise source blend + hybrid ridge refinement round4
    current best reference 4M+4K 0.718172를 정확히 재현
    pure top은 w0.18125/li0.55/d0.8125/gain1.15, 4M+4K 0.718962
    tie-rule selected는 w0.20625/li0.525/d0.8625/gain1.20, 4M+4K 0.718759
    selected는 직전 best 대비 4M+4K +0.0817%, Deep +2.5074%, Wake+REM +0.0031%
    pooled Deep 정답 +10, Deep→Light -10으로 동시에 회복
    총점 개선폭이 tie band를 넘고 Wake+REM도 상승해 selected를 새 best로 채택
    반 간격의 마지막 static ridge refinement round5로 전환

33. direct4 classwise source blend + final hybrid ridge refinement round5
    current best reference 4M+4K 0.718759를 정확히 재현
    pure top과 tie-rule selected가 같은 w0.1875/li0.54375/d0.81875/gain1.15
    4M 0.4388 / 4K 0.2805 / 4M+4K 0.7193 / Wake+REM 0.9084
    직전 best 대비 4M+4K +0.0781%, 4K +0.1880%, Wake+REM +0.0424%
    Light +0.1191%, Deep -0.5871%; validation 4M+4K +0.1536%
    명확한 새 best로 채택하고 static ridge 탐색을 종료
    Light-vs-Deep binary specialist의 conditional probability fusion으로 전환

34. Light-vs-Deep binary specialist conditional fusion
    static round5 best 4M+4K 0.719320을 정확히 재현
    pure top과 tie-rule selected가 같은 original_h128_ce/beta1/scale0.5/bias0.5
    4M 0.4558 / 4K 0.2872 / 4M+4K 0.7430 / Wake+REM 0.9138
    직전 best 대비 4M+4K +3.2861%, Deep +41.7825%, Wake+REM +0.5895%
    Deep 정답 +206, Deep→Light -245로 병목을 크게 완화
    Light -1.6443%, Light→Deep +316, validation 총점 -1.4448%
    프로젝트 test 선택 규칙에 따라 새 best로 채택
    selected edge와 cross-split robust ridge를 함께 덮는 calibration refinement로 전환

35. Light-vs-Deep specialist calibration refinement
    current best reference 4M+4K 0.742958을 정확히 재현
    pure top과 tie-rule selected가 같은 beta1.00/scale0.55/bias0.25
    4M 0.4550 / 4K 0.2897 / 4M+4K 0.7447 / Wake+REM 0.9128
    직전 best 대비 4M+4K +0.2354%, 4K +0.8849%, Light +1.7678%
    Deep -5.5615%, Wake+REM -0.1018%지만 총점 차이는 tie band를 넘음
    Light→Deep -302, Deep false positive -370, validation 총점 +3.0769%
    test/validation 동시 개선으로 새 best 채택
    selected 주변 fine calibration round2로 전환

36. Light-vs-Deep specialist calibration refinement round2
    current best reference 4M+4K 0.744707을 정확히 재현
    pure top beta0.975/scale0.60/bias0.30, 4M+4K 0.745184
    tie-rule selected beta1.00/scale0.5375/bias0.25, 4M+4K 0.744948
    selected는 pure top과 0.000236 차이, Wake+REM은 0.001286 높음
    직전 best 대비 4M+4K +0.0324%, Deep +0.2026%, Wake+REM +0.0677%
    Deep 정답 +3, Deep→Light -5, validation 총점 -0.0865%
    프로젝트 규칙에 따라 selected를 새 best로 채택
    calibration 포화로 종료하고 specialist same-split multi-init ensemble로 전환

37. Light-vs-Deep specialist same-split multi-init ensemble
    current best 4M+4K 0.744948을 정확히 재현
    pure top, tie-rule selected, tie-band Deep top이 모두 기존 single current best
    가장 나은 새 init2002 후보도 4M+4K 0.730053, current 대비 -1.9995%
    ensemble6 top은 4M+4K 0.723296, current 대비 -2.9066%
    ensemble6 Deep precision +20.2701%지만 recall -35.3643%, Deep F1 -20.6251%
    단순 probability average가 약한 replica에 끌려가므로 새 best 채택 없음
    single을 보존하며 replica를 학습 가중하는 subject-OOF logistic stacking으로 전환

38. Light-vs-Deep specialist subject-OOF logistic stacking
    current best 4M+4K 0.744948을 정확히 재현
    5,587개 후보의 pure top/selected/tie-band Deep top이 모두 기존 single
    best stack none/C0.003도 4M+4K 0.720798, current 대비 -3.2418%
    best stack Deep precision +12.5231%지만 recall -36.8518%, Deep F1 -22.2464%
    OOF validation도 -12.5716%로 test와 같은 하락 방향
    weak replica의 상보 신호를 stacking으로 회수하지 못해 새 best 채택 없음
    실제 N3 오답의 핵심인 N2/N3 hard boundary를 직접 학습하는 specialist로 전환

39. N2-vs-N3 hard-boundary specialist
    current best 4M+4K 0.744948을 정확히 재현
    3,235개 후보의 pure top/selected/tie-band Deep top이 모두 기존 single
    best N2/N3 score는 h256 2-layer 0.727870, current 대비 -2.2925%
    best N2/N3 Deep은 h256 1-layer 0.214710, current 대비 -9.1921%
    N1 제외 hard-negative 학습은 current를 개선하지 못해 종료
    Light-vs-Deep h256 2-layer는 test -1.0742%지만 validation +2.3462%
    current h128과 validation-strong h256 2-layer pairwise blend로 전환

40. Light-vs-Deep h128+h256 pairwise specialist blend
    current best 4M+4K 0.744948을 정확히 재현
    4,999개 후보의 pure top/selected/tie-band Deep top이 모두 같은 후보
    h128 0.40 + h256 2-layer 0.60, beta1.00/scale0.75/bias0.25
    4M 0.4574 / 4K 0.2915 / 4M+4K 0.7489 / Wake+REM 0.9104
    직전 best 대비 4M+4K +0.5313%, 4M +0.5071%, 4K +0.5693%
    Deep +4.8504%, precision +9.0029%, recall +4.4785%
    Wake +0.3971%, Light +0.1195%, REM -1.3567%, Wake+REM -0.3327%
    Deep 정답 +29, Deep→Light -17, Light→Deep -59, Deep false positive -122
    validation 4M+4K도 +0.7700%로 test와 함께 개선되어 새 best 채택
    alpha 0.60과 calibration 주변의 pair-blend fine refinement로 전환

41. Light-vs-Deep h128+h256 pairwise specialist blend refinement
    current best 4M+4K 0.748906을 정확히 재현
    pure top/tie-band Deep top은 alpha0.60/beta1.00/scale0.775/bias0.25
    pure top 4M+4K 0.749230, 직전 best 대비 +0.0433%
    tie-rule selected는 alpha0.60/beta0.975/scale0.75/bias0.25
    selected는 pure top 대비 총점 -0.000352, Wake+REM +0.000580
    직전 best 대비 4M+4K -0.0037%, Wake+REM +0.0267%
    4K +0.0611%, Light +0.0789%, REM +0.0421%
    Deep -0.6443%, precision +0.0632%, recall -1.2587%
    Deep 정답 -6, Deep→Light +6, Light→Deep -13, Deep false positive -15
    validation 4M+4K +0.0205%
    tie 규칙에 따라 selected를 새 best로 채택하고 두 경쟁점 주변 round2로 전환

42. 앱 목적 중심 Light-vs-rest 트랙으로 전환
    4-class fusion refinement round2는 실행 전 보류
    primary target을 Light(N1/N2) vs Other(Wake/Deep/REM)로 변경
    Deep을 제외하지 않고 negative loss multiplier와 subgroup metric으로 관리
    direct binary와 4-class auxiliary multitask를 같은 outer split에서 비교
    threshold와 model selection은 validation 3-seed 평균만 사용
    test는 최종 보고에만 사용해 threshold test leakage를 방지
    기존 current best argmax baseline:
      pooled Light F1 0.671859 / binary Kappa 0.237415
      Light precision 0.702532 / recall 0.643752
      Deep→Light 0.607715

43. App-oriented Light-vs-rest objective audit
    9 configs x 3 seeds, validation-only model/threshold selection
    best validation Light objective:
      multitask_h256_lstm2_deep2_aux025 / threshold 0.10
    test pooled Light F1 0.750424, baseline 대비 +11.6937%
    test pooled Light recall 0.921068, baseline 대비 +43.0780%
    binary Kappa -36.4450%, precision -9.8794%, Light objective -0.8755%
    Deep→Light 0.607715 -> 0.944807(+55.4688%)
    Wake→Light +109.2282%, REM→Light +96.5762%
    낮은 threshold로 대부분을 Light로 보내 F1만 높인 unsafe solution
    validation Deep leak 제한 0.10/0.20/0.30/0.40도 test에서 모두 초과
    direct binary/multitask 단독 새 best 채택 없음
    Light proposal + current staging Deep conditional veto audit로 전환
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

same-split multi-init ensemble은 기존 single-checkpoint weight를 그대로 적용했지만 선택 기준상 새
benchmark best가 됐다.

```text
candidate:
classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13

test 3-seed:
4M 0.4151 / 4K 0.2649 / 4M+4K 0.6799
Wake 0.5131 / Light 0.6740 / Deep 0.1000 / REM 0.3731
Wake+REM 0.8862
```

이전 single-checkpoint best 대비 4M+4K는 +0.0065 (+0.9696%)다. 4K와 Light가 각각
+0.0068 (+2.6300%), +0.0326 (+5.0756%) 개선된 반면 Deep은 -0.0274 (-21.4834%) 하락했다.

role ensemble 단독 Deep F1은 아래와 같다.

```text
original 0.0828 / full_w20 0.1016 / capacity_h128 0.1327 / h128_ls003 0.1106
```

현재 fusion Deep weight는 full_w20 0.82 / h128_ls003 0.18이고, role 중 Deep이 가장 높은
capacity_h128 weight가 0이다. 따라서 single-checkpoint에서 찾은 Deep weight가 ensemble에는
최적이 아닐 가능성이 높다.

현재 ensemble best의 3개 outer test confusion matrix를 합산한 실제 N3 행:

```text
실제 N3 총 1,685:
Wake 102 (6.05%)
N1   216 (12.82%)
N2 1,074 (63.74%)
N3   176 (10.45%)
REM  117 (6.94%)

4-class:
Light 1,290 (76.56%) / Deep 176 (10.45%)
```

Deep 오답 1,509개 중 85.49%가 Light이며, 그중 N2가 71.17%를 차지한다. 무작위 stage 혼동보다
N2/N3 경계와 ensemble averaging에 의한 rare Deep probability 약화가 핵심 병목으로 보인다.

Deep probability/threshold audit 결과, current fusion의 N3 probability에는 argmax가 버리는 ranking
신호가 남아 있지만 calibration subject shift가 크다.

```text
current fusion:
validation ROC-AUC 0.8120 / AP 0.1594
test       ROC-AUC 0.7362 / AP 0.1055

test Deep prevalence 0.0480
test argmax:
precision 0.1187 / recall 0.1037 / specificity 0.9583 / F1 0.1000
predicted-positive-rate 0.0446
```

test role별 ranking:

```text
original       ROC-AUC 0.7151 / AP 0.1019
full_w20       ROC-AUC 0.7281 / AP 0.1024
capacity_h128  ROC-AUC 0.7283 / AP 0.1055
h128_ls003     ROC-AUC 0.7348 / AP 0.1071
current fusion ROC-AUC 0.7362 / AP 0.1055
```

fusion은 평균 ROC-AUC가 가장 높고 AP도 h128_ls003/capacity와 거의 같으므로 Deep ranking 신호가
fusion에서 특별히 소실된 것은 아니다. AP는 prevalence 대비 약 2.20배지만 절대 분리력은 아직 낮다.

validation recall floor로 선택한 current fusion threshold의 test 결과:

```text
policy      threshold  precision  recall  specificity  F1      predicted-positive
recall 50%  0.1457     0.1224     0.3348  0.8810       0.1781  0.1291
recall 70%  0.0745     0.1034     0.5357  0.7671       0.1728  0.2475
recall 80%  0.0555     0.0970     0.6189  0.7118       0.1672  0.3038
recall 90%  0.0213     0.0829     0.7875  0.5545       0.1499  0.4617
```

validation target 대비 test recall은 50/70/80/90 policy에서 각각 약 19.0/17.7/18.5/11.6%p
하락했다. max-F1 threshold는 seed별 threshold 편차가 매우 크고 test F1 0.0759로 argmax보다도
낮아 calibration overfit이 확인됐다. recall 90% policy도 평균 test recall은 78.75%까지 복구하지만
전체 epoch의 46.17%를 Deep으로 차단하고 specificity가 55.45%에 그쳐 단독 alarm veto로 채택하지 않는다.

다음 단계는 현재 probability를 버리지 않고 subject/gap 경계를 지키는 causal moving-average/EMA로
Deep 구간의 연속성을 사용해 ROC-AUC/AP와 recall-specificity tradeoff를 개선할 수 있는지 확인한다.

Causal temporal Deep probability audit 결과, smoothing은 raw N3 probability를 개선하지 못했다.

```text
variant          test ROC-AUC  test AP
raw              0.7362        0.1055
causal mean 3    0.7311        0.1042
causal mean 5    0.7282        0.1034
causal mean 10   0.7234        0.1023
EMA 0.20         0.7270        0.1038
EMA 0.40         0.7306        0.1044
EMA 0.60         0.7331        0.1049
EMA 0.80         0.7350        0.1053
```

가장 가까운 EMA 0.80도 raw 대비 ROC-AUC -0.0012 (-0.17%), AP -0.0002 (-0.16%)로 낮았다.
threshold 결과의 미세한 차이도 실질적인 개선으로 보지 않는다.

```text
recall-70 policy:
raw F1 0.1728 / EMA 0.80 F1 0.1732 (+0.0004, +0.25%)

recall-90 policy:
raw F1 0.1499 / EMA 0.80 F1 0.1502 (+0.0003, +0.19%)
raw onset/run recall 0.6826/0.7052
EMA 0.80 onset/run recall 0.6872/0.7076
```

이 차이는 seed 변동보다 훨씬 작으며 EMA 0.80도 recall-70에서 test recall이 낮아지고, recall-90에서는
specificity와 predicted-positive-rate가 소폭 나빠졌다. validation variant 선택도 recall-70/90에서
3 seed 중 2 seed가 raw를 선택했다. 긴 mean/강한 smoothing은 Deep 진입 감지 지연을 늘리면서
ranking까지 낮췄다. 따라서 현재 probability에 smoothing/hysteresis를 더 쌓는 방향은 중단한다.

Deep 문제는 후처리보다 학습 표현과 목적함수 문제로 판단한다. 다음은 N1/N2를 loss 전에 Light로 합치는
direct 4-class baseline을 실행하고, 그래도 Deep ranking/recall이 부족하면 N3-vs-rest binary specialist를
별도 alarm veto 모델로 학습한다.

direct 4-class baseline 결과, `original_4class`가 Deep을 뚜렷하게 복구했지만 전체 benchmark는 넘지 못했다.

```text
candidate          4M      4K      4M+4K  Wake    Light   Deep    REM
benchmark best     0.4151  0.2649  0.6799  0.5131  0.6740  0.1000  0.3731
original direct4   0.4160  0.2481  0.6641  0.5208  0.6363  0.1767  0.3301
mapped fusion      0.3896  0.2469  0.6365  0.5188  0.6558  0.0418  0.3419
```

`original_4class`의 benchmark 대비 변화:

```text
4M+4K -0.0158 (-2.3221%)
4 Macro +0.0009 (+0.2232%)
4 Kappa -0.0167 (-6.3109%)
Wake +0.0076 (+1.4908%)
Light -0.0377 (-5.5874%)
Deep +0.0767 (+76.6818%)
REM -0.0430 (-11.5192%)
Wake+REM -0.0353 (-3.9863%)
```

pooled confusion에서 실제 Deep 1,685개 중 정답은 `176 -> 268`로 92개 증가(+52.27%)했고,
Deep→Light 오답은 `1,290 -> 1,101`, 비율은 `76.56% -> 65.34%`로 11.22%p 감소했다.
즉 direct4 학습 자체는 Deep specialist로 유효하다. 다만 outer seed별 Deep F1이 0.2613/0.1070/0.1618로
편차가 크고 Light/Kappa/REM 손실 때문에 단독 benchmark best로는 채택하지 않는다.

기존 mapped fusion이 실패한 주원인은 5-class best의 Deep weight 82%를 direct4의 `full_w20_4class`에
그대로 배정한 것이다. 이 role의 test Deep F1은 0.0420뿐이므로 direct4에서는 기존 role weight를
재사용할 수 없다. 다음은 current benchmark의 기존 argmax를 정확히 보존하면서 가장 강한
`original_4class` 확률을 stage별로 주입하는 no-training hybrid grid다.

hybrid Deep refinement 결과, alpha=0 baseline이 이전 best를 소수점 끝까지 정확히 재현했고
pure 4M+4K top, tie-rule selected, tie band 내 Deep top이 모두 같은 새 best로 일치했다.

```text
hybrid_w0.20_li0.10_d1.00_rem0.00
4M 0.4331 / 4K 0.2751 / 4M+4K 0.7082
Wake 0.5233 / Light 0.6746 / Deep 0.1592 / REM 0.3752
Deep precision 0.2199 / Deep recall 0.1506 / Wake+REM 0.8984
```

기존 best 대비 4M+4K +0.0283(+4.1600%), Deep +0.0592(+59.1888%),
Wake+REM +0.0122(+1.3796%)로 전체 성능과 alarm-oriented 성능이 동시에 개선됐다.
pooled confusion에서도 Deep 정답은 `176 -> 242`, Deep→Light는 `1,290 -> 1,230`으로 개선됐다.

grid 단면은 Wake `0 -> 0.10 -> 0.20`, Light `0 -> 0.05 -> 0.10`,
Deep `0.80 -> 0.90 -> 1.00`에서 계속 상승했고 REM은 0이 최선이었다. 따라서 round2에서는
Wake/Light 상단을 확장하고 direct4 Deep score에 별도 gain 0.80~1.60을 적용해 alpha=1 이후의
Deep 판정 임계값까지 탐색한다.

주의할 점은 validation 4M+4K가 `0.6710 -> 0.6696`(-0.20%)로 소폭 낮아졌고 test score 표준편차가
`0.0278 -> 0.0554`로 커졌다는 것이다. 현재 프로젝트 선택 기준에 따라 새 best로 채택하되, 최종 배포
전에는 고정된 독립 holdout으로 repeated test-grid tuning의 과대평가 여부를 확인한다.

hybrid refinement round2 결과, round1 best를 정확히 재현했고 pure top, tie-rule selected,
tie band 내 Deep top이 다시 같은 후보로 일치했다.

```text
hybrid_w0.30_li0.24_d0.95_rem0.00_dg1.30
4M 0.4382 / 4K 0.2787 / 4M+4K 0.7169
Wake 0.5306 / Light 0.6700 / Deep 0.1776 / REM 0.3745
Deep precision 0.2086 / Deep recall 0.1847 / Wake+REM 0.9052
```

round1 best 대비 4M+4K +0.0087(+1.2352%), Deep +0.0184(+11.5637%),
Wake+REM +0.0067(+0.7500%)로 다시 동시 개선됐다. Deep precision은 5.12% 낮아졌지만 recall이
22.68% 높아져 Deep F1과 실제 Deep 정답 수가 개선됐다. pooled Deep 정답은 `242 -> 298`,
Deep→Light는 `1,230 -> 1,173`으로 줄었다.

pure top과 4M+4K 차이가 0.00004뿐인 두 번째 ridge는 아래 후보였다.

```text
hybrid_w0.30_li0.18_d1.00_rem0.00_dg1.30
4M+4K 0.716891 / Deep 0.1771 / Wake+REM 0.9047
```

pure top은 Wake 0.30과 gain 1.30이 내부 최적점이지만 Light 0.24가 상단, Deep alpha 0.95가
하단 edge다. round3에서는 두 near-tied ridge를 함께 덮도록 Wake 0.25~0.35, Light 0.16~0.34,
Deep alpha 0.80~1.00, gain 1.15~1.45를 세밀 탐색한다.

round2 validation 4M+4K는 round1 대비 `0.6696 -> 0.6672`(-0.37%)로 다시 낮아졌다.
현재 성능-only 선택 규칙에 따라 새 best로 채택하지만 독립 holdout 필요성은 더 커졌다.

hybrid refinement round3 결과, round2 best를 정확히 재현했다. 이번에는 pure top과 프로젝트
tie-rule selected가 달랐다.

```text
pure top:
hybrid_w0.32_li0.24_d1.00_rem0.00_dg1.25
4M+4K 0.717944 / Wake+REM 0.9052 / Deep 0.1771

selected:
hybrid_w0.30_li0.34_d0.98_rem0.00_dg1.25
4M 0.4383 / 4K 0.2794 / 4M+4K 0.717671
Wake 0.5313 / Light 0.6703 / Deep 0.1766 / REM 0.3750
Deep precision 0.2127 / Deep recall 0.1797 / Wake+REM 0.9063
```

selected는 pure top보다 4M+4K가 0.000272 낮아 0.0005 tie band 안에 있고, Wake+REM이
`0.9052 -> 0.9063`으로 더 높아 프로젝트 규칙상 우선한다.

round2 best 대비 selected의 4M+4K는 +0.0007(+0.1026%), Wake+REM은 +0.0011(+0.1254%)다.
Deep F1은 -0.0010(-0.5566%), pooled Deep 정답은 `298 -> 289`로 감소했다. 이번 개선은 주로
Kappa와 Wake+REM에서 왔으며 Deep 개선 round는 아니다.

selected의 Wake 0.30, Deep alpha 0.975, gain 1.25는 내부에 있지만 Light 0.34가 상단 edge다.
round4는 두 top ridge를 보존하면서 Light를 0.22~0.50으로 확장하는 마지막 static-grid pass다.
그 뒤에는 추가 test-grid tuning보다 original direct4 same-split multi-init으로 specialist variance를
줄이는 방향을 우선한다.

round3 validation 4M+4K는 round2 selected 대비 `0.6672 -> 0.6664`(-0.11%)이고 test score
표준편차도 `0.0558 -> 0.0601`로 커졌다. 독립 holdout 경고는 계속 유지한다.

hybrid refinement round4 결과 pure top은 아래 후보였다.

```text
hybrid_w0.31_li0.34_d0.98_rem0.00_dg1.25
4M 0.4384 / 4K 0.2796 / 4M+4K 0.718054
Wake 0.5318 / Light 0.6704 / Deep 0.1766 / REM 0.3747
Wake+REM 0.9066
```

프로젝트 tie rule로 선택한 새 best는 아래 후보다.

```text
hybrid_w0.31_li0.34_d0.85_rem0.00_dg1.20
4M 0.4378 / 4K 0.2799 / 4M+4K 0.717680
Wake 0.5318 / Light 0.6707 / Deep 0.1734 / REM 0.3754
Deep precision 0.2083 / Deep recall 0.1769 / Wake+REM 0.9072
```

selected는 pure top보다 4M+4K가 0.000374 낮아 tie band 안이고 Wake+REM이
`0.9066 -> 0.9072`로 더 높다. 직전 round3 best 대비 4M+4K는
`+0.000009(+0.0012%)`, Wake+REM은 `+0.000894(+0.0986%)`지만 Deep은
`-0.003258(-1.8444%)`다. pooled Deep 정답은 `289 -> 287`, Deep→Light는
`1,183 -> 1,186`으로 악화됐다.

Light를 0.50까지 확장했지만 top은 다시 0.34에 남았다. test 4M+4K 개선은 0.000009에
불과하고 validation 4M+4K도 `0.6664 -> 0.6660`(-0.0655%)로 낮아져 static grid는
포화됐다고 판단한다. 이후에는 original direct4 specialist의 same-split multi-init
probability ensemble로 모델 자체의 분산을 줄인 뒤 hybrid를 다시 보정한다.

original direct4 same-split multi-init ensemble 결과, pure top, tie-rule selected,
tie band 내 Deep top이 모두 아래 후보로 일치했다.

```text
hybrid_w0.15_li0.55_d0.85_rem0.00_dg1.20
4M 0.4264 / 4K 0.2748 / 4M+4K 0.701199
Wake 0.5265 / Light 0.6745 / Deep 0.1316 / REM 0.3731
Deep precision 0.1612 / Deep recall 0.1264 / Wake+REM 0.8996
```

current best 대비:

```text
4M+4K -0.016481 (-2.2964%)
4 Macro -0.011397 (-2.6032%)
4 Kappa -0.005083 (-1.8164%)
Wake -0.005271 (-0.9911%)
Light +0.003771 (+0.5623%)
Deep -0.041755 (-24.0839%)
Deep precision -0.047125 (-22.6256%)
Deep recall -0.050505 (-28.5562%)
REM -0.002335 (-0.6220%)
Wake+REM -0.007606 (-0.8384%)
```

pooled Deep 정답은 `287 -> 209`로 78개(-27.18%) 줄었고, Deep→Light는
`1,186 -> 1,264`, `70.39% -> 75.01%`로 4.63%p 증가했다. 따라서 새 hybrid는
채택하지 않고 25-checkpoint current best를 유지한다.

6-checkpoint direct4 단독을 기존 single direct4와 비교하면 결과가 양면적이다.

```text
4M+4K 0.6641 -> 0.6740 (+1.4886%)
4 Macro 0.4160 -> 0.4133 (-0.6551%)
4 Kappa 0.2481 -> 0.2607 (+5.0822%)
Wake 0.5208 -> 0.5300 (+1.7783%)
Light 0.6363 -> 0.6585 (+3.4872%)
Deep 0.1767 -> 0.1175 (-33.4829%)
REM 0.3301 -> 0.3469 (+5.0937%)
```

즉 초기화 평균은 Wake/Light/REM과 Kappa에는 유효하지만 기존 single direct4의 핵심 역할인
Deep specialist 성능을 희석했다. 전체 앙상블을 버리기보다 Wake/Light/REM은 ensemble 쪽,
Deep은 single 쪽을 더 쓰도록 direct4 source 자체를 stage별로 혼합한 뒤 hybrid를 재보정한다.
이전 실험의 validation 4M+4K는 `0.6831`로 current best의 `0.6660`보다 높아, stage별
정보 회수 가능성도 남아 있다.

direct4 classwise source blend 결과 current best reference를 정확히 재현했다. pure top,
tie-rule selected, tie band 내 Deep top은 모두 아래 후보로 일치했다.

```text
source:
  Wake ensemble beta 0.00
  Light ensemble beta 0.00
  Deep ensemble beta 0.25
  REM ensemble beta 0.50

hybrid:
  Wake alpha 0.15
  Light alpha 0.55
  Deep alpha 0.85
  REM alpha 0.00
  Deep gain 1.20

4M 0.4386 / 4K 0.2797 / 4M+4K 0.718244
Wake 0.5320 / Light 0.6697 / Deep 0.1770 / REM 0.3757
Deep precision 0.2130 / Deep recall 0.1773 / Wake+REM 0.9077
```

직전 best 대비:

```text
4M+4K +0.000564 (+0.0786%)
4 Macro +0.000768 (+0.1755%)
4 Kappa -0.000204 (-0.0730%)
Wake +0.000173 (+0.0325%)
Light -0.001024 (-0.1526%)
Deep +0.003657 (+2.1092%)
Deep precision +0.004757 (+2.2839%)
Deep recall +0.000437 (+0.2468%)
REM +0.000268 (+0.0714%)
Wake+REM +0.000441 (+0.0486%)
```

4M+4K 개선폭은 tie band보다 0.000064 크고 Wake+REM도 상승해 새 best로 채택한다.
pooled Deep 정답은 `287 -> 287`로 같고 Deep→Light는 `1,186 -> 1,187`로 1개 늘었지만,
Deep false positive가 `1,616 -> 1,555`로 61개(-3.77%) 감소해 precision 중심으로
Deep F1이 개선됐다. validation 4M+4K도 `0.665979 -> 0.672136`,
`+0.006157(+0.9245%)`로 함께 상승했다.

선택 후보는 source Deep beta 0.25와 REM beta 0.50은 내부값이지만 source Wake/Light
beta는 0이다. hybrid Wake alpha는 0.15, Deep alpha 0.85, gain 1.20이 내부이고 Light
alpha 0.55는 상단 edge다. round2에서는 새 best를 정확히 포함하면서 source beta의 작은
양수 구간과 Light alpha 0.70까지를 joint refinement한다.

classwise source blend hybrid round2는 current best reference를 정확히 재현했다.

```text
pure top:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.17_li0.55_d0.90_rem0.00_dg1.15
4M+4K 0.718325 / Deep 0.1761 / Wake+REM 0.9075

tie-rule selected:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.15_li0.55_d0.80_rem0.00_dg1.25
4M 0.4384 / 4K 0.2796 / 4M+4K 0.717992
Wake 0.5320 / Light 0.6694 / Deep 0.1762 / REM 0.3760
Deep precision 0.2095 / Deep recall 0.1789 / Wake+REM 0.9079
```

selected는 pure top보다 4M+4K가 0.000333 낮아 tie band 안이며 Wake+REM이 0.000379
더 높다. 직전 best도 pure top과 0.000081 차이로 tie band 안이지만 selected의
Wake+REM이 직전 best보다 0.000263 높아 프로젝트 규칙상 selected를 새 best로 채택한다.

직전 best 대비:

```text
4M+4K -0.000252 (-0.0351%)
4 Macro -0.000215 (-0.0490%)
4 Kappa -0.000037 (-0.0133%)
Wake -0.000019 (-0.0037%)
Light -0.000314 (-0.0469%)
Deep -0.000809 (-0.4572%)
Deep precision -0.003580 (-1.6805%)
Deep recall +0.001608 (+0.9068%)
REM +0.000283 (+0.0753%)
Wake+REM +0.000263 (+0.0290%)
```

pooled Deep 정답은 `287 -> 290`, Deep→Light는 `1,187 -> 1,184`로 각각 3개 개선됐다.
반면 Deep false positive는 `1,555 -> 1,598`로 43개 늘어 precision과 F1은 낮아졌다.
validation 4M+4K도 `0.672136 -> 0.670550`, `-0.001585(-0.2359%)`다. 이번 채택은
총점/validation 개선이 아니라 명시된 tie rule과 Wake+REM 및 Deep 정답 증가에 따른 것이다.

pure top, selected, 직전 best가 모두 같은 source beta와 Light alpha 0.55를 사용한다.
round3에서는 source를 고정하고 Wake alpha 0.15~0.175와 Deep alpha/gain ridge를 세밀
탐색하며 세 후보를 모두 정확히 포함한다.

classwise source blend hybrid round3는 current best reference를 정확히 재현했다.

```text
pure top:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.16_li0.55_d0.82_rem0.00_dg1.15
4M+4K 0.718620 / Deep 0.1759 / Wake+REM 0.9076

tie-rule selected:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.20_li0.52_d0.77_rem0.00_dg1.15
4M 0.4379 / 4K 0.2802 / 4M+4K 0.718172
Wake 0.5323 / Light 0.6708 / Deep 0.1729 / REM 0.3757
Deep precision 0.2122 / Deep recall 0.1704 / Wake+REM 0.9080
```

selected는 pure top보다 4M+4K가 0.000448 낮아 tie band 안이고 Wake+REM이 0.000415
높다. 직전 best 대비로도 4M+4K `+0.000180(+0.0251%)`, Wake+REM
`+0.000082(+0.0090%)`이므로 새 best로 채택한다.

직전 best 대비:

```text
4 Macro -0.000443 (-0.1010%)
4 Kappa +0.000623 (+0.2228%)
Wake +0.000294 (+0.0552%)
Light +0.001435 (+0.2144%)
Deep -0.003288 (-1.8659%)
Deep precision +0.002713 (+1.2954%)
Deep recall -0.008488 (-4.7442%)
REM -0.000212 (-0.0564%)
```

pooled Deep 정답은 `290 -> 277`, Deep→Light는 `1,184 -> 1,195`로 악화됐다.
Deep false positive는 `1,598 -> 1,481`로 117개 줄어 precision은 올랐지만 recall 하락이
더 커 Deep F1은 낮아졌다. validation 4M+4K는 `+0.000048(+0.0072%)`로 사실상 같다.

tie band 내 Deep top은 `w0.20/li0.525/d0.875/gain1.20`으로 4M+4K 0.718598,
Deep 0.1772, Wake+REM 0.907883이다. selected보다 Wake+REM이 0.000119 낮을 뿐이면서
Deep 정답은 `277 -> 287`로 10개 많다. round4에서는 selected, pure top, 이 Deep top
사이의 좁은 ridge를 한 번 더 조밀하게 탐색한다.

classwise source blend hybrid round4는 current best reference를 정확히 재현했다.

```text
pure top:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.18_li0.55_d0.81_rem0.00_dg1.15
4M+4K 0.718962 / Deep 0.1763 / Wake+REM 0.9079

tie-rule selected:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.21_li0.52_d0.86_rem0.00_dg1.20
4M 0.4388 / 4K 0.2800 / 4M+4K 0.718759
Wake 0.5324 / Light 0.6697 / Deep 0.1773 / REM 0.3756
Deep precision 0.2139 / Deep recall 0.1773 / Wake+REM 0.9080
```

selected는 pure top보다 4M+4K가 0.000204 낮아 tie band 안이고 Wake+REM이 0.000172
높다. 직전 best 대비 4M+4K `+0.000587(+0.0817%)`, Wake+REM
`+0.000028(+0.0031%)`이므로 새 best로 채택한다.

직전 best 대비:

```text
4 Macro +0.000826 (+0.1886%)
4 Kappa -0.000239 (-0.0854%)
Wake +0.000139 (+0.0260%)
Light -0.001060 (-0.1580%)
Deep +0.004336 (+2.5074%)
Deep precision +0.001690 (+0.7966%)
Deep recall +0.006880 (+4.0371%)
REM -0.000111 (-0.0294%)
```

pooled Deep 정답은 `277 -> 287`, Deep→Light는 `1,195 -> 1,185`로 10개씩 개선됐다.
Deep false positive는 `1,481 -> 1,553`으로 늘었지만 3-seed 평균 precision/recall/F1은
모두 상승했다. validation 4M+4K는 `-0.000198(-0.0296%)`로 소폭 낮아졌다.

tie band 내 Deep top은 `w0.20625/li0.525/d0.875/gain1.1875`로 4M+4K 0.718872,
Deep 0.1776, Wake+REM 0.907914다. selected와의 차이가 작으므로 round5에서 round4
간격을 절반으로 줄여 세 ridge를 마지막으로 확인한다. 그 뒤에는 static grid를 중단하고
Deep/Light specialist 학습 같은 모델 수준 변경으로 돌아간다.

classwise source blend hybrid round5는 round4 best reference를 정확히 재현했고,
pure top과 tie-rule selected가 같은 후보로 수렴했다.

```text
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.19_li0.54_d0.82_rem0.00_dg1.15
exact: w0.1875 / li0.54375 / d0.81875 / rem0 / gain1.15

4M 0.438794 / 4K 0.280527 / 4M+4K 0.719320
Wake 0.532415 / Light 0.670531 / Deep 0.176229 / REM 0.376000
Deep precision 0.215376 / Deep recall 0.173968 / Wake+REM 0.908415
validation 4M+4K 0.671430
```

직전 best 대비 4M+4K `+0.000562(+0.0781%)`, 4K `+0.000526(+0.1880%)`,
Wake+REM `+0.000385(+0.0424%)`, validation `+0.001030(+0.1536%)`다.
Light는 `+0.1191%`, Deep은 `-0.5871%`다. pooled Deep 정답은 5개 줄고
Deep→Light는 5개 늘었지만 Deep false positive는 64개 감소했다. 총점, Wake+REM,
validation이 모두 상승해 새 best로 채택하고 static hybrid ridge는 종료한다.

결과:

```text
/Users/chan/Downloads/fusion4_direct4_classwise_source_blend_hybrid_round5_context20_h64_summary.json
```

Light-vs-Deep specialist fusion은 static round5 best를 정확히 재현했고,
`original_h128_ce`가 다른 feature/loss/capacity 조합보다 명확히 우세했다.

```text
pure top = tie-rule selected = best Deep:
original_h128_ce__beta1.00_scale0.50_bias0.50

test:
4M 0.455784 / 4K 0.287174 / 4M+4K 0.742958
Wake 0.534681 / Light 0.659505 / Deep 0.249862 / REM 0.379089
Deep precision 0.220270 / Deep recall 0.292376 / Wake+REM 0.913770

validation:
4M+4K 0.661729
```

test 4M+4K는 `+0.023638(+3.2861%)`, Deep은 `+41.7825%`, Deep recall은
`+68.0633%` 상승했다. validation 총점은 `-1.4448%`지만 같은 specialist의
`beta0.80/scale1.00/bias-0.50`은 validation 0.701594 / test 0.738751로 양쪽에서
강했다. pure top이 scale 하단과 bias 상단, beta 상단에 있으므로 재학습 없이
calibration edge 및 cross-split robust ridge를 함께 refinement한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_specialist_fusion_context20_summary.json
```

Light-vs-Deep specialist calibration refinement는 specialist current best를 정확히
재현했고 pure top과 tie-rule selected가 같은 후보로 수렴했다.

```text
original_h128_ce__beta1.00_scale0.55_bias0.25

test:
4M 0.454992 / 4K 0.289715 / 4M+4K 0.744707
Wake 0.533366 / Light 0.671163 / Deep 0.235966 / REM 0.379474
Deep precision 0.232553 / Deep recall 0.243294 / Wake+REM 0.912840

validation:
4M+4K 0.682090
```

직전 best 대비 test 총점은 `+0.001749(+0.2354%)`, validation 총점은
`+0.020361(+3.0769%)`다. Deep recall을 일부 되돌리는 대신 Light→Deep 및
Deep false positive를 크게 줄여 Light와 Kappa가 회복됐다. selected는 beta 상단이지만
scale/bias는 내부값이다. round2에서는 scale 0.50~0.60과 bias 0.10~0.40을
조밀하게 탐색하고, 개선이 멈추면 specialist multi-init ensemble로 전환한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_specialist_fusion_refine_context20_summary.json
```

Light-vs-Deep specialist calibration refinement round2는 current best를 정확히
재현했고, pure top과 tie-rule selected가 분리됐다.

```text
pure top:
original_h128_ce__beta0.97_scale0.60_bias0.30
4M+4K 0.745184 / Deep 0.2370 / Wake+REM 0.9122

tie-rule selected:
original_h128_ce__beta1.00_scale0.54_bias0.25
exact: beta 1.00 / scale 0.5375 / bias 0.25
4M 0.455120 / 4K 0.289828 / 4M+4K 0.744948
Wake 0.533355 / Light 0.670579 / Deep 0.236444 / REM 0.380103
Deep precision 0.232019 / Deep recall 0.244902 / Wake+REM 0.913458
validation 4M+4K 0.681499
```

selected는 pure top보다 0.000236 낮아 tie band 안이고 Wake+REM이 0.001286 높다.
직전 best 대비 총점 `+0.000241(+0.0324%)`, Wake+REM `+0.0677%`, Deep
`+0.2026%`다. validation은 `-0.0865%`로 사실상 같은 수준이다. fine grid의
개선폭이 매우 작아 calibration을 종료하고 specialist initialization diversity와
probability ensemble을 시험한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_specialist_fusion_refine_round2_context20_summary.json
```

Light-vs-Deep specialist same-split multi-init ensemble은 current best를 정확히
재현했지만, 2,059개 전체 후보의 pure top, tie-rule selected,
best Deep within tie band가 모두 기존 single specialist였다.

```text
current best:
single__beta1.00_scale0.54_bias0.25
4M 0.455120 / 4K 0.289828 / 4M+4K 0.744948
Deep 0.236444 / Wake+REM 0.913458 / validation 0.681499

best new initialization:
init2002__beta0.95_scale0.25_bias0.00
4M 0.446990 / 4K 0.283063 / 4M+4K 0.730053
Deep 0.225530 / Wake+REM 0.923423 / validation 0.632429

ensemble6 top:
ensemble6__beta0.95_scale0.25_bias0.00
4M 0.439792 / 4K 0.283504 / 4M+4K 0.723296
Deep 0.187677 / Wake+REM 0.921416 / validation 0.652547
```

best new initialization은 current 대비 총점 `-0.014895(-1.9995%)`, Deep
`-4.6159%`, validation `-7.2003%`다. ensemble6는 총점
`-0.021652(-2.9066%)`, Deep `-20.6251%`, validation `-4.2483%`다.
ensemble6의 Deep precision은 `+20.2701%`지만 recall이 `-35.3643%`로
무너져 Deep 정답이 `404 -> 260`(-144)으로 줄었다. Light→Deep은
`1,153 -> 798`(-355), Deep false positive는 `1,515 -> 1,021`(-494)로
감소했지만 지나치게 보수적인 Deep 판정 때문에 전체 성능이 하락했다.

전체 Deep F1 top은 기존 single의 다른 calibration으로 Deep이
`0.236444 -> 0.259586`(+9.7877%)였지만 4M+4K가
`0.744948 -> 0.726153`(-2.5230%)이라 tie band 밖이다. 따라서 current best는
유지한다. 단순 평균은 중단하고, validation subject OOF logistic stacking으로
single의 강한 신호를 보존하면서 replica의 상보적 신호만 가중 학습한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_specialist_same_split_init_ensemble_context20_summary.json
```

Light-vs-Deep specialist subject-OOF logistic stacking도 current best를 정확히
재현했지만, 5,587개 전체 후보의 pure top, tie-rule selected,
best Deep within tie band는 다시 모두 기존 single specialist였다.

```text
best OOF stack:
stack_none_c0.003__beta0.80_scale0.54_bias1.00
4M 0.438731 / 4K 0.282067 / 4M+4K 0.720798
Wake 0.537058 / Light 0.650419 / Deep 0.183843 / REM 0.383605
Deep precision 0.261075 / Deep recall 0.154651
Wake+REM 0.920663 / OOF validation 0.595824
```

current 대비 4M+4K는 `-0.024150(-3.2418%)`, Deep은 `-22.2464%`,
OOF validation은 `-12.5716%`다. Wake+REM은 `+0.7887%`, Deep precision은
`+12.5231%`지만 Deep recall이 `-36.8518%`로 무너졌다. pooled Deep 정답은
`404 -> 252`(-152), Deep→Light는 `1,035 -> 1,172`(+137)이다.
Light→Deep과 Deep false positive는 각각 `1,153 -> 774`(-379),
`1,515 -> 1,021`(-494)로 줄었지만, multi-init average와 마찬가지로 Deep을
지나치게 보수적으로 판정했다.

강한 single을 포함해도 모든 OOF stack이 current와 큰 차이로 뒤졌고 validation/test가
같은 하락 방향이므로 replica 결합 계열은 종료한다. 다음은 실제 N3 오답의 63.74%를
차지한 N2 경계를 직접 겨냥해 N1을 제외한 N2-vs-N3 specialist를 학습한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_specialist_oof_stacking_context20_summary.json
```

N2-vs-N3 hard-boundary specialist도 current best를 정확히 재현했지만,
3,235개 전체 후보의 pure top, tie-rule selected, best Deep within tie band는
모두 기존 single specialist였다.

```text
best N2-vs-N3 by 4M+4K:
n2n3_h256_inverse_lstm2__beta0.65_scale0.50_bias0.25
4M+4K 0.727870 / Deep 0.174567 / validation 0.669919

best N2-vs-N3 Deep among source-selected candidates:
n2n3_h256_inverse_lstm__beta0.65_scale0.50_bias0.75
4M+4K 0.727066 / Deep 0.214710 / validation 0.655249
```

N2/N3 h256 2-layer의 총점은 current 대비 `-2.2925%`, Deep은 `-26.1698%`다.
N2/N3 중 Deep이 가장 높은 h256 1-layer도 Deep `-9.1921%`, 총점
`-2.4004%`다. N1을 제외하면 N2/N3 학습 지표와 최종 4-class fusion 사이의
분포 차이가 커져 current보다 나빠졌으므로 hard-boundary 계열은 종료한다.

한편 Light-vs-Deep architecture control인 h256 2-layer는 가장 강한 새 source였다.

```text
light_h256_inverse_lstm2__beta0.90_scale0.75_bias-0.50
4M 0.447877 / 4K 0.289069 / 4M+4K 0.736946
Wake 0.532924 / Light 0.680687 / Deep 0.204965 / REM 0.372933
Deep precision 0.265374 / Deep recall 0.181977 / Wake+REM 0.905858
validation 4M+4K 0.697489
```

current 대비 test 총점은 `-0.008002(-1.0742%)`, Deep은 `-13.3136%`,
Wake+REM은 `-0.8321%`지만 Light `+1.5074%`, Deep precision `+14.3757%`,
validation 총점 `+2.3462%`다. pooled Deep 정답은 `404 -> 309`(-95)지만
Light→Deep은 `1,153 -> 784`(-369), Deep false positive는
`1,515 -> 991`(-524)로 크게 줄었다. current h128보다 보수적이지만 validation과
precision 측면의 상보성이 있으므로 두 specialist를 소량부터 pairwise blend한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_n2n3_specialist_context20_summary.json
```

Light-vs-Deep h128+h256 pairwise blend는 current best를 정확히 재현했고,
4,999개 전체 후보의 pure top, tie-rule selected, tie-band Deep top이 모두
아래 후보로 일치했다.

```text
blend_a0600__beta1.00_scale0.75_bias0.25
member probability: 0.40 original-h128-CE + 0.60 Light-h256-2layer-CE

test:
4M 0.457428 / 4K 0.291478 / 4M+4K 0.748906
Wake 0.535473 / Light 0.671380 / Deep 0.247912 / REM 0.374946
Deep precision 0.252908 / Deep recall 0.255870 / Wake+REM 0.910419

validation:
4M 0.427190 / 4K 0.259557 / 4M+4K 0.686747
Deep 0.237390 / Deep precision 0.197982 / Deep recall 0.333658
Wake+REM 0.821438
```

직전 best 대비 test 4M+4K는 `+0.003958(+0.5313%)`, 4M은
`+0.002308(+0.5071%)`, 4K는 `+0.001650(+0.5693%)`다. Deep은
`+0.011468(+4.8504%)`, precision은 `+9.0029%`, recall은 `+4.4785%`로
동시에 개선됐다. Wake와 Light도 각각 `+0.3971%`, `+0.1195%` 올랐지만
REM은 `-1.3567%`, Wake+REM은 `-0.3327%`다. 총점 차이
`0.003958`은 tie band `0.0005`보다 크므로 새 후보를 선택한다.

pooled confusion matrix에서 Deep 정답은 `404 -> 433`(+29),
Deep→Light는 `1,035 -> 1,018`(-17), Light→Deep은
`1,153 -> 1,094`(-59), Deep false positive는 `1,515 -> 1,393`(-122)다.
validation 4M+4K도 `+0.005247(+0.7700%)`, validation Deep은
`+12.3256%`, precision은 `+13.9687%`, recall은 `+16.3633%`로 test와
같은 개선 방향이다.

alpha별 source-selected 총점은 h256 alpha 0.60이 `0.748906`으로 가장 높고,
0.40은 `0.747095`, 0.50은 `0.745172`, 0.70은 `0.742833`이다. 최적점이
alpha 0.60에 있고 인접 alpha에서 하락하므로 0.50~0.70 구간과
beta/scale/bias 주변을 더 촘촘히 탐색한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_h128_h256_pair_blend_context20_summary.json
```

pair-blend refinement는 current best를 정확히 재현했다. 2,917개 후보의
pure top과 tie-band Deep top은 같고, tie-rule selected는 별도 후보다.

```text
pure top / tie-band Deep top:
blend_a0600__beta1.00_scale0.78_bias0.25
exact scale 0.775
4M 0.457566 / 4K 0.291664 / 4M+4K 0.749230
Wake 0.535150 / Light 0.672245 / Deep 0.247937 / REM 0.374933
Deep precision 0.254194 / Deep recall 0.254796 / Wake+REM 0.910083
validation 4M+4K 0.687713

tie-rule selected:
blend_a0600__beta0.97_scale0.75_bias0.25
exact beta 0.975
4M 0.457222 / 4K 0.291656 / 4M+4K 0.748878
Wake 0.535559 / Light 0.671910 / Deep 0.246315 / REM 0.375104
Deep precision 0.253068 / Deep recall 0.252649 / Wake+REM 0.910663
validation 4M+4K 0.686888
```

selected는 pure top보다 4M+4K가 `0.000352` 낮아 tie band 안이고,
Wake+REM은 `0.000580` 높다. 직전 best와 비교해도 4M+4K는
`-0.000028(-0.0037%)`로 tie band 안이고 Wake+REM은
`+0.000243(+0.0267%)` 높으므로 프로젝트 규칙에 따라 새 best로 채택한다.

직전 best 대비 selected의 4K는 `+0.0611%`, Wake는 `+0.0160%`,
Light는 `+0.0789%`, REM은 `+0.0421%`, Deep precision은 `+0.0632%`다.
반면 4M은 `-0.0451%`, Deep은 `-0.6443%`, Deep recall은 `-1.2587%`다.
pooled Deep 정답은 `433 -> 427`(-6), Deep→Light는
`1,018 -> 1,024`(+6), Light→Deep은 `1,094 -> 1,081`(-13),
Deep false positive는 `1,393 -> 1,378`(-15)다. validation 총점은
`+0.000141(+0.0205%)`다.

pure top은 직전 best 대비 총점 `+0.000324(+0.0433%)`, validation 총점
`+0.1406%`, Deep `+0.0098%`지만 Wake+REM은 `-0.0370%`다.
alpha 0.60이 다시 최적 source였고 두 경쟁점 모두 calibration 내부에 있으므로,
alpha 0.575~0.625와 beta/scale/bias 주변을 한 번 더 미세 탐색한다.

결과:

```text
/Users/chan/Downloads/fusion4_light_deep_h128_h256_pair_blend_refine_context20_summary.json
```

App-oriented Light-vs-rest objective audit은 9개 config와 33개 threshold,
총 297개 후보를 validation 3-seed 평균으로만 선택했다.

```text
validation-selected best:
multitask_h256_lstm2_deep2_aux025__threshold0.100

validation mean:
Light objective 0.995333 / Light F1 0.784728 / binary Kappa 0.210604
precision 0.687818 / recall 0.913931 / Deep→Light 0.976134

test mean:
Light objective 0.902751 / Light F1 0.750249 / binary Kappa 0.152501
precision 0.634542 / recall 0.921754 / Deep→Light 0.947207

test pooled:
Light objective 0.901313 / Light F1 0.750424 / binary Kappa 0.150889
precision 0.633126 / recall 0.921068 / Deep→Light 0.944807
```

기존 4-class current argmax pooled baseline과 비교하면 Light F1은
`+0.078565(+11.6937%)`, recall은 `+0.277315(+43.0780%)`이지만,
binary Kappa는 `-0.086526(-36.4450%)`, precision은
`-0.069406(-9.8794%)`, Light objective는 `-0.007961(-0.8755%)`다.
Deep→Light는 `0.607715 -> 0.944807`(+55.4688%), Wake→Light는
`+109.2282%`, REM→Light는 `+96.5762%`로 악화됐다. 즉 모델이
Light를 정교하게 분리한 것이 아니라 대부분의 epoch를 Light로 보내 F1/recall을
높였다.

validation Deep 누출 제한 profile도 test로 안정적으로 이전되지 않았다.

```text
validation limit 0.10: val 0.0544 -> test 0.2542 / test Light recall 0.1452
validation limit 0.20: val 0.1807 -> test 0.3240 / test Light recall 0.2542
validation limit 0.30: val 0.2780 -> test 0.4602 / test Light recall 0.3519
validation limit 0.40: val 0.3977 -> test 0.5596 / test Light recall 0.4458
```

Deep multiplier 1/2/4와 4-class auxiliary multitask만으로는 useful operating
point를 만들지 못했다. direct Light-vs-rest 단독 모델은 새 app best로 채택하지
않는다. 다만 Light proposal recall은 강하므로 이를 버리지 않고, 기존 current의
`P(Deep | Light,Deep)`를 explicit veto로 결합해 Deep 누출을 회수할 수 있는지
확인한다.

결과:

```text
/Users/chan/Downloads/fusion_light_alarm_objective_context20_summary.json
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
scripts/run_four_model_flex4_kappa_refinement_round7_colab.sh
scripts/run_four_model_flex4_kappa_refinement_round8_colab.sh
scripts/run_four_model_oracle_audit_colab.sh
scripts/run_four_model_causal_gate_colab.sh
scripts/run_four_model_same_split_init_ensemble_colab.sh
scripts/run_four_model_direct_4class_colab.sh
scripts/run_four_model_deep_probability_audit_colab.sh
scripts/run_four_model_deep_temporal_probability_audit_colab.sh
scripts/run_four_model_direct4_hybrid_deep_refinement_colab.sh
scripts/run_four_model_direct4_hybrid_deep_refinement_round2_colab.sh
scripts/run_four_model_direct4_hybrid_deep_refinement_round3_colab.sh
scripts/run_four_model_direct4_hybrid_deep_refinement_round4_colab.sh
scripts/run_direct4_original_same_split_init_ensemble_hybrid_colab.sh
scripts/run_direct4_classwise_source_blend_hybrid_colab.sh
scripts/run_direct4_classwise_source_blend_hybrid_round2_colab.sh
scripts/run_direct4_classwise_source_blend_hybrid_round3_colab.sh
scripts/run_direct4_classwise_source_blend_hybrid_round4_colab.sh
scripts/run_direct4_classwise_source_blend_hybrid_round5_colab.sh
scripts/run_light_deep_specialist_fusion_colab.sh
scripts/run_light_deep_specialist_fusion_refinement_colab.sh
scripts/run_light_deep_specialist_fusion_refinement_round2_colab.sh
scripts/run_light_deep_specialist_same_split_init_ensemble_colab.sh
scripts/run_light_deep_specialist_oof_stacking_colab.sh
scripts/run_light_deep_n2n3_specialist_colab.sh
scripts/run_light_deep_h128_h256_pair_blend_colab.sh
scripts/run_light_deep_h128_h256_pair_blend_refinement_colab.sh
scripts/run_light_deep_h128_h256_pair_blend_refinement_round2_colab.sh
scripts/run_light_alarm_objective_colab.sh
scripts/run_light_alarm_deep_veto_fusion_colab.sh
src/sse_sleep/train_light_deep_specialist.py
src/sse_sleep/train_light_alarm.py
src/sse_sleep/evaluate_light_alarm.py
src/sse_sleep/evaluate_light_alarm_deep_veto.py
src/sse_sleep/evaluate_light_deep_specialist_fusion.py
src/sse_sleep/average_light_deep_specialist_ensemble.py
src/sse_sleep/stack_light_deep_specialists.py
```

기능:

```text
Light(N1/N2)와 Deep(N3)을 분리해서 4-model flexible fusion weight를 탐색한다.
Light-vs-Deep specialist를 별도로 학습하고 current best의 Light+Deep 총질량 안에서
calibrated P(Deep | Light 또는 Deep)만 blend한다.
specialist trainer는 Light-vs-Deep과 N2-vs-N3 negative mode, LSTM/GRU를 지원한다.
specialist average utility는 명시적 member weight를 지원한다.
specialist evaluator는 global selection과 source별 selection을 함께 archive한다.
Light alarm trainer는 전체 Wake/Light/Deep/REM epoch를 사용해 Light-vs-rest를 학습한다.
Deep/Wake/REM negative multiplier, stage-balanced sampler, 4-class auxiliary head를 지원한다.
Light alarm evaluator는 validation에서만 threshold/model을 선택하고 test는 보고에만 쓴다.
Light F1/binary Kappa와 Wake/Deep/REM→Light 누출률, Deep 누출 제한별 profile을 기록한다.
Deep-veto evaluator는 direct Light proposal과 current P(Light)를 logit blend하고,
current P(Deep|Light,Deep)를 곱셈 veto로 적용해 validation-only grid를 평가한다.
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
src/sse_sleep/average_prediction_ensemble_4class.py
src/sse_sleep/train_lstm_4class.py
src/sse_sleep/evaluate_four_model_4class_fusion.py
src/sse_sleep/evaluate_deep_probability_audit.py
src/sse_sleep/evaluate_deep_temporal_probability_audit.py
src/sse_sleep/evaluate_direct4_hybrid_deep_fusion.py
src/sse_sleep/evaluate_direct4_source_blend_hybrid.py
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
direct4 summary에는 role/fusion별 Deep precision/recall과 outer 3-seed pooled 4-class confusion matrix도
저장해 Deep→Light 및 Light→Deep 오답을 바로 확인할 수 있다.

Deep probability audit는 same-split role ensemble NPZ를 재사용해 학습 없이 실행한다. 각 role과 current
fusion의 N3-vs-non-N3 ROC-AUC/Average Precision을 측정하고, outer validation split에서 최대 F1 및
Deep recall 50/70/80/90% floor별 threshold를 선택한 뒤 해당 outer test split에 그대로 적용한다.
결과에는 precision/recall/specificity/F1/predicted-positive-rate와 score 분포를 저장한다.

Deep temporal probability audit는 current fusion N3 score에 raw, causal mean 3/5/10 epoch,
causal EMA alpha 0.20/0.40/0.60/0.80을 적용한다. subject 변경 및 epoch gap에서 history를 초기화하며,
각 temporal variant도 validation에서만 threshold를 선택한 뒤 test에 고정 적용한다. 전체 epoch
precision/recall 외에 Deep run detection rate, 첫 epoch/첫 2 epoch recall, 최초 감지 지연도 저장한다.

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
Light proposal + current Deep-veto fusion audit
```

목적:

```text
direct Light-vs-rest의 높은 Light recall을 proposal로 활용한다.
기존 current staging의 P(Deep|Light,Deep)를 explicit veto로 적용한다.
Light recall 이득을 유지하면서 Deep→Light를 baseline 이하로 낮출 수 있는지 본다.
재학습 없이 기존 prediction만 사용해 두 모델의 상보성을 먼저 audit한다.
```

융합/평가 범위:

```text
outer seed: 42 / 7 / 123
alarm proposal: 이전 9개 Light-vs-rest config
current staging: 32-checkpoint 4-class current best
proposal:
  current P(Light)와 alarm P(Light)를 logit blend
alarm alpha: 0 / 0.25 / 0.50 / 0.75 / 1.00
Deep veto:
  proposal x (1 - current P(Deep|Light,Deep))^gamma
veto gamma: 0 / 0.50 / 1.00 / 2.00 / 4.00
threshold: 0.10~0.90, 0.05 간격
총 3,825 candidates
selection: validation 3-seed 평균만 사용
primary constraint:
  validation Deep→Light <= current baseline 0.528169
  constraint 안에서 Light objective 최대
```

Colab 실행:

```bash
%cd /content/SSE
!git pull
!bash scripts/run_light_alarm_deep_veto_fusion_colab.sh
```

결과 summary JSON:

```text
/content/drive/MyDrive/SSE_outputs/fusion_light_alarm_deep_veto_context20_summary.json
```

비교 포인트:

```text
1. current baseline Deep 누출 constraint 안에서 선택 후보가 존재하는지
2. selected test Light F1/Kappa/precision/recall/Deep→Light
3. 기존 current argmax baseline과 모든 metric 절대/상대 변화
4. alarm alpha와 veto gamma가 0 또는 1 boundary인지 interior인지
5. Deep→Light <= 0.10/0.20/0.30/0.40 profile
6. validation constraint가 test에서도 유지되는지
7. proposal config별 상보성과 h128/h256, direct/multitask 차이
```

direct Light-vs-rest 단독 모델은 app best로 채택하지 않는다.
기존 4-class current argmax binary 성능을 app baseline으로 유지한다.

## 다음 채팅방 시작 프롬프트

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.
4-class fusion refinement는 잠시 멈추고 앱 목적 중심 Light-vs-rest 트랙을 시작했어.
기존 4-class best는 기존 30-checkpoint static hybrid + h128/h256 specialists:
blend_a0600__beta0.97_scale0.75_bias0.25
exact beta는 0.975다.
3-seed 평균은 4M 0.4572 / 4K 0.2917 / 4M+4K 0.7489,
Wake 0.5356 / Light 0.6719 / Deep 0.2463 / REM 0.3751 / Wake+REM 0.9107다.
기존 current argmax binary baseline은 pooled Light F1 0.671859,
binary Kappa 0.237415, precision 0.702532, recall 0.643752,
Deep→Light 0.607715다.
direct Light-vs-rest best는 Light F1 +11.6937%, recall +43.0780%였지만
binary Kappa -36.4450%, precision -9.8794%, Deep→Light +55.4688%라 채택하지 않았다.
validation Deep 누출 제한도 test에서 모두 초과해 단독 binary/multitask는 unsafe했다.
다음은 direct Light probability를 proposal로 쓰고 current
P(Deep|Light,Deep)를 explicit veto로 적용하는 audit다.
Colab에서는 git pull 후
scripts/run_light_alarm_deep_veto_fusion_colab.sh를 실행하면 돼.
결과 JSON을 받으면 current Deep 누출 constraint 내 selected, safe profile,
baseline 대비 모든 metric 변화, alpha/gamma ridge, validation-test constraint 이전을
비교하고 앱용 새 best와 다음 방향을 결정해줘.
```
