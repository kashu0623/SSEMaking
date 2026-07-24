# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 24-checkpoint same-split ensemble + original direct4 hybrid:
hybrid_w0.31_li0.34_d0.85_rem0.00_dg1.20

3-seed 평균:
4M 0.4378 / 4K 0.2799 / 4M+4K 0.7177
Wake 0.5318 / Light 0.6707 / Deep 0.1734 / REM 0.3754 / Wake+REM 0.9072

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

round3 best 대비 4M+4K +0.0012%, Deep -1.8444%, Wake+REM +0.0986%다.
round4 pure top과 selected 차이는 0.000374이고 selected의 Wake+REM이 더 높아 tie rule로 채택했다.
static grid는 포화되어 중단한다.

6-checkpoint direct4 평균은 single 대비 단독 4M+4K +1.4886%, Kappa +5.0822%였지만
Deep F1이 -33.4829%로 붕괴했다. 최상 hybrid도 current best 대비 4M+4K -2.2964%,
Deep -24.0839%라 채택하지 않았고 current best를 유지한다.

다음 실험은 single/6-checkpoint direct4 classwise source blend + hybrid recalibration이야.
Wake/Light/Deep/REM마다 single과 ensemble의 혼합 beta를 따로 탐색하고, 유망 source만 hybrid한다.
기존 single source와 current best weight도 정확히 포함한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_direct4_classwise_source_blend_hybrid_colab.sh

결과 summary JSON을 받으면 current best 정확 재현, source beta, pure top/tie-rule selected,
current best 대비 4M+4K/Wake+REM/Light/Deep의 절대/상대 변화율과 Deep confusion 변화를
비교하고 새 best 및 다음 방향을 정한 뒤 docs/current_progress_summary.md를 갱신해줘.
```
