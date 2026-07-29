# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 기존 30-checkpoint static hybrid + h128/h256 Light/Deep specialists:
blend_a0600__beta0.97_scale0.75_bias0.25
exact beta는 0.975다.
member probability는 0.40 original-h128-CE + 0.60 Light-h256-2layer-CE다.
outer split 하나당 총 32 checkpoints다.

3-seed 평균:
4M 0.4572 / 4K 0.2917 / 4M+4K 0.7489
Wake 0.5356 / Light 0.6719 / Deep 0.2463 / REM 0.3751 / Wake+REM 0.9107

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

직전 best 대비 4M+4K -0.0037%, Wake+REM +0.0267%라 tie 규칙으로 채택했다.
4K +0.0611%, Light +0.0789%, REM +0.0421%다.
Deep -0.6443%, Deep precision +0.0632%, Deep recall -1.2587%다.
validation 4M+4K는 +0.0205%다.

pure top은 alpha0.60/beta1.00/scale0.775/bias0.25로 4M+4K 0.749230이지만
selected보다 Wake+REM이 0.000580 낮다.

다음 실험은 selected와 pure top 주변 pair-blend refinement round2야.
h256 alpha 0.575~0.625와 beta/scale/bias를 더 촘촘히 탐색한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_light_deep_h128_h256_pair_blend_refinement_round2_colab.sh

결과 summary JSON을 받으면 current best 정확 재현, alpha별 source,
pure top/tie-rule selected, current best 대비 모든 metric의 절대/상대 변화율과
Light/Deep confusion 및 REM/Wake+REM 변화를 비교하고 새 best 및 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
