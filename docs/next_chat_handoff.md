# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 기존 30-checkpoint static hybrid + original-h128-CE Light/Deep specialist:
original_h128_ce__beta1.00_scale0.54_bias0.25
exact scale은 0.5375다.
outer split 하나당 총 31 checkpoints다.

3-seed 평균:
4M 0.4551 / 4K 0.2898 / 4M+4K 0.7449
Wake 0.5334 / Light 0.6706 / Deep 0.2364 / REM 0.3801 / Wake+REM 0.9135

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

직전 best 대비 4M+4K +0.0324%, Deep +0.2026%, Wake+REM +0.0677%이며
validation 총점은 -0.0865%다.

다음 실험은 Light-vs-Deep specialist same-split multi-init ensemble이야.
기존 single, initialization replica 5개 각각, 6-checkpoint ensemble을 비교한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_light_deep_specialist_same_split_init_ensemble_colab.sh

결과 summary JSON을 받으면 current best 정확 재현, single/init별/ensemble6 source,
pure top/tie-rule selected, current best 대비 모든 metric의 절대/상대 변화율과
Light/Deep confusion 변화를 비교하고 새 best 및 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
