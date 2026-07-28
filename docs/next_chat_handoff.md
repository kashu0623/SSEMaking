# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 24-checkpoint current ensemble + classwise-blended direct4 6 checkpoints:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.21_li0.52_d0.86_rem0.00_dg1.20

3-seed 평균:
4M 0.4388 / 4K 0.2800 / 4M+4K 0.7188
Wake 0.5324 / Light 0.6697 / Deep 0.1773 / REM 0.3756 / Wake+REM 0.9080

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

직전 best 대비 4M+4K +0.0817%, Deep +2.5074%, Wake+REM +0.0031%다.
round4 pure top보다 총점이 0.000204 낮아 tie band 안이고 Wake+REM이 더 높아 새 best로 채택했다.
Deep 정답은 277->287, Deep->Light는 1195->1185로 개선됐다.

다음 실험은 마지막 static hybrid ridge refinement round5야.
source를 고정하고 round4 selected/pure top/tie-band Deep top 사이를 반 간격으로 탐색한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_direct4_classwise_source_blend_hybrid_round5_colab.sh

결과 summary JSON을 받으면 current best 정확 재현, source beta/hybrid alpha, pure top/tie-rule selected,
current best 대비 4M+4K/Wake+REM/Light/Deep의 절대/상대 변화율과 Deep confusion 변화를
비교하고 새 best 및 다음 방향을 정한 뒤 docs/current_progress_summary.md를 갱신해줘.
```
