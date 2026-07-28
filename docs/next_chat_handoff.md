# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 24-checkpoint current ensemble + classwise-blended direct4 6 checkpoints:
source_w0.00_li0.00_d0.25_rem0.50__hybrid_w0.19_li0.54_d0.82_rem0.00_dg1.15
exact hybrid alpha는 Wake 0.1875 / Light 0.54375 / Deep 0.81875 / REM 0,
Deep gain 1.15다.

3-seed 평균:
4M 0.4388 / 4K 0.2805 / 4M+4K 0.7193
Wake 0.5324 / Light 0.6705 / Deep 0.1762 / REM 0.3760 / Wake+REM 0.9084

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

직전 best 대비 4M+4K +0.0781%, Deep -0.5871%, Wake+REM +0.0424%다.
round5 pure top과 tie-rule selected가 같고 validation도 상승해 새 best로 채택했다.
Deep 정답은 287->282, Deep->Light는 1185->1190으로 악화됐고 Deep FP는 64개 감소했다.

다음 실험은 Light-vs-Deep binary specialist conditional fusion이야.
현재 best의 Wake/REM 및 Light+Deep 총질량을 보존하고 내부 Deep 조건부 확률만 보정한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_light_deep_specialist_fusion_colab.sh

결과 summary JSON을 받으면 current best 정확 재현, feature/loss/capacity별 specialist,
pure top/tie-rule selected, current best 대비 모든 metric의 절대/상대 변화율과
Light/Deep confusion 변화를 비교하고 새 best 및 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
