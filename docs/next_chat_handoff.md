# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 4-model stage-split flexible fusion:
classwise4_w_p0.77_c0.04_l0.00_li_p0.78_c0.02_l0.17_d_p0.76_c0.01_l0.18_rem_p0.00_c0.44_l0.11

3-seed 평균:
4M 0.4142 / 4K 0.2571 / Wake 0.5086 / Light 0.6414 / Deep 0.1244 / REM 0.3822

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

다음 실험은 round5 best와 pure top 사이를 같이 덮는 flex4_stage_refine_round6부터 진행해줘.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_four_model_flex4_stage_refinement_round6_colab.sh

결과 summary JSON을 받으면 current best 대비 4M+4K, Wake+REM, Light/Deep/REM 변화를 비교하고,
새 best 채택 여부를 판단한 뒤 docs/current_progress_summary.md를 갱신해줘.
```
