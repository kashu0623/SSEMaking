# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 benchmark best는 4-role x 6-checkpoint same-split ensemble stage-split flexible fusion:
classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13

3-seed 평균:
4M 0.4151 / 4K 0.2649 / Wake 0.5131 / Light 0.6740 / Deep 0.1000 / REM 0.3731

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

현재 ensemble은 실제 N3 1,685개 중 1,290개(76.56%)를 Light로 오인한다.
Deep probability audit에서 fusion test ROC-AUC 0.7362 / AP 0.1055로 신호는 남아 있지만,
validation recall 90% threshold가 test recall 78.75% / specificity 55.45% / positive-rate 46.17%여서
raw threshold는 alarm veto로 채택하지 않았다.

다음 실험은 current ensemble의 causal temporal Deep probability audit야.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_four_model_deep_temporal_probability_audit_colab.sh

결과 summary JSON을 받으면 raw 대비 causal mean/EMA의 ROC-AUC/AP와 validation-selected threshold의
test Deep precision/recall/specificity/F1을 분석하고, hysteresis/ensemble Deep weight refinement/
direct4/binary specialist 중 다음 방향을 정한 뒤 docs/current_progress_summary.md를 갱신해줘.
```
