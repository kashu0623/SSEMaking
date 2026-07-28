# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 기존 30-checkpoint static hybrid + original-h128-CE Light/Deep specialist:
original_h128_ce__beta1.00_scale0.50_bias0.50
outer split 하나당 총 31 checkpoints다.

3-seed 평균:
4M 0.4558 / 4K 0.2872 / 4M+4K 0.7430
Wake 0.5347 / Light 0.6595 / Deep 0.2499 / REM 0.3791 / Wake+REM 0.9138

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

직전 best 대비 4M+4K +3.2861%, Deep +41.7825%, Wake+REM +0.5895%다.
Deep 정답은 282->488, Deep->Light는 1190->945로 개선됐다.
Light->Deep은 1131->1447로 늘었고 validation 총점은 -1.4448%다.

다음 실험은 Light-vs-Deep specialist calibration refinement야.
pure top의 grid edge와 validation/test가 함께 좋았던 robust ridge를 같이 탐색한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_light_deep_specialist_fusion_refinement_colab.sh

결과 summary JSON을 받으면 current best 정확 재현, pure edge/cross-split robust ridge,
pure top/tie-rule selected, current best 대비 모든 metric의 절대/상대 변화율과
Light/Deep confusion 변화를 비교하고 새 best 및 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
