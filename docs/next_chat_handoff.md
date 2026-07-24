# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 24-checkpoint same-split ensemble + original direct4 hybrid:
hybrid_w0.20_li0.10_d1.00_rem0.00

3-seed 평균:
4M 0.4331 / 4K 0.2751 / 4M+4K 0.7082
Wake 0.5233 / Light 0.6746 / Deep 0.1592 / REM 0.3752 / Wake+REM 0.8984

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

이전 best 대비 4M+4K +4.1600%, Deep +59.1888%, Wake+REM +1.3796%다.
pooled Deep 정답은 176->242, Deep->Light는 1,290->1,230으로 개선됐다.
round1 best가 Wake 0.20/Light 0.10/Deep 1.00 grid 상단에 걸렸다.

다음 실험은 Wake/Light edge 확장 + Deep gain hybrid refinement round2야.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_four_model_direct4_hybrid_deep_refinement_round2_colab.sh

결과 summary JSON을 받으면 round1 best 재현, pure top/tie-rule selected, round1 best 대비
4M+4K, Wake+REM, Light/Deep의 절대/상대 변화율을 비교하고 새 best 및 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
