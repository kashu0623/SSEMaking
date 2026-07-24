# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

현재 목표는 비용, 모델 수, 추론량을 무시하고 성능-only 기준으로 DreamT sleep stage estimation fixed/flexible fusion 성능을 개선하는 거야.

현재 best는 24-checkpoint same-split ensemble + original direct4 hybrid:
hybrid_w0.30_li0.34_d0.98_rem0.00_dg1.25

3-seed 평균:
4M 0.4383 / 4K 0.2794 / 4M+4K 0.7177
Wake 0.5313 / Light 0.6703 / Deep 0.1766 / REM 0.3750 / Wake+REM 0.9063

선택 기준:
3-seed 평균에서 4M+4K가 가장 높은 후보를 best로 둔다.
단, 4M+4K 차이가 0.0005 이하이면 Wake+REM이 더 높은 후보를 우선한다.

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

round2 best 대비 4M+4K +0.1026%, Deep -0.5566%, Wake+REM +0.1254%다.
pure top과 selected 차이는 0.000272이고 selected의 Wake+REM이 더 높아 tie rule로 채택했다.
selected는 Light 0.34 상단에 걸렸고 pure top은 Light 0.24 ridge다.

다음 실험은 Light를 0.50까지 확장하는 마지막 static hybrid refinement round4야.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_four_model_direct4_hybrid_deep_refinement_round4_colab.sh

결과 summary JSON을 받으면 round3 selected best 재현, pure top/tie-rule selected, round3 best 대비
4M+4K, Wake+REM, Light/Deep의 절대/상대 변화율을 비교하고 새 best 및 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
