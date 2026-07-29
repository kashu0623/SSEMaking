# 다음 채팅방 전달 내용

아래 내용을 다음 채팅방에 그대로 전달한다.
상세한 현재 진행 상황, current best, 실험 히스토리, 다음 실험 기준은 `docs/current_progress_summary.md`를 기준으로 확인하고 갱신한다.

```text
docs/current_progress_summary.md를 읽고 이어서 진행해줘.

4-class fusion refinement는 잠시 멈추고 앱 목적 중심 Light-vs-rest 트랙을 시작했어.
비용, 모델 수, 추론량은 무시하고 성능만 본다.

기존 4-class best는 기존 30-checkpoint static hybrid + specialist 2개:
blend_a0600__beta0.97_scale0.75_bias0.25
exact beta는 0.975다.

3-seed 평균:
4M 0.4572 / 4K 0.2917 / 4M+4K 0.7489
Wake 0.5356 / Light 0.6719 / Deep 0.2463 / REM 0.3751 / Wake+REM 0.9107

최종 알람은 수면 단계 AI 단독이 아니라 미세 움직임/RR/RRV/HR/HRV/피부온도 변화를
0~1 정규화 후 가중합하는 PotchArousalCalculator의 각성 점수와 함께 판단할 예정이다.

기존 current argmax의 binary baseline:
pooled Light F1 0.671859 / binary Kappa 0.237415
Light precision 0.702532 / recall 0.643752 / Deep→Light 0.607715

direct Light-vs-rest best는 Light F1 +11.6937%, recall +43.0780%였지만
binary Kappa -36.4450%, precision -9.8794%, Deep→Light +55.4688%라 채택하지 않았다.
Deep-veto selected도 Light F1 +1.8440%, recall +10.3812%였지만
objective -5.6238%, Kappa -26.7569%, precision -6.0831%,
Deep→Light +15.9180%라 채택하지 않았다.
alpha=1 경계, global best gamma=0으로 post-hoc veto 상보성도 없었다.

다음은 Other(Wake+REM)/Light/Deep 3-class primary objective를 직접 학습한다.
Deep을 독립 출력으로 강제하고 P(Light)를 기존 앱 binary 지표로 평가한다.

Colab 실행:
%cd /content/SSE
!git pull
!bash scripts/run_light_alarm_3class_colab.sh

결과 summary JSON을 받으면 selected와 safe profiles를 current argmax baseline 대비
objective/F1/Kappa/precision/recall 및 Wake/Deep/REM→Light 변화로 비교하고
앱용 새 best와 다음 방향을 정한 뒤
docs/current_progress_summary.md를 갱신해줘.
```
