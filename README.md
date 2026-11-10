# rpb2025

이 저장소는 기계시스템 설계 및 로봇 프로그래밍 기초를 위한 공간입니다. Hello

## Go1 보행 강화학습

MuJoCo + PPO로 Unitree Go1의 트로트 보행 파라미터를 학습하는 실험입니다.

- [설치·학습·평가 방법](go1_walk/README.md)
- [실험 기록](go1_walk/results/실험기록.md)
- [학습된 보행 영상](go1_walk/results/improved.mp4)
- [4,096회 추가 학습과 장시간 평가 기록](go1_walk/results_extended/추가학습기록.md)
- [추가 학습 전후 비교 그래프](go1_walk/results_extended/comparison.png)
- [보존된 최우수 정책의 새 평가 영상](go1_walk/results_extended/after.mp4)
- [4,096회 추가 학습 종료 모델의 영상](go1_walk/results_extended/final.mp4)

PPT 작성은 보류하고 보행 학습과 평가를 진행합니다.

추가 학습은 기존 최우수 체크포인트에서 학습률을 0.0005로 낮춰 진행했습니다.
기존 최우수 모델의 검증 보상을 넘지 못해 기존 정책을 최우수로 유지하고,
추가 학습 종료 모델도 별도로 저장해 12초·60초 평가에서 비교합니다.

GPU가 없는 환경에서 실행했으며, 사전 정의한 보행 궤적의 파라미터를 PPO로 학습합니다.
