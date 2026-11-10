# 관절 떨림 완화

기존 Warp 학습 모델의 떨림을 줄이기 위해 **위치 액추에이터 속도 감쇠 `kv=0.5 N m s/rad`**를 추가했다. 기존 kp=100, 관절의 수동 감쇠, 관절·토크 제한, 접촉·마찰, 발 궤적, 50Hz 목표 갱신, 250Hz 물리는 유지했다. 양수 kv에서는 `implicitfast` 적분을 사용한다. 원본 XML과 학습 결과는 보존한다.

새 모델과 12초·60초 비교 결과는 [안정화 평가](results_stability/assessment/안정화평가.md)에 기록한다. [전후 동시 비교 영상](results_stability/assessment/side_by_side.mp4)은 왼쪽이 변경 전, 오른쪽이 안정화 모델이다. [안정화 모델 영상](results_stability/assessment/stabilized_12s.mp4)도 별도 보존한다.

## 변경 이유와 실험 범위

기존 제어는 관절 위치만 액추에이터에 지시했고 별도의 액추에이터 속도 피드백이 없었다. 관절 자체의 수동 감쇠는 이미 있었다. 빠른 관절 움직임에 반대 방향 토크를 더하는 속도 감쇠로 진동을 줄인다. MuJoCo의 [position actuator와 kv 설명](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-position)에 따른 설정이다.

원래 발 궤적에는 스윙/지지 전환 시 목표 발 속도의 불연속이 있다. 현재 정책에서 수평 약 0.344m/s, 수직 약 0.637m/s의 경계 속도 차이가 계산됐다. 궤적·제어 주파수·감쇠를 탐색한 단일 시드 5초 실험을 `results_stability/go1-jitter-sweep*.json`에 보존했다. 단순히 제어 주파수를 올리거나 timestep을 줄이는 것만으로는 개선되지 않았다. 더 부드러운 발 궤적을 기존 정책에 바로 적용한 경우 전진 속도 감소도 관측돼 이번 수정에는 채택하지 않았다.

원래 정책에 kv=0.5만 적용한 조건과, 같은 제어에서 기존 정책을 CUDA로 512개 에피소드 더 학습한 조건을 비교했다. 학습률은 0.0005이며 64개 병렬 환경을 사용했다. 검증 시드는 900~902다. **검증에서 넘어짐 없이 원래 속도의 85% 이상을 유지하고, 10Hz 초과 관절 속도 RMS를 25% 이상 줄인 후보만 허용한 뒤 보상이 높은 후보를 선택**했다. 최종 시드 5000~5009의 결과로 후보를 다시 선택하지 않는다.

고주파 RMS는 250Hz 물리 상태에서 첫 0.5초를 제외하고 Hann window를 적용한 뒤 10Hz 초과 관절 속도 성분의 크기로 계산한다. 보행의 고조파도 포함되므로 이 값 전체를 수치 오류로 해석하지 않는다. 초기 탐색 JSON은 window를 사용하지 않은 지표이므로 최종 평가와 절대값을 직접 비교하지 않는다. CPU·Warp 일치도 테스트는 원래 제어와 kv=0.5 제어 모두 실행한다.

영상은 프레임 보간이나 시간 축 변경 없이 CPU MuJoCo에서 실제 제어를 재생한다. 모델 선택과 새 평가 시드를 분리하고, 장기 직진성도 함께 확인한다. 떨림 완화가 지형·마찰·외란에 대한 일반화나 실물 로봇의 안정성을 보장하지는 않는다.

## 실행 방법

기존 모델에서 안정화 제어로 추가 학습하려면 저장소 루트에서:

```bash
.venv/bin/python -m go1_walk.train --backend warp --num-envs 64 \
  --mode improved --actuator-kv 0.5 --learning-rate 0.0005 --steps 512 \
  --resume go1_walk/results_warp_new/improved/best.zip \
  --output go1_walk/results_stability_new
```

이미 안정화된 모델에서 이어 학습하면 `--actuator-kv` 생략 시 체크포인트 옆 `config.json`에서 kv를 복원한다. 새 학습은 기본값 0으로 기존 동작을 유지하며, `--actuator-kv 0`으로 명시적으로 원래 액추에이터 제어를 선택할 수 있다. CPU 학습에도 같은 옵션을 사용한다.

새 결과를 기존 제어와 평가하고 녹화:

```bash
.venv/bin/python -m go1_walk.stability \
  --candidate go1_walk/results_stability_new/improved/best.zip \
  --output go1_walk/results_stability_new/assessment
```

일반 `assess`, `compare`, `evaluate`도 체크포인트 옆 설정의 kv를 복원한다. 체크포인트 ZIP에는 MuJoCo 모델 설정이 들어 있지 않으므로 **config.json과 ZIP을 함께 보관**한다. 이전 설정 파일에 kv가 없으면 0으로 처리한다. 기존 출력 폴더에 덮어쓰지 말고 새 경로를 사용한다.
