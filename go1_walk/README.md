# Unitree Go1 보행 강화학습

목표: MuJoCo에서 Go1을 전진시키고, PPO의 보상 설계에 따른 속도·안정성·기계적 에너지 사용 차이를 기록한다. **실제 로봇에 배포하는 코드는 아니다.**

## 실행 환경과 설치

- RTX 4050 노트북의 WSL2 로컬 설치 및 CUDA 병렬 실행: [LOCAL_SETUP.md](LOCAL_SETUP.md).
- 로컬 설치는 Python 3.10과 `requirements-local.lock`을 사용한다. 초기 실험의 Python 3.12 / MuJoCo 4.0.0 / PyTorch CPU 환경은 원본 `requirements.lock`에 보존했다.
- `--backend cpu`는 물리와 PPO 모두 CPU, `--backend warp`는 CUDA 병렬 물리와 CPU PPO를 사용한다. 작은 정책의 GPU 학습 여부와 물리의 GPU 실행 여부를 구분한다.
- 외부 로봇 모델 및 라이선스: [assets/SOURCE.md](assets/SOURCE.md).
- 네트워크: 최초 설치 시 `download.pytorch.org`, `download-r2.pytorch.org`, `pypi.org`, `files.pythonhosted.org` 필요. 모델은 저장소에 포함되어 추가 다운로드하지 않는다. API 키 불필요.

저장소 루트에서:

```bash
bash go1_walk/setup.sh
source .venv/bin/activate
python -m unittest go1_walk.test_env -v
```

설치 위치를 바꾸려면 `GO1_VENV=/원하는/경로 bash go1_walk/setup.sh`를 실행한다. Python 3.10의 venv 모듈이 없으면 `python3.10-venv`가 필요하다. EGL 영상 렌더링에는 시스템 OpenGL/EGL 라이브러리가 필요하다.

## 학습과 평가

```bash
# 각 실험에 2,048개의 물리 롤아웃. 기존 결과를 보존하기 위해 새 경로 사용.
python -m go1_walk.train --steps 2048 --seed 42 --output go1_walk/results_new
MUJOCO_GL=egl python -m go1_walk.evaluate --results go1_walk/results_new
```

RTX 4050에서 CUDA 병렬 물리로 학습(6GB VRAM에서는 64개 환경부터 시작):

```bash
python -m go1_walk.train --backend warp --num-envs 64 --mode improved \
  --steps 2048 --output go1_walk/results_warp_new
```

이 경로는 동일한 행동·관측·보상과 4ms 물리 / 20ms 제어 주기를 유지한다. 검증 체크포인트 선택은 CPU MuJoCo의 고정된 검증 시드로 진행한다. `--resume`으로 기존 CPU 정책을 GPU 물리 학습에 이어 쓸 수 있다. Warp는 float32이므로 CPU의 float64 궤적과 장시간 완전히 일치하지 않을 수 있다. 요청한 `--steps`는 병렬 환경 수 × PPO rollout 길이의 배수로 올림하며 실제 횟수와 장치를 `config.json`에 기록한다. CUDA 접근에 실패하면 오류를 내며 CPU로 자동 대체하지 않는다.

`improved`만 학습한 결과의 영상·비교 평가는 다음 명령으로 생성한다. 새 출력 폴더를 지정해야 한다. 기존 CPU best와 이번 best/final을 같은 CPU MuJoCo에서 새 시드 3000~3019로 12초·60초 평가하고, 12초 MP4 세 개와 CSV·JSON·그래프·한국어 보고서를 만든다. 기존 `evaluate` 명령은 baseline과 improved가 모두 있는 결과 폴더용이다.

```bash
python -m go1_walk.assess --results go1_walk/results_warp_new \
  --output go1_walk/results_warp_new/assessment_new
```

기존 학습 결과만 재평가:

```bash
MUJOCO_GL=egl python -m go1_walk.evaluate --results go1_walk/results
```

렌더링이 없는 서버에서는 `--no-video`로 수치 평가만 실행한다. `--steps`는 시뮬레이션의 제어 스텝 수가 아니라 **5초짜리 전체 에피소드 수**다. PPO rollout buffer 64개 단위로 반올림되며 `config.json`에 실제 개수를 기록한다. 학습은 CPU에서도 가능하지만 수 분 이상 걸린다. `best.zip`은 검증 시드 900~902의 평균 보상으로 선택하며, 최종 성능 평가에는 별도 시드 1000~1009를 사용한다.

### 기존 모델에서 추가 학습

```bash
python -m go1_walk.train --mode improved --steps 4096 \
  --resume go1_walk/results/improved/best.zip \
  --learning-rate 0.0005 --validation-every 256 \
  --output go1_walk/results_extended
python -m go1_walk.compare --final go1_walk/results_extended/improved/final.zip
```

체크포인트의 정책과 optimizer 상태를 불러오고 **추가로** 4,096개 에피소드를 학습한다. 기존 출력은 덮어쓰지 않는다. 추가 실험의 `best.zip`은 시작 모델도 후보에 포함해 검증 보상이 가장 높은 모델을 보존한다. `final.zip`은 추가 학습을 모두 마친 모델이며, 항상 best보다 좋은 것은 아니다. `best_metrics.json`에 선택된 시점과 검증 성능을 기록한다.

`compare`는 추가 학습 전후 정책을 새 평가 시드 2000~2019에서 각 12초·60초 실행한다. `--final`을 주면 검증 최우수 정책뿐 아니라 종료 시점의 정책도 함께 비교한다. 기본 출력은 `results_extended/`의 원자료, 그래프, `추가학습기록.md`, `after.mp4`이며 `--final`을 주면 `final.mp4`도 저장한다. **PPT는 생성하지 않는다.** 평가 시드는 체크포인트 선택에 사용하지 않는다.

원래 2,048개 학습 에피소드 중 선택된 최우수 체크포인트는 1,024번째 시점의 정책이다. 이어 학습할 때의 누적 카운터는 이 체크포인트에서 시작한다. 따라서 `actual_rollouts`(이번 추가 학습 횟수), `previous_checkpoint_rollouts`(복원 시점), `total_checkpoint_rollouts`(학습 종료 시점)를 구분한다.

## 무엇을 학습하는가

PPO가 매 에피소드 시작에 **보행 파라미터 6개**를 선택한다. 이 파라미터로 발 궤적을 생성하고 역기구학으로 12개 관절 목표각을 계산한다. MuJoCo의 원본 위치 액추에이터가 물리적으로 로봇을 움직인다.

| 파라미터 | 허용 범위 |
|---|---|
| 발의 전후 이동 길이 | 0.03~0.17 m |
| 보행 주파수 | 1.0~2.8 Hz |
| 다리의 수직 길이 | 0.245~0.310 m |
| 스윙 발 높이 | 0.015~0.075 m |
| 발의 전후 중심 오프셋 | -0.045~0.045 m |
| 주기 중 지지 구간 비율 | 0.50~0.72 |

대각선 두 다리가 교대로 움직이는 **트로트(trot)** 위상은 사람이 지정했다. 엄밀한 네 발 순차 보행(walk)이나 관절 제어를 처음부터 학습하는 end-to-end RL과는 다르다. CPU에서 재현 가능한 첫 보행 실험으로 범위를 정했다. 원한다면 다음 실험에서 위상 자체를 행동 공간에 넣어 walk와 trot를 비교할 수 있다.

관측은 상수 하나다. 하나의 행동이 전체 에피소드에 적용되는 파라미터 탐색 문제이므로 `gamma=0`을 사용한다. 매 20ms 목표각 갱신, 물리는 4ms 간격으로 진행한다. 동일한 약한 롤 자세 피드백을 두 실험에 모두 적용한다. 초기 관절각에 표준편차 0.008rad 잡음을 주며 평평한 지면과 고정된 마찰계수만 사용한다.

## 보상 설계 비교

두 조건에 동일한 PPO, 학습 예산, 시드, 파라미터 범위를 사용한다. 자세 피드백과 주기 궤적은 공통이다.

**baseline**: 각 제어 스텝의 전진 속도 `vx`를 보상으로 사용한다.

**improved**:

```text
exp(-(vx - 0.25)^2 / 0.04)       # 0.25 m/s 목표 속도 추종
- 2.0 * (1 - body_up_z)          # 몸통 기울기
- 0.5 * vy^2                    # 옆 방향 속도
- 0.003 * sum(abs(torque*qvel))  # 절대 기계적 파워
- 0.1 * yaw_rate^2              # 회전 속도
```

에피소드 보상은 전체 예정 제어 스텝 수로 나눈다. 몸통 높이가 0.16m 미만이거나 몸통 수직 축의 z성분이 0.65 미만이면 종료하고 -2를 추가한다. 조기 종료로 나쁜 후반부를 회피하는 것을 억제한다. PPO에 전달할 때만 전체 보상을 10배한다. 학습 그래프에는 배율 적용 전 보상을 표시한다. 두 보상함수의 숫자 자체는 서로 직접 비교하지 않는다.

기계적 에너지는 모든 물리 스텝에서 `sum(abs(관절 토크 × 관절 속도)) × dt`를 적분한다. 전기 소비전력이나 배터리 사용량과 같지 않으며 모터 손실·회생제동을 모델링하지 않는다. 보상에 쓰는 파워는 제어 스텝 끝의 샘플 값이라, 평가에서 적분한 평균 파워와 정확히 같지는 않다.

## 결과 파일

- `results/baseline/`, `results/improved/`: 모델 체크포인트, 실제 설정, 롤아웃별 CSV.
- `results/evaluation.json`, `evaluation.csv`: 세 정책을 각각 12초 × 10개 별도 초기 상태로 평가한 원자료.
- `initial.mp4`, `baseline.mp4`, `improved.mp4`: 같은 평가 시드 1000에서의 실제 시뮬레이션 영상. 25fps, 이동 카메라.
- `comparison.png`, `learning_curves.png`: 평가 비교 및 학습 곡선.
- `실험기록.md`: 첫 실험에서 관측된 결과 정리. 추가 학습 기록은 `results_extended/추가학습기록.md`에 별도로 저장한다.

PPT 작업은 현재 보류했다. 나중에 필요하면 `python -m go1_walk.presentation --results go1_walk/results`로 별도 생성할 수 있다. PPT 파일은 Git에 올리지 않는다.

속도는 준비 자세에서 0.5초 정착한 뒤의 순 전방 변위 / 실제 생존 시간이다. `tracking_score`는 목표 속도 추종 항의 전체 예정 평가 구간 평균이며 1에 가까울수록 좋다. `mean_tilt`는 라디안이 아니라 `1 - body_up_z`다. 결과의 표준편차는 같은 학습 모델의 서로 다른 초기 상태에 대한 값이다. **독립적인 학습 시드 10개를 반복한 결과가 아니다.**

## 발표할 때 설명할 내용

1. GPU 부재에 맞춰 CPU MuJoCo와 저차원 PPO 파라미터 학습을 선택했다.
2. 사람이 제공한 트로트 위상·역기구학과 PPO가 실제로 찾은 파라미터를 구분한다.
3. 전진 속도만 보상하면 목표 속도를 지나치게 넘거나 에너지를 많이 쓰는지 확인한다.
4. 속도 추종·자세·에너지 항을 넣었을 때 실제 측정값이 어떻게 바뀌었는지 표와 영상으로 보여준다.
5. 개선이 관측된 지표와 악화된 지표를 모두 기록한다. 자연스러운 보행이나 일반화가 증명됐다고 과장하지 않는다.

후속 실험: 3개 이상 학습 시드 반복, 더 긴 평가, 지면 마찰·질량·외력 변화, 발 접촉/미끄럼 페널티, 실제 상태 관측을 사용하는 residual PPO, GPU 환경에서 대규모 병렬 학습. 현재 결과는 평지 시뮬레이션의 제한된 실험이다.
