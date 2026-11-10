# RTX 4050 로컬 실행

이 문서와 `local_validation/`은 로컬 설치 검증 기록이다. `results/`와 `results_extended/`는 기존 실험 결과로 보존한다. PPT는 생성하지 않는다.

## 실제 검증 결과

[검증 기록과 원자료](local_validation/README.md): CPU 테스트 4개, CUDA 테스트 3개 통과. CPU baseline·improved 각각 64개 에피소드, CUDA 64개 환경에서 256개 에피소드, 기존 모델에서 CPU 및 CUDA 각각 64개 추가 학습을 실행했다. 기존 체크포인트 4개 복원과 평가 CLI도 검증했다. 기존 결과 37개의 SHA256은 모두 유지됐다.

64개 환경의 물리 롤아웃 처리량은 CPU 12.53개/초, CUDA 128.38개/초로 약 10.25배였다. 최초 컴파일·PPO는 제외하고 리셋·전송은 포함한 단일 측정이다. CPU는 단일 환경을 순차 실행했다. CUDA 신규 256개 학습 구간은 약 4.59초였으며 환경 초기화 시간은 제외한다. `nvidia-smi` 1초 간격 관측 최대 GPU 사용률은 100%였다. 보행 성능 향상 검증과 실행 성공은 구분하며, 기존 best에서 짧게 이어 학습한 결과는 시작 모델의 검증 보상을 넘지 못했다.

## 설치 환경

- WSL2 Ubuntu 22.04.5 LTS, Linux `6.6.87.2-microsoft-standard-WSL2`.
- NVIDIA GeForce RTX 4050 Laptop GPU, VRAM 6141 MiB, Windows 드라이버 577.05. `nvidia-smi`의 CUDA 12.9는 드라이버 지원 버전이며 별도 CUDA Toolkit 설치를 의미하지 않는다.
- Python 3.10.12, 저장소의 `.venv/`. `python` 명령은 가상환경 활성화 후 사용한다.
- 이 Codex 세션의 샌드박스에서는 GPU 접근이 막힌다. GPU 검증은 권한을 확장한 프로세스로 실행한다. 일반 WSL 터미널에서도 `nvidia-smi`와 아래 진단 명령으로 확인한다.
- WSL 커널로 WSL2 설치·실행은 확인했다. Windows 실행 파일 interop이 등록되지 않아 `powershell.exe`와 `wsl.exe --version`은 `Exec format error`로 실패했다. Windows의 정확한 에디션·빌드와 WSL 앱 버전은 확인되지 않았다. 설치된 servicing 폴더에서 26100 계열 흔적은 확인했지만 실행 중인 OS 버전으로 단정하지 않는다.

## 실행

저장소 루트의 WSL 터미널에서:

```bash
source .venv/bin/activate
python -m go1_walk.diagnose --output /tmp/go1-diagnostics.json
python -m unittest go1_walk.test_env -v
python -m go1_walk.train --mode improved --steps 64 --output go1_walk/results_cpu_new
python -m unittest go1_walk.test_warp -v
python -m go1_walk.train --backend warp --num-envs 64 --mode improved \
  --steps 256 --output go1_walk/results_warp_new
```

매 실행마다 새로운 `--output`을 사용한다. 학습기는 비어 있지 않은 출력 폴더에 덮어쓰기를 거부한다. 결과는 `best.zip`, `final.zip`, `training.csv`, `validation.csv`, `config.json`에 저장한다. `--steps`는 5초 물리 에피소드 수이며 CPU 기본은 64개 단위, Warp 64 환경도 64개 단위로 올림된다. GPU 첫 실행의 커널 컴파일은 오래 걸릴 수 있다.

기존 정책에서 이어 학습:

```bash
python -m go1_walk.train --backend warp --num-envs 64 --mode improved \
  --resume go1_walk/results/improved/best.zip --learning-rate 0.0005 \
  --steps 2048 --validation-every 256 --output go1_walk/results_warp_resume_new
```

체크포인트의 정책·optimizer를 복원한다. 물리 엔진은 변경되므로 기존 결과와 동일한 궤적을 보장하지 않는다. 검증 최우수 정책 선택은 CPU의 시드 900~902를 계속 사용한다. 최종 평가 시드는 이들과 분리해야 한다. 짧은 실행 성공은 충분한 보행 학습이나 성능 향상을 의미하지 않는다.

로컬 MuJoCo 버전도 초기 결과의 4.0.0과 다르므로, 과거 평가 수치를 현재 수치와 직접 동일시하지 않는다. 원본 모델 파일과 초기 결과·lockfile은 보존했다.

벤치마크:

```bash
python -m go1_walk.benchmark --num-envs 64 --batches 3 \
  --output /tmp/go1-benchmark-new.json
```

## CPU와 GPU의 역할

| 실행 경로 | 물리 / IK / 자세 피드백 / 보상 | PPO 정책·optimizer |
|---|---|---|
| `--backend cpu` | CPU MuJoCo, 1개 환경 | CPU PyTorch |
| `--backend warp --num-envs 64` | MuJoCo Warp, CUDA 64개 환경 | CPU PyTorch |

따라서 CPU 빌드 PyTorch에서 `torch.cuda.is_available()`이 False여도 Warp 물리는 GPU에서 실행될 수 있다. `warp.get_cuda_devices()`, 실제 Warp 실행, NVIDIA 메모리·사용률로 판단한다. 정책이 32×32의 작은 신경망이므로 이번 전환에서는 CPU PPO를 유지했다. CUDA 정책 학습 전체를 구성한 것은 아니다.

Warp 경로는 6개 행동으로 한 에피소드의 트로트 파라미터를 선택하는 기존 문제를 유지한다. 새로운 상태 기반 관절 제어 문제로 바꾸지 않았다. GPU에서 IK, feedback, 에너지 적분, 보상을 계산하며 5개의 물리 스텝과 제어·측정을 CUDA graph로 반복한다. 에피소드 시작/끝에서만 배열을 전송한다. 넘어짐 이후 해당 환경의 지표는 동결되지만 배치 물리는 전체 예정 구간까지 진행한다. MuJoCo의 원본 모델·solver 설정은 유지하며 Warp의 float32에 따른 차이가 있다. 짧은 CPU/GPU 비교 테스트와 비유한 상태·버퍼 overflow 검사를 제공한다.

6GB VRAM에서는 64개 환경부터 시작한다. 환경 수를 늘릴 때 메모리·처리량을 실측한다. `nconmax=64`, `njmax=256`이며 overflow 발생 시 성공으로 처리하지 않고 오류를 낸다. CUDA가 없을 때 CPU로 자동 대체하지 않는다.

## 재설치

`requirements.lock`은 초기 실험의 Python 3.12 전용 기록이다. 로컬용 Python 3.10 패키지는 `requirements-local.lock`으로 별도 고정한다. 이 세션에서는 시스템 pip와 ensurepip가 없어 `.venv` 안에 공식 PyPA bootstrap으로 pip를 설치했다. 시스템 Python은 변경하지 않았다.

새 Ubuntu에서 venv 모듈이 없으면 먼저 `sudo apt install python3.10-venv`를 실행한 후:

```bash
bash go1_walk/setup.sh
source .venv/bin/activate
python -m pip check
```

Warp는 NVIDIA 드라이버를 사용한다. WSL 안에 Linux용 NVIDIA 디스플레이 드라이버를 설치하지 않는다. 현재 드라이버로 실제 실행 가능한 패키지 조합을 검증한다.

중요한 설치 차이: PyPI의 Warp 1.19 기본 wheel은 CUDA 13.4용이며 현재 드라이버에서는 CUDA 장치를 활성화하지 못했다. [NVIDIA 공식 릴리스](https://github.com/NVIDIA/warp/releases/tag/v1.19.0)의 `1.19.0+cu12` Linux x86_64 wheel을 사용한다. 로컬 lockfile은 이 다운로드 URL과 SHA256을 포함하므로 단순한 `pip install -U warp-lang`으로 바꾸지 않는다. Python 3.10 지원은 Warp 1.20에서 제거될 예정이므로 이번 환경은 1.19로 고정한다.

참고: [MuJoCo Warp 공식 문서](https://mujoco.readthedocs.io/en/latest/mjwarp/), [SB3 2.7.1 설치 요구사항](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/install.html).
