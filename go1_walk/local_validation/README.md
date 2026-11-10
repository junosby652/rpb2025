# 로컬 검증 기록 (2026-11-10)

WSL2 Ubuntu 22.04.5, Python 3.10.12, RTX 4050 Laptop GPU 6141 MiB, Windows NVIDIA 드라이버 577.05.

가상환경: 저장소 `.venv/`. MuJoCo / MuJoCo Warp 3.17.0, Warp 1.19.0+cu12, SB3 2.7.1, PyTorch 2.6.0+cpu. 패키지 출처와 버전은 `../requirements-local.lock`에 고정했다.

| 검증 | 결과 |
|---|---|
| CPU 물리·Gymnasium 테스트 | 4개 통과 (`cpu_tests.log`) |
| CUDA 물리·CPU 일치도·리셋·VecEnv 테스트 | 3개 통과 (`gpu_tests.log`) |
| 기존 체크포인트 복원 | baseline best, improved best, extended best/final 각각 5초 실행 |
| 기존 평가 CLI | 새 CPU smoke 모델에서 3개 정책 × 10시드 × 1초, 영상 없이 실행 |
| 기존 결과 보존 | 37개 파일 SHA256 모두 일치 |

| 학습 실행 | 추가 에피소드 | 학습 구간 시간 |
|---|---:|---:|
| cpu_smoke/baseline | 64 | 6.55초 |
| cpu_smoke/improved | 64 | 6.36초 |
| cpu_resume/improved | 64 | 6.45초 |
| warp_smoke/improved | 256 | 4.59초 |
| warp_resume/improved | 64 | 1.50초 |

학습 구간 시간은 환경/정책 초기화를 제외하고 검증·optimizer 업데이트·체크포인트 저장을 포함한다. 출력 폴더별 `config.json`, `best.zip`, `final.zip`, `training.csv`, `validation.csv`를 보존했다.

64개 환경 × 3개 배치, 에피소드 5초, 동일한 중간 행동·초기 자세로 물리 처리량을 비교했다. GPU 128.38 에피소드/초, CPU 12.53 에피소드/초, 속도비 10.25배. GPU 총 1.496초 / CPU 총 15.323초. 초기 컴파일과 PPO는 제외하고 리셋과 CPU↔GPU 전송은 포함한다. CPU 비교는 단일 환경 순차 실행이며 멀티코어 CPU 벤치마크가 아니다. 단일 측정이므로 환경·전력 상태에 따라 달라진다.

같은 5초 rollout의 CPU/GPU 전진 거리 차이는 0.00005160m, 에너지 차이는 0.00191J였다. 장시간 궤적의 완전한 동등성을 보장하는 수치는 아니다.

1초 간격 nvidia-smi 관측 최대 GPU 사용률은 100%, 메모리는 165MiB, 전력은 29.03W였다. WSL nvidia-smi의 샘플값이며 일부 행에 Unknown Error가 있어 해당 값은 집계에서 제외했다. GPU 메모리 전체를 정밀 프로파일링한 결과는 아니다.

PPO 정책·optimizer는 CPU다. CUDA에서는 물리·IK·feedback·보상을 계산한다. PyTorch CUDA=False와 Warp CUDA=True가 동시에 성립한다. `hardware.json`과 `gpu_telemetry.csv`에 기록했다.

GPU 신규 256개 학습은 CPU 검증 보상 약 0.356까지 실행됐다. 기존 best에서 GPU로 64개를 더 학습한 모델은 시작 정책의 약 0.441을 넘지 못했으므로 시작 정책을 best로 유지했다. final은 별도 저장한다. 성능 향상을 주장하지 않는다.

Windows OS 정확한 에디션·빌드와 WSL 앱 버전은 interop 비활성화로 조회하지 못했다. WSL2 실행·커널 버전, Ubuntu, GPU/VRAM/드라이버는 확인했다. Windows 드라이버·시스템 Python·WSL 설정은 변경하지 않았다. PPT를 생성하지 않았다.

`hardware_initial_cuda13_incompatible.json`은 최초 기본 Warp wheel의 드라이버 불일치 진단이다. 최종 환경은 `hardware.json`을 사용한다.
