"""Paired held-out evaluation of an original and continued PPO checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .evaluate import plt, record
import numpy as np
import torch
from stable_baselines3 import PPO
from .env import Go1GaitEnv


def compare(before, after, output, episodes=20, durations=(12., 60.), video=True):
    if episodes <= 0 or any(t <= 0 for t in durations):
        raise ValueError("Use positive episodes and durations")
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(2000, 2000 + episodes))
    checkpoints = {"before": Path(before), "after": Path(after)}
    actions = {name: PPO.load(path, device="cpu").predict(np.zeros(1, np.float32), deterministic=True)[0]
               for name, path in checkpoints.items()}
    rows, summary = [], {}
    metrics = ("speed_m_s", "distance_m", "mean_power_W", "mean_tilt",
               "lateral_drift_m", "survived_s", "tracking_score", "reward")
    for duration in durations:
        horizon = str(int(duration)) if float(duration).is_integer() else str(duration)
        summary[horizon] = {}
        for name, action in actions.items():
            env = Go1GaitEnv(duration=duration)
            trials = []
            for seed in seeds:
                env.reset(seed=seed)
                _, info = env.simulate(action)
                trials.append(info)
                rows.append({"policy": name, "duration_s": duration, "seed": seed,
                             **{k: v for k, v in info.items() if k != "parameters"}})
            summary[horizon][name] = {
                k: {"mean": float(np.mean([r[k] for r in trials])),
                    "std": float(np.std([r[k] for r in trials]))} for k in metrics}
            summary[horizon][name].update(falls=sum(r["fell"] for r in trials), episodes=episodes,
                                          parameters=trials[0]["parameters"], action=action.tolist())
            print(horizon, name, json.dumps(summary[horizon][name]), flush=True)
            env.close()
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"evaluation_seeds": seeds, "durations_s": list(durations),
              "checkpoints": {n: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for n, p in checkpoints.items()}, "results": summary}
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(len(durations), 3, figsize=(12, 4 * len(durations)), squeeze=False)
    for row, (horizon, policies) in enumerate(summary.items()):
        for ax, metric, title in zip(axes[row], ("speed_m_s", "mean_power_W", "tracking_score"),
                                     ("Speed (m/s)", "Absolute mechanical power (W)", "Tracking score")):
            ax.bar(list(policies), [p[metric]["mean"] for p in policies.values()],
                   yerr=[p[metric]["std"] for p in policies.values()], capsize=4,
                   color=["#94a3b8", "#06b6d4"])
            ax.set_title(f"{horizon}s: {title}")
            ax.grid(axis="y", alpha=.2)
        axes[row, 0].axhline(.25, linestyle="--", color="black", linewidth=1)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    report = ["# 추가 학습 전후 비교", "", "기존 체크포인트에서 PPO를 이어 학습했다. 기존 실험 파일은 보존했다. PPT는 새로 생성하지 않았다.",
              "", "## 평가 방법", "", f"동일한 별도 초기 상태 {episodes}개(시드 {seeds[0]}~{seeds[-1]})에서 각 정책을 평가했다. 학습 시드는 하나이며 평가 시드는 독립 학습 반복을 뜻하지 않는다.",
              "", "체크포인트 선택은 학습 중 검증 시드 900~902의 5초 평균 보상만 사용했다. 아래 12초·60초 결과는 모델 선택에 사용하지 않았다.",
              "", "| 평가 길이 | 정책 | 속도 m/s | 기계적 파워 W | 속도 추종 점수 | 넘어짐 | 옆 변위 m |",
              "|---|---|---:|---:|---:|---:|---:|"]
    config_path = checkpoints["after"].parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        report[3:3] = ["", f"이번 실행: 추가 {config['actual_rollouts']:,}개 에피소드, 학습률 {config['learning_rate']}, seed={config['seed']}. 복원 시점 카운터 {config['previous_checkpoint_rollouts']:,} → 최종 모델 카운터 {config['total_checkpoint_rollouts']:,}."]
    best_metadata = checkpoints["after"].parent / "best_metrics.json"
    if best_metadata.exists():
        best = json.loads(best_metadata.read_text())
        report[3:3] = ["", f"선택된 best 모델: 추가 학습 {best['additional_rollouts']:,}개 시점, 검증 보상 {best['mean_reward']:.6f}. 시작 모델이 선택되었다면 추가 학습에서 검증 개선을 얻지 못했다는 뜻이다."]
    for horizon, policies in summary.items():
        for name, r in policies.items():
            report.append(f"| {horizon}s | {name} | {r['speed_m_s']['mean']:.4f} ± {r['speed_m_s']['std']:.4f} | {r['mean_power_W']['mean']:.2f} | {r['tracking_score']['mean']:.4f} | {r['falls']}/{episodes} | {r['lateral_drift_m']['mean']:.3f} |")
    report += ["", "## 해석 시 주의", "", "- 목표 속도는 0.25m/s다. 더 빠르다고 더 좋은 정책은 아니다.",
               "- 학습 횟수를 늘려도 모든 지표가 함께 좋아지는 것은 아니다. 넘어짐, 속도 추종, 파워, 옆 변위를 함께 확인한다.",
               "- 파워는 절대 기계적 일률이며 배터리 소비전력과 다르다.",
               "- 대각선 트로트 위상·역기구학은 미리 제공했다. PPO가 보행 파라미터 6개를 학습하며 평지에서만 검증했다.",
               "- CSV에 모든 개별 평가를 기록했고 JSON에 체크포인트 SHA-256을 기록했다."]
    (output / "추가학습기록.md").write_text("\n".join(report), encoding="utf-8")
    if video:
        env = Go1GaitEnv(duration=12.)
        record(env, actions["after"], output / "after.mp4", seed=seeds[0])
        env.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=Path("go1_walk/results/improved/best.zip"))
    parser.add_argument("--after", type=Path, default=Path("go1_walk/results_extended/improved/best.zip"))
    parser.add_argument("--output", type=Path, default=Path("go1_walk/results_extended"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--durations", type=float, nargs="+", default=[12., 60.])
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    compare(args.before, args.after, args.output, args.episodes, args.durations, not args.no_video)
