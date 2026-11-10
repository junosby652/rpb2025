"""Evaluate a fresh Warp training run against a preserved CPU policy, with MP4s."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from .evaluate import plt, record
from .env import Go1GaitEnv
import numpy as np
import torch
from stable_baselines3 import PPO


METRICS = ('speed_m_s', 'distance_m', 'energy_J', 'mean_power_W', 'mean_tilt',
           'lateral_drift_m', 'abs_lateral_drift_m', 'survived_s', 'tracking_score', 'reward')
LABELS = {'reference': '기존 CPU best', 'best': '이번 Warp best', 'final': '이번 Warp final'}


def assess(results, reference, output, episodes=20, seed_start=3000, durations=(12., 60.)):
    if episodes < 1 or seed_start < 0 or any(t <= 0 for t in durations) or not durations:
        raise ValueError('Use positive episodes/durations and nonnegative seeds')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    results = Path(results)
    checkpoints = {'reference': Path(reference), 'best': results/'improved/best.zip',
                   'final': results/'improved/final.zip'}
    torch.set_num_threads(1)
    actions = {name: PPO.load(path, device='cpu').predict(np.zeros(1, np.float32), deterministic=True)[0]
               for name, path in checkpoints.items()}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints.values()}
    seeds = list(range(seed_start, seed_start+episodes))
    if set(seeds) & {900, 901, 902}:
        raise ValueError('Evaluation seeds must exclude checkpoint-selection seeds 900..902')
    output.mkdir(parents=True)
    summaries, rows, cache = {}, [], {}
    for duration in durations:
        summaries[str(duration)] = {}
        for name, action in actions.items():
            key = (duration, action.tobytes())
            if key not in cache:
                env = Go1GaitEnv(duration=duration)
                trials = []
                try:
                    for seed in seeds:
                        env.reset(seed=seed)
                        _, info = env.simulate(action)
                        if not all(np.isfinite(info[k]) for k in ('reward', 'energy_J', 'speed_m_s')):
                            raise FloatingPointError(info)
                        info['abs_lateral_drift_m'] = abs(info['lateral_drift_m'])
                        trials.append(info)
                finally:
                    env.close()
                cache[key] = trials
            trials = cache[key]
            summary = {k: {'mean': float(np.mean([t[k] for t in trials])),
                           'std': float(np.std([t[k] for t in trials]))} for k in METRICS}
            summary.update(falls=sum(t['fell'] for t in trials), episodes=episodes,
                           parameters=trials[0]['parameters'], action=action.tolist())
            summaries[str(duration)][name] = summary
            for seed, info in zip(seeds, trials):
                rows.append(dict(policy=name, duration_s=duration, seed=seed,
                                 **{k: v for k, v in info.items() if k != 'parameters'}))
            print(f"{duration:g}s {name}: speed={summary['speed_m_s']['mean']:.4f}, "
                  f"power={summary['mean_power_W']['mean']:.2f}, "
                  f"tracking={summary['tracking_score']['mean']:.4f}, "
                  f"falls={summary['falls']}/{episodes}, "
                  f"abs_drift={summary['abs_lateral_drift_m']['mean']:.3f}", flush=True)
        # Preserve completed horizons even if a later rendering operation fails.
        with (output/'evaluation.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    equal = bool(np.array_equal(actions['best'], actions['final']))
    result = dict(evaluation_backend='CPU MuJoCo', target_speed_m_s=.25,
                  evaluation_seeds=seeds, durations_s=list(durations),
                  training_config=json.loads((results/'improved/config.json').read_text()),
                  packages={n: importlib.metadata.version(n) for n in ('mujoco', 'numpy', 'torch', 'stable-baselines3')},
                  checkpoints={n: dict(path=str(p), sha256=hashes[str(p)]) for n, p in checkpoints.items()},
                  best_final_deterministic_actions_identical=equal, results=summaries)
    (output/'evaluation.json').write_text(json.dumps(result, indent=2))
    figure, axes = plt.subplots(len(durations), 4, figsize=(16, 4*len(durations)), squeeze=False)
    for axis_row, (duration, policies) in zip(axes, summaries.items()):
        for ax, metric, title in zip(axis_row,
                ('speed_m_s', 'mean_power_W', 'tracking_score', 'abs_lateral_drift_m'),
                ('Forward speed (m/s)', 'Mechanical power (W)', 'Speed tracking (0-1)', 'Absolute lateral drift (m)')):
            ax.bar(list(policies), [r[metric]['mean'] for r in policies.values()],
                   yerr=[r[metric]['std'] for r in policies.values()], capsize=4,
                   color=['#94a3b8', '#06b6d4', '#f59e0b'])
            ax.set_title(f'{float(duration):g}s: {title}')
            ax.grid(axis='y', alpha=.2)
        axis_row[0].axhline(.25, color='black', linestyle='--', linewidth=1)
    figure.tight_layout()
    figure.savefig(output/'comparison.png', dpi=170)
    plt.close(figure)
    with (results/'improved/training.csv').open() as f:
        training = list(csv.DictReader(f))
    with (results/'improved/validation.csv').open() as f:
        validation = list(csv.DictReader(f))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    rewards = np.array([float(r['reward']) for r in training])
    axes[0].plot(np.arange(len(rewards))+1, rewards, alpha=.15, linewidth=.5)
    window = min(64, len(rewards))
    axes[0].plot(np.arange(window, len(rewards)+1),
                 np.convolve(rewards, np.ones(window)/window, mode='valid'))
    axes[0].set(title='Training reward (64-episode moving mean)', xlabel='Episodes', ylabel='Unscaled reward')
    axes[1].plot([int(r['additional_rollouts']) for r in validation],
                 [float(r['mean_reward']) for r in validation], marker='.', linewidth=1)
    axes[1].set(title='CPU validation reward (seeds 900-902)', xlabel='Episodes', ylabel='Unscaled reward')
    for ax in axes:
        ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output/'learning_curves.png', dpi=170)
    plt.close(figure)
    recorded = {}
    for name, action in actions.items():
        destination = output/f'{name}_12s.mp4'
        key = action.tobytes()
        if key in recorded:
            shutil.copyfile(recorded[key], destination)
            shutil.copyfile(recorded[key].with_suffix('.png'), destination.with_suffix('.png'))
        else:
            env = Go1GaitEnv(duration=12.)
            try:
                record(env, action, destination, seed=seeds[0])
            finally:
                env.close()
            recorded[key] = destination
        print('Video:', destination, flush=True)
    for p in checkpoints.values():
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
            raise RuntimeError(f'Checkpoint changed during evaluation: {p}')
    config = result['training_config']
    start_description = (f"{config['resume_from']}에서 이어 학습한" if config.get('resume_from')
                         else '새로 시작한')
    report = ['# Warp 학습 모델 평가', '',
              f"이번 학습은 seed={config['seed']}로 {start_description} {config['actual_rollouts']:,}개 에피소드 실험이다. 정확한 설정은 evaluation.json의 training_config에 보존했다.", '',
              f'평가 시드 {seeds[0]}~{seeds[-1]}의 동일한 초기 관절 잡음을 각 정책에 적용했다. 학습 중 모델 선택 시드 900~902와 분리했다. 모든 정책을 현재 CPU MuJoCo 버전으로 다시 평가했으므로 과거 문서의 다른 버전 평가값을 그대로 가져오지 않았다.', '',
              '| 길이 | 정책 | 속도 m/s (평균±표준편차) | 파워 W | 속도 추종 | 넘어짐 | 절대 옆 변위 m |',
              '|---|---|---:|---:|---:|---:|---:|']
    for duration, policies in summaries.items():
        for name, r in policies.items():
            report.append(f"| {float(duration):g}s | {LABELS[name]} | {r['speed_m_s']['mean']:.4f} ± {r['speed_m_s']['std']:.4f} | {r['mean_power_W']['mean']:.2f} | {r['tracking_score']['mean']:.4f} | {r['falls']}/{episodes} | {r['abs_lateral_drift_m']['mean']:.3f} |")
    report += ['', '## 관측 결과', '']
    for duration, policies in summaries.items():
        old, new = policies['reference'], policies['best']
        power_change = (new['mean_power_W']['mean']/old['mean_power_W']['mean']-1)*100
        report.append(f"- {float(duration):g}초: 기존 → 이번 best 속도 {old['speed_m_s']['mean']:.4f} → {new['speed_m_s']['mean']:.4f}m/s, 속도 추종 {old['tracking_score']['mean']:.4f} → {new['tracking_score']['mean']:.4f}, 기계적 파워 {power_change:+.2f}%, 절대 옆 변위 {old['abs_lateral_drift_m']['mean']:.3f} → {new['abs_lateral_drift_m']['mean']:.3f}m.")
    report += ['', '## 영상과 해석 범위', '',
               f"- best와 final의 결정적 행동이 {'완전히 같아 평가 결과와 영상도 동일하다' if equal else '달라 각각 평가·녹화했다'}. 같은 행동의 중복 물리 평가는 재사용했다.",
               '- reference_12s.mp4는 기존 CPU best, best_12s.mp4와 final_12s.mp4는 이번 학습 결과다. 영상 시드는 평가의 첫 시드다. 한 영상은 전체 20개 초기 상태를 대표하지 않는다.',
               '- 영상은 CUDA 학습 중 화면 녹화가 아니라 학습한 정책을 CPU MuJoCo에서 재생한 결과다. 평가는 CPU 전이 성능을 측정하며 Warp 장기 궤적 평가와 구분한다.',
               '- 목표 속도는 0.25m/s다. 빠른 속도만으로 우수하다고 판단하지 않는다. 속도 추종 점수는 높을수록, 넘어짐·몸체 기울기·옆 변위는 낮을수록 좋다.',
               '- 에너지는 절대 기계적 일이며 배터리 소비전력과 다르다. CSV/JSON에 에너지·자세 지표를 포함했다.',
               '- 각 방법의 학습 시드는 하나다. 표준편차는 평가 초기 상태에 따른 변동이며 독립 학습 반복의 불확실성이 아니다. 일반적인 우월성이나 통계적 유의성을 주장하지 않는다.',
               '- 고정 트로트 위상과 역기구학을 이용한 6개 보행 파라미터 학습이며 평지에서만 평가했다. 원본 결과·체크포인트를 덮어쓰지 않았고 PPT는 생성하지 않았다.']
    (output/'평가결과.md').write_text('\n'.join(report)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path('go1_walk/results_warp_new'))
    p.add_argument('--reference', type=Path, default=Path('go1_walk/results/improved/best.zip'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed-start', type=int, default=3000)
    p.add_argument('--durations', type=float, nargs='+', default=[12., 60.])
    args = p.parse_args()
    assess(args.results, args.reference, args.output, args.episodes, args.seed_start, args.durations)
