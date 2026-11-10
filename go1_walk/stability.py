"""Paired vibration audit, validation-only selection, and comparison videos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
import torch
from .evaluate import plt, record
from stable_baselines3 import PPO
from .env import Go1GaitEnv, controller_options


def vibration_metrics(trace, dt=.004):
    # Exclude the first 0.5 s; Hann taper reduces leakage from the slow gait cycle.
    values = np.asarray(trace, dtype=np.float64)[round(.5/dt):]
    if len(values) < 8:
        raise ValueError('Rollout too short for vibration measurements')
    velocity = values[:, :18]
    window = np.hanning(len(values))
    spectrum = np.fft.rfft((velocity-velocity.mean(axis=0))*window[:, None], axis=0)
    frequency = np.fft.rfftfreq(len(values), dt)
    weights = np.full(len(frequency), 2.)
    weights[0] = 1.
    if len(values) % 2 == 0:
        weights[-1] = 1.
    high = frequency > 10.
    mean_square = np.sum(weights[high, None]*np.abs(spectrum[high])**2, axis=0)
    mean_square /= len(values)*np.sum(window**2)
    return dict(joint_high_frequency_rms=float(np.sqrt(np.mean(mean_square[6:18]))),
                body_high_frequency_rms=float(np.sqrt(np.mean(mean_square[:3]))),
                torque_rate_rms=float(np.sqrt(np.mean((np.diff(values[:, 18:], axis=0)/dt)**2))))


def trial(action, kv, duration, seed):
    env = Go1GaitEnv(duration=duration, actuator_kv=kv)
    env.reset(seed=seed)
    samples = []
    def sample(model, data):
        samples.append(np.concatenate((data.qvel.copy(), data.actuator_force.copy())))
    try:
        _, info = env.simulate(action, physics_callback=sample)
    finally:
        env.close()
    trace = np.asarray(samples)
    info.update(vibration_metrics(trace), abs_lateral_drift_m=abs(info['lateral_drift_m']))
    return info, trace


def summarize(trials):
    keys = [k for k, v in trials[0].items() if k not in ('parameters', 'fell')]
    return {**{k: dict(mean=float(np.mean([r[k] for r in trials])),
                       std=float(np.std([r[k] for r in trials]))) for k in keys},
            'falls': sum(t['fell'] for t in trials), 'episodes': len(trials)}


def audit(reference, candidate, output, episodes=10, seed_start=5000):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if episodes < 1 or seed_start <= 902:
        raise ValueError('Use positive episodes and held-out seeds above 902')
    reference, candidate = Path(reference), Path(candidate)
    torch.set_num_threads(1)
    policy = {}
    for name, path in (('original', reference), ('retrained', candidate)):
        policy[name] = PPO.load(path, device='cpu').predict(np.zeros(1, np.float32), deterministic=True)[0]
    kv = controller_options(candidate)['actuator_kv']
    if kv <= 0:
        raise ValueError('Candidate config must specify positive actuator_kv')
    variants = {'original': (policy['original'], 0.), 'damped': (policy['original'], kv),
                'retrained': (policy['retrained'], kv)}
    output.mkdir(parents=True)
    validation = {}
    for name, (action, damping) in variants.items():
        validation[name] = summarize([trial(action, damping, 5., seed)[0] for seed in (900, 901, 902)])
    baseline = validation['original']
    eligible = [n for n in ('damped', 'retrained') if validation[n]['falls'] == 0
                and validation[n]['speed_m_s']['mean'] >= .85*baseline['speed_m_s']['mean']
                and validation[n]['joint_high_frequency_rms']['mean'] <= .75*baseline['joint_high_frequency_rms']['mean']]
    if not eligible:
        raise RuntimeError('No candidate retained speed and reduced vibration on validation seeds')
    selected = max(eligible, key=lambda n: validation[n]['reward']['mean'])
    print('Selected on validation seeds 900-902:', selected, flush=True)
    results, rows, traces = {}, [], {}
    seeds = list(range(seed_start, seed_start+episodes))
    for duration in (12., 60.):
        results[str(int(duration))] = {}
        for name, (action, damping) in variants.items():
            trials = []
            for seed in seeds:
                info, trace = trial(action, damping, duration, seed)
                trials.append(info)
                rows.append(dict(policy=name, duration_s=duration, seed=seed,
                                 **{k: v for k, v in info.items() if k != 'parameters'}))
                if duration == 12. and seed == seeds[0]:
                    traces[name] = trace
            result = summarize(trials)
            results[str(int(duration))][name] = result
            print(f"{duration:g}s {name}: speed={result['speed_m_s']['mean']:.4f}, "
                  f"joint_HF={result['joint_high_frequency_rms']['mean']:.4f}, "
                  f"power={result['mean_power_W']['mean']:.1f}, "
                  f"drift={result['abs_lateral_drift_m']['mean']:.3f}, falls={result['falls']}/{episodes}", flush=True)
        with (output/'evaluation.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    result = dict(selected=selected, selection_seeds=[900, 901, 902], validation=validation,
        selection_rule='No falls; retain >=85% original speed; reduce >10Hz joint velocity RMS >=25%; then maximize original improved reward.',
        evaluation_seeds=seeds, results=results,
        variants={n: dict(action=a.tolist(), actuator_kv=k) for n, (a, k) in variants.items()},
        checkpoints={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (reference, candidate)},
        evaluation_backend='CPU MuJoCo', sample_hz=250,
        vibration_definition='Hann-windowed >10Hz joint-velocity RMS, excluding first 0.5 s; includes gait harmonics, not only numerical error.')
    (output/'evaluation.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(output/'traces_250hz.npz', **traces)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis_row, (duration, policies) in zip(axes, results.items()):
        for ax, metric, label in zip(axis_row,
                ('joint_high_frequency_rms', 'speed_m_s', 'mean_power_W', 'abs_lateral_drift_m'),
                ('Joint velocity >10Hz RMS (rad/s)', 'Speed (m/s)', 'Mechanical power (W)', 'Absolute lateral drift (m)')):
            ax.bar(list(policies), [r[metric]['mean'] for r in policies.values()],
                   yerr=[r[metric]['std'] for r in policies.values()], capsize=4,
                   color=['#94a3b8', '#06b6d4', '#f59e0b'])
            ax.set_title(f'{duration}s: {label}')
            ax.grid(axis='y', alpha=.2)
        axis_row[1].axhline(.25, color='black', linestyle='--', linewidth=1)
    figure.tight_layout()
    figure.savefig(output/'comparison.png', dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for name in ('original', selected):
        trace = traces[name]
        t = np.arange(len(trace))*.004
        mask = (t >= 8.) & (t < 9.)
        axes[0].plot(t[mask], trace[mask, 8], label=name, linewidth=1)
        axes[1].plot(t[mask], trace[mask, 20], label=name, linewidth=1)
    axes[0].set_ylabel('FR knee velocity (rad/s)')
    axes[1].set(ylabel='FR knee torque (N m)', xlabel='Episode time (s)')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output/'joint_trace.png', dpi=170)
    plt.close(figure)
    return finalize(result, output)


def finalize(result, output):
    """Render saved evaluations without repeating the long physics trials."""
    output = Path(output)
    selected, results, seeds = result['selected'], result['results'], result['evaluation_seeds']
    episodes = len(seeds)
    variants = {n: (np.asarray(v['action'], np.float32), v['actuator_kv'])
                for n, v in result['variants'].items()}
    kv = variants['retrained'][1]
    for name, filename in (('original', 'before_12s.mp4'), (selected, 'stabilized_12s.mp4')):
        action, damping = variants[name]
        env = Go1GaitEnv(duration=12., actuator_kv=damping)
        try:
            record(env, action, output/filename, seed=seeds[0])
        finally:
            env.close()
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(output/filename))
        if frames != 300 or abs(seconds-12.) > .1:
            raise RuntimeError(f'Unexpected video length: {filename}, {frames}, {seconds}')
        print('Video:', output/filename, flush=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-n',
        '-i', str(output/'before_12s.mp4'), '-i', str(output/'stabilized_12s.mp4'),
        '-filter_complex', "[0:v]drawtext=text='Before':x=16:y=16:fontsize=24:fontcolor=white[l];[1:v]drawtext=text='Stabilized':x=16:y=16:fontsize=24:fontcolor=white[r];[l][r]hstack=inputs=2[v]",
        '-map', '[v]', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        str(output/'side_by_side.mp4')], check=True)
    report = ['# 보행 떨림 완화 평가', '',
        f'기존 Warp 학습 정책을 유지한 감쇠 제어(damped)와 해당 제어에서 추가 학습한 정책(retrained)을 비교했다. 선택은 검증 시드 900~902의 5초 실험으로만 했고 **{selected}**를 선택했다. 최종 평가는 새 시드 {seeds[0]}~{seeds[-1]}에서 12초·60초씩 수행했다.', '',
        f'제어 변경: 기존 kp=100과 토크 제한·관절 마찰·접촉·50Hz 목표 갱신·250Hz 물리를 유지하고 위치 액추에이터에 kv={kv:g} N m s/rad 속도 감쇠를 추가했다. 양수 kv에서는 implicitfast 적분을 사용한다. 발 궤적과 보상식은 유지했다. 원본 XML과 기존 체크포인트는 변경하지 않았다.', '',
        '| 길이 | 조건 | 관절 >10Hz RMS rad/s | 속도 m/s | 기계적 파워 W | 넘어짐 | 절대 옆 변위 m |',
        '|---|---|---:|---:|---:|---:|---:|']
    for duration, policies in results.items():
        for name, r in policies.items():
            report.append(f"| {duration}s | {name} | {r['joint_high_frequency_rms']['mean']:.4f} ± {r['joint_high_frequency_rms']['std']:.4f} | {r['speed_m_s']['mean']:.4f} | {r['mean_power_W']['mean']:.2f} | {r['falls']}/{episodes} | {r['abs_lateral_drift_m']['mean']:.3f} |")
    report += ['', '## 관측 결과', '']
    for duration, policies in results.items():
        old, new = policies['original'], policies[selected]
        vibration_change = (new['joint_high_frequency_rms']['mean']/old['joint_high_frequency_rms']['mean']-1)*100
        power_change = (new['mean_power_W']['mean']/old['mean_power_W']['mean']-1)*100
        report.append(f"- {duration}초: 고주파 관절 속도 {vibration_change:+.1f}%, 기계적 파워 {power_change:+.1f}%, 평균 전진 속도 {old['speed_m_s']['mean']:.4f} → {new['speed_m_s']['mean']:.4f}m/s, 절대 옆 변위 {old['abs_lateral_drift_m']['mean']:.3f} → {new['abs_lateral_drift_m']['mean']:.3f}m.")
    report += ['', '## 해석과 재현', '',
        '- 관절 떨림은 250Hz 물리 상태에서 첫 0.5초를 제외하고 Hann window로 10Hz 초과 속도 성분의 RMS를 계산했다. 토크 변화율도 CSV/JSON에 기록했다. 고주파 성분에는 보행의 고조파도 포함되며 모두 수치 오류라는 뜻은 아니다.',
        '- 화면 프레임을 보간하거나 영상을 느리게 해서 부드러워 보이게 만든 것이 아니다. 영상은 실제 제어 변경을 CPU MuJoCo에서 재생한 25fps 결과다. side_by_side.mp4의 왼쪽은 원본, 오른쪽은 선택한 제어·정책이다.',
        '- CPU·CUDA 일치도는 별도 테스트에서 검증한다. 최종 장기 지표는 CPU MuJoCo에서 측정했으며 GPU 장기 궤적과 동일함을 보장하지 않는다.',
        '- 감쇠가 추가되면 모델의 물리·제어 조건이 달라진다. config.json과 체크포인트를 함께 보관해야 한다. 평가·영상 도구는 이 설정을 자동 복원한다.',
        '- 에너지는 절대 기계적 일이며 전기·배터리 소비량이 아니다. 추가된 액추에이터 감쇠의 모델링을 포함하므로 실물 로봇 효율 향상을 뜻하지 않는다.',
        '- 각 정책의 학습 시드는 하나다. 최종 평가 시드는 독립 학습 반복이 아니며, 마찰·지형·외란 일반화를 검증한 결과가 아니다.',
        '- 이 작업은 떨림 완화를 다룬다. 상수 관측·에피소드당 한 행동의 한계로 누적 heading 오차가 남을 수 있으며, 직진 문제를 해결했다고 가정하지 않는다.',
        '- 원본 학습 결과는 보존했으며 PPT는 생성하지 않았다.']
    (output/'안정화평가.md').write_text('\n'.join(report)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=Path('go1_walk/results_warp_new/improved/best.zip'))
    parser.add_argument('--candidate', type=Path, default=Path('go1_walk/results_stability/retrained/improved/best.zip'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=5000)
    parser.add_argument('--render-only', action='store_true', help='Regenerate videos/report from existing evaluation.json')
    args = parser.parse_args()
    if args.render_only:
        finalize(json.loads((args.output/'evaluation.json').read_text()), args.output)
    else:
        audit(args.reference, args.candidate, args.output, args.episodes, args.seed_start)
