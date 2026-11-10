"""Measure batched GPU vs sequential CPU throughput with identical actions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .env import Go1GaitEnv
from .warp_env import WarpGo1VecEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--batches', type=int, default=3)
    p.add_argument('--duration', type=float, default=5.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.num_envs < 1 or args.batches < 1 or args.duration <= 0:
        p.error('num-envs, batches, and duration must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    actions = np.zeros((args.num_envs, 6), np.float32)
    started = time.perf_counter()
    gpu = WarpGo1VecEnv(args.num_envs, duration=args.duration)
    gpu.simulate(actions, perturb=False)
    warmup = time.perf_counter()-started
    started = time.perf_counter()
    for _ in range(args.batches):
        rewards, gpu_info = gpu.simulate(actions, perturb=False)
    gpu_time = time.perf_counter()-started
    cpu = Go1GaitEnv(duration=args.duration)
    started = time.perf_counter()
    for _ in range(args.batches):
        for action in actions:
            _, cpu_info = cpu.simulate(action, perturb=False)
    cpu_time = time.perf_counter()-started
    result = dict(num_envs=args.num_envs, batches=args.batches, duration_s=args.duration,
        warmup_and_compile_s=warmup, gpu_s=gpu_time, cpu_s=cpu_time,
        gpu_episodes_per_s=args.num_envs*args.batches/gpu_time,
        cpu_episodes_per_s=args.num_envs*args.batches/cpu_time, speedup=cpu_time/gpu_time,
        gpu_sample=gpu_info[0], cpu_sample=cpu_info,
        physics_device=str(gpu.device), policy_device='cpu',
        note='Includes reset and transfers; excludes compilation and PPO. CPU stops on falls; GPU runs the fixed horizon with masked metrics.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    gpu.close()
    cpu.close()


if __name__ == '__main__':
    main()
