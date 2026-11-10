"""Run with python -m go1_walk.train --steps 2048 --seed 42."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .env import Go1GaitEnv


class Progress(BaseCallback):
    def __init__(self, directory, mode):
        super().__init__()
        self.directory, self.mode = directory, mode
        self.rows = []
        self.best = -float("inf")
        self.eval_env = Go1GaitEnv(mode)

    def _on_step(self):
        info = self.locals["infos"][0]
        self.rows.append({"episode": self.num_timesteps, **{k: v for k, v in info.items() if k not in ("parameters", "episode", "terminal_observation", "TimeLimit.truncated")}})
        return True

    def _on_rollout_end(self):
        action, _ = self.model.predict(np.zeros(1, np.float32), deterministic=True)
        scores = []
        for seed in [900, 901, 902]:
            self.eval_env.reset(seed=seed)
            r, _ = self.eval_env.simulate(action)
            scores.append(r)
        score = float(np.mean(scores))
        if score > self.best:
            self.best = score
            self.model.save(self.directory / "best")
        print(f"{self.mode}: episodes={self.num_timesteps} validation_reward={score:.3f} best={self.best:.3f}", flush=True)

    def _on_training_end(self):
        self._on_rollout_end()
        with (self.directory / "training.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2048, help="Whole physical rollouts, each up to 5 seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["baseline", "improved", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("go1_walk/results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    for mode in (["baseline", "improved"] if args.mode == "both" else [args.mode]):
        directory = args.output / mode
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "best.zip").exists():
            raise FileExistsError(f"Use another --output directory to preserve {directory}")
        env = Monitor(Go1GaitEnv(mode), str(directory / "monitor.csv"))
        model = PPO("MlpPolicy", env, device="cpu", seed=args.seed,
                    learning_rate=.003, n_steps=64, batch_size=32, n_epochs=10,
                    gamma=0., gae_lambda=1., ent_coef=.002, clip_range=.2,
                    policy_kwargs={"net_arch": [32, 32], "log_std_init": -.5}, verbose=0)
        started = time.time()
        model.learn(args.steps, callback=Progress(directory, mode))
        model.save(directory / "final")
        (directory / "config.json").write_text(json.dumps({
            "seed": args.seed, "requested_rollouts": args.steps,
            "actual_rollouts": model.num_timesteps,
            "sim_seconds_per_rollout": 5., "mode": mode,
            "wall_time_s": time.time() - started,
            "algorithm": "PPO; one action per episode; gamma=0; six gait parameters",
            "validation_seeds": [900, 901, 902], "target_speed_m_s": .25,
        }, indent=2))
        env.close()


if __name__ == "__main__":
    main()
