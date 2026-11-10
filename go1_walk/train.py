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
from .env import Go1GaitEnv, decode, PARAMETERS


class Progress(BaseCallback):
    def __init__(self, directory, mode, validation_every=64):
        super().__init__()
        self.directory, self.mode = directory, mode
        self.rows = []
        self.best = -float("inf")
        self.eval_env = Go1GaitEnv(mode)
        self.validation_every = validation_every
        self.validation_rows = []

    def _on_training_start(self):
        # Preserve the starting policy as a candidate, even if learning regresses.
        self._evaluate()

    def _on_step(self):
        info = self.locals["infos"][0]
        self.rows.append({"episode": self.num_timesteps, **{k: v for k, v in info.items() if k not in ("parameters", "episode", "terminal_observation", "TimeLimit.truncated")}})
        return True

    def _on_rollout_end(self):
        if len(self.rows) % self.validation_every == 0:
            self._evaluate()

    def _evaluate(self):
        action, _ = self.model.predict(np.zeros(1, np.float32), deterministic=True)
        scores = []
        for seed in [900, 901, 902]:
            self.eval_env.reset(seed=seed)
            r, _ = self.eval_env.simulate(action)
            scores.append(r)
        score = float(np.mean(scores))
        self.validation_rows.append({"additional_rollouts": len(self.rows),
                                     "checkpoint_rollouts": self.num_timesteps,
                                     "mean_reward": score})
        if score > self.best:
            self.best = score
            self.model.save(self.directory / "best")
            (self.directory / "best_metrics.json").write_text(json.dumps({
                **self.validation_rows[-1], "validation_seeds": [900, 901, 902],
                "parameters": dict(zip(PARAMETERS, map(float, decode(action)))),
            }, indent=2))
        print(f"{self.mode}: episodes={self.num_timesteps} validation_reward={score:.3f} best={self.best:.3f}", flush=True)

    def _on_training_end(self):
        self._evaluate()
        with (self.directory / "training.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)
        with (self.directory / "validation.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.validation_rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.validation_rows)
        self.eval_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2048, help="Whole physical rollouts, each up to 5 seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["baseline", "improved", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("go1_walk/results"))
    parser.add_argument("--resume", type=Path, help="Continue from a PPO checkpoint, retaining optimizer state")
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--validation-every", type=int, default=64)
    args = parser.parse_args()
    if args.steps <= 0 or args.validation_every <= 0 or args.learning_rate <= 0:
        parser.error("steps, validation-every and learning-rate must be positive")
    if args.resume and args.mode == "both":
        parser.error("Choose the checkpoint's --mode when using --resume")
    torch.set_num_threads(1)
    for mode in (["baseline", "improved"] if args.mode == "both" else [args.mode]):
        directory = args.output / mode
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Use another --output directory to preserve {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        env = Monitor(Go1GaitEnv(mode), str(directory / "monitor.csv"))
        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu", seed=args.seed,
                             learning_rate=args.learning_rate)
        else:
            model = PPO("MlpPolicy", env, device="cpu", seed=args.seed,
                        learning_rate=args.learning_rate, n_steps=64, batch_size=32, n_epochs=10,
                        gamma=0., gae_lambda=1., ent_coef=.002, clip_range=.2,
                        policy_kwargs={"net_arch": [32, 32], "log_std_init": -.5}, verbose=0)
        previous_rollouts = model.num_timesteps
        started = time.time()
        model.learn(args.steps, reset_num_timesteps=not bool(args.resume),
                    callback=Progress(directory, mode, args.validation_every))
        model.save(directory / "final")
        (directory / "config.json").write_text(json.dumps({
            "seed": args.seed, "requested_rollouts": args.steps,
            "actual_rollouts": model.num_timesteps - previous_rollouts,
            "previous_checkpoint_rollouts": previous_rollouts,
            "total_checkpoint_rollouts": model.num_timesteps,
            "resume_from": str(args.resume) if args.resume else None,
            "learning_rate": args.learning_rate,
            "sim_seconds_per_rollout": 5., "mode": mode,
            "wall_time_s": time.time() - started,
            "algorithm": "PPO; one action per episode; gamma=0; six gait parameters",
            "validation_seeds": [900, 901, 902], "target_speed_m_s": .25,
        }, indent=2))
        env.close()


if __name__ == "__main__":
    main()
