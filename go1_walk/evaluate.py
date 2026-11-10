"""Held-out physical evaluations, plots, and headless MP4 recordings."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/go1-mesa-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/go1-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/go1-cache")
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from .env import Go1GaitEnv, controller_options


def evaluate(root, seeds=range(1000, 1010), duration=12., video=True):
    torch.set_num_threads(1)
    root = Path(root)
    summary, rows = {}, []
    for name in ["initial", "baseline", "improved"]:
        action = np.zeros(6, np.float32) if name == "initial" else PPO.load(root / name / "best", device="cpu").predict(np.zeros(1, np.float32), deterministic=True)[0]
        settings = controller_options(root / ("improved" if name == "initial" else name) / "best.zip")
        env = Go1GaitEnv("improved", duration=duration, **settings)
        trials = []
        for seed in seeds:
            env.reset(seed=seed)
            _, info = env.simulate(action)
            rows.append({"policy": name, "seed": seed, **{k: v for k, v in info.items() if k != "parameters"}})
            trials.append(info)
        keys = ["speed_m_s", "distance_m", "mean_power_W", "energy_J", "mean_tilt", "lateral_drift_m", "survived_s", "tracking_score"]
        summary[name] = {k: {"mean": float(np.mean([r[k] for r in trials])), "std": float(np.std([r[k] for r in trials]))} for k in keys}
        summary[name].update(falls=sum(r["fell"] for r in trials), episodes=len(trials),
                             duration_s=duration, action=action.tolist(), parameters=trials[0]["parameters"],
                             controller=settings)
        if video:
            record(env, action, root / f"{name}.mp4")
        print(name, json.dumps(summary[name]), flush=True)
    with (root / "evaluation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (root / "evaluation.json").write_text(json.dumps(summary, indent=2))
    plot(root, summary)
    return summary


def record(env, action, destination, seed=1000):
    env.reset(seed=seed)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -20
    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
        with imageio.get_writer(str(destination), fps=25, codec="libx264", quality=7) as writer:
            frame_count = 0
            def frame(model, data):
                nonlocal frame_count
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                # Software rendering on CPU: omit expensive visual effects.
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                pixels = renderer.render()
                writer.append_data(pixels)
                if frame_count in (0, 75):
                    imageio.imwrite(destination.with_suffix(".png"), pixels)
                frame_count += 1
            env.simulate(action, frame_callback=frame)


def plot(root, summary):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    names = list(summary)
    for ax, metric, title in zip(axes, ["speed_m_s", "mean_power_W", "tracking_score"], ["Forward speed (m/s)", "Absolute mechanical power (W)", "Speed tracking score (0–1)"]):
        ax.bar(names, [summary[n][metric]["mean"] for n in names], yerr=[summary[n][metric]["std"] for n in names], color=["#94a3b8", "#f59e0b", "#06b6d4"], capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].axhline(.25, color="black", linestyle="--", label="Target 0.25 m/s")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ["baseline", "improved"]):
        with (root / name / "training.csv").open() as f:
            records = list(csv.DictReader(f))
        y = np.array([float(r["reward"]) for r in records])
        ax.plot(np.arange(len(y)) + 1, y, alpha=.15, linewidth=.5)
        window = min(64, len(y))
        ax.plot(np.arange(window, len(y) + 1), np.convolve(y, np.ones(window)/window, mode="valid"))
        ax.set(title=f"{name}: own training objective", xlabel="Physical rollouts", ylabel="Unscaled reward")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("go1_walk/results"))
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--duration", type=float, default=12.)
    args = p.parse_args()
    evaluate(args.results, duration=args.duration, video=not args.no_video)
