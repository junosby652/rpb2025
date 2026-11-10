"""PPO chooses six gait parameters once per physical rollout.

This is an episodic gait-parameter task, not end-to-end joint control.
The diagonal trot phase and inverse kinematics are supplied as a prior.
"""
from pathlib import Path
import json
import math
import gymnasium as gym
import mujoco
import numpy as np

MODEL = Path(__file__).parent / "assets/unitree_go1/scene.xml"
PARAMETERS = ("stride_m", "frequency_hz", "leg_length_m", "clearance_m", "foot_offset_m", "stance_fraction")
LOW = np.array([.03, 1., .245, .015, -.045, .50])
HIGH = np.array([.17, 2.8, .31, .075, .045, .72])


def configure_actuators(model, actuator_kv=0.):
    """Optional velocity damping on position servos; retain the original kp/limits."""
    if not np.isfinite(actuator_kv) or actuator_kv < 0:
        raise ValueError("actuator_kv must be finite and nonnegative")
    if actuator_kv > 0:
        model.actuator_biasprm[:, 2] = -actuator_kv
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def controller_options(checkpoint):
    """Restore physics/controller settings stored beside a policy checkpoint."""
    config = Path(checkpoint).parent / "config.json"
    return {"actuator_kv": json.loads(config.read_text()).get("actuator_kv", 0.) if config.exists() else 0.}


def decode(action):
    return LOW + (np.clip(action, -1, 1) + 1) * .5 * (HIGH - LOW)


def targets(t, parameters):
    stride, frequency, height, clearance, offset, duty = parameters
    phase = (t * frequency + np.array([0., .5, .5, 0.])) % 1
    stance = phase < duty
    u = np.where(stance, phase / duty, (phase - duty) / (1 - duty))
    # Smooth swing, linear stance: feet move backward while on the ground.
    x = offset + np.where(stance, stride * (.5 - u), -.5 * stride * np.cos(np.pi * u))
    z = -height + np.where(stance, 0., clearance * np.sin(np.pi * u))
    radius = np.minimum(np.hypot(x, z), .423)
    knee = -np.arccos(np.clip((radius**2 - 2 * .213**2) / (2 * .213**2), -1, 1))
    hip = np.arctan2(-x, -z) - knee * .5
    result = np.zeros((4, 3))
    result[:, 1], result[:, 2] = hip, knee
    return result.ravel()


class Go1GaitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward_mode="improved", duration=5., target_speed=.25, actuator_kv=0.):
        super().__init__()
        if reward_mode not in ("baseline", "improved"):
            raise ValueError(reward_mode)
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.model.opt.timestep = .004
        configure_actuators(self.model, actuator_kv)
        self.actuator_kv = actuator_kv
        self.data = mujoco.MjData(self.model)
        self.duration, self.target_speed, self.reward_mode = duration, target_speed, reward_mode
        self.action_space = gym.spaces.Box(-1., 1., (6,), np.float32)
        self.observation_space = gym.spaces.Box(-1., 1., (1,), np.float32)
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def simulate(self, action, frame_callback=None, perturb=True, physics_callback=None):
        p = decode(action)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[7:] = targets(0, p)
        self.data.qpos[2] = p[2] + .023
        if perturb:
            self.data.qpos[7:] += self.np_random.normal(0, .008, 12)
        self.data.ctrl[:] = self.data.qpos[7:]
        mujoco.mj_forward(self.model, self.data)
        # Let gravity establish contact before measuring displacement and work.
        for _ in range(125):
            mujoco.mj_step(self.model, self.data)
        start_x = float(self.data.qpos[0])
        start_y = float(self.data.qpos[1])
        reward_sum = energy = tilt_sum = tracking_sum = 0.
        fell = False
        n = math.ceil(self.duration / .02)
        for step in range(n):
            control = targets(step * .02, p)
            # Small attitude feedback; same fixed controller in both experiments.
            rot = self.data.xmat[self.trunk].reshape(3, 3)
            control[0::3] = np.clip(-.25 * rot[2, 1] - .03 * self.data.qvel[3], -.15, .15)
            self.data.ctrl[:] = np.clip(control, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
                if physics_callback is not None:
                    physics_callback(self.model, self.data)
                energy += float(np.sum(np.abs(self.data.actuator_force * self.data.qvel[6:]))) * .004
            vx, vy = self.data.qvel[:2]
            upright = float(self.data.xmat[self.trunk, 8])
            tilt = max(0., 1 - upright)
            track = math.exp(-float((vx - self.target_speed)**2) / .04)
            tracking_sum += track
            tilt_sum += tilt
            if self.reward_mode == "baseline":
                reward_sum += float(vx)
            else:
                power = float(np.sum(np.abs(self.data.actuator_force * self.data.qvel[6:])))
                reward_sum += track - 2. * tilt - .5 * float(vy**2) - .003 * power - .1 * float(self.data.qvel[5]**2)
            if frame_callback is not None and step % 2 == 0:
                frame_callback(self.model, self.data)
            if self.data.qpos[2] < .16 or upright < .65 or not np.isfinite(self.data.qpos).all():
                fell = True
                break
        elapsed = (step + 1) * .02
        # Normalize by the FULL horizon so early termination cannot hide bad steps.
        reward = reward_sum / n - (2. if fell else 0.)
        distance = float(self.data.qpos[0]) - start_x
        info = dict(distance_m=distance, speed_m_s=distance / elapsed,
                    lateral_drift_m=float(self.data.qpos[1]) - start_y,
                    survived_s=elapsed, fell=fell, energy_J=energy,
                    mean_power_W=energy / elapsed, mean_tilt=tilt_sum / (step + 1),
                    tracking_score=tracking_sum / n, reward=reward,
                    parameters=dict(zip(PARAMETERS, map(float, p))))
        return float(reward), info

    def step(self, action):
        reward, info = self.simulate(action)
        return np.zeros(1, np.float32), reward * 10, True, False, info
