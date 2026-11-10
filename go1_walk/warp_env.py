"""Batched CUDA physics for the same episodic gait-parameter PPO task.

The small SB3 policy runs on CPU; physics, IK, feedback, and reward run on GPU.
Only actions/reset states and end-of-episode metrics cross the device boundary.
"""
import math
import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from stable_baselines3.common.vec_env import VecEnv
from .env import MODEL, PARAMETERS, decode, targets, configure_actuators


@wp.kernel
def control(p: wp.array2d(dtype=float), ticks: wp.array(dtype=int),
            xmat: wp.array2d(dtype=wp.mat33), vel: wp.array2d(dtype=float),
            bounds: wp.array(dtype=wp.vec2), ctrl: wp.array2d(dtype=float), trunk: int):
    w, leg = wp.tid()
    phase_offset = float(0.0)
    if leg == 1 or leg == 2:
        phase_offset = 0.5
    phase = (float(ticks[w]) * 0.02 * p[w, 1] + phase_offset) % 1.0
    duty = p[w, 5]
    x = float(0.0)
    z = -p[w, 2]
    if phase < duty:
        u = phase / duty
        x = p[w, 4] + p[w, 0] * (0.5 - u)
    else:
        u = (phase - duty) / (1.0 - duty)
        x = p[w, 4] - 0.5 * p[w, 0] * wp.cos(wp.pi * u)
        z += p[w, 3] * wp.sin(wp.pi * u)
    radius = wp.min(wp.sqrt(x*x + z*z), 0.423)
    knee = -wp.acos(wp.clamp((radius*radius - 2.0*0.213*0.213) / (2.0*0.213*0.213), -1.0, 1.0))
    hip = wp.atan2(-x, -z) - 0.5*knee
    roll = wp.clamp(-0.25*xmat[w, trunk][2, 1] - 0.03*vel[w, 3], -0.15, 0.15)
    j = leg*3
    ctrl[w, j] = wp.clamp(roll, bounds[j][0], bounds[j][1])
    ctrl[w, j+1] = wp.clamp(hip, bounds[j+1][0], bounds[j+1][1])
    ctrl[w, j+2] = wp.clamp(knee, bounds[j+2][0], bounds[j+2][1])


@wp.kernel
def integrate(force: wp.array2d(dtype=float), vel: wp.array2d(dtype=float),
              stats: wp.array2d(dtype=float)):
    w = wp.tid()
    if stats[w, 7] == 0.0:
        power = float(0.0)
        for j in range(12):
            power += wp.abs(force[w, j]*vel[w, j+6])
        stats[w, 1] += power*0.004


@wp.kernel
def measure(q: wp.array2d(dtype=float), vel: wp.array2d(dtype=float),
            force: wp.array2d(dtype=float), xmat: wp.array2d(dtype=wp.mat33),
            ticks: wp.array(dtype=int), stats: wp.array2d(dtype=float), trunk: int, baseline: int):
    w = wp.tid()
    ticks[w] += 1
    if stats[w, 7] == 0.0:
        upright = xmat[w, trunk][2, 2]
        tilt = wp.max(0.0, 1.0-upright)
        vx = vel[w, 0]
        track = wp.exp(-(vx-0.25)*(vx-0.25)/0.04)
        power = float(0.0)
        for j in range(12):
            power += wp.abs(force[w, j]*vel[w, j+6])
        reward = track - 2.0*tilt - 0.5*vel[w, 1]*vel[w, 1] - 0.003*power - 0.1*vel[w, 5]*vel[w, 5]
        if baseline == 1:
            reward = vx
        stats[w, 0] += reward
        stats[w, 2] += tilt
        stats[w, 3] += track
        stats[w, 4] += 1.0
        stats[w, 5] = q[w, 0]
        stats[w, 6] = q[w, 1]
        if q[w, 2] < 0.16 or upright < 0.65:
            stats[w, 7] = 1.0


class WarpGo1VecEnv(VecEnv):
    def __init__(self, nworld=64, reward_mode="improved", duration=5., seed=42,
                 nconmax=64, njmax=256, actuator_kv=0.):
        if nworld < 1 or duration <= 0 or reward_mode not in ("baseline", "improved"):
            raise ValueError("Invalid world count, duration, or reward mode")
        wp.init()
        self.device = wp.get_device("cuda:0")  # fail explicitly; never fall back to CPU
        self.reward_mode, self.duration = reward_mode, duration
        self.horizon = math.ceil(duration/.02)
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.model.opt.timestep = .004
        configure_actuators(self.model, actuator_kv)
        self.actuator_kv = actuator_kv
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.render_mode = None
        super().__init__(nworld, gym.spaces.Box(-1., 1., (1,), np.float32),
                         gym.spaces.Box(-1., 1., (6,), np.float32))
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.model)
            self.d = mjw.make_data(self.model, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self.p = wp.zeros((nworld, 6), dtype=float)
            self.ticks = wp.zeros(nworld, dtype=int)
            self.stats = wp.zeros((nworld, 8), dtype=float)
            self.bounds = wp.array(self.model.actuator_ctrlrange.astype(np.float32), dtype=wp.vec2)
            self._initialize(np.zeros((nworld, 6), np.float32))
            # Compile before capture; captured work advances five 4 ms physics steps.
            self._control_step()
            with wp.ScopedCapture() as capture:
                self._control_step()
            self.graph = capture.graph
            with wp.ScopedCapture() as capture:
                mjw.step(self.m, self.d)
            self.settle_graph = capture.graph
        self.actions = None

    def _initialize(self, actions, perturb=True):
        parameters = np.asarray([decode(a) for a in actions], np.float32)
        mjw.reset_data(self.m, self.d)
        qpos = np.tile(self.model.key_qpos[0], (self.num_envs, 1)).astype(np.float32)
        for w, p in enumerate(parameters):
            qpos[w, 7:] = targets(0, p)
            qpos[w, 2] = p[2] + .023
        if perturb:
            qpos[:, 7:] += self.rng.normal(0, .008, (self.num_envs, 12))
        self.d.qpos.assign(qpos)
        self.d.ctrl.assign(np.ascontiguousarray(qpos[:, 7:]))
        self.p.assign(parameters)
        self.ticks.zero_()
        self.stats.zero_()
        mjw.forward(self.m, self.d)
        return parameters

    def _control_step(self):
        wp.launch(control, (self.num_envs, 4), [self.p, self.ticks, self.d.xmat,
                  self.d.qvel, self.bounds, self.d.ctrl, self.trunk])
        for _ in range(5):
            mjw.step(self.m, self.d)
            wp.launch(integrate, self.num_envs, [self.d.actuator_force, self.d.qvel, self.stats])
        wp.launch(measure, self.num_envs, [self.d.qpos, self.d.qvel, self.d.actuator_force,
                  self.d.xmat, self.ticks, self.stats, self.trunk, int(self.reward_mode == "baseline")])

    def reset(self):
        if any(s is not None for s in self._seeds):
            self.rng = np.random.default_rng(self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        return np.zeros((self.num_envs, 1), np.float32)

    def step_async(self, actions):
        self.actions = np.clip(np.asarray(actions, np.float32), -1., 1.)

    def simulate(self, actions, perturb=True):
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.num_envs, 6) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {(self.num_envs, 6)}")
        with wp.ScopedDevice(self.device):
            parameters = self._initialize(actions, perturb)
            for _ in range(125):
                wp.capture_launch(self.settle_graph)
            start = self.d.qpos.numpy()[:, :2].copy()
            for _ in range(self.horizon):
                wp.capture_launch(self.graph)
            stats = self.stats.numpy()
            if hasattr(self.d, "overflow") and np.any(self.d.overflow.numpy()):
                raise RuntimeError("MuJoCo Warp buffer overflow; increase nconmax/njmax")
            if not np.isfinite(stats).all() or not np.isfinite(self.d.qpos.numpy()).all():
                raise FloatingPointError("Non-finite GPU simulation state")
        rewards, infos = [], []
        for w, s in enumerate(stats):
            elapsed = float(s[4])*.02
            reward = float(s[0])/self.horizon - 2.*float(s[7])
            distance = float(s[5]-start[w, 0])
            infos.append(dict(distance_m=distance, speed_m_s=distance/elapsed,
                lateral_drift_m=float(s[6]-start[w, 1]), survived_s=elapsed, fell=bool(s[7]),
                energy_J=float(s[1]), mean_power_W=float(s[1])/elapsed,
                mean_tilt=float(s[2])/float(s[4]), tracking_score=float(s[3])/self.horizon,
                reward=reward, parameters=dict(zip(PARAMETERS, map(float, parameters[w])))))
            rewards.append(reward)
        return np.asarray(rewards, np.float32), infos

    def step_wait(self):
        rewards, infos = self.simulate(self.actions)
        for info in infos:
            info.update(terminal_observation=np.zeros(1, np.float32), **{"TimeLimit.truncated": False})
        return np.zeros((self.num_envs, 1), np.float32), rewards*10., np.ones(self.num_envs, bool), infos

    def close(self):
        wp.synchronize_device(self.device)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Shared batched attributes cannot be set per world")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("Use the batched simulate method")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
