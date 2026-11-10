import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from .env import Go1GaitEnv, decode, LOW, HIGH, targets


class PhysicalEnvironmentTests(unittest.TestCase):
    def test_servo_velocity_damping_force(self):
        import mujoco
        env = Go1GaitEnv(actuator_kv=.5)
        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        env.data.ctrl[:] = env.data.qpos[7:]
        mujoco.mj_forward(env.model, env.data)
        static = env.data.actuator_force.copy()
        env.data.qvel[6:] = 1.
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.data.actuator_force - static, -.5, atol=1e-9)
        env.close()

    def test_invalid_damping(self):
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                Go1GaitEnv(actuator_kv=value)

    def test_action_bounds(self):
        np.testing.assert_allclose(decode(np.full(6, -2.)), LOW)
        np.testing.assert_allclose(decode(np.full(6, 2.)), HIGH)

    def test_kinematics_and_diagonal_phase(self):
        parameters = decode(np.zeros(6))
        for t in np.linspace(0, 1, 30):
            q = targets(t, parameters).reshape(4, 3)
            np.testing.assert_allclose(q[0], q[3])
            np.testing.assert_allclose(q[1], q[2])
            x = -.213 * (np.sin(q[:, 1]) + np.sin(q[:, 1] + q[:, 2]))
            z = -.213 * (np.cos(q[:, 1]) + np.cos(q[:, 1] + q[:, 2]))
            self.assertTrue(np.all(np.abs(x) <= parameters[0] / 2 + 1e-8))
            self.assertTrue(np.all(z >= -parameters[2] - 1e-8))
            self.assertTrue(np.all(z <= -parameters[2] + parameters[3] + 1e-8))

    def test_reproducible_physics_and_action_effect(self):
        env = Go1GaitEnv(duration=2.)
        env.reset(seed=77)
        r1, first = env.simulate(np.zeros(6))
        env.reset(seed=77)
        r2, second = env.simulate(np.zeros(6))
        self.assertEqual(first, second)
        self.assertEqual(r1, r2)
        self.assertGreater(first["energy_J"], 0)
        env.reset(seed=77)
        _, different = env.simulate(np.ones(6) * .5)
        self.assertNotAlmostEqual(first["distance_m"], different["distance_m"], places=3)

    def test_gymnasium_contract(self):
        check_env(Go1GaitEnv(duration=.2), warn=True)


if __name__ == "__main__":
    unittest.main()
