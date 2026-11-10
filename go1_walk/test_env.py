import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from .env import Go1GaitEnv, decode, LOW, HIGH, targets


class PhysicalEnvironmentTests(unittest.TestCase):
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
