"""Run explicitly on a CUDA-capable host: python -m unittest go1_walk.test_warp -v."""
import unittest
import numpy as np
from .env import Go1GaitEnv
from .warp_env import WarpGo1VecEnv


class WarpPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = WarpGo1VecEnv(nworld=4, duration=.2)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_matches_short_cpu_rollouts(self):
        actions = np.array([np.zeros(6), np.ones(6)*.5, np.ones(6)*-.5, np.ones(6)])
        rewards, gpu = self.env.simulate(actions, perturb=False)
        cpu = Go1GaitEnv(duration=.2, actuator_kv=self.env.actuator_kv)
        for action, reward, info in zip(actions, rewards, gpu):
            expected, reference = cpu.simulate(action, perturb=False)
            self.assertAlmostEqual(float(reward), expected, delta=.05)
            self.assertAlmostEqual(info['distance_m'], reference['distance_m'], delta=.01)
            self.assertAlmostEqual(info['energy_J'], reference['energy_J'], delta=max(.2, reference['energy_J']*.1))
            self.assertEqual(info['fell'], reference['fell'])
        cpu.close()

    def test_reset_and_action_effect(self):
        actions = np.zeros((4, 6), np.float32)
        first, first_info = self.env.simulate(actions, perturb=False)
        self.env.simulate(np.ones((4, 6), np.float32), perturb=False)
        second, second_info = self.env.simulate(actions, perturb=False)
        np.testing.assert_allclose(first, second, atol=1e-5)
        np.testing.assert_allclose([i['energy_J'] for i in first_info],
                                   [i['energy_J'] for i in second_info], rtol=1e-5)
        other, _ = self.env.simulate(np.ones((4, 6), np.float32)*.5, perturb=False)
        self.assertGreater(float(np.max(np.abs(first-other))), .001)

    def test_vector_contract(self):
        self.env.seed(12)
        self.assertEqual(self.env.reset().shape, (4, 1))
        obs, rewards, dones, infos = self.env.step(np.zeros((4, 6), np.float32))
        self.assertEqual(obs.shape, (4, 1))
        self.assertTrue(dones.all())
        self.assertTrue(np.isfinite(rewards).all())
        self.assertEqual(len(infos), 4)
        self.assertTrue(all(i['survived_s'] <= .2 for i in infos))


class DampedWarpPhysicsTests(WarpPhysicsTests):
    @classmethod
    def setUpClass(cls):
        cls.env = WarpGo1VecEnv(nworld=4, duration=.2, actuator_kv=.5)


if __name__ == '__main__':
    unittest.main()
