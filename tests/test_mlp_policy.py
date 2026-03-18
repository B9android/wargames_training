# tests/test_mlp_policy.py
"""Tests for models/mlp_policy.py — BattalionMlpPolicy."""

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mlp_policy import BattalionFeaturesExtractor, BattalionMlpPolicy


def _make_ppo_model(verbose: int = 0):
    """Create a PPO model with BattalionMlpPolicy against BattalionEnv."""
    from stable_baselines3 import PPO
    from envs.battalion_env import BattalionEnv

    env = BattalionEnv()
    model = PPO("BattalionMlpPolicy", env, verbose=verbose)
    return model, env


class TestPPOInit(unittest.TestCase):
    """PPO('BattalionMlpPolicy', env) should initialise without errors."""

    def test_ppo_init_by_string(self) -> None:
        model, env = _make_ppo_model()
        self.assertIsInstance(model.policy, BattalionMlpPolicy)
        env.close()

    def test_ppo_init_by_class(self) -> None:
        from stable_baselines3 import PPO
        from envs.battalion_env import BattalionEnv

        env = BattalionEnv()
        model = PPO(BattalionMlpPolicy, env, verbose=0)
        self.assertIsInstance(model.policy, BattalionMlpPolicy)
        env.close()

    def test_policy_aliases_registered(self) -> None:
        from stable_baselines3 import PPO

        self.assertIn("BattalionMlpPolicy", PPO.policy_aliases)
        self.assertIs(PPO.policy_aliases["BattalionMlpPolicy"], BattalionMlpPolicy)


class TestForwardPassShapes(unittest.TestCase):
    """Verify output shapes at every layer of the network."""

    def setUp(self) -> None:
        self.model, self.env = _make_ppo_model()
        self.policy = self.model.policy
        self.policy.eval()

    def tearDown(self) -> None:
        self.env.close()

    def _obs_tensor(self, batch_size: int = 64) -> torch.Tensor:
        obs = np.random.rand(batch_size, 12).astype("float32")
        return torch.tensor(obs)

    def test_features_extractor_output_shape(self) -> None:
        obs = self._obs_tensor(64)
        with torch.no_grad():
            features = self.policy.extract_features(
                obs, self.policy.features_extractor
            )
        self.assertEqual(features.shape, (64, 64))

    def test_latent_pi_shape(self) -> None:
        obs = self._obs_tensor(64)
        with torch.no_grad():
            features = self.policy.extract_features(
                obs, self.policy.features_extractor
            )
            latent_pi, _ = self.policy.mlp_extractor(features)
        self.assertEqual(latent_pi.shape, (64, 32))

    def test_latent_vf_shape(self) -> None:
        obs = self._obs_tensor(64)
        with torch.no_grad():
            features = self.policy.extract_features(
                obs, self.policy.features_extractor
            )
            _, latent_vf = self.policy.mlp_extractor(features)
        self.assertEqual(latent_vf.shape, (64, 32))

    def test_value_output_shape(self) -> None:
        obs = self._obs_tensor(64)
        with torch.no_grad():
            features = self.policy.extract_features(
                obs, self.policy.features_extractor
            )
            _, latent_vf = self.policy.mlp_extractor(features)
            values = self.policy.value_net(latent_vf)
        self.assertEqual(values.shape, (64, 1))

    def test_action_mean_shape(self) -> None:
        obs = self._obs_tensor(64)
        with torch.no_grad():
            features = self.policy.extract_features(
                obs, self.policy.features_extractor
            )
            latent_pi, _ = self.policy.mlp_extractor(features)
            action_mean = self.policy.action_net(latent_pi)
        # action space has 3 dimensions (move, rotate, fire)
        self.assertEqual(action_mean.shape, (64, 3))

    def test_action_mean_tanh_bounded(self) -> None:
        """Actor output must lie in [-1, 1] due to Tanh activation."""
        obs = self._obs_tensor(64)
        with torch.no_grad():
            features = self.policy.extract_features(
                obs, self.policy.features_extractor
            )
            latent_pi, _ = self.policy.mlp_extractor(features)
            action_mean = self.policy.action_net(latent_pi)
        self.assertGreaterEqual(float(action_mean.min()), -1.0)
        self.assertLessEqual(float(action_mean.max()), 1.0)


class TestArchitecture(unittest.TestCase):
    """Verify the network structure matches the specification."""

    def setUp(self) -> None:
        self.model, self.env = _make_ppo_model()
        self.policy = self.model.policy

    def tearDown(self) -> None:
        self.env.close()

    def test_features_extractor_is_battalion_extractor(self) -> None:
        self.assertIsInstance(
            self.policy.features_extractor, BattalionFeaturesExtractor
        )

    def test_features_extractor_trunk_structure(self) -> None:
        net = self.policy.features_extractor.net
        # 6 sub-modules: Linear, ReLU, LayerNorm, Linear, ReLU, LayerNorm
        modules = list(net.children())
        self.assertEqual(len(modules), 6)
        self.assertIsInstance(modules[0], nn.Linear)
        self.assertEqual(modules[0].in_features, 12)
        self.assertEqual(modules[0].out_features, 128)
        self.assertIsInstance(modules[1], nn.ReLU)
        self.assertIsInstance(modules[2], nn.LayerNorm)
        self.assertIsInstance(modules[3], nn.Linear)
        self.assertEqual(modules[3].in_features, 128)
        self.assertEqual(modules[3].out_features, 64)
        self.assertIsInstance(modules[4], nn.ReLU)
        self.assertIsInstance(modules[5], nn.LayerNorm)

    def test_action_net_ends_with_tanh(self) -> None:
        last = list(self.policy.action_net.children())[-1]
        self.assertIsInstance(last, nn.Tanh)

    def test_parameter_count_logged(self) -> None:
        """Parameter count is computable (acts as a proxy for W&B logging)."""
        n_params = sum(p.numel() for p in self.policy.parameters())
        # Sanity check: architecture has a predictable size
        self.assertGreater(n_params, 0)


class TestInferenceSpeed(unittest.TestCase):
    """Forward pass must be < 1 ms on CPU for batch size 64."""

    def setUp(self) -> None:
        self.model, self.env = _make_ppo_model()
        self.policy = self.model.policy
        self.policy.eval()

    def tearDown(self) -> None:
        self.env.close()

    def test_forward_pass_speed(self) -> None:
        """Average forward pass for batch size 64 must be under 1 ms on CPU."""
        obs = torch.tensor(
            np.random.rand(64, 12).astype("float32")
        )
        # Warm-up
        for _ in range(5):
            with torch.no_grad():
                self.policy(obs)

        n_trials = 200
        t0 = time.perf_counter()
        for _ in range(n_trials):
            with torch.no_grad():
                self.policy(obs)
        elapsed_ms = (time.perf_counter() - t0) / n_trials * 1000
        self.assertLess(
            elapsed_ms,
            1.0,
            msg=f"Forward pass took {elapsed_ms:.3f} ms (spec: < 1 ms)",
        )


if __name__ == "__main__":
    unittest.main()
