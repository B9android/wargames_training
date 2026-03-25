# SPDX-License-Identifier: MIT
# tests/test_saliency.py
"""Tests for analysis/saliency.py (Epic E5.3).

Coverage
--------
* OBSERVATION_FEATURES — correct length and strings
* _extract_mlp_network — SB3 PPO, ActorCriticPolicy, plain nn.Module
* compute_gradient_saliency — shape, non-negativity, reduce modes
* compute_integrated_gradients — shape, completeness approximation
* compute_shap_importance — shape, fallback permutation path
* _permutation_importance — shape, changes when feature matters
* SaliencyAnalyzer — construction, gradient_saliency, integrated_gradients,
  shap_importance, top_features, summary, plot_saliency, plot_importance
* plot_saliency_map / plot_feature_importance — return Figure without error
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.saliency import (
    OBSERVATION_FEATURES,
    SaliencyAnalyzer,
    _extract_mlp_network,
    _permutation_importance,
    compute_gradient_saliency,
    compute_integrated_gradients,
    compute_shap_importance,
    plot_feature_importance,
    plot_saliency_map,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 12
ACTION_DIM = 3


def _make_simple_net(obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM) -> nn.Module:
    """Tiny 2-layer MLP — fast to test and fully differentiable."""
    return nn.Sequential(
        nn.Linear(obs_dim, 16),
        nn.Tanh(),
        nn.Linear(16, action_dim),
    )


def _make_ppo_model():
    """Return a minimal untrained SB3 PPO model using BattalionMlpPolicy."""
    try:
        from envs.battalion_env import BattalionEnv
        from models.mlp_policy import BattalionMlpPolicy
        from stable_baselines3 import PPO

        env = BattalionEnv(randomize_terrain=False, max_steps=10)
        return PPO(BattalionMlpPolicy, env, verbose=0), env
    except ImportError:
        return None, None


def _random_obs(n: int = 1, obs_dim: int = OBS_DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=(n, obs_dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# OBSERVATION_FEATURES
# ---------------------------------------------------------------------------


class TestObservationFeatures(unittest.TestCase):

    def test_length(self) -> None:
        self.assertEqual(len(OBSERVATION_FEATURES), OBS_DIM)

    def test_all_strings(self) -> None:
        for name in OBSERVATION_FEATURES:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_no_duplicates(self) -> None:
        self.assertEqual(len(OBSERVATION_FEATURES), len(set(OBSERVATION_FEATURES)))

    def test_known_entries(self) -> None:
        self.assertIn("dist_norm", OBSERVATION_FEATURES)
        self.assertIn("blue_strength", OBSERVATION_FEATURES)
        self.assertIn("red_strength", OBSERVATION_FEATURES)


# ---------------------------------------------------------------------------
# _extract_mlp_network
# ---------------------------------------------------------------------------


class TestExtractMlpNetwork(unittest.TestCase):

    def test_plain_module_returned_as_is(self) -> None:
        net = _make_simple_net()
        extracted = _extract_mlp_network(net)
        self.assertIs(extracted, net)

    def test_unsupported_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            _extract_mlp_network("not a policy")

    def test_ppo_model_extraction(self) -> None:
        model, _env = _make_ppo_model()
        if model is None:
            self.skipTest("SB3 / BattalionEnv not available")
        net = _extract_mlp_network(model)
        self.assertIsInstance(net, nn.Module)
        # Forward pass should not raise
        obs = torch.zeros(1, OBS_DIM)
        with torch.no_grad():
            out = net(obs)
        self.assertEqual(out.shape[-1], ACTION_DIM)


# ---------------------------------------------------------------------------
# compute_gradient_saliency
# ---------------------------------------------------------------------------


class TestComputeGradientSaliency(unittest.TestCase):

    def setUp(self) -> None:
        self.net = _make_simple_net()
        self.obs_1d = _random_obs(1)[0]         # (obs_dim,)
        self.obs_batch = _random_obs(5)         # (5, obs_dim)

    def test_shape_single_obs(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_1d)
        self.assertEqual(sal.shape, (OBS_DIM,))

    def test_shape_batch_obs(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_batch)
        self.assertEqual(sal.shape, (OBS_DIM,))

    def test_non_negative(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_1d)
        self.assertTrue((sal >= 0).all(), "Saliency scores must be non-negative")

    def test_reduce_mean_abs(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_batch, reduce="mean_abs")
        self.assertEqual(sal.shape, (OBS_DIM,))

    def test_reduce_max_abs(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_batch, reduce="max_abs")
        self.assertEqual(sal.shape, (OBS_DIM,))
        self.assertTrue((sal >= 0).all())

    def test_reduce_sum_abs(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_batch, reduce="sum_abs")
        self.assertEqual(sal.shape, (OBS_DIM,))

    def test_invalid_reduce_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_gradient_saliency(self.net, self.obs_1d, reduce="invalid")

    def test_reduce_none_returns_batch(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_batch, reduce="none")
        self.assertEqual(sal.shape, (5, OBS_DIM))

    def test_returns_numpy(self) -> None:
        sal = compute_gradient_saliency(self.net, self.obs_1d)
        self.assertIsInstance(sal, np.ndarray)

    def test_accepts_torch_tensor(self) -> None:
        t = torch.from_numpy(self.obs_1d)
        sal = compute_gradient_saliency(self.net, t)
        self.assertEqual(sal.shape, (OBS_DIM,))


# ---------------------------------------------------------------------------
# compute_integrated_gradients
# ---------------------------------------------------------------------------


class TestComputeIntegratedGradients(unittest.TestCase):

    def setUp(self) -> None:
        self.net = _make_simple_net()
        self.obs = _random_obs(1)[0]   # (obs_dim,)

    def test_shape(self) -> None:
        ig = compute_integrated_gradients(self.net, self.obs)
        self.assertEqual(ig.shape, (OBS_DIM,))

    def test_returns_numpy(self) -> None:
        ig = compute_integrated_gradients(self.net, self.obs)
        self.assertIsInstance(ig, np.ndarray)

    def test_custom_baseline(self) -> None:
        baseline = np.ones(OBS_DIM, dtype=np.float32) * 0.5
        ig = compute_integrated_gradients(self.net, self.obs, baseline=baseline)
        self.assertEqual(ig.shape, (OBS_DIM,))

    def test_few_steps_still_works(self) -> None:
        ig = compute_integrated_gradients(self.net, self.obs, n_steps=5)
        self.assertEqual(ig.shape, (OBS_DIM,))

    def test_zero_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_integrated_gradients(self.net, self.obs, n_steps=0)

    def test_batch_obs_mean_reduction(self) -> None:
        obs_batch = _random_obs(4)
        ig = compute_integrated_gradients(self.net, obs_batch)
        self.assertEqual(ig.shape, (OBS_DIM,))

    def test_completeness_approximation(self) -> None:
        """IG attributions sum ≈ f(obs) - f(baseline) (within tolerance)."""
        obs = torch.from_numpy(self.obs).unsqueeze(0)
        baseline = torch.zeros_like(obs)
        ig = compute_integrated_gradients(self.net, self.obs, n_steps=200)
        with torch.no_grad():
            f_obs = self.net(obs).sum().item()
            f_base = self.net(baseline).sum().item()
        expected_diff = f_obs - f_base
        # Allow 10% relative tolerance due to finite-step approximation
        tol = abs(expected_diff) * 0.10 + 1e-3
        self.assertAlmostEqual(ig.sum(), expected_diff, delta=tol)


# ---------------------------------------------------------------------------
# compute_shap_importance / _permutation_importance
# ---------------------------------------------------------------------------


class TestComputeShapImportance(unittest.TestCase):

    def setUp(self) -> None:
        self.net = _make_simple_net()
        self.obs = _random_obs(5)  # (5, obs_dim)

    def test_shape(self) -> None:
        scores = compute_shap_importance(self.net, self.obs)
        self.assertEqual(scores.shape, (OBS_DIM,))

    def test_non_negative(self) -> None:
        scores = compute_shap_importance(self.net, self.obs)
        self.assertTrue((scores >= 0).all())

    def test_custom_background(self) -> None:
        bg = np.ones((1, OBS_DIM), dtype=np.float32) * 0.5
        scores = compute_shap_importance(self.net, self.obs, background=bg)
        self.assertEqual(scores.shape, (OBS_DIM,))

    def test_returns_numpy(self) -> None:
        scores = compute_shap_importance(self.net, self.obs)
        self.assertIsInstance(scores, np.ndarray)


class TestPermutationImportance(unittest.TestCase):

    def setUp(self) -> None:
        self.net = _make_simple_net()
        self.obs = _random_obs(3)
        self.bg = np.zeros((1, OBS_DIM), dtype=np.float32)

    def test_shape(self) -> None:
        imp = _permutation_importance(self.net, self.obs, self.bg)
        self.assertEqual(imp.shape, (OBS_DIM,))

    def test_non_negative(self) -> None:
        imp = _permutation_importance(self.net, self.obs, self.bg)
        self.assertTrue((imp >= 0).all())

    def test_n_samples_accepted(self) -> None:
        """n_samples parameter is accepted and produces correct output shape."""
        imp = _permutation_importance(self.net, self.obs, self.bg, n_samples=5)
        self.assertEqual(imp.shape, (OBS_DIM,))

    def test_sensitive_feature_has_high_importance(self) -> None:
        """A network that only uses feature 0 should rank feature 0 highest."""
        # Build a 1-hot network: maps only feature 0 to output
        weights = torch.zeros(ACTION_DIM, OBS_DIM)
        weights[:, 0] = 1.0
        net = nn.Linear(OBS_DIM, ACTION_DIM, bias=False)
        with torch.no_grad():
            net.weight.copy_(weights)

        obs = _random_obs(5)
        imp = _permutation_importance(net, obs, self.bg)
        top_idx = int(np.argmax(imp))
        self.assertEqual(top_idx, 0, "Feature 0 should be ranked most important")


# ---------------------------------------------------------------------------
# plot_saliency_map / plot_feature_importance
# ---------------------------------------------------------------------------


class TestPlotFunctions(unittest.TestCase):

    def _dummy_scores(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(0.0, 1.0, size=OBS_DIM).astype(np.float32)

    def test_plot_saliency_returns_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = self._dummy_scores()
        fig = plot_saliency_map(scores)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_saliency_custom_title(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = self._dummy_scores()
        fig = plot_saliency_map(scores, title="My Title")
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "My Title")
        plt.close("all")

    def test_plot_saliency_no_normalise(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = self._dummy_scores() * 5.0
        fig = plot_saliency_map(scores, normalise=False)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_feature_importance_returns_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = self._dummy_scores()
        fig = plot_feature_importance(scores)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_feature_importance_top_k(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = self._dummy_scores()
        fig = plot_feature_importance(scores, top_k=5)
        ax = fig.axes[0]
        self.assertEqual(len(ax.get_yticklabels()), 5)
        plt.close("all")


# ---------------------------------------------------------------------------
# SaliencyAnalyzer facade
# ---------------------------------------------------------------------------


class TestSaliencyAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        self.net = _make_simple_net()
        self.analyzer = SaliencyAnalyzer(self.net)
        self.obs = _random_obs(3)

    def test_construction_default_feature_names(self) -> None:
        self.assertEqual(self.analyzer.feature_names, OBSERVATION_FEATURES)

    def test_construction_custom_feature_names(self) -> None:
        names = tuple(f"f{i}" for i in range(OBS_DIM))
        analyzer = SaliencyAnalyzer(self.net, feature_names=names)
        self.assertEqual(analyzer.feature_names, names)

    def test_gradient_saliency_shape(self) -> None:
        sal = self.analyzer.gradient_saliency(self.obs)
        self.assertEqual(sal.shape, (OBS_DIM,))

    def test_integrated_gradients_shape(self) -> None:
        ig = self.analyzer.integrated_gradients(self.obs, n_steps=10)
        self.assertEqual(ig.shape, (OBS_DIM,))

    def test_shap_importance_shape(self) -> None:
        shap = self.analyzer.shap_importance(self.obs)
        self.assertEqual(shap.shape, (OBS_DIM,))

    def test_top_features_length(self) -> None:
        sal = self.analyzer.gradient_saliency(self.obs)
        top = self.analyzer.top_features(sal, k=3)
        self.assertEqual(len(top), 3)

    def test_top_features_sorted_descending(self) -> None:
        sal = self.analyzer.gradient_saliency(self.obs)
        top = self.analyzer.top_features(sal, k=5)
        scores = [s for _, s in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_features_returns_tuples(self) -> None:
        sal = self.analyzer.gradient_saliency(self.obs)
        top = self.analyzer.top_features(sal, k=2)
        for name, score in top:
            self.assertIsInstance(name, str)
            self.assertIsInstance(score, float)

    def test_summary_keys(self) -> None:
        result = self.analyzer.summary(self.obs, n_steps=5, n_samples=10)
        self.assertIn("gradient_saliency", result)
        self.assertIn("integrated_gradients", result)
        self.assertIn("shap_importance", result)

    def test_summary_shapes(self) -> None:
        result = self.analyzer.summary(self.obs, n_steps=5, n_samples=10)
        for key, val in result.items():
            self.assertEqual(val.shape, (OBS_DIM,), msg=f"Shape mismatch for {key}")

    def test_plot_saliency_with_scores(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sal = self.analyzer.gradient_saliency(self.obs)
        fig = self.analyzer.plot_saliency(sal)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_saliency_with_obs(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = self.analyzer.plot_saliency(obs=self.obs)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_saliency_no_args_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.analyzer.plot_saliency()

    def test_plot_importance_with_scores(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imp = self.analyzer.shap_importance(self.obs)
        fig = self.analyzer.plot_importance(imp)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_importance_with_obs(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = self.analyzer.plot_importance(obs=self.obs)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_importance_no_args_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.analyzer.plot_importance()

    def test_with_ppo_model(self) -> None:
        """SaliencyAnalyzer works with a real SB3 PPO model."""
        model, env = _make_ppo_model()
        if model is None:
            self.skipTest("SB3 / BattalionEnv not available")
        obs, _ = env.reset(seed=7)
        analyzer = SaliencyAnalyzer(model)
        sal = analyzer.gradient_saliency(obs)
        self.assertEqual(sal.shape, (OBS_DIM,))
        top = analyzer.top_features(sal, k=3)
        self.assertEqual(len(top), 3)


if __name__ == "__main__":
    unittest.main()
