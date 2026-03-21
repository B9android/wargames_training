# tests/test_coa_generator.py
"""Tests for analysis/coa_generator.py and api/coa_endpoint.py (Epic E5.2).

Coverage
--------
* COAScore / CourseOfAction data classes
* _BiasedPolicy — action bias application and clipping
* _run_single_rollout — basic shape / type checks
* _aggregate_rollouts — statistics correctness
* COAGenerator — construction, generate(), ranking, n_coas variants
* generate_coas() — convenience wrapper
* Flask /health and /coas endpoints — happy path and error paths
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.coa_generator import (
    COAGenerator,
    COAScore,
    CourseOfAction,
    STRATEGY_LABELS,
    _BiasedPolicy,
    _aggregate_rollouts,
    _run_single_rollout,
    generate_coas,
)
from envs.battalion_env import BattalionEnv

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_N_ROLLOUTS = 3   # tiny to keep tests fast
_N_COAS     = 3   # fewer than the 7 built-in strategies


def _make_env(**kwargs) -> BattalionEnv:
    return BattalionEnv(randomize_terrain=False, max_steps=30, **kwargs)


class _ConstantPolicy:
    """Deterministic policy that always returns the same fixed action."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Any]:
        return self._action.copy(), None


# ---------------------------------------------------------------------------
# 1. Data classes
# ---------------------------------------------------------------------------


class TestCOAScore(unittest.TestCase):
    def _make(self, **overrides) -> COAScore:
        defaults = dict(
            win_rate=0.5, draw_rate=0.2, loss_rate=0.3,
            blue_casualties=0.1, red_casualties=0.4,
            terrain_control=0.6, composite=0.48, n_rollouts=5,
        )
        defaults.update(overrides)
        return COAScore(**defaults)

    def test_as_dict_keys(self) -> None:
        score = self._make()
        d = score.as_dict()
        for key in (
            "win_rate", "draw_rate", "loss_rate",
            "blue_casualties", "red_casualties",
            "terrain_control", "composite", "n_rollouts",
        ):
            self.assertIn(key, d)

    def test_as_dict_values(self) -> None:
        score = self._make(win_rate=0.7)
        self.assertAlmostEqual(score.as_dict()["win_rate"], 0.7)


class TestCourseOfAction(unittest.TestCase):
    def _make(self) -> CourseOfAction:
        score = COAScore(
            win_rate=0.6, draw_rate=0.1, loss_rate=0.3,
            blue_casualties=0.05, red_casualties=0.3,
            terrain_control=0.55, composite=0.5, n_rollouts=4,
        )
        return CourseOfAction(label="aggressive", rank=1, score=score,
                              action_summary={}, seed=42)

    def test_as_dict_structure(self) -> None:
        coa = self._make()
        d = coa.as_dict()
        self.assertEqual(d["label"], "aggressive")
        self.assertEqual(d["rank"], 1)
        self.assertIn("score", d)
        self.assertIn("win_rate", d["score"])

    def test_as_dict_json_serialisable(self) -> None:
        import json
        coa = self._make()
        json_str = json.dumps(coa.as_dict())
        self.assertIn("aggressive", json_str)


# ---------------------------------------------------------------------------
# 2. _BiasedPolicy
# ---------------------------------------------------------------------------


class TestBiasedPolicy(unittest.TestCase):
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(0)

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            _BiasedPolicy(None, "does_not_exist", self._rng())

    def test_random_base_policy_returns_valid_action(self) -> None:
        p = _BiasedPolicy(None, "aggressive", self._rng())
        obs = np.zeros(12, dtype=np.float32)
        action, state = p.predict(obs)
        self.assertEqual(action.shape, (3,))
        self.assertGreaterEqual(float(action[0]), -1.0)
        self.assertLessEqual(float(action[0]),  1.0)
        self.assertGreaterEqual(float(action[2]),  0.0)
        self.assertLessEqual(float(action[2]),  1.0)

    def test_aggressive_bias_raises_fire(self) -> None:
        """Aggressive strategy should produce high fire values."""
        base = _ConstantPolicy(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        p = _BiasedPolicy(base, "aggressive", self._rng())
        obs = np.zeros(12, dtype=np.float32)
        action, _ = p.predict(obs)
        self.assertGreater(float(action[2]), 0.5)   # fire_bias = 0.8

    def test_defensive_bias_moves_backward(self) -> None:
        base = _ConstantPolicy(np.array([0.0, 0.0, 0.5], dtype=np.float32))
        p = _BiasedPolicy(base, "defensive", self._rng())
        obs = np.zeros(12, dtype=np.float32)
        action, _ = p.predict(obs)
        self.assertLess(float(action[0]), 0.0)   # move_bias = -0.5

    def test_clipping_prevents_out_of_range(self) -> None:
        """Even with a policy that saturates actions, biased output must stay in bounds."""
        base = _ConstantPolicy(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        p = _BiasedPolicy(base, "rapid_assault", self._rng())
        obs = np.zeros(12, dtype=np.float32)
        action, _ = p.predict(obs)
        self.assertLessEqual(float(action[0]), 1.0)
        self.assertLessEqual(float(action[2]), 1.0)
        self.assertGreaterEqual(float(action[2]), 0.0)

    def test_all_strategies_produce_valid_actions(self) -> None:
        obs = np.zeros(12, dtype=np.float32)
        for strat in STRATEGY_LABELS:
            p = _BiasedPolicy(None, strat, np.random.default_rng(0))
            action, _ = p.predict(obs)
            self.assertEqual(action.shape, (3,), msg=f"strategy={strat}")
            self.assertTrue(np.all(np.isfinite(action)), msg=f"strategy={strat}")


# ---------------------------------------------------------------------------
# 3. _run_single_rollout
# ---------------------------------------------------------------------------


class TestRunSingleRollout(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        env = _make_env()
        p = _BiasedPolicy(None, "aggressive", np.random.default_rng(1))
        result = _run_single_rollout(env, p, seed=0)
        env.close()
        for key in ("outcome", "blue_strength", "red_strength", "steps",
                    "actions", "terrain_control_frac"):
            self.assertIn(key, result, msg=f"Missing key: {key}")

    def test_outcome_in_valid_set(self) -> None:
        env = _make_env()
        p = _BiasedPolicy(None, "aggressive", np.random.default_rng(2))
        result = _run_single_rollout(env, p, seed=1)
        env.close()
        self.assertIn(result["outcome"], (-1, 0, 1))

    def test_strengths_in_unit_interval(self) -> None:
        env = _make_env()
        p = _BiasedPolicy(None, "defensive", np.random.default_rng(3))
        result = _run_single_rollout(env, p, seed=2)
        env.close()
        self.assertGreaterEqual(result["blue_strength"], 0.0)
        self.assertLessEqual(result["blue_strength"], 1.0)
        self.assertGreaterEqual(result["red_strength"], 0.0)
        self.assertLessEqual(result["red_strength"], 1.0)

    def test_actions_shape(self) -> None:
        env = _make_env()
        p = _BiasedPolicy(None, "standoff", np.random.default_rng(4))
        result = _run_single_rollout(env, p, seed=3)
        env.close()
        self.assertEqual(result["actions"].ndim, 2)
        self.assertEqual(result["actions"].shape[1], 3)
        self.assertEqual(result["actions"].shape[0], result["steps"])

    def test_terrain_control_frac_in_unit_interval(self) -> None:
        env = _make_env()
        p = _BiasedPolicy(None, "flanking_left", np.random.default_rng(5))
        result = _run_single_rollout(env, p, seed=4)
        env.close()
        self.assertGreaterEqual(result["terrain_control_frac"], 0.0)
        self.assertLessEqual(result["terrain_control_frac"], 1.0)


# ---------------------------------------------------------------------------
# 4. _aggregate_rollouts
# ---------------------------------------------------------------------------


class TestAggregateRollouts(unittest.TestCase):
    def _make_results(self, outcomes: list, strengths: list | None = None) -> list:
        """Build synthetic rollout dicts."""
        if strengths is None:
            strengths = [(0.8, 0.5)] * len(outcomes)
        results = []
        for outcome, (blue_s, red_s) in zip(outcomes, strengths):
            n_steps = 10
            results.append({
                "outcome": outcome,
                "blue_strength": blue_s,
                "red_strength": red_s,
                "steps": n_steps,
                "actions": np.random.default_rng(0).random((n_steps, 3)).astype(np.float32),
                "terrain_control_frac": 0.5,
            })
        return results

    def test_win_rate_correct(self) -> None:
        results = self._make_results([1, 1, 0, -1])
        score, _ = _aggregate_rollouts(results)
        self.assertAlmostEqual(score.win_rate, 0.5)
        self.assertEqual(score.n_rollouts, 4)

    def test_loss_rate_correct(self) -> None:
        results = self._make_results([1, 1, 0, -1])
        score, _ = _aggregate_rollouts(results)
        self.assertAlmostEqual(score.loss_rate, 0.25)

    def test_casualties_correct(self) -> None:
        results = self._make_results(
            [1, -1], strengths=[(0.9, 0.3), (0.7, 0.5)]
        )
        score, _ = _aggregate_rollouts(results)
        expected_blue = (1.0 - 0.9 + 1.0 - 0.7) / 2   # 0.2
        self.assertAlmostEqual(score.blue_casualties, expected_blue, places=4)

    def test_composite_in_unit_interval(self) -> None:
        results = self._make_results([1, 0, -1, 1, 0])
        score, _ = _aggregate_rollouts(results)
        self.assertGreaterEqual(score.composite, 0.0)
        self.assertLessEqual(score.composite, 1.0)

    def test_action_summary_keys_present(self) -> None:
        results = self._make_results([1])
        _, summary = _aggregate_rollouts(results)
        for q in range(1, 5):
            for dim in ("move", "rotate", "fire"):
                self.assertIn(f"{dim}_q{q}", summary)


# ---------------------------------------------------------------------------
# 5. COAGenerator
# ---------------------------------------------------------------------------


class TestCOAGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _make_env()

    def tearDown(self) -> None:
        self.env.close()

    def test_construction_defaults(self) -> None:
        gen = COAGenerator(self.env)
        self.assertEqual(gen.n_rollouts, 20)
        self.assertEqual(gen.n_coas, 5)

    def test_n_rollouts_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            COAGenerator(self.env, n_rollouts=0)

    def test_n_coas_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            COAGenerator(self.env, n_coas=0)

    def test_n_coas_too_large_raises(self) -> None:
        with self.assertRaises(ValueError):
            COAGenerator(self.env, n_coas=100)

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            COAGenerator(self.env, strategies=["does_not_exist"])

    def test_duplicate_strategies_raises(self) -> None:
        """Duplicate strategies that reduce distinct count below n_coas must raise ValueError."""
        with self.assertRaises(ValueError):
            COAGenerator(self.env, n_coas=2, strategies=["aggressive", "aggressive"])

    def test_strategies_deduplication_preserves_order(self) -> None:
        """Duplicate entries in strategies are removed but order is preserved."""
        gen = COAGenerator(
            self.env, n_rollouts=_N_ROLLOUTS, n_coas=2,
            strategies=["standoff", "aggressive", "standoff"],
        )
        self.assertEqual(gen._strategies, ["standoff", "aggressive"])

    def test_generate_returns_n_coas(self) -> None:
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=_N_COAS, seed=0)
        coas = gen.generate()
        self.assertEqual(len(coas), _N_COAS)

    def test_generate_all_are_course_of_action(self) -> None:
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=_N_COAS, seed=1)
        for coa in gen.generate():
            self.assertIsInstance(coa, CourseOfAction)

    def test_generate_ranks_are_unique_and_ordered(self) -> None:
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=_N_COAS, seed=2)
        coas = gen.generate()
        ranks = [c.rank for c in coas]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(ranks[0], 1)

    def test_generate_sorted_by_composite(self) -> None:
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=_N_COAS, seed=3)
        coas = gen.generate()
        composites = [c.score.composite for c in coas]
        self.assertEqual(composites, sorted(composites, reverse=True))

    def test_generate_labels_distinct(self) -> None:
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=_N_COAS, seed=4)
        coas = gen.generate()
        labels = [c.label for c in coas]
        self.assertEqual(len(labels), len(set(labels)), "All COA labels should be distinct")

    def test_generate_with_constant_policy(self) -> None:
        policy = _ConstantPolicy(np.array([0.5, 0.0, 0.8], dtype=np.float32))
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=2, seed=5)
        coas = gen.generate(policy=policy)
        self.assertEqual(len(coas), 2)

    def test_generate_reproducible_with_seed(self) -> None:
        gen1 = COAGenerator(_make_env(), n_rollouts=_N_ROLLOUTS, n_coas=2, seed=99)
        gen2 = COAGenerator(_make_env(), n_rollouts=_N_ROLLOUTS, n_coas=2, seed=99)
        coas1 = gen1.generate()
        coas2 = gen2.generate()
        gen1.env.close()
        gen2.env.close()
        for c1, c2 in zip(coas1, coas2):
            self.assertEqual(c1.label, c2.label)
            self.assertAlmostEqual(c1.score.win_rate, c2.score.win_rate)

    def test_generate_scores_in_range(self) -> None:
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=_N_COAS, seed=6)
        for coa in gen.generate():
            s = coa.score
            self.assertGreaterEqual(s.win_rate, 0.0)
            self.assertLessEqual(s.win_rate, 1.0)
            self.assertGreaterEqual(s.composite, 0.0)
            self.assertLessEqual(s.composite, 1.0)
            self.assertGreaterEqual(s.terrain_control, 0.0)
            self.assertLessEqual(s.terrain_control, 1.0)

    def test_generate_five_coas(self) -> None:
        """Acceptance criterion: generator produces ≥ 5 distinct COAs."""
        gen = COAGenerator(self.env, n_rollouts=_N_ROLLOUTS, n_coas=5, seed=7)
        coas = gen.generate()
        self.assertGreaterEqual(len(coas), 5)
        labels = {c.label for c in coas}
        self.assertEqual(len(labels), 5, "All 5 COAs must have distinct labels")


# ---------------------------------------------------------------------------
# 6. generate_coas() convenience wrapper
# ---------------------------------------------------------------------------


class TestGenerateCoas(unittest.TestCase):
    def test_generates_expected_count(self) -> None:
        coas = generate_coas(n_rollouts=_N_ROLLOUTS, n_coas=3, seed=0)
        self.assertEqual(len(coas), 3)

    def test_with_pre_built_env(self) -> None:
        env = _make_env()
        coas = generate_coas(env=env, n_rollouts=_N_ROLLOUTS, n_coas=2, seed=1)
        env.close()
        self.assertEqual(len(coas), 2)

    def test_env_kwargs_forwarded(self) -> None:
        coas = generate_coas(
            n_rollouts=_N_ROLLOUTS, n_coas=2, seed=2,
            env_kwargs={"curriculum_level": 1, "randomize_terrain": False, "max_steps": 20},
        )
        self.assertEqual(len(coas), 2)

    def test_custom_strategies(self) -> None:
        strats = ["standoff", "rapid_assault"]
        coas = generate_coas(n_rollouts=_N_ROLLOUTS, n_coas=2, seed=3, strategies=strats)
        labels = [c.label for c in coas]
        # Both requested labels should appear in the results
        for s in strats:
            self.assertIn(s, labels)


# ---------------------------------------------------------------------------
# 7. Flask API endpoint
# ---------------------------------------------------------------------------


class TestCOAEndpoint(unittest.TestCase):
    """Smoke tests for the Flask REST API."""

    @classmethod
    def setUpClass(cls) -> None:
        from api.coa_endpoint import app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_returns_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_coas_default_request(self) -> None:
        resp = self.client.post(
            "/coas",
            json={"n_rollouts": _N_ROLLOUTS, "n_coas": 3, "seed": 0,
                  "env_kwargs": {"randomize_terrain": False, "max_steps": 20}},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("coas", data)
        self.assertEqual(data["n_coas"], 3)
        self.assertEqual(len(data["coas"]), 3)

    def test_coas_response_structure(self) -> None:
        resp = self.client.post(
            "/coas",
            json={"n_rollouts": _N_ROLLOUTS, "n_coas": 2, "seed": 1,
                  "env_kwargs": {"randomize_terrain": False, "max_steps": 15}},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        coa = data["coas"][0]
        for field in ("label", "rank", "score", "action_summary", "seed"):
            self.assertIn(field, coa, msg=f"Missing field '{field}' in COA")
        score = coa["score"]
        for metric in ("win_rate", "draw_rate", "loss_rate",
                       "blue_casualties", "red_casualties",
                       "terrain_control", "composite", "n_rollouts"):
            self.assertIn(metric, score, msg=f"Missing score field '{metric}'")

    def test_coas_ranked_ascending(self) -> None:
        resp = self.client.post(
            "/coas",
            json={"n_rollouts": _N_ROLLOUTS, "n_coas": 3, "seed": 2,
                  "env_kwargs": {"randomize_terrain": False, "max_steps": 15}},
        )
        data = resp.get_json()
        ranks = [c["rank"] for c in data["coas"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_coas_custom_strategies(self) -> None:
        resp = self.client.post(
            "/coas",
            json={"n_rollouts": _N_ROLLOUTS, "n_coas": 2, "seed": 3,
                  "strategies": ["aggressive", "defensive"],
                  "env_kwargs": {"randomize_terrain": False, "max_steps": 15}},
        )
        self.assertEqual(resp.status_code, 200)
        labels = {c["label"] for c in resp.get_json()["coas"]}
        self.assertEqual(labels, {"aggressive", "defensive"})

    def test_coas_empty_body_uses_defaults(self) -> None:
        """Empty JSON body with small overrides should return 200."""
        resp = self.client.post(
            "/coas",
            json={"n_rollouts": _N_ROLLOUTS, "n_coas": 2,
                  "env_kwargs": {"randomize_terrain": False, "max_steps": 15}},
        )
        self.assertEqual(resp.status_code, 200)

    def test_coas_non_object_body_returns_400(self) -> None:
        """A JSON array (not object) as body must return 400."""
        resp = self.client.post("/coas", data=b"[1,2,3]",
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_coas_env_kwargs_non_dict_returns_400(self) -> None:
        """env_kwargs as a non-dict (e.g. list) must return 400."""
        resp = self.client.post("/coas", json={"env_kwargs": ["bad"]})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_coas_duplicate_strategies_deduplicated(self) -> None:
        """Duplicate strategies resulting in fewer distinct labels than n_coas must return 400."""
        resp = self.client.post(
            "/coas",
            json={"n_coas": 2, "strategies": ["aggressive", "aggressive"],
                  "n_rollouts": _N_ROLLOUTS,
                  "env_kwargs": {"randomize_terrain": False, "max_steps": 15}},
        )
        self.assertEqual(resp.status_code, 400)

    def test_coas_invalid_n_rollouts_returns_400(self) -> None:
        resp = self.client.post("/coas", json={"n_rollouts": 0})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_coas_invalid_strategy_returns_400(self) -> None:
        resp = self.client.post(
            "/coas",
            json={"n_coas": 1, "strategies": ["does_not_exist"]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_coas_n_coas_too_large_returns_400(self) -> None:
        resp = self.client.post("/coas", json={"n_coas": 999})
        self.assertEqual(resp.status_code, 400)

    def test_coas_unknown_env_kwargs_returns_400(self) -> None:
        resp = self.client.post("/coas", json={"env_kwargs": {"unknown_param": 1}})
        self.assertEqual(resp.status_code, 400)

    def test_health_method_not_allowed(self) -> None:
        resp = self.client.post("/health")
        self.assertEqual(resp.status_code, 405)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
