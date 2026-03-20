"""Tests for training/adaptive_temporal.py — AdaptiveTemporalScheduler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adaptive_temporal import AdaptiveTemporalScheduler, SWEEP_RATIOS
from envs.options import Option


# ---------------------------------------------------------------------------
# 1. Construction and validation
# ---------------------------------------------------------------------------


class TestAdaptiveTemporalSchedulerConstruction(unittest.TestCase):
    """Verify AdaptiveTemporalScheduler validates constructor inputs."""

    def test_default_construction(self) -> None:
        sched = AdaptiveTemporalScheduler()
        self.assertEqual(sched.base_ratio, 10)
        self.assertEqual(sched.min_ratio, 5)
        self.assertEqual(sched.max_ratio, 20)
        self.assertEqual(sched.adaptation, "fixed")

    def test_current_ratio_initial_value_fixed(self) -> None:
        """current_ratio before get_ratio call equals base_ratio for fixed."""
        sched = AdaptiveTemporalScheduler(base_ratio=15, adaptation="fixed")
        self.assertEqual(sched.current_ratio, 15)

    def test_current_ratio_initial_value_linear_decrease(self) -> None:
        """current_ratio before get_ratio call equals max_ratio for linear_decrease."""
        sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_decrease"
        )
        self.assertEqual(sched.current_ratio, 20)

    def test_current_ratio_initial_value_linear_increase(self) -> None:
        """current_ratio before get_ratio call equals min_ratio for linear_increase."""
        sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_increase"
        )
        self.assertEqual(sched.current_ratio, 5)

    def test_custom_construction(self) -> None:
        sched = AdaptiveTemporalScheduler(
            base_ratio=15, min_ratio=3, max_ratio=30, adaptation="linear_decrease"
        )
        self.assertEqual(sched.base_ratio, 15)
        self.assertEqual(sched.min_ratio, 3)
        self.assertEqual(sched.max_ratio, 30)
        self.assertEqual(sched.adaptation, "linear_decrease")

    def test_invalid_base_ratio(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTemporalScheduler(base_ratio=0)

    def test_invalid_min_ratio(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTemporalScheduler(min_ratio=0)

    def test_max_less_than_min(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTemporalScheduler(min_ratio=20, max_ratio=10)

    def test_invalid_adaptation(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTemporalScheduler(adaptation="cosine")  # type: ignore[arg-type]

    def test_repr(self) -> None:
        sched = AdaptiveTemporalScheduler(base_ratio=10)
        r = repr(sched)
        self.assertIn("AdaptiveTemporalScheduler", r)
        self.assertIn("base_ratio=10", r)


# ---------------------------------------------------------------------------
# 2. Fixed adaptation strategy
# ---------------------------------------------------------------------------


class TestFixedAdaptation(unittest.TestCase):
    """Fixed adaptation returns base_ratio regardless of episode progress."""

    def setUp(self) -> None:
        self.sched = AdaptiveTemporalScheduler(base_ratio=10, adaptation="fixed")

    def test_start_of_episode(self) -> None:
        self.assertEqual(self.sched.get_ratio(0.0), 10)

    def test_mid_episode(self) -> None:
        self.assertEqual(self.sched.get_ratio(0.5), 10)

    def test_end_of_episode(self) -> None:
        self.assertEqual(self.sched.get_ratio(1.0), 10)

    def test_current_ratio_property(self) -> None:
        self.sched.get_ratio(0.3)
        self.assertEqual(self.sched.current_ratio, 10)

    def test_sweep_ratios_all_fixed(self) -> None:
        for ratio in SWEEP_RATIOS:
            sched = AdaptiveTemporalScheduler(base_ratio=ratio, adaptation="fixed")
            self.assertEqual(sched.get_ratio(0.0), ratio)
            self.assertEqual(sched.get_ratio(1.0), ratio)


# ---------------------------------------------------------------------------
# 3. Linear decrease adaptation
# ---------------------------------------------------------------------------


class TestLinearDecreaseAdaptation(unittest.TestCase):
    """Linear decrease goes from max_ratio to min_ratio over the episode."""

    def setUp(self) -> None:
        self.sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_decrease"
        )

    def test_episode_start_is_max(self) -> None:
        self.assertEqual(self.sched.get_ratio(0.0), 20)

    def test_episode_end_is_min(self) -> None:
        self.assertEqual(self.sched.get_ratio(1.0), 5)

    def test_midpoint_is_between(self) -> None:
        ratio = self.sched.get_ratio(0.5)
        self.assertGreater(ratio, 5)
        self.assertLess(ratio, 20)

    def test_monotone_decreasing(self) -> None:
        progresses = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ratios = [self.sched.get_ratio(p) for p in progresses]
        for i in range(len(ratios) - 1):
            self.assertGreaterEqual(ratios[i], ratios[i + 1])

    def test_out_of_range_progress_clipped(self) -> None:
        self.assertEqual(self.sched.get_ratio(-0.5), 20)
        self.assertEqual(self.sched.get_ratio(1.5), 5)


# ---------------------------------------------------------------------------
# 4. Linear increase adaptation
# ---------------------------------------------------------------------------


class TestLinearIncreaseAdaptation(unittest.TestCase):
    """Linear increase goes from min_ratio to max_ratio over the episode."""

    def setUp(self) -> None:
        self.sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_increase"
        )

    def test_episode_start_is_min(self) -> None:
        self.assertEqual(self.sched.get_ratio(0.0), 5)

    def test_episode_end_is_max(self) -> None:
        self.assertEqual(self.sched.get_ratio(1.0), 20)

    def test_midpoint_is_between(self) -> None:
        ratio = self.sched.get_ratio(0.5)
        self.assertGreater(ratio, 5)
        self.assertLess(ratio, 20)

    def test_monotone_increasing(self) -> None:
        progresses = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ratios = [self.sched.get_ratio(p) for p in progresses]
        for i in range(len(ratios) - 1):
            self.assertLessEqual(ratios[i], ratios[i + 1])


# ---------------------------------------------------------------------------
# 5. make_options integration
# ---------------------------------------------------------------------------


class TestMakeOptions(unittest.TestCase):
    """make_options returns a valid option list with the correct max_steps."""

    def test_fixed_returns_six_options(self) -> None:
        sched = AdaptiveTemporalScheduler(base_ratio=10, adaptation="fixed")
        opts = sched.make_options(0.0)
        self.assertEqual(len(opts), 6)
        self.assertTrue(all(isinstance(o, Option) for o in opts))

    def test_max_steps_matches_ratio_for_non_flanking(self) -> None:
        """Advance, defend, withdraw, concentrate options use the full ratio."""
        sched = AdaptiveTemporalScheduler(base_ratio=20, adaptation="fixed")
        opts = sched.make_options(0.0)
        non_flanking = [o for o in opts if "flank" not in o.name]
        flanking = [o for o in opts if "flank" in o.name]
        self.assertEqual(len(non_flanking), 4)
        self.assertEqual(len(flanking), 2)
        for opt in non_flanking:
            self.assertEqual(opt.max_steps, 20, msg=f"Failed for {opt.name}")

    def test_flanking_options_use_half_ratio(self) -> None:
        """Flanking options use max(1, ratio // 2) as their cap."""
        sched = AdaptiveTemporalScheduler(base_ratio=20, adaptation="fixed")
        opts = sched.make_options(0.0)
        flanking = {o.name: o for o in opts if "flank" in o.name}
        self.assertEqual(len(flanking), 2)
        for name, opt in flanking.items():
            self.assertEqual(opt.max_steps, 10, msg=f"Failed for {name}")

    def test_linear_decrease_changes_options_over_episode(self) -> None:
        sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_decrease"
        )
        opts_start = sched.make_options(0.0)
        opts_end = sched.make_options(1.0)
        # At start, advance option should have max_steps=20; at end, 5.
        advance_start = next(o for o in opts_start if o.name == "advance_sector")
        advance_end = next(o for o in opts_end if o.name == "advance_sector")
        self.assertEqual(advance_start.max_steps, 20)
        self.assertEqual(advance_end.max_steps, 5)

    def test_ratio_1_is_minimum(self) -> None:
        """Even with min_ratio=1, make_options must not raise."""
        sched = AdaptiveTemporalScheduler(base_ratio=1, min_ratio=1, max_ratio=1)
        opts = sched.make_options(0.0)
        self.assertEqual(len(opts), 6)


# ---------------------------------------------------------------------------
# 6. W&B config dict
# ---------------------------------------------------------------------------


class TestWandbConfig(unittest.TestCase):
    """wandb_config returns the expected keys."""

    def test_fixed_adaptation_config(self) -> None:
        sched = AdaptiveTemporalScheduler(base_ratio=20, adaptation="fixed")
        cfg = sched.wandb_config()
        # For fixed, current_ratio = base_ratio
        self.assertEqual(cfg["temporal_ratio"], 20)
        self.assertEqual(cfg["temporal_adaptation"], "fixed")

    def test_linear_decrease_reports_initial_ratio(self) -> None:
        """For linear_decrease, temporal_ratio in config is the start ratio (max_ratio)."""
        sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_decrease"
        )
        cfg = sched.wandb_config()
        # current_ratio before any call = max_ratio (episode start value)
        self.assertEqual(cfg["temporal_ratio"], 20)
        self.assertEqual(cfg["temporal_ratio_min"], 5)
        self.assertEqual(cfg["temporal_ratio_max"], 20)
        self.assertEqual(cfg["temporal_adaptation"], "linear_decrease")

    def test_linear_increase_reports_initial_ratio(self) -> None:
        """For linear_increase, temporal_ratio in config is the start ratio (min_ratio)."""
        sched = AdaptiveTemporalScheduler(
            min_ratio=5, max_ratio=20, adaptation="linear_increase"
        )
        cfg = sched.wandb_config()
        # current_ratio before any call = min_ratio (episode start value)
        self.assertEqual(cfg["temporal_ratio"], 5)
        self.assertEqual(cfg["temporal_adaptation"], "linear_increase")

    def test_keys_present(self) -> None:
        sched = AdaptiveTemporalScheduler(
            base_ratio=10, min_ratio=5, max_ratio=20, adaptation="linear_decrease"
        )
        cfg = sched.wandb_config()
        self.assertIn("temporal_ratio", cfg)
        self.assertIn("temporal_ratio_min", cfg)
        self.assertIn("temporal_ratio_max", cfg)
        self.assertIn("temporal_adaptation", cfg)


# ---------------------------------------------------------------------------
# 7. BrigadeEnv integration
# ---------------------------------------------------------------------------


class TestBrigadeEnvTemporalRatio(unittest.TestCase):
    """BrigadeEnv respects the temporal_ratio parameter."""

    def test_default_temporal_ratio(self) -> None:
        from envs.brigade_env import BrigadeEnv
        env = BrigadeEnv(n_blue=2, n_red=2)
        self.assertEqual(env.temporal_ratio, 10)
        env.close()

    def test_custom_temporal_ratio_sets_option_max_steps(self) -> None:
        from envs.brigade_env import BrigadeEnv
        env = BrigadeEnv(n_blue=2, n_red=2, temporal_ratio=20)
        self.assertEqual(env.temporal_ratio, 20)
        # advance_sector (index 0) should have max_steps = 20
        advance_opt = env._options[0]
        self.assertEqual(advance_opt.name, "advance_sector")
        self.assertEqual(advance_opt.max_steps, 20)
        env.close()

    def test_sweep_ratios_are_valid(self) -> None:
        from envs.brigade_env import BrigadeEnv
        for ratio in SWEEP_RATIOS:
            env = BrigadeEnv(n_blue=2, n_red=2, temporal_ratio=ratio)
            self.assertEqual(env.temporal_ratio, ratio)
            env.close()

    def test_temporal_ratio_zero_raises(self) -> None:
        from envs.brigade_env import BrigadeEnv
        with self.assertRaises(ValueError):
            BrigadeEnv(n_blue=2, n_red=2, temporal_ratio=0)

    def test_temporal_ratio_zero_with_explicit_options_does_not_raise(self) -> None:
        """temporal_ratio is not validated when explicit options override it."""
        from envs.brigade_env import BrigadeEnv
        from envs.options import make_default_options
        custom_opts = make_default_options(max_steps=5)
        # temporal_ratio=0 is meaningless but harmless when options are explicit
        env = BrigadeEnv(n_blue=2, n_red=2, temporal_ratio=0, options=custom_opts)
        env.close()

    def test_explicit_options_override_temporal_ratio(self) -> None:
        """Explicitly provided options take precedence over temporal_ratio."""
        from envs.brigade_env import BrigadeEnv
        from envs.options import make_default_options
        custom_opts = make_default_options(max_steps=7)
        env = BrigadeEnv(n_blue=2, n_red=2, temporal_ratio=50, options=custom_opts)
        # The options passed explicitly have max_steps=7, not 50
        advance_opt = env._options[0]
        self.assertEqual(advance_opt.max_steps, 7)
        env.close()

    def test_episode_runs_with_temporal_ratio(self) -> None:
        """Smoke test: an episode completes without error for each sweep ratio."""
        from envs.brigade_env import BrigadeEnv
        for ratio in SWEEP_RATIOS:
            env = BrigadeEnv(n_blue=2, n_red=2, temporal_ratio=ratio, max_steps=50)
            obs, _ = env.reset(seed=0)
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            env.close()


# ---------------------------------------------------------------------------
# 8. SWEEP_RATIOS constant
# ---------------------------------------------------------------------------


class TestSweepRatios(unittest.TestCase):
    def test_contains_expected_values(self) -> None:
        self.assertEqual(set(SWEEP_RATIOS), {5, 10, 20, 50})

    def test_all_positive(self) -> None:
        self.assertTrue(all(r > 0 for r in SWEEP_RATIOS))


if __name__ == "__main__":
    unittest.main()
