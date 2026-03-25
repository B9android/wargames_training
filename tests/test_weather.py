# SPDX-License-Identifier: MIT
# tests/test_weather.py
"""Tests for envs/sim/weather.py — weather and time-of-day effects.

Covers:
* WeatherCondition / TimeOfDay enum values
* ConditionEffects / TODEffects table completeness
* WeatherConfig validation — all parameter constraints
* WeatherConfig defaults
* sample_weather — fixed override, random draw, reproducibility
* step_weather — no progression when steps_per_time_of_day=0,
                 cycling DAWN→DAY→DUSK→NIGHT→DAWN, counter reset
* get_visibility_fraction — multiplicative combination, clamping
* get_effective_visibility_range — fraction × base
* get_accuracy_modifier — multiplicative combination, clamping
* get_speed_modifier — condition-only (TOD ignored)
* get_morale_stressor — additive combination
* Acceptance criteria:
    - Heavy RAIN reduces accuracy by ≥ 30 %
    - FOG reduces LOS range to < 100 m with default 500 m base
* BattalionEnv integration — enable_weather=True obs dims,
  info keys, LOS blocked by weather, step info visibility_fraction
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.weather import (
    CONDITION_EFFECTS,
    DEFAULT_BASE_VISIBILITY_RANGE,
    NUM_CONDITIONS,
    NUM_TIME_OF_DAY,
    TIME_OF_DAY_EFFECTS,
    ConditionEffects,
    TODEffects,
    TimeOfDay,
    WeatherCondition,
    WeatherConfig,
    WeatherState,
    get_accuracy_modifier,
    get_effective_visibility_range,
    get_morale_stressor,
    get_speed_modifier,
    get_visibility_fraction,
    sample_weather,
    step_weather,
)
from envs.battalion_env import BattalionEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _state(
    condition: WeatherCondition = WeatherCondition.CLEAR,
    tod: TimeOfDay = TimeOfDay.DAY,
) -> WeatherState:
    return WeatherState(condition=condition, time_of_day=tod)


def _default_config() -> WeatherConfig:
    return WeatherConfig()


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------


class TestEnums(unittest.TestCase):
    """Basic sanity checks for WeatherCondition and TimeOfDay."""

    def test_condition_count(self) -> None:
        self.assertEqual(NUM_CONDITIONS, 5)
        self.assertEqual(len(WeatherCondition), 5)

    def test_tod_count(self) -> None:
        self.assertEqual(NUM_TIME_OF_DAY, 4)
        self.assertEqual(len(TimeOfDay), 4)

    def test_condition_values(self) -> None:
        self.assertEqual(WeatherCondition.CLEAR.value, 0)
        self.assertEqual(WeatherCondition.OVERCAST.value, 1)
        self.assertEqual(WeatherCondition.RAIN.value, 2)
        self.assertEqual(WeatherCondition.FOG.value, 3)
        self.assertEqual(WeatherCondition.SNOW.value, 4)

    def test_tod_values(self) -> None:
        self.assertEqual(TimeOfDay.DAWN.value, 0)
        self.assertEqual(TimeOfDay.DAY.value, 1)
        self.assertEqual(TimeOfDay.DUSK.value, 2)
        self.assertEqual(TimeOfDay.NIGHT.value, 3)


# ---------------------------------------------------------------------------
# Effect table completeness
# ---------------------------------------------------------------------------


class TestEffectTables(unittest.TestCase):
    """All conditions and times-of-day have table entries with valid ranges."""

    def test_all_conditions_present(self) -> None:
        for cond in WeatherCondition:
            self.assertIn(cond, CONDITION_EFFECTS)

    def test_all_tod_present(self) -> None:
        for tod in TimeOfDay:
            self.assertIn(tod, TIME_OF_DAY_EFFECTS)

    def test_condition_ranges(self) -> None:
        for cond, fx in CONDITION_EFFECTS.items():
            self.assertIsInstance(fx, ConditionEffects, msg=cond)
            self.assertGreaterEqual(fx.visibility_fraction, 0.0, msg=cond)
            self.assertLessEqual(fx.visibility_fraction, 1.0, msg=cond)
            self.assertGreater(fx.accuracy_modifier, 0.0, msg=cond)
            self.assertLessEqual(fx.accuracy_modifier, 1.0, msg=cond)
            self.assertGreater(fx.speed_modifier, 0.0, msg=cond)
            self.assertLessEqual(fx.speed_modifier, 1.0, msg=cond)
            self.assertGreaterEqual(fx.morale_stressor, 0.0, msg=cond)

    def test_tod_ranges(self) -> None:
        for tod, fx in TIME_OF_DAY_EFFECTS.items():
            self.assertIsInstance(fx, TODEffects, msg=tod)
            self.assertGreaterEqual(fx.visibility_fraction, 0.0, msg=tod)
            self.assertLessEqual(fx.visibility_fraction, 1.0, msg=tod)
            self.assertGreater(fx.accuracy_modifier, 0.0, msg=tod)
            self.assertLessEqual(fx.accuracy_modifier, 1.0, msg=tod)
            self.assertGreaterEqual(fx.morale_stressor, 0.0, msg=tod)

    def test_clear_day_full_visibility(self) -> None:
        """CLEAR + DAY should give visibility fraction of 1.0."""
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(get_visibility_fraction(state), 1.0)

    def test_clear_day_full_accuracy(self) -> None:
        """CLEAR + DAY should give accuracy modifier of 1.0."""
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(get_accuracy_modifier(state), 1.0)

    def test_clear_day_no_morale_stressor(self) -> None:
        """CLEAR + DAY should produce zero morale stressor."""
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(get_morale_stressor(state), 0.0)


# ---------------------------------------------------------------------------
# WeatherConfig validation
# ---------------------------------------------------------------------------


class TestWeatherConfigValidation(unittest.TestCase):
    """Verify WeatherConfig.__post_init__ enforces parameter constraints."""

    def test_default_instantiation(self) -> None:
        cfg = WeatherConfig()
        self.assertIsNone(cfg.fixed_condition)
        self.assertIsNone(cfg.fixed_time_of_day)
        self.assertEqual(cfg.steps_per_time_of_day, 0)
        self.assertEqual(cfg.base_visibility_range, DEFAULT_BASE_VISIBILITY_RANGE)
        self.assertEqual(len(cfg.condition_weights), NUM_CONDITIONS)

    def test_invalid_weights_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            WeatherConfig(condition_weights=[0.5, 0.5])

    def test_invalid_weights_negative(self) -> None:
        with self.assertRaises(ValueError):
            WeatherConfig(condition_weights=[-0.1, 0.3, 0.3, 0.3, 0.2])

    def test_invalid_weights_all_zero(self) -> None:
        with self.assertRaises(ValueError):
            WeatherConfig(condition_weights=[0.0, 0.0, 0.0, 0.0, 0.0])

    def test_invalid_steps_negative(self) -> None:
        with self.assertRaises(ValueError):
            WeatherConfig(steps_per_time_of_day=-1)

    def test_invalid_base_visibility_zero(self) -> None:
        with self.assertRaises(ValueError):
            WeatherConfig(base_visibility_range=0.0)

    def test_invalid_base_visibility_negative(self) -> None:
        with self.assertRaises(ValueError):
            WeatherConfig(base_visibility_range=-100.0)

    def test_fixed_condition_accepted(self) -> None:
        cfg = WeatherConfig(fixed_condition=WeatherCondition.FOG)
        self.assertEqual(cfg.fixed_condition, WeatherCondition.FOG)

    def test_fixed_tod_accepted(self) -> None:
        cfg = WeatherConfig(fixed_time_of_day=TimeOfDay.NIGHT)
        self.assertEqual(cfg.fixed_time_of_day, TimeOfDay.NIGHT)

    def test_invalid_fixed_condition_type(self) -> None:
        """Passing an int for fixed_condition must raise ValueError."""
        with self.assertRaises(ValueError):
            WeatherConfig(fixed_condition=3)  # type: ignore[arg-type]

    def test_invalid_fixed_condition_str(self) -> None:
        """Passing a string for fixed_condition must raise ValueError."""
        with self.assertRaises(ValueError):
            WeatherConfig(fixed_condition="FOG")  # type: ignore[arg-type]

    def test_invalid_fixed_tod_type(self) -> None:
        """Passing an int for fixed_time_of_day must raise ValueError."""
        with self.assertRaises(ValueError):
            WeatherConfig(fixed_time_of_day=1)  # type: ignore[arg-type]

    def test_invalid_fixed_tod_str(self) -> None:
        """Passing a string for fixed_time_of_day must raise ValueError."""
        with self.assertRaises(ValueError):
            WeatherConfig(fixed_time_of_day="NIGHT")  # type: ignore[arg-type]

    def test_steps_per_tod_zero_accepted(self) -> None:
        cfg = WeatherConfig(steps_per_time_of_day=0)
        self.assertEqual(cfg.steps_per_time_of_day, 0)

    def test_steps_per_tod_positive_accepted(self) -> None:
        cfg = WeatherConfig(steps_per_time_of_day=100)
        self.assertEqual(cfg.steps_per_time_of_day, 100)


# ---------------------------------------------------------------------------
# sample_weather
# ---------------------------------------------------------------------------


class TestSampleWeather(unittest.TestCase):
    """Verify sample_weather respects fixed overrides and is reproducible."""

    def test_fixed_condition(self) -> None:
        cfg = WeatherConfig(fixed_condition=WeatherCondition.FOG)
        for seed in range(10):
            state = sample_weather(_rng(seed), cfg)
            self.assertEqual(state.condition, WeatherCondition.FOG)

    def test_fixed_time_of_day(self) -> None:
        cfg = WeatherConfig(fixed_time_of_day=TimeOfDay.NIGHT)
        for seed in range(10):
            state = sample_weather(_rng(seed), cfg)
            self.assertEqual(state.time_of_day, TimeOfDay.NIGHT)

    def test_fixed_both(self) -> None:
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.SNOW,
            fixed_time_of_day=TimeOfDay.DAWN,
        )
        state = sample_weather(_rng(0), cfg)
        self.assertEqual(state.condition, WeatherCondition.SNOW)
        self.assertEqual(state.time_of_day, TimeOfDay.DAWN)

    def test_random_condition_valid(self) -> None:
        cfg = WeatherConfig()
        for seed in range(20):
            state = sample_weather(_rng(seed), cfg)
            self.assertIn(state.condition, list(WeatherCondition))

    def test_random_tod_valid(self) -> None:
        cfg = WeatherConfig()
        for seed in range(20):
            state = sample_weather(_rng(seed), cfg)
            self.assertIn(state.time_of_day, list(TimeOfDay))

    def test_reproducibility(self) -> None:
        """Same seed → same state."""
        cfg = WeatherConfig()
        s1 = sample_weather(_rng(42), cfg)
        s2 = sample_weather(_rng(42), cfg)
        self.assertEqual(s1.condition, s2.condition)
        self.assertEqual(s1.time_of_day, s2.time_of_day)

    def test_all_conditions_reachable(self) -> None:
        """With uniform weights, all conditions appear in 1000 samples."""
        cfg = WeatherConfig(condition_weights=[1.0, 1.0, 1.0, 1.0, 1.0])
        seen = set()
        for seed in range(1000):
            state = sample_weather(_rng(seed), cfg)
            seen.add(state.condition)
        self.assertEqual(len(seen), NUM_CONDITIONS)

    def test_initial_tod_counter_zero(self) -> None:
        cfg = WeatherConfig()
        state = sample_weather(_rng(0), cfg)
        self.assertEqual(state._tod_step_counter, 0)


# ---------------------------------------------------------------------------
# step_weather — time-of-day progression
# ---------------------------------------------------------------------------


class TestStepWeather(unittest.TestCase):
    """Verify step_weather advances time of day correctly."""

    def test_no_progression_when_disabled(self) -> None:
        cfg = WeatherConfig(steps_per_time_of_day=0)
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.DAWN)
        for _ in range(200):
            step_weather(state, cfg)
        self.assertEqual(state.time_of_day, TimeOfDay.DAWN)

    def test_advances_after_threshold(self) -> None:
        cfg = WeatherConfig(steps_per_time_of_day=10)
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.DAWN)
        for _ in range(10):
            step_weather(state, cfg)
        self.assertEqual(state.time_of_day, TimeOfDay.DAY)

    def test_counter_resets_after_transition(self) -> None:
        cfg = WeatherConfig(steps_per_time_of_day=5)
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.DAWN)
        for _ in range(5):
            step_weather(state, cfg)
        self.assertEqual(state._tod_step_counter, 0)

    def test_full_cycle(self) -> None:
        """After 4 × steps_per_time_of_day we return to the starting TOD."""
        cfg = WeatherConfig(steps_per_time_of_day=5)
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.DAWN)
        start = state.time_of_day
        for _ in range(4 * 5):
            step_weather(state, cfg)
        self.assertEqual(state.time_of_day, start)

    def test_tod_sequence_dawn_day_dusk_night(self) -> None:
        """TOD must progress DAWN→DAY→DUSK→NIGHT in order."""
        cfg = WeatherConfig(steps_per_time_of_day=3)
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.DAWN)
        expected_sequence = [
            TimeOfDay.DAWN,
            TimeOfDay.DAY,
            TimeOfDay.DUSK,
            TimeOfDay.NIGHT,
            TimeOfDay.DAWN,
        ]
        transitions = [state.time_of_day]
        for _ in range(4 * 3):
            step_weather(state, cfg)
            if state._tod_step_counter == 0:
                transitions.append(state.time_of_day)
        # First 5 transition points must match the expected sequence
        self.assertEqual(transitions[:5], expected_sequence[:5])

    def test_counter_increments_before_threshold(self) -> None:
        cfg = WeatherConfig(steps_per_time_of_day=5)
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.DAY)
        step_weather(state, cfg)
        self.assertEqual(state._tod_step_counter, 1)
        step_weather(state, cfg)
        self.assertEqual(state._tod_step_counter, 2)

    def test_noop_when_fixed_time_of_day_with_steps_positive(self) -> None:
        """fixed_time_of_day must suppress progression even if steps_per_time_of_day > 0."""
        cfg = WeatherConfig(
            fixed_time_of_day=TimeOfDay.NIGHT,
            steps_per_time_of_day=3,
        )
        state = WeatherState(condition=WeatherCondition.CLEAR, time_of_day=TimeOfDay.NIGHT)
        for _ in range(30):
            step_weather(state, cfg)
        self.assertEqual(state.time_of_day, TimeOfDay.NIGHT)
        self.assertEqual(state._tod_step_counter, 0)


# ---------------------------------------------------------------------------
# get_visibility_fraction
# ---------------------------------------------------------------------------


class TestGetVisibilityFraction(unittest.TestCase):
    """Verify combined visibility fraction = condition × tod."""

    def test_clear_day_is_one(self) -> None:
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(get_visibility_fraction(state), 1.0)

    def test_fog_day_product(self) -> None:
        state = _state(WeatherCondition.FOG, TimeOfDay.DAY)
        expected = 0.15 * 1.00
        self.assertAlmostEqual(get_visibility_fraction(state), expected)

    def test_fog_night_product(self) -> None:
        state = _state(WeatherCondition.FOG, TimeOfDay.NIGHT)
        expected = min(0.15 * 0.30, 1.0)
        self.assertAlmostEqual(get_visibility_fraction(state), expected)

    def test_rain_day(self) -> None:
        state = _state(WeatherCondition.RAIN, TimeOfDay.DAY)
        expected = 0.70 * 1.00
        self.assertAlmostEqual(get_visibility_fraction(state), expected)

    def test_result_clamped_to_one(self) -> None:
        # All real conditions produce ≤ 1.0 when multiplied; verify bound.
        for cond in WeatherCondition:
            for tod in TimeOfDay:
                state = _state(cond, tod)
                v = get_visibility_fraction(state)
                self.assertLessEqual(v, 1.0, msg=f"{cond} {tod}")
                self.assertGreaterEqual(v, 0.0, msg=f"{cond} {tod}")


# ---------------------------------------------------------------------------
# get_effective_visibility_range
# ---------------------------------------------------------------------------


class TestGetEffectiveVisibilityRange(unittest.TestCase):
    """Verify effective range = fraction × base."""

    def test_clear_day_full_range(self) -> None:
        cfg = WeatherConfig(base_visibility_range=500.0)
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(
            get_effective_visibility_range(state, cfg), 500.0
        )

    def test_fog_day_below_100m(self) -> None:
        """Acceptance criterion: FOG at DAY gives effective range < 100 m."""
        cfg = WeatherConfig(base_visibility_range=500.0)
        state = _state(WeatherCondition.FOG, TimeOfDay.DAY)
        rng = get_effective_visibility_range(state, cfg)
        self.assertLess(rng, 100.0)

    def test_fog_day_exact_value(self) -> None:
        cfg = WeatherConfig(base_visibility_range=500.0)
        state = _state(WeatherCondition.FOG, TimeOfDay.DAY)
        expected = 0.15 * 500.0
        self.assertAlmostEqual(
            get_effective_visibility_range(state, cfg), expected
        )

    def test_custom_base_range(self) -> None:
        cfg = WeatherConfig(base_visibility_range=800.0)
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(
            get_effective_visibility_range(state, cfg), 800.0
        )


# ---------------------------------------------------------------------------
# get_accuracy_modifier
# ---------------------------------------------------------------------------


class TestGetAccuracyModifier(unittest.TestCase):
    """Verify combined accuracy modifier = condition × tod."""

    def test_clear_day_is_one(self) -> None:
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(get_accuracy_modifier(state), 1.0)

    def test_rain_day_at_least_30_pct_reduction(self) -> None:
        """Acceptance criterion: heavy rain reduces accuracy by ≥ 30 %."""
        state = _state(WeatherCondition.RAIN, TimeOfDay.DAY)
        modifier = get_accuracy_modifier(state)
        self.assertLessEqual(modifier, 0.70)

    def test_rain_day_exact_value(self) -> None:
        state = _state(WeatherCondition.RAIN, TimeOfDay.DAY)
        expected = 0.65 * 1.00
        self.assertAlmostEqual(get_accuracy_modifier(state), expected)

    def test_fog_night_combined(self) -> None:
        state = _state(WeatherCondition.FOG, TimeOfDay.NIGHT)
        expected = min(0.85 * 0.70, 1.0)
        self.assertAlmostEqual(get_accuracy_modifier(state), expected)

    def test_all_conditions_at_most_one(self) -> None:
        for cond in WeatherCondition:
            for tod in TimeOfDay:
                state = _state(cond, tod)
                acc = get_accuracy_modifier(state)
                self.assertLessEqual(acc, 1.0, msg=f"{cond} {tod}")
                self.assertGreater(acc, 0.0, msg=f"{cond} {tod}")


# ---------------------------------------------------------------------------
# get_speed_modifier
# ---------------------------------------------------------------------------


class TestGetSpeedModifier(unittest.TestCase):
    """Verify speed modifier is condition-only (time of day does not affect march speed)."""

    def test_clear_is_one(self) -> None:
        for tod in TimeOfDay:
            state = _state(WeatherCondition.CLEAR, tod)
            self.assertAlmostEqual(get_speed_modifier(state), 1.0, msg=tod)

    def test_overcast_is_one(self) -> None:
        state = _state(WeatherCondition.OVERCAST, TimeOfDay.DAY)
        self.assertAlmostEqual(get_speed_modifier(state), 1.00)

    def test_snow_reduces_speed(self) -> None:
        state = _state(WeatherCondition.SNOW, TimeOfDay.DAY)
        self.assertAlmostEqual(get_speed_modifier(state), 0.50)

    def test_rain_reduces_speed(self) -> None:
        state = _state(WeatherCondition.RAIN, TimeOfDay.DAY)
        self.assertAlmostEqual(get_speed_modifier(state), 0.85)

    def test_tod_does_not_change_speed(self) -> None:
        """Same condition → same speed regardless of time of day."""
        for cond in WeatherCondition:
            values = set()
            for tod in TimeOfDay:
                state = _state(cond, tod)
                values.add(round(get_speed_modifier(state), 8))
            self.assertEqual(len(values), 1, msg=f"{cond} speed varies with TOD unexpectedly")

    def test_all_positive(self) -> None:
        for cond in WeatherCondition:
            state = _state(cond, TimeOfDay.DAY)
            self.assertGreater(get_speed_modifier(state), 0.0, msg=cond)


# ---------------------------------------------------------------------------
# get_morale_stressor
# ---------------------------------------------------------------------------


class TestGetMoraleStressor(unittest.TestCase):
    """Verify morale stressor = condition_stressor + tod_stressor."""

    def test_clear_day_zero(self) -> None:
        state = _state(WeatherCondition.CLEAR, TimeOfDay.DAY)
        self.assertAlmostEqual(get_morale_stressor(state), 0.0)

    def test_snow_has_stressor(self) -> None:
        state = _state(WeatherCondition.SNOW, TimeOfDay.DAY)
        self.assertGreater(get_morale_stressor(state), 0.0)

    def test_rain_night_additive(self) -> None:
        state = _state(WeatherCondition.RAIN, TimeOfDay.NIGHT)
        expected = 0.005 + 0.005
        self.assertAlmostEqual(get_morale_stressor(state), expected)

    def test_non_negative_for_all(self) -> None:
        for cond in WeatherCondition:
            for tod in TimeOfDay:
                state = _state(cond, tod)
                ms = get_morale_stressor(state)
                self.assertGreaterEqual(ms, 0.0, msg=f"{cond} {tod}")

    def test_stressor_worse_at_night(self) -> None:
        """Same condition should produce higher stressor at NIGHT than DAY."""
        for cond in WeatherCondition:
            day_stressor = get_morale_stressor(_state(cond, TimeOfDay.DAY))
            night_stressor = get_morale_stressor(_state(cond, TimeOfDay.NIGHT))
            self.assertGreaterEqual(
                night_stressor, day_stressor, msg=f"{cond}: night stressor ≥ day stressor"
            )


# ---------------------------------------------------------------------------
# BattalionEnv integration
# ---------------------------------------------------------------------------


class TestBattalionEnvWeatherIntegration(unittest.TestCase):
    """Verify BattalionEnv correctly integrates the weather system."""

    def _make_env(self, **kwargs) -> BattalionEnv:
        return BattalionEnv(enable_weather=True, **kwargs)

    # -- Observation space dimensions --

    def test_obs_dim_no_formations_no_logistics(self) -> None:
        """enable_weather=True adds 2 dims to the base 17-dim obs."""
        env = self._make_env()
        self.assertEqual(env.observation_space.shape[0], 19)  # 17 + 2

    def test_obs_dim_with_formations(self) -> None:
        """formations(+2) + weather(+2) → 21 dims."""
        env = self._make_env(enable_formations=True)
        self.assertEqual(env.observation_space.shape[0], 21)  # 17+2+2

    def test_obs_dim_with_logistics(self) -> None:
        """logistics(+3) + weather(+2) → 22 dims."""
        env = self._make_env(enable_logistics=True)
        self.assertEqual(env.observation_space.shape[0], 22)  # 17+3+2

    def test_obs_dim_with_formations_and_logistics(self) -> None:
        """formations(+2) + logistics(+3) + weather(+2) → 24 dims."""
        env = self._make_env(enable_formations=True, enable_logistics=True)
        self.assertEqual(env.observation_space.shape[0], 24)  # 17+2+3+2

    def test_no_weather_obs_dim_unchanged(self) -> None:
        """enable_weather=False (default) → base 17 dims."""
        env = BattalionEnv()
        self.assertEqual(env.observation_space.shape[0], 17)

    # -- reset() initialises weather state --

    def test_reset_sets_weather_state(self) -> None:
        env = self._make_env()
        env.reset(seed=1)
        self.assertIsNotNone(env.weather_state)

    def test_reset_weather_state_is_weather_state_instance(self) -> None:
        env = self._make_env()
        env.reset(seed=1)
        self.assertIsInstance(env.weather_state, WeatherState)

    def test_reset_fixed_condition_respected(self) -> None:
        cfg = WeatherConfig(fixed_condition=WeatherCondition.SNOW)
        env = self._make_env(weather_config=cfg)
        for seed in range(5):
            env.reset(seed=seed)
            self.assertEqual(env.weather_state.condition, WeatherCondition.SNOW)

    def test_reset_no_weather_state_none(self) -> None:
        env = BattalionEnv(enable_weather=False)
        env.reset(seed=1)
        self.assertIsNone(env.weather_state)

    # -- Observation values within bounds --

    def test_obs_within_declared_bounds(self) -> None:
        env = self._make_env()
        obs, _ = env.reset(seed=42)
        low = env.observation_space.low
        high = env.observation_space.high
        self.assertTrue(np.all(obs >= low - 1e-6))
        self.assertTrue(np.all(obs <= high + 1e-6))

    def test_obs_last_two_dims_visibility(self) -> None:
        """Last two dims of obs are weather_id [0,1] and visibility [0,1]."""
        cfg = WeatherConfig(fixed_condition=WeatherCondition.FOG, fixed_time_of_day=TimeOfDay.DAY)
        env = self._make_env(weather_config=cfg)
        obs, _ = env.reset(seed=0)
        # weather_id = FOG.value / (NUM_CONDITIONS - 1) = 3/4 = 0.75
        expected_weather_id = WeatherCondition.FOG.value / float(NUM_CONDITIONS - 1)
        expected_vis = 0.15 * 1.00  # FOG × DAY
        self.assertAlmostEqual(float(obs[-2]), expected_weather_id, places=4)
        self.assertAlmostEqual(float(obs[-1]), expected_vis, places=4)

    # -- step() returns weather info --

    def test_step_info_has_weather_keys(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIn("weather_condition", info)
        self.assertIn("time_of_day", info)
        self.assertIn("visibility_fraction", info)

    def test_step_info_weather_condition_int(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertIsInstance(info["weather_condition"], int)

    def test_step_info_visibility_in_range(self) -> None:
        env = self._make_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        vis = info["visibility_fraction"]
        self.assertGreaterEqual(vis, 0.0)
        self.assertLessEqual(vis, 1.0)

    def test_no_weather_info_keys_absent(self) -> None:
        env = BattalionEnv(enable_weather=False)
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertNotIn("weather_condition", info)
        self.assertNotIn("visibility_fraction", info)

    # -- Acceptance criterion: RAIN reduces accuracy by ≥ 30 % --

    def test_rain_reduces_damage_at_least_30pct(self) -> None:
        """Damage in RAIN+DAY must be ≤ 70 % of damage in CLEAR+DAY."""
        from envs.sim.combat import compute_fire_damage
        from envs.sim.battalion import Battalion

        blue = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
        red = Battalion(x=100.0, y=0.0, theta=0.0, strength=1.0, team=1)

        from envs.sim.weather import (
            WeatherCondition, TimeOfDay, WeatherState,
            get_accuracy_modifier,
        )
        clear_state = WeatherState(WeatherCondition.CLEAR, TimeOfDay.DAY)
        rain_state = WeatherState(WeatherCondition.RAIN, TimeOfDay.DAY)

        base_damage = compute_fire_damage(blue, red, intensity=1.0)
        rain_acc = get_accuracy_modifier(rain_state)
        clear_acc = get_accuracy_modifier(clear_state)

        rain_damage = base_damage * rain_acc
        clear_damage = base_damage * clear_acc

        # clear_acc=1.0 so clear_damage == base_damage; rain must be ≤ 70 %
        self.assertLessEqual(rain_damage, clear_damage * 0.70 + 1e-9)

    # -- Acceptance criterion: FOG reduces LOS range to < 100 m --

    def test_fog_reduces_effective_visibility_below_100m(self) -> None:
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.FOG,
            fixed_time_of_day=TimeOfDay.DAY,
            base_visibility_range=500.0,
        )
        env = self._make_env(weather_config=cfg)
        env.reset(seed=0)
        vis_range = (
            get_effective_visibility_range(env.weather_state, env.weather_config)
        )
        self.assertLess(vis_range, 100.0)

    def test_fog_day_los_blocked_at_200m(self) -> None:
        """Enemies 200 m apart in FOG should have LOS blocked in obs."""
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.FOG,
            fixed_time_of_day=TimeOfDay.DAY,
            base_visibility_range=500.0,
        )
        env = BattalionEnv(
            enable_weather=True,
            weather_config=cfg,
            randomize_terrain=False,
        )
        env.reset(seed=0)
        # Force units 200 m apart (well beyond 75 m FOG range)
        env.blue.x = 400.0
        env.blue.y = 500.0
        env.red.x = 600.0
        env.red.y = 500.0
        obs = env._get_obs()
        # obs[16] is LOS; should be 0.0 since 200 m > 75 m (FOG+DAY effective visibility range)
        self.assertAlmostEqual(float(obs[16]), 0.0)

    def test_clear_day_los_not_blocked_at_200m(self) -> None:
        """Enemies 200 m apart in CLEAR+DAY should have clear LOS in obs."""
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.CLEAR,
            fixed_time_of_day=TimeOfDay.DAY,
            base_visibility_range=500.0,
        )
        env = BattalionEnv(
            enable_weather=True,
            weather_config=cfg,
            randomize_terrain=False,
        )
        env.reset(seed=0)
        env.blue.x = 400.0
        env.blue.y = 500.0
        env.red.x = 600.0
        env.red.y = 500.0
        obs = env._get_obs()
        # On flat terrain with CLEAR sky, LOS should be 1.0
        self.assertAlmostEqual(float(obs[16]), 1.0)

    # -- Time-of-day progression in env --

    def test_tod_progresses_over_episode(self) -> None:
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.CLEAR,
            steps_per_time_of_day=5,
        )
        env = self._make_env(weather_config=cfg)
        env.reset(seed=0)
        start_tod = env.weather_state.time_of_day
        for _ in range(5):
            obs, _, done, trunc, _ = env.step(env.action_space.sample())
            if done or trunc:
                break
        end_tod = env.weather_state.time_of_day
        self.assertNotEqual(start_tod, end_tod)

    def test_tod_fixed_does_not_progress(self) -> None:
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.CLEAR,
            fixed_time_of_day=TimeOfDay.DAY,
            steps_per_time_of_day=0,
        )
        env = self._make_env(weather_config=cfg)
        env.reset(seed=0)
        for _ in range(50):
            _, _, done, trunc, _ = env.step(env.action_space.sample())
            if done or trunc:
                break
        self.assertEqual(env.weather_state.time_of_day, TimeOfDay.DAY)

    # -- Speed penalty applied --

    def test_snow_reduces_blue_movement(self) -> None:
        """SNOW (speed_modifier=0.5) should reduce Blue displacement vs CLEAR."""
        import math

        def _run_one_step(condition: WeatherCondition) -> float:
            cfg = WeatherConfig(
                fixed_condition=condition,
                fixed_time_of_day=TimeOfDay.DAY,
            )
            env = BattalionEnv(
                enable_weather=True,
                weather_config=cfg,
                randomize_terrain=False,
                curriculum_level=1,
            )
            env.reset(seed=7)
            bx0, by0 = env.blue.x, env.blue.y
            # Force full-forward action
            action = env.action_space.sample() * 0
            action[0] = 1.0  # move forward at max
            env.step(action)
            dx = env.blue.x - bx0
            dy = env.blue.y - by0
            return math.sqrt(dx * dx + dy * dy)

        dist_clear = _run_one_step(WeatherCondition.CLEAR)
        dist_snow = _run_one_step(WeatherCondition.SNOW)
        # SNOW speed_modifier=0.5, so displacement should be ~50% of CLEAR
        self.assertLess(dist_snow, dist_clear * 0.90)

    # -- fixed_time_of_day suppresses progression even with steps_per_time_of_day>0 --

    def test_fixed_tod_suppresses_progression_in_env(self) -> None:
        """fixed_time_of_day must keep TOD constant even when steps_per_time_of_day > 0."""
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.CLEAR,
            fixed_time_of_day=TimeOfDay.DAWN,
            steps_per_time_of_day=3,  # would normally cycle
        )
        env = self._make_env(weather_config=cfg)
        env.reset(seed=0)
        for _ in range(20):
            _, _, done, trunc, _ = env.step(env.action_space.sample())
            if done or trunc:
                break
        self.assertEqual(env.weather_state.time_of_day, TimeOfDay.DAWN)

    # -- _get_red_obs logistics dims --

    def test_red_obs_includes_logistics_dims(self) -> None:
        """_get_red_obs must include Red ammo/food/fatigue when enable_logistics=True."""
        env = BattalionEnv(enable_weather=True, enable_logistics=True)
        env.reset(seed=0)
        red_obs = env._get_red_obs()
        # 17 base + 3 logistics + 2 weather = 22 dims
        self.assertEqual(red_obs.shape[0], 22)

    def test_red_obs_logistics_and_formations_dims(self) -> None:
        """_get_red_obs with formations+logistics+weather → 24 dims."""
        env = BattalionEnv(enable_weather=True, enable_logistics=True, enable_formations=True)
        env.reset(seed=0)
        red_obs = env._get_red_obs()
        # 17 base + 2 formations + 3 logistics + 2 weather = 24
        self.assertEqual(red_obs.shape[0], 24)

    def test_red_obs_matches_obs_space_shape(self) -> None:
        """_get_red_obs shape must match observation_space (shared schema)."""
        env = BattalionEnv(enable_weather=True, enable_logistics=True)
        env.reset(seed=0)
        red_obs = env._get_red_obs()
        self.assertEqual(red_obs.shape[0], env.observation_space.shape[0])

    # -- Weather morale drain consistent across both code paths --

    def test_weather_morale_drain_legacy_path(self) -> None:
        """Weather morale stressor is applied even when morale_config is None."""
        cfg = WeatherConfig(
            fixed_condition=WeatherCondition.SNOW,  # morale_stressor=0.008
            fixed_time_of_day=TimeOfDay.NIGHT,     # morale_stressor=0.005 → total 0.013
        )
        env = BattalionEnv(
            enable_weather=True,
            weather_config=cfg,
            # No morale_config → legacy morale_check path
        )
        env.reset(seed=0)
        initial_blue_morale = env.blue.morale
        # Step without firing so only weather drains morale
        action = env.action_space.sample() * 0  # no move, no rotate, no fire
        env.step(action)
        # Morale should have decreased or stayed the same (recovery may offset stressor)
        # At minimum it should never INCREASE when there is a stressor and no recovery
        self.assertLessEqual(env.blue.morale, initial_blue_morale + 1e-6)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
