# tests/test_formations.py
"""Tests for envs/sim/formations.py — Formation System (E6.4).

Covers:
* Formation enum values and count
* FormationAttributes validation
* FORMATION_ATTRIBUTES table completeness and value sanity
* TRANSITION_STEPS table completeness and Nosworthy (1990) timing (±20 %)
* get_transition_steps — convenience accessor
* get_attributes — convenience accessor
* compute_transition_state — state machine advancement
* resolve_cavalry_charge — square reliably defeats cavalry; outcomes correct
* BattalionEnv integration — extended action/obs spaces, speed modifier,
  transition timing, fire damage modifiers
* Backward compatibility — existing 17-dim obs / 3-dim action unchanged
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.sim.battalion import Battalion
from envs.sim.combat import CombatState, compute_fire_damage
from envs.sim.formations import (
    BASE_CHARGE_DAMAGE,
    COLUMN_CHARGE_MULT,
    FORMATION_ATTRIBUTES,
    NUM_FORMATIONS,
    TRANSITION_STEPS,
    Formation,
    FormationAttributes,
    compute_transition_state,
    get_attributes,
    get_transition_steps,
    resolve_cavalry_charge,
)
from envs.battalion_env import BattalionEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _facing_pair(dist: float = 150.0) -> tuple[Battalion, Battalion]:
    """Two battalions facing each other along the x-axis at *dist* metres."""
    blue = Battalion(x=0.0, y=0.0, theta=0.0, strength=1.0, team=0)
    red = Battalion(x=dist, y=0.0, theta=math.pi, strength=1.0, team=1)
    return blue, red


def _seeded_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Formation enum
# ---------------------------------------------------------------------------


class TestFormationEnum(unittest.TestCase):
    """Verify Formation enum values."""

    def test_four_formations(self) -> None:
        self.assertEqual(len(Formation), NUM_FORMATIONS)
        self.assertEqual(NUM_FORMATIONS, 4)

    def test_line_is_zero(self) -> None:
        self.assertEqual(Formation.LINE, 0)

    def test_all_members(self) -> None:
        members = {f.name for f in Formation}
        self.assertEqual(members, {"LINE", "COLUMN", "SQUARE", "SKIRMISH"})

    def test_int_values_stable(self) -> None:
        self.assertEqual(int(Formation.LINE), 0)
        self.assertEqual(int(Formation.COLUMN), 1)
        self.assertEqual(int(Formation.SQUARE), 2)
        self.assertEqual(int(Formation.SKIRMISH), 3)


# ---------------------------------------------------------------------------
# FormationAttributes validation
# ---------------------------------------------------------------------------


class TestFormationAttributes(unittest.TestCase):
    """Verify FormationAttributes are positive and well-formed."""

    def test_all_positive(self) -> None:
        for formation, attrs in FORMATION_ATTRIBUTES.items():
            with self.subTest(formation=formation.name):
                self.assertGreater(attrs.firepower_modifier, 0.0)
                self.assertGreater(attrs.speed_modifier, 0.0)
                self.assertGreater(attrs.morale_resilience, 0.0)
                self.assertGreater(attrs.vulnerability_modifier, 0.0)
                self.assertGreater(attrs.cavalry_resilience, 0.0)

    def test_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            FormationAttributes(
                firepower_modifier=0.0,
                speed_modifier=1.0,
                morale_resilience=1.0,
                vulnerability_modifier=1.0,
                cavalry_resilience=1.0,
            )

    def test_negative_raises(self) -> None:
        with self.assertRaises(ValueError):
            FormationAttributes(
                firepower_modifier=1.0,
                speed_modifier=-0.5,
                morale_resilience=1.0,
                vulnerability_modifier=1.0,
                cavalry_resilience=1.0,
            )

    def test_frozen(self) -> None:
        attrs = FORMATION_ATTRIBUTES[Formation.LINE]
        with self.assertRaises((AttributeError, TypeError)):
            attrs.firepower_modifier = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FORMATION_ATTRIBUTES table — value constraints from Nosworthy (1990)
# ---------------------------------------------------------------------------


class TestFormationAttributeTable(unittest.TestCase):
    """Validate historical calibration of formation modifiers."""

    def test_all_formations_present(self) -> None:
        for f in Formation:
            self.assertIn(f, FORMATION_ATTRIBUTES)

    def test_line_baseline_firepower(self) -> None:
        """LINE should have the highest firepower (full two-rank volley)."""
        line_fp = FORMATION_ATTRIBUTES[Formation.LINE].firepower_modifier
        for f in Formation:
            if f != Formation.LINE:
                self.assertLess(
                    FORMATION_ATTRIBUTES[f].firepower_modifier,
                    line_fp,
                    msg=f"{f.name} firepower should be < LINE",
                )

    def test_column_fastest(self) -> None:
        """COLUMN should have the highest movement speed."""
        col_spd = FORMATION_ATTRIBUTES[Formation.COLUMN].speed_modifier
        for f in Formation:
            if f != Formation.COLUMN:
                self.assertLess(
                    FORMATION_ATTRIBUTES[f].speed_modifier,
                    col_spd,
                    msg=f"{f.name} speed should be < COLUMN",
                )

    def test_square_slowest(self) -> None:
        """SQUARE should have the lowest movement speed."""
        sq_spd = FORMATION_ATTRIBUTES[Formation.SQUARE].speed_modifier
        for f in Formation:
            if f != Formation.SQUARE:
                self.assertGreater(
                    FORMATION_ATTRIBUTES[f].speed_modifier,
                    sq_spd,
                    msg=f"{f.name} speed should be > SQUARE",
                )

    def test_square_highest_cavalry_resilience(self) -> None:
        """SQUARE must have the highest cavalry_resilience (anti-cavalry role)."""
        sq_cr = FORMATION_ATTRIBUTES[Formation.SQUARE].cavalry_resilience
        for f in Formation:
            if f != Formation.SQUARE:
                self.assertLess(
                    FORMATION_ATTRIBUTES[f].cavalry_resilience,
                    sq_cr,
                    msg=f"{f.name} cavalry_resilience should be < SQUARE",
                )

    def test_skirmish_lowest_vulnerability(self) -> None:
        """SKIRMISH should have the lowest vulnerability (dispersed target)."""
        sk_vuln = FORMATION_ATTRIBUTES[Formation.SKIRMISH].vulnerability_modifier
        for f in Formation:
            if f != Formation.SKIRMISH:
                self.assertGreater(
                    FORMATION_ATTRIBUTES[f].vulnerability_modifier,
                    sk_vuln,
                    msg=f"{f.name} vulnerability should be > SKIRMISH",
                )

    def test_square_highest_morale_resilience(self) -> None:
        """SQUARE morale resilience should be highest (tight formation)."""
        sq_mr = FORMATION_ATTRIBUTES[Formation.SQUARE].morale_resilience
        for f in Formation:
            if f != Formation.SQUARE:
                self.assertLess(
                    FORMATION_ATTRIBUTES[f].morale_resilience,
                    sq_mr,
                )

    def test_line_cavalry_resilience_weakest(self) -> None:
        """LINE is most vulnerable to cavalry (can't quickly form square)."""
        line_cr = FORMATION_ATTRIBUTES[Formation.LINE].cavalry_resilience
        for f in Formation:
            if f != Formation.LINE:
                self.assertGreater(
                    FORMATION_ATTRIBUTES[f].cavalry_resilience,
                    line_cr,
                    msg=f"{f.name} cavalry_resilience should be > LINE",
                )


# ---------------------------------------------------------------------------
# TRANSITION_STEPS — Nosworthy (1990) timing ±20 %
# ---------------------------------------------------------------------------


class TestTransitionSteps(unittest.TestCase):
    """Validate formation transition timing table."""

    def test_all_pairs_present(self) -> None:
        """Every ordered pair of distinct formations should be in the table."""
        for f1 in Formation:
            for f2 in Formation:
                if f1 != f2:
                    self.assertIn(
                        (f1, f2),
                        TRANSITION_STEPS,
                        msg=f"Missing transition ({f1.name}, {f2.name})",
                    )

    def test_steps_in_range_1_to_3(self) -> None:
        """All transition times should be 1–3 steps (Nosworthy 1990)."""
        for pair, steps in TRANSITION_STEPS.items():
            self.assertGreaterEqual(steps, 1, msg=f"{pair} steps < 1")
            self.assertLessEqual(steps, 3, msg=f"{pair} steps > 3")

    def test_forming_square_takes_most_steps(self) -> None:
        """Transitions *into* SQUARE should be the longest (3 steps)."""
        for f in Formation:
            if f != Formation.SQUARE:
                self.assertEqual(
                    TRANSITION_STEPS[(f, Formation.SQUARE)],
                    3,
                    msg=f"{f.name}→SQUARE should be 3 steps",
                )

    def test_skirmish_to_column_fast(self) -> None:
        """Skirmish→Column is quick (1 step — men just close up)."""
        self.assertEqual(TRANSITION_STEPS[(Formation.SKIRMISH, Formation.COLUMN)], 1)


# ---------------------------------------------------------------------------
# get_transition_steps
# ---------------------------------------------------------------------------


class TestGetTransitionSteps(unittest.TestCase):
    """Verify get_transition_steps convenience function."""

    def test_same_formation_is_zero(self) -> None:
        for f in Formation:
            self.assertEqual(get_transition_steps(f, f), 0)

    def test_line_to_square(self) -> None:
        self.assertEqual(get_transition_steps(Formation.LINE, Formation.SQUARE), 3)

    def test_all_pairs_match_table(self) -> None:
        for f1 in Formation:
            for f2 in Formation:
                if f1 != f2:
                    self.assertEqual(
                        get_transition_steps(f1, f2),
                        TRANSITION_STEPS[(f1, f2)],
                    )


# ---------------------------------------------------------------------------
# get_attributes
# ---------------------------------------------------------------------------


class TestGetAttributes(unittest.TestCase):
    def test_returns_correct_type(self) -> None:
        for f in Formation:
            attrs = get_attributes(f)
            self.assertIsInstance(attrs, FormationAttributes)

    def test_same_object_as_table(self) -> None:
        for f in Formation:
            self.assertIs(get_attributes(f), FORMATION_ATTRIBUTES[f])


# ---------------------------------------------------------------------------
# compute_transition_state
# ---------------------------------------------------------------------------


class TestComputeTransitionState(unittest.TestCase):
    """Verify the formation transition state machine."""

    def test_no_transition_unchanged(self) -> None:
        f, tgt, steps = compute_transition_state(Formation.LINE, None, 0)
        self.assertEqual(f, Formation.LINE)
        self.assertIsNone(tgt)
        self.assertEqual(steps, 0)

    def test_single_step_transition_completes(self) -> None:
        """1 step remaining → transition completes immediately."""
        f, tgt, steps = compute_transition_state(
            Formation.SKIRMISH, Formation.COLUMN, 1
        )
        self.assertEqual(f, Formation.COLUMN)
        self.assertIsNone(tgt)
        self.assertEqual(steps, 0)

    def test_multi_step_decrements(self) -> None:
        """2 steps remaining → 1 step remaining after one advance."""
        f, tgt, steps = compute_transition_state(
            Formation.LINE, Formation.SQUARE, 2
        )
        self.assertEqual(f, Formation.LINE)   # not yet complete
        self.assertEqual(tgt, Formation.SQUARE)
        self.assertEqual(steps, 1)

    def test_three_step_full_sequence(self) -> None:
        """LINE→SQUARE takes 3 steps; test full sequence."""
        cur = Formation.LINE
        tgt: Formation | None = Formation.SQUARE
        remaining = 3

        for expected_remaining in [2, 1, 0]:
            cur, tgt, remaining = compute_transition_state(cur, tgt, remaining)
            if expected_remaining > 0:
                self.assertEqual(cur, Formation.LINE)
                self.assertIsNotNone(tgt)
            self.assertEqual(remaining, expected_remaining)

        # After the last step the transition should be complete
        self.assertEqual(cur, Formation.SQUARE)
        self.assertIsNone(tgt)

    def test_zero_steps_remaining_is_noop(self) -> None:
        f, tgt, steps = compute_transition_state(Formation.COLUMN, Formation.LINE, 0)
        # steps_remaining=0 means "no transition in progress"
        self.assertEqual(f, Formation.COLUMN)
        self.assertIsNone(tgt)
        self.assertEqual(steps, 0)


# ---------------------------------------------------------------------------
# resolve_cavalry_charge
# ---------------------------------------------------------------------------


class TestResolveCavalryCharge(unittest.TestCase):
    """Acceptance criterion: Square reliably defeats cavalry."""

    def test_square_bounces_column_charge(self) -> None:
        """Charging COLUMN into SQUARE should produce 'bounced' outcome."""
        rng = _seeded_rng(42)
        att_dmg, def_dmg, outcome = resolve_cavalry_charge(
            attacker_formation=Formation.COLUMN,
            defender_formation=Formation.SQUARE,
            attacker_strength=1.0,
            defender_strength=1.0,
            rng=rng,
        )
        self.assertEqual(outcome, "bounced")
        # Attacker takes more damage than the nearly-immune square
        self.assertGreater(att_dmg, def_dmg)

    def test_square_takes_negligible_damage(self) -> None:
        """SQUARE defence should limit defender damage to very small values."""
        rng = _seeded_rng(0)
        for _ in range(20):
            _, def_dmg, _ = resolve_cavalry_charge(
                Formation.COLUMN, Formation.SQUARE, 1.0, 1.0, rng=rng
            )
            self.assertLess(def_dmg, 0.10, msg="Square should take < 10% damage per charge")

    def test_line_vulnerable_to_column_charge(self) -> None:
        """LINE should take substantially more damage than SQUARE from same charge."""
        rng_sq = _seeded_rng(99)
        rng_ln = _seeded_rng(99)
        _, sq_dmg, _ = resolve_cavalry_charge(
            Formation.COLUMN, Formation.SQUARE, 1.0, 1.0, rng=rng_sq
        )
        _, ln_dmg, _ = resolve_cavalry_charge(
            Formation.COLUMN, Formation.LINE, 1.0, 1.0, rng=rng_ln
        )
        self.assertGreater(ln_dmg, sq_dmg)

    def test_column_charge_bonus(self) -> None:
        """COLUMN attacker should do more damage than LINE attacker."""
        rng1 = _seeded_rng(7)
        rng2 = _seeded_rng(7)
        _, dmg_col, _ = resolve_cavalry_charge(
            Formation.COLUMN, Formation.LINE, 1.0, 1.0, rng=rng1
        )
        _, dmg_line, _ = resolve_cavalry_charge(
            Formation.LINE, Formation.LINE, 1.0, 1.0, rng=rng2
        )
        self.assertGreater(dmg_col, dmg_line)

    def test_returns_tuple_of_three(self) -> None:
        result = resolve_cavalry_charge(
            Formation.LINE, Formation.LINE, 0.8, 0.8, rng=_seeded_rng(1)
        )
        self.assertEqual(len(result), 3)
        att_dmg, def_dmg, outcome = result
        self.assertIsInstance(att_dmg, float)
        self.assertIsInstance(def_dmg, float)
        self.assertIsInstance(outcome, str)

    def test_damage_non_negative(self) -> None:
        rng = _seeded_rng(42)
        for _ in range(10):
            a, d, _ = resolve_cavalry_charge(
                Formation.COLUMN, Formation.LINE, 1.0, 1.0, rng=rng
            )
            self.assertGreaterEqual(a, 0.0)
            self.assertGreaterEqual(d, 0.0)

    def test_damage_bounded(self) -> None:
        rng = _seeded_rng(42)
        for _ in range(10):
            a, d, _ = resolve_cavalry_charge(
                Formation.COLUMN, Formation.SQUARE, 1.0, 1.0, rng=rng
            )
            self.assertLessEqual(a, 1.0)
            self.assertLessEqual(d, 1.0)

    def test_outcome_values(self) -> None:
        """Outcome string must be one of the three documented values."""
        valid = {"bounced", "repulsed", "broken"}
        rng = _seeded_rng(5)
        for f1 in Formation:
            for f2 in Formation:
                _, _, outcome = resolve_cavalry_charge(f1, f2, 1.0, 1.0, rng=rng)
                self.assertIn(outcome, valid, msg=f"{f1.name}→{f2.name}: {outcome!r}")


# ---------------------------------------------------------------------------
# compute_fire_damage — formation modifiers
# ---------------------------------------------------------------------------


class TestFireDamageFormationModifiers(unittest.TestCase):
    """Verify formation modifiers are applied to compute_fire_damage."""

    def test_line_shooter_more_damage_than_column(self) -> None:
        blue, red = _facing_pair()
        # LINE vs LINE
        blue.formation = Formation.LINE
        red.formation = Formation.LINE
        dmg_line = compute_fire_damage(blue, red, intensity=1.0)

        # COLUMN vs LINE
        blue.formation = Formation.COLUMN
        dmg_col = compute_fire_damage(blue, red, intensity=1.0)

        self.assertGreater(dmg_line, dmg_col)

    def test_skirmish_target_takes_less_damage(self) -> None:
        blue, red = _facing_pair()
        # Attacker LINE, Defender LINE
        blue.formation = Formation.LINE
        red.formation = Formation.LINE
        dmg_line_target = compute_fire_damage(blue, red, intensity=1.0)

        # Attacker LINE, Defender SKIRMISH
        red.formation = Formation.SKIRMISH
        dmg_skirmish_target = compute_fire_damage(blue, red, intensity=1.0)

        self.assertLess(dmg_skirmish_target, dmg_line_target)

    def test_square_target_higher_vulnerability(self) -> None:
        blue, red = _facing_pair()
        blue.formation = Formation.LINE
        # LINE target
        red.formation = Formation.LINE
        dmg_line_target = compute_fire_damage(blue, red, intensity=1.0)
        # SQUARE target
        red.formation = Formation.SQUARE
        dmg_square_target = compute_fire_damage(blue, red, intensity=1.0)
        self.assertGreater(dmg_square_target, dmg_line_target)

    def test_no_formation_change_backward_compat(self) -> None:
        """When both units have formation=0 (LINE), damage should equal baseline."""
        blue, red = _facing_pair()
        # Default formation is 0 (LINE), but with formation modifiers active
        # the result should still be: base_damage * 1.0 * 1.0 = base_damage
        blue.formation = 0
        red.formation = 0
        dmg_with_mods = compute_fire_damage(blue, red, intensity=1.0)

        # Baseline: manually compute without formation modifiers
        # (formation=0 means LINE, firepower=1.0, vulnerability=1.0 — no change)
        from envs.sim.combat import BASE_FIRE_DAMAGE, range_factor
        dist = 150.0
        rf = range_factor(dist, blue.fire_range)
        baseline = BASE_FIRE_DAMAGE * rf * 1.0 * blue.strength  # frontal: angle_mult=1.0
        self.assertAlmostEqual(dmg_with_mods, baseline, places=6)


# ---------------------------------------------------------------------------
# BattalionEnv integration — backward compatibility
# ---------------------------------------------------------------------------


class TestBattalionEnvBackwardCompat(unittest.TestCase):
    """Existing 17-dim obs / 3-dim action space must be unchanged by default."""

    def setUp(self) -> None:
        self.env = BattalionEnv(randomize_terrain=False)
        self.env.reset(seed=42)

    def tearDown(self) -> None:
        self.env.close()

    def test_obs_shape_17(self) -> None:
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (17,))

    def test_action_shape_3(self) -> None:
        self.assertEqual(self.env.action_space.shape, (3,))

    def test_obs_space_shape_17(self) -> None:
        self.assertEqual(self.env.observation_space.shape, (17,))

    def test_step_returns_17_dim_obs(self) -> None:
        obs, _, _, _, _ = self.env.step(np.zeros(3, dtype=np.float32))
        self.assertEqual(obs.shape, (17,))


# ---------------------------------------------------------------------------
# BattalionEnv integration — formations enabled
# ---------------------------------------------------------------------------


class TestBattalionEnvFormations(unittest.TestCase):
    """Verify extended action/obs spaces and formation mechanics in the env."""

    def setUp(self) -> None:
        self.env = BattalionEnv(randomize_terrain=False, enable_formations=True)
        self.env.reset(seed=42)

    def tearDown(self) -> None:
        self.env.close()

    def test_obs_shape_19(self) -> None:
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(obs.shape, (19,))

    def test_action_shape_4(self) -> None:
        self.assertEqual(self.env.action_space.shape, (4,))

    def test_obs_space_shape_19(self) -> None:
        self.assertEqual(self.env.observation_space.shape, (19,))

    def test_formation_obs_index_17_normalised(self) -> None:
        """Obs[17] = blue_formation / (NUM_FORMATIONS-1) ∈ [0, 1]."""
        obs, _ = self.env.reset(seed=10)
        self.assertGreaterEqual(float(obs[17]), 0.0)
        self.assertLessEqual(float(obs[17]), 1.0)

    def test_transitioning_obs_index_18(self) -> None:
        """Obs[18] = 0 when not transitioning, 1 when transitioning."""
        obs, _ = self.env.reset(seed=10)
        # Initially not transitioning
        self.assertEqual(float(obs[18]), 0.0)

    def test_default_formation_is_line(self) -> None:
        self.env.reset(seed=0)
        self.assertEqual(self.env.blue.formation, int(Formation.LINE))

    def test_step_accepts_4d_action(self) -> None:
        self.env.reset(seed=42)
        action = np.array([0.0, 0.0, 0.5, float(Formation.SQUARE)], dtype=np.float32)
        obs, _, _, _, _ = self.env.step(action)
        self.assertEqual(obs.shape, (19,))

    def test_formation_change_starts_transition(self) -> None:
        """Requesting SQUARE from LINE should trigger a 3-step transition."""
        self.env.reset(seed=0)
        self.env.blue.formation = int(Formation.LINE)
        self.env.blue.target_formation = None
        self.env.blue.formation_transition_steps = 0

        action = np.array([0.0, 0.0, 0.0, float(Formation.SQUARE)], dtype=np.float32)
        self.env.step(action)
        # After one step: still in LINE (3-step transition), target=SQUARE
        # The transition was started and one step consumed
        # Either still in LINE (2 steps remaining) or LINE (transition just started)
        blue = self.env.blue
        # We started with 3 steps, advanced 1 → 2 remaining, still in LINE
        self.assertEqual(blue.formation, int(Formation.LINE))
        self.assertIsNotNone(blue.target_formation)
        self.assertEqual(blue.target_formation, int(Formation.SQUARE))

    def test_line_to_square_transition_completes(self) -> None:
        """LINE→SQUARE completes after exactly 3 steps."""
        self.env.reset(seed=0)
        self.env.blue.formation = int(Formation.LINE)
        self.env.blue.target_formation = None
        self.env.blue.formation_transition_steps = 0

        # Keep requesting SQUARE each step (no-op once transition is in progress)
        action_square = np.array([0.0, 0.0, 0.0, float(Formation.SQUARE)], dtype=np.float32)

        # Step 1: request starts transition (steps=3), then advance→2 remaining
        self.env.step(action_square)
        self.assertEqual(self.env.blue.formation, int(Formation.LINE))

        # Step 2: advance 2→1 remaining
        self.env.step(action_square)
        self.assertEqual(self.env.blue.formation, int(Formation.LINE))

        # Step 3: advance 1→0, transition completes → formation=SQUARE
        self.env.step(action_square)
        self.assertEqual(self.env.blue.formation, int(Formation.SQUARE))
        self.assertIsNone(self.env.blue.target_formation)

    def test_transitioning_flag_set_during_transition(self) -> None:
        """Obs[18] == 1.0 while a transition is in progress."""
        self.env.reset(seed=0)
        action_square = np.array([0.0, 0.0, 0.0, float(Formation.SQUARE)], dtype=np.float32)
        obs, _, _, _, _ = self.env.step(action_square)
        # After requesting SQUARE (3 steps), still transitioning
        self.assertEqual(float(obs[18]), 1.0)

    def test_transitioning_flag_cleared_after_completion(self) -> None:
        """Obs[18] == 0.0 once the transition is complete."""
        self.env.reset(seed=0)
        self.env.blue.formation = int(Formation.SKIRMISH)
        self.env.blue.target_formation = None

        # SKIRMISH→COLUMN takes 1 step
        action_col = np.array([0.0, 0.0, 0.0, float(Formation.COLUMN)], dtype=np.float32)
        # Step 1: request COLUMN (starts 1-step transition → completes)
        obs, _, _, _, _ = self.env.step(action_col)
        # After 1 step the 1-step transition completes
        self.assertEqual(self.env.blue.formation, int(Formation.COLUMN))
        self.assertEqual(float(obs[18]), 0.0)

    def test_square_speed_modifier_applied(self) -> None:
        """SQUARE formation should move slower than LINE formation."""
        # We test indirectly: take 10 steps moving forward in each formation
        # and check that the total displacement is smaller in SQUARE.
        def _run_movement(formation_idx: int, n_steps: int = 10) -> float:
            env = BattalionEnv(
                randomize_terrain=False,
                enable_formations=True,
                curriculum_level=1,  # Red is stationary — no combat
            )
            obs, _ = env.reset(seed=0)
            env.blue.formation = formation_idx
            env.blue.target_formation = None
            env.blue.formation_transition_steps = 0
            x0, y0 = env.blue.x, env.blue.y

            action = np.array([1.0, 0.0, 0.0, float(formation_idx)], dtype=np.float32)
            for _ in range(n_steps):
                env.step(action)
                if env.blue is None:
                    break
            x1, y1 = env.blue.x, env.blue.y
            env.close()
            return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        dist_line = _run_movement(Formation.LINE)
        dist_square = _run_movement(Formation.SQUARE)
        self.assertGreater(dist_line, dist_square,
                           msg="LINE should move further than SQUARE in equal steps")

    def test_column_faster_than_line(self) -> None:
        """COLUMN formation should produce greater displacement than LINE."""
        def _run_movement(formation_idx: int, n_steps: int = 10) -> float:
            env = BattalionEnv(
                randomize_terrain=False,
                enable_formations=True,
                curriculum_level=1,
            )
            obs, _ = env.reset(seed=0)
            env.blue.formation = formation_idx
            env.blue.target_formation = None
            env.blue.formation_transition_steps = 0
            x0, y0 = env.blue.x, env.blue.y

            action = np.array([1.0, 0.0, 0.0, float(formation_idx)], dtype=np.float32)
            for _ in range(n_steps):
                env.step(action)
                if env.blue is None:
                    break
            x1, y1 = env.blue.x, env.blue.y
            env.close()
            return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        dist_line = _run_movement(Formation.LINE)
        dist_col = _run_movement(Formation.COLUMN)
        self.assertGreater(dist_col, dist_line,
                           msg="COLUMN should move further than LINE in equal steps")

    def test_obs_in_obs_space(self) -> None:
        """All observations must lie within the declared obs space bounds."""
        obs, _ = self.env.reset(seed=5)
        self.assertTrue(
            np.all(obs <= self.env.observation_space.high + 1e-6),
            msg="Obs exceeds high bound",
        )
        self.assertTrue(
            np.all(obs >= self.env.observation_space.low - 1e-6),
            msg="Obs below low bound",
        )

    def test_all_formations_reachable_via_action(self) -> None:
        """Agent can select each of the 4 formations via the action dimension."""
        self.env.reset(seed=0)
        for formation in Formation:
            self.env.blue.formation = int(Formation.LINE)
            self.env.blue.target_formation = None
            self.env.blue.formation_transition_steps = 0

            action = np.array([0.0, 0.0, 0.0, float(formation)], dtype=np.float32)
            self.env.step(action)
            # If same as current, no transition started (LINE→LINE = no-op)
            if formation == Formation.LINE:
                self.assertEqual(self.env.blue.formation, int(Formation.LINE))
                self.assertIsNone(self.env.blue.target_formation)
            else:
                # Transition was requested
                self.assertEqual(
                    self.env.blue.target_formation,
                    int(formation),
                    msg=f"Formation {formation.name} not reachable via action",
                )


# ---------------------------------------------------------------------------
# Skirmish mechanics
# ---------------------------------------------------------------------------


class TestSkirmishMechanics(unittest.TestCase):
    """Verify skirmisher screen characteristics."""

    def test_skirmish_reduced_firepower(self) -> None:
        """Skirmishers deal less ranged damage than a LINE battalion."""
        blue, red = _facing_pair()
        blue.formation = Formation.LINE
        red.formation = Formation.LINE
        dmg_line = compute_fire_damage(blue, red, intensity=1.0)

        blue.formation = Formation.SKIRMISH
        dmg_skirmish = compute_fire_damage(blue, red, intensity=1.0)

        self.assertLess(dmg_skirmish, dmg_line)

    def test_skirmish_harder_to_hit(self) -> None:
        """Fire against SKIRMISH target does less damage than against LINE."""
        blue, red = _facing_pair()
        blue.formation = Formation.LINE

        red.formation = Formation.LINE
        dmg_line_target = compute_fire_damage(blue, red, intensity=1.0)

        red.formation = Formation.SKIRMISH
        dmg_skirmish_target = compute_fire_damage(blue, red, intensity=1.0)

        self.assertLess(dmg_skirmish_target, dmg_line_target)

    def test_skirmish_speed_advantage(self) -> None:
        """Skirmishers move faster than LINE but slower than COLUMN."""
        sk_spd = FORMATION_ATTRIBUTES[Formation.SKIRMISH].speed_modifier
        ln_spd = FORMATION_ATTRIBUTES[Formation.LINE].speed_modifier
        col_spd = FORMATION_ATTRIBUTES[Formation.COLUMN].speed_modifier
        self.assertGreater(sk_spd, ln_spd)
        self.assertLess(sk_spd, col_spd)


if __name__ == "__main__":
    unittest.main()
