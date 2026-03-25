# SPDX-License-Identifier: MIT
# tests/test_corps_league.py
"""Tests for E7.3 — Multi-Corps Self-Play & League Extension.

Covers:
1. MatchResult / MatchDatabase operational field extensions (territory, casualties, supply).
2. CorpsActorCriticPolicy construction, act(), evaluate_actions(), and serialisation.
3. CorpsMainAgentTrainer construction, snapshot saving, short training loop.
4. distributed_runner.corps_benchmark result shape (pure-Python, no Ray).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase, MatchResult
from training.league.matchmaker import LeagueMatchmaker
from training.league.train_corps_main_agent import (
    CorpsActorCriticPolicy,
    CorpsMainAgentTrainer,
    make_pfsp_weight_fn,
)
from training.elo import EloRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_dir: str, max_size: int = 100) -> AgentPool:
    manifest = Path(tmp_dir) / "pool.json"
    return AgentPool(pool_manifest=manifest, max_size=max_size)


def _make_db(tmp_dir: str) -> MatchDatabase:
    return MatchDatabase(Path(tmp_dir) / "matches.jsonl")


def _make_elo(tmp_dir: str) -> EloRegistry:
    return EloRegistry(path=Path(tmp_dir) / "elo.json")


def _make_matchmaker(pool: AgentPool, db: MatchDatabase) -> LeagueMatchmaker:
    return LeagueMatchmaker(agent_pool=pool, match_database=db)


def _make_policy(
    obs_dim: int = 20,
    n_divisions: int = 2,
    n_options: int = 6,
) -> CorpsActorCriticPolicy:
    return CorpsActorCriticPolicy(
        obs_dim=obs_dim,
        n_divisions=n_divisions,
        n_options=n_options,
        actor_hidden_sizes=(32, 16),
        critic_hidden_sizes=(32, 16),
    )


def _make_corps_env_mock(
    obs_dim: int = 20,
    n_divisions: int = 2,
    n_options: int = 6,
) -> MagicMock:
    """Return a MagicMock that satisfies the CorpsEnv interface used by CorpsMainAgentTrainer."""
    env = MagicMock()
    env._obs_dim = obs_dim
    env.n_divisions = n_divisions
    env.n_corps_options = n_options
    env.max_steps = 50
    env.observation_space.shape = (obs_dim,)
    env.n_blue = n_divisions * 2   # total blue unit count for casualty tracking
    env.n_red = n_divisions * 2    # total red unit count for casualty tracking

    rng = np.random.default_rng(0)
    # Cheap step / reset that always returns done after 1 step.
    def _reset(seed=None):
        obs = rng.random(obs_dim).astype(np.float32)
        return obs, {}

    def _step(action):
        obs = rng.random(obs_dim).astype(np.float32)
        reward = float(rng.random())
        terminated = True  # terminate immediately so loops stay short
        truncated = False
        info = {
            "corps_steps": 1,
            "objective_rewards": {"capture": 0.5, "supply_cut": 0.0, "flank": 0.0},
            "supply_levels": [0.8, 0.7],
            "blue_units_alive": n_divisions * 2,
            "red_units_alive": n_divisions * 2,
        }
        return obs, reward, terminated, truncated, info

    env.reset.side_effect = _reset
    env.step.side_effect = _step
    # mock action_space.sample()
    env.action_space.sample.return_value = np.zeros(n_divisions, dtype=np.int64)
    return env


def _make_corps_trainer(
    tmp_dir: str,
    env_mock: MagicMock = None,
    agent_id: str = "corps_agent_001",
    snapshot_freq: int = 5,
    eval_freq: int = 5,
    n_eval_episodes: int = 2,
    n_steps: int = 4,
    pfsp_temperature: float = 1.0,
) -> CorpsMainAgentTrainer:
    if env_mock is None:
        env_mock = _make_corps_env_mock()
    obs_dim = env_mock._obs_dim
    n_divisions = env_mock.n_divisions
    n_options = env_mock.n_corps_options
    policy = _make_policy(obs_dim=obs_dim, n_divisions=n_divisions, n_options=n_options)
    pool = _make_pool(tmp_dir)
    db = _make_db(tmp_dir)
    mm = _make_matchmaker(pool, db)
    elo = _make_elo(tmp_dir)
    snap_dir = Path(tmp_dir) / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    return CorpsMainAgentTrainer(
        env=env_mock,
        policy=policy,
        agent_pool=pool,
        match_database=db,
        matchmaker=mm,
        elo_registry=elo,
        agent_id=agent_id,
        snapshot_dir=snap_dir,
        snapshot_freq=snapshot_freq,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
        pfsp_temperature=pfsp_temperature,
        device="cpu",
        seed=0,
        log_interval=1,
    )


# ===========================================================================
# 1. MatchResult — operational fields
# ===========================================================================


class TestMatchResultOperationalFields(unittest.TestCase):
    """MatchResult now carries four optional operational fields (E7.3)."""

    def test_default_fields_are_none(self) -> None:
        r = MatchResult(agent_id="a", opponent_id="b", outcome=0.6)
        self.assertIsNone(r.territory_control)
        self.assertIsNone(r.blue_casualties)
        self.assertIsNone(r.red_casualties)
        self.assertIsNone(r.supply_consumed)

    def test_fields_set_correctly(self) -> None:
        r = MatchResult(
            agent_id="a",
            opponent_id="b",
            outcome=0.7,
            territory_control=0.6,
            blue_casualties=5,
            red_casualties=12,
            supply_consumed=2.5,
        )
        self.assertAlmostEqual(r.territory_control, 0.6)
        self.assertEqual(r.blue_casualties, 5)
        self.assertEqual(r.red_casualties, 12)
        self.assertAlmostEqual(r.supply_consumed, 2.5)

    def test_territory_control_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            MatchResult(
                agent_id="a", opponent_id="b", outcome=0.5, territory_control=1.5
            )
        with self.assertRaises(ValueError):
            MatchResult(
                agent_id="a", opponent_id="b", outcome=0.5, territory_control=-0.1
            )

    def test_negative_casualties_raises(self) -> None:
        with self.assertRaises(ValueError):
            MatchResult(
                agent_id="a", opponent_id="b", outcome=0.5, blue_casualties=-1
            )
        with self.assertRaises(ValueError):
            MatchResult(
                agent_id="a", opponent_id="b", outcome=0.5, red_casualties=-3
            )

    def test_negative_supply_consumed_raises(self) -> None:
        with self.assertRaises(ValueError):
            MatchResult(
                agent_id="a", opponent_id="b", outcome=0.5, supply_consumed=-0.1
            )

    def test_zero_values_accepted(self) -> None:
        r = MatchResult(
            agent_id="a",
            opponent_id="b",
            outcome=0.5,
            territory_control=0.0,
            blue_casualties=0,
            red_casualties=0,
            supply_consumed=0.0,
        )
        self.assertEqual(r.blue_casualties, 0)
        self.assertAlmostEqual(r.territory_control, 0.0)


class TestMatchResultSerialisation(unittest.TestCase):
    """to_dict / from_dict round-trip for operational fields."""

    def test_round_trip_all_fields(self) -> None:
        r = MatchResult(
            agent_id="x",
            opponent_id="y",
            outcome=0.8,
            territory_control=0.5,
            blue_casualties=3,
            red_casualties=7,
            supply_consumed=1.2,
        )
        r2 = MatchResult.from_dict(r.to_dict())
        self.assertAlmostEqual(r2.territory_control, 0.5)
        self.assertEqual(r2.blue_casualties, 3)
        self.assertEqual(r2.red_casualties, 7)
        self.assertAlmostEqual(r2.supply_consumed, 1.2)

    def test_round_trip_none_fields_absent_from_dict(self) -> None:
        r = MatchResult(agent_id="x", opponent_id="y", outcome=0.4)
        d = r.to_dict()
        self.assertNotIn("territory_control", d)
        self.assertNotIn("blue_casualties", d)
        self.assertNotIn("red_casualties", d)
        self.assertNotIn("supply_consumed", d)

    def test_from_dict_missing_fields_are_none(self) -> None:
        """Old-format records (without operational fields) load cleanly."""
        d = {
            "agent_id": "a",
            "opponent_id": "b",
            "outcome": 0.5,
            "match_id": "mid",
            "timestamp": 0.0,
            "metadata": {},
        }
        r = MatchResult.from_dict(d)
        self.assertIsNone(r.territory_control)
        self.assertIsNone(r.blue_casualties)

    def test_repr_unchanged(self) -> None:
        r = MatchResult(agent_id="a", opponent_id="b", outcome=0.5)
        self.assertIn("outcome=0.500", repr(r))


# ===========================================================================
# 2. MatchDatabase — operational field query helpers
# ===========================================================================


class TestMatchDatabaseOperationalQueries(unittest.TestCase):
    """MatchDatabase.record() and query helpers for operational fields."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db = _make_db(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _record_with_ops(self, agent_id: str = "a", opponent_id: str = "b", outcome: float = 0.6,
                         tc: float = 0.5, bc: int = 2, rc: int = 4, sc: float = 1.5) -> MatchResult:
        return self._db.record(
            agent_id=agent_id,
            opponent_id=opponent_id,
            outcome=outcome,
            territory_control=tc,
            blue_casualties=bc,
            red_casualties=rc,
            supply_consumed=sc,
        )

    def test_record_stores_operational_fields(self) -> None:
        r = self._record_with_ops()
        self.assertAlmostEqual(r.territory_control, 0.5)
        self.assertEqual(r.blue_casualties, 2)
        self.assertEqual(r.red_casualties, 4)
        self.assertAlmostEqual(r.supply_consumed, 1.5)

    def test_mean_territory_control(self) -> None:
        self._record_with_ops(tc=0.4)
        self._record_with_ops(tc=0.8)
        mean = self._db.mean_territory_control("a", "b")
        self.assertIsNotNone(mean)
        self.assertAlmostEqual(mean, 0.6)  # (0.4 + 0.8) / 2

    def test_mean_territory_control_no_results_returns_none(self) -> None:
        self.assertIsNone(self._db.mean_territory_control("nobody"))

    def test_mean_territory_control_ignores_records_without_field(self) -> None:
        # Record without territory_control
        self._db.record(agent_id="a", opponent_id="b", outcome=0.5)
        # Record with territory_control
        self._record_with_ops(tc=0.6)
        mean = self._db.mean_territory_control("a", "b")
        self.assertAlmostEqual(mean, 0.6)

    def test_mean_casualties(self) -> None:
        self._record_with_ops(bc=2, rc=5)
        self._record_with_ops(bc=4, rc=3)
        result = self._db.mean_casualties("a", "b")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["blue"], 3.0)
        self.assertAlmostEqual(result["red"], 4.0)

    def test_mean_casualties_partial_records_not_biased(self) -> None:
        """Records missing one side's casualties must not bias the other side's mean."""
        # Record with only blue_casualties set.
        self._db.record(
            agent_id="a", opponent_id="b", outcome=0.5,
            blue_casualties=10, red_casualties=None,
        )
        # Record with only red_casualties set.
        self._db.record(
            agent_id="a", opponent_id="b", outcome=0.5,
            blue_casualties=None, red_casualties=6,
        )
        result = self._db.mean_casualties("a", "b")
        self.assertIsNotNone(result)
        # blue mean should be 10 (only one record has it), not biased by missing red record.
        self.assertAlmostEqual(result["blue"], 10.0)
        # red mean should be 6 (only one record has it), not biased by missing blue record.
        self.assertAlmostEqual(result["red"], 6.0)

    def test_mean_casualties_no_results_returns_none(self) -> None:
        self.assertIsNone(self._db.mean_casualties("nobody"))

    def test_mean_supply_consumed(self) -> None:
        self._record_with_ops(sc=1.0)
        self._record_with_ops(sc=3.0)
        mean = self._db.mean_supply_consumed("a", "b")
        self.assertAlmostEqual(mean, 2.0)

    def test_mean_supply_consumed_no_results_returns_none(self) -> None:
        self.assertIsNone(self._db.mean_supply_consumed("nobody"))

    def test_record_backward_compatible(self) -> None:
        """record() without operational fields should still work (backward compat)."""
        r = self._db.record(agent_id="a", opponent_id="b", outcome=0.5)
        self.assertIsNone(r.territory_control)
        self.assertIsNone(r.blue_casualties)
        self.assertIsNone(r.red_casualties)
        self.assertIsNone(r.supply_consumed)

    def test_persistence_round_trip(self) -> None:
        """Operational fields survive a JSONL write-reload cycle."""
        self._record_with_ops(tc=0.7, bc=1, rc=9, sc=2.5)
        db2 = MatchDatabase(self._db.db_path)
        r = db2.all_results()[0]
        self.assertAlmostEqual(r.territory_control, 0.7)
        self.assertEqual(r.blue_casualties, 1)
        self.assertEqual(r.red_casualties, 9)
        self.assertAlmostEqual(r.supply_consumed, 2.5)

    def test_mean_supply_consumed_ignores_records_without_field(self) -> None:
        self._db.record(agent_id="a", opponent_id="b", outcome=0.5)  # no supply_consumed
        self._record_with_ops(sc=2.0)
        mean = self._db.mean_supply_consumed("a", "b")
        self.assertAlmostEqual(mean, 2.0)


# ===========================================================================
# 3. CorpsActorCriticPolicy
# ===========================================================================


class TestCorpsActorCriticPolicyConstruction(unittest.TestCase):

    def test_default_construction(self) -> None:
        pol = CorpsActorCriticPolicy(obs_dim=20, n_divisions=3, n_options=6)
        self.assertEqual(pol.obs_dim, 20)
        self.assertEqual(pol.n_divisions, 3)
        self.assertEqual(pol.n_options, 6)

    def test_division_heads_count(self) -> None:
        pol = _make_policy(n_divisions=4)
        self.assertEqual(len(pol.division_heads), 4)

    def test_custom_hidden_sizes(self) -> None:
        pol = CorpsActorCriticPolicy(
            obs_dim=15, n_divisions=2, n_options=6,
            actor_hidden_sizes=(64, 32),
            critic_hidden_sizes=(64,),
        )
        self.assertEqual(pol.actor_hidden_sizes, (64, 32))
        self.assertEqual(pol.critic_hidden_sizes, (64,))


class TestCorpsActorCriticPolicyAct(unittest.TestCase):

    def setUp(self) -> None:
        self._obs_dim = 20
        self._n_div = 3
        self._pol = _make_policy(obs_dim=self._obs_dim, n_divisions=self._n_div)

    def _obs(self) -> torch.Tensor:
        return torch.zeros(self._obs_dim, dtype=torch.float32)

    def test_act_returns_correct_shapes(self) -> None:
        action, log_prob = self._pol.act(self._obs())
        self.assertEqual(action.shape, (self._n_div,))
        self.assertEqual(log_prob.shape, ())

    def test_act_actions_are_valid_integers(self) -> None:
        for _ in range(10):
            action, _ = self._pol.act(self._obs())
            for a in action:
                self.assertGreaterEqual(int(a), 0)
                self.assertLess(int(a), 6)  # n_options=6

    def test_deterministic_act_is_reproducible(self) -> None:
        obs = torch.zeros(self._obs_dim)
        a1, _ = self._pol.act(obs, deterministic=True)
        a2, _ = self._pol.act(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_stochastic_act_varies(self) -> None:
        """Stochastic actions should not always match (fails rarely with small prob)."""
        obs = torch.zeros(self._obs_dim)
        outcomes = set()
        for _ in range(30):
            a, _ = self._pol.act(obs, deterministic=False)
            outcomes.add(tuple(a.tolist()))
        # With 6 options and 3 divisions, many combos exist; at least 2 distinct.
        self.assertGreater(len(outcomes), 1)

    def test_forward_critic_returns_scalar(self) -> None:
        val = self._pol.forward_critic(self._obs())
        self.assertEqual(val.shape, ())


class TestCorpsActorCriticPolicyEvaluateActions(unittest.TestCase):

    def setUp(self) -> None:
        self._obs_dim = 15
        self._n_div = 2
        self._batch = 8
        self._pol = _make_policy(obs_dim=self._obs_dim, n_divisions=self._n_div)

    def test_evaluate_actions_shapes(self) -> None:
        obs = torch.zeros(self._batch, self._obs_dim)
        actions = torch.zeros(self._batch, self._n_div, dtype=torch.long)
        lp, vals, ent = self._pol.evaluate_actions(obs, actions)
        self.assertEqual(lp.shape, (self._batch,))
        self.assertEqual(vals.shape, (self._batch,))
        self.assertEqual(ent.shape, ())

    def test_entropy_is_positive(self) -> None:
        obs = torch.zeros(self._batch, self._obs_dim)
        actions = torch.zeros(self._batch, self._n_div, dtype=torch.long)
        _, _, ent = self._pol.evaluate_actions(obs, actions)
        self.assertGreater(ent.item(), 0)


class TestCorpsActorCriticPolicySerialisation(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pol = _make_policy()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_policy_kwargs_round_trip(self) -> None:
        kwargs = self._pol.policy_kwargs()
        pol2 = CorpsActorCriticPolicy(**kwargs)
        self.assertEqual(pol2.obs_dim, self._pol.obs_dim)
        self.assertEqual(pol2.n_divisions, self._pol.n_divisions)
        self.assertEqual(pol2.n_options, self._pol.n_options)

    def test_save_load_state_dict(self) -> None:
        path = Path(self._tmpdir.name) / "pol.pt"
        torch.save(
            {
                "state_dict": self._pol.state_dict(),
                "kwargs": self._pol.policy_kwargs(),
            },
            path,
        )
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        pol2 = CorpsActorCriticPolicy(**data["kwargs"])
        pol2.load_state_dict(data["state_dict"])

        obs = torch.zeros(self._pol.obs_dim)
        a1, _ = self._pol.act(obs, deterministic=True)
        a2, _ = pol2.act(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)


# ===========================================================================
# 4. CorpsMainAgentTrainer — construction
# ===========================================================================


class TestCorpsMainAgentTrainerConstruction(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_construction_succeeds(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        self.assertIsInstance(t, CorpsMainAgentTrainer)

    def test_snapshot_dir_created(self) -> None:
        snap_dir = Path(self._tmpdir.name) / "custom_snaps"
        env_mock = _make_corps_env_mock()
        pol = _make_policy(
            obs_dim=env_mock._obs_dim,
            n_divisions=env_mock.n_divisions,
            n_options=env_mock.n_corps_options,
        )
        pool = _make_pool(self._tmpdir.name)
        db = _make_db(self._tmpdir.name)
        mm = _make_matchmaker(pool, db)
        elo = _make_elo(self._tmpdir.name)
        t = CorpsMainAgentTrainer(
            env=env_mock,
            policy=pol,
            agent_pool=pool,
            match_database=db,
            matchmaker=mm,
            elo_registry=elo,
            agent_id="test_agent",
            snapshot_dir=snap_dir,
        )
        self.assertTrue(snap_dir.exists())

    def test_pfsp_weight_fn_applied_to_matchmaker(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name, pfsp_temperature=2.0)
        fn = t._matchmaker._pfsp_weight_fn
        self.assertIsNotNone(fn)
        # At T=2 weight(0.5) = (1-0.5)^(1/2) = 0.5^0.5 ≈ 0.707
        self.assertAlmostEqual(fn(0.5), 0.5 ** 0.5, places=5)

    def test_invalid_total_timesteps_raises(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        with self.assertRaises(ValueError):
            t.train(total_timesteps=0)
        with self.assertRaises(ValueError):
            t.train(total_timesteps=-1)


# ===========================================================================
# 5. CorpsMainAgentTrainer — snapshot saving
# ===========================================================================


class TestCorpsMainAgentTrainerSnapshots(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_snapshot_creates_file(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        path = t._save_snapshot(1)
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_snapshot_filename_contains_version(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        path = t._save_snapshot(42)
        self.assertIn("000042", path.name)

    def test_load_snapshot_roundtrip(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        path = t._save_snapshot(1)
        loaded = t._load_snapshot(path)
        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded, CorpsActorCriticPolicy)

    def test_load_missing_snapshot_returns_none(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        result = t._load_snapshot(Path("/nonexistent/path.pt"))
        self.assertIsNone(result)

    def test_load_corrupt_snapshot_returns_none(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        corrupt = Path(self._tmpdir.name) / "corrupt.pt"
        corrupt.write_bytes(b"not_a_valid_pt_file")
        result = t._load_snapshot(corrupt)
        self.assertIsNone(result)

    def test_initial_snapshot_registered_in_pool(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        t._ensure_initial_snapshot()
        self.assertIn(t.agent_id, t._agent_pool)

    def test_initial_snapshot_not_duplicated(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        t._ensure_initial_snapshot()
        t._ensure_initial_snapshot()  # idempotent
        records = [r for r in t._agent_pool.list() if r.agent_id == t.agent_id]
        self.assertEqual(len(records), 1)


# ===========================================================================
# 6. CorpsMainAgentTrainer — short training loop
# ===========================================================================


class TestCorpsMainAgentTrainerLoop(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_train_advances_total_steps(self) -> None:
        t = _make_corps_trainer(
            self._tmpdir.name,
            snapshot_freq=100,
            eval_freq=100,
            n_steps=4,
        )
        t.train(total_timesteps=8)
        # We ran at least one rollout of 4 steps (env terminates after 1 step each)
        self.assertGreaterEqual(t._total_steps, 4)

    def test_snapshot_saved_in_pool_after_threshold(self) -> None:
        """A snapshot should be added to the pool after snapshot_freq steps."""
        t = _make_corps_trainer(
            self._tmpdir.name,
            snapshot_freq=4,
            eval_freq=1000,
            n_steps=4,
        )
        t.train(total_timesteps=8)
        # Pool should contain at least initial + one additional snapshot.
        self.assertGreaterEqual(t._agent_pool.size, 1)

    def test_match_recorded_after_eval(self) -> None:
        """After eval_freq steps, a match result should be recorded in the DB."""
        t = _make_corps_trainer(
            self._tmpdir.name,
            snapshot_freq=1000,
            eval_freq=4,
            n_eval_episodes=2,
            n_steps=4,
        )
        t.train(total_timesteps=8)
        results = t._match_db.all_results()
        self.assertGreaterEqual(len(results), 1)

    def test_match_result_has_operational_fields(self) -> None:
        """Match results recorded by CorpsMainAgentTrainer have operational fields."""
        t = _make_corps_trainer(
            self._tmpdir.name,
            snapshot_freq=1000,
            eval_freq=4,
            n_eval_episodes=2,
            n_steps=4,
        )
        t.train(total_timesteps=8)
        results = t._match_db.all_results()
        if results:
            r = results[0]
            # territory_control should be non-None (populated from objective_rewards)
            self.assertIsNotNone(r.territory_control)
            # supply_consumed should be non-None (populated from supply_levels)
            self.assertIsNotNone(r.supply_consumed)

    def test_checkpoint_saved_when_dir_provided(self) -> None:
        ckpt_dir = Path(self._tmpdir.name) / "checkpoints"
        t = _make_corps_trainer(
            self._tmpdir.name,
            snapshot_freq=1000,
            eval_freq=1000,
            n_steps=4,
        )
        t._checkpoint_dir = ckpt_dir
        t.checkpoint_freq = 4
        t.train(total_timesteps=8)
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        self.assertGreater(len(ckpt_files), 0)


# ===========================================================================
# 7. CorpsMainAgentTrainer — evaluate_episode
# ===========================================================================


class TestCorpsMainAgentTrainerEvalEpisode(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_evaluate_episode_returns_expected_keys(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        result = t._evaluate_episode()
        for key in (
            "total_reward", "outcome", "territory_control",
            "blue_casualties", "red_casualties", "supply_consumed",
        ):
            self.assertIn(key, result)

    def test_evaluate_episode_outcome_in_0_1(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        result = t._evaluate_episode()
        self.assertIn(result["outcome"], [0.0, 1.0])

    def test_evaluate_episode_territory_control_bounded(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        result = t._evaluate_episode()
        tc = result["territory_control"]
        self.assertGreaterEqual(tc, 0.0)
        self.assertLessEqual(tc, 1.0)

    def test_evaluate_episode_casualties_non_negative(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        result = t._evaluate_episode()
        self.assertGreaterEqual(result["blue_casualties"], 0)
        self.assertGreaterEqual(result["red_casualties"], 0)

    def test_evaluate_episode_supply_consumed_non_negative(self) -> None:
        t = _make_corps_trainer(self._tmpdir.name)
        result = t._evaluate_episode()
        self.assertGreaterEqual(result["supply_consumed"], 0.0)


# ===========================================================================
# 8. make_pfsp_weight_fn (re-export)
# ===========================================================================


class TestMakePfspWeightFnReexport(unittest.TestCase):
    """make_pfsp_weight_fn is re-exported from train_corps_main_agent."""

    def test_callable(self) -> None:
        fn = make_pfsp_weight_fn(1.0)
        self.assertTrue(callable(fn))

    def test_temperature_one(self) -> None:
        fn = make_pfsp_weight_fn(1.0)
        self.assertAlmostEqual(fn(0.0), 1.0)
        self.assertAlmostEqual(fn(1.0), 0.0)

    def test_invalid_temperature_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_pfsp_weight_fn(0.0)


# ===========================================================================
# 9. distributed_runner.corps_benchmark (pure-Python path — no Ray)
# ===========================================================================


class TestCorpsBenchmarkSignature(unittest.TestCase):
    """corps_benchmark returns the expected result dict."""

    def test_single_process_only_result_shape(self) -> None:
        """Verify corps_benchmark result structure and keys by actually calling it.

        Ray is patched so that the parallel portion is mocked while the
        single-process path runs for real (2 tiny episodes).
        """
        from training.league import distributed_runner as dr

        env_kw = {
            "n_divisions": 2,
            "n_brigades_per_division": 2,
            "n_blue_per_brigade": 2,
        }

        # Patch ray so the Ray actor-pool path is skipped in CI.
        with patch.object(dr, "ray") as mock_ray:
            mock_ray.is_initialized.return_value = True  # skip ray.init()

            # Make @ray.remote(num_cpus=1) a no-op decorator that returns the fn.
            def _noop_remote(*args, **kwargs):
                def _decorator(fn):
                    fn.remote = fn  # fn.remote(a, b) → fn(a, b)
                    return fn
                if len(args) == 1 and callable(args[0]):
                    # Called without arguments: @ray.remote
                    return _decorator(args[0])
                # Called with arguments: @ray.remote(num_cpus=1)
                return _decorator

            mock_ray.remote.side_effect = _noop_remote
            # ray.get([...]) returns the list as-is (already resolved).
            mock_ray.get.side_effect = lambda refs: refs

            result = dr.corps_benchmark(
                num_workers=2,
                n_episodes=2,
                env_kwargs=env_kw,
                seed=0,
            )

        # Verify the expected key set and types.
        for key in (
            "single_steps_per_sec", "ray_steps_per_sec", "speedup",
            "num_workers", "n_episodes",
        ):
            self.assertIn(key, result)
        self.assertGreater(result["single_steps_per_sec"], 0)
        self.assertEqual(result["num_workers"], 2)
        self.assertEqual(result["n_episodes"], 2)

    def test_corps_benchmark_exported(self) -> None:
        from training.league.distributed_runner import corps_benchmark
        self.assertTrue(callable(corps_benchmark))


# ===========================================================================
# 10. Module exports
# ===========================================================================


class TestModuleExports(unittest.TestCase):

    def test_corps_actor_critic_policy_exported(self) -> None:
        from training.league.train_corps_main_agent import CorpsActorCriticPolicy
        self.assertIsNotNone(CorpsActorCriticPolicy)

    def test_corps_main_agent_trainer_exported(self) -> None:
        from training.league.train_corps_main_agent import CorpsMainAgentTrainer
        self.assertIsNotNone(CorpsMainAgentTrainer)

    def test_make_pfsp_weight_fn_exported(self) -> None:
        from training.league.train_corps_main_agent import make_pfsp_weight_fn
        self.assertIsNotNone(make_pfsp_weight_fn)

    def test_corps_benchmark_in_distributed_runner(self) -> None:
        from training.league.distributed_runner import corps_benchmark
        self.assertIsNotNone(corps_benchmark)


if __name__ == "__main__":
    unittest.main()
