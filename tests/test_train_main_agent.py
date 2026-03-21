# tests/test_train_main_agent.py
"""Tests for training/league/train_main_agent.py (E4.2)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase
from training.league.matchmaker import LeagueMatchmaker
from training.league.train_main_agent import MainAgentTrainer, make_pfsp_weight_fn
from training.elo import EloRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_dir: str, max_size: int = 100) -> AgentPool:
    manifest = Path(tmp_dir) / "pool.json"
    return AgentPool(pool_manifest=manifest, max_size=max_size)


def _make_db(tmp_dir: str) -> MatchDatabase:
    return MatchDatabase(Path(tmp_dir) / "matches.jsonl")


def _dummy_path(tmp_dir: str, name: str = "snap.pt") -> Path:
    p = Path(tmp_dir) / name
    p.write_bytes(b"")
    return p


def _make_elo(tmp_dir: str) -> EloRegistry:
    return EloRegistry(path=Path(tmp_dir) / "elo.json")


def _make_matchmaker(pool: AgentPool, db: MatchDatabase) -> LeagueMatchmaker:
    return LeagueMatchmaker(agent_pool=pool, match_database=db)


def _make_trainer_mock(seed: int = 42, total_steps: int = 0) -> MagicMock:
    """Return a MagicMock that mimics the MAPPOTrainer interface."""
    trainer = MagicMock()
    trainer._total_steps = total_steps
    trainer._rng = np.random.default_rng(seed)
    trainer.seed = seed
    trainer.n_blue = 2
    trainer.n_red = 2
    trainer.device = "cpu"
    trainer.policy = MagicMock()
    # Make policy behave enough like a real MAPPOPolicy for snapshot saving
    trainer.policy.state_dict.return_value = {}
    trainer.policy.obs_dim = 17
    trainer.policy.action_dim = 5
    trainer.policy.state_dim = 13
    trainer.policy.n_agents = 2
    trainer.policy.share_parameters = True
    trainer.policy.actor_hidden_sizes = (128, 64)
    trainer.policy.critic_hidden_sizes = (128, 64)
    trainer.update_policy.return_value = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.5,
        "total_loss": 0.3,
    }

    def _collect_rollout_side_effect():
        trainer._total_steps += 256  # advance step counter

    trainer.collect_rollout.side_effect = _collect_rollout_side_effect
    trainer.set_red_policy = MagicMock()
    trainer._save_checkpoint = MagicMock()
    return trainer


def _make_league_trainer(
    tmp_dir: str,
    trainer: Optional[MagicMock] = None,
    agent_id: str = "agent_001",
    snapshot_freq: int = 1000,
    eval_freq: int = 500,
    n_eval_episodes: int = 5,
    pfsp_temperature: float = 1.0,
) -> MainAgentTrainer:
    pool = _make_pool(tmp_dir)
    db = _make_db(tmp_dir)
    matchmaker = _make_matchmaker(pool, db)
    elo = _make_elo(tmp_dir)
    snap_dir = Path(tmp_dir) / "snapshots"
    snap_dir.mkdir()
    if trainer is None:
        trainer = _make_trainer_mock()
    return MainAgentTrainer(
        trainer=trainer,
        agent_pool=pool,
        match_database=db,
        matchmaker=matchmaker,
        elo_registry=elo,
        agent_id=agent_id,
        snapshot_dir=snap_dir,
        snapshot_freq=snapshot_freq,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        pfsp_temperature=pfsp_temperature,
        log_interval=500,
    )


# ---------------------------------------------------------------------------
# make_pfsp_weight_fn
# ---------------------------------------------------------------------------


class TestMakePfspWeightFn(unittest.TestCase):
    """Tests for make_pfsp_weight_fn factory."""

    def test_temperature_one_matches_hard_first(self) -> None:
        """T=1 should give (1-w) which is the standard hard-first function."""
        fn = make_pfsp_weight_fn(1.0)
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertAlmostEqual(fn(w), max(1.0 - w, 0.0), places=8)

    def test_temperature_high_approaches_uniform(self) -> None:
        """Very high T should produce nearly equal weights across win rates."""
        fn = make_pfsp_weight_fn(1000.0)
        # At T=1000 all reasonable win rates should give ≈1.0 weight
        w_low = fn(0.1)
        w_high = fn(0.9)
        self.assertAlmostEqual(w_low, w_high, places=1)

    def test_temperature_low_concentrates_on_hard(self) -> None:
        """Very low T should give near-zero weight to easy opponents."""
        fn = make_pfsp_weight_fn(0.1)
        # At T=0.1 a 0.9 win-rate (easy) gets very low weight vs 0.1 win-rate
        self.assertGreater(fn(0.1), fn(0.9) * 10)

    def test_win_rate_one_gives_zero_weight(self) -> None:
        """win_rate=1.0 should give weight 0 (agent already dominates)."""
        fn = make_pfsp_weight_fn(1.0)
        self.assertAlmostEqual(fn(1.0), 0.0, places=8)

    def test_non_positive_temperature_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_pfsp_weight_fn(0.0)
        with self.assertRaises(ValueError):
            make_pfsp_weight_fn(-1.0)

    def test_returns_callable(self) -> None:
        fn = make_pfsp_weight_fn(1.5)
        self.assertTrue(callable(fn))
        self.assertIsInstance(fn(0.5), float)


# ---------------------------------------------------------------------------
# MainAgentTrainer construction
# ---------------------------------------------------------------------------


class TestMainAgentTrainerInit(unittest.TestCase):
    """Tests for MainAgentTrainer construction."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_construction_succeeds(self) -> None:
        trainer = _make_league_trainer(self._tmpdir.name)
        self.assertIsInstance(trainer, MainAgentTrainer)

    def test_pfsp_weight_fn_is_applied(self) -> None:
        """Constructor should replace the matchmaker's weight function."""
        t = _make_league_trainer(self._tmpdir.name, pfsp_temperature=2.0)
        fn = t._matchmaker._pfsp_weight_fn
        self.assertIsNotNone(fn)
        # At T=2 weight(0.5) = (1-0.5)^(1/2) = 0.5^0.5 ≈ 0.707
        self.assertAlmostEqual(fn(0.5), 0.5 ** 0.5, places=6)

    def test_snapshot_dir_created(self) -> None:
        snap_dir = Path(self._tmpdir.name) / "snaps"
        pool = _make_pool(self._tmpdir.name)
        db = _make_db(self._tmpdir.name)
        mm = _make_matchmaker(pool, db)
        elo = _make_elo(self._tmpdir.name)
        trainer_mock = _make_trainer_mock()
        t = MainAgentTrainer(
            trainer=trainer_mock,
            agent_pool=pool,
            match_database=db,
            matchmaker=mm,
            elo_registry=elo,
            agent_id="agent_x",
            snapshot_dir=snap_dir,
        )
        self.assertTrue(snap_dir.exists())

    def test_invalid_total_timesteps_raises(self) -> None:
        t = _make_league_trainer(self._tmpdir.name)
        with self.assertRaises(ValueError):
            t.train(total_timesteps=0)


# ---------------------------------------------------------------------------
# Snapshot saving
# ---------------------------------------------------------------------------


class TestSnapshotSaving(unittest.TestCase):
    """Tests for policy snapshot persistence."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_snapshot_creates_file(self) -> None:
        """_save_snapshot should create a .pt file."""
        import torch

        t = _make_league_trainer(self._tmpdir.name)

        # Provide a real state_dict so torch.save works
        t._trainer.policy.state_dict.return_value = {"w": torch.zeros(3)}

        path = t._save_snapshot(version=1)
        self.assertTrue(path.exists())
        self.assertTrue(str(path).endswith(".pt"))

    def test_snapshot_filename_contains_version(self) -> None:
        import torch

        t = _make_league_trainer(self._tmpdir.name)
        t._trainer.policy.state_dict.return_value = {}
        path = t._save_snapshot(version=7)
        self.assertIn("000007", path.name)

    def test_initial_snapshot_registered_in_pool(self) -> None:
        import torch

        t = _make_league_trainer(self._tmpdir.name, agent_id="main_a")
        t._trainer.policy.state_dict.return_value = {}
        t._ensure_initial_snapshot()
        self.assertIn("main_a", t._agent_pool)

    def test_initial_snapshot_not_duplicated(self) -> None:
        """Calling _ensure_initial_snapshot twice should not raise."""
        import torch

        t = _make_league_trainer(self._tmpdir.name, agent_id="main_b")
        t._trainer.policy.state_dict.return_value = {}
        t._ensure_initial_snapshot()
        # Should silently skip if already registered
        t._ensure_initial_snapshot()
        self.assertEqual(t._agent_pool.size, 1)


# ---------------------------------------------------------------------------
# Opponent loading
# ---------------------------------------------------------------------------


class TestOpponentLoading(unittest.TestCase):
    """Tests for _load_opponent."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_load_opponent_returns_none_for_missing_file(self) -> None:
        t = _make_league_trainer(self._tmpdir.name)
        result = t._load_opponent(Path(self._tmpdir.name) / "nonexistent.pt")
        self.assertIsNone(result)

    def test_load_opponent_returns_none_for_corrupt_file(self) -> None:
        corrupt = Path(self._tmpdir.name) / "bad.pt"
        corrupt.write_bytes(b"not a pytorch file")
        t = _make_league_trainer(self._tmpdir.name)
        result = t._load_opponent(corrupt)
        self.assertIsNone(result)

    def test_load_valid_snapshot(self) -> None:
        """A valid snapshot saved by _save_snapshot should load back."""
        import torch
        from models.mappo_policy import MAPPOPolicy

        # Build a real (tiny) policy to snapshot.
        policy = MAPPOPolicy(
            obs_dim=10,
            action_dim=3,
            state_dim=8,
            n_agents=2,
            share_parameters=True,
            actor_hidden_sizes=(16,),
            critic_hidden_sizes=(16,),
        )
        snap_path = Path(self._tmpdir.name) / "snap_v1.pt"
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "kwargs": {
                    "obs_dim": policy.obs_dim,
                    "action_dim": policy.action_dim,
                    "state_dim": policy.state_dim,
                    "n_agents": policy.n_agents,
                    "share_parameters": policy.share_parameters,
                    "actor_hidden_sizes": policy.actor_hidden_sizes,
                    "critic_hidden_sizes": policy.critic_hidden_sizes,
                },
            },
            snap_path,
        )

        t = _make_league_trainer(self._tmpdir.name)
        loaded = t._load_opponent(snap_path)
        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded, MAPPOPolicy)


# ---------------------------------------------------------------------------
# Training loop (mocked)
# ---------------------------------------------------------------------------


class TestTrainLoop(unittest.TestCase):
    """Tests for the MainAgentTrainer.train() loop using a mocked trainer."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_train_calls_collect_rollout_and_update_policy(self) -> None:
        """train() must call collect_rollout and update_policy at least once."""
        trainer_mock = _make_trainer_mock(total_steps=0)
        t = _make_league_trainer(
            self._tmpdir.name,
            trainer=trainer_mock,
            snapshot_freq=10000,
            eval_freq=10000,
        )
        with patch.object(t, "_save_snapshot", return_value=Path(self._tmpdir.name) / "s.pt"):
            with patch.object(t, "_ensure_initial_snapshot"):
                # Run for just a few rollouts
                target = 1024
                t.train(total_timesteps=target)
        self.assertGreater(trainer_mock.collect_rollout.call_count, 0)
        self.assertGreater(trainer_mock.update_policy.call_count, 0)

    def test_snapshot_added_to_pool(self) -> None:
        """A snapshot should be added to the pool every snapshot_freq steps."""
        import torch

        trainer_mock = _make_trainer_mock(total_steps=0)
        t = _make_league_trainer(
            self._tmpdir.name,
            trainer=trainer_mock,
            snapshot_freq=512,  # one snapshot per ~512 steps
            eval_freq=100000,   # disable eval
        )
        trainer_mock.policy.state_dict.return_value = {}

        # Disable opponent loading (no real policies in pool)
        with patch.object(t, "_load_opponent", return_value=None):
            with patch.object(t, "_ensure_initial_snapshot"):
                t.train(total_timesteps=1024)

        # Should have at least one periodic snapshot plus possibly more
        self.assertGreater(t._snapshot_version, 0)

    def test_checkpoint_saved_when_dir_provided(self) -> None:
        """_save_checkpoint should be called when checkpoint_dir is configured."""
        import torch

        trainer_mock = _make_trainer_mock(total_steps=0)
        ckpt_dir = Path(self._tmpdir.name) / "ckpts"
        pool = _make_pool(self._tmpdir.name)
        db = _make_db(self._tmpdir.name)
        mm = _make_matchmaker(pool, db)
        elo = _make_elo(self._tmpdir.name)
        snap_dir = Path(self._tmpdir.name) / "snaps"
        snap_dir.mkdir()
        trainer_mock.policy.state_dict.return_value = {}

        t = MainAgentTrainer(
            trainer=trainer_mock,
            agent_pool=pool,
            match_database=db,
            matchmaker=mm,
            elo_registry=elo,
            agent_id="agent_ckpt",
            snapshot_dir=snap_dir,
            snapshot_freq=100000,
            eval_freq=100000,
            checkpoint_dir=ckpt_dir,
            checkpoint_freq=512,
        )

        with patch.object(t, "_load_opponent", return_value=None):
            with patch.object(t, "_ensure_initial_snapshot"):
                t.train(total_timesteps=1024)

        # Final checkpoint is always saved
        trainer_mock._save_checkpoint.assert_called()

    def test_eval_records_match_result(self) -> None:
        """An eval round should record an entry in the MatchDatabase."""
        import torch
        from models.mappo_policy import MAPPOPolicy

        trainer_mock = _make_trainer_mock(total_steps=0)
        t = _make_league_trainer(
            self._tmpdir.name,
            trainer=trainer_mock,
            snapshot_freq=100000,
            eval_freq=512,
            n_eval_episodes=1,
        )
        trainer_mock.policy.state_dict.return_value = {}

        # Add a real policy snapshot as a "current opponent"
        policy = MAPPOPolicy(
            obs_dim=10,
            action_dim=3,
            state_dim=8,
            n_agents=2,
            share_parameters=True,
            actor_hidden_sizes=(16,),
            critic_hidden_sizes=(16,),
        )
        snap_path = Path(self._tmpdir.name) / "opp.pt"
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "kwargs": {
                    "obs_dim": policy.obs_dim,
                    "action_dim": policy.action_dim,
                    "state_dim": policy.state_dim,
                    "n_agents": policy.n_agents,
                    "share_parameters": policy.share_parameters,
                    "actor_hidden_sizes": policy.actor_hidden_sizes,
                    "critic_hidden_sizes": policy.critic_hidden_sizes,
                },
            },
            snap_path,
        )
        t._agent_pool.add(snap_path, AgentType.MAIN_AGENT, agent_id="opp_001")
        t._current_opponent_id = "opp_001"

        # Patch eval to avoid running the full env
        with patch.object(t, "_evaluate_vs_opponent", return_value=0.6):
            with patch.object(t, "_ensure_initial_snapshot"):
                with patch.object(t, "_refresh_opponent"):
                    t.train(total_timesteps=1024)

        results = t._match_db.results_for(t.agent_id)
        self.assertGreater(len(results), 0)

    def test_elo_updated_after_eval(self) -> None:
        """Elo registry should be updated after an evaluation round."""
        trainer_mock = _make_trainer_mock(total_steps=0)
        t = _make_league_trainer(
            self._tmpdir.name,
            trainer=trainer_mock,
            snapshot_freq=100000,
            eval_freq=512,
            n_eval_episodes=1,
        )
        trainer_mock.policy.state_dict.return_value = {}

        # Add opponent to pool
        snap_path = _dummy_path(self._tmpdir.name, "opp2.pt")
        t._agent_pool.add(snap_path, AgentType.MAIN_AGENT, agent_id="opp_002")
        t._current_opponent_id = "opp_002"

        initial_rating = t._elo.get_rating(t.agent_id)

        with patch.object(t, "_evaluate_vs_opponent", return_value=1.0):
            t._run_evaluation(total_steps=512)

        updated_rating = t._elo.get_rating(t.agent_id)
        # Win → Elo should increase
        self.assertGreater(updated_rating, initial_rating)

    def test_matchup_win_rates_accumulated(self) -> None:
        """_matchup_outcomes should accumulate per-opponent outcomes."""
        trainer_mock = _make_trainer_mock(total_steps=0)
        t = _make_league_trainer(self._tmpdir.name, trainer=trainer_mock)

        snap_path = _dummy_path(self._tmpdir.name, "opp3.pt")
        t._agent_pool.add(snap_path, AgentType.MAIN_AGENT, agent_id="opp_003")
        t._current_opponent_id = "opp_003"

        with patch.object(t, "_evaluate_vs_opponent", return_value=0.7):
            t._run_evaluation(total_steps=100)
        with patch.object(t, "_evaluate_vs_opponent", return_value=0.8):
            t._run_evaluation(total_steps=200)

        outcomes = t._matchup_outcomes.get("opp_003", [])
        self.assertEqual(outcomes, [0.7, 0.8])


# ---------------------------------------------------------------------------
# PFSP temperature integration
# ---------------------------------------------------------------------------


class TestPfspTemperature(unittest.TestCase):
    """Integration tests verifying temperature affects opponent sampling."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_high_temperature_near_uniform(self) -> None:
        """With T=100 sampling should be nearly uniform over opponents."""
        pool = _make_pool(self._tmpdir.name)
        db = _make_db(self._tmpdir.name)
        mm = _make_matchmaker(pool, db)
        elo = _make_elo(self._tmpdir.name)
        snap_dir = Path(self._tmpdir.name) / "snaps"
        snap_dir.mkdir()
        trainer_mock = _make_trainer_mock()

        t = MainAgentTrainer(
            trainer=trainer_mock,
            agent_pool=pool,
            match_database=db,
            matchmaker=mm,
            elo_registry=elo,
            agent_id="focal",
            snapshot_dir=snap_dir,
            pfsp_temperature=100.0,
        )

        # Add two opponents with very different win rates
        p1 = _dummy_path(self._tmpdir.name, "p1.pt")
        p2 = _dummy_path(self._tmpdir.name, "p2.pt")
        pool.add(p1, AgentType.MAIN_AGENT, agent_id="focal")  # focal agent itself
        pool.add(p1, AgentType.MAIN_AGENT, agent_id="easy_opp")
        pool.add(p2, AgentType.MAIN_AGENT, agent_id="hard_opp")

        # Record history: focal wins 90% vs easy, 10% vs hard
        for _ in range(10):
            db.record("focal", "easy_opp", outcome=0.9)
        for _ in range(10):
            db.record("focal", "hard_opp", outcome=0.1)

        # With T=100 the probabilities should be nearly equal
        probs = mm.opponent_probabilities("focal")
        if "easy_opp" in probs and "hard_opp" in probs:
            ratio = probs["hard_opp"] / probs["easy_opp"]
            # Near-uniform → ratio should be between 0.5 and 2.0
            self.assertGreater(ratio, 0.5)
            self.assertLess(ratio, 2.0)

    def test_low_temperature_hard_first(self) -> None:
        """With T=0.1 sampling should heavily favour the hardest opponent."""
        pool = _make_pool(self._tmpdir.name)
        db = _make_db(self._tmpdir.name)
        mm = _make_matchmaker(pool, db)
        elo = _make_elo(self._tmpdir.name)
        snap_dir = Path(self._tmpdir.name) / "snaps2"
        snap_dir.mkdir()
        trainer_mock = _make_trainer_mock()

        t = MainAgentTrainer(
            trainer=trainer_mock,
            agent_pool=pool,
            match_database=db,
            matchmaker=mm,
            elo_registry=elo,
            agent_id="focal2",
            snapshot_dir=snap_dir,
            pfsp_temperature=0.1,
        )

        p1 = _dummy_path(self._tmpdir.name, "q1.pt")
        p2 = _dummy_path(self._tmpdir.name, "q2.pt")
        pool.add(p1, AgentType.MAIN_AGENT, agent_id="focal2")
        pool.add(p1, AgentType.MAIN_AGENT, agent_id="easy2")
        pool.add(p2, AgentType.MAIN_AGENT, agent_id="hard2")

        for _ in range(10):
            db.record("focal2", "easy2", outcome=0.9)
        for _ in range(10):
            db.record("focal2", "hard2", outcome=0.1)

        probs = mm.opponent_probabilities("focal2")
        if "easy2" in probs and "hard2" in probs:
            # Low T → hard opponent should be sampled much more
            self.assertGreater(probs["hard2"], probs["easy2"])


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports(unittest.TestCase):
    """Verify that the public API is exported correctly."""

    def test_main_agent_trainer_exported(self) -> None:
        from training.league import MainAgentTrainer as MAT
        self.assertIs(MAT, MainAgentTrainer)

    def test_make_pfsp_weight_fn_exported(self) -> None:
        from training.league import make_pfsp_weight_fn as mpwf
        self.assertIs(mpwf, make_pfsp_weight_fn)


if __name__ == "__main__":
    unittest.main()
