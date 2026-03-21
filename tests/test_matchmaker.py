# tests/test_matchmaker.py
"""Tests for training/league/matchmaker.py and training/league/match_database.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.league.agent_pool import AgentPool, AgentRecord, AgentType
from training.league.match_database import MatchDatabase, MatchResult
from training.league.matchmaker import LeagueMatchmaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_dir: str, max_size: int = 100) -> AgentPool:
    manifest = Path(tmp_dir) / "pool.json"
    return AgentPool(pool_manifest=manifest, max_size=max_size)


def _make_db(tmp_dir: str, name: str = "matches.jsonl") -> MatchDatabase:
    return MatchDatabase(Path(tmp_dir) / name)


def _dummy_path(tmp_dir: str, name: str = "snap.pt") -> Path:
    p = Path(tmp_dir) / name
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


class TestMatchResult(unittest.TestCase):
    """Unit tests for MatchResult."""

    def test_construction(self) -> None:
        r = MatchResult("a", "b", 0.7)
        self.assertEqual(r.agent_id, "a")
        self.assertEqual(r.opponent_id, "b")
        self.assertAlmostEqual(r.outcome, 0.7)
        self.assertIsInstance(r.match_id, str)
        self.assertIsInstance(r.timestamp, float)

    def test_outcome_boundary_values(self) -> None:
        MatchResult("a", "b", 0.0)
        MatchResult("a", "b", 1.0)
        MatchResult("a", "b", 0.5)

    def test_invalid_outcome_raises(self) -> None:
        with self.assertRaises(ValueError):
            MatchResult("a", "b", -0.1)
        with self.assertRaises(ValueError):
            MatchResult("a", "b", 1.1)

    def test_round_trip_serialisation(self) -> None:
        r = MatchResult("a", "b", 0.6, metadata={"map": "hill"})
        r2 = MatchResult.from_dict(r.to_dict())
        self.assertEqual(r2.match_id, r.match_id)
        self.assertEqual(r2.agent_id, r.agent_id)
        self.assertEqual(r2.opponent_id, r.opponent_id)
        self.assertAlmostEqual(r2.outcome, r.outcome)
        self.assertEqual(r2.metadata, r.metadata)

    def test_equality_by_match_id(self) -> None:
        r1 = MatchResult("a", "b", 1.0, match_id="same")
        r2 = MatchResult("x", "y", 0.0, match_id="same")
        r3 = MatchResult("a", "b", 1.0, match_id="other")
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)

    def test_hashable(self) -> None:
        r = MatchResult("a", "b", 0.5, match_id="m1")
        s = {r}
        self.assertIn(r, s)


# ---------------------------------------------------------------------------
# MatchDatabase
# ---------------------------------------------------------------------------


class TestMatchDatabase(unittest.TestCase):
    """Tests for MatchDatabase persistence and queries."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_empty_database_size(self) -> None:
        db = _make_db(self._tmpdir.name)
        self.assertEqual(db.size, 0)

    def test_record_match(self) -> None:
        db = _make_db(self._tmpdir.name)
        r = db.record("a", "b", 1.0)
        self.assertIsInstance(r, MatchResult)
        self.assertEqual(db.size, 1)

    def test_win_rate_no_history(self) -> None:
        db = _make_db(self._tmpdir.name)
        wr = db.win_rate("a", "b")
        self.assertIsNone(wr)

    def test_win_rate_single_match(self) -> None:
        db = _make_db(self._tmpdir.name)
        db.record("a", "b", 1.0)
        self.assertAlmostEqual(db.win_rate("a", "b"), 1.0)

    def test_win_rate_average_over_multiple(self) -> None:
        db = _make_db(self._tmpdir.name)
        db.record("a", "b", 1.0)
        db.record("a", "b", 0.0)
        db.record("a", "b", 1.0)
        self.assertAlmostEqual(db.win_rate("a", "b"), 2.0 / 3.0)

    def test_win_rates_for_multiple_opponents(self) -> None:
        db = _make_db(self._tmpdir.name)
        db.record("focal", "opp1", 1.0)
        db.record("focal", "opp1", 0.0)
        db.record("focal", "opp2", 0.5)
        wr = db.win_rates_for("focal")
        self.assertIn("opp1", wr)
        self.assertIn("opp2", wr)
        self.assertAlmostEqual(wr["opp1"], 0.5)
        self.assertAlmostEqual(wr["opp2"], 0.5)

    def test_results_for_filters_correctly(self) -> None:
        db = _make_db(self._tmpdir.name)
        db.record("a", "b", 1.0)
        db.record("a", "c", 0.0)
        db.record("x", "b", 1.0)
        self.assertEqual(len(db.results_for("a")), 2)
        self.assertEqual(len(db.results_for("a", "b")), 1)
        self.assertEqual(len(db.results_for("x")), 1)

    def test_persistence_across_reload(self) -> None:
        """Records written in one DB instance should be readable in another."""
        db_path = Path(self._tmpdir.name) / "persistent.jsonl"
        db1 = MatchDatabase(db_path)
        db1.record("a", "b", 1.0)
        db1.record("a", "b", 0.0)

        db2 = MatchDatabase(db_path)
        self.assertEqual(db2.size, 2)
        self.assertAlmostEqual(db2.win_rate("a", "b"), 0.5)

    def test_prune_reduces_in_memory_count(self) -> None:
        db = _make_db(self._tmpdir.name)
        for i in range(10):
            db.record("a", "b", float(i % 2))
        self.assertEqual(db.size, 10)
        dropped = db.prune(keep_last=5)
        self.assertEqual(dropped, 5)
        self.assertEqual(db.size, 5)

    def test_prune_keep_last_zero(self) -> None:
        db = _make_db(self._tmpdir.name)
        for i in range(5):
            db.record("a", "b", 1.0)
        dropped = db.prune(keep_last=0)
        self.assertEqual(dropped, 5)
        self.assertEqual(db.size, 0)

    def test_prune_negative_raises(self) -> None:
        db = _make_db(self._tmpdir.name)
        db.record("a", "b", 1.0)
        with self.assertRaises(ValueError):
            db.prune(keep_last=-1)

    def test_rewrite_after_prune(self) -> None:
        db = _make_db(self._tmpdir.name)
        for i in range(10):
            db.record("a", "b", float(i % 2))
        db.prune(keep_last=3)
        db.rewrite()

        db2 = MatchDatabase(db.db_path)
        self.assertEqual(db2.size, 3)

    def test_all_results_returns_copy(self) -> None:
        db = _make_db(self._tmpdir.name)
        db.record("a", "b", 1.0)
        all_r = db.all_results()
        all_r.clear()
        self.assertEqual(db.size, 1)


# ---------------------------------------------------------------------------
# LeagueMatchmaker
# ---------------------------------------------------------------------------


class TestLeagueMatchmaker(unittest.TestCase):
    """Tests for LeagueMatchmaker PFSP opponent selection."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.pool = _make_pool(self._tmpdir.name)
        self.db = _make_db(self._tmpdir.name)
        path = _dummy_path(self._tmpdir.name)

        # Populate pool with 3 main agents, 2 main exploiters, 2 league exploiters.
        for i in range(3):
            self.pool.add(path, AgentType.MAIN_AGENT, agent_id=f"main-{i}")
        for i in range(2):
            self.pool.add(path, AgentType.MAIN_EXPLOITER, agent_id=f"mexpl-{i}")
        for i in range(2):
            self.pool.add(path, AgentType.LEAGUE_EXPLOITER, agent_id=f"lexpl-{i}")

        self.mm = LeagueMatchmaker(self.pool, self.db)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_select_opponent_returns_record(self) -> None:
        rec = self.mm.select_opponent("main-0")
        self.assertIsInstance(rec, AgentRecord)

    def test_select_opponent_excludes_focal(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            rec = self.mm.select_opponent("main-0", rng=rng)
            self.assertIsNotNone(rec)
            self.assertNotEqual(rec.agent_id, "main-0")  # type: ignore[union-attr]

    def test_main_exploiter_plays_only_main_agents(self) -> None:
        """MAIN_EXPLOITER should only receive MAIN_AGENT opponents."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            rec = self.mm.select_opponent("mexpl-0", rng=rng)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.agent_type, AgentType.MAIN_AGENT)  # type: ignore[union-attr]

    def test_league_exploiter_plays_all_types(self) -> None:
        """LEAGUE_EXPLOITER may play any agent in the pool."""
        rng = np.random.default_rng(2)
        seen_types = set()
        for _ in range(200):
            rec = self.mm.select_opponent("lexpl-0", rng=rng)
            self.assertIsNotNone(rec)
            seen_types.add(rec.agent_type)  # type: ignore[union-attr]
        # Should eventually see all three types (excluding focal lexpl-0 itself).
        expected = {AgentType.MAIN_AGENT, AgentType.MAIN_EXPLOITER, AgentType.LEAGUE_EXPLOITER}
        self.assertEqual(seen_types, expected)

    def test_pfsp_weights_harder_opponents_more(self) -> None:
        """Opponent with lower recorded win rate should be sampled more."""
        # Record that main-0 always beats main-1 but never beats main-2.
        for _ in range(10):
            self.db.record("main-0", "main-1", 1.0)
            self.db.record("main-0", "main-2", 0.0)

        rng = np.random.default_rng(42)
        counts: Dict[str, int] = {"main-1": 0, "main-2": 0}
        n = 2000
        for _ in range(n):
            rec = self.mm.select_opponent("main-0", candidate_types=[AgentType.MAIN_AGENT], rng=rng)
            if rec and rec.agent_id in counts:
                counts[rec.agent_id] += 1

        # main-2 (hard, win_rate=0 → weight=1) should be sampled more
        # than main-1 (easy, win_rate=1 → weight=0).
        # main-1 may get weight 0, so count could be 0.
        self.assertGreater(counts["main-2"], counts["main-1"])

    def test_pfsp_unknown_win_rate_default(self) -> None:
        """Opponents without history use unknown_win_rate (default 0.5)."""
        mm = LeagueMatchmaker(self.pool, self.db, unknown_win_rate=0.5)
        probs = mm.opponent_probabilities("main-0")
        self.assertGreater(len(probs), 0)
        total = sum(probs.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_opponent_probabilities_sum_to_one(self) -> None:
        probs = self.mm.opponent_probabilities("main-0")
        self.assertGreater(len(probs), 0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_opponent_probabilities_excludes_focal(self) -> None:
        probs = self.mm.opponent_probabilities("main-0")
        self.assertNotIn("main-0", probs)

    def test_select_opponent_unknown_focal_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.mm.select_opponent("does-not-exist")

    def test_select_opponent_no_candidates_returns_none(self) -> None:
        """A main exploiter pool with no main agents → no candidates."""
        pool = _make_pool(self._tmpdir.name + "/empty")
        db = _make_db(self._tmpdir.name, "empty.jsonl")
        path = _dummy_path(self._tmpdir.name)
        # Only add a main exploiter (no main agents).
        pool.add(path, AgentType.MAIN_EXPLOITER, agent_id="lone-exp")
        mm = LeagueMatchmaker(pool, db)
        rec = mm.select_opponent("lone-exp")
        self.assertIsNone(rec)

    def test_custom_pfsp_weight_fn(self) -> None:
        """Custom weight fn should be respected."""
        mm = LeagueMatchmaker(
            self.pool,
            self.db,
            pfsp_weight_fn=lambda w: w,  # easy-first
        )
        probs = mm.opponent_probabilities("main-0")
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_candidate_types_override(self) -> None:
        """Explicit candidate_types overrides default matchup rules."""
        rng = np.random.default_rng(5)
        for _ in range(30):
            rec = self.mm.select_opponent(
                "main-0",
                candidate_types=[AgentType.MAIN_EXPLOITER],
                rng=rng,
            )
            self.assertIsNotNone(rec)
            self.assertEqual(rec.agent_type, AgentType.MAIN_EXPLOITER)  # type: ignore[union-attr]

    def test_invalid_unknown_win_rate_raises(self) -> None:
        with self.assertRaises(ValueError):
            LeagueMatchmaker(self.pool, self.db, unknown_win_rate=1.5)

    def test_pfsp_distribution_proportional_to_weights(self) -> None:
        """Verify empirical sampling distribution matches PFSP probabilities."""
        # Record different win rates for main-1 and main-2.
        for _ in range(10):
            self.db.record("main-0", "main-1", 0.8)  # win_rate=0.8 → weight=0.2
            self.db.record("main-0", "main-2", 0.2)  # win_rate=0.2 → weight=0.8

        theoretical = self.mm.opponent_probabilities(
            "main-0", candidate_types=[AgentType.MAIN_AGENT]
        )
        rng = np.random.default_rng(123)
        counts: Dict[str, int] = {k: 0 for k in theoretical}
        n = 5000
        for _ in range(n):
            rec = self.mm.select_opponent(
                "main-0",
                candidate_types=[AgentType.MAIN_AGENT],
                rng=rng,
            )
            if rec and rec.agent_id in counts:
                counts[rec.agent_id] += 1

        for aid, expected_p in theoretical.items():
            empirical_p = counts.get(aid, 0) / n
            self.assertAlmostEqual(
                empirical_p,
                expected_p,
                delta=0.05,
                msg=f"Empirical vs theoretical probability for {aid}",
            )


if __name__ == "__main__":
    unittest.main()
