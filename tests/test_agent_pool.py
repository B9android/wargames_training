# tests/test_agent_pool.py
"""Tests for training/league/agent_pool.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.league.agent_pool import (
    AgentPool,
    AgentRecord,
    AgentType,
    _default_pfsp_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(tmp_dir: str, max_size: int = 50) -> AgentPool:
    manifest = Path(tmp_dir) / "pool_manifest.json"
    return AgentPool(pool_manifest=manifest, max_size=max_size)


def _dummy_path(tmp_dir: str, name: str = "snapshot.pt") -> Path:
    p = Path(tmp_dir) / name
    p.write_bytes(b"")  # create empty file as placeholder
    return p


# ---------------------------------------------------------------------------
# AgentType
# ---------------------------------------------------------------------------


class TestAgentType(unittest.TestCase):
    """Tests for the AgentType enum."""

    def test_values_are_strings(self) -> None:
        self.assertEqual(AgentType.MAIN_AGENT.value, "main_agent")
        self.assertEqual(AgentType.MAIN_EXPLOITER.value, "main_exploiter")
        self.assertEqual(AgentType.LEAGUE_EXPLOITER.value, "league_exploiter")

    def test_from_str_exact(self) -> None:
        self.assertEqual(AgentType.from_str("main_agent"), AgentType.MAIN_AGENT)
        self.assertEqual(AgentType.from_str("main_exploiter"), AgentType.MAIN_EXPLOITER)
        self.assertEqual(AgentType.from_str("league_exploiter"), AgentType.LEAGUE_EXPLOITER)

    def test_from_str_case_insensitive(self) -> None:
        self.assertEqual(AgentType.from_str("MAIN_AGENT"), AgentType.MAIN_AGENT)
        self.assertEqual(AgentType.from_str("Main_Exploiter"), AgentType.MAIN_EXPLOITER)

    def test_from_str_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            AgentType.from_str("unknown_type")


# ---------------------------------------------------------------------------
# AgentRecord
# ---------------------------------------------------------------------------


class TestAgentRecord(unittest.TestCase):
    """Tests for AgentRecord construction and serialisation."""

    def test_construction(self) -> None:
        r = AgentRecord(
            agent_id="abc",
            agent_type=AgentType.MAIN_AGENT,
            version=3,
            snapshot_path="/tmp/snap.pt",
        )
        self.assertEqual(r.agent_id, "abc")
        self.assertEqual(r.agent_type, AgentType.MAIN_AGENT)
        self.assertEqual(r.version, 3)
        self.assertEqual(r.snapshot_path, Path("/tmp/snap.pt"))
        self.assertIsInstance(r.created_at, float)
        self.assertEqual(r.metadata, {})

    def test_round_trip_serialisation(self) -> None:
        r = AgentRecord(
            agent_id="xyz",
            agent_type=AgentType.LEAGUE_EXPLOITER,
            version=7,
            snapshot_path="/tmp/le.pt",
            metadata={"run_id": "wandb-123"},
        )
        data = r.to_dict()
        r2 = AgentRecord.from_dict(data)
        self.assertEqual(r2.agent_id, r.agent_id)
        self.assertEqual(r2.agent_type, r.agent_type)
        self.assertEqual(r2.version, r.version)
        self.assertEqual(str(r2.snapshot_path), str(r.snapshot_path))
        self.assertEqual(r2.metadata, r.metadata)

    def test_equality_by_agent_id(self) -> None:
        r1 = AgentRecord("id1", AgentType.MAIN_AGENT, 1, "/p")
        r2 = AgentRecord("id1", AgentType.MAIN_AGENT, 2, "/q")
        r3 = AgentRecord("id2", AgentType.MAIN_AGENT, 1, "/p")
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)

    def test_hashable(self) -> None:
        r = AgentRecord("id1", AgentType.MAIN_AGENT, 1, "/p")
        s = {r}
        self.assertIn(r, s)


# ---------------------------------------------------------------------------
# AgentPool — basic operations
# ---------------------------------------------------------------------------


class TestAgentPoolBasic(unittest.TestCase):
    """Basic add/remove/get/list tests for AgentPool."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _pool(self, max_size: int = 50) -> AgentPool:
        return _make_pool(self._tmpdir.name, max_size=max_size)

    def test_empty_pool_size(self) -> None:
        pool = self._pool()
        self.assertEqual(pool.size, 0)
        self.assertEqual(len(pool), 0)

    def test_add_single_record(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        rec = pool.add(path, AgentType.MAIN_AGENT, version=1)
        self.assertEqual(pool.size, 1)
        self.assertIsInstance(rec, AgentRecord)
        self.assertEqual(rec.agent_type, AgentType.MAIN_AGENT)

    def test_add_returns_correct_record(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        rec = pool.add(path, agent_type="main_exploiter", version=5, metadata={"x": 1})
        self.assertEqual(rec.agent_type, AgentType.MAIN_EXPLOITER)
        self.assertEqual(rec.version, 5)
        self.assertEqual(rec.metadata, {"x": 1})

    def test_add_explicit_agent_id(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        rec = pool.add(path, agent_id="my-agent-001")
        self.assertEqual(rec.agent_id, "my-agent-001")
        self.assertIn("my-agent-001", pool)

    def test_add_duplicate_id_raises(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        pool.add(path, agent_id="dup")
        with self.assertRaises(ValueError):
            pool.add(path, agent_id="dup")

    def test_remove_existing_record(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        rec = pool.add(path, agent_id="to-remove")
        self.assertEqual(pool.size, 1)
        pool.remove("to-remove")
        self.assertEqual(pool.size, 0)
        self.assertNotIn("to-remove", pool)

    def test_remove_nonexistent_raises(self) -> None:
        pool = self._pool()
        with self.assertRaises(KeyError):
            pool.remove("ghost")

    def test_get_existing(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        rec = pool.add(path, agent_id="agent-a")
        fetched = pool.get("agent-a")
        self.assertEqual(fetched.agent_id, "agent-a")
        self.assertIs(fetched, rec)

    def test_get_nonexistent_raises(self) -> None:
        pool = self._pool()
        with self.assertRaises(KeyError):
            pool.get("ghost")

    def test_list_all(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        pool.add(path, AgentType.MAIN_AGENT, agent_id="a1")
        pool.add(path, AgentType.MAIN_EXPLOITER, agent_id="a2")
        pool.add(path, AgentType.LEAGUE_EXPLOITER, agent_id="a3")
        self.assertEqual(len(pool.list()), 3)

    def test_list_filtered_by_type(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        pool.add(path, AgentType.MAIN_AGENT, agent_id="main1")
        pool.add(path, AgentType.MAIN_AGENT, agent_id="main2")
        pool.add(path, AgentType.MAIN_EXPLOITER, agent_id="exp1")
        mains = pool.list(agent_type=AgentType.MAIN_AGENT)
        self.assertEqual(len(mains), 2)
        exploiters = pool.list(agent_type=AgentType.MAIN_EXPLOITER)
        self.assertEqual(len(exploiters), 1)

    def test_contains_operator(self) -> None:
        pool = self._pool()
        path = _dummy_path(self._tmpdir.name)
        pool.add(path, agent_id="here")
        self.assertIn("here", pool)
        self.assertNotIn("not-here", pool)


# ---------------------------------------------------------------------------
# AgentPool — capacity / eviction
# ---------------------------------------------------------------------------


class TestAgentPoolCapacity(unittest.TestCase):
    """Test capacity enforcement and force-eviction."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _pool(self, max_size: int) -> AgentPool:
        return _make_pool(self._tmpdir.name, max_size=max_size)

    def test_max_size_enforced(self) -> None:
        pool = self._pool(max_size=3)
        path = _dummy_path(self._tmpdir.name)
        for i in range(3):
            pool.add(path, agent_id=f"a{i}")
        with self.assertRaises(RuntimeError):
            pool.add(path, agent_id="overflow")

    def test_force_evicts_oldest(self) -> None:
        pool = self._pool(max_size=3)
        path = _dummy_path(self._tmpdir.name)
        for i in range(3):
            pool.add(path, agent_id=f"a{i}")
        pool.add(path, agent_id="new", force=True)
        # Oldest (a0) should be gone.
        self.assertNotIn("a0", pool)
        self.assertIn("new", pool)
        self.assertEqual(pool.size, 3)

    def test_force_evicts_multiple_when_oversized(self) -> None:
        """Reloading a manifest created with a larger max_size and then force-adding
        must evict as many records as needed to stay at max_size."""
        # Write a 5-entry manifest, then re-open with max_size=3 and force-add.
        manifest = Path(self._tmpdir.name) / "ovr_manifest.json"
        path = _dummy_path(self._tmpdir.name)
        pool_large = AgentPool(manifest, max_size=5)
        for i in range(5):
            pool_large.add(path, agent_id=f"old-{i}")

        # Re-open with smaller max_size (simulates config change).
        pool_small = AgentPool(manifest, max_size=3)
        # Pool loaded 5 records; adding with force=True must evict until < 3 and
        # then insert, leaving exactly 3.
        pool_small.add(path, agent_id="new-entry", force=True)
        self.assertEqual(pool_small.size, 3)
        self.assertIn("new-entry", pool_small)

    def test_pool_supports_50_concurrent_snapshots(self) -> None:
        """Pool must handle ≥ 50 agents without errors."""
        pool = self._pool(max_size=50)
        path = _dummy_path(self._tmpdir.name)
        for i in range(50):
            pool.add(path, agent_id=f"agent-{i:03d}")
        self.assertEqual(pool.size, 50)

    def test_max_size_one(self) -> None:
        pool = self._pool(max_size=1)
        path = _dummy_path(self._tmpdir.name)
        pool.add(path, agent_id="first")
        pool.add(path, agent_id="second", force=True)
        self.assertEqual(pool.size, 1)
        self.assertIn("second", pool)
        self.assertNotIn("first", pool)

    def test_invalid_max_size_raises(self) -> None:
        manifest = Path(self._tmpdir.name) / "m.json"
        with self.assertRaises(ValueError):
            AgentPool(pool_manifest=manifest, max_size=0)


# ---------------------------------------------------------------------------
# AgentPool — persistence
# ---------------------------------------------------------------------------


class TestAgentPoolPersistence(unittest.TestCase):
    """Test that the pool survives a process restart (reload from disk)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_reload_from_manifest(self) -> None:
        manifest = Path(self._tmpdir.name) / "manifest.json"
        path = _dummy_path(self._tmpdir.name)

        pool1 = AgentPool(manifest, max_size=10)
        pool1.add(path, AgentType.MAIN_AGENT, agent_id="persist-me", version=3)

        # Simulate restart.
        pool2 = AgentPool(manifest, max_size=10)
        self.assertIn("persist-me", pool2)
        rec = pool2.get("persist-me")
        self.assertEqual(rec.version, 3)
        self.assertEqual(rec.agent_type, AgentType.MAIN_AGENT)

    def test_remove_persisted_after_reload(self) -> None:
        manifest = Path(self._tmpdir.name) / "manifest.json"
        path = _dummy_path(self._tmpdir.name)

        pool1 = AgentPool(manifest, max_size=10)
        pool1.add(path, agent_id="keep")
        pool1.add(path, agent_id="drop")
        pool1.remove("drop")

        pool2 = AgentPool(manifest, max_size=10)
        self.assertIn("keep", pool2)
        self.assertNotIn("drop", pool2)

    def test_empty_manifest_on_fresh_start(self) -> None:
        manifest = Path(self._tmpdir.name) / "new_manifest.json"
        pool = AgentPool(manifest, max_size=10)
        self.assertEqual(pool.size, 0)
        # Manifest file should not yet exist before first add.
        self.assertFalse(manifest.exists())


# ---------------------------------------------------------------------------
# AgentPool — PFSP sampling
# ---------------------------------------------------------------------------


class TestAgentPoolPFSP(unittest.TestCase):
    """Tests for PFSP opponent sampling."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.pool = _make_pool(self._tmpdir.name)
        path = _dummy_path(self._tmpdir.name)
        # Add 5 main agents with known IDs.
        for i in range(5):
            self.pool.add(path, AgentType.MAIN_AGENT, agent_id=f"main-{i}")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_sample_pfsp_returns_a_record(self) -> None:
        rec = self.pool.sample_pfsp()
        self.assertIsInstance(rec, AgentRecord)

    def test_sample_pfsp_excludes_focal(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            rec = self.pool.sample_pfsp(
                win_rates={"main-0": 0.9},
                exclude_ids=["main-0"],
                rng=rng,
            )
            self.assertIsNotNone(rec)
            self.assertNotEqual(rec.agent_id, "main-0")  # type: ignore[union-attr]

    def test_pfsp_hard_first_biases_harder_opponents(self) -> None:
        """Opponents with lower win rates should be sampled more often."""
        rng = np.random.default_rng(0)
        # main-0: win_rate=0.9 (easy) → weight 0.1
        # main-1: win_rate=0.1 (hard) → weight 0.9
        win_rates = {"main-0": 0.9, "main-1": 0.1}
        counts: Dict[str, int] = {f"main-{i}": 0 for i in range(5)}
        n = 2000
        for _ in range(n):
            rec = self.pool.sample_pfsp(
                win_rates=win_rates,
                exclude_ids=["main-0"],
                rng=rng,
            )
            counts[rec.agent_id] += 1  # type: ignore[union-attr]
        # main-1 should be sampled far more than main-2/3/4 (unknown → 0.5 → weight 0.5).
        self.assertGreater(counts["main-1"], counts["main-2"])

    def test_pfsp_all_zero_weights_falls_back_to_uniform(self) -> None:
        """If all win rates are 1.0 (all weights zero) → uniform fallback."""
        rng = np.random.default_rng(7)
        win_rates = {f"main-{i}": 1.0 for i in range(5)}
        # focal = main-0, exclude it
        counts: Dict[str, int] = {f"main-{i}": 0 for i in range(1, 5)}
        n = 2000
        for _ in range(n):
            rec = self.pool.sample_pfsp(
                win_rates=win_rates,
                exclude_ids=["main-0"],
                rng=rng,
            )
            counts[rec.agent_id] += 1  # type: ignore[union-attr]
        # All should be sampled roughly equally.
        expected = n / 4
        for aid, cnt in counts.items():
            self.assertAlmostEqual(cnt / n, 0.25, delta=0.05, msg=f"agent {aid}")

    def test_sample_pfsp_filter_by_type(self) -> None:
        path = _dummy_path(self._tmpdir.name)
        self.pool.add(path, AgentType.MAIN_EXPLOITER, agent_id="exp-0")
        rng = np.random.default_rng(1)
        for _ in range(20):
            rec = self.pool.sample_pfsp(agent_type=AgentType.MAIN_EXPLOITER, rng=rng)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.agent_type, AgentType.MAIN_EXPLOITER)  # type: ignore[union-attr]

    def test_sample_pfsp_empty_pool_returns_none(self) -> None:
        empty_pool = _make_pool(self._tmpdir.name + "/sub")
        rec = empty_pool.sample_pfsp()
        self.assertIsNone(rec)

    def test_sample_uniform(self) -> None:
        rng = np.random.default_rng(42)
        counts: Dict[str, int] = {f"main-{i}": 0 for i in range(5)}
        n = 2000
        for _ in range(n):
            rec = self.pool.sample_uniform(rng=rng)
            counts[rec.agent_id] += 1  # type: ignore[union-attr]
        for aid, cnt in counts.items():
            self.assertAlmostEqual(cnt / n, 0.2, delta=0.05, msg=f"agent {aid}")

    def test_custom_pfsp_weight_fn(self) -> None:
        """Custom weight fn: easy opponents (high win rate) are preferred."""
        rng = np.random.default_rng(99)
        win_rates = {"main-1": 0.9, "main-2": 0.1}
        # Easy-first: f(w) = w
        rec = self.pool.sample_pfsp(
            win_rates=win_rates,
            pfsp_weight_fn=lambda w: w,
            rng=rng,
        )
        self.assertIsNotNone(rec)


# ---------------------------------------------------------------------------
# Default PFSP weight function
# ---------------------------------------------------------------------------


class TestDefaultPFSPWeight(unittest.TestCase):
    """Unit tests for _default_pfsp_weight."""

    def test_zero_win_rate_gives_max_weight(self) -> None:
        self.assertAlmostEqual(_default_pfsp_weight(0.0), 1.0)

    def test_full_win_rate_gives_zero_weight(self) -> None:
        self.assertAlmostEqual(_default_pfsp_weight(1.0), 0.0)

    def test_half_win_rate_gives_half_weight(self) -> None:
        self.assertAlmostEqual(_default_pfsp_weight(0.5), 0.5)

    def test_monotone_decreasing(self) -> None:
        for w in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            prev = _default_pfsp_weight(w)
            nxt = _default_pfsp_weight(min(w + 0.2, 1.0))
            self.assertGreaterEqual(prev, nxt)


if __name__ == "__main__":
    unittest.main()
