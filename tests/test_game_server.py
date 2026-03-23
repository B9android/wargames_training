# tests/test_game_server.py
"""Tests for server/game_server.py and server/replay.py (E9.1)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# server.replay
# ---------------------------------------------------------------------------

from server.replay import Replay, ReplayMetadata, ReplayPlayer, ReplayRecorder


class TestReplayMetadata(unittest.TestCase):
    """Validate ReplayMetadata dataclass."""

    def test_defaults(self) -> None:
        meta = ReplayMetadata(
            scenario="open_field",
            difficulty=5,
            map_width=1000.0,
            map_height=1000.0,
        )
        self.assertEqual(meta.outcome, "unknown")
        self.assertEqual(meta.total_steps, 0)
        self.assertIsInstance(meta.recorded_at, str)
        self.assertGreater(len(meta.recorded_at), 0)

    def test_custom_outcome(self) -> None:
        meta = ReplayMetadata(
            scenario="last_stand",
            difficulty=3,
            map_width=500.0,
            map_height=500.0,
            outcome="blue_wins",
            total_steps=200,
        )
        self.assertEqual(meta.outcome, "blue_wins")
        self.assertEqual(meta.total_steps, 200)


class TestReplayRecorder(unittest.TestCase):
    """Tests for ReplayRecorder."""

    def _make_recorder(self) -> ReplayRecorder:
        return ReplayRecorder(
            scenario="open_field",
            difficulty=5,
            map_width=1000.0,
            map_height=1000.0,
        )

    def test_initial_frame_count(self) -> None:
        rec = self._make_recorder()
        self.assertEqual(rec.frame_count, 0)

    def test_record_increments_count(self) -> None:
        rec = self._make_recorder()
        rec.record({"step": 0, "blue": {}, "red": {}})
        rec.record({"step": 1, "blue": {}, "red": {}})
        self.assertEqual(rec.frame_count, 2)

    def test_finish_returns_replay(self) -> None:
        rec = self._make_recorder()
        rec.record({"step": 0})
        replay = rec.finish(outcome="blue_wins")
        self.assertIsInstance(replay, Replay)
        self.assertEqual(replay.metadata.outcome, "blue_wins")
        self.assertEqual(replay.metadata.total_steps, 1)

    def test_finish_default_outcome(self) -> None:
        rec = self._make_recorder()
        replay = rec.finish()
        self.assertEqual(replay.metadata.outcome, "unknown")

    def test_frames_copied_on_finish(self) -> None:
        rec = self._make_recorder()
        frame = {"step": 0, "data": "test"}
        rec.record(frame)
        replay = rec.finish()
        self.assertEqual(len(replay.frames), 1)
        self.assertEqual(replay.frames[0]["data"], "test")

    def test_multiple_finishes_independent(self) -> None:
        rec = self._make_recorder()
        rec.record({"step": 0})
        replay1 = rec.finish(outcome="blue_wins")
        replay2 = rec.finish(outcome="red_wins")
        # Each finish captures current state.
        self.assertEqual(replay1.metadata.outcome, "blue_wins")
        self.assertEqual(replay2.metadata.outcome, "red_wins")


class TestReplay(unittest.TestCase):
    """Tests for Replay serialisation / deserialisation."""

    def _make_replay(self, n_frames: int = 3) -> Replay:
        meta = ReplayMetadata(
            scenario="open_field",
            difficulty=5,
            map_width=1000.0,
            map_height=1000.0,
            outcome="blue_wins",
            total_steps=n_frames,
        )
        frames = [{"step": i, "blue": {"x": float(i)}} for i in range(n_frames)]
        return Replay(metadata=meta, frames=frames)

    def test_to_dict_has_required_keys(self) -> None:
        replay = self._make_replay()
        d = replay.to_dict()
        self.assertIn("metadata", d)
        self.assertIn("frames", d)
        self.assertEqual(d["metadata"]["scenario"], "open_field")

    def test_to_json_is_valid_json(self) -> None:
        replay = self._make_replay()
        text = replay.to_json()
        parsed = json.loads(text)
        self.assertIn("metadata", parsed)

    def test_from_dict_roundtrip(self) -> None:
        replay = self._make_replay(n_frames=5)
        d = replay.to_dict()
        restored = Replay.from_dict(d)
        self.assertEqual(restored.metadata.scenario, "open_field")
        self.assertEqual(restored.metadata.outcome, "blue_wins")
        self.assertEqual(len(restored.frames), 5)

    def test_from_json_roundtrip(self) -> None:
        replay = self._make_replay()
        text = replay.to_json()
        restored = Replay.from_json(text)
        self.assertEqual(restored.metadata.difficulty, 5)
        self.assertEqual(len(restored.frames), 3)

    def test_from_dict_missing_metadata_raises(self) -> None:
        with self.assertRaises(KeyError):
            Replay.from_dict({"frames": []})

    def test_from_dict_minimal_metadata(self) -> None:
        d = {
            "metadata": {
                "scenario": "last_stand",
                "difficulty": 3,
                "map_width": 500.0,
                "map_height": 500.0,
            },
            "frames": [],
        }
        replay = Replay.from_dict(d)
        self.assertEqual(replay.metadata.outcome, "unknown")
        self.assertEqual(replay.metadata.total_steps, 0)

    def test_from_dict_missing_recorded_at_uses_default(self) -> None:
        d = {
            "metadata": {
                "scenario": "open_field",
                "difficulty": 5,
                "map_width": 1000.0,
                "map_height": 1000.0,
                # no "recorded_at" key
            },
            "frames": [],
        }
        replay = Replay.from_dict(d)
        # Should not be empty string — should be a valid timestamp.
        self.assertGreater(len(replay.metadata.recorded_at), 0)
        self.assertNotEqual(replay.metadata.recorded_at, "")

    def test_from_dict_preserves_recorded_at_when_provided(self) -> None:
        ts = "2026-01-01T00:00:00Z"
        d = {
            "metadata": {
                "scenario": "open_field",
                "difficulty": 5,
                "map_width": 1000.0,
                "map_height": 1000.0,
                "recorded_at": ts,
            },
            "frames": [],
        }
        replay = Replay.from_dict(d)
        self.assertEqual(replay.metadata.recorded_at, ts)

    def test_frames_not_shared_with_input(self) -> None:
        meta = ReplayMetadata(scenario="open_field", difficulty=5, map_width=1000.0, map_height=1000.0)
        original_frames = [{"step": 0}]
        replay = Replay(metadata=meta, frames=original_frames)
        d = replay.to_dict()
        # Mutating d should not affect the original.
        d["frames"].append({"step": 99})
        self.assertEqual(len(original_frames), 1)


class TestReplayPlayer(unittest.TestCase):
    """Tests for ReplayPlayer."""

    def _make_player(self, n_frames: int = 5) -> ReplayPlayer:
        meta = ReplayMetadata(scenario="open_field", difficulty=5, map_width=1000.0, map_height=1000.0)
        frames = [{"step": i} for i in range(n_frames)]
        replay = Replay(metadata=meta, frames=frames)
        return ReplayPlayer(replay)

    def test_initial_index(self) -> None:
        player = self._make_player()
        self.assertEqual(player.current_index, 0)

    def test_total_frames(self) -> None:
        player = self._make_player(n_frames=7)
        self.assertEqual(player.total_frames, 7)

    def test_step_advances_index(self) -> None:
        player = self._make_player()
        frame = player.step()
        self.assertIsNotNone(frame)
        self.assertEqual(frame["step"], 0)
        self.assertEqual(player.current_index, 1)

    def test_step_returns_none_when_done(self) -> None:
        player = self._make_player(n_frames=2)
        player.step()
        player.step()
        self.assertTrue(player.done)
        self.assertIsNone(player.step())

    def test_seek_clamps_low(self) -> None:
        player = self._make_player()
        player.seek(-5)
        self.assertEqual(player.current_index, 0)

    def test_seek_clamps_high(self) -> None:
        player = self._make_player(n_frames=5)
        player.seek(100)
        self.assertEqual(player.current_index, 5)

    def test_seek_to_middle(self) -> None:
        player = self._make_player(n_frames=10)
        player.seek(4)
        frame = player.step()
        self.assertIsNotNone(frame)
        self.assertEqual(frame["step"], 4)

    def test_iter_frames_yields_all(self) -> None:
        player = self._make_player(n_frames=4)
        frames = list(player.iter_frames())
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0]["step"], 0)
        self.assertEqual(frames[3]["step"], 3)

    def test_iter_frames_from_middle(self) -> None:
        player = self._make_player(n_frames=6)
        player.seek(3)
        frames = list(player.iter_frames())
        self.assertEqual(len(frames), 3)

    def test_metadata_accessible(self) -> None:
        player = self._make_player()
        self.assertEqual(player.metadata.scenario, "open_field")

    def test_not_done_initially(self) -> None:
        player = self._make_player()
        self.assertFalse(player.done)

    def test_done_after_all_frames(self) -> None:
        player = self._make_player(n_frames=2)
        list(player.iter_frames())
        self.assertTrue(player.done)

    def test_seek_then_done(self) -> None:
        player = self._make_player(n_frames=3)
        player.seek(3)
        self.assertTrue(player.done)


# ---------------------------------------------------------------------------
# server.game_server — unit helpers (no env required)
# ---------------------------------------------------------------------------

from server.game_server import PolicyClient, _clamp, _determine_outcome


class TestClamp(unittest.TestCase):
    def test_clamp_within(self) -> None:
        self.assertEqual(_clamp(0.5, -1.0, 1.0), 0.5)

    def test_clamp_low(self) -> None:
        self.assertEqual(_clamp(-2.0, -1.0, 1.0), -1.0)

    def test_clamp_high(self) -> None:
        self.assertEqual(_clamp(3.0, -1.0, 1.0), 1.0)

    def test_clamp_edge(self) -> None:
        self.assertEqual(_clamp(-1.0, -1.0, 1.0), -1.0)
        self.assertEqual(_clamp(1.0, -1.0, 1.0), 1.0)


class TestDetermineOutcome(unittest.TestCase):
    def test_blue_wins_when_red_routed(self) -> None:
        info = {"blue_routed": False, "red_routed": True}
        self.assertEqual(_determine_outcome(info), "blue_wins")

    def test_red_wins_when_blue_routed(self) -> None:
        info = {"blue_routed": True, "red_routed": False}
        self.assertEqual(_determine_outcome(info), "red_wins")

    def test_draw_when_neither_routed(self) -> None:
        info = {"blue_routed": False, "red_routed": False}
        self.assertEqual(_determine_outcome(info), "draw")

    def test_draw_when_both_routed(self) -> None:
        info = {"blue_routed": True, "red_routed": True}
        self.assertEqual(_determine_outcome(info), "draw")

    def test_fallback_blue_wins_via_strength(self) -> None:
        info = {"blue_strength": 0.5, "red_strength": 0.0}
        self.assertEqual(_determine_outcome(info), "blue_wins")

    def test_fallback_red_wins_via_strength(self) -> None:
        info = {"blue_strength": 0.0, "red_strength": 0.8}
        self.assertEqual(_determine_outcome(info), "red_wins")

    def test_fallback_draw_via_strength(self) -> None:
        info = {"blue_strength": 0.5, "red_strength": 0.5}
        self.assertEqual(_determine_outcome(info), "draw")

    def test_missing_keys_returns_unknown(self) -> None:
        info: dict = {}
        self.assertEqual(_determine_outcome(info), "unknown")

    def test_numpy_bool_routed_flag(self) -> None:
        import numpy as np
        info = {"blue_routed": np.bool_(False), "red_routed": np.bool_(True)}
        self.assertEqual(_determine_outcome(info), "blue_wins")


class TestPolicyClient(unittest.TestCase):
    """Tests for PolicyClient (mocked HTTP)."""

    def setUp(self) -> None:
        self.client = PolicyClient(url="http://localhost:8080", timeout=0.5)

    def test_predict_returns_zeros_on_failure(self) -> None:
        import numpy as np

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = self.client.predict(np.zeros(22, dtype=np.float32))
        self.assertEqual(result.shape, (3,))
        self.assertTrue((result == 0).all())

    def test_predict_parses_valid_response(self) -> None:
        import io
        import numpy as np

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"output": [[0.1, -0.2, 0.5]]}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.client.predict(np.zeros(22, dtype=np.float32))

        self.assertAlmostEqual(result[0], 0.1, places=5)
        self.assertAlmostEqual(result[1], -0.2, places=5)
        self.assertAlmostEqual(result[2], 0.5, places=5)

    def test_is_available_returns_false_on_failure(self) -> None:
        with patch("urllib.request.urlopen", side_effect=Exception("refused")):
            self.assertFalse(self.client.is_available())

    def test_is_available_returns_true_on_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            self.assertTrue(self.client.is_available())


class TestOnnxRedPolicy(unittest.TestCase):
    """Tests for the _OnnxRedPolicy adapter."""

    def test_predict_wraps_client(self) -> None:
        import numpy as np
        from server.game_server import PolicyClient, _OnnxRedPolicy

        client = PolicyClient(url="http://localhost:8080")
        policy = _OnnxRedPolicy(client)

        obs = np.zeros(22, dtype=np.float32)
        with patch("urllib.request.urlopen", side_effect=Exception("refused")):
            action, state = policy.predict(obs, deterministic=True)

        self.assertIsNone(state)
        # Falls back to zeros.
        self.assertTrue((action == 0).all())

    def test_predict_returns_action_on_success(self) -> None:
        import numpy as np
        from server.game_server import PolicyClient, _OnnxRedPolicy

        client = PolicyClient(url="http://localhost:8080")
        policy = _OnnxRedPolicy(client)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"output": [[0.5, -0.3, 0.9]]}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        obs = np.zeros(22, dtype=np.float32)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            action, state = policy.predict(obs)

        self.assertIsNone(state)
        self.assertAlmostEqual(float(action[0]), 0.5, places=5)


# ---------------------------------------------------------------------------
# GameSession — integration test (requires gymnasium + envs)
# ---------------------------------------------------------------------------

try:
    import gymnasium  # noqa: F401
    import numpy as np
    _GYMNASIUM_AVAILABLE = True
except ImportError:
    _GYMNASIUM_AVAILABLE = False


@unittest.skipUnless(_GYMNASIUM_AVAILABLE, "gymnasium not installed")
class TestGameSession(unittest.TestCase):
    """Integration tests for GameSession (requires full env stack)."""

    def _make_session(self, scenario: str = "open_field") -> "GameSession":
        from server.game_server import GameSession

        return GameSession(scenario=scenario, difficulty=3, policy_url=None)

    def test_reset_returns_frame(self) -> None:
        session = self._make_session()
        frame = session.reset(seed=42)
        self.assertEqual(frame["type"], "frame")
        self.assertIn("blue", frame)
        self.assertIn("red", frame)
        self.assertIn("step", frame)

    def test_step_returns_frame_and_done_flag(self) -> None:
        session = self._make_session()
        session.reset(seed=42)
        frame, done = session.step()
        self.assertEqual(frame["type"], "frame")
        self.assertIsInstance(done, bool)

    def test_apply_action_clamps_values(self) -> None:
        session = self._make_session()
        session.reset(seed=42)
        session.apply_action(move=5.0, rotate=-99.0, fire=10.0)
        import numpy as np

        action = session._human_action
        self.assertAlmostEqual(float(action[0]), 1.0)
        self.assertAlmostEqual(float(action[1]), -1.0)
        self.assertAlmostEqual(float(action[2]), 1.0)

    def test_recorder_tracks_frames(self) -> None:
        session = self._make_session()
        session.reset(seed=0)
        n_steps = 3
        for _ in range(n_steps):
            _, done = session.step()
            if done:
                break
        # At least one frame was recorded.
        self.assertGreater(session._recorder.frame_count, 0)

    def test_export_replay_none_before_episode_ends(self) -> None:
        session = self._make_session()
        session.reset(seed=0)
        # No step taken to end episode yet (on first step after reset it shouldn't be done).
        self.assertIsNone(session.export_replay())

    def test_load_and_playback_replay(self) -> None:
        import numpy as np

        session = self._make_session()
        session.reset(seed=0)
        # Record a short sequence.
        for _ in range(5):
            _, done = session.step()
            if done:
                break
        # For the test, manually finish the recorder.
        replay_obj = session._recorder.finish(outcome="draw")
        session._last_replay = replay_obj

        replay_dict = session.export_replay()
        self.assertIsNotNone(replay_dict)

        # Load into a fresh session.
        session2 = self._make_session()
        ok = session2.load_replay(replay_dict)
        self.assertTrue(ok)
        self.assertIsNotNone(session2._replay_player)

    def test_replay_step_advances(self) -> None:
        session = self._make_session()
        session.reset(seed=0)
        for _ in range(3):
            _, done = session.step()
            if done:
                break
        replay_obj = session._recorder.finish(outcome="draw")
        session._last_replay = replay_obj
        replay_dict = session.export_replay()

        session2 = self._make_session()
        session2.load_replay(replay_dict)
        msg = session2.replay_step()
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "replay_frame")
        self.assertEqual(msg["index"], 0)

    def test_replay_seek(self) -> None:
        session = self._make_session()
        session.reset(seed=0)
        for _ in range(5):
            _, done = session.step()
            if done:
                break
        replay_obj = session._recorder.finish(outcome="draw")
        session._last_replay = replay_obj
        replay_dict = session.export_replay()

        session2 = self._make_session()
        session2.load_replay(replay_dict)
        msg = session2.replay_seek(2)
        if msg is not None:
            self.assertEqual(msg["type"], "replay_frame")
            self.assertGreaterEqual(msg["index"], 0)

    def test_invalid_replay_load_returns_false(self) -> None:
        session = self._make_session()
        ok = session.load_replay({"invalid": "data"})
        self.assertFalse(ok)

    def test_replay_step_returns_none_without_replay(self) -> None:
        session = self._make_session()
        self.assertIsNone(session.replay_step())

    def test_replay_seek_returns_none_without_replay(self) -> None:
        session = self._make_session()
        self.assertIsNone(session.replay_seek(0))

    def test_unknown_scenario_falls_back(self) -> None:
        from server.game_server import GameSession

        session = GameSession(scenario="nonexistent_xyz", difficulty=3)
        frame = session.reset(seed=0)
        self.assertEqual(frame["type"], "frame")

    def test_difficulty_clamped_low(self) -> None:
        from server.game_server import GameSession

        session = GameSession(scenario="open_field", difficulty=-99)
        self.assertEqual(session._difficulty, 1)

    def test_difficulty_clamped_high(self) -> None:
        from server.game_server import GameSession

        session = GameSession(scenario="open_field", difficulty=999)
        self.assertEqual(session._difficulty, 5)

    def test_weather_condition_accepted(self) -> None:
        from server.game_server import GameSession

        # Should not raise; weather enables fine.
        session = GameSession(scenario="open_field", difficulty=3, weather_condition="RAIN")
        frame = session.reset(seed=0)
        self.assertEqual(frame["type"], "frame")

    def test_invalid_weather_condition_graceful(self) -> None:
        from server.game_server import GameSession

        # Invalid condition name should not crash.
        session = GameSession(scenario="open_field", difficulty=3, weather_condition="TORNADO")
        frame = session.reset(seed=0)
        self.assertEqual(frame["type"], "frame")

    def test_randomize_terrain_override(self) -> None:
        from server.game_server import GameSession

        session = GameSession(scenario="open_field", difficulty=3, randomize_terrain=True)
        frame = session.reset(seed=0)
        self.assertEqual(frame["type"], "frame")

    def test_onnx_policy_wired_as_red_policy(self) -> None:
        """_OnnxRedPolicy should be set on the inner env when policy_url given."""
        from server.game_server import GameSession, _OnnxRedPolicy

        # Use a non-existent URL; we just check the red_policy type, not behaviour.
        session = GameSession(scenario="open_field", difficulty=3, policy_url="http://localhost:19999")
        inner_env = session._env._env  # BattalionEnv
        self.assertIsInstance(inner_env.red_policy, _OnnxRedPolicy)


class TestGameServerInit(unittest.TestCase):
    """Tests for GameServer constructor (no event loop required)."""

    def test_init_raises_without_websockets(self) -> None:
        from server import game_server as gs

        orig = gs._WS_AVAILABLE
        try:
            gs._WS_AVAILABLE = False
            from server.game_server import GameServer

            with self.assertRaises(ImportError):
                GameServer()
        finally:
            gs._WS_AVAILABLE = orig

    def test_init_succeeds_with_websockets(self) -> None:
        from server.game_server import GameServer, _WS_AVAILABLE

        if not _WS_AVAILABLE:
            self.skipTest("websockets not installed")
        server = GameServer(host="127.0.0.1", port=9999)
        self.assertEqual(server._host, "127.0.0.1")
        self.assertEqual(server._port, 9999)


if __name__ == "__main__":
    unittest.main()
