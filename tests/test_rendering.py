# tests/test_rendering.py
"""Tests for envs/rendering/renderer.py, recorder.py, and the --render CLI."""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

# ---------------------------------------------------------------------------
# Headless pygame — must be set before pygame is imported by any module.
# ---------------------------------------------------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.battalion_env import BattalionEnv
from envs.rendering.recorder import (
    EpisodeRecorder,
    EpisodeReplayer,
    _battalion_to_dict,
    _dict_to_battalion,
    _is_json_serialisable,
)
from envs.rendering.renderer import BattalionRenderer
from envs.sim.battalion import Battalion
from envs.sim.terrain import TerrainMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blue() -> Battalion:
    return Battalion(x=300.0, y=400.0, theta=math.pi / 4, strength=0.8, team=0)


def _make_red() -> Battalion:
    return Battalion(x=700.0, y=600.0, theta=math.pi, strength=0.6, team=1)


def _make_flat_terrain() -> TerrainMap:
    return TerrainMap.flat(1000.0, 1000.0)


# ---------------------------------------------------------------------------
# BattalionRenderer tests
# ---------------------------------------------------------------------------


class TestBattalionRenderer(unittest.TestCase):
    """Tests for BattalionRenderer (runs in headless dummy mode)."""

    def setUp(self) -> None:
        self.renderer = BattalionRenderer(
            map_width=1000.0, map_height=1000.0, fps=0
        )

    def tearDown(self) -> None:
        self.renderer.close()

    def test_init_creates_renderer(self) -> None:
        """BattalionRenderer can be instantiated without raising."""
        self.assertIsNotNone(self.renderer)

    def test_world_to_screen_origin(self) -> None:
        """World (0, 0) maps to the bottom-left of the screen (clamped)."""
        sx, sy = self.renderer._world_to_screen(0.0, 0.0)
        self.assertEqual(sx, 0)
        # y=0 → sy = win_h, clamped to win_h - 1
        self.assertEqual(sy, self.renderer._win_h - 1)

    def test_world_to_screen_top_right(self) -> None:
        """World (width, height) maps to the top-right corner (clamped)."""
        sx, sy = self.renderer._world_to_screen(1000.0, 1000.0)
        self.assertEqual(sx, self.renderer._win_w - 1)
        self.assertEqual(sy, 0)

    def test_world_to_screen_clamped(self) -> None:
        """Out-of-bounds world coordinates are clamped to screen boundaries."""
        sx, sy = self.renderer._world_to_screen(-100.0, -100.0)
        self.assertEqual(sx, 0)
        self.assertEqual(sy, self.renderer._win_h - 1)
        sx2, sy2 = self.renderer._world_to_screen(9999.0, 9999.0)
        self.assertEqual(sx2, self.renderer._win_w - 1)
        self.assertEqual(sy2, 0)

    def test_world_to_screen_centre(self) -> None:
        """World centre maps to screen centre."""
        sx, sy = self.renderer._world_to_screen(500.0, 500.0)
        self.assertEqual(sx, self.renderer._win_w // 2)
        self.assertEqual(sy, self.renderer._win_h // 2)

    def test_render_frame_returns_bool(self) -> None:
        """render_frame() returns a bool (True in dummy mode)."""
        blue = _make_blue()
        red = _make_red()
        result = self.renderer.render_frame(blue, red, step=1)
        self.assertIsInstance(result, bool)

    def test_render_frame_with_flat_terrain(self) -> None:
        """render_frame() works with a flat TerrainMap."""
        blue = _make_blue()
        red = _make_red()
        terrain = _make_flat_terrain()
        result = self.renderer.render_frame(blue, red, terrain=terrain, step=0)
        self.assertIsInstance(result, bool)

    def test_set_terrain_caches_surface(self) -> None:
        """set_terrain() caches a surface and records the terrain id."""
        terrain = _make_flat_terrain()
        self.assertIsNone(self.renderer._terrain_surface)
        self.renderer.set_terrain(terrain)
        self.assertIsNotNone(self.renderer._terrain_surface)
        self.assertEqual(self.renderer._cached_terrain_id, id(terrain))

    def test_terrain_cache_refreshed_on_new_terrain(self) -> None:
        """render_frame() rebuilds the terrain surface when terrain changes."""
        terrain_a = _make_flat_terrain()
        terrain_b = _make_flat_terrain()
        self.assertIsNot(terrain_a, terrain_b)  # confirm distinct objects
        blue = _make_blue()
        red = _make_red()
        self.renderer.render_frame(blue, red, terrain=terrain_a, step=0)
        surface_a = self.renderer._terrain_surface
        self.renderer.render_frame(blue, red, terrain=terrain_b, step=1)
        surface_b = self.renderer._terrain_surface
        # Different terrain objects → cache must have been rebuilt
        self.assertIsNot(surface_a, surface_b)

    def test_terrain_cache_not_rebuilt_for_same_object(self) -> None:
        """render_frame() reuses the cached surface when terrain is unchanged."""
        terrain = _make_flat_terrain()
        blue = _make_blue()
        red = _make_red()
        self.renderer.render_frame(blue, red, terrain=terrain, step=0)
        surface_first = self.renderer._terrain_surface
        self.renderer.render_frame(blue, red, terrain=terrain, step=1)
        self.assertIs(self.renderer._terrain_surface, surface_first)

    def test_render_routed_battalion(self) -> None:
        """render_frame() handles a routed battalion without crashing."""
        blue = _make_blue()
        red = _make_red()
        red.routed = True
        result = self.renderer.render_frame(blue, red, step=5)
        self.assertIsInstance(result, bool)

    def test_render_zero_strength(self) -> None:
        """render_frame() handles zero-strength battalions without crashing."""
        blue = _make_blue()
        blue.strength = 0.0
        blue.morale = 0.0
        red = _make_red()
        result = self.renderer.render_frame(blue, red, step=10)
        self.assertIsInstance(result, bool)

    def test_build_terrain_surface_with_elevation(self) -> None:
        """_build_terrain_surface() handles a non-trivial elevation grid."""
        import numpy as np

        grid = np.linspace(0.0, 1.0, 25).reshape(5, 5).astype(np.float32)
        terrain = TerrainMap.from_arrays(
            width=1000.0,
            height=1000.0,
            elevation=grid,
            cover=np.zeros((5, 5), dtype=np.float32),
        )
        surf = self.renderer._build_terrain_surface(terrain)
        self.assertIsNotNone(surf)

    def test_build_terrain_surface_with_cover(self) -> None:
        """_build_terrain_surface() handles a cover grid without errors."""
        import numpy as np

        terrain = TerrainMap.from_arrays(
            width=1000.0,
            height=1000.0,
            elevation=np.zeros((4, 4), dtype=np.float32),
            cover=np.full((4, 4), 0.5, dtype=np.float32),
        )
        surf = self.renderer._build_terrain_surface(terrain)
        self.assertIsNotNone(surf)


# ---------------------------------------------------------------------------
# EpisodeRecorder tests
# ---------------------------------------------------------------------------


class TestEpisodeRecorder(unittest.TestCase):
    """Tests for EpisodeRecorder."""

    def test_initial_state(self) -> None:
        """A fresh recorder has zero frames."""
        rec = EpisodeRecorder()
        self.assertEqual(rec.n_frames, 0)
        self.assertEqual(rec.frames(), [])

    def test_record_step_increments_n_frames(self) -> None:
        """record_step() increases n_frames by 1."""
        rec = EpisodeRecorder()
        rec.record_step(0, _make_blue(), _make_red())
        self.assertEqual(rec.n_frames, 1)

    def test_record_multiple_steps(self) -> None:
        """record_step() can be called many times."""
        rec = EpisodeRecorder()
        for i in range(10):
            rec.record_step(i, _make_blue(), _make_red(), reward=float(i))
        self.assertEqual(rec.n_frames, 10)

    def test_frame_structure(self) -> None:
        """Recorded frames contain expected keys."""
        rec = EpisodeRecorder()
        rec.record_step(3, _make_blue(), _make_red(), reward=1.5, info={"key": "val"})
        frame = rec.frames()[0]
        self.assertEqual(frame["step"], 3)
        self.assertAlmostEqual(frame["reward"], 1.5)
        self.assertIn("blue", frame)
        self.assertIn("red", frame)
        self.assertIn("info", frame)

    def test_frames_returns_copy(self) -> None:
        """frames() returns a copy so mutation does not affect the recorder."""
        rec = EpisodeRecorder()
        rec.record_step(0, _make_blue(), _make_red())
        copy = rec.frames()
        copy.clear()
        self.assertEqual(rec.n_frames, 1)

    def test_save_creates_json_file(self) -> None:
        """save() writes a valid JSON file."""
        rec = EpisodeRecorder()
        rec.record_step(0, _make_blue(), _make_red())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep.json"
            returned = rec.save(path)
            self.assertTrue(path.exists())
            self.assertEqual(returned, path.resolve())
            with path.open() as fh:
                data = json.load(fh)
            self.assertIn("frames", data)
            self.assertEqual(len(data["frames"]), 1)

    def test_save_creates_parent_directories(self) -> None:
        """save() creates missing parent directories."""
        rec = EpisodeRecorder()
        rec.record_step(0, _make_blue(), _make_red())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b" / "ep.json"
            rec.save(path)
            self.assertTrue(path.exists())

    def test_non_serialisable_info_values_dropped(self) -> None:
        """Non-JSON-serialisable info values are silently omitted."""
        rec = EpisodeRecorder()
        rec.record_step(
            0,
            _make_blue(),
            _make_red(),
            info={"ok": 1, "bad": object()},
        )
        frame = rec.frames()[0]
        self.assertIn("ok", frame["info"])
        self.assertNotIn("bad", frame["info"])


# ---------------------------------------------------------------------------
# EpisodeReplayer tests
# ---------------------------------------------------------------------------


class TestEpisodeReplayer(unittest.TestCase):
    """Tests for EpisodeReplayer."""

    def _save_recording(self, n: int = 5) -> tuple[Path, "EpisodeReplayer"]:
        """Helper: record n steps, save to a tmp file, return path + replayer."""
        rec = EpisodeRecorder()
        for i in range(n):
            rec.record_step(i, _make_blue(), _make_red(), reward=float(i))
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "test_ep.json"
        rec.save(path)
        replayer = EpisodeReplayer.from_file(path)
        return path, replayer

    def test_from_file_loads_frames(self) -> None:
        """from_file() correctly loads the frame count."""
        _, replayer = self._save_recording(7)
        self.assertEqual(replayer.n_frames, 7)

    def test_replay_calls_renderer_for_each_frame(self) -> None:
        """replay() calls render_frame once per recorded frame."""
        _, replayer = self._save_recording(4)
        renderer = BattalionRenderer(1000.0, 1000.0, fps=0)
        try:
            # Count render_frame invocations by wrapping it
            call_count = {"n": 0}
            original = renderer.render_frame

            def counting_render(*args, **kwargs):
                call_count["n"] += 1
                return original(*args, **kwargs)

            renderer.render_frame = counting_render  # type: ignore[method-assign]
            replayer.replay(renderer)
            self.assertEqual(call_count["n"], 4)
        finally:
            renderer.close()

    def test_replay_stops_on_window_close(self) -> None:
        """replay() stops early when render_frame returns False."""
        _, replayer = self._save_recording(10)
        renderer = BattalionRenderer(1000.0, 1000.0, fps=0)
        try:
            call_count = {"n": 0}

            def fake_render(*args, **kwargs):
                call_count["n"] += 1
                return call_count["n"] < 3  # stop after 2 frames

            renderer.render_frame = fake_render  # type: ignore[method-assign]
            replayer.replay(renderer)
            self.assertEqual(call_count["n"], 3)  # 2 True + 1 False
        finally:
            renderer.close()

    def test_replayer_round_trip(self) -> None:
        """Recording then replaying preserves step count and battalion state."""
        rec = EpisodeRecorder()
        blue = Battalion(x=123.0, y=456.0, theta=1.23, strength=0.7, team=0)
        red = Battalion(x=789.0, y=100.0, theta=2.34, strength=0.4, team=1)
        red.routed = True
        rec.record_step(0, blue, red, reward=0.5)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rt.json"
            rec.save(path)
            replayer = EpisodeReplayer.from_file(path)

        self.assertEqual(replayer.n_frames, 1)
        frame = replayer._frames[0]
        self.assertAlmostEqual(frame["blue"]["x"], 123.0)
        self.assertAlmostEqual(frame["red"]["theta"], 2.34)
        self.assertTrue(frame["red"]["routed"])


# ---------------------------------------------------------------------------
# Serialisation helper tests
# ---------------------------------------------------------------------------


class TestSerialisationHelpers(unittest.TestCase):
    def test_battalion_to_dict_fields(self) -> None:
        b = Battalion(x=1.0, y=2.0, theta=0.5, strength=0.9, team=0)
        d = _battalion_to_dict(b)
        self.assertEqual(d["team"], 0)
        self.assertAlmostEqual(d["x"], 1.0)
        self.assertAlmostEqual(d["theta"], 0.5)
        self.assertFalse(d["routed"])

    def test_dict_to_battalion_roundtrip(self) -> None:
        b = Battalion(x=10.0, y=20.0, theta=1.0, strength=0.5, team=1)
        b.morale = 0.3
        b.routed = True
        d = _battalion_to_dict(b)
        b2 = _dict_to_battalion(d)
        self.assertAlmostEqual(b2.x, 10.0)
        self.assertAlmostEqual(b2.morale, 0.3)
        self.assertTrue(b2.routed)
        self.assertEqual(b2.team, 1)

    def test_is_json_serialisable_primitives(self) -> None:
        self.assertTrue(_is_json_serialisable(1))
        self.assertTrue(_is_json_serialisable(1.5))
        self.assertTrue(_is_json_serialisable("hello"))
        self.assertTrue(_is_json_serialisable(True))
        self.assertTrue(_is_json_serialisable(None))
        self.assertTrue(_is_json_serialisable([1, 2, 3]))

    def test_is_json_serialisable_rejects_objects(self) -> None:
        self.assertFalse(_is_json_serialisable(object()))
        self.assertFalse(_is_json_serialisable({1, 2, 3}))  # set is not serialisable


# ---------------------------------------------------------------------------
# BattalionEnv render_mode="human" tests
# ---------------------------------------------------------------------------


class TestBattalionEnvRenderMode(unittest.TestCase):
    """Verify BattalionEnv accepts render_mode='human' and calls the renderer."""

    def test_human_render_mode_accepted(self) -> None:
        """BattalionEnv(render_mode='human') does not raise on construction."""
        env = BattalionEnv(render_mode="human")
        env.close()

    def test_render_before_reset_is_noop(self) -> None:
        """Calling render() before reset() does not crash."""
        env = BattalionEnv(render_mode="human")
        env.render()  # blue/red are None — should be a no-op
        env.close()

    def test_render_after_reset(self) -> None:
        """render() after reset() creates a renderer and renders without error."""
        env = BattalionEnv(render_mode="human")
        env.reset(seed=0)
        env.render()
        self.assertIsNotNone(env._renderer)
        env.close()

    def test_close_destroys_renderer(self) -> None:
        """close() disposes of the renderer."""
        env = BattalionEnv(render_mode="human")
        env.reset(seed=0)
        env.render()
        env.close()
        self.assertIsNone(env._renderer)

    def test_none_render_mode_no_renderer(self) -> None:
        """render_mode=None never creates a renderer."""
        env = BattalionEnv(render_mode=None)
        env.reset(seed=0)
        env.render()
        self.assertIsNone(env._renderer)
        env.close()

    def test_invalid_render_mode_raises(self) -> None:
        """An unrecognised render_mode raises ValueError."""
        with self.assertRaises(ValueError):
            BattalionEnv(render_mode="rgb_array")


# ---------------------------------------------------------------------------
# CLI --render / --record flag tests
# ---------------------------------------------------------------------------


class TestEvaluateCLIFlags(unittest.TestCase):
    """Tests for the --render and --record flags in evaluate.py main()."""

    _tmpdir: tempfile.TemporaryDirectory
    _checkpoint: str

    @classmethod
    def setUpClass(cls) -> None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from models.mlp_policy import BattalionMlpPolicy

        cls._tmpdir = tempfile.TemporaryDirectory()
        vec_env = make_vec_env(BattalionEnv, n_envs=1, seed=0)
        model = PPO(BattalionMlpPolicy, vec_env, n_steps=32, batch_size=16, verbose=0)
        model.learn(total_timesteps=32)
        ckpt = Path(cls._tmpdir.name) / "render_test_model"
        model.save(str(ckpt))
        vec_env.close()
        cls._checkpoint = str(ckpt)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _ckpt(self) -> str:
        return self._checkpoint

    def test_render_flag_runs_episode(self) -> None:
        """--render flag causes main() to run an episode and print a win rate."""
        from training.evaluate import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "--checkpoint", self._ckpt(),
                "--n-episodes", "1",
                "--opponent", "scripted_l1",
                "--seed", "0",
                "--render",
            ])
        output = buf.getvalue()
        self.assertIn("Win rate:", output)

    def test_record_flag_saves_json(self) -> None:
        """--record saves episode JSON file(s) under the specified directory."""
        from training.evaluate import main

        with tempfile.TemporaryDirectory() as rec_dir:
            buf = io.StringIO()
            with redirect_stdout(buf):
                main([
                    "--checkpoint", self._ckpt(),
                    "--n-episodes", "2",
                    "--opponent", "scripted_l1",
                    "--seed", "0",
                    "--record", rec_dir,
                ])
            json_files = list(Path(rec_dir).glob("*.json"))
            self.assertEqual(len(json_files), 2)
            output = buf.getvalue()
            self.assertIn("Recorded:", output)
            # Each file should be a valid recording
            for jf in json_files:
                with jf.open() as fh:
                    data = json.load(fh)
                self.assertIn("frames", data)
                self.assertGreater(len(data["frames"]), 0)

    def test_render_flag_registered_in_argparser(self) -> None:
        """--render is a valid CLI argument (parse_args does not raise)."""
        import argparse
        # We just need to confirm the flag exists; importing main and calling
        # parse_args with --help would exit, so use a minimal parse.
        from training.evaluate import main  # noqa: F401
        # Indirect check: the flag appears in the help text.
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "training" / "evaluate.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertIn("--render", result.stdout)
        self.assertIn("--record", result.stdout)


if __name__ == "__main__":
    unittest.main()
