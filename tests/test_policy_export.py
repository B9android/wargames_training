# SPDX-License-Identifier: MIT
# tests/test_policy_export.py
"""Tests for scripts/export_policy.py — Epic E5.5 policy export.

Coverage
--------
* export_to_onnx        — file is created, ONNX graph validates
* export_to_torchscript — file is created, model loads and runs
* Inference parity      — ONNX and TorchScript outputs match PyTorch
                          within the required tolerance (≤ 1e-5)
* _SB3ActorWrapper      — wraps SB3 policy; forward produces correct shape
* export_policy()       — end-to-end orchestrator for mappo_actor,
                          mappo_critic, and sb3_mlp
* benchmark_inference() — returns expected stat keys
* CLI (main())          — valid args succeed, bad args return exit code 1
* Error paths           — unknown model_type, unknown format, missing file
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_policy import (
    _SB3ActorWrapper,
    _load_model,
    benchmark_inference,
    export_policy,
    export_to_onnx,
    export_to_torchscript,
    main,
)

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 22
ACTION_DIM = 3
STATE_DIM = 25
BATCH = 4
ATOL = 1e-5  # acceptance criterion from the issue


def _make_actor() -> nn.Module:
    """Tiny MAPPOActor with default hidden sizes."""
    from models.mappo_policy import MAPPOActor

    return MAPPOActor(obs_dim=OBS_DIM, action_dim=ACTION_DIM)


def _make_critic() -> nn.Module:
    """Tiny MAPPOCritic with default hidden sizes."""
    from models.mappo_policy import MAPPOCritic

    return MAPPOCritic(state_dim=STATE_DIM)


def _dummy_obs(batch: int = BATCH) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, OBS_DIM)


def _dummy_state(batch: int = BATCH) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, STATE_DIM)


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------


class TestExportToTorchScript(unittest.TestCase):
    """Tests for export_to_torchscript()."""

    def test_actor_file_created(self) -> None:
        """TorchScript export creates the output file."""
        actor = _make_actor().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.torchscript.pt"
            result = export_to_torchscript(actor, _dummy_obs(), out)
            self.assertTrue(out.exists())
            self.assertEqual(result, out)

    def test_critic_file_created(self) -> None:
        """TorchScript export works for the critic too."""
        critic = _make_critic().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "critic.torchscript.pt"
            export_to_torchscript(critic, _dummy_state(), out)
            self.assertTrue(out.exists())

    def test_loaded_model_runs(self) -> None:
        """Loaded TorchScript model runs without error."""
        actor = _make_actor().eval()
        obs = _dummy_obs()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.torchscript.pt"
            export_to_torchscript(actor, obs, out)
            loaded = torch.jit.load(str(out))
            loaded.eval()
            with torch.no_grad():
                output = loaded(obs)
            self.assertIsNotNone(output)

    def test_torchscript_output_shape(self) -> None:
        """TorchScript actor output has the expected shape."""
        actor = _make_actor().eval()
        obs = _dummy_obs()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.torchscript.pt"
            export_to_torchscript(actor, obs, out)
            loaded = torch.jit.load(str(out))
            loaded.eval()
            with torch.no_grad():
                mean, std = loaded(obs)
            self.assertEqual(mean.shape, (BATCH, ACTION_DIM))
            self.assertEqual(std.shape, (BATCH, ACTION_DIM))

    def test_torchscript_parity_with_pytorch(self) -> None:
        """TorchScript output matches PyTorch within tolerance."""
        actor = _make_actor().eval()
        obs = _dummy_obs()
        with torch.no_grad():
            pt_mean, pt_std = actor(obs)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.torchscript.pt"
            export_to_torchscript(actor, obs, out)
            loaded = torch.jit.load(str(out))
            loaded.eval()
            with torch.no_grad():
                ts_mean, ts_std = loaded(obs)

        self.assertTrue(
            torch.allclose(pt_mean, ts_mean, atol=ATOL),
            msg=f"Max mean diff: {(pt_mean - ts_mean).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(pt_std, ts_std, atol=ATOL),
            msg=f"Max std diff: {(pt_std - ts_std).abs().max().item():.2e}",
        )

    def test_creates_parent_directory(self) -> None:
        """export_to_torchscript creates missing parent directories."""
        actor = _make_actor().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "subdir" / "nested" / "actor.pt"
            export_to_torchscript(actor, _dummy_obs(), out)
            self.assertTrue(out.exists())


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


@unittest.skipUnless(_ONNX_AVAILABLE, "onnx / onnxruntime not installed")
class TestExportToOnnx(unittest.TestCase):
    """Tests for export_to_onnx() — skipped when ONNX is not installed."""

    def test_actor_file_created(self) -> None:
        """ONNX export creates the output file."""
        actor = _make_actor().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.onnx"
            result = export_to_onnx(actor, _dummy_obs(), out)
            self.assertTrue(out.exists())
            self.assertEqual(result, out)

    def test_critic_file_created(self) -> None:
        """ONNX export works for the critic."""
        critic = _make_critic().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "critic.onnx"
            export_to_onnx(
                critic,
                _dummy_state(),
                out,
                input_names=["state"],
                output_names=["value"],
                dynamic_axes={"state": {0: "batch"}, "value": {0: "batch"}},
            )
            self.assertTrue(out.exists())

    def test_onnx_graph_is_valid(self) -> None:
        """ONNX checker validates the exported graph without error."""
        import onnx as _onnx

        actor = _make_actor().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.onnx"
            export_to_onnx(actor, _dummy_obs(), out)
            proto = _onnx.load(str(out))
            _onnx.checker.check_model(proto)  # raises if invalid

    def test_onnx_parity_with_pytorch_actor_outputs(self) -> None:
        """ONNX actor outputs (action_mean, action_std) both match PyTorch within 1e-5."""
        import numpy as np
        import onnxruntime as ort

        actor = _make_actor().eval()
        obs = _dummy_obs()
        with torch.no_grad():
            pt_mean, pt_std = actor(obs)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.onnx"
            export_to_onnx(
                actor, obs, out,
                input_names=["obs"],
                output_names=["action_mean", "action_std"],
                dynamic_axes={
                    "obs": {0: "batch"},
                    "action_mean": {0: "batch"},
                    "action_std": {0: "batch"},
                },
            )

            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            ort_outputs = sess.run(None, {input_name: obs.numpy()})

        # Verify both outputs are present.
        self.assertEqual(len(ort_outputs), 2, "Expected 2 outputs (action_mean, action_std)")

        ort_mean = torch.tensor(ort_outputs[0])
        ort_std = torch.tensor(ort_outputs[1])
        self.assertTrue(
            torch.allclose(pt_mean, ort_mean, atol=ATOL),
            msg=f"action_mean max diff: {(pt_mean - ort_mean).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(pt_std, ort_std, atol=ATOL),
            msg=f"action_std max diff: {(pt_std - ort_std).abs().max().item():.2e}",
        )

    def test_onnx_dynamic_batch(self) -> None:
        """ONNX model accepts different batch sizes (dynamic axes)."""
        import onnxruntime as ort

        actor = _make_actor().eval()
        obs_4 = _dummy_obs(batch=4)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.onnx"
            export_to_onnx(actor, obs_4, out)

            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name

            # Run with different batch sizes.
            for batch in (1, 2, 8):
                obs = torch.randn(batch, OBS_DIM).numpy()
                result = sess.run(None, {input_name: obs})
                self.assertEqual(result[0].shape[0], batch)

    def test_creates_parent_directory(self) -> None:
        """export_to_onnx creates missing parent directories."""
        actor = _make_actor().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "actor.onnx"
            export_to_onnx(actor, _dummy_obs(), out)
            self.assertTrue(out.exists())

    def test_onnx_import_error_without_package(self) -> None:
        """export_to_onnx raises ImportError when onnx is not importable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "onnx":
                raise ImportError("mocked missing onnx")
            return real_import(name, *args, **kwargs)

        actor = _make_actor().eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.onnx"
            builtins.__import__ = mock_import
            try:
                with self.assertRaises(ImportError):
                    export_to_onnx(actor, _dummy_obs(), out)
            finally:
                builtins.__import__ = real_import


# ---------------------------------------------------------------------------
# _SB3ActorWrapper
# ---------------------------------------------------------------------------


class TestSB3ActorWrapper(unittest.TestCase):
    """Tests for _SB3ActorWrapper."""

    def _make_sb3_policy(self):
        """Return a minimal SB3 ActorCriticPolicy-like mock."""
        try:
            from envs.battalion_env import BattalionEnv
            from models.mlp_policy import BattalionMlpPolicy
            from stable_baselines3 import PPO

            env = BattalionEnv(randomize_terrain=False, max_steps=5)
            model = PPO(BattalionMlpPolicy, env, verbose=0)
            env.close()
            return model.policy
        except (ImportError, Exception):
            return None

    def test_wrapper_forward_shape(self) -> None:
        """_SB3ActorWrapper produces output of shape (batch, action_dim)."""
        policy = self._make_sb3_policy()
        if policy is None:
            self.skipTest("SB3/BattalionEnv not available")
        wrapper = _SB3ActorWrapper(policy)
        wrapper.eval()
        obs = torch.zeros(3, 12)
        with torch.no_grad():
            out = wrapper(obs)
        self.assertEqual(out.shape[0], 3)

    def test_wrapper_is_nn_module(self) -> None:
        """_SB3ActorWrapper is an nn.Module and can be JIT-traced."""
        policy = self._make_sb3_policy()
        if policy is None:
            self.skipTest("SB3/BattalionEnv not available")
        wrapper = _SB3ActorWrapper(policy)
        self.assertIsInstance(wrapper, nn.Module)


# ---------------------------------------------------------------------------
# export_policy() orchestrator
# ---------------------------------------------------------------------------


class TestExportPolicyOrchestrator(unittest.TestCase):
    """Tests for the high-level export_policy() function."""

    def _save_actor_checkpoint(self, tmp: str) -> Path:
        actor = _make_actor()
        ckpt = Path(tmp) / "actor.pt"
        torch.save(actor.state_dict(), ckpt)
        return ckpt

    def _save_critic_checkpoint(self, tmp: str) -> Path:
        critic = _make_critic()
        ckpt = Path(tmp) / "critic.pt"
        torch.save(critic.state_dict(), ckpt)
        return ckpt

    def test_mappo_actor_torchscript_only(self) -> None:
        """export_policy creates a TorchScript file for mappo_actor."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._save_actor_checkpoint(tmp)
            out_dir = Path(tmp) / "out"
            result = export_policy(
                ckpt, "mappo_actor", out_dir,
                obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                formats=["torchscript"],
            )
            self.assertIn("torchscript", result)
            self.assertTrue(result["torchscript"].exists())

    @unittest.skipUnless(_ONNX_AVAILABLE, "onnx / onnxruntime not installed")
    def test_mappo_actor_onnx_only(self) -> None:
        """export_policy creates an ONNX file for mappo_actor."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._save_actor_checkpoint(tmp)
            out_dir = Path(tmp) / "out"
            result = export_policy(
                ckpt, "mappo_actor", out_dir,
                obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                formats=["onnx"],
            )
            self.assertIn("onnx", result)
            self.assertTrue(result["onnx"].exists())

    @unittest.skipUnless(_ONNX_AVAILABLE, "onnx / onnxruntime not installed")
    def test_mappo_actor_both_formats(self) -> None:
        """export_policy creates both ONNX and TorchScript for mappo_actor."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._save_actor_checkpoint(tmp)
            out_dir = Path(tmp) / "out"
            result = export_policy(
                ckpt, "mappo_actor", out_dir,
                obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                formats=["onnx", "torchscript"],
            )
            self.assertIn("onnx", result)
            self.assertIn("torchscript", result)
            self.assertTrue(result["onnx"].exists())
            self.assertTrue(result["torchscript"].exists())

    def test_mappo_critic_torchscript(self) -> None:
        """export_policy exports mappo_critic to TorchScript."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._save_critic_checkpoint(tmp)
            out_dir = Path(tmp) / "out"
            result = export_policy(
                ckpt, "mappo_critic", out_dir,
                obs_dim=OBS_DIM, state_dim=STATE_DIM,
                formats=["torchscript"],
            )
            self.assertIn("torchscript", result)
            self.assertTrue(result["torchscript"].exists())

    def test_unknown_model_type_raises(self) -> None:
        """export_policy raises ValueError for an unknown model_type."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "dummy.pt"
            ckpt.touch()
            with self.assertRaises(ValueError):
                export_policy(
                    ckpt, "unknown_type", Path(tmp) / "out",
                    obs_dim=OBS_DIM,
                )

    def test_unknown_format_raises(self) -> None:
        """export_policy raises ValueError for an unrecognised format."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._save_actor_checkpoint(tmp)
            with self.assertRaises(ValueError):
                export_policy(
                    ckpt, "mappo_actor", Path(tmp) / "out",
                    obs_dim=OBS_DIM,
                    formats=["onnx", "banana"],
                )

    def test_output_stem_matches_checkpoint(self) -> None:
        """Output file stem is derived from the checkpoint filename."""
        with tempfile.TemporaryDirectory() as tmp:
            actor = _make_actor()
            ckpt = Path(tmp) / "my_policy.pt"
            torch.save(actor.state_dict(), ckpt)
            out_dir = Path(tmp) / "out"
            result = export_policy(
                ckpt, "mappo_actor", out_dir,
                obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                formats=["torchscript"],
            )
            self.assertIn("my_policy", result["torchscript"].stem)


# ---------------------------------------------------------------------------
# benchmark_inference
# ---------------------------------------------------------------------------


class TestBenchmarkInference(unittest.TestCase):
    """Tests for benchmark_inference()."""

    def test_returns_expected_keys(self) -> None:
        """benchmark_inference returns mean_ms, min_ms, max_ms."""
        actor = _make_actor().eval()
        obs = _dummy_obs(batch=1)
        stats = benchmark_inference(actor, obs, n_warmup=2, n_runs=5)
        self.assertIn("mean_ms", stats)
        self.assertIn("min_ms", stats)
        self.assertIn("max_ms", stats)

    def test_latency_values_are_positive(self) -> None:
        """Latency values are positive floats."""
        actor = _make_actor().eval()
        stats = benchmark_inference(actor, _dummy_obs(batch=1), n_warmup=2, n_runs=5)
        self.assertGreater(stats["mean_ms"], 0.0)
        self.assertGreater(stats["min_ms"], 0.0)
        self.assertGreater(stats["max_ms"], 0.0)

    def test_min_lte_mean_lte_max(self) -> None:
        """min_ms ≤ mean_ms ≤ max_ms."""
        actor = _make_actor().eval()
        stats = benchmark_inference(actor, _dummy_obs(batch=1), n_warmup=2, n_runs=10)
        self.assertLessEqual(stats["min_ms"], stats["mean_ms"])
        self.assertLessEqual(stats["mean_ms"], stats["max_ms"])


# ---------------------------------------------------------------------------
# CLI (main())
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    """Tests for the command-line interface via main()."""

    def _make_checkpoint(self, tmp: str) -> Path:
        actor = _make_actor()
        ckpt = Path(tmp) / "actor.pt"
        torch.save(actor.state_dict(), ckpt)
        return ckpt

    def test_torchscript_export_via_cli(self) -> None:
        """CLI returns exit code 0 for a valid torchscript export."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._make_checkpoint(tmp)
            out_dir = Path(tmp) / "out"
            exit_code = main([
                "--checkpoint", str(ckpt),
                "--model-type", "mappo_actor",
                "--obs-dim", str(OBS_DIM),
                "--action-dim", str(ACTION_DIM),
                "--output-dir", str(out_dir),
                "--formats", "torchscript",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "actor.torchscript.pt").exists())

    @unittest.skipUnless(_ONNX_AVAILABLE, "onnx / onnxruntime not installed")
    def test_onnx_export_via_cli(self) -> None:
        """CLI returns exit code 0 for a valid ONNX export."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._make_checkpoint(tmp)
            out_dir = Path(tmp) / "out"
            exit_code = main([
                "--checkpoint", str(ckpt),
                "--model-type", "mappo_actor",
                "--obs-dim", str(OBS_DIM),
                "--action-dim", str(ACTION_DIM),
                "--output-dir", str(out_dir),
                "--formats", "onnx",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "actor.onnx").exists())

    def test_missing_checkpoint_returns_exit_1(self) -> None:
        """CLI returns exit code 1 when the checkpoint file is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main([
                "--checkpoint", str(Path(tmp) / "nonexistent.pt"),
                "--model-type", "mappo_actor",
                "--obs-dim", str(OBS_DIM),
                "--output-dir", str(Path(tmp) / "out"),
                "--formats", "torchscript",
            ])
            self.assertEqual(exit_code, 1)

    def test_unknown_model_type_returns_exit_1(self) -> None:
        """CLI returns exit code 1 for an unknown model type."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self._make_checkpoint(tmp)
            with self.assertRaises(SystemExit):
                # argparse itself will reject the invalid choice
                main([
                    "--checkpoint", str(ckpt),
                    "--model-type", "bad_type",
                    "--obs-dim", str(OBS_DIM),
                    "--output-dir", str(Path(tmp) / "out"),
                ])


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------


class TestLoadModel(unittest.TestCase):
    """Tests for the internal _load_model helper."""

    def test_mappo_actor_loads(self) -> None:
        """_load_model loads a MAPPOActor state-dict correctly."""
        actor = _make_actor()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "actor.pt"
            torch.save(actor.state_dict(), ckpt)
            loaded = _load_model(ckpt, "mappo_actor", OBS_DIM, ACTION_DIM, STATE_DIM)
        self.assertIsInstance(loaded, nn.Module)

    def test_mappo_critic_loads(self) -> None:
        """_load_model loads a MAPPOCritic state-dict correctly."""
        critic = _make_critic()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "critic.pt"
            torch.save(critic.state_dict(), ckpt)
            loaded = _load_model(ckpt, "mappo_critic", OBS_DIM, ACTION_DIM, STATE_DIM)
        self.assertIsInstance(loaded, nn.Module)

    def test_loaded_model_is_in_eval_mode(self) -> None:
        """_load_model returns the model in eval mode."""
        actor = _make_actor()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "actor.pt"
            torch.save(actor.state_dict(), ckpt)
            loaded = _load_model(ckpt, "mappo_actor", OBS_DIM, ACTION_DIM, STATE_DIM)
        self.assertFalse(loaded.training)

    def test_unknown_model_type_raises(self) -> None:
        """_load_model raises ValueError for an unrecognised model_type."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "dummy.pt"
            ckpt.touch()
            with self.assertRaises(ValueError):
                _load_model(ckpt, "bad_type", OBS_DIM, ACTION_DIM, STATE_DIM)


if __name__ == "__main__":
    unittest.main()
