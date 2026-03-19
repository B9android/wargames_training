"""Unit tests for training.artifacts helper utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.artifacts import (
    CheckpointManifest,
    checkpoint_best_filename,
    checkpoint_final_stem,
    checkpoint_name_prefix,
    parse_step_from_checkpoint_name,
)


class TestArtifactNaming(unittest.TestCase):
    """Naming helper contracts for checkpoint artifacts."""

    def test_v2_prefix_includes_seed_and_curriculum(self) -> None:
        self.assertEqual(
            checkpoint_name_prefix(seed=42, curriculum_level=3, enable_v2=True),
            "ppo_battalion_s42_c3",
        )

    def test_legacy_prefix_is_unchanged(self) -> None:
        self.assertEqual(
            checkpoint_name_prefix(seed=42, curriculum_level=3, enable_v2=False),
            "ppo_battalion",
        )

    def test_final_and_best_names(self) -> None:
        self.assertEqual(
            checkpoint_final_stem(seed=7, curriculum_level=2, enable_v2=True),
            "ppo_battalion_s7_c2_final",
        )
        self.assertEqual(
            checkpoint_best_filename(seed=7, curriculum_level=2, enable_v2=True),
            "ppo_battalion_s7_c2_best.zip",
        )

    def test_parse_step_from_checkpoint_name(self) -> None:
        self.assertEqual(
            parse_step_from_checkpoint_name(Path("ppo_battalion_s7_c2_12345_steps.zip")),
            12345,
        )
        self.assertIsNone(parse_step_from_checkpoint_name(Path("best_model.zip")))


class TestCheckpointManifest(unittest.TestCase):
    """Manifest append/query behavior."""

    def test_register_and_deduplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            ckpt_path = Path(tmp) / "ppo_battalion_s1_c1_100_steps.zip"
            ckpt_path.write_text("x", encoding="utf-8")

            manifest = CheckpointManifest(manifest_path)
            first = manifest.register(
                ckpt_path,
                artifact_type="periodic",
                seed=1,
                curriculum_level=1,
                run_id="abc",
                config_hash="hash",
                step=100,
            )
            second = manifest.register(
                ckpt_path,
                artifact_type="periodic",
                seed=1,
                curriculum_level=1,
                run_id="abc",
                config_hash="hash",
                step=100,
            )

            self.assertTrue(first)
            self.assertFalse(second)

    def test_latest_periodic_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "ppo_battalion_s5_c2_100_steps.zip"
            p2 = root / "ppo_battalion_s5_c2_200_steps.zip"
            p1.write_text("a", encoding="utf-8")
            p2.write_text("b", encoding="utf-8")

            manifest = CheckpointManifest(root / "manifest.jsonl")
            manifest.register(
                p1,
                artifact_type="periodic",
                seed=5,
                curriculum_level=2,
                run_id="r1",
                config_hash="h",
                step=100,
            )
            manifest.register(
                p2,
                artifact_type="periodic",
                seed=5,
                curriculum_level=2,
                run_id="r1",
                config_hash="h",
                step=200,
            )

            latest = manifest.latest_periodic(root, "ppo_battalion_s5_c2")
            self.assertEqual(latest, p2)


class TestResumeResolution(unittest.TestCase):
    """Tests for _resolve_resume_checkpoint helper in training.train."""

    def _import_resolver(self):
        from training.train import _resolve_resume_checkpoint
        return _resolve_resume_checkpoint

    def _make_cfg(self, auto: bool = False, checkpoint: str | None = None):
        from omegaconf import OmegaConf
        return OmegaConf.create({"resume": {"auto": auto, "checkpoint": checkpoint}})

    def test_returns_none_when_disabled(self) -> None:
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_cfg(auto=False)
            result = resolver(
                cfg=cfg,
                checkpoint_dir=Path(tmp),
                manifest=None,
                periodic_prefix="ppo_battalion_s42_c5",
                current_hash="abc",
            )
            self.assertIsNone(result)

    def test_explicit_checkpoint_returned(self) -> None:
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ppo_battalion_s42_c5_100_steps.zip"
            ckpt.write_text("x", encoding="utf-8")
            cfg = self._make_cfg(checkpoint=str(ckpt))
            result = resolver(
                cfg=cfg,
                checkpoint_dir=Path(tmp),
                manifest=None,
                periodic_prefix="ppo_battalion_s42_c5",
                current_hash="abc",
            )
            self.assertEqual(result, ckpt)

    def test_explicit_checkpoint_missing_raises(self) -> None:
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.zip"
            cfg = self._make_cfg(checkpoint=str(missing))
            with self.assertRaises(FileNotFoundError):
                resolver(
                    cfg=cfg,
                    checkpoint_dir=Path(tmp),
                    manifest=None,
                    periodic_prefix="ppo_battalion_s42_c5",
                    current_hash="abc",
                )

    def test_auto_uses_manifest_over_glob(self) -> None:
        """When manifest has a higher-step entry than glob, manifest wins."""
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "ppo_battalion_s42_c5"
            p100 = root / f"{prefix}_100_steps.zip"
            p200 = root / f"{prefix}_200_steps.zip"
            p100.write_text("a", encoding="utf-8")
            p200.write_text("b", encoding="utf-8")

            manifest = CheckpointManifest(root / "manifest.jsonl")
            manifest.register(p200, artifact_type="periodic", seed=42,
                               curriculum_level=5, run_id="r", config_hash="h", step=200)

            cfg = self._make_cfg(auto=True)
            result = resolver(
                cfg=cfg,
                checkpoint_dir=root,
                manifest=manifest,
                periodic_prefix=prefix,
                current_hash="h",
            )
            self.assertEqual(result, p200)

    def test_auto_falls_back_to_glob_when_no_manifest(self) -> None:
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "ppo_battalion_s42_c5"
            p50 = root / f"{prefix}_50_steps.zip"
            p150 = root / f"{prefix}_150_steps.zip"
            p50.write_text("a", encoding="utf-8")
            p150.write_text("b", encoding="utf-8")

            cfg = self._make_cfg(auto=True)
            result = resolver(
                cfg=cfg,
                checkpoint_dir=root,
                manifest=None,  # no manifest
                periodic_prefix=prefix,
                current_hash="h",
            )
            self.assertEqual(result, p150)

    def test_auto_returns_none_when_no_checkpoints_exist(self) -> None:
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_cfg(auto=True)
            result = resolver(
                cfg=cfg,
                checkpoint_dir=Path(tmp),
                manifest=None,
                periodic_prefix="ppo_battalion_s42_c5",
                current_hash="h",
            )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
