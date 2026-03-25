# SPDX-License-Identifier: MIT
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

    def test_register_allows_same_path_with_different_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            best_path = Path(tmp) / "best_model.zip"
            best_path.write_text("x", encoding="utf-8")

            manifest = CheckpointManifest(manifest_path)
            first = manifest.register(
                best_path,
                artifact_type="best_alias",
                seed=1,
                curriculum_level=1,
                run_id="abc",
                config_hash="hash",
                step=100,
            )
            second = manifest.register(
                best_path,
                artifact_type="best_alias",
                seed=1,
                curriculum_level=1,
                run_id="abc",
                config_hash="hash",
                step=200,
            )

            self.assertTrue(first)
            self.assertTrue(second)

    def test_latest_entry_for_path_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            best_path = Path(tmp) / "best_model.zip"
            best_path.write_text("x", encoding="utf-8")

            manifest = CheckpointManifest(manifest_path)
            manifest.register(
                best_path,
                artifact_type="best_alias",
                seed=1,
                curriculum_level=1,
                run_id="abc",
                config_hash="hash-a",
                step=100,
            )
            manifest.register(
                best_path,
                artifact_type="best_alias",
                seed=2,
                curriculum_level=3,
                run_id="abc",
                config_hash="hash-b",
                step=200,
            )

            latest = manifest.latest_entry_for_path(best_path)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["step"], 200)
            self.assertEqual(latest["seed"], 2)

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

    def test_auto_logs_seed_and_curriculum_mismatch_warning(self) -> None:
        resolver = self._import_resolver()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = "ppo_battalion_s42_c5"
            checkpoint = root / f"{prefix}_100_steps.zip"
            checkpoint.write_text("x", encoding="utf-8")

            manifest = CheckpointManifest(root / "manifest.jsonl")
            manifest.register(
                checkpoint,
                artifact_type="periodic",
                seed=7,
                curriculum_level=2,
                run_id="r",
                config_hash="h",
                step=100,
            )

            from omegaconf import OmegaConf

            cfg = OmegaConf.create(
                {
                    "resume": {"auto": True, "checkpoint": None},
                    "training": {"seed": 42},
                    "env": {"curriculum_level": 5},
                }
            )
            with self.assertLogs("training.train", level="WARNING") as captured:
                result = resolver(
                    cfg=cfg,
                    checkpoint_dir=root,
                    manifest=manifest,
                    periodic_prefix=prefix,
                    current_hash="h",
                )

            self.assertEqual(result, checkpoint)
            joined = "\n".join(captured.output)
            self.assertIn("Seed mismatch", joined)
            self.assertIn("Curriculum mismatch", joined)


class TestCheckpointManifestPruning(unittest.TestCase):
    """Tests for prune_periodic() and prune_self_play_snapshots()."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _make_periodic(
        self,
        root: Path,
        prefix: str,
        steps: list[int],
        manifest: CheckpointManifest,
        *,
        seed: int = 42,
        curriculum_level: int = 5,
    ) -> list[Path]:
        """Write fake periodic checkpoint zips and register them."""
        paths: list[Path] = []
        for step in steps:
            p = root / f"{prefix}_{step}_steps.zip"
            p.write_text("x", encoding="utf-8")
            manifest.register(
                p,
                artifact_type="periodic",
                seed=seed,
                curriculum_level=curriculum_level,
                run_id="r",
                config_hash="h",
                step=step,
            )
            paths.append(p)
        return paths

    def _make_snapshots(
        self,
        root: Path,
        steps: list[int],
        manifest: CheckpointManifest,
    ) -> list[Path]:
        """Write fake self-play snapshot zips and register them."""
        paths: list[Path] = []
        for step in steps:
            p = root / f"snapshot_v{step}.zip"
            p.write_text("x", encoding="utf-8")
            manifest.register(
                p,
                artifact_type="self_play_snapshot",
                seed=0,
                curriculum_level=5,
                run_id="r",
                config_hash="h",
                step=step,
            )
            paths.append(p)
        return paths

    # ------------------------------------------------------------------ #
    # prune_periodic tests                                                 #
    # ------------------------------------------------------------------ #

    def test_prune_periodic_keeps_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            prefix = "ppo_battalion_s42_c5"
            paths = self._make_periodic(root, prefix, [100, 200, 300, 400, 500], manifest)

            deleted = manifest.prune_periodic(root, prefix, keep_last=3)

            self.assertEqual(len(deleted), 2)
            # The two oldest (100, 200) must be gone.
            self.assertFalse(paths[0].exists())  # step 100
            self.assertFalse(paths[1].exists())  # step 200
            # The three newest must still exist.
            for p in paths[2:]:
                self.assertTrue(p.exists(), f"Expected {p} to survive pruning")

    def test_prune_periodic_noop_when_keep_exceeds_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            prefix = "ppo_battalion_s42_c5"
            paths = self._make_periodic(root, prefix, [100, 200], manifest)

            deleted = manifest.prune_periodic(root, prefix, keep_last=10)

            self.assertEqual(deleted, [])
            for p in paths:
                self.assertTrue(p.exists())

    def test_prune_periodic_ignores_missing_files(self) -> None:
        """Paths registered in the manifest but deleted externally are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            prefix = "ppo_battalion_s42_c5"
            paths = self._make_periodic(root, prefix, [100, 200, 300], manifest)
            # Delete step-100 externally before pruning.
            paths[0].unlink()

            deleted = manifest.prune_periodic(root, prefix, keep_last=1)

            # Only steps 200 and 300 are visible; keep_last=1 → delete step-200.
            self.assertEqual(len(deleted), 1)
            self.assertFalse(paths[1].exists())  # step 200 pruned
            self.assertTrue(paths[2].exists())   # step 300 kept

    def test_prune_periodic_ignores_other_prefixes(self) -> None:
        """Files from a different prefix must not be pruned."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            # Register two prefixes.
            target = "ppo_battalion_s42_c5"
            other = "ppo_battalion_s7_c3"
            self._make_periodic(root, target, [100, 200, 300], manifest)
            other_paths = self._make_periodic(root, other, [100, 200, 300], manifest,
                                              seed=7, curriculum_level=3)

            manifest.prune_periodic(root, target, keep_last=1)

            # Other-prefix files must be untouched.
            for p in other_paths:
                self.assertTrue(p.exists(), f"Other-prefix file {p} was wrongly deleted")

    # ------------------------------------------------------------------ #
    # prune_self_play_snapshots tests                                      #
    # ------------------------------------------------------------------ #

    def test_prune_snapshots_keeps_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            paths = self._make_snapshots(root, [50, 100, 150, 200, 250], manifest)

            deleted = manifest.prune_self_play_snapshots(root, keep_last=2)

            self.assertEqual(len(deleted), 3)
            for p in paths[:3]:
                self.assertFalse(p.exists(), f"Expected {p} to be pruned")
            for p in paths[3:]:
                self.assertTrue(p.exists(), f"Expected {p} to survive")

    def test_prune_snapshots_noop_when_keep_exceeds_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            paths = self._make_snapshots(root, [50, 100], manifest)

            deleted = manifest.prune_self_play_snapshots(root, keep_last=5)

            self.assertEqual(deleted, [])
            for p in paths:
                self.assertTrue(p.exists())

    def test_prune_snapshots_empty_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = CheckpointManifest(root / "manifest.jsonl")
            deleted = manifest.prune_self_play_snapshots(root, keep_last=3)
            self.assertEqual(deleted, [])


if __name__ == "__main__":
    unittest.main()
