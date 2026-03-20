# tests/test_policy_registry.py
"""Tests for training/policy_registry.py (E3.6)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.policy_registry import Echelon, PolicyEntry, PolicyRegistry, main


# ---------------------------------------------------------------------------
# TestEchelon
# ---------------------------------------------------------------------------


class TestEchelon(unittest.TestCase):
    """Tests for the Echelon enum."""

    def test_values(self) -> None:
        self.assertEqual(Echelon.BATTALION.value, "battalion")
        self.assertEqual(Echelon.BRIGADE.value, "brigade")
        self.assertEqual(Echelon.DIVISION.value, "division")

    def test_from_str_lowercase(self) -> None:
        self.assertIs(Echelon.from_str("battalion"), Echelon.BATTALION)
        self.assertIs(Echelon.from_str("brigade"), Echelon.BRIGADE)
        self.assertIs(Echelon.from_str("division"), Echelon.DIVISION)

    def test_from_str_uppercase(self) -> None:
        self.assertIs(Echelon.from_str("BATTALION"), Echelon.BATTALION)
        self.assertIs(Echelon.from_str("Brigade"), Echelon.BRIGADE)

    def test_from_str_invalid(self) -> None:
        with self.assertRaises(ValueError):
            Echelon.from_str("corps")


# ---------------------------------------------------------------------------
# TestPolicyEntry
# ---------------------------------------------------------------------------


class TestPolicyEntry(unittest.TestCase):
    """Tests for PolicyEntry."""

    def test_fields(self) -> None:
        entry = PolicyEntry(
            echelon="battalion",
            version="v2_final",
            path="/tmp/mappo.pt",
            run_id="run-abc",
        )
        self.assertEqual(entry.echelon, "battalion")
        self.assertEqual(entry.version, "v2_final")
        self.assertEqual(entry.path, "/tmp/mappo.pt")
        self.assertEqual(entry.run_id, "run-abc")

    def test_fields_no_run_id(self) -> None:
        entry = PolicyEntry(echelon="brigade", version="v1", path="/tmp/ppo.zip", run_id=None)
        self.assertIsNone(entry.run_id)

    def test_str_contains_key_fields(self) -> None:
        entry = PolicyEntry(
            echelon="battalion",
            version="v2_final",
            path="checkpoints/mappo.pt",
            run_id="run-xyz",
        )
        s = str(entry)
        self.assertIn("battalion", s)
        self.assertIn("v2_final", s)
        self.assertIn("checkpoints/mappo.pt", s)
        self.assertIn("run-xyz", s)

    def test_str_no_run_id_shows_dash(self) -> None:
        entry = PolicyEntry(echelon="division", version="v3", path="ckpt.zip", run_id=None)
        self.assertIn("—", str(entry))


# ---------------------------------------------------------------------------
# TestPolicyRegistryInMemory
# ---------------------------------------------------------------------------


class TestPolicyRegistryInMemory(unittest.TestCase):
    """Tests for PolicyRegistry without a backing file."""

    def setUp(self) -> None:
        self.reg = PolicyRegistry(path=None)

    def test_can_save_false(self) -> None:
        self.assertFalse(self.reg.can_save)

    def test_save_raises_without_path(self) -> None:
        with self.assertRaises(ValueError):
            self.reg.save()

    def test_register_and_get(self) -> None:
        entry = self.reg.register("battalion", "v1", "/tmp/ckpt.pt")
        self.assertEqual(entry.echelon, "battalion")
        fetched = self.reg.get(Echelon.BATTALION, "v1")
        self.assertEqual(fetched, entry)

    def test_register_with_run_id(self) -> None:
        entry = self.reg.register("brigade", "v2", "/tmp/ppo.zip", run_id="run-999")
        self.assertEqual(entry.run_id, "run-999")

    def test_register_duplicate_raises(self) -> None:
        self.reg.register("battalion", "v1", "/tmp/a.pt")
        with self.assertRaises(ValueError):
            self.reg.register("battalion", "v1", "/tmp/b.pt")

    def test_register_overwrite(self) -> None:
        self.reg.register("battalion", "v1", "/tmp/a.pt")
        self.reg.register("battalion", "v1", "/tmp/b.pt", overwrite=True)
        entry = self.reg.get("battalion", "v1")
        self.assertEqual(entry.path, "/tmp/b.pt")

    def test_register_echelon_string_normalized(self) -> None:
        """Echelon strings are normalised to lowercase."""
        self.reg.register("BATTALION", "vX", "/tmp/x.pt")
        entry = self.reg.get("battalion", "vX")
        self.assertEqual(entry.echelon, "battalion")

    def test_get_missing_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.reg.get("battalion", "nonexistent")

    def test_list_empty(self) -> None:
        self.assertEqual(self.reg.list(), [])

    def test_list_all(self) -> None:
        self.reg.register("battalion", "v1", "/tmp/a.pt")
        self.reg.register("brigade", "v1", "/tmp/b.zip")
        entries = self.reg.list()
        self.assertEqual(len(entries), 2)

    def test_list_filter_echelon(self) -> None:
        self.reg.register("battalion", "v1", "/tmp/a.pt")
        self.reg.register("brigade", "v1", "/tmp/b.zip")
        self.reg.register("division", "v1", "/tmp/c.zip")
        self.assertEqual(len(self.reg.list("battalion")), 1)
        self.assertEqual(len(self.reg.list(Echelon.BRIGADE)), 1)

    def test_list_filter_invalid_echelon(self) -> None:
        with self.assertRaises(ValueError):
            self.reg.list("invalid_echelon")

    def test_remove(self) -> None:
        self.reg.register("battalion", "v1", "/tmp/a.pt")
        self.reg.remove("battalion", "v1")
        self.assertEqual(self.reg.list(), [])

    def test_remove_missing_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.reg.remove("battalion", "nope")

    def test_list_returns_copy(self) -> None:
        """Mutating the returned list must not affect the registry."""
        self.reg.register("battalion", "v1", "/tmp/a.pt")
        lst = self.reg.list()
        lst.clear()
        self.assertEqual(len(self.reg.list()), 1)


# ---------------------------------------------------------------------------
# TestPolicyRegistryPersistence
# ---------------------------------------------------------------------------


class TestPolicyRegistryPersistence(unittest.TestCase):
    """Tests for PolicyRegistry JSON persistence."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "policy_registry.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_reg(self) -> PolicyRegistry:
        return PolicyRegistry(path=self._path)

    def test_can_save_true(self) -> None:
        self.assertTrue(self._make_reg().can_save)

    def test_save_creates_file(self) -> None:
        reg = self._make_reg()
        reg.register("battalion", "v1", "/tmp/a.pt")
        reg.save()
        self.assertTrue(self._path.exists())

    def test_round_trip(self) -> None:
        reg1 = self._make_reg()
        reg1.register("battalion", "v1", "/tmp/a.pt", run_id="run-001")
        reg1.register("brigade", "v2", "/tmp/b.zip")
        reg1.save()

        reg2 = self._make_reg()
        entries = reg2.list()
        self.assertEqual(len(entries), 2)
        batt = reg2.get("battalion", "v1")
        self.assertEqual(batt.path, "/tmp/a.pt")
        self.assertEqual(batt.run_id, "run-001")

    def test_save_json_schema(self) -> None:
        reg = self._make_reg()
        reg.register("division", "v3", "/tmp/c.zip", run_id="run-xyz")
        reg.save()
        with open(self._path) as fh:
            data = json.load(fh)
        self.assertIn("entries", data)
        self.assertEqual(len(data["entries"]), 1)
        self.assertEqual(data["entries"][0]["echelon"], "division")
        self.assertEqual(data["entries"][0]["version"], "v3")
        self.assertEqual(data["entries"][0]["run_id"], "run-xyz")

    def test_load_invalid_json_raises(self) -> None:
        self._path.write_text("not-json", encoding="utf-8")
        with self.assertRaises(ValueError):
            PolicyRegistry(path=self._path)

    def test_load_malformed_entry_raises(self) -> None:
        data = {"entries": [{"echelon": "battalion"}]}  # missing version/path
        self._path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            PolicyRegistry(path=self._path)

    def test_save_creates_parent_dirs(self) -> None:
        nested_path = Path(self._tmp.name) / "subdir" / "nested" / "reg.json"
        reg = PolicyRegistry(path=nested_path)
        reg.register("brigade", "v1", "/tmp/b.zip")
        reg.save()
        self.assertTrue(nested_path.exists())

    def test_overwrite_persists(self) -> None:
        reg1 = self._make_reg()
        reg1.register("battalion", "v1", "/tmp/old.pt")
        reg1.save()

        reg2 = self._make_reg()
        reg2.register("battalion", "v1", "/tmp/new.pt", overwrite=True)
        reg2.save()

        reg3 = self._make_reg()
        self.assertEqual(reg3.get("battalion", "v1").path, "/tmp/new.pt")


# ---------------------------------------------------------------------------
# TestPolicyRegistryLoad
# ---------------------------------------------------------------------------


class TestPolicyRegistryLoad(unittest.TestCase):
    """Tests for PolicyRegistry.load()."""

    def setUp(self) -> None:
        self.reg = PolicyRegistry(path=None)
        # Inject a mock for training.utils.freeze_policy to avoid torch dependency.
        self._mock_freeze_module = MagicMock()
        self._mock_freeze_mappo = MagicMock(return_value="mappo_policy")
        self._mock_freeze_sb3 = MagicMock(return_value="sb3_model")
        self._mock_freeze_module.load_and_freeze_mappo = self._mock_freeze_mappo
        self._mock_freeze_module.load_and_freeze_sb3 = self._mock_freeze_sb3
        self._sys_modules_patcher = patch.dict(
            sys.modules,
            {"training.utils.freeze_policy": self._mock_freeze_module},
        )
        self._sys_modules_patcher.start()

    def tearDown(self) -> None:
        self._sys_modules_patcher.stop()

    def test_load_missing_entry_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.reg.load("battalion", "nonexistent")

    def test_load_battalion_missing_kwargs_raises(self) -> None:
        self.reg.register("battalion", "v1", "/tmp/fake.pt")
        with self.assertRaises(ValueError) as ctx:
            self.reg.load("battalion", "v1")
        self.assertIn("obs_dim", str(ctx.exception))

    def test_load_brigade_dispatches_sb3(self) -> None:
        """load() for brigade dispatches to load_and_freeze_sb3, not load_and_freeze_mappo."""
        self.reg.register("brigade", "v1", "/tmp/ppo.zip")
        result = self.reg.load("brigade", "v1")

        self._mock_freeze_sb3.assert_called_once_with(
            checkpoint_path=Path("/tmp/ppo.zip"), device="cpu"
        )
        self._mock_freeze_mappo.assert_not_called()
        self.assertEqual(result, "sb3_model")

    def test_load_division_dispatches_sb3(self) -> None:
        """load() for division dispatches to load_and_freeze_sb3, not load_and_freeze_mappo."""
        self.reg.register("division", "v1", "/tmp/div.zip")
        result = self.reg.load("division", "v1")

        self._mock_freeze_sb3.assert_called_once_with(
            checkpoint_path=Path("/tmp/div.zip"), device="cpu"
        )
        self._mock_freeze_mappo.assert_not_called()
        self.assertEqual(result, "sb3_model")

    def test_load_battalion_calls_load_and_freeze_mappo(self) -> None:
        """load() for battalion dispatches to load_and_freeze_mappo with correct kwargs."""
        self.reg.register("battalion", "v1", "/tmp/fake.pt")
        result = self.reg.load(
            "battalion",
            "v1",
            obs_dim=10,
            action_dim=3,
            state_dim=15,
        )

        self._mock_freeze_mappo.assert_called_once_with(
            checkpoint_path=Path("/tmp/fake.pt"),
            device="cpu",
            obs_dim=10,
            action_dim=3,
            state_dim=15,
        )
        self._mock_freeze_sb3.assert_not_called()
        self.assertEqual(result, "mappo_policy")

    def test_load_custom_device(self) -> None:
        """device kwarg is forwarded to the underlying loader."""
        self.reg.register("brigade", "v1", "/tmp/b.zip")
        self.reg.load("brigade", "v1", device="cuda")
        self._mock_freeze_sb3.assert_called_once_with(
            checkpoint_path=Path("/tmp/b.zip"), device="cuda"
        )


# ---------------------------------------------------------------------------
# TestPolicyRegistryCLI
# ---------------------------------------------------------------------------


class TestPolicyRegistryCLI(unittest.TestCase):
    """Tests for the CLI entry point."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._reg_path = str(Path(self._tmp.name) / "test_registry.json")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_empty(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--registry", self._reg_path, "list"])
        self.assertIn("No policies registered", buf.getvalue())

    def test_list_with_echelon_filter_empty(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--registry", self._reg_path, "list", "--echelon", "battalion"])
        self.assertIn("No policies", buf.getvalue())
        self.assertIn("battalion", buf.getvalue())

    def test_register_and_list(self) -> None:
        main([
            "--registry", self._reg_path,
            "register",
            "--echelon", "battalion",
            "--version", "v1",
            "--path", "/tmp/ckpt.pt",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--registry", self._reg_path, "list"])
        output = buf.getvalue()
        self.assertIn("battalion", output)
        self.assertIn("v1", output)
        self.assertIn("/tmp/ckpt.pt", output)

    def test_register_creates_json_file(self) -> None:
        main([
            "--registry", self._reg_path,
            "register",
            "--echelon", "brigade",
            "--version", "v2",
            "--path", "/tmp/ppo.zip",
            "--run-id", "run-abc",
        ])
        self.assertTrue(Path(self._reg_path).exists())

    def test_register_run_id_persisted(self) -> None:
        main([
            "--registry", self._reg_path,
            "register",
            "--echelon", "division",
            "--version", "v3",
            "--path", "/tmp/div.zip",
            "--run-id", "run-xyz",
        ])
        reg = PolicyRegistry(path=self._reg_path)
        entry = reg.get("division", "v3")
        self.assertEqual(entry.run_id, "run-xyz")

    def test_list_shows_header(self) -> None:
        main([
            "--registry", self._reg_path,
            "register",
            "--echelon", "battalion",
            "--version", "v1",
            "--path", "/tmp/a.pt",
        ])
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--registry", self._reg_path, "list"])
        output = buf.getvalue()
        self.assertIn("Echelon", output)
        self.assertIn("Version", output)
        self.assertIn("W&B Run ID", output)

    def test_register_overwrite_flag(self) -> None:
        main([
            "--registry", self._reg_path,
            "register", "--echelon", "battalion", "--version", "v1",
            "--path", "/tmp/old.pt",
        ])
        main([
            "--registry", self._reg_path,
            "register", "--echelon", "battalion", "--version", "v1",
            "--path", "/tmp/new.pt", "--overwrite",
        ])
        reg = PolicyRegistry(path=self._reg_path)
        self.assertEqual(reg.get("battalion", "v1").path, "/tmp/new.pt")

    def test_missing_command_exits(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--registry", self._reg_path])

    def test_list_filters_by_echelon(self) -> None:
        for echelon in ("battalion", "brigade", "division"):
            main([
                "--registry", self._reg_path,
                "register", "--echelon", echelon, "--version", "v1",
                "--path", f"/tmp/{echelon}.pt",
            ])

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--registry", self._reg_path, "list", "--echelon", "brigade"])
        output = buf.getvalue()
        self.assertIn("brigade", output)
        # battalion and division should NOT appear (filtered out)
        self.assertNotIn("battalion", output)
        self.assertNotIn("division", output)


try:
    import stable_baselines3 as _sb3  # noqa: F401
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False


# ---------------------------------------------------------------------------
# TestEvaluateCLIIntegration
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_SB3, "stable_baselines3 not installed")
class TestEvaluateCLIIntegration(unittest.TestCase):
    """Test --battalion-policy / --brigade-policy / --division-policy flags
    on the evaluate.py CLI."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._reg_path = str(Path(self._tmp.name) / "test_registry.json")
        # Pre-populate the registry with a dummy battalion entry
        reg = PolicyRegistry(path=self._reg_path)
        # We point to a non-existent checkpoint so we can test the resolution
        # logic without actually running an episode.
        reg.register("battalion", "v1", "/tmp/fake_battalion.pt", run_id="run-001")
        reg.register("brigade", "v1", "/tmp/fake_brigade.zip")
        reg.register("division", "v1", "/tmp/fake_division.zip")
        reg.save()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_battalion_policy_without_registry_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main([
                "--battalion-policy", "v1",
                "--n-episodes", "1",
                "--checkpoint", "/tmp/dummy.zip",
            ])

    def test_brigade_policy_without_registry_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main([
                "--brigade-policy", "v1",
                "--n-episodes", "1",
                "--checkpoint", "/tmp/dummy.zip",
            ])

    def test_division_policy_without_registry_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main([
                "--division-policy", "v1",
                "--n-episodes", "1",
                "--checkpoint", "/tmp/dummy.zip",
            ])

    def test_missing_checkpoint_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main(["--n-episodes", "1"])

    def test_battalion_policy_unknown_version_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main([
                "--policy-registry", self._reg_path,
                "--battalion-policy", "nonexistent_version",
                "--n-episodes", "1",
                "--checkpoint", "/tmp/dummy.zip",
            ])

    def test_brigade_policy_unknown_version_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main([
                "--policy-registry", self._reg_path,
                "--brigade-policy", "nonexistent_version",
                "--n-episodes", "1",
                "--checkpoint", "/tmp/dummy.zip",
            ])

    def test_division_policy_unknown_version_exits(self) -> None:
        from training.evaluate import main as eval_main

        with self.assertRaises(SystemExit):
            eval_main([
                "--policy-registry", self._reg_path,
                "--division-policy", "nonexistent_version",
                "--n-episodes", "1",
                "--checkpoint", "/tmp/dummy.zip",
            ])


if __name__ == "__main__":
    unittest.main()
