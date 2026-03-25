# SPDX-License-Identifier: MIT
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from static_analysis_agent import _env_bool, _env_scan_dirs  # noqa: E402  # type: ignore[reportMissingImports]
from static_analysis_agent import _filter_findings_to_changed_files, _normalize_changed_paths  # noqa: E402  # type: ignore[reportMissingImports]


class StaticAnalysisAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_env_bool_parses_truthy_and_falsy_values(self) -> None:
        os.environ["FLAG"] = "true"
        self.assertTrue(_env_bool("FLAG"))

        os.environ["FLAG"] = "0"
        self.assertFalse(_env_bool("FLAG", default=True))

    def test_env_scan_dirs_defaults_and_parsing(self) -> None:
        os.environ.pop("STATIC_ANALYZER_PATHS", None)
        defaults = _env_scan_dirs()
        self.assertIn("scripts", defaults)

        os.environ["STATIC_ANALYZER_PATHS"] = "envs, scripts ,tests"
        parsed = _env_scan_dirs()
        self.assertEqual(parsed, ["envs", "scripts", "tests"])

    def test_normalize_changed_paths_only_keeps_python(self) -> None:
        lines = ["scripts/project_agent/a.py", "README.md", "tests\\test_x.py", ""]
        normalized = _normalize_changed_paths(lines)
        self.assertEqual(normalized, {"scripts/project_agent/a.py", "tests/test_x.py"})

    def test_filter_findings_to_changed_files(self) -> None:
        repo_root = PROJECT_ROOT
        findings = [
            {
                "rule": "dead_private_function",
                "confidence": "high",
                "file": str(PROJECT_ROOT / "scripts" / "project_agent" / "static_analyzer.py"),
                "line": 1,
                "module": "scripts.project_agent.static_analyzer",
                "reason": "x",
            },
            {
                "rule": "unused_module",
                "confidence": "low",
                "file": str(PROJECT_ROOT / "training" / "train.py"),
                "line": 1,
                "module": "training.train",
                "reason": "y",
            },
        ]

        filtered = _filter_findings_to_changed_files(
            findings,
            repo_root,
            {"scripts/project_agent/static_analyzer.py"},
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["module"], "scripts.project_agent.static_analyzer")


if __name__ == "__main__":
    unittest.main()
