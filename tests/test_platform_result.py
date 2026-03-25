# SPDX-License-Identifier: MIT
﻿import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_platform.result import ActionStatus, RunResult  # noqa: E402


class PlatformResultTests(unittest.TestCase):
    def test_success_builder(self) -> None:
        result = RunResult.success("triage", False, "apply_labels", "issue #1")
        self.assertTrue(result.ok)
        self.assertEqual(len(result.successes), 1)
        self.assertEqual(result.successes[0].status, ActionStatus.SUCCESS)

    def test_failure_builder(self) -> None:
        result = RunResult.failure("triage", False, "apply_labels", "issue #1", "boom")
        self.assertFalse(result.ok)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].error, "boom")


if __name__ == "__main__":
    unittest.main()

