# SPDX-License-Identifier: MIT
﻿import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_platform.result import RunResult  # noqa: E402
from agent_platform.summary import ExecutionSummary  # noqa: E402


class PlatformSummaryTests(unittest.TestCase):
    def test_render_and_artifacts(self) -> None:
        summary = ExecutionSummary("triage", "issue #1", dry_run=True)
        summary.checkpoint("start", "begin")
        summary.decision("rule-based triage")
        summary.set_result(RunResult.success("triage", True, "apply_labels", "issue #1"))

        terminal = summary.render_terminal()
        comment = summary.render_github_comment()

        self.assertIn("AGENT EXECUTION REPORT", terminal)
        self.assertIn("rule-based triage", terminal)
        self.assertIn("rule-based triage", comment)

        with tempfile.TemporaryDirectory() as td:
            json_path, md_path = summary.write_artifacts(td)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()

