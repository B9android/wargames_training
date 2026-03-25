# SPDX-License-Identifier: MIT
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import triage_agent as ta


class TriageAgentTests(unittest.TestCase):
    def test_choose_milestone_prefers_m1_for_exp(self) -> None:
        ta.all_milestones = {"M1: 1v1 Competence": 1, "M2: Other": 2}
        self.assertEqual(ta.choose_milestone("[EXP] run"), "M1: 1v1 Competence")

    def test_choose_milestone_falls_back_to_first(self) -> None:
        ta.all_milestones = {"M2: Other": 2}
        self.assertEqual(ta.choose_milestone("[BUG] issue"), "M2: Other")


class EmitStepOutputsTests(unittest.TestCase):
    """Tests for triage_agent.emit_step_outputs() — the GitHub Actions output writer."""

    def _call(self, labels: list[str], issue_number: int) -> dict[str, str]:
        """Call emit_step_outputs with a temp file and return parsed key=value pairs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            output_path = tmp.name
        try:
            ta.emit_step_outputs(labels, issue_number, output_path=output_path)
            with open(output_path, encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if "=" in line]
            return dict(pair.split("=", 1) for pair in lines)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_epic_labels_emit_is_epic_true(self) -> None:
        result = self._call(["type: epic", "priority: high"], issue_number=42)
        self.assertEqual(result["is_epic"], "true")
        self.assertEqual(result["is_experiment"], "false")
        self.assertEqual(result["issue_number"], "42")

    def test_experiment_labels_emit_is_experiment_true(self) -> None:
        result = self._call(["type: experiment", "priority: medium"], issue_number=7)
        self.assertEqual(result["is_epic"], "false")
        self.assertEqual(result["is_experiment"], "true")
        self.assertEqual(result["issue_number"], "7")

    def test_non_epic_non_experiment_labels_emit_false(self) -> None:
        result = self._call(["type: bug", "priority: high"], issue_number=99)
        self.assertEqual(result["is_epic"], "false")
        self.assertEqual(result["is_experiment"], "false")
        self.assertEqual(result["issue_number"], "99")

    def test_pre_filter_labels_respected_even_if_not_in_repo(self) -> None:
        """is_epic must be true even when the epic label would be filtered out downstream."""
        # Simulate: triage classified as epic but repo doesn't have 'type: epic' yet.
        classified = ["type: epic", "priority: high"]
        result = self._call(classified, issue_number=10)
        self.assertEqual(result["is_epic"], "true")

    def test_no_output_path_and_no_env_var_is_noop(self) -> None:
        """emit_step_outputs must be silent when neither arg nor env var is set."""
        orig = os.environ.pop("GITHUB_OUTPUT", None)
        try:
            # Should not raise even though there is nowhere to write.
            ta.emit_step_outputs(["type: epic"], 1)
        finally:
            if orig is not None:
                os.environ["GITHUB_OUTPUT"] = orig


if __name__ == "__main__":
    unittest.main()
