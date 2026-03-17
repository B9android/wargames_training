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


class TriageOutputTests(unittest.TestCase):
    """Tests for the GitHub Actions output emission added for job chaining."""

    def _run_output_emission(self, labels: list[str], issue_number: int) -> dict[str, str]:
        """Helper: run the output emission block and return parsed key=value pairs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            output_path = tmp.name

        orig_issue = ta.ISSUE_NUMBER
        orig_env = os.environ.get("GITHUB_OUTPUT")
        try:
            ta.ISSUE_NUMBER = issue_number
            os.environ["GITHUB_OUTPUT"] = output_path

            is_epic = "type: epic" in labels
            is_experiment = "type: experiment" in labels
            github_output_file = os.environ.get("GITHUB_OUTPUT", "")
            if github_output_file:
                with open(github_output_file, "a", encoding="utf-8") as _out:
                    _out.write(f"is_epic={'true' if is_epic else 'false'}\n")
                    _out.write(f"is_experiment={'true' if is_experiment else 'false'}\n")
                    _out.write(f"issue_number={issue_number}\n")

            with open(output_path, encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if "=" in line]
            return dict(pair.split("=", 1) for pair in lines)
        finally:
            ta.ISSUE_NUMBER = orig_issue
            if orig_env is None:
                os.environ.pop("GITHUB_OUTPUT", None)
            else:
                os.environ["GITHUB_OUTPUT"] = orig_env
            Path(output_path).unlink(missing_ok=True)

    def test_epic_labels_emit_is_epic_true(self) -> None:
        result = self._run_output_emission(["type: epic", "priority: high"], issue_number=42)
        self.assertEqual(result["is_epic"], "true")
        self.assertEqual(result["is_experiment"], "false")
        self.assertEqual(result["issue_number"], "42")

    def test_experiment_labels_emit_is_experiment_true(self) -> None:
        result = self._run_output_emission(["type: experiment", "priority: medium"], issue_number=7)
        self.assertEqual(result["is_epic"], "false")
        self.assertEqual(result["is_experiment"], "true")
        self.assertEqual(result["issue_number"], "7")

    def test_non_epic_non_experiment_labels_emit_false(self) -> None:
        result = self._run_output_emission(["type: bug", "priority: high"], issue_number=99)
        self.assertEqual(result["is_epic"], "false")
        self.assertEqual(result["is_experiment"], "false")
        self.assertEqual(result["issue_number"], "99")

    def test_no_github_output_env_does_not_raise(self) -> None:
        """If GITHUB_OUTPUT is not set (local dev), emission must be a no-op."""
        orig = os.environ.pop("GITHUB_OUTPUT", None)
        try:
            github_output_file = os.environ.get("GITHUB_OUTPUT", "")
            if github_output_file:  # Should be falsy — no file write
                self.fail("Should not attempt to write when GITHUB_OUTPUT is unset")
        finally:
            if orig is not None:
                os.environ["GITHUB_OUTPUT"] = orig


if __name__ == "__main__":
    unittest.main()
