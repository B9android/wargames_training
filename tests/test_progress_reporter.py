import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import progress_reporter as pr


class ProgressReporterTests(unittest.TestCase):
    def test_issue_lines_returns_empty_fallback(self) -> None:
        self.assertEqual(pr.issue_lines([], "- None"), "- None")

    def test_milestone_lines_formats_values(self) -> None:
        lines = pr.milestone_lines(
            [{"title": "M1", "open": 2, "closed": 3, "pct": 60, "due": "2026-03-20"}]
        )
        self.assertIn("M1: 3/5 closed (60%), due 2026-03-20", lines)

    def test_deterministic_report_contains_sections(self) -> None:
        context = {
            "week_ending": "2026-03-17",
            "closed_this_week": [],
            "opened_this_week": [],
            "currently_blocked": [],
            "in_progress": [],
            "milestones": [],
            "total_open_issues": 0,
            "dependency_blockers": [],
        }
        report = pr.deterministic_report(context)
        self.assertIn("Weekly Snapshot", report)
        self.assertIn("Dependency Blockers", report)


if __name__ == "__main__":
    unittest.main()
