# SPDX-License-Identifier: MIT
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_create_issue_called_with_list_labels(self) -> None:
        """Regression: repo.create_issue() must receive labels as a list, not a tuple.

        PyGitHub 2.x is_optional_list() requires isinstance(v, list). Passing a tuple
        raised AssertionError, producing the CI error "('type: docs', 'status: agent-created')".
        """
        mock_label_docs = MagicMock()
        mock_label_docs.name = "type: docs"
        mock_label_agent = MagicMock()
        mock_label_agent.name = "status: agent-created"
        mock_created_issue = MagicMock()
        mock_created_issue.number = 99

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = []
        mock_repo.get_milestones.return_value = []
        mock_repo.get_labels.return_value = [mock_label_docs, mock_label_agent]
        mock_repo.create_issue.return_value = mock_created_issue

        mock_gh_instance = MagicMock()
        mock_gh_instance.get_repo.return_value = mock_repo

        mock_ai = MagicMock()
        mock_ai.chat.completions.create.return_value.choices[0].message.content = "AI text"

        env = {
            "REPO_NAME": "test/repo",
            "GITHUB_TOKEN": "fake-token",
            "OPENAI_API_KEY": "fake-key",
            "DRY_RUN": "false",
        }
        with patch.dict("os.environ", env, clear=False), \
                patch("github.Github", return_value=mock_gh_instance), \
                patch("github.Auth.Token", return_value=MagicMock()), \
                patch("openai.OpenAI", return_value=mock_ai):
            pr.main()

        mock_repo.create_issue.assert_called_once()
        _, call_kwargs = mock_repo.create_issue.call_args
        labels_arg = call_kwargs.get("labels")
        self.assertIsInstance(labels_arg, list, "labels passed to create_issue must be a list")
        self.assertEqual(sorted(labels_arg), sorted(pr.REQUIRED_LABELS))

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
