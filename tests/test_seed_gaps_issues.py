"""Tests for scripts/project_agent/seed_gaps_issues.py."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import seed_gaps_issues as sgi


class AllIssuesStructureTests(unittest.TestCase):
    """Verify ALL_ISSUES is well-formed without touching GitHub."""

    def test_all_issues_is_nonempty(self) -> None:
        self.assertGreater(len(sgi.ALL_ISSUES), 0)

    def test_every_issue_has_required_keys(self) -> None:
        for issue in sgi.ALL_ISSUES:
            with self.subTest(title=issue.get("title", "<missing>")):
                self.assertIn("title", issue, "Missing 'title' key")
                self.assertIn("labels", issue, "Missing 'labels' key")
                self.assertIn("body", issue, "Missing 'body' key")

    def test_titles_are_unique(self) -> None:
        titles = [i["title"] for i in sgi.ALL_ISSUES]
        self.assertEqual(
            len(titles),
            len(set(titles)),
            "Duplicate issue titles found in ALL_ISSUES",
        )

    def test_every_issue_has_status_agent_created_label(self) -> None:
        for issue in sgi.ALL_ISSUES:
            with self.subTest(title=issue["title"]):
                self.assertIn(
                    "status: agent-created",
                    issue["labels"],
                    "Every seeded issue must carry 'status: agent-created'",
                )

    def test_every_issue_body_contains_attribution(self) -> None:
        for issue in sgi.ALL_ISSUES:
            with self.subTest(title=issue["title"]):
                self.assertIn(
                    "Strategist Forge",
                    issue["body"],
                    "Issue body must include the attribution footer",
                )

    def test_section_lists_are_nonempty(self) -> None:
        sections = [
            sgi.CICD_ISSUES,
            sgi.SECURITY_ISSUES,
            sgi.MODEL_ISSUES,
            sgi.SIM_INTEGRATION_ISSUES,
            sgi.CORRECTNESS_ISSUES,
            sgi.DOCS_ISSUES,
            sgi.ANALYSIS_ISSUES,
            sgi.FRONTEND_ISSUES,
            sgi.TESTING_ISSUES,
        ]
        for section in sections:
            self.assertGreater(len(section), 0, "Found an empty section list")

    def test_all_issues_equals_sum_of_sections(self) -> None:
        sections = [
            sgi.CICD_ISSUES,
            sgi.SECURITY_ISSUES,
            sgi.MODEL_ISSUES,
            sgi.SIM_INTEGRATION_ISSUES,
            sgi.CORRECTNESS_ISSUES,
            sgi.DOCS_ISSUES,
            sgi.ANALYSIS_ISSUES,
            sgi.FRONTEND_ISSUES,
            sgi.TESTING_ISSUES,
        ]
        total = sum(len(s) for s in sections)
        self.assertEqual(
            total,
            len(sgi.ALL_ISSUES),
            "ALL_ISSUES count does not match the sum of individual section lists",
        )


class LabelColorTests(unittest.TestCase):
    def test_known_prefixes_return_non_default_colors(self) -> None:
        for prefix in ("type:", "priority:", "status:", "domain:", "v1:", "v6:"):
            color = sgi._label_color(f"{prefix} something")
            self.assertNotEqual(color, "ededed", f"Expected non-default color for {prefix}")

    def test_unknown_prefix_returns_fallback(self) -> None:
        self.assertEqual(sgi._label_color("unknown: thing"), "ededed")


class CreateIssueLogicTests(unittest.TestCase):
    """Test create_issue() without real GitHub calls."""

    def _make_repo(self):
        repo = MagicMock()
        return repo

    def test_returns_exists_when_title_already_known(self) -> None:
        repo = self._make_repo()
        issue_def = {
            "title": "Already Existing Issue",
            "labels": [],
            "body": "body text",
        }
        known = {"already existing issue"}
        result = sgi.create_issue(repo, issue_def, known, dry_run=False)
        self.assertEqual(result, "exists")
        repo.create_issue.assert_not_called()

    def test_returns_skipped_in_dry_run(self) -> None:
        repo = self._make_repo()
        issue_def = {
            "title": "New Issue Title",
            "labels": [],
            "body": "body text",
        }
        result = sgi.create_issue(repo, issue_def, set(), dry_run=True)
        self.assertEqual(result, "skipped")
        repo.create_issue.assert_not_called()

    def test_returns_created_on_success(self) -> None:
        repo = self._make_repo()
        repo.get_label.return_value = MagicMock()
        repo.create_issue.return_value = MagicMock()
        issue_def = {
            "title": "Brand New Issue",
            "labels": ["type: chore"],
            "body": "body text",
        }
        result = sgi.create_issue(repo, issue_def, set(), dry_run=False)
        self.assertEqual(result, "created")
        repo.create_issue.assert_called_once()

    def test_returns_skipped_on_create_exception(self) -> None:
        repo = self._make_repo()
        repo.get_label.return_value = MagicMock()
        repo.create_issue.side_effect = RuntimeError("API error")
        issue_def = {
            "title": "Failing Issue",
            "labels": ["type: chore"],
            "body": "body text",
        }
        result = sgi.create_issue(repo, issue_def, set(), dry_run=False)
        self.assertEqual(result, "skipped")

    def test_milestone_attached_when_found(self) -> None:
        repo = self._make_repo()
        ms = MagicMock()
        ms.title = "M0: Project Bootstrap"
        repo.get_milestones.return_value = [ms]
        repo.get_label.return_value = MagicMock()
        repo.create_issue.return_value = MagicMock()
        issue_def = {
            "title": "Issue With Milestone",
            "labels": [],
            "body": "body",
            "milestone": "M0: Project Bootstrap",
        }
        sgi.create_issue(repo, issue_def, set(), dry_run=False)
        call_kwargs = repo.create_issue.call_args[1]
        self.assertEqual(call_kwargs["milestone"], ms)

    def test_issue_created_without_milestone_when_not_found(self) -> None:
        repo = self._make_repo()
        repo.get_milestones.return_value = []
        repo.create_milestone.side_effect = RuntimeError("cannot create")
        repo.get_label.return_value = MagicMock()
        repo.create_issue.return_value = MagicMock()
        issue_def = {
            "title": "Issue Without Milestone",
            "labels": [],
            "body": "body",
            "milestone": "M99: Nonexistent",
        }
        result = sgi.create_issue(repo, issue_def, set(), dry_run=False)
        self.assertEqual(result, "created")
        call_kwargs = repo.create_issue.call_args[1]
        self.assertNotIn("milestone", call_kwargs)


class RunFunctionTests(unittest.TestCase):
    """Test the run() entry point without real GitHub calls."""

    def test_run_returns_1_without_env_vars(self) -> None:
        import os
        with patch.dict(os.environ, {}, clear=True):
            result = sgi.run()
            self.assertEqual(result, 1)

    def test_run_returns_1_when_pygithub_missing(self) -> None:
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "github":
                raise ImportError("mocked missing")
            return real_import(name, *args, **kwargs)

        import os
        with patch.dict(
            os.environ,
            {"GITHUB_TOKEN": "fake", "REPO_NAME": "owner/repo"},
        ):
            with patch("builtins.__import__", side_effect=mock_import):
                result = sgi.run()
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
