import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import release_coordinator as rc


class _Milestone:
    def __init__(self, title: str) -> None:
        self.title = title


class _Issue:
    def __init__(self, number: int, title: str) -> None:
        self.number = number
        self.title = title


class ReleaseCoordinatorTests(unittest.TestCase):
    def test_issue_lines_returns_none_when_empty(self) -> None:
        self.assertEqual(rc.issue_lines([]), "- None")

    def test_issue_lines_formats_issue_bullets(self) -> None:
        lines = rc.issue_lines([_Issue(1, "A"), _Issue(2, "B")])
        self.assertIn("- #1 A", lines)
        self.assertIn("- #2 B", lines)

    def test_build_changelog_block_contains_marker_and_signature(self) -> None:
        block = rc.build_changelog_block(
            _Milestone("M1: 1v1 Competence"),
            [_Issue(3, "Finish rollout")],
            "<!-- marker -->",
            "2026-03-17",
        )
        self.assertIn("## [M1: 1v1 Competence] - 2026-03-17", block)
        self.assertIn("<!-- marker -->", block)
        self.assertIn("Chronicler Release", block)

    def test_insert_roadmap_update_inserts_after_first_separator(self) -> None:
        roadmap = "Header\n\n---\n\nRest"
        inserted = rc.insert_roadmap_update(roadmap, "UPDATE\n")
        self.assertIn("---\n\nUPDATE\nRest", inserted)

    def test_insert_roadmap_update_prepends_when_separator_missing(self) -> None:
        roadmap = "No separator roadmap"
        inserted = rc.insert_roadmap_update(roadmap, "UPDATE\n")
        self.assertTrue(inserted.startswith("UPDATE\n"))

    def test_already_synced_requires_marker_in_both_docs(self) -> None:
        marker = "<!-- agent:release-sync:12 -->"
        self.assertTrue(rc.already_synced(f"x {marker}", f"y {marker}", marker))
        self.assertFalse(rc.already_synced(f"x {marker}", "y", marker))
        self.assertFalse(rc.already_synced("x", f"y {marker}", marker))

    def test_sync_release_docs_contents_noop_when_marker_already_in_both(self) -> None:
        marker = "<!-- agent:release-sync:5 -->"
        changelog = f"Header\n{marker}\n"
        roadmap = f"Roadmap\n{marker}\n"

        updated_changelog, updated_roadmap, has_changes = rc.sync_release_docs_contents(
            changelog,
            roadmap,
            milestone=_Milestone("M1"),
            closed_issues=[_Issue(1, "Done")],
            marker=marker,
            run_date="2026-03-17",
        )

        self.assertFalse(has_changes)
        self.assertEqual(updated_changelog, changelog)
        self.assertEqual(updated_roadmap, roadmap)

    def test_sync_release_docs_contents_updates_when_marker_missing(self) -> None:
        marker = "<!-- agent:release-sync:9 -->"
        changelog = "# Changelog\n"
        roadmap = "# Roadmap\n\n---\n\nBody\n"

        updated_changelog, updated_roadmap, has_changes = rc.sync_release_docs_contents(
            changelog,
            roadmap,
            milestone=_Milestone("M2"),
            closed_issues=[_Issue(2, "Ship feature")],
            marker=marker,
            run_date="2026-03-17",
        )

        self.assertTrue(has_changes)
        self.assertIn("## [M2] - 2026-03-17", updated_changelog)
        self.assertIn(marker, updated_changelog)
        self.assertIn("Milestone `M2` closed", updated_roadmap)
        self.assertIn(marker, updated_roadmap)

    @patch("release_coordinator.require_env")
    @patch("release_coordinator.Github")
    @patch("release_coordinator.Path.read_text")
    @patch("release_coordinator.Path.write_text")
    @patch("release_coordinator.log_event")
    def test_main_writes_files_when_marker_missing(
        self, mock_log, mock_write, mock_read, mock_github, mock_require_env
    ) -> None:
        """Integration test: main() updates files when release marker is missing."""
        # Setup environment variables
        mock_require_env.return_value = {
            "REPO_NAME": "B9android/wargames_training",
            "GITHUB_TOKEN": "fake-token-123",
            "MILESTONE_NUMBER": "5",
        }

        # Setup GitHub API mocks
        mock_gh = MagicMock()
        mock_github.return_value = mock_gh

        mock_milestone = MagicMock()
        mock_milestone.title = "M1: 1v1 Competence"
        mock_milestone.state = "closed"
        mock_milestone.number = 5

        mock_repo = MagicMock()
        mock_repo.get_milestone.return_value = mock_milestone
        mock_gh.get_repo.return_value = mock_repo

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.title = "Fix critic baseline"
        mock_issue.pull_request = None
        mock_repo.get_issues.return_value = [mock_issue]

        # Setup file system mocks
        initial_changelog = "# Changelog\n"
        initial_roadmap = "# Roadmap\n\n---\n\nRelease Notes\n"
        mock_read.side_effect = [initial_changelog, initial_roadmap]

        # Run main()
        rc.main()

        # Verify require_env was called correctly
        mock_require_env.assert_called_once()

        # Verify Github repo was fetched
        mock_gh.get_repo.assert_called_once_with("B9android/wargames_training")

        # Verify milestone was fetched
        mock_repo.get_milestone.assert_called_once_with(5)

        # Verify files were written (not skipped)
        self.assertEqual(mock_write.call_count, 2)

        # Get the written content
        write_calls = mock_write.call_args_list
        updated_changelog = write_calls[0][0][0]
        updated_roadmap = write_calls[1][0][0]

        # Verify marker is in both files
        marker = "<!-- agent:release-sync:5 -->"
        self.assertIn(marker, updated_changelog)
        self.assertIn(marker, updated_roadmap)

        # Verify content includes issue and milestone info
        self.assertIn("#42 Fix critic baseline", updated_changelog)
        self.assertIn("M1: 1v1 Competence", updated_changelog)

        # Verify log_event was called for completion
        self.assertEqual(mock_log.call_count, 1)
        log_call = mock_log.call_args
        self.assertEqual(log_call[0][0], "release_sync_complete")
        self.assertEqual(log_call[1]["milestone_number"], 5)
        self.assertEqual(log_call[1]["closed_issue_count"], 1)

    @patch("release_coordinator.require_env")
    @patch("release_coordinator.Github")
    @patch("release_coordinator.Path.read_text")
    @patch("release_coordinator.Path.write_text")
    @patch("release_coordinator.log_event")
    def test_main_skips_when_marker_already_present(
        self, mock_log, mock_write, mock_read, mock_github, mock_require_env
    ) -> None:
        """Integration test: main() skips write when release marker already present."""
        # Setup environment variables
        mock_require_env.return_value = {
            "REPO_NAME": "B9android/wargames_training",
            "GITHUB_TOKEN": "fake-token-123",
            "MILESTONE_NUMBER": "5",
        }

        # Setup GitHub API mocks
        mock_gh = MagicMock()
        mock_github.return_value = mock_gh

        mock_milestone = MagicMock()
        mock_milestone.title = "M1: 1v1 Competence"
        mock_milestone.state = "closed"
        mock_milestone.number = 5

        mock_repo = MagicMock()
        mock_repo.get_milestone.return_value = mock_milestone
        mock_gh.get_repo.return_value = mock_repo

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.title = "Fix critic baseline"
        mock_issue.pull_request = None
        mock_repo.get_issues.return_value = [mock_issue]

        # Setup file system mocks with marker already present
        marker = "<!-- agent:release-sync:5 -->"
        existing_changelog = f"# Changelog\n{marker}\n"
        existing_roadmap = f"# Roadmap\n{marker}\n"
        mock_read.side_effect = [existing_changelog, existing_roadmap]

        # Run main()
        rc.main()

        # Verify files were NOT written (skipped due to marker already present)
        mock_write.assert_not_called()

        # Verify log_event was called for skip
        self.assertEqual(mock_log.call_count, 1)
        log_call = mock_log.call_args
        self.assertEqual(log_call[0][0], "release_sync_skipped")
        self.assertEqual(log_call[1]["reason"], "already_synced")


if __name__ == "__main__":
    unittest.main()
