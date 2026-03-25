# SPDX-License-Identifier: MIT
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import milestone_checker as mc


class _Label:
    def __init__(self, name: str) -> None:
        self.name = name


class _Issue:
    def __init__(
        self,
        number: int,
        title: str,
        body: str,
        labels: list[str] | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.number = number
        self.title = title
        self.body = body
        self.labels = [_Label(label) for label in (labels or [])]
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.added_labels: list[str] = []
        self.comments: list[str] = []

    def add_to_labels(self, *labels: str) -> None:
        self.added_labels.extend(labels)

    def create_comment(self, message: str) -> None:
        self.comments.append(message)


_DEFAULT_REPO_LABELS = [
    "type: bug", "type: feature", "type: experiment", "type: research",
    "type: epic", "type: chore",
    "priority: critical", "priority: high", "priority: medium", "priority: low",
    "status: needs-manual-triage", "status: approved", "status: in-progress",
    "status: blocked", "status: stale", "status: complete", "status: failed",
    "status: cancelled", "status: agent-created",
]


class _Milestone:
    def __init__(
        self,
        number: int,
        title: str,
        *,
        due_on: datetime | None,
        open_issues: int,
        closed_issues: int,
        url: str,
    ) -> None:
        self.number = number
        self.title = title
        self.due_on = due_on
        self.open_issues = open_issues
        self.closed_issues = closed_issues
        self.url = url


class _Repo:
    def __init__(
        self,
        open_issues: list[_Issue],
        agent_issues: list[_Issue],
        milestone_issues: list[_Issue],
        repo_labels: list[str] | None = None,
    ) -> None:
        self._open_issues = open_issues
        self._agent_issues = agent_issues
        self._milestone_issues = milestone_issues
        self._repo_labels = repo_labels if repo_labels is not None else _DEFAULT_REPO_LABELS
        self.created: list[dict[str, object]] = []

    def get_issues(self, *, state: str, labels=None, milestone=None):
        if labels is not None:
            return list(self._agent_issues)
        if milestone is not None:
            return list(self._milestone_issues)
        return list(self._open_issues)

    def get_labels(self):
        return [_Label(name) for name in self._repo_labels]

    def create_issue(self, *, title: str, body: str, labels: list[str]):
        number = 900 + len(self.created)
        self.created.append({"number": number, "title": title, "body": body, "labels": labels})
        issue = _Issue(number, title, body, labels=[])
        issue.number = number
        return issue


class MilestoneCheckerTests(unittest.TestCase):
    def test_marker_for_milestone(self) -> None:
        marker = mc.marker_for_milestone(12, "2026-03-17")
        self.assertEqual(marker, "<!-- agent:milestone-risk:12:2026-03-17 -->")

    def test_dependency_blocker_lines_formats_rollup(self) -> None:
        parent = _Issue(1, "Parent", "", labels=[])
        child = _Issue(2, "Child", "Part of #1", labels=["status: blocked"])
        lines = mc.dependency_blocker_lines([parent, child])
        self.assertIn("#1 Parent blocked by: #2 Child", lines)

    def test_maybe_create_at_risk_issue_dry_run_does_not_create(self) -> None:
        now = datetime(2026, 3, 17, tzinfo=timezone.utc)
        milestone = _Milestone(
            1,
            "M1",
            due_on=now + timedelta(days=5),
            open_issues=9,
            closed_issues=1,
            url="https://example.com/m1",
        )
        repo = _Repo(open_issues=[], agent_issues=[], milestone_issues=[])
        created_number = mc.maybe_create_at_risk_issue(repo, milestone, now, dry_run=True)
        self.assertIsNone(created_number)
        self.assertEqual(repo.created, [])

    def test_mark_stale_issues_updates_only_stale_non_in_progress(self) -> None:
        now = datetime(2026, 3, 17, tzinfo=timezone.utc)
        stale_issue = _Issue(
            11,
            "Stale",
            "",
            labels=[],
            updated_at=now - timedelta(days=30),
        )
        in_progress_issue = _Issue(
            12,
            "Active",
            "",
            labels=["status: in-progress"],
            updated_at=now - timedelta(days=30),
        )
        repo = _Repo(open_issues=[stale_issue, in_progress_issue], agent_issues=[], milestone_issues=[])

        mc.mark_stale_issues(repo, now, dry_run=False)

        self.assertEqual(stale_issue.added_labels, ["status: stale"])
        self.assertEqual(len(stale_issue.comments), 1)
        self.assertEqual(in_progress_issue.added_labels, [])

    def test_report_unlabeled_issues_creates_when_threshold_met(self) -> None:
        now = datetime(2026, 3, 17, tzinfo=timezone.utc)
        unlabeled = [_Issue(i, f"Issue {i}", "", labels=[]) for i in range(1, 6)]
        repo = _Repo(open_issues=unlabeled, agent_issues=[], milestone_issues=[])

        created = mc.report_unlabeled_issues(repo, now, dry_run=False)

        self.assertEqual(len(created), 1)
        self.assertEqual(len(repo.created), 1)
        self.assertIn("issues need labels", repo.created[0]["title"])


class LoadRuleMapTests(unittest.TestCase):
    def test_returns_dict_with_bug_marker(self) -> None:
        rule_map = mc.load_rule_map()
        self.assertIn("[BUG]", rule_map)
        self.assertIn("type: bug", rule_map["[BUG]"])

    def test_wip_marker_present(self) -> None:
        rule_map = mc.load_rule_map()
        self.assertIn("[WIP]", rule_map)
        self.assertIn("status: in-progress", rule_map["[WIP]"])

    def test_all_keys_are_uppercase(self) -> None:
        rule_map = mc.load_rule_map()
        for key in rule_map:
            self.assertEqual(key, key.upper(), f"Key '{key}' is not uppercase")

    def test_uses_loaded_yaml_when_available(self) -> None:
        """Ensure load_rule_map actually uses yaml.safe_load, not just the fallback."""
        fake_rule_map = {"[TEST_ONLY]": ["test: from-yaml"]}

        # Patch the yaml module used inside milestone_checker to control safe_load output.
        with patch.object(mc, "yaml", autospec=True) as mock_yaml:
            mock_yaml.safe_load.return_value = fake_rule_map
            rule_map = mc.load_rule_map()

        self.assertIn("[TEST_ONLY]", rule_map)
        self.assertIn("test: from-yaml", rule_map["[TEST_ONLY]"])


class TriageUnlabeledIssuesTests(unittest.TestCase):
    def _make_repo(self, issues: list[_Issue]) -> _Repo:
        return _Repo(open_issues=issues, agent_issues=[], milestone_issues=[])

    def test_empty_list_returns_empty(self) -> None:
        repo = self._make_repo([])
        result = mc.triage_unlabeled_issues(repo, [], dry_run=False)
        self.assertEqual(result, [])

    def test_bug_marker_applies_bug_labels(self) -> None:
        issue = _Issue(10, "[BUG] Something broken", "")
        repo = self._make_repo([issue])
        triaged = mc.triage_unlabeled_issues(repo, [issue], dry_run=False)
        self.assertIn(10, triaged)
        self.assertIn("type: bug", issue.added_labels)
        self.assertIn("priority: high", issue.added_labels)

    def test_wip_marker_applies_in_progress_label(self) -> None:
        issue = _Issue(11, "[WIP] Feature in flight", "")
        repo = self._make_repo([issue])
        mc.triage_unlabeled_issues(repo, [issue], dry_run=False)
        self.assertIn("status: in-progress", issue.added_labels)

    def test_wip_and_bug_markers_combine(self) -> None:
        issue = _Issue(12, "[WIP] [BUG] Fix in progress", "")
        repo = self._make_repo([issue])
        mc.triage_unlabeled_issues(repo, [issue], dry_run=False)
        self.assertIn("type: bug", issue.added_labels)
        self.assertIn("priority: high", issue.added_labels)
        self.assertIn("status: in-progress", issue.added_labels)

    def test_no_marker_applies_needs_manual_triage(self) -> None:
        issue = _Issue(13, "Add some feature", "")
        repo = self._make_repo([issue])
        mc.triage_unlabeled_issues(repo, [issue], dry_run=False)
        self.assertIn("status: needs-manual-triage", issue.added_labels)
        self.assertIn("priority: medium", issue.added_labels)

    def test_triage_posts_comment(self) -> None:
        issue = _Issue(14, "[BUG] Crash on start", "")
        repo = self._make_repo([issue])
        mc.triage_unlabeled_issues(repo, [issue], dry_run=False)
        self.assertEqual(len(issue.comments), 1)
        self.assertIn("Labels applied automatically", issue.comments[0])

    def test_dry_run_does_not_modify_issue(self) -> None:
        issue = _Issue(15, "[BUG] Crash on start", "")
        repo = self._make_repo([issue])
        mc.triage_unlabeled_issues(repo, [issue], dry_run=True)
        self.assertEqual(issue.added_labels, [])
        self.assertEqual(issue.comments, [])

    def test_label_not_in_repo_is_skipped(self) -> None:
        issue = _Issue(16, "[BUG] Something", "")
        repo = _Repo(
            open_issues=[issue],
            agent_issues=[],
            milestone_issues=[],
            repo_labels=[],  # no labels exist
        )
        result = mc.triage_unlabeled_issues(repo, [issue], dry_run=False)
        self.assertEqual(result, [])
        self.assertEqual(issue.added_labels, [])

    def test_returns_triaged_issue_numbers(self) -> None:
        issues = [_Issue(i, f"Issue {i}", "") for i in range(20, 23)]
        repo = self._make_repo(issues)
        result = mc.triage_unlabeled_issues(repo, issues, dry_run=False)
        self.assertEqual(sorted(result), [20, 21, 22])


if __name__ == "__main__":
    unittest.main()
