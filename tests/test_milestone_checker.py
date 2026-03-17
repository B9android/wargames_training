import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

    def add_to_labels(self, label: str) -> None:
        self.added_labels.append(label)

    def create_comment(self, message: str) -> None:
        self.comments.append(message)


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
    def __init__(self, open_issues: list[_Issue], agent_issues: list[_Issue], milestone_issues: list[_Issue]) -> None:
        self._open_issues = open_issues
        self._agent_issues = agent_issues
        self._milestone_issues = milestone_issues
        self.created: list[dict[str, object]] = []

    def get_issues(self, *, state: str, labels=None, milestone=None):
        if labels is not None:
            return list(self._agent_issues)
        if milestone is not None:
            return list(self._milestone_issues)
        return list(self._open_issues)

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


if __name__ == "__main__":
    unittest.main()
