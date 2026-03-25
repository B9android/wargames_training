# SPDX-License-Identifier: MIT
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import training_monitor as tm


class _Label:
    def __init__(self, name: str) -> None:
        self.name = name


class _Issue:
    def __init__(self, body: str = "", labels: list[str] | None = None) -> None:
        self.body = body
        self.labels = [_Label(label) for label in (labels or [])]
        self.edited_bodies: list[str] = []
        self.added: list[str] = []
        self.removed: list[str] = []
        self.comments: list[str] = []

    def edit(self, *, body: str) -> None:
        self.body = body
        self.edited_bodies.append(body)

    def add_to_labels(self, label: str) -> None:
        self.added.append(label)

    def remove_from_labels(self, label: str) -> None:
        self.removed.append(label)

    def create_comment(self, message: str) -> None:
        self.comments.append(message)


class _Run:
    def __init__(self, name: str, url: str, state: str) -> None:
        self.name = name
        self.url = url
        self.state = state


class TrainingMonitorTests(unittest.TestCase):
    def test_desired_target_state_maps_run_states(self) -> None:
        self.assertEqual(tm.desired_target_state("running"), "in-progress")
        self.assertEqual(tm.desired_target_state("finished"), "complete")
        self.assertEqual(tm.desired_target_state("failed"), "failed")
        self.assertEqual(tm.desired_target_state("crashed"), "failed")
        self.assertIsNone(tm.desired_target_state("queued"))

    def test_plan_label_changes_adds_target_and_removes_other_statuses(self) -> None:
        labels_to_add, labels_to_remove = tm.plan_label_changes(
            {"status: approved", "status: blocked", "type: experiment"},
            "in-progress",
        )
        self.assertEqual(labels_to_add, ["status: in-progress"])
        self.assertEqual(labels_to_remove, ["status: approved", "status: blocked"])

    def test_update_issue_outcome_updates_body_when_section_present(self) -> None:
        issue = _Issue(body="### Outcome\nRunning\n\n### Results\nPending")
        tm.update_issue_outcome(
            issue,
            "finished",
            run_failed=False,
            dry_run=False,
            issue_number=1,
        )
        self.assertEqual(len(issue.edited_bodies), 1)
        self.assertIn("Success - hypothesis confirmed", issue.body)

    def test_update_issue_outcome_noop_when_section_missing(self) -> None:
        issue = _Issue(body="No outcome section here")
        tm.update_issue_outcome(
            issue,
            "finished",
            run_failed=False,
            dry_run=False,
            issue_number=1,
        )
        self.assertEqual(issue.edited_bodies, [])

    def test_format_runtime_uses_minutes_seconds(self) -> None:
        self.assertEqual(tm.format_runtime({"_runtime": 125}), "2m 5s")
        self.assertEqual(tm.format_runtime({}), "N/A")

    def test_build_results_comment_includes_transition_note_when_present(self) -> None:
        run = _Run("run-1", "https://wandb.ai/fake", "finished")
        comment = tm.build_results_comment(
            run,
            {"score": 0.98765, "_runtime": 61},
            {"seed": 42},
            "Invalid transition",
        )
        self.assertIn("Training Run Results", comment)
        self.assertIn("0.9877", comment)
        self.assertIn("Lifecycle sync note", comment)

    def test_sync_issue_labels_applies_expected_changes_when_transition_allowed(self) -> None:
        issue = _Issue(labels=["status: approved", "status: blocked", "type: experiment"])
        with patch.object(tm, "validate_run_transition", return_value=(True, "Transition allowed")):
            reason = tm.sync_issue_labels(
                issue,
                "finished",
                repo_root=PROJECT_ROOT,
                dry_run=False,
                issue_number=77,
            )
        self.assertIsNone(reason)
        self.assertEqual(issue.added, ["status: complete"])
        self.assertEqual(issue.removed, ["status: approved", "status: blocked"])

    def test_sync_issue_labels_posts_comment_when_transition_blocked(self) -> None:
        issue = _Issue(labels=["status: complete", "type: experiment"])
        with patch.object(
            tm,
            "validate_run_transition",
            return_value=(False, "Invalid transition for experiment: complete -> in-progress"),
        ):
            reason = tm.sync_issue_labels(
                issue,
                "running",
                repo_root=PROJECT_ROOT,
                dry_run=False,
                issue_number=88,
            )

        self.assertIn("Invalid transition", reason or "")
        self.assertEqual(issue.added, [])
        self.assertEqual(issue.removed, [])
        self.assertEqual(len(issue.comments), 1)
        self.assertIn("Lifecycle update was skipped", issue.comments[0])


if __name__ == "__main__":
    unittest.main()
