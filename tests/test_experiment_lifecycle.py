import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import experiment_lifecycle as lifecycle


class _Issue:
    def __init__(self, number: int) -> None:
        self.number = number
        self.added: list[str] = []
        self.removed: list[str] = []
        self.comments: list[str] = []

    def add_to_labels(self, label: str) -> None:
        self.added.append(label)

    def remove_from_labels(self, label: str) -> None:
        self.removed.append(label)

    def create_comment(self, message: str) -> None:
        self.comments.append(message)


class ExperimentLifecycleTests(unittest.TestCase):
    def test_load_experiment_policy_from_repo_config(self) -> None:
        policy = lifecycle.load_experiment_policy(PROJECT_ROOT)
        self.assertTrue(policy.get("require_approval_before_kickoff"))
        self.assertTrue(policy.get("require_parent_issue_reference"))

    def test_extract_parent_refs_ignores_fenced_code(self) -> None:
        body = """
Part of #101

```text
#999 should be ignored
```

Related to #202
"""
        refs = lifecycle.extract_parent_refs(body)
        self.assertEqual(refs, [101, 202])

    def test_validate_experiment_transition_allows_approved_to_in_progress(self) -> None:
        is_allowed, _reason, current_state = lifecycle.validate_experiment_transition(
            PROJECT_ROOT,
            ["status: approved"],
            "in-progress",
        )
        self.assertTrue(is_allowed)
        self.assertEqual(current_state, "approved")

    def test_validate_experiment_transition_blocks_complete_to_in_progress(self) -> None:
        is_allowed, reason, current_state = lifecycle.validate_experiment_transition(
            PROJECT_ROOT,
            ["status: complete"],
            "in-progress",
        )
        self.assertFalse(is_allowed)
        self.assertEqual(current_state, "complete")
        self.assertIn("Invalid transition", reason)

    def test_apply_label_changes_mutates_issue_when_not_dry_run(self) -> None:
        issue = _Issue(number=7)
        lifecycle.apply_label_changes(
            issue,
            current_label_names=["status: approved"],
            add=["status: in-progress", "status: agent-created"],
            remove=["status: approved"],
            dry_run=False,
        )
        self.assertEqual(issue.added, ["status: in-progress", "status: agent-created"])
        self.assertEqual(issue.removed, ["status: approved"])

    def test_post_comment_writes_issue_comment_when_not_dry_run(self) -> None:
        issue = _Issue(number=8)
        lifecycle.post_comment(issue, "hello", dry_run=False, description="test")
        self.assertEqual(issue.comments, ["hello"])


if __name__ == "__main__":
    unittest.main()
