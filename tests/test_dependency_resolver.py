import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from dependency_resolver import (  # noqa: E402
    blocked_parent_rollup,
    extract_issue_references,
    strip_fenced_code_blocks,
)


class _Label:
    def __init__(self, name: str) -> None:
        self.name = name


class _Issue:
    def __init__(self, number: int, title: str, body: str, labels: list[str]) -> None:
        self.number = number
        self.title = title
        self.body = body
        self.labels = [_Label(label) for label in labels]


class DependencyResolverTests(unittest.TestCase):
    def test_extract_issue_references_ignores_code_fences(self) -> None:
        text = """
Part of #12

```python
#999 should not count as a dependency reference
```

Related to #34
"""
        refs = extract_issue_references(text)
        self.assertEqual(refs, [12, 34])

    def test_strip_fenced_code_blocks_removes_all_fenced_content(self) -> None:
        text = "before\n```bash\n#123\n```\nafter"
        stripped = strip_fenced_code_blocks(text)
        self.assertNotIn("#123", stripped)
        self.assertIn("before", stripped)
        self.assertIn("after", stripped)

    def test_blocked_parent_rollup_groups_blocked_children(self) -> None:
        issues = [
            _Issue(1, "Parent", "", []),
            _Issue(2, "Child blocked", "Part of #1", ["status: blocked"]),
            _Issue(3, "Child active", "Part of #1", ["status: in-progress"]),
            _Issue(4, "Other parent", "", []),
            _Issue(5, "Blocked child 2", "Part of #4", ["status: blocked"]),
        ]

        rollup = blocked_parent_rollup(issues)
        self.assertEqual(len(rollup), 2)

        first = rollup[0]
        self.assertIn(first["parent_number"], {1, 4})

        parent_one = next(item for item in rollup if item["parent_number"] == 1)
        blocked_children_numbers = [child["number"] for child in parent_one["blocked_children"]]
        self.assertEqual(blocked_children_numbers, [2])


if __name__ == "__main__":
    unittest.main()
