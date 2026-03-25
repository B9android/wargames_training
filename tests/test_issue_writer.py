# SPDX-License-Identifier: MIT
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import issue_writer as iw


class IssueWriterTests(unittest.TestCase):
    def test_extract_parent_issue_numbers_ignores_code_fences(self) -> None:
        context = """
Part of #12

```text
#999 should be ignored
```

Related to #44
"""
        self.assertEqual(iw.extract_parent_issue_numbers(context), [12, 44])

    def test_build_user_prompt_contains_expected_sections(self) -> None:
        prompt = iw.build_user_prompt(
            "context text",
            "v1",
            ["type: feature"],
            {"M1: 1v1 Competence": 1},
            [{"number": 5, "title": "Existing"}],
            [],
        )
        self.assertIn("Target version: v1", prompt)
        self.assertIn("Available labels:", prompt)
        self.assertIn("Dependency blockers", prompt)


if __name__ == "__main__":
    unittest.main()
