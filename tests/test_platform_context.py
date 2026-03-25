# SPDX-License-Identifier: MIT
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_platform.context import AgentContext  # noqa: E402
from agent_platform.errors import ContractError  # noqa: E402


class PlatformContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_from_env_builds_context(self) -> None:
        os.environ["GITHUB_TOKEN"] = "token"
        os.environ["REPO_NAME"] = "owner/repo"
        os.environ["ISSUE_NUMBER"] = "42"
        os.environ["DRY_RUN"] = "true"

        ctx = AgentContext.from_env()
        self.assertEqual(ctx.repo_owner, "owner")
        self.assertEqual(ctx.repo_slug, "repo")
        self.assertEqual(ctx.issue_number, 42)
        self.assertTrue(ctx.dry_run)

    def test_require_issue_raises_when_missing(self) -> None:
        os.environ["GITHUB_TOKEN"] = "token"
        os.environ["REPO_NAME"] = "owner/repo"

        ctx = AgentContext.from_env()
        with self.assertRaises(ContractError):
            ctx.require_issue()


if __name__ == "__main__":
    unittest.main()

