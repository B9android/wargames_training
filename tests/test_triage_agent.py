import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import triage_agent as ta


class TriageAgentTests(unittest.TestCase):
    def test_choose_milestone_prefers_m1_for_exp(self) -> None:
        ta.all_milestones = {"M1: 1v1 Competence": 1, "M2: Other": 2}
        self.assertEqual(ta.choose_milestone("[EXP] run"), "M1: 1v1 Competence")

    def test_choose_milestone_falls_back_to_first(self) -> None:
        ta.all_milestones = {"M2: Other": 2}
        self.assertEqual(ta.choose_milestone("[BUG] issue"), "M2: Other")


if __name__ == "__main__":
    unittest.main()
