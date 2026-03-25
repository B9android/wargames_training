# SPDX-License-Identifier: MIT
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_platform import telemetry  # noqa: E402


class PlatformTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = dict(os.environ)
        telemetry._run_id = ""
        telemetry._agent_name = ""
        telemetry._start_time = 0.0

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old)

    def test_end_run_handles_summary_with_reserved_keys(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            telemetry.begin_run("static_analysis")
            telemetry.end_run(
                success=True,
                summary={
                    "agent": "override",
                    "run_id": "override",
                    "ok": True,
                },
            )

        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)

        complete_payload = json.loads(lines[1])
        self.assertEqual(complete_payload["event"], "agent_run_complete")
        self.assertEqual(complete_payload["agent"], "static_analysis")
        self.assertEqual(complete_payload["summary_agent"], "override")
        self.assertEqual(complete_payload["summary_run_id"], "override")
        self.assertTrue(complete_payload["ok"])


if __name__ == "__main__":
    unittest.main()
