"""Platform telemetry — structured logging with correlation IDs and severity."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

# Populated by begin_run() for the lifetime of a single agent invocation.
_run_id: str = ""
_agent_name: str = ""
_start_time: float = 0.0


def begin_run(agent_name: str) -> str:
    """Start a telemetry run context.  Returns the correlation run_id."""
    global _run_id, _agent_name, _start_time
    _run_id = os.environ.get("GITHUB_RUN_ID") or str(uuid.uuid4())[:8]
    _agent_name = agent_name
    _start_time = time.monotonic()
    log_event("agent_run_started", agent=agent_name, run_id=_run_id)
    return _run_id


def end_run(*, success: bool, summary: dict[str, Any] | str | None = None) -> None:
    """Emit run-complete telemetry with elapsed time."""
    elapsed = time.monotonic() - _start_time if _start_time else 0.0
    if isinstance(summary, dict):
        summary_payload = summary
    elif isinstance(summary, str) and summary:
        summary_payload = {"summary": summary}
    else:
        summary_payload = {}

    # Strip any keys from summary_payload that log_event injects automatically
    # (agent, run_id, dry_run, ts) to avoid "got multiple values for keyword
    # argument" errors when callers embed those keys in their summary dicts.
    _auto_keys = {"agent", "run_id", "dry_run", "ts", "event"}
    safe_payload = {k: v for k, v in summary_payload.items() if k not in _auto_keys}

    log_event(
        "agent_run_complete",
        success=success,
        elapsed_seconds=round(elapsed, 2),
        **safe_payload,
    )


def log_event(event: str, **data: object) -> None:
    """Emit a structured JSON log line with standard context fields."""
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "agent": _agent_name or os.environ.get("AGENT_NAME", "unknown"),
        "run_id": _run_id or os.environ.get("GITHUB_RUN_ID", ""),
        "dry_run": os.environ.get("DRY_RUN", "false").lower() == "true",
    }
    payload.update(data)
    print(json.dumps(payload, sort_keys=True, default=str))
