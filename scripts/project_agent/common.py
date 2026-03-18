"""Shared helpers for GitHub project agents."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


AGENT_IDENTITIES = {
    "triage": "Marshal Triage",
    "experiment_approval": "Councillor Approval",
    "experiment_kickoff": "Captain Kickoff",
    "training_monitor": "Quartermaster Watch",
    "milestone_checker": "Steward Milestone",
    "progress_reporter": "Archivist Weekly",
    "issue_writer": "Strategist Forge",
    "release_coordinator": "Chronicler Release",
    "epic_decomposer": "Cartographer Epic",
    "sprint_manager": "Chronometer Sprint",
    "pr_linker": "Courier PR",
}

# Project board custom field names (for consistency across agents)
PROJECT_FIELDS = {
    "version": "Version",
    "sprint": "Sprint",
    "status": "Status",
    "story_points": "Story Points",
    "experiment_status": "Experiment Status",
    "git_commit": "Git Commit",
    "wb_run": "W&B Run",
}

# Backward-compatible alias for older imports.
PROJECTS_V2_FIELDS = PROJECT_FIELDS


class AgentError(RuntimeError):
    """Raised for agent failures that should produce a controlled exit."""


def require_env(keys: Iterable[str]) -> dict[str, str]:
    """Return required environment values or raise a helpful error."""
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        value = os.environ.get(key)
        if not value:
            missing.append(key)
        else:
            values[key] = value
    if missing:
        raise AgentError(f"Missing required environment variables: {', '.join(missing)}")
    return values


def log_event(event: str, **data: object) -> None:
    """Emit structured logs for easier workflow diagnostics."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    }
    print(json.dumps(payload, sort_keys=True))


def agent_identity(agent_key: str) -> str:
    """Return the friendly display name for an automation agent."""
    return AGENT_IDENTITIES.get(agent_key, agent_key)


def agent_signature(agent_key: str, *, context: str | None = None) -> str:
    """Render a short attribution footer for issue-facing automation output."""
    identity = agent_identity(agent_key)
    if context:
        return f"> {identity} handled this via {context}."
    return f"> {identity} handled this automatically."


def retry(
    operation: Callable[[], T],
    *,
    retries: int = 3,
    base_delay_seconds: float = 1.0,
    retriable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Run operation with exponential backoff for transient failures."""
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except retriable_exceptions as exc:  # pragma: no cover - runtime guard
            last_error = exc
            if attempt == retries:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            log_event("retry", attempt=attempt, delay_seconds=delay, error=str(exc))
            time.sleep(delay)
    if last_error:
        raise last_error
    raise AgentError("retry called without operation result or error")


def extract_marker(body: str, marker_name: str) -> str | None:
    """Extract idempotency marker or metadata from issue body."""
    import re
    pattern = rf"<!--\s*{re.escape(marker_name)}:(.+?)\s*-->"
    match = re.search(pattern, body)
    return match.group(1).strip() if match else None


def has_marker(body: str, marker_name: str) -> bool:
    """Check if idempotency marker exists in body."""
    return extract_marker(body, marker_name) is not None


def add_marker(body: str, marker_name: str, marker_value: str = "") -> str:
    """Add idempotency marker to issue body."""
    marker = f"<!-- {marker_name}:{marker_value} -->"
    if body:
        return f"{body}\n\n{marker}"
    return marker
