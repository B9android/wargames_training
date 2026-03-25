# SPDX-License-Identifier: MIT
"""Shared experiment lifecycle helpers for approval and kickoff agents."""

from __future__ import annotations

from pathlib import Path

import yaml

from common import retry
from dependency_resolver import extract_issue_references
from state_machine import load_state_machine


AGENT_LABEL = "status: agent-created"
APPROVED_LABEL = "status: approved"
IN_PROGRESS_LABEL = "status: in-progress"
STATUS_LABEL_TO_STATE = {
    "status: approved": "approved",
    "status: in-progress": "in-progress",
    "status: blocked": "blocked",
    "status: stale": "stale",
    "status: complete": "complete",
    "status: failed": "failed",
    "status: cancelled": "cancelled",
}


def repo_root_from_file(file_path: str) -> Path:
    """Resolve the repository root from an agent script path."""
    return Path(file_path).resolve().parents[2]


def load_experiment_policy(repo_root: Path) -> dict[str, object]:
    """Load experiment policy settings from the orchestration contract."""
    config_path = repo_root / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    policies = raw.get("policies", {})
    experiment_policy = policies.get("experiment", {})
    return experiment_policy if isinstance(experiment_policy, dict) else {}


def infer_experiment_state(label_names: list[str] | set[str]) -> str:
    """Infer the current lifecycle state from issue labels."""
    for label_name, state in STATUS_LABEL_TO_STATE.items():
        if label_name in label_names:
            return state
    return "triaged"


def extract_parent_refs(issue_body: str | None) -> list[int]:
    """Extract linked parent issue references from an issue body."""
    return extract_issue_references(issue_body)


def validate_experiment_transition(
    repo_root: Path,
    label_names: list[str] | set[str],
    target_state: str,
) -> tuple[bool, str, str]:
    """Validate an experiment lifecycle transition and return current state."""
    current_state = infer_experiment_state(label_names)
    decision = load_state_machine(repo_root).can_transition("experiment", current_state, target_state)
    return decision.is_allowed, decision.reason, current_state


def apply_label_changes(
    issue,
    current_label_names: list[str] | set[str],
    *,
    add: list[str],
    remove: list[str] | None = None,
    dry_run: bool,
) -> list[str]:
    """Apply label additions/removals with retry and dry-run support."""
    current_labels = set(current_label_names)
    removed_labels = remove or []

    for label in add:
        if label not in current_labels:
            if dry_run:
                print(f"[dry-run] Would add label: {label}")
            else:
                retry(lambda label=label: issue.add_to_labels(label))

    for label in removed_labels:
        if label in current_labels:
            if dry_run:
                print(f"[dry-run] Would remove label: {label}")
            else:
                retry(lambda label=label: issue.remove_from_labels(label))

    return add


def post_comment(issue, message: str, *, dry_run: bool, description: str) -> None:
    """Post an issue comment or emit a dry-run message."""
    if dry_run:
        print(f"[dry-run] Would post {description} comment to issue #{issue.number}")
        return
    retry(lambda: issue.create_comment(message))