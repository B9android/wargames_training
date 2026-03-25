# SPDX-License-Identifier: MIT
﻿"""Sprint Assigner: Helper for auto-assigning issues to active sprint."""

from __future__ import annotations

import importlib
from pathlib import Path

from common import log_event
from agent_platform.graphql_gateway import GraphQLGateway


def get_priority_order(config: dict) -> dict[str, int]:
    """Get priority ranking from orchestration config."""
    priority_labels = config.get("labels", {}).get("priority", [])
    # Map label name to numeric priority (lower = higher priority)
    priority_map = {}
    for i, label in enumerate(reversed(priority_labels)):  # Reverse so "critical" gets 0
        priority_map[label] = i
    return priority_map


def _load_orchestration_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        yaml_module = importlib.import_module("yaml")
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml_module.safe_load(handle) or {}
    except Exception as exc:
        log_event("config_load_failed", error=str(exc))
        return {}


def _issue_labels(issue) -> list[str]:
    return [label.name for label in issue.labels]


def _is_assignable(issue, filter_version: str | None) -> bool:
    if issue.pull_request or "<!-- sprint:" in (issue.body or ""):
        return False
    if not filter_version:
        return True
    version_labels = [label for label in _issue_labels(issue) if label.startswith("v")]
    return filter_version in version_labels


def _get_priority(issue, priority_map: dict[str, int]) -> int:
    labels = _issue_labels(issue)
    return min((priority_map.get(label, 999) for label in labels), default=999)


def _assign_issue_to_sprint(client: GraphQLGateway, issue, sprint_id: str, priority: int) -> bool:
    issue_node_id = client.get_issue_node_id(issue.number)
    if not issue_node_id:
        return False

    item_id = client.get_project_item_id(issue_node_id)
    if not item_id:
        return False

    success = client.update_field_value(item_id, "Sprint", sprint_id, "ITERATION")
    event_name = "issue_assigned_to_sprint" if success else "issue_assignment_failed"
    log_event(event_name, issue=issue.number, sprint_id=sprint_id, priority=priority)
    return success


def auto_assign_issues_to_sprint(
    client: GraphQLGateway,
    repo,
    sprint_id: str,
    filter_version: str | None = None,
    max_issues: int = 20,
) -> int:
    """Auto-assign unassigned issues to active sprint.
    
    Args:
        client: GraphQL gateway instance
        repo: PyGithub repo object
        sprint_id: Target sprint iteration ID
        filter_version: Optional version to filter by (e.g., "v1")
        max_issues: Maximum number of issues to assign
        
    Returns:
        Number of issues assigned
    """
    if not sprint_id:
        log_event("sprint_assign_skipped", reason="no_sprint_id")
        return 0

    priority_map = get_priority_order(_load_orchestration_config())

    # Query unassigned issues in backlog (no sprint assignment)
    try:
        issues = repo.get_issues(state="open", sort="updated", direction="desc")
        
        assigned_count = 0
        for issue in issues:
            if assigned_count >= max_issues:
                break

            if not _is_assignable(issue, filter_version):
                continue

            priority = _get_priority(issue, priority_map)
            if _assign_issue_to_sprint(client, issue, sprint_id, priority):
                assigned_count += 1

        log_event("sprint_auto_assign_complete", assigned=assigned_count, max=max_issues, sprint_id=sprint_id)
        return assigned_count

    except Exception as exc:
        log_event("sprint_auto_assign_failed", error=str(exc), sprint_id=sprint_id)
        return 0


def batch_update_sprint_assignment(
    client: GraphQLGateway,
    issue_numbers: list[int],
    sprint_id: str,
) -> dict[int, bool]:
    """Update sprint assignment for multiple issues.
    
    Args:
        client: GraphQL gateway instance
        issue_numbers: List of issue numbers
        sprint_id: Target sprint iteration ID
        
    Returns:
        Dict of issue_number to success boolean
    """
    results = {}
    
    for issue_num in issue_numbers:
        issue_node_id = client.get_issue_node_id(issue_num)
        if not issue_node_id:
            results[issue_num] = False
            continue

        item_id = client.get_project_item_id(issue_node_id)
        if not item_id:
            results[issue_num] = False
            continue

        success = client.update_field_value(item_id, "Sprint", sprint_id, "ITERATION")
        results[issue_num] = success
        log_event(
            "batch_sprint_update",
            issue=issue_num,
            sprint_id=sprint_id,
            success=success,
        )

    log_event("batch_sprint_update_complete", total=len(issue_numbers), successful=sum(results.values()))
    return results

