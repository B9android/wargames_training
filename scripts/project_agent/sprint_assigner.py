"""Sprint Assigner: Helper for auto-assigning issues to active sprint."""

from __future__ import annotations

from pathlib import Path

import yaml

from common import log_event
from projects_v2 import ProjectsV2Client


def get_priority_order(config: dict) -> dict[str, int]:
    """Get priority ranking from orchestration config."""
    priority_labels = config.get("labels", {}).get("priority", [])
    # Map label name to numeric priority (lower = higher priority)
    priority_map = {}
    for i, label in enumerate(reversed(priority_labels)):  # Reverse so "critical" gets 0
        priority_map[label] = i
    return priority_map


def auto_assign_issues_to_sprint(
    client: ProjectsV2Client,
    repo,
    sprint_id: str,
    filter_version: str | None = None,
    max_issues: int = 20,
) -> int:
    """Auto-assign unassigned issues to active sprint.
    
    Args:
        client: ProjectsV2Client instance
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

    # Load config for priority ordering
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception as exc:
        log_event("config_load_failed", error=str(exc))
        config = {}

    priority_map = get_priority_order(config)

    # Query unassigned issues in backlog (no sprint assignment)
    try:
        # Query issues sorted by priority
        issues = repo.get_issues(state="open", sort="updated", direction="desc")
        
        assigned_count = 0
        for issue in issues:
            if assigned_count >= max_issues:
                break

            # Skip if already has sprint marker or pull request
            if issue.pull_request or "<!-- sprint:" in (issue.body or ""):
                continue

            # Skip if filtered by version and version doesn't match
            if filter_version:
                labels = [label.name for label in issue.labels]
                version_labels = [l for l in labels if l.startswith("v")]
                if filter_version not in version_labels:
                    continue

            # Get priority for sorting
            labels = [label.name for label in issue.labels]
            priority = min((priority_map.get(l, 999) for l in labels), default=999)

            # Get issue node ID and assign to sprint
            issue_node_id = client.get_issue_node_id(client.repo_name, issue.number)
            if not issue_node_id:
                continue

            item_id = client.get_project_item_id(issue_node_id)
            if not item_id:
                continue

            # Assign to sprint
            if client.update_field_value(item_id, "Sprint", sprint_id, "ITERATION"):
                log_event("issue_assigned_to_sprint", issue=issue.number, sprint_id=sprint_id, priority=priority)
                assigned_count += 1
            else:
                log_event("issue_assignment_failed", issue=issue.number, sprint_id=sprint_id)

        log_event("sprint_auto_assign_complete", assigned=assigned_count, max=max_issues, sprint_id=sprint_id)
        return assigned_count

    except Exception as exc:
        log_event("sprint_auto_assign_failed", error=str(exc), sprint_id=sprint_id)
        return 0


def batch_update_sprint_assignment(
    client: ProjectsV2Client,
    issue_numbers: list[int],
    sprint_id: str,
) -> dict[int, bool]:
    """Update sprint assignment for multiple issues.
    
    Args:
        client: ProjectsV2Client instance
        issue_numbers: List of issue numbers
        sprint_id: Target sprint iteration ID
        
    Returns:
        Dict of issue_number → success boolean
    """
    results = {}
    
    for issue_num in issue_numbers:
        issue_node_id = client.get_issue_node_id(client.repo_name, issue_num)
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
