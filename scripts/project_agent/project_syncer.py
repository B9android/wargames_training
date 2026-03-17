"""Project Syncer: Reactive sync of GitHub labels and metadata to Projects v2 fields."""

from __future__ import annotations

import os
from pathlib import Path

from github import Github

from common import extract_marker, log_event, require_env
from projects_v2 import ProjectsV2Client


DRY_RUN = False
REPO_NAME = ""
ISSUE_NUMBER = 0


def load_label_to_field_mappings() -> dict[str, tuple[str, str]]:
    """Load mappings of GitHub labels to project field updates.
    
    Returns:
        Dict mapping label → (field_name, field_value) for known labels
    """
    # Standard mappings
    return {
        # Status mappings (GitHub label → project Status field)
        "status: triaged": ("Status", "Triaged"),
        "status: approved": ("Status", "Approved"),
        "status: in-progress": ("Status", "In Progress"),
        "status: blocked": ("Status", "Blocked"),
        "status: complete": ("Status", "Done"),
        # Version mappings (GitHub label → project Version field)
        "v1": ("Version", "v1"),
        "v2": ("Version", "v2"),
        "v3": ("Version", "v3"),
        "v4": ("Version", "v4"),
        # Priority → Story Points (heuristic mapping)
        "priority: critical": ("Story Points", "8"),
        "priority: high": ("Story Points", "5"),
        "priority: medium": ("Story Points", "3"),
        "priority: low": ("Story Points", "1"),
    }


def sync_issue_labels_to_fields(repo, issue_number: int, projects_client: ProjectsV2Client | None = None) -> bool:
    """Sync issue labels to project board custom fields.
    
    Args:
        repo: PyGithub repo object
        issue_number: Issue number to sync
        projects_client: ProjectsV2Client instance
        
    Returns:
        True if successful
    """
    if not projects_client:
        log_event("sync_skipped", issue=issue_number, reason="no_projects_client")
        return False

    try:
        issue = repo.get_issue(issue_number)
    except Exception as exc:
        log_event("issue_lookup_failed", issue=issue_number, error=str(exc))
        return False

    label_mappings = load_label_to_field_mappings()
    applied_labels = {label.name for label in issue.labels}

    # Get issue node ID and project item ID
    try:
        issue_node_id = projects_client.get_issue_node_id(REPO_NAME, issue_number)
        if not issue_node_id:
            log_event("sync_failed", issue=issue_number, reason="node_id_not_found")
            return False

        item_id = projects_client.get_project_item_id(issue_node_id)
        if not item_id:
            log_event("sync_failed", issue=issue_number, reason="item_id_not_found")
            return False
    except Exception as exc:
        log_event("sync_ids_query_failed", issue=issue_number, error=str(exc))
        return False

    # Apply field updates for matching labels
    updated_fields = {}
    for label_name, (field_name, field_value) in label_mappings.items():
        if label_name in applied_labels:
            # Determine field type
            field_type = "SINGLE_SELECT"  # Most are single-select
            if field_name == "Story Points":
                field_type = "NUMBER"

            try:
                if not DRY_RUN:
                    success = projects_client.update_field_value(item_id, field_name, field_value, field_type)
                    if success:
                        updated_fields[field_name] = field_value
                        log_event("field_synced", issue=issue_number, field=field_name, value=field_value)
                else:
                    log_event("field_sync_dry_run", issue=issue_number, field=field_name, value=field_value)
                    updated_fields[field_name] = field_value
            except Exception as exc:
                log_event("field_sync_failed", issue=issue_number, field=field_name, error=str(exc))

    if updated_fields:
        log_event("sync_complete", issue=issue_number, fields=len(updated_fields), updated=updated_fields)
        return True

    log_event("sync_no_matching_labels", issue=issue_number)
    return True


def sync_issue_milestone_to_version(repo, issue_number: int, projects_client: ProjectsV2Client | None = None) -> bool:
    """Sync issue milestone to project Version field if milestone contains version info.
    
    Args:
        repo: PyGithub repo object
        issue_number: Issue number to sync
        projects_client: ProjectsV2Client instance
        
    Returns:
        True if successful
    """
    if not projects_client:
        return False

    try:
        issue = repo.get_issue(issue_number)
        if not issue.milestone:
            return True  # No milestone, nothing to sync

        milestone_title = issue.milestone.title
        
        # Try to extract version from milestone (e.g., "M1: 1v1 Competence" → "v1")
        version = None
        for v in ["v1", "v2", "v3", "v4"]:
            if v.lower() in milestone_title.lower():
                version = v
                break

        if not version:
            return True  # No version found in milestone

        issue_node_id = projects_client.get_issue_node_id(REPO_NAME, issue_number)
        if not issue_node_id:
            return False

        item_id = projects_client.get_project_item_id(issue_node_id)
        if not item_id:
            return False

        if not DRY_RUN:
            projects_client.update_field_value(item_id, "Version", version, "SINGLE_SELECT")
            log_event("milestone_version_synced", issue=issue_number, milestone=milestone_title, version=version)
        else:
            log_event("milestone_version_sync_dry_run", issue=issue_number, milestone=milestone_title, version=version)

        return True

    except Exception as exc:
        log_event("milestone_sync_failed", issue=issue_number, error=str(exc))
        return False


def main() -> None:
    global DRY_RUN, REPO_NAME, ISSUE_NUMBER

    env = require_env(["REPO_NAME", "ISSUE_NUMBER", "GITHUB_TOKEN"])
    REPO_NAME = env["REPO_NAME"]
    ISSUE_NUMBER = int(env["ISSUE_NUMBER"])
    DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

    gh = Github(env["GITHUB_TOKEN"])
    repo = gh.get_repo(REPO_NAME)

    # Initialize projects v2 client
    projects_client = None
    try:
        projects_client = ProjectsV2Client(env["GITHUB_TOKEN"], REPO_NAME)
    except Exception as exc:
        log_event("projects_client_init_failed", error=str(exc))
        raise

    # Sync labels to fields
    labels_synced = sync_issue_labels_to_fields(repo, ISSUE_NUMBER, projects_client)

    # Sync milestone to version field
    milestone_synced = sync_issue_milestone_to_version(repo, ISSUE_NUMBER, projects_client)

    if labels_synced or milestone_synced:
        print(f"{'[dry-run] ' if DRY_RUN else ''}✅ Issue #{ISSUE_NUMBER} project fields synced")
    else:
        print(f"{'[dry-run] ' if DRY_RUN else ''}⚠️ Issue #{ISSUE_NUMBER} sync completed with warnings")

    log_event("project_syncer_complete", issue=ISSUE_NUMBER, dry_run=DRY_RUN)


if __name__ == "__main__":
    main()
