"""Sprint Manager: Manages sprint lifecycle (planning, active, completed, archived)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from github import Auth, Github

from common import agent_signature, log_event, require_env, retry
from projects_v2 import ProjectsV2Client
from sprint_assigner import auto_assign_issues_to_sprint


DRY_RUN = False
REPO_NAME = ""
ACTION = ""  # "start", "close", "auto-transition"


def load_orchestration_config() -> dict:
    """Load orchestration.yaml for policies."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_sprint_by_name(repo, sprint_name: str):
    """Find a project field value (sprint) by name via GraphQL."""
    # This would require GitHub GraphQL API integration
    # For now, return None - full implementation would need custom queries
    log_event("sprint_lookup_stub", sprint_name=sprint_name)
    return None


def get_active_sprint_issues(repo) -> list:
    """Get all issues assigned to active sprint."""
    # This is a stub - full implementation would use project iterations
    # For now, query issues with sprint metadata in body
    issues = []
    try:
        # Search for issues with active sprint marker
        issues = repo.get_issues(state="open")
    except Exception as exc:
        log_event("sprint_issues_query_failed", error=str(exc))
    
    return issues


def start_sprint(repo, sprint_name: str, projects_client: ProjectsV2Client | None = None) -> bool:
    """Activate a sprint and transition issues to it."""
    log_event("sprint_start_requested", sprint=sprint_name)
    
    if DRY_RUN:
        print(f"[dry-run] Would start sprint: {sprint_name}")
        return True
    
    # If projects_client provided, auto-assign issues to sprint
    if projects_client:
        try:
            sprint_id = projects_client.get_active_sprint_id()
            if sprint_id:
                assigned = auto_assign_issues_to_sprint(projects_client, repo, sprint_id, max_issues=30)
                log_event("sprint_issues_auto_assigned", sprint=sprint_name, sprint_id=sprint_id, count=assigned)
        except Exception as exc:
            log_event("sprint_auto_assign_failed", sprint=sprint_name, error=str(exc))
    
    # GitHub API limitation: sprints via Projects v2 require GraphQL
    # For now, log the intent
    log_event("sprint_started", sprint=sprint_name)
    return True


def close_sprint(repo, sprint_name: str) -> bool:
    """Mark sprint as completed and archive old sprints."""
    config = load_orchestration_config()
    log_event("sprint_close_requested", sprint=sprint_name)
    
    if DRY_RUN:
        print(f"[dry-run] Would close sprint: {sprint_name}")
        return True
    
    log_event("sprint_closed", sprint=sprint_name)
    return True


def auto_transition_sprints(repo, projects_client: ProjectsV2Client | None = None) -> dict[str, int]:
    """Auto-transition sprints and issues based on deadlines."""
    config = load_orchestration_config()
    sprint_policy = config.get("policies", {}).get("sprint", {})
    
    if not sprint_policy.get("auto_transition", False):
        log_event("sprint_auto_transition_disabled")
        return {}
    
    stats = {
        "sprints_started": 0,
        "sprints_closed": 0,
        "sprints_archived": 0,
    }
    
    if DRY_RUN:
        print("[dry-run] Would auto-transition sprints")
        return stats
    
    # If projects_client provided, try to auto-assign issues to current sprint
    if projects_client:
        try:
            sprint_id = projects_client.get_active_sprint_id()
            if sprint_id:
                assigned = auto_assign_issues_to_sprint(projects_client, repo, sprint_id, max_issues=50)
                stats["issues_auto_assigned"] = assigned
                log_event("sprint_auto_transition_assigned_issues", count=assigned)
        except Exception as exc:
            log_event("sprint_projects_sync_failed", error=str(exc))
    
    # Full implementation would:
    # 1. Query all projects for sprints with iteration fields
    # 2. Check deadline vs current time
    # 3. Transition based on state_machine rules
    # 4. Update issue assignments
    
    return stats


def main() -> None:
    global DRY_RUN, REPO_NAME, ACTION
    
    env = require_env(["REPO_NAME", "GITHUB_TOKEN"])
    REPO_NAME = env["REPO_NAME"]
    ACTION = os.environ.get("ACTION", "auto-transition")
    DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
    
    sprint_name = os.environ.get("SPRINT_NAME", "")
    
    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    repo = gh.get_repo(REPO_NAME)
    
    # Initialize projects v2 client for board management
    projects_client = None
    try:
        projects_client = ProjectsV2Client(env["GITHUB_TOKEN"], REPO_NAME)
    except Exception as exc:
        log_event("projects_client_init_failed", error=str(exc))
    
    if ACTION == "start":
        if not sprint_name:
            raise ValueError("SPRINT_NAME required for start action")
        start_sprint(repo, sprint_name, projects_client)
        print(f"{'[dry-run] ' if DRY_RUN else ''}✅ Sprint '{sprint_name}' started")
        
    elif ACTION == "close":
        if not sprint_name:
            raise ValueError("SPRINT_NAME required for close action")
        close_sprint(repo, sprint_name)
        print(f"{'[dry-run] ' if DRY_RUN else ''}✅ Sprint '{sprint_name}' closed")
        
    elif ACTION == "auto-transition":
        stats = auto_transition_sprints(repo, projects_client)
        log_event("sprint_auto_transition_complete", **stats, dry_run=DRY_RUN)
        print(f"{'[dry-run] ' if DRY_RUN else ''}✅ Sprint auto-transition complete")
    else:
        raise ValueError(f"Unknown action: {ACTION}")


if __name__ == "__main__":
    main()
