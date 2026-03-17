"""Event Router: Central dispatcher for GitHub webhook events."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from common import log_event, require_env


def load_orchestration_config() -> dict:
    """Load orchestration.yaml for policies."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dispatch_gh_workflow(workflow_name: str, inputs: dict) -> int:
    """Dispatch a GitHub Actions workflow with given inputs."""
    cmd = ["gh", "workflow", "run", workflow_name, "-r", "main"]
    
    for key, value in inputs.items():
        cmd.extend(["-f", f"{key}={value}"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log_event("workflow_dispatched", workflow=workflow_name, inputs=inputs)
        return 0
    except subprocess.CalledProcessError as exc:
        log_event("workflow_dispatch_failed", workflow=workflow_name, error=exc.stderr)
        return 1


def handle_issue_opened(event: dict) -> None:
    """Route new issue to triage."""
    issue = event.get("issue", {})
    repo = event.get("repository", {})
    
    issue_number = issue.get("number")
    repo_name = repo.get("full_name")
    
    if not (issue_number and repo_name):
        log_event("issue_opened_missing_data", event=event)
        return
    
    # Check for issue markers to determine routing
    title = issue.get("title", "").upper()
    
    if "[EPIC]" in title:
        # Triage as epic
        log_event("routing_epic_issue", issue=issue_number, repo=repo_name)
        dispatch_gh_workflow("agent-triage.yml", {
            "issue_number": str(issue_number),
            "repo_name": repo_name,
            "dry_run": "false"
        })
    elif "[EXP]" in title:
        # Triage as experiment
        log_event("routing_experiment_issue", issue=issue_number, repo=repo_name)
        dispatch_gh_workflow("agent-triage.yml", {
            "issue_number": str(issue_number),
            "repo_name": repo_name,
            "dry_run": "false"
        })
    else:
        # Regular triage
        log_event("routing_regular_issue", issue=issue_number, repo=repo_name)
        dispatch_gh_workflow("agent-triage.yml", {
            "issue_number": str(issue_number),
            "repo_name": repo_name,
            "dry_run": "false"
        })


def handle_issue_labeled(event: dict) -> None:
    """Route labeled issues to decomposer if epic."""
    issue = event.get("issue", {})
    label = event.get("label", {})
    repo = event.get("repository", {})
    
    issue_number = issue.get("number")
    repo_name = repo.get("full_name")
    label_name = label.get("name", "")
    
    if not (issue_number and repo_name):
        log_event("issue_labeled_missing_data", event=event)
        return
    
    # If epic and labeled with "status: triaged", decompose
    if label_name == "status: triaged" and any(
        label.get("name", "").startswith("type: epic") 
        for label in issue.get("labels", [])
    ):
        log_event("routing_epic_decomposition", issue=issue_number, repo=repo_name)
        dispatch_gh_workflow("agent-epic-decomposer.yml", {
            "issue_number": str(issue_number),
            "repo_name": repo_name,
            "dry_run": "false"
        })


def handle_pull_request_opened(event: dict) -> None:
    """Route new PR to linker."""
    pr = event.get("pull_request", {})
    repo = event.get("repository", {})
    
    pr_number = pr.get("number")
    repo_name = repo.get("full_name")
    
    if not (pr_number and repo_name):
        log_event("pr_opened_missing_data", event=event)
        return
    
    log_event("routing_pr_opened", pr=pr_number, repo=repo_name)
    dispatch_gh_workflow("agent-pr-linker.yml", {
        "pr_number": str(pr_number),
        "repo_name": repo_name,
        "dry_run": "false"
    })


def handle_pull_request_closed(event: dict) -> None:
    """Route merged PR to linker for status sync."""
    pr = event.get("pull_request", {})
    repo = event.get("repository", {})
    action = event.get("action", "")
    
    pr_number = pr.get("number")
    repo_name = repo.get("full_name")
    merged = pr.get("merged", False)
    
    if not (pr_number and repo_name):
        log_event("pr_closed_missing_data", event=event)
        return
    
    if merged:
        log_event("routing_pr_merged", pr=pr_number, repo=repo_name)
        dispatch_gh_workflow("agent-pr-linker.yml", {
            "pr_number": str(pr_number),
            "repo_name": repo_name,
            "dry_run": "false"
        })


def main() -> None:
    # Read GitHub webhook payload from stdin or env
    env = require_env(["GITHUB_TOKEN"])
    
    # GitHub Actions provides the entire event as environment variable or stdin
    event_json = os.environ.get("GITHUB_EVENT_JSON")
    if not event_json:
        # Try reading from stdin (for direct webhook handlers)
        import sys
        try:
            event_json = sys.stdin.read()
        except Exception:
            log_event("event_router_no_event")
            return
    
    try:
        event = json.loads(event_json)
    except json.JSONDecodeError as exc:
        log_event("event_parser_failed", error=str(exc))
        return
    
    event_type = event.get("action", "")
    
    # Route based on event type
    if "issue" in event and event_type in ["opened", "reopened"]:
        handle_issue_opened(event)
    elif "issue" in event and event_type == "labeled":
        handle_issue_labeled(event)
    elif "pull_request" in event and event_type == "opened":
        handle_pull_request_opened(event)
    elif "pull_request" in event and event_type in ["closed", "synchronize"]:
        handle_pull_request_closed(event)
    else:
        log_event("event_unhandled", action=event_type, keys=list(event.keys()))
    
    log_event("event_router_complete", action=event_type)


if __name__ == "__main__":
    main()
