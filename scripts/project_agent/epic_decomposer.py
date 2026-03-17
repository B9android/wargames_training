"""Epic Decomposer: Auto-creates child issues from epic decomposition templates."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from github import Github

from common import add_marker, agent_signature, extract_marker, has_marker, log_event, require_env, retry
from projects_v2 import ProjectsV2Client


DRY_RUN = False
REPO_NAME = ""
EPIC_NUMBER = 0


def load_orchestration_config() -> dict:
    """Load orchestration.yaml for policies."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_epic_sections(body: str) -> dict[str, str]:
    """Parse epic body for titled sections (e.g., ## Goal, ## Implementation Plan)."""
    sections = {}
    current_section = None
    current_content = []
    
    for line in (body or "").split("\n"):
        # Detect section headers (## Header format)
        header_match = re.match(r"^##\s+(.+)$", line)
        if header_match:
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = header_match.group(1).strip()
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


def parse_decomposition_items(implementation_plan: str) -> list[dict]:
    """Parse Implementation Plan section into individual issues.
    
    Expected format:
    - [ ] Task 1: Description
    - [ ] Task 2: Description
    or
    - Task 1: Description
    - Task 2: Description
    """
    items = []
    for line in implementation_plan.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Remove checkbox markers if present
        line = re.sub(r"^-\s*\[\s*\]\s*", "", line)
        line = re.sub(r"^-\s*", "", line).strip()
        
        if line:
            # Split title from optional description
            if ":" in line:
                title, description = line.split(":", 1)
                items.append({
                    "title": title.strip(),
                    "description": description.strip()
                })
            else:
                items.append({
                    "title": line,
                    "description": ""
                })
    
    return items


def get_active_sprint(repo, config: dict) -> str | None:
    """Find active (or upcoming) sprint to assign decomposed issues."""
    # Try to find a sprint with "active" status via project iterations
    # For now, return None and let it be manually assigned
    # This would integrate with GitHub Projects API in a full implementation
    return None


def create_child_issue(repo, epic_issue, item: dict, config: dict, projects_client: ProjectsV2Client | None = None) -> int | None:
    """Create a child issue from decomposition item."""
    config_epic = config.get("policies", {}).get("epic", {})
    
    # Build child issue body
    child_body = f"""{item.get('description', '')}

**Parent Epic:** #{epic_issue.number}

<!-- parent-epic:epic-{epic_issue.number} -->
<!-- status:triaged -->
"""
    
    # Create the issue
    try:
        if DRY_RUN:
            log_event(
                "child_issue_create_dry_run",
                epic=epic_issue.number,
                title=item["title"],
                description=item.get("description", "")
            )
            return None
        
        child_issue = repo.create_issue(
            title=item["title"],
            body=child_body,
            labels=["type: feature", "status: triaged"]
        )
        
        log_event(
            "child_issue_created",
            epic=epic_issue.number,
            child_issue=child_issue.number,
            title=item["title"]
        )
        
        # Add child issue to project board with parent epic's version
        if projects_client:
            try:
                # Extract version from parent epic if available
                epic_version = extract_marker(epic_issue.body or "", "version")
                field_updates = {}
                if epic_version:
                    field_updates["Version"] = (epic_version, "SINGLE_SELECT")
                
                # Also try to get active sprint for assignment
                active_sprint_id = projects_client.get_active_sprint_id()
                if active_sprint_id:
                    field_updates["Sprint"] = (active_sprint_id, "ITERATION")
                
                if field_updates:
                    projects_client.ensure_issue_in_project_with_fields(
                        child_issue.number,
                        field_updates=field_updates,
                    )
                    log_event("child_issue_added_to_project", child=child_issue.number)
                else:
                    # Add to project even without field updates
                    projects_client.ensure_issue_in_project_with_fields(
                        child_issue.number,
                    )
                    log_event("child_issue_added_to_project", child=child_issue.number)
            except Exception as exc:
                log_event("child_project_sync_failed", child=child_issue.number, error=str(exc))
        
        return child_issue.number
    except Exception as exc:
        log_event(
            "child_issue_creation_failed",
            epic=epic_issue.number,
            title=item["title"],
            error=str(exc)
        )
        return None


def decompose_epic(repo, epic_issue, projects_client: ProjectsV2Client | None = None) -> list[int]:
    """Decompose epic into child issues."""
    config = load_orchestration_config()
    child_issue_numbers = []
    
    # Skip if already decomposed
    if has_marker(epic_issue.body or "", "decomposition-complete"):
        log_event("epic_already_decomposed", epic=epic_issue.number)
        return []
    
    # Extract sections from epic body
    sections = extract_epic_sections(epic_issue.body or "")
    
    # Look for Implementation Plan section (or similar)
    implementation_plan = None
    for key in ["Implementation Plan", "Implementation", "Tasks", "Breakdown"]:
        if key in sections:
            implementation_plan = sections[key]
            break
    
    if not implementation_plan:
        log_event("epic_no_implementation_plan", epic=epic_issue.number)
        return []
    
    # Parse items
    items = parse_decomposition_items(implementation_plan)
    log_event("epic_decomposition_items_parsed", epic=epic_issue.number, count=len(items))
    
    # Create child issues
    for item in items:
        child_number = create_child_issue(repo, epic_issue, item, config, projects_client)
        if child_number:
            child_issue_numbers.append(child_number)
    
    # Mark epic as decomposed
    if not DRY_RUN and child_issue_numbers:
        new_body = add_marker(epic_issue.body or "", "decomposition-complete", str(len(child_issue_numbers)))
        # Update epic body with marker (if we have access to edit)
        try:
            retry(lambda body=new_body, issue=epic_issue: issue.edit(body=body))
            log_event("epic_marked_decomposed", epic=epic_issue.number, children=len(child_issue_numbers))
        except Exception as exc:
            log_event("epic_mark_decomposed_failed", epic=epic_issue.number, error=str(exc))
    
    return child_issue_numbers


def main() -> None:
    global DRY_RUN, REPO_NAME, EPIC_NUMBER
    
    env = require_env(["REPO_NAME", "EPIC_NUMBER", "GITHUB_TOKEN"])
    REPO_NAME = env["REPO_NAME"]
    EPIC_NUMBER = int(env["EPIC_NUMBER"])
    DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
    
    gh = Github(env["GITHUB_TOKEN"])
    repo = gh.get_repo(REPO_NAME)
    epic_issue = repo.get_issue(EPIC_NUMBER)
    
    # Initialize projects v2 client for board management
    projects_client = ProjectsV2Client(env["GITHUB_TOKEN"], REPO_NAME)
    
    # Verify this is actually an epic
    if not any(label.name.startswith("type: epic") for label in epic_issue.labels):
        log_event("not_an_epic", issue=EPIC_NUMBER)
        raise ValueError(f"Issue #{EPIC_NUMBER} is not labeled as an epic")
    
    # Decompose the epic
    child_issues = decompose_epic(repo, epic_issue, projects_client)
    
    # Post comment
    comment_body = f"""## Epic Decomposition

Decomposed into **{len(child_issues)}** child issue(s):
"""
    for child_num in child_issues:
        comment_body += f"\n- #{child_num}"
    
    if len(child_issues) == 0:
        comment_body += "\n*(No items to decompose; check Implementation Plan section)*"
    
    comment_body += f"\n\n{agent_signature('epic_decomposer', context='epic decomposition')}"
    
    if DRY_RUN:
        print(f"[dry-run] Would post decomposition comment to epic #{EPIC_NUMBER}")
        print(comment_body)
    else:
        try:
            retry(lambda body=comment_body: epic_issue.create_comment(body))
        except Exception as exc:
            log_event("decomposition_comment_failed", epic=EPIC_NUMBER, error=str(exc))
    
    log_event("epic_decomposition_complete", epic=EPIC_NUMBER, children=len(child_issues), dry_run=DRY_RUN)
    print(f"{'[dry-run] ' if DRY_RUN else ''}✅ Decomposed epic #{EPIC_NUMBER} into {len(child_issues)} issues")


if __name__ == "__main__":
    main()
