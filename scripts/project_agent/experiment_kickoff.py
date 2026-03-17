#!/usr/bin/env python
"""
🧪 Experiment Kickoff Agent
Validates experiment issue completeness and initializes run tracking.

Responsibilities:
- Validate hypothesis and setup fields are present
- Extract and validate W&B run URL/ID
- Set issue to "in-progress" status
- Post initialization comment with extracted metadata
- Prevent duplicate runs via idempotency markers
"""

import os
import re
from github import Auth, Github
from common import agent_signature, log_event, require_env, retry
from experiment_lifecycle import (
    AGENT_LABEL,
    APPROVED_LABEL,
    IN_PROGRESS_LABEL,
    apply_label_changes,
    extract_parent_refs,
    load_experiment_policy,
    post_comment,
    repo_root_from_file,
    validate_experiment_transition,
)

# ============================================================================
# Configuration
# ============================================================================

REQUIRED_FIELDS = ["hypothesis", "setup"]

# Regex to extract W&B run ID from URLs/references
# Formats: https://wandb.ai/org/project/runs/RUN_ID or just RUN_ID
WANDB_RUN_PATTERN = r"(?:https://wandb\.ai/[^/]+/[^/]+/runs/)?([a-z0-9]+)"


# ============================================================================
# Utilities
# ============================================================================

def extract_field_from_body(body: str, field_name: str) -> str:
    """
    Extract the content of a field from a GitHub issue body.
    
    GitHub's form rendering creates sections like:
    ### Field Label
    field content here
    
    Args:
        body: The issue body text
        field_name: The field name from the template (e.g., "hypothesis", "setup")
    
    Returns:
        The field content, stripped of whitespace. Empty string if not found.
    """
    # Normalize field name to title case
    field_header = f"### {field_name.replace('_', ' ').title()}"
    
    if field_header not in body:
        return ""
    
    # Extract text after the header until the next ### or end of string
    pattern = re.escape(field_header) + r"\n(.+?)(?=\n###|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""


def extract_wandb_run_id(setup_text: str) -> str:
    """
    Extract W&B run ID from the setup field.
    
    Looks for patterns like:
    - W&B run: https://wandb.ai/org/project/runs/abc123
    - W&B run: abc123
    - wandb.ai/.../runs/abc123
    
    Args:
        setup_text: The experiment setup field content
    
    Returns:
        The W&B run ID if found, empty string otherwise
    """
    # Look for wandb.ai URLs or direct run IDs
    lines = setup_text.split("\n")
    for line in lines:
        if "wandb" in line.lower() or "w&b" in line.lower():
            match = re.search(WANDB_RUN_PATTERN, line)
            if match:
                return match.group(1)
    return ""


def validate_experiment_completeness(issue) -> tuple[bool, list[str]]:
    """
    Validate that required fields are filled in the experiment issue.
    
    Args:
        issue: PyGithub Issue object
    
    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    missing = []
    for field in REQUIRED_FIELDS:
        content = extract_field_from_body(issue.body or "", field)
        if not content:
            missing.append(field)
    
    return len(missing) == 0, missing


def create_idempotency_marker(run_id: str) -> str:
    """Create HTML comment marker for idempotency tracking."""
    return f"<!-- agent:experiment-kickoff:{run_id} -->"


def has_kickoff_marker(body: str, run_id: str) -> bool:
    """Check if this experiment has already been processed."""
    marker = create_idempotency_marker(run_id)
    return marker in (body or "")


def post_validation_failure(issue, *, dry_run: bool, issue_number: str, missing_fields: list[str]) -> None:
    """Comment and log when required experiment fields are missing."""
    error_msg = (
        f"❌ Experiment validation failed. Missing required fields:\n\n"
        f"{chr(10).join(f'- {field}' for field in missing_fields)}\n\n"
        "Please fill in these fields in the issue description before submitting the run.\n\n"
        + agent_signature("experiment_kickoff", context="experiment validation")
    )
    post_comment(issue, error_msg, dry_run=dry_run, description="validation error")
    log_event(
        "experiment_kickoff_failed",
        issue_number=issue_number,
        missing_fields=missing_fields,
    )


def block_missing_parent_reference(issue, *, dry_run: bool, issue_number: str) -> None:
    """Comment and log when parent linkage is missing."""
    linkage_msg = (
        "⏸️ Experiment kickoff is blocked until this issue is linked to parent work.\n\n"
        "Add a reference like `Part of #123` or `Related to #123` in the issue body, then rerun kickoff.\n\n"
        + agent_signature("experiment_kickoff", context="dependency validation")
    )
    post_comment(issue, linkage_msg, dry_run=dry_run, description="dependency validation")
    log_event(
        "experiment_kickoff_blocked",
        issue_number=issue_number,
        reason="missing_parent_reference",
    )


def block_missing_approval(issue, *, dry_run: bool, issue_number: str) -> None:
    """Comment and log when approval is required before kickoff."""
    approval_msg = (
        "⏸️ Experiment kickoff is blocked until approval is recorded.\n\n"
        f"Add the `{APPROVED_LABEL}` label after review, then rerun kickoff for this issue.\n\n"
        + agent_signature("experiment_kickoff", context="approval gate")
    )
    post_comment(issue, approval_msg, dry_run=dry_run, description="approval gate")
    log_event(
        "experiment_kickoff_blocked",
        issue_number=issue_number,
        reason="approval_required",
        required_label=APPROVED_LABEL,
    )


def block_invalid_transition(issue, *, dry_run: bool, issue_number: str, reason: str) -> None:
    """Comment and log when kickoff transition is invalid."""
    blocked_msg = (
        "⏸️ Experiment kickoff could not advance the issue lifecycle.\n\n"
        f"Transition rejected: `{reason}`\n\n"
        + agent_signature("experiment_kickoff", context="state transition validation")
    )
    post_comment(issue, blocked_msg, dry_run=dry_run, description="transition validation")
    log_event(
        "experiment_kickoff_blocked",
        issue_number=issue_number,
        reason="invalid_transition",
        detail=reason,
    )


def build_success_comment(wandb_run_id: str, linked_parent_refs: list[int]) -> str:
    """Build the kickoff success comment body."""
    success_msg = "✅ Experiment initialized and validated.\n\n"
    success_msg += "**Extracted Metadata:**\n"
    success_msg += "- Status: In Progress\n"
    if wandb_run_id:
        success_msg += f"- W&B Run ID: `{wandb_run_id}`\n"
        success_msg += f"- W&B URL: https://wandb.ai/runs/{wandb_run_id}\n"
    else:
        success_msg += "- W&B Run ID: Not found in setup\n"
    success_msg += f"- Linked parent issues: {', '.join(f'#{ref}' for ref in linked_parent_refs)}\n"

    success_msg += "\nNext steps:\n"
    success_msg += "1. Monitor training progress via W&B dashboard\n"
    if wandb_run_id:
        success_msg += f"2. Use `/monitor {wandb_run_id}` command to sync results to this issue\n"
    success_msg += "3. Fill in Results section when training completes\n"
    success_msg += "4. Update Outcome dropdown with result status\n"
    return success_msg


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Main agent workflow."""
    
    # Validate environment
    env = require_env(["REPO_NAME", "GITHUB_TOKEN", "ISSUE_NUMBER"])
    REPO_NAME = env["REPO_NAME"]
    ISSUE_NUMBER = env["ISSUE_NUMBER"]
    DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
    repo_root = repo_root_from_file(__file__)
    
    # Connect to GitHub
    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    repo = gh.get_repo(REPO_NAME)
    issue = retry(lambda: repo.get_issue(int(ISSUE_NUMBER)))
    policy = load_experiment_policy(repo_root)
    
    log_event("experiment_kickoff_started", issue_number=ISSUE_NUMBER, title=issue.title)
    
    # Validate experiment completeness
    is_valid, missing_fields = validate_experiment_completeness(issue)
    
    if not is_valid:
        post_validation_failure(
            issue,
            dry_run=DRY_RUN,
            issue_number=ISSUE_NUMBER,
            missing_fields=missing_fields,
        )
        return
    
    # Extract W&B run ID
    setup_text = extract_field_from_body(issue.body or "", "setup")
    wandb_run_id = extract_wandb_run_id(setup_text)
    
    # Check for duplicate processing
    if has_kickoff_marker(issue.body or "", wandb_run_id or "unknown"):
        log_event(
            "experiment_kickoff_skipped",
            issue_number=ISSUE_NUMBER,
            reason="already_processed"
        )
        return

    approval_required = bool(policy.get("require_approval_before_kickoff", False))
    linked_parent_refs = extract_parent_refs(issue.body)
    current_label_names = [label.name for label in issue.labels]
    require_parent_issue_reference = bool(policy.get("require_parent_issue_reference", False))
    if require_parent_issue_reference and not linked_parent_refs:
        block_missing_parent_reference(
            issue,
            dry_run=DRY_RUN,
            issue_number=ISSUE_NUMBER,
        )
        return

    if approval_required and APPROVED_LABEL not in current_label_names:
        block_missing_approval(
            issue,
            dry_run=DRY_RUN,
            issue_number=ISSUE_NUMBER,
        )
        return

    is_allowed, reason, _current_state = validate_experiment_transition(
        repo_root,
        current_label_names,
        "in-progress",
    )
    if not is_allowed:
        block_invalid_transition(
            issue,
            dry_run=DRY_RUN,
            issue_number=ISSUE_NUMBER,
            reason=reason,
        )
        return
    
    labels_to_add = [IN_PROGRESS_LABEL]
    if AGENT_LABEL not in current_label_names:
        labels_to_add.append(AGENT_LABEL)
    apply_label_changes(
        issue,
        current_label_names,
        add=labels_to_add,
        remove=[APPROVED_LABEL],
        dry_run=DRY_RUN,
    )
    
    # Add idempotency marker + success comment
    final_comment = (
        build_success_comment(wandb_run_id, linked_parent_refs)
        + "\n"
        + agent_signature("experiment_kickoff", context="experiment kickoff")
        + "\n"
        + create_idempotency_marker(wandb_run_id or "unknown")
    )
    post_comment(issue, final_comment, dry_run=DRY_RUN, description="kickoff")
    
    log_event(
        "experiment_kickoff_complete",
        issue_number=ISSUE_NUMBER,
        wandb_run_id=wandb_run_id or None,
        labels_applied=labels_to_add
    )


if __name__ == "__main__":
    main()
