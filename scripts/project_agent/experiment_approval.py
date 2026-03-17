"""Experiment approval agent for gated kickoff transitions."""

from __future__ import annotations

from github import Auth, Github

from common import AgentError, agent_signature, log_event, require_env, retry
from experiment_lifecycle import (
    AGENT_LABEL,
    APPROVED_LABEL,
    apply_label_changes,
    extract_parent_refs,
    post_comment,
    repo_root_from_file,
    validate_experiment_transition,
)


def block_missing_parent_reference(issue, *, dry_run: bool, issue_number: int) -> None:
    """Comment and log when an experiment is missing parent linkage."""
    message = (
        "## Experiment Approval\n\n"
        "Approval was blocked because this experiment is not linked to any parent issue or epic.\n\n"
        "Add a reference like `Part of #123` or `Related to #123` in the issue body, then rerun approval.\n\n"
        + agent_signature("experiment_approval", context="dependency validation")
    )
    post_comment(issue, message, dry_run=dry_run, description="missing-link approval")
    log_event(
        "experiment_approval_blocked",
        issue_number=issue_number,
        reason="missing_parent_reference",
        dry_run=dry_run,
    )


def block_invalid_transition(issue, *, dry_run: bool, issue_number: int, current_state: str, reason: str) -> None:
    """Comment and log when approval transition is invalid."""
    message = (
        "## Experiment Approval\n\n"
        "Approval was rejected because the lifecycle transition is invalid.\n\n"
        f"- Current state: `{current_state}`\n"
        f"- Reason: `{reason}`\n\n"
        + agent_signature("experiment_approval", context="experiment approval")
    )
    post_comment(issue, message, dry_run=dry_run, description="invalid approval")
    log_event(
        "experiment_approval_blocked",
        issue_number=issue_number,
        current_state=current_state,
        reason=reason,
        dry_run=dry_run,
    )


def main() -> None:
    """Approve an experiment issue for kickoff."""
    env = require_env(["REPO_NAME", "GITHUB_TOKEN", "ISSUE_NUMBER"])
    repo_name = env["REPO_NAME"]
    issue_number = int(env["ISSUE_NUMBER"])
    dry_run = env.get("DRY_RUN", "false").lower() == "true"
    repo_root = repo_root_from_file(__file__)

    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)
    issue = retry(lambda: repo.get_issue(issue_number))

    if not issue.title.upper().startswith("[EXP]"):
        raise AgentError(f"Issue #{issue_number} is not an [EXP] experiment issue")

    parent_refs = extract_parent_refs(issue.body)
    if not parent_refs:
        block_missing_parent_reference(issue, dry_run=dry_run, issue_number=issue_number)
        return

    current_label_names = [label.name for label in issue.labels]
    is_allowed, reason, current_state = validate_experiment_transition(
        repo_root,
        current_label_names,
        "approved",
    )
    if not is_allowed:
        block_invalid_transition(
            issue,
            dry_run=dry_run,
            issue_number=issue_number,
            current_state=current_state,
            reason=reason,
        )
        return

    labels_to_add = [APPROVED_LABEL]
    if AGENT_LABEL not in current_label_names:
        labels_to_add.append(AGENT_LABEL)
    apply_label_changes(issue, current_label_names, add=labels_to_add, dry_run=dry_run)

    comment = (
        "## Experiment Approval\n\n"
        "This experiment is approved for kickoff.\n\n"
        f"- Previous state: `{current_state}`\n"
        f"- New state label: `{APPROVED_LABEL}`\n\n"
        f"- Linked parent issues: {', '.join(f'#{ref}' for ref in parent_refs)}\n\n"
        + agent_signature("experiment_approval", context="experiment approval")
    )
    post_comment(issue, comment, dry_run=dry_run, description="approval")

    log_event(
        "experiment_approval_complete",
        issue_number=issue_number,
        previous_state=current_state,
        labels_added=labels_to_add,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()