# SPDX-License-Identifier: MIT
"""Experiment approval agent for gated kickoff transitions."""

from __future__ import annotations

from agent_platform.context import AgentContext
from agent_platform.errors import AgentError
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from common import agent_signature
from experiment_lifecycle import (
    AGENT_LABEL,
    APPROVED_LABEL,
    apply_label_changes,
    extract_parent_refs,
    post_comment,
    repo_root_from_file,
    validate_experiment_transition,
)


def _block_missing_parent(issue, *, dry_run: bool, issue_number: int) -> None:
    message = (
        "## Experiment Approval\n\n"
        "Approval was blocked because this experiment is not linked to any parent issue or epic.\n\n"
        "Add a reference like `Part of #123` or `Related to #123` in the issue body, then rerun approval.\n\n"
        + agent_signature("experiment_approval", context="dependency validation")
    )
    post_comment(issue, message, dry_run=dry_run, description="missing-link approval")
    log_event("experiment_approval_blocked", issue_number=issue_number, reason="missing_parent_reference", dry_run=dry_run)


def _block_invalid_transition(
    issue,
    *,
    dry_run: bool,
    issue_number: int,
    current_state: str,
    reason: str,
) -> None:
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


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    issue_number = ctx.require_issue()
    results: list[ActionResult] = []

    summary.checkpoint("fetch_issue", f"Fetching experiment issue #{issue_number}")
    issue = rest._repo_obj.get_issue(issue_number)

    if not issue.title.upper().startswith("[EXP]"):
        raise AgentError(f"Issue #{issue_number} is not an [EXP] experiment issue")

    parent_refs = extract_parent_refs(issue.body)
    if not parent_refs:
        _block_missing_parent(issue, dry_run=ctx.dry_run, issue_number=issue_number)
        results.append(ActionResult("validate_parent_link", f"issue #{issue_number}", ActionStatus.SKIPPED, "Missing parent reference"))
        return RunResult(agent="experiment_approval", dry_run=ctx.dry_run, actions=results)

    current_label_names = [label.name for label in issue.labels]
    repo_root = repo_root_from_file(__file__)
    is_allowed, reason, current_state = validate_experiment_transition(repo_root, current_label_names, "approved")
    if not is_allowed:
        _block_invalid_transition(
            issue,
            dry_run=ctx.dry_run,
            issue_number=issue_number,
            current_state=current_state,
            reason=reason,
        )
        results.append(ActionResult("validate_transition", f"issue #{issue_number}", ActionStatus.SKIPPED, reason))
        return RunResult(agent="experiment_approval", dry_run=ctx.dry_run, actions=results)

    labels_to_add = [APPROVED_LABEL]
    if AGENT_LABEL not in current_label_names:
        labels_to_add.append(AGENT_LABEL)

    apply_label_changes(issue, current_label_names, add=labels_to_add, dry_run=ctx.dry_run)
    results.append(ActionResult("apply_labels", f"issue #{issue_number}", ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS, metadata={"added": labels_to_add}))

    comment = (
        "## Experiment Approval\n\n"
        "This experiment is approved for kickoff.\n\n"
        f"- Previous state: `{current_state}`\n"
        f"- New state label: `{APPROVED_LABEL}`\n\n"
        f"- Linked parent issues: {', '.join(f'#{ref}' for ref in parent_refs)}\n\n"
        + agent_signature("experiment_approval", context="experiment approval")
    )
    post_comment(issue, comment, dry_run=ctx.dry_run, description="approval")
    results.append(ActionResult("post_approval_comment", f"issue #{issue_number}", ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS))

    summary.decision(f"Approved experiment #{issue_number}; labels added={labels_to_add}")
    log_event("experiment_approval_complete", issue_number=issue_number, previous_state=current_state, labels_added=labels_to_add, dry_run=ctx.dry_run)
    return RunResult(agent="experiment_approval", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("experiment_approval", _run)
