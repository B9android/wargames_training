"""Experiment kickoff agent: validate readiness, move to in-progress, and comment metadata."""

from __future__ import annotations

import re

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
    IN_PROGRESS_LABEL,
    apply_label_changes,
    extract_parent_refs,
    load_experiment_policy,
    post_comment,
    repo_root_from_file,
    validate_experiment_transition,
)

REQUIRED_FIELDS = ["hypothesis", "setup"]
WANDB_RUN_PATTERN = r"(?:https://wandb\.ai/[^/]+/[^/]+/runs/)?([a-z0-9]+)"


def _extract_field(body: str, field_name: str) -> str:
    field_header = f"### {field_name.replace('_', ' ').title()}"
    if field_header not in body:
        return ""
    pattern = re.escape(field_header) + r"\n(.+?)(?=\n###|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_wandb_run_id(setup_text: str) -> str:
    for line in setup_text.split("\n"):
        if "wandb" in line.lower() or "w&b" in line.lower():
            match = re.search(WANDB_RUN_PATTERN, line)
            if match:
                return match.group(1)
    return ""


def _validate_completeness(issue) -> tuple[bool, list[str]]:
    missing = []
    for field in REQUIRED_FIELDS:
        if not _extract_field(issue.body or "", field):
            missing.append(field)
    return len(missing) == 0, missing


def _marker(run_id: str) -> str:
    return f"<!-- agent:experiment-kickoff:{run_id} -->"


def _has_marker(body: str, run_id: str) -> bool:
    return _marker(run_id) in (body or "")


def _build_success_comment(wandb_run_id: str, linked_parent_refs: list[int]) -> str:
    msg = "✅ Experiment initialized and validated.\n\n"
    msg += "**Extracted Metadata:**\n"
    msg += "- Status: In Progress\n"
    if wandb_run_id:
        msg += f"- W&B Run ID: `{wandb_run_id}`\n"
        msg += f"- W&B URL: https://wandb.ai/runs/{wandb_run_id}\n"
    else:
        msg += "- W&B Run ID: Not found in setup\n"
    msg += f"- Linked parent issues: {', '.join(f'#{ref}' for ref in linked_parent_refs)}\n\n"
    msg += "Next steps:\n"
    msg += "1. Monitor training progress via W&B dashboard\n"
    if wandb_run_id:
        msg += f"2. Use /monitor {wandb_run_id} command to sync results to this issue\n"
    msg += "3. Fill in Results section when training completes\n"
    msg += "4. Update Outcome dropdown with result status\n"
    return msg


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
    repo_root = repo_root_from_file(__file__)
    policy = load_experiment_policy(repo_root)

    is_valid, missing_fields = _validate_completeness(issue)
    if not is_valid:
        error_msg = (
            "❌ Experiment validation failed. Missing required fields:\n\n"
            + "\n".join(f"- {f}" for f in missing_fields)
            + "\n\nPlease fill in these fields in the issue description before submitting the run.\n\n"
            + agent_signature("experiment_kickoff", context="experiment validation")
        )
        post_comment(issue, error_msg, dry_run=ctx.dry_run, description="validation error")
        results.append(ActionResult("validate_fields", f"issue #{issue_number}", ActionStatus.SKIPPED, f"Missing fields: {missing_fields}"))
        return RunResult(agent="experiment_kickoff", dry_run=ctx.dry_run, actions=results)

    setup_text = _extract_field(issue.body or "", "setup")
    wandb_run_id = _extract_wandb_run_id(setup_text)

    if _has_marker(issue.body or "", wandb_run_id or "unknown"):
        results.append(ActionResult("idempotency_check", f"issue #{issue_number}", ActionStatus.SKIPPED, "Already processed"))
        return RunResult(agent="experiment_kickoff", dry_run=ctx.dry_run, actions=results)

    parent_refs = extract_parent_refs(issue.body)
    current_labels = [label.name for label in issue.labels]

    if policy.get("require_parent_issue_reference", False) and not parent_refs:
        msg = (
            "⏸️ Experiment kickoff is blocked until this issue is linked to parent work.\n\n"
            "Add a reference like `Part of #123` or `Related to #123` in the issue body, then rerun kickoff.\n\n"
            + agent_signature("experiment_kickoff", context="dependency validation")
        )
        post_comment(issue, msg, dry_run=ctx.dry_run, description="dependency validation")
        results.append(ActionResult("validate_parent_link", f"issue #{issue_number}", ActionStatus.SKIPPED, "Missing parent reference"))
        return RunResult(agent="experiment_kickoff", dry_run=ctx.dry_run, actions=results)

    if policy.get("require_approval_before_kickoff", False) and APPROVED_LABEL not in current_labels:
        msg = (
            "⏸️ Experiment kickoff is blocked until approval is recorded.\n\n"
            f"Add the `{APPROVED_LABEL}` label after review, then rerun kickoff for this issue.\n\n"
            + agent_signature("experiment_kickoff", context="approval gate")
        )
        post_comment(issue, msg, dry_run=ctx.dry_run, description="approval gate")
        results.append(ActionResult("validate_approval", f"issue #{issue_number}", ActionStatus.SKIPPED, "Approval label missing"))
        return RunResult(agent="experiment_kickoff", dry_run=ctx.dry_run, actions=results)

    allowed, reason, _ = validate_experiment_transition(repo_root, current_labels, "in-progress")
    if not allowed:
        msg = (
            "⏸️ Experiment kickoff could not advance the issue lifecycle.\n\n"
            f"Transition rejected: `{reason}`\n\n"
            + agent_signature("experiment_kickoff", context="state transition validation")
        )
        post_comment(issue, msg, dry_run=ctx.dry_run, description="transition validation")
        results.append(ActionResult("validate_transition", f"issue #{issue_number}", ActionStatus.SKIPPED, reason))
        return RunResult(agent="experiment_kickoff", dry_run=ctx.dry_run, actions=results)

    labels_to_add = [IN_PROGRESS_LABEL]
    if AGENT_LABEL not in current_labels:
        labels_to_add.append(AGENT_LABEL)
    apply_label_changes(issue, current_labels, add=labels_to_add, remove=[APPROVED_LABEL], dry_run=ctx.dry_run)
    results.append(ActionResult("apply_labels", f"issue #{issue_number}", ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS, metadata={"added": labels_to_add, "removed": [APPROVED_LABEL]}))

    comment = (
        _build_success_comment(wandb_run_id, parent_refs)
        + "\n"
        + agent_signature("experiment_kickoff", context="experiment kickoff")
        + "\n"
        + _marker(wandb_run_id or "unknown")
    )
    post_comment(issue, comment, dry_run=ctx.dry_run, description="kickoff")
    results.append(ActionResult("post_kickoff_comment", f"issue #{issue_number}", ActionStatus.DRY_RUN if ctx.dry_run else ActionStatus.SUCCESS))

    log_event("experiment_kickoff_complete", issue_number=issue_number, wandb_run_id=wandb_run_id or None, labels_applied=labels_to_add)
    summary.decision(f"Kickoff completed for issue #{issue_number}; wandb_run_id={wandb_run_id or 'none'}")
    return RunResult(agent="experiment_kickoff", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("experiment_kickoff", _run)
