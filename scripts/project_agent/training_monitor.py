# SPDX-License-Identifier: MIT
"""Training monitor agent for syncing W&B results to experiment issues."""

from __future__ import annotations

import re
from pathlib import Path

from agent_platform.context import AgentContext
from agent_platform.errors import AgentError
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from common import agent_signature
from state_machine import load_state_machine

STATUS_LABEL_TO_STATE = {
    "status: approved": "approved",
    "status: in-progress": "in-progress",
    "status: blocked": "blocked",
    "status: stale": "stale",
    "status: complete": "complete",
    "status: failed": "failed",
    "status: cancelled": "cancelled",
}
TARGET_STATE_TO_LABEL = {
    "approved": "status: approved",
    "in-progress": "status: in-progress",
    "blocked": "status: blocked",
    "stale": "status: stale",
    "complete": "status: complete",
    "failed": "status: failed",
    "cancelled": "status: cancelled",
}
RUN_STATE_TO_TARGET_STATE = {
    "running": "in-progress",
    "finished": "complete",
    "failed": "failed",
    "crashed": "failed",
}


# Backward-compatible public helper names used by existing tests.


def desired_target_state(run_state: str) -> str | None:
    return _desired_target_state(run_state)


def plan_label_changes(current_labels: set[str], target_state: str) -> tuple[list[str], list[str]]:
    return _plan_label_changes(current_labels, target_state)


def validate_run_transition(repo_root: Path, current_state: str, target_state: str) -> tuple[bool, str]:
    return _validate_transition(repo_root, current_state, target_state)


def build_results_comment(run, summary: dict, config: dict, transition_error: str | None) -> str:
    return _build_comment(run, summary, config, transition_error)


def format_runtime(summary: dict) -> str:
    runtime_seconds = summary.get("_runtime")
    return f"{int(runtime_seconds // 60)}m {int(runtime_seconds % 60)}s" if runtime_seconds else "N/A"


def update_issue_outcome(issue, run_state: str, *, run_failed: bool, dry_run: bool, issue_number: int | None = None) -> None:
    _ = issue_number  # compatibility-only argument
    _update_issue_outcome(issue, run_state, run_failed=run_failed, dry_run=dry_run)


def sync_issue_labels(issue, run_state: str, *, repo_root: Path, dry_run: bool, issue_number: int) -> str | None:
    _ = issue_number  # compatibility-only argument
    target_state = _desired_target_state(run_state)
    if target_state is None:
        return None

    current_labels = {label.name for label in issue.labels}
    current_state = _infer_state(current_labels)
    allowed, reason = validate_run_transition(repo_root, current_state, target_state)
    if not allowed:
        if not dry_run:
            issue.create_comment(
                "## Training Run State Sync\n\n"
                "Lifecycle update was skipped because the target transition is invalid.\n\n"
                f"- Current state: `{current_state}`\n"
                f"- Requested state: `{target_state}`\n"
                f"- Reason: `{reason}`\n\n"
                + agent_signature("training_monitor", context="state transition validation")
            )
        return reason

    labels_to_add, labels_to_remove = _plan_label_changes(current_labels, target_state)
    if not dry_run:
        for label in labels_to_remove:
            issue.remove_from_labels(label)
        for label in labels_to_add:
            issue.add_to_labels(label)
    return None


def _update_issue_outcome(issue, run_state: str, *, run_failed: bool, dry_run: bool) -> None:
    outcome_map = {
        "running": "Running",
        "finished": "Success - hypothesis confirmed" if not run_failed else "Mixed - partial confirmation",
        "failed": "Crashed - technical failure",
        "crashed": "Crashed - technical failure",
    }
    new_outcome = outcome_map.get(run_state, "Running")
    body = issue.body or ""
    pattern = r"(### Outcome\n)(.*)(?=\n###|\Z)"
    if not re.search(pattern, body, re.DOTALL):
        return
    updated = re.sub(pattern, r"\1" + new_outcome, body, flags=re.DOTALL)
    if updated == body:
        return
    if not dry_run:
        issue.edit(body=updated)


def _infer_state(label_names: set[str]) -> str:
    for label_name, state in STATUS_LABEL_TO_STATE.items():
        if label_name in label_names:
            return state
    return "triaged"


def _desired_target_state(run_state: str) -> str | None:
    return RUN_STATE_TO_TARGET_STATE.get(run_state)


def _plan_label_changes(current_labels: set[str], target_state: str) -> tuple[list[str], list[str]]:
    status_labels = set(STATUS_LABEL_TO_STATE.keys())
    target_label = TARGET_STATE_TO_LABEL[target_state]
    labels_to_remove = sorted(label for label in current_labels if label in status_labels and label != target_label)
    labels_to_add = [] if target_label in current_labels else [target_label]
    return labels_to_add, labels_to_remove


def _validate_transition(repo_root: Path, current_state: str, target_state: str) -> tuple[bool, str]:
    decision = load_state_machine(repo_root).can_transition("experiment", current_state, target_state)
    return decision.is_allowed, decision.reason


def _build_comment(run, summary: dict, config: dict, transition_error: str | None) -> str:
    metric_rows = "\n".join(
        f"| `{key}` | `{round(value, 4) if isinstance(value, float) else value}` |"
        for key, value in sorted(summary.items())
        if not key.startswith("_") and isinstance(value, (int, float, str, bool))
    )
    config_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in list(config.items())[:20])
    runtime_seconds = summary.get("_runtime")
    runtime = f"{int(runtime_seconds // 60)}m {int(runtime_seconds % 60)}s" if runtime_seconds else "N/A"
    transition_line = f"> Lifecycle sync note: `{transition_error}`\n\n" if transition_error else ""

    return f"""## Training Run Results

**W&B Run:** [{run.name}]({run.url})
**State:** `{run.state}`
**Runtime:** {runtime}

### Summary Metrics

| Metric | Value |
|---|---|
{metric_rows or "| - | No scalar metrics recorded |"}

### Config Highlights

| Parameter | Value |
|---|---|
{config_rows or "| - | No config recorded |"}

---
{agent_signature("training_monitor", context="training result synchronization")}

{transition_line}> [View full run on W&B]({run.url})
"""


def _load_wandb_run(issue, wandb_run_id: str, dry_run: bool):
    import importlib

    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:
        raise AgentError("wandb is required for training_monitor; install it in the runtime environment") from exc

    api = wandb.Api()
    try:
        return api.run(wandb_run_id)
    except Exception as exc:
        if not dry_run:
            issue.create_comment(
                "## Training Run Results\n\n"
                f"Failed to fetch W&B run `{wandb_run_id}`: `{exc}`\n\n"
                "Please verify the run id and WANDB_API_KEY configuration.\n\n"
                + agent_signature("training_monitor", context="training monitor failure handling")
            )
        raise AgentError(f"Could not fetch W&B run {wandb_run_id}: {exc}") from exc


def _sync_lifecycle_labels(issue, run_state: str, dry_run: bool, repo_root: Path) -> str | None:
    target_state = _desired_target_state(run_state)
    if target_state is None:
        return None

    current_labels = {label.name for label in issue.labels}
    current_state = _infer_state(current_labels)
    allowed, reason = _validate_transition(repo_root, current_state, target_state)
    if not allowed:
        if not dry_run:
            issue.create_comment(
                "## Training Run State Sync\n\n"
                "Lifecycle update was skipped because the target transition is invalid.\n\n"
                f"- Current state: `{current_state}`\n"
                f"- Requested state: `{target_state}`\n"
                f"- Reason: `{reason}`\n\n"
                + agent_signature("training_monitor", context="state transition validation")
            )
        return reason

    labels_to_add, labels_to_remove = _plan_label_changes(current_labels, target_state)
    if not dry_run:
        for label in labels_to_remove:
            issue.remove_from_labels(label)
        for label in labels_to_add:
            issue.add_to_labels(label)
    return None


def _record_result(results: list[ActionResult], issue_number: int, run_id: str, run_state: str, dry_run: bool) -> None:
    status = ActionStatus.DRY_RUN if dry_run else ActionStatus.SUCCESS
    results.append(
        ActionResult(
            "post_training_results",
            f"issue #{issue_number}",
            status,
            metadata={"run": run_id, "state": run_state},
        )
    )


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    issue_number = ctx.require_issue()
    wandb_run_id = ctx.require_wandb_run()
    results: list[ActionResult] = []

    issue = rest._repo_obj.get_issue(issue_number)
    run = _load_wandb_run(issue, wandb_run_id, ctx.dry_run)

    summary_data = dict(run.summary)
    config_data = dict(run.config)

    run_failed = getattr(run, "failed", False) or run.state in ("failed", "crashed")
    _update_issue_outcome(issue, run.state, run_failed=run_failed, dry_run=ctx.dry_run)

    transition_error = _sync_lifecycle_labels(
        issue,
        run.state,
        ctx.dry_run,
        Path(__file__).resolve().parents[2],
    )

    comment = _build_comment(run, summary_data, config_data, transition_error)
    if not ctx.dry_run:
        issue.create_comment(comment)
    _record_result(results, issue_number, wandb_run_id, run.state, ctx.dry_run)

    log_event("training_monitor_posted", issue=issue_number, run=wandb_run_id, state=run.state, dry_run=ctx.dry_run)
    summary.decision(f"Training results synced for issue #{issue_number}; run={wandb_run_id}; state={run.state}")
    return RunResult(agent="training_monitor", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("training_monitor", _run)
