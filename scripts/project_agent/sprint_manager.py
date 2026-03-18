"""Sprint Manager â€” start/close/auto-transition sprint assignments via project fields."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from agent_platform.context import AgentContext
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.github_gateway import GitHubGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from sprint_assigner import auto_assign_issues_to_sprint


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    action = os.environ.get("ACTION", "auto-transition")
    results: list[ActionResult] = []

    if action == "start":
        sprint_name = ctx.require_sprint()
        summary.checkpoint("start", f"Starting sprint '{sprint_name}'")
        sprint_id = gql.get_active_sprint_id()
        if not sprint_id:
            results.append(ActionResult("resolve_active_sprint", sprint_name, ActionStatus.FAILED,
                                        "No active sprint/iteration found"))
            return RunResult(agent="sprint_manager", dry_run=ctx.dry_run, actions=results)
        if ctx.dry_run:
            summary.checkpoint("dry_run", f"Would auto-assign issues to sprint '{sprint_name}'")
            results.append(ActionResult("start_sprint", sprint_name, ActionStatus.DRY_RUN))
        else:
            # auto_assign expects a PyGithub repo object; use internal one from rest
            assigned = auto_assign_issues_to_sprint(gql, rest._repo_obj, sprint_id, max_issues=30)
            summary.checkpoint("assigned", f"Assigned {assigned} issues to sprint '{sprint_name}'")
            results.append(ActionResult("start_sprint", sprint_name, ActionStatus.SUCCESS,
                                        metadata={"assigned": assigned}))

    elif action == "close":
        sprint_name = ctx.require_sprint()
        summary.checkpoint("close", f"Closing sprint '{sprint_name}'")
        if ctx.dry_run:
            results.append(ActionResult("close_sprint", sprint_name, ActionStatus.DRY_RUN))
        else:
            # State transition logic remains policy-based/manual for now.
            log_event("sprint_closed", sprint=sprint_name)
            results.append(ActionResult("close_sprint", sprint_name, ActionStatus.SUCCESS))

    elif action == "auto-transition":
        summary.checkpoint("auto_transition", "Running sprint auto-transition")
        cfg = _load_config()
        enabled = cfg.get("policies", {}).get("sprint", {}).get("auto_transition", False)
        if not enabled:
            summary.decision("Sprint auto-transition disabled by policy")
            results.append(ActionResult("auto_transition", ctx.repo_name, ActionStatus.SKIPPED,
                                        "Disabled in orchestration config"))
            return RunResult(agent="sprint_manager", dry_run=ctx.dry_run, actions=results)

        sprint_id = gql.get_active_sprint_id()
        if not sprint_id:
            results.append(ActionResult("resolve_active_sprint", ctx.repo_name, ActionStatus.SKIPPED,
                                        "No active sprint/iteration found"))
            return RunResult(agent="sprint_manager", dry_run=ctx.dry_run, actions=results)

        if ctx.dry_run:
            results.append(ActionResult("auto_assign", sprint_id, ActionStatus.DRY_RUN))
        else:
            assigned = auto_assign_issues_to_sprint(gql, rest._repo_obj, sprint_id, max_issues=50)
            summary.checkpoint("assigned", f"Auto-assigned {assigned} issues")
            results.append(ActionResult("auto_assign", sprint_id, ActionStatus.SUCCESS,
                                        metadata={"assigned": assigned}))
    else:
        results.append(ActionResult("validate_action", action, ActionStatus.FAILED,
                                    f"Unknown action: {action}"))

    return RunResult(agent="sprint_manager", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("sprint_manager", _run)

