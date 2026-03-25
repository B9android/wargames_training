"""Standard agent runner — boilerplate-free entry point for every agent.

Usage
-----
from agent_platform.runner import run_agent

def _run(ctx, gql, rest, summary):
    ...

if __name__ == "__main__":
    run_agent("my_agent", _run)
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Any

from agent_platform.context import AgentContext
from agent_platform.errors import (
    AgentError,
    ContractError,
    DryRunViolation,
    PolicyViolation,
    TransitionError,
)
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionStatus, RunResult
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import begin_run, end_run, log_event

AgentFn = Callable[
    [AgentContext, GraphQLGateway, GitHubGateway, ExecutionSummary],
    RunResult | None,
]


def run_agent(agent_name: str, fn: AgentFn) -> None:
    """Bootstrap, run, and teardown a single agent invocation.

    1. Initialise telemetry (run_id, correlation)
    2. Build AgentContext from environment variables
    3. Construct gateway objects
    4. Call fn(ctx, gql, rest, summary)
    5. Flush UX surfaces (terminal, Actions summary, artifact)
    6. Exit with the appropriate code

    Exit-code contract
    ------------------
    0 — success (or dry-run preview)
    1 — agent error / contract violation / unexpected exception
    2 — dry-run violation (mutation attempted with DRY_RUN=true)
    3 — policy / state-machine violation
    """
    begin_run(agent_name)
    target: str | None = None
    summary: ExecutionSummary | None = None
    result: RunResult | None = None

    try:
        ctx = AgentContext.from_env()
        # Derive a human-readable target label from whatever context we have.
        target = _derive_target(ctx)

        summary = ExecutionSummary(agent_name=agent_name, target=target, dry_run=ctx.dry_run)
        log_event("agent_started", target=target, dry_run=ctx.dry_run)

        gql = GraphQLGateway(
            github_token=ctx.github_token,
            repo_name=ctx.repo_name,
            dry_run=ctx.dry_run,
        )
        rest = GitHubGateway(ctx)

        result = fn(ctx, gql, rest, summary) or RunResult.skip(
            agent=agent_name, dry_run=ctx.dry_run, reason="agent returned nothing"
        )

        summary.set_result(result)

    except ContractError as exc:
        log_event("contract_error", error=str(exc))
        _finalise(summary, agent_name, target, f"Contract error: {exc}")
        end_run(success=False, summary=str(exc))
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(1)

    except DryRunViolation as exc:
        log_event("dry_run_violation", error=str(exc))
        _finalise(summary, agent_name, target, f"Dry-run violation: {exc}")
        end_run(success=False, summary=str(exc))
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(2)

    except PolicyViolation as exc:
        log_event("policy_violation", error=str(exc))
        _finalise(summary, agent_name, target, f"Policy violation: {exc}")
        end_run(success=False, summary=str(exc))
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(3)

    except TransitionError as exc:
        log_event("transition_error", error=str(exc))
        _finalise(summary, agent_name, target, f"State-machine error: {exc}")
        end_run(success=False, summary=str(exc))
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(3)

    except AgentError as exc:
        log_event("agent_error", error=str(exc))
        _finalise(summary, agent_name, target, f"Agent error: {exc}")
        end_run(success=False, summary=str(exc))
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(exc.exit_code)

    except Exception as exc:
        log_event("unexpected_error", error=str(exc), traceback=traceback.format_exc())
        _finalise(summary, agent_name, target, f"Unexpected: {exc}")
        end_run(success=False, summary=str(exc))
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(1)

    else:
        ok = result is not None and result.ok
        end_run(success=ok, summary={
            "ok": ok,
            "successes": len(result.successes) if result else 0,
            "failures": len(result.failures) if result else 0,
            "skipped": len(result.skipped) if result else 0,
        })
        _flush(summary, artifact_dir="agent-artifacts")
        sys.exit(0)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _derive_target(ctx: AgentContext) -> str:
    if ctx.issue_number:
        return f"issue #{ctx.issue_number}"
    if ctx.pr_number:
        return f"PR #{ctx.pr_number}"
    if ctx.milestone_number:
        return f"milestone #{ctx.milestone_number}"
    if ctx.sprint_name:
        return f"sprint {ctx.sprint_name!r}"
    if ctx.wandb_run_id:
        return f"wandb/{ctx.wandb_run_id}"
    return ctx.repo_name


def _finalise(
    summary: ExecutionSummary | None,
    agent_name: str,
    target: str | None,
    error_message: str,
) -> None:
    if summary is None:
        summary = ExecutionSummary(
            agent_name=agent_name,
            target=target or "unknown",
            dry_run=False,
        )
    from agent_platform.result import RunResult

    summary.set_result(
        RunResult.failure(
            agent=agent_name,
            dry_run=summary.dry_run,
            action="run",
            resource_id=target or "unknown",
            error=error_message,
        )
    )


def _flush(summary: ExecutionSummary | None, artifact_dir: str) -> None:
    if summary is None:
        return
    print(summary.render_terminal(), flush=True)
    summary.write_artifacts(artifact_dir)
    summary.write_actions_summary()

