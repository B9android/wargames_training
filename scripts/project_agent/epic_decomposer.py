"""Epic Decomposer â€” creates child issues from an epic's Implementation Plan.

Entry point for orchestration.yml:epic_decompose* jobs.
Platform-native: per-child checkpoints in ExecutionSummary, GraphQL project sync,
idempotency via has_marker("decomposition-complete").
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from common import add_marker, extract_marker, has_marker
from agent_platform.context import AgentContext
from agent_platform.errors import ResourceNotFound
from agent_platform.github_gateway import GitHubGateway, IssueData
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ------------------------------------------------------------------
# Body parsing
# ------------------------------------------------------------------

def _extract_sections(body: str) -> dict[str, str]:
    """Parse ## Section headers from issue body."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in (body or "").split("\n"):
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = m.group(1).strip()
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def _parse_tasks(plan: str) -> list[dict[str, str]]:
    """Parse bullet items from Implementation Plan section."""
    items: list[dict[str, str]] = []
    for line in plan.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^-\s*\[\s*\]\s*", "", line)
        line = re.sub(r"^-\s*", "", line).strip()
        if not line:
            continue
        if ":" in line:
            title, desc = line.split(":", 1)
            items.append({"title": title.strip(), "description": desc.strip()})
        else:
            items.append({"title": line, "description": ""})
    return items


# ------------------------------------------------------------------
# Child issue creation
# ------------------------------------------------------------------

def _create_child(
    rest: GitHubGateway,
    gql: GraphQLGateway,
    epic: IssueData,
    item: dict[str, str],
    epic_version: str | None,
    summary: ExecutionSummary,
) -> ActionResult:
    title = item["title"]
    body = (
        f"{item.get('description', '')}\n\n"
        f"**Parent Epic:** #{epic.number}\n\n"
        f"<!-- parent-epic:epic-{epic.number} -->\n"
        f"<!-- status:triaged -->"
    )

    summary.checkpoint("create_child", f"Creating child: '{title}'")

    # Create issue
    try:
        child = rest.create_issue(title=title, body=body, labels=["type: feature", "status: triaged"])
    except Exception as exc:
        log_event("child_issue_create_failed", epic=epic.number, title=title, error=str(exc))
        return ActionResult("create_child_issue", title, ActionStatus.FAILED, str(exc))

    log_event("child_issue_created", epic=epic.number, child=child.number, title=title)

    # Sync to project board
    field_updates: dict[str, tuple[str, str]] = {}
    if epic_version:
        field_updates["Version"] = (epic_version, "SINGLE_SELECT")
    sprint_id = gql.get_active_sprint_id()
    if sprint_id:
        field_updates["Sprint"] = (sprint_id, "ITERATION")

    try:
        gql.ensure_issue_in_project_with_fields(child.number, field_updates=field_updates or None)
        summary.checkpoint("board_sync", f"#{child.number} synced to project board")
    except Exception as exc:
        log_event("child_project_sync_failed", child=child.number, error=str(exc))
        # Not fatal â€” child issue was created successfully

    return ActionResult(
        "create_child_issue",
        f"issue #{child.number}",
        ActionStatus.SUCCESS,
        metadata={"title": title, "number": child.number},
    )


# ------------------------------------------------------------------
# Decompose orchestration
# ------------------------------------------------------------------

def _decompose(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    epic_number: int,
    summary: ExecutionSummary,
) -> tuple[list[ActionResult], list[int]]:
    results: list[ActionResult] = []
    created_numbers: list[int] = []

    summary.checkpoint("fetch_epic", f"Fetching epic #{epic_number}")
    try:
        epic = rest.get_issue(epic_number)
    except ResourceNotFound as exc:
        return [ActionResult("fetch_epic", f"issue #{epic_number}", ActionStatus.FAILED, str(exc))], []

    # Idempotency guard
    if has_marker(epic.body, "decomposition-complete"):
        summary.decision(f"Epic #{epic_number} already decomposed â€” skipping")
        log_event("epic_already_decomposed", epic=epic_number)
        results.append(ActionResult("idempotency_check", f"issue #{epic_number}", ActionStatus.SKIPPED,
                                    "Already decomposed"))
        return results, []

    # Verify it's an epic
    if not any(lbl.lower().startswith("type: epic") for lbl in epic.labels):
        summary.decision(f"Issue #{epic_number} is not labeled 'type: epic' â€” aborting")
        results.append(ActionResult("verify_epic", f"issue #{epic_number}", ActionStatus.FAILED,
                                    "Not labeled as an epic"))
        return results, []

    # Parse implementation plan
    sections = _extract_sections(epic.body)
    plan_text: str | None = None
    for key in ("Implementation Plan", "Implementation", "Tasks", "Breakdown"):
        if key in sections:
            plan_text = sections[key]
            break

    if not plan_text:
        summary.decision("No Implementation Plan section found in epic body")
        results.append(ActionResult("parse_plan", f"issue #{epic_number}", ActionStatus.SKIPPED,
                                    "No Implementation Plan section"))
        return results, []

    tasks = _parse_tasks(plan_text)
    if not tasks:
        summary.decision("Implementation Plan has no parseable task items")
        results.append(ActionResult("parse_plan", f"issue #{epic_number}", ActionStatus.SKIPPED,
                                    "No task items found"))
        return results, []

    log_event("epic_tasks_parsed", epic=epic_number, count=len(tasks))
    summary.checkpoint("tasks_parsed", f"Found {len(tasks)} tasks to decompose")

    # Epic metadata for child issues
    epic_version = extract_marker(epic.body, "version")

    # Create child issues
    for i, task in enumerate(tasks, 1):
        summary.checkpoint("progress", f"Creating child {i}/{len(tasks)}: {task['title']}")
        result = _create_child(rest, gql, epic, task, epic_version, summary)
        results.append(result)
        if result.status == ActionStatus.SUCCESS and result.metadata:
            created_numbers.append(result.metadata["number"])

    # Mark epic as decomposed
    if created_numbers and not ctx.dry_run:
        new_body = add_marker(epic.body, "decomposition-complete", str(len(created_numbers)))
        try:
            rest.update_issue_body(epic_number, new_body)
            results.append(ActionResult("mark_decomposed", f"issue #{epic_number}", ActionStatus.SUCCESS))
            log_event("epic_marked_decomposed", epic=epic_number, children=len(created_numbers))
        except Exception as exc:
            results.append(ActionResult("mark_decomposed", f"issue #{epic_number}", ActionStatus.FAILED, str(exc)))

    # Post decomposition comment
    if created_numbers or ctx.dry_run:
        refs = "\n".join(f"- #{n}" for n in created_numbers) if created_numbers else "*(dry-run â€” no issues created)*"
        comment = (
            f"## \U0001f9e9 Epic Decomposed\n\n"
            f"Created **{len(created_numbers)}** child issue(s):\n\n"
            f"{refs}\n\n"
            f"<!-- agent:epic_decomposer -->"
        )
        if ctx.dry_run:
            summary.checkpoint("comment", "[dry-run] Would post decomposition comment")
        else:
            try:
                rest.add_issue_comment(epic_number, comment)
                results.append(ActionResult("post_decomposition_comment", f"issue #{epic_number}", ActionStatus.SUCCESS))
                summary.checkpoint("comment", f"Posted decomposition comment on epic #{epic_number}")
            except Exception as exc:
                results.append(ActionResult("post_decomposition_comment", f"issue #{epic_number}",
                                            ActionStatus.FAILED, str(exc)))

    return results, created_numbers


# ------------------------------------------------------------------
# Main agent function
# ------------------------------------------------------------------

def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    # EPIC_NUMBER env var used by this agent (not ISSUE_NUMBER)
    import os
    epic_env = os.environ.get("EPIC_NUMBER", "")
    if not epic_env:
        epic_number = ctx.require_issue()
    else:
        epic_number = int(epic_env)

    results, created = _decompose(ctx, gql, rest, epic_number, summary)
    summary.decision(f"Decomposed epic #{epic_number} into {len(created)} child issues")
    return RunResult(agent="epic_decomposer", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("epic_decomposer", _run)

