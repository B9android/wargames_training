"""Project Syncer â€” sync issue labels/milestone into project board custom fields."""
from __future__ import annotations

from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary

STORY_POINTS_FIELD = "Story Points"


def _label_mappings() -> dict[str, tuple[str, str, str]]:
    """Map labels to project field updates: label -> (field, value, type)."""
    return {
        "status: triaged": ("Status", "Triaged", "SINGLE_SELECT"),
        "status: approved": ("Status", "Approved", "SINGLE_SELECT"),
        "status: in-progress": ("Status", "In Progress", "SINGLE_SELECT"),
        "status: blocked": ("Status", "Blocked", "SINGLE_SELECT"),
        "status: complete": ("Status", "Done", "SINGLE_SELECT"),
        "v1": ("Version", "v1", "SINGLE_SELECT"),
        "v2": ("Version", "v2", "SINGLE_SELECT"),
        "v3": ("Version", "v3", "SINGLE_SELECT"),
        "v4": ("Version", "v4", "SINGLE_SELECT"),
        "priority: critical": (STORY_POINTS_FIELD, "8", "NUMBER"),
        "priority: high": (STORY_POINTS_FIELD, "5", "NUMBER"),
        "priority: medium": (STORY_POINTS_FIELD, "3", "NUMBER"),
        "priority: low": (STORY_POINTS_FIELD, "1", "NUMBER"),
    }


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    issue_number = ctx.require_issue()
    results: list[ActionResult] = []

    summary.checkpoint("fetch_issue", f"Fetching issue #{issue_number}")
    issue = rest.get_issue(issue_number)
    labels = set(issue.labels)

    field_updates: dict[str, tuple[str, str]] = {}
    for label_name, (field, value, field_type) in _label_mappings().items():
        if label_name in labels:
            field_updates[field] = (value, field_type)

    # Milestone -> version heuristic
    if issue.milestone_number is not None:
        milestone = rest.get_milestone(issue.milestone_number)
        title = milestone.get("title", "")
        for v in ("v1", "v2", "v3", "v4"):
            if v in title.lower():
                field_updates["Version"] = (v, "SINGLE_SELECT")
                break

    if not field_updates:
        results.append(ActionResult("derive_field_updates", f"issue #{issue_number}", ActionStatus.SKIPPED,
                                    "No matching labels or milestone mappings"))
        return RunResult(agent="project_syncer", dry_run=ctx.dry_run, actions=results)

    summary.checkpoint("sync", f"Syncing {len(field_updates)} field(s) to project board")
    if ctx.dry_run:
        results.append(ActionResult("sync_fields", f"issue #{issue_number}", ActionStatus.DRY_RUN,
                                    metadata={"field_updates": field_updates}))
    else:
        ok = gql.ensure_issue_in_project_with_fields(issue_number, field_updates=field_updates)
        if ok:
            results.append(ActionResult("sync_fields", f"issue #{issue_number}", ActionStatus.SUCCESS,
                                        metadata={"field_updates": field_updates}))
        else:
            results.append(ActionResult("sync_fields", f"issue #{issue_number}", ActionStatus.FAILED,
                                        "Project sync failed"))

    return RunResult(agent="project_syncer", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("project_syncer", _run)

