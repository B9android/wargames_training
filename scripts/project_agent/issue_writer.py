# SPDX-License-Identifier: MIT
"""Issue Writer Agent — generates and creates follow-up GitHub issues from context."""
from __future__ import annotations

import json
import importlib
from datetime import datetime, timezone

from dependency_resolver import blocked_parent_rollup, dependency_blocker_context, extract_issue_references
from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event

AGENT_CREATED_LABEL = "status: agent-created"

SYSTEM_PROMPT = """
You are a project management agent for wargames_training.
Generate 1-5 actionable follow-up GitHub issues.
Respond ONLY with JSON:
{
  "issues": [
    {
      "title": "Issue title",
      "body": "Markdown body",
      "labels": ["label1"],
      "milestone": "M1: 1v1 Competence",
      "priority": "high"
    }
  ]
}
"""


def _build_prompt(
    context: str,
    target_version: str,
    all_labels: list[str],
    all_milestones: list[str],
    open_issues: list[dict[str, object]],
    dependency_rollup: list[dict[str, object]],
) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Target version: {target_version}\n"
        f"Available labels: {json.dumps(all_labels)}\n"
        f"Available milestones: {json.dumps(all_milestones)}\n"
        f"Existing open issues (avoid duplicates): {json.dumps(open_issues[:30])}\n"
        f"Dependency blockers:\n{dependency_blocker_context(dependency_rollup)}\n"
        "Generate follow-up issues."
    )


# Backward-compatible helpers used by existing tests.
def extract_parent_issue_numbers(text: str) -> list[int]:
    return extract_issue_references(text)


def build_user_prompt(
    context: str,
    target_version: str,
    all_labels: list[str],
    all_milestones: dict[str, int] | list[str],
    open_issues: list[dict[str, object]],
    dependency_rollup: list[dict[str, object]],
) -> str:
    milestone_titles = list(all_milestones.keys()) if isinstance(all_milestones, dict) else list(all_milestones)
    return _build_prompt(
        context,
        target_version,
        all_labels,
        milestone_titles,
        open_issues,
        dependency_rollup,
    )


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    if not ctx.agent_context_text:
        raise ValueError("AGENT_CONTEXT is required for issue_writer")

    target_version = ctx.target_version or "v1"
    parent_issues = extract_issue_references(ctx.agent_context_text)
    results: list[ActionResult] = []

    all_labels = [lbl.name for lbl in rest._repo_obj.get_labels()]
    all_milestones = [m.title for m in rest._repo_obj.get_milestones(state="open")]
    open_issue_objects = list(rest._repo_obj.get_issues(state="open"))
    open_issues = [{"number": i.number, "title": i.title} for i in open_issue_objects[:30]]
    dependency_rollup = blocked_parent_rollup(open_issue_objects)

    prompt = _build_prompt(
        ctx.agent_context_text,
        target_version,
        all_labels,
        all_milestones,
        open_issues,
        dependency_rollup,
    )

    # Real OpenAI key from environment (decoupled from AgentContext requirement)
    import os
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is required for issue_writer")
    openai_cls = importlib.import_module("openai").OpenAI
    ai = openai_cls(api_key=openai_key)

    summary.checkpoint("generate", "Generating follow-up issues with AI")
    response = ai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        timeout=60,
    )
    parsed = json.loads(response.choices[0].message.content)

    created_numbers: list[int] = []
    active_sprint_id = gql.get_active_sprint_id()

    for issue_data in parsed.get("issues", []):
        title = issue_data["title"]
        labels = [l for l in issue_data.get("labels", []) if l in all_labels]
        if AGENT_CREATED_LABEL not in labels and AGENT_CREATED_LABEL in all_labels:
            labels.append(AGENT_CREATED_LABEL)

        milestone_number = None
        milestone_title = issue_data.get("milestone")
        if milestone_title in all_milestones:
            milestone_obj = next(m for m in rest._repo_obj.get_milestones(state="open") if m.title == milestone_title)
            milestone_number = milestone_obj.number

        parent_links = "\n".join(f"- Part of #{n}" for n in parent_issues) if parent_issues else "- None"
        body = (
            issue_data["body"]
            + "\n\n## Parent Work\n"
            + parent_links
            + "\n\n---\n"
            + f"> Target version: `{target_version}`\n"
            + f"> Generated at: `{datetime.now(timezone.utc).isoformat()}`\n"
            + "<!-- agent:issue_writer -->"
        )

        if ctx.dry_run:
            results.append(ActionResult("create_issue", title, ActionStatus.DRY_RUN,
                                        metadata={"labels": labels, "milestone": milestone_title}))
            continue

        created = rest.create_issue(
            title=title,
            body=body,
            labels=labels,
            milestone_number=milestone_number,
        )
        created_numbers.append(created.number)
        log_event("issue_created", issue_number=created.number, title=title)

        # Project board sync
        fields = {"Version": (target_version, "SINGLE_SELECT")}
        if active_sprint_id:
            fields["Sprint"] = (active_sprint_id, "ITERATION")
        gql.ensure_issue_in_project_with_fields(created.number, field_updates=fields)

        results.append(ActionResult("create_issue", f"issue #{created.number}", ActionStatus.SUCCESS,
                                    metadata={"title": title}))

    summary.decision(f"Issue writer created {len(created_numbers)} issues")
    return RunResult(agent="issue_writer", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("issue_writer", _run)

