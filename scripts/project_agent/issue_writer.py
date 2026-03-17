"""Issue Writer Agent: generates and creates follow-up GitHub issues from context."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from common import agent_signature, log_event, require_env, retry
from dependency_resolver import (
    blocked_parent_rollup,
    dependency_blocker_context,
    extract_issue_references,
)
from projects_v2 import ProjectsV2Client

AGENT_CREATED_LABEL = "status: agent-created"


def extract_parent_issue_numbers(text: str) -> list[int]:
    return extract_issue_references(text)


SYSTEM_PROMPT = """
You are a project management agent for wargames_training, a reinforcement learning
research project training AI agents to control Napoleonic military battalions.

Given a context (training results, findings, blockers, etc.), generate 1-5
GitHub issues that represent the logical next steps. Each issue should be
actionable, specific, and well-scoped for a single developer.

Respond ONLY with valid JSON:
{
  "issues": [
    {
      "title": "Issue title",
      "body": "Full GitHub markdown body with ## sections",
      "labels": ["label1", "label2"],
      "milestone": "M1: 1v1 Competence",
      "priority": "high"
    }
  ]
}

Issue body should include:
## Context
## Task
## Acceptance Criteria
- [ ] item
## Notes

If parent references are provided, include a `## Parent Work` section that links them.
"""

def build_user_prompt(
    context: str,
    target_version: str,
    all_labels: list[str],
    all_milestones: dict[str, int],
    open_issues: list[dict[str, object]],
    dependency_rollup: list[dict[str, object]],
) -> str:
    return f"""
Context provided:
{context}

Target version: {target_version}
Available labels: {json.dumps(all_labels)}
Available milestones: {json.dumps(list(all_milestones.keys()))}
Existing open issues (avoid duplicates): {json.dumps(open_issues[:30])}
Dependency blockers (prioritize these when proposing next work):
{dependency_blocker_context(dependency_rollup)}

Generate follow-up issues.
"""


def main() -> None:
    from github import Auth, Github
    from openai import OpenAI

    env = require_env(["REPO_NAME", "GITHUB_TOKEN", "OPENAI_API_KEY"])
    repo_name = env["REPO_NAME"]
    context = os.environ.get("AGENT_CONTEXT", "")
    target_version = os.environ.get("TARGET_VERSION", "v1")
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    parent_issues = extract_parent_issue_numbers(context)

    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    ai = OpenAI(api_key=env["OPENAI_API_KEY"])
    repo = gh.get_repo(repo_name)
    
    # Initialize projects v2 client for board management
    projects_client = ProjectsV2Client(env["GITHUB_TOKEN"], repo_name)
    
    # Get active sprint ID for auto-assignment
    active_sprint_id = None
    try:
        active_sprint_id = projects_client.get_active_sprint_id()
    except Exception as exc:
        log_event("active_sprint_query_failed", error=str(exc))

    all_labels = [label.name for label in retry(lambda: list(repo.get_labels()))]
    all_milestones = {
        milestone.title: milestone.number
        for milestone in retry(lambda: list(repo.get_milestones(state="open")))
    }
    open_issue_objects = retry(lambda: list(repo.get_issues(state="open")))
    open_issues = [
        {"number": issue.number, "title": issue.title}
        for issue in open_issue_objects[:30]
    ]
    dependency_rollup = blocked_parent_rollup(open_issue_objects)

    user_prompt = build_user_prompt(
        context,
        target_version,
        all_labels,
        all_milestones,
        open_issues,
        dependency_rollup,
    )

    try:
        response = retry(
            lambda: ai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                timeout=60,
            )
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as exc:
        log_event("issue_writer_ai_failed", error=str(exc))
        raise

    created: list[int] = []
    planned_titles: list[str] = []

    for issue_data in result.get("issues", []):
        labels_to_apply = [label for label in issue_data.get("labels", []) if label in all_labels]
        if AGENT_CREATED_LABEL not in labels_to_apply:
            labels_to_apply.append(AGENT_CREATED_LABEL)

        milestone_obj = None
        milestone_title = issue_data.get("milestone")
        if milestone_title and milestone_title in all_milestones:
            milestone_obj = retry(
                lambda milestone_key=milestone_title: repo.get_milestone(all_milestones[milestone_key])
            )

        if parent_issues:
            parent_links = "\n".join(
                f"- #{number}: https://github.com/{repo_name}/issues/{number}"
                for number in parent_issues
            )
            parent_section = "\n\n## Parent Work\n" + "\n".join(
                f"- Part of #{number}" for number in parent_issues
            )
        else:
            parent_links = "- None detected from context"
            parent_section = ""

        attribution = (
            "\n\n---\n"
            + agent_signature("issue_writer", context="follow-up issue generation")
            + "\n"
            f"> Target version: `{target_version}`\n"
            f"> Generated at: `{datetime.now(timezone.utc).isoformat()}`\n"
            "> Parent references:\n"
            f"{parent_links}"
        )

        body = issue_data["body"] + parent_section + attribution
        issue_title = issue_data["title"]

        if dry_run:
            planned_titles.append(issue_title)
            print(f"[dry-run] Would create issue: {issue_title}")
            continue

        new_issue = retry(
            lambda title=issue_title, body=body, labels=tuple(labels_to_apply), milestone=milestone_obj: repo.create_issue(
                title=title,
                body=body,
                labels=labels,
                milestone=milestone,
            )
        )
        created.append(new_issue.number)
        log_event("issue_created", issue_number=new_issue.number, title=issue_title)
        print(f"Created issue #{new_issue.number}: {issue_title}")
        
        # Add to project board with custom fields
        if projects_client:
            field_updates = {
                "Version": (target_version, "SINGLE_SELECT"),
            }
            if active_sprint_id:
                field_updates["Sprint"] = (active_sprint_id, "ITERATION")
            
            try:
                projects_client.ensure_issue_in_project_with_fields(
                    new_issue.number,
                    field_updates=field_updates,
                )
                log_event("issue_added_to_project", issue=new_issue.number)
            except Exception as exc:
                log_event("project_board_sync_failed", issue=new_issue.number, error=str(exc))

    log_event(
        "issue_writer_complete",
        created_count=len(created),
        created=created,
        dry_run=dry_run,
        planned_count=len(planned_titles),
        planned_titles=planned_titles,
    )
    prefix = "[dry-run] " if dry_run else ""
    total = len(planned_titles) if dry_run else len(created)
    print(f"\n{prefix}Total issues processed: {total}")


if __name__ == "__main__":
    main()
