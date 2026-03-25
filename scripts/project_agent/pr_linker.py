"""PR Linker â€” auto-links PRs to issues and syncs project board status.

Entry point for orchestration.yml:pr_link job.
All logic is platform-native: no global state, no broad exception catches.
"""
from __future__ import annotations

import re

from common import add_marker, extract_marker, has_marker
from agent_platform.context import AgentContext
from agent_platform.errors import ResourceNotFound
from agent_platform.github_gateway import GitHubGateway, PRData
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event


# ------------------------------------------------------------------
# Issue discovery â€” GraphQL first, body-parse fallback
# ------------------------------------------------------------------

def _find_linked_issue_graphql(gql: GraphQLGateway, pr_number: int) -> int | None:
    """Use GitHub's closing-issues-references via GraphQL (most reliable)."""
    linked = gql.get_pr_linked_issues(pr_number)
    if linked:
        log_event("pr_linked_issue_via_graphql", pr=pr_number, issues=linked)
        return linked[0]
    return None


def _find_linked_issue_body(body: str, pr_number: int) -> int | None:
    """Fallback: parse PR body for issue references."""
    patterns = [
        r"(?:closes|fixes|resolves)\s+#(\d+)",
        r"linked\s+to\s+#(\d+)",
        r"(?<![/\w])#(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, body or "", re.IGNORECASE)
        if m:
            candidate = int(m.group(1))
            if candidate != pr_number:
                log_event("pr_linked_issue_via_body", pr=pr_number, issue=candidate)
                return candidate
    return None


def _find_linked_issue(
    gql: GraphQLGateway, pr_number: int, pr_body: str
) -> int | None:
    issue_num = _find_linked_issue_graphql(gql, pr_number)
    if issue_num:
        return issue_num
    return _find_linked_issue_body(pr_body, pr_number)


# ------------------------------------------------------------------
# Link operations
# ------------------------------------------------------------------

def _link_pr_to_issue(
    rest: GitHubGateway,
    pr: PRData,
    issue_num: int,
    summary: ExecutionSummary,
) -> list[ActionResult]:
    results: list[ActionResult] = []
    summary.checkpoint("link", f"Linking PR #{pr.number} \u2192 issue #{issue_num}")

    # 1. Annotate PR body (use PyGithub directly for PR edit)
    if not has_marker(pr.body, "linked-issue"):
        new_body = add_marker(pr.body, "linked-issue", str(issue_num))
        try:
            rest.update_pr_body(pr.number, new_body)
            results.append(ActionResult("annotate_pr_body", f"PR #{pr.number}", ActionStatus.SUCCESS))
            log_event("pr_body_annotated", pr=pr.number, issue=issue_num)
        except Exception as exc:
            results.append(ActionResult("annotate_pr_body", f"PR #{pr.number}", ActionStatus.FAILED, str(exc)))
            log_event("pr_body_annotate_failed", pr=pr.number, error=str(exc))

    # 2. Annotate issue body
    try:
        issue = rest.get_issue(issue_num)
    except ResourceNotFound as exc:
        results.append(ActionResult("lookup_issue", f"issue #{issue_num}", ActionStatus.FAILED, str(exc)))
        return results

    if not has_marker(issue.body, "linked-pr"):
        new_body = add_marker(issue.body, "linked-pr", str(pr.number))
        try:
            rest.update_issue_body(issue_num, new_body)
            results.append(ActionResult("annotate_issue_body", f"issue #{issue_num}", ActionStatus.SUCCESS))
        except Exception as exc:
            results.append(ActionResult("annotate_issue_body", f"issue #{issue_num}", ActionStatus.FAILED, str(exc)))

    # 3. Cross-reference comment on issue
    comment = (
        f"## \U0001f517 Linked PR\n\n"
        f"PR #{pr.number} ([{pr.title}]({pr.html_url})) is linked to this issue.\n\n"
        f"<!-- agent:pr_linker -->"
    )
    try:
        rest.add_issue_comment(issue_num, comment)
        results.append(ActionResult("post_link_comment", f"issue #{issue_num}", ActionStatus.SUCCESS))
        summary.checkpoint("commented", f"Posted link comment on issue #{issue_num}")
    except Exception as exc:
        results.append(ActionResult("post_link_comment", f"issue #{issue_num}", ActionStatus.FAILED, str(exc)))

    summary.decision(f"PR #{pr.number} linked to issue #{issue_num}")
    return results


def _sync_merge_status(
    rest: GitHubGateway,
    gql: GraphQLGateway,
    pr: PRData,
    summary: ExecutionSummary,
) -> list[ActionResult]:
    results: list[ActionResult] = []
    summary.checkpoint("merge_sync", f"PR #{pr.number} merged \u2014 syncing linked issue status")

    # Discover linked issue
    issue_num: int | None = None
    raw_marker = extract_marker(pr.body, "linked-issue")
    if raw_marker:
        try:
            issue_num = int(raw_marker)
        except ValueError:
            pass
    if not issue_num:
        issue_num = _find_linked_issue(gql, pr.number, pr.body)

    if not issue_num:
        log_event("pr_merged_no_linked_issue", pr=pr.number)
        summary.decision("No linked issue found; skipping status sync")
        results.append(ActionResult(
            "find_linked_issue", f"PR #{pr.number}", ActionStatus.SKIPPED,
            "No linked issue found"
        ))
        return results

    # Update issue status marker
    try:
        issue = rest.get_issue(issue_num)
        current_status = extract_marker(issue.body, "status")
        if current_status not in ("complete", "done"):
            new_body = add_marker(issue.body, "status", "in-progress")
            rest.update_issue_body(issue_num, new_body)
            results.append(ActionResult("update_issue_status", f"issue #{issue_num}", ActionStatus.SUCCESS))
    except Exception as exc:
        results.append(ActionResult("update_issue_status", f"issue #{issue_num}", ActionStatus.FAILED, str(exc)))

    # Update project board Status field
    try:
        gql.ensure_issue_in_project_with_fields(
            issue_num,
            field_updates={"Status": ("In Progress", "SINGLE_SELECT")},
        )
        results.append(ActionResult("update_project_status", f"issue #{issue_num}", ActionStatus.SUCCESS))
        summary.checkpoint("board_synced", f"Board Status \u2192 'In Progress' for issue #{issue_num}")
    except Exception as exc:
        results.append(ActionResult(
            "update_project_status", f"issue #{issue_num}", ActionStatus.FAILED, str(exc)
        ))

    # Merge comment on issue
    merge_comment = (
        f"## \U0001f389 PR Merged\n\n"
        f"PR #{pr.number} has been merged!\n\n"
        f"<!-- agent:pr_linker -->"
    )
    try:
        rest.add_issue_comment(issue_num, merge_comment)
        results.append(ActionResult("post_merge_comment", f"issue #{issue_num}", ActionStatus.SUCCESS))
    except Exception as exc:
        results.append(ActionResult("post_merge_comment", f"issue #{issue_num}", ActionStatus.FAILED, str(exc)))

    summary.decision(f"Merge status synced for issue #{issue_num}")
    return results


# ------------------------------------------------------------------
# Main agent function
# ------------------------------------------------------------------

def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    pr_number = ctx.require_pr()
    results: list[ActionResult] = []

    summary.checkpoint("fetch_pr", f"Fetching PR #{pr_number}")
    pr = rest.get_pr(pr_number)

    if pr.merged:
        summary.decision("PR is merged \u2192 running merge-status sync")
        results.extend(_sync_merge_status(rest, gql, pr, summary))
    else:
        summary.checkpoint("find_issue", "Discovering linked issue")
        issue_num = _find_linked_issue(gql, pr_number, pr.body)
        if issue_num:
            results.extend(_link_pr_to_issue(rest, pr, issue_num, summary))
        else:
            summary.decision("No linked issue found \u2014 PR body has no issue reference")
            results.append(ActionResult(
                "find_linked_issue", f"PR #{pr_number}", ActionStatus.SKIPPED,
                "No issue reference in PR body or GraphQL closing references"
            ))

    return RunResult(agent="pr_linker", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("pr_linker", _run)

