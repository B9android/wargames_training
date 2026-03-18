"""Release coordinator: sync milestone closeouts into changelog and roadmap."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from pathlib import Path

from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event
from common import agent_signature, require_env

try:
    github_module = importlib.import_module("github")
    Auth = github_module.Auth
    Github = github_module.Github
except Exception:  # pragma: no cover - optional in local analysis envs
    Auth = None  # type: ignore[assignment]
    Github = None  # type: ignore[assignment]


def _issue_lines(issues: list[object], *, limit: int = 30) -> str:
    if not issues:
        return "- None"
    return "\n".join(f"- #{issue.number} {issue.title}" for issue in issues[:limit])


def _build_changelog_block(milestone, closed_issues: list[object], marker: str, run_date: str) -> str:
    return (
        "\n---\n\n"
        f"## [{milestone.title}] - {run_date}\n\n"
        "### Completed\n"
        f"{_issue_lines(closed_issues)}\n\n"
        f"{agent_signature('release_coordinator', context='milestone release sync')}\n"
        f"{marker}\n"
    )


def _build_roadmap_block(milestone, closed_count: int, marker: str, run_date: str) -> str:
    return (
        "\n## Automation Updates\n\n"
        f"- {run_date}: Milestone `{milestone.title}` closed with {closed_count} completed issues.\n"
        f"  {agent_signature('release_coordinator', context='milestone release sync')}\n"
        f"{marker}\n\n"
        "---\n\n"
    )


def _insert_roadmap_update(roadmap_content: str, update_block: str) -> str:
    separator = "\n---\n\n"
    split_idx = roadmap_content.find(separator)
    if split_idx == -1:
        return update_block + roadmap_content
    insert_at = split_idx + len(separator)
    return roadmap_content[:insert_at] + update_block + roadmap_content[insert_at:]


def _sync_docs(
    changelog_content: str,
    roadmap_content: str,
    *,
    milestone,
    closed_issues: list[object],
    marker: str,
    run_date: str,
) -> tuple[str, str, bool]:
    if marker in changelog_content and marker in roadmap_content:
        return changelog_content, roadmap_content, False

    updated_changelog = changelog_content
    updated_roadmap = roadmap_content

    if marker not in updated_changelog:
        updated_changelog += _build_changelog_block(milestone, closed_issues, marker, run_date)
    if marker not in updated_roadmap:
        updated_roadmap = _insert_roadmap_update(
            updated_roadmap,
            _build_roadmap_block(milestone, len(closed_issues), marker, run_date),
        )

    changed = updated_changelog != changelog_content or updated_roadmap != roadmap_content
    return updated_changelog, updated_roadmap, changed


# Backward-compatible helper names used by existing unit tests.
issue_lines = _issue_lines
build_changelog_block = _build_changelog_block
insert_roadmap_update = _insert_roadmap_update
sync_release_docs_contents = _sync_docs


def already_synced(changelog_content: str, roadmap_content: str, marker: str) -> bool:
    return marker in changelog_content and marker in roadmap_content


def main() -> None:
    """Legacy test-friendly entrypoint preserved for compatibility."""
    env = require_env(["REPO_NAME", "GITHUB_TOKEN", "MILESTONE_NUMBER"])
    repo_name = env["REPO_NAME"]
    milestone_number = int(env["MILESTONE_NUMBER"])
    dry_run = False

    if Github is None or Auth is None:
        raise RuntimeError("PyGithub is required for legacy main()")
    gh = Github(auth=Auth.Token(env["GITHUB_TOKEN"]))
    repo = gh.get_repo(repo_name)

    repo_root = Path(__file__).resolve().parents[2]
    changelog_path = repo_root / "docs" / "CHANGELOG.md"
    roadmap_path = repo_root / "docs" / "ROADMAP.md"
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    marker = f"<!-- agent:release-sync:{milestone_number} -->"

    milestone = repo.get_milestone(milestone_number)
    if milestone.state != "closed":
        log_event("release_sync_skipped", reason="milestone_not_closed", milestone_number=milestone_number)
        return

    closed_issues = [
        issue
        for issue in repo.get_issues(state="closed", milestone=milestone)
        if getattr(issue, "pull_request", None) is None
    ]

    changelog_content = changelog_path.read_text(encoding="utf-8")
    roadmap_content = roadmap_path.read_text(encoding="utf-8")

    if already_synced(changelog_content, roadmap_content, marker):
        log_event("release_sync_skipped", reason="already_synced", milestone_number=milestone_number)
        return

    new_changelog, new_roadmap, has_changes = _sync_docs(
        changelog_content,
        roadmap_content,
        milestone=milestone,
        closed_issues=closed_issues,
        marker=marker,
        run_date=run_date,
    )

    if has_changes and not dry_run:
        changelog_path.write_text(new_changelog, encoding="utf-8")
        roadmap_path.write_text(new_roadmap, encoding="utf-8")

    log_event(
        "release_sync_complete",
        milestone_number=milestone_number,
        milestone_title=milestone.title,
        closed_issue_count=len(closed_issues),
        dry_run=dry_run,
    )


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    milestone_number = ctx.require_milestone()
    results: list[ActionResult] = []

    repo_root = Path(__file__).resolve().parents[2]
    changelog_path = repo_root / "docs" / "CHANGELOG.md"
    roadmap_path = repo_root / "docs" / "ROADMAP.md"
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    marker = f"<!-- agent:release-sync:{milestone_number} -->"

    milestone = rest._repo_obj.get_milestone(milestone_number)
    if milestone.state != "closed":
        results.append(ActionResult("validate_milestone_closed", f"milestone #{milestone_number}", ActionStatus.SKIPPED, "Milestone is not closed"))
        return RunResult(agent="release_coordinator", dry_run=ctx.dry_run, actions=results)

    closed_issues = [
        issue
        for issue in rest._repo_obj.get_issues(state="closed", milestone=milestone)
        if getattr(issue, "pull_request", None) is None
    ]

    changelog_content = changelog_path.read_text(encoding="utf-8")
    roadmap_content = roadmap_path.read_text(encoding="utf-8")

    new_changelog, new_roadmap, has_changes = _sync_docs(
        changelog_content,
        roadmap_content,
        milestone=milestone,
        closed_issues=closed_issues,
        marker=marker,
        run_date=run_date,
    )

    if not has_changes:
        results.append(ActionResult("sync_release_docs", f"milestone #{milestone_number}", ActionStatus.SKIPPED, "Already synced"))
        return RunResult(agent="release_coordinator", dry_run=ctx.dry_run, actions=results)

    if ctx.dry_run:
        results.append(ActionResult("sync_release_docs", f"milestone #{milestone_number}", ActionStatus.DRY_RUN))
    else:
        changelog_path.write_text(new_changelog, encoding="utf-8")
        roadmap_path.write_text(new_roadmap, encoding="utf-8")
        results.append(ActionResult("sync_release_docs", f"milestone #{milestone_number}", ActionStatus.SUCCESS, metadata={"closed_issue_count": len(closed_issues)}))

    log_event(
        "release_sync_complete",
        milestone_number=milestone_number,
        milestone_title=milestone.title,
        closed_issue_count=len(closed_issues),
        dry_run=ctx.dry_run,
    )
    summary.decision(f"Release docs synced for milestone #{milestone_number}")
    return RunResult(agent="release_coordinator", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("release_coordinator", _run)
