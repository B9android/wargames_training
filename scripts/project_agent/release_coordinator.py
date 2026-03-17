"""Release coordinator: sync milestone closeouts into changelog and roadmap."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from github import Auth, Github

from common import agent_signature, log_event, require_env, retry


def issue_lines(issues: list[object], *, limit: int = 30) -> str:
    """Render closed issue bullets for release notes."""
    if not issues:
        return "- None"
    return "\n".join(f"- #{issue.number} {issue.title}" for issue in issues[:limit])


def build_changelog_block(milestone, closed_issues: list[object], marker: str, run_date: str) -> str:
    """Build a changelog block for a closed milestone."""
    return (
        "\n---\n\n"
        f"## [{milestone.title}] - {run_date}\n\n"
        "### Completed\n"
        f"{issue_lines(closed_issues)}\n\n"
        f"{agent_signature('release_coordinator', context='milestone release sync')}\n"
        f"{marker}\n"
    )


def build_roadmap_block(milestone, closed_count: int, marker: str, run_date: str) -> str:
    """Build a roadmap automation update block for a closed milestone."""
    return (
        "\n## Automation Updates\n\n"
        f"- {run_date}: Milestone `{milestone.title}` closed with {closed_count} completed issues.\n"
        f"  {agent_signature('release_coordinator', context='milestone release sync')}\n"
        f"{marker}\n\n"
        "---\n\n"
    )


def insert_roadmap_update(roadmap_content: str, update_block: str) -> str:
    """Insert roadmap update block after the document intro separator."""
    separator = "\n---\n\n"
    split_idx = roadmap_content.find(separator)
    if split_idx == -1:
        return update_block + roadmap_content
    insert_at = split_idx + len(separator)
    return roadmap_content[:insert_at] + update_block + roadmap_content[insert_at:]


def already_synced(changelog_content: str, roadmap_content: str, marker: str) -> bool:
    """Return True if both docs already contain the milestone sync marker."""
    return marker in changelog_content and marker in roadmap_content


def sync_release_docs_contents(
    changelog_content: str,
    roadmap_content: str,
    *,
    milestone,
    closed_issues: list[object],
    marker: str,
    run_date: str,
) -> tuple[str, str, bool]:
    """Return updated doc contents and whether a change is needed."""
    if already_synced(changelog_content, roadmap_content, marker):
        return changelog_content, roadmap_content, False

    updated_changelog = changelog_content
    updated_roadmap = roadmap_content

    if marker not in updated_changelog:
        updated_changelog += build_changelog_block(milestone, closed_issues, marker, run_date)
    if marker not in updated_roadmap:
        updated_roadmap = insert_roadmap_update(
            updated_roadmap,
            build_roadmap_block(milestone, len(closed_issues), marker, run_date),
        )

    has_changes = updated_changelog != changelog_content or updated_roadmap != roadmap_content
    return updated_changelog, updated_roadmap, has_changes


def main() -> None:
    """Sync docs from a closed milestone."""
    env = require_env(["REPO_NAME", "GITHUB_TOKEN", "MILESTONE_NUMBER"])
    repo_name = env["REPO_NAME"]
    milestone_number = int(env["MILESTONE_NUMBER"])
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"

    repo_root = Path(__file__).resolve().parents[2]
    changelog_path = repo_root / "docs" / "CHANGELOG.md"
    roadmap_path = repo_root / "docs" / "ROADMAP.md"
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    marker = f"<!-- agent:release-sync:{milestone_number} -->"

    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)
    milestone = retry(lambda: repo.get_milestone(milestone_number))

    if milestone.state != "closed":
        log_event(
            "release_sync_skipped",
            milestone_number=milestone_number,
            milestone_state=milestone.state,
            reason="milestone_not_closed",
        )
        print(f"Skipped: milestone #{milestone_number} is not closed")
        return

    closed_issues = [
        issue
        for issue in retry(lambda: list(repo.get_issues(state="closed", milestone=milestone)))
        if getattr(issue, "pull_request", None) is None
    ]

    changelog_content = changelog_path.read_text(encoding="utf-8")
    roadmap_content = roadmap_path.read_text(encoding="utf-8")

    changelog_content, roadmap_content, has_changes = sync_release_docs_contents(
        changelog_content,
        roadmap_content,
        milestone=milestone,
        closed_issues=closed_issues,
        marker=marker,
        run_date=run_date,
    )

    if not has_changes:
        log_event(
            "release_sync_skipped",
            milestone_number=milestone_number,
            reason="already_synced",
        )
        print(f"Skipped: release sync marker already present for milestone #{milestone_number}")
        return

    if dry_run:
        print(f"[dry-run] Would update {changelog_path}")
        print(f"[dry-run] Would update {roadmap_path}")
    else:
        changelog_path.write_text(changelog_content, encoding="utf-8")
        roadmap_path.write_text(roadmap_content, encoding="utf-8")

    log_event(
        "release_sync_complete",
        milestone_number=milestone_number,
        milestone_title=milestone.title,
        closed_issue_count=len(closed_issues),
        dry_run=dry_run,
    )
    print(f"{'[dry-run] ' if dry_run else ''}Synced release docs for milestone #{milestone_number}")


if __name__ == "__main__":
    main()