"""PR Linker: Auto-links PRs to issues, syncs status on merge."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from github import Auth, Github

from common import add_marker, agent_signature, extract_marker, has_marker, log_event, require_env, retry
from projects_v2 import ProjectsV2Client


DRY_RUN = False
REPO_NAME = ""
PR_NUMBER = 0


def load_orchestration_config() -> dict:
    """Load orchestration.yaml for policies."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_linked_issue(pr, repo) -> int | None:
    """Extract linked issue number from PR body or commits."""
    # Check PR body for issue references (#123, closes #123, etc.)
    body = pr.body or ""
    
    # Patterns: Closes #123, Fixes #123, Resolves #123, or just #123
    patterns = [
        r"(?:closes|fixes|resolves)\s+#(\d+)",
        r"linked to\s+#(\d+)",
        r"(?<!/)#(\d+)(?=/|\s|$)",  # Bare #123
    ]
    
    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            issue_num = int(match.group(1))
            # Verify issue exists and is not the PR itself
            try:
                issue = repo.get_issue(issue_num)
                if not issue.pull_request:  # Ensure it's an issue, not a PR
                    return issue_num
            except Exception:
                pass
    
    # Check PR commit messages
    try:
        for commit in pr.get_commits():
            msg = commit.commit.message
            for pattern in patterns:
                match = re.search(pattern, msg, re.IGNORECASE)
                if match:
                    issue_num = int(match.group(1))
                    try:
                        issue = repo.get_issue(issue_num)
                        if not issue.pull_request:
                            return issue_num
                    except Exception:
                        pass
    except Exception as exc:
        log_event("pr_commits_query_failed", pr=PR_NUMBER, error=str(exc))
    
    return None


def link_pr_to_issue(pr, repo, issue_num: int) -> bool:
    """Create bidirectional link between PR and issue."""
    try:
        issue = repo.get_issue(issue_num)
    except Exception as exc:
        log_event("issue_lookup_failed", issue=issue_num, pr=PR_NUMBER, error=str(exc))
        return False
    
    # Add link marker to PR body
    if not has_marker(pr.body or "", "linked-issue"):
        new_pr_body = add_marker(pr.body or "", "linked-issue", str(issue_num))
        if not DRY_RUN:
            try:
                retry(lambda body=new_pr_body: pr.edit(body=body))
                log_event("pr_body_updated", pr=PR_NUMBER, issue=issue_num)
            except Exception as exc:
                log_event("pr_body_update_failed", pr=PR_NUMBER, error=str(exc))
    
    # Add link marker to issue body
    if not has_marker(issue.body or "", "linked-pr"):
        new_issue_body = add_marker(issue.body or "", "linked-pr", str(PR_NUMBER))
        if not DRY_RUN:
            try:
                retry(lambda body=new_issue_body: issue.edit(body=body))
                log_event("issue_body_updated", issue=issue_num, pr=PR_NUMBER)
            except Exception as exc:
                log_event("issue_body_update_failed", issue=issue_num, error=str(exc))
    
    # Post cross-reference comment on issue
    comment = f"""## Linked PR

PR #{PR_NUMBER} ([{pr.title}]({pr.html_url})) is linked to this issue.

{agent_signature('pr_linker', context='PR linking')}
"""
    
    if DRY_RUN:
        print(f"[dry-run] Would post link comment to issue #{issue_num}")
    else:
        try:
            retry(lambda body=comment: issue.create_comment(body))
        except Exception as exc:
            log_event("pr_link_comment_failed", issue=issue_num, pr=PR_NUMBER, error=str(exc))
    
    return True


def sync_pr_status_on_merge(pr, repo, projects_client: ProjectsV2Client | None = None) -> bool:
    """Update linked issue status based on PR merge."""
    if not pr.merged:
        log_event("pr_not_merged", pr=PR_NUMBER)
        return False
    
    # Find linked issue
    issue_num = extract_marker(pr.body or "", "linked-issue")
    if not issue_num:
        # Try to find issue from body
        issue_num = find_linked_issue(pr, repo)
    
    if not issue_num:
        log_event("pr_merged_no_linked_issue", pr=PR_NUMBER)
        return False
    
    try:
        issue_num = int(issue_num)
        issue = repo.get_issue(issue_num)
    except Exception as exc:
        log_event("issue_lookup_on_merge_failed", pr=PR_NUMBER, error=str(exc))
        return False
    
    # Update issue status if still in-progress
    current_status = extract_marker(issue.body or "", "status")
    if current_status in [None, "triaged", "in-progress"]:
        new_status = "in-progress"  # PR merged, but issue may not be complete
        new_body = add_marker(issue.body or "", "status", new_status)
        
        if not DRY_RUN:
            try:
                retry(lambda body=new_body: issue.edit(body=body))
                log_event("issue_status_updated_on_pr_merge", issue=issue_num, pr=PR_NUMBER, status=new_status)
            except Exception as exc:
                log_event("issue_status_update_failed", issue=issue_num, error=str(exc))
    
    # Update project board Status field if projects_client available
    if projects_client and not DRY_RUN:
        try:
            # Get PR metadata for project updates
            git_commit = pr.merge_commit_sha or ""
            
            # Get issue node ID and project item ID
            issue_node_id = projects_client.get_issue_node_id(REPO_NAME, issue_num)
            if issue_node_id:
                item_id = projects_client.get_project_item_id(issue_node_id)
                if item_id:
                    # Update Status field to "In Progress" (PR merged)
                    # Note: This assumes "In Progress" is a valid Status option
                    projects_client.update_field_value(item_id, "Status", "In Progress")
                    
                    # Update Git Commit field if available
                    if git_commit:
                        projects_client.update_field_value(item_id, "Git Commit", git_commit, "TEXT")
                    
                    log_event("pr_merge_project_status_updated", issue=issue_num, pr=PR_NUMBER)
        except Exception as exc:
            log_event("pr_merge_project_sync_failed", issue=issue_num, error=str(exc))
    
    # Post merge comment on issue
    merge_comment = f"""## PR Merged

PR #{PR_NUMBER} has been merged! 🎉

{agent_signature('pr_linker', context='PR merge tracking')}
"""
    
    if DRY_RUN:
        print(f"[dry-run] Would post merge comment to issue #{issue_num}")
    else:
        try:
            retry(lambda body=merge_comment: issue.create_comment(body))
        except Exception as exc:
            log_event("merge_comment_failed", issue=issue_num, pr=PR_NUMBER, error=str(exc))
    
    return True


def main() -> None:
    global DRY_RUN, REPO_NAME, PR_NUMBER
    
    env = require_env(["REPO_NAME", "PR_NUMBER", "GITHUB_TOKEN"])
    REPO_NAME = env["REPO_NAME"]
    PR_NUMBER = int(env["PR_NUMBER"])
    DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
    
    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    repo = gh.get_repo(REPO_NAME)
    pr = repo.get_pull(PR_NUMBER)
    
    # Initialize projects v2 client for board management
    projects_client = None
    try:
        projects_client = ProjectsV2Client(env["GITHUB_TOKEN"], REPO_NAME)
    except Exception as exc:
        log_event("projects_client_init_failed", error=str(exc))
    
    log_event("pr_linker_start", pr=PR_NUMBER, merged=pr.merged)
    
    if pr.merged:
        # Sync status from merge
        sync_pr_status_on_merge(pr, repo, projects_client)
        print(f"{'[dry-run] ' if DRY_RUN else ''}✅ PR #{PR_NUMBER} status synced")
    else:
        # Find and link issue
        issue_num = find_linked_issue(pr, repo)
        if issue_num:
            link_pr_to_issue(pr, repo, issue_num)
            print(f"{'[dry-run] ' if DRY_RUN else ''}✅ PR #{PR_NUMBER} linked to issue #{issue_num}")
        else:
            log_event("pr_no_issue_found", pr=PR_NUMBER)
            print(f"{'[dry-run] ' if DRY_RUN else ''}⚠️ PR #{PR_NUMBER}: no linked issue found")
    
    log_event("pr_linker_complete", pr=PR_NUMBER, dry_run=DRY_RUN)


if __name__ == "__main__":
    main()
