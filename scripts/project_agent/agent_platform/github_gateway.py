"""GitHub REST gateway — thin PyGithub wrapper for operations not available in GraphQL.

All write paths respect dry_run.  Callers receive typed objects rather
than raw dicts so the rest of the code doesn't depend on github.Github.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_platform.context import AgentContext
from agent_platform.errors import AgentError, DryRunViolation, ResourceNotFound
from agent_platform.telemetry import log_event


@dataclass(frozen=True)
class IssueData:
    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    milestone_number: int | None
    assignees: list[str]
    html_url: str
    node_id: str


@dataclass(frozen=True)
class PRData:
    number: int
    title: str
    body: str
    state: str
    merged: bool
    head_ref: str
    base_ref: str
    html_url: str
    node_id: str
    labels: list[str]


@dataclass(frozen=True)
class CommentData:
    id: int
    body: str
    user: str
    html_url: str


class GitHubGateway:
    """REST-over-PyGithub wrapper.  Instantiate once per agent run."""

    def __init__(self, ctx: AgentContext) -> None:
        from github import Auth, Github  # deferred import — keeps module testable

        self._gh = Github(auth=Auth.Token(ctx.github_token), per_page=100)
        self._repo_obj = self._gh.get_repo(ctx.repo_name)
        self.dry_run = ctx.dry_run
        self.repo_name = ctx.repo_name

    def _guard_write(self, operation: str) -> None:
        if self.dry_run:
            raise DryRunViolation(f"REST write {operation!r} blocked: DRY_RUN=true")

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def get_issue(self, issue_number: int) -> IssueData:
        try:
            raw = self._repo_obj.get_issue(issue_number)
            return self._issue_to_dataclass(raw)
        except Exception as exc:
            raise ResourceNotFound(f"Issue #{issue_number}: {exc}") from exc

    def create_issue(
        self,
        title: str,
        body: str,
        *,
        labels: list[str] | None = None,
        milestone_number: int | None = None,
        assignees: list[str] | None = None,
    ) -> IssueData:
        self._guard_write("create_issue")
        kwargs: dict[str, Any] = {}
        if labels:
            kwargs["labels"] = labels
        if milestone_number is not None:
            kwargs["milestone"] = self._repo_obj.get_milestone(milestone_number)
        if assignees:
            kwargs["assignees"] = assignees

        try:
            raw = self._repo_obj.create_issue(title=title, body=body, **kwargs)
            log_event("issue_created", number=raw.number, title=title)
            return self._issue_to_dataclass(raw)
        except Exception as exc:
            raise AgentError(f"create_issue failed: {exc}") from exc

    def add_issue_comment(self, issue_number: int, body: str) -> CommentData:
        self._guard_write("add_issue_comment")
        try:
            raw = self._repo_obj.get_issue(issue_number)
            comment = raw.create_comment(body)
            log_event("issue_commented", issue=issue_number, comment_id=comment.id)
            return CommentData(
                id=comment.id,
                body=comment.body,
                user=comment.user.login if comment.user else "",
                html_url=comment.html_url,
            )
        except DryRunViolation:
            raise
        except Exception as exc:
            raise AgentError(f"add_issue_comment failed: {exc}") from exc

    def update_issue_body(self, issue_number: int, body: str) -> None:
        self._guard_write("update_issue_body")
        try:
            raw = self._repo_obj.get_issue(issue_number)
            raw.edit(body=body)
            log_event("issue_body_updated", issue=issue_number)
        except DryRunViolation:
            raise
        except Exception as exc:
            raise AgentError(f"update_issue_body failed: {exc}") from exc

    def add_labels(self, issue_number: int, label_names: list[str]) -> None:
        self._guard_write("add_labels")
        if not label_names:
            return
        try:
            raw = self._repo_obj.get_issue(issue_number)
            for name in label_names:
                raw.add_to_labels(name)
            log_event("labels_added_rest", issue=issue_number, labels=label_names)
        except DryRunViolation:
            raise
        except Exception as exc:
            raise AgentError(f"add_labels failed: {exc}") from exc

    def remove_labels(self, issue_number: int, label_names: list[str]) -> None:
        self._guard_write("remove_labels")
        if not label_names:
            return
        try:
            raw = self._repo_obj.get_issue(issue_number)
            for name in label_names:
                try:
                    raw.remove_from_labels(name)
                except Exception:
                    pass  # label not present — idempotent
            log_event("labels_removed_rest", issue=issue_number, labels=label_names)
        except DryRunViolation:
            raise
        except Exception as exc:
            raise AgentError(f"remove_labels failed: {exc}") from exc

    def get_issue_comments(self, issue_number: int) -> list[CommentData]:
        try:
            raw = self._repo_obj.get_issue(issue_number)
            return [
                CommentData(
                    id=c.id,
                    body=c.body,
                    user=c.user.login if c.user else "",
                    html_url=c.html_url,
                )
                for c in raw.get_comments()
            ]
        except Exception as exc:
            raise AgentError(f"get_issue_comments failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    def get_pr(self, pr_number: int) -> PRData:
        try:
            raw = self._repo_obj.get_pull(pr_number)
            return self._pr_to_dataclass(raw)
        except Exception as exc:
            raise ResourceNotFound(f"PR #{pr_number}: {exc}") from exc

    def add_pr_comment(self, pr_number: int, body: str) -> CommentData:
        self._guard_write("add_pr_comment")
        try:
            raw = self._repo_obj.get_pull(pr_number)
            comment = raw.create_issue_comment(body)
            log_event("pr_commented", pr=pr_number, comment_id=comment.id)
            return CommentData(
                id=comment.id,
                body=comment.body,
                user=comment.user.login if comment.user else "",
                html_url=comment.html_url,
            )
        except DryRunViolation:
            raise
        except Exception as exc:
            raise AgentError(f"add_pr_comment failed: {exc}") from exc

    def update_pr_body(self, pr_number: int, body: str) -> None:
        self._guard_write("update_pr_body")
        try:
            raw = self._repo_obj.get_pull(pr_number)
            raw.edit(body=body)
            log_event("pr_body_updated", pr=pr_number)
        except DryRunViolation:
            raise
        except Exception as exc:
            raise AgentError(f"update_pr_body failed: {exc}") from exc

    def get_pr_comments(self, pr_number: int) -> list[CommentData]:
        try:
            raw = self._repo_obj.get_pull(pr_number)
            return [
                CommentData(
                    id=c.id,
                    body=c.body,
                    user=c.user.login if c.user else "",
                    html_url=c.html_url,
                )
                for c in raw.get_issue_comments()
            ]
        except Exception as exc:
            raise AgentError(f"get_pr_comments failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Milestones
    # ------------------------------------------------------------------

    def get_milestone(self, milestone_number: int) -> dict[str, Any]:
        try:
            raw = self._repo_obj.get_milestone(milestone_number)
            return {
                "number": raw.number,
                "title": raw.title,
                "due_on": raw.due_on.isoformat() if raw.due_on else None,
                "open_issues": raw.open_issues,
                "closed_issues": raw.closed_issues,
            }
        except Exception as exc:
            raise ResourceNotFound(f"Milestone #{milestone_number}: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _issue_to_dataclass(raw: Any) -> IssueData:
        return IssueData(
            number=raw.number,
            title=raw.title,
            body=raw.body or "",
            state=raw.state,
            labels=[lbl.name for lbl in raw.labels],
            milestone_number=raw.milestone.number if raw.milestone else None,
            assignees=[a.login for a in raw.assignees],
            html_url=raw.html_url,
            node_id=raw.raw_data.get("node_id", ""),
        )

    @staticmethod
    def _pr_to_dataclass(raw: Any) -> PRData:
        return PRData(
            number=raw.number,
            title=raw.title,
            body=raw.body or "",
            state=raw.state,
            merged=raw.merged,
            head_ref=raw.head.ref,
            base_ref=raw.base.ref,
            html_url=raw.html_url,
            node_id=raw.raw_data.get("node_id", ""),
            labels=[lbl.name for lbl in raw.labels],
        )

