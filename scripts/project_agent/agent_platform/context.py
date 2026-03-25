# SPDX-License-Identifier: MIT
"""Platform agent context — parses and validates the environment contract for every agent invocation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from agent_platform.errors import ContractError
from agent_platform.telemetry import log_event


# ---------------------------------------------------------------------------
# Required env vars shared by every agent
# ---------------------------------------------------------------------------
_UNIVERSAL_REQUIRED = ("GITHUB_TOKEN", "REPO_NAME")


@dataclass(frozen=True)
class AgentContext:
    """Immutable runtime context for one agent invocation.

    Construct via :func:`AgentContext.from_env` — never build manually in
    production code.
    """

    github_token: str
    repo_name: str
    dry_run: bool

    # Optional identifiers populated by action-specific agents
    issue_number: int | None = None
    pr_number: int | None = None
    milestone_number: int | None = None
    sprint_name: str | None = None
    wandb_run_id: str | None = None
    action: str | None = None
    agent_context_text: str | None = None   # free-text context for issue_writer
    target_version: str = "v1"

    # Derived from REPO_NAME
    repo_owner: str = field(init=False, default="")
    repo_slug: str = field(init=False, default="")

    def __post_init__(self) -> None:
        owner, _, slug = self.repo_name.partition("/")
        object.__setattr__(self, "repo_owner", owner)
        object.__setattr__(self, "repo_slug", slug)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, *, extra_required: Iterable[str] = ()) -> "AgentContext":
        """Build a validated context from environment variables.

        Raises :class:`ContractError` if any required env var is absent.
        """
        missing: list[str] = []
        env: dict[str, str] = {}

        for key in (*_UNIVERSAL_REQUIRED, *extra_required):
            val = os.environ.get(key, "").strip()
            if not val:
                missing.append(key)
            else:
                env[key] = val

        if missing:
            raise ContractError(f"Missing required environment variables: {', '.join(missing)}")

        def _int_env(key: str) -> int | None:
            raw = os.environ.get(key, "").strip()
            try:
                return int(raw) if raw else None
            except ValueError:
                raise ContractError(f"Environment variable {key}={raw!r} must be an integer")

        ctx = cls(
            github_token=env["GITHUB_TOKEN"],
            repo_name=env["REPO_NAME"],
            dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
            issue_number=_int_env("ISSUE_NUMBER"),
            pr_number=_int_env("PR_NUMBER"),
            milestone_number=_int_env("MILESTONE_NUMBER"),
            sprint_name=os.environ.get("SPRINT_NAME", "").strip() or None,
            wandb_run_id=os.environ.get("WANDB_RUN_ID", "").strip() or None,
            action=os.environ.get("ACTION", "").strip() or None,
            agent_context_text=os.environ.get("AGENT_CONTEXT", "").strip() or None,
            target_version=os.environ.get("TARGET_VERSION", "v1").strip() or "v1",
        )
        log_event(
            "agent_context_loaded",
            repo=ctx.repo_name,
            dry_run=ctx.dry_run,
            issue=ctx.issue_number,
            pr=ctx.pr_number,
        )
        return ctx

    # ------------------------------------------------------------------
    # Assertions helpers
    # ------------------------------------------------------------------

    def require_issue(self) -> int:
        if self.issue_number is None:
            raise ContractError("ISSUE_NUMBER is required for this action but was not provided")
        return self.issue_number

    def require_pr(self) -> int:
        if self.pr_number is None:
            raise ContractError("PR_NUMBER is required for this action but was not provided")
        return self.pr_number

    def require_milestone(self) -> int:
        if self.milestone_number is None:
            raise ContractError("MILESTONE_NUMBER is required for this action but was not provided")
        return self.milestone_number

    def require_sprint(self) -> str:
        if not self.sprint_name:
            raise ContractError("SPRINT_NAME is required for this action but was not provided")
        return self.sprint_name

    def require_wandb_run(self) -> str:
        if not self.wandb_run_id:
            raise ContractError("WANDB_RUN_ID is required for this action but was not provided")
        return self.wandb_run_id

