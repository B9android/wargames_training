"""Platform result and action-outcome models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


@dataclass
class ActionResult:
    """Outcome of a single atomic agent action (create issue, update field, etc.)."""

    action: str          # e.g. "create_issue", "update_field", "add_label"
    resource_id: str     # e.g. "#45", "field:Status", "label:priority: high"
    status: ActionStatus
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (ActionStatus.SUCCESS, ActionStatus.DRY_RUN)


@dataclass
class RunResult:
    """Aggregate result across all actions in one agent invocation."""

    agent: str
    dry_run: bool
    actions: list[ActionResult] = field(default_factory=list)

    def record(self, result: ActionResult) -> ActionResult:
        self.actions.append(result)
        return result

    def record_success(self, action: str, resource_id: str, **metadata: Any) -> ActionResult:
        status = ActionStatus.DRY_RUN if self.dry_run else ActionStatus.SUCCESS
        return self.record(ActionResult(action=action, resource_id=resource_id, status=status, metadata=metadata))

    def record_failure(self, action: str, resource_id: str, error: str, **metadata: Any) -> ActionResult:
        return self.record(ActionResult(action=action, resource_id=resource_id, status=ActionStatus.FAILED, error=error, metadata=metadata))

    def record_skip(self, action: str, reason: str, **metadata: Any) -> ActionResult:
        return self.record(ActionResult(action=action, resource_id="—", status=ActionStatus.SKIPPED, error=reason, metadata=metadata))

    @classmethod
    def success(cls, agent: str, dry_run: bool, action: str, resource_id: str, **metadata: Any) -> "RunResult":
        result = cls(agent=agent, dry_run=dry_run)
        result.record_success(action, resource_id, **metadata)
        return result

    @classmethod
    def failure(
        cls,
        agent: str,
        dry_run: bool,
        action: str,
        resource_id: str,
        error: str,
        **metadata: Any,
    ) -> "RunResult":
        result = cls(agent=agent, dry_run=dry_run)
        result.record_failure(action, resource_id, error, **metadata)
        return result

    @classmethod
    def skip(cls, agent: str, dry_run: bool, reason: str, action: str = "noop", **metadata: Any) -> "RunResult":
        result = cls(agent=agent, dry_run=dry_run)
        result.record_skip(action, reason, **metadata)
        return result

    @property
    def successes(self) -> list[ActionResult]:
        return [a for a in self.actions if a.status in (ActionStatus.SUCCESS, ActionStatus.DRY_RUN)]

    @property
    def failures(self) -> list[ActionResult]:
        return [a for a in self.actions if a.status == ActionStatus.FAILED]

    @property
    def skipped(self) -> list[ActionResult]:
        return [a for a in self.actions if a.status == ActionStatus.SKIPPED]

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    @property
    def partial(self) -> bool:
        return bool(self.successes) and bool(self.failures)
