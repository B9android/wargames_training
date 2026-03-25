# SPDX-License-Identifier: MIT
"""Platform error taxonomy — typed exceptions for controlled exit paths."""
from __future__ import annotations


class AgentError(RuntimeError):
    """Base for all agent failures that should produce a controlled exit with a user-facing message."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ContractError(AgentError):
    """A required environment variable or input is missing or invalid."""


class DryRunViolation(AgentError):
    """A mutation was attempted while DRY_RUN=true. Indicates a code-path bug."""


class PolicyViolation(AgentError):
    """An orchestration policy rejected a lifecycle transition or action."""


class TransitionError(PolicyViolation):
    """State-machine transition was not permitted."""

    def __init__(self, current: str, target: str, reason: str) -> None:
        super().__init__(f"Invalid transition {current!r} → {target!r}: {reason}")
        self.current = current
        self.target = target
        self.reason = reason


class GraphQLError(AgentError):
    """A GitHub GraphQL query or mutation returned an error."""


class ResourceNotFound(AgentError):
    """A referenced GitHub resource (issue, PR, milestone) does not exist."""
