"""State transition validator for project-management automation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common import AgentError


@dataclass(frozen=True)
class TransitionDecision:
    is_allowed: bool
    workflow_type: str
    from_state: str | None
    to_state: str
    reason: str


class StateMachine:
    """Validates lifecycle transitions against configs/orchestration.yaml."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        if not config_path.exists():
            raise AgentError(f"State machine config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        transitions = (raw or {}).get("state_machine", {})
        if not isinstance(transitions, dict) or not transitions:
            raise AgentError("Invalid state machine config: missing state_machine section")

        self._transitions: dict[str, dict[str | None, set[str]]] = {}
        for workflow_type, cfg in transitions.items():
            allowed = cfg.get("allowed_transitions", {}) if isinstance(cfg, dict) else {}
            normalized: dict[str | None, set[str]] = {}
            for source, targets in allowed.items():
                from_state = None if source in (None, "null", "None") else str(source)
                normalized[from_state] = {str(target) for target in (targets or [])}
            self._transitions[workflow_type] = normalized

    @classmethod
    def from_repo_root(cls, repo_root: Path, *, config_rel_path: str = "configs/orchestration.yaml") -> "StateMachine":
        return cls(repo_root / config_rel_path)

    def can_transition(self, workflow_type: str, from_state: str | None, to_state: str) -> TransitionDecision:
        type_config = self._transitions.get(workflow_type)
        if type_config is None:
            return TransitionDecision(
                is_allowed=False,
                workflow_type=workflow_type,
                from_state=from_state,
                to_state=to_state,
                reason=f"Unknown workflow type: {workflow_type}",
            )

        allowed_targets = type_config.get(from_state, set())
        if to_state in allowed_targets:
            return TransitionDecision(
                is_allowed=True,
                workflow_type=workflow_type,
                from_state=from_state,
                to_state=to_state,
                reason="Transition allowed",
            )

        return TransitionDecision(
            is_allowed=False,
            workflow_type=workflow_type,
            from_state=from_state,
            to_state=to_state,
            reason=(
                f"Invalid transition for {workflow_type}: "
                f"{from_state or 'null'} -> {to_state}. "
                f"Allowed: {sorted(allowed_targets)}"
            ),
        )


def load_state_machine(repo_root: str | Path) -> StateMachine:
    """Helper for agents to load the canonical transition config."""
    return StateMachine.from_repo_root(Path(repo_root))
