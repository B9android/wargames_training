"""Platform execution summary — three-surface UX output for every agent run.

Surfaces:
  1. Terminal  — human-readable ANSI report with timeline, decisions, mutations.
  2. GitHub issue/PR comment markdown — explains what happened and why.
  3. Artifact JSON + markdown — downloadable run report for every workflow job.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.telemetry import log_event


# ---------------------------------------------------------------------------
# Checkpoint — inline progress milestone for long-running actions
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    label: str
    detail: str = ""
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# ExecutionSummary
# ---------------------------------------------------------------------------

class ExecutionSummary:
    """Accumulates checkpoints and result data; renders three output surfaces."""

    def __init__(self, agent_name: str, target: str, *, dry_run: bool) -> None:
        self.agent_name = agent_name
        self.target = target
        self.dry_run = dry_run
        self._start = time.monotonic()
        self._checkpoints: list[Checkpoint] = []
        self._decisions: list[str] = []   # "why" statements
        self._result: RunResult | None = None

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def checkpoint(self, label: str, detail: str = "") -> None:
        """Record a named progress milestone (shown in Actions step summary)."""
        elapsed = time.monotonic() - self._start
        self._checkpoints.append(Checkpoint(label=label, detail=detail, elapsed=elapsed))
        log_event("checkpoint", label=label, detail=detail, elapsed_s=round(elapsed, 2))

    def decision(self, reason: str) -> None:
        """Record a policy/rule decision for transparency in user-facing output."""
        self._decisions.append(reason)
        log_event("decision", reason=reason)

    def set_result(self, run_result: RunResult) -> None:
        self._result = run_result

    # ------------------------------------------------------------------
    # Surface 1: Terminal / Actions step summary
    # ------------------------------------------------------------------

    def render_terminal(self) -> str:
        elapsed = time.monotonic() - self._start
        prefix = "[dry-run] " if self.dry_run else ""
        r = self._result
        lines: list[str] = [
            "",
            f"{prefix}{'='*62}",
            f"{prefix}  AGENT EXECUTION REPORT — {self.agent_name.upper().replace('_', ' ')}",
            f"{prefix}  Target: {self.target}  |  Elapsed: {elapsed:.1f}s  |  DRY_RUN={self.dry_run}",
            f"{prefix}{'='*62}",
        ]

        # Checkpoints
        if self._checkpoints:
            lines.append(f"{prefix}")
            lines.append(f"{prefix}  PROGRESS")
            for cp in self._checkpoints:
                tag = f"[+{cp.elapsed:.1f}s]"
                lines.append(f"{prefix}  {tag:>10}  \u2713 {cp.label}" + (f"  — {cp.detail}" if cp.detail else ""))

        # Decisions
        if self._decisions:
            lines.append(f"{prefix}")
            lines.append(f"{prefix}  DECISIONS")
            for d in self._decisions:
                lines.append(f"{prefix}    · {d}")

        # Results
        if r:
            lines.append(f"{prefix}")
            if r.successes:
                label = "DRY-RUN WOULD" if self.dry_run else "COMPLETED"
                lines.append(f"{prefix}  \u2705 {label} ({len(r.successes)})")
                for a in r.successes[:12]:
                    lines.append(f"{prefix}    · {a.action}  →  {a.resource_id}")
                if len(r.successes) > 12:
                    lines.append(f"{prefix}    … and {len(r.successes)-12} more")

            if r.failures:
                lines.append(f"{prefix}")
                lines.append(f"{prefix}  \u274c FAILED ({len(r.failures)})")
                for a in r.failures[:6]:
                    lines.append(f"{prefix}    · {a.action}  →  {a.resource_id}:  {a.error}")
                if len(r.failures) > 6:
                    lines.append(f"{prefix}    … and {len(r.failures)-6} more")

            if r.skipped:
                lines.append(f"{prefix}")
                lines.append(f"{prefix}  \u23ed  SKIPPED ({len(r.skipped)})")
                for a in r.skipped[:4]:
                    lines.append(f"{prefix}    · {a.error}")

            lines.append(f"{prefix}")
            if r.ok:
                total = len(r.successes)
                lines.append(f"{prefix}  {'[DRY-RUN] ALL PLANNED' if self.dry_run else '\u2705 ALL'} {total} ACTION(S) SUCCEEDED")
            elif r.partial:
                ok, bad = len(r.successes), len(r.failures)
                lines.append(f"{prefix}  \u26a0\ufe0f  PARTIAL: {ok} succeeded / {bad} failed — review errors above")
            else:
                lines.append(f"{prefix}  \u274c FAILED — no actions completed successfully")

        lines.append(f"{prefix}{'='*62}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Surface 2: GitHub issue/PR comment
    # ------------------------------------------------------------------

    def render_github_comment(self, *, agent_identity: str = "") -> str:
        """Return GitHub-flavored markdown for posting to an issue or PR."""
        r = self._result
        prefix = "\U0001f4dd Dry-run preview" if self.dry_run else "\u2705 Completed"
        identity = agent_identity or self.agent_name.replace("_", " ").title()

        lines = [
            f"## {identity}",
            "",
            f"**Action:** `{self.agent_name}`  |  **Target:** {self.target}  |  **Mode:** `{'dry-run' if self.dry_run else 'live'}`",
            "",
        ]

        # Decisions / policy reasoning
        if self._decisions:
            lines.append("**Why this ran:**")
            for d in self._decisions:
                lines.append(f"- {d}")
            lines.append("")

        # Results table
        if r:
            if r.successes:
                action_label = "Would perform" if self.dry_run else "Performed"
                lines.append(f"**{prefix} — {action_label} {len(r.successes)} action(s):**")
                for a in r.successes[:15]:
                    lines.append(f"- `{a.action}` → {a.resource_id}")
                if len(r.successes) > 15:
                    lines.append(f"- _(\u2026{len(r.successes)-15} more)_")
                lines.append("")

            if r.failures:
                lines.append(f"**\u26a0\ufe0f {len(r.failures)} action(s) failed:**")
                for a in r.failures[:5]:
                    lines.append(f"- `{a.action}` → {a.resource_id}: _{a.error}_")
                lines.append("")

            if r.skipped:
                lines.append(f"**Skipped:** {len(r.skipped)} item(s)")
                for a in r.skipped[:3]:
                    lines.append(f"- _{a.error}_")
                lines.append("")

        # Next actions for the human
        if not self.dry_run and r and r.ok:
            lines.append("**What to do next:** No action needed — automation handled this.")
        elif self.dry_run:
            lines.append("**What to do next:** Re-run this action with `dry_run=false` to apply changes.")
        elif r and r.failures:
            lines.append("**What to do next:** Review failures above and re-run, or handle manually.")

        lines.extend(["", f"> _{self.agent_name}_ handled this automatically via the agent platform."])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Surface 3: Artifact (JSON + markdown run report)
    # ------------------------------------------------------------------

    def write_artifacts(self, out_dir: str | Path = ".") -> tuple[Path, Path]:
        """Write run-report.json and run-report.md to *out_dir*.

        Returns paths to (json_path, md_path).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - self._start
        r = self._result

        payload: dict[str, Any] = {
            "agent": self.agent_name,
            "target": self.target,
            "dry_run": self.dry_run,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "decisions": self._decisions,
            "checkpoints": [
                {"label": c.label, "detail": c.detail, "elapsed_s": round(c.elapsed, 2)}
                for c in self._checkpoints
            ],
            "result": {
                "ok": r.ok if r else None,
                "partial": r.partial if r else None,
                "successes": len(r.successes) if r else 0,
                "failures": len(r.failures) if r else 0,
                "skipped": len(r.skipped) if r else 0,
                "actions": [
                    {
                        "action": a.action,
                        "resource_id": a.resource_id,
                        "status": a.status.value,
                        "error": a.error,
                        "metadata": a.metadata,
                    }
                    for a in (r.actions if r else [])
                ],
            },
        }

        json_path = out_dir / "run-report.json"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        md_path = out_dir / "run-report.md"
        md_path.write_text(self.render_terminal() + "\n\n---\n\n```json\n" + json.dumps(payload, indent=2, default=str) + "\n```\n", encoding="utf-8")

        log_event("artifacts_written", json=str(json_path), md=str(md_path))
        return json_path, md_path

    # ------------------------------------------------------------------
    # GitHub Actions step summary
    # ------------------------------------------------------------------

    def write_actions_summary(self) -> None:
        """Append the terminal report to $GITHUB_STEP_SUMMARY if running in Actions."""
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
        if not summary_path:
            return
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("```\n")
            fh.write(self.render_terminal())
            fh.write("\n```\n\n")
            if self._result:
                fh.write(self.render_github_comment())
        log_event("actions_summary_written", path=summary_path)
