#!/usr/bin/env python3
"""Idempotent bootstrap of GitHub labels and milestones for wargames_training.

Creates all labels defined in configs/orchestration.yaml (type:, priority:,
status: groups) and the five project milestones (M0–M4) with target due dates.

Safe to run repeatedly — existing resources are left untouched.

Usage:
    GITHUB_TOKEN=<token> REPO_NAME=B9android/wargames_training python setup_labels_and_milestones.py
    GITHUB_TOKEN=<token> REPO_NAME=B9android/wargames_training DRY_RUN=true python setup_labels_and_milestones.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Label definitions
# Colours follow GitHub's conventional palette so the label board looks tidy.
# ---------------------------------------------------------------------------

LABEL_DEFINITIONS: dict[str, tuple[str, str]] = {
    # ── type: ──────────────────────────────────────────────────────────────
    "type: bug":        ("d73a4a", "Something isn't working"),
    "type: experiment": ("0075ca", "Training experiment"),
    "type: research":   ("cfd3d7", "Research / investigation task"),
    "type: feature":    ("a2eeef", "New feature or enhancement"),
    "type: epic":       ("7057ff", "Large multi-task initiative"),
    "type: chore":      ("e4e669", "Maintenance / housekeeping"),
    # ── priority: ──────────────────────────────────────────────────────────
    "priority: critical": ("b60205", "Drop-everything severity"),
    "priority: high":     ("d93f0b", "High priority"),
    "priority: medium":   ("fbca04", "Medium priority"),
    "priority: low":      ("0e8a16", "Low priority"),
    # ── status: ────────────────────────────────────────────────────────────
    "status: needs-manual-triage": ("e4e669", "Needs human review before work begins"),
    "status: approved":            ("0075ca", "Approved and ready to start"),
    "status: in-progress":         ("1d76db", "Actively being worked on"),
    "status: blocked":             ("b60205", "Blocked by a dependency"),
    "status: stale":               ("cfd3d7", "No activity in 14+ days"),
    "status: complete":            ("0e8a16", "Work is done"),
    "status: failed":              ("d73a4a", "Attempt failed"),
    "status: cancelled":           ("cfd3d7", "No longer relevant"),
    "status: agent-created":       ("0075ca", "Created by an automation agent"),
    # ── domain: ────────────────────────────────────────────────────────────
    "domain: sim":   ("bfd4f2", "Simulation engine"),
    "domain: ml":    ("bfd4f2", "Machine learning"),
    "domain: env":   ("bfd4f2", "Gymnasium environment"),
    "domain: viz":   ("bfd4f2", "Visualisation"),
    "domain: infra": ("bfd4f2", "Infrastructure / tooling"),
    "domain: eval":  ("bfd4f2", "Evaluation / metrics"),
    # ── v1: work-stream labels ──────────────────────────────────────────────
    "v1: simulation":    ("fef2c0", "v1 simulation engine work"),
    "v1: environment":   ("fef2c0", "v1 Gymnasium environment work"),
    "v1: training":      ("fef2c0", "v1 training pipeline work"),
    "v1: terrain":       ("fef2c0", "v1 terrain system work"),
    "v1: rewards":       ("fef2c0", "v1 reward shaping work"),
    "v1: self-play":     ("fef2c0", "v1 self-play work"),
    "v1: evaluation":    ("fef2c0", "v1 evaluation framework"),
    "v1: visualization": ("fef2c0", "v1 visualisation work"),
    "v1: documentation": ("fef2c0", "v1 documentation work"),
    # ── misc ───────────────────────────────────────────────────────────────
    "status: needs-experiment": ("0075ca", "Requires an experiment to proceed"),
    "enhancement":              ("a2eeef", "Improvement to existing functionality"),
}

# ---------------------------------------------------------------------------
# Milestone definitions  (title → (description, due_date ISO-8601))
# Dates are targets; adjust once real sprint cadence is established.
# ---------------------------------------------------------------------------

MILESTONE_DEFINITIONS: list[dict] = [
    {
        "title": "M0: Project Bootstrap",
        "description": "All foundational tooling, repo structure, CI, and dev environment in place.",
        "due_date": "2026-04-15",
    },
    {
        "title": "M1: 1v1 Competence",
        "description": "Agent reliably defeats the scripted opponent in 1v1 battalion combat.",
        "due_date": "2026-06-30",
    },
    {
        "title": "M2: Terrain & Generalization",
        "description": "Agent performs well across diverse terrain configurations.",
        "due_date": "2026-08-31",
    },
    {
        "title": "M3: Self-Play Baseline",
        "description": "Stable self-play loop producing measurable improvement over time.",
        "due_date": "2026-10-31",
    },
    {
        "title": "M4: v1 Complete",
        "description": "All v1 acceptance criteria met; system ready for v2 multi-agent work.",
        "due_date": "2026-12-31",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_extra_labels_from_config() -> dict[str, tuple[str, str]]:
    """Read any additional label names from orchestration.yaml.

    Returns a dict of name → (color, description) for labels not already
    in LABEL_DEFINITIONS.  Unknown colours default to the GitHub grey.
    """
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        extra: dict[str, tuple[str, str]] = {}
        for group_labels in raw.get("labels", {}).values():
            for label_name in group_labels:
                if label_name not in LABEL_DEFINITIONS:
                    extra[label_name] = ("ededed", "")
        return extra
    except Exception as exc:
        print(f"  [warn] Could not read orchestration.yaml: {exc}", file=sys.stderr)
        return {}


def ensure_label(repo, name: str, color: str, description: str, *, dry_run: bool) -> str:
    """Create label if it doesn't exist. Returns 'created', 'exists', or 'skipped'."""
    try:
        repo.get_label(name)
        return "exists"
    except Exception:
        pass

    if dry_run:
        print(f"  [dry-run] Would create label: {name}")
        return "skipped"

    try:
        repo.create_label(name=name, color=color, description=description)
        print(f"  + created label: {name}")
        return "created"
    except Exception as exc:
        print(f"  ! Could not create label '{name}': {exc}", file=sys.stderr)
        return "skipped"


def ensure_milestone(repo, title: str, description: str, due_date: str, *, dry_run: bool) -> str:
    """Create milestone if it doesn't exist. Returns 'created', 'exists', or 'skipped'."""
    for ms in repo.get_milestones(state="open"):
        if ms.title == title:
            return "exists"

    if dry_run:
        print(f"  [dry-run] Would create milestone: {title} (due {due_date})")
        return "skipped"

    try:
        due_dt = datetime.strptime(due_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        repo.create_milestone(title=title, description=description, due_on=due_dt)
        print(f"  + created milestone: {title} (due {due_date})")
        return "created"
    except Exception as exc:
        print(f"  ! Could not create milestone '{title}': {exc}", file=sys.stderr)
        return "skipped"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo_name = os.environ.get("REPO_NAME")
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"

    if not token or not repo_name:
        print("ERROR: GITHUB_TOKEN and REPO_NAME environment variables must be set.", file=sys.stderr)
        return 1

    try:
        from github import Auth, Github
    except ImportError:
        print("ERROR: PyGithub is not installed. Run: pip install PyGithub", file=sys.stderr)
        return 1

    auth = Auth.Token(token)
    gh = Github(auth=auth)
    repo = gh.get_repo(repo_name)
    print(f"Connected to {repo.full_name}{'  [DRY RUN]' if dry_run else ''}")

    # ------------------------------------------------------------------
    # 1. Labels
    # ------------------------------------------------------------------
    print("\n── Labels ──────────────────────────────────────────────────")
    all_labels = {**LABEL_DEFINITIONS, **load_extra_labels_from_config()}
    stats = {"created": 0, "exists": 0, "skipped": 0}
    for name, (color, description) in sorted(all_labels.items()):
        result = ensure_label(repo, name, color, description, dry_run=dry_run)
        stats[result] += 1
    print(f"Labels: {stats['created']} created, {stats['exists']} already exist, {stats['skipped']} skipped.")

    # ------------------------------------------------------------------
    # 2. Milestones
    # ------------------------------------------------------------------
    print("\n── Milestones ───────────────────────────────────────────────")
    ms_stats = {"created": 0, "exists": 0, "skipped": 0}
    for ms_def in MILESTONE_DEFINITIONS:
        result = ensure_milestone(
            repo,
            ms_def["title"],
            ms_def["description"],
            ms_def["due_date"],
            dry_run=dry_run,
        )
        ms_stats[result] += 1
    print(
        f"Milestones: {ms_stats['created']} created, {ms_stats['exists']} already exist, "
        f"{ms_stats['skipped']} skipped."
    )

    print(f"\n{'[dry-run] ' if dry_run else ''}✅ Bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
