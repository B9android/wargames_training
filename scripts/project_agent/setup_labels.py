"""
Bootstrap Script — Create all required GitHub labels.

Usage:
    GITHUB_TOKEN=<token> REPO_NAME=<owner/repo> python setup_labels.py

Safe to re-run: existing labels are updated in-place; missing ones are created.
"""
import os
from github import Github, GithubException

REPO_NAME = os.environ["REPO_NAME"]
gh = Github(os.environ["GITHUB_TOKEN"])
repo = gh.get_repo(REPO_NAME)

# ---------------------------------------------------------------------------
# Label definitions  (name, hex color without #, description)
# ---------------------------------------------------------------------------
LABELS = [
    # type:
    ("type: bug",            "d73a4a", "Something isn't working"),
    ("type: feature",        "0075ca", "New feature or request"),
    ("type: experiment",     "7057ff", "Training experiment"),
    ("type: epic",           "003d82", "Large, multi-issue body of work"),
    ("type: research",       "0e8a8a", "Research / investigation task"),
    ("type: infrastructure", "586069", "CI, tooling, repo maintenance"),
    ("type: documentation",  "cfd3d7", "Documentation improvement"),
    ("type: chore",          "e4e669", "Routine maintenance task"),
    # priority:
    ("priority: critical",   "b60205", "Drop everything — production blocker"),
    ("priority: high",       "e4560a", "Important, do soon"),
    ("priority: medium",     "fbca04", "Normal priority"),
    ("priority: low",        "0e8a16", "Nice to have"),
    # status:
    ("status: in-progress",  "fef2c0", "Actively being worked on"),
    ("status: blocked",      "ee0701", "Cannot proceed — waiting on something"),
    ("status: needs-review", "0075ca", "Needs a human review"),
    ("status: ready",        "0e8a16", "Ready to be picked up"),
    ("status: on-hold",      "bfd4f2", "Intentionally paused"),
    ("status: stale",        "ededed", "No activity for 14+ days"),
    ("status: agent-created","c2e0c6", "Created automatically by a GitHub agent"),
]

existing = {label.name: label for label in repo.get_labels()}
created = 0
updated = 0

for name, color, description in LABELS:
    if name in existing:
        label = existing[name]
        if label.color != color or label.description != description:
            label.edit(name=name, color=color, description=description)
            print(f"  ✏️  Updated  : {name}")
            updated += 1
        else:
            print(f"  ✔️  Exists   : {name}")
    else:
        try:
            repo.create_label(name=name, color=color, description=description)
            print(f"  ✅ Created  : {name}")
            created += 1
        except GithubException as exc:
            print(f"  ❌ Failed   : {name} — {exc}")

print(f"\nDone. {created} created, {updated} updated.")
