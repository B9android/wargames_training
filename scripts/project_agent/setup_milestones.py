"""
Bootstrap Script — Create M0–M4 GitHub Milestones with due dates.

Usage:
    GITHUB_TOKEN=<token> REPO_NAME=<owner/repo> python setup_milestones.py

Safe to re-run: existing milestones (matched by title) are left unchanged.
Due dates are relative to the START_DATE constant below — adjust as needed.
"""
import os
from datetime import datetime, timedelta, timezone
from github import Github, GithubException

REPO_NAME = os.environ["REPO_NAME"]
gh = Github(os.environ["GITHUB_TOKEN"])
repo = gh.get_repo(REPO_NAME)

# ---------------------------------------------------------------------------
# Milestone schedule — START_DATE is configurable via env var.
# Each offset is in calendar weeks from START_DATE.
# ---------------------------------------------------------------------------
START_DATE_ENV = os.getenv("START_DATE")
if START_DATE_ENV:
    try:
        parsed_start = datetime.fromisoformat(START_DATE_ENV)
        if parsed_start.tzinfo is None:
            parsed_start = parsed_start.replace(tzinfo=timezone.utc)
        START_DATE = parsed_start
    except ValueError as exc:
        raise ValueError(
            "Invalid START_DATE env var format. Use ISO 8601, e.g. "
            "'2026-03-17' or '2026-03-17T00:00:00+00:00'."
        ) from exc
else:
    START_DATE = datetime.now(timezone.utc)

MILESTONES = [
    {
        "title": "M0: Project Bootstrap",
        "description": "Repo structure, CI, tooling, dev environment, labels, and project board.",
        "weeks_offset": 2,
    },
    {
        "title": "M1: 1v1 Competence",
        "description": "Agent reliably beats the scripted opponent in 1v1 battalion combat.",
        "weeks_offset": 8,
    },
    {
        "title": "M2: Self-Play",
        "description": "Stable self-play training loop with improving win rate over time.",
        "weeks_offset": 16,
    },
    {
        "title": "M3: League Training",
        "description": "Multi-agent league with historical opponents and robust evaluation.",
        "weeks_offset": 26,
    },
    {
        "title": "M4: HRL",
        "description": "Hierarchical RL — high-level commander directing battalion-level agents.",
        "weeks_offset": 38,
    },
]

existing = {m.title: m for m in repo.get_milestones(state="open")}
existing.update({m.title: m for m in repo.get_milestones(state="closed")})

created = 0

for ms in MILESTONES:
    due_on = START_DATE + timedelta(weeks=ms["weeks_offset"])
    if ms["title"] in existing:
        print(f"  ✔️  Exists   : {ms['title']} (due {existing[ms['title']].due_on})")
    else:
        try:
            repo.create_milestone(
                title=ms["title"],
                description=ms["description"],
                due_on=due_on,
            )
            print(f"  ✅ Created  : {ms['title']} (due {due_on.strftime('%Y-%m-%d')})")
            created += 1
        except GithubException as exc:
            print(f"  ❌ Failed   : {ms['title']} — {exc}")

print(f"\nDone. {created} milestones created.")
