"""
Milestone Health Check Agent.
Runs daily. Detects:
- Milestones at risk (due soon, low completion)
- Stale issues (no activity in 14+ days)
- Missing labels on open issues
Creates blocker/warning issues automatically.
"""
import os
import json
from datetime import datetime, timedelta, timezone
from github import Github
from openai import OpenAI

REPO_NAME = os.environ["REPO_NAME"]
gh = Github(os.environ["GITHUB_TOKEN"])
ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
repo = gh.get_repo(REPO_NAME)
now = datetime.now(timezone.utc)

issues_created = []

# --- Check 1: Milestones at risk ---
for m in repo.get_milestones(state="open"):
    if not m.due_on:
        continue
    due = m.due_on.replace(tzinfo=timezone.utc)
    days_left = (due - now).days
    total = m.open_issues + m.closed_issues
    pct = (m.closed_issues / total * 100) if total > 0 else 0

    at_risk = (days_left <= 14 and pct < 50) or (days_left <= 7 and pct < 80)
    if at_risk:
        title = f"⚠️ MILESTONE AT RISK: {m.title} ({pct:.0f}% complete, {days_left}d left)"
        # Don't duplicate — check if this issue already exists
        existing = list(repo.get_issues(state="open", labels=["status: blocked"]))
        already_exists = any(title[:50] in i.title for i in existing)
        if not already_exists:
            body = f"""## ⚠️ Milestone At Risk

**Milestone:** [{m.title}]({m.url})
**Due:** {due.strftime('%Y-%m-%d')} ({days_left} days remaining)
**Progress:** {m.closed_issues}/{total} issues closed ({pct:.0f}%)

### Open Issues Remaining
{chr(10).join(f'- #{i.number} {i.title}' for i in repo.get_issues(state="open", milestone=m))}

### Recommended Actions
- Review and close or defer non-critical issues
- Ensure all in-progress work is tracked with `status: in-progress`
- Consider adjusting milestone due date if scope has changed

> *Created automatically by milestone health check agent.*
"""
            new_issue = repo.create_issue(
                title=title,
                body=body,
                labels=["priority: critical", "type: chore", "status: agent-created"],
            )
            issues_created.append(new_issue.number)

# --- Check 2: Mark stale issues ---
stale_cutoff = now - timedelta(days=14)
for issue in repo.get_issues(state="open"):
    label_names = [l.name for l in issue.labels]
    if "status: stale" in label_names:
        continue
    if "status: in-progress" in label_names:
        continue
    if issue.updated_at < stale_cutoff:
        issue.add_to_labels("status: stale")
        issue.create_comment(
            "🤖 **Agent:** This issue has had no activity in 14+ days and has been marked `status: stale`. "
            "Please update, close, or defer it. Remove this label if work is resuming."
        )

# --- Check 3: Issues missing labels ---
unlabeled = [i for i in repo.get_issues(state="open") if not i.labels]
if len(unlabeled) > 3:
    body = f"""## 🏷️ Unlabeled Issues Detected

The following {len(unlabeled)} issues have no labels and may not be properly triaged:

{chr(10).join(f'- #{i.number} {i.title}' for i in unlabeled[:20])}

Please label these issues so they appear in the correct project views.

> *Created automatically by milestone health check agent.*
"""
    new_issue = repo.create_issue(
        title=f"🏷️ {len(unlabeled)} issues need labels",
        body=body,
        labels=["type: chore", "priority: low", "status: agent-created"],
    )
    issues_created.append(new_issue.number)

print(f"✅ Milestone check complete. Created {len(issues_created)} issues: {issues_created}")
