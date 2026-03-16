"""
Weekly Progress Report Agent.
Reads all open/closed issues from the past week, generates a summary,
and posts it as a new issue with label `type: docs`.
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
one_week_ago = now - timedelta(days=7)

# Gather data
closed_this_week = [
    i for i in repo.get_issues(state="closed")
    if i.closed_at and i.closed_at > one_week_ago
]
opened_this_week = [
    i for i in repo.get_issues(state="open")
    if i.created_at > one_week_ago
]
all_open = list(repo.get_issues(state="open"))
blocked = [i for i in all_open if any(l.name == "status: blocked" for l in i.labels)]
in_progress = [i for i in all_open if any(l.name == "status: in-progress" for l in i.labels)]

# Milestone progress
milestone_data = []
for m in repo.get_milestones(state="open"):
    pct = 0
    if m.open_issues + m.closed_issues > 0:
        pct = round(m.closed_issues / (m.open_issues + m.closed_issues) * 100)
    milestone_data.append({
        "title": m.title,
        "open": m.open_issues,
        "closed": m.closed_issues,
        "pct": pct,
        "due": str(m.due_on.date()) if m.due_on else "no due date",
    })

context = {
    "week_ending": now.strftime("%Y-%m-%d"),
    "closed_this_week": [{"number": i.number, "title": i.title} for i in closed_this_week],
    "opened_this_week": [{"number": i.number, "title": i.title} for i in opened_this_week],
    "currently_blocked": [{"number": i.number, "title": i.title} for i in blocked],
    "in_progress": [{"number": i.number, "title": i.title} for i in in_progress],
    "milestones": milestone_data,
    "total_open_issues": len(all_open),
}

SYSTEM_PROMPT = """
You are a project management agent for wargames_training, a reinforcement learning
research project. Write a professional, concise weekly progress report in GitHub
Markdown. Include:
- Executive summary (2-3 sentences)
- What was completed this week
- What is in progress
- Blockers
- Milestone progress bars
- Recommended focus for next week

Be direct and specific. Use emojis sparingly. Format for GitHub markdown.
"""

response = ai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Generate the weekly report from this data:\n{json.dumps(context, indent=2)}"},
    ],
)

report_body = response.choices[0].message.content

title = f"📊 Weekly Progress Report — Week of {now.strftime('%Y-%m-%d')}"
labels_to_apply = []
for label in repo.get_labels():
    if label.name in ["type: docs", "status: agent-created"]:
        labels_to_apply.append(label.name)

new_issue = repo.create_issue(
    title=title,
    body=report_body,
    labels=labels_to_apply,
)
print(f"✅ Weekly report created: #{new_issue.number}")
