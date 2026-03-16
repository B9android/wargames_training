"""
Triage Agent — runs on every new issue.
Labels it, assigns it to the correct milestone, posts a structured comment.
"""
import os
import json
from github import Github
from openai import OpenAI

REPO_NAME = os.environ["REPO_NAME"]
ISSUE_NUMBER = int(os.environ["ISSUE_NUMBER"])

gh = Github(os.environ["GITHUB_TOKEN"])
ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
repo = gh.get_repo(REPO_NAME)
issue = repo.get_issue(ISSUE_NUMBER)

# Fetch all available labels and milestones for context
all_labels = [l.name for l in repo.get_labels()]
all_milestones = {m.title: m.number for m in repo.get_milestones(state="open")}

SYSTEM_PROMPT = """
You are a project management agent for a reinforcement learning research project
called wargames_training. The project trains AI agents to control Napoleonic-era
military battalions in a continuous 2D simulation.

Your job is to triage new GitHub issues by:
1. Suggesting the most appropriate labels from the available list
2. Suggesting the most appropriate milestone
3. Writing a brief, helpful triage comment

Respond ONLY with valid JSON in this exact format:
{
  "labels": ["label1", "label2"],
  "milestone": "M1: 1v1 Competence",
  "comment": "Triage comment here",
  "priority": "high"
}
"""

USER_PROMPT = f"""
New issue opened:
Title: {issue.title}
Body: {issue.body or '(no body)'}

Available labels: {json.dumps(all_labels)}
Available milestones: {json.dumps(list(all_milestones.keys()))}

Triage this issue.
"""

response = ai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    response_format={"type": "json_object"},
)

result = json.loads(response.choices[0].message.content)

# Apply labels
existing_label_names = [l.name for l in repo.get_labels()]
labels_to_apply = [l for l in result.get("labels", []) if l in existing_label_names]
if labels_to_apply:
    issue.add_to_labels(*labels_to_apply)

# Apply milestone
milestone_title = result.get("milestone")
if milestone_title and milestone_title in all_milestones:
    milestone = repo.get_milestone(all_milestones[milestone_title])
    issue.edit(milestone=milestone)

# Post triage comment
comment_body = f"""## 🤖 Agent Triage

{result.get('comment', '')}

---
**Labels applied:** {', '.join(f'`{l}`' for l in labels_to_apply) or 'none'}
**Milestone:** `{milestone_title or 'none'}`
**Suggested priority:** `{result.get('priority', 'medium')}`

> *This triage was performed automatically. Please review and adjust as needed.*
"""
issue.create_comment(comment_body)
print(f"✅ Triaged issue #{ISSUE_NUMBER}")
