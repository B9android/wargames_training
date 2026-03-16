"""
Issue Writer Agent.
Given a context string (e.g. training results, a research finding, a blocker),
generates and creates 1-5 well-structured follow-up GitHub issues.
"""
import os
import json
from github import Github
from openai import OpenAI

REPO_NAME = os.environ["REPO_NAME"]
CONTEXT = os.environ.get("AGENT_CONTEXT", "")
TARGET_VERSION = os.environ.get("TARGET_VERSION", "v1")

gh = Github(os.environ["GITHUB_TOKEN"])
ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
repo = gh.get_repo(REPO_NAME)

all_labels = [l.name for l in repo.get_labels()]
all_milestones = {m.title: m.number for m in repo.get_milestones(state="open")}
open_issues = [{"number": i.number, "title": i.title} for i in repo.get_issues(state="open")]

SYSTEM_PROMPT = """
You are a project management agent for wargames_training, a reinforcement learning
research project training AI agents to control Napoleonic military battalions.

Given a context (training results, findings, blockers, etc.), generate 1-5
GitHub issues that represent the logical next steps. Each issue should be
actionable, specific, and well-scoped for a single developer.

Respond ONLY with valid JSON:
{
  "issues": [
    {
      "title": "Issue title",
      "body": "Full GitHub markdown body with ## sections",
      "labels": ["label1", "label2"],
      "milestone": "M1: 1v1 Competence",
      "priority": "high"
    }
  ]
}

Issue body should include:
## Context
## Task
## Acceptance Criteria
- [ ] item
## Notes
"""

USER_PROMPT = f"""
Context provided:
{CONTEXT}

Target version: {TARGET_VERSION}
Available labels: {json.dumps(all_labels)}
Available milestones: {json.dumps(list(all_milestones.keys()))}
Existing open issues (avoid duplicates): {json.dumps(open_issues[:30])}

Generate follow-up issues.
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
created = []

for issue_data in result.get("issues", []):
    labels_to_apply = [l for l in issue_data.get("labels", []) if l in all_labels]
    labels_to_apply.append("status: agent-created")

    milestone_obj = None
    milestone_title = issue_data.get("milestone")
    if milestone_title and milestone_title in all_milestones:
        milestone_obj = repo.get_milestone(all_milestones[milestone_title])

    body = issue_data["body"] + "\n\n---\n> *Created automatically by issue writer agent.*"

    new_issue = repo.create_issue(
        title=issue_data["title"],
        body=body,
        labels=labels_to_apply,
        milestone=milestone_obj,
    )
    created.append(new_issue.number)
    print(f"✅ Created issue #{new_issue.number}: {issue_data['title']}")

print(f"\n✅ Total issues created: {len(created)}")
