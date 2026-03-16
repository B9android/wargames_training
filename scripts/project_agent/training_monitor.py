"""
Training Monitor Agent.
Fetches W&B run results and posts them as a comment to the linked GitHub issue.

Triggered via workflow_dispatch with:
  WANDB_RUN_ID  — W&B run path: "entity/project/run_id"
  ISSUE_NUMBER  — GitHub issue number for the linked experiment
"""
import os
from github import Github
import wandb

REPO_NAME = os.environ["REPO_NAME"]
WANDB_RUN_ID = os.environ["WANDB_RUN_ID"]
ISSUE_NUMBER = int(os.environ["ISSUE_NUMBER"])

gh = Github(os.environ["GITHUB_TOKEN"])
repo = gh.get_repo(REPO_NAME)
issue = repo.get_issue(ISSUE_NUMBER)

# Authenticate and fetch the W&B run
api = wandb.Api()
run = api.run(WANDB_RUN_ID)

summary = dict(run.summary)
config = dict(run.config)

# Build a readable metrics table (skip internal W&B keys)
metric_rows = "\n".join(
    f"| `{k}` | `{round(v, 4) if isinstance(v, float) else v}` |"
    for k, v in sorted(summary.items())
    if not k.startswith("_") and isinstance(v, (int, float, str, bool))
)

config_rows = "\n".join(
    f"| `{k}` | `{v}` |"
    for k, v in list(config.items())[:20]
)

runtime_s = summary.get("_runtime")
runtime_str = f"{int(runtime_s // 60)}m {int(runtime_s % 60)}s" if runtime_s else "N/A"

comment_body = f"""## 📊 Training Run Results

**W&B Run:** [{run.name}]({run.url})
**State:** `{run.state}`
**Runtime:** {runtime_str}

### Summary Metrics

| Metric | Value |
|---|---|
{metric_rows or "| — | No scalar metrics recorded |"}

### Config Highlights

| Parameter | Value |
|---|---|
{config_rows or "| — | No config recorded |"}

---
> *Posted automatically by training monitor agent. [View full run on W&B]({run.url})*
"""

issue.create_comment(comment_body)
print(f"✅ Posted W&B results for run '{run.name}' to issue #{ISSUE_NUMBER}")
