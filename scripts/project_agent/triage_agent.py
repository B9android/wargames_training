"""Triage Agent: rule-first triage with AI fallback for ambiguous issues."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from common import agent_signature, log_event, require_env, retry
from projects_v2 import ProjectsV2Client


PRIORITY_HIGH = "priority: high"
PRIORITY_MEDIUM = "priority: medium"

RULE_MAP = {
    "[BUG]": ["type: bug", PRIORITY_HIGH],
    "[EXP]": ["type: experiment", PRIORITY_MEDIUM],
    "[RESEARCH]": ["type: research", "priority: low"],
    "[FEAT]": ["type: feature", PRIORITY_MEDIUM],
    "[FEATURE]": ["type: feature", PRIORITY_MEDIUM],
    "[EPIC]": ["type: epic", PRIORITY_HIGH],
}

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


REPO_NAME = ""
ISSUE_NUMBER = 0
DRY_RUN = False
ai = None
repo = None
issue = None
all_labels: list[str] = []
all_milestones: dict[str, int] = {}


def load_rule_map() -> dict[str, list[str]]:
    """Load title marker rules from the canonical orchestration contract."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        markers = raw.get("markers", {})
        if not isinstance(markers, dict):
            raise ValueError("markers section is not a mapping")

        loaded_rule_map: dict[str, list[str]] = {}
        for marker, marker_config in markers.items():
            labels = marker_config.get("labels", []) if isinstance(marker_config, dict) else []
            loaded_rule_map[str(marker).upper()] = [str(label) for label in labels]

        if loaded_rule_map:
            return loaded_rule_map
    except Exception as exc:  # pragma: no cover - config fallback
        log_event("triage_rule_map_fallback", error=str(exc), config=str(config_path))
    return RULE_MAP


def choose_milestone(title: str) -> str | None:
    if "[EXP]" in title and "M1: 1v1 Competence" in all_milestones:
        return "M1: 1v1 Competence"
    if all_milestones:
        return next(iter(all_milestones.keys()))
    return None


def triage_by_rules() -> dict[str, object]:
    upper_title = issue.title.upper()
    labels: list[str] = []
    for marker, mapped_labels in load_rule_map().items():
        if marker in upper_title:
            labels.extend(mapped_labels)
    labels = sorted(set(labels))

    if not labels:
        return {
            "labels": ["status: needs-manual-triage", PRIORITY_MEDIUM],
            "milestone": choose_milestone(issue.title),
            "comment": "Issue could not be confidently triaged by rules. Marked for manual review.",
            "priority": "medium",
            "rule_only": True,
        }

    return {
        "labels": labels,
        "milestone": choose_milestone(issue.title),
        "comment": "Issue triaged using deterministic title markers.",
        "priority": "high" if PRIORITY_HIGH in labels else "medium",
        "rule_only": True,
    }


def build_triage_prompt() -> str:
    return f"""
New issue opened:
Title: {issue.title}
Body: {issue.body or '(no body)'}

Available labels: {json.dumps(all_labels)}
Available milestones: {json.dumps(list(all_milestones.keys()))}

Triage this issue.
"""


def maybe_ai_triage(result: dict[str, object], prompt: str) -> dict[str, object]:
    if not (ai and "status: needs-manual-triage" in result.get("labels", [])):
        return result
    try:
        response = retry(
            lambda: ai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                timeout=30,
            )
        )
        parsed = json.loads(response.choices[0].message.content)
        parsed["rule_only"] = False
        return parsed
    except Exception as exc:  # pragma: no cover
        log_event("triage_ai_fallback", error=str(exc), issue=ISSUE_NUMBER)
        return result


def apply_labels(result: dict[str, object]) -> list[str]:
    existing_label_names = [label.name for label in repo.get_labels()]
    labels_to_apply = [label for label in result.get("labels", []) if label in existing_label_names]
    if not labels_to_apply:
        return []

    if DRY_RUN:
        print(f"[dry-run] Would apply labels: {labels_to_apply}")
    else:
        retry(lambda labels=tuple(labels_to_apply): issue.add_to_labels(*labels))
    return labels_to_apply


def apply_milestone(result: dict[str, object]) -> str | None:
    milestone_title = result.get("milestone")
    if not (milestone_title and milestone_title in all_milestones):
        return milestone_title

    milestone = repo.get_milestone(all_milestones[milestone_title])
    if DRY_RUN:
        print(f"[dry-run] Would set milestone: {milestone_title}")
    else:
        retry(lambda milestone=milestone: issue.edit(milestone=milestone))
    return milestone_title


def main() -> None:
    global REPO_NAME, ISSUE_NUMBER, DRY_RUN, ai, repo, issue, all_labels, all_milestones

    from github import Auth, Github
    from openai import OpenAI

    env = require_env(["REPO_NAME", "ISSUE_NUMBER", "GITHUB_TOKEN"])
    REPO_NAME = env["REPO_NAME"]
    ISSUE_NUMBER = int(env["ISSUE_NUMBER"])
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

    auth = Auth.Token(env["GITHUB_TOKEN"])
    gh = Github(auth=auth)
    ai = OpenAI(api_key=openai_api_key) if openai_api_key else None
    repo = gh.get_repo(REPO_NAME)
    issue = repo.get_issue(ISSUE_NUMBER)

    all_labels = [label.name for label in repo.get_labels()]
    all_milestones = {
        milestone.title: milestone.number for milestone in repo.get_milestones(state="open")
    }

    prompt = build_triage_prompt()
    result = maybe_ai_triage(triage_by_rules(), prompt)
    labels_to_apply = apply_labels(result)
    milestone_title = apply_milestone(result)
    
    # Initialize projects v2 client for board management
    projects_client = None
    try:
        projects_client = ProjectsV2Client(env["GITHUB_TOKEN"], REPO_NAME)
        
        # Extract version from labels or milestone
        version_label = next((l for l in labels_to_apply if l.startswith("v")), None)
        if not version_label and milestone_title:
            # Try to extract version from milestone (e.g., "M1: 1v1 Competence" -> version extraction logic)
            version_label = None
        
        if version_label:
            # Add issue to project and set Version field
            if not DRY_RUN:
                projects_client.ensure_issue_in_project_with_fields(
                    ISSUE_NUMBER,
                    field_updates={"Version": (version_label, "SINGLE_SELECT")},
                )
                log_event("triaged_issue_version_set", issue=ISSUE_NUMBER, version=version_label)
    except Exception as exc:
        log_event("triage_projects_sync_failed", issue=ISSUE_NUMBER, error=str(exc))

    mode = "rules" if result.get("rule_only", False) else "ai"
    comment_body = f"""## Agent Triage

{result.get('comment', '')}

---
**Labels applied:** {', '.join(f'`{label}`' for label in labels_to_apply) or 'none'}
**Milestone:** `{milestone_title or 'none'}`
**Suggested priority:** `{result.get('priority', 'medium')}`
**Triage mode:** `{mode}`

{agent_signature("triage", context="issue triage")}

Please review and adjust as needed.
"""
    if DRY_RUN:
        print(f"[dry-run] Would post triage comment to issue #{ISSUE_NUMBER}")
    else:
        retry(lambda body=comment_body: issue.create_comment(body))

    log_event("triage_complete", issue=ISSUE_NUMBER, mode=mode, labels=labels_to_apply, dry_run=DRY_RUN)
    print(f"{'[dry-run] ' if DRY_RUN else ''}✅ Triaged issue #{ISSUE_NUMBER}")


if __name__ == "__main__":
    main()
