"""Triage Agent — rule-first triage with optional AI fallback for ambiguous issues."""
from __future__ import annotations

import json
import importlib
import os
from pathlib import Path

yaml = importlib.import_module("yaml")

from agent_platform.context import AgentContext
from agent_platform.github_gateway import GitHubGateway
from agent_platform.graphql_gateway import GraphQLGateway
from agent_platform.result import ActionResult, ActionStatus, RunResult
from agent_platform.runner import run_agent
from agent_platform.summary import ExecutionSummary
from agent_platform.telemetry import log_event

PRIORITY_HIGH = "priority: high"
PRIORITY_MEDIUM = "priority: medium"
TYPE_EXPERIMENT = "type: experiment"
TYPE_EPIC = "type: epic"

RULE_MAP = {
    "[BUG]": ["type: bug", PRIORITY_HIGH],
    "[EXP]": [TYPE_EXPERIMENT, PRIORITY_MEDIUM],
    "[RESEARCH]": ["type: research", "priority: low"],
    "[FEAT]": ["type: feature", PRIORITY_MEDIUM],
    "[FEATURE]": ["type: feature", PRIORITY_MEDIUM],
    "[EPIC]": [TYPE_EPIC, PRIORITY_HIGH],
}

SYSTEM_PROMPT = """
You are a project management agent for a reinforcement learning research project
called wargames_training.

Respond ONLY with valid JSON in this format:
{
  "labels": ["label1", "label2"],
  "milestone": "M1: 1v1 Competence",
  "comment": "Triage comment here",
  "priority": "high"
}
"""


# Legacy test compatibility: tests mutate this directly.
all_milestones: dict[str, int] = {}


def _load_rule_map() -> dict[str, list[str]]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "orchestration.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        markers = raw.get("markers", {})
        if not isinstance(markers, dict):
            raise ValueError("markers section is not a mapping")
        loaded: dict[str, list[str]] = {}
        for marker, cfg in markers.items():
            labels = cfg.get("labels", []) if isinstance(cfg, dict) else []
            loaded[str(marker).upper()] = [str(label) for label in labels]
        if loaded:
            return loaded
    except Exception as exc:
        log_event("triage_rule_map_fallback", error=str(exc), config=str(config_path))
    return RULE_MAP


def _choose_milestone(title: str, milestone_titles: list[str]) -> str | None:
    if "[EXP]" in title and "M1: 1v1 Competence" in milestone_titles:
        return "M1: 1v1 Competence"
    return milestone_titles[0] if milestone_titles else None


def _triage_by_rules(title: str, milestone_titles: list[str]) -> dict[str, object]:
    upper = title.upper()
    labels: list[str] = []
    for marker, mapped in _load_rule_map().items():
        if marker in upper:
            labels.extend(mapped)
    labels = sorted(set(labels))

    if not labels:
        return {
            "labels": ["status: needs-manual-triage", PRIORITY_MEDIUM],
            "milestone": _choose_milestone(title, milestone_titles),
            "comment": "Issue could not be confidently triaged by rules. Marked for manual review.",
            "priority": "medium",
            "rule_only": True,
        }

    return {
        "labels": labels,
        "milestone": _choose_milestone(title, milestone_titles),
        "comment": "Issue triaged using deterministic title markers.",
        "priority": "high" if PRIORITY_HIGH in labels else "medium",
        "rule_only": True,
    }


def _maybe_ai_triage(
    result: dict[str, object],
    *,
    title: str,
    body: str,
    all_labels: list[str],
    all_milestones: list[str],
) -> dict[str, object]:
    if "status: needs-manual-triage" not in result.get("labels", []):
        return result

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return result

    try:
        openai_cls = importlib.import_module("openai").OpenAI
        ai = openai_cls(api_key=api_key)
        prompt = (
            f"Title: {title}\n"
            f"Body: {body or '(no body)'}\n"
            f"Available labels: {json.dumps(all_labels)}\n"
            f"Available milestones: {json.dumps(all_milestones)}\n"
            "Triage this issue."
        )
        response = ai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )
        parsed = json.loads(response.choices[0].message.content)
        parsed["rule_only"] = False
        return parsed
    except Exception as exc:
        log_event("triage_ai_fallback", error=str(exc))
        return result


def _emit_step_outputs(classified_labels: list[str], issue_number: int) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "")
    if not output_path:
        return
    labels = [str(lbl) for lbl in classified_labels] if isinstance(classified_labels, list) else []
    is_epic = TYPE_EPIC in labels
    is_experiment = TYPE_EXPERIMENT in labels
    with open(output_path, "a", encoding="utf-8") as out:
        out.write(f"is_epic={'true' if is_epic else 'false'}\n")
        out.write(f"is_experiment={'true' if is_experiment else 'false'}\n")
        out.write(f"issue_number={issue_number}\n")


# Backward-compatible helper names used by existing tests.
def choose_milestone(title: str) -> str | None:
    return _choose_milestone(title, list(all_milestones.keys()))


def emit_step_outputs(classified_labels: list[str], issue_number: int, *, output_path: str | None = None) -> None:
    if output_path:
        labels = [str(lbl) for lbl in classified_labels] if isinstance(classified_labels, list) else []
        is_epic = TYPE_EPIC in labels
        is_experiment = TYPE_EXPERIMENT in labels
        with open(output_path, "a", encoding="utf-8") as out:
            out.write(f"is_epic={'true' if is_epic else 'false'}\n")
            out.write(f"is_experiment={'true' if is_experiment else 'false'}\n")
            out.write(f"issue_number={issue_number}\n")
        return
    _emit_step_outputs(classified_labels, issue_number)


def _apply_labels(
    ctx: AgentContext,
    rest: GitHubGateway,
    issue_number: int,
    labels: list[str],
    results: list[ActionResult],
) -> None:
    if not labels:
        return
    if ctx.dry_run:
        results.append(
            ActionResult("apply_labels", f"issue #{issue_number}", ActionStatus.DRY_RUN, metadata={"labels": labels})
        )
        return
    rest.add_labels(issue_number, labels)
    results.append(
        ActionResult("apply_labels", f"issue #{issue_number}", ActionStatus.SUCCESS, metadata={"labels": labels})
    )


def _apply_milestone(
    ctx: AgentContext,
    rest: GitHubGateway,
    issue_number: int,
    milestone_title: str | None,
    all_milestones: list[str],
    results: list[ActionResult],
) -> None:
    if not (milestone_title and milestone_title in all_milestones):
        return
    if ctx.dry_run:
        results.append(
            ActionResult("set_milestone", f"issue #{issue_number}", ActionStatus.DRY_RUN, metadata={"milestone": milestone_title})
        )
        return
    milestone_obj = next(m for m in rest._repo_obj.get_milestones(state="open") if m.title == milestone_title)
    rest._repo_obj.get_issue(issue_number).edit(milestone=milestone_obj)
    results.append(
        ActionResult("set_milestone", f"issue #{issue_number}", ActionStatus.SUCCESS, metadata={"milestone": milestone_title})
    )


def _sync_version_field(
    ctx: AgentContext,
    gql: GraphQLGateway,
    issue_number: int,
    labels: list[str],
    results: list[ActionResult],
) -> None:
    version_label = next((l for l in labels if l.startswith("v")), None)
    if not version_label:
        return
    if ctx.dry_run:
        results.append(
            ActionResult("sync_version_field", f"issue #{issue_number}", ActionStatus.DRY_RUN, metadata={"version": version_label})
        )
        return
    gql.ensure_issue_in_project_with_fields(
        issue_number,
        field_updates={"Version": (version_label, "SINGLE_SELECT")},
    )
    results.append(
        ActionResult("sync_version_field", f"issue #{issue_number}", ActionStatus.SUCCESS, metadata={"version": version_label})
    )


def _post_triage_comment(
    ctx: AgentContext,
    rest: GitHubGateway,
    issue_number: int,
    labels: list[str],
    milestone_title: str | None,
    triage_result: dict[str, object],
    results: list[ActionResult],
) -> str:
    mode = "rules" if triage_result.get("rule_only", False) else "ai"
    comment_body = (
        "## Agent Triage\n\n"
        f"{triage_result.get('comment', '')}\n\n"
        "---\n"
        f"**Labels applied:** {', '.join(f'`{l}`' for l in labels) or 'none'}\n"
        f"**Milestone:** `{milestone_title or 'none'}`\n"
        f"**Suggested priority:** `{triage_result.get('priority', 'medium')}`\n"
        f"**Triage mode:** `{mode}`\n\n"
        "<!-- agent:triage -->\n\n"
        "Please review and adjust as needed."
    )
    if ctx.dry_run:
        results.append(ActionResult("post_triage_comment", f"issue #{issue_number}", ActionStatus.DRY_RUN))
    else:
        rest.add_issue_comment(issue_number, comment_body)
        results.append(ActionResult("post_triage_comment", f"issue #{issue_number}", ActionStatus.SUCCESS))
    return mode


def _run(
    ctx: AgentContext,
    gql: GraphQLGateway,
    rest: GitHubGateway,
    summary: ExecutionSummary,
) -> RunResult:
    issue_number = ctx.require_issue()
    results: list[ActionResult] = []

    issue = rest.get_issue(issue_number)
    all_labels = [lbl.name for lbl in rest._repo_obj.get_labels()]
    all_milestones = [m.title for m in rest._repo_obj.get_milestones(state="open")]

    summary.checkpoint("classify", f"Classifying issue #{issue_number}")
    triage_result = _triage_by_rules(issue.title, all_milestones)
    triage_result = _maybe_ai_triage(
        triage_result,
        title=issue.title,
        body=issue.body,
        all_labels=all_labels,
        all_milestones=all_milestones,
    )

    labels = [label for label in triage_result.get("labels", []) if label in all_labels]
    milestone_title = triage_result.get("milestone")

    _apply_labels(ctx, rest, issue_number, labels, results)
    _apply_milestone(ctx, rest, issue_number, milestone_title, all_milestones, results)
    _sync_version_field(ctx, gql, issue_number, labels, results)
    mode = _post_triage_comment(ctx, rest, issue_number, labels, milestone_title, triage_result, results)

    _emit_step_outputs(triage_result.get("labels", []), issue_number)
    summary.decision(f"Triage mode={mode}; labels={labels}; milestone={milestone_title}")

    return RunResult(agent="triage_agent", dry_run=ctx.dry_run, actions=results)


if __name__ == "__main__":
    run_agent("triage_agent", _run)

