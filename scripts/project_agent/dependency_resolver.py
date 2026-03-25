# SPDX-License-Identifier: MIT
"""Helpers for extracting and validating issue references in automation payloads."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


ISSUE_REFERENCE_PATTERN = re.compile(r"#(\d+)")
FENCED_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def strip_fenced_code_blocks(text: str | None) -> str:
    """Remove fenced code blocks before dependency parsing."""
    return FENCED_CODE_BLOCK_PATTERN.sub("", text or "")


def extract_issue_references(text: str | None) -> list[int]:
    """Return sorted unique issue references found in free-form text."""
    refs = ISSUE_REFERENCE_PATTERN.findall(strip_fenced_code_blocks(text))
    return sorted({int(ref) for ref in refs})


def has_issue_reference(text: str | None) -> bool:
    """Check whether free-form text contains at least one issue reference."""
    return bool(extract_issue_references(text))


def issue_label_names(issue: Any) -> set[str]:
    """Return label names for an issue-like object."""
    labels = getattr(issue, "labels", []) or []
    return {getattr(label, "name", "") for label in labels if getattr(label, "name", "")}


def build_parent_child_map(issues: list[Any]) -> dict[int, set[int]]:
    """Build a parent -> child issue mapping using #refs in child issue bodies."""
    known_issue_numbers = {getattr(issue, "number", -1) for issue in issues}
    parents_to_children: dict[int, set[int]] = defaultdict(set)

    for issue in issues:
        child_number = getattr(issue, "number", None)
        if child_number is None:
            continue
        parent_refs = extract_issue_references(getattr(issue, "body", ""))
        for parent_number in parent_refs:
            if parent_number in known_issue_numbers and parent_number != child_number:
                parents_to_children[parent_number].add(child_number)

    return dict(parents_to_children)


def blocked_parent_rollup(issues: list[Any]) -> list[dict[str, object]]:
    """Return parent issues blocked by blocked children discovered from #refs."""
    by_number = {getattr(issue, "number", -1): issue for issue in issues}
    parent_child = build_parent_child_map(issues)
    rollup: list[dict[str, object]] = []

    for parent_number, child_numbers in parent_child.items():
        blocked_children = []
        for child_number in sorted(child_numbers):
            child_issue = by_number.get(child_number)
            if child_issue is None:
                continue
            if "status: blocked" in issue_label_names(child_issue):
                blocked_children.append(
                    {
                        "number": child_number,
                        "title": getattr(child_issue, "title", ""),
                    }
                )

        if not blocked_children:
            continue

        parent_issue = by_number.get(parent_number)
        rollup.append(
            {
                "parent_number": parent_number,
                "parent_title": getattr(parent_issue, "title", "(not loaded)"),
                "blocked_children": blocked_children,
            }
        )

    return sorted(rollup, key=lambda item: len(item["blocked_children"]), reverse=True)


def dependency_blocker_context(rollup: list[dict[str, object]], *, limit: int = 8) -> str:
    """Format blocker rollup entries for prompt/context payloads."""
    if not rollup:
        return "None"

    lines = []
    for entry in rollup[:limit]:
        blocked_children = ", ".join(
            f"#{child['number']} {child['title']}" for child in entry["blocked_children"]
        )
        lines.append(f"#{entry['parent_number']} {entry['parent_title']} blocked by {blocked_children}")
    return "\n".join(lines)