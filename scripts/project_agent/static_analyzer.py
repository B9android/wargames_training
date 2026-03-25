# SPDX-License-Identifier: MIT
"""Deterministic static analyzer for cross-file integration TODO generation.

This module intentionally avoids AI heuristics. It performs AST-based checks,
produces a JSON report, and can insert deterministic TODO comments for
high-confidence findings.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SCAN_DIRS = ("envs", "models", "training", "scripts", "tests")
ENTRYPOINT_MODULES = {
    "training.train",
    "training.evaluate",
    "scripts.scenario_runner",
}

HIGH_CONFIDENCE_RULES = {
    "unresolved_local_import",
    "unresolved_from_symbol",
    "dead_private_function",
}


@dataclass(frozen=True)
class ImportBinding:
    line: int
    local_name: str
    target_module: str | None
    imported_name: str | None


@dataclass(frozen=True)
class FunctionDefInfo:
    name: str
    line: int
    is_private: bool


@dataclass
class ModuleInfo:
    path: Path
    module_name: str
    imports: list[ImportBinding]
    functions: list[FunctionDefInfo]
    exported_symbols: set[str]
    local_calls: set[str]
    imported_symbol_calls: list[tuple[str, str]]
    module_attr_calls: list[tuple[str, str]]
    has_main_guard: bool


def path_to_module(repo_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(repo_root)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_python_files(repo_root: Path, scan_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for rel_dir in scan_dirs:
        abs_dir = repo_root / rel_dir
        if not abs_dir.exists() or not abs_dir.is_dir():
            continue
        files.extend(sorted(abs_dir.rglob("*.py")))
    return files


def _iter_top_level_assigned_names(node: ast.stmt) -> set[str]:
    names: set[str] = set()
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign) and node.target is not None:
        targets = [node.target]

    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Tuple):
            names.update(
                elt.id for elt in target.elts if isinstance(elt, ast.Name)
            )
    return names


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    comparators = node.test.comparators
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and bool(comparators)
        and isinstance(comparators[0], ast.Constant)
        and comparators[0].value == "__main__"
    )


def _collect_import_node(
    node: ast.Import,
    imports: list[ImportBinding],
    exported_symbols: set[str],
) -> None:
    for alias in node.names:
        module_target = alias.name
        local_name = alias.asname or alias.name.split(".")[0]
        imports.append(
            ImportBinding(
                line=node.lineno,
                local_name=local_name,
                target_module=module_target,
                imported_name=None,
            )
        )
        exported_symbols.add(local_name)


def _collect_from_import_node(
    node: ast.ImportFrom,
    imports: list[ImportBinding],
    exported_symbols: set[str],
) -> None:
    if node.module is None:
        return
    if node.level and node.level > 0:
        # Keep v1 deterministic and conservative: skip relative imports.
        return
    for alias in node.names:
        local_name = alias.asname or alias.name
        imports.append(
            ImportBinding(
                line=node.lineno,
                local_name=local_name,
                target_module=node.module,
                imported_name=alias.name,
            )
        )
        exported_symbols.add(local_name)


def _collect_top_level_node(
    node: ast.stmt,
    *,
    imports: list[ImportBinding],
    functions: list[FunctionDefInfo],
    exported_symbols: set[str],
) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        functions.append(
            FunctionDefInfo(
                name=node.name,
                line=node.lineno,
                is_private=node.name.startswith("_") and not node.name.startswith("__"),
            )
        )
        exported_symbols.add(node.name)
        return _is_main_guard(node)

    if isinstance(node, ast.ClassDef):
        exported_symbols.add(node.name)
        return _is_main_guard(node)

    if isinstance(node, ast.Import):
        _collect_import_node(node, imports, exported_symbols)
        return _is_main_guard(node)

    if isinstance(node, ast.ImportFrom):
        _collect_from_import_node(node, imports, exported_symbols)
        return _is_main_guard(node)

    exported_symbols.update(_iter_top_level_assigned_names(node))
    return _is_main_guard(node)


def parse_module(repo_root: Path, file_path: Path) -> ModuleInfo:
    source = file_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(file_path))

    imports: list[ImportBinding] = []
    functions: list[FunctionDefInfo] = []
    exported_symbols: set[str] = set()
    local_calls: set[str] = set()
    imported_symbol_calls: list[tuple[str, str]] = []
    module_attr_calls: list[tuple[str, str]] = []
    has_main_guard = False

    for node in tree.body:
        has_main_guard = has_main_guard or _collect_top_level_node(
            node,
            imports=imports,
            functions=functions,
            exported_symbols=exported_symbols,
        )

    # Walk calls through full tree to detect usage patterns.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            local_calls.add(node.func.id)
            imported_symbol_calls.append((node.func.id, node.func.id))
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module_attr_calls.append((node.func.value.id, node.func.attr))

    return ModuleInfo(
        path=file_path,
        module_name=path_to_module(repo_root, file_path),
        imports=imports,
        functions=functions,
        exported_symbols=exported_symbols,
        local_calls=local_calls,
        imported_symbol_calls=imported_symbol_calls,
        module_attr_calls=module_attr_calls,
        has_main_guard=has_main_guard,
    )


def build_module_index(repo_root: Path, scan_dirs: list[str]) -> dict[str, ModuleInfo]:
    index: dict[str, ModuleInfo] = {}
    for path in discover_python_files(repo_root, scan_dirs):
        info = parse_module(repo_root, path)
        index[info.module_name] = info
    return index


def build_module_index_with_parse_findings(
    repo_root: Path,
    scan_dirs: list[str],
) -> tuple[dict[str, ModuleInfo], list[dict[str, Any]]]:
    index: dict[str, ModuleInfo] = {}
    parse_findings: list[dict[str, Any]] = []

    for path in discover_python_files(repo_root, scan_dirs):
        try:
            info = parse_module(repo_root, path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_findings.append(
                {
                    "rule": "parse_error",
                    "confidence": "high",
                    "file": str(path),
                    "line": getattr(exc, "lineno", 1) or 1,
                    "module": path_to_module(repo_root, path),
                    "reason": f"File could not be parsed deterministically: {exc}",
                }
            )
            continue

        index[info.module_name] = info

    return index, parse_findings


def build_basename_index(module_index: dict[str, ModuleInfo]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for module_name in module_index:
        grouped[module_name.split(".")[-1]].append(module_name)

    resolved: dict[str, str] = {}
    for base_name, names in grouped.items():
        if len(names) == 1:
            resolved[base_name] = names[0]
    return resolved


def build_suffix_index(module_index: dict[str, ModuleInfo]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for module_name in module_index:
        parts = module_name.split(".")
        for idx in range(1, len(parts)):
            suffix = ".".join(parts[idx:])
            grouped[suffix].append(module_name)

    resolved: dict[str, str] = {}
    for suffix, names in grouped.items():
        if len(names) == 1:
            resolved[suffix] = names[0]
    return resolved


def _local_root_tokens(module_index: dict[str, ModuleInfo]) -> set[str]:
    roots: set[str] = set()
    for module_name in module_index:
        parts = module_name.split(".")
        if parts:
            roots.add(parts[0])
        if len(parts) > 1:
            roots.add(parts[1])
    return roots


def is_likely_local_module_ref(
    module_ref: str,
    module_index: dict[str, ModuleInfo],
    basename_index: dict[str, str],
    suffix_index: dict[str, str],
) -> bool:
    if module_ref in module_index:
        return True
    if module_ref in suffix_index:
        return True
    first = module_ref.split(".")[0]
    if first in basename_index:
        return True
    return first in _local_root_tokens(module_index)


def resolve_local_module(
    module_ref: str,
    module_index: dict[str, ModuleInfo],
    basename_index: dict[str, str],
    suffix_index: dict[str, str],
) -> str | None:
    if module_ref in module_index:
        return module_ref
    if module_ref in suffix_index:
        return suffix_index[module_ref]
    if module_ref in basename_index:
        return basename_index[module_ref]
    return None


def _is_test_module(module_name: str) -> bool:
    return module_name.startswith("tests.")


def _is_entry_module(module_info: ModuleInfo) -> bool:
    return module_info.has_main_guard or module_info.module_name in ENTRYPOINT_MODULES


def _make_finding(
    *,
    rule: str,
    confidence: str,
    file: Path,
    line: int,
    module: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "rule": rule,
        "confidence": confidence,
        "file": str(file),
        "line": line,
        "module": module,
        "reason": reason,
    }


def _analyze_import_bindings(
    source_name: str,
    info: ModuleInfo,
    module_index: dict[str, ModuleInfo],
    basename_index: dict[str, str],
    suffix_index: dict[str, str],
    *,
    findings: list[dict[str, Any]],
    inbound_module_refs: dict[str, int],
    import_graph: dict[str, set[str]],
) -> None:
    for binding in info.imports:
        if not binding.target_module:
            continue

        target_module = resolve_local_module(
            binding.target_module,
            module_index,
            basename_index,
            suffix_index,
        )

        if binding.imported_name is None:
            if target_module is None:
                if not is_likely_local_module_ref(
                    binding.target_module,
                    module_index,
                    basename_index,
                    suffix_index,
                ):
                    continue
                findings.append(
                    _make_finding(
                        rule="unresolved_local_import",
                        confidence="high",
                        file=info.path,
                        line=binding.line,
                        module=source_name,
                        reason=f"Local import '{binding.target_module}' cannot be resolved in repository",
                    )
                )
                continue

            import_graph[source_name].add(target_module)
            inbound_module_refs[target_module] += 1
            continue

        if target_module is None:
            if not is_likely_local_module_ref(
                binding.target_module,
                module_index,
                basename_index,
                suffix_index,
            ):
                continue
            findings.append(
                _make_finding(
                    rule="unresolved_local_import",
                    confidence="high",
                    file=info.path,
                    line=binding.line,
                    module=source_name,
                    reason=f"From-import module '{binding.target_module}' cannot be resolved in repository",
                )
            )
            continue

        import_graph[source_name].add(target_module)
        inbound_module_refs[target_module] += 1

        if binding.imported_name == "*":
            continue

        target_info = module_index[target_module]
        submodule_candidate = f"{target_module}.{binding.imported_name}"
        symbol_found = (
            binding.imported_name in target_info.exported_symbols
            or submodule_candidate in module_index
        )
        if not symbol_found:
            findings.append(
                _make_finding(
                    rule="unresolved_from_symbol",
                    confidence="high",
                    file=info.path,
                    line=binding.line,
                    module=source_name,
                    reason=(
                        f"Symbol '{binding.imported_name}' is not exported by local module "
                        f"'{target_module}'"
                    ),
                )
            )


def _resolve_function_inbound_refs(
    source_name: str,
    info: ModuleInfo,
    module_index: dict[str, ModuleInfo],
    basename_index: dict[str, str],
    suffix_index: dict[str, str],
    function_inbound: dict[tuple[str, str], int],
) -> None:
    import_map = {binding.local_name: binding for binding in info.imports}

    for called_name in info.local_calls:
        if any(fn.name == called_name for fn in info.functions):
            function_inbound[(source_name, called_name)] += 1
            continue

        binding = import_map.get(called_name)
        if (
            binding
            and binding.target_module
            and binding.imported_name
            and binding.imported_name != "*"
        ):
            target_module = resolve_local_module(
                binding.target_module,
                module_index,
                basename_index,
                suffix_index,
            )
            if target_module:
                function_inbound[(target_module, binding.imported_name)] += 1

    for module_alias, attr_name in info.module_attr_calls:
        binding = import_map.get(module_alias)
        if not binding or not binding.target_module or binding.imported_name is not None:
            continue

        target_module = resolve_local_module(
            binding.target_module,
            module_index,
            basename_index,
            suffix_index,
        )
        if target_module:
            function_inbound[(target_module, attr_name)] += 1


def _detect_import_cycles(
    module_index: dict[str, ModuleInfo],
    import_graph: dict[str, set[str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    visited: set[str] = set()
    stack: list[str] = []
    in_stack: set[str] = set()
    seen_cycle_keys: set[str] = set()

    def dfs(node: str) -> None:
        visited.add(node)
        stack.append(node)
        in_stack.add(node)

        for neighbor in sorted(import_graph.get(node, set())):
            if neighbor not in visited:
                dfs(neighbor)
                continue
            if neighbor in in_stack:
                cycle_start = stack.index(neighbor)
                cycle_path = stack[cycle_start:] + [neighbor]
                cycle_key = " -> ".join(cycle_path)
                if cycle_key not in seen_cycle_keys:
                    seen_cycle_keys.add(cycle_key)
                    findings.append(
                        _make_finding(
                            rule="import_cycle",
                            confidence="high",
                            file=module_index[node].path,
                            line=1,
                            module=node,
                            reason=f"Import cycle detected: {' -> '.join(cycle_path)}",
                        )
                    )

        stack.pop()
        in_stack.remove(node)

    for module_name in sorted(module_index):
        if module_name not in visited:
            dfs(module_name)

    return findings


def _detect_dead_functions(
    module_index: dict[str, ModuleInfo],
    function_inbound: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for module_name, info in module_index.items():
        for fn in info.functions:
            inbound = function_inbound.get((module_name, fn.name), 0)
            if inbound > 0:
                continue

            if fn.is_private:
                findings.append(
                    _make_finding(
                        rule="dead_private_function",
                        confidence="high",
                        file=info.path,
                        line=fn.line,
                        module=module_name,
                        reason=f"Private function '{fn.name}' has no call sites",
                    )
                )
                continue

            if _is_test_module(module_name):
                continue

            findings.append(
                _make_finding(
                    rule="dead_public_function",
                    confidence="medium",
                    file=info.path,
                    line=fn.line,
                    module=module_name,
                    reason=f"Public function '{fn.name}' has no detected call sites",
                )
            )
    return findings


def _detect_unused_modules(
    module_index: dict[str, ModuleInfo],
    inbound_module_refs: dict[str, int],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for module_name, info in module_index.items():
        if _is_test_module(module_name):
            continue
        if info.path.name == "__init__.py":
            continue
        if _is_entry_module(info):
            continue

        inbound_count = inbound_module_refs.get(module_name, 0)
        if inbound_count == 0:
            findings.append(
                _make_finding(
                    rule="unused_module",
                    confidence="low",
                    file=info.path,
                    line=1,
                    module=module_name,
                    reason="Module has no inbound local imports",
                )
            )
    return findings


def analyze(module_index: dict[str, ModuleInfo]) -> list[dict[str, Any]]:
    basename_index = build_basename_index(module_index)
    suffix_index = build_suffix_index(module_index)
    findings: list[dict[str, Any]] = []
    inbound_module_refs: dict[str, int] = defaultdict(int)
    import_graph: dict[str, set[str]] = defaultdict(set)
    function_inbound: dict[tuple[str, str], int] = defaultdict(int)

    for source_name, info in module_index.items():
        _analyze_import_bindings(
            source_name,
            info,
            module_index,
            basename_index,
            suffix_index,
            findings=findings,
            inbound_module_refs=inbound_module_refs,
            import_graph=import_graph,
        )
        _resolve_function_inbound_refs(
            source_name,
            info,
            module_index,
            basename_index,
            suffix_index,
            function_inbound,
        )

    findings.extend(_detect_import_cycles(module_index, import_graph))
    findings.extend(_detect_dead_functions(module_index, function_inbound))
    findings.extend(_detect_unused_modules(module_index, inbound_module_refs))

    return sorted(findings, key=lambda item: (item["file"], item["line"], item["rule"]))


def build_todo_comment(finding: dict[str, Any]) -> str:
    rule = finding["rule"]
    reason = finding["reason"]
    return f"# TODO(analyzer:{rule}): {reason}"


def _group_high_confidence_todo_edits(
    findings: list[dict[str, Any]],
) -> dict[Path, list[dict[str, Any]]]:
    edits_by_file: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        if finding["rule"] not in HIGH_CONFIDENCE_RULES:
            continue
        line = int(finding["line"])
        if line <= 0:
            continue
        path = Path(str(finding["file"]))
        edits_by_file[path].append(finding)
    return edits_by_file


def _apply_file_todo_edits(path: Path, file_findings: list[dict[str, Any]]) -> bool:
    if not path.exists():
        return False

    lines = path.read_text(encoding="utf-8").splitlines()
    changed = False
    ordered = sorted(file_findings, key=lambda item: int(item["line"]), reverse=True)

    for finding in ordered:
        insertion_idx = max(0, int(finding["line"]) - 1)
        todo_line = build_todo_comment(finding)

        prev_line = lines[insertion_idx - 1].strip() if insertion_idx - 1 >= 0 and lines else ""
        current_line = lines[insertion_idx].strip() if insertion_idx < len(lines) and lines else ""
        marker = f"TODO(analyzer:{finding['rule']})"
        if marker in prev_line or marker in current_line:
            continue

        lines.insert(insertion_idx, todo_line)
        changed = True

    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def apply_todos(findings: list[dict[str, Any]]) -> list[Path]:
    edits_by_file = _group_high_confidence_todo_edits(findings)

    changed_files: list[Path] = []
    for path, file_findings in edits_by_file.items():
        if _apply_file_todo_edits(path, file_findings):
            changed_files.append(path)

    return sorted(changed_files)


def write_json_report(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "total": len(findings),
            "high": sum(1 for f in findings if f["confidence"] == "high"),
            "medium": sum(1 for f in findings if f["confidence"] == "medium"),
            "low": sum(1 for f in findings if f["confidence"] == "low"),
        },
        "findings": findings,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_summary(path: Path, findings: list[dict[str, Any]]) -> None:
    by_rule: dict[str, int] = defaultdict(int)
    by_conf: dict[str, int] = defaultdict(int)

    for finding in findings:
        by_rule[finding["rule"]] += 1
        by_conf[finding["confidence"]] += 1

    lines = [
        "Deterministic Static Analyzer Summary",
        "",
        f"Total findings: {len(findings)}",
        f"High confidence: {by_conf.get('high', 0)}",
        f"Medium confidence: {by_conf.get('medium', 0)}",
        f"Low confidence: {by_conf.get('low', 0)}",
        "",
        "Findings by rule:",
    ]

    for rule in sorted(by_rule):
        lines.append(f"- {rule}: {by_rule[rule]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic cross-file static analyzer")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(DEFAULT_SCAN_DIRS),
        help="Relative directories to scan",
    )
    parser.add_argument(
        "--report-path",
        default="logs/static_analysis_report.json",
        help="Path to JSON findings report",
    )
    parser.add_argument(
        "--summary-path",
        default="logs/static_analysis_summary.txt",
        help="Path to text summary report",
    )
    parser.add_argument(
        "--insert-todos",
        action="store_true",
        help="Insert TODO comments for high-confidence findings",
    )
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit non-zero when high-confidence findings exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    module_index, parse_findings = build_module_index_with_parse_findings(
        repo_root,
        list(args.paths),
    )
    findings = parse_findings + analyze(module_index)

    report_path = (repo_root / args.report_path).resolve()
    summary_path = (repo_root / args.summary_path).resolve()
    write_json_report(report_path, findings)
    write_text_summary(summary_path, findings)

    changed_files: list[Path] = []
    if args.insert_todos:
        changed_files = apply_todos(findings)

    print(
        json.dumps(
            {
                "total_findings": len(findings),
                "high": sum(1 for f in findings if f["confidence"] == "high"),
                "inserted_files": [str(path) for path in changed_files],
                "report": str(report_path),
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )

    if args.fail_on_high and any(f["confidence"] == "high" for f in findings):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
