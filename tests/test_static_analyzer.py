# SPDX-License-Identifier: MIT
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "scripts" / "project_agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from static_analyzer import analyze, apply_todos, build_module_index  # noqa: E402  # type: ignore[reportMissingImports]


class StaticAnalyzerTests(unittest.TestCase):
    def _write(self, root: Path, rel: str, content: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_detects_unresolved_import_and_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "pkg/a.py",
                "import pkg.missing_local\nfrom pkg.b import nope\n",
            )
            self._write(
                root,
                "pkg/b.py",
                "def ok():\n    return 1\n",
            )

            index = build_module_index(root, ["pkg"])
            findings = analyze(index)
            rules = {f["rule"] for f in findings}

            self.assertIn("unresolved_local_import", rules)
            self.assertIn("unresolved_from_symbol", rules)

    def test_ignores_external_and_stdlib_imports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "pkg/a.py",
                "from __future__ import annotations\nimport math\nfrom typing import Any\n",
            )

            index = build_module_index(root, ["pkg"])
            findings = analyze(index)
            unresolved = [f for f in findings if f["rule"] == "unresolved_local_import"]

            self.assertEqual(unresolved, [])

    def test_resolves_unique_suffix_alias_import(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root, "scripts/project_agent/agent_platform/context.py", "def ok():\n    return 1\n")
            self._write(
                root,
                "scripts/project_agent/consumer.py",
                "from agent_platform.context import ok\n\n\ndef run():\n    return ok()\n",
            )

            index = build_module_index(root, ["scripts"])
            findings = analyze(index)
            unresolved = [f for f in findings if f["rule"] == "unresolved_local_import"]
            unresolved_symbols = [f for f in findings if f["rule"] == "unresolved_from_symbol"]

            self.assertEqual(unresolved, [])
            self.assertEqual(unresolved_symbols, [])

    def test_detects_dead_private_function(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root,
                "pkg/a.py",
                "def _hidden():\n    return 1\n",
            )

            index = build_module_index(root, ["pkg"])
            findings = analyze(index)
            dead_private = [f for f in findings if f["rule"] == "dead_private_function"]

            self.assertEqual(len(dead_private), 1)
            self.assertEqual(dead_private[0]["line"], 1)

    def test_detects_import_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(root, "pkg/a.py", "import b\n")
            self._write(root, "pkg/b.py", "import a\n")

            index = build_module_index(root, ["pkg"])
            findings = analyze(index)
            cycle = [f for f in findings if f["rule"] == "import_cycle"]
            self.assertTrue(cycle)

    def test_applies_todos_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "pkg" / "a.py"
            self._write(root, "pkg/a.py", "def _hidden():\n    return 1\n")

            findings = [
                {
                    "rule": "dead_private_function",
                    "confidence": "high",
                    "file": str(target),
                    "line": 1,
                    "module": "pkg.a",
                    "reason": "Private function '_hidden' has no call sites",
                }
            ]

            changed_first = apply_todos(findings)
            changed_second = apply_todos(findings)

            self.assertEqual(changed_first, [target])
            self.assertEqual(changed_second, [])

            text = target.read_text(encoding="utf-8")
            self.assertIn("TODO(analyzer:dead_private_function)", text)


if __name__ == "__main__":
    unittest.main()
