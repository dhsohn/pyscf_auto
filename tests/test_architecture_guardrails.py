from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_ENGINE = ROOT / "src" / "runner" / "attempt_engine.py"
EXECUTION_INIT = ROOT / "src" / "execution" / "__init__.py"
EXECUTION_ENTRYPOINT = ROOT / "src" / "execution" / "entrypoint.py"
EXECUTION_PLUGINS = ROOT / "src" / "execution" / "plugins.py"


def _parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_runner_uses_execution_entrypoint_boundary() -> None:
    tree = _parse_file(ATTEMPT_ENGINE)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0:
                imported_modules.add(module)

    assert "execution.entrypoint" in imported_modules
    assert "execution" not in imported_modules


def test_execution_module_does_not_import_stage_modules_directly() -> None:
    tree = _parse_file(EXECUTION_INIT)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            module = node.module or ""
            assert not module.startswith(
                "stage_"
            ), f"Direct stage import found in execution/__init__.py: {module}"


def test_boundary_modules_keep_function_size_reasonable() -> None:
    limits = {
        ATTEMPT_ENGINE: 210,
        EXECUTION_ENTRYPOINT: 80,
        EXECUTION_PLUGINS: 90,
    }
    for path, max_lines in limits.items():
        tree = _parse_file(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.end_lineno is None:
                continue
            size = node.end_lineno - node.lineno + 1
            assert (
                size <= max_lines
            ), f"{path.name}:{node.name} too large ({size} lines > {max_lines})"
