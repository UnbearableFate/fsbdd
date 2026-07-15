from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/fsbdd"
PRODUCTION = PACKAGE / "diloco"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            values.append(node.module)
    return tuple(values)


def test_production_tree_has_no_auxiliary_dependency() -> None:
    violations = {
        path.relative_to(ROOT).as_posix(): imported
        for path in PRODUCTION.rglob("*.py")
        for imported in _imports(path)
        if imported == "fsbdd.auxiliary" or imported.startswith("fsbdd.auxiliary.")
    }
    assert violations == {}


def test_root_package_contains_only_entrypoint_and_namespaces() -> None:
    assert {path.name for path in PACKAGE.glob("*.py")} == {"__init__.py", "cli.py"}
    assert {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    } >= {
        "auxiliary",
        "diloco",
    }


def test_cpu_syncer_modules_do_not_import_torch_at_module_scope() -> None:
    for path in (PRODUCTION / "syncer").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top_level = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert "torch" not in top_level
