import ast
from pathlib import Path

import realtime_orcahand as rt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "tests"


def test_realtime_orcahand_is_the_only_runtime_implementation():
    assert rt.PROJECT_ROOT == PROJECT_ROOT
    assert not (PROJECT_ROOT / "orca_realtime").exists()
    assert not (PROJECT_ROOT / "realtime_orcahand_control.py").exists()
    assert not (PROJECT_ROOT / "realtime_orcahand_control").exists()


def test_every_retained_test_imports_the_main_runtime_module():
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            (alias.name, alias.asname)
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_from = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

        assert ("realtime_orcahand", "rt") in imports, path.name
        assert not any(
            name == "orca_realtime" or name.startswith("orca_realtime.")
            for name, _alias in imports
        ), path.name
        assert not any(
            name == "orca_realtime" or (name or "").startswith("orca_realtime.")
            for name in imported_from
        ), path.name
        assert not any(
            name == "realtime_orcahand_control"
            or name.startswith("realtime_orcahand_control.")
            for name, _alias in imports
        ), path.name
        assert not any(
            name == "realtime_orcahand_control"
            or (name or "").startswith("realtime_orcahand_control.")
            for name in imported_from
        ), path.name
