# Single-Source Test Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the duplicated `orca_realtime` package and legacy CLI, then make every retained test import and exercise `realtime_orcahand.py`.

**Architecture:** Treat `realtime_orcahand.py` as the sole importable runtime module. Migrate modular and dynamically loaded tests to `import realtime_orcahand as rt`, remove tests for intentionally deleted legacy behavior, and enforce the single-source layout with a structural test.

**Tech Stack:** Python 3, pytest, PowerShell, Git.

## Global Constraints

- Delete `orca_realtime/` and `realtime_orcahand_control.py` only after resolving their absolute paths and proving they are direct children of the workspace.
- Use PowerShell end-to-end for recursive deletion.
- Every retained `tests/test_*.py` must contain `import realtime_orcahand as rt`.
- Do not restore `_handle_key()` or other deleted legacy CLI behavior.
- Do not add package-only `cleanup_generated_logs()` to the main program.
- Preserve current safety, kinematics, calibration, controller, logging, video, and GUI assertions.
- The complete pytest suite must finish with zero failures.

---

### Task 1: Add the Single-Source Enforcement Test

**Files:**
- Create: `tests/test_single_source.py`

**Interfaces:**
- Consumes: repository paths relative to `Path(__file__).resolve().parents[1]`.
- Produces: structural enforcement that deleted implementations stay absent and every retained test imports `realtime_orcahand as rt`.

- [ ] **Step 1: Create the failing structural test**

```python
from pathlib import Path

import realtime_orcahand as rt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "tests"


def test_realtime_orcahand_is_the_only_runtime_implementation():
    assert rt.PROJECT_ROOT == PROJECT_ROOT
    assert not (PROJECT_ROOT / "orca_realtime").exists()
    assert not (PROJECT_ROOT / "realtime_orcahand_control.py").exists()


def test_every_retained_test_imports_the_main_runtime_module():
    for path in sorted(TEST_ROOT.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        assert "import realtime_orcahand as rt" in source, path.name
        assert "from orca_realtime" not in source, path.name
        assert "import orca_realtime" not in source, path.name
        assert "realtime_orcahand_control" not in source, path.name
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_single_source.py -q
```

Expected: failures report the existing `orca_realtime` directory, legacy CLI,
and test files that do not import the main module.

### Task 2: Migrate Current Behavior Tests to the Main Module

**Files:**
- Modify: `tests/test_calibration_status.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_filters_state.py`
- Modify: `tests/test_kinematics.py`
- Modify: `tests/test_logging_and_controller.py`
- Modify: `tests/test_safety.py`
- Modify: `tests/test_visual_calibration.py`
- Modify: `tests/test_realtime_orcahand_calibration_stop.py`
- Modify: `tests/test_realtime_orcahand_gui_controls.py`
- Modify: `tests/test_realtime_orcahand_missing_calibration.py`
- Modify: `tests/test_realtime_orcahand_video_import.py`
- Modify: `tests/test_single_file_entrypoint.py`
- Modify: `tests/test_entrypoint.py`
- Delete: `tests/test_gui_entrypoint.py`

**Interfaces:**
- Consumes: public classes/functions defined directly in `realtime_orcahand.py`.
- Produces: all retained test modules import `realtime_orcahand as rt`; helper loaders return `rt` rather than dynamically loading a second module instance.

- [ ] **Step 1: Replace package imports**

Use this import pattern in each retained test:

```python
import realtime_orcahand as rt
```

Where a file currently uses unqualified imported names, bind them explicitly
from the main module to keep the test body focused:

```python
RuntimeSafetySettings = rt.RuntimeSafetySettings
load_realtime_config = rt.load_realtime_config
SafetyController = rt.SafetyController
HandKinematics = rt.HandKinematics
```

Apply the equivalent bindings only for names each file uses. Remove
`importlib.util`, `sys`, and file-loader boilerplate when no longer used.

- [ ] **Step 2: Retarget legacy entrypoint coverage**

In `tests/test_entrypoint.py`, replace imports from
`realtime_orcahand_control` and `tools.mediapipe_hand` with `rt`. Assert the
current parser contract:

```python
def test_entrypoint_defaults_to_live_gui_runtime():
    args = rt.build_arg_parser().parse_args([])
    assert args.live is True
    assert rt.build_arg_parser().parse_args(["--preview-only"]).live is False
```

Use `rt.detect_frame` for the MediaPipe conversion test. Delete
`tests/test_gui_entrypoint.py` because it exclusively targets a missing legacy
GUI file.

- [ ] **Step 3: Remove package-only and legacy-key tests**

Delete the `cleanup_generated_logs()` import and its dedicated test from
`tests/test_visual_calibration.py`.

Delete these obsolete functions from `tests/test_single_file_entrypoint.py`:

```text
test_single_file_o_and_c_capture_user_flex_range
test_single_file_a_and_s_capture_user_abd_range
test_single_file_m_and_l_block_until_user_range_calibration
test_single_file_m_blocks_when_abd_range_calibration_is_missing
test_single_file_terminal_colors_and_mapping_message
test_single_file_m_key_does_not_capture_camera_hand_as_visual_neutral
test_single_file_l_key_allows_live_without_camera_neutral_and_starts_ramp
test_single_file_l_key_does_not_capture_neutral_before_enabling_live
test_single_file_preview_safety_rejection_does_not_auto_arm_mapping
test_single_file_m_and_l_keys_report_fault_instead_of_fake_mapping_messages
```

Update the drawing assertion to expect `rt.DEFAULT_DRAW_LINE_TYPE`, and update
the status-line assertion to the current two-line `OUTPUT OFF`/`OUTPUT ON`
contract.

- [ ] **Step 4: Run migrated tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected at this intermediate stage: behavior tests pass; the structural test
still fails only because the old runtime paths have not yet been deleted.

### Task 3: Delete Duplicate Implementations and Update Documentation

**Files:**
- Delete: `orca_realtime/`
- Delete: `realtime_orcahand_control.py`
- Modify: `README.md`

**Interfaces:**
- Produces: one runtime implementation, `realtime_orcahand.py`.

- [ ] **Step 1: Verify resolved deletion targets**

Run:

```powershell
$workspace = (Resolve-Path '.').Path
$package = (Resolve-Path '.\orca_realtime').Path
$legacyCli = (Resolve-Path '.\realtime_orcahand_control.py').Path
$separator = [IO.Path]::DirectorySeparatorChar
if (-not $package.StartsWith($workspace + $separator)) { throw "package target escaped workspace" }
if (-not $legacyCli.StartsWith($workspace + $separator)) { throw "CLI target escaped workspace" }
if ((Split-Path $package -Parent) -ne $workspace) { throw "package is not a direct child" }
if ((Split-Path $legacyCli -Parent) -ne $workspace) { throw "CLI is not a direct child" }
Write-Output "Workspace: $workspace"
Write-Output "Package: $package"
Write-Output "Legacy CLI: $legacyCli"
```

Expected: all paths begin with the workspace path and both targets are direct
children.

- [ ] **Step 2: Delete the verified targets with PowerShell**

Run in the same PowerShell environment using the verified literal paths:

```powershell
Remove-Item -LiteralPath $package -Recurse -Force
Remove-Item -LiteralPath $legacyCli -Force
```

- [ ] **Step 3: Update README**

Remove the note that the older CLI is retained. State that
`realtime_orcahand.py` is the sole application entry point and sole runtime
implementation exercised by tests.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_single_source.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile realtime_orcahand.py
rg -n "^(from|import) orca_realtime|^from realtime_orcahand_control" tests README.md realtime_orcahand.py
git diff --check
git status --short
```

Expected: both pytest commands pass with zero failures; compilation and diff
check exit successfully; `rg` returns no matches; Git lists only the intended
test/document changes and deletions.

- [ ] **Step 5: Commit**

```powershell
git add -A -- README.md realtime_orcahand.py tests orca_realtime realtime_orcahand_control.py
git commit -m "refactor: make realtime app the single tested source"
```
