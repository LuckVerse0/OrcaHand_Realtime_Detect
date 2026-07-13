# Single-Source Test Migration Design

## Goal

Make `realtime_orcahand.py` the only application implementation and the only
source imported by product-behavior tests. Remove the duplicated
`orca_realtime` package and the legacy `realtime_orcahand_control.py` CLI so a
feature change never requires updating two implementations.

## Deletion Scope

Delete these tracked application paths:

- `orca_realtime/` and every file below it.
- `realtime_orcahand_control.py`.

Before recursive deletion, resolve both the workspace root and deletion target
to absolute Windows paths and verify that each target is a direct child of the
workspace. Use PowerShell path operations end-to-end.

The package-only `cleanup_generated_logs()` function is not used by the current
application. It and its dedicated test are deleted instead of adding a
test-only utility to `realtime_orcahand.py`.

## Test Import Policy

Every retained Python test module under `tests/` imports the current program as:

```python
import realtime_orcahand as rt
```

Tests call classes and functions through `rt`, including configuration,
filters, state machine, safety, kinematics, calibration, visual calibration,
logging, controller, MediaPipe helpers, video processing, and GUI behavior.
Importing the module does not start Tk or the camera because execution remains
guarded by `if __name__ == "__main__"`.

Dynamic `importlib.util.spec_from_file_location()` loaders are removed unless a
test specifically verifies import isolation that cannot be expressed through a
normal import. No retained behavior test imports `orca_realtime`,
`realtime_orcahand_control`, or a missing legacy GUI module.

## Legacy Test Handling

- Rewrite `tests/test_entrypoint.py` to validate the current
  `realtime_orcahand.py` parser and MediaPipe helpers, not the deleted CLI.
- Delete `tests/test_gui_entrypoint.py`; it only targets the absent
  `realtime_orcahand_gui.py`, and the current GUI already has active coverage.
- Remove old `_handle_key()` tests from `tests/test_single_file_entrypoint.py`.
  They test behavior intentionally removed with the legacy OpenCV CLI.
- Update stale drawing and status assertions to the current GUI/preview
  contract only when the behavior remains part of `realtime_orcahand.py`.
- Remove the `cleanup_generated_logs()` test from
  `tests/test_visual_calibration.py`.
- Preserve meaningful tests for current behavior; do not weaken safety,
  calibration, kinematics, controller, logging, video, or GUI assertions merely
  to make the suite pass.

Redundant tests may be consolidated or deleted when an equivalent current-main
test already covers the same behavior.

## Documentation

Update `README.md` to remove references to `realtime_orcahand_control.py` and to
describe `realtime_orcahand.py` as the sole application entry point and source
of runtime behavior.

## Enforcement

Add or update a structural test that verifies:

1. `orca_realtime/` does not exist.
2. `realtime_orcahand_control.py` does not exist.
3. Retained test source contains no imports from either deleted implementation.
4. `realtime_orcahand.py` remains importable without starting the GUI.

The enforcement test is written and observed failing before the deletion and
import migration.

## Verification

The migration is complete only when all of the following hold:

- `rg` finds no runtime or test import of `orca_realtime` or
  `realtime_orcahand_control`.
- Every retained `tests/test_*.py` imports `realtime_orcahand as rt`.
- `python -m py_compile realtime_orcahand.py` exits successfully.
- The complete pytest suite passes with zero failures.
- `git diff --check` reports no whitespace errors.
- The resolved deletion targets were recorded as inside the workspace before
  deletion, and `git status` shows only the intended removals and migrations.
