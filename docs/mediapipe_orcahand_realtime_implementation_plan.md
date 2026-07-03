# MediaPipe OrcaHand Realtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dry-run-first realtime MediaPipe to OrcaHand controller with smoothing, state gating, safety offsets, motor limit checks, logging, and an optional live hardware path.

**Architecture:** Add a focused `orca_realtime` package for testable core logic, then add a CLI entrypoint that reuses the existing MediaPipe model workflow. Core modules are pure Python/Numpy where possible; hardware control is isolated behind an `OrcaController` wrapper and only used in `--live`.

**Tech Stack:** Python, Numpy, OpenCV, MediaPipe Tasks, PyYAML, unittest/pytest-compatible tests, Orca core loaded optionally for live mode.

---

## File Structure

- Create `orca_realtime/config.py`: load `config/config.yaml` and `config/calibration.yaml`, normalize signed joint-to-motor mappings, build safety defaults, validate live readiness.
- Create `orca_realtime/filters.py`: exponential smoothing for landmarks and joint dictionaries.
- Create `orca_realtime/state.py`: runtime state machine for `preview`, `armed`, `live`, `tracking_lost`, and `fault`.
- Create `orca_realtime/safety.py`: joint safe ROMs, motor safe limits, shared offset conversion, max-delta limiting, disabled joint behavior, wrist neutral enforcement, abrupt-change rejection.
- Create `orca_realtime/kinematics.py`: convert MediaPipe landmarks to 17 Orca joint commands using stable geometric control signals.
- Create `orca_realtime/logging_utils.py`: CSV/JSON session logging.
- Create `orca_realtime/orca_controller.py`: optional Orca core live wrapper.
- Create `realtime_orcahand_control.py`: CLI app with dry-run/live modes, buttons/keys, preview overlays, emergency stop.
- Add tests under `tests/` for config loading, safety, filters, state transitions, and kinematics smoke behavior.
- Modify `requirements.txt`: add `PyYAML` and `pytest`.

## Task 1: Dependencies and Core Config

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_config.py`
- Create: `orca_realtime/__init__.py`
- Create: `orca_realtime/config.py`

- [ ] Add `PyYAML` and `pytest` to requirements.
- [ ] Write failing tests that load the current `config` files, require 17 joints, normalize signed motor mappings, and reject incomplete live calibration.
- [ ] Implement `load_realtime_config()` and dataclasses.
- [ ] Run `python -m pytest tests/test_config.py -v`.

## Task 2: Safety Controller

**Files:**
- Create: `tests/test_safety.py`
- Create: `orca_realtime/safety.py`

- [ ] Write failing tests for shared joint/motor offset conversion, safe neutral clamping, max-delta limiting, disabled joint behavior, wrist neutral enforcement, and abrupt-change rejection.
- [ ] Implement `SafetyController`.
- [ ] Run `python -m pytest tests/test_safety.py -v`.

## Task 3: Filters and State Machine

**Files:**
- Create: `tests/test_filters_state.py`
- Create: `orca_realtime/filters.py`
- Create: `orca_realtime/state.py`

- [ ] Write failing tests for exponential smoothing, reset behavior, legal state transitions, tracking lost, fault, and reset.
- [ ] Implement filters and runtime state machine.
- [ ] Run `python -m pytest tests/test_filters_state.py -v`.

## Task 4: Kinematics Mapper

**Files:**
- Create: `tests/test_kinematics.py`
- Create: `orca_realtime/kinematics.py`

- [ ] Write failing tests with synthetic open/curled hand landmarks and verify 17 joint keys, wrist neutral, and increased finger curl.
- [ ] Implement vector helpers, finger curl estimation, abduction estimation, neutral baseline capture, and joint gain application.
- [ ] Run `python -m pytest tests/test_kinematics.py -v`.

## Task 5: Logging and Orca Wrapper

**Files:**
- Create: `tests/test_logging_and_controller.py`
- Create: `orca_realtime/logging_utils.py`
- Create: `orca_realtime/orca_controller.py`

- [ ] Write failing tests for log row creation and dry-run safety around missing Orca core.
- [ ] Implement CSV/JSON logging and optional live wrapper with emergency `disable_torque()`.
- [ ] Run `python -m pytest tests/test_logging_and_controller.py -v`.

## Task 6: Realtime Entrypoint

**Files:**
- Create: `realtime_orcahand_control.py`

- [ ] Implement CLI args: `--camera`, `--mirror`, `--model`, `--live`, `--orca-core-root`, `--log-dir`, `--once`, `--config-dir`.
- [ ] Reuse MediaPipe model download and drawing style from `mediapipe_hand.py`.
- [ ] Wire state machine keys: `m`, `l`, `n`, `Space`, `Esc`, `q`.
- [ ] Ensure default run is dry-run preview and live requires explicit `--live`.
- [ ] Run `python -m pytest tests -v`.
- [ ] Run `python -m compileall orca_realtime realtime_orcahand_control.py`.

## Self-Review

Spec coverage:
- Configuration source is current `config`: Task 1 and Task 6.
- Smoothing: Task 3 and Task 6.
- Start mapping button/state gates: Task 3 and Task 6.
- Abrupt-change rejection: Task 2.
- Joint ROM and motor limits with shared offset: Task 2.
- 17 DOF mapping and wrist fixed neutral: Task 4.
- Single joint enable/gain: Task 2 and Task 4.
- Logs and replay data: Task 5.
- Emergency stop: Task 3, Task 5, and Task 6.

No placeholders remain. All tasks have concrete files and verification commands.
