# Tracking-Lost Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop new OrcaHand commands immediately when live tracking is lost, show a two-second blinking `TRACKING LOST` alert, recover automatically when detection returns, and fault after a continuous two-second loss.

**Architecture:** Keep command gating in `RuntimeStateMachine`: entering `TRACKING_LOST` makes `can_send_to_hardware` false immediately. Add a GUI-owned, tokenized Tk `after()` blink notification that may outlive an early runtime recovery, while a separate monotonic two-second loss timeout controls escalation to the existing emergency-stop path.

**Tech Stack:** Python 3, Tkinter, pytest, existing single-file OrcaHand GUI.

## Global Constraints

- Keep the application self-contained in `realtime_orcahand.py`.
- Do not disable motor torque on a recoverable tracking loss; hold the last command and keep the controller connected.
- Use a 2.0-second continuous-loss timeout and a 250 ms GUI blink interval.
- `FAULT` must override and cancel any active tracking-loss blink.
- Do not change landmark detection or joint-estimation behavior.

---

### Task 1: Two-Second Recoverable Tracking Pause

**Files:**
- Modify: `realtime_orcahand.py:68`
- Modify: `realtime_orcahand.py:3409-3450`
- Test: `tests/test_single_file_entrypoint.py`

**Interfaces:**
- Consumes: `RuntimeStateMachine.tracking_lost(reason)`, `recover_tracking()`, and `can_send_to_hardware`.
- Produces: `DEFAULT_TRACKING_LOST_STOP_S == 2.0`; `_handle_no_hand(now_s=...)` pauses output immediately and faults only after two continuous seconds.

- [ ] **Step 1: Write the failing timeout tests**

Rename and extend the existing
`test_single_file_gui_sustained_no_hand_safety_stops_after_grace_window` test.
Keep its complete local `FakeResettable`, `FakeVar`, and `FakeGui` definitions,
then replace the assertions after `gui = FakeGui()` with:

```python
assert module.DEFAULT_TRACKING_LOST_STOP_S == 2.0
module.OrcaRealtimeGui._handle_no_hand(gui, now_s=10.0)
module.OrcaRealtimeGui._handle_no_hand(gui, now_s=11.99)
assert gui.state.state == module.RuntimeState.TRACKING_LOST
assert gui.safety_stop_reason is None

module.OrcaRealtimeGui._handle_no_hand(gui, now_s=12.0)
assert gui.state.state == module.RuntimeState.FAULT
assert gui.safety_stop_reason == "no hand detected"


def test_single_file_gui_tracking_recovery_restores_live_before_timeout():
    module = load_single_file_module()
    state = module.RuntimeStateMachine()
    state.start_mapping()
    state.enable_live()
    state.tracking_lost("no hand detected")
    assert state.can_send_to_hardware is False
    state.recover_tracking()
    assert state.state == module.RuntimeState.LIVE
    assert state.can_send_to_hardware is True
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_single_file_entrypoint.py -k "tracking_loss_waits_two_seconds or tracking_recovery_restores_live" -q
```

Expected: the timeout assertion fails because the current value is `0.3`.

- [ ] **Step 3: Implement the minimal timeout change**

In `realtime_orcahand.py`:

```python
DEFAULT_TRACKING_LOST_STOP_S = 2.0
```

Retain the current immediate `LIVE -> TRACKING_LOST` transition in
`_handle_no_hand()` and the existing recovery call in `_process_landmarks()`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: both selected tests pass.

- [ ] **Step 5: Commit the recoverable timeout**

```powershell
git add -- realtime_orcahand.py tests/test_single_file_entrypoint.py
git commit -m "fix: allow two-second tracking recovery"
```

### Task 2: Immediate, Tokenized Tracking-Lost Blink

**Files:**
- Modify: `realtime_orcahand.py:68-72`
- Modify: `realtime_orcahand.py:2290-2330`
- Modify: `realtime_orcahand.py:2390-2520`
- Modify: `realtime_orcahand.py:3409-3450`
- Modify: `realtime_orcahand.py:3690-3780`
- Test: `tests/test_realtime_orcahand_gui_controls.py`

**Interfaces:**
- Produces: `DEFAULT_TRACKING_LOST_ALERT_MS: int`, `DEFAULT_TRACKING_LOST_BLINK_MS: int`.
- Produces: `OrcaRealtimeGui._start_tracking_lost_alert() -> None`.
- Produces: `OrcaRealtimeGui._advance_tracking_lost_alert(token: int) -> None`.
- Produces: `OrcaRealtimeGui._cancel_tracking_lost_alert(*, refresh: bool = True) -> None`.
- Produces: `OrcaRealtimeGui._refresh_runtime_state_display() -> None`.

- [ ] **Step 1: Write failing GUI alert tests**

Add lightweight fake Tk labels and an `after()` scheduler alongside the
existing `FakeVar`, then test the alert lifecycle:

```python
class FakeLabel:
    def __init__(self):
        self.foreground = None

    def configure(self, **kwargs):
        self.foreground = kwargs.get("fg", self.foreground)

    def winfo_exists(self):
        return True


class FakeRoot:
    def __init__(self):
        self.scheduled = []
        self.delays = []

    def after(self, delay_ms, callback):
        self.delays.append(delay_ms)
        self.scheduled.append(callback)

    def run_next(self):
        self.scheduled.pop(0)()


def make_state_display_gui(module, state):
    class FakeGui:
        pass

    gui = FakeGui()
    gui.root = FakeRoot()
    gui.state = module.RuntimeStateMachine(state=state)
    gui.state_var = FakeVar()
    gui.state_card_var = FakeVar()
    gui.preview_state_var = FakeVar()
    gui.state_card_label = FakeLabel()
    gui.preview_state_label = FakeLabel()
    gui._tracking_lost_alert_token = 0
    gui._tracking_lost_alert_active = False
    gui._tracking_lost_blink_visible = True
    gui._tracking_lost_blink_steps_remaining = 0
    return gui


def test_tracking_lost_alert_is_immediate_and_runs_for_eight_blink_steps():
    module = load_realtime_module()
    gui = make_state_display_gui(module, module.RuntimeState.TRACKING_LOST)

    module.OrcaRealtimeGui._start_tracking_lost_alert(gui)

    assert gui.preview_state_var.value == "STATE TRACKING LOST"
    assert gui.state_card_var.value == "TRACKING LOST"
    assert gui.root.delays == [250]
    gui.state.recover_tracking()

    for _ in range(7):
        gui.root.run_next()
        assert gui.preview_state_var.value == "STATE TRACKING LOST"

    gui.root.run_next()
    assert gui.preview_state_var.value == "STATE LIVE"


def test_fault_cancels_tracking_lost_alert_and_remains_visible():
    module = load_realtime_module()
    gui = make_state_display_gui(module, module.RuntimeState.TRACKING_LOST)
    module.OrcaRealtimeGui._start_tracking_lost_alert(gui)
    gui.state.fault("safety stop: no hand detected")

    module.OrcaRealtimeGui._cancel_tracking_lost_alert(gui)

    assert gui.preview_state_var.value == "STATE FAULT"
    assert gui.state_card_var.value == "FAULT"
```

- [ ] **Step 2: Run the GUI alert tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_realtime_orcahand_gui_controls.py -k "tracking_lost_alert or fault_cancels_tracking" -q
```

Expected: tests fail because the four alert methods do not exist.

- [ ] **Step 3: Implement the minimal alert controller**

Add constants:

```python
DEFAULT_TRACKING_LOST_ALERT_MS = 2000
DEFAULT_TRACKING_LOST_BLINK_MS = 250
```

Store references to the preview state label and snapshot state value label.
Initialize an alert token, active flag, visible flag, and remaining step count.
Use eight scheduled 250 ms advances. Each callback verifies its token before
changing UI. `_refresh_runtime_state_display()` shows the alert text while the
alert is active, but reads the actual state after the alert ends. `FAULT` uses
`CONSOLE_DANGER` and always overrides the alert.

- [ ] **Step 4: Wire every relevant transition to immediate display refresh**

- Start the alert exactly once when `_handle_no_hand()` first moves `LIVE` to
  `TRACKING_LOST`.
- Let `_process_landmarks()` recover the runtime state without cancelling the
  two-second visual alert.
- Make `_safety_stop()` and `_handle_runtime_error()` cancel the alert and
  refresh `FAULT` immediately.
- Replace `_update_button_states()` direct state-variable assignments with the
  centralized refresh method.
- In `close()`, invalidate the token before destroying Tk.

- [ ] **Step 5: Run the GUI alert tests and verify GREEN**

Run the command from Step 2. Expected: selected tests pass.

- [ ] **Step 6: Run all affected tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_realtime_orcahand_gui_controls.py tests\test_single_file_entrypoint.py -q
```

Expected: all tests pass with no warnings or errors.

- [ ] **Step 7: Run the full suite and syntax verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile realtime_orcahand.py
git diff --check
```

Expected: pytest exits with zero failures, compilation exits `0`, and
`git diff --check` prints no errors.

- [ ] **Step 8: Commit the state alert implementation**

```powershell
git add -- realtime_orcahand.py tests/test_realtime_orcahand_gui_controls.py
git commit -m "feat: flash recoverable tracking loss state"
```
