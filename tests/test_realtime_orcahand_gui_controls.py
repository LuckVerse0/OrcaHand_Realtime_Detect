from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REALTIME_FILE = Path("realtime_orcahand.py")


def load_realtime_module():
    spec = importlib.util.spec_from_file_location("realtime_orcahand_current", REALTIME_FILE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeButton:
    def __init__(self):
        self.state = None
        self.text = None

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]
        if "text" in kwargs:
            self.text = kwargs["text"]


class FakeVar:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


def test_neutral_button_sends_safe_neutral_and_exits_live_output():
    module = load_realtime_module()

    class FakeController:
        connected = True

        def __init__(self):
            self.sent = []

        def send(self, joints):
            self.sent.append(dict(joints))

    class FakeSafety:
        def __init__(self):
            self.reset_count = 0

        def safe_neutral(self):
            return {"index_mcp": 1.5, "wrist": 0.0}

        def reset_to_safe_neutral(self):
            self.reset_count += 1

    class FakeSmoother:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class FakeGui:
        hardware_allowed = True

        def __init__(self):
            self.controller = FakeController()
            self.safety = FakeSafety()
            self.joint_smoother = FakeSmoother()
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.status_messages = []
            self.telemetry = None
            self.update_count = 0
            self.stop_motion_calls = []

        def stop_processed_video_playback(self, **kwargs):
            self.stop_motion_calls.append(kwargs)

        def _update_safety_telemetry(self, safety_result):
            self.telemetry = safety_result

        def _set_status(self, message):
            self.status_messages.append(message)

        def _update_button_states(self):
            self.update_count += 1

    gui = FakeGui()

    module.OrcaRealtimeGui.move_to_neutral(gui)

    assert gui.controller.sent == [{"index_mcp": 1.5, "wrist": 0.0}]
    assert gui.safety.reset_count == 1
    assert gui.joint_smoother.reset_count == 1
    assert gui.state.state == module.RuntimeState.PREVIEW
    assert gui.telemetry.joints == {"index_mcp": 1.5, "wrist": 0.0}
    assert gui.telemetry.accepted is True
    assert gui.stop_motion_calls == [{"update_status": False}]
    assert gui.status_messages[-1] == "Moved OrcaHand to neutral."
    assert gui.update_count == 1


def test_neutral_button_is_enabled_only_when_hardware_can_receive_commands():
    module = load_realtime_module()

    class FakeController:
        connected = False

    class FakeGui:
        running = False
        hardware_allowed = True
        _background_busy = False
        _calibration_busy = False
        _video_processing_busy = False
        _video_playing = False
        state = module.RuntimeStateMachine()
        controller = FakeController()
        state_var = FakeVar()
        hardware_var = FakeVar()
        start_button = FakeButton()
        stop_button = FakeButton()
        connect_button = FakeButton()
        neutral_button = FakeButton()
        map_button = FakeButton()
        stop_map_button = FakeButton()
        output_button = FakeButton()
        stop_output_button = FakeButton()
        calibrate_button = FakeButton()
        stop_calibrate_button = FakeButton()
        import_video_button = FakeButton()
        stop_video_button = FakeButton()

    gui = FakeGui()

    module.OrcaRealtimeGui._update_button_states(gui)
    assert gui.neutral_button.state == "disabled"

    gui.controller.connected = True
    module.OrcaRealtimeGui._update_button_states(gui)
    assert gui.neutral_button.state == "normal"

    gui._background_busy = True
    module.OrcaRealtimeGui._update_button_states(gui)
    assert gui.neutral_button.state == "disabled"


def test_motion_controls_can_be_collapsed_and_expanded():
    module = load_realtime_module()

    class FakeFrame:
        def __init__(self):
            self.visible = None
            self.grid_kwargs = None

        def grid(self, **kwargs):
            self.visible = True
            self.grid_kwargs = kwargs

        def grid_remove(self):
            self.visible = False

    class FakeGui:
        def __init__(self):
            self.motion_controls_visible = False
            self.motion_controls_frame = FakeFrame()
            self.motion_toggle_button = FakeButton()

    gui = FakeGui()

    module.OrcaRealtimeGui._sync_motion_controls_visibility(gui)
    assert gui.motion_controls_frame.visible is False
    assert gui.motion_toggle_button.text == "Show Motion Source"

    module.OrcaRealtimeGui._toggle_motion_controls(gui)
    assert gui.motion_controls_visible is True
    assert gui.motion_controls_frame.visible is True
    assert gui.motion_controls_frame.grid_kwargs["row"] == 8
    assert gui.motion_toggle_button.text == "Hide Motion Source"

    module.OrcaRealtimeGui._toggle_motion_controls(gui)
    assert gui.motion_controls_visible is False
    assert gui.motion_controls_frame.visible is False
    assert gui.motion_toggle_button.text == "Show Motion Source"


def test_stop_recognition_after_live_disconnect_returns_to_preview_for_mapping():
    module = load_realtime_module()

    class FakeController:
        def __init__(self):
            self.connected = True
            self.stop_count = 0

        def emergency_stop(self):
            self.stop_count += 1
            self.connected = False

    class FakeResettable:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class FakeGui:
        def __init__(self):
            self.running = True
            self.controller = FakeController()
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.latest_landmarks = object()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.release_count = 0
            self.status_messages = []

        def _stop_live_hardware_output(self):
            module.OrcaRealtimeGui._stop_live_hardware_output(self)

        def _release_recognition_resources(self):
            self.release_count += 1

        def _set_status(self, message):
            self.status_messages.append(message)

    gui = FakeGui()

    module.OrcaRealtimeGui.stop_recognition(gui)

    assert gui.controller.connected is False
    assert gui.controller.stop_count == 1
    assert gui.state.state == module.RuntimeState.PREVIEW
    assert gui.running is False
    assert gui.status_messages[-1] == "Recognition stopped."


def test_successful_reconnect_clears_fault_so_mapping_can_start_again():
    module = load_realtime_module()

    class FakeController:
        connected = True

    class FakeGui:
        def __init__(self):
            self.controller = FakeController()
            self.state = module.RuntimeStateMachine()
            self.state.fault("emergency stop")
            self._play_video_after_connect = False
            self.reloaded = False
            self.status_messages = []

        def _reload_runtime_config(self):
            self.reloaded = True

        def _set_status(self, message):
            self.status_messages.append(message)

    gui = FakeGui()

    module.OrcaRealtimeGui._handle_background_done(
        gui,
        "OrcaHand connected and moved to neutral.",
    )

    assert gui.reloaded is True
    assert gui.state.state == module.RuntimeState.PREVIEW
    assert gui.status_messages[-1] == "OrcaHand connected and moved to neutral."


def test_disable_output_after_live_disconnect_returns_to_preview():
    module = load_realtime_module()

    class FakeController:
        def __init__(self):
            self.connected = True

        def emergency_stop(self):
            self.connected = False

    class FakeGui:
        def __init__(self):
            self.controller = FakeController()
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.status_messages = []

        def _stop_live_hardware_output(self):
            module.OrcaRealtimeGui._stop_live_hardware_output(self)

        def _set_status(self, message):
            self.status_messages.append(message)

    gui = FakeGui()

    module.OrcaRealtimeGui.disable_output(gui)

    assert gui.controller.connected is False
    assert gui.state.state == module.RuntimeState.PREVIEW
    assert gui.status_messages[-1] == "Hardware output disabled."
