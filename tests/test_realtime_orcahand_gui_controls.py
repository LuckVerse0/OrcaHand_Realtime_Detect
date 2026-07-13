from __future__ import annotations

import realtime_orcahand as rt


def load_realtime_module():
    return rt


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


class FakeLabel:
    def __init__(self):
        self.foreground = None

    def configure(self, **kwargs):
        if "fg" in kwargs:
            self.foreground = kwargs["fg"]

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


def make_state_display_gui(module):
    class FakeGui:
        def _safety_stop(self, reason):
            self.safety_stop_reasons.append(reason)
            self.state.fault(f"safety stop: {reason}")

    gui = FakeGui()
    gui.root = FakeRoot()
    gui.state = module.RuntimeStateMachine()
    gui.state.start_mapping()
    gui.state.enable_live()
    gui.state.tracking_lost("no hand detected")
    gui.state_var = FakeVar()
    gui.state_card_var = FakeVar()
    gui.preview_state_var = FakeVar()
    gui.state_card_label = FakeLabel()
    gui.preview_state_label = FakeLabel()
    gui._tracking_lost_alert_token = 0
    gui._tracking_lost_alert_active = False
    gui._tracking_lost_blink_visible = True
    gui._tracking_lost_blink_steps_remaining = 0
    gui.safety_stop_reasons = []
    return gui


def test_tracking_lost_alert_is_immediate_and_runs_for_two_seconds():
    module = load_realtime_module()
    gui = make_state_display_gui(module)

    module.OrcaRealtimeGui._start_tracking_lost_alert(gui)

    assert gui.preview_state_var.value == "STATE TRACKING LOST"
    assert gui.state_card_var.value == "TRACKING LOST"
    assert gui.root.delays == [250]
    assert gui.preview_state_label.foreground == module.CONSOLE_DANGER

    gui.state.recover_tracking()
    gui.root.run_next()
    assert gui.preview_state_label.foreground == "#101822"
    gui.root.run_next()
    assert gui.preview_state_label.foreground == module.CONSOLE_DANGER

    for _ in range(5):
        gui.root.run_next()
        assert gui.preview_state_var.value == "STATE TRACKING LOST"

    gui.root.run_next()

    assert gui.preview_state_var.value == "STATE LIVE"
    assert gui.state_card_var.value == "LIVE"
    assert gui._tracking_lost_alert_active is False


def test_fault_cancels_tracking_lost_alert_and_remains_visible():
    module = load_realtime_module()
    gui = make_state_display_gui(module)
    module.OrcaRealtimeGui._start_tracking_lost_alert(gui)
    gui.state.fault("safety stop: no hand detected")

    module.OrcaRealtimeGui._cancel_tracking_lost_alert(gui)

    assert gui.preview_state_var.value == "STATE FAULT"
    assert gui.state_card_var.value == "FAULT"
    assert gui.preview_state_label.foreground == module.CONSOLE_DANGER
    assert gui.state_card_label.foreground == module.CONSOLE_DANGER

    gui.root.run_next()
    assert gui.preview_state_var.value == "STATE FAULT"


def test_tracking_lost_alert_deadline_faults_when_tracking_does_not_recover():
    module = load_realtime_module()
    gui = make_state_display_gui(module)

    module.OrcaRealtimeGui._start_tracking_lost_alert(gui)
    for _ in range(8):
        gui.root.run_next()

    assert gui.safety_stop_reasons == ["no hand detected"]
    assert gui.state.state == module.RuntimeState.FAULT


def test_stop_recognition_safely_stops_live_output_during_tracking_loss():
    module = load_realtime_module()

    class FakeController:
        def __init__(self):
            self.connected = True
            self.stop_count = 0

        def emergency_stop(self):
            self.stop_count += 1
            self.connected = False

    class FakeResettable:
        def reset(self):
            pass

    class FakeGui:
        def __init__(self):
            self.running = True
            self.controller = FakeController()
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.state.tracking_lost("no hand detected")
            self.latest_landmarks = object()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.release_count = 0
            self.status_messages = []
            self.alert_cancelled_in_state = None

        def _stop_live_hardware_output(self):
            module.OrcaRealtimeGui._stop_live_hardware_output(self)

        def _release_recognition_resources(self):
            self.release_count += 1

        def _cancel_tracking_lost_alert(self):
            self.alert_cancelled_in_state = self.state.state

        def _set_status(self, message):
            self.status_messages.append(message)

    gui = FakeGui()

    module.OrcaRealtimeGui.stop_recognition(gui)

    assert gui.controller.stop_count == 1
    assert gui.controller.connected is False
    assert gui.state.state == module.RuntimeState.PREVIEW
    assert gui.alert_cancelled_in_state == module.RuntimeState.PREVIEW


def test_camera_read_failure_safety_stops_live_output():
    module = load_realtime_module()

    class FakeCapture:
        def read(self):
            return False, None

    class FakeGui:
        def __init__(self):
            self.running = True
            self.cap = FakeCapture()
            self.capture_reader = None
            self.landmarker = object()
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.safety_stop_reasons = []
            self.stop_recognition_count = 0

        def _safety_stop(self, reason):
            self.safety_stop_reasons.append(reason)
            self.state.fault(f"safety stop: {reason}")

        def stop_recognition(self):
            self.stop_recognition_count += 1

    gui = FakeGui()

    module.OrcaRealtimeGui._tick(gui)

    assert gui.safety_stop_reasons == ["camera read failed"]
    assert gui.stop_recognition_count == 1


def test_runtime_error_safety_stops_output_during_tracking_loss():
    module = load_realtime_module()

    class FakeGui:
        def __init__(self):
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.state.tracking_lost("no hand detected")
            self.running = True
            self.safety_stop_reasons = []
            self.release_count = 0
            self.status_messages = []

        def _safety_stop(self, reason):
            self.safety_stop_reasons.append(reason)
            self.state.fault(f"safety stop: {reason}")

        def _release_recognition_resources(self):
            self.release_count += 1

        def _set_status(self, message):
            self.status_messages.append(message)

    gui = FakeGui()

    module.OrcaRealtimeGui._handle_runtime_error(gui, RuntimeError("boom"))

    assert gui.safety_stop_reasons == ["runtime error: boom"]
    assert gui.state.state == module.RuntimeState.FAULT
    assert gui.running is False


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
