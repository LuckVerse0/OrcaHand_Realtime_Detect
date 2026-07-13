from __future__ import annotations

from types import SimpleNamespace
import time
from pathlib import Path

import numpy as np
import pytest
import realtime_orcahand as rt


SINGLE_FILE = Path("realtime_orcahand.py")


def load_single_file_module():
    return rt


def synthetic_hand(curled: bool = False) -> np.ndarray:
    points = np.zeros((21, 3), dtype=float)
    points[0] = [0.0, 0.0, 0.0]
    for start, x in ((1, -1.0), (5, -0.45), (9, 0.0), (13, 0.45), (17, 0.9)):
        points[start] = [x, 1.0, 0.0]
        points[start + 1] = [x, 2.0, 0.0]
        points[start + 2] = [x, 3.0, 0.0]
        points[start + 3] = [x, 4.0, 0.0]

    if curled:
        for start in (5, 9, 13, 17):
            x = points[start, 0]
            points[start + 1] = [x, 1.8, 0.0]
            points[start + 2] = [x + 0.45, 2.0, 0.0]
            points[start + 3] = [x + 0.8, 1.65, 0.0]

    return points


def tracked_hand_keypoints(offset_x: float = 0.0) -> np.ndarray:
    points = synthetic_hand(curled=False) * 60.0
    points[:, 0] += 220.0 + offset_x
    points[:, 1] += 120.0
    return points.astype(np.float32)


def abducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[6, 7, 8], 0] -= 0.45
    hand[[10, 11, 12], 0] += 0.35
    hand[[14, 15, 16], 0] += 0.45
    hand[[18, 19, 20], 0] += 0.45
    hand[[1, 2, 3, 4], 0] -= 0.45
    return hand


def adducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[6, 7, 8], 0] += 0.35
    hand[[10, 11, 12], 0] -= 0.35
    hand[[14, 15, 16], 0] -= 0.35
    hand[[18, 19, 20], 0] -= 0.35
    hand[[1, 2, 3, 4], 0] += 0.55
    return hand


def capture_complete_visual_calibration(module, kinematics) -> None:
    neutral = synthetic_hand(curled=False)
    kinematics.capture_neutral(neutral)
    kinematics.capture_open_pose(neutral)
    kinematics.capture_closed_pose(synthetic_hand(curled=True))
    kinematics.capture_abd_spread_pose(abducted_hand(neutral))
    kinematics.capture_abd_together_pose(adducted_hand(neutral))


def test_single_file_entrypoint_is_self_contained_and_parseable():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    assert "from orca_realtime" not in source
    assert "import orca_realtime" not in source
    assert "from mediapipe_hand" not in source
    assert "import mediapipe_hand" not in source
    compile(source, str(SINGLE_FILE), "exec")


def test_single_file_user_messages_do_not_reference_removed_cli_options():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    assert "--orca-core-root" not in source
    assert "--mock-orca" not in source


def test_single_file_entrypoint_only_exposes_live_user_option():
    module = load_single_file_module()
    parser = module.build_arg_parser()

    assert parser.parse_args([]).live is True
    assert parser.parse_args(["--live"]).live is True
    assert parser.parse_args(["--preview-only"]).live is False
    assert parser.parse_args(["--no-live"]).live is False

    help_text = parser.format_help()
    assert "--live" in help_text
    assert "--preview-only" in help_text

    hidden_options = [
        "--camera",
        "--camera-width",
        "--camera-height",
        "--capture-fps",
        "--inference-width",
        "--display-width",
        "--display-height",
        "--display-fps",
        "--max-detected-hands",
        "--camera-backend",
        "--use-gpu",
        "--perf-log",
        "--mirror",
        "--model",
        "--config-dir",
        "--mock-orca",
        "--orca-core-root",
        "--log-dir",
        "--once",
        "--max-delta",
        "--abrupt-delta",
        "--offset-deg",
        "--startup-ramp-frames",
        "--startup-max-delta",
        "--force-calibrate",
        "--skip-neutral-on-connect",
        "--send-steps",
        "--send-step-size",
    ]
    for option in hidden_options:
        assert option not in help_text
        with pytest.raises(SystemExit):
            parser.parse_args([option])
        with pytest.raises(SystemExit):
            parser.parse_args([option])


def test_single_file_runtime_defaults_are_internal_and_safe_for_live_use():
    module = load_single_file_module()

    assert module.DEFAULT_MIRROR is False
    assert module.DEFAULT_CAMERA == 0
    assert module.DEFAULT_USE_GPU is False
    assert module.DEFAULT_CONFIG_DIR == Path("config")
    assert module.DEFAULT_LOG_DIR == Path("logs")
    assert module.DEFAULT_VISUAL_CALIBRATION_DIR == Path("profiles") / "visual"
    assert module.DEFAULT_ORCA_CORE_ROOT == module.PROJECT_ROOT / "vendor" / "orca_core"
    assert (module.DEFAULT_ORCA_CORE_ROOT / "orca_core" / "__init__.py").exists()
    assert module.DEFAULT_STARTUP_RAMP_FRAMES == 60
    assert module.DEFAULT_STARTUP_MAX_DELTA_DEG_PER_FRAME == 0.5
    assert module.DEFAULT_CAPTURE_FPS == 30
    assert module.DEFAULT_INFERENCE_MAX_WIDTH <= 480
    assert module.DEFAULT_DISPLAY_MAX_WIDTH <= module.DEFAULT_CAPTURE_WIDTH
    assert module.DEFAULT_DISPLAY_MAX_HEIGHT <= module.DEFAULT_CAPTURE_HEIGHT


def test_single_file_loads_runtime_safety_settings_from_config(tmp_path):
    module = load_single_file_module()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """
type: right
joint_ids: [index_mcp]
motor_ids: [1]
joint_to_motor_map:
  index_mcp: 1
joint_roms:
  index_mcp: [-20, 95]
neutral_position:
  index_mcp: 0
runtime_safety:
  default_offset_deg: 3.0
  max_delta_deg_per_frame: 2.0
  startup_max_delta_deg_per_frame: 0.25
  startup_ramp_frames: 12
  abrupt_delta_deg: 45.0
  offset_overrides_deg:
    index_mcp: 6.0
  joint_controls:
    index_mcp:
      enabled: false
      gain: 0.4
""",
        encoding="utf-8",
    )
    (config_dir / "calibration.yaml").write_text(
        """
calibrated: true
wrist_calibrated: true
motor_limits:
  1: [0.0, 1.5]
joint_to_motor_ratios:
  1: 0.01
""",
        encoding="utf-8",
    )

    config = module.load_realtime_config(config_dir)

    assert config.runtime_safety.default_offset_deg == 3.0
    assert config.runtime_safety.max_delta_deg_per_frame == 2.0
    assert config.runtime_safety.startup_max_delta_deg_per_frame == 0.25
    assert config.runtime_safety.startup_ramp_frames == 12
    assert config.runtime_safety.abrupt_delta_deg == 45.0
    assert config.runtime_safety.offset_for_joint("index_mcp") == 6.0
    assert config.runtime_safety.control_for_joint("index_mcp") == (False, 0.4)


def test_single_file_is_the_unified_gui_program_entrypoint():
    module = load_single_file_module()

    assert hasattr(module, "OrcaRealtimeGui")
    assert hasattr(module, "CalibrationResult")
    assert hasattr(module, "save_visual_calibration")
    assert hasattr(module, "load_visual_calibration")
    assert hasattr(module, "apply_visual_calibration")


def test_single_file_runtime_uses_configured_safety_settings():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    assert "self.settings = self.config.runtime_safety" in source
    assert "settings = config.runtime_safety" in source


def test_single_file_visual_calibration_round_trip(tmp_path):
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(config, module.RuntimeSafetySettings())
    source = module.HandKinematics(config, safety.safe_neutral())
    capture_complete_visual_calibration(module, source)

    path = module.save_visual_calibration(tmp_path, "single gui profile", source)
    profile = module.load_visual_calibration(path)
    target = module.HandKinematics(config, safety.safe_neutral())
    module.apply_visual_calibration(profile, target)

    assert path == tmp_path / "single_gui_profile.yaml"
    assert target.has_neutral is True
    assert target.has_range_calibration is True
    assert target.has_abd_range_calibration is True
    assert target.estimate(abducted_hand(synthetic_hand(False)))["thumb_abd"] == pytest.approx(
        source.estimate(abducted_hand(synthetic_hand(False)))["thumb_abd"]
    )


def test_single_file_gui_draws_skeleton_without_text_overlay_on_camera_frame():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    gui_source = source.split("class OrcaRealtimeGui:", 1)[1].split(
        "def build_arg_parser",
        1,
    )[0]

    assert "_draw_overlay(frame)" not in gui_source
    assert "draw_hand(" in gui_source
    assert "draw_label=False" in gui_source
    assert "cv2.putText" not in gui_source
    assert "cv2.imencode" not in gui_source
    assert "ImageTk.PhotoImage" in gui_source
    assert 'text="Camera preview"' not in gui_source


def test_single_file_gui_uses_console_dashboard_layout():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    gui_source = source.split("class OrcaRealtimeGui:", 1)[1].split(
        "def build_arg_parser",
        1,
    )[0]

    assert "_build_console_styles" in gui_source
    assert "_create_metric_card" in gui_source
    assert "_create_status_pill" in gui_source
    assert "System Snapshot" in gui_source
    assert "Run Controls" in gui_source
    assert "Joint Telemetry" in gui_source
    assert "self.preview_state_var" in gui_source
    assert "self.preview_fps_var" in gui_source
    assert "self.preview_safety_var" in gui_source


def test_single_file_gui_keeps_operator_labels_readable_in_chinese():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    gui_source = source.split("class OrcaRealtimeGui:", 1)[1].split(
        "def build_arg_parser",
        1,
    )[0]

    expected_labels = [
        "开始识别",
        "结束识别",
        "连接 OrcaHand",
        "开始映射",
        "启用机器手输出",
        "紧急停止",
        "记录张开手",
        "记录握拳",
        "保存视觉校准",
        "使用",
        "停止校准",
    ]
    for label in expected_labels:
        assert label in gui_source


def test_single_file_gui_right_control_panel_is_scrollable():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    gui_source = source.split("class OrcaRealtimeGui:", 1)[1].split(
        "def build_arg_parser",
        1,
    )[0]

    assert "tk.Canvas" in gui_source
    assert "ttk.Scrollbar" in gui_source
    assert "yscrollcommand" in gui_source
    assert "_on_panel_mousewheel" in gui_source
    assert "_queue_panel_scrollregion_update" in gui_source


def test_single_file_gui_video_preview_does_not_reflow_right_panel():
    source = SINGLE_FILE.read_text(encoding="utf-8")

    build_ui_source = source.split("def _build_ui", 1)[1].split(
        "def _update_panel_scrollregion",
        1,
    )[0]

    assert "video_frame.grid_propagate(False)" in build_ui_source
    assert "width=DEFAULT_DISPLAY_MAX_WIDTH" in build_ui_source
    assert "height=DEFAULT_DISPLAY_MAX_HEIGHT" in build_ui_source


def test_single_file_gui_coalesces_scrollregion_repaints():
    module = load_single_file_module()

    class FakeRoot:
        def __init__(self):
            self.idle_callbacks = []

        def after_idle(self, callback):
            self.idle_callbacks.append(callback)

    class FakeCanvas:
        def __init__(self):
            self.configure_calls = []

        def bbox(self, value):
            assert value == "all"
            return (0, 0, 390, 900)

        def configure(self, **kwargs):
            self.configure_calls.append(kwargs)

    class FakeGui:
        def __init__(self):
            self.root = FakeRoot()
            self._panel_canvas = FakeCanvas()
            self._panel_scrollregion_pending = False

        def _update_panel_scrollregion(self):
            module.OrcaRealtimeGui._update_panel_scrollregion(self)

    gui = FakeGui()

    module.OrcaRealtimeGui._queue_panel_scrollregion_update(gui)
    module.OrcaRealtimeGui._queue_panel_scrollregion_update(gui)

    assert len(gui.root.idle_callbacks) == 1
    assert gui._panel_canvas.configure_calls == []

    gui.root.idle_callbacks.pop()()

    assert gui._panel_scrollregion_pending is False
    assert gui._panel_canvas.configure_calls == [
        {"scrollregion": (0, 0, 390, 900)}
    ]


def test_single_file_gui_keeps_start_mapping_clickable_for_calibration_guidance():
    module = load_single_file_module()

    class FakeVar:
        def __init__(self):
            self.value = None

        def set(self, value):
            self.value = value

    class FakeButton:
        def __init__(self):
            self.state = None

        def configure(self, *, state):
            self.state = state

    class FakeController:
        connected = False

    class FakeGui:
        running = False
        hardware_allowed = False
        _background_busy = False
        state = module.RuntimeStateMachine()
        controller = FakeController()
        state_var = FakeVar()
        hardware_var = FakeVar()
        start_button = FakeButton()
        stop_button = FakeButton()
        connect_button = FakeButton()
        map_button = FakeButton()
        stop_map_button = FakeButton()
        output_button = FakeButton()
        stop_output_button = FakeButton()
        calibrate_button = FakeButton()
        stop_calibrate_button = FakeButton()

        def _mapping_calibration_ready(self):
            return False

    gui = FakeGui()

    module.OrcaRealtimeGui._update_button_states(gui)

    assert gui.map_button.state == "normal"


def test_single_file_gui_start_mapping_enables_hardware_output_when_connected():
    module = load_single_file_module()

    class FakeController:
        connected = True

    class FakeResettable:
        def __init__(self):
            self.reset_count = 0

        def reset_to_safe_neutral(self):
            self.reset_count += 1

        def reset(self):
            self.reset_count += 1

    class FakeGui:
        running = True
        hardware_allowed = True
        controller = FakeController()
        state = module.RuntimeStateMachine()
        safety = FakeResettable()
        joint_smoother = FakeResettable()

        def __init__(self):
            self.status_messages = []

        def _mapping_calibration_ready(self):
            return True

        def _set_status(self, message):
            self.status_messages.append(message)

    gui = FakeGui()

    module.OrcaRealtimeGui.start_mapping(gui)

    assert gui.state.state == module.RuntimeState.LIVE
    assert gui.safety.reset_count == 1
    assert gui.joint_smoother.reset_count == 1
    assert "Hardware output enabled" in gui.status_messages[-1]


def test_single_file_gui_stop_actions_disable_live_hardware_output():
    module = load_single_file_module()

    class FakeController:
        connected = True

        def __init__(self):
            self.stop_count = 0

        def emergency_stop(self):
            self.stop_count += 1
            self.connected = False

    class FakeResettable:
        def reset(self):
            pass

    class FakeGui:
        running = True
        cap = None
        landmarker = None
        logger = None
        latest_landmarks = None
        landmark_smoother = FakeResettable()
        joint_smoother = FakeResettable()

        def __init__(self):
            self.controller = FakeController()
            self.state = module.RuntimeStateMachine()
            self.status_messages = []

        def _release_recognition_resources(self):
            pass

        def _stop_live_hardware_output(self):
            module.OrcaRealtimeGui._stop_live_hardware_output(self)

        def _set_status(self, message):
            self.status_messages.append(message)

    for action_name in ("stop_recognition", "stop_mapping", "disable_output"):
        gui = FakeGui()
        gui.state.start_mapping()
        gui.state.enable_live()

        getattr(module.OrcaRealtimeGui, action_name)(gui)

        assert gui.controller.stop_count == 1, action_name
        assert gui.controller.connected is False, action_name


def test_single_file_emergency_stop_disconnects_and_releases_hand():
    module = load_single_file_module()

    class FakeHand:
        def __init__(self):
            self.stop_count = 0
            self.disable_count = 0
            self.disconnect_count = 0

        def stop_task(self):
            self.stop_count += 1

        def disable_torque(self):
            self.disable_count += 1

        def disconnect(self):
            self.disconnect_count += 1

    controller = module.OrcaController("config/config.yaml", live=True)
    hand = FakeHand()
    controller.hand = hand
    controller.connected = True

    controller.emergency_stop()

    assert hand.stop_count == 1
    assert hand.disable_count == 1
    assert hand.disconnect_count == 1
    assert controller.connected is False
    assert controller.hand is None


def test_single_file_disconnect_clears_hand_even_when_disconnect_fails():
    module = load_single_file_module()

    class FakeHand:
        def stop_task(self):
            pass

        def disconnect(self):
            raise RuntimeError("disconnect failed")

    controller = module.OrcaController("config/config.yaml", live=True)
    controller.hand = FakeHand()
    controller.connected = True

    with pytest.raises(RuntimeError, match="disconnect failed"):
        controller.disconnect()

    assert controller.connected is False
    assert controller.hand is None


def test_single_file_gui_skips_button_state_work_when_snapshot_is_unchanged():
    module = load_single_file_module()

    class FakeVar:
        def __init__(self):
            self.set_count = 0

        def set(self, value):
            self.set_count += 1

    class FakeButton:
        def __init__(self):
            self.configure_count = 0

        def configure(self, *, state):
            self.configure_count += 1

    class FakeController:
        connected = False

    class FakeGui:
        running = False
        hardware_allowed = False
        _background_busy = False
        state = module.RuntimeStateMachine()
        controller = FakeController()
        state_var = FakeVar()
        hardware_var = FakeVar()
        start_button = FakeButton()
        stop_button = FakeButton()
        connect_button = FakeButton()
        map_button = FakeButton()
        stop_map_button = FakeButton()
        output_button = FakeButton()
        stop_output_button = FakeButton()
        calibrate_button = FakeButton()
        stop_calibrate_button = FakeButton()

    gui = FakeGui()

    module.OrcaRealtimeGui._update_button_states(gui)
    first_configure_count = gui.start_button.configure_count
    module.OrcaRealtimeGui._update_button_states(gui)

    assert first_configure_count == 1
    assert gui.start_button.configure_count == first_configure_count
    assert gui.state_var.set_count == 1


def test_single_file_landmarker_prefers_gpu_and_falls_back_to_cpu(monkeypatch):
    module = load_single_file_module()
    calls = []

    def fake_create(model_path, max_hands, prefer_gpu):
        calls.append(prefer_gpu)
        if prefer_gpu:
            raise RuntimeError("GPU delegate failed")
        return "cpu-landmarker"

    monkeypatch.setattr(module, "_create_landmarker_with_delegate", fake_create)

    landmarker = module.create_landmarker(Path("model.task"), 1, prefer_gpu=True)

    assert landmarker == "cpu-landmarker"
    assert calls == [True, False]


def test_single_file_landmarker_uses_cpu_when_gpu_not_preferred(monkeypatch):
    module = load_single_file_module()
    calls = []

    def fake_create(model_path, max_hands, prefer_gpu):
        calls.append(prefer_gpu)
        return "cpu-landmarker"

    monkeypatch.setattr(module, "_create_landmarker_with_delegate", fake_create)

    landmarker = module.create_landmarker(Path("model.task"), 1, prefer_gpu=False)

    assert landmarker == "cpu-landmarker"
    assert calls == [False]


def test_single_file_resize_to_fill_returns_exact_display_size():
    module = load_single_file_module()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = module._resize_to_fill(frame, target_width=300, target_height=300)

    assert resized.shape == (300, 300, 3)


def test_single_file_resize_to_fill_reuses_exact_size_frame():
    module = load_single_file_module()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    resized = module._resize_to_fill(frame, target_width=1280, target_height=720)

    assert resized is frame


def test_single_file_draw_hand_uses_fast_line_type(monkeypatch):
    module = load_single_file_module()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    keypoints = np.zeros((21, 3), dtype=np.float32)
    keypoints[:, 0] = np.linspace(10, 190, 21)
    keypoints[:, 1] = np.linspace(10, 90, 21)
    scores = np.ones(21, dtype=np.float32)
    line_types = []

    def fake_line(*args):
        line_types.append(args[-1])

    def fake_circle(*args):
        line_types.append(args[-1])

    monkeypatch.setattr(module.cv2, "line", fake_line)
    monkeypatch.setattr(module.cv2, "circle", fake_circle)

    module.draw_hand(frame, keypoints, scores, "Right 0.95", draw_label=False)

    assert line_types
    assert set(line_types) == {module.DEFAULT_DRAW_LINE_TYPE}


def test_single_file_select_one_hand_rejects_wrong_handedness_for_right_hand():
    module = load_single_file_module()
    keypoints = np.stack([tracked_hand_keypoints(0), tracked_hand_keypoints(240)])
    scores = np.ones((2, 21), dtype=np.float32)

    selected = module.select_one_hand(
        keypoints,
        scores,
        ["Left 0.96", "Right 0.91"],
        locked_wrist=None,
        max_hands=1,
        preferred_handedness="right",
    )

    assert len(selected) == 1
    assert selected[0]["label"] == "Right 0.91"

    selected = module.select_one_hand(
        keypoints[:1],
        scores[:1],
        ["Left 0.96"],
        locked_wrist=None,
        max_hands=1,
        preferred_handedness="right",
    )

    assert selected == []


def test_single_file_select_one_hand_rejects_low_quality_landmarks():
    module = load_single_file_module()
    good = tracked_hand_keypoints()
    low_scores = np.full((1, 21), 0.2, dtype=np.float32)
    degenerate = np.zeros((1, 21, 3), dtype=np.float32)

    assert (
        module.select_one_hand(
            np.stack([good]),
            low_scores,
            ["Right 0.95"],
            locked_wrist=None,
            max_hands=1,
            preferred_handedness="right",
        )
        == []
    )
    assert (
        module.select_one_hand(
            degenerate,
            np.ones((1, 21), dtype=np.float32),
            ["Right 0.95"],
            locked_wrist=None,
            max_hands=1,
            preferred_handedness="right",
        )
        == []
    )


def test_single_file_configures_capture_for_bounded_realtime_input():
    module = load_single_file_module()

    class FakeCapture:
        def __init__(self):
            self.calls = []

        def set(self, prop, value):
            self.calls.append((prop, value))
            return True

    capture = FakeCapture()

    module._configure_capture(capture)

    assert module.DEFAULT_CAPTURE_WIDTH <= 640
    assert module.DEFAULT_CAPTURE_HEIGHT <= 480
    assert (module.cv2.CAP_PROP_FRAME_WIDTH, module.DEFAULT_CAPTURE_WIDTH) in capture.calls
    assert (module.cv2.CAP_PROP_FRAME_HEIGHT, module.DEFAULT_CAPTURE_HEIGHT) in capture.calls
    assert (module.cv2.CAP_PROP_FPS, module.DEFAULT_CAPTURE_FPS) in capture.calls
    assert (module.cv2.CAP_PROP_BUFFERSIZE, 1) in capture.calls


def test_single_file_configures_capture_with_requested_dimensions():
    module = load_single_file_module()

    class FakeCapture:
        def __init__(self):
            self.calls = []

        def set(self, prop, value):
            self.calls.append((prop, value))
            return True

    capture = FakeCapture()

    module._configure_capture(
        capture,
        width=800,
        height=600,
        fps=24,
        buffer_size=2,
        fourcc="MJPG",
    )

    assert (module.cv2.CAP_PROP_FRAME_WIDTH, 800) in capture.calls
    assert (module.cv2.CAP_PROP_FRAME_HEIGHT, 600) in capture.calls
    assert (module.cv2.CAP_PROP_FPS, 24) in capture.calls
    assert (module.cv2.CAP_PROP_BUFFERSIZE, 2) in capture.calls
    assert any(prop == module.cv2.CAP_PROP_FOURCC for prop, _value in capture.calls)


def test_single_file_resize_for_display_caps_tk_image_work():
    module = load_single_file_module()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    resized = module._resize_for_display(frame, max_width=640, max_height=360)

    assert resized.shape == (360, 640, 3)


def test_single_file_display_target_fills_label_without_explicit_cap():
    module = load_single_file_module()

    assert module._display_target_size(
        label_width=960,
        label_height=540,
        max_width=0,
        max_height=0,
    ) == (960, 540)


def test_single_file_display_target_respects_explicit_cap():
    module = load_single_file_module()

    assert module._display_target_size(
        label_width=960,
        label_height=540,
        max_width=640,
        max_height=360,
    ) == (640, 360)


def test_single_file_perf_telemetry_updates_less_often_than_frames():
    module = load_single_file_module()

    class FakeGui:
        _next_perf_telemetry_update = 0.0

    gui = FakeGui()

    assert module.OrcaRealtimeGui._should_update_perf_telemetry(gui, 10.0) is True
    assert module.OrcaRealtimeGui._should_update_perf_telemetry(gui, 10.1) is False
    assert module.OrcaRealtimeGui._should_update_perf_telemetry(gui, 10.6) is True


def test_single_file_rolling_fps_average_drops_startup_outlier():
    module = load_single_file_module()
    counter = module.RollingFpsCounter(window_s=0.2)

    counter.update(0.0)
    counter.update(0.5)
    _instant_fps, average_fps = counter.update(0.5 + 1.0 / 30.0)

    assert average_fps == pytest.approx(30.0, rel=0.05)


def test_single_file_gui_throttles_repeated_telemetry_updates():
    module = load_single_file_module()

    class FakeGui:
        _next_telemetry_update = 0.0

    gui = FakeGui()

    assert module.OrcaRealtimeGui._should_update_telemetry(gui, 10.0) is True
    assert module.OrcaRealtimeGui._should_update_telemetry(gui, 10.01) is False
    assert module.OrcaRealtimeGui._should_update_telemetry(gui, 10.2) is True


def test_single_file_opens_capture_with_preferred_backend(monkeypatch):
    module = load_single_file_module()
    created = []

    class FakeCapture:
        def __init__(self, *args):
            self.args = args
            self.calls = []

        def set(self, prop, value):
            self.calls.append((prop, value))
            return True

    def fake_video_capture(*args):
        capture = FakeCapture(*args)
        created.append(capture)
        return capture

    monkeypatch.setattr(module, "DEFAULT_CAMERA_BACKEND", 12345)
    monkeypatch.setattr(module.cv2, "VideoCapture", fake_video_capture)

    capture = module._open_capture(2)

    assert capture is created[0]
    assert capture.args == (2, 12345)
    assert (module.cv2.CAP_PROP_FRAME_WIDTH, module.DEFAULT_CAPTURE_WIDTH) in capture.calls


def test_single_file_tick_delay_respects_target_capture_fps():
    module = load_single_file_module()

    assert module._next_frame_delay_ms(elapsed_s=0.010, target_fps=30) == 23
    assert module._next_frame_delay_ms(elapsed_s=0.050, target_fps=30) == 1
    assert module._next_frame_delay_ms(elapsed_s=0.0, target_fps=0) == 1


def test_single_file_latest_frame_capture_keeps_only_newest_frame():
    module = load_single_file_module()

    class FakeCapture:
        def __init__(self):
            self.frames = [
                np.full((2, 2, 3), 1, dtype=np.uint8),
                np.full((2, 2, 3), 2, dtype=np.uint8),
            ]
            self.index = 0

        def read(self):
            frame = self.frames[min(self.index, len(self.frames) - 1)]
            self.index += 1
            return True, frame.copy()

    reader = module.LatestFrameCapture(FakeCapture())

    assert reader.capture_once_for_test() is True
    assert reader.capture_once_for_test() is True

    ok, frame, _read_ms = reader.latest()

    assert ok is True
    assert frame[0, 0, 0] == 2
    frame[0, 0, 0] = 99
    assert reader.latest()[1][0, 0, 0] == 2


def test_single_file_tick_uses_latest_frame_reader_without_blocking_cap_read(monkeypatch):
    module = load_single_file_module()

    class BlockingCapture:
        def read(self):
            raise AssertionError("tick should not call cap.read when reader is active")

    class FakeReader:
        def latest(self):
            return True, np.zeros((80, 120, 3), dtype=np.uint8), 0.2

    class FakeVar:
        def set(self, _value):
            pass

    class FakeRoot:
        def __init__(self):
            self.after_calls = []

        def after(self, delay, callback):
            self.after_calls.append((delay, callback))

    class FakeGui:
        def __init__(self):
            self.running = True
            self.cap = BlockingCapture()
            self.capture_reader = FakeReader()
            self.landmarker = object()
            self.start_time = time.monotonic()
            self.frame_count = 0
            self.state = module.RuntimeStateMachine()
            self.locked_wrist = None
            self.config = SimpleNamespace(hand_type="right")
            self.latest_landmarks = None
            self.hand_var = FakeVar()
            self.tracking_card_var = FakeVar()
            self.fps_var = FakeVar()
            self.preview_fps_var = FakeVar()
            self.inference_var = FakeVar()
            self.root = FakeRoot()
            self._fps_counter = module.RollingFpsCounter()
            self._next_perf_telemetry_update = 0.0
            self.button_updates = 0
            self.shown_frames = 0

        def _process_landmarks(self, _landmarks):
            pass

        def _show_frame(self, _frame):
            self.shown_frames += 1

        def _update_button_states(self):
            self.button_updates += 1

        def _should_update_perf_telemetry(self, _now_s):
            return False

        def _tick(self):
            pass

    landmarks = np.asarray([tracked_hand_keypoints(0)], dtype=np.float32)
    scores = np.ones((1, 21), dtype=np.float32)
    monkeypatch.setattr(module, "detect_frame", lambda *_args, **_kwargs: (landmarks, scores, ["Right 0.95"]))

    gui = FakeGui()

    module.OrcaRealtimeGui._tick(gui)

    assert gui.shown_frames == 1
    assert gui.button_updates == 1
    assert gui.root.after_calls


def test_single_file_gui_tick_safety_stops_live_hardware_on_runtime_error(monkeypatch):
    module = load_single_file_module()

    class FakeCapture:
        def read(self):
            return True, np.zeros((80, 120, 3), dtype=np.uint8)

    class FakeGui:
        def __init__(self):
            self.running = True
            self.cap = FakeCapture()
            self.landmarker = object()
            self.start_time = 0.0
            self.frame_count = 0
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.status_messages = []
            self.safety_stop_reason = None
            self.resources_released = False

        def _safety_stop(self, reason):
            self.safety_stop_reason = reason
            self.state.fault(f"safety stop: {reason}")

        def _set_status(self, message):
            self.status_messages.append(message)

        def _release_recognition_resources(self):
            self.resources_released = True
            self.running = False

    def raising_detect(*_args, **_kwargs):
        raise RuntimeError("landmarker exploded")

    monkeypatch.setattr(module, "detect_frame", raising_detect)
    gui = FakeGui()

    module.OrcaRealtimeGui._tick(gui)

    assert gui.safety_stop_reason == "runtime error: landmarker exploded"
    assert gui.resources_released is True
    assert gui.running is False
    assert gui.status_messages[-1] == "Runtime stopped: landmarker exploded"


def test_single_file_start_recognition_updates_ui_before_camera_open(monkeypatch):
    module = load_single_file_module()

    class FakeRoot:
        def __init__(self):
            self.updated = False

        def update_idletasks(self):
            self.updated = True

    class FakeLabel:
        def __init__(self):
            self.configures = []

        def configure(self, **kwargs):
            self.configures.append(kwargs)

    class FakeVar:
        def __init__(self):
            self.values = []

        def set(self, value):
            self.values.append(value)

    class FakeCapture:
        def isOpened(self):
            return True

    class FakeResettable:
        def reset(self):
            pass

    class FakeGui:
        def __init__(self):
            self.running = False
            self.root = FakeRoot()
            self.video_label = FakeLabel()
            self.args = module.build_arg_parser().parse_args(["--preview-only"])
            self.delegate_var = FakeVar()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.status_messages = []
            self.updated_before_camera_open = None

        def _set_status(self, message):
            self.status_messages.append(message)

        def _release_recognition_resources(self):
            pass

        def _update_button_states(self):
            pass

        def _tick(self):
            pass

    gui = FakeGui()

    def fake_open_capture(*_args, **_kwargs):
        gui.updated_before_camera_open = gui.root.updated
        return FakeCapture()

    monkeypatch.setattr(module, "ensure_model", lambda _model_path: None)
    monkeypatch.setattr(module, "create_landmarker", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "_open_capture", fake_open_capture)
    monkeypatch.setattr(module, "SessionLogger", lambda *_args, **_kwargs: object())

    module.OrcaRealtimeGui.start_recognition(gui)

    assert gui.status_messages[0] == "Starting recognition: opening camera..."
    assert gui.video_label.configures[0]["text"] == "Opening camera..."
    assert gui.updated_before_camera_open is True


def test_single_file_start_recognition_detects_two_hands_for_handedness_selection(monkeypatch):
    module = load_single_file_module()

    class FakeRoot:
        def update_idletasks(self):
            pass

    class FakeLabel:
        def configure(self, **_kwargs):
            pass

    class FakeVar:
        def set(self, _value):
            pass

    class FakeCapture:
        def isOpened(self):
            return True

    class FakeResettable:
        def reset(self):
            pass

    class FakeGui:
        def __init__(self):
            self.running = False
            self.root = FakeRoot()
            self.video_label = FakeLabel()
            self.args = module.build_arg_parser().parse_args(["--preview-only"])
            self.delegate_var = FakeVar()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.status_messages = []

        def _set_status(self, message):
            self.status_messages.append(message)

        def _release_recognition_resources(self):
            pass

        def _update_button_states(self):
            pass

        def _tick(self):
            pass

    max_hands_requested = []

    def fake_create_landmarker(_model_path, max_hands, **_kwargs):
        max_hands_requested.append(max_hands)
        return object()

    monkeypatch.setattr(module, "ensure_model", lambda _model_path: None)
    monkeypatch.setattr(module, "create_landmarker", fake_create_landmarker)
    monkeypatch.setattr(module, "_open_capture", lambda *_args, **_kwargs: FakeCapture())
    monkeypatch.setattr(module, "SessionLogger", lambda *_args, **_kwargs: object())

    module.OrcaRealtimeGui.start_recognition(FakeGui())

    assert max_hands_requested == [2]


def test_single_file_start_recognition_shows_first_camera_frame_before_landmarker(monkeypatch):
    module = load_single_file_module()
    events = []

    class FakeRoot:
        def update_idletasks(self):
            pass

    class FakeLabel:
        def configure(self, **_kwargs):
            pass

    class FakeVar:
        def set(self, _value):
            pass

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            events.append("read")
            return True, np.zeros((80, 120, 3), dtype=np.uint8)

    class FakeResettable:
        def reset(self):
            pass

    class FakeGui:
        def __init__(self):
            self.running = False
            self.root = FakeRoot()
            self.video_label = FakeLabel()
            self.args = module.build_arg_parser().parse_args(["--preview-only"])
            self.delegate_var = FakeVar()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.status_messages = []

        def _set_status(self, message):
            self.status_messages.append(message)

        def _release_recognition_resources(self):
            pass

        def _update_button_states(self):
            pass

        def _tick(self):
            pass

        def _show_frame(self, frame):
            events.append(("show", frame.shape))

    def fake_create_landmarker(*_args, **_kwargs):
        events.append("landmarker")
        return object()

    monkeypatch.setattr(module, "ensure_model", lambda _model_path: None)
    monkeypatch.setattr(module, "create_landmarker", fake_create_landmarker)
    monkeypatch.setattr(module, "_open_capture", lambda *_args, **_kwargs: FakeCapture())
    monkeypatch.setattr(module, "SessionLogger", lambda *_args, **_kwargs: object())

    module.OrcaRealtimeGui.start_recognition(FakeGui())

    assert events[0] == "read"
    assert events[1] == ("show", (80, 120, 3))
    assert events.index("landmarker") > events.index(("show", (80, 120, 3)))


def test_single_file_release_recognition_resources_stops_frame_reader_first():
    module = load_single_file_module()
    events = []

    class FakeReader:
        def stop(self):
            events.append("stop_reader")

    class FakeCapture:
        def release(self):
            events.append("release_capture")

    class FakeLandmarker:
        def close(self):
            events.append("close_landmarker")

    class FakeLogger:
        def close(self):
            events.append("close_logger")

    class FakeGui:
        def __init__(self):
            self.capture_reader = FakeReader()
            self.cap = FakeCapture()
            self.landmarker = FakeLandmarker()
            self.logger = FakeLogger()

    gui = FakeGui()

    module.OrcaRealtimeGui._release_recognition_resources(gui)

    assert events == [
        "stop_reader",
        "release_capture",
        "close_landmarker",
        "close_logger",
    ]
    assert gui.capture_reader is None
    assert gui.cap is None
    assert gui.landmarker is None
    assert gui.logger is None


def test_single_file_session_logger_batches_flushes_and_flushes_on_close(tmp_path):
    module = load_single_file_module()
    logger = module.SessionLogger(
        tmp_path,
        "session",
        flush_every=2,
        flush_interval_s=999.0,
    )
    row = {
        "timestamp": 1.23,
        "state": "live",
        "accepted": True,
        "reasons": [],
        "joints": {"index_mcp": 12.0},
        "motor_positions": {15: 1.5},
    }

    logger.write(row)

    assert "index_mcp" not in (tmp_path / "session.csv").read_text(encoding="utf-8")
    assert "index_mcp" not in (tmp_path / "session.jsonl").read_text(encoding="utf-8")

    logger.write(row)

    assert "index_mcp" in (tmp_path / "session.csv").read_text(encoding="utf-8")
    assert "index_mcp" in (tmp_path / "session.jsonl").read_text(encoding="utf-8")

    logger.write(row)
    logger.close()

    assert (tmp_path / "session.jsonl").read_text(encoding="utf-8").count("index_mcp") == 3


def test_single_file_joint_summary_formats_angles_and_safety_reasons():
    module = load_single_file_module()
    config = module.load_realtime_config("config")

    assert list(module.JOINT_DISPLAY_ORDER) == config.joint_ids

    result = module.SafetyResult(
        joints={joint: float(index) for index, joint in enumerate(module.JOINT_DISPLAY_ORDER, start=1)},
        accepted=False,
        reasons=["abrupt joint change on thumb_abd"],
        motor_positions={},
    )

    assert "thumb_abd:2.0" in module._format_joint_summary(result)
    assert "index_mcp:6.0" in module._format_joint_summary(result)
    assert "pinky_abd:14.0" in module._format_joint_summary(result)
    assert "wrist:17.0" in module._format_joint_summary(result)
    assert "abrupt joint change" in module._format_safety_summary(result)


def test_single_file_status_lines_explain_mapping_and_output_gate():
    module = load_single_file_module()
    state = module.RuntimeStateMachine()

    preview_lines = module._status_lines(state, live_allowed=True, safety_result=None)
    assert "state:preview" in preview_lines[0]
    assert "OUTPUT OFF" in preview_lines[1]

    state.start_mapping()
    state.enable_live()
    live_lines = module._status_lines(state, live_allowed=True, safety_result=None)
    assert "OUTPUT ON" in live_lines[1]


def test_single_file_mapping_helpers_reverse_flex_without_changing_config():
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(
        config,
        module.RuntimeSafetySettings(default_offset_deg=1.0),
    )
    kinematics = module.HandKinematics(config, safety.safe_neutral())
    neutral_pose = safety.safe_neutral()

    for joint in (
        "thumb_mcp",
        "thumb_pip",
        "thumb_dip",
        "index_mcp",
        "index_pip",
        "middle_mcp",
        "middle_pip",
        "ring_mcp",
        "ring_pip",
        "pinky_mcp",
        "pinky_pip",
    ):
        assert kinematics._flex_to_joint(joint, 0.0) > neutral_pose[joint], joint
        assert kinematics._flex_to_joint(joint, 100.0) < neutral_pose[joint], joint

    for joint in ("thumb_abd", "index_abd", "middle_abd", "ring_abd", "pinky_abd"):
        center = kinematics._abd_to_joint(joint, 0.0)
        command = kinematics._abd_to_joint(joint, 10.0)
        if joint == "thumb_abd":
            assert command < center, joint
        else:
            assert command < center, joint
        assert command >= safety.bounds[joint].joint_min, joint


def test_single_file_open_thumb_pose_maps_to_open_side():
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(
        config,
        module.RuntimeSafetySettings(default_offset_deg=1.0),
    )
    kinematics = module.HandKinematics(config, safety.safe_neutral())
    points = np.zeros((21, 3), dtype=float)
    points[0] = [0.0, 0.0, 0.0]
    for start, x in ((1, -1.0), (5, -0.45), (9, 0.0), (13, 0.45), (17, 0.9)):
        points[start] = [x, 1.0, 0.0]
        points[start + 1] = [x, 2.0, 0.0]
        points[start + 2] = [x, 3.0, 0.0]
        points[start + 3] = [x, 4.0, 0.0]

    joints = kinematics.estimate(points)
    neutral = safety.safe_neutral()

    assert joints["thumb_mcp"] > neutral["thumb_mcp"]
    assert joints["thumb_pip"] > neutral["thumb_pip"]


def test_single_file_thumb_abd_raw_uses_lateral_gap_when_thumb_axis_does_not_rotate():
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(
        config,
        module.RuntimeSafetySettings(default_offset_deg=1.0),
    )
    kinematics = module.HandKinematics(config, safety.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)
    spread_hand = neutral_hand.copy()
    spread_hand[[1, 2, 3, 4], 0] -= 0.45
    together_hand = neutral_hand.copy()
    together_hand[[1, 2, 3, 4], 0] += 0.55

    neutral_raw = kinematics._raw_controls(neutral_hand)["thumb_abd"]
    spread_raw = kinematics._raw_controls(spread_hand)["thumb_abd"]
    together_raw = kinematics._raw_controls(together_hand)["thumb_abd"]

    assert spread_raw > neutral_raw + 5.0
    assert together_raw < neutral_raw - 5.0


def test_single_file_thumb_abd_arc_uses_index_ray_as_together_reference():
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(
        config,
        module.RuntimeSafetySettings(default_offset_deg=1.0),
    )
    kinematics = module.HandKinematics(config, safety.safe_neutral())
    hand = synthetic_hand(curled=False)
    hand[6] = [-0.60, 2.0, 0.0]
    hand[7] = [-0.75, 3.0, 0.0]
    hand[8] = [-0.90, 4.0, 0.0]
    hand[4] = [-0.90, 4.0, 0.0]

    together_raw = kinematics._raw_controls(hand)["thumb_abd"]

    assert abs(together_raw) < 2.0


def test_single_file_detect_frame_keeps_mediapipe_z_for_3d_kinematics():
    module = load_single_file_module()

    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            landmarks = [
                SimpleNamespace(x=0.25, y=0.5, z=-0.1, visibility=0.9, presence=0.8)
                for _ in range(21)
            ]
            handedness = [[SimpleNamespace(category_name="Right", score=0.95)]]
            return SimpleNamespace(hand_landmarks=[landmarks], handedness=handedness)

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    keypoints, scores, labels = module.detect_frame(FakeLandmarker(), frame, 0)

    assert keypoints.shape == (1, 21, 3)
    assert keypoints[0, 0, 0] == pytest.approx(50.0)
    assert keypoints[0, 0, 1] == pytest.approx(50.0)
    assert keypoints[0, 0, 2] == pytest.approx(-20.0)
    assert scores[0, 0] == pytest.approx(0.9)
    assert labels == ["Right 0.95"]


def test_single_file_detect_frame_downscales_hd_input_for_inference():
    module = load_single_file_module()

    class FakeLandmarker:
        def __init__(self):
            self.image_shape = None

        def detect_for_video(self, image, timestamp_ms):
            self.image_shape = image.numpy_view().shape
            landmarks = [
                SimpleNamespace(x=0.5, y=0.25, z=-0.1, visibility=0.9, presence=0.8)
                for _ in range(21)
            ]
            handedness = [[SimpleNamespace(category_name="Right", score=0.95)]]
            return SimpleNamespace(hand_landmarks=[landmarks], handedness=handedness)

    landmarker = FakeLandmarker()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    keypoints, _scores, _labels = module.detect_frame(landmarker, frame, 0)

    assert landmarker.image_shape[:2] == (270, 480)
    assert keypoints[0, 0, 0] == pytest.approx(960.0)
    assert keypoints[0, 0, 1] == pytest.approx(270.0)
    assert keypoints[0, 0, 2] == pytest.approx(-192.0)


def test_single_file_detect_frame_skips_incomplete_mediapipe_hands():
    module = load_single_file_module()

    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            complete = [
                SimpleNamespace(x=0.25, y=0.5, z=-0.1, visibility=0.9, presence=0.8)
                for _ in range(21)
            ]
            incomplete = complete[:-1]
            handedness = [
                [SimpleNamespace(category_name="Right", score=0.95)],
                [SimpleNamespace(category_name="Left", score=0.95)],
            ]
            return SimpleNamespace(
                hand_landmarks=[incomplete, complete],
                handedness=handedness,
            )

    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    keypoints, scores, labels = module.detect_frame(FakeLandmarker(), frame, 0)

    assert keypoints.shape == (1, 21, 3)
    assert scores.shape == (1, 21)
    assert labels == ["Left 0.95"]


def test_single_file_safety_can_reset_to_neutral_for_live_ramp():
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(
        config,
        module.RuntimeSafetySettings(default_offset_deg=1.0, max_delta_deg_per_frame=5.0),
    )
    neutral = safety.safe_neutral()
    far_target = dict(neutral)
    far_target["index_mcp"] = neutral["index_mcp"] + 60.0

    for _ in range(20):
        safety.apply(far_target)

    safety.reset_to_safe_neutral()
    result = safety.apply(far_target)

    assert result.joints["index_mcp"] == neutral["index_mcp"] + 0.5


def test_single_file_rejected_safety_result_stops_live_hardware_and_warns(capsys):
    module = load_single_file_module()
    state = module.RuntimeStateMachine()
    state.start_mapping()
    state.enable_live()
    result = module.SafetyResult(
        joints={},
        accepted=False,
        reasons=["index_mcp outside safe ROM"],
        motor_positions={},
    )

    class FakeController:
        def __init__(self):
            self.stopped = False

        def emergency_stop(self):
            self.stopped = True

    controller = FakeController()

    module._react_to_safety_result(state, controller, result, live_allowed=True)

    assert state.state == module.RuntimeState.FAULT
    assert controller.stopped is True
    out = capsys.readouterr().out
    assert "\033[" in out
    assert "SAFETY STOP" in out
    assert "index_mcp outside safe ROM" in out


def test_single_file_gui_clears_preview_safety_reason_after_accepted_frame():
    module = load_single_file_module()

    class FakeSmoother:
        def update(self, value):
            return value

    class FakeKinematics:
        def estimate(self, _landmarks):
            return {"index_mcp": 0.0}

    class FakeSafety:
        def __init__(self):
            self.results = [
                module.SafetyResult(
                    joints={},
                    accepted=False,
                    reasons=["index_mcp outside safe ROM"],
                    motor_positions={},
                ),
                module.SafetyResult(
                    joints={"index_mcp": 1.0},
                    accepted=True,
                    reasons=[],
                    motor_positions={},
                ),
            ]

        def apply(self, _joints):
            return self.results.pop(0)

    class FakeVar:
        def set(self, _value):
            pass

    class FakeController:
        def send(self, _joints):
            raise AssertionError("preview mode must not send hardware commands")

    class FakeGui:
        def __init__(self):
            self.landmark_smoother = FakeSmoother()
            self.joint_smoother = FakeSmoother()
            self.kinematics = FakeKinematics()
            self.safety = FakeSafety()
            self.state = module.RuntimeStateMachine()
            self.controller = FakeController()
            self.logger = None
            self.latest_safety_result = None
            self.joint_angles_var = FakeVar()
            self.safety_var = FakeVar()
            self.safety_card_var = FakeVar()
            self.preview_safety_var = FakeVar()

        def _should_update_telemetry(self, _now):
            return True

        def _update_safety_telemetry(self, safety_result):
            module.OrcaRealtimeGui._update_safety_telemetry(self, safety_result)

        def _safety_stop(self, reason):
            raise AssertionError(f"preview mode must not safety stop: {reason}")

    gui = FakeGui()
    landmarks = np.zeros((21, 3), dtype=float)

    module.OrcaRealtimeGui._process_landmarks(gui, landmarks)
    assert gui.state.reason == "index_mcp outside safe ROM"

    module.OrcaRealtimeGui._process_landmarks(gui, landmarks)
    assert gui.state.reason == ""


def test_single_file_gui_logs_rejected_safety_frames():
    module = load_single_file_module()

    class FakeSmoother:
        def update(self, value):
            return value

    class FakeKinematics:
        def estimate(self, _landmarks):
            return {"index_mcp": 200.0}

    class FakeSafety:
        def apply(self, _joints):
            return module.SafetyResult(
                joints={"index_mcp": 1.0},
                accepted=False,
                reasons=["index_mcp outside safe ROM"],
                motor_positions={15: 0.5},
            )

    class FakeLogger:
        def __init__(self):
            self.rows = []

        def write(self, row):
            self.rows.append(row)

    class FakeVar:
        def set(self, _value):
            pass

    class FakeController:
        def send(self, _joints):
            raise AssertionError("rejected frames must not send hardware commands")

    class FakeGui:
        def __init__(self):
            self.landmark_smoother = FakeSmoother()
            self.joint_smoother = FakeSmoother()
            self.kinematics = FakeKinematics()
            self.safety = FakeSafety()
            self.state = module.RuntimeStateMachine()
            self.controller = FakeController()
            self.logger = FakeLogger()
            self.latest_safety_result = None
            self.joint_angles_var = FakeVar()
            self.safety_var = FakeVar()
            self.safety_card_var = FakeVar()
            self.preview_safety_var = FakeVar()

        def _should_update_telemetry(self, _now):
            return True

        def _update_safety_telemetry(self, safety_result):
            module.OrcaRealtimeGui._update_safety_telemetry(self, safety_result)

        def _safety_stop(self, reason):
            raise AssertionError(f"preview mode must not safety stop: {reason}")

    gui = FakeGui()

    module.OrcaRealtimeGui._process_landmarks(gui, np.zeros((21, 3), dtype=float))

    assert len(gui.logger.rows) == 1
    assert gui.logger.rows[0]["accepted"] is False
    assert gui.logger.rows[0]["reasons"] == ["index_mcp outside safe ROM"]


def test_single_file_gui_transient_no_hand_enters_tracking_lost_without_fault():
    module = load_single_file_module()

    class FakeResettable:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class FakeVar:
        def __init__(self):
            self.value = None

        def set(self, value):
            self.value = value

    class FakeGui:
        def __init__(self):
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.latest_landmarks = np.ones((21, 3), dtype=float)
            self.latest_safety_result = object()
            self.hand_var = FakeVar()
            self.tracking_card_var = FakeVar()
            self.joint_angles_var = FakeVar()
            self.safety_var = FakeVar()
            self.safety_card_var = FakeVar()
            self.preview_safety_var = FakeVar()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.safety_stop_reason = None
            self._hand_missing_since = None
            self.tracking_lost_alert_count = 0

        def _start_tracking_lost_alert(self):
            self.tracking_lost_alert_count += 1

        def _safety_stop(self, reason):
            self.safety_stop_reason = reason
            self.state.fault(f"safety stop: {reason}")

    gui = FakeGui()

    module.OrcaRealtimeGui._handle_no_hand(gui, now_s=10.0)

    assert gui.state.state == module.RuntimeState.TRACKING_LOST
    assert gui.state.reason == "no hand detected"
    assert gui.safety_stop_reason is None
    assert gui.tracking_lost_alert_count == 1
    assert gui.latest_landmarks is None
    assert gui.latest_safety_result is None
    assert gui.landmark_smoother.reset_count == 1
    assert gui.joint_smoother.reset_count == 1


def test_single_file_gui_tracking_loss_waits_two_seconds_before_safety_stop():
    module = load_single_file_module()

    class FakeResettable:
        def reset(self):
            pass

    class FakeVar:
        def set(self, _value):
            pass

    class FakeGui:
        def __init__(self):
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.latest_landmarks = None
            self.latest_safety_result = None
            self.hand_var = FakeVar()
            self.tracking_card_var = FakeVar()
            self.joint_angles_var = FakeVar()
            self.safety_var = FakeVar()
            self.safety_card_var = FakeVar()
            self.preview_safety_var = FakeVar()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self.safety_stop_reason = None
            self._hand_missing_since = None

        def _safety_stop(self, reason):
            self.safety_stop_reason = reason
            self.state.fault(f"safety stop: {reason}")

    gui = FakeGui()

    assert module.DEFAULT_TRACKING_LOST_STOP_S == 2.0
    module.OrcaRealtimeGui._handle_no_hand(gui, now_s=10.0)
    module.OrcaRealtimeGui._handle_no_hand(gui, now_s=11.99)

    assert gui.state.state == module.RuntimeState.TRACKING_LOST
    assert gui.safety_stop_reason is None

    module.OrcaRealtimeGui._handle_no_hand(gui, now_s=12.0)

    assert gui.safety_stop_reason == "no hand detected"
    assert gui.state.state == module.RuntimeState.FAULT


def test_single_file_tracking_lost_recovers_to_original_preview_state():
    module = load_single_file_module()
    state = module.RuntimeStateMachine()

    state.tracking_lost("temporary issue")
    state.recover_tracking()

    assert state.state == module.RuntimeState.PREVIEW


def test_single_file_tracking_lost_pauses_output_and_recovers_live_state():
    module = load_single_file_module()
    state = module.RuntimeStateMachine()
    state.start_mapping()
    state.enable_live()

    state.tracking_lost("no hand detected")

    assert state.state == module.RuntimeState.TRACKING_LOST
    assert state.can_send_to_hardware is False

    state.recover_tracking()

    assert state.state == module.RuntimeState.LIVE
    assert state.can_send_to_hardware is True


def test_single_file_tracking_loss_deadline_blocks_late_recovery_and_send():
    module = load_single_file_module()

    class FakeController:
        def __init__(self):
            self.sent = []

        def send(self, joints):
            self.sent.append(dict(joints))

    class FakeGui:
        def __init__(self):
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.state.tracking_lost("no hand detected")
            self._tracking_lost_deadline_s = 12.0
            self.controller = FakeController()
            self.safety_stop_reasons = []

        def _safety_stop(self, reason):
            self.safety_stop_reasons.append(reason)
            self.state.fault(f"safety stop: {reason}")

    gui = FakeGui()

    module.OrcaRealtimeGui._process_landmarks(
        gui,
        np.zeros((21, 3), dtype=float),
        now_s=12.0,
    )

    assert gui.safety_stop_reasons == ["no hand detected"]
    assert gui.state.state == module.RuntimeState.FAULT
    assert gui.controller.sent == []


def test_single_file_tracking_loss_timeout_starts_when_live_tracking_is_lost():
    module = load_single_file_module()

    class FakeResettable:
        def reset(self):
            pass

    class FakeVar:
        def set(self, _value):
            pass

    class FakeGui:
        def __init__(self):
            self.state = module.RuntimeStateMachine()
            self.latest_landmarks = None
            self.latest_safety_result = None
            self.hand_var = FakeVar()
            self.tracking_card_var = FakeVar()
            self.joint_angles_var = FakeVar()
            self.safety_var = FakeVar()
            self.safety_card_var = FakeVar()
            self.preview_safety_var = FakeVar()
            self.landmark_smoother = FakeResettable()
            self.joint_smoother = FakeResettable()
            self._hand_missing_since = None
            self._tracking_lost_deadline_s = None
            self.safety_stop_reason = None

        def _start_tracking_lost_alert(self):
            pass

        def _safety_stop(self, reason):
            self.safety_stop_reason = reason
            self.state.fault(f"safety stop: {reason}")

    gui = FakeGui()
    module.OrcaRealtimeGui._handle_no_hand(gui, now_s=10.0)
    gui.state.start_mapping()
    gui.state.enable_live()

    module.OrcaRealtimeGui._handle_no_hand(gui, now_s=20.0)

    assert gui.state.state == module.RuntimeState.TRACKING_LOST
    assert gui.safety_stop_reason is None
    assert gui._hand_missing_since == 20.0
    assert gui._tracking_lost_deadline_s == 22.0


def test_single_file_startup_ramp_uses_smaller_delta_before_normal_delta():
    module = load_single_file_module()
    config = module.load_realtime_config("config")
    safety = module.SafetyController(
        config,
        module.RuntimeSafetySettings(
            default_offset_deg=1.0,
            max_delta_deg_per_frame=5.0,
            startup_max_delta_deg_per_frame=0.5,
            startup_ramp_frames=2,
        ),
    )
    neutral = safety.safe_neutral()
    target = dict(neutral)
    target["index_mcp"] = neutral["index_mcp"] + 50.0

    safety.reset_to_safe_neutral()
    first = safety.apply(target)
    second = safety.apply(target)
    third = safety.apply(target)

    assert first.joints["index_mcp"] == neutral["index_mcp"] + 0.5
    assert second.joints["index_mcp"] == neutral["index_mcp"] + 1.0
    assert third.joints["index_mcp"] == neutral["index_mcp"] + 6.0
