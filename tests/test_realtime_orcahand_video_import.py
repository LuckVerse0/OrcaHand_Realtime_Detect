from __future__ import annotations

from pathlib import Path

import numpy as np
import realtime_orcahand as rt


def load_realtime_module():
    return rt


def fake_processed_video(module):
    return module.ProcessedVideo(
        source_path=Path("source.mp4"),
        output_path=Path("processed.mp4"),
        fps=60.0,
        width=1920,
        height=1080,
        frames=[
            module.ProcessedVideoFrame(
                frame_index=0,
                timestamp_s=0.0,
                joints={"index_mcp": 12.0},
            ),
            module.ProcessedVideoFrame(
                frame_index=1,
                timestamp_s=1.0 / 60.0,
                joints=None,
            ),
        ],
    )


def test_processed_video_frame_timestamps_are_60fps():
    module = load_realtime_module()

    frames = [
        module.ProcessedVideoFrame(frame_index=index, timestamp_s=index / 60.0, joints=None)
        for index in range(3)
    ]

    assert frames[0].timestamp_s == 0.0
    assert frames[1].timestamp_s == 1.0 / 60.0
    assert frames[2].timestamp_s == 2.0 / 60.0


def test_letterbox_to_1080p_preserves_aspect_ratio_for_portrait_frame():
    module = load_realtime_module()
    frame = np.full((100, 50, 3), 255, dtype=np.uint8)

    output = module._letterbox_to_size(frame, target_width=1920, target_height=1080)

    assert output.shape == (1080, 1920, 3)
    content_columns = np.where(output.max(axis=(0, 2)) > 0)[0]
    assert content_columns[0] == 690
    assert content_columns[-1] == 1229


def test_video_import_button_allows_idle_disconnected_hardware():
    module = load_realtime_module()

    class FakeVar:
        def set(self, _value):
            pass

    class FakeButton:
        def __init__(self):
            self.state = None

        def configure(self, *, state):
            self.state = state

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
    assert gui.import_video_button.state == "normal"

    gui.controller.connected = True
    module.OrcaRealtimeGui._update_button_states(gui)
    assert gui.import_video_button.state == "normal"

    gui._video_processing_busy = True
    module.OrcaRealtimeGui._update_button_states(gui)
    assert gui.import_video_button.state == "disabled"


def test_video_ready_requests_connection_before_playback():
    module = load_realtime_module()

    class FakeController:
        connected = False

    class FakeGui:
        def __init__(self):
            self.controller = FakeController()
            self.processed_video = None
            self._play_video_after_connect = False
            self.background_calls = []
            self.playback_started = False

        def _run_background(self, start_message, worker):
            self.background_calls.append((start_message, worker))

        def _connect_hardware_worker(self):
            return "connected"

        def start_processed_video_playback(self):
            self.playback_started = True

    gui = FakeGui()
    video = fake_processed_video(module)

    module.OrcaRealtimeGui._handle_processed_video_ready(gui, video)

    assert gui.processed_video is video
    assert gui._play_video_after_connect is True
    assert gui.background_calls[0][0] == "Motion ready. Connecting OrcaHand..."
    assert not gui.playback_started


def test_pending_video_playback_starts_after_connect_done():
    module = load_realtime_module()

    class FakeGui:
        def __init__(self):
            self._play_video_after_connect = True
            self.reloaded = False
            self.status_messages = []
            self.playback_started = False

        def _reload_runtime_config(self):
            self.reloaded = True

        def _set_status(self, message):
            self.status_messages.append(message)

        def start_processed_video_playback(self):
            self.playback_started = True

    gui = FakeGui()

    module.OrcaRealtimeGui._handle_background_done(
        gui,
        "OrcaHand connected and moved to neutral.",
    )

    assert gui.reloaded is True
    assert gui._play_video_after_connect is False
    assert gui.playback_started is True


def test_video_playback_enters_live_state_without_exposing_file_path(monkeypatch):
    module = load_realtime_module()

    class FakeCapture:
        def __init__(self, _path):
            self.released = False

        def isOpened(self):
            return True

        def release(self):
            self.released = True

    class FakeController:
        connected = True

    class FakeRoot:
        def after_cancel(self, _after_id):
            pass

    class FakeSafety:
        def reset_to_safe_neutral(self):
            pass

    class FakeSmoother:
        def reset(self):
            pass

    class FakeGui:
        def __init__(self):
            self.processed_video = fake_processed_video(module)
            self.hardware_allowed = True
            self.controller = FakeController()
            self.state = module.RuntimeStateMachine()
            self.root = FakeRoot()
            self.video_playback_cap = None
            self._video_after_id = None
            self._video_processing_busy = False
            self._video_playing = False
            self.safety = FakeSafety()
            self.joint_smoother = FakeSmoother()
            self.status_messages = []
            self.play_frame_called = False

        def _set_status(self, message):
            self.status_messages.append(message)

        def _update_button_states(self):
            pass

        def stop_processed_video_playback(self, **kwargs):
            module.OrcaRealtimeGui.stop_processed_video_playback(self, **kwargs)

        def _play_processed_video_frame(self):
            self.play_frame_called = True

    monkeypatch.setattr(module.cv2, "VideoCapture", FakeCapture)
    gui = FakeGui()

    module.OrcaRealtimeGui.start_processed_video_playback(gui)

    assert gui.state.state == module.RuntimeState.LIVE
    assert gui.play_frame_called is True
    assert gui.status_messages == ["Motion playback started."]
    assert "processed" not in gui.status_messages[0].lower()
    assert "processed.mp4" not in gui.status_messages[0]


def test_video_playback_sends_cached_joints_for_frame():
    module = load_realtime_module()

    class FakeController:
        connected = True

        def __init__(self):
            self.sent = []

        def send(self, joints):
            self.sent.append(dict(joints))

    class FakeVar:
        def __init__(self):
            self.value = None

        def set(self, value):
            self.value = value

    class FakeGui:
        hardware_allowed = True

        def __init__(self):
            self.controller = FakeController()
            self.joint_angles_var = FakeVar()
            self.safety_var = FakeVar()
            self.safety_card_var = FakeVar()
            self.preview_safety_var = FakeVar()
            self.state = module.RuntimeStateMachine()
            self.state.start_mapping()
            self.state.enable_live()
            self.processed_video = fake_processed_video(module)

    gui = FakeGui()

    assert module.OrcaRealtimeGui._send_processed_video_command(gui, 0) is True
    assert module.OrcaRealtimeGui._send_processed_video_command(gui, 1) is False
    gui.state.disable_live()
    assert module.OrcaRealtimeGui._send_processed_video_command(gui, 0) is False

    assert gui.controller.sent == [{"index_mcp": 12.0}]
    assert "index_mcp" in gui.joint_angles_var.value


def test_process_video_to_1080p60_writes_letterboxed_frames(monkeypatch, tmp_path):
    module = load_realtime_module()
    written_frames = []

    class FakeCapture:
        def __init__(self, _path):
            self.frames = [
                np.full((40, 20, 3), 200, dtype=np.uint8),
                np.full((40, 20, 3), 100, dtype=np.uint8),
            ]

        def isOpened(self):
            return True

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def get(self, prop):
            if prop == module.cv2.CAP_PROP_FRAME_COUNT:
                return 2
            return 0

        def release(self):
            pass

    class FakeWriter:
        def __init__(self, *_args):
            pass

        def isOpened(self):
            return True

        def write(self, frame):
            written_frames.append(frame.copy())

        def release(self):
            pass

    class FakeLandmarker:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(module, "ensure_model", lambda _path: None)
    monkeypatch.setattr(module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(module.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(module, "create_landmarker", lambda *_args, **_kwargs: FakeLandmarker())
    monkeypatch.setattr(
        module,
        "detect_frame",
        lambda *_args, **_kwargs: (
            np.empty((0, 21, 3), dtype=np.float32),
            np.empty((0, 21), dtype=np.float32),
            [],
        ),
    )
    config = module.load_realtime_config("config")

    processed = module.process_video_to_1080p60(
        tmp_path / "source.mp4",
        output_dir=tmp_path,
        config=config,
        settings=config.runtime_safety,
    )

    assert processed.fps == 60.0
    assert processed.width == 1920
    assert processed.height == 1080
    assert processed.frame_count == 2
    assert [frame.shape for frame in written_frames] == [(1080, 1920, 3), (1080, 1920, 3)]
    assert all(frame.joints is None for frame in processed.frames)


def test_process_video_to_1080p60_preserves_duration_for_30fps_source(monkeypatch, tmp_path):
    module = load_realtime_module()
    written_frames = []

    class FakeCapture:
        def __init__(self, _path):
            self.frames = [
                np.full((20, 20, 3), 200, dtype=np.uint8),
                np.full((20, 20, 3), 100, dtype=np.uint8),
            ]

        def isOpened(self):
            return True

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def get(self, prop):
            if prop == module.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == module.cv2.CAP_PROP_FRAME_COUNT:
                return 2
            return 0

        def release(self):
            pass

    class FakeWriter:
        def __init__(self, *_args):
            pass

        def isOpened(self):
            return True

        def write(self, frame):
            written_frames.append(frame.copy())

        def release(self):
            pass

    class FakeLandmarker:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(module, "ensure_model", lambda _path: None)
    monkeypatch.setattr(module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(module.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(module, "create_landmarker", lambda *_args, **_kwargs: FakeLandmarker())
    monkeypatch.setattr(
        module,
        "detect_frame",
        lambda *_args, **_kwargs: (
            np.empty((0, 21, 3), dtype=np.float32),
            np.empty((0, 21), dtype=np.float32),
            [],
        ),
    )
    config = module.load_realtime_config("config")

    processed = module.process_video_to_1080p60(
        tmp_path / "source.mp4",
        output_dir=tmp_path,
        config=config,
        settings=config.runtime_safety,
    )

    assert processed.fps == 60.0
    assert processed.frame_count == 4
    assert len(written_frames) == 4
    assert [frame.timestamp_s for frame in processed.frames] == [
        0.0,
        1.0 / 60.0,
        2.0 / 60.0,
        3.0 / 60.0,
    ]
