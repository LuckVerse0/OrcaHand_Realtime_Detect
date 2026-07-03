from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.mediapipe_hand as mediapipe_hand
from realtime_orcahand_control import DEFAULT_ORCA_CORE_ROOT, build_arg_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_entrypoint_defaults_to_dry_run_runtime_config():
    args = build_arg_parser().parse_args([])

    assert args.live is False
    assert args.config_dir == Path("config")
    assert args.orca_core_root == DEFAULT_ORCA_CORE_ROOT


def test_root_directory_keeps_tools_and_large_models_out_of_top_level():
    root_names = {path.name for path in PROJECT_ROOT.iterdir()}

    assert "tools" in root_names
    assert "mediapipe_hand.py" not in root_names
    assert "prepare_models.py" not in root_names
    assert "rtmpose_hand_cuda.py" not in root_names
    assert "vitpose_hand_cuda.py" not in root_names
    assert "vitpose-s-wholebody.onnx" not in root_names
    assert (
        "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
        not in root_names
    )
    assert (PROJECT_ROOT / "tools" / "mediapipe_hand.py").exists()


def test_entrypoint_live_flag_is_explicit():
    args = build_arg_parser().parse_args(["--live"])

    assert args.live is True


def test_entrypoint_exposes_orca_core_lifecycle_options():
    args = build_arg_parser().parse_args([
        "--mock-orca",
        "--force-calibrate",
        "--skip-neutral-on-connect",
        "--send-steps",
        "3",
        "--send-step-size",
        "0.02",
        "--startup-ramp-frames",
        "12",
        "--startup-max-delta",
        "0.5",
    ])

    assert args.mock_orca is True
    assert args.force_calibrate is True
    assert args.skip_neutral_on_connect is True
    assert args.send_steps == 3
    assert args.send_step_size == 0.02
    assert args.startup_ramp_frames == 12
    assert args.startup_max_delta == 0.5


def test_entrypoint_defaults_to_realtime_performance_budget():
    args = build_arg_parser().parse_args([])

    assert args.capture_width == 640
    assert args.capture_height == 480
    assert args.capture_fps == 30
    assert args.capture_buffer_size == 1
    assert args.inference_width <= 480
    assert args.max_detected_hands == 2


def test_mediapipe_tool_detect_frame_downscales_and_skips_incomplete_hands():
    class FakeLandmarker:
        def __init__(self):
            self.image_shape = None

        def detect_for_video(self, image, timestamp_ms):
            self.image_shape = image.numpy_view().shape
            complete = [
                SimpleNamespace(x=0.5, y=0.25, z=-0.1, visibility=0.9, presence=0.8)
                for _ in range(21)
            ]
            incomplete = complete[:-1]
            return SimpleNamespace(
                hand_landmarks=[incomplete, complete],
                handedness=[
                    [SimpleNamespace(category_name="Right", score=0.95)],
                    [SimpleNamespace(category_name="Left", score=0.95)],
                ],
            )

    landmarker = FakeLandmarker()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    keypoints, scores, labels = mediapipe_hand.detect_frame(landmarker, frame, 0)

    assert landmarker.image_shape[:2] == (270, 480)
    assert keypoints.shape == (1, 21, 3)
    assert scores.shape == (1, 21)
    assert labels == ["Left 0.95"]
    assert keypoints[0, 0, 0] == pytest.approx(960.0)
    assert keypoints[0, 0, 2] == pytest.approx(-192.0)
