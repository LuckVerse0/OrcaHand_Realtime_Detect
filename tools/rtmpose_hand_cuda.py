from __future__ import annotations

import ctypes
import os
import site
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import Hand


DLL_DIRECTORY_HANDLES = []


def add_nvidia_dll_directories() -> None:
    for site_packages in site.getsitepackages():
        nvidia_root = Path(site_packages) / "nvidia"
        if not nvidia_root.exists():
            continue

        for bin_dir in nvidia_root.glob("*/bin"):
            if bin_dir.exists():
                bin_path = str(bin_dir)
                DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_path))
                if bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]


add_nvidia_dll_directories()
ort.preload_dlls(directory="")
ort.set_default_logger_severity(3)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "models" / "rtmlib"))
MODEL_DIR = PROJECT_ROOT / "models" / "rtmlib" / "hub" / "checkpoints"
DET_MODEL = MODEL_DIR / "rtmdet_nano_8xb32-300e_hand-267f9c8f.onnx"
POSE_MODEL = (
    MODEL_DIR
    / "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx"
)

CAMERA_INDEX = 0
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAPTURE_BUFFER_SIZE = 1
BACKEND = "onnxruntime"
DEVICE = "cuda"
PREFER_TENSORRT = True
TENSORRT_FP16 = True
TENSORRT_CACHE_DIR = PROJECT_ROOT / "models" / "onnxruntime_tensorrt_cache"
WINDOW_NAME = "RTMPose-Hand"
MAX_HANDS_TO_DRAW = 1
MIRROR_INPUT = False

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def windows_dll_is_loadable(dll_name: str) -> tuple[bool, str | None]:
    if os.name != "nt":
        return True, None

    try:
        ctypes.WinDLL(dll_name)
    except OSError as exc:
        return False, str(exc)

    return True, None


def provider_name(provider: Any) -> str:
    if isinstance(provider, tuple):
        return str(provider[0])
    return str(provider)


def build_onnxruntime_providers() -> list[Any]:
    available = ort.get_available_providers()
    providers: list[Any] = []

    if PREFER_TENSORRT:
        if "TensorrtExecutionProvider" not in available:
            print("TensorRT requested but TensorrtExecutionProvider is unavailable.")
        else:
            can_load_tensorrt, load_error = windows_dll_is_loadable("nvinfer_10.dll")
            if can_load_tensorrt:
                TENSORRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": "1" if TENSORRT_FP16 else "0",
                            "trt_engine_cache_enable": "1",
                            "trt_engine_cache_path": str(TENSORRT_CACHE_DIR),
                        },
                    )
                )
            else:
                print("TensorRT requested but nvinfer_10.dll is not loadable.")
                print(f"TensorRT load check: {load_error}")

    if DEVICE.startswith("cuda") and "CUDAExecutionProvider" in available:
        if ":" in DEVICE:
            device_id = int(DEVICE.split(":", 1)[1])
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        else:
            providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


@contextmanager
def override_onnxruntime_providers(providers: list[Any]):
    if BACKEND != "onnxruntime":
        yield
        return

    original_inference_session = ort.InferenceSession

    def create_inference_session(*args, **kwargs):
        kwargs["providers"] = providers
        return original_inference_session(*args, **kwargs)

    ort.InferenceSession = create_inference_session
    try:
        yield
    finally:
        ort.InferenceSession = original_inference_session


def create_hand_model() -> tuple[Hand, list[Any]]:
    providers = build_onnxruntime_providers()
    with override_onnxruntime_providers(providers):
        hand_model = Hand(
            det=str(DET_MODEL),
            pose=str(POSE_MODEL),
            backend=BACKEND,
            device=DEVICE,
            to_openpose=False,
        )

    return hand_model, providers


def draw_hand(frame, keypoints: np.ndarray, scores: np.ndarray) -> None:
    visible = scores > 0.35

    for start, end in HAND_CONNECTIONS:
        if visible[start] and visible[end]:
            p1 = tuple(keypoints[start, :2].astype(int))
            p2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(frame, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)

    for point, is_visible in zip(keypoints[:, :2], visible):
        if is_visible:
            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 0), -1)


def select_one_hand(keypoints: np.ndarray, scores: np.ndarray, locked_wrist):
    candidates = []
    for hand_keypoints, hand_scores in zip(keypoints, scores):
        score = float(np.mean(hand_scores))
        wrist = tuple(hand_keypoints[0, :2].astype(float))
        candidates.append(
            {
                "keypoints": hand_keypoints,
                "scores": hand_scores,
                "score": score,
                "wrist": wrist,
            }
        )

    if locked_wrist and candidates:
        candidates.sort(
            key=lambda hand: (hand["wrist"][0] - locked_wrist[0]) ** 2
            + (hand["wrist"][1] - locked_wrist[1]) ** 2
        )
    else:
        candidates.sort(key=lambda hand: hand["score"], reverse=True)

    return candidates[:MAX_HANDS_TO_DRAW]


def active_runtime_label(hand: Hand) -> str:
    pose_providers = hand.pose_model.session.get_providers()
    if "TensorrtExecutionProvider" in pose_providers:
        return "TensorRT"
    if "CUDAExecutionProvider" in pose_providers:
        return "CUDA"
    return "CPU"


def configure_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DEFAULT_CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, DEFAULT_CAPTURE_BUFFER_SIZE)


def print_runtime_info(hand: Hand, requested_providers: list[Any]) -> None:
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    print("Requested providers:", [provider_name(p) for p in requested_providers])
    print("Detector providers:", hand.det_model.session.get_providers())
    print("Pose providers:", hand.pose_model.session.get_providers())
    if PREFER_TENSORRT and active_runtime_label(hand) != "TensorRT":
        print("TensorRT is not active; running with the best available fallback.")


def main() -> int:
    if not DET_MODEL.exists() or not POSE_MODEL.exists():
        print("Missing local RTMPose-Hand model files.")
        print(f"Detector: {DET_MODEL}")
        print(f"Pose:     {POSE_MODEL}")
        print("Run: .\\.venv\\Scripts\\python.exe tools\\prepare_models.py")
        return 1

    hand_model, requested_providers = create_hand_model()
    runtime_label = active_runtime_label(hand_model)
    print_runtime_info(hand_model, requested_providers)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    configure_capture(cap)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return 1

    frame_count = 0
    start_time = time.monotonic()
    locked_wrist = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from camera.")
            break

        if MIRROR_INPUT:
            frame = cv2.flip(frame, 1)

        keypoints, scores = hand_model(frame)
        selected_hands = select_one_hand(keypoints, scores, locked_wrist)

        for hand in selected_hands:
            locked_wrist = hand["wrist"]
            draw_hand(frame, hand["keypoints"], hand["scores"])

        frame_count += 1
        elapsed = max(time.monotonic() - start_time, 0.001)
        fps = frame_count / elapsed
        status = (
            f"RTMPose-Hand {runtime_label} hands:{len(selected_hands)}/{MAX_HANDS_TO_DRAW} "
            f"FPS:{fps:.1f}"
        )
        cv2.putText(
            frame,
            status,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if selected_hands else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        window_closed = cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE,
        ) < 1
        if window_closed or key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
