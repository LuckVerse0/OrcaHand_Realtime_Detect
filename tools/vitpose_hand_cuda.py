from __future__ import annotations

import argparse
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
from rtmlib import RTMDet, ViTPose


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
POSE_MODEL = MODEL_DIR / "vitpose-s-wholebody.onnx"

BACKEND = "onnxruntime"
DEVICE = "cuda"
PREFER_TENSORRT = True
TENSORRT_FP16 = True
TENSORRT_CACHE_DIR = PROJECT_ROOT / "models" / "onnxruntime_tensorrt_cache"
WINDOW_NAME = "ViTPose-WholeBody Hand"
CAMERA_INDEX = 0
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAPTURE_BUFFER_SIZE = 1
MAX_HANDS_TO_DRAW = 1
MIRROR_INPUT = False
DRAW_BOXES = True
KEYPOINT_SCORE_THRESHOLD = 0.2

DET_INPUT_SIZE = (320, 320)
POSE_INPUT_SIZE = (192, 256)

LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)

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


class ViTPoseHand:
    def __init__(
        self,
        det_model: str,
        pose_model: str,
        backend: str,
        device: str,
    ) -> None:
        self.det_model = RTMDet(
            det_model,
            model_input_size=DET_INPUT_SIZE,
            backend=backend,
            device=device,
        )
        self.pose_model = ViTPose(
            pose_model,
            model_input_size=POSE_INPUT_SIZE,
            to_openpose=False,
            backend=backend,
            device=device,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        bboxes = np.asarray(self.det_model(image), dtype=np.float32)
        if len(bboxes) == 0:
            return empty_hand_result()

        keypoints133, scores133 = self.pose_model(image, bboxes=bboxes)

        hand_keypoints = []
        hand_scores = []
        hand_sides = []
        for keypoints, scores, bbox in zip(keypoints133, scores133, bboxes):
            kpts, kpt_scores, side = choose_hand_slice(keypoints, scores, bbox)
            hand_keypoints.append(kpts)
            hand_scores.append(kpt_scores)
            hand_sides.append(side)

        return (
            np.stack(hand_keypoints).astype(np.float32),
            np.stack(hand_scores).astype(np.float32),
            bboxes,
            hand_sides,
        )


def empty_hand_result() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    return (
        np.empty((0, 21, 2), dtype=np.float32),
        np.empty((0, 21), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        [],
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


def build_onnxruntime_providers(
    *,
    device: str,
    prefer_tensorrt: bool,
    tensorrt_fp16: bool,
) -> list[Any]:
    available = ort.get_available_providers()
    providers: list[Any] = []

    if prefer_tensorrt:
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
                            "trt_fp16_enable": "1" if tensorrt_fp16 else "0",
                            "trt_engine_cache_enable": "1",
                            "trt_engine_cache_path": str(TENSORRT_CACHE_DIR),
                        },
                    )
                )
            else:
                print("TensorRT requested but nvinfer_10.dll is not loadable.")
                print(f"TensorRT load check: {load_error}")

    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        if ":" in device:
            device_id = int(device.split(":", 1)[1])
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


def choose_hand_slice(
    keypoints133: np.ndarray,
    scores133: np.ndarray,
    bbox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    left = (
        keypoints133[LEFT_HAND_SLICE],
        scores133[LEFT_HAND_SLICE],
        "L",
    )
    right = (
        keypoints133[RIGHT_HAND_SLICE],
        scores133[RIGHT_HAND_SLICE],
        "R",
    )

    candidates = [left, right]
    candidates.sort(
        key=lambda candidate: hand_slice_score(
            candidate[0],
            candidate[1],
            bbox,
        ),
        reverse=True,
    )
    return candidates[0]


def hand_slice_score(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: np.ndarray,
) -> float:
    x1, y1, x2, y2 = expand_bbox(bbox, ratio=0.15)
    inside = (
        (keypoints[:, 0] >= x1)
        & (keypoints[:, 0] <= x2)
        & (keypoints[:, 1] >= y1)
        & (keypoints[:, 1] <= y2)
    )
    mean_score = float(np.mean(scores))
    inside_ratio = float(np.mean(inside))
    return mean_score * (0.5 + inside_ratio)


def expand_bbox(bbox: np.ndarray, ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox.astype(float)
    width = x2 - x1
    height = y2 - y1
    pad_x = width * ratio
    pad_y = height * ratio
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def create_hand_model(
    *,
    device: str,
    prefer_tensorrt: bool,
    tensorrt_fp16: bool,
) -> tuple[ViTPoseHand, list[Any]]:
    providers = build_onnxruntime_providers(
        device=device,
        prefer_tensorrt=prefer_tensorrt,
        tensorrt_fp16=tensorrt_fp16,
    )
    with override_onnxruntime_providers(providers):
        hand_model = ViTPoseHand(
            det_model=str(DET_MODEL),
            pose_model=str(POSE_MODEL),
            backend=BACKEND,
            device=device,
        )

    return hand_model, providers


def draw_hand(frame, keypoints: np.ndarray, scores: np.ndarray) -> None:
    visible = scores > KEYPOINT_SCORE_THRESHOLD

    for start, end in HAND_CONNECTIONS:
        if visible[start] and visible[end]:
            p1 = tuple(keypoints[start, :2].astype(int))
            p2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(frame, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)

    for point, is_visible in zip(keypoints[:, :2], visible):
        if is_visible:
            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 0), -1)


def draw_bbox(frame, bbox: np.ndarray, label: str) -> None:
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )


def select_one_hand(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bboxes: np.ndarray,
    sides: list[str],
    locked_wrist,
    max_hands: int,
):
    candidates = []
    for index, (hand_keypoints, hand_scores, bbox, side) in enumerate(
        zip(keypoints, scores, bboxes, sides)
    ):
        score = float(np.mean(hand_scores))
        wrist = tuple(hand_keypoints[0, :2].astype(float))
        candidates.append(
            {
                "index": index,
                "keypoints": hand_keypoints,
                "scores": hand_scores,
                "bbox": bbox,
                "side": side,
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

    return candidates[:max_hands]


def active_runtime_label(hand: ViTPoseHand) -> str:
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


def print_runtime_info(hand: ViTPoseHand, requested_providers: list[Any]) -> None:
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    print("Requested providers:", [provider_name(p) for p in requested_providers])
    print("Detector providers:", hand.det_model.session.get_providers())
    print("Pose providers:", hand.pose_model.session.get_providers())
    if "TensorrtExecutionProvider" in [provider_name(p) for p in requested_providers]:
        if active_runtime_label(hand) != "TensorRT":
            print("TensorRT is not active; running with the best available fallback.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTMDet-Hand + ViTPose-WholeBody hand keypoint webcam test."
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--max-hands", type=int, default=MAX_HANDS_TO_DRAW)
    parser.add_argument("--mirror", action="store_true", default=MIRROR_INPUT)
    parser.add_argument("--no-boxes", action="store_true")
    parser.add_argument("--no-tensorrt", action="store_true")
    parser.add_argument("--fp32-trt", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save-frame", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cpu" if args.cpu else DEVICE
    prefer_tensorrt = PREFER_TENSORRT and not args.no_tensorrt and not args.cpu
    tensorrt_fp16 = TENSORRT_FP16 and not args.fp32_trt

    if not DET_MODEL.exists() or not POSE_MODEL.exists():
        print("Missing local model files.")
        print(f"Detector: {DET_MODEL}")
        print(f"Pose:     {POSE_MODEL}")
        return 1

    hand_model, requested_providers = create_hand_model(
        device=device,
        prefer_tensorrt=prefer_tensorrt,
        tensorrt_fp16=tensorrt_fp16,
    )
    runtime_label = active_runtime_label(hand_model)
    print_runtime_info(hand_model, requested_providers)

    if args.check:
        return 0

    cap = cv2.VideoCapture(args.camera)
    configure_capture(cap)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    if args.once:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("Could not read from camera.")
            return 1

        if args.mirror:
            frame = cv2.flip(frame, 1)

        inference_start = time.monotonic()
        keypoints, scores, bboxes, sides = hand_model(frame)
        elapsed = max(time.monotonic() - inference_start, 0.001)
        selected_hands = select_one_hand(
            keypoints,
            scores,
            bboxes,
            sides,
            None,
            args.max_hands,
        )

        for hand in selected_hands:
            if DRAW_BOXES and not args.no_boxes:
                draw_bbox(
                    frame,
                    hand["bbox"],
                    f"{hand['side']} {hand['score']:.2f}",
                )
            draw_hand(frame, hand["keypoints"], hand["scores"])

        print(f"Detected hand boxes: {len(bboxes)}")
        print(f"Selected hands: {len(selected_hands)}")
        print(f"Inference time: {elapsed * 1000:.1f} ms")
        for hand in selected_hands:
            print(
                f"Hand {hand['index']}: side={hand['side']} "
                f"score={hand['score']:.3f} bbox={hand['bbox'].round(1).tolist()}"
            )

        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_frame), frame)
            print(f"Saved frame: {args.save_frame}")
        return 0

    frame_count = 0
    start_time = time.monotonic()
    locked_wrist = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from camera.")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        keypoints, scores, bboxes, sides = hand_model(frame)
        selected_hands = select_one_hand(
            keypoints,
            scores,
            bboxes,
            sides,
            locked_wrist,
            args.max_hands,
        )

        for hand in selected_hands:
            locked_wrist = hand["wrist"]
            if DRAW_BOXES and not args.no_boxes:
                draw_bbox(
                    frame,
                    hand["bbox"],
                    f"{hand['side']} {hand['score']:.2f}",
                )
            draw_hand(frame, hand["keypoints"], hand["scores"])

        frame_count += 1
        elapsed = max(time.monotonic() - start_time, 0.001)
        fps = frame_count / elapsed
        status = (
            f"ViTPose-Hand {runtime_label} hands:{len(selected_hands)}/{args.max_hands} "
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
