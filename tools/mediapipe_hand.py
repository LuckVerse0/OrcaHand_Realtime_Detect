from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

CAMERA_INDEX = 0
WINDOW_NAME = "MediaPipe Hands"
MAX_HANDS_TO_DRAW = 1
MIRROR_INPUT = False
KEYPOINT_SCORE_THRESHOLD = 0.0
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAPTURE_BUFFER_SIZE = 1
DEFAULT_INFERENCE_MAX_WIDTH = 480
MIN_HANDEDNESS_SCORE = 0.5

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


def download_model(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(model_path.suffix + ".part")

    print(f"Downloading MediaPipe hand model to {model_path} ...")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with temp_path.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(f"\r{percent:5.1f}%", end="")
        print()

    temp_path.replace(model_path)


def ensure_model(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return
    download_model(model_path)


def create_landmarker(
    model_path: Path,
    max_hands: int,
) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def configure_capture(
    cap: cv2.VideoCapture,
    *,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    fps: int = DEFAULT_CAPTURE_FPS,
    buffer_size: int = DEFAULT_CAPTURE_BUFFER_SIZE,
) -> None:
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    if buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))


def _resize_for_inference(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = float(max_width) / float(width)
    target_size = (int(max_width), max(1, int(round(height * scale))))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def frame_to_mediapipe_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_frame(
    landmarker: vision.HandLandmarker,
    frame: np.ndarray,
    timestamp_ms: int,
    *,
    max_inference_width: int = DEFAULT_INFERENCE_MAX_WIDTH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    inference_frame = _resize_for_inference(frame, max_inference_width)
    image = frame_to_mediapipe_image(inference_frame)
    result = landmarker.detect_for_video(image, timestamp_ms)

    frame_h, frame_w = frame.shape[:2]
    handedness_groups = getattr(result, "handedness", [])
    valid_hands = []
    for hand_index, landmarks in enumerate(result.hand_landmarks):
        if len(landmarks) != 21:
            continue
        handedness_group = (
            handedness_groups[hand_index]
            if hand_index < len(handedness_groups)
            else []
        )
        valid_hands.append((landmarks, handedness_group))

    hand_count = len(valid_hands)
    if hand_count <= 0:
        return (
            np.empty((0, 21, 3), dtype=np.float32),
            np.empty((0, 21), dtype=np.float32),
            [],
        )

    keypoints = np.empty((hand_count, 21, 3), dtype=np.float32)
    scores = np.empty((hand_count, 21), dtype=np.float32)
    handedness = []

    for hand_index, (landmarks, handedness_group) in enumerate(valid_hands):
        hand_keypoints = keypoints[hand_index]
        hand_scores = scores[hand_index]
        for point_index, landmark in enumerate(landmarks):
            hand_keypoints[point_index, 0] = landmark.x * frame_w
            hand_keypoints[point_index, 1] = landmark.y * frame_h
            hand_keypoints[point_index, 2] = landmark.z * frame_w
            hand_scores[point_index] = landmark_score(landmark)

        label = handedness_group[0].category_name if handedness_group else "Unknown"
        score = handedness_group[0].score if handedness_group else 0.0
        handedness.append(f"{label} {score:.2f}")

    return keypoints, scores, handedness


def landmark_score(landmark) -> float:
    values = []
    for attribute in ("visibility", "presence"):
        value = getattr(landmark, attribute, None)
        if value is not None:
            values.append(float(value))
    return max(values) if values else 1.0


def draw_hand(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    label: str,
) -> None:
    visible = scores >= KEYPOINT_SCORE_THRESHOLD

    for start, end in HAND_CONNECTIONS:
        if visible[start] and visible[end]:
            p1 = tuple(keypoints[start, :2].astype(int))
            p2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(frame, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)

    for point, is_visible in zip(keypoints[:, :2], visible):
        if is_visible:
            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 0), -1)

    wrist = keypoints[0, :2].astype(int)
    cv2.putText(
        frame,
        label,
        (int(wrist[0]) + 8, max(int(wrist[1]) - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )


def select_one_hand(
    keypoints: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    locked_wrist,
    max_hands: int,
    preferred_handedness: str | None = None,
):
    candidates = []
    for index, (hand_keypoints, hand_scores, label) in enumerate(
        zip(keypoints, scores, labels)
    ):
        handedness, handedness_score = _parse_handedness_label(label)
        score = float(np.mean(hand_scores))
        wrist = tuple(hand_keypoints[0, :2].astype(float))
        candidates.append(
            {
                "index": index,
                "keypoints": hand_keypoints,
                "scores": hand_scores,
                "label": label,
                "handedness": handedness,
                "handedness_score": handedness_score,
                "score": score,
                "wrist": wrist,
            }
        )

    preferred = _normalize_handedness(preferred_handedness)
    if preferred and candidates:
        matching = [
            hand
            for hand in candidates
            if hand["handedness"] == preferred
            and hand["handedness_score"] >= MIN_HANDEDNESS_SCORE
        ]
        unknown = [
            hand
            for hand in candidates
            if hand["handedness"] is None
            or hand["handedness_score"] < MIN_HANDEDNESS_SCORE
        ]
        if matching:
            candidates = matching
        elif unknown:
            candidates = unknown
        else:
            return []

    if locked_wrist and candidates:
        candidates.sort(
            key=lambda hand: (hand["wrist"][0] - locked_wrist[0]) ** 2
            + (hand["wrist"][1] - locked_wrist[1]) ** 2
        )
    else:
        candidates.sort(key=lambda hand: hand["score"], reverse=True)

    return candidates[:max_hands]


def _normalize_handedness(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("left", "right"):
        return text
    return None


def _parse_handedness_label(label: str) -> tuple[str | None, float]:
    parts = str(label).strip().split()
    handedness = _normalize_handedness(parts[0] if parts else None)
    score = 0.0
    if len(parts) >= 2:
        try:
            score = float(parts[1])
        except ValueError:
            score = 0.0
    return handedness, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe hand landmark webcam test.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--max-hands", type=int, default=MAX_HANDS_TO_DRAW)
    parser.add_argument("--mirror", action="store_true", default=MIRROR_INPUT)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--capture-width", type=int, default=DEFAULT_CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=int, default=DEFAULT_CAPTURE_HEIGHT)
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS)
    parser.add_argument(
        "--capture-buffer-size",
        type=int,
        default=DEFAULT_CAPTURE_BUFFER_SIZE,
    )
    parser.add_argument("--inference-width", type=int, default=DEFAULT_INFERENCE_MAX_WIDTH)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save-frame", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_model(args.model)

    print("MediaPipe:", mp.__version__)
    print(f"Model: {args.model}")

    with create_landmarker(args.model, max(1, args.max_hands)) as landmarker:
        if args.check:
            return 0

        cap = cv2.VideoCapture(args.camera)
        configure_capture(
            cap,
            width=args.capture_width,
            height=args.capture_height,
            fps=args.capture_fps,
            buffer_size=args.capture_buffer_size,
        )
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}.")
            return 1

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

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            inference_start = time.monotonic()
            keypoints, scores, labels = detect_frame(
                landmarker,
                frame,
                max(timestamp_ms, frame_count),
                max_inference_width=args.inference_width,
            )
            inference_ms = (time.monotonic() - inference_start) * 1000
            selected_hands = select_one_hand(
                keypoints,
                scores,
                labels,
                locked_wrist,
                args.max_hands,
            )

            for hand in selected_hands:
                locked_wrist = hand["wrist"]
                draw_hand(frame, hand["keypoints"], hand["scores"], hand["label"])

            frame_count += 1
            elapsed = max(time.monotonic() - start_time, 0.001)
            fps = frame_count / elapsed
            status = (
                f"MediaPipe Hands hands:{len(selected_hands)}/{args.max_hands} "
                f"FPS:{fps:.1f} infer:{inference_ms:.1f}ms"
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

            if args.once:
                print(f"Detected hands: {len(keypoints)}")
                print(f"Selected hands: {len(selected_hands)}")
                print(f"Inference time: {inference_ms:.1f} ms")
                for hand in selected_hands:
                    print(f"Hand {hand['index']}: {hand['label']}")
                if args.save_frame:
                    args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.save_frame), frame)
                    print(f"Saved frame: {args.save_frame}")
                break

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
