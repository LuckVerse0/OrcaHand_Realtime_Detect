from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from tools.mediapipe_hand import (
    DEFAULT_CAPTURE_BUFFER_SIZE,
    DEFAULT_CAPTURE_FPS,
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    DEFAULT_INFERENCE_MAX_WIDTH,
    HAND_CONNECTIONS,
    MODEL_PATH,
    configure_capture,
    create_landmarker,
    detect_frame,
    draw_hand,
    ensure_model,
    select_one_hand,
)
from orca_realtime.config import RuntimeSafetySettings, load_realtime_config
from orca_realtime.filters import ExponentialSmoother, JointSmoother
from orca_realtime.kinematics import HandKinematics
from orca_realtime.logging_utils import SessionLogger
from orca_realtime.orca_controller import OrcaController
from orca_realtime.safety import SafetyController
from orca_realtime.state import RuntimeState, RuntimeStateMachine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ORCA_CORE_ROOT = PROJECT_ROOT / "vendor" / "orca_core"
WINDOW_NAME = "MediaPipe OrcaHand Realtime"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime MediaPipe hand tracking to OrcaHand joint commands."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--capture-width", type=int, default=DEFAULT_CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=int, default=DEFAULT_CAPTURE_HEIGHT)
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS)
    parser.add_argument(
        "--capture-buffer-size",
        type=int,
        default=DEFAULT_CAPTURE_BUFFER_SIZE,
    )
    parser.add_argument("--inference-width", type=int, default=DEFAULT_INFERENCE_MAX_WIDTH)
    parser.add_argument("--max-detected-hands", type=int, default=2)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--mock-orca",
        action="store_true",
        help="Exercise the OrcaHand control path with MockOrcaHand instead of physical motors.",
    )
    parser.add_argument("--orca-core-root", type=Path, default=DEFAULT_ORCA_CORE_ROOT)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-delta", type=float, default=5.0)
    parser.add_argument("--abrupt-delta", type=float, default=120.0)
    parser.add_argument("--offset-deg", type=float, default=5.0)
    parser.add_argument("--startup-ramp-frames", type=int, default=30)
    parser.add_argument("--startup-max-delta", type=float, default=1.0)
    parser.add_argument("--force-calibrate", action="store_true")
    parser.add_argument("--skip-neutral-on-connect", action="store_true")
    parser.add_argument("--send-steps", type=int, default=1)
    parser.add_argument("--send-step-size", type=float, default=1e-2)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    ensure_model(args.model)

    config = load_realtime_config(args.config_dir)
    hardware_enabled = args.live or args.mock_orca
    if hardware_enabled:
        config.validate_for_live()

    settings = RuntimeSafetySettings(
        default_offset_deg=args.offset_deg,
        max_delta_deg_per_frame=args.max_delta,
        startup_max_delta_deg_per_frame=args.startup_max_delta,
        startup_ramp_frames=args.startup_ramp_frames,
        abrupt_delta_deg=args.abrupt_delta,
    )
    safety = SafetyController(config, settings)
    kinematics = HandKinematics(config, safety.safe_neutral())
    landmark_smoother = ExponentialSmoother(alpha=0.45)
    joint_smoother = JointSmoother(default_alpha=0.35, abd_alpha=0.25)
    state = RuntimeStateMachine()
    logger = SessionLogger(args.log_dir, time.strftime("orcahand_%Y%m%d_%H%M%S"))
    controller = OrcaController(
        config.config_path,
        live=hardware_enabled,
        orca_core_root=args.orca_core_root,
        mock=args.mock_orca,
        force_calibrate=args.force_calibrate,
        move_to_neutral=not args.skip_neutral_on_connect,
        send_num_steps=args.send_steps,
        send_step_size=args.send_step_size,
    )

    if hardware_enabled:
        controller.connect()

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
        logger.close()
        controller.disconnect()
        return 1

    start_time = time.monotonic()
    frame_count = 0
    locked_wrist = None
    latest_landmarks: np.ndarray | None = None

    try:
        with create_landmarker(args.model, max(1, args.max_detected_hands)) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    state.fault("camera read failed")
                    break
                if args.mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                keypoints, scores, labels = detect_frame(
                    landmarker,
                    frame,
                    max(timestamp_ms, frame_count),
                    max_inference_width=args.inference_width,
                )
                selected_hands = select_one_hand(
                    keypoints,
                    scores,
                    labels,
                    locked_wrist,
                    1,
                    preferred_handedness=config.hand_type,
                )

                safety_result = None
                if selected_hands:
                    hand = selected_hands[0]
                    locked_wrist = hand["wrist"]
                    latest_landmarks = np.asarray(hand["keypoints"], dtype=float)
                    draw_hand(frame, hand["keypoints"], hand["scores"], hand["label"])

                    smoothed_landmarks = landmark_smoother.update(latest_landmarks)
                    raw_joints = kinematics.estimate(smoothed_landmarks)
                    smoothed_joints = joint_smoother.update(raw_joints)
                    safety_result = safety.apply(smoothed_joints)

                    if not safety_result.accepted:
                        state.tracking_lost("; ".join(safety_result.reasons))
                    elif state.state == RuntimeState.TRACKING_LOST:
                        state.recover_tracking()

                    if state.can_send_to_hardware and safety_result.accepted:
                        try:
                            controller.send(safety_result.joints)
                        except Exception as exc:
                            state.fault(str(exc))
                            controller.emergency_stop()

                    logger.write(
                        {
                            "timestamp": time.time(),
                            "state": state.state.value,
                            "accepted": safety_result.accepted,
                            "reasons": safety_result.reasons,
                            "joints": safety_result.joints,
                            "motor_positions": safety_result.motor_positions,
                        }
                    )
                else:
                    landmark_smoother.reset()
                    joint_smoother.reset()
                    if state.state == RuntimeState.LIVE:
                        state.tracking_lost("no hand detected")

                _draw_status(frame, state, hardware_enabled, safety_result)

                if args.once:
                    break

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if _handle_key(
                    key,
                    state,
                    kinematics,
                    latest_landmarks,
                    controller,
                    safety,
                    hardware_enabled,
                    joint_smoother=joint_smoother,
                ):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        controller.disconnect()

    return 0


def _handle_key(
    key: int,
    state: RuntimeStateMachine,
    kinematics: HandKinematics,
    latest_landmarks: np.ndarray | None,
    controller: OrcaController,
    safety: SafetyController,
    live_allowed: bool,
    joint_smoother=None,
) -> bool:
    if key == 255:
        return False
    if key == ord("q"):
        return True
    if key == ord("m"):
        if state.state == RuntimeState.PREVIEW:
            state.start_mapping()
        else:
            state.stop_mapping()
    elif key == ord("l"):
        if live_allowed and state.state == RuntimeState.ARMED:
            safety.reset_to_safe_neutral()
            if joint_smoother is not None:
                joint_smoother.reset()
            state.enable_live()
        elif state.state == RuntimeState.LIVE:
            state.disable_live()
    elif key == ord("n") and latest_landmarks is not None:
        _capture_visual_neutral(kinematics, latest_landmarks, joint_smoother)
        safety.reset_to_safe_neutral()
    elif key in (27, 32):
        state.fault("emergency stop")
        controller.emergency_stop()
    return False


def _capture_visual_neutral(
    kinematics: HandKinematics,
    latest_landmarks: np.ndarray,
    joint_smoother=None,
) -> None:
    kinematics.capture_neutral(latest_landmarks)
    if joint_smoother is not None:
        joint_smoother.reset()


def _draw_status(
    frame: np.ndarray,
    state: RuntimeStateMachine,
    live_allowed: bool,
    safety_result,
) -> None:
    color = (0, 255, 0)
    if state.state in (RuntimeState.TRACKING_LOST, RuntimeState.FAULT):
        color = (0, 0, 255)
    text = f"state:{state.state.value} live_allowed:{live_allowed}"
    cv2.putText(
        frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    if state.reason:
        cv2.putText(
            frame,
            state.reason[:90],
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    if safety_result is not None:
        preview = " ".join(
            f"{joint}:{safety_result.joints[joint]:.0f}"
            for joint in ("index_mcp", "index_pip", "thumb_abd", "thumb_pip")
        )
        cv2.putText(
            frame,
            preview,
            (16, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    raise SystemExit(main())
