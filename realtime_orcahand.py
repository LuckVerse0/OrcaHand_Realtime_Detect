"""Single-file realtime OrcaHand control.

The runtime keeps the camera preview high resolution, runs MediaPipe on a
bounded inference frame, maps landmarks into safe OrcaHand joint targets, and
only sends commands when the state machine is explicitly live.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import queue
import re
import sys
import threading
import time
import tkinter as tk
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW_NAME = "MediaPipe OrcaHand Realtime"
WINDOW_TITLE = "OrcaHand Realtime Control"
CONSOLE_BG = "#0f141b"
CONSOLE_PANEL = "#151c26"
CONSOLE_CARD = "#1b2531"
CONSOLE_PREVIEW = "#05080d"
CONSOLE_BORDER = "#2f3a49"
CONSOLE_TEXT = "#edf3fb"
CONSOLE_MUTED = "#93a0b3"
CONSOLE_ACCENT = "#66a3ff"
CONSOLE_SUCCESS = "#57d281"
CONSOLE_WARNING = "#f0b84b"
CONSOLE_DANGER = "#ef5b5b"
KEYPOINT_SCORE_THRESHOLD = 0.0
MIN_HAND_SCORE = 0.5
MIN_LANDMARK_SCORE = 0.5
MIN_VISIBLE_LANDMARK_FRACTION = 0.8
MIN_PALM_SIZE_PX = 20.0
MAX_LOCKED_WRIST_JUMP_PX = 350.0
MIN_HANDEDNESS_SCORE = 0.5

# Camera stays bounded for realtime feedback; inference and Tk display work are
# capped separately so the CPU path can stay inside a 30 FPS frame budget.
DEFAULT_CAMERA = 0
DEFAULT_MIRROR = False
# The current Windows MediaPipe wheel reports GPU support disabled at build time.
DEFAULT_USE_GPU = False
DEFAULT_MAX_DETECTED_HANDS = 2
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480
DEFAULT_CAPTURE_FPS = 30
DEFAULT_CAPTURE_BUFFER_SIZE = 1
DEFAULT_CAPTURE_FOURCC = ""
DEFAULT_INFERENCE_MAX_WIDTH = 480
DEFAULT_DISPLAY_MAX_WIDTH = 640
DEFAULT_DISPLAY_MAX_HEIGHT = 480
DEFAULT_PERF_TELEMETRY_FPS = 2
DEFAULT_FPS_AVG_WINDOW_S = 2.0

# Paths are intentionally repo-relative so the single file can run in-place.
DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FLUSH_EVERY = 30
DEFAULT_LOG_FLUSH_INTERVAL_S = 1.0
DEFAULT_UI_TELEMETRY_FPS = 10
DEFAULT_TRACKING_LOST_STOP_S = 0.3
DEFAULT_VISUAL_CALIBRATION_DIR = Path("profiles") / "visual"
DEFAULT_PROCESSED_VIDEO_DIR = DEFAULT_LOG_DIR / "processed_videos"
DEFAULT_ORCA_CORE_ROOT = PROJECT_ROOT / "vendor" / "orca_core"
DEFAULT_VIDEO_OUTPUT_WIDTH = 1920
DEFAULT_VIDEO_OUTPUT_HEIGHT = 1080
DEFAULT_VIDEO_OUTPUT_FPS = 60.0

# Runtime safety gates keep visual tracking errors from becoming abrupt motion.
DEFAULT_OFFSET_DEG = 3.0
DEFAULT_ABD_OFFSET_DEG = 2.0
DEFAULT_MAX_DELTA_DEG_PER_FRAME = 5.0
DEFAULT_STARTUP_MAX_DELTA_DEG_PER_FRAME = 0.5
DEFAULT_STARTUP_RAMP_FRAMES = 60
DEFAULT_ABRUPT_DELTA_DEG = 120.0
DEFAULT_FLEX_DEADZONE_DEG = 20.0
DEFAULT_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
DEFAULT_COMMAND_LIMIT_MARGIN_DEG = DEFAULT_OFFSET_DEG
ABD_DIRECTION_SCALE = -2.0
ABD_DIRECTION_SCALE_BY_JOINT = {
    "thumb_abd": -1.0,
}
THUMB_ABD_ARC_GAIN = 2.0
THUMB_ABD_ARC_LIMIT_DEG = 90.0
FLEX_DEADZONE_BY_JOINT = {
    "thumb_mcp": 0.0,
    "thumb_pip": 0.0,
    "thumb_dip": 0.0,
}
FLEX_ACTIVE_RANGE_BY_JOINT = {
    "thumb_mcp": 120.0,
    "thumb_pip": 120.0,
    "thumb_dip": 120.0,
}
FLEX_DECREASE_JOINTS = {
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
}
MIN_FLEX_CALIBRATION_RANGE_DEG = 8.0
MIN_ABD_CALIBRATION_RANGE_DEG = 4.0
DEFAULT_FORCE_CALIBRATE = False
DEFAULT_MOVE_TO_NEUTRAL_ON_CONNECT = True
DEFAULT_SEND_STEPS = 1
DEFAULT_SEND_STEP_SIZE = 1e-2
DEFAULT_DRAW_LINE_TYPE = cv2.LINE_8
ANSI_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
}
VISUAL_CALIBRATION_VERSION = 1
PROFILE_SUFFIX = ".yaml"
LAST_LANDMARKER_DELEGATE = "unknown"

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

FINGER_STARTS = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

JOINT_DISPLAY_ORDER = (
    "thumb_mcp",
    "thumb_abd",
    "thumb_pip",
    "thumb_dip",
    "index_abd",
    "index_mcp",
    "index_pip",
    "middle_abd",
    "middle_mcp",
    "middle_pip",
    "ring_abd",
    "ring_mcp",
    "ring_pip",
    "pinky_abd",
    "pinky_mcp",
    "pinky_pip",
    "wrist",
)


@dataclass(frozen=True)
class RuntimeSafetySettings:
    """Tunable runtime limits applied after visual joint estimation."""

    default_offset_deg: float = DEFAULT_OFFSET_DEG
    max_delta_deg_per_frame: float = DEFAULT_MAX_DELTA_DEG_PER_FRAME
    startup_max_delta_deg_per_frame: float = DEFAULT_STARTUP_MAX_DELTA_DEG_PER_FRAME
    startup_ramp_frames: int = DEFAULT_STARTUP_RAMP_FRAMES
    abrupt_delta_deg: float = DEFAULT_ABRUPT_DELTA_DEG
    joint_controls: dict[str, dict[str, float | bool]] = field(default_factory=dict)
    offset_overrides_deg: dict[str, float] = field(default_factory=dict)

    def offset_for_joint(self, joint: str) -> float:
        if joint in self.offset_overrides_deg:
            return float(self.offset_overrides_deg[joint])
        if joint.endswith("_abd"):
            return min(float(self.default_offset_deg), DEFAULT_ABD_OFFSET_DEG)
        return float(self.default_offset_deg)

    def control_for_joint(self, joint: str) -> tuple[bool, float]:
        control = self.joint_controls.get(joint, {})
        enabled = bool(control.get("enabled", True))
        gain = float(control.get("gain", 1.0))
        return enabled, gain


@dataclass(frozen=True)
class RealtimeConfig:
    """Normalized config/calibration data needed for live joint commands."""

    config_dir: Path
    config_path: Path
    calibration_path: Path
    hand_type: str
    joint_ids: list[str]
    motor_ids: list[int]
    joint_roms: dict[str, tuple[float, float]]
    neutral_position: dict[str, float]
    joint_to_motor_map: dict[str, int]
    joint_inversions: dict[str, bool]
    motor_limits: dict[int, tuple[float, float]]
    joint_to_motor_ratios: dict[int, float]
    calibrated: bool
    wrist_calibrated: bool
    max_current: int | None = None
    control_mode: str | None = None
    runtime_safety: RuntimeSafetySettings = field(default_factory=RuntimeSafetySettings)

    @property
    def motor_to_joint_map(self) -> dict[int, str]:
        return {motor_id: joint for joint, motor_id in self.joint_to_motor_map.items()}

    def validate_for_live(self) -> None:
        if not self.calibration_path.exists():
            raise ValueError(
                f"{self.calibration_path} is missing. Run Orca calibration first before connecting OrcaHand."
            )
        if not self.calibrated:
            raise ValueError(
                "Run Orca calibration first before connecting OrcaHand: calibration.yaml says calibrated is false."
            )
        for joint in self.joint_ids:
            if joint not in self.joint_roms:
                raise ValueError(f"missing ROM for joint {joint}")
            if joint not in self.joint_to_motor_map:
                raise ValueError(f"missing motor mapping for joint {joint}")
            motor_id = self.joint_to_motor_map[joint]
            if motor_id not in self.motor_limits:
                raise ValueError(f"missing motor limit for motor {motor_id}")
            ratio = self.joint_to_motor_ratios.get(motor_id)
            if ratio is None or float(ratio) == 0.0:
                raise ValueError(f"missing or zero ratio for motor {motor_id}")


@dataclass(frozen=True)
class CalibrationResult:
    """Small adapter for the official Orca calibration YAML schema."""

    motor_limits_dict: dict[int, list]
    joint_to_motor_ratios_dict: dict[int, float]
    calibrated: bool
    wrist_calibrated: bool

    @classmethod
    def empty(cls, motor_ids: list[int]) -> "CalibrationResult":
        return cls(
            motor_limits_dict={int(motor_id): [None, None] for motor_id in motor_ids},
            joint_to_motor_ratios_dict={int(motor_id): 0.0 for motor_id in motor_ids},
            calibrated=False,
            wrist_calibrated=False,
        )

    @classmethod
    def from_calibration_path(
        cls,
        calibration_path: str | Path,
        motor_ids: list[int],
    ) -> "CalibrationResult":
        path = Path(calibration_path)
        if not path.exists():
            return cls.empty([int(motor_id) for motor_id in motor_ids])

        with path.open("r", encoding="utf-8") as file:
            calibration = yaml.safe_load(file) or {}

        motor_limits_raw = calibration.get("motor_limits", {}) or {}
        ratios_raw = calibration.get("joint_to_motor_ratios", {}) or {}
        return cls(
            motor_limits_dict={
                int(motor_id): _read_raw_value(
                    motor_limits_raw,
                    int(motor_id),
                    [None, None],
                )
                for motor_id in motor_ids
            },
            joint_to_motor_ratios_dict={
                int(motor_id): float(
                    _read_raw_value(ratios_raw, int(motor_id), 0.0) or 0.0
                )
                for motor_id in motor_ids
            },
            calibrated=bool(calibration.get("calibrated", False)),
            wrist_calibrated=bool(calibration.get("wrist_calibrated", False)),
        )


@dataclass(frozen=True)
class VisualCalibrationProfile:
    """Metadata shown in the GUI profile selector."""

    name: str
    path: Path
    created_at: str


@dataclass(frozen=True)
class ProcessedVideoFrame:
    """One prepared motion frame and the cached OrcaHand command for playback."""

    frame_index: int
    timestamp_s: float
    joints: dict[str, float] | None
    accepted: bool = True
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessedVideo:
    """A 1080p60 skeleton-overlay video plus per-frame hardware commands."""

    source_path: Path
    output_path: Path
    fps: float
    width: int
    height: int
    frames: list[ProcessedVideoFrame]

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def load_realtime_config(config_dir: str | Path) -> RealtimeConfig:
    config_dir = Path(config_dir).resolve()
    config_path = config_dir / "config.yaml"
    calibration_path = config_dir / "calibration.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if calibration_path.exists():
        with calibration_path.open("r", encoding="utf-8") as file:
            calibration = yaml.safe_load(file) or {}
    else:
        calibration = {}

    joint_to_motor_map, joint_inversions = _normalize_joint_to_motor_map(
        config.get("joint_to_motor_map", {})
    )
    joint_ids = [str(joint) for joint in config.get("joint_ids", [])]
    motor_ids = [int(motor_id) for motor_id in config.get("motor_ids", [])]

    return RealtimeConfig(
        config_dir=config_dir,
        config_path=config_path,
        calibration_path=calibration_path,
        hand_type=str(config.get("type", "")),
        joint_ids=joint_ids,
        motor_ids=motor_ids,
        joint_roms=_read_joint_roms(config.get("joint_roms", {})),
        neutral_position={
            str(joint): float(value)
            for joint, value in (config.get("neutral_position", {}) or {}).items()
        },
        joint_to_motor_map=joint_to_motor_map,
        joint_inversions=joint_inversions,
        motor_limits=_read_int_pair_map(calibration.get("motor_limits", {})),
        joint_to_motor_ratios={
            int(motor_id): float(ratio)
            for motor_id, ratio in (
                calibration.get("joint_to_motor_ratios", {}) or {}
            ).items()
        },
        calibrated=bool(calibration.get("calibrated", False)),
        wrist_calibrated=bool(calibration.get("wrist_calibrated", False)),
        max_current=_optional_int(config.get("max_current")),
        control_mode=config.get("control_mode"),
        runtime_safety=_read_runtime_safety_settings(config),
    )


def _read_runtime_safety_settings(config: dict[str, Any]) -> RuntimeSafetySettings:
    raw = config.get("runtime_safety", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    return RuntimeSafetySettings(
        default_offset_deg=float(raw.get("default_offset_deg", DEFAULT_OFFSET_DEG)),
        max_delta_deg_per_frame=float(
            raw.get("max_delta_deg_per_frame", DEFAULT_MAX_DELTA_DEG_PER_FRAME)
        ),
        startup_max_delta_deg_per_frame=float(
            raw.get(
                "startup_max_delta_deg_per_frame",
                DEFAULT_STARTUP_MAX_DELTA_DEG_PER_FRAME,
            )
        ),
        startup_ramp_frames=int(
            raw.get("startup_ramp_frames", DEFAULT_STARTUP_RAMP_FRAMES)
        ),
        abrupt_delta_deg=float(raw.get("abrupt_delta_deg", DEFAULT_ABRUPT_DELTA_DEG)),
        joint_controls=_read_joint_controls(
            raw.get("joint_controls", config.get("joint_controls", {}))
        ),
        offset_overrides_deg=_read_float_dict(
            raw.get("offset_overrides_deg", config.get("offset_overrides_deg", {}))
        ),
    )


def _read_joint_controls(values: object) -> dict[str, dict[str, float | bool]]:
    if not isinstance(values, dict):
        return {}
    controls: dict[str, dict[str, float | bool]] = {}
    for joint, raw_control in values.items():
        if not isinstance(raw_control, dict):
            continue
        control: dict[str, float | bool] = {}
        if "enabled" in raw_control:
            control["enabled"] = bool(raw_control["enabled"])
        if "gain" in raw_control:
            control["gain"] = float(raw_control["gain"])
        controls[str(joint)] = control
    return controls


def _normalize_joint_to_motor_map(
    raw_map: dict[str, Any],
) -> tuple[dict[str, int], dict[str, bool]]:
    normalized: dict[str, int] = {}
    inversions: dict[str, bool] = {}
    for joint, raw_motor_id in (raw_map or {}).items():
        motor_id = int(raw_motor_id)
        normalized[str(joint)] = abs(motor_id)
        inversions[str(joint)] = motor_id < 0
    return normalized, inversions


def _read_joint_roms(raw_roms: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(joint): (float(values[0]), float(values[1]))
        for joint, values in (raw_roms or {}).items()
    }


def _read_int_pair_map(raw_values: dict[Any, Any]) -> dict[int, tuple[float, float]]:
    pairs: dict[int, tuple[float, float]] = {}
    for key, values in (raw_values or {}).items():
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            continue
        if values[0] is None or values[1] is None:
            continue
        pairs[int(key)] = (float(values[0]), float(values[1]))
    return pairs


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _read_raw_value(raw: dict[Any, Any], key: int, default: Any) -> Any:
    if key in raw:
        return raw[key]
    text_key = str(key)
    if text_key in raw:
        return raw[text_key]
    return default


@dataclass
class ExponentialSmoother:
    """Low-pass filter for landmark coordinates."""

    alpha: float
    _value: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self._value is None:
            self._value = value.copy()
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value.copy()

    def reset(self) -> None:
        self._value = None


@dataclass
class JointSmoother:
    """Low-pass filter for estimated joint commands."""

    default_alpha: float = 0.35
    abd_alpha: float = 0.25
    _values: dict[str, float] = field(default_factory=dict)

    def update(self, joints: dict[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for joint, value in joints.items():
            value = float(value)
            if joint not in self._values:
                output[joint] = value
            else:
                alpha = self.abd_alpha if joint.endswith("_abd") else self.default_alpha
                output[joint] = alpha * value + (1.0 - alpha) * self._values[joint]
        self._values.update(output)
        return dict(output)

    def reset(self) -> None:
        self._values.clear()


@dataclass
class RollingFpsCounter:
    """Recent-window FPS counter that ignores old startup outliers."""

    window_s: float = DEFAULT_FPS_AVG_WINDOW_S
    _times: deque[float] = field(default_factory=deque)
    _last_time: float | None = None

    def update(self, now_s: float) -> tuple[float, float]:
        now = float(now_s)
        if self._last_time is None:
            instant_fps = 0.0
        else:
            instant_fps = 1.0 / max(now - self._last_time, 0.001)
        self._last_time = now

        self._times.append(now)
        cutoff = now - max(float(self.window_s), 0.001)
        while len(self._times) > 1 and self._times[0] < cutoff:
            self._times.popleft()

        if len(self._times) >= 2:
            span_s = max(self._times[-1] - self._times[0], 0.001)
            average_fps = (len(self._times) - 1) / span_s
        else:
            average_fps = instant_fps
        return instant_fps, average_fps

    def reset(self) -> None:
        self._times.clear()
        self._last_time = None


class RuntimeState(str, Enum):
    PREVIEW = "preview"
    ARMED = "armed"
    LIVE = "live"
    TRACKING_LOST = "tracking_lost"
    FAULT = "fault"


@dataclass
class RuntimeStateMachine:
    """Small safety-oriented state machine for preview, armed, live, and fault."""

    state: RuntimeState = RuntimeState.PREVIEW
    reason: str = ""
    _return_state: RuntimeState = RuntimeState.ARMED

    def start_mapping(self) -> None:
        if self.state != RuntimeState.FAULT:
            self.state = RuntimeState.ARMED
            self.reason = ""

    def stop_mapping(self) -> None:
        if self.state != RuntimeState.FAULT:
            self.state = RuntimeState.PREVIEW
            self.reason = ""

    def enable_live(self) -> None:
        if self.state == RuntimeState.ARMED:
            self.state = RuntimeState.LIVE
            self.reason = ""

    def disable_live(self) -> None:
        if self.state == RuntimeState.LIVE:
            self.state = RuntimeState.ARMED
            self.reason = ""

    def tracking_lost(self, reason: str) -> None:
        if self.state == RuntimeState.FAULT:
            return
        self._return_state = self.state
        self.state = RuntimeState.TRACKING_LOST
        self.reason = reason

    def recover_tracking(self) -> None:
        if self.state == RuntimeState.TRACKING_LOST:
            self.state = self._return_state
            self.reason = ""

    def fault(self, reason: str) -> None:
        self.state = RuntimeState.FAULT
        self.reason = reason

    def reset_fault(self) -> None:
        if self.state == RuntimeState.FAULT:
            self.state = RuntimeState.PREVIEW
            self.reason = ""

    @property
    def can_send_to_hardware(self) -> bool:
        return self.state == RuntimeState.LIVE


@dataclass(frozen=True)
class JointSafetyBounds:
    joint: str
    motor_id: int
    joint_min: float
    joint_max: float
    motor_min: float
    motor_max: float
    offset_deg: float
    motor_offset_rad: float


@dataclass(frozen=True)
class SafetyResult:
    joints: dict[str, float]
    accepted: bool
    reasons: list[str]
    motor_positions: dict[int, float]


class SafetyController:
    """Clamp visual targets into safe joint and motor ranges before output."""

    def __init__(
        self,
        config: RealtimeConfig,
        settings: RuntimeSafetySettings | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or RuntimeSafetySettings()
        self.bounds = self._build_bounds()
        self._safe_neutral = self._compute_safe_neutral()
        self._previous_safe = dict(self._safe_neutral)
        self._previous_raw = dict(self._safe_neutral)
        self._startup_ramp_remaining = 0

    def safe_neutral(self) -> dict[str, float]:
        return dict(self._safe_neutral)

    def reset_to_safe_neutral(self) -> None:
        self._previous_safe = dict(self._safe_neutral)
        self._previous_raw = dict(self._safe_neutral)
        self._startup_ramp_remaining = max(0, int(self.settings.startup_ramp_frames))

    def apply(self, targets: dict[str, float]) -> SafetyResult:
        desired = self._desired_targets(targets)
        abrupt_reasons = self._abrupt_reasons(desired)
        if abrupt_reasons:
            return SafetyResult(
                joints=dict(self._previous_safe),
                accepted=False,
                reasons=abrupt_reasons,
                motor_positions=self.estimate_motor_positions(self._previous_safe),
            )

        limited: dict[str, float] = {}
        reasons: list[str] = []
        for joint in self.config.joint_ids:
            bounds = self.bounds[joint]
            if not bounds.joint_min <= desired[joint] <= bounds.joint_max:
                return SafetyResult(
                    joints=dict(self._previous_safe),
                    accepted=False,
                    reasons=[f"{joint} outside safe ROM"],
                    motor_positions=self.estimate_motor_positions(self._previous_safe),
                )
            value = desired[joint]

            max_delta = self._max_delta_for_joint(joint)
            previous = self._previous_safe[joint]
            value = previous + self._clamp(value - previous, -max_delta, max_delta)
            limited[joint] = value

        motor_positions = self.estimate_motor_positions(limited)
        for joint in self.config.joint_ids:
            motor_id = self.config.joint_to_motor_map[joint]
            if motor_id not in motor_positions:
                continue
            bounds = self.bounds[joint]
            motor_pos = motor_positions[motor_id]
            if not bounds.motor_min <= motor_pos <= bounds.motor_max:
                return SafetyResult(
                    joints=dict(self._previous_safe),
                    accepted=False,
                    reasons=[f"{joint} motor {motor_id} outside safe limits"],
                    motor_positions=motor_positions,
                )

        self._previous_safe = dict(limited)
        self._previous_raw = dict(desired)
        if self._startup_ramp_remaining > 0:
            self._startup_ramp_remaining -= 1
        return SafetyResult(
            joints=dict(limited),
            accepted=True,
            reasons=reasons,
            motor_positions=motor_positions,
        )

    def estimate_motor_positions(self, joints: dict[str, float]) -> dict[int, float]:
        positions: dict[int, float] = {}
        for joint, value in joints.items():
            motor_id = self.config.joint_to_motor_map[joint]
            if self._motor_calibration_for_joint(joint) is None:
                continue
            positions[motor_id] = self.estimate_motor_position(joint, value)
        return positions

    def estimate_motor_position(self, joint: str, value: float) -> float:
        motor_id = self.config.joint_to_motor_map[joint]
        rom_min, rom_max = self.config.joint_roms[joint]
        calibration = self._motor_calibration_for_joint(joint)
        if calibration is None:
            raise ValueError(f"missing motor calibration for joint {joint}")
        motor_min, _motor_max, ratio = calibration
        if self.config.joint_inversions.get(joint, False):
            return motor_min + (rom_max - float(value)) * ratio
        return motor_min + (float(value) - rom_min) * ratio

    def _build_bounds(self) -> dict[str, JointSafetyBounds]:
        bounds: dict[str, JointSafetyBounds] = {}
        for joint in self.config.joint_ids:
            motor_id = self.config.joint_to_motor_map[joint]
            rom_min, rom_max = self.config.joint_roms[joint]
            offset_deg = self.settings.offset_for_joint(joint)
            joint_min = rom_min + offset_deg
            joint_max = rom_max - offset_deg
            calibration = self._motor_calibration_for_joint(joint)
            if calibration is None:
                motor_offset_rad = 0.0
                safe_motor_min = float("-inf")
                safe_motor_max = float("inf")
            else:
                motor_min, motor_max, ratio = calibration
                motor_offset_rad = offset_deg * ratio
                safe_motor_min = motor_min + motor_offset_rad
                safe_motor_max = motor_max - motor_offset_rad
            if joint_min >= joint_max:
                raise ValueError(f"safety offset leaves no joint range for {joint}")
            if safe_motor_min >= safe_motor_max:
                raise ValueError(f"safety offset leaves no motor range for {motor_id}")
            bounds[joint] = JointSafetyBounds(
                joint=joint,
                motor_id=motor_id,
                joint_min=joint_min,
                joint_max=joint_max,
                motor_min=safe_motor_min,
                motor_max=safe_motor_max,
                offset_deg=offset_deg,
                motor_offset_rad=motor_offset_rad,
            )
        return bounds

    def _motor_calibration_for_joint(
        self,
        joint: str,
    ) -> tuple[float, float, float] | None:
        motor_id = self.config.joint_to_motor_map[joint]
        motor_limits = self.config.motor_limits.get(motor_id)
        ratio = self.config.joint_to_motor_ratios.get(motor_id)
        if motor_limits is None or ratio is None or float(ratio) == 0.0:
            return None
        motor_min, motor_max = motor_limits
        return float(motor_min), float(motor_max), float(ratio)

    def _compute_safe_neutral(self) -> dict[str, float]:
        neutral: dict[str, float] = {}
        for joint in self.config.joint_ids:
            bounds = self.bounds[joint]
            configured = float(self.config.neutral_position.get(joint, 0.0))
            neutral[joint] = self._clamp(configured, bounds.joint_min, bounds.joint_max)
        return neutral

    def _desired_targets(self, targets: dict[str, float]) -> dict[str, float]:
        desired = dict(self._safe_neutral)
        for joint in self.config.joint_ids:
            enabled, gain = self.settings.control_for_joint(joint)
            if joint == "wrist" or not enabled:
                desired[joint] = self._safe_neutral[joint]
                continue
            raw = float(targets.get(joint, self._safe_neutral[joint]))
            neutral = self._safe_neutral[joint]
            desired[joint] = neutral + (raw - neutral) * gain
        return desired

    def _abrupt_reasons(self, desired: dict[str, float]) -> list[str]:
        if self._startup_ramp_remaining > 0:
            return []
        reasons: list[str] = []
        threshold = float(self.settings.abrupt_delta_deg)
        for joint, value in desired.items():
            if joint == "wrist":
                continue
            previous = self._previous_raw.get(joint, self._safe_neutral[joint])
            if abs(value - previous) > threshold:
                reasons.append(f"abrupt joint change on {joint}")
        return reasons

    def _max_delta_for_joint(self, joint: str) -> float:
        if self._startup_ramp_remaining > 0:
            return float(self.settings.startup_max_delta_deg_per_frame)
        return float(self.settings.max_delta_deg_per_frame)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))


@dataclass
class HandKinematics:
    """Convert normalized hand landmarks into OrcaHand joint-space targets."""

    config: RealtimeConfig
    safe_neutral: dict[str, float]
    _neutral_raw: dict[str, float] = field(default_factory=dict)
    _open_raw: dict[str, float] = field(default_factory=dict)
    _closed_raw: dict[str, float] = field(default_factory=dict)
    _abd_spread_raw: dict[str, float] = field(default_factory=dict)
    _abd_together_raw: dict[str, float] = field(default_factory=dict)
    _calibration_warnings: list[str] = field(default_factory=list)
    has_neutral: bool = False

    def estimate(self, landmarks: np.ndarray) -> dict[str, float]:
        points = _as_points(landmarks)
        raw = self._raw_controls(points)
        joints = dict(self.safe_neutral)

        for finger in ("index", "middle", "ring", "pinky"):
            mcp_joint = f"{finger}_mcp"
            pip_joint = f"{finger}_pip"
            abd_joint = f"{finger}_abd"
            joints[mcp_joint] = self._flex_raw_to_joint(
                mcp_joint, raw[f"{finger}_mcp"]
            )
            joints[pip_joint] = self._flex_raw_to_joint(
                pip_joint, raw[f"{finger}_pip"]
            )
            joints[abd_joint] = self._abd_raw_to_joint(
                abd_joint, raw[f"{finger}_abd"]
            )

        for joint in ("thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip"):
            if joint == "thumb_abd":
                joints[joint] = self._abd_raw_to_joint(joint, raw[joint])
            else:
                joints[joint] = self._flex_raw_to_joint(joint, raw[joint])

        if "wrist" in joints:
            joints["wrist"] = self.safe_neutral["wrist"]
        return joints

    @property
    def has_range_calibration(self) -> bool:
        return bool(self._open_raw) and bool(self._closed_raw)

    def capture_open_pose(self, landmarks: np.ndarray) -> None:
        self._open_raw = self._raw_controls(_as_points(landmarks))
        self._calibration_warnings.clear()

    def capture_closed_pose(self, landmarks: np.ndarray) -> None:
        self._closed_raw = self._raw_controls(_as_points(landmarks))
        self._calibration_warnings.clear()

    @property
    def has_abd_range_calibration(self) -> bool:
        return bool(self._abd_spread_raw) and bool(self._abd_together_raw)

    def capture_abd_spread_pose(self, landmarks: np.ndarray) -> None:
        self._abd_spread_raw = self._raw_controls(_as_points(landmarks))
        self._calibration_warnings.clear()

    def capture_abd_together_pose(self, landmarks: np.ndarray) -> None:
        self._abd_together_raw = self._raw_controls(_as_points(landmarks))
        self._calibration_warnings.clear()

    def range_calibration_status(self) -> str:
        if self.has_range_calibration:
            return "ready"
        if self._open_raw:
            return "open captured, press c"
        if self._closed_raw:
            return "fist captured, press o"
        return "press o(open) then c(fist)"

    def calibration_warnings(self) -> list[str]:
        return list(self._calibration_warnings)

    def capture_neutral(self, landmarks: np.ndarray) -> None:
        self._neutral_raw = self._raw_controls(_as_points(landmarks))
        self.has_neutral = True

    def export_visual_calibration(self) -> dict[str, object]:
        return {
            "neutral_raw": _copy_float_dict(self._neutral_raw),
            "open_raw": _copy_float_dict(self._open_raw),
            "closed_raw": _copy_float_dict(self._closed_raw),
            "abd_spread_raw": _copy_float_dict(self._abd_spread_raw),
            "abd_together_raw": _copy_float_dict(self._abd_together_raw),
            "has_neutral": bool(self.has_neutral),
        }

    def import_visual_calibration(self, state: dict[str, object]) -> None:
        self._neutral_raw = _read_float_dict(state.get("neutral_raw", {}))
        self._open_raw = _read_float_dict(state.get("open_raw", {}))
        self._closed_raw = _read_float_dict(state.get("closed_raw", {}))
        self._abd_spread_raw = _read_float_dict(state.get("abd_spread_raw", {}))
        self._abd_together_raw = _read_float_dict(
            state.get("abd_together_raw", {})
        )
        self.has_neutral = bool(state.get("has_neutral", bool(self._neutral_raw)))
        self._calibration_warnings.clear()

    def _baseline(self, name: str) -> float:
        return float(self._neutral_raw.get(name, 0.0))

    def _raw_controls(self, points: np.ndarray) -> dict[str, float]:
        raw: dict[str, float] = {}
        palm_frame = _palm_frame(points)

        for finger, start in FINGER_STARTS.items():
            curl = _finger_curl(points, start)
            raw[f"{finger}_mcp"] = 0.55 * curl
            raw[f"{finger}_pip"] = curl
            raw[f"{finger}_abd"] = _finger_abduction_from_frame(
                points,
                start,
                palm_frame,
            )

        thumb_mcp_flex = _joint_flexion(points[1], points[2], points[3])
        thumb_curl = _joint_flexion(points[2], points[3], points[4])
        raw["thumb_mcp"] = thumb_mcp_flex
        raw["thumb_abd"] = _thumb_abduction_from_frame(points, palm_frame)
        raw["thumb_pip"] = thumb_curl
        raw["thumb_dip"] = 0.75 * thumb_curl

        return raw

    def _flex_raw_to_joint(self, joint: str, raw_value: float) -> float:
        fraction = self._calibrated_flex_fraction(joint, raw_value)
        if fraction is not None:
            return self._flex_fraction_to_joint(joint, fraction)
        return self._flex_to_joint(joint, float(raw_value) - self._baseline(joint))

    def _calibrated_flex_fraction(
        self, joint: str, raw_value: float
    ) -> float | None:
        if not self.has_range_calibration:
            return None
        open_value = self._open_raw.get(joint)
        closed_value = self._closed_raw.get(joint)
        if open_value is None or closed_value is None:
            return None
        span = float(closed_value) - float(open_value)
        if abs(span) < MIN_FLEX_CALIBRATION_RANGE_DEG:
            self._add_calibration_warning(
                f"{joint} calibration span too small ({abs(span):.1f} deg); "
                "using default mapping."
            )
            return None
        return _clamp((float(raw_value) - float(open_value)) / span, 0.0, 1.0)

    def _add_calibration_warning(self, warning: str) -> None:
        if warning not in self._calibration_warnings:
            self._calibration_warnings.append(warning)

    def _flex_to_joint(self, joint: str, flex_deg: float) -> float:
        deadzone = FLEX_DEADZONE_BY_JOINT.get(joint, DEFAULT_FLEX_DEADZONE_DEG)
        active_range = FLEX_ACTIVE_RANGE_BY_JOINT.get(joint, 100.0 - deadzone)
        flex_after_deadzone = max(0.0, float(flex_deg) - deadzone)
        active_range = max(1.0, active_range)
        fraction = _clamp(flex_after_deadzone / active_range, 0.0, 1.0)
        return self._flex_fraction_to_joint(joint, fraction)

    def _flex_fraction_to_joint(self, joint: str, fraction: float) -> float:
        command_min, command_max = self._command_range(joint)
        fraction = _clamp(fraction, 0.0, 1.0)
        if joint in FLEX_DECREASE_JOINTS:
            open_target, close_target = command_max, command_min
        else:
            open_target, close_target = command_min, command_max
        return open_target + fraction * (close_target - open_target)

    def _abd_raw_to_joint(self, joint: str, raw_value: float) -> float:
        calibrated = self._calibrated_abd_to_joint(joint, raw_value)
        if calibrated is not None:
            return calibrated
        return self._abd_to_joint(joint, float(raw_value) - self._baseline(joint))

    def _calibrated_abd_to_joint(
        self, joint: str, raw_value: float
    ) -> float | None:
        if not self.has_abd_range_calibration:
            return None
        spread_value = self._abd_spread_raw.get(joint)
        together_value = self._abd_together_raw.get(joint)
        if spread_value is None or together_value is None:
            return None

        neutral_value = self._neutral_raw.get(
            joint, 0.5 * (float(spread_value) + float(together_value))
        )
        spread_delta = float(spread_value) - float(neutral_value)
        together_delta = float(together_value) - float(neutral_value)
        if (
            abs(spread_delta) < MIN_ABD_CALIBRATION_RANGE_DEG
            or abs(together_delta) < MIN_ABD_CALIBRATION_RANGE_DEG
        ):
            self._add_calibration_warning(
                f"{joint} ABD calibration span too small; using default mapping."
            )
            return None

        spread_direction = self._abd_command_direction(joint, spread_delta)
        together_direction = self._abd_command_direction(joint, together_delta)
        if spread_direction == 0 or together_direction == 0 or spread_direction == together_direction:
            self._add_calibration_warning(
                f"{joint} ABD calibration endpoints overlap; using default mapping."
            )
            return None

        raw_delta = float(raw_value) - float(neutral_value)
        command_min, command_max = self._command_range(joint)
        center = 0.5 * (command_min + command_max)
        half_range = 0.5 * (command_max - command_min)

        if _same_direction(raw_delta, spread_delta):
            fraction = _clamp(abs(raw_delta) / abs(spread_delta), 0.0, 1.0)
            direction = spread_direction
        elif _same_direction(raw_delta, together_delta):
            fraction = _clamp(abs(raw_delta) / abs(together_delta), 0.0, 1.0)
            direction = together_direction
        else:
            return center
        return _clamp(center + direction * half_range * fraction, command_min, command_max)

    def _abd_command_direction(self, joint: str, raw_delta: float) -> int:
        command_min, command_max = self._command_range(joint)
        center = 0.5 * (command_min + command_max)
        command = self._abd_to_joint(joint, raw_delta)
        if command > center:
            return 1
        if command < center:
            return -1
        scale = ABD_DIRECTION_SCALE_BY_JOINT.get(joint, ABD_DIRECTION_SCALE)
        signed = float(raw_delta) * scale
        if signed > 0.0:
            return 1
        if signed < 0.0:
            return -1
        return 0

    def _abd_to_joint(self, joint: str, value: float) -> float:
        scale = ABD_DIRECTION_SCALE_BY_JOINT.get(joint, ABD_DIRECTION_SCALE)
        command_min, command_max = self._command_range(joint)
        center = 0.5 * (command_min + command_max)
        return _clamp(center + float(value) * scale, command_min, command_max)

    def _relative_to_joint(self, joint: str, value: float, scale: float) -> float:
        neutral = self.safe_neutral[joint]
        command_min, command_max = self._command_range(joint)
        return _clamp(neutral + float(value) * scale, command_min, command_max)

    def _command_range(self, joint: str) -> tuple[float, float]:
        rom_min, rom_max = self.config.joint_roms[joint]
        neutral = self.safe_neutral[joint]
        margin = min(DEFAULT_COMMAND_LIMIT_MARGIN_DEG, DEFAULT_ABD_OFFSET_DEG) if joint.endswith("_abd") else DEFAULT_COMMAND_LIMIT_MARGIN_DEG
        return (
            min(neutral, rom_min + margin),
            max(neutral, rom_max - margin),
        )


def _as_points(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=float)
    if points.shape[0] != 21 or points.shape[1] not in (2, 3):
        raise ValueError("landmarks must have shape (21, 2) or (21, 3)")
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(21, dtype=float)])
    return points


def _copy_float_dict(values: dict[str, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _read_float_dict(values: object) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {str(key): float(value) for key, value in values.items()}


def _finger_curl(points: np.ndarray, start: int) -> float:
    pip_flex = _joint_flexion(points[start], points[start + 1], points[start + 3])
    dip_flex = _joint_flexion(points[start + 1], points[start + 2], points[start + 3])
    return max(pip_flex, 0.5 * (pip_flex + dip_flex))


def _palm_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    palm_forward = points[9] - points[0]
    palm_side = points[17] - points[5]
    forward = _unit_vector(palm_forward)
    side = palm_side - np.dot(palm_side, forward) * forward
    side = _unit_vector(side)
    normal = _unit_vector(np.cross(side, forward))
    valid = not (
        _near_zero_vector(forward)
        or _near_zero_vector(side)
        or _near_zero_vector(normal)
    )
    return forward, side, normal, valid


def _finger_abduction(points: np.ndarray, start: int) -> float:
    return _finger_abduction_from_frame(points, start, _palm_frame(points))


def _finger_abduction_from_frame(
    points: np.ndarray,
    start: int,
    palm_frame: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
) -> float:
    forward, side, _normal, valid = palm_frame
    if not valid:
        return 0.0

    finger_axis = _unit_vector(points[start + 1] - points[start])
    if _near_zero_vector(finger_axis):
        return 0.0
    side_component = float(np.dot(finger_axis, side))
    forward_component = float(np.dot(finger_axis, forward))
    return float(np.degrees(np.arctan2(side_component, forward_component)))


def _thumb_abduction(points: np.ndarray) -> float:
    return _thumb_abduction_from_frame(points, _palm_frame(points))


def _thumb_abduction_from_frame(
    points: np.ndarray,
    palm_frame: tuple[np.ndarray, np.ndarray, np.ndarray, bool],
) -> float:
    forward, _side, normal, valid = palm_frame
    if not valid:
        thumb_axis = points[4] - points[1]
        index_axis = points[8] - points[5]
        return -_signed_angle_2d(index_axis, thumb_axis)

    index_ray = _project_to_plane(points[6] - points[5], normal)
    if _near_zero_vector(index_ray):
        index_ray = forward
    thumb_ray = _project_to_plane(points[4] - points[5], normal)
    if _near_zero_vector(thumb_ray):
        return 0.0
    arc_angle = _signed_angle_about_axis(index_ray, thumb_ray, normal)
    return _clamp(
        arc_angle * THUMB_ABD_ARC_GAIN,
        -THUMB_ABD_ARC_LIMIT_DEG,
        THUMB_ABD_ARC_LIMIT_DEG,
    )


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=float) - np.dot(vector, normal) * normal


def _signed_angle_about_axis(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    a_unit = _unit_vector(a)
    b_unit = _unit_vector(b)
    axis_unit = _unit_vector(axis)
    if (
        np.linalg.norm(a_unit) <= 1e-9
        or np.linalg.norm(b_unit) <= 1e-9
        or np.linalg.norm(axis_unit) <= 1e-9
    ):
        return 0.0
    sin_theta = float(np.dot(np.cross(a_unit, b_unit), axis_unit))
    cos_theta = float(np.dot(a_unit, b_unit))
    return float(np.degrees(np.arctan2(sin_theta, cos_theta)))


def _joint_flexion(prev_point: np.ndarray, joint: np.ndarray, next_point: np.ndarray) -> float:
    angle = _angle_between(prev_point - joint, next_point - joint)
    return max(0.0, 180.0 - angle)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _near_zero_vector(vector: np.ndarray) -> bool:
    return float(np.dot(vector, vector)) <= 1e-18


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-9:
        return 180.0
    cos_theta = float(np.dot(a, b) / norm)
    cos_theta = _clamp(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _signed_angle_2d(a: np.ndarray, b: np.ndarray) -> float:
    a2 = np.asarray(a[:2], dtype=float)
    b2 = np.asarray(b[:2], dtype=float)
    if np.linalg.norm(a2) <= 1e-9 or np.linalg.norm(b2) <= 1e-9:
        return 0.0
    cross = a2[0] * b2[1] - a2[1] * b2[0]
    dot = float(np.dot(a2, b2))
    return float(np.degrees(np.arctan2(cross, dot)))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _same_direction(value: float, reference: float) -> bool:
    return abs(value) > 1e-9 and float(value) * float(reference) > 0.0


class SessionLogger:
    """Write CSV/JSONL rows without flushing on every realtime frame."""

    def __init__(
        self,
        log_dir: str | Path,
        stem: str,
        *,
        flush_every: int = DEFAULT_LOG_FLUSH_EVERY,
        flush_interval_s: float = DEFAULT_LOG_FLUSH_INTERVAL_S,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{stem}.csv"
        self.jsonl_path = self.log_dir / f"{stem}.jsonl"
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_s = max(0.0, float(flush_interval_s))
        self._pending_rows = 0
        self._last_flush = time.monotonic()
        self._fieldnames = [
            "timestamp",
            "state",
            "state_reason",
            "accepted",
            "reasons",
            "joints",
            "motor_positions",
        ]
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        normalized = {
            key: self._serialize(row.get(key, "")) for key in self._fieldnames
        }
        self._writer.writerow(normalized)
        self._jsonl_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._pending_rows += 1
        if self._should_flush():
            self.flush()

    def flush(self) -> None:
        self._jsonl_file.flush()
        self._csv_file.flush()
        self._pending_rows = 0
        self._last_flush = time.monotonic()

    def close(self) -> None:
        self.flush()
        self._csv_file.close()
        self._jsonl_file.close()

    def _should_flush(self) -> bool:
        return self._pending_rows >= self._flush_every or (
            time.monotonic() - self._last_flush
        ) >= self._flush_interval_s

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)


def sanitize_profile_name(name: str) -> str:
    cleaned = str(name).strip()
    if cleaned.endswith((".yaml", ".yml")):
        cleaned = Path(cleaned).stem
    if not cleaned:
        raise ValueError("profile name cannot be empty")
    if any(separator in cleaned for separator in ("/", "\\")) or ".." in cleaned:
        raise ValueError("profile name must not contain path separators")
    if re.match(r"^[A-Za-z]:", cleaned):
        raise ValueError("profile name must not be an absolute path")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("profile name has no usable characters")
    return cleaned


def profile_path(profile_dir: str | Path, name: str) -> Path:
    directory = Path(profile_dir).resolve()
    sanitized = sanitize_profile_name(name)
    path = (directory / f"{sanitized}{PROFILE_SUFFIX}").resolve()
    if path.parent != directory:
        raise ValueError("profile path escaped the visual calibration directory")
    return path


def save_visual_calibration(
    profile_dir: str | Path,
    name: str,
    kinematics: "HandKinematics",
) -> Path:
    path = profile_path(profile_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "version": VISUAL_CALIBRATION_VERSION,
        "name": path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kinematics": kinematics.export_visual_calibration(),
    }
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(profile, file, sort_keys=True, allow_unicode=False)
    return path


def load_visual_calibration(path_or_dir: str | Path, name: str | None = None) -> dict[str, Any]:
    path = (
        profile_path(path_or_dir, name)
        if name is not None
        else Path(path_or_dir).resolve()
    )
    with path.open("r", encoding="utf-8") as file:
        profile = yaml.safe_load(file) or {}
    if int(profile.get("version", 0)) != VISUAL_CALIBRATION_VERSION:
        raise ValueError(f"unsupported visual calibration version in {path}")
    if "kinematics" not in profile:
        raise ValueError(f"visual calibration profile {path} has no kinematics block")
    return profile


def apply_visual_calibration(profile: dict[str, Any], kinematics: "HandKinematics") -> None:
    kinematics.import_visual_calibration(profile["kinematics"])


def list_visual_calibrations(profile_dir: str | Path) -> list[VisualCalibrationProfile]:
    directory = Path(profile_dir)
    if not directory.exists():
        return []

    profiles: list[VisualCalibrationProfile] = []
    for path in sorted(directory.glob(f"*{PROFILE_SUFFIX}")):
        if not path.is_file():
            continue
        try:
            profile = load_visual_calibration(path)
        except Exception:
            continue
        profiles.append(
            VisualCalibrationProfile(
                name=str(profile.get("name", path.stem)),
                path=path,
                created_at=str(profile.get("created_at", "")),
            )
        )
    return sorted(profiles, key=lambda profile: profile.name)


class CalibrationStopped(RuntimeError):
    """Raised when the operator requests official Orca calibration to stop."""


class VideoProcessingStopped(RuntimeError):
    """Raised when the operator requests motion preparation to stop."""


class OrcaController:
    """Thin wrapper around orca_core with a preview-safe dry-run mode."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        live: bool,
        orca_core_root: str | Path | None = None,
        mock: bool = False,
        force_calibrate: bool = False,
        move_to_neutral: bool = True,
        send_num_steps: int = 1,
        send_step_size: float = 1e-2,
    ) -> None:
        self.config_path = Path(config_path)
        self.live = bool(live)
        self.orca_core_root = Path(orca_core_root) if orca_core_root else None
        self.mock = bool(mock)
        self.force_calibrate = bool(force_calibrate)
        self.move_to_neutral = bool(move_to_neutral)
        self.send_num_steps = int(send_num_steps)
        self.send_step_size = float(send_step_size)
        self.hand: Any | None = None
        self.connected = False
        if self.send_num_steps < 1:
            raise ValueError("send_num_steps must be >= 1")
        if self.send_step_size < 0:
            raise ValueError("send_step_size must be >= 0")

    def connect(self) -> None:
        if not self.live:
            self.connected = False
            return

        self._connect_hand()
        self.hand.init_joints(
            force_calibrate=self.force_calibrate,
            move_to_neutral=self.move_to_neutral,
        )
        self.connected = True

    def connect_for_calibration(self) -> None:
        if not self.live:
            self.connected = False
            return
        self._connect_hand()
        self.connected = True

    def calibrate(self, *, force_wrist: bool = False, joints: list[str] | None = None) -> None:
        if not self.live:
            return
        if self.hand is None or not self.connected:
            self.connect_for_calibration()
        if self.hand is None:
            raise RuntimeError("OrcaController is not connected")
        if self._start_stoppable_calibration_task(force_wrist=force_wrist, joints=joints):
            if self._wait_for_task_completion():
                raise CalibrationStopped("Official Orca calibration stopped.")
            return
        self.hand.calibrate(force_wrist=force_wrist, joints=joints)

    def stop_task(self) -> None:
        if self.hand is not None:
            task_thread = getattr(self.hand, "_task_thread", None)
            stop_event = getattr(self.hand, "_task_stop_event", None)
            if (
                task_thread is not None
                and task_thread.is_alive()
                and stop_event is not None
            ):
                stop_event.set()
                return
            self._stop_task_quietly()

    def _start_stoppable_calibration_task(
        self,
        *,
        force_wrist: bool,
        joints: list[str] | None,
    ) -> bool:
        start_task = getattr(self.hand, "_start_task", None)
        calibrate_task = getattr(self.hand, "_calibrate", None)
        if callable(start_task) and callable(calibrate_task):
            start_task(calibrate_task, force_wrist=force_wrist, joints=joints)
            return True

        calibrate = getattr(self.hand, "calibrate", None)
        if not callable(calibrate):
            return False
        try:
            calibrate(blocking=False, force_wrist=force_wrist, joints=joints)
        except TypeError:
            return False
        return getattr(self.hand, "_task_thread", None) is not None

    def _wait_for_task_completion(self) -> bool:
        task_thread = getattr(self.hand, "_task_thread", None)
        if task_thread is None:
            return False
        while task_thread.is_alive():
            task_thread.join(timeout=0.1)
        stop_event = getattr(self.hand, "_task_stop_event", None)
        return bool(stop_event is not None and stop_event.is_set())

    def _connect_hand(self) -> None:
        if self.hand is not None and self.connected:
            return
        OrcaHand = self._load_orca_hand()
        self.hand = OrcaHand(config_path=str(self.config_path))
        success, message = self.hand.connect()
        if not success:
            self.hand = None
            raise RuntimeError(message)

    def send(self, joints: dict[str, float]) -> None:
        if not self.live:
            return
        if self.hand is None or not self.connected:
            raise RuntimeError("OrcaController is not connected")
        self.hand.set_joint_positions(
            joints,
            num_steps=self.send_num_steps,
            step_size=self.send_step_size,
        )

    def emergency_stop(self) -> None:
        if self.hand is not None:
            try:
                self._stop_task_quietly()
                try:
                    self.hand.disable_torque()
                except Exception:
                    pass
                try:
                    self.hand.disconnect()
                except Exception:
                    pass
            finally:
                self.hand = None
                self.connected = False
        else:
            self.connected = False

    def disconnect(self) -> None:
        hand = self.hand
        try:
            if hand is not None:
                self._stop_task_quietly()
                hand.disconnect()
        finally:
            self.hand = None
            self.connected = False

    def _load_orca_hand(self):
        candidates = []
        if self.orca_core_root is not None:
            candidates.append(self.orca_core_root)
        candidates.append(DEFAULT_ORCA_CORE_ROOT)

        for root in candidates:
            if root.exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))

        try:
            if self.mock:
                from orca_core.hardware_hand import MockOrcaHand

                return MockOrcaHand
            from orca_core import OrcaHand
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"orca_core is required for --live. Check DEFAULT_ORCA_CORE_ROOT ({DEFAULT_ORCA_CORE_ROOT}) or install orca_core."
            ) from exc
        return OrcaHand

    def _stop_task_quietly(self) -> None:
        stop_task = getattr(self.hand, "stop_task", None)
        if stop_task is None:
            return
        try:
            stop_task()
        except Exception:
            pass


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


def _configure_capture(
    cap: cv2.VideoCapture,
    *,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    fps: int = DEFAULT_CAPTURE_FPS,
    buffer_size: int = DEFAULT_CAPTURE_BUFFER_SIZE,
    fourcc: str = DEFAULT_CAPTURE_FOURCC,
) -> None:
    """Apply low-latency camera settings best-effort across OpenCV backends."""

    if fourcc:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*str(fourcc)[:4]),
        )
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    if buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))


def _camera_backend_from_name(name: str | None) -> int:
    normalized = (name or "auto").strip().lower()
    if normalized == "auto":
        return DEFAULT_CAMERA_BACKEND
    if normalized == "any":
        return cv2.CAP_ANY
    if normalized == "dshow":
        return cv2.CAP_DSHOW
    if normalized == "msmf":
        return cv2.CAP_MSMF
    raise ValueError(f"Unsupported camera backend: {name}")


def _open_capture(
    camera: int = DEFAULT_CAMERA,
    *,
    backend: str | int | None = None,
    width: int = DEFAULT_CAPTURE_WIDTH,
    height: int = DEFAULT_CAPTURE_HEIGHT,
    fps: int = DEFAULT_CAPTURE_FPS,
    buffer_size: int = DEFAULT_CAPTURE_BUFFER_SIZE,
    fourcc: str = DEFAULT_CAPTURE_FOURCC,
) -> cv2.VideoCapture:
    """Open and configure the camera through the preferred platform backend."""

    selected_backend = (
        _camera_backend_from_name(backend)
        if backend is None or isinstance(backend, str)
        else int(backend)
    )
    if selected_backend == cv2.CAP_ANY:
        cap = cv2.VideoCapture(camera)
    else:
        cap = cv2.VideoCapture(camera, selected_backend)
    _configure_capture(
        cap,
        width=width,
        height=height,
        fps=fps,
        buffer_size=buffer_size,
        fourcc=fourcc,
    )
    return cap


class LatestFrameCapture:
    """Background camera reader that keeps only the newest frame."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        *,
        initial_frame: np.ndarray | None = None,
        initial_read_ms: float = 0.0,
    ) -> None:
        self.cap = cap
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_frame = (
            initial_frame.copy() if initial_frame is not None else None
        )
        self._latest_read_ms = float(initial_read_ms)
        self._latest_ok = initial_frame is not None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_s)))
        self._thread = None

    def latest(self) -> tuple[bool, np.ndarray | None, float]:
        with self._lock:
            if self._latest_frame is None:
                return False, None, self._latest_read_ms
            return True, self._latest_frame.copy(), self._latest_read_ms

    def capture_once_for_test(self) -> bool:
        return self._capture_once()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            ok = self._capture_once()
            if not ok:
                time.sleep(0.005)

    def _capture_once(self) -> bool:
        read_start = time.monotonic()
        ok, frame = self.cap.read()
        read_ms = (time.monotonic() - read_start) * 1000.0
        with self._lock:
            self._latest_ok = bool(ok)
            self._latest_read_ms = read_ms
            if ok:
                self._latest_frame = frame
        return bool(ok)


def _next_frame_delay_ms(*, elapsed_s: float, target_fps: int) -> int:
    if target_fps <= 0:
        return 1
    target_interval_s = 1.0 / float(target_fps)
    remaining_s = target_interval_s - max(0.0, float(elapsed_s))
    return max(1, int(remaining_s * 1000.0))


def create_landmarker(
    model_path: Path,
    max_hands: int,
    prefer_gpu: bool = DEFAULT_USE_GPU,
) -> vision.HandLandmarker:
    """Create MediaPipe HandLandmarker with optional GPU fallback."""

    global LAST_LANDMARKER_DELEGATE
    if prefer_gpu:
        try:
            landmarker = _create_landmarker_with_delegate(model_path, max_hands, True)
            LAST_LANDMARKER_DELEGATE = "GPU"
            return landmarker
        except Exception as exc:
            print(_color_text(f"GPU delegate unavailable, falling back to CPU: {exc}", "yellow"))

    landmarker = _create_landmarker_with_delegate(model_path, max_hands, False)
    LAST_LANDMARKER_DELEGATE = "CPU"
    return landmarker


def _create_landmarker_with_delegate(
    model_path: Path,
    max_hands: int,
    prefer_gpu: bool,
) -> vision.HandLandmarker:
    base_options = BaseOptions(
        model_asset_path=str(model_path),
        delegate=BaseOptions.Delegate.GPU if prefer_gpu else None,
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def frame_to_mediapipe_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _resize_for_inference(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale only the inference input; display coordinates use full frame."""

    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = float(max_width) / float(width)
    target_size = (int(max_width), max(1, int(round(height * scale))))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def detect_frame(
    landmarker: vision.HandLandmarker,
    frame: np.ndarray,
    timestamp_ms: int,
    *,
    max_inference_width: int = DEFAULT_INFERENCE_MAX_WIDTH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run MediaPipe and rescale normalized landmarks into display pixels."""

    inference_frame = _resize_for_inference(frame, max_inference_width)
    image = frame_to_mediapipe_image(inference_frame)
    result = landmarker.detect_for_video(image, timestamp_ms)

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

    frame_h, frame_w = frame.shape[:2]
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
    *,
    draw_label: bool = True,
) -> None:
    """Draw the tracked hand skeleton in-place on the display frame."""

    visible = scores >= KEYPOINT_SCORE_THRESHOLD

    for start, end in HAND_CONNECTIONS:
        if visible[start] and visible[end]:
            p1 = (int(keypoints[start, 0]), int(keypoints[start, 1]))
            p2 = (int(keypoints[end, 0]), int(keypoints[end, 1]))
            cv2.line(frame, p1, p2, (0, 180, 255), 2, DEFAULT_DRAW_LINE_TYPE)

    for point, is_visible in zip(keypoints[:, :2], visible):
        if is_visible:
            cv2.circle(
                frame,
                (int(point[0]), int(point[1])),
                4,
                (0, 255, 0),
                -1,
                DEFAULT_DRAW_LINE_TYPE,
            )

    if draw_label:
        wrist_x = int(keypoints[0, 0])
        wrist_y = int(keypoints[0, 1])
        cv2.putText(
            frame,
            label,
            (wrist_x + 8, max(wrist_y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
            DEFAULT_DRAW_LINE_TYPE,
        )


def process_video_to_1080p60(
    source_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_PROCESSED_VIDEO_DIR,
    config: RealtimeConfig,
    settings: RuntimeSafetySettings,
    visual_calibration_state: dict[str, object] | None = None,
    stop_event: threading.Event | None = None,
    progress_callback=None,
) -> ProcessedVideo:
    """Process a complete video into a 1080p60 skeleton overlay and joint track."""

    ensure_model(MODEL_PATH)
    source = Path(source_path)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = _processed_video_output_path(source, output_directory)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Could not open selected motion source.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        DEFAULT_VIDEO_OUTPUT_FPS,
        (DEFAULT_VIDEO_OUTPUT_WIDTH, DEFAULT_VIDEO_OUTPUT_HEIGHT),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create motion stream.")

    safety = SafetyController(config, settings)
    kinematics = HandKinematics(config, safety.safe_neutral())
    if visual_calibration_state:
        kinematics.import_visual_calibration(visual_calibration_state)
    landmark_smoother = ExponentialSmoother(alpha=0.45)
    joint_smoother = JointSmoother(default_alpha=0.35, abd_alpha=0.25)
    frames: list[ProcessedVideoFrame] = []
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = DEFAULT_VIDEO_OUTPUT_FPS
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = 0
    if source_frame_count > 0:
        total_frames = max(
            1,
            int(round(source_frame_count * DEFAULT_VIDEO_OUTPUT_FPS / source_fps)),
        )

    try:
        with create_landmarker(
            MODEL_PATH,
            DEFAULT_MAX_DETECTED_HANDS,
            prefer_gpu=DEFAULT_USE_GPU,
        ) as landmarker:
            frame_index = 0
            source_frame_index = -1
            source_frame: np.ndarray | None = None
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise VideoProcessingStopped("Video processing stopped.")
                if total_frames > 0 and frame_index >= total_frames:
                    break

                target_source_index = int(
                    frame_index * source_fps / DEFAULT_VIDEO_OUTPUT_FPS
                )
                while source_frame_index < target_source_index:
                    ok, next_frame = cap.read()
                    if not ok:
                        source_frame = None
                        break
                    source_frame_index += 1
                    source_frame = next_frame
                if source_frame is None:
                    break

                frame = source_frame.copy()
                timestamp_s = frame_index / DEFAULT_VIDEO_OUTPUT_FPS
                timestamp_ms = int(timestamp_s * 1000.0)
                keypoints, scores, labels = detect_frame(
                    landmarker,
                    frame,
                    timestamp_ms,
                )
                selected_hands = select_one_hand(
                    keypoints,
                    scores,
                    labels,
                    locked_wrist=None,
                    max_hands=1,
                    preferred_handedness=config.hand_type or "right",
                )

                joints: dict[str, float] | None = None
                accepted = False
                reasons: list[str] = []
                if selected_hands:
                    hand = selected_hands[0]
                    landmarks = np.asarray(hand["keypoints"], dtype=float)
                    draw_hand(
                        frame,
                        hand["keypoints"],
                        hand["scores"],
                        hand["label"],
                        draw_label=False,
                    )
                    smoothed_landmarks = landmark_smoother.update(landmarks)
                    raw_joints = kinematics.estimate(smoothed_landmarks)
                    smoothed_joints = joint_smoother.update(raw_joints)
                    safety_result = safety.apply(smoothed_joints)
                    accepted = safety_result.accepted
                    reasons = list(safety_result.reasons)
                    if safety_result.accepted:
                        joints = dict(safety_result.joints)
                else:
                    landmark_smoother.reset()
                    joint_smoother.reset()
                    safety.reset_to_safe_neutral()
                    reasons = ["no right hand detected"]

                writer.write(
                    _letterbox_to_size(
                        frame,
                        target_width=DEFAULT_VIDEO_OUTPUT_WIDTH,
                        target_height=DEFAULT_VIDEO_OUTPUT_HEIGHT,
                    )
                )
                frames.append(
                    ProcessedVideoFrame(
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                        joints=joints,
                        accepted=accepted,
                        reasons=reasons,
                    )
                )
                frame_index += 1
                if progress_callback and (
                    frame_index == 1 or frame_index % int(DEFAULT_VIDEO_OUTPUT_FPS) == 0
                ):
                    progress_callback(_format_video_progress(frame_index, total_frames))
    except VideoProcessingStopped:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        cap.release()
        writer.release()

    return ProcessedVideo(
        source_path=source,
        output_path=output_path,
        fps=DEFAULT_VIDEO_OUTPUT_FPS,
        width=DEFAULT_VIDEO_OUTPUT_WIDTH,
        height=DEFAULT_VIDEO_OUTPUT_HEIGHT,
        frames=frames,
    )


def _processed_video_output_path(source_path: Path, output_dir: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("_")
    if not safe_stem:
        safe_stem = "video"
    return output_dir / f"{safe_stem}_orcahand_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


def _format_video_progress(frame_index: int, total_frames: int) -> str:
    if total_frames > 0:
        percent = min(100.0, frame_index / total_frames * 100.0)
        return f"Preparing motion: {frame_index}/{total_frames} frames ({percent:.1f}%)."
    return f"Preparing motion: {frame_index} frames."


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


def _hand_quality_reasons(
    keypoints: np.ndarray,
    scores: np.ndarray,
    locked_wrist: tuple[float, float] | None,
) -> list[str]:
    reasons: list[str] = []
    points = np.asarray(keypoints, dtype=np.float32)
    hand_scores = np.asarray(scores, dtype=np.float32)
    if points.shape[0] != 21 or hand_scores.shape[0] != 21:
        return ["incomplete hand landmarks"]

    mean_score = float(np.mean(hand_scores))
    visible_fraction = float(np.mean(hand_scores >= MIN_LANDMARK_SCORE))
    if mean_score < MIN_HAND_SCORE:
        reasons.append("low hand landmark score")
    if visible_fraction < MIN_VISIBLE_LANDMARK_FRACTION:
        reasons.append("too few confident landmarks")

    min_palm_size_sq = MIN_PALM_SIZE_PX * MIN_PALM_SIZE_PX
    palm_size_sq = max(
        _distance_sq_2d(points[9], points[0]),
        _distance_sq_2d(points[17], points[5]),
        _distance_sq_2d(points[5], points[0]),
    )
    if palm_size_sq < min_palm_size_sq:
        reasons.append("hand too small or degenerate")

    if locked_wrist is not None:
        jump_limit_sq = MAX_LOCKED_WRIST_JUMP_PX * MAX_LOCKED_WRIST_JUMP_PX
        if _distance_sq_xy(points[0], locked_wrist[0], locked_wrist[1]) > jump_limit_sq:
            reasons.append("wrist jump too large")
    return reasons


def _distance_sq_2d(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def _distance_sq_xy(a: np.ndarray, x: float, y: float) -> float:
    dx = float(a[0]) - float(x)
    dy = float(a[1]) - float(y)
    return dx * dx + dy * dy


def select_one_hand(
    keypoints: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    locked_wrist,
    max_hands: int,
    preferred_handedness: str | None = None,
):
    """Keep controlling the same hand by wrist proximity when possible."""

    candidates = []
    for index, (hand_keypoints, hand_scores, label) in enumerate(
        zip(keypoints, scores, labels)
    ):
        if _hand_quality_reasons(hand_keypoints, hand_scores, locked_wrist):
            continue
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


class OrcaRealtimeGui:
    """Tk dashboard that owns realtime capture, inference, safety, and output."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title(WINDOW_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.config = load_realtime_config(DEFAULT_CONFIG_DIR)
        self.hardware_allowed = bool(args.live)
        self.settings = self.config.runtime_safety
        self.safety = SafetyController(self.config, self.settings)
        self.kinematics = HandKinematics(self.config, self.safety.safe_neutral())
        self.landmark_smoother = ExponentialSmoother(alpha=0.45)
        self.joint_smoother = JointSmoother(default_alpha=0.35, abd_alpha=0.25)
        self.state = RuntimeStateMachine()
        self.controller = OrcaController(
            self.config.config_path,
            live=self.hardware_allowed,
            orca_core_root=DEFAULT_ORCA_CORE_ROOT,
            mock=False,
            force_calibrate=False,
            move_to_neutral=DEFAULT_MOVE_TO_NEUTRAL_ON_CONNECT,
            send_num_steps=DEFAULT_SEND_STEPS,
            send_step_size=DEFAULT_SEND_STEP_SIZE,
        )

        self.cap: cv2.VideoCapture | None = None
        self.capture_reader: LatestFrameCapture | None = None
        self.landmarker: Any | None = None
        self.logger: SessionLogger | None = None
        self.running = False
        self.frame_count = 0
        self.start_time = 0.0
        self.locked_wrist: tuple[float, float] | None = None
        self.latest_landmarks: np.ndarray | None = None
        self.latest_safety_result: SafetyResult | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._background_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._background_busy = False
        self._calibration_busy = False
        self._video_processing_busy = False
        self._video_playing = False
        self._video_stop_event = threading.Event()
        self._video_after_id: str | None = None
        self._play_video_after_connect = False
        self.motion_controls_visible = False
        self.video_playback_cap: cv2.VideoCapture | None = None
        self.video_playback_index = 0
        self.video_playback_start_s = 0.0
        self.processed_video: ProcessedVideo | None = None
        self._next_telemetry_update = 0.0
        self._next_perf_telemetry_update = 0.0
        self._panel_scrollregion_pending = False
        self._hand_missing_since: float | None = None
        self._fps_counter = RollingFpsCounter()

        self.status_var = tk.StringVar(value="Ready.")
        self.state_var = tk.StringVar(value="state: preview")
        self.hardware_var = tk.StringVar(value="hardware: disconnected")
        self.calibration_var = tk.StringVar(value="")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.inference_var = tk.StringVar(value="Inference: -- ms")
        self.delegate_var = tk.StringVar(value="Delegate: --")
        self.hand_var = tk.StringVar(value="Hand: none")
        self.joint_angles_var = tk.StringVar(value="Angles: --")
        self.safety_var = tk.StringVar(value="Safety: --")
        self.profile_var = tk.StringVar(value="")
        self.preview_state_var = tk.StringVar(value="STATE PREVIEW")
        self.preview_fps_var = tk.StringVar(value="FPS --")
        self.preview_safety_var = tk.StringVar(value="Safety: --")
        self.state_card_var = tk.StringVar(value="PREVIEW")
        self.hardware_card_var = tk.StringVar(value="DISCONNECTED")
        self.tracking_card_var = tk.StringVar(value="NO HAND")
        self.safety_card_var = tk.StringVar(value="--")

        self._build_ui()
        self._refresh_profiles()
        self._refresh_calibration_status()
        self._update_button_states()
        self._poll_background_queue()

    def _build_console_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Console.TFrame", background=CONSOLE_BG)
        style.configure("Panel.TFrame", background=CONSOLE_PANEL)
        style.configure(
            "Console.TLabelframe",
            background=CONSOLE_PANEL,
            bordercolor=CONSOLE_BORDER,
            relief="solid",
        )
        style.configure(
            "Console.TLabelframe.Label",
            background=CONSOLE_PANEL,
            foreground=CONSOLE_TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Console.TLabel",
            background=CONSOLE_PANEL,
            foreground=CONSOLE_TEXT,
        )
        style.configure(
            "Muted.TLabel",
            background=CONSOLE_PANEL,
            foreground=CONSOLE_MUTED,
        )
        style.configure(
            "Primary.TButton",
            background=CONSOLE_ACCENT,
            foreground=CONSOLE_TEXT,
            padding=(10, 6),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#7db2ff"), ("disabled", "#2c3542")],
            foreground=[("disabled", "#748094")],
        )
        style.configure(
            "Danger.TButton",
            background=CONSOLE_DANGER,
            foreground=CONSOLE_TEXT,
            padding=(10, 6),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#ff7373"), ("disabled", "#3b3238")],
            foreground=[("disabled", "#83717a")],
        )

    def _create_metric_card(
        self,
        parent,
        *,
        title: str,
        variable: tk.StringVar,
        accent: str,
    ) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=CONSOLE_CARD,
            highlightbackground=CONSOLE_BORDER,
            highlightcolor=CONSOLE_BORDER,
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        tk.Label(
            card,
            text=title,
            bg=CONSOLE_CARD,
            fg=CONSOLE_MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            textvariable=variable,
            bg=CONSOLE_CARD,
            fg=accent,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(4, 0))
        return card

    def _create_status_pill(
        self,
        parent,
        *,
        label: str,
        variable: tk.StringVar,
        accent: str = CONSOLE_TEXT,
    ) -> tk.Frame:
        pill = tk.Frame(
            parent,
            bg=CONSOLE_CARD,
            highlightbackground=CONSOLE_BORDER,
            highlightcolor=CONSOLE_BORDER,
            highlightthickness=1,
            padx=10,
            pady=7,
        )
        tk.Label(
            pill,
            text=label,
            bg=CONSOLE_CARD,
            fg=CONSOLE_MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            pill,
            textvariable=variable,
            bg=CONSOLE_CARD,
            fg=accent,
            font=("Segoe UI", 10),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))
        return pill

    def _build_ui(self) -> None:
        self._build_console_styles()
        self.root.configure(bg=CONSOLE_BG)
        self.root.geometry("1180x760")
        self.root.minsize(980, 620)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12, style="Console.TFrame")
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        video_frame = tk.Frame(
            main,
            bg=CONSOLE_PREVIEW,
            highlightbackground=CONSOLE_BORDER,
            highlightthickness=1,
            width=DEFAULT_DISPLAY_MAX_WIDTH,
            height=DEFAULT_DISPLAY_MAX_HEIGHT,
        )
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        video_frame.grid_propagate(False)
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_frame,
            anchor="center",
            text="",
            bg=CONSOLE_PREVIEW,
            fg=CONSOLE_TEXT,
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        preview_top = tk.Frame(
            video_frame,
            bg="#101822",
            highlightbackground=CONSOLE_BORDER,
            highlightthickness=1,
            padx=10,
            pady=7,
        )
        preview_top.place(x=16, y=16, relwidth=1.0, width=-32, height=46)
        preview_top.columnconfigure(0, weight=1)
        preview_top.columnconfigure(1, weight=1)
        preview_top.columnconfigure(2, weight=1)
        for column, (variable, color, anchor) in enumerate(
            (
                (self.preview_state_var, CONSOLE_SUCCESS, "w"),
                (self.preview_fps_var, CONSOLE_ACCENT, "center"),
                (self.preview_safety_var, CONSOLE_WARNING, "e"),
            )
        ):
            tk.Label(
                preview_top,
                textvariable=variable,
                bg="#101822",
                fg=color,
                font=("Segoe UI", 10, "bold"),
                anchor=anchor,
            ).grid(row=0, column=column, sticky="ew")

        preview_bottom = tk.Frame(
            video_frame,
            bg="#101822",
            highlightbackground=CONSOLE_BORDER,
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        preview_bottom.place(x=16, rely=1.0, y=-94, relwidth=1.0, width=-32, height=78)
        tk.Label(
            preview_bottom,
            text="Joint Telemetry",
            bg="#101822",
            fg=CONSOLE_MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            preview_bottom,
            textvariable=self.joint_angles_var,
            bg="#101822",
            fg=CONSOLE_TEXT,
            font=("Consolas", 9),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        panel_holder = ttk.Frame(main, style="Console.TFrame")
        panel_holder.grid(row=0, column=1, sticky="ns")
        panel_holder.rowconfigure(0, weight=1)
        panel_holder.columnconfigure(0, weight=1)

        self._panel_canvas = tk.Canvas(
            panel_holder,
            highlightthickness=0,
            width=390,
            bg=CONSOLE_BG,
            bd=0,
        )
        self._panel_scrollbar = ttk.Scrollbar(
            panel_holder,
            orient="vertical",
            command=self._panel_canvas.yview,
        )
        self._panel_canvas.configure(yscrollcommand=self._panel_scrollbar.set)
        self._panel_canvas.grid(row=0, column=0, sticky="ns")
        self._panel_scrollbar.grid(row=0, column=1, sticky="ns")

        panel = ttk.Frame(self._panel_canvas, style="Panel.TFrame", padding=10)
        panel.columnconfigure(0, weight=1)
        self._panel_window = self._panel_canvas.create_window(
            (0, 0),
            window=panel,
            anchor="nw",
        )
        panel.bind("<Configure>", self._queue_panel_scrollregion_update)
        self._panel_canvas.bind("<Configure>", self._resize_panel_window)
        self._panel_canvas.bind("<Enter>", self._bind_panel_mousewheel)
        self._panel_canvas.bind("<Leave>", self._unbind_panel_mousewheel)

        snapshot = ttk.LabelFrame(
            panel,
            text="System Snapshot",
            padding=10,
            style="Console.TLabelframe",
        )
        snapshot.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        snapshot.columnconfigure(0, weight=1)
        snapshot.columnconfigure(1, weight=1)
        cards = (
            ("STATE", self.state_card_var, CONSOLE_SUCCESS),
            ("HARDWARE", self.hardware_card_var, CONSOLE_ACCENT),
            ("TRACKING", self.tracking_card_var, CONSOLE_TEXT),
            ("SAFETY", self.safety_card_var, CONSOLE_WARNING),
        )
        for index, (title, variable, accent) in enumerate(cards):
            card = self._create_metric_card(
                snapshot,
                title=title,
                variable=variable,
                accent=accent,
            )
            card.grid(
                row=index // 2,
                column=index % 2,
                sticky="ew",
                padx=(0, 6) if index % 2 == 0 else (6, 0),
                pady=(0, 8) if index < 2 else (0, 0),
            )

        status = ttk.LabelFrame(
            panel,
            text="Operator Message",
            padding=10,
            style="Console.TLabelframe",
        )
        status.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for row, variable in enumerate((self.status_var, self.calibration_var)):
            ttk.Label(
                status,
                textvariable=variable,
                wraplength=340,
                style="Console.TLabel" if row == 0 else "Muted.TLabel",
            ).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 8, 0))

        metrics = ttk.LabelFrame(
            panel,
            text="Runtime Telemetry",
            padding=10,
            style="Console.TLabelframe",
        )
        metrics.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        metrics.columnconfigure(0, weight=1)
        runtime_rows = (
            ("FRAME RATE", self.fps_var, CONSOLE_ACCENT),
            ("INFERENCE", self.inference_var, CONSOLE_TEXT),
            ("DELEGATE", self.delegate_var, CONSOLE_TEXT),
            ("HAND", self.hand_var, CONSOLE_SUCCESS),
        )
        for row, (label, variable, accent) in enumerate(runtime_rows):
            pill = self._create_status_pill(
                metrics,
                label=label,
                variable=variable,
                accent=accent,
            )
            pill.grid(row=row, column=0, sticky="ew", pady=(0, 6))

        angles = ttk.LabelFrame(
            panel,
            text="Joint Telemetry",
            padding=10,
            style="Console.TLabelframe",
        )
        angles.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            angles,
            textvariable=self.joint_angles_var,
            wraplength=340,
            style="Console.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        safety_panel = ttk.LabelFrame(
            panel,
            text="Safety",
            padding=10,
            style="Console.TLabelframe",
        )
        safety_panel.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            safety_panel,
            textvariable=self.safety_var,
            wraplength=340,
            style="Console.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        run_controls = ttk.LabelFrame(
            panel,
            text="Run Controls",
            padding=10,
            style="Console.TLabelframe",
        )
        run_controls.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        run_controls.columnconfigure(0, weight=1)
        run_controls.columnconfigure(1, weight=1)
        ttk.Label(run_controls, text="Recognition", style="Muted.TLabel").grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 4),
        )
        self.start_button = ttk.Button(
            run_controls,
            text="开始识别",
            command=self.start_recognition,
            style="Primary.TButton",
        )
        self.start_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.stop_button = ttk.Button(
            run_controls,
            text="结束识别",
            command=self.stop_recognition,
        )
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=2)
        ttk.Label(run_controls, text="Hardware", style="Muted.TLabel").grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(10, 4),
        )
        self.connect_button = ttk.Button(
            run_controls,
            text="连接 OrcaHand",
            command=self.connect_hardware,
        )
        self.connect_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        self.map_button = ttk.Button(
            run_controls,
            text="开始映射",
            command=self.start_mapping,
            style="Primary.TButton",
        )
        self.map_button.grid(row=4, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.stop_map_button = ttk.Button(
            run_controls,
            text="停止映射",
            command=self.stop_mapping,
        )
        self.stop_map_button.grid(row=4, column=1, sticky="ew", padx=(4, 0), pady=2)
        self.output_button = ttk.Button(
            run_controls,
            text="启用机器手输出",
            command=self.enable_output,
            style="Primary.TButton",
        )
        self.output_button.grid(row=5, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.stop_output_button = ttk.Button(
            run_controls,
            text="停止输出",
            command=self.disable_output,
        )
        self.stop_output_button.grid(row=5, column=1, sticky="ew", padx=(4, 0), pady=2)
        self.neutral_button = ttk.Button(
            run_controls,
            text="Neutral",
            command=self.move_to_neutral,
        )
        self.neutral_button.grid(row=6, column=0, sticky="ew", padx=(0, 4), pady=(10, 2))
        self.estop_button = ttk.Button(
            run_controls,
            text="紧急停止",
            command=self.emergency_stop,
            style="Danger.TButton",
        )
        self.estop_button.grid(row=6, column=1, sticky="ew", padx=(4, 0), pady=(10, 2))
        self.motion_toggle_button = ttk.Button(
            run_controls,
            text="Show Motion Source",
            command=self._toggle_motion_controls,
        )
        self.motion_toggle_button.grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 4),
        )
        self.motion_controls_frame = ttk.Frame(run_controls, style="Console.TFrame")
        self.motion_controls_frame.columnconfigure(0, weight=1)
        self.motion_controls_frame.columnconfigure(1, weight=1)
        self.import_video_button = ttk.Button(
            self.motion_controls_frame,
            text="Load Motion",
            command=self.import_video,
        )
        self.import_video_button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.stop_video_button = ttk.Button(
            self.motion_controls_frame,
            text="Stop Motion",
            command=self.stop_processed_video_playback,
        )
        self.stop_video_button.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)
        self._sync_motion_controls_visibility()

        visual = ttk.LabelFrame(
            panel,
            text="Visual Calibration",
            padding=10,
            style="Console.TLabelframe",
        )
        visual.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        visual.columnconfigure(0, weight=1)
        visual.columnconfigure(1, weight=1)
        visual_buttons = [
            ("记录张开手", self.capture_open_pose),
            ("记录握拳", self.capture_closed_pose),
            ("记录 ABD 中立", self.capture_abd_neutral),
            ("记录手指横向张开", self.capture_abd_spread),
            ("记录手指并拢", self.capture_abd_together),
            ("保存视觉校准", self.save_visual_profile),
        ]
        for row, (text, command) in enumerate(visual_buttons):
            ttk.Button(visual, text=text, command=command).grid(
                row=row // 2,
                column=row % 2,
                sticky="ew",
                padx=(0, 4) if row % 2 == 0 else (4, 0),
                pady=2,
            )

        profiles = ttk.Frame(visual)
        profiles.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        profiles.columnconfigure(0, weight=1)
        self.profile_combo = ttk.Combobox(
            profiles,
            textvariable=self.profile_var,
            state="readonly",
            width=26,
        )
        self.profile_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(profiles, text="使用", command=self.load_visual_profile).grid(
            row=0,
            column=1,
            padx=(6, 0),
        )

        official = ttk.LabelFrame(
            panel,
            text="Orca Calibration",
            padding=10,
            style="Console.TLabelframe",
        )
        official.grid(row=7, column=0, sticky="ew")
        official.columnconfigure(0, weight=1)
        official.columnconfigure(1, weight=1)
        self.calibrate_button = ttk.Button(
            official,
            text="Calibrate",
            command=self.run_official_calibration,
        )
        self.calibrate_button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)
        self.stop_calibrate_button = ttk.Button(
            official,
            text="停止校准",
            command=self.stop_official_calibration,
            style="Danger.TButton",
        )
        self.stop_calibrate_button.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)

    def _queue_panel_scrollregion_update(self, _event=None) -> None:
        if self._panel_scrollregion_pending:
            return
        self._panel_scrollregion_pending = True
        self.root.after_idle(self._update_panel_scrollregion)

    def _update_panel_scrollregion(self, _event=None) -> None:
        self._panel_scrollregion_pending = False
        self._panel_canvas.configure(scrollregion=self._panel_canvas.bbox("all"))

    def _resize_panel_window(self, event) -> None:
        self._panel_canvas.itemconfigure(self._panel_window, width=event.width)

    def _bind_panel_mousewheel(self, _event=None) -> None:
        self._panel_canvas.bind_all("<MouseWheel>", self._on_panel_mousewheel)

    def _unbind_panel_mousewheel(self, _event=None) -> None:
        self._panel_canvas.unbind_all("<MouseWheel>")

    def _on_panel_mousewheel(self, event) -> None:
        delta = int(-event.delta / 120)
        if delta:
            self._panel_canvas.yview_scroll(delta, "units")

    def start_recognition(self) -> None:
        if self.running:
            return
        self._set_status("Starting recognition: opening camera...")
        self.video_label.configure(text="Opening camera...", image="")
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        try:
            ensure_model(MODEL_PATH)
            self.cap = _open_capture(DEFAULT_CAMERA)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {DEFAULT_CAMERA}")
            initial_frame, initial_read_ms = OrcaRealtimeGui._show_initial_camera_frame(self)
            if callable(getattr(self.cap, "read", None)):
                self.capture_reader = LatestFrameCapture(
                    self.cap,
                    initial_frame=initial_frame,
                    initial_read_ms=initial_read_ms,
                )
                self.capture_reader.start()
            self.landmarker = create_landmarker(
                MODEL_PATH,
                DEFAULT_MAX_DETECTED_HANDS,
                prefer_gpu=DEFAULT_USE_GPU,
            )
            self.delegate_var.set(f"Delegate: {LAST_LANDMARKER_DELEGATE}")
            self.logger = SessionLogger(
                DEFAULT_LOG_DIR,
                time.strftime("orcahand_%Y%m%d_%H%M%S"),
            )
        except Exception as exc:
            self._set_status(f"Start failed: {exc}")
            messagebox.showerror("Start failed", str(exc))
            self._release_recognition_resources()
            return

        self.running = True
        self.frame_count = 0
        self.start_time = time.monotonic()
        if not hasattr(self, "_fps_counter"):
            self._fps_counter = RollingFpsCounter()
        self._fps_counter.reset()
        self.locked_wrist = None
        self.latest_landmarks = None
        self._hand_missing_since = None
        self._set_status("Recognition started.")
        self._update_button_states()
        self._tick()

    def _show_initial_camera_frame(self) -> tuple[np.ndarray | None, float]:
        if self.cap is None:
            return None, 0.0
        read = getattr(self.cap, "read", None)
        if read is None:
            return None, 0.0
        read_start = time.monotonic()
        ok, frame = read()
        read_ms = (time.monotonic() - read_start) * 1000.0
        if not ok:
            return None, read_ms
        if DEFAULT_MIRROR:
            frame = cv2.flip(frame, 1)
        self._show_frame(frame)
        return frame, read_ms

    def stop_recognition(self) -> None:
        self.running = False
        self._stop_live_hardware_output()
        self._release_recognition_resources()
        self.latest_landmarks = None
        self.landmark_smoother.reset()
        self.joint_smoother.reset()
        if not self.controller.connected and self.state.state in (
            RuntimeState.ARMED,
            RuntimeState.LIVE,
            RuntimeState.TRACKING_LOST,
        ):
            self.state.stop_mapping()
        elif self.state.state == RuntimeState.LIVE:
            self.state.disable_live()
        self._set_status("Recognition stopped.")

    def connect_hardware(self) -> None:
        if not self.hardware_allowed:
            messagebox.showwarning(
                "Hardware unavailable",
                "Hardware is disabled in preview-only mode. Run without --preview-only to connect to the real OrcaHand.",
            )
            return
        if not self._background_busy:
            self._run_background("Connecting OrcaHand...", self._connect_hardware_worker)

    def _connect_hardware_worker(self) -> str:
        self.config.validate_for_live()
        self.controller.connect()
        return "OrcaHand connected and moved to neutral."

    def start_mapping(self) -> None:
        if not self._mapping_calibration_ready():
            self._set_status("Calibrate first: open, fist, ABD neutral, spread, together.")
            return
        if not self.hardware_allowed or not self.controller.connected:
            self._set_status("Connect OrcaHand first.")
            return
        if not self.running:
            self.start_recognition()
            if not self.running:
                return
        self.state.start_mapping()
        self.safety.reset_to_safe_neutral()
        self.joint_smoother.reset()
        self.state.enable_live()
        self._set_status("Mapping started. Hardware output enabled. Ramping from neutral.")

    def stop_mapping(self) -> None:
        self._stop_live_hardware_output()
        self.state.stop_mapping()
        self._set_status("Mapping stopped.")

    def enable_output(self) -> None:
        if not self.hardware_allowed or not self.controller.connected:
            self._set_status("Connect OrcaHand first.")
            return
        if self.state.state != RuntimeState.ARMED:
            self._set_status("Start mapping first, then enable output.")
            return
        self.safety.reset_to_safe_neutral()
        self.joint_smoother.reset()
        self.state.enable_live()
        self._set_status("Hardware output enabled. Ramping from neutral.")

    def disable_output(self) -> None:
        self._stop_live_hardware_output()
        if not self.controller.connected and self.state.state in (
            RuntimeState.ARMED,
            RuntimeState.LIVE,
            RuntimeState.TRACKING_LOST,
        ):
            self.state.stop_mapping()
        else:
            self.state.disable_live()
        self._set_status("Hardware output disabled.")

    def move_to_neutral(self) -> None:
        if not self.hardware_allowed or not self.controller.connected:
            self._set_status("Connect OrcaHand first.")
            return
        self.stop_processed_video_playback(update_status=False)
        if self.state.state in (
            RuntimeState.ARMED,
            RuntimeState.LIVE,
            RuntimeState.TRACKING_LOST,
        ):
            self.state.stop_mapping()
        neutral = self.safety.safe_neutral()
        self.safety.reset_to_safe_neutral()
        self.joint_smoother.reset()
        try:
            self.controller.send(neutral)
        except Exception as exc:
            self._safety_stop(f"neutral send failed: {exc}")
            return
        estimate_motor_positions = getattr(self.safety, "estimate_motor_positions", None)
        motor_positions = (
            estimate_motor_positions(neutral)
            if callable(estimate_motor_positions)
            else {}
        )
        summary = SafetyResult(
            joints=dict(neutral),
            accepted=True,
            reasons=[],
            motor_positions=motor_positions,
        )
        self.latest_safety_result = summary
        self._update_safety_telemetry(summary)
        self._set_status("Moved OrcaHand to neutral.")
        self._update_button_states()

    def _stop_live_hardware_output(self) -> None:
        if self.state.state == RuntimeState.LIVE:
            self.controller.emergency_stop()

    def emergency_stop(self) -> None:
        self.stop_processed_video_playback(update_status=False)
        self.state.fault("emergency stop")
        self.controller.emergency_stop()
        self._set_status("Emergency stop: torque disabled.")

    def capture_open_pose(self) -> None:
        if self._require_landmarks("Show a fully open hand first."):
            self.kinematics.capture_open_pose(self.latest_landmarks)
            self._reset_after_visual_capture()
            self._set_status("Open hand pose captured.")

    def capture_closed_pose(self) -> None:
        if self._require_landmarks("Make a fist first."):
            self.kinematics.capture_closed_pose(self.latest_landmarks)
            self._reset_after_visual_capture()
            self._set_status("Fist pose captured.")

    def capture_abd_neutral(self) -> None:
        if self._require_landmarks("Show the ABD neutral pose first."):
            self.kinematics.capture_neutral(self.latest_landmarks)
            self._reset_after_visual_capture()
            self._set_status("ABD neutral pose captured.")

    def capture_abd_spread(self) -> None:
        if self._require_landmarks("Spread fingers sideways first."):
            self.kinematics.capture_abd_spread_pose(self.latest_landmarks)
            self._reset_after_visual_capture()
            self._set_status("ABD spread pose captured.")

    def capture_abd_together(self) -> None:
        if self._require_landmarks("Bring fingers together first."):
            self.kinematics.capture_abd_together_pose(self.latest_landmarks)
            self._reset_after_visual_capture()
            self._set_status("ABD together pose captured.")

    def save_visual_profile(self) -> None:
        if not self._mapping_calibration_ready():
            self._set_status("Complete visual calibration before saving.")
            return
        name = simpledialog.askstring("Save visual calibration", "Profile name:", parent=self.root)
        if not name:
            return
        try:
            path = save_visual_calibration(
                DEFAULT_VISUAL_CALIBRATION_DIR,
                name,
                self.kinematics,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._refresh_profiles()
        self.profile_var.set(path.stem)
        self._set_status(f"Visual calibration saved: {path}")

    def load_visual_profile(self) -> None:
        name = self.profile_var.get()
        if not name:
            self._set_status("Choose a visual calibration profile first.")
            return
        try:
            profile = load_visual_calibration(DEFAULT_VISUAL_CALIBRATION_DIR, name)
            apply_visual_calibration(profile, self.kinematics)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.safety.reset_to_safe_neutral()
        self.joint_smoother.reset()
        self._set_status(f"Visual calibration loaded: {name}")

    def run_official_calibration(self) -> None:
        if not self.hardware_allowed:
            messagebox.showwarning(
                "Hardware unavailable",
                "Hardware is disabled in preview-only mode. Run without --preview-only to calibrate the real OrcaHand.",
            )
            return
        if self._background_busy:
            return
        ok = messagebox.askyesno(
            "Run Orca calibration",
            "This will move the OrcaHand through its mechanical limits and write config/calibration.yaml. Continue?",
        )
        if ok:
            self.state.stop_mapping()
            self._calibration_busy = True
            self._run_background("Running official Orca calibration...", self._official_calibration_worker)

    def _official_calibration_worker(self) -> str:
        try:
            self.controller.calibrate(force_wrist=False, joints=None)
        except CalibrationStopped:
            return "Official Orca calibration stopped."
        return "Official Orca calibration complete. config/calibration.yaml updated."

    def stop_official_calibration(self) -> None:
        self.controller.stop_task()
        self._set_status("Stop calibration requested.")

    def _toggle_motion_controls(self) -> None:
        self.motion_controls_visible = not bool(self.motion_controls_visible)
        OrcaRealtimeGui._sync_motion_controls_visibility(self)

    def _sync_motion_controls_visibility(self) -> None:
        if self.motion_controls_visible:
            self.motion_controls_frame.grid(row=8, column=0, columnspan=2, sticky="ew")
            self.motion_toggle_button.configure(text="Hide Motion Source")
        else:
            self.motion_controls_frame.grid_remove()
            self.motion_toggle_button.configure(text="Show Motion Source")

    def import_video(self) -> None:
        if not OrcaRealtimeGui._video_import_allowed(self):
            self._set_status("Stop live recognition before loading motion.")
            return
        path = filedialog.askopenfilename(
            title="Select hand motion source",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._video_stop_event.clear()
        self._play_video_after_connect = False
        self._video_processing_busy = True
        self._background_busy = True
        self._set_status("Preparing motion stream at 1080p60...")
        visual_state = self.kinematics.export_visual_calibration()
        config = self.config
        settings = self.settings

        def run() -> None:
            try:
                video = process_video_to_1080p60(
                    path,
                    output_dir=DEFAULT_PROCESSED_VIDEO_DIR,
                    config=config,
                    settings=settings,
                    visual_calibration_state=visual_state,
                    stop_event=self._video_stop_event,
                    progress_callback=lambda message: self._background_queue.put(
                        ("video_progress", message)
                    ),
                )
            except VideoProcessingStopped:
                self._background_queue.put(("video_stopped", "Motion preparation stopped."))
            except Exception as exc:
                self._background_queue.put(("video_error", str(exc)))
            else:
                self._background_queue.put(("video_done", video))

        threading.Thread(target=run, daemon=True).start()

    def start_processed_video_playback(self) -> None:
        if self.processed_video is None:
            self._set_status("Load a motion source first.")
            return
        if not self.hardware_allowed:
            self._set_status("Hardware is disabled in preview-only mode.")
            return
        if not self.controller.connected:
            self._play_video_after_connect = True
            if not self._background_busy:
                self._run_background("Motion ready. Connecting OrcaHand...", self._connect_hardware_worker)
            else:
                self._set_status("Motion ready. Waiting for OrcaHand connection...")
            return
        self.stop_processed_video_playback(update_status=False)
        self.video_playback_cap = cv2.VideoCapture(str(self.processed_video.output_path))
        if not self.video_playback_cap.isOpened():
            self.video_playback_cap = None
            self._set_status("Motion stream could not start.")
            return
        self._video_playing = True
        self.video_playback_index = 0
        self.video_playback_start_s = time.monotonic()
        self.safety.reset_to_safe_neutral()
        self.joint_smoother.reset()
        if self.state.state == RuntimeState.PREVIEW:
            self.state.start_mapping()
        if self.state.state == RuntimeState.ARMED:
            self.state.enable_live()
        self._play_video_after_connect = False
        self._set_status("Motion playback started.")
        self._play_processed_video_frame()

    def stop_processed_video_playback(self, completed: bool = False, update_status: bool = True) -> None:
        if self._video_processing_busy:
            self._video_stop_event.set()
            if update_status:
                self._set_status("Stop motion preparation requested.")
        after_id = self._video_after_id
        self._video_after_id = None
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        cap = self.video_playback_cap
        self.video_playback_cap = None
        was_playing = self._video_playing
        self._video_playing = False
        if cap is not None:
            cap.release()
        if was_playing and self.state.state in (
            RuntimeState.ARMED,
            RuntimeState.LIVE,
            RuntimeState.TRACKING_LOST,
        ):
            self._stop_live_hardware_output()
            self.state.stop_mapping()
        if update_status and was_playing:
            self._set_status("Motion playback complete." if completed else "Motion playback stopped.")
        else:
            self._update_button_states()

    def _play_processed_video_frame(self) -> None:
        if not self._video_playing or self.video_playback_cap is None:
            return
        ok, frame = self.video_playback_cap.read()
        if not ok:
            self.stop_processed_video_playback(completed=True)
            return
        try:
            self._show_frame(frame)
            OrcaRealtimeGui._send_processed_video_command(self, self.video_playback_index)
        except Exception as exc:
            self.stop_processed_video_playback(update_status=False)
            self._safety_stop(f"motion playback failed: {exc}")
            return

        self.video_playback_index += 1
        fps = float(self.processed_video.fps) if self.processed_video else DEFAULT_VIDEO_OUTPUT_FPS
        target_elapsed = self.video_playback_index / max(1.0, fps)
        delay_ms = max(
            1,
            int((self.video_playback_start_s + target_elapsed - time.monotonic()) * 1000.0),
        )
        self._video_after_id = self.root.after(delay_ms, self._play_processed_video_frame)

    def _tick(self) -> None:
        """Process one realtime frame and schedule the next frame by FPS budget."""

        if not self.running or self.cap is None or self.landmarker is None:
            return
        try:
            tick_start = time.monotonic()
            ok, frame, read_ms = OrcaRealtimeGui._read_realtime_frame(self)
            if not ok:
                self.state.fault("camera read failed")
                self.stop_recognition()
                return
            if DEFAULT_MIRROR:
                frame = cv2.flip(frame, 1)

            timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
            inference_start = time.monotonic()
            keypoints, scores, labels = detect_frame(
                self.landmarker,
                frame,
                max(timestamp_ms, self.frame_count),
            )
            inference_ms = (time.monotonic() - inference_start) * 1000.0
            selected_hands = select_one_hand(
                keypoints,
                scores,
                labels,
                self.locked_wrist,
                1,
                preferred_handedness=self.config.hand_type,
            )
            if selected_hands:
                hand = selected_hands[0]
                self.locked_wrist = hand["wrist"]
                self._hand_missing_since = None
                self.latest_landmarks = np.asarray(hand["keypoints"], dtype=float)
                self.hand_var.set(f"Hand: {hand['label']}")
                self.tracking_card_var.set(str(hand["label"]).upper())
                draw_hand(
                    frame,
                    hand["keypoints"],
                    hand["scores"],
                    hand["label"],
                    draw_label=False,
                )
                self._process_landmarks(self.latest_landmarks)
            else:
                OrcaRealtimeGui._handle_no_hand(self, now_s=time.monotonic())

            display_start = time.monotonic()
            self._show_frame(frame)
            display_ms = (time.monotonic() - display_start) * 1000.0
            frame_end = time.monotonic()
            frame_ms = (frame_end - tick_start) * 1000.0
            instant_fps, average_fps = self._fps_counter.update(frame_end)
            if self._should_update_perf_telemetry(frame_end):
                fps_text = f"FPS: {instant_fps:.1f} (avg {average_fps:.1f})"
                self.fps_var.set(fps_text)
                self.preview_fps_var.set(fps_text.upper())
                self.inference_var.set(
                    "Inference: "
                    f"{inference_ms:.1f} ms | camera(bg) {read_ms:.1f} | "
                    f"show {display_ms:.1f} | frame {frame_ms:.1f}"
                )
            self.frame_count += 1
            self._update_button_states()
            self.root.after(
                _next_frame_delay_ms(
                    elapsed_s=time.monotonic() - tick_start,
                    target_fps=DEFAULT_CAPTURE_FPS,
                ),
                self._tick,
            )
        except Exception as exc:
            OrcaRealtimeGui._handle_runtime_error(self, exc)

    def _read_realtime_frame(self) -> tuple[bool, np.ndarray | None, float]:
        reader = getattr(self, "capture_reader", None)
        if reader is not None:
            return reader.latest()
        if self.cap is None:
            return False, None, 0.0
        read_start = time.monotonic()
        ok, frame = self.cap.read()
        read_ms = (time.monotonic() - read_start) * 1000.0
        return bool(ok), frame, read_ms

    def _handle_runtime_error(self, exc: Exception) -> None:
        message = str(exc) or exc.__class__.__name__
        if self.state.state == RuntimeState.LIVE:
            try:
                self._safety_stop(f"runtime error: {message}")
            except Exception:
                self.state.fault(f"runtime error: {message}")
        else:
            self.state.fault(f"runtime error: {message}")
        self.running = False
        self._release_recognition_resources()
        self._set_status(f"Runtime stopped: {message}")

    def _handle_no_hand(self, *, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else float(now_s)
        missing_since = getattr(self, "_hand_missing_since", None)
        if missing_since is None:
            self._hand_missing_since = now
            missing_since = now
        missing_s = max(0.0, now - float(missing_since))

        self.latest_landmarks = None
        self.latest_safety_result = None
        self.hand_var.set("Hand: none")
        self.tracking_card_var.set("NO HAND")
        self.joint_angles_var.set("Angles: --")
        self.safety_var.set("Safety: no hand detected")
        self.safety_card_var.set("NO HAND")
        self.preview_safety_var.set("Safety: no hand")
        self.landmark_smoother.reset()
        self.joint_smoother.reset()

        if self.state.state == RuntimeState.LIVE:
            self.state.tracking_lost("no hand detected")
        elif self.state.state == RuntimeState.TRACKING_LOST:
            self.state.reason = "no hand detected"

        if (
            self.state.state == RuntimeState.TRACKING_LOST
            and missing_s >= DEFAULT_TRACKING_LOST_STOP_S
        ):
            self._safety_stop("no hand detected")

    def _process_landmarks(self, landmarks: np.ndarray) -> None:
        """Map landmarks to safe joint commands; UI text is throttled separately."""

        if hasattr(self, "_hand_missing_since"):
            self._hand_missing_since = None
        smoothed_landmarks = self.landmark_smoother.update(landmarks)
        raw_joints = self.kinematics.estimate(smoothed_landmarks)
        smoothed_joints = self.joint_smoother.update(raw_joints)
        safety_result = self.safety.apply(smoothed_joints)
        self.latest_safety_result = safety_result

        if self._should_update_telemetry(time.monotonic()) or not safety_result.accepted:
            self._update_safety_telemetry(safety_result)

        if not safety_result.accepted:
            reason = "; ".join(safety_result.reasons) or "safety rejected frame"
            if self.state.state == RuntimeState.LIVE:
                self._safety_stop(reason)
            else:
                self.state.reason = reason
            OrcaRealtimeGui._write_safety_log(self, safety_result)
            return
        if self.state.state == RuntimeState.TRACKING_LOST:
            self.state.recover_tracking()
        elif self.state.state in (RuntimeState.PREVIEW, RuntimeState.ARMED):
            self.state.reason = ""
        if self.state.can_send_to_hardware:
            try:
                self.controller.send(safety_result.joints)
            except Exception as exc:
                self._safety_stop(f"hardware send failed: {exc}")
        OrcaRealtimeGui._write_safety_log(self, safety_result)

    def _write_safety_log(self, safety_result: SafetyResult) -> None:
        if self.logger is None:
            return
        self.logger.write(
            {
                "timestamp": time.time(),
                "state": self.state.state.value,
                "state_reason": self.state.reason,
                "accepted": safety_result.accepted,
                "reasons": safety_result.reasons,
                "joints": safety_result.joints,
                "motor_positions": safety_result.motor_positions,
            }
        )

    def _send_processed_video_command(self, frame_index: int) -> bool:
        video = getattr(self, "processed_video", None)
        if video is None or frame_index < 0 or frame_index >= len(video.frames):
            return False
        frame = video.frames[frame_index]
        if not frame.joints:
            return False
        if not self.hardware_allowed or not self.controller.connected:
            return False
        state = getattr(self, "state", None)
        if state is not None and not state.can_send_to_hardware:
            return False
        self.controller.send(frame.joints)
        summary = SafetyResult(
            joints=dict(frame.joints),
            accepted=True,
            reasons=[],
            motor_positions={},
        )
        OrcaRealtimeGui._update_safety_telemetry(self, summary)
        return True

    def _should_update_telemetry(self, now_s: float) -> bool:
        if DEFAULT_UI_TELEMETRY_FPS <= 0:
            return True
        if now_s < self._next_telemetry_update:
            return False
        self._next_telemetry_update = now_s + 1.0 / float(DEFAULT_UI_TELEMETRY_FPS)
        return True

    def _should_update_perf_telemetry(self, now_s: float) -> bool:
        if DEFAULT_PERF_TELEMETRY_FPS <= 0:
            return True
        next_update = getattr(self, "_next_perf_telemetry_update", 0.0)
        if now_s < next_update:
            return False
        self._next_perf_telemetry_update = (
            now_s + 1.0 / float(DEFAULT_PERF_TELEMETRY_FPS)
        )
        return True

    def _update_safety_telemetry(self, safety_result: SafetyResult) -> None:
        self.joint_angles_var.set(_format_joint_summary(safety_result))
        safety_text = _format_safety_summary(safety_result)
        self.safety_var.set(safety_text)
        self.safety_card_var.set("ACCEPTED" if safety_result.accepted else "BLOCKED")
        self.preview_safety_var.set(safety_text)

    def _show_frame(self, frame: np.ndarray) -> None:
        """Resize and convert one frame for Tk display."""

        max_width = DEFAULT_DISPLAY_MAX_WIDTH
        max_height = DEFAULT_DISPLAY_MAX_HEIGHT
        label_width = int(self.video_label.winfo_width())
        label_height = int(self.video_label.winfo_height())
        target_width, target_height = _display_target_size(
            label_width=label_width,
            label_height=label_height,
            max_width=max_width,
            max_height=max_height,
        )
        frame = _resize_to_fill(
            frame,
            target_width=target_width,
            target_height=target_height,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo = photo
        self.video_label.configure(image=photo, text="")

    def _release_recognition_resources(self) -> None:
        if self.capture_reader is not None:
            self.capture_reader.stop()
            self.capture_reader = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            close = getattr(self.landmarker, "close", None)
            if callable(close):
                close()
            self.landmarker = None
        if self.logger is not None:
            self.logger.close()
            self.logger = None

    def _require_landmarks(self, message: str) -> bool:
        if self.latest_landmarks is None:
            self._set_status(message)
            return False
        return True

    def _reset_after_visual_capture(self) -> None:
        if self.state.state in (RuntimeState.ARMED, RuntimeState.LIVE):
            self.state.stop_mapping()
        self.safety.reset_to_safe_neutral()
        self.joint_smoother.reset()
        self._refresh_calibration_status()
        self._update_button_states()

    def _mapping_calibration_ready(self) -> bool:
        return bool(self.kinematics.has_range_calibration) and bool(
            self.kinematics.has_abd_range_calibration
        )

    def _visual_calibration_status(self) -> str:
        flex = "ready" if self.kinematics.has_range_calibration else "missing flex"
        abd = "ready" if self.kinematics.has_abd_range_calibration else "missing abd"
        neutral = "ready" if self.kinematics.has_neutral else "missing neutral"
        return f"visual calibration: flex {flex}, abd {abd}, neutral {neutral}"

    def _refresh_calibration_status(self) -> None:
        result = CalibrationResult.from_calibration_path(
            self.config.calibration_path,
            self.config.motor_ids,
        )
        if not self.config.calibration_path.exists():
            hardware = "hardware calibration: missing config/calibration.yaml"
        else:
            hardware = f"hardware calibrated:{result.calibrated} wrist:{result.wrist_calibrated}"
        self.calibration_var.set(f"{hardware}\n{self._visual_calibration_status()}")

    def _refresh_profiles(self) -> None:
        profiles = list_visual_calibrations(DEFAULT_VISUAL_CALIBRATION_DIR)
        names = [profile.name for profile in profiles]
        self.profile_combo.configure(values=names)
        if names and self.profile_var.get() not in names:
            self.profile_var.set(names[0])

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        self._refresh_calibration_status()
        self._update_button_states()

    def _run_background(self, start_message: str, worker) -> None:
        self._background_busy = True
        self._set_status(start_message)

        def run() -> None:
            try:
                message = worker()
            except Exception as exc:
                self._background_queue.put(("error", str(exc)))
            else:
                self._background_queue.put(("done", message))

        threading.Thread(target=run, daemon=True).start()

    def _poll_background_queue(self) -> None:
        while True:
            try:
                kind, message = self._background_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "video_progress":
                self._set_status(str(message))
                continue
            if kind == "video_done":
                self._background_busy = False
                self._video_processing_busy = False
                self._video_stop_event.clear()
                self._handle_processed_video_ready(message)
                continue
            if kind == "video_stopped":
                self._background_busy = False
                self._video_processing_busy = False
                self._video_stop_event.clear()
                self._set_status(str(message))
                continue
            if kind == "video_error":
                self._background_busy = False
                self._video_processing_busy = False
                self._video_stop_event.clear()
                self._play_video_after_connect = False
                self._set_status(str(message))
                messagebox.showerror("Motion setup failed", str(message))
                continue
            self._background_busy = False
            self._calibration_busy = False
            if kind == "error":
                self._play_video_after_connect = False
                self._set_status(message)
                messagebox.showerror("OrcaHand", message)
            else:
                self._handle_background_done(message)
        self.root.after(100, self._poll_background_queue)

    def _handle_processed_video_ready(self, video: ProcessedVideo) -> None:
        self.processed_video = video
        if self.controller.connected:
            self.start_processed_video_playback()
            return
        self._play_video_after_connect = True
        if getattr(self, "hardware_allowed", True):
            self._run_background("Motion ready. Connecting OrcaHand...", self._connect_hardware_worker)
        else:
            self._set_status("Motion ready. Hardware is disabled in preview-only mode.")

    def _handle_background_done(self, message: str) -> None:
        self._reload_runtime_config()
        controller = getattr(self, "controller", None)
        state = getattr(self, "state", None)
        if (
            controller is not None
            and state is not None
            and controller.connected
            and state.state == RuntimeState.FAULT
        ):
            state.reset_fault()
        self._set_status(message)
        if self._play_video_after_connect:
            self._play_video_after_connect = False
            self.start_processed_video_playback()

    def _reload_runtime_config(self) -> None:
        visual_state = self.kinematics.export_visual_calibration()
        self.config = load_realtime_config(DEFAULT_CONFIG_DIR)
        self.settings = self.config.runtime_safety
        self.safety = SafetyController(self.config, self.settings)
        self.kinematics = HandKinematics(self.config, self.safety.safe_neutral())
        self.kinematics.import_visual_calibration(visual_state)
        self.joint_smoother.reset()
        self.controller.config_path = self.config.config_path
        self._refresh_calibration_status()

    def _safety_stop(self, reason: str) -> None:
        self.state.fault(f"safety stop: {reason}")
        self.controller.emergency_stop()
        self.safety_var.set(f"Safety: STOP - {reason}")
        if hasattr(self, "safety_card_var"):
            self.safety_card_var.set("STOP")
        if hasattr(self, "preview_safety_var"):
            self.preview_safety_var.set("Safety: STOP")
        self._set_status(f"SAFETY STOP: {reason}")

    def _video_import_allowed(self) -> bool:
        return bool(
            self.hardware_allowed
            and not self.running
            and not self._background_busy
            and not getattr(self, "_video_processing_busy", False)
            and not getattr(self, "_video_playing", False)
            and self.state.state == RuntimeState.PREVIEW
        )

    def _update_button_states(self) -> None:
        state_text = self.state.state.value
        connected = self.controller.connected
        snapshot = (
            bool(self.running),
            bool(self.hardware_allowed),
            bool(connected),
            bool(self._background_busy),
            bool(getattr(self, "_calibration_busy", False)),
            bool(getattr(self, "_video_processing_busy", False)),
            bool(getattr(self, "_video_playing", False)),
            state_text,
        )
        if getattr(self, "_button_state_snapshot", None) == snapshot:
            return
        self._button_state_snapshot = snapshot

        self.state_var.set(f"state: {state_text}")
        if hasattr(self, "state_card_var"):
            self.state_card_var.set(state_text.upper())
        if hasattr(self, "preview_state_var"):
            self.preview_state_var.set(f"STATE {state_text.upper()}")
        self.hardware_var.set("hardware: connected" if connected else "hardware: disconnected")
        if hasattr(self, "hardware_card_var"):
            self.hardware_card_var.set("CONNECTED" if connected else "DISCONNECTED")
        self.start_button.configure(state="disabled" if self.running else "normal")
        self.stop_button.configure(state="normal" if self.running else "disabled")
        self.connect_button.configure(
            state="normal"
            if self.hardware_allowed and not connected and not self._background_busy
            else "disabled"
        )
        if hasattr(self, "neutral_button"):
            self.neutral_button.configure(
                state="normal"
                if self.hardware_allowed and connected and not self._background_busy
                else "disabled"
            )
        self.map_button.configure(
            state="normal"
            if self.state.state == RuntimeState.PREVIEW
            else "disabled"
        )
        self.stop_map_button.configure(
            state="normal"
            if self.state.state in (RuntimeState.ARMED, RuntimeState.LIVE)
            else "disabled"
        )
        self.output_button.configure(
            state="normal" if connected and self.state.state == RuntimeState.ARMED else "disabled"
        )
        self.stop_output_button.configure(
            state="normal" if self.state.state == RuntimeState.LIVE else "disabled"
        )
        self.calibrate_button.configure(
            state="normal" if self.hardware_allowed and not self._background_busy else "disabled"
        )
        self.stop_calibrate_button.configure(
            state="normal" if getattr(self, "_calibration_busy", False) else "disabled"
        )
        if hasattr(self, "import_video_button"):
            self.import_video_button.configure(
                state="normal"
                if OrcaRealtimeGui._video_import_allowed(self)
                else "disabled"
            )
        if hasattr(self, "stop_video_button"):
            self.stop_video_button.configure(
                state="normal"
                if getattr(self, "_video_processing_busy", False)
                or getattr(self, "_video_playing", False)
                else "disabled"
            )

    def close(self) -> None:
        self.running = False
        self.stop_processed_video_playback(update_status=False)
        self._release_recognition_resources()
        self.controller.disconnect()
        self.root.destroy()


def _format_joint_summary(safety_result: SafetyResult | None) -> str:
    if safety_result is None or not safety_result.joints:
        return "Angles: --"
    parts = [
        f"{joint}:{float(safety_result.joints[joint]):.1f}"
        for joint in JOINT_DISPLAY_ORDER
        if joint in safety_result.joints
    ]
    return "Angles: " + " | ".join(parts)


def _format_safety_summary(safety_result: SafetyResult | None) -> str:
    if safety_result is None:
        return "Safety: --"
    if safety_result.accepted:
        return "Safety: accepted"
    if safety_result.reasons:
        return "Safety: " + "; ".join(safety_result.reasons)
    return "Safety: rejected"


def _resize_to_fill(
    frame: np.ndarray,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize with center crop; reuse exact-size frames to save display work."""

    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if width == target_width and height == target_height:
        return frame

    scale = max(target_width / width, target_height / height)
    resized_width = max(target_width, int(np.ceil(width * scale)))
    resized_height = max(target_height, int(np.ceil(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x0 = max(0, (resized_width - target_width) // 2)
    y0 = max(0, (resized_height - target_height) // 2)
    return resized[y0 : y0 + target_height, x0 : x0 + target_width]


def _resize_for_display(
    frame: np.ndarray,
    *,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return frame
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _letterbox_to_size(
    frame: np.ndarray,
    *,
    target_width: int = DEFAULT_VIDEO_OUTPUT_WIDTH,
    target_height: int = DEFAULT_VIDEO_OUTPUT_HEIGHT,
) -> np.ndarray:
    """Resize with black padding, preserving the complete source frame."""

    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    height, width = frame.shape[:2]
    channels = frame.shape[2] if frame.ndim == 3 else 1
    if height <= 0 or width <= 0:
        return np.zeros((target_height, target_width, channels), dtype=frame.dtype)

    scale = min(float(target_width) / float(width), float(target_height) / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((target_height, target_width, channels), dtype=frame.dtype)
    x0 = (target_width - new_width) // 2
    y0 = (target_height - new_height) // 2
    output[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return output


def _display_target_size(
    *,
    label_width: int,
    label_height: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int]:
    width = int(label_width) if int(label_width) > 1 else DEFAULT_DISPLAY_MAX_WIDTH
    height = int(label_height) if int(label_height) > 1 else DEFAULT_DISPLAY_MAX_HEIGHT
    if width <= 1:
        width = 640
    if height <= 1:
        height = 360
    if max_width > 0:
        width = min(width, int(max_width))
    if max_height > 0:
        height = min(height, int(max_height))
    return max(1, width), max(1, height)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file realtime MediaPipe hand tracking to OrcaHand joint commands."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        dest="live",
        default=True,
        help="Allow OrcaHand hardware connection. This is enabled by default.",
    )
    parser.add_argument(
        "--preview-only",
        "--no-live",
        action="store_false",
        dest="live",
        help="Disable OrcaHand hardware connection and only preview tracking.",
    )
    return parser


def run_gui(args: argparse.Namespace) -> int:
    root = tk.Tk()
    OrcaRealtimeGui(root, args)
    root.mainloop()
    return 0


def main() -> int:
    return run_gui(build_arg_parser().parse_args())


def run_cv2_window(args: argparse.Namespace) -> int:
    ensure_model(MODEL_PATH)

    config = load_realtime_config(DEFAULT_CONFIG_DIR)
    hardware_enabled = args.live
    if hardware_enabled:
        config.validate_for_live()

    settings = config.runtime_safety
    safety = SafetyController(config, settings)
    kinematics = HandKinematics(config, safety.safe_neutral())
    landmark_smoother = ExponentialSmoother(alpha=0.45)
    joint_smoother = JointSmoother(default_alpha=0.35, abd_alpha=0.25)
    state = RuntimeStateMachine()
    logger = SessionLogger(DEFAULT_LOG_DIR, time.strftime("orcahand_%Y%m%d_%H%M%S"))
    controller = OrcaController(
        config.config_path,
        live=hardware_enabled,
        orca_core_root=DEFAULT_ORCA_CORE_ROOT,
        mock=False,
        force_calibrate=DEFAULT_FORCE_CALIBRATE,
        move_to_neutral=DEFAULT_MOVE_TO_NEUTRAL_ON_CONNECT,
        send_num_steps=DEFAULT_SEND_STEPS,
        send_step_size=DEFAULT_SEND_STEP_SIZE,
    )

    if hardware_enabled:
        controller.connect()

    cap = _open_capture(DEFAULT_CAMERA)
    if not cap.isOpened():
        print(f"Could not open camera {DEFAULT_CAMERA}.")
        logger.close()
        controller.disconnect()
        return 1

    start_time = time.monotonic()
    frame_count = 0
    locked_wrist = None
    latest_landmarks: np.ndarray | None = None

    try:
        with create_landmarker(
            MODEL_PATH,
            DEFAULT_MAX_DETECTED_HANDS,
            prefer_gpu=DEFAULT_USE_GPU,
        ) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    state.fault("camera read failed")
                    break
                if DEFAULT_MIRROR:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                keypoints, scores, labels = detect_frame(
                    landmarker,
                    frame,
                    max(timestamp_ms, frame_count),
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

                    _react_to_safety_result(
                        state,
                        controller,
                        safety_result,
                        hardware_enabled,
                    )

                    if state.can_send_to_hardware and safety_result.accepted:
                        try:
                            controller.send(safety_result.joints)
                        except Exception as exc:
                            _print_status(f"Hardware send failed: {exc}", "red")
                            state.fault(str(exc))
                            controller.emergency_stop()

                    logger.write(
                        {
                            "timestamp": time.time(),
                            "state": state.state.value,
                            "state_reason": state.reason,
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
                        _safety_stop(state, controller, "no hand detected")

                _draw_status(frame, state, hardware_enabled, safety_result, kinematics)

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
        _print_status("Quit requested. Stopping program.", "yellow")
        return True
    if state.state == RuntimeState.FAULT and key in (
        ord("m"),
        ord("l"),
        ord("n"),
        ord("o"),
        ord("c"),
        ord("a"),
        ord("s"),
    ):
        _print_status(f"FAULT active: {state.reason}. Restart the program to continue.", "red")
        return False
    if key == ord("o"):
        if latest_landmarks is None:
            _print_status("No hand detected. Show a fully open hand, then press 'o'.", "yellow")
            return False
        kinematics.capture_open_pose(latest_landmarks)
        _reset_after_calibration_capture(state, safety, joint_smoother)
        _print_status(
            "Open hand calibration captured. Make a tight fist and press 'c'.",
            "cyan",
        )
    elif key == ord("c"):
        if latest_landmarks is None:
            _print_status("No hand detected. Make a tight fist, then press 'c'.", "yellow")
            return False
        kinematics.capture_closed_pose(latest_landmarks)
        _reset_after_calibration_capture(state, safety, joint_smoother)
        if _range_calibration_ready(kinematics):
            _print_status(
                "Fist calibration captured. Flex calibration complete. Spread fingers and press 'a'.",
                "green",
            )
        else:
            _print_status(
                "Fist calibration captured. Fully open your hand and press 'o'.",
                "cyan",
            )
    elif key == ord("a"):
        if latest_landmarks is None:
            _print_status("No hand detected. Spread fingers sideways, then press 'a'.", "yellow")
            return False
        kinematics.capture_abd_spread_pose(latest_landmarks)
        _reset_after_calibration_capture(state, safety, joint_smoother)
        _print_status(
            "Spread calibration captured. Bring fingers together and press 's'.",
            "cyan",
        )
    elif key == ord("s"):
        if latest_landmarks is None:
            _print_status("No hand detected. Bring fingers together, then press 's'.", "yellow")
            return False
        kinematics.capture_abd_together_pose(latest_landmarks)
        _reset_after_calibration_capture(state, safety, joint_smoother)
        if _mapping_calibration_ready(kinematics):
            _print_status(
                "Together calibration captured. All calibration complete. Press 'm' to arm mapping.",
                "green",
            )
        else:
            _print_status(
                "Together calibration captured. Finish flex calibration with 'o' and 'c'.",
                "cyan",
            )
    elif key == ord("m"):
        if state.state == RuntimeState.PREVIEW:
            if not _mapping_calibration_ready(kinematics):
                _print_status(
                    "Calibrate first: open 'o', fist 'c', side-spread 'a', side-together 's'.",
                    "yellow",
                )
                return False
            state.start_mapping()
            _print_status(
                "Mapping armed. OrcaHand stays at mechanical neutral until 'l'.",
                "cyan",
            )
        else:
            state.stop_mapping()
            _print_status("Mapping stopped. Hardware output is off.", "yellow")
    elif key == ord("l"):
        if state.state != RuntimeState.LIVE and not _mapping_calibration_ready(kinematics):
            _print_status(
                "Calibrate first: open 'o', fist 'c', side-spread 'a', side-together 's'.",
                "yellow",
            )
        elif live_allowed and state.state == RuntimeState.ARMED:
            safety.reset_to_safe_neutral()
            if joint_smoother is not None:
                joint_smoother.reset()
            state.enable_live()
            _print_status(
                "Hardware output enabled. Ramping from mechanical neutral.",
                "green",
            )
        elif state.state == RuntimeState.LIVE:
            state.disable_live()
            _print_status("Hardware output disabled. Preview remains active.", "yellow")
        elif not live_allowed:
            _print_status("Hardware output is disabled in preview-only mode.", "red")
        else:
            _print_status("Press 'm' first to arm mapping, then press 'l'.", "yellow")
    elif key == ord("n") and latest_landmarks is not None:
        _capture_visual_neutral(kinematics, latest_landmarks, joint_smoother)
        safety.reset_to_safe_neutral()
        _print_status("Captured current hand pose as abduction neutral.", "cyan")
    elif key in (27, 32):
        state.fault("emergency stop")
        controller.emergency_stop()
        _print_status("Emergency stop: torque disabled.", "red")
    return False


def _react_to_safety_result(
    state: RuntimeStateMachine,
    controller: OrcaController,
    safety_result: SafetyResult,
    live_allowed: bool,
) -> None:
    if safety_result.accepted:
        if state.state == RuntimeState.TRACKING_LOST:
            state.recover_tracking()
        elif state.state in (RuntimeState.PREVIEW, RuntimeState.ARMED):
            state.reason = ""
        return

    reason = "; ".join(safety_result.reasons) or "safety rejected frame"
    if live_allowed and state.state == RuntimeState.LIVE:
        _safety_stop(state, controller, reason)
    elif state.state == RuntimeState.TRACKING_LOST:
        state.reason = reason
    elif state.state in (RuntimeState.PREVIEW, RuntimeState.ARMED):
        state.reason = reason


def _safety_stop(
    state: RuntimeStateMachine,
    controller: OrcaController,
    reason: str,
) -> None:
    state.fault(f"safety stop: {reason}")
    controller.emergency_stop()
    _print_status(f"SAFETY STOP: {reason}", "red")


def _capture_visual_neutral(
    kinematics: HandKinematics,
    latest_landmarks: np.ndarray,
    joint_smoother=None,
) -> None:
    kinematics.capture_neutral(latest_landmarks)
    if joint_smoother is not None:
        joint_smoother.reset()


def _reset_after_calibration_capture(
    state: RuntimeStateMachine,
    safety: SafetyController,
    joint_smoother=None,
) -> None:
    if state.state in (RuntimeState.ARMED, RuntimeState.LIVE):
        state.stop_mapping()
    safety.reset_to_safe_neutral()
    if joint_smoother is not None:
        joint_smoother.reset()


def _range_calibration_ready(kinematics: HandKinematics) -> bool:
    return bool(getattr(kinematics, "has_range_calibration", True))


def _abd_range_calibration_ready(kinematics: HandKinematics) -> bool:
    return bool(getattr(kinematics, "has_abd_range_calibration", True))


def _mapping_calibration_ready(kinematics: HandKinematics) -> bool:
    return _range_calibration_ready(kinematics) and _abd_range_calibration_ready(
        kinematics
    )


def _range_calibration_status(kinematics: HandKinematics | None) -> str:
    if kinematics is None:
        return "unknown"
    status = getattr(kinematics, "range_calibration_status", None)
    flex_status = str(status()) if callable(status) else (
        "ready" if _range_calibration_ready(kinematics) else "press o/c"
    )
    abd_status = "ready" if _abd_range_calibration_ready(kinematics) else "press a/s"
    return f"flex:{flex_status} abd:{abd_status}"


def _draw_status(
    frame: np.ndarray,
    state: RuntimeStateMachine,
    live_allowed: bool,
    safety_result,
    kinematics: HandKinematics | None = None,
) -> None:
    color = (0, 255, 0)
    if state.state in (RuntimeState.TRACKING_LOST, RuntimeState.FAULT):
        color = (0, 0, 255)

    for index, line in enumerate(
        _status_lines(state, live_allowed, safety_result, kinematics)
    ):
        cv2.putText(
            frame,
            line,
            (16, 32 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 if index else 0.7,
            color if index != 2 else ((0, 255, 0) if state.can_send_to_hardware else (0, 180, 255)),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    if state.reason:
        cv2.putText(
            frame,
            state.reason[:90],
            (16, 120),
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


def _status_lines(
    state: RuntimeStateMachine,
    live_allowed: bool,
    safety_result,
    kinematics: HandKinematics | None = None,
) -> list[str]:
    output_state = "OUTPUT ON" if state.can_send_to_hardware else "OUTPUT OFF"
    lines = [
        f"state:{state.state.value} live_allowed:{live_allowed} calib:{_range_calibration_status(kinematics)}",
        "o:open | c:fist | n:abd neutral | a:spread | s:together | m:start | l:output | q:quit",
        output_state,
    ]
    if safety_result is not None and safety_result.reasons:
        lines.append("safety: " + "; ".join(safety_result.reasons[:2]))
    return lines


def _color_text(text: str, color: str) -> str:
    prefix = ANSI_COLORS.get(color, "")
    suffix = ANSI_COLORS["reset"] if prefix else ""
    return f"{prefix}{text}{suffix}"


def _print_status(message: str, color: str = "cyan") -> None:
    print(_color_text(message, color), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
