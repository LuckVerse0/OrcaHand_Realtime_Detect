from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_ABD_OFFSET_DEG, DEFAULT_OFFSET_DEG, RealtimeConfig


FINGER_STARTS = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

DEFAULT_FLEX_DEADZONE_DEG = 20.0
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


@dataclass
class HandKinematics:
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

        for finger, start in FINGER_STARTS.items():
            curl = _finger_curl(points, start)
            raw[f"{finger}_mcp"] = 0.55 * curl
            raw[f"{finger}_pip"] = curl
            raw[f"{finger}_abd"] = _finger_abduction(points, start)

        thumb_mcp_flex = _joint_flexion(points[1], points[2], points[3])
        thumb_curl = _joint_flexion(points[2], points[3], points[4])
        raw["thumb_mcp"] = thumb_mcp_flex
        raw["thumb_abd"] = _thumb_abduction(points)
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


def _finger_abduction(points: np.ndarray, start: int) -> float:
    palm_forward = points[9] - points[0]
    palm_side = points[17] - points[5]
    forward = _unit_vector(palm_forward)
    side = palm_side - np.dot(palm_side, forward) * forward
    side = _unit_vector(side)
    if np.linalg.norm(forward) <= 1e-9 or np.linalg.norm(side) <= 1e-9:
        return 0.0

    finger_axis = _unit_vector(points[start + 1] - points[start])
    if np.linalg.norm(finger_axis) <= 1e-9:
        return 0.0
    side_component = float(np.dot(finger_axis, side))
    forward_component = float(np.dot(finger_axis, forward))
    return float(np.degrees(np.arctan2(side_component, forward_component)))


def _thumb_abduction(points: np.ndarray) -> float:
    palm_forward = points[9] - points[0]
    palm_side = points[17] - points[5]
    forward = _unit_vector(palm_forward)
    side = palm_side - np.dot(palm_side, forward) * forward
    side = _unit_vector(side)
    normal = _unit_vector(np.cross(side, forward))
    if (
        np.linalg.norm(forward) <= 1e-9
        or np.linalg.norm(side) <= 1e-9
        or np.linalg.norm(normal) <= 1e-9
    ):
        thumb_axis = points[4] - points[1]
        index_axis = points[8] - points[5]
        return -_signed_angle_2d(index_axis, thumb_axis)

    index_ray = _project_to_plane(points[6] - points[5], normal)
    if np.linalg.norm(index_ray) <= 1e-9:
        index_ray = forward
    thumb_ray = _project_to_plane(points[4] - points[5], normal)
    if np.linalg.norm(thumb_ray) <= 1e-9:
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
