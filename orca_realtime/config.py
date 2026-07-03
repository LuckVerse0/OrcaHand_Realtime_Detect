from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OFFSET_DEG = 3.0
DEFAULT_ABD_OFFSET_DEG = 2.0


@dataclass(frozen=True)
class RuntimeSafetySettings:
    default_offset_deg: float = DEFAULT_OFFSET_DEG
    max_delta_deg_per_frame: float = 5.0
    startup_max_delta_deg_per_frame: float = 1.0
    startup_ramp_frames: int = 30
    abrupt_delta_deg: float = 120.0
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

    @property
    def motor_to_joint_map(self) -> dict[int, str]:
        return {motor_id: joint for joint, motor_id in self.joint_to_motor_map.items()}

    def validate_for_live(self) -> None:
        if not self.calibrated:
            raise ValueError("calibration.yaml says calibrated is false")
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


def load_realtime_config(config_dir: str | Path) -> RealtimeConfig:
    config_dir = Path(config_dir).resolve()
    config_path = config_dir / "config.yaml"
    calibration_path = config_dir / "calibration.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = yaml.safe_load(file) or {}

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
    )


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
    return {
        int(key): (float(values[0]), float(values[1]))
        for key, values in (raw_values or {}).items()
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
