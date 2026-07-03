from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CalibrationResult:
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


def _read_raw_value(raw: dict[Any, Any], key: int, default: Any) -> Any:
    if key in raw:
        return raw[key]
    text_key = str(key)
    if text_key in raw:
        return raw[text_key]
    return default
