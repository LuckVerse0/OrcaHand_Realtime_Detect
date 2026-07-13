from __future__ import annotations

from pathlib import Path

import realtime_orcahand as rt


CalibrationResult = rt.CalibrationResult


def test_calibration_result_loads_yaml_with_integer_motor_ids(tmp_path: Path):
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        """
calibrated: true
wrist_calibrated: false
motor_limits:
  1: [0.1, 1.2]
  "2": [0.2, 1.4]
joint_to_motor_ratios:
  1: 0.5
  "2": 0.75
""",
        encoding="utf-8",
    )

    result = CalibrationResult.from_calibration_path(calibration_path, [1, 2, 3])

    assert result.calibrated is True
    assert result.wrist_calibrated is False
    assert result.motor_limits_dict == {1: [0.1, 1.2], 2: [0.2, 1.4], 3: [None, None]}
    assert result.joint_to_motor_ratios_dict == {1: 0.5, 2: 0.75, 3: 0.0}


def test_calibration_result_empty_marks_all_motors_uncalibrated():
    result = CalibrationResult.empty([4, 7])

    assert result.calibrated is False
    assert result.wrist_calibrated is False
    assert result.motor_limits_dict == {4: [None, None], 7: [None, None]}
    assert result.joint_to_motor_ratios_dict == {4: 0.0, 7: 0.0}
