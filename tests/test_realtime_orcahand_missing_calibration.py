from __future__ import annotations

from pathlib import Path

import pytest
import realtime_orcahand as rt


def load_realtime_module():
    return rt


def write_minimal_config(config_dir: Path) -> None:
    (config_dir / "config.yaml").write_text(
        """
type: right
joint_ids: [index_mcp]
motor_ids: [1]
joint_to_motor_map:
  index_mcp: 1
joint_roms:
  index_mcp: [-20, 95]
neutral_position:
  index_mcp: 0
""",
        encoding="utf-8",
    )


def test_load_realtime_config_allows_missing_calibration_yaml(tmp_path: Path):
    module = load_realtime_module()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_minimal_config(config_dir)

    config = module.load_realtime_config(config_dir)

    assert config.calibrated is False
    assert config.wrist_calibrated is False
    assert config.motor_limits == {}
    assert config.joint_to_motor_ratios == {}
    with pytest.raises(ValueError, match="Run Orca calibration first"):
        config.validate_for_live()


def test_safety_controller_allows_preview_without_calibration_yaml(tmp_path: Path):
    module = load_realtime_module()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_minimal_config(config_dir)
    config = module.load_realtime_config(config_dir)

    safety = module.SafetyController(config)

    assert safety.safe_neutral() == {"index_mcp": 0.0}
    result = safety.apply({"index_mcp": 10.0})
    assert result.accepted is True
    assert result.motor_positions == {}


def test_load_realtime_config_allows_partial_calibration_yaml(tmp_path: Path):
    module = load_realtime_module()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    write_minimal_config(config_dir)
    (config_dir / "calibration.yaml").write_text(
        """
calibrated: false
wrist_calibrated: false
motor_limits:
  1: [null, null]
  2: [0.1, 1.2]
joint_to_motor_ratios:
  1: 0.0
  2: 0.25
""",
        encoding="utf-8",
    )

    config = module.load_realtime_config(config_dir)

    assert config.calibrated is False
    assert config.motor_limits == {2: (0.1, 1.2)}
