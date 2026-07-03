from pathlib import Path

import pytest

from orca_realtime.config import load_realtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_current_runtime_config_and_calibration():
    cfg = load_realtime_config(PROJECT_ROOT / "config")

    assert cfg.config_path == PROJECT_ROOT / "config" / "config.yaml"
    assert cfg.calibration_path == PROJECT_ROOT / "config" / "calibration.yaml"
    assert len(cfg.joint_ids) == 17
    assert len(cfg.motor_ids) == 17
    assert cfg.calibrated is True
    assert cfg.wrist_calibrated is True


def test_normalizes_signed_joint_to_motor_mapping_and_inversion():
    cfg = load_realtime_config(PROJECT_ROOT / "config")

    assert cfg.joint_to_motor_map["thumb_mcp"] == 4
    assert cfg.joint_inversions["thumb_mcp"] is True
    assert cfg.joint_to_motor_map["index_abd"] == 14
    assert cfg.joint_inversions["index_abd"] is False
    for joint in (
        "index_mcp",
        "index_pip",
        "middle_mcp",
        "middle_pip",
        "ring_mcp",
        "ring_pip",
        "pinky_mcp",
        "pinky_pip",
    ):
        assert cfg.joint_inversions[joint] is True


def test_live_validation_requires_all_motor_limits_and_ratios(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """
joint_ids: [index_mcp]
motor_ids: [1]
joint_to_motor_map:
  index_mcp: 1
joint_roms:
  index_mcp: [-20, 95]
neutral_position:
  index_mcp: 0
type: right
""",
        encoding="utf-8",
    )
    (config_dir / "calibration.yaml").write_text(
        """
calibrated: true
wrist_calibrated: true
motor_limits:
  1: [0.0, 1.0]
joint_to_motor_ratios:
  1: 0.0
""",
        encoding="utf-8",
    )

    cfg = load_realtime_config(config_dir)

    with pytest.raises(ValueError, match="ratio"):
        cfg.validate_for_live()
