from pathlib import Path

import pytest

from orca_realtime.config import RuntimeSafetySettings, load_realtime_config
from orca_realtime.safety import SafetyController


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_controller(**kwargs):
    cfg = load_realtime_config(PROJECT_ROOT / "config")
    settings = RuntimeSafetySettings(**kwargs)
    return SafetyController(cfg, settings)


def test_joint_and_motor_offsets_share_same_angle_offset():
    controller = make_controller(default_offset_deg=5.0)
    bounds = controller.bounds["index_mcp"]
    cfg = controller.config
    motor_id = cfg.joint_to_motor_map["index_mcp"]
    ratio = cfg.joint_to_motor_ratios[motor_id]

    assert bounds.joint_min == pytest.approx(-15.0)
    assert bounds.joint_max == pytest.approx(90.0)
    assert bounds.motor_min == pytest.approx(cfg.motor_limits[motor_id][0] + 5.0 * ratio)
    assert bounds.motor_max == pytest.approx(cfg.motor_limits[motor_id][1] - 5.0 * ratio)


def test_default_safety_offsets_are_smaller_and_motor_rad_stays_synced():
    controller = make_controller()
    cfg = controller.config

    index_bounds = controller.bounds["index_mcp"]
    index_motor_id = cfg.joint_to_motor_map["index_mcp"]
    index_ratio = cfg.joint_to_motor_ratios[index_motor_id]

    assert index_bounds.offset_deg == pytest.approx(3.0)
    assert index_bounds.joint_min == pytest.approx(-17.0)
    assert index_bounds.joint_max == pytest.approx(92.0)
    assert index_bounds.motor_offset_rad == pytest.approx(3.0 * index_ratio)
    assert index_bounds.motor_min == pytest.approx(
        cfg.motor_limits[index_motor_id][0] + 3.0 * index_ratio
    )
    assert index_bounds.motor_max == pytest.approx(
        cfg.motor_limits[index_motor_id][1] - 3.0 * index_ratio
    )

    index_abd_bounds = controller.bounds["index_abd"]
    index_abd_motor_id = cfg.joint_to_motor_map["index_abd"]
    index_abd_ratio = cfg.joint_to_motor_ratios[index_abd_motor_id]

    assert index_abd_bounds.offset_deg == pytest.approx(2.0)
    assert index_abd_bounds.joint_min == pytest.approx(-35.0)
    assert index_abd_bounds.joint_max == pytest.approx(35.0)
    assert index_abd_bounds.motor_offset_rad == pytest.approx(2.0 * index_abd_ratio)


def test_safe_neutral_is_clamped_when_config_neutral_is_outside_rom():
    controller = make_controller(default_offset_deg=4.0)

    neutral = controller.safe_neutral()

    assert neutral["thumb_abd"] <= controller.bounds["thumb_abd"].joint_max
    assert neutral["pinky_abd"] >= controller.bounds["pinky_abd"].joint_min


def test_limits_per_frame_delta_and_keeps_wrist_neutral():
    controller = make_controller(max_delta_deg_per_frame=4.0, abrupt_delta_deg=200.0)
    start = controller.safe_neutral()
    target = dict(start)
    target["index_mcp"] = start["index_mcp"] + 60.0

    result = controller.apply(target)

    assert result.accepted is True
    assert result.joints["index_mcp"] <= start["index_mcp"] + 4.0
    assert result.joints["wrist"] == pytest.approx(start["wrist"])


def test_reversed_program_flex_commands_increase_motor_position_with_official_config():
    controller = make_controller(default_offset_deg=1.0)
    neutral = controller.safe_neutral()

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
        close_target = max(controller.bounds[joint].joint_min, neutral[joint] - 10.0)
        neutral_motor = controller.estimate_motor_position(joint, neutral[joint])
        close_motor = controller.estimate_motor_position(joint, close_target)

        assert close_motor > neutral_motor, joint


def test_disabled_joint_stays_at_safe_neutral():
    controller = make_controller(
        joint_controls={"index_mcp": {"enabled": False, "gain": 1.0}}
    )
    neutral = controller.safe_neutral()

    result = controller.apply({"index_mcp": 90.0})

    assert result.joints["index_mcp"] == pytest.approx(neutral["index_mcp"])


def test_abrupt_joint_change_rejects_frame_and_preserves_previous_output():
    controller = make_controller(abrupt_delta_deg=10.0)
    first = controller.apply({"index_mcp": 2.0})

    second = controller.apply({"index_mcp": 80.0})

    assert first.accepted is True
    assert second.accepted is False
    assert "abrupt" in " ".join(second.reasons)
    assert second.joints["index_mcp"] == pytest.approx(first.joints["index_mcp"])


def test_joint_safe_limit_violation_rejects_frame_instead_of_clamping_and_sending():
    controller = make_controller(default_offset_deg=5.0, abrupt_delta_deg=200.0)
    neutral = controller.safe_neutral()

    result = controller.apply({"index_mcp": controller.bounds["index_mcp"].joint_max + 10.0})

    assert result.accepted is False
    assert "outside safe ROM" in " ".join(result.reasons)
    assert result.joints["index_mcp"] == pytest.approx(neutral["index_mcp"])


def test_reset_to_safe_neutral_restarts_delta_limit_from_neutral():
    controller = make_controller(max_delta_deg_per_frame=5.0, abrupt_delta_deg=200.0)
    neutral = controller.safe_neutral()
    target = dict(neutral)
    target["index_mcp"] = neutral["index_mcp"] + 60.0

    for _ in range(20):
        controller.apply(target)

    controller.reset_to_safe_neutral()
    result = controller.apply(target)

    assert result.joints["index_mcp"] == pytest.approx(neutral["index_mcp"] + 1.0)


def test_startup_ramp_uses_smaller_delta_before_normal_delta():
    controller = make_controller(
        max_delta_deg_per_frame=5.0,
        startup_max_delta_deg_per_frame=0.5,
        startup_ramp_frames=2,
        abrupt_delta_deg=200.0,
    )
    neutral = controller.safe_neutral()
    target = dict(neutral)
    target["index_mcp"] = neutral["index_mcp"] + 50.0

    controller.reset_to_safe_neutral()
    first = controller.apply(target)
    second = controller.apply(target)
    third = controller.apply(target)

    assert first.joints["index_mcp"] == pytest.approx(neutral["index_mcp"] + 0.5)
    assert second.joints["index_mcp"] == pytest.approx(neutral["index_mcp"] + 1.0)
    assert third.joints["index_mcp"] == pytest.approx(neutral["index_mcp"] + 6.0)


def test_startup_ramp_does_not_reject_first_live_target_as_abrupt():
    controller = make_controller(
        max_delta_deg_per_frame=5.0,
        startup_max_delta_deg_per_frame=0.5,
        startup_ramp_frames=2,
        abrupt_delta_deg=10.0,
    )
    neutral = controller.safe_neutral()
    target = dict(neutral)
    target["thumb_abd"] = controller.bounds["thumb_abd"].joint_min

    controller.reset_to_safe_neutral()
    result = controller.apply(target)

    assert result.accepted is True
    assert result.reasons == []
    assert result.joints["thumb_abd"] == pytest.approx(neutral["thumb_abd"] - 0.5)
