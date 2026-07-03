import numpy as np
import pytest

from orca_realtime.config import RuntimeSafetySettings, load_realtime_config
from orca_realtime.kinematics import HandKinematics
from orca_realtime.safety import SafetyController


def synthetic_hand(curled: bool = False) -> np.ndarray:
    points = np.zeros((21, 3), dtype=float)
    points[0] = [0.0, 0.0, 0.0]
    bases = {
        "thumb": (1, -1.0),
        "index": (5, -0.45),
        "middle": (9, 0.0),
        "ring": (13, 0.45),
        "pinky": (17, 0.9),
    }
    for _name, (start, x) in bases.items():
        points[start] = [x, 1.0, 0.0]
        points[start + 1] = [x, 2.0, 0.0]
        points[start + 2] = [x, 3.0, 0.0]
        points[start + 3] = [x, 4.0, 0.0]

    if curled:
        for start in (5, 9, 13, 17):
            x = points[start, 0]
            points[start + 1] = [x, 1.8, 0.0]
            points[start + 2] = [x + 0.45, 2.0, 0.0]
            points[start + 3] = [x + 0.8, 1.65, 0.0]

    return points


def abducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[6, 7, 8], 0] -= 0.45
    hand[[10, 11, 12], 0] += 0.35
    hand[[14, 15, 16], 0] += 0.45
    hand[[18, 19, 20], 0] += 0.45
    return hand


def adducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[6, 7, 8], 0] += 0.35
    hand[[10, 11, 12], 0] -= 0.35
    hand[[14, 15, 16], 0] -= 0.35
    hand[[18, 19, 20], 0] -= 0.35
    return hand


def thumb_abducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[1, 2, 3, 4], 0] -= 0.45
    return hand


def thumb_adducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[1, 2, 3, 4], 0] += 0.55
    return hand


def test_kinematics_outputs_all_orca_joints_and_keeps_wrist_neutral():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())

    joints = kinematics.estimate(synthetic_hand(curled=False))

    assert set(joints) == set(cfg.joint_ids)
    assert joints["wrist"] == controller.safe_neutral()["wrist"]


def test_open_and_curled_finger_span_open_close_commands_for_official_config():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral = controller.safe_neutral()

    open_joints = kinematics.estimate(synthetic_hand(curled=False))
    curled_joints = kinematics.estimate(synthetic_hand(curled=True))

    assert open_joints["index_mcp"] > neutral["index_mcp"]
    assert open_joints["index_pip"] > neutral["index_pip"]
    assert curled_joints["index_mcp"] < open_joints["index_mcp"]
    assert curled_joints["index_pip"] < open_joints["index_pip"]


def test_open_thumb_pose_maps_to_open_side_instead_of_neutral():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral = controller.safe_neutral()

    joints = kinematics.estimate(synthetic_hand(curled=False))

    assert joints["thumb_mcp"] > neutral["thumb_mcp"]
    assert joints["thumb_pip"] > neutral["thumb_pip"]


def test_curled_finger_commands_remain_inside_safety_bounds():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())

    result = controller.apply(kinematics.estimate(synthetic_hand(curled=True)))

    assert result.accepted is True
    assert result.reasons == []


def test_abduction_helpers_keep_large_raw_values_inside_safety_bounds():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())

    for joint in ("thumb_abd", "index_abd", "middle_abd", "ring_abd", "pinky_abd"):
        bounds = controller.bounds[joint]
        for raw_value in (-500.0, 500.0):
            command = kinematics._relative_to_joint(joint, raw_value, scale=1.0)
            assert bounds.joint_min <= command <= bounds.joint_max, joint


def test_finger_abduction_uses_hardware_observed_reverse_direction_and_gain():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)
    kinematics.capture_neutral(neutral_hand)
    base_joints = kinematics.estimate(neutral_hand)

    abducted = neutral_hand.copy()
    abducted[[6, 7, 8], 0] -= 0.45
    abducted[[14, 15, 16], 0] += 0.45
    joints = kinematics.estimate(abducted)

    assert joints["index_abd"] > base_joints["index_abd"] + 20.0
    assert joints["ring_abd"] < base_joints["ring_abd"] - 20.0


def test_index_and_middle_abd_both_move_with_hardware_observed_reverse_direction():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)
    kinematics.capture_neutral(neutral_hand)
    base_joints = kinematics.estimate(neutral_hand)

    together = neutral_hand.copy()
    together[[6, 7, 8], 0] += 0.35
    together[[10, 11, 12], 0] -= 0.35
    joints = kinematics.estimate(together)

    assert joints["index_abd"] < base_joints["index_abd"] - 20.0
    assert joints["middle_abd"] > base_joints["middle_abd"] + 20.0


def test_thumb_abd_raw_uses_lateral_gap_when_thumb_axis_does_not_rotate():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)

    neutral_raw = kinematics._raw_controls(neutral_hand)["thumb_abd"]
    spread_raw = kinematics._raw_controls(thumb_abducted_hand(neutral_hand))[
        "thumb_abd"
    ]
    together_raw = kinematics._raw_controls(thumb_adducted_hand(neutral_hand))[
        "thumb_abd"
    ]

    assert spread_raw > neutral_raw + 5.0
    assert together_raw < neutral_raw - 5.0


def test_thumb_abd_range_calibration_maps_lateral_gap_to_safe_span():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)
    spread_hand = thumb_abducted_hand(neutral_hand)
    together_hand = thumb_adducted_hand(neutral_hand)

    kinematics.capture_neutral(neutral_hand)
    kinematics.capture_abd_spread_pose(spread_hand)
    kinematics.capture_abd_together_pose(together_hand)

    center = kinematics._abd_to_joint("thumb_abd", 0.0)
    command_min, command_max = kinematics._command_range("thumb_abd")
    half_range = 0.5 * (command_max - command_min)
    spread_command = kinematics.estimate(spread_hand)["thumb_abd"]
    together_command = kinematics.estimate(together_hand)["thumb_abd"]

    assert spread_command < center - 0.8 * half_range
    assert together_command > center + 0.8 * half_range


def test_thumb_abd_arc_uses_index_ray_as_together_reference():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    hand = synthetic_hand(curled=False)
    hand[6] = [-0.60, 2.0, 0.0]
    hand[7] = [-0.75, 3.0, 0.0]
    hand[8] = [-0.90, 4.0, 0.0]
    hand[4] = [-0.90, 4.0, 0.0]

    together_raw = kinematics._raw_controls(hand)["thumb_abd"]

    assert abs(together_raw) < 2.0


def test_thumb_abd_default_arc_mapping_has_enough_close_authority():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)
    together_hand = thumb_adducted_hand(neutral_hand)

    kinematics.capture_neutral(neutral_hand)
    center = kinematics._abd_to_joint("thumb_abd", 0.0)
    together_command = kinematics.estimate(together_hand)["thumb_abd"]

    assert together_command > center + 15.0


def test_kinematics_helpers_reverse_flex_in_program_without_changing_config():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral = controller.safe_neutral()

    for joint in (
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
    ):
        assert kinematics._flex_to_joint(joint, 0.0) > neutral[joint], joint
        assert kinematics._flex_to_joint(joint, 100.0) < neutral[joint], joint

    for joint in ("thumb_abd", "index_abd", "middle_abd", "ring_abd", "pinky_abd"):
        center = kinematics._abd_to_joint(joint, 0.0)
        command = kinematics._abd_to_joint(joint, 10.0)
        if joint == "thumb_abd":
            assert command < center, joint
        else:
            assert command < center, joint
        assert command >= controller.bounds[joint].joint_min, joint


def test_thumb_flex_mapping_is_less_sensitive_than_fingers():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())

    thumb_open = kinematics._flex_to_joint("thumb_pip", 0.0)
    thumb_partial = kinematics._flex_to_joint("thumb_pip", 10.0)
    thumb_moderate = kinematics._flex_to_joint("thumb_pip", 100.0)
    index_open = kinematics._flex_to_joint("index_pip", 0.0)
    index_small = kinematics._flex_to_joint("index_pip", 10.0)

    assert thumb_partial < thumb_open
    assert thumb_moderate > controller.bounds["thumb_pip"].joint_min + 10.0
    assert index_small == pytest.approx(index_open)


def test_abd_mapping_uses_safe_midpoint_center_and_stronger_reverse_gain():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings(default_offset_deg=1.0))
    kinematics = HandKinematics(cfg, controller.safe_neutral())

    thumb_command = kinematics._abd_to_joint("thumb_abd", 5.0)
    index_command = kinematics._abd_to_joint("index_abd", 5.0)
    middle_command = kinematics._abd_to_joint("middle_abd", 5.0)
    thumb_center = kinematics._abd_to_joint("thumb_abd", 0.0)
    index_center = kinematics._abd_to_joint("index_abd", 0.0)
    middle_center = kinematics._abd_to_joint("middle_abd", 0.0)

    assert thumb_center - thumb_command == pytest.approx(5.0)
    assert index_center - index_command == pytest.approx(10.0)
    assert middle_center - middle_command == pytest.approx(10.0)


def test_abd_range_calibration_maps_user_lateral_limits_to_safe_range():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)
    spread_hand = abducted_hand(neutral_hand)
    together_hand = adducted_hand(neutral_hand)

    kinematics.capture_neutral(neutral_hand)
    kinematics.capture_abd_spread_pose(spread_hand)
    kinematics.capture_abd_together_pose(together_hand)

    neutral_joints = kinematics.estimate(neutral_hand)
    spread_joints = kinematics.estimate(spread_hand)
    together_joints = kinematics.estimate(together_hand)

    assert kinematics.has_abd_range_calibration is True
    assert neutral_joints["index_abd"] == pytest.approx(
        kinematics._abd_to_joint("index_abd", 0.0)
    )
    assert spread_joints["index_abd"] == pytest.approx(
        controller.bounds["index_abd"].joint_max
    )
    assert together_joints["index_abd"] == pytest.approx(
        controller.bounds["index_abd"].joint_min
    )
    assert spread_joints["middle_abd"] == pytest.approx(
        controller.bounds["middle_abd"].joint_min
    )
    assert together_joints["middle_abd"] == pytest.approx(
        controller.bounds["middle_abd"].joint_max
    )


def test_abd_range_calibration_warns_and_falls_back_when_span_is_too_small():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    neutral_hand = synthetic_hand(curled=False)

    kinematics.capture_neutral(neutral_hand)
    kinematics.capture_abd_spread_pose(neutral_hand)
    kinematics.capture_abd_together_pose(neutral_hand)

    joints = kinematics.estimate(neutral_hand)

    assert joints["index_abd"] == pytest.approx(
        kinematics._abd_to_joint("index_abd", 0.0)
    )
    assert any("index_abd" in warning for warning in kinematics.calibration_warnings())


def test_range_calibration_maps_user_open_and_fist_to_full_flex_span():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    open_hand = synthetic_hand(curled=False)
    fist_hand = synthetic_hand(curled=True)

    kinematics.capture_open_pose(open_hand)
    kinematics.capture_closed_pose(fist_hand)

    open_joints = kinematics.estimate(open_hand)
    fist_joints = kinematics.estimate(fist_hand)

    assert kinematics.has_range_calibration is True
    assert open_joints["index_mcp"] == pytest.approx(
        controller.bounds["index_mcp"].joint_max
    )
    assert open_joints["index_pip"] == pytest.approx(
        controller.bounds["index_pip"].joint_max
    )
    assert fist_joints["index_mcp"] == pytest.approx(
        controller.bounds["index_mcp"].joint_min
    )
    assert fist_joints["index_pip"] == pytest.approx(
        controller.bounds["index_pip"].joint_min
    )


def test_range_calibration_ignores_visual_neutral_for_flex_mapping():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    open_hand = synthetic_hand(curled=False)
    fist_hand = synthetic_hand(curled=True)

    kinematics.capture_open_pose(open_hand)
    kinematics.capture_closed_pose(fist_hand)
    kinematics.capture_neutral(fist_hand)

    open_joints = kinematics.estimate(open_hand)

    assert open_joints["index_pip"] == pytest.approx(
        controller.bounds["index_pip"].joint_max
    )


def test_range_calibration_warns_and_falls_back_when_span_is_too_small():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    open_hand = synthetic_hand(curled=False)

    kinematics.capture_open_pose(open_hand)
    kinematics.capture_closed_pose(open_hand)

    joints = kinematics.estimate(open_hand)

    assert joints["index_pip"] == pytest.approx(
        kinematics._flex_to_joint("index_pip", 0.0)
    )
    assert any("index_pip" in warning for warning in kinematics.calibration_warnings())


def test_capture_neutral_reduces_initial_abduction_bias():
    cfg = load_realtime_config("config")
    controller = SafetyController(cfg, RuntimeSafetySettings())
    kinematics = HandKinematics(cfg, controller.safe_neutral())
    hand = synthetic_hand(curled=False)
    hand[5, 0] -= 0.3

    neutral = kinematics._abd_to_joint("index_abd", 0.0)
    before = abs(kinematics.estimate(hand)["index_abd"] - neutral)
    kinematics.capture_neutral(hand)
    after = abs(kinematics.estimate(hand)["index_abd"] - neutral)

    assert after < before
