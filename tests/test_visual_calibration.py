from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import realtime_orcahand as rt


RuntimeSafetySettings = rt.RuntimeSafetySettings
load_realtime_config = rt.load_realtime_config
HandKinematics = rt.HandKinematics
SafetyController = rt.SafetyController
apply_visual_calibration = rt.apply_visual_calibration
list_visual_calibrations = rt.list_visual_calibrations
load_visual_calibration = rt.load_visual_calibration
save_visual_calibration = rt.save_visual_calibration
sanitize_profile_name = rt.sanitize_profile_name


def synthetic_hand(curled: bool = False) -> np.ndarray:
    points = np.zeros((21, 3), dtype=float)
    points[0] = [0.0, 0.0, 0.0]
    for start, x in ((1, -1.0), (5, -0.45), (9, 0.0), (13, 0.45), (17, 0.9)):
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
    hand[[1, 2, 3, 4], 0] -= 0.45
    return hand


def adducted_hand(points: np.ndarray) -> np.ndarray:
    hand = points.copy()
    hand[[6, 7, 8], 0] += 0.35
    hand[[10, 11, 12], 0] -= 0.35
    hand[[14, 15, 16], 0] -= 0.35
    hand[[18, 19, 20], 0] -= 0.35
    hand[[1, 2, 3, 4], 0] += 0.55
    return hand


def make_kinematics() -> tuple[HandKinematics, SafetyController]:
    cfg = load_realtime_config("config")
    safety = SafetyController(cfg, RuntimeSafetySettings())
    return HandKinematics(cfg, safety.safe_neutral()), safety


def capture_complete_visual_calibration(kinematics: HandKinematics) -> None:
    neutral = synthetic_hand(curled=False)
    kinematics.capture_neutral(neutral)
    kinematics.capture_open_pose(neutral)
    kinematics.capture_closed_pose(synthetic_hand(curled=True))
    kinematics.capture_abd_spread_pose(abducted_hand(neutral))
    kinematics.capture_abd_together_pose(adducted_hand(neutral))


def test_sanitize_profile_name_keeps_profile_inside_visual_calibration_folder():
    assert sanitize_profile_name("my calibration 01") == "my_calibration_01"
    assert sanitize_profile_name("thumb-good.yaml") == "thumb-good"

    for bad_name in ("", "../escape", "C:/escape", "a\\b"):
        with pytest.raises(ValueError):
            sanitize_profile_name(bad_name)


def test_visual_calibration_profile_round_trip_applies_to_kinematics(tmp_path):
    source, safety = make_kinematics()
    capture_complete_visual_calibration(source)
    neutral = synthetic_hand(curled=False)
    spread = abducted_hand(neutral)

    path = save_visual_calibration(tmp_path, "daily thumb test", source)
    profile = load_visual_calibration(path)
    target = HandKinematics(source.config, safety.safe_neutral())
    apply_visual_calibration(profile, target)

    assert path == tmp_path / "daily_thumb_test.yaml"
    assert target.has_neutral is True
    assert target.has_range_calibration is True
    assert target.has_abd_range_calibration is True
    assert target.estimate(spread)["index_abd"] == pytest.approx(
        source.estimate(spread)["index_abd"]
    )
    assert target.estimate(spread)["thumb_abd"] == pytest.approx(
        source.estimate(spread)["thumb_abd"]
    )


def test_visual_calibration_list_returns_named_profiles(tmp_path):
    first, _safety = make_kinematics()
    capture_complete_visual_calibration(first)
    second, _safety = make_kinematics()
    capture_complete_visual_calibration(second)

    save_visual_calibration(tmp_path, "b profile", second)
    save_visual_calibration(tmp_path, "a profile", first)

    profiles = list_visual_calibrations(tmp_path)

    assert [profile.name for profile in profiles] == ["a_profile", "b_profile"]
    assert all(profile.path.parent == tmp_path for profile in profiles)
