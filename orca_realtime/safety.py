from __future__ import annotations

from dataclasses import dataclass

from .config import RealtimeConfig, RuntimeSafetySettings


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
            positions[motor_id] = self.estimate_motor_position(joint, value)
        return positions

    def estimate_motor_position(self, joint: str, value: float) -> float:
        motor_id = self.config.joint_to_motor_map[joint]
        rom_min, rom_max = self.config.joint_roms[joint]
        motor_min, _motor_max = self.config.motor_limits[motor_id]
        ratio = self.config.joint_to_motor_ratios[motor_id]
        if self.config.joint_inversions.get(joint, False):
            return motor_min + (rom_max - float(value)) * ratio
        return motor_min + (float(value) - rom_min) * ratio

    def _build_bounds(self) -> dict[str, JointSafetyBounds]:
        bounds: dict[str, JointSafetyBounds] = {}
        for joint in self.config.joint_ids:
            motor_id = self.config.joint_to_motor_map[joint]
            rom_min, rom_max = self.config.joint_roms[joint]
            motor_min, motor_max = self.config.motor_limits[motor_id]
            ratio = self.config.joint_to_motor_ratios[motor_id]
            offset_deg = self.settings.offset_for_joint(joint)
            motor_offset_rad = offset_deg * ratio
            joint_min = rom_min + offset_deg
            joint_max = rom_max - offset_deg
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
