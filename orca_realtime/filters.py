from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExponentialSmoother:
    alpha: float
    _value: np.ndarray | None = None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self._value is None:
            self._value = value.copy()
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value.copy()

    def reset(self) -> None:
        self._value = None


@dataclass
class JointSmoother:
    default_alpha: float = 0.35
    abd_alpha: float = 0.25
    _values: dict[str, float] = field(default_factory=dict)

    def update(self, joints: dict[str, float]) -> dict[str, float]:
        output: dict[str, float] = {}
        for joint, value in joints.items():
            value = float(value)
            if joint not in self._values:
                output[joint] = value
            else:
                alpha = self.abd_alpha if joint.endswith("_abd") else self.default_alpha
                output[joint] = alpha * value + (1.0 - alpha) * self._values[joint]
        self._values.update(output)
        return dict(output)

    def reset(self) -> None:
        self._values.clear()
