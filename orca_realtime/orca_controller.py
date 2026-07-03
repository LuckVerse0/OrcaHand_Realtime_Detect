from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORCA_CORE_ROOT = PROJECT_ROOT / "vendor" / "orca_core"


class OrcaController:
    def __init__(
        self,
        config_path: str | Path,
        *,
        live: bool,
        orca_core_root: str | Path | None = None,
        mock: bool = False,
        force_calibrate: bool = False,
        move_to_neutral: bool = True,
        send_num_steps: int = 1,
        send_step_size: float = 1e-2,
    ) -> None:
        self.config_path = Path(config_path)
        self.live = bool(live)
        self.orca_core_root = Path(orca_core_root) if orca_core_root else None
        self.mock = bool(mock)
        self.force_calibrate = bool(force_calibrate)
        self.move_to_neutral = bool(move_to_neutral)
        self.send_num_steps = int(send_num_steps)
        self.send_step_size = float(send_step_size)
        self.hand: Any | None = None
        self.connected = False
        if self.send_num_steps < 1:
            raise ValueError("send_num_steps must be >= 1")
        if self.send_step_size < 0:
            raise ValueError("send_step_size must be >= 0")

    def connect(self) -> None:
        if not self.live:
            self.connected = False
            return

        self._connect_hand()
        self.hand.init_joints(
            force_calibrate=self.force_calibrate,
            move_to_neutral=self.move_to_neutral,
        )
        self.connected = True

    def connect_for_calibration(self) -> None:
        if not self.live:
            self.connected = False
            return
        self._connect_hand()
        self.connected = True

    def calibrate(self, *, force_wrist: bool = False, joints: list[str] | None = None) -> None:
        if not self.live:
            return
        if self.hand is None or not self.connected:
            self.connect_for_calibration()
        if self.hand is None:
            raise RuntimeError("OrcaController is not connected")
        self.hand.calibrate(force_wrist=force_wrist, joints=joints)

    def stop_task(self) -> None:
        if self.hand is not None:
            self._stop_task_quietly()

    def _connect_hand(self) -> None:
        if self.hand is not None and self.connected:
            return
        OrcaHand = self._load_orca_hand()
        self.hand = OrcaHand(config_path=str(self.config_path))
        success, message = self.hand.connect()
        if not success:
            self.hand = None
            raise RuntimeError(message)

    def send(self, joints: dict[str, float]) -> None:
        if not self.live:
            return
        if self.hand is None or not self.connected:
            raise RuntimeError("OrcaController is not connected")
        self.hand.set_joint_positions(
            joints,
            num_steps=self.send_num_steps,
            step_size=self.send_step_size,
        )

    def emergency_stop(self) -> None:
        if self.hand is not None:
            try:
                self._stop_task_quietly()
                try:
                    self.hand.disable_torque()
                except Exception:
                    pass
                disconnect = getattr(self.hand, "disconnect", None)
                if disconnect is not None:
                    try:
                        disconnect()
                    except Exception:
                        pass
            finally:
                self.hand = None
                self.connected = False
        else:
            self.connected = False

    def disconnect(self) -> None:
        hand = self.hand
        try:
            if hand is not None:
                self._stop_task_quietly()
                hand.disconnect()
        finally:
            self.hand = None
            self.connected = False

    def _load_orca_hand(self):
        candidates = []
        if self.orca_core_root is not None:
            candidates.append(self.orca_core_root)
        candidates.append(DEFAULT_ORCA_CORE_ROOT)

        for root in candidates:
            if root.exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))

        try:
            if self.mock:
                from orca_core.hardware_hand import MockOrcaHand

                return MockOrcaHand
            from orca_core import OrcaHand
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "orca_core is required for --live. Pass --orca-core-root or install orca_core."
            ) from exc
        return OrcaHand

    def _stop_task_quietly(self) -> None:
        stop_task = getattr(self.hand, "stop_task", None)
        if stop_task is None:
            return
        try:
            stop_task()
        except Exception:
            pass
