from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeState(str, Enum):
    PREVIEW = "preview"
    ARMED = "armed"
    LIVE = "live"
    TRACKING_LOST = "tracking_lost"
    FAULT = "fault"


@dataclass
class RuntimeStateMachine:
    state: RuntimeState = RuntimeState.PREVIEW
    reason: str = ""
    _return_state: RuntimeState = RuntimeState.ARMED

    def start_mapping(self) -> None:
        if self.state != RuntimeState.FAULT:
            self.state = RuntimeState.ARMED
            self.reason = ""

    def stop_mapping(self) -> None:
        if self.state != RuntimeState.FAULT:
            self.state = RuntimeState.PREVIEW
            self.reason = ""

    def enable_live(self) -> None:
        if self.state == RuntimeState.ARMED:
            self.state = RuntimeState.LIVE
            self.reason = ""

    def disable_live(self) -> None:
        if self.state == RuntimeState.LIVE:
            self.state = RuntimeState.ARMED
            self.reason = ""

    def tracking_lost(self, reason: str) -> None:
        if self.state == RuntimeState.FAULT:
            return
        self._return_state = self.state if self.state == RuntimeState.LIVE else RuntimeState.ARMED
        self.state = RuntimeState.TRACKING_LOST
        self.reason = reason

    def recover_tracking(self) -> None:
        if self.state == RuntimeState.TRACKING_LOST:
            self.state = self._return_state
            self.reason = ""

    def fault(self, reason: str) -> None:
        self.state = RuntimeState.FAULT
        self.reason = reason

    def reset_fault(self) -> None:
        if self.state == RuntimeState.FAULT:
            self.state = RuntimeState.PREVIEW
            self.reason = ""

    @property
    def can_send_to_hardware(self) -> bool:
        return self.state == RuntimeState.LIVE
