from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SyncPhase(str, Enum):
    WAITING = "waiting"
    RECORDING = "recording"
    COMPLETE = "complete"


@dataclass
class SyncStateMachine:
    stable_frames_required: int = 3

    def __post_init__(self) -> None:
        self.phase = SyncPhase.WAITING
        self._candidate_id: int | None = None
        self._candidate_count = 0
        self.csv_started = False
        self.csv_completed = False

    def update(self, sync_ids: set[int]) -> list[str]:
        messages: list[str] = []
        detected_id = None
        for wanted in (0, 1, 2):
            if wanted in sync_ids:
                detected_id = wanted
                break
        if detected_id is None:
            self._candidate_id = None
            self._candidate_count = 0
            return messages
        if detected_id == 0 and self.phase == SyncPhase.WAITING:
            messages.append("Sync marker ID0 detected... waiting")
        if self._candidate_id != detected_id:
            self._candidate_id = detected_id
            self._candidate_count = 1
            return messages
        self._candidate_count += 1
        if self._candidate_count < self.stable_frames_required:
            return messages

        if detected_id == 1 and self.phase != SyncPhase.RECORDING:
            self.phase = SyncPhase.RECORDING
            self.csv_started = True
            messages.append("Sync marker ID1 detected... csv writing started")
        elif detected_id == 2 and self.phase != SyncPhase.COMPLETE:
            self.phase = SyncPhase.COMPLETE
            self.csv_completed = True
            messages.append("Sync marker ID2 detected... csv writing complete")
        self._candidate_count = 0
        return messages
