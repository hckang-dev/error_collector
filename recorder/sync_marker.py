from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import tkinter as tk


class SyncState(str, Enum):
    BEFORE = "before"
    RECORDING = "recording"
    DONE = "done"


@dataclass(frozen=True)
class SyncVisual:
    marker_id: int
    text: str


SYNC_VISUALS = {
    SyncState.BEFORE: SyncVisual(marker_id=0, text="Before Recording"),
    SyncState.RECORDING: SyncVisual(marker_id=1, text="Recording"),
    SyncState.DONE: SyncVisual(marker_id=2, text="Recording Finished"),
}


class SyncMarkerWindow:
    def __init__(self, root: tk.Misc, marker_size_px: int = 420) -> None:
        self.root = root
        self.marker_size_px = int(marker_size_px)
        self.window = tk.Toplevel(root)
        self.window.title("Sync Marker")
        self.window.geometry("460x520")
        self.window.configure(bg="white")
        self.window.transient(root)
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)

        self.marker_label = tk.Label(self.window, bg="white")
        self.marker_label.pack(padx=18, pady=(18, 10))
        self.text_var = tk.StringVar(value="")
        self.text_label = tk.Label(
            self.window,
            textvariable=self.text_var,
            font=("Helvetica", 22, "bold"),
            anchor="center",
            bg="white",
        )
        self.text_label.pack(fill="x", padx=18, pady=(0, 18))

        self._image: Optional[tk.PhotoImage] = None
        self.state = SyncState.BEFORE
        self.set_state(SyncState.BEFORE)

    def set_state(self, state: SyncState) -> None:
        visual = SYNC_VISUALS[state]
        self._image = self._render_marker(visual.marker_id)
        self.marker_label.configure(image=self._image)
        self.text_var.set(visual.text)
        self.state = state
        self.window.deiconify()

    def _render_marker(self, marker_id: int) -> tk.PhotoImage:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        image = cv2.aruco.generateImageMarker(dictionary, int(marker_id), self.marker_size_px)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        ok, encoded_png = cv2.imencode(".png", np.asarray(image_rgb, dtype=np.uint8))
        if not ok:
            raise RuntimeError("failed to encode sync marker image")
        encoded = base64.b64encode(encoded_png.tobytes()).decode("ascii")
        return tk.PhotoImage(data=encoded, format="png")
