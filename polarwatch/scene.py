"""Renders the simulated camera frame with YOLO-style detection boxes.

Uses OpenCV only (per the project's stated tech stack) — a small synthetic
Antarctic scene with one shape per detected animal, boxed and labelled the
way a real detector's output would be drawn for a researcher's dashboard.
"""

from __future__ import annotations

import cv2
import numpy as np

from .data import FRAME_H, FRAME_W, SPECIES
from .detection import Detection

_SPECIES_COLOR = {s["name"]: s["color"] for s in SPECIES}  # stored as RGB
_SKY_TOP = np.array([10, 18, 30])       # RGB, deep polar-night blue
_SKY_HORIZON = np.array([168, 158, 140])  # RGB, pale dusk
_GROUND = (214, 224, 228)                # RGB, snow/ice
_GROUND_SHADE = (188, 200, 206)

FLAG_COLOR = (197, 60, 56)     # RGB — matches --critical
NORMAL_BOX_COLOR = (86, 150, 92)  # RGB — muted good-green


def render_frame(detections: list[Detection], flagged_species: str | None = None) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    sky_h = int(FRAME_H * 0.62)
    ramp = np.linspace(0, 1, sky_h)[:, None]
    sky = (_SKY_TOP[None, :] * (1 - ramp) + _SKY_HORIZON[None, :] * ramp).astype(np.uint8)
    frame[:sky_h, :, :] = sky[:, None, :]
    frame[sky_h:, :, :] = _GROUND

    for i in range(1, 5):
        x0 = int(FRAME_W * i / 5)
        cv2.line(frame, (x0, sky_h), (x0 - 30, FRAME_H), _GROUND_SHADE, 2, lineType=cv2.LINE_AA)

    for det in detections:
        x, y, w, h = det.bbox
        color = tuple(int(c) for c in _SPECIES_COLOR.get(det.species, (90, 90, 90)))
        cx, cy = x + w // 2, y + int(h * 0.6)

        cv2.ellipse(frame, (cx, cy), (w // 3, h // 3), 0, 0, 360, color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (w // 3, h // 3), 0, 0, 360, (245, 248, 250), 1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, y + h // 6), max(w // 7, 7), color, -1, lineType=cv2.LINE_AA)

        is_flagged = flagged_species is not None and det.species == flagged_species
        box_color = FLAG_COLOR if is_flagged else NORMAL_BOX_COLOR
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2, lineType=cv2.LINE_AA)

        label = f"{det.species}  {det.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        label_y = max(y - 6, th + 6)
        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 8, label_y), box_color, -1)
        cv2.putText(frame, label, (x + 4, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (250, 250, 250), 1, cv2.LINE_AA)

    return frame
