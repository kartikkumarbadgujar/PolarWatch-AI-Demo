"""Stand-in for the YOLO real-time species detector (slide 3, step 2).

A trained YOLO model would take a video frame and return bounding boxes with
species labels and confidence scores. This module returns data shaped the
same way, drawn from a random generator, so the rest of the pipeline and the
dashboard can be built and demoed before a model is trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DETECTION_SLOTS, SPECIES_NAMES


@dataclass
class Detection:
    species: str
    bbox: tuple[int, int, int, int]
    confidence: float


class SimulatedDetector:
    """Fills the fixed camera slots with a species guess each cycle."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def detect(self) -> list[Detection]:
        picks = self.rng.choice(SPECIES_NAMES, size=len(DETECTION_SLOTS), replace=True)
        detections = []
        for slot, species in zip(DETECTION_SLOTS, picks):
            confidence = float(self.rng.uniform(0.86, 0.99))
            detections.append(Detection(species=species, bbox=slot, confidence=confidence))
        return detections
