"""Stand-in for the RNN-based temporal behaviour analysis (slide 3, step 4).

A trained RNN would look at a sequence of observations for one animal or
colony and score how far its recent movement/behaviour pattern deviates from
normal. This module keeps a rolling window per species and derives a
comparable anomaly score from the current visual-concern reading plus a
synthetic movement-variance term.
"""

from __future__ import annotations

from collections import deque

import numpy as np

HISTORY_LENGTH = 12


class BehaviorAnalyzer:
    def __init__(self):
        self.history: dict[str, deque[float]] = {}

    def update(self, species: str, visual_score: float, rng: np.random.Generator) -> tuple[float, list[float]]:
        series = self.history.setdefault(species, deque(maxlen=HISTORY_LENGTH))
        # Sicker animals also move more erratically, so variance rises with
        # the visual-concern score rather than being independent noise.
        movement_variance = float(rng.uniform(2, 14)) + max(0.0, visual_score - 35) * 0.55
        noise = float(rng.normal(0, 3))
        anomaly = 0.5 * visual_score + 0.4 * movement_variance + noise
        anomaly = float(np.clip(anomaly, 0, 100))
        series.append(anomaly)
        return anomaly, list(series)
