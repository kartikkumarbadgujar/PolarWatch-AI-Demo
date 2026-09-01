"""Stand-in for the CNN-based visual health assessment (slide 3, step 3).

A trained CNN would classify the cropped detection region for plumage
condition, lesions, and posture. Here a numeric "visual concern" score
(0 = no visible issue, 100 = severe) is drawn from a distribution that can be
pushed upward for one species to simulate a developing outbreak — the same
role a real classifier's output would play downstream.
"""

from __future__ import annotations

import numpy as np


class HealthAssessor:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def assess(self, species: str, outbreak_species: str | None, outbreak_level: float) -> float:
        baseline = float(self.rng.uniform(4, 18))
        if outbreak_species is not None and species == outbreak_species:
            baseline += outbreak_level
        return float(np.clip(baseline, 0, 100))
