"""The multimodal risk engine (slide 4, step 5).

Combines visual, behavioural, environmental, and historical signals into one
0-100 wildlife health-risk score, then maps that score onto the four zones
used throughout the dashboard: Normal, Monitoring, High Risk, Critical.
"""

from __future__ import annotations

from .data import RISK_ZONES

WEIGHTS = {
    "visual": 0.40,
    "behavioral": 0.35,
    "environmental": 0.15,
    "historical": 0.10,
}


class RiskEngine:
    def compute(self, visual: float, behavioral: float, environmental: float, historical: float) -> tuple[float, str, str]:
        score = (
            WEIGHTS["visual"] * visual
            + WEIGHTS["behavioral"] * behavioral
            + WEIGHTS["environmental"] * environmental
            + WEIGHTS["historical"] * historical
        )
        score = float(min(max(score, 0.0), 100.0))
        for lo, hi, label, key in RISK_ZONES:
            if lo <= score < hi:
                return score, label, key
        return score, "Critical", "critical"
