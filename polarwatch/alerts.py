"""The early-warning alerting stage (slide 4, step 7).

Any cycle whose worst-case risk score lands in "High Risk" or "Critical"
produces an Alert carrying exactly what the proposal calls for: species,
location, time, and risk level.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class Alert:
    time: str
    species: str
    station: str
    score: float
    zone_label: str
    zone_key: str


class AlertManager:
    def __init__(self):
        self.alerts: list[Alert] = []

    def evaluate(self, species: str, station: str, score: float, zone_label: str, zone_key: str, now: dt.datetime) -> Alert | None:
        if zone_key not in ("serious", "critical"):
            return None
        alert = Alert(
            time=now.strftime("%H:%M:%S"),
            species=species,
            station=station,
            score=score,
            zone_label=zone_label,
            zone_key=zone_key,
        )
        self.alerts.insert(0, alert)
        return alert
