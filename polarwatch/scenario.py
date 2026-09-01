"""Orchestrates one full pipeline pass: Capture → Detect → Identify →
Assess Health → Analyze Anomalies → Predict Risk → Alert Researchers.

Holds the running state for a session (detection log, anomaly timeline,
alert history) and scripts a slow-building outbreak in one species so
repeated runs demonstrate Normal → Monitoring → High Risk → Critical
escalation and a resulting alert, rather than relying on pure chance.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .alerts import Alert, AlertManager
from .behavior import BehaviorAnalyzer
from .data import RESEARCH_STATIONS, SPECIES_NAMES
from .detection import Detection, SimulatedDetector
from .health import HealthAssessor
from .risk_engine import RiskEngine

ANOMALY_WINDOW = 12


class Scenario:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.detector = SimulatedDetector(self.rng)
        self.health = HealthAssessor(self.rng)
        self.behavior = BehaviorAnalyzer()
        self.risk_engine = RiskEngine()
        self.alert_manager = AlertManager()

        self.cycle = 0
        self.outbreak_species: str = str(self.rng.choice(SPECIES_NAMES))
        self.outbreak_station: str = str(self.rng.choice(RESEARCH_STATIONS))
        self.outbreak_level = 0.0

        self.detection_log: list[dict] = []
        self.anomaly_history: list[float] = []
        self.last_detections: list[Detection] = []
        self.last_flagged: str | None = None
        self.last_score = 0.0
        self.last_zone_label = "Normal"
        self.last_zone_key = "good"
        self.last_factors = {
            "Visual Features": 0.0,
            "Behavioural Patterns": 0.0,
            "Environmental Data": 0.0,
            "Historical Records": 0.0,
        }
        self.last_alert: Alert | None = None

    def run_cycle(self, station: str) -> None:
        self.cycle += 1
        self.outbreak_level = min(self.outbreak_level + self.rng.uniform(11, 17), 95)
        now = dt.datetime.now()

        detections = self.detector.detect()
        self.last_detections = detections

        tracked_anomaly = None
        worst_score, worst_species, worst_zone_label, worst_zone_key, worst_factors = -1.0, None, "Normal", "good", None

        for det in detections:
            visual = self.health.assess(det.species, self.outbreak_species, self.outbreak_level)
            anomaly, _series = self.behavior.update(det.species, visual, self.rng)

            # An active outbreak also shows up in the colony's environmental
            # readings and its deviation from historical baselines, not just
            # in the animal that was detected — so both signals climb with
            # the tracked species' outbreak level, on top of normal variability.
            is_tracked = det.species == self.outbreak_species
            environmental = float(self.rng.uniform(10, 28)) + (self.outbreak_level * 0.55 if is_tracked else 0.0)
            historical = float(self.rng.uniform(5, 16)) + (self.outbreak_level * 0.40 if is_tracked else 0.0)
            environmental = min(environmental, 100.0)
            historical = min(historical, 100.0)

            score, zone_label, zone_key = self.risk_engine.compute(visual, anomaly, environmental, historical)

            self.detection_log.insert(0, {"time": now.strftime("%H:%M:%S"), "species": det.species, "confidence": det.confidence})

            if det.species == self.outbreak_species:
                tracked_anomaly = anomaly

            if score > worst_score:
                worst_score, worst_species, worst_zone_label, worst_zone_key = score, det.species, zone_label, zone_key
                worst_factors = {
                    "Visual Features": visual,
                    "Behavioural Patterns": anomaly,
                    "Environmental Data": environmental,
                    "Historical Records": historical,
                }

        if tracked_anomaly is None:
            visual = self.health.assess(self.outbreak_species, self.outbreak_species, self.outbreak_level)
            tracked_anomaly, _series = self.behavior.update(self.outbreak_species, visual, self.rng)

        self.anomaly_history.append(tracked_anomaly)
        self.anomaly_history = self.anomaly_history[-ANOMALY_WINDOW:]
        self.detection_log = self.detection_log[:6]

        self.last_score = worst_score
        self.last_zone_label = worst_zone_label
        self.last_zone_key = worst_zone_key
        self.last_factors = worst_factors or self.last_factors
        self.last_flagged = worst_species if worst_zone_key in ("serious", "critical") else None

        self.last_alert = self.alert_manager.evaluate(
            species=worst_species or self.outbreak_species,
            station=station,
            score=worst_score,
            zone_label=worst_zone_label,
            zone_key=worst_zone_key,
            now=now,
        )
