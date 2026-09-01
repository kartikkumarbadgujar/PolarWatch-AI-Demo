"""Static reference data: monitored species, stations, and detection layout."""

from __future__ import annotations

# Frame size used by the simulated camera feed (scene.py).
FRAME_W, FRAME_H = 640, 400

# Fixed bounding-box slots the simulated detector "finds" animals in.
# (x, y, w, h) in frame pixels — three animals per camera frame.
DETECTION_SLOTS = [
    (70, 230, 95, 80),
    (260, 245, 85, 65),
    (440, 210, 115, 95),
]

RESEARCH_STATIONS = [
    "Halley Research Station",
    "Rothera Point",
    "Casey Station",
    "McMurdo Sound",
]

SPECIES = [
    {"name": "Adélie Penguin", "latin": "Pygoscelis adeliae", "baseline_population": 3400, "color": (60, 60, 60)},
    {"name": "Emperor Penguin", "latin": "Aptenodytes forsteri", "baseline_population": 1180, "color": (40, 40, 50)},
    {"name": "Weddell Seal", "latin": "Leptonychotes weddellii", "baseline_population": 640, "color": (120, 120, 130)},
    {"name": "Leopard Seal", "latin": "Hydrurga leptonyx", "baseline_population": 95, "color": (100, 105, 90)},
    {"name": "South Polar Skua", "latin": "Stercorarius maccormicki", "baseline_population": 210, "color": (70, 55, 45)},
]

SPECIES_NAMES = [s["name"] for s in SPECIES]

# Risk-zone thresholds shared by the risk engine, the gauge, and the alert log.
RISK_ZONES = [
    (0, 30, "Normal", "good"),
    (30, 55, "Monitoring", "warning"),
    (55, 80, "High Risk", "serious"),
    (80, 101, "Critical", "critical"),
]

PIPELINE_STEPS = [
    "Capture",
    "Detect",
    "Identify",
    "Assess Health",
    "Analyze Anomalies",
    "Predict Risk",
    "Alert Researchers",
]
