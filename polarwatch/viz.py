"""Matplotlib rendering for the risk gauge — the one place a chart needs
custom drawing rather than a native Streamlit element."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from .data import RISK_ZONES

ZONE_COLORS = {
    "good": "#1F9D6B",
    "warning": "#B07A10",
    "serious": "#C15A1F",
    "critical": "#C53832",
}

NEEDLE_COLOR = "#0F2530"


def draw_gauge(score: float):
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for lo, hi, _label, key in RISK_ZONES:
        hi_clamped = min(hi, 100)
        theta1 = 180 - 1.8 * hi_clamped
        theta2 = 180 - 1.8 * lo
        ax.add_patch(
            Wedge((0, 0), 1.0, theta1, theta2, width=0.32, facecolor=ZONE_COLORS[key], edgecolor="white", linewidth=1.4)
        )

    clamped = max(0.0, min(score, 100.0))
    angle = math.radians(180 - 1.8 * clamped)
    x2, y2 = 0.82 * math.cos(angle), 0.82 * math.sin(angle)
    ax.plot([0, x2], [0, y2], color=NEEDLE_COLOR, linewidth=3, solid_capstyle="round")
    ax.add_patch(Circle((0, 0), 0.045, color=NEEDLE_COLOR))

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.08, 1.15)
    return fig
