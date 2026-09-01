"""PolarWatch AI — pipeline simulation dashboard.

Run with:  streamlit run app.py

Everything in this file is Streamlit's native component API — no HTML/CSS/JS
is authored by hand. The look comes from .streamlit/config.toml and from
Streamlit's own layout primitives (columns, tabs, metrics, progress bars).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from polarwatch.data import PIPELINE_STEPS, RESEARCH_STATIONS, SPECIES
from polarwatch.scenario import Scenario
from polarwatch.scene import render_frame
from polarwatch.viz import draw_gauge

ALERT_THRESHOLD = 55

st.set_page_config(page_title="PolarWatch AI", page_icon="🐧", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- state ----

if "scenario" not in st.session_state:
    st.session_state.scenario = Scenario(seed=7)

if "population" not in st.session_state:
    rng = np.random.default_rng(3)
    rows = []
    for sp in SPECIES:
        walk = [sp["baseline_population"]]
        for _ in range(6):
            step = int(rng.normal(0, sp["baseline_population"] * 0.012 + 1))
            walk.append(max(0, walk[-1] + step))
        rows.append(
            {
                "Species": sp["name"],
                "Latin Name": sp["latin"],
                "Population (est.)": walk[-1],
                "7-Day Trend": walk,
                "Δ 7d": walk[-1] - walk[0],
            }
        )
    st.session_state.population = pd.DataFrame(rows)

scenario: Scenario = st.session_state.scenario

population_df = st.session_state.population.copy()
population_df["Status"] = population_df["Species"].apply(
    lambda s: scenario.last_zone_label if s == scenario.outbreak_species else "Normal"
)

# -------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.markdown("## Control Room")
    default_index = RESEARCH_STATIONS.index(scenario.outbreak_station)
    station = st.selectbox("Research station", RESEARCH_STATIONS, index=default_index)
    run_clicked = st.button("Run Detection Cycle", use_container_width=True, type="primary")
    reset_clicked = st.button("Reset Simulation", use_container_width=True)
    st.markdown("---")
    st.caption(f"Cycle **{scenario.cycle}** completed")
    with st.expander("How this demo works"):
        st.write(
            "Every score here comes from a synthetic pipeline in `polarwatch/`, "
            "not a trained model — the goal is to prove out the end-to-end flow "
            "before YOLO/CNN/RNN training happens. Press **Run Detection Cycle** "
            "repeatedly to watch one species' risk climb from Normal toward "
            "Critical and trigger an automated alert, the same escalation the "
            "finished system is designed to catch early."
        )

if reset_clicked:
    del st.session_state["scenario"]
    st.rerun()

# --------------------------------------------------------------- header ----

st.title("PolarWatch AI")
st.caption("AI-powered Antarctic wildlife monitoring & health database — live pipeline simulation")

open_alerts = len(scenario.alert_manager.alerts)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Camera Nodes Online", "18 / 20")
k2.metric("Species Tracked", str(len(SPECIES)))
k3.metric("Colonies Monitored", "12")
k4.metric(
    "Open Alerts",
    str(open_alerts),
    delta=("+1" if scenario.last_alert else None),
    delta_color="inverse",
)

tab_live, tab_population, tab_about = st.tabs(["Live Monitoring", "Population & Health", "About the Pipeline"])

# ---------------------------------------------------------- live monitor ----

with tab_live:
    if run_clicked:
        step_placeholder = st.empty()
        for i, step in enumerate(PIPELINE_STEPS):
            rendered = "  →  ".join(f"**{s}**" if j <= i else s for j, s in enumerate(PIPELINE_STEPS))
            step_placeholder.markdown(rendered)
            time.sleep(0.16)
        scenario.run_cycle(station)
        step_placeholder.empty()

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.subheader("Live Detection Feed")
        st.caption(f"{station} · simulated frame-by-frame species detection")

        frame = render_frame(scenario.last_detections, scenario.last_flagged)
        st.image(frame, use_container_width=True)

        if scenario.cycle == 0:
            st.caption("Press **Run Detection Cycle** in the sidebar to bring the feed online.")
        else:
            done_line = "  →  ".join(f"**{s}**" for s in PIPELINE_STEPS)
            st.markdown(done_line)

        st.markdown("**Recent detections**")
        if scenario.detection_log:
            log_df = pd.DataFrame(scenario.detection_log)
            log_df["confidence"] = (log_df["confidence"] * 100).round(1)
            log_df = log_df.rename(columns={"time": "Time", "species": "Species", "confidence": "Confidence (%)"})
            st.dataframe(log_df, hide_index=True, use_container_width=True, height=180)
        else:
            st.caption("No detections yet.")

    with right:
        st.subheader("Health Risk Engine")
        st.pyplot(draw_gauge(scenario.last_score), use_container_width=True)

        st.metric("Risk Score", f"{scenario.last_score:.0f} / 100")

        zone_box = {"good": st.success, "warning": st.info, "serious": st.warning, "critical": st.error}[scenario.last_zone_key]
        subject = scenario.last_flagged or scenario.outbreak_species
        zone_box(f"{scenario.last_zone_label} — assessed for {subject}")

        st.markdown("**Contributing factors**")
        for label, value in scenario.last_factors.items():
            st.progress(min(int(value), 100) / 100, text=f"{label} · {value:.0f}")

    st.divider()

    chart_col, alert_col = st.columns([1.3, 1], gap="large")

    with chart_col:
        st.subheader("Behavioural Anomaly Timeline")
        st.caption(f"Tracking: {scenario.outbreak_species}")
        if scenario.anomaly_history:
            chart_df = pd.DataFrame(
                {
                    "Cycle": list(range(1, len(scenario.anomaly_history) + 1)),
                    "Anomaly Score": scenario.anomaly_history,
                    "Alert Threshold": [ALERT_THRESHOLD] * len(scenario.anomaly_history),
                }
            ).set_index("Cycle")
            st.line_chart(chart_df, color=["#0E8FA3", "#C15A1F"], use_container_width=True, height=260)
        else:
            st.caption("Timeline will populate once detection cycles run.")

    with alert_col:
        st.subheader("Alert Log")
        if scenario.alert_manager.alerts:
            for alert in scenario.alert_manager.alerts[:6]:
                box = st.error if alert.zone_key == "critical" else st.warning
                box(f"**{alert.species}** · {alert.zone_label} ({alert.score:.0f}/100)  \n{alert.station} · {alert.time}")
        else:
            st.caption("No alerts raised yet — alerts fire once a risk score crosses into High Risk or Critical.")

# ---------------------------------------------------------- population ----

with tab_population:
    st.subheader("Population & Health Overview")
    st.caption("Estimated colony sizes, a 7-day trend, and each species' latest automated health status.")
    st.dataframe(
        population_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "7-Day Trend": st.column_config.LineChartColumn("7-Day Trend", width="medium"),
            "Population (est.)": st.column_config.NumberColumn("Population (est.)", format="%d"),
            "Δ 7d": st.column_config.NumberColumn("Δ 7d", format="%+d"),
        },
    )

# ---------------------------------------------------------------- about ----

with tab_about:
    st.subheader("How the pipeline maps to this demo")
    mapping = pd.DataFrame(
        [
            ("Data Acquisition & Ingestion", "polarwatch/data.py", "Static reference data for now — will become live camera/sensor ingestion."),
            ("Real-Time Species Detection", "polarwatch/detection.py", "Simulates YOLO output shape; a trained model swaps in behind the same interface."),
            ("Visual Health Assessment", "polarwatch/health.py", "Simulates a CNN's visual-concern score; scripted to escalate for one species."),
            ("Temporal Behaviour & Anomaly Analysis", "polarwatch/behavior.py", "Simulates an RNN's anomaly score from a rolling per-species history."),
            ("Multimodal Risk Engine", "polarwatch/risk_engine.py", "Real weighted combination of the four signal types into one 0-100 score."),
            ("Alerting", "polarwatch/alerts.py", "Real threshold logic — fires whenever a score crosses High Risk or Critical."),
        ],
        columns=["Pipeline stage", "Module", "Status in this demo"],
    )
    st.dataframe(mapping, hide_index=True, use_container_width=True)
    st.info(
        "No YOLO/CNN/RNN model is loaded or trained here — every score is generated "
        "by the lightweight synthetic logic in `polarwatch/`. This dashboard exists "
        "to prove out the end-to-end UX and data flow ahead of model training."
    )
    st.markdown(
        "**Team FlashCoderX** · Build with Bharat 2.0 · "
        "[PolarWatch-AI on GitHub](https://github.com/ojasviv3-ctrl/PolarWatch-AI-)"
    )
