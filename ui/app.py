from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from env.digital_twin_env import DigitalTwinDiabetesEnv  # noqa: E402


ACTION_PRESETS: Dict[str, Dict[str, Any]] = {
    "NOOP": {"type": "noop"},
    "Start Metformin + Lifestyle": {"type": "start", "drug": "metformin", "dose": 1.0, "lifestyle": 0.7},
    "Add GLP-1": {"type": "add", "drug": "glp1", "dose": 1.0},
    "Add SGLT2": {"type": "add", "drug": "sglt2", "dose": 1.0},
    "Start Insulin": {"type": "start", "drug": "insulin", "dose": 0.7},
    "Add DPP-4": {"type": "add", "drug": "dpp4", "dose": 1.0},
    "Add Sulfonylurea": {"type": "add", "drug": "sulfonylurea", "dose": 0.7},
}


@dataclass
class RolloutLog:
    obs: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)


def _inject_css() -> None:
    st.markdown(
        """
<style>
  :root {
    --bg: #0b1020;
    --panel: rgba(255,255,255,0.06);
    --panel2: rgba(255,255,255,0.08);
    --stroke: rgba(255,255,255,0.09);
    --text: rgba(255,255,255,0.92);
    --muted: rgba(255,255,255,0.65);
    --accent: #7c3aed;
    --good: #22c55e;
    --bad: #ef4444;
    --warn: #f59e0b;
  }
  .stApp { background: radial-gradient(1000px 600px at 20% 10%, rgba(124,58,237,0.22), transparent 60%),
                    radial-gradient(800px 500px at 90% 0%, rgba(34,197,94,0.10), transparent 55%),
                    var(--bg); color: var(--text); }
  [data-testid="stSidebar"] { background: rgba(255,255,255,0.03) !important; border-right: 1px solid var(--stroke); }
  .card {
    background: linear-gradient(180deg, var(--panel), rgba(255,255,255,0.03));
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 14px 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  .card h4 { margin: 0 0 4px 0; font-size: 12px; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
  .card .big { font-size: 22px; font-weight: 700; margin: 0; }
  .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--stroke); background: rgba(255,255,255,0.04); color: var(--muted); font-size: 12px; }
  .pill.good { color: rgba(34,197,94,0.95); border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.08); }
  .pill.bad { color: rgba(239,68,68,0.95); border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.08); }
  .pill.warn { color: rgba(245,158,11,0.95); border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.08); }
  .section-title { font-size: 14px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); margin: 4px 0 10px 0; }
  .muted { color: var(--muted); }
</style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_env():
    if "env" not in st.session_state:
        st.session_state.env = DigitalTwinDiabetesEnv(seed=0, max_steps=24)
        o, info = st.session_state.env.reset(seed=0)
        st.session_state.patient_id = info.get("patient_id")
        st.session_state.log = RolloutLog(obs=[o])


def _reset(seed: int, max_steps: int):
    st.session_state.env = DigitalTwinDiabetesEnv(seed=seed, max_steps=max_steps)
    o, info = st.session_state.env.reset(seed=seed)
    st.session_state.patient_id = info.get("patient_id")
    st.session_state.log = RolloutLog(obs=[o])


def _step(action: Dict[str, Any]):
    env = st.session_state.env
    log: RolloutLog = st.session_state.log
    o, r, term, trunc, info = env.step(action)
    log.obs.append(o)
    log.rewards.append(float(r))
    log.infos.append(info)
    return term, trunc


def _kpi_card(title: str, value: str, delta: str | None = None, pill: str | None = None) -> None:
    extra = f"<span class='muted'>{delta}</span>" if delta else ""
    pill_html = ""
    if pill:
        klass = "pill"
        if pill.lower() in {"ok", "good", "stable", "responding"}:
            klass += " good"
        elif pill.lower() in {"fail", "bad", "critical"}:
            klass += " bad"
        else:
            klass += " warn"
        pill_html = f"<span class='{klass}' style='float:right'>{pill}</span>"
    st.markdown(
        f"""
<div class="card">
  {pill_html}
  <h4>{title}</h4>
  <div class="big">{value}</div>
  <div style="margin-top:6px">{extra}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _sparkline(y: List[float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", line=dict(color="#a78bfa", width=2)))
    fig.update_layout(
        template="plotly_dark",
        height=160,
        margin=dict(l=10, r=10, t=35, b=10),
        title=dict(text=title, x=0.02, font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _render_overview(df: pd.DataFrame, log: RolloutLog) -> None:
    st.markdown("<div class='section-title'>Overview</div>", unsafe_allow_html=True)

    latest = df.iloc[-1].to_dict() if len(df) else {}
    prev = df.iloc[-2].to_dict() if len(df) > 1 else latest

    h = float(latest.get("hba1c", 0.0))
    g = float(latest.get("fasting_glucose", 0.0))
    egfr = float(latest.get("egfr", 0.0))
    bmi = float(latest.get("bmi", 0.0))
    ret = float(df["return_to_date"].iloc[-1]) if "return_to_date" in df else 0.0

    dh = float(prev.get("hba1c", h)) - h
    dg = float(prev.get("fasting_glucose", g)) - g
    deg = egfr - float(prev.get("egfr", egfr))
    dbmi = float(prev.get("bmi", bmi)) - bmi

    fd = (log.infos[-1].get("fall_detection") if log.infos else {}) or {}
    fd_status = "Stable"
    if fd.get("failed"):
        fd_status = str(fd.get("category", "warning")).title()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi_card("Return", f"{ret:+.2f}", delta="episode-to-date", pill=fd_status)
    with k2:
        _kpi_card("HbA1c", f"{h:.2f} %", delta=f"Δ {dh:+.2f} this week")
    with k3:
        _kpi_card("FPG", f"{g:.0f} mg/dL", delta=f"Δ {dg:+.0f} this week")
    with k4:
        _kpi_card("eGFR", f"{egfr:.0f}", delta=f"Δ {deg:+.2f} this week")
    with k5:
        _kpi_card("BMI", f"{bmi:.1f}", delta=f"Δ {dbmi:+.2f} this week")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(
            _sparkline([float(x) for x in df["hba1c"].tolist()], "HbA1c (sparkline)"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _sparkline([0.0] + [float(x) for x in log.rewards], "Reward (sparkline)"),
            use_container_width=True,
        )

    st.markdown("<div class='section-title'>Latest decision</div>", unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if log.infos:
            last = log.infos[-1]
            st.write("**Action**")
            st.code(json.dumps(last.get("action", {}), indent=2), language="json")
            st.write("**Fall detection**")
            st.json(last.get("fall_detection", {}))
        else:
            st.info("Step once to populate decision info.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if log.infos:
            last = log.infos[-1]
            st.write("**Reward components**")
            st.json(last.get("reward", {}))
            st.write("**Cost / side-effects**")
            st.json(
                {
                    "weekly_cost_usd": last.get("weekly_cost_usd"),
                    "side_effects": last.get("side_effects"),
                }
            )
        st.markdown("</div>", unsafe_allow_html=True)


def _render_patient(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Patient trajectory</div>", unsafe_allow_html=True)
    if len(df) < 2:
        st.info("Step a few times to see trajectories.")
        return

    tabs = st.tabs(["HbA1c", "Fasting glucose", "Kidney (eGFR)", "Vitals"])
    with tabs[0]:
        st.plotly_chart(px.line(df, x="week", y="hba1c", markers=True, template="plotly_dark"), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(px.line(df, x="week", y="fasting_glucose", markers=True, template="plotly_dark"), use_container_width=True)
    with tabs[2]:
        st.plotly_chart(px.line(df, x="week", y="egfr", markers=True, template="plotly_dark"), use_container_width=True)
    with tabs[3]:
        sub = df[["week", "bmi", "systolic_bp"]].copy()
        fig = px.line(sub.melt(id_vars=["week"]), x="week", y="value", color="variable", markers=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Raw observations</div>", unsafe_allow_html=True)
    st.dataframe(df.tail(20), use_container_width=True, height=280)


def _render_tools(log: RolloutLog) -> None:
    st.markdown("<div class='section-title'>Tool outputs (last step)</div>", unsafe_allow_html=True)
    if not log.infos:
        st.info("Step once to populate tool-related info.")
        return
    last = log.infos[-1]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("These are the structured signals the agent uses (mock tools are implemented in `tools/`).")
    st.json(
        {
            "fall_detection": last.get("fall_detection"),
            "side_effects": last.get("side_effects"),
            "cvd_event": last.get("cvd_event"),
            "action_parse": last.get("action_parse"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Digital Twin Medicine (T2DM)", layout="wide")
    _inject_css()

    top = st.container()
    with top:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("## 🧬 Digital Twin Medicine")
            st.markdown("<div class='muted'>A Personalized Treatment RL Agent • OpenEnv-style world modeling</div>", unsafe_allow_html=True)
        with right:
            pid = st.session_state.get("patient_id", "unknown")
            st.markdown(f"<div style='text-align:right'><span class='pill'>patient_id: {pid}</span></div>", unsafe_allow_html=True)

    _ensure_env()

    with st.sidebar:
        st.markdown("### Control Panel")
        page = st.radio("Navigate", ["Dashboard", "Patient", "Tools"], label_visibility="collapsed")

        st.markdown("#### Episode")
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=0, step=1)
        max_steps = st.slider("Max steps (weeks)", min_value=8, max_value=52, value=24, step=1)
        if st.button("Reset episode", use_container_width=True):
            _reset(int(seed), int(max_steps))

        st.markdown("#### Action")
        preset = st.selectbox("Preset", list(ACTION_PRESETS.keys()))
        action = dict(ACTION_PRESETS[preset])

        # Optional knobs
        if action.get("type") in {"start", "add", "dose_adjust"}:
            action["dose"] = st.slider("Dose (0..1)", 0.0, 1.0, float(action.get("dose", 1.0)), 0.05)
        if "lifestyle" in action or st.checkbox("Set lifestyle", value="lifestyle" in action):
            action["lifestyle"] = st.slider("Lifestyle (0..1)", 0.0, 1.0, float(action.get("lifestyle", 0.4)), 0.05)

        if st.button("Step", use_container_width=True):
            term, trunc = _step(action)
            if term or trunc:
                st.warning("Episode finished. Reset to continue.")

        with st.expander("Action JSON", expanded=False):
            st.code(json.dumps(action, indent=2), language="json")

    log: RolloutLog = st.session_state.log
    df = pd.DataFrame(log.obs)
    df["reward"] = [None] + log.rewards
    df["return_to_date"] = pd.Series([0.0] + list(pd.Series(log.rewards).cumsum()))

    if page == "Dashboard":
        _render_overview(df, log)
    elif page == "Patient":
        _render_patient(df)
    else:
        _render_tools(log)


if __name__ == "__main__":
    main()

