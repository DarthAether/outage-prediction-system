"""
GridShield AI — Outage Prediction Dashboard
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

st.set_page_config(
    page_title="GridShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DESIGN SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset ── */
.stApp { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 1rem; max-width: 1200px; }

/* ── Colors ── */
:root {
    --bg-deep: #060a13;
    --bg-card: rgba(15,20,35,0.6);
    --border: rgba(99,102,241,0.08);
    --border-hover: rgba(99,102,241,0.25);
    --accent: #6366f1;
    --accent-light: #818cf8;
    --text-primary: #f1f5f9;
    --text-secondary: #64748b;
    --text-muted: #334155;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
}

/* ── Top Bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-logo {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}
.topbar-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.topbar-badge {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    color: var(--green);
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Navigation ── */
.nav-row {
    display: flex;
    gap: 2px;
    background: rgba(15,20,35,0.5);
    padding: 3px;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 36px;
}
/* Override Streamlit button styles for nav */
.stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 9px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    padding: 10px 18px !important;
    transition: all 0.15s ease !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.01em !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.08) !important;
    color: var(--text-primary) !important;
}
.stButton > button:active, .stButton > button:focus {
    background: rgba(99,102,241,0.15) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}

/* ── Hero Number ── */
.hero-num {
    font-size: 6rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 8px;
}
.hero-sublabel {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Stat Row ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 24px 0;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.2s ease;
}
.stat-card:hover { border-color: var(--border-hover); }
.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}
.stat-delta {
    font-size: 0.65rem;
    color: var(--green);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section ── */
.sec-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.sec-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 24px;
    line-height: 1.5;
}

/* ── Contribution Blocks ── */
.cb-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.cb-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 24px;
    transition: border-color 0.2s ease, transform 0.2s ease;
    position: relative;
    overflow: hidden;
}
.cb-card:hover { border-color: var(--border-hover); transform: translateY(-2px); }
.cb-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-light));
    opacity: 0;
    transition: opacity 0.2s ease;
}
.cb-card:hover::before { opacity: 1; }
.cb-num {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--accent-light);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
}
.cb-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
    line-height: 1.3;
}
.cb-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 40px 0;
}

/* ── Metric Override ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs Override ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    color: var(--text-secondary);
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
}

/* ── Risk Indicator ── */
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.risk-low { background: rgba(16,185,129,0.1); color: var(--green); border: 1px solid rgba(16,185,129,0.2); }
.risk-mod { background: rgba(245,158,11,0.1); color: var(--amber); border: 1px solid rgba(245,158,11,0.2); }
.risk-high { background: rgba(239,68,68,0.1); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }
.risk-crit { background: rgba(220,38,38,0.15); color: #fca5a5; border: 1px solid rgba(220,38,38,0.3); }
</style>
""",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_models():
    xgb = pickle.load(open(ROOT / "models" / "xgboost_model.pkl", "rb"))
    lgb = pickle.load(open(ROOT / "models" / "lightgbm_model.pkl", "rb"))
    scaler = pickle.load(open(ROOT / "models" / "scaler.pkl", "rb"))
    calibrator = pickle.load(open(ROOT / "models" / "calibrator.pkl", "rb"))
    return xgb, lgb, scaler, calibrator


@st.cache_data
def load_data():
    r = json.load(open(ROOT / "models" / "results.json"))
    cs = json.load(open(ROOT / "models" / "cross_state_results.json"))
    imp = pd.read_csv(ROOT / "models" / "feature_importance.csv")
    ds = pd.read_parquet(ROOT / "data" / "processed" / "training_dataset.parquet")
    wx = pd.read_parquet(ROOT / "data" / "processed" / "weather_events.parquet")
    out = pd.read_parquet(ROOT / "data" / "processed" / "outage_observations.parquet")
    shap = json.load(open(ROOT / "models" / "shap_values.json"))
    abl = pd.read_csv(ROOT / "models" / "ablation_results.csv")
    return r, cs, imp, ds, wx, out, shap, abl


xgb_model, lgb_model, scaler, calibrator = load_models()
(
    results,
    cross_state,
    importance_df,
    dataset,
    weather_df,
    outage_df,
    shap_data,
    ablation_df,
) = load_data()

PLT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#64748b", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(99,102,241,0.06)", zerolinecolor="rgba(99,102,241,0.08)"
    ),
    yaxis=dict(
        gridcolor="rgba(99,102,241,0.06)", zerolinecolor="rgba(99,102,241,0.08)"
    ),
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOP BAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    """
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-logo">🛡️</div>
        <span class="topbar-name">GridShield AI</span>
    </div>
    <div class="topbar-right">
        <span class="topbar-badge">Paper #822 — manuscript under review</span>
        <span>Kommuri · Mahadev · Chase · Dr. Adam Baba</span>
        <span>Malla Reddy University</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NAVIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAGES = ["Overview", "Predict", "Performance", "Features", "Cross-State", "Data"]
cols = st.columns(len(PAGES))
for i, (c, p) in enumerate(zip(cols, PAGES)):
    if c.button(p, use_container_width=True, key=f"n{i}"):
        st.session_state["p"] = p
page = st.session_state.get("p", "Overview")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "Overview":
    # Hero
    left, right = st.columns([2, 3])
    with left:
        st.markdown("")
        st.markdown("")
        st.markdown('<div class="hero-num">0.966</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-label">Ensemble AUC-ROC · synthetic targets</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-sublabel">95% CI: [0.944, 0.984]</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">0.947</div>
                <div class="stat-label">F1 Score</div>
                <div class="stat-delta">threshold = 0.39</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">0.004</div>
                <div class="stat-label">ECE Calibrated</div>
                <div class="stat-delta">↓ 98.4% from raw</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">138</div>
                <div class="stat-label">Features</div>
                <div class="stat-delta">4 groups</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">1.00</div>
                <div class="stat-label">Precision</div>
                <div class="stat-delta">recall = 0.90</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Contributions
    st.markdown(
        '<div class="sec-title">Research Contributions</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sec-desc">Three novel contributions to the power systems reliability literature.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="cb-grid">
        <div class="cb-card">
            <div class="cb-num">Contribution 01</div>
            <div class="cb-title">Compound Event Features</div>
            <div class="cb-desc">Co-occurrence matrices, pairwise severity products, and sequential escalation scores across wind, ice, heat, flood, drought, and fire. The composite severity index ranks 18th among 138 features.</div>
        </div>
        <div class="cb-card">
            <div class="cb-num">Contribution 02</div>
            <div class="cb-title">Calibrated Uncertainty</div>
            <div class="cb-desc">Ensemble disagreement between XGBoost and LightGBM with post-hoc isotonic calibration. ECE reduced from 0.267 to 0.004 — a 98.4% improvement over raw predictions.</div>
        </div>
        <div class="cb-card">
            <div class="cb-num">Contribution 03</div>
            <div class="cb-title">Simulated Transfer Experiment</div>
            <div class="cb-desc">YAML configuration files encode region-specific thresholds and hazard profiles. Transfer was explored across state-context datasets produced by the same synthetic target generator; this is not measured utility validation.</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Dataset
    st.markdown('<div class="sec-title">Dataset</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Training Samples", f"{results['dataset_info']['total_samples']:,}")
    d2.metric("NOAA Events", f"{len(weather_df):,}")
    d3.metric("Outage Records", f"{len(outage_df):,}")
    d4.metric("Positive Rate", f"{results['dataset_info']['positive_rate']:.1%}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Predict":
    st.markdown('<div class="sec-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-desc">Adjust conditions. The trained ensemble predicts in real-time.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 3], gap="large")

    with left:
        weather_count = st.slider("Active storms (3h window)", 0, 20, 3)
        max_mag = st.slider("Max magnitude (knots)", 0.0, 150.0, 45.0)
        trend_3h = st.slider("Outage trend (3h)", -0.5, 0.5, 0.1, 0.01)
        trend_6h = st.slider("Outage trend (6h)", -0.5, 0.5, 0.05, 0.01)
        compound = st.slider("Compound event categories", 0, 6, 1)
        grid_load = st.slider("Grid load (MW)", 25000, 85000, 55000, 1000)
        reserve = st.slider("Reserve margin (%)", 0.0, 30.0, 12.0, 0.5)
        c1, c2 = st.columns(2)
        hour = c1.slider("Hour", 0, 23, 14)
        month = c2.slider("Month", 1, 12, 6)

    # Build features
    fd = {
        col: 0.0
        for col in dataset.columns
        if col
        not in ("h3_cell", "timestamp", "target_outage", "target_max_outage_fraction")
    }
    fd.update(
        {
            "weather_count_3h": float(weather_count),
            "weather_max_mag_3h": max_mag,
            "trend_outage_3h": trend_3h,
            "trend_outage_6h": trend_6h,
            "compound_event_count": float(compound),
            "compound_severity_index": min(1.0, compound / 6.0),
            "has_compound_event": float(compound >= 2),
            "current_load_mw": float(grid_load),
            "reserve_margin_pct": reserve,
            "load_capacity_ratio": grid_load / 85000.0,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        }
    )

    from src.ml.dataset import OutageDataset

    fcols = OutageDataset(dataset).feature_cols
    X = scaler.transform(np.array([[fd.get(c, 0.0) for c in fcols]], dtype=np.float32))

    xp = xgb_model.predict_proba(X)[0, 1]
    lp = lgb_model.predict_proba(X)[0, 1]
    ep = (xp + lp) / 2.0
    cp = float(np.clip(calibrator.predict([ep])[0], 0, 1))
    unc = abs(xp - lp) / 2.0

    if cp < 0.25:
        rl, rc, rcls = "LOW", "#10b981", "risk-low"
    elif cp < 0.55:
        rl, rc, rcls = "MODERATE", "#f59e0b", "risk-mod"
    elif cp < 0.80:
        rl, rc, rcls = "HIGH", "#ef4444", "risk-high"
    else:
        rl, rc, rcls = "CRITICAL", "#dc2626", "risk-crit"

    with right:
        # Big risk number
        st.markdown(
            f"""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size: 5rem; font-weight: 900; color: {rc}; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.04em; line-height: 1;">{cp:.0%}</div>
            <div style="margin-top: 12px;"><span class="risk-badge {rcls}">{rl} RISK</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("XGBoost", f"{xp:.3f}")
        m2.metric("LightGBM", f"{lp:.3f}")
        m3.metric("Ensemble", f"{ep:.3f}")
        m4.metric("Uncertainty", f"{unc:.4f}")

        if rl == "LOW":
            st.success("Normal operations. No action required.")
        elif rl == "MODERATE":
            st.warning("Pre-stage crews. Monitor NWS alerts. Notify on-call teams.")
        elif rl == "HIGH":
            st.error(
                "Deploy field teams. Issue customer alerts. Activate backup generation."
            )
        else:
            st.error(
                "EMERGENCY: Full response. Coordinate with first responders. Open shelters."
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Performance":
    st.markdown(
        '<div class="sec-title">Model Performance</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sec-desc">Evaluated on 1,800 held-out samples using the chronological row split implemented in the repository; no 72-hour embargo is currently applied.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Metrics", "Calibration", "Ablation"])

    with tab1:
        rows = []
        for name, m in results["metrics"].items():
            rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        if "bootstrap_ci" in results:
            st.markdown("##### Bootstrap 95% CI (1,000 resamples)")
            ci = []
            for met, v in results["bootstrap_ci"].items():
                if v["mean"] > 0:
                    ci.append(
                        {
                            "Metric": met,
                            "Mean": f"{v['mean']:.4f}",
                            "95% CI": f"[{v['lower']:.4f}, {v['upper']:.4f}]",
                        }
                    )
            st.dataframe(pd.DataFrame(ci).set_index("Metric"), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        c1.metric("ECE Before", f"{results['ece_before_calibration']:.4f}")
        c2.metric("ECE After", f"{results['ece_after_calibration']:.4f}")
        f = ROOT / "paper" / "figures" / "fig1_reliability_diagram.png"
        if f.exists():
            st.image(
                str(f),
                caption="Reliability diagram — before vs after isotonic calibration",
            )

    with tab3:
        a = ablation_df[ablation_df["metric"] == "auc_roc"].copy()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=a["group_removed"],
                y=a["full_value"],
                name="Full Model",
                marker_color="#6366f1",
            )
        )
        fig.add_trace(
            go.Bar(
                x=a["group_removed"],
                y=a["ablated_value"],
                name="Removed",
                marker_color="#ef4444",
            )
        )
        fig.update_layout(
            barmode="group",
            height=380,
            title="AUC-ROC: Full vs Feature Group Removed",
            **PLT,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Removing temporal features drops AUC from 0.968 → 0.489, confirming outage prediction is a temporal pattern recognition task."
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Features":
    st.markdown('<div class="sec-title">Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-desc">138 features across temporal, spatial, compound, and socioeconomic groups.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Importance", "SHAP"])

    with tab1:
        n = st.slider("Top features", 10, 40, 20)
        top = importance_df.nlargest(n, "xgb_importance").copy()

        def grp(name):
            if any(
                k in name
                for k in [
                    "weather_",
                    "trend_",
                    "lag_",
                    "hour_",
                    "dow_",
                    "month_",
                    "is_weekend",
                    "current_load",
                    "reserve_margin",
                    "load_capacity",
                ]
            ):
                return "Temporal"
            elif any(
                k in name
                for k in [
                    "neighbor_",
                    "transmission",
                    "distribution",
                    "substation",
                    "vegetation",
                    "line_density",
                    "infrastructure",
                ]
            ):
                return "Spatial"
            elif any(
                k in name
                for k in [
                    "cooccur_",
                    "interact_",
                    "cat_",
                    "seq_",
                    "compound_",
                    "has_compound",
                ]
            ):
                return "Compound"
            return "Socioeconomic"

        top["group"] = top["feature"].apply(grp)
        fig = px.bar(
            top,
            x="xgb_importance",
            y="feature",
            color="group",
            orientation="h",
            color_discrete_map={
                "Temporal": "#6366f1",
                "Spatial": "#10b981",
                "Compound": "#ef4444",
                "Socioeconomic": "#f59e0b",
            },
        )
        fig.update_layout(
            height=max(400, n * 26), yaxis=dict(autorange="reversed"), **PLT
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "XGBoost importance only. The stored LightGBM split counts use a different scale and are not averaged here."
        )

    with tab2:
        sdf = pd.DataFrame(shap_data["top_features"]).head(20)
        fig = px.bar(
            sdf,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale=["#1e1b4b", "#6366f1", "#818cf8"],
        )
        fig.update_layout(
            height=520,
            yaxis=dict(autorange="reversed"),
            title="Mean |SHAP| Value",
            **PLT,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        for path, cap, col in [
            ("fig_shap_beeswarm.png", "SHAP Beeswarm", c1),
            ("fig_shap_dependence_weather.png", "Dependence Plot", c2),
        ]:
            fp = ROOT / "paper" / "figures" / path
            if fp.exists():
                col.image(str(fp), caption=cap)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CROSS-STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Cross-State":
    st.markdown(
        '<div class="sec-title">Preliminary Simulated Transfer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-desc">A Texas-context model evaluated on California- and Florida-context datasets produced by the same synthetic target generator.</div>',
        unsafe_allow_html=True,
    )

    if "summary" in cross_state:
        for state, data in cross_state["summary"].items():
            xd = data.get("XGBoost", {})
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            c1.markdown(f"#### {state}")
            c2.metric("TX Model", f"{xd.get('tx_model_auc', 0):.4f}")
            c3.metric("Local Model", f"{xd.get('local_model_auc', 0):.4f}")
            c4.metric("Gap", f"{abs(xd.get('auc_gap', 0)):.4f}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    cs_df = pd.DataFrame(cross_state["all_results"])
    cs_xgb = cs_df[cs_df["model"].str.contains("XGBoost")]
    fig = px.bar(
        cs_xgb,
        x="dataset",
        y="auc_roc",
        color="model",
        barmode="group",
        color_discrete_sequence=["#6366f1", "#818cf8", "#a5b4fc", "#10b981"],
    )
    fig.update_layout(height=380, title="AUC-ROC Across States", **PLT)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "These small within-generator gaps are preliminary results in a simulated setup; they do not validate real-world geographic transfer."
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "Data":
    st.markdown('<div class="sec-title">Data Explorer</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Weather Events", "Outages", "Training Data"])

    with tab1:
        st.metric("Events", f"{len(weather_df):,}")
        ec = weather_df["event_type"].value_counts()
        fig = px.bar(
            x=ec.index,
            y=ec.values,
            labels={"x": "Type", "y": "Count"},
            color=ec.values,
            color_continuous_scale=["#1e1b4b", "#6366f1"],
        )
        fig.update_layout(height=360, title="NOAA Storm Events — Texas 2022", **PLT)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(weather_df.head(50), use_container_width=True)

    with tab2:
        st.metric("Records", f"{len(outage_df):,}")
        fig = px.histogram(
            outage_df,
            x="outage_fraction",
            nbins=50,
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(height=320, title="Outage Severity Distribution", **PLT)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.metric("Samples", f"{len(dataset):,}")
        tc = dataset["target_outage"].value_counts()
        fig = px.pie(
            values=tc.values,
            names=["No Outage", "Outage"],
            hole=0.6,
            color_discrete_sequence=["#10b981", "#ef4444"],
        )
        fig.update_layout(
            height=320,
            title="Target Distribution",
            **{k: v for k, v in PLT.items() if "axis" not in k},
        )
        st.plotly_chart(fig, use_container_width=True)
