"""
Outage Prediction System — Demo Dashboard
Run: streamlit run demo/app.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GridShield AI — Outage Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: rgba(148,163,184,0.9) !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
}

/* Cards */
.glass-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.6), rgba(15,23,42,0.8));
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 32px;
    margin: 12px 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* Hero section */
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 50%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(148,163,184,0.8);
    font-weight: 400;
    line-height: 1.6;
    max-width: 700px;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s ease-in-out infinite;
}
.status-green { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
.status-red { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Risk gauge labels */
.risk-low { color: #22c55e; font-weight: 700; }
.risk-moderate { color: #f59e0b; font-weight: 700; }
.risk-high { color: #ef4444; font-weight: 700; }
.risk-critical { color: #dc2626; font-weight: 700; font-size: 1.2em; }

/* Contribution cards */
.contrib-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.5), rgba(15,23,42,0.7));
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    transition: border-color 0.3s ease;
}
.contrib-card:hover {
    border-color: rgba(99,102,241,0.4);
}
.contrib-number {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.contrib-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 6px;
}
.contrib-desc {
    font-size: 0.85rem;
    color: rgba(148,163,184,0.7);
    line-height: 1.5;
}

/* Divider */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 32px 0;
    border: none;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.9rem;
    color: rgba(148,163,184,0.6);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models & Data ───────────────────────────────────────
@st.cache_resource
def load_models():
    xgb = pickle.load(open(ROOT / "models" / "xgboost_model.pkl", "rb"))
    lgb = pickle.load(open(ROOT / "models" / "lightgbm_model.pkl", "rb"))
    scaler = pickle.load(open(ROOT / "models" / "scaler.pkl", "rb"))
    calibrator = pickle.load(open(ROOT / "models" / "calibrator.pkl", "rb"))
    return xgb, lgb, scaler, calibrator

@st.cache_data
def load_data():
    results = json.load(open(ROOT / "models" / "results.json"))
    cross_state = json.load(open(ROOT / "models" / "cross_state_results.json"))
    importance = pd.read_csv(ROOT / "models" / "feature_importance.csv")
    dataset = pd.read_parquet(ROOT / "data" / "processed" / "training_dataset.parquet")
    weather = pd.read_parquet(ROOT / "data" / "processed" / "weather_events.parquet")
    outages = pd.read_parquet(ROOT / "data" / "processed" / "outage_observations.parquet")
    shap_data = json.load(open(ROOT / "models" / "shap_values.json"))
    ablation = pd.read_csv(ROOT / "models" / "ablation_results.csv")
    return results, cross_state, importance, dataset, weather, outages, shap_data, ablation

xgb_model, lgb_model, scaler, calibrator = load_models()
results, cross_state, importance_df, dataset, weather_df, outage_df, shap_data, ablation_df = load_data()

# ── Plotly Theme ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    title_font=dict(size=16, color="#e2e8f0"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.1)"),
    yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zerolinecolor="rgba(99,102,241,0.1)"),
)
COLORS = ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#22c55e", "#f59e0b", "#ef4444"]

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0;">
        <div style="font-size: 2.5rem;">🛡️</div>
        <div style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">GridShield AI</div>
        <div style="font-size: 0.75rem; color: rgba(148,163,184,0.6); margin-top: 4px;">Outage Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Overview",
        "⚡ Live Prediction",
        "📊 Model Performance",
        "🔬 Feature Analysis",
        "🌎 Cross-State",
        "📁 Data Explorer",
    ], label_visibility="collapsed")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 12px; background: rgba(99,102,241,0.08); border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);">
        <div style="font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #818cf8; margin-bottom: 8px;">IEEE SPICES 2026</div>
        <div style="font-size: 0.8rem; color: #94a3b8;">Paper ID: <strong style="color: #e2e8f0;">822</strong></div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;"><span class="status-dot status-green"></span>Submitted</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div style="font-size: 0.7rem; color: rgba(148,163,184,0.4); line-height: 1.6;">
        <strong style="color: rgba(148,163,184,0.6);">Authors</strong><br>
        Kommuri &middot; Mahadev &middot; Chase<br><br>
        <strong style="color: rgba(148,163,184,0.6);">Guide</strong><br>
        Dr. Mohammed Adam Baba<br><br>
        <strong style="color: rgba(148,163,184,0.6);">Institution</strong><br>
        Malla Reddy University
    </div>
    """, unsafe_allow_html=True)

# ── Page: Overview ───────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="hero-badge">Research Paper — IEEE SPICES 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Compound Weather Event<br>Interaction Modeling</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Calibrated uncertainty estimation for state-agnostic power outage prediction using XGBoost/LightGBM ensemble with H3 spatial indexing and novel compound event features.</div>', unsafe_allow_html=True)

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", "0.967", "95% CI: 0.944-0.984")
    c2.metric("F1 Score", "0.947", "Threshold: 0.39")
    c3.metric("ECE", "0.004", "-98.4% from raw")
    c4.metric("Features", "138", "4 groups")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Contributions
    st.markdown('<div class="section-header">Research Contributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Three novel contributions to the power systems reliability literature</div>', unsafe_allow_html=True)

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("""
        <div class="contrib-card">
            <div class="contrib-number">01</div>
            <div class="contrib-title">Compound Event Features</div>
            <div class="contrib-desc">Co-occurrence matrices, pairwise severity interactions, and sequential escalation across 6 weather categories. CSI ranks 18th/138 features.</div>
        </div>
        """, unsafe_allow_html=True)
    with cc2:
        st.markdown("""
        <div class="contrib-card">
            <div class="contrib-number">02</div>
            <div class="contrib-title">Calibrated Uncertainty</div>
            <div class="contrib-desc">Ensemble disagreement + isotonic calibration reduces Expected Calibration Error from 0.267 to 0.004 — a 98.4% reduction.</div>
        </div>
        """, unsafe_allow_html=True)
    with cc3:
        st.markdown("""
        <div class="contrib-card">
            <div class="contrib-number">03</div>
            <div class="contrib-title">State-Agnostic Design</div>
            <div class="contrib-desc">YAML-driven config enables deployment to new states without retraining. TX model generalizes to CA/FL with &lt;0.7% AUC gap.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Dataset stats
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Training Samples", f"{results['dataset_info']['total_samples']:,}")
    dc2.metric("NOAA Events (TX)", f"{len(weather_df):,}")
    dc3.metric("Outage Records", f"{len(outage_df):,}")
    dc4.metric("Positive Rate", f"{results['dataset_info']['positive_rate']:.1%}")

# ── Page: Live Prediction ────────────────────────────────────
elif page == "⚡ Live Prediction":
    st.markdown('<div class="section-header">Live Outage Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Adjust weather and grid conditions — the trained ensemble predicts in real-time</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### Weather Conditions")
        weather_count = st.slider("Active storms (3h)", 0, 20, 3)
        max_magnitude = st.slider("Max magnitude (kts)", 0.0, 150.0, 45.0)
        trend_3h = st.slider("Outage trend (3h)", -0.5, 0.5, 0.1, 0.01)
        trend_6h = st.slider("Outage trend (6h)", -0.5, 0.5, 0.05, 0.01)
        compound_events = st.slider("Compound categories", 0, 6, 1)

        st.markdown("##### Grid Status")
        grid_load = st.slider("Load (MW)", 25000, 85000, 55000, 1000)
        reserve_margin = st.slider("Reserve margin (%)", 0.0, 30.0, 12.0, 0.5)

        st.markdown("##### Time")
        tc1, tc2 = st.columns(2)
        hour = tc1.slider("Hour", 0, 23, 14)
        month = tc2.slider("Month", 1, 12, 6)
        is_weekend = st.checkbox("Weekend")

    # Build feature vector
    feature_dict = {col: 0.0 for col in dataset.columns
                    if col not in ("h3_cell", "timestamp", "target_outage", "target_max_outage_fraction")}
    feature_dict.update({
        "weather_count_3h": float(weather_count),
        "weather_max_mag_3h": max_magnitude,
        "trend_outage_3h": trend_3h,
        "trend_outage_6h": trend_6h,
        "compound_event_count": float(compound_events),
        "compound_severity_index": min(1.0, compound_events / 6.0),
        "has_compound_event": float(compound_events >= 2),
        "current_load_mw": float(grid_load),
        "reserve_margin_pct": reserve_margin,
        "load_capacity_ratio": grid_load / 85000.0,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "is_weekend": float(is_weekend),
    })

    from src.ml.dataset import OutageDataset
    temp_ds = OutageDataset(dataset)
    feature_cols = temp_ds.feature_cols
    X = np.array([[feature_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
    lgb_prob = lgb_model.predict_proba(X_scaled)[0, 1]
    ensemble_prob = (xgb_prob + lgb_prob) / 2.0
    calibrated_prob = float(np.clip(calibrator.predict([ensemble_prob])[0], 0, 1))
    disagreement = abs(xgb_prob - lgb_prob) / 2.0

    if calibrated_prob < 0.25:
        risk_level, risk_color, risk_class = "LOW", "#22c55e", "risk-low"
    elif calibrated_prob < 0.55:
        risk_level, risk_color, risk_class = "MODERATE", "#f59e0b", "risk-moderate"
    elif calibrated_prob < 0.80:
        risk_level, risk_color, risk_class = "HIGH", "#ef4444", "risk-high"
    else:
        risk_level, risk_color, risk_class = "CRITICAL", "#dc2626", "risk-critical"

    with col2:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=calibrated_prob * 100,
            number={"suffix": "%", "font": {"size": 48, "family": "Inter", "color": "#e2e8f0"}},
            title={"text": "OUTAGE RISK", "font": {"size": 14, "color": "#94a3b8", "family": "Inter"}},
            delta={"reference": 50, "decreasing": {"color": "#22c55e"}, "increasing": {"color": "#ef4444"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar": {"color": risk_color, "thickness": 0.7},
                "bgcolor": "rgba(15,23,42,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(34,197,94,0.1)"},
                    {"range": [25, 55], "color": "rgba(245,158,11,0.1)"},
                    {"range": [55, 80], "color": "rgba(239,68,68,0.1)"},
                    {"range": [80, 100], "color": "rgba(220,38,38,0.15)"},
                ],
                "threshold": {"line": {"color": "#e2e8f0", "width": 2}, "thickness": 0.8, "value": calibrated_prob * 100},
            },
        ))
        fig.update_layout(height=280, **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis" and k != "yaxis"})
        st.plotly_chart(fig, use_container_width=True)

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Risk Level", risk_level)
        rc2.metric("XGBoost", f"{xgb_prob:.3f}")
        rc3.metric("LightGBM", f"{lgb_prob:.3f}")
        rc4.metric("Uncertainty", f"{disagreement:.4f}")

        # Recommended actions
        st.markdown("##### Recommended Actions")
        if risk_level == "LOW":
            st.success("Normal operations. No action required.")
        elif risk_level == "MODERATE":
            st.warning("Pre-stage restoration crews. Monitor NWS alerts.")
        elif risk_level == "HIGH":
            st.error("Activate emergency protocol. Notify affected customers. Deploy field teams.")
        else:
            st.error("CRITICAL: Full emergency response. Coordinate with emergency services. Open shelters.")

# ── Page: Model Performance ──────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comprehensive evaluation on 1,800 held-out test samples</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🎯 Calibration", "🧪 Ablation"])

    with tab1:
        metrics_data = []
        for model_name, metrics in results["metrics"].items():
            row = {"Model": model_name}
            row.update({k: round(v, 4) for k, v in metrics.items()})
            metrics_data.append(row)
        st.dataframe(pd.DataFrame(metrics_data).set_index("Model"), use_container_width=True)

        if "bootstrap_ci" in results:
            st.markdown("##### Bootstrap 95% Confidence Intervals")
            ci_data = []
            for metric, ci in results["bootstrap_ci"].items():
                if ci["mean"] > 0:
                    ci_data.append({
                        "Metric": metric, "Mean": f"{ci['mean']:.4f}",
                        "95% CI": f"[{ci['lower']:.4f}, {ci['upper']:.4f}]",
                    })
            st.dataframe(pd.DataFrame(ci_data).set_index("Metric"), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        c1.metric("ECE Before", f"{results['ece_before_calibration']:.4f}")
        c2.metric("ECE After", f"{results['ece_after_calibration']:.4f}")

        fig_path = ROOT / "paper" / "figures" / "fig1_reliability_diagram.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Reliability diagram: before vs after isotonic calibration")

    with tab3:
        abl_auc = ablation_df[ablation_df["metric"] == "auc_roc"].copy()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=abl_auc["group_removed"], y=abl_auc["full_value"],
            name="Full Model", marker_color="#6366f1", marker_line_width=0,
        ))
        fig.add_trace(go.Bar(
            x=abl_auc["group_removed"], y=abl_auc["ablated_value"],
            name="After Removal", marker_color="#ef4444", marker_line_width=0,
        ))
        fig.update_layout(
            title="AUC-ROC Impact of Removing Each Feature Group",
            barmode="group", height=400,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("> Removing **temporal features** collapses AUC from 0.968 to 0.489 — confirming outage prediction is fundamentally a temporal pattern recognition task.")

# ── Page: Feature Analysis ───────────────────────────────────
elif page == "🔬 Feature Analysis":
    st.markdown('<div class="section-header">Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">138 engineered features across 4 groups — temporal, spatial, compound, socioeconomic</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Importance Ranking", "🔍 SHAP Analysis"])

    with tab1:
        top_n = st.slider("Show top N features", 10, 50, 20)
        top_features = importance_df.head(top_n).copy()

        # Color by feature group
        def get_group(name):
            if any(k in name for k in ["weather_", "trend_", "lag_", "hour_", "dow_", "month_", "is_weekend", "current_load", "reserve_margin", "load_capacity"]):
                return "Temporal"
            elif any(k in name for k in ["neighbor_", "transmission", "distribution", "substation", "vegetation", "line_density", "infrastructure"]):
                return "Spatial"
            elif any(k in name for k in ["cooccur_", "interact_", "cat_", "seq_", "compound_", "has_compound"]):
                return "Compound"
            else:
                return "Socioeconomic"

        top_features["group"] = top_features["feature"].apply(get_group)
        group_colors = {"Temporal": "#6366f1", "Spatial": "#22c55e", "Compound": "#ef4444", "Socioeconomic": "#f59e0b"}

        fig = px.bar(
            top_features, x="avg_importance", y="feature", color="group",
            orientation="h", title=f"Top {top_n} Features",
            color_discrete_map=group_colors,
        )
        fig.update_layout(height=max(450, top_n * 28), yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        shap_df = pd.DataFrame(shap_data["top_features"])
        fig = px.bar(
            shap_df.head(20), x="mean_abs_shap", y="feature",
            orientation="h", title="Top 20 Features by Mean |SHAP| Value",
            color="mean_abs_shap", color_continuous_scale=["#1e1b4b", "#6366f1", "#818cf8"],
        )
        fig.update_layout(height=550, yaxis=dict(autorange="reversed"), **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        for path, caption, col in [
            ("fig_shap_beeswarm.png", "SHAP Beeswarm", col1),
            ("fig_shap_dependence_weather.png", "weather_count_3h vs compound_severity_index", col2),
        ]:
            fpath = ROOT / "paper" / "figures" / path
            if fpath.exists():
                col.image(str(fpath), caption=caption)

# ── Page: Cross-State ────────────────────────────────────────
elif page == "🌎 Cross-State":
    st.markdown('<div class="section-header">Cross-State Generalization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A single Texas-trained model evaluated on California and Florida data</div>', unsafe_allow_html=True)

    if "summary" in cross_state:
        for state, data in cross_state["summary"].items():
            xgb_data = data.get("XGBoost", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"### {state}")
            c2.metric("TX Model AUC", f"{xgb_data.get('tx_model_auc', 0):.4f}")
            c3.metric("Local Model AUC", f"{xgb_data.get('local_model_auc', 0):.4f}")
            c4.metric("Gap", f"{xgb_data.get('auc_gap', 0):.4f}")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    results_list = cross_state["all_results"]
    cs_df = pd.DataFrame(results_list)
    cs_xgb = cs_df[cs_df["model"].str.contains("XGBoost")].copy()

    fig = px.bar(
        cs_xgb, x="dataset", y="auc_roc", color="model",
        barmode="group", title="AUC-ROC Across States",
        color_discrete_sequence=COLORS,
    )
    fig.update_layout(height=400, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.info("The TX-trained model matches or exceeds locally-trained models on both CA and FL, validating the state-agnostic architecture.")

# ── Page: Data Explorer ──────────────────────────────────────
elif page == "📁 Data Explorer":
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse the actual data powering the predictions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌪️ Weather Events", "⚡ Outages", "📊 Training Data"])

    with tab1:
        st.metric("Total Events", f"{len(weather_df):,}")
        event_counts = weather_df["event_type"].value_counts()
        fig = px.bar(
            x=event_counts.index, y=event_counts.values,
            title="NOAA Storm Events by Type (Texas 2022)",
            labels={"x": "Event Type", "y": "Count"},
            color=event_counts.values, color_continuous_scale=["#1e1b4b", "#6366f1"],
        )
        fig.update_layout(height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(weather_df.head(50), use_container_width=True)

    with tab2:
        st.metric("Total Records", f"{len(outage_df):,}")
        fig = px.histogram(
            outage_df, x="outage_fraction", nbins=50,
            title="Distribution of Outage Severity",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.metric("Samples", f"{len(dataset):,}")
        target_counts = dataset["target_outage"].value_counts()
        fig = px.pie(
            values=target_counts.values, names=["No Outage", "Outage"],
            title="Target Distribution",
            color_discrete_sequence=["#22c55e", "#ef4444"],
            hole=0.55,
        )
        fig.update_layout(height=350, **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis" and k != "yaxis"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dataset.head(30), use_container_width=True)
