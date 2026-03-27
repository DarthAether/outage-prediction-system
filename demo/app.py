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
    page_title="Outage Prediction System",
    page_icon="⚡",
    layout="wide",
)

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
    return results, cross_state, importance, dataset, weather, outages, shap_data

@st.cache_data
def load_ablation():
    return pd.read_csv(ROOT / "models" / "ablation_results.csv")

xgb_model, lgb_model, scaler, calibrator = load_models()
results, cross_state, importance_df, dataset, weather_df, outage_df, shap_data = load_data()
ablation_df = load_ablation()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚡ Outage Prediction")
st.sidebar.markdown("**IEEE SPICES 2026** | Paper ID: 822")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Live Prediction",
    "Model Performance",
    "Feature Analysis",
    "Cross-State Generalization",
    "Data Explorer",
])
st.sidebar.divider()
st.sidebar.markdown("**Authors:** Kommuri, Mahadev, Chase")
st.sidebar.markdown("**Guide:** Dr. Mohammed Adam Baba")
st.sidebar.markdown("Malla Reddy University, Hyderabad")

# ── Page: Overview ───────────────────────────────────────────
if page == "Overview":
    st.title("Compound Weather Event Interaction Modeling for State-Agnostic Power Outage Prediction")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", "0.967", "95% CI: 0.944-0.984")
    c2.metric("F1 Score", "0.947", "Threshold: 0.39")
    c3.metric("ECE (calibrated)", "0.004", "-98.4% from raw")
    c4.metric("Features", "138", "4 groups")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Research Contributions")
        st.markdown("""
        1. **Compound Event Features** — Co-occurrence matrices + pairwise severity interactions + sequential escalation across 6 weather categories
        2. **Calibrated Uncertainty** — Ensemble disagreement + isotonic calibration (ECE: 0.267 → 0.004)
        3. **State-Agnostic Design** — YAML-driven config, TX model generalizes to CA/FL with <0.7% AUC gap
        """)

    with col2:
        st.subheader("System Architecture")
        st.code("""
NOAA Storm Events → Data Pipeline → Feature Store
                                         |
              ┌──────────────────────────┤
              v                          v
        [Temporal 48d]           [Compound 71d]
              |                          |
              └──────────┬───────────────┘
                         v
              [XGBoost + LightGBM Ensemble]
                         |
              [Isotonic Calibration]
                         |
              [FastAPI + Dashboard]
        """, language=None)

    st.markdown("---")
    st.subheader("Dataset Summary")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Training Samples", f"{results['dataset_info']['total_samples']:,}")
    dc2.metric("NOAA Events (TX)", f"{len(weather_df):,}")
    dc3.metric("Outage Records", f"{len(outage_df):,}")
    dc4.metric("Positive Rate", f"{results['dataset_info']['positive_rate']:.1%}")

# ── Page: Live Prediction ────────────────────────────────────
elif page == "Live Prediction":
    st.title("Live Outage Risk Prediction")
    st.markdown("Adjust weather conditions to see real-time model predictions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Weather Conditions")
        weather_count = st.slider("Active weather events (3h window)", 0, 20, 3)
        max_magnitude = st.slider("Max event magnitude (knots/inches)", 0.0, 150.0, 45.0)
        trend_3h = st.slider("Outage trend (3h)", -0.5, 0.5, 0.1, 0.01)
        trend_6h = st.slider("Outage trend (6h)", -0.5, 0.5, 0.05, 0.01)
        compound_events = st.slider("Compound event count", 0, 6, 1)
        grid_load = st.slider("Grid load (MW)", 25000, 85000, 55000, 1000)
        reserve_margin = st.slider("Reserve margin (%)", 0.0, 30.0, 12.0, 0.5)

        st.subheader("Time Context")
        hour = st.slider("Hour of day", 0, 23, 14)
        month = st.slider("Month", 1, 12, 6)
        is_weekend = st.checkbox("Weekend")

    # Build feature vector
    feature_dict = {}
    # Fill all 138 features with defaults
    for col in dataset.columns:
        if col not in ("h3_cell", "timestamp", "target_outage", "target_max_outage_fraction"):
            feature_dict[col] = 0.0

    # Override with user inputs
    feature_dict["weather_count_3h"] = float(weather_count)
    feature_dict["weather_max_mag_3h"] = max_magnitude
    feature_dict["trend_outage_3h"] = trend_3h
    feature_dict["trend_outage_6h"] = trend_6h
    feature_dict["compound_event_count"] = float(compound_events)
    feature_dict["compound_severity_index"] = min(1.0, compound_events / 6.0)
    feature_dict["has_compound_event"] = float(compound_events >= 2)
    feature_dict["current_load_mw"] = float(grid_load)
    feature_dict["reserve_margin_pct"] = reserve_margin
    feature_dict["load_capacity_ratio"] = grid_load / 85000.0
    feature_dict["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feature_dict["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feature_dict["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    feature_dict["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    feature_dict["is_weekend"] = float(is_weekend)

    # Get feature columns in correct order
    from src.ml.dataset import OutageDataset
    temp_ds = OutageDataset(dataset)
    feature_cols = temp_ds.feature_cols
    X = np.array([[feature_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    # Predict
    xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
    lgb_prob = lgb_model.predict_proba(X_scaled)[0, 1]
    ensemble_prob = (xgb_prob + lgb_prob) / 2.0
    calibrated_prob = float(calibrator.predict([ensemble_prob])[0])
    calibrated_prob = np.clip(calibrated_prob, 0, 1)
    disagreement = abs(xgb_prob - lgb_prob) / 2.0

    # Risk level
    if calibrated_prob < 0.25:
        risk_level, risk_color = "LOW", "green"
    elif calibrated_prob < 0.55:
        risk_level, risk_color = "MODERATE", "orange"
    elif calibrated_prob < 0.80:
        risk_level, risk_color = "HIGH", "red"
    else:
        risk_level, risk_color = "CRITICAL", "darkred"

    with col2:
        st.subheader("Prediction Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("Outage Risk", f"{calibrated_prob:.1%}")
        r2.metric("Risk Level", risk_level)
        r3.metric("Model Uncertainty", f"{disagreement:.3f}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=calibrated_prob * 100,
            title={"text": "Outage Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 25], "color": "#d4edda"},
                    {"range": [25, 55], "color": "#fff3cd"},
                    {"range": [55, 80], "color": "#f8d7da"},
                    {"range": [80, 100], "color": "#721c24"},
                ],
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Model breakdown
        st.markdown("**Model Breakdown:**")
        mb1, mb2, mb3 = st.columns(3)
        mb1.metric("XGBoost", f"{xgb_prob:.3f}")
        mb2.metric("LightGBM", f"{lgb_prob:.3f}")
        mb3.metric("Ensemble (calibrated)", f"{calibrated_prob:.3f}")

# ── Page: Model Performance ──────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")

    tab1, tab2, tab3 = st.tabs(["Metrics", "Calibration", "Ablation Study"])

    with tab1:
        st.subheader("Test Set Results (n=1,800)")
        metrics_data = []
        for model_name, metrics in results["metrics"].items():
            row = {"Model": model_name}
            row.update({k: round(v, 4) for k, v in metrics.items()})
            metrics_data.append(row)
        st.dataframe(pd.DataFrame(metrics_data).set_index("Model"), use_container_width=True)

        if "bootstrap_ci" in results:
            st.subheader("Bootstrap 95% Confidence Intervals (1,000 resamples)")
            ci_data = []
            for metric, ci in results["bootstrap_ci"].items():
                ci_data.append({
                    "Metric": metric,
                    "Mean": round(ci["mean"], 4),
                    "Lower": round(ci["lower"], 4),
                    "Upper": round(ci["upper"], 4),
                })
            st.dataframe(pd.DataFrame(ci_data).set_index("Metric"), use_container_width=True)

    with tab2:
        st.subheader("Calibration Analysis")
        c1, c2 = st.columns(2)
        c1.metric("ECE Before Calibration", f"{results['ece_before_calibration']:.4f}")
        c2.metric("ECE After Calibration", f"{results['ece_after_calibration']:.4f}")

        st.image(str(ROOT / "paper" / "figures" / "fig1_reliability_diagram.png"),
                 caption="Reliability diagram before and after isotonic calibration")

    with tab3:
        st.subheader("Feature Group Ablation")
        abl_auc = ablation_df[ablation_df["metric"] == "auc_roc"].copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=abl_auc["group_removed"],
            y=abl_auc["full_value"],
            name="Full Model",
            marker_color="#2196F3",
        ))
        fig.add_trace(go.Bar(
            x=abl_auc["group_removed"],
            y=abl_auc["ablated_value"],
            name="After Removal",
            marker_color="#FF5722",
        ))
        fig.update_layout(
            title="AUC-ROC: Full Model vs Feature Group Removed",
            barmode="group", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(abl_auc[["group_removed", "full_value", "ablated_value", "delta"]].round(4),
                     use_container_width=True)

# ── Page: Feature Analysis ───────────────────────────────────
elif page == "Feature Analysis":
    st.title("Feature Analysis")

    tab1, tab2 = st.tabs(["Feature Importance", "SHAP Analysis"])

    with tab1:
        top_n = st.slider("Top N features", 10, 50, 20)
        top_features = importance_df.head(top_n)

        fig = px.bar(
            top_features, x="avg_importance", y="feature",
            orientation="h", title=f"Top {top_n} Features by Average Importance",
            color="avg_importance", color_continuous_scale="Blues",
        )
        fig.update_layout(height=max(400, top_n * 25), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("SHAP Explainability")

        shap_df = pd.DataFrame(shap_data["top_features"])
        fig = px.bar(
            shap_df.head(20), x="mean_abs_shap", y="feature",
            orientation="h", title="Top 20 Features by Mean |SHAP| Value",
            color="mean_abs_shap", color_continuous_scale="Reds",
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(str(ROOT / "paper" / "figures" / "fig_shap_beeswarm.png"),
                     caption="SHAP Beeswarm Plot")
        with col2:
            st.image(str(ROOT / "paper" / "figures" / "fig_shap_dependence_weather.png"),
                     caption="SHAP Dependence: weather_count_3h vs compound_severity_index")

# ── Page: Cross-State Generalization ─────────────────────────
elif page == "Cross-State Generalization":
    st.title("Cross-State Generalization")
    st.markdown("A model trained on **Texas** data evaluated on California and Florida.")

    results_list = cross_state["all_results"]
    cs_df = pd.DataFrame(results_list)
    cs_df = cs_df[cs_df["model"].str.contains("XGBoost")]

    fig = px.bar(
        cs_df, x="dataset", y="auc_roc", color="model",
        barmode="group", title="AUC-ROC: TX-Trained vs Locally-Trained Models",
        color_discrete_sequence=["#2196F3", "#FF9800", "#4CAF50"],
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fair Comparison (Test Split)")
    if "summary" in cross_state:
        for state, data in cross_state["summary"].items():
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{state}**")
            xgb_data = data.get("XGBoost", {})
            c2.metric("TX Model AUC", f"{xgb_data.get('tx_model_auc', 0):.4f}")
            c3.metric("Local Model AUC", f"{xgb_data.get('local_model_auc', 0):.4f}")

# ── Page: Data Explorer ──────────────────────────────────────
elif page == "Data Explorer":
    st.title("Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Weather Events", "Outage Observations", "Training Dataset"])

    with tab1:
        st.subheader(f"NOAA Storm Events — Texas 2022 ({len(weather_df):,} events)")

        event_counts = weather_df["event_type"].value_counts()
        fig = px.bar(x=event_counts.index, y=event_counts.values,
                     title="Events by Type", labels={"x": "Event Type", "y": "Count"},
                     color=event_counts.values, color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(weather_df.head(100), use_container_width=True)

    with tab2:
        st.subheader(f"Outage Observations ({len(outage_df):,} records)")

        fig = px.histogram(outage_df, x="outage_fraction", nbins=50,
                           title="Distribution of Outage Severity",
                           labels={"outage_fraction": "Outage Fraction"})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"Training Dataset ({len(dataset):,} samples, {len(dataset.columns)} columns)")

        target_dist = dataset["target_outage"].value_counts()
        fig = px.pie(values=target_dist.values, names=["No Outage", "Outage"],
                     title="Target Distribution", color_discrete_sequence=["#4CAF50", "#F44336"])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(dataset.head(50), use_container_width=True)
