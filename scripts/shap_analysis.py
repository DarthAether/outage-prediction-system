"""
SHAP Analysis for Outage Prediction Models.

Computes SHAP values on the test set using TreeExplainer,
generates publication-quality figures, and saves feature importance JSON.

Usage:
    PYTHONPATH=backend python scripts/shap_analysis.py
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.ml.dataset import OutageDataset

# ---------- paths ----------
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "training_dataset.parquet"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- matplotlib academic style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ===================================================================
# 1. Load models and data
# ===================================================================
print("=" * 70)
print("SHAP ANALYSIS — Outage Prediction System")
print("=" * 70)

print("\n[1] Loading models...")
with open(MODEL_DIR / "xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open(MODEL_DIR / "lightgbm_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)
with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("    Loaded XGBoost, LightGBM, and scaler.")

print("\n[2] Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"    {len(df):,} samples, {len(df.columns)} columns")

# ===================================================================
# 2. Temporal split (same as training)
# ===================================================================
print("\n[3] Creating temporal split...")
dataset = OutageDataset(df, target_col="target_outage", timestamp_col="timestamp")
train_df, val_df, test_df = dataset.temporal_split(train_frac=0.7, val_frac=0.15)

# For tree models we use unscaled feature DataFrames (TreeExplainer works on raw features)
X_test_df = dataset.get_feature_dataframe(test_df)
y_test = test_df["target_outage"].values

feature_names = list(X_test_df.columns)
print(f"    Test set: {len(X_test_df):,} samples, {len(feature_names)} features")

# ===================================================================
# 3. SHAP TreeExplainer on XGBoost
# ===================================================================
print("\n[4] Computing SHAP values (XGBoost TreeExplainer)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_df)
print(f"    SHAP values shape: {shap_values.shape}")

# Build an Explanation object for shap's plotting API
explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(shap_values.shape[0], explainer.expected_value),
    data=X_test_df.values,
    feature_names=feature_names,
)

# ===================================================================
# 4. Mean |SHAP| ranking (top 30)
# ===================================================================
print("\n[5] Computing mean |SHAP| feature ranking...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_order = np.argsort(mean_abs_shap)[::-1]

top30 = []
for rank, idx in enumerate(importance_order[:30]):
    entry = {
        "rank": rank + 1,
        "feature": feature_names[idx],
        "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
    }
    top30.append(entry)
    if rank < 20:
        print(f"    {rank+1:2d}. {feature_names[idx]:40s}  {mean_abs_shap[idx]:.6f}")

shap_json_path = MODEL_DIR / "shap_values.json"
with open(shap_json_path, "w") as f:
    json.dump(top30, f, indent=2)
print(f"\n    Saved top-30 to {shap_json_path}")

# ===================================================================
# 5. Beeswarm plot (top 20)
# ===================================================================
print("\n[6] Generating beeswarm plot (top 20)...")

fig, ax = plt.subplots(figsize=(7, 5))
shap.plots.beeswarm(explanation, max_display=20, show=False)
# Adjust current figure produced by shap
fig_bee = plt.gcf()
fig_bee.set_size_inches(7, 5)
fig_bee.tight_layout()

for ext in ("png", "pdf"):
    out = FIG_DIR / f"fig_shap_beeswarm.{ext}"
    fig_bee.savefig(out, dpi=300, bbox_inches="tight")
    print(f"    Saved {out}")
plt.close("all")

# ===================================================================
# 6. Bar plot — mean |SHAP|
# ===================================================================
print("\n[7] Generating bar plot (mean |SHAP|, top 20)...")

fig_bar, ax_bar = plt.subplots(figsize=(3.5, 5))
shap.plots.bar(explanation, max_display=20, show=False)
fig_bar2 = plt.gcf()
fig_bar2.set_size_inches(3.5, 5)
fig_bar2.tight_layout()

for ext in ("png", "pdf"):
    out = FIG_DIR / f"fig_shap_bar.{ext}"
    fig_bar2.savefig(out, dpi=300, bbox_inches="tight")
    print(f"    Saved {out}")
plt.close("all")

# ===================================================================
# 7. Dependence plot — weather_count_3h
# ===================================================================
print("\n[8] Generating dependence plot: weather_count_3h...")

weather_col = "weather_count_3h"
interaction_col = "compound_severity_index"

if weather_col in feature_names:
    fig_dep1, ax_dep1 = plt.subplots(figsize=(3.5, 3))
    w_idx = feature_names.index(weather_col)
    if interaction_col in feature_names:
        shap.dependence_plot(
            w_idx, shap_values, X_test_df.values,
            feature_names=feature_names,
            interaction_index=feature_names.index(interaction_col),
            ax=ax_dep1, show=False,
        )
    else:
        shap.dependence_plot(
            w_idx, shap_values, X_test_df.values,
            feature_names=feature_names,
            ax=ax_dep1, show=False,
        )
    ax_dep1.set_title("")
    fig_dep1.tight_layout()
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"fig_shap_dependence_weather.{ext}"
        fig_dep1.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    Saved {out}")
    plt.close("all")
else:
    print(f"    [WARN] '{weather_col}' not in features, skipping.")

# ===================================================================
# 8. Dependence plot — trend_outage_3h
# ===================================================================
print("\n[9] Generating dependence plot: trend_outage_3h...")

trend_col = "trend_outage_3h"
if trend_col in feature_names:
    fig_dep2, ax_dep2 = plt.subplots(figsize=(3.5, 3))
    t_idx = feature_names.index(trend_col)
    shap.dependence_plot(
        t_idx, shap_values, X_test_df.values,
        feature_names=feature_names,
        ax=ax_dep2, show=False,
    )
    ax_dep2.set_title("")
    fig_dep2.tight_layout()
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"fig_shap_dependence_trend.{ext}"
        fig_dep2.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    Saved {out}")
    plt.close("all")
else:
    print(f"    [WARN] '{trend_col}' not in features, skipping.")

# ===================================================================
# Done
# ===================================================================
print(f"\n{'=' * 70}")
print("SHAP ANALYSIS COMPLETE")
print(f"{'=' * 70}")
print(f"  Figures saved to: {FIG_DIR}")
print(f"  SHAP JSON saved to: {shap_json_path}")
