"""
Case Study Analysis: Compound Storm Event for Research Paper.

Identifies the most significant compound weather event cluster from the
training data, analyzes model predictions for that period, and generates
a timeline figure for the paper.

Two-pronged approach:
  A) Find the densest 48h storm window (most events, cells, damage)
  B) Anchor on training samples where compound features actually fired

Usage:
    PYTHONPATH=backend python scripts/case_study.py
"""

import json
import pickle
import sys
from datetime import timedelta
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.features.compound_events import classify_event, EVENT_CATEGORIES
from src.ml.dataset import OutageDataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ser(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# 1. Load all data
# ---------------------------------------------------------------------------

def load_data():
    proc = ROOT / "data" / "processed"
    weather = pd.read_parquet(proc / "weather_events.parquet")
    outages = pd.read_parquet(proc / "outage_observations.parquet")
    training = pd.read_parquet(proc / "training_dataset.parquet")

    weather["event_time"] = pd.to_datetime(weather["event_time"])
    outages["observed_at"] = pd.to_datetime(outages["observed_at"])
    training["timestamp"] = pd.to_datetime(training["timestamp"])

    return weather, outages, training


# ---------------------------------------------------------------------------
# 2. Find the most significant compound event cluster
# ---------------------------------------------------------------------------

def find_compound_event_cluster(weather: pd.DataFrame, outages: pd.DataFrame):
    """Slide a 48h window and pick the one with the highest compound score."""
    ws = weather.sort_values("event_time").copy()
    ws["category"] = ws["event_type"].apply(classify_event)
    t_min, t_max = ws["event_time"].min(), ws["event_time"].max()

    best_score, best = -1, None
    cur = t_min
    while cur + timedelta(hours=48) <= t_max:
        we = cur + timedelta(hours=48)
        m = (ws["event_time"] >= cur) & (ws["event_time"] < we)
        ev = ws.loc[m]
        if len(ev) < 3:
            cur += timedelta(hours=6)
            continue

        cats = set(ev["category"].dropna().unique())
        om = (outages["observed_at"] >= cur) & (outages["observed_at"] < we + timedelta(hours=24))
        wo = outages.loc[om]
        mean_of = wo["outage_fraction"].mean() if len(wo) else 0
        max_of = wo["outage_fraction"].max() if len(wo) else 0

        score = (
            len(cats) * 10
            + ev["h3_index_res7"].nunique() * 2
            + len(ev)
            + np.log1p(ev["damage_property"].sum())
            + mean_of * 50
            + max_of * 20
        )

        if score > best_score:
            best_score = score
            best = {
                "start": cur, "end": we,
                "n_categories": len(cats),
                "categories": sorted(cats),
                "n_cells": int(ev["h3_index_res7"].nunique()),
                "n_events": len(ev),
                "total_damage": float(ev["damage_property"].sum()),
                "max_magnitude": float(ev["magnitude"].max()),
                "mean_outage_fraction": float(mean_of),
                "max_outage_fraction": float(max_of),
                "n_outage_cells": int(wo["h3_index_res7"].nunique()) if len(wo) else 0,
                "n_outage_records": len(wo),
                "score": float(score),
            }
        cur += timedelta(hours=6)

    return best


# ---------------------------------------------------------------------------
# 3. Find training samples where compound features fired
# ---------------------------------------------------------------------------

def find_compound_anchor_samples(training):
    """Find training rows where has_compound_event == 1."""
    comp = training[training["has_compound_event"] > 0].copy()
    return comp


# ---------------------------------------------------------------------------
# 4. Expand around anchors for broader context
# ---------------------------------------------------------------------------

def get_context_samples(training, anchor_timestamps, hours_before=48, hours_after=48):
    """Get all training samples in a time neighbourhood of the anchors."""
    masks = []
    for ts in anchor_timestamps:
        m = (training["timestamp"] >= ts - timedelta(hours=hours_before)) & \
            (training["timestamp"] <= ts + timedelta(hours=hours_after))
        masks.append(m)
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return training.loc[combined].copy()


# ---------------------------------------------------------------------------
# 5. Model predictions
# ---------------------------------------------------------------------------

def model_predictions(samples, feature_cols, scaler):
    """Run XGBoost on given samples, return probas and analysis."""
    with open(ROOT / "models" / "xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    available = [c for c in feature_cols if c in samples.columns]
    X = samples[available].fillna(0).values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    actuals = samples["target_outage"].values

    # Feature contribution: importance * mean |value|
    imp = model.feature_importances_
    raw = samples[available].fillna(0).values.astype(np.float32)
    mean_abs = np.abs(raw).mean(axis=0)[:len(imp)]
    contrib = imp * mean_abs
    top_idx = np.argsort(contrib)[::-1][:15]
    top_features = [
        {"feature": available[i], "importance": float(imp[i]),
         "mean_value": float(raw[:, i].mean()),
         "contribution_score": float(contrib[i])}
        for i in top_idx
    ]

    summary = {
        "n_samples": len(samples),
        "mean_predicted_probability": float(proba.mean()),
        "max_predicted_probability": float(proba.max()),
        "min_predicted_probability": float(proba.min()),
        "predicted_positive_rate": float(preds.mean()),
        "actual_positive_rate": float(actuals.mean()),
        "n_true_positives": int(((preds == 1) & (actuals == 1)).sum()),
        "n_false_negatives": int(((preds == 0) & (actuals == 1)).sum()),
        "n_false_positives": int(((preds == 1) & (actuals == 0)).sum()),
        "n_true_negatives": int(((preds == 0) & (actuals == 0)).sum()),
        "high_risk_predictions": int((proba >= 0.7).sum()),
        "top_contributing_features": top_features,
    }
    return proba, preds, summary


# ---------------------------------------------------------------------------
# 6. Compound features analysis
# ---------------------------------------------------------------------------

def analyze_compound_features(samples):
    cols = [c for c in samples.columns
            if c.startswith(("cooccur_", "compound_", "has_compound", "seq_", "interact_", "cat_"))]
    out = {}
    for c in cols:
        v = samples[c]
        if v.max() > 0:
            out[c] = {"mean": float(v.mean()), "max": float(v.max()),
                       "nonzero_pct": float((v > 0).mean())}
    return out


# ---------------------------------------------------------------------------
# 7. Timeline figure
# ---------------------------------------------------------------------------

def create_timeline_figure(weather, outages, context_samples, proba,
                           window, anchor_times, output_path):
    start, end = window["start"], window["end"]
    aftermath_end = end + timedelta(hours=24)

    # Full plot range: union of storm window and anchor windows
    all_times = [start - timedelta(hours=12), aftermath_end + timedelta(hours=12)]
    for at in anchor_times:
        all_times += [at - timedelta(hours=48), at + timedelta(hours=48)]
    plot_start = min(all_times)
    plot_end = max(all_times)

    # Filter data to plot range
    wm = (weather["event_time"] >= plot_start) & (weather["event_time"] <= plot_end)
    we_plot = weather.loc[wm].copy()
    we_plot["category"] = we_plot["event_type"].apply(classify_event)

    om = (outages["observed_at"] >= plot_start) & (outages["observed_at"] <= plot_end)
    oo_plot = outages.loc[om].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1.1]})
    fig.suptitle("Case Study: Compound Storm Event Analysis\n"
                 f"({start.strftime('%b %d')} \u2013 {end.strftime('%b %d, %Y')}  |  "
                 f"Categories: {', '.join(window['categories'])})",
                 fontsize=13, fontweight="bold", y=0.99)

    # ---- Panel A: Weather events ----
    ax1 = axes[0]
    cat_colors = {"wind": "#E74C3C", "ice": "#3498DB", "heat": "#F39C12",
                  "flood": "#27AE60", "drought": "#8B4513", "fire": "#FF6347",
                  None: "#95A5A6"}
    seen = set()
    for _, r in we_plot.iterrows():
        cat = r["category"]
        col = cat_colors.get(cat, "#95A5A6")
        lbl = cat if cat and cat not in seen else None
        if cat:
            seen.add(cat)
        mag = max(float(r["magnitude"]) if pd.notna(r["magnitude"]) else 0, 3)
        ax1.scatter(r["event_time"], mag, c=col, s=35 + mag * 2, alpha=0.7,
                    edgecolors="black", linewidths=0.4, label=lbl, zorder=5)

    ax1.axvspan(start, end, alpha=0.15, color="red", label="Compound event window (48 h)")
    ax1.axvspan(end, aftermath_end, alpha=0.08, color="orange", label="Aftermath (+24 h)")
    for at in anchor_times:
        ax1.axvline(at, color="purple", linewidth=1.2, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Event Magnitude", fontsize=10)
    ax1.set_title("(a) Weather Events by Category", fontsize=11, fontweight="bold", loc="left")
    ax1.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: Outage fraction ----
    ax2 = axes[1]
    if len(oo_plot) > 0:
        oo_plot["hour_bin"] = oo_plot["observed_at"].dt.floor("3h")
        hourly = oo_plot.groupby("hour_bin").agg(
            mean_frac=("outage_fraction", "mean"),
            max_frac=("outage_fraction", "max"),
            count=("outage_fraction", "size"),
        ).reset_index()
        ax2.fill_between(hourly["hour_bin"], 0, hourly["max_frac"],
                         alpha=0.2, color="#E74C3C", label="Max outage fraction")
        ax2.plot(hourly["hour_bin"], hourly["mean_frac"],
                 color="#E74C3C", linewidth=2, marker="o", markersize=3,
                 label="Mean outage fraction")
    ax2.axvspan(start, end, alpha=0.15, color="red")
    ax2.axvspan(end, aftermath_end, alpha=0.08, color="orange")
    for at in anchor_times:
        ax2.axvline(at, color="purple", linewidth=1.2, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Outage Fraction", fontsize=10)
    ax2.set_title("(b) Observed Outage Severity", fontsize=11, fontweight="bold", loc="left")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # ---- Panel C: Model predictions ----
    ax3 = axes[2]
    if proba is not None and len(context_samples) > 0:
        cs = context_samples.copy()
        cs["proba"] = proba
        cs = cs.sort_values("timestamp")

        ax3.plot(cs["timestamp"], cs["proba"], color="#2980B9", linewidth=1.5,
                 marker="s", markersize=3, alpha=0.8, label="Predicted P(outage)", zorder=4)

        pos = cs[cs["target_outage"] == 1]
        neg = cs[cs["target_outage"] == 0]
        ax3.scatter(pos["timestamp"], pos["proba"], c="#E74C3C", s=70,
                    marker="^", zorder=6, edgecolors="black", linewidths=0.7,
                    label="Actual outage")
        ax3.scatter(neg["timestamp"], neg["proba"], c="#27AE60", s=25,
                    marker="v", zorder=3, alpha=0.4, edgecolors="black",
                    linewidths=0.3, label="No outage")

        # Highlight compound anchor samples
        anchor_mask = cs["has_compound_event"] > 0
        if anchor_mask.any():
            ax3.scatter(cs.loc[anchor_mask, "timestamp"],
                        cs.loc[anchor_mask, "proba"],
                        c="purple", s=120, marker="D", zorder=7,
                        edgecolors="black", linewidths=1.0,
                        label="Compound event sample")

        ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.5)")

    ax3.axvspan(start, end, alpha=0.15, color="red")
    ax3.axvspan(end, aftermath_end, alpha=0.08, color="orange")
    for at in anchor_times:
        ax3.axvline(at, color="purple", linewidth=1.2, linestyle="--", alpha=0.6,
                     label="Compound sample time" if at == anchor_times[0] else None)
    ax3.set_ylabel("P(Outage)", fontsize=10)
    ax3.set_xlabel("Date / Time", fontsize=10)
    ax3.set_title("(c) XGBoost Predictions vs. Actual Outages", fontsize=11,
                  fontweight="bold", loc="left")
    ax3.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved to {output_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CASE STUDY ANALYSIS: Compound Storm Event")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading data...")
    weather, outages, training = load_data()
    print(f"    Weather events: {len(weather):,}")
    print(f"    Outage observations: {len(outages):,}")
    print(f"    Training samples: {len(training):,}")

    # 2. Best 48h storm window
    print("\n[2] Searching for densest 48h storm window...")
    window = find_compound_event_cluster(weather, outages)
    print(f"    Window: {window['start']} \u2192 {window['end']}")
    print(f"    Categories: {window['categories']} ({window['n_categories']})")
    print(f"    H3 cells: {window['n_cells']}")
    print(f"    Events: {window['n_events']}")
    print(f"    Mean outage fraction: {window['mean_outage_fraction']:.4f}")
    print(f"    Max outage fraction: {window['max_outage_fraction']:.4f}")
    print(f"    Total damage: ${window['total_damage']:,.0f}")

    # 3. Compound-feature anchor samples
    print("\n[3] Finding training samples with compound features...")
    anchors = find_compound_anchor_samples(training)
    anchor_times = sorted(anchors["timestamp"].tolist())
    print(f"    Found {len(anchors)} samples with has_compound_event=1")
    for _, row in anchors.iterrows():
        print(f"      {row['timestamp']}  cell={row['h3_cell']}  "
              f"compound_count={row['compound_event_count']:.0f}  "
              f"cooccur_flood_wind={row['cooccur_flood_wind']:.0f}  "
              f"target_outage={row['target_outage']}")

    # 4. Context: samples around the storm window AND around compound anchors
    print("\n[4] Gathering context samples...")
    # Union of storm window range and anchor ranges
    all_anchor_ts = anchor_times + [window["start"], window["end"]]
    context = get_context_samples(training, all_anchor_ts,
                                  hours_before=72, hours_after=72)
    print(f"    Context samples: {len(context)}")
    print(f"    Time range: {context['timestamp'].min()} \u2192 {context['timestamp'].max()}")
    print(f"    Positive rate: {context['target_outage'].mean():.4f}")

    # 5. Analyse compound features in context
    print("\n[5] Compound features analysis (context samples)...")
    cf = analyze_compound_features(context)
    active_cf = {k: v for k, v in cf.items() if v["max"] > 0}
    print(f"    Active compound features: {len(active_cf)}")
    for feat, stats in sorted(active_cf.items(), key=lambda x: -x[1]["max"])[:15]:
        print(f"      {feat:45s}: max={stats['max']:.4f}  mean={stats['mean']:.4f}  nonzero={stats['nonzero_pct']:.1%}")

    # Also analyse compound features specifically on the anchor samples
    print("\n    Compound features on anchor samples only:")
    cf_anchor = analyze_compound_features(anchors)
    for feat, stats in sorted(cf_anchor.items(), key=lambda x: -x[1]["max"])[:15]:
        print(f"      {feat:45s}: max={stats['max']:.4f}  mean={stats['mean']:.4f}")

    # 6. Model predictions on context
    print("\n[6] Running model predictions on context samples...")
    dataset = OutageDataset(training, target_col="target_outage", timestamp_col="timestamp")
    scaler = None
    sp = ROOT / "models" / "scaler.pkl"
    if sp.exists():
        with open(sp, "rb") as f:
            scaler = pickle.load(f)

    proba, preds, pred_summary = model_predictions(context, dataset.feature_cols, scaler)
    print(f"    Samples: {pred_summary['n_samples']}")
    print(f"    Mean P(outage): {pred_summary['mean_predicted_probability']:.4f}")
    print(f"    Max P(outage): {pred_summary['max_predicted_probability']:.4f}")
    print(f"    Predicted positive rate: {pred_summary['predicted_positive_rate']:.4f}")
    print(f"    Actual positive rate: {pred_summary['actual_positive_rate']:.4f}")
    print(f"    High risk (>=0.7): {pred_summary['high_risk_predictions']}")
    print(f"    TP={pred_summary['n_true_positives']}  FN={pred_summary['n_false_negatives']}  "
          f"FP={pred_summary['n_false_positives']}  TN={pred_summary['n_true_negatives']}")

    # predictions specifically on anchor samples
    if len(anchors) > 0:
        anchor_idx = context.index.isin(anchors.index)
        anchor_proba = proba[anchor_idx]
        print(f"\n    Predictions on compound-event anchor samples:")
        for i, (_, row) in enumerate(anchors.iterrows()):
            p = anchor_proba[i] if i < len(anchor_proba) else float("nan")
            print(f"      {row['timestamp']}  P(outage)={p:.4f}  actual={row['target_outage']}")

    print(f"\n    Top contributing features:")
    for feat in pred_summary["top_contributing_features"][:10]:
        print(f"      {feat['feature']:40s}: imp={feat['importance']:.4f}  "
              f"mean_val={feat['mean_value']:.4f}  contrib={feat['contribution_score']:.4f}")

    # 7. Detailed event breakdown in storm window
    print("\n[7] Event breakdown in storm window...")
    wm = (weather["event_time"] >= window["start"]) & (weather["event_time"] < window["end"])
    we_win = weather.loc[wm].copy()
    we_win["category"] = we_win["event_type"].apply(classify_event)
    et_break = we_win.groupby("event_type").agg(
        count=("event_type", "size"),
        max_magnitude=("magnitude", "max"),
        total_damage=("damage_property", "sum"),
    )
    for et, row in et_break.iterrows():
        print(f"      {et:25s}: count={int(row['count'])}  max_mag={row['max_magnitude']:.1f}  "
              f"damage=${row['total_damage']:,.0f}")

    cat_break = we_win.groupby("category").agg(
        count=("category", "size"), cells=("h3_index_res7", "nunique")
    )
    cooccur_pairs = []
    active_cats = set(we_win["category"].dropna().unique())
    for a, b in combinations(sorted(active_cats), 2):
        cooccur_pairs.append(f"{a}+{b}")
    print(f"    Active categories: {sorted(active_cats)}")
    print(f"    Co-occurrence pairs: {cooccur_pairs}")
    print(f"    Compound event: {'YES' if len(active_cats) >= 2 else 'NO'}")

    # 8. Generate figure
    print("\n[8] Generating timeline figure...")
    fig_path = ROOT / "paper" / "figures" / "fig_case_study_timeline.png"
    create_timeline_figure(weather, outages, context, proba,
                           window, anchor_times, fig_path)

    # 9. Save JSON
    print("\n[9] Saving case_study.json...")
    result = {
        "storm_window": {
            "start": str(window["start"]),
            "end": str(window["end"]),
            "duration_hours": 48,
            "n_events": window["n_events"],
            "n_categories": window["n_categories"],
            "categories": window["categories"],
            "n_h3_cells_affected": window["n_cells"],
            "total_damage_usd": window["total_damage"],
            "max_magnitude": window["max_magnitude"],
            "compound_score": window["score"],
        },
        "outage_impact": {
            "mean_outage_fraction": window["mean_outage_fraction"],
            "max_outage_fraction": window["max_outage_fraction"],
            "n_outage_observations": window["n_outage_records"],
            "n_outage_cells": window["n_outage_cells"],
        },
        "event_type_breakdown": {
            et: {"count": int(r["count"]), "max_magnitude": float(r["max_magnitude"]),
                 "total_damage": float(r["total_damage"])}
            for et, r in et_break.iterrows()
        },
        "category_breakdown": {
            cat: {"count": int(r["count"]), "cells": int(r["cells"])}
            for cat, r in cat_break.iterrows()
        },
        "compound_event_analysis": {
            "has_compound_event": len(active_cats) >= 2,
            "n_active_categories": len(active_cats),
            "cooccurrence_pairs": cooccur_pairs,
            "n_compound_anchor_samples": len(anchors),
            "anchor_timestamps": [str(t) for t in anchor_times],
            "compound_features_fired": {
                k: v for k, v in cf_anchor.items()
            } if cf_anchor else {},
        },
        "model_predictions": pred_summary,
        "compound_features_in_context": active_cf,
        "context_window": {
            "n_samples": len(context),
            "time_start": str(context["timestamp"].min()),
            "time_end": str(context["timestamp"].max()),
            "positive_rate": float(context["target_outage"].mean()),
        },
        "figure_path": str(fig_path),
    }

    json_path = ROOT / "models" / "case_study.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_ser)
    print(f"  Saved to {json_path}")

    print(f"\n{'=' * 70}")
    print("CASE STUDY COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Storm window: {window['start']} \u2192 {window['end']}")
    print(f"  Categories: {', '.join(window['categories'])}")
    print(f"  Compound event: {'YES' if len(active_cats) >= 2 else 'NO'}")
    print(f"  Compound anchor samples: {len(anchors)}")
    if pred_summary:
        print(f"  Mean P(outage) in context: {pred_summary['mean_predicted_probability']:.4f}")
        print(f"  High-risk predictions: {pred_summary['high_risk_predictions']}")
    print(f"  Output JSON: {json_path}")
    print(f"  Figure: {fig_path}")


if __name__ == "__main__":
    main()
