"""
Cross-State Generalization Evaluation.

Evaluates TX-trained models on CA and FL data, then trains state-specific
models and compares cross-state vs in-state performance.

Usage:
    python scripts/cross_state_eval.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.ml.dataset import OutageDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXCLUDE_COLS = ("h3_cell", "timestamp", "target_outage", "target_max_outage_fraction")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a parquet dataset."""
    df = pd.read_parquet(path)
    print(f"  Loaded {path.name}: {len(df):,} rows, positive rate {df['target_outage'].mean():.4f}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get feature columns (excluding metadata and targets)."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]


def prepare_data(df: pd.DataFrame, feature_cols: list[str], scaler=None):
    """Extract X, y from a dataframe using given feature columns and optional scaler."""
    # Ensure all feature cols exist; fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["target_outage"].values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    return X, y


def evaluate_model(model, X, y, model_name: str, dataset_name: str) -> dict:
    """Evaluate a model and return metrics dict."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "auc_roc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
    }
    return metrics


def train_state_model(X_train, y_train, X_val, y_val, model_type: str = "xgboost"):
    """Train a state-specific model."""
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = (n_neg / n_pos) if n_pos > 0 else 1.0

    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "auc"],
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            scale_pos_weight=scale_pos,
            tree_method="hist",
            random_state=42,
            n_estimators=500,
            early_stopping_rounds=30,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            objective="binary",
            metric=["binary_logloss", "auc"],
            max_depth=8,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
    return model


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    header = f"{'Model':30s} {'Dataset':8s} {'AUC-ROC':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'Accuracy':>9s} {'N':>7s}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['model']:30s} {r['dataset']:8s} "
            f"{r['auc_roc']:8.4f} {r['f1']:8.4f} {r['precision']:10.4f} "
            f"{r['recall']:8.4f} {r['accuracy']:9.4f} {r['n_samples']:7d}"
        )
    print(sep)


def print_generalization_summary(results: list[dict]):
    """Print a summary of cross-state generalization gaps."""
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP ANALYSIS")
    print("=" * 70)

    for state in ["CA", "FL"]:
        print(f"\n  --- {state} ---")
        for mtype in ["XGBoost", "LightGBM"]:
            tx_key = f"TX-trained {mtype}"
            local_key = f"{state}-trained {mtype}"
            tx_row = next((r for r in results if r["model"] == tx_key and r["dataset"] == state), None)
            local_row = next((r for r in results if r["model"] == local_key and r["dataset"] == state), None)
            if tx_row and local_row:
                auc_gap = local_row["auc_roc"] - tx_row["auc_roc"]
                f1_gap = local_row["f1"] - tx_row["f1"]
                print(f"  {mtype:10s}: TX AUC={tx_row['auc_roc']:.4f}, {state} AUC={local_row['auc_roc']:.4f}, gap={auc_gap:+.4f}")
                print(f"  {mtype:10s}: TX F1 ={tx_row['f1']:.4f}, {state} F1 ={local_row['f1']:.4f}, gap={f1_gap:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models_dir = ROOT / "models"
    data_dir = ROOT / "data" / "processed"

    print("=" * 70)
    print("CROSS-STATE GENERALIZATION EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load TX models and scaler
    # ------------------------------------------------------------------
    print("\n[1] Loading TX-trained models...")
    with open(models_dir / "xgboost_model.pkl", "rb") as f:
        tx_xgb = pickle.load(f)
    with open(models_dir / "lightgbm_model.pkl", "rb") as f:
        tx_lgb = pickle.load(f)
    with open(models_dir / "scaler.pkl", "rb") as f:
        tx_scaler = pickle.load(f)
    print("  Loaded: xgboost_model.pkl, lightgbm_model.pkl, scaler.pkl")

    # ------------------------------------------------------------------
    # 2. Load TX dataset to get feature column order
    # ------------------------------------------------------------------
    print("\n[2] Loading datasets...")
    tx_df = load_dataset(data_dir / "training_dataset.parquet")
    ca_df = load_dataset(data_dir / "training_dataset_CA.parquet")
    fl_df = load_dataset(data_dir / "training_dataset_FL.parquet")

    # Determine feature columns from TX dataset (same order as training)
    tx_dataset = OutageDataset(tx_df, target_col="target_outage", timestamp_col="timestamp")
    feature_cols = tx_dataset.feature_cols
    print(f"  Feature columns: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 3. Evaluate TX models on all states
    # ------------------------------------------------------------------
    print("\n[3] Evaluating TX-trained models on all states...")
    all_results = []

    for state, state_df in [("TX", tx_df), ("CA", ca_df), ("FL", fl_df)]:
        X, y = prepare_data(state_df.copy(), feature_cols, scaler=tx_scaler)
        for name, model in [("TX-trained XGBoost", tx_xgb), ("TX-trained LightGBM", tx_lgb)]:
            metrics = evaluate_model(model, X, y, name, state)
            all_results.append(metrics)
            print(f"    {name} on {state}: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

    # ------------------------------------------------------------------
    # 4. Train state-specific models for CA and FL
    # ------------------------------------------------------------------
    for state, state_df in [("CA", ca_df), ("FL", fl_df)]:
        print(f"\n[4] Training {state}-specific models...")
        ds = OutageDataset(state_df, target_col="target_outage", timestamp_col="timestamp")
        train_df, val_df, test_df = ds.temporal_split(train_frac=0.7, val_frac=0.15)

        X_train, y_train = ds.get_X_y(train_df, fit_scaler=True)
        X_val, y_val = ds.get_X_y(val_df)
        X_test, y_test = ds.get_X_y(test_df)

        print(f"    Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        print(f"    Train pos rate: {y_train.mean():.4f}, Test pos rate: {y_test.mean():.4f}")

        for mtype in ["xgboost", "lightgbm"]:
            t0 = time.time()
            model = train_state_model(X_train, y_train, X_val, y_val, model_type=mtype)
            elapsed = time.time() - t0
            print(f"    {mtype} trained in {elapsed:.1f}s")

            # Evaluate on test split
            metrics = evaluate_model(model, X_test, y_test, f"{state}-trained {mtype.title().replace('gbm', 'GBM').replace('xgb', 'XGB')}", state)
            # Fix naming
            mname = "XGBoost" if mtype == "xgboost" else "LightGBM"
            metrics["model"] = f"{state}-trained {mname}"
            all_results.append(metrics)
            print(f"    {state}-trained {mname} on {state}: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

        # Also evaluate TX models on this state's TEST split only (for fair comparison)
        # The earlier TX eval was on full dataset; redo on test split for apples-to-apples
        X_test_tx_scaled, y_test_check = prepare_data(test_df.copy(), feature_cols, scaler=tx_scaler)
        for name, model in [("TX-trained XGBoost (test split)", tx_xgb), ("TX-trained LightGBM (test split)", tx_lgb)]:
            metrics = evaluate_model(model, X_test_tx_scaled, y_test_check, name, state)
            all_results.append(metrics)
            print(f"    {name} on {state} test: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

    # ------------------------------------------------------------------
    # 5. Print comparison tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)
    print_comparison_table(all_results)

    # Fair comparison: TX-model test split vs state-model test split
    print("\n" + "=" * 70)
    print("FAIR COMPARISON (test split only)")
    print("=" * 70)
    fair_results = [r for r in all_results if "test split" in r["model"] or "-trained" in r["model"]]
    # Filter to only state-specific test split comparisons
    fair_ca = [r for r in all_results if r["dataset"] == "CA" and ("test split" in r["model"] or "CA-trained" in r["model"])]
    fair_fl = [r for r in all_results if r["dataset"] == "FL" and ("test split" in r["model"] or "FL-trained" in r["model"])]

    if fair_ca:
        print("\n  California:")
        print_comparison_table(fair_ca)
    if fair_fl:
        print("\n  Florida:")
        print_comparison_table(fair_fl)

    print_generalization_summary(all_results)

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output_path = models_dir / "cross_state_results.json"
    payload = {
        "description": "Cross-state generalization evaluation",
        "all_results": all_results,
        "fair_comparison": {
            "CA": fair_ca,
            "FL": fair_fl,
        },
        "summary": {},
    }

    # Compute summary gaps
    for state in ["CA", "FL"]:
        payload["summary"][state] = {}
        for mtype in ["XGBoost", "LightGBM"]:
            tx_test = next(
                (r for r in all_results if r["model"] == f"TX-trained {mtype} (test split)" and r["dataset"] == state),
                None,
            )
            local = next(
                (r for r in all_results if r["model"] == f"{state}-trained {mtype}" and r["dataset"] == state),
                None,
            )
            if tx_test and local:
                payload["summary"][state][mtype] = {
                    "tx_model_auc": tx_test["auc_roc"],
                    "local_model_auc": local["auc_roc"],
                    "auc_gap": round(local["auc_roc"] - tx_test["auc_roc"], 4),
                    "tx_model_f1": tx_test["f1"],
                    "local_model_f1": local["f1"],
                    "f1_gap": round(local["f1"] - tx_test["f1"], 4),
                }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print(f"\n{'=' * 70}")
    print("CROSS-STATE EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
