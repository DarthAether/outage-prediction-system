"""
Train Outage Prediction Models.

Trains XGBoost + LightGBM on the materialized feature dataset,
runs ablation studies, computes evaluation metrics with bootstrap CIs,
calibrates uncertainty, and saves everything for the research paper.

Usage:
    python scripts/train.py
    python scripts/train.py --dataset data/processed/training_dataset.parquet
    python scripts/train.py --ablation --bootstrap
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from src.ml.dataset import OutageDataset
from src.ml.evaluation import ModelEvaluator
from src.ml.uncertainty import UncertaintyEstimator


def train_xgboost(X_train, y_train, X_val, y_val, class_weights):
    """Train XGBoost with early stopping."""
    import xgboost as xgb

    scale_pos = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "scale_pos_weight": scale_pos,
        "tree_method": "hist",
        "random_state": 42,
        "n_estimators": 500,
        "early_stopping_rounds": 30,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"    XGBoost best iteration: {model.best_iteration}")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, class_weights):
    """Train LightGBM with early stopping."""
    import lightgbm as lgb

    scale_pos = class_weights.get(1, 1.0) / class_weights.get(0, 1.0)

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
    print(f"    LightGBM best iteration: {model.best_iteration_}")
    return model


def run_ablation(
    evaluator: ModelEvaluator,
    dataset: OutageDataset,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_groups: dict,
) -> pd.DataFrame:
    """Run ablation study by removing each feature group."""
    from src.features.feature_store import FeatureStore

    print("\n[ABLATION] Running ablation study...")
    groups = FeatureStore.get_feature_groups()

    # Build dynamic compound features list
    compound_cols = [c for c in train_df.columns if c.startswith(("cooccur_", "interact_", "cat_", "seq_", "compound_"))]
    groups["compound"] = compound_cols

    # Filter groups to only include columns that exist
    filtered_groups = {}
    for name, cols in groups.items():
        existing = [c for c in cols if c in train_df.columns]
        if existing:
            filtered_groups[name] = existing
            print(f"    Group '{name}': {len(existing)} features")

    def train_fn(X, y):
        import xgboost as xgb
        m = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=200,
            tree_method="hist", random_state=42, verbosity=0,
        )
        m.fit(X, y)
        return m

    X_train_feat = dataset.get_feature_dataframe(train_df)
    X_test_feat = dataset.get_feature_dataframe(test_df)

    ablation_results = evaluator.ablation_study(
        train_fn, X_train_feat, y_train, X_test_feat, y_test, filtered_groups
    )

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="Train outage prediction models")
    parser.add_argument("--dataset", type=str, default="data/processed/training_dataset.parquet")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CIs")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OUTAGE PREDICTION - Model Training")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 1. Load dataset
    # -------------------------------------------------------------------
    print(f"\n[1] Loading dataset from {dataset_path}...")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found. Run `python scripts/build_dataset.py` first.")
        sys.exit(1)

    df = pd.read_parquet(dataset_path)
    print(f"    Loaded {len(df):,} samples, {len(df.columns)} columns")
    print(f"    Positive rate: {df['target_outage'].mean():.4f}")

    # -------------------------------------------------------------------
    # 2. Temporal split
    # -------------------------------------------------------------------
    print("\n[2] Creating temporal train/val/test split...")
    dataset = OutageDataset(df, target_col="target_outage", timestamp_col="timestamp")
    train_df, val_df, test_df = dataset.temporal_split(train_frac=0.7, val_frac=0.15)

    X_train, y_train = dataset.get_X_y(train_df, fit_scaler=True)
    X_val, y_val = dataset.get_X_y(val_df)
    X_test, y_test = dataset.get_X_y(test_df)

    print(f"    Train: {len(train_df):,} (pos rate {y_train.mean():.4f})")
    print(f"    Val:   {len(val_df):,} (pos rate {y_val.mean():.4f})")
    print(f"    Test:  {len(test_df):,} (pos rate {y_test.mean():.4f})")
    print(f"    Features: {dataset.n_features}")

    class_weights = dataset.get_class_weights(y_train)
    print(f"    Class weights: {class_weights}")

    # -------------------------------------------------------------------
    # 3. Train models
    # -------------------------------------------------------------------
    print("\n[3] Training XGBoost...")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, class_weights)
    print(f"    XGBoost trained in {time.time() - t0:.1f}s")

    print("\n[4] Training LightGBM...")
    t0 = time.time()
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, class_weights)
    print(f"    LightGBM trained in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------
    # 4. Evaluate individual models
    # -------------------------------------------------------------------
    print("\n[5] Evaluating models...")
    evaluator = ModelEvaluator()

    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]

    # Ensemble: simple average
    ensemble_preds = (xgb_preds + lgb_preds) / 2.0

    # Find optimal threshold on validation set
    from sklearn.metrics import f1_score as sk_f1
    val_ens_preds_thresh = (
        xgb_model.predict_proba(X_val)[:, 1] + lgb_model.predict_proba(X_val)[:, 1]
    ) / 2.0
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1_t = sk_f1(y_val, (val_ens_preds_thresh >= t).astype(int))
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t
    print(f"    Optimal threshold (from val): {best_thresh:.2f} (val F1={best_f1:.4f})")

    results = {}
    for name, preds in [("XGBoost", xgb_preds), ("LightGBM", lgb_preds), ("Ensemble", ensemble_preds)]:
        metrics = evaluator.evaluate(y_test, preds, threshold=best_thresh)
        results[name] = metrics
        print(f"\n    {name}:")
        for k, v in metrics.items():
            print(f"      {k:15s}: {v:.4f}")

    # -------------------------------------------------------------------
    # 5. McNemar's test: ensemble vs individual
    # -------------------------------------------------------------------
    print("\n[6] Statistical significance (McNemar's test)...")
    for name, preds in [("XGBoost", xgb_preds), ("LightGBM", lgb_preds)]:
        test_result = evaluator.mcnemar_test(y_test, ensemble_preds, preds)
        sig = "YES" if test_result["significant"] else "no"
        print(f"    Ensemble vs {name}: chi2={test_result['chi2']:.3f}, p={test_result['p_value']:.4f} (significant: {sig})")

    # -------------------------------------------------------------------
    # 6. Uncertainty estimation + calibration
    # -------------------------------------------------------------------
    print("\n[7] Uncertainty estimation and calibration...")
    uncertainty = UncertaintyEstimator(confidence_level=0.90)

    # Fit calibration on validation set
    val_ensemble_preds = (
        xgb_model.predict_proba(X_val)[:, 1] + lgb_model.predict_proba(X_val)[:, 1]
    ) / 2.0
    uncertainty.calibrate(y_val, val_ensemble_preds)

    # Compute ECE before and after calibration
    ece_before = UncertaintyEstimator.expected_calibration_error(y_test, ensemble_preds)
    calibrated_preds = uncertainty._calibrator.predict(ensemble_preds)
    ece_after = UncertaintyEstimator.expected_calibration_error(y_test, calibrated_preds)
    print(f"    ECE before calibration: {ece_before:.4f}")
    print(f"    ECE after calibration:  {ece_after:.4f}")

    # Ensemble uncertainty from member disagreement
    ens_preds_matrix = np.column_stack([xgb_preds, lgb_preds])
    uncertainty_results = uncertainty.predict_with_uncertainty(ens_preds_matrix)

    avg_epistemic = np.mean([u.epistemic for u in uncertainty_results])
    avg_std = np.mean([u.std for u in uncertainty_results])
    print(f"    Average epistemic uncertainty: {avg_epistemic:.4f}")
    print(f"    Average total uncertainty: {avg_std:.4f}")

    # Reliability diagram data
    reliability = UncertaintyEstimator.reliability_diagram_data(y_test, calibrated_preds)

    # -------------------------------------------------------------------
    # 7. Bootstrap confidence intervals
    # -------------------------------------------------------------------
    if args.bootstrap:
        print("\n[8] Bootstrap confidence intervals (1000 resamples)...")
        t0 = time.time()
        ci_results = evaluator.bootstrap_confidence_intervals(
            y_test, ensemble_preds, n_bootstrap=1000
        )
        print(f"    Completed in {time.time() - t0:.1f}s")
        for metric, ci in ci_results.items():
            print(f"    {metric:15s}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    else:
        ci_results = None

    # -------------------------------------------------------------------
    # 8. Ablation study
    # -------------------------------------------------------------------
    ablation_df = None
    if args.ablation:
        ablation_df = run_ablation(
            evaluator, dataset, train_df, test_df, y_train, y_test, {}
        )
        print("\n    Ablation results (AUC-ROC delta):")
        auc_rows = ablation_df[ablation_df["metric"] == "auc_roc"]
        for _, row in auc_rows.iterrows():
            print(f"      Remove '{row['group_removed']}': AUC {row['full_value']:.4f} -> {row['ablated_value']:.4f} (delta {row['delta']:+.4f})")

    # -------------------------------------------------------------------
    # 9. Feature importance
    # -------------------------------------------------------------------
    print("\n[9] Feature importance (top 20)...")
    xgb_importance = xgb_model.feature_importances_
    feature_names = dataset.feature_cols
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(xgb_importance)],
        "xgb_importance": xgb_importance,
        "lgb_importance": lgb_model.feature_importances_[:len(xgb_importance)],
    })
    importance_df["avg_importance"] = (importance_df["xgb_importance"] + importance_df["lgb_importance"]) / 2
    importance_df = importance_df.sort_values("avg_importance", ascending=False)
    for _, row in importance_df.head(20).iterrows():
        print(f"    {row['feature']:40s}: {row['avg_importance']:.4f}")

    # -------------------------------------------------------------------
    # 10. Save everything
    # -------------------------------------------------------------------
    print(f"\n[10] Saving models and results to {output_dir}...")

    # Models
    pickle.dump(xgb_model, open(output_dir / "xgboost_model.pkl", "wb"))
    pickle.dump(lgb_model, open(output_dir / "lightgbm_model.pkl", "wb"))
    pickle.dump(dataset.scaler, open(output_dir / "scaler.pkl", "wb"))
    pickle.dump(uncertainty._calibrator, open(output_dir / "calibrator.pkl", "wb"))

    # Results
    results_payload = {
        "metrics": results,
        "ece_before_calibration": ece_before,
        "ece_after_calibration": ece_after,
        "avg_epistemic_uncertainty": avg_epistemic,
        "avg_total_uncertainty": avg_std,
        "reliability_diagram": reliability,
        "feature_importance": importance_df.head(30).to_dict(orient="records"),
        "dataset_info": {
            "total_samples": len(df),
            "n_features": dataset.n_features,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "positive_rate": float(df["target_outage"].mean()),
        },
    }

    if ci_results:
        results_payload["bootstrap_ci"] = ci_results

    if ablation_df is not None:
        results_payload["ablation"] = ablation_df.to_dict(orient="records")
        ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_payload, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Ensemble AUC-ROC: {results['Ensemble']['auc_roc']:.4f}")
    print(f"  Ensemble F1:      {results['Ensemble']['f1']:.4f}")
    print(f"  ECE (calibrated): {ece_after:.4f}")
    print(f"  Models saved to:  {output_dir}")


if __name__ == "__main__":
    main()
