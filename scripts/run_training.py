#!/usr/bin/env python3
"""Training pipeline for outage prediction models.

Usage:
    python scripts/run_training.py --region TX --data-dir data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from src.ml.baselines import LightGBMOutageModel, XGBoostOutageModel
from src.ml.dataset import OutageDataset
from src.ml.ensemble import OutageEnsemble
from src.ml.evaluation import ModelEvaluator
from src.ml.registry import ModelRegistry
from src.ml.uncertainty import UncertaintyEstimator


def main():
    parser = argparse.ArgumentParser(description="Train outage prediction models")
    parser.add_argument("--region", type=str, default="TX", help="Region code")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--n-optuna-trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Model version")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Outage Prediction Model Training Pipeline")
    print(f"  Region: {args.region} | Version: {args.version}")
    print(f"{'='*60}\n")

    # Load dataset
    data_path = Path(args.data_dir) / f"features_{args.region.lower()}.csv"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Run the feature store builder first.")
        sys.exit(1)

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    dataset = OutageDataset(df)
    print(f"Dataset loaded: {dataset.summary()['total_samples']} samples, "
          f"{dataset.n_features} features, "
          f"{dataset.summary()['positive_rate']:.1%} positive rate\n")

    # Split
    train_df, val_df, test_df = dataset.temporal_split()
    X_train, y_train = dataset.get_X_y(train_df, fit_scaler=True)
    X_val, y_val = dataset.get_X_y(val_df)
    X_test, y_test = dataset.get_X_y(test_df)

    X_train_df = dataset.get_feature_dataframe(train_df)
    X_val_df = dataset.get_feature_dataframe(val_df)
    X_test_df = dataset.get_feature_dataframe(test_df)

    registry = ModelRegistry(args.model_dir)
    evaluator = ModelEvaluator()

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = XGBoostOutageModel()
    xgb_metrics = xgb_model.train(X_train_df, y_train, X_val_df, y_val)
    xgb_test_preds = xgb_model.predict_proba(X_test_df)
    xgb_eval = evaluator.evaluate(y_test, xgb_test_preds)
    registry.save_model(xgb_model.model, "xgboost", args.version, args.region, xgb_eval)
    print(f"  AUC-ROC: {xgb_eval['auc_roc']:.4f} | F1: {xgb_eval['f1']:.4f}\n")

    # Train LightGBM
    print("Training LightGBM...")
    lgb_model = LightGBMOutageModel()
    lgb_metrics = lgb_model.train(X_train_df, y_train, X_val_df, y_val)
    lgb_test_preds = lgb_model.predict_proba(X_test_df)
    lgb_eval = evaluator.evaluate(y_test, lgb_test_preds)
    registry.save_model(lgb_model.model, "lightgbm", args.version, args.region, lgb_eval)
    print(f"  AUC-ROC: {lgb_eval['auc_roc']:.4f} | F1: {lgb_eval['f1']:.4f}\n")

    # Train LSTM
    print("Training LSTM with attention...")
    from torch.utils.data import DataLoader
    from src.ml.temporal_models import LSTMOutageModel, TemporalModelTrainer, TimeSeriesDataset

    seq_len = 24
    X_train_seq, y_train_seq = dataset.create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = dataset.create_sequences(X_val, y_val, seq_len)
    X_test_seq, y_test_seq = dataset.create_sequences(X_test, y_test, seq_len)

    train_ds = TimeSeriesDataset(X_train.astype(np.float32), y_train.astype(np.float32), seq_len)
    val_ds = TimeSeriesDataset(X_val.astype(np.float32), y_val.astype(np.float32), seq_len)
    test_ds = TimeSeriesDataset(X_test.astype(np.float32), y_test.astype(np.float32), seq_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    lstm_model = LSTMOutageModel(input_dim=dataset.n_features)
    trainer = TemporalModelTrainer(lstm_model, learning_rate=1e-3, max_epochs=100, patience=10)
    history = trainer.train(train_loader, val_loader)
    lstm_test_preds = trainer.predict(test_loader)

    if len(lstm_test_preds) == len(y_test_seq):
        lstm_eval = evaluator.evaluate(y_test_seq, lstm_test_preds)
    else:
        # Align lengths
        min_len = min(len(lstm_test_preds), len(y_test_seq))
        lstm_eval = evaluator.evaluate(y_test_seq[:min_len], lstm_test_preds[:min_len])

    import torch
    registry.save_model(lstm_model, "lstm", args.version, args.region, lstm_eval, model_type="sequential")
    print(f"  AUC-ROC: {lstm_eval['auc_roc']:.4f} | F1: {lstm_eval['f1']:.4f}")
    print(f"  Best epoch: {history.get('best_epoch', 'N/A')} | Best val loss: {history.get('best_val_loss', 'N/A'):.4f}\n")

    # Build Ensemble
    print("Building ensemble with stacking...")
    ensemble = OutageEnsemble(
        models={"xgboost": xgb_model.model, "lightgbm": lgb_model.model, "lstm": lstm_model},
        model_types={"xgboost": "tabular", "lightgbm": "tabular", "lstm": "sequential"},
    )
    ensemble.fit_stacking(X_val_df.values, X_val_seq, y_val)

    # Uncertainty estimation
    print("Computing uncertainty estimates...")
    ue = UncertaintyEstimator(n_mc_samples=50)

    ensemble_preds = ue.ensemble_predictions(
        {"xgboost": xgb_model.model, "lightgbm": lgb_model.model},
        X_test_df.values,
    )
    mc_preds = ue.mc_dropout_predictions(lstm_model, X_test_seq)

    min_len = min(ensemble_preds.shape[0], mc_preds.shape[0])
    predictions = ue.predict_with_uncertainty(
        ensemble_preds[:min_len], mc_preds[:min_len]
    )

    pred_means = np.array([p.mean for p in predictions])
    final_eval = evaluator.evaluate(y_test[:min_len], pred_means)

    # Calibrate
    ue.calibrate(y_val, xgb_model.predict_proba(X_val_df))
    ece_before = UncertaintyEstimator.expected_calibration_error(y_test[:min_len], pred_means)

    print(f"\n{'='*60}")
    print(f"  Final Results (Ensemble with Uncertainty)")
    print(f"{'='*60}")
    for metric, value in final_eval.items():
        print(f"  {metric:>15}: {value:.4f}")
    print(f"  {'ECE':>15}: {ece_before:.4f}")

    # Bootstrap CIs
    print(f"\nBootstrap 95% Confidence Intervals:")
    ci = evaluator.bootstrap_confidence_intervals(y_test[:min_len], pred_means, n_bootstrap=1000)
    for metric, vals in ci.items():
        print(f"  {metric:>15}: {vals['mean']:.4f} [{vals['lower']:.4f}, {vals['upper']:.4f}]")

    # Promote best model
    registry.promote("xgboost", args.version, args.region)
    registry.promote("lightgbm", args.version, args.region)
    registry.promote("lstm", args.version, args.region)

    print(f"\nModels saved to {args.model_dir}/{args.region}/")
    print("Training complete.\n")


if __name__ == "__main__":
    main()
