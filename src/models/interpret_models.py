import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

MODEL_DIR = "models"
DATA_PATH = "data/processed/dataset.csv"

# Load data (only for feature names)
df = pd.read_csv(DATA_PATH)
features = ["MAGNITUDE", "DAMAGE_PROPERTY"]

# ---- Load Models ----
logreg = joblib.load(f"{MODEL_DIR}/logistic_regression.pkl")
rf = joblib.load(f"{MODEL_DIR}/random_forest.pkl")

# ---- Logistic Regression Coefficients ----
lr_coef = pd.Series(
    logreg.coef_[0],
    index=features
).sort_values(ascending=False)

print("Logistic Regression Coefficients:")
print(lr_coef)

# ---- Random Forest Feature Importance ----
rf_importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

# ---- Plot (Paper-Ready) ----
plt.figure(figsize=(6,4))
rf_importance.plot(kind="bar")
plt.ylabel("Importance Score")
plt.title("Feature Importance for Outage Risk Prediction")
plt.tight_layout()
plt.show()
