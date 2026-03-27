import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/dataset.csv"

df = pd.read_csv(DATA_PATH)

# Drop timestamp for correlation
corr_df = df.drop(columns=["TIMESTAMP"])

# Correlation matrix
corr = corr_df.corr()

print("Correlation matrix:")
print(corr)

print("\nCorrelation with OUTAGE_RISK:")
print(corr["OUTAGE_RISK"].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Storm Severity Indicators and Outage Risk")
plt.tight_layout()
plt.show()
