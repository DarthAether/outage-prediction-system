import pandas as pd
import os

RAW_DATA_PATH = "data/raw/storm_events_details.csv"
OUTPUT_PATH = "data/processed/dataset.csv"

SEVERE_EVENTS = [
    "Thunderstorm Wind",
    "High Wind",
    "Heavy Snow",
    "Ice Storm",
    "Winter Storm",
    "Hurricane",
    "Tornado"
]

def main():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = df.columns.str.upper()

    # Parse timestamp
    df["TIMESTAMP"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")

    # Binary outage-risk label
    df["OUTAGE_RISK"] = df["EVENT_TYPE"].isin(SEVERE_EVENTS).astype(int)

    # Clean numeric fields
    df["MAGNITUDE"] = pd.to_numeric(df["MAGNITUDE"], errors="coerce")
    df["DAMAGE_PROPERTY"] = (
        df["DAMAGE_PROPERTY"]
        .astype(str)
        .str.replace("K", "e3")
        .str.replace("M", "e6")
        .str.replace("B", "e9")
    )

    df["DAMAGE_PROPERTY"] = pd.to_numeric(df["DAMAGE_PROPERTY"], errors="coerce")

    # Final analytical dataset
    dataset = df[
        [
            "TIMESTAMP",
            "MAGNITUDE",
            "DAMAGE_PROPERTY",
            "OUTAGE_RISK"
        ]
    ].dropna()

    dataset.to_csv(OUTPUT_PATH, index=False)

    print("Dataset built successfully")
    print(dataset.head())
    print(dataset.describe())

if __name__ == "__main__":
    main()
