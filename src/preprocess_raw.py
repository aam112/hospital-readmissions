# src/preprocess_raw.py
import pandas as pd
from pathlib import Path

def main():
    raw_path = Path("data/raw/diabetic_data.csv")
    processed_path = Path("data/processed/readmission_dataset.csv")

    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Replace missing values represented as '?'
    df = df.replace("?", pd.NA)

    # Drop unique identifiers
    drop_cols = ["encounter_id", "patient_nbr"]
    df = df.drop(columns=drop_cols)

    # Create binary target
    df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Optional: drop columns with too many missing values
    missing_fraction = df.isna().mean()
    high_missing = missing_fraction[missing_fraction > 0.4].index
    df = df.drop(columns=high_missing)
    print(f"Dropped high-missing columns: {list(high_missing)}")

    # Save processed version
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned data to {processed_path}")

if __name__ == "__main__":
    main()
