from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

DATA_PATH = RAW / "diabetic_data.csv"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Expected data/raw/diabetic_data.csv")

    df = pd.read_csv(DATA_PATH)

    # target: readmitted within 30 days
    # column 'readmitted' has: 'NO', '>30', '<30'
    df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

    # pick a small set of structured features to start
    keep_cols = [
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "readmit_30d",
    ]
    df = df[keep_cols].copy()

    # age is a bucket like "[60-70)" -> map to midpoint
    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }
    df["age"] = df["age"].map(age_map)

    # drop rows where age failed to map
    df = df.dropna(subset=["age"])

    out_path = PROC / "readmission_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows")
    print(df["readmit_30d"].value_counts(normalize=True))

if __name__ == "__main__":
    main()
