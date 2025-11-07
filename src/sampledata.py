# src/make_new_admissions.py
import pandas as pd
from pathlib import Path

processed = Path("data/processed/readmission_dataset.csv")
new_path = Path("data/processed/new_admissions.csv")

df = pd.read_csv(processed)

# Drop target and prediction columns if they exist
cols_to_drop = ["readmit_30d", "readmit_30_pred", "readmit_30_proba"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# Take a random sample of 50 admissions for demonstration
sample_df = df.sample(n=50, random_state=42)

new_path.parent.mkdir(parents=True, exist_ok=True)
sample_df.to_csv(new_path, index=False)

print(f"Saved sample new admissions file to {new_path}")
