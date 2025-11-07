# src/explore.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data_path = Path("data/processed/readmission_dataset.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # Drop prediction columns if they exist
    drop_for_viz = ["readmit_30_pred", "readmit_30_proba"]
    df = df.drop(columns=[c for c in drop_for_viz if c in df.columns], errors="ignore")

    # Basic info
    print(df.info())

    target = "readmit_30d" if "readmit_30d" in df.columns else None

    # Target distribution
    if target:
        plt.figure(figsize=(4, 3))
        sns.countplot(x=target, data=df)
        plt.title("Target distribution (readmit_30d)")
        plt.tight_layout()
        plt.show()

    # Correlation on numeric
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (numeric)")
        plt.tight_layout()
        plt.show()

    # Categorical vs target (first few)
    if target:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols[:5]:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col, hue=target, data=df)
            plt.title(f"{col} vs {target}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
