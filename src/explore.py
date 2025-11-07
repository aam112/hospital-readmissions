# src/explore.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

def main():
    data_path = Path("data/processed/readmission_dataset.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # drop prediction columns if they exist
    drop_cols = ["readmit_30_pred", "readmit_30_proba"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # reset figures folder
    fig_dir = Path("reports/figures")
    if fig_dir.exists():
        shutil.rmtree(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    target = "readmit_30d" if "readmit_30d" in df.columns else None

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # 1. target distribution
    if target:
        plt.figure(figsize=(4, 3))
        sns.countplot(x=target, data=df)
        plt.title("Target distribution (readmit_30d)")
        plt.tight_layout()
        plt.savefig(fig_dir / "target_distribution.png", dpi=150)
        plt.close()

    # 2. numeric correlation heatmap
    num_df = df.select_dtypes(include=["int64", "float64"])
    if not num_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (numeric)")
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    # 3. categorical vs target (first 5)
    if target:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols[:5]:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col, hue=target, data=df)
            plt.title(f"{col} vs {target}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out_file = fig_dir / f"{col}_vs_{target}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()

    print("EDA complete. Plots saved to reports/figures/")

if __name__ == "__main__":
    main()
