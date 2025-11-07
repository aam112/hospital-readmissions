import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_path = Path("data/processed/readmission_dataset.csv")
df = pd.read_csv(data_path)
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

print("\nBasic info:")
print(df.info())

print("\nSummary statistics (numeric):")
print(df.describe())

print("\nCategorical column unique counts:")
print(df.select_dtypes(include=['object']).nunique())

target_candidates = [c for c in df.columns if 'readmit' in c or 'readmission' in c or 'target' in c]
if len(target_candidates) > 0:
    target_col = target_candidates[0]
    print(f"\nDetected target column: {target_col}")
    print(df[target_col].value_counts(normalize=True))

    plt.figure(figsize=(5,4))
    sns.countplot(x=target_col, data=df)
    plt.title("Target distribution")
    plt.show()
else:
    print("No target column detected, skipping target plots.")

numeric_df = df.select_dtypes(include=["int64", "float64"])
if not numeric_df.empty:
    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlations")
    plt.show()

if len(target_candidates) > 0:
    target = target_candidates[0]
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols[:5]:  # limit to first 5 for clarity
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

if numeric_df.shape[1] <= 6:
    sns.pairplot(df, hue=target_candidates[0] if target_candidates else None)
    plt.show()
