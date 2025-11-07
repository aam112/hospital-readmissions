# src/train.py
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. load processed data
    data_path = Path("data/processed/readmission_dataset.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded processed data: {df.shape}")

    target_col = "readmit_30d"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in processed data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. build preprocessing
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 3. model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )

    # 4. split and fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    # 5. evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report_txt = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    print("Classification report:")
    print(report_txt)
    print(f"ROC-AUC: {auc:.4f}")

    # 6. save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "readmission_model.pkl")
    print("Saved model to models/readmission_model.pkl")

    # 7. save metrics (txt + json)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # txt
    metrics_txt_path = reports_dir / "metrics.txt"
    with open(metrics_txt_path, "w") as f:
        f.write("Classification report\n")
        f.write(report_txt)
        f.write(f"\nROC-AUC: {auc:.4f}\n")
    print(f"Saved metrics to {metrics_txt_path}")

    # json
    metrics_json_path = reports_dir / "metrics.json"
    metrics_payload = {
        "classification_report": report_dict,
        "roc_auc": auc,
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics JSON to {metrics_json_path}")

    # 8. feature importance (top 20) to reports/figures
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # extract feature names from the trained preprocessing
    preproc = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # numeric names
    num_names = numeric_features

    # categorical names (expanded)
    ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features)

    feature_names = np.concatenate([num_names, cat_names])

    coefs = clf.coef_.flatten()
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": coefs,
        "abs_importance": np.abs(coefs)
    }).sort_values("abs_importance", ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("Coefficient (importance)")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances (Logistic Regression)")
    plt.tight_layout()
    fi_path = figures_dir / "feature_importance.png"
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {fi_path}")


if __name__ == "__main__":
    main()
