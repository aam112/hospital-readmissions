import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

def load_data():

    paths = [
        Path("data/processed/readmission_dataset.csv"),
        Path("data/raw/readmission_dataset.csv"),
        Path("readmission_dataset.csv")
    ]
    for p in paths:
        if p.exists():
            print(f"Loading {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("readmission_dataset.csv not found in data folders.")

def main():
    df = load_data()

    # Ensure target column exists
    # Assuming your cleaned dataset has 'readmit_30d' as binary target
    target_col = "readmit_30d"
    if target_col not in df.columns:
        raise ValueError(f"Expected a target column named '{target_col}'.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Separate column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=200, class_weight="balanced"))
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit model
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/readmission_model.pkl")
    print("\nModel saved to models/readmission_model.pkl")

if __name__ == "__main__":
    main()