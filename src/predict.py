# src/predict.py
from pathlib import Path
import pandas as pd
import joblib
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/input.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found.")

    model_path = Path("models/readmission_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train the model first.")

    model = joblib.load(model_path)
    df_new = pd.read_csv(input_path)

    # Drop target/pred columns if present
    for col in ["readmit_30d", "readmit_30_pred", "readmit_30_proba"]:
        if col in df_new.columns:
            df_new = df_new.drop(columns=[col])

    preds = model.predict(df_new)
    probas = model.predict_proba(df_new)[:, 1]

    df_out = df_new.copy()
    df_out["readmit_30_pred"] = preds
    df_out["readmit_30_proba"] = probas

    out_path = input_path.parent / f"{input_path.stem}_with_preds.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")

if __name__ == "__main__":
    main()
