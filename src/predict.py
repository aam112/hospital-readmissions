from pathlib import Path
import pandas as pd
import joblib
import sys


def load_model(model_path="models/readmission_model.pkl"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    return joblib.load(model_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/new_data.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file {input_path}")

    clf = load_model()

    df_new = pd.read_csv(input_path)

    for tgt in ["readmit_30", "readmission_30", "readmitted", "target"]:
        if tgt in df_new.columns:
            df_new = df_new.drop(columns=[tgt])

    preds = clf.predict(df_new)
    probas = clf.predict_proba(df_new)[:, 1]

    df_out = df_new.copy()
    df_out["readmit_30_pred"] = preds
    df_out["readmit_30_proba"] = probas

    out_path = input_path.parent / f"{input_path.stem}_with_preds.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
