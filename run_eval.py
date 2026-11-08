import argparse
import os

import joblib
import pandas as pd

from config import ATTACK_DIR, DATA_DIR, MODEL_DIR, RESULTS_DIR
from metrics import compute_metrics

ATTACKS = ["leetspeak", "cyrillic", "base64", "wrapped", "hindi"]


def load_eval_sets():
    sets = []
    for name in ["test", "ood_jailbreak"]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run data.py (Person 1) first.")
        sets.append((name, pd.read_csv(path)))
    for name in ATTACKS:
        path = os.path.join(ATTACK_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run attacks.py (Person 2) first.")
        sets.append((name, pd.read_csv(path)))
    return sets


def score(model_name, model_path, df):
    texts = df["text"].astype(str).tolist()
    if model_name == "baseline":
        pipe = joblib.load(model_path)
        return pipe.predict_proba(texts)[:, 1]
    from guard import predict_proba
    return predict_proba(texts, model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_sets = load_eval_sets()

    models = [
        ("baseline", os.path.join(MODEL_DIR, "baseline.joblib")),
        ("distilbert", os.path.join(MODEL_DIR, "distilbert")),
        ("distilbert_multilingual", os.path.join(MODEL_DIR, "distilbert_multilingual")),
    ]

    records = []
    for model_name, model_path in models:
        if not os.path.exists(model_path):
            print(f"SKIP {model_name}: {model_path} not found "
                  f"(train it with train.py, or download it as described in MODELS.md)")
            continue
        print(f"\nevaluating {model_name} ...")
        try:
            all_probs = [score(model_name, model_path, df) for _, df in eval_sets]
        except Exception as e:
            print(f"SKIP {model_name}: could not run it ({type(e).__name__}: {e})")
            continue
        for (set_name, df), probs in zip(eval_sets, all_probs):
            y_pred = (probs >= args.threshold).astype(int)
            m = compute_metrics(df["label"].values, y_pred)
            m.pop("precision")
            m["n"] = len(df)
            records.append({"model": model_name, "test_set": set_name, **m})

    if not records:
        print("No models available to evaluate. Nothing written.")
        return

    out = pd.DataFrame(records)
    path = os.path.join(RESULTS_DIR, "main_results.csv")
    out.to_csv(path, index=False)

    print(f"\nthreshold = {args.threshold}")
    print("=" * 78)
    print(f"{'model':<26}{'test_set':<17}{'recall':>9}{'FPR':>11}{'f1':>9}{'n':>6}")
    print("=" * 78)
    for r in records:
        fpr = "-" if pd.isna(r["false_positive_rate"]) else f"{r['false_positive_rate']:.4f}"
        f1 = "-" if pd.isna(r["f1"]) else f"{r['f1']:.4f}"
        print(f"{r['model']:<26}{r['test_set']:<17}{r['recall']:>9.4f}{fpr:>11}{f1:>9}{r['n']:>6}")
    print("=" * 78)
    print(f"\nSaved {len(records)} rows to {path}")
    print("run_eval.py finished.")


if __name__ == "__main__":
    main()
