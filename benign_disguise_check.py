"""Generates disguised benign controls and evaluates model false positive rates."""

import os
import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, SEED
from attacks import to_leetspeak, to_cyrillic, to_base64, to_wrapped, to_hindi
from guard import predict_proba

DISGUISE_DIR = os.path.join(DATA_DIR, "benign_disguised")
DISGUISES = ["leetspeak", "cyrillic", "base64", "wrapped", "hindi"]


def generate_benign_disguised_sets():
    os.makedirs(DISGUISE_DIR, exist_ok=True)
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing {test_path}")

    test_df = pd.read_csv(test_path)
    benign_df = test_df[test_df["label"] == 0].copy()

    n_sample = min(190, len(benign_df))
    sample_df = benign_df.sample(n=n_sample, random_state=SEED).reset_index(drop=True)
    print(f"Sampled {n_sample} benign prompts from {test_path} (random_state={SEED}).")

    generators = {
        "leetspeak": to_leetspeak,
        "cyrillic": to_cyrillic,
        "base64": to_base64,
        "wrapped": to_wrapped,
        "hindi": to_hindi,
    }

    created_files = {}
    for disguise_name, gen_fn in generators.items():
        transformed = [gen_fn(t) for t in sample_df["text"]]
        df_disguised = pd.DataFrame({
            "text": transformed,
            "label": 0,
            "original_text": sample_df["text"],
            "attack_type": disguise_name,
        })
        out_csv = os.path.join(DISGUISE_DIR, f"{disguise_name}.csv")
        df_disguised.to_csv(out_csv, index=False)
        created_files[disguise_name] = df_disguised
        print(f"Saved {len(df_disguised)} rows to {out_csv}")

    return created_files


def evaluate_models_on_disguised_benign(disguised_sets):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = [
        ("baseline", os.path.join(MODEL_DIR, "baseline.joblib")),
        ("distilbert", os.path.join(MODEL_DIR, "distilbert")),
        ("distilbert_multilingual", os.path.join(MODEL_DIR, "distilbert_multilingual")),
    ]

    records = []
    threshold = 0.5

    for model_name, model_path in models:
        print(f"\nEvaluating {model_name} on disguised benign sets...")
        if not os.path.exists(model_path):
            print(f"Warning: model path {model_path} does not exist, skipping.")
            continue

        if model_name == "baseline":
            pipe = joblib.load(model_path)
            for disguise_name in DISGUISES:
                df = disguised_sets[disguise_name]
                probs = pipe.predict_proba(df["text"].astype(str).tolist())[:, 1]
                fpr = float(np.mean(probs >= threshold))
                records.append({
                    "model": model_name,
                    "disguise": disguise_name,
                    "false_positive_rate": fpr,
                    "n": len(df),
                })
        else:
            for disguise_name in DISGUISES:
                df = disguised_sets[disguise_name]
                probs = predict_proba(df["text"].astype(str).tolist(), model_path)
                fpr = float(np.mean(probs >= threshold))
                records.append({
                    "model": model_name,
                    "disguise": disguise_name,
                    "false_positive_rate": fpr,
                    "n": len(df),
                })

    out_df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "benign_disguise_check.csv")
    out_df.to_csv(out_csv, index=False)

    print("")
    print("BENIGN DISGUISE CHECK RESULTS (threshold = 0.5)")
    print("-" * 60)
    print(f"{'Model':<26}{'Disguise':<15}{'FPR':>10}{'N':>8}")
    print("-" * 60)
    for r in records:
        print(f"{r['model']:<26}{r['disguise']:<15}{r['false_positive_rate']:>10.4f}{r['n']:>8}")
    print("-" * 60)
    print(f"Saved results to {out_csv}")


def main():
    disguised_sets = generate_benign_disguised_sets()
    evaluate_models_on_disguised_benign(disguised_sets)
    print("\nbenign_disguise_check.py finished.")


if __name__ == "__main__":
    main()
