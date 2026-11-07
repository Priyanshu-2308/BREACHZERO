"""
Baseline TF-IDF + Logistic Regression model for BREACHZERO.

Pipeline:
- TfidfVectorizer (word 1-2 grams, sublinear_tf, tuned with optional char n-grams)
- LogisticRegression(class_weight="balanced")

Tuned lightly on data/val.csv.
Evaluated on:
- data/test.csv
- data/ood_jailbreak.csv
- data/attacks/leetspeak.csv
- data/attacks/cyrillic.csv
- data/attacks/base64.csv
- data/attacks/wrapped.csv
- data/attacks/hindi.csv

Saves:
- models/baseline.joblib
- results/baseline_metrics.csv (columns: model,test_set,recall,false_positive_rate,f1,n)
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from config import ATTACK_DIR, DATA_DIR, MODEL_DIR, RESULTS_DIR, SEED
from metrics import compute_metrics


def build_candidate_pipelines():
    """
    Define candidate vectorization + classifier setups for tuning on val.csv.
    """
    candidates = {}

    # Word 1-2 grams only
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    # Word 1-2 grams + Char 3-5 grams
    word_char_vec = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    ngram_range=(3, 5),
                    analyzer="char_wb",
                    sublinear_tf=True,
                    min_df=3,
                ),
            ),
        ]
    )

    for c_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        candidates[f"word_only_C_{c_val}"] = Pipeline(
            [
                ("tfidf", word_vec),
                (
                    "clf",
                    LogisticRegression(
                        C=c_val,
                        class_weight="balanced",
                        random_state=SEED,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        candidates[f"word_char_C_{c_val}"] = Pipeline(
            [
                ("tfidf", word_char_vec),
                (
                    "clf",
                    LogisticRegression(
                        C=c_val,
                        class_weight="balanced",
                        random_state=SEED,
                        max_iter=1000,
                    ),
                ),
            ]
        )

    return candidates


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    ood_path = os.path.join(DATA_DIR, "ood_jailbreak.csv")

    for p in [train_path, val_path, test_path, ood_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required data file: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    ood_df = pd.read_csv(ood_path)

    print(f"Loaded datasets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, ood={len(ood_df)}")

    # 2. Light tuning on val.csv
    print("\nTuning baseline candidates on val.csv...")
    candidates = build_candidate_pipelines()
    best_name = None
    best_pipe = None
    best_val_f1 = -1.0
    best_val_metrics = None

    for name, pipe in candidates.items():
        pipe.fit(train_df["text"], train_df["label"])
        val_pred = pipe.predict(val_df["text"])
        m = compute_metrics(val_df["label"], val_pred)
        print(f"  [{name}] val F1={m['f1']:.4f}, recall={m['recall']:.4f}, FPR={m['false_positive_rate']:.4f}")

        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_name = name
            best_pipe = pipe
            best_val_metrics = m

    print(f"\nBest model on val: {best_name} (val F1={best_val_f1:.4f})")

    # 3. Save best pipeline to models/baseline.joblib
    model_path = os.path.join(MODEL_DIR, "baseline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"Saved trained baseline pipeline to {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    # 4. Evaluation on test, ood, and 5 attacks
    eval_sets = [
        ("test", test_df),
        ("ood_jailbreak", ood_df),
    ]

    attack_types = ["leetspeak", "cyrillic", "base64", "wrapped", "hindi"]
    for at in attack_types:
        at_path = os.path.join(ATTACK_DIR, f"{at}.csv")
        if not os.path.exists(at_path):
            raise FileNotFoundError(f"Missing attack file: {at_path}. Run attacks.py first.")
        at_df = pd.read_csv(at_path)
        eval_sets.append((at, at_df))

    records = []
    print("\n" + "=" * 75)
    print(f"{'Test Set':<15} {'N':<6} {'Recall':<10} {'FPR':<10} {'Precision':<12} {'F1':<10}")
    print("=" * 75)

    for test_set_name, df_eval in eval_sets:
        y_true = df_eval["label"].values
        y_pred = best_pipe.predict(df_eval["text"].astype(str).values)
        m = compute_metrics(y_true, y_pred)

        fpr_str = f"{m['false_positive_rate']:.4f}" if not pd.isna(m["false_positive_rate"]) else "NaN"
        prec_str = f"{m['precision']:.4f}" if not pd.isna(m["precision"]) else "NaN"
        f1_str = f"{m['f1']:.4f}" if not pd.isna(m["f1"]) else "NaN"

        print(
            f"{test_set_name:<15} {m['n']:<6} {m['recall']:<10.4f} {fpr_str:<10} {prec_str:<12} {f1_str:<10}"
        )

        records.append(
            {
                "model": "baseline",
                "test_set": test_set_name,
                "recall": m["recall"],
                "false_positive_rate": m["false_positive_rate"],
                "f1": m["f1"],
                "n": m["n"],
            }
        )

    print("=" * 75)

    # 5. Save results/baseline_metrics.csv
    results_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(results_path, index=False)
    print(f"\nSaved metrics to {results_path} (columns: {list(metrics_df.columns)})")
    print("\nbaseline.py finished successfully.")


if __name__ == "__main__":
    main()

