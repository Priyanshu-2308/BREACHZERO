"""
Task 5 for Person 4: Augmentation Experiment (Leave-One-Disguise-Out)
Builds adversarial augmentation datasets and tests recovery of robustness:

1. Samples 200 injections and 200 benign prompts from data/train.csv (NOT test.csv).
2. Applies all 5 disguises (leetspeak, cyrillic, base64, wrapped, hindi) to both injections and benigns.
3. Generates combined data/augmented_train_extra.csv (2000 rows: 1000 attacks, 1000 benigns).
4. Generates leave-one-out subsets for held-out testing (hindi and base64).
5. Trains:
   - models/distilbert_aug_hindi (held out: hindi)
   - models/distilbert_aug_base64 (held out: base64)
   - models/distilbert_aug_all (all 5 types, NOT an unseen-type test)
6. Evaluates before vs after on:
   - data/attacks/*.csv (Recall)
   - data/test.csv (FPR)
   - data/ood_jailbreak.csv (FPR)
   - data/benign_disguised/*.csv (FPR)
7. Saves results/augment_results.csv and results/augment_results.png.
"""

import os
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ATTACK_DIR, DATA_DIR, MODEL_DIR, RESULTS_DIR, SEED
from attacks import to_leetspeak, to_cyrillic, to_base64, to_wrapped, to_hindi
from guard import predict_proba
from metrics import compute_metrics

DISGUISES = ["leetspeak", "cyrillic", "base64", "wrapped", "hindi"]
DISGUISE_FNS = {
    "leetspeak": to_leetspeak,
    "cyrillic": to_cyrillic,
    "base64": to_base64,
    "wrapped": to_wrapped,
    "hindi": to_hindi,
}


def build_augmented_datasets():
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")

    train_df = pd.read_csv(train_path)
    injections_df = train_df[train_df["label"] == 1].sample(n=200, random_state=SEED).reset_index(drop=True)
    benign_df = train_df[train_df["label"] == 0].sample(n=200, random_state=SEED).reset_index(drop=True)
    print(f"Sampled 200 injections and 200 benign prompts from {train_path} for augmentation.", flush=True)

    pool = []  # list of dicts with text, label, disguise_type
    for name, fn in DISGUISE_FNS.items():
        # Disguised attacks (label 1)
        for t in injections_df["text"]:
            pool.append({"text": fn(t), "label": 1, "disguise_type": name})
        # Disguised benign controls (label 0)
        for t in benign_df["text"]:
            pool.append({"text": fn(t), "label": 0, "disguise_type": name})

    pool_df = pd.DataFrame(pool).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 1. Combined all-types extra set -> data/augmented_train_extra.csv
    all_extra_path = os.path.join(DATA_DIR, "augmented_train_extra.csv")
    pool_df[["text", "label"]].to_csv(all_extra_path, index=False)
    print(f"Saved combined all-types extra dataset to {all_extra_path} ({len(pool_df)} rows).", flush=True)

    # 2. Leave-one-out extra sets for 2 held-out types: hindi and base64
    extra_files = {"all": all_extra_path}
    for held_out in ["hindi", "base64"]:
        leave_df = pool_df[pool_df["disguise_type"] != held_out][["text", "label"]]
        p = os.path.join(DATA_DIR, f"augmented_train_extra_leave_{held_out}.csv")
        leave_df.to_csv(p, index=False)
        extra_files[held_out] = p
        print(f"Saved leave-{held_out}-out dataset to {p} ({len(leave_df)} rows).", flush=True)

    return extra_files


def train_augmented_model(extra_csv, out_dir, python_bin):
    """Fine-tune DistilBERT on extra augmented data."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = f"{out_dir.rstrip('/')}_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
        python_bin, "train.py",
        "--model", "models/distilbert",
        "--extra_train", extra_csv,
        "--out", out_dir,
        "--epochs", "1",
        "--limit", "100",
    ]
    print(f"\nRunning: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for {out_dir}")

    # Remove temporary checkpoint directory to keep models/ clean
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def evaluate_model_on_suite(model_name, model_dir):
    """Evaluates a model across attacks, test set, ood set, and benign disguised sets."""
    print(f"\nEvaluating suite for {model_name} ({model_dir})...", flush=True)
    records = []
    threshold = 0.5

    # 1. Attack sets (Recall)
    for disguise in DISGUISES:
        p = os.path.join(ATTACK_DIR, f"{disguise}.csv")
        df = pd.read_csv(p)
        probs = predict_proba(df["text"].astype(str).tolist(), model_dir, batch_size=64)
        y_pred = (probs >= threshold).astype(int)
        m = compute_metrics(df["label"].values, y_pred)
        records.append({
            "model": model_name,
            "eval_set": f"attack_{disguise}",
            "metric_type": "recall",
            "score": m["recall"],
            "n": len(df),
        })

    # 2. Test set FPR
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    probs_test = predict_proba(test_df["text"].astype(str).tolist(), model_dir, batch_size=64)
    y_pred_test = (probs_test >= threshold).astype(int)
    m_test = compute_metrics(test_df["label"].values, y_pred_test)
    records.append({
        "model": model_name,
        "eval_set": "test_set",
        "metric_type": "fpr",
        "score": m_test["false_positive_rate"],
        "n": len(test_df),
    })

    # 3. OOD Jailbreak FPR
    ood_df = pd.read_csv(os.path.join(DATA_DIR, "ood_jailbreak.csv"))
    probs_ood = predict_proba(ood_df["text"].astype(str).tolist(), model_dir, batch_size=64)
    y_pred_ood = (probs_ood >= threshold).astype(int)
    m_ood = compute_metrics(ood_df["label"].values, y_pred_ood)
    records.append({
        "model": model_name,
        "eval_set": "ood_jailbreak",
        "metric_type": "fpr",
        "score": m_ood["false_positive_rate"],
        "n": len(ood_df),
    })

    # 4. Benign disguised sets FPR
    benign_disguised_dir = os.path.join(DATA_DIR, "benign_disguised")
    for disguise in DISGUISES:
        p = os.path.join(benign_disguised_dir, f"{disguise}.csv")
        df = pd.read_csv(p)
        probs = predict_proba(df["text"].astype(str).tolist(), model_dir, batch_size=64)
        fpr = float(np.mean(probs >= threshold))
        records.append({
            "model": model_name,
            "eval_set": f"benign_{disguise}",
            "metric_type": "fpr",
            "score": fpr,
            "n": len(df),
        })

    return records


def plot_results(df_results, out_png):
    """Generate a clean grouped bar chart comparing before vs after augmentation."""
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Recall on Disguised Attacks
    ax1 = axes[0]
    attack_data = df_results[df_results["eval_set"].str.startswith("attack_")].copy()
    attack_data["disguise"] = attack_data["eval_set"].str.replace("attack_", "")
    
    models = attack_data["model"].unique()
    x = np.arange(len(DISGUISES))
    width = 0.8 / len(models)

    for i, m in enumerate(models):
        m_subset = attack_data[attack_data["model"] == m].set_index("disguise").reindex(DISGUISES)
        ax1.bar(x + i * width, m_subset["score"], width, label=m)

    ax1.set_title("Recall on Disguised Attacks (Higher is Better)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(DISGUISES, fontsize=10)
    ax1.set_ylabel("Recall (Threshold 0.5)")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc="lower right", fontsize=8)

    # Plot 2: FPR on Disguised Benign Controls + Test + OOD
    ax2 = axes[1]
    benign_eval_sets = [f"benign_{d}" for d in DISGUISES] + ["test_set", "ood_jailbreak"]
    fpr_data = df_results[df_results["eval_set"].isin(benign_eval_sets)].copy()
    labels = [s.replace("benign_", "") for s in benign_eval_sets]

    x2 = np.arange(len(benign_eval_sets))
    width2 = 0.8 / len(models)

    for i, m in enumerate(models):
        m_subset = fpr_data[fpr_data["model"] == m].set_index("eval_set").reindex(benign_eval_sets)
        ax2.bar(x2 + i * width2, m_subset["score"], width2, label=m)

    ax2.set_title("FPR on Benign Prompts (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x2 + width2 * (len(models) - 1) / 2)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("False Positive Rate")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved grouped bar chart to {out_png}", flush=True)


def main():
    python_bin = sys.executable
    print("=" * 70, flush=True)
    print("STARTING TASK 5: AUGMENTATION EXPERIMENT", flush=True)
    print("=" * 70, flush=True)

    extra_files = build_augmented_datasets()

    models_to_train = [
        ("distilbert_aug_hindi", extra_files["hindi"]),
        ("distilbert_aug_base64", extra_files["base64"]),
        ("distilbert_aug_all", extra_files["all"]),
    ]

    trained_models = []
    for model_name, extra_csv in models_to_train:
        out_path = os.path.join(MODEL_DIR, model_name)
        train_augmented_model(extra_csv, out_path, python_bin)
        trained_models.append((model_name, out_path))

    # Evaluate before (base distilbert) and all after models
    all_eval_records = []
    base_distilbert_path = os.path.join(MODEL_DIR, "distilbert")
    all_eval_records.extend(evaluate_model_on_suite("distilbert_before", base_distilbert_path))

    for model_name, model_path in trained_models:
        all_eval_records.extend(evaluate_model_on_suite(model_name, model_path))

    out_df = pd.DataFrame(all_eval_records)
    out_csv = os.path.join(RESULTS_DIR, "augment_results.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved evaluation metrics to {out_csv}", flush=True)

    out_png = os.path.join(RESULTS_DIR, "augment_results.png")
    plot_results(out_df, out_png)

    # Clean up temporary leave-one-out CSVs (only keep data/augmented_train_extra.csv as requested)
    for held_out in ["hindi", "base64"]:
        temp_csv = extra_files[held_out]
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            print(f"Removed temporary file: {temp_csv}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("AUGMENTATION EXPERIMENT SUMMARY RESULTS")
    print("=" * 70, flush=True)
    summary_pivot = out_df.pivot(index="eval_set", columns="model", values="score")
    print(summary_pivot.to_string(), flush=True)
    print("\naugment_experiment.py finished.", flush=True)


if __name__ == "__main__":
    main()
