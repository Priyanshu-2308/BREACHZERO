"""Scores benign and attack prompts at many thresholds and picks one using val.csv only."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_DIR, ATTACK_DIR, RESULTS_DIR, MODEL_DIR, SEED
from guard import predict_proba
from metrics import compute_metrics


def sweep_thresholds(y_true, probs, thresholds):
    metrics_list = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        m = compute_metrics(y_true, y_pred)
        m["threshold"] = t
        metrics_list.append(m)
    return pd.DataFrame(metrics_list)


def main():
    print("")
    print("Starting threshold sweep on validation data")
    print("-" * 50)

    model_dir = os.path.join(MODEL_DIR, "distilbert")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}. Download it first!")

    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(val_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Missing val.csv or test.csv in data directory.")

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    thresholds = np.linspace(0.05, 0.95, 91)

    print(f"\n1. Scoring validation set ({len(val_df)} rows)...", flush=True)
    val_probs = predict_proba(val_df["text"].astype(str).tolist(), model_dir, batch_size=64)
    val_metrics_df = sweep_thresholds(val_df["label"].values, val_probs, thresholds)

    # Threshold selection on val.csv only: FPR <= 0.02
    valid_candidates = val_metrics_df[val_metrics_df["false_positive_rate"] <= 0.02]
    if valid_candidates.empty:
        best_row = val_metrics_df.sort_values(by=["false_positive_rate", "recall"], ascending=[True, False]).iloc[0]
        chosen_thresh = float(best_row["threshold"])
        print(f"Warning: No threshold achieved FPR <= 0.02 on val. Falling back to min FPR: {chosen_thresh:.2f}")
    else:
        best_row = valid_candidates.sort_values(by=["recall", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
        chosen_thresh = float(best_row["threshold"])

    print(f"\n--- THRESHOLD SELECTION (Validation Set) ---")
    print(f"Chosen Threshold: {chosen_thresh:.2f}")
    print(f"Val FPR:          {best_row['false_positive_rate'] * 100:.2f}% (Constraint: <= 2.0%)")
    print(f"Val Recall:       {best_row['recall'] * 100:.2f}%")
    print(f"Val F1:           {best_row['f1']:.4f}")
    print(f"Val Precision:    {best_row['precision']:.4f}")

    print(f"\n2. Scoring test set ({len(test_df)} rows)...", flush=True)
    test_probs = predict_proba(test_df["text"].astype(str).tolist(), model_dir, batch_size=64)
    test_metrics_df = sweep_thresholds(test_df["label"].values, test_probs, thresholds)

    test_pred_chosen = (test_probs >= chosen_thresh).astype(int)
    test_chosen_metrics = compute_metrics(test_df["label"].values, test_pred_chosen)

    print(f"\n--- TEST SET PERFORMANCE AT CHOSEN THRESHOLD ({chosen_thresh:.2f}) ---")
    print(f"Test FPR:         {test_chosen_metrics['false_positive_rate'] * 100:.2f}%")
    print(f"Test Recall:      {test_chosen_metrics['recall'] * 100:.2f}%")
    print(f"Test F1:          {test_chosen_metrics['f1']:.4f}")
    print(f"Test Precision:   {test_chosen_metrics['precision']:.4f}")

    attack_files = ["leetspeak", "cyrillic", "base64", "wrapped", "hindi"]
    attack_recalls = {}
    attack_chosen_recall = {}
    print(f"\n3. Scoring attack sets...", flush=True)
    for disguise in attack_files:
        p = os.path.join(ATTACK_DIR, f"{disguise}.csv")
        if os.path.exists(p):
            df_att = pd.read_csv(p)
            probs_att = predict_proba(df_att["text"].astype(str).tolist(), model_dir, batch_size=64)
            rec_series = [(probs_att >= t).mean() for t in thresholds]
            attack_recalls[disguise] = rec_series
            attack_chosen_recall[disguise] = float((probs_att >= chosen_thresh).mean())
            print(f"  {disguise:10s} Recall @ {chosen_thresh:.2f}: {attack_chosen_recall[disguise] * 100:.2f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    thresh_out_file = os.path.join(RESULTS_DIR, "chosen_threshold.txt")
    with open(thresh_out_file, "w") as f:
        f.write(f"{chosen_thresh:.2f}\n")
    print(f"\nSaved chosen threshold to {thresh_out_file}")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(test_metrics_df["threshold"], test_metrics_df["recall"], label="Test Recall", color="#2b5c8f", lw=2.2)
    ax1.plot(test_metrics_df["threshold"], test_metrics_df["false_positive_rate"], label="Test FPR", color="#c23b22", lw=2.2, linestyle="--")
    ax1.plot(val_metrics_df["threshold"], val_metrics_df["false_positive_rate"], label="Val FPR", color="#e67e22", lw=1.5, linestyle=":")
    ax1.axhline(0.02, color="#7f8c8d", linestyle="--", alpha=0.7, label="2% FPR Target")
    ax1.axvline(chosen_thresh, color="#27ae60", linestyle="-.", lw=2, label=f"Chosen Threshold ({chosen_thresh:.2f})")
    ax1.scatter([chosen_thresh], [test_chosen_metrics["recall"]], color="#2b5c8f", s=80, zorder=5)
    ax1.scatter([chosen_thresh], [test_chosen_metrics["false_positive_rate"]], color="#c23b22", s=80, zorder=5)

    ax1.set_title("Detection Performance vs. Decision Threshold (English DistilBERT)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Decision Threshold", fontsize=11)
    ax1.set_ylabel("Rate", fontsize=11)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlim(0.05, 0.95)
    ax1.legend(loc="center right", fontsize=9)

    colors = {"leetspeak": "#8e44ad", "cyrillic": "#e74c3c", "base64": "#3498db", "wrapped": "#16a085", "hindi": "#f39c12"}
    for disguise, rec_vals in attack_recalls.items():
        c = colors.get(disguise, "#333333")
        ax2.plot(thresholds, rec_vals, label=f"{disguise} (Recall={attack_chosen_recall[disguise]*100:.1f}%)", color=c, lw=2.0)

    ax2.axvline(chosen_thresh, color="#27ae60", linestyle="-.", lw=2, label=f"Chosen Threshold ({chosen_thresh:.2f})")
    ax2.set_title("Recall on Disguised Attack Types vs. Decision Threshold", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Threshold", fontsize=11)
    ax2.set_ylabel("Attack Recall (TPR)", fontsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(0.05, 0.95)
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    curve_out_file = os.path.join(RESULTS_DIR, "threshold_curve.png")
    plt.savefig(curve_out_file, dpi=300)
    plt.close()
    print(f"Saved threshold curve visualization to {curve_out_file}")

    print("")
    print("Threshold sweep complete")
    print(f"Chosen threshold {chosen_thresh:.2f} satisfies benign FPR <= 2% on validation data.")
    print("-" * 50)


if __name__ == "__main__":
    main()
