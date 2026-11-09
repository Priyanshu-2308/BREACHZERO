"""
Task 4 for Person 4: OOD False Positive Diagnosis
Diagnoses why transformers flag 36-38% of benign prompts in data/ood_jailbreak.csv.

Without training on this held-out file:
- Scores label-0 rows with English DistilBERT and splits into flagged (P>=0.5) vs not flagged (P<0.5).
- Reports median length (characters and tokens) of both groups.
- Finds the 15 most over-represented words in the flagged group.
- Prints 10 example flagged prompts (truncated to 200 chars).
- Saves summary statistics to results/ood_fpr_diagnosis.csv.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from guard import predict_proba

STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "as", "was",
    "with", "on", "are", "by", "this", "be", "or", "from", "at", "an", "your",
    "all", "have", "not", "but", "can", "they", "we", "has", "more", "will",
    "if", "their", "so", "what", "which", "when", "there", "were", "who", "been",
    "than", "its", "also", "into", "any", "other", "some", "time", "about", "would"
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "ood_jailbreak.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    model_dir = os.path.join(MODEL_DIR, "distilbert")

    # Check if cached first_256 probabilities from long_input_check exist
    cache_path = os.path.join(RESULTS_DIR, "_ood_distilbert_probs.npy")
    if os.path.exists(cache_path):
        print(f"Loading precomputed probabilities from {cache_path}...")
        all_probs = np.load(cache_path)
    else:
        print("Scoring all rows with English DistilBERT...")
        all_probs = predict_proba(df["text"].astype(str).tolist(), model_dir)

    df["prob"] = all_probs
    df["flagged"] = df["prob"] >= 0.5

    # Filter to benign rows (label == 0)
    benign_df = df[df["label"] == 0].copy()
    n_benign = len(benign_df)
    n_flagged = int(benign_df["flagged"].sum())
    n_unflagged = n_benign - n_flagged
    fpr = n_flagged / n_benign

    print(f"Total benign rows in OOD set: {n_benign}")
    print(f"Flagged as attack (False Positives): {n_flagged} ({fpr:.4f} FPR)")
    print(f"Correctly unflagged: {n_unflagged}")

    # Tokenizer for token length calculation
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    flagged_texts = benign_df[benign_df["flagged"]]["text"].astype(str).tolist()
    unflagged_texts = benign_df[~benign_df["flagged"]]["text"].astype(str).tolist()

    # Character lengths
    char_lens_flagged = [len(t) for t in flagged_texts]
    char_lens_unflagged = [len(t) for t in unflagged_texts]
    med_char_flagged = float(np.median(char_lens_flagged))
    med_char_unflagged = float(np.median(char_lens_unflagged))

    # Token lengths
    tok_lens_flagged = [len(tokenizer.encode(t, add_special_tokens=False)) for t in flagged_texts]
    tok_lens_unflagged = [len(tokenizer.encode(t, add_special_tokens=False)) for t in unflagged_texts]
    med_tok_flagged = float(np.median(tok_lens_flagged))
    med_tok_unflagged = float(np.median(tok_lens_unflagged))

    print("\n" + "=" * 70)
    print("LENGTH COMPARISON: FLAGGED (FP) vs UNFLAGGED BENIGN PROMPTS")
    print("=" * 70)
    print(f"{'Group':<25}{'Count':>10}{'Median Chars':>16}{'Median Tokens':>16}")
    print("-" * 70)
    print(f"{'Flagged (False Positives)':<25}{n_flagged:>10}{med_char_flagged:>16.1f}{med_tok_flagged:>16.1f}")
    print(f"{'Not Flagged (True Negatives)':<25}{n_unflagged:>10}{med_char_unflagged:>16.1f}{med_tok_unflagged:>16.1f}")
    print("=" * 70)

    # Word over-representation analysis
    def get_words(texts):
        words = []
        for t in texts:
            found = re.findall(r"\b[a-z]{3,}\b", t.lower())
            words.extend(found)
        return words

    flagged_words = [w for w in get_words(flagged_texts) if w not in STOPWORDS]
    unflagged_words = [w for w in get_words(unflagged_texts) if w not in STOPWORDS]

    count_flagged = Counter(flagged_words)
    count_unflagged = Counter(unflagged_words)

    total_flagged_words = max(1, len(flagged_words))
    total_unflagged_words = max(1, len(unflagged_words))

    # Compute relative frequency ratio for words appearing at least 5 times in flagged
    candidates = []
    for word, c_f in count_flagged.items():
        if c_f >= 5:
            p_f = c_f / total_flagged_words
            c_u = count_unflagged.get(word, 0)
            p_u = (c_u + 1) / total_unflagged_words
            ratio = p_f / p_u
            candidates.append((word, ratio, c_f, c_u))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_15_words = candidates[:15]

    print("\n" + "=" * 70)
    print("TOP 15 MOST OVER-REPRESENTED WORDS IN FLAGGED BENIGN PROMPTS")
    print("=" * 70)
    print(f"{'Rank':<6}{'Word':<20}{'Ratio':>10}{'Count in FP':>14}{'Count in TN':>14}")
    print("-" * 70)
    for i, (w, r, cf, cu) in enumerate(top_15_words, 1):
        print(f"{i:<6}{w:<20}{r:>10.2f}{cf:>14}{cu:>14}")
    print("=" * 70)

    # 10 example flagged prompts truncated to 200 characters
    print("\n" + "=" * 70)
    print("10 EXAMPLE FLAGGED BENIGN PROMPTS (truncated to 200 chars):")
    print("=" * 70)
    examples_for_csv = []
    for i, t in enumerate(flagged_texts[:10], 1):
        sample_short = t[:200].replace("\n", " ")
        examples_for_csv.append(sample_short)
        print(f"[{i:2d}] {sample_short}...")

    # Save summary statistics to results/ood_fpr_diagnosis.csv
    records = [
        {"metric": "total_benign", "value": n_benign},
        {"metric": "flagged_false_positives", "value": n_flagged},
        {"metric": "false_positive_rate", "value": fpr},
        {"metric": "median_chars_flagged", "value": med_char_flagged},
        {"metric": "median_chars_unflagged", "value": med_char_unflagged},
        {"metric": "median_tokens_flagged", "value": med_tok_flagged},
        {"metric": "median_tokens_unflagged", "value": med_tok_unflagged},
    ]
    for rank, (w, r, cf, cu) in enumerate(top_15_words, 1):
        records.append({
            "metric": f"top_{rank:02d}_word",
            "value": f"{w} (ratio={r:.2f}, FP={cf}, TN={cu})"
        })

    out_df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "ood_fpr_diagnosis.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved diagnosis summary to {out_csv}")
    print("ood_fpr_diagnosis.py finished.")


if __name__ == "__main__":
    main()
