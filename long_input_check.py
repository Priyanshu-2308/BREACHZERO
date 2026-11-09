"""
Task 3 for Person 4: Long Input Check
Compares three strategies for long inputs on data/ood_jailbreak.csv using English DistilBERT:
(a) first 256 tokens (default predict_proba truncation)
(b) last 256 tokens
(c) split into overlapping 256-token chunks and take the max score

Reports recall, FPR, precision, F1, and N to results/long_input_check.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from metrics import compute_metrics


def score_batches(model, input_ids_list, device, batch_size=64, desc="Scoring"):
    """
    High-performance bucketed forward pass:
    Sorts sequences by length to minimize padding on CPU,
    prints progress on each batch to prevent idle timeouts,
    then maps probabilities back to original indices.
    """
    if not input_ids_list:
        return np.array([])

    total = len(input_ids_list)
    indexed = sorted(enumerate(input_ids_list), key=lambda x: len(x[1]))
    probs_ordered = [0.0] * total
    n_batches = (total + batch_size - 1) // batch_size

    with torch.inference_mode():
        for b_idx in range(n_batches):
            i = b_idx * batch_size
            batch_tuples = indexed[i : i + batch_size]
            orig_indices = [x[0] for x in batch_tuples]
            batch_ids = [x[1] for x in batch_tuples]
            max_l = max(len(ids) for ids in batch_ids)

            padded = [ids + [0] * (max_l - len(ids)) for ids in batch_ids]
            masks = [[1] * len(ids) + [0] * (max_l - len(ids)) for ids in batch_ids]

            t_ids = torch.tensor(padded, dtype=torch.long, device=device)
            t_mask = torch.tensor(masks, dtype=torch.long, device=device)

            logits = model(input_ids=t_ids, attention_mask=t_mask).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            for orig_idx, prob in zip(orig_indices, p):
                probs_ordered[orig_idx] = float(prob)

            processed = min(i + batch_size, total)
            print(f"[{desc}] Batch {b_idx + 1}/{n_batches} ({processed}/{total} sequences, {100*processed/total:.1f}%)", flush=True)

    return np.array(probs_ordered)


def evaluate_strategies():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "ood_jailbreak.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].values
    n = len(df)
    print(f"Loaded {n} rows from {csv_path} (labels: {pd.Series(y_true).value_counts().to_dict()}).", flush=True)

    model_dir = os.path.join(MODEL_DIR, "distilbert")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Missing {model_dir}")

    print("Loading English DistilBERT tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # Tokenize raw inputs without special tokens
    print("Tokenizing texts to identify length distribution...", flush=True)
    all_raw_tokens = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    long_indices = [i for i, toks in enumerate(all_raw_tokens) if len(toks) > 254]
    short_indices = [i for i, toks in enumerate(all_raw_tokens) if len(toks) <= 254]
    print(f"Short texts (<=254 tokens): {len(short_indices)} | Long texts (>254 tokens): {len(long_indices)}", flush=True)

    # Strategy (a): First 256 tokens
    print("\n--- Strategy (a): First 256 tokens ---", flush=True)
    first_256_ids = []
    for toks in all_raw_tokens:
        sub = toks[:254]
        first_256_ids.append([cls_id] + sub + [sep_id])
    probs_first = score_batches(model, first_256_ids, device, batch_size=64, desc="Strategy (a)")

    # Strategy (b): Last 256 tokens
    print("\n--- Strategy (b): Last 256 tokens ---", flush=True)
    probs_last = np.copy(probs_first)
    if long_indices:
        last_256_long_ids = []
        for i in long_indices:
            sub = all_raw_tokens[i][-254:]
            last_256_long_ids.append([cls_id] + sub + [sep_id])
        long_probs_last = score_batches(model, last_256_long_ids, device, batch_size=64, desc="Strategy (b)")
        for i, p in zip(long_indices, long_probs_last):
            probs_last[i] = p

    # Strategy (c): Overlapping 256-token chunks, max aggregation
    print("\n--- Strategy (c): Overlapping 256-token chunks ---", flush=True)
    probs_chunk = np.copy(probs_first)
    stride = 200  # 56 tokens overlap between consecutive chunks
    if long_indices:
        chunk_list = []
        chunk_text_idx = []
        for i in long_indices:
            toks = all_raw_tokens[i]
            start = 0
            while start < len(toks):
                sub = toks[start : start + 254]
                chunk_list.append([cls_id] + sub + [sep_id])
                chunk_text_idx.append(i)
                if start + 254 >= len(toks):
                    break
                start += stride

        print(f"Scoring {len(chunk_list)} chunks for the {len(long_indices)} long texts...", flush=True)
        long_chunk_probs = score_batches(model, chunk_list, device, batch_size=64, desc="Strategy (c)")

        # Aggregate max per text
        text_chunk_map = {}
        for c_prob, t_i in zip(long_chunk_probs, chunk_text_idx):
            text_chunk_map.setdefault(t_i, []).append(c_prob)
        for i in long_indices:
            probs_chunk[i] = max(text_chunk_map[i])

    # Evaluate each strategy at threshold 0.5
    threshold = 0.5
    strategies = [
        ("first_256", probs_first),
        ("last_256", probs_last),
        ("chunk_max_256", probs_chunk),
    ]

    records = []
    for name, probs in strategies:
        y_pred = (probs >= threshold).astype(int)
        m = compute_metrics(y_true, y_pred)
        records.append({
            "strategy": name,
            "recall": m["recall"],
            "false_positive_rate": m["false_positive_rate"],
            "precision": m["precision"],
            "f1": m["f1"],
            "n": m["n"],
        })

    out_df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "long_input_check.csv")
    out_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78, flush=True)
    print("LONG INPUT CHECK RESULTS (data/ood_jailbreak.csv, English DistilBERT)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Strategy':<20}{'Recall':>10}{'FPR':>12}{'Precision':>12}{'F1':>10}{'N':>8}", flush=True)
    print("-" * 78, flush=True)
    for r in records:
        print(f"{r['strategy']:<20}{r['recall']:>10.4f}{r['false_positive_rate']:>12.4f}"
              f"{r['precision']:>12.4f}{r['f1']:>10.4f}{r['n']:>8}", flush=True)
    print("=" * 78, flush=True)
    print(f"Saved results to {out_csv}", flush=True)


def main():
    evaluate_strategies()
    print("long_input_check.py finished.", flush=True)


if __name__ == "__main__":
    main()
