"""
Shared evaluation metrics helper for BREACHZERO.

Label convention: label = 1 means injection/attack, label = 0 means benign.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    """
    Compute binary classification metrics for prompt injection detection.

    Parameters:
        y_true: array-like of true labels (0 = benign, 1 = injection)
        y_pred: array-like of predicted labels (0 = benign, 1 = injection)

    Returns:
        dict: {
            "recall": recall on label 1,
            "false_positive_rate": false positive rate (FP / all label 0),
            "precision": precision on label 1,
            "f1": F1 score on label 1,
            "n": total number of samples
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(len(y_true))

    # Total negatives (label 0)
    num_neg = np.sum(y_true == 0)
    if num_neg == 0:
        fpr = float("nan")
    else:
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fpr = float(fp / num_neg)

    # Total positives (label 1)
    num_pos = np.sum(y_true == 1)
    if num_pos == 0:
        recall = float("nan")
        precision = float("nan")
        f1 = float("nan")
    else:
        recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    return {
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "f1": f1,
        "n": n,
    }


if __name__ == "__main__":
    # Self-test
    y_t = [1, 0, 1, 1, 0, 0]
    y_p = [1, 0, 0, 1, 1, 0]
    m = compute_metrics(y_t, y_p)
    print("Self-test balanced:", m)

    # All positive test (attack files)
    y_t_pos = [1, 1, 1, 1]
    y_p_pos = [1, 1, 0, 1]
    m_pos = compute_metrics(y_t_pos, y_p_pos)
    print("Self-test all-positive:", m_pos)

