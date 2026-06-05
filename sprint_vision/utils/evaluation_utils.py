"""
Pure metric computation utilities for SPRInT VLM validation.

No torch, no llava, no PIL — only math.  All functions are
deterministic, stateless, and importable before the model is loaded.

Metric definitions for multi-label datasets (chest / endo):
  - Accuracy     : exact match — ALL N labels must agree with ground truth.
                   Near 0% for 19-class chest; expected, not a bug.
                   NOT the same as MedFMC paper "Accuracy" (paper uses AUC).
  - macro_f1     : per-label binary F1 averaged over all N classes.
                   Comparable across training strategies; NOT comparable to
                   MedFMC discriminative-classifier baselines.
  - AUC / mAP    : logit-based ranking metrics (computed in validation.py).
                   These ARE the MedFMC primary metrics and ARE comparable
                   to the paper's ViT/ResNet sigmoid-head baselines.
"""

import math


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_multi_label(text: str, label_names: list) -> set:
    """
    Parse a comma-separated model output into a set of known label strings.

    Uses fuzzy substring matching so minor formatting differences
    (spaces, underscores vs spaces) still map to the right label.
    """
    cleaned = text.lower().strip().rstrip(".")
    if cleaned in ("none", "no finding", "no findings", "normal", ""):
        return set()
    parts = [p.strip() for p in cleaned.replace(";", ",").split(",") if p.strip()]
    found = set()
    for part in parts:
        if part in label_names:
            found.add(part)
        else:
            for lbl in label_names:
                if lbl in part or part in lbl:
                    found.add(lbl)
                    break
    return found


# ── Text-generation metrics ────────────────────────────────────────────────────

def compute_macro_f1_binary(results: list) -> float:
    """
    Macro F1 for binary colon (class 0 / class 1).

    Each result dict must have keys 'pred' (str "0"/"1") and 'gt' (str "0"/"1").
    Returns (F1_class0 + F1_class1) / 2.
    """
    preds = [r["pred"] for r in results]
    gts   = [r["gt"]   for r in results]

    tp = sum(p == "1" and g == "1" for p, g in zip(preds, gts))
    fp = sum(p == "1" and g == "0" for p, g in zip(preds, gts))
    tn = sum(p == "0" and g == "0" for p, g in zip(preds, gts))
    fn = sum(p == "0" and g == "1" for p, g in zip(preds, gts))

    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1  = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0

    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0  = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    return (f1_0 + f1_1) / 2


def compute_macro_f1_multilabel(results: list, label_names: list) -> float:
    """
    Per-label binary F1 averaged over all N classes (standard multi-label macro-F1).

    Each result dict must have keys 'pred_set' (set) and 'gt_set' (set).
    Equivalent to sklearn f1_score(..., average='macro') with each label
    treated as an independent binary problem.
    """
    f1_sum = 0.0
    for lbl in label_names:
        tp = sum(lbl in r["pred_set"] and lbl in r["gt_set"]     for r in results)
        fp = sum(lbl in r["pred_set"] and lbl not in r["gt_set"] for r in results)
        fn = sum(lbl not in r["pred_set"] and lbl in r["gt_set"] for r in results)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_sum += f1
    return f1_sum / len(label_names) if label_names else 0.0


def compute_all_metrics(results: list, label_names: list, is_multi_label: bool) -> dict:
    """
    Compute macro_f1 and exact-match accuracy from text-generation results.

    Returns:
        {
          'macro_f1': float,   — primary text-gen metric
          'accuracy': float,   — exact-match (strict; near 0% for multi-label)
        }
    """
    if not results:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    total   = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total

    if is_multi_label:
        macro_f1 = compute_macro_f1_multilabel(results, label_names)
    else:
        macro_f1 = compute_macro_f1_binary(results)

    return {"macro_f1": macro_f1, "accuracy": accuracy}


# ── Ranking / logit-based metrics ─────────────────────────────────────────────

def compute_auc_trapz(scores: list, labels: list) -> float:
    """
    Trapezoidal ROC-AUC (Wilcoxon-Mann-Whitney rank formula).

    Equivalent to sklearn roc_auc_score.
    Returns float('nan') if all labels are the same class (undefined AUC).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = sum(i + 1 for i, (_, l) in enumerate(pairs) if l == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_ap(scores: list, labels: list) -> float:
    """
    Average Precision (all-points interpolation).

    Equivalent to sklearn average_precision_score.
    Returns 0.0 if there are no positive labels.
    """
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    precision_sum = 0.0
    for _, lbl in pairs:
        if lbl:
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1
    return precision_sum / n_pos
