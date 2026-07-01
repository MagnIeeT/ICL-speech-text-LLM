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


# ── Tie-correct reference metrics (verified to match sklearn to 1e-16, incl. ties)
# These are NOT yet used for the reported metric — they exist only so the tie
# diagnostic below can estimate how far the current (tie-naive) metric is from
# the sklearn-correct value on our own predictions, with no sklearn dependency.

def auc_avg_rank(scores: list, labels: list) -> float:
    """
    Tie-correct ROC-AUC via the Mann-Whitney U formula with AVERAGE ranks for
    tied scores.  Equals sklearn roc_auc_score exactly, including under ties.
    Returns nan if a single class.
    """
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0   # 1-based average rank for this tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ap_tie_correct(scores: list, labels: list) -> float:
    """
    Tie-correct Average Precision: positives/negatives at the SAME score are
    grouped at one threshold before precision/recall are read off.  Equals
    sklearn average_precision_score exactly, including under ties.
    Returns 0.0 if no positives.
    """
    n = len(scores)
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: -scores[i])
    ap = 0.0
    prev_recall = 0.0
    tp = fp = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return ap


def _fmt(x, width, prec=None):
    """Right-justified fixed-width cell; blank for None, fixed precision for floats."""
    if x is None:
        return " " * width
    if prec is not None:
        return ("{:>%d.%df}" % (width, prec)).format(x)
    return ("{:>%d}" % width).format(x)


def tie_diagnostics(scores_matrix, labels_matrix, label_names,
                    auc_fn=None, ap_fn=None, print_fn=print, header=""):
    """
    Per-class P(Yes) tie report + estimated metric impact, on OUR OWN predictions.

    For each class column it reports:
      n        : number of scored samples
      unique   : number of distinct P(Yes) values
      dups     : n - unique  (count of samples sharing a value with another)
      %P=1     : percent of samples with P(Yes) exactly 1.0  (softmax saturation)
      %P=0     : percent of samples with P(Yes) exactly 0.0
    and, for classes with both labels present, the AUC/AP computed by the CURRENT
    (tie-naive) functions vs the tie-correct (sklearn-equivalent) reference, so
    the delta shows whether ties actually move the reported metric.

    Args:
      scores_matrix : [n_samples][n_classes] of P(Yes) floats.
      labels_matrix : [n_samples][n_classes] of 0/1 ground truth.
      auc_fn/ap_fn  : the CURRENT functions used by the calling path
                      (validation: compute_auc_trapz/compute_ap;
                       inference: sprint_eval._compute_auc/_compute_average_precision).
                      Defaults to compute_auc_trapz/compute_ap.
    Returns a dict with per-class rows and macro current/reference/delta.
    """
    auc_fn = auc_fn or compute_auc_trapz
    ap_fn  = ap_fn  or compute_ap
    n = len(scores_matrix)
    rows = []
    auc_cur_l = []; auc_ref_l = []; ap_cur_l = []; ap_ref_l = []

    for j, lbl in enumerate(label_names):
        col = [scores_matrix[i][j] for i in range(n)]
        lab = [labels_matrix[i][j] for i in range(n)]
        ns = len(col)
        uniq = len(set(col))
        dups = ns - uniq
        pct1 = round(100.0 * sum(1 for s in col if s == 1.0) / ns, 1) if ns else 0.0
        pct0 = round(100.0 * sum(1 for s in col if s == 0.0) / ns, 1) if ns else 0.0
        npos = sum(lab)
        row = {"class": lbl, "n": ns, "unique": uniq, "dups": dups,
               "pct_p1": pct1, "pct_p0": pct0,
               "auc_cur": None, "auc_ref": None, "d_auc": None,
               "ap_cur": None, "ap_ref": None, "d_ap": None}
        if 0 < npos < ns:
            ac = auc_fn(col, lab); ar = auc_avg_rank(col, lab)
            pc = ap_fn(col, lab);  pr = ap_tie_correct(col, lab)
            ac_ok = ac == ac  # not nan
            row["auc_cur"] = round(ac, 4) if ac_ok else None
            row["auc_ref"] = round(ar, 4)
            row["d_auc"]   = round(ac - ar, 4) if ac_ok else None
            row["ap_cur"]  = round(pc, 4)
            row["ap_ref"]  = round(pr, 4)
            row["d_ap"]    = round(pc - pr, 4)
            if ac_ok:
                auc_cur_l.append(ac); auc_ref_l.append(ar)
            ap_cur_l.append(pc); ap_ref_l.append(pr)
        rows.append(row)

    print_fn("=" * 110)
    print_fn(f"[SPRINT::TIE-DIAG] {header}per-class P(Yes) ties + metric impact (current vs tie-correct/sklearn-equiv reference)")
    print_fn(f"  {'class':<22}{'n':>5}{'uniq':>6}{'dups':>6}{'%P=1':>7}{'%P=0':>7}"
             f"{'AUCcur':>8}{'AUCref':>8}{'dAUC':>8}{'APcur':>8}{'APref':>8}{'dAP':>8}")
    for r in rows:
        print_fn(
            f"  {r['class']:<22}{r['n']:>5}{r['unique']:>6}{r['dups']:>6}"
            f"{r['pct_p1']:>7}{r['pct_p0']:>7}"
            f"{_fmt(r['auc_cur'],8,4)}{_fmt(r['auc_ref'],8,4)}{_fmt(r['d_auc'],8,4)}"
            f"{_fmt(r['ap_cur'],8,4)}{_fmt(r['ap_ref'],8,4)}{_fmt(r['d_ap'],8,4)}"
        )

    macro_auc_cur = sum(auc_cur_l) / len(auc_cur_l) if auc_cur_l else float("nan")
    macro_auc_ref = sum(auc_ref_l) / len(auc_ref_l) if auc_ref_l else float("nan")
    mAP_cur = sum(ap_cur_l) / len(ap_cur_l) if ap_cur_l else float("nan")
    mAP_ref = sum(ap_ref_l) / len(ap_ref_l) if ap_ref_l else float("nan")
    print_fn(f"  MACRO  macro_AUC current={macro_auc_cur:.4f}  tie-correct={macro_auc_ref:.4f}  "
             f"delta={macro_auc_cur - macro_auc_ref:+.4f}")
    print_fn(f"  MACRO  mAP       current={mAP_cur:.4f}  tie-correct={mAP_ref:.4f}  "
             f"delta={mAP_cur - mAP_ref:+.4f}")
    print_fn("=" * 110)

    return {
        "per_class": rows,
        "macro_auc_current": macro_auc_cur, "macro_auc_reference": macro_auc_ref,
        "map_current": mAP_cur, "map_reference": mAP_ref,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL metric aggregators — the SINGLE evaluation implementation shared by
# BOTH training-time validation (validation.py::SPRInTValidationManager) AND
# standalone inference (sprint_eval.py). Each takes per-sample result records and
# returns the metric dict. Moved here so there is exactly one scoring code path
# (advisor's requirement); the inference output format is unchanged because
# sprint_eval's _compute_* now delegate to these verbatim.
# ─────────────────────────────────────────────────────────────────────────────

def _auc_0p5_fallback(scores: list, labels: list) -> float:
    """Tie-averaged ROC-AUC (== sklearn/MedFMC); returns 0.5 for a single-class
    column (legacy contract of the binary aggregator)."""
    n_pos = sum(labels)
    if n_pos == 0 or n_pos == len(labels):
        return 0.5
    return auc_avg_rank(scores, labels)


def compute_binary_metrics(results: list, auc_token_id=None) -> dict:
    """Accuracy, AUC, macro F1, sensitivity, specificity for binary tasks (colon).

    Prediction for accuracy/sens/spec is the logit-argmax over the two answer-token
    logits when both are saved (== MedFMC top-1), else the decoded-text prediction.
    AUC/mAP (when auc_token_id is set) use the saved positive-class logit. This is
    the canonical implementation called by both inference and training validation."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    def _pred_for(r):
        lp, ln = r.get("logit_pos"), r.get("logit_neg")
        if lp is not None and ln is not None:
            return "1" if lp > ln else "0"
        return r["pred"]

    preds = [_pred_for(r) for r in results]
    gts   = [r["gt"]   for r in results]

    # Self-consistency: logit-argmax vs decoded-text prediction (diagnostic only).
    _logit_avail = sum(
        1 for r in results
        if r.get("logit_pos") is not None and r.get("logit_neg") is not None
    )
    _comparable = [
        r for r in results
        if r.get("logit_pos") is not None and r.get("logit_neg") is not None
        and r.get("pred") in ("0", "1")
    ]
    if _comparable:
        _disagree = sum(
            1 for r in _comparable
            if ("1" if r["logit_pos"] > r["logit_neg"] else "0") != r["pred"]
        )
        _rate = _disagree / len(_comparable)
        print(
            f"[SPRINT EVAL] colon self-consistency: logit-argmax vs text-parse "
            f"disagree {_disagree}/{len(_comparable)} ({_rate*100:.2f}%)  "
            f"[logits available: {_logit_avail}/{total}]"
        )
    else:
        print(
            f"[SPRINT EVAL] colon self-consistency: not computed "
            f"(logits available: {_logit_avail}/{total}; "
            f"parseable text preds: "
            f"{sum(1 for r in results if r.get('pred') in ('0', '1'))}/{total}) "
            f"— accuracy uses "
            f"{'logit-argmax' if _logit_avail else 'text-parse fallback'}"
        )

    correct = sum(p == g for p, g in zip(preds, gts))
    accuracy = correct / total

    tp = sum(p == "1" and g == "1" for p, g in zip(preds, gts))
    fp = sum(p == "1" and g == "0" for p, g in zip(preds, gts))
    tn = sum(p == "0" and g == "0" for p, g in zip(preds, gts))
    fn = sum(p == "0" and g == "1" for p, g in zip(preds, gts))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_pos = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2

    metrics = {
        "accuracy":            accuracy,
        "accuracy_aacc":       accuracy,   # binary: per-class-avg accuracy == accuracy
        "accuracy_exactmatch": accuracy,   # binary: exact-match == accuracy
        "macro_f1":    macro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total":       total,
    }

    if auc_token_id is not None:
        valid_pairs = [
            (r["logit_pos"], int(r["gt"]))
            for r in results
            if r.get("logit_pos") is not None and r["gt"] in ("0", "1")
        ]
        if len(valid_pairs) > 10:
            scores_list = [p[0] for p in valid_pairs]
            labels_list = [p[1] for p in valid_pairs]
            auc = _auc_0p5_fallback(scores_list, labels_list)
            ap_pos = compute_ap(scores_list, labels_list)
            ap_neg = compute_ap([-s for s in scores_list],
                                [1 - l for l in labels_list])
            metrics["auc"]       = auc
            metrics["macro_auc"] = auc                 # binary: macro_AUC == binary AUC
            metrics["map"]       = (ap_pos + ap_neg) / 2
            metrics["ap_pos"]    = ap_pos

    return metrics


def compute_multilabel_metrics(results: list, valid_labels: list,
                               tie_header: str = "INFERENCE ") -> dict:
    """Exact-match accuracy, aACC, macro F1, and (when per-class scores are present)
    per-class AUC + mAP for multi-label tasks (chest/endo). Canonical implementation
    called by both inference and training validation."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    n_labels = len(valid_labels)
    has_scores = bool(results) and all("scores" in r for r in results)

    if has_scores:
        y_true        = [r["gt_binary"]   for r in results]
        y_pred        = [r["pred_binary"] for r in results]
        scores_matrix = [r["scores"]      for r in results]
    else:
        y_true = []
        y_pred = []
        for r in results:
            gt_set   = set(r.get("gt_parsed",   []))
            pred_set = set(r.get("pred_parsed", []))
            y_true.append([1 if lbl in gt_set   else 0 for lbl in valid_labels])
            y_pred.append([1 if lbl in pred_set else 0 for lbl in valid_labels])
        scores_matrix = None

    exact_match = sum(gt == pd for gt, pd in zip(y_true, y_pred)) / total

    per_label = {}
    f1_sum  = 0.0
    acc_sum = 0.0
    auc_sum = 0.0
    ap_sum  = 0.0
    n_auc   = 0
    n_ap    = 0
    for j, lbl in enumerate(valid_labels):
        tp = sum(y_true[i][j] == 1 and y_pred[i][j] == 1 for i in range(total))
        fp = sum(y_true[i][j] == 0 and y_pred[i][j] == 1 for i in range(total))
        tn = sum(y_true[i][j] == 0 and y_pred[i][j] == 0 for i in range(total))
        fn = sum(y_true[i][j] == 1 and y_pred[i][j] == 0 for i in range(total))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc_j = (tp + tn) / total

        entry = {
            "f1": round(f1, 4),
            "accuracy": round(acc_j, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

        if scores_matrix is not None:
            col_scores = [scores_matrix[i][j] for i in range(total)]
            col_labels = [y_true[i][j] for i in range(total)]
            n_pos = sum(col_labels)
            if 0 < n_pos < total:
                auc_j = _auc_0p5_fallback(col_scores, col_labels)
                ap_j  = compute_ap(col_scores, col_labels)
                entry["auc"] = round(auc_j, 4)
                entry["ap"]  = round(ap_j, 4)
                auc_sum += auc_j
                ap_sum  += ap_j
                n_auc += 1
                n_ap  += 1
            else:
                entry["auc"] = None
                entry["ap"]  = None

        per_label[lbl] = entry
        f1_sum += f1
        acc_sum += acc_j

    macro_f1 = f1_sum / n_labels if n_labels > 0 else 0.0
    aacc = acc_sum / n_labels if n_labels > 0 else 0.0

    out = {
        "accuracy":            aacc,
        "accuracy_aacc":       aacc,
        "accuracy_exactmatch": exact_match,
        "macro_f1":  macro_f1,
        "per_label": per_label,
        "total":     total,
    }

    if scores_matrix is not None and n_auc > 0:
        out["macro_auc"] = auc_sum / n_auc
        out["map"]       = ap_sum  / n_ap

    return out
