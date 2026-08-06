import logging
import re
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config.data_config.master_config import DatasetType, get_dataset_config

logger = logging.getLogger(__name__)


def _to_label_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,;|]", value)
        return [p.strip().lower() for p in parts if p.strip()]
    return [str(value).strip().lower()]


def clean_prediction(prediction: str, dataset_type: DatasetType = None) -> str:
    """Normalize raw model output to label text."""
    if prediction is None:
        return ""

    text = str(prediction).strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Remove common generation prefixes.
    for prefix in ["output:", "label:", "answer:", "prediction:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Keep only first answer segment for single-label tasks.
    if dataset_type in [DatasetType.VOXCELEB, DatasetType.MELD_EMOTION, DatasetType.RAVDESS, DatasetType.ESD, DatasetType.CREMAD, DatasetType.RAVDESS_SONG]:
        text = re.split(r"[,;|]", text)[0].strip()

    return text


def _canonical(s: Any) -> str:
    """Lowercase and collapse whitespace/hyphens to underscores so trivial
    separator differences ('ifsc code' vs 'ifsc_code') aren't counted invalid."""
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def _evaluate_single_label(predictions: List[Dict[str, Any]], valid_labels: List[str], label_map: Dict[str, str] = None) -> Dict[str, Any]:
    # Canonicalize separators + apply optional dataset aliases (e.g. spelling variants)
    # so only genuine label mismatches are marked invalid.
    label_map = {_canonical(k): _canonical(v) for k, v in (label_map or {}).items()}
    valid_labels = [_canonical(v) for v in valid_labels]
    true_labels = [_canonical(p.get("true_label", "")) for p in predictions]
    pred_labels = []
    for p in predictions:
        pc = _canonical(p.get("predicted_label", ""))
        pred_labels.append(label_map.get(pc, pc))

    true_filtered = []
    pred_filtered = []
    pred_with_invalid = []
    skipped_true_labels = []

    for gt, pd in zip(true_labels, pred_labels):
        if gt not in valid_labels:
            skipped_true_labels.append(gt)
            continue
        true_filtered.append(gt)
        pred_with_invalid.append(pd if pd in valid_labels else "invalid")
        if pd in valid_labels:
            pred_filtered.append(pd)
        else:
            pred_filtered.append(None)

    if skipped_true_labels:
        from collections import Counter
        counts = Counter(skipped_true_labels)
        logger.warning(
            "Skipped %d sample(s) with out-of-vocab true labels (excluded from both metrics): %s",
            len(skipped_true_labels), dict(counts),
        )

    if not true_filtered:
        return {
            "accuracy": 0.0,
            "accuracy_with_invalid": 0.0,
            "macro_f1": 0.0,
            "macro_f1_with_invalid": 0.0,
            "invalid_predictions": 0,
            "total_samples": len(predictions),
            "valid_samples": 0,
        }

    keep_idx = [i for i, v in enumerate(pred_filtered) if v is not None]
    if keep_idx:
        true_valid = [true_filtered[i] for i in keep_idx]
        pred_valid = [pred_filtered[i] for i in keep_idx]
        accuracy = accuracy_score(true_valid, pred_valid)
        macro_f1 = f1_score(true_valid, pred_valid, average="macro", labels=valid_labels, zero_division=0)
    else:
        accuracy = 0.0
        macro_f1 = 0.0

    macro_f1_with_invalid = f1_score(
        true_filtered,
        pred_with_invalid,
        average="macro",
        labels=valid_labels,
        zero_division=0,
    )

    invalid_predictions = sum(1 for x in pred_with_invalid if x == "invalid")

    # True accuracy over all valid-true-label samples, counting invalid predictions
    # as wrong (denominator = true_filtered, mirroring macro_f1_with_invalid's universe).
    correct_with_invalid = sum(1 for t, p in zip(true_filtered, pred_with_invalid) if t == p)
    accuracy_with_invalid = correct_with_invalid / len(true_filtered)

    return {
        "accuracy": float(accuracy),
        "accuracy_with_invalid": float(accuracy_with_invalid),
        "macro_f1": float(macro_f1),
        "macro_f1_with_invalid": float(macro_f1_with_invalid),
        "invalid_predictions": int(invalid_predictions),
        "total_samples": len(predictions),
        "valid_samples": len(true_filtered),
    }


def _evaluate_multi_label(predictions: List[Dict[str, Any]], valid_labels: List[str]) -> Dict[str, Any]:
    y_true = []
    y_pred = []
    invalid_samples = 0

    for row in predictions:
        gt_labels = _to_label_list(row.get("true_label", ""))
        pd_labels = _to_label_list(row.get("predicted_label", ""))

        gt_vec = np.array([1 if label in gt_labels else 0 for label in valid_labels])
        pd_vec = np.array([1 if label in pd_labels else 0 for label in valid_labels])

        if gt_vec.sum() == 0:
            continue
        if pd_vec.sum() == 0:
            invalid_samples += 1

        y_true.append(gt_vec)
        y_pred.append(pd_vec)

    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_f1_with_invalid": 0.0,
            "invalid_samples": invalid_samples,
            "total_samples": len(predictions),
            "valid_samples": 0,
        }

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))

    return {
        "accuracy": exact_match,
        "macro_f1": float(macro_f1),
        "macro_f1_with_invalid": float(macro_f1),
        "invalid_samples": int(invalid_samples),
        "total_samples": len(predictions),
        "valid_samples": int(len(y_true)),
    }


def _normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop articles/punctuation, collapse whitespace."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _evaluate_qa(predictions: List[Dict[str, Any]], max_answer_words: int = 2) -> Dict[str, Any]:
    """Free-form extractive-QA metrics on the RAW generated text:
      exact_match, token_f1 (max over gold answers), and format_compliance
      (predicted answer is 1..max_answer_words tokens — the instruction-following signal)."""
    em, f1s, comp = [], [], []
    for item in predictions:
        pred = str(item.get("raw_prediction") or item.get("predicted_label") or "")
        golds = item.get("true_label")
        golds = golds if isinstance(golds, list) else [golds]
        golds = [g for g in golds if g is not None] or [""]
        np_ = _normalize_answer(pred)
        em.append(max(int(np_ == _normalize_answer(g)) for g in golds))
        f1s.append(max(_token_f1(np_, _normalize_answer(g)) for g in golds))
        comp.append(int(0 < len(pred.split()) <= max_answer_words))
    return {
        "exact_match": float(np.mean(em)) if em else 0.0,
        "token_f1": float(np.mean(f1s)) if f1s else 0.0,
        "format_compliance": float(np.mean(comp)) if comp else 0.0,
        # aliases so the pipeline's primary-metric/aggregation plumbing still finds a value
        "macro_f1_with_invalid": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy_with_invalid": float(np.mean(em)) if em else 0.0,
        "total_samples": len(predictions),
    }


def evaluate_predictions(predictions: List[Dict[str, Any]], dataset_type: DatasetType) -> Dict[str, Any]:
    """Evaluate predictions for active dataset types: voxceleb, hvb, voxpopuli, meld_emotion."""
    if not predictions:
        return {"error": "Empty predictions list", "accuracy": 0.0}

    try:
        config = get_dataset_config(dataset_type)
        valid_labels = [str(v).strip().lower() for v in (config.valid_labels or [])]

        normalized_rows = []
        for item in predictions:
            normalized_rows.append(
                {
                    **item,
                    "predicted_label": clean_prediction(item.get("predicted_label", ""), dataset_type),
                }
            )

        if getattr(config, "task_type", "classification") == "qa":
            return _evaluate_qa(normalized_rows)

        if not config.is_multi_label:
            return _evaluate_single_label(normalized_rows, valid_labels, config.label_mapping)

        if config.is_multi_label:
            if dataset_type == DatasetType.VOXPOPULI and "none" not in valid_labels:
                valid_labels = valid_labels + ["none"]
            return _evaluate_multi_label(normalized_rows, valid_labels)

        return {"accuracy": 0.0, "error": f"Unsupported dataset configuration for: {dataset_type}"}

    except Exception as exc:
        logger.error("Error in evaluate_predictions: %s", exc)
        return {"error": str(exc), "accuracy": 0.0}
