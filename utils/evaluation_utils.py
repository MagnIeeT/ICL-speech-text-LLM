import logging
import re
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
    if dataset_type in [DatasetType.VOXCELEB, DatasetType.MELD_EMOTION]:
        text = re.split(r"[,;|]", text)[0].strip()

    return text


def _evaluate_single_label(predictions: List[Dict[str, Any]], valid_labels: List[str]) -> Dict[str, Any]:
    true_labels = [str(p.get("true_label", "")).strip().lower() for p in predictions]
    pred_labels = [str(p.get("predicted_label", "")).strip().lower() for p in predictions]

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

    return {
        "accuracy": float(accuracy),
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
        "invalid_samples": int(invalid_samples),
        "total_samples": len(predictions),
        "valid_samples": int(len(y_true)),
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

        if not config.is_multi_label:
            return _evaluate_single_label(normalized_rows, valid_labels)

        if config.is_multi_label:
            if dataset_type == DatasetType.VOXPOPULI and "none" not in valid_labels:
                valid_labels = valid_labels + ["none"]
            return _evaluate_multi_label(normalized_rows, valid_labels)

        return {"accuracy": 0.0, "error": f"Unsupported dataset configuration for: {dataset_type}"}

    except Exception as exc:
        logger.error("Error in evaluate_predictions: %s", exc)
        return {"error": str(exc), "accuracy": 0.0}
