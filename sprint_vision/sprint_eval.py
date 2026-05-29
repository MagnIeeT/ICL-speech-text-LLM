"""
SPRInT VLM Inference + Evaluation script.

Handles binary tasks (colon) and multi-label tasks (chest, endo).

Multi-label evaluation modes
----------------------------
For chest/endo with --icl-shots 0, the script runs one binary yes/no query
PER CLASS (19 for chest, 4 for endo) and extracts the softmax probability over
the {0, 1} answer tokens at the first generation step. This produces calibrated
per-class probability scores → enables AUC and mAP (matches MedFMC's reported
metrics). For --icl-shots > 0 it falls back to the original single-prompt text
output (accuracy + macro F1 only).

Usage examples
--------------
# Zero-shot, base model, colon (binary; AUC via logit for token "1"):
python sprint_eval.py \
    --model-path /home/harinis/.cache/huggingface/hub/llava-v1.5-13b \
    --image-folder /home/harinis/MedFM/data/MedFMC \
    --question-file sprint_vision/data/colon_test.json \
    --dataset colon --strategy regular

# Zero-shot, base model, chest (per-class binary → AUC + mAP):
python sprint_eval.py \
    --model-path /home/harinis/.cache/huggingface/hub/llava-v1.5-13b \
    --image-folder /home/harinis/MedFM/data/MedFMC \
    --question-file sprint_vision/data/chest_test.json \
    --dataset chest --strategy regular --icl-shots 0

# 5-shot, fine-tuned, chest (text-output mode; no AUC/mAP):
python sprint_eval.py \
    --model-base /home/harinis/.cache/huggingface/hub/llava-v1.5-13b \
    --model-path sprint_vision/checkpoints/llava-chest-regular-percent100 \
    --image-folder /home/harinis/MedFM/data/MedFMC \
    --question-file sprint_vision/data/chest_test.json \
    --train-file sprint_vision/data/chest_train_percent100.json \
    --dataset chest --strategy regular --icl-shots 5

Output
------
Results JSON → {LLAVA_DIR}/logs/json/results_{dataset}_{strategy}_{shots}shot_{timestamp}.json
"""

import argparse
import datetime
import json
import os
import random
import sys

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataload.example_selector import ExampleSelector
from dataload.prompt_builder import LLaVAPromptBuilder
from models.symbolAdapter.symbol_manager import SymbolManager
from config.data_config import get_dataset_config, DatasetName

VALID_DATASETS = [e.value for e in DatasetName]


# ─────────────────────────────────────────────────────────────────────────────
# Prediction parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_binary_pred(text: str) -> str:
    """
    Extract '0' or '1' from decoded model output.
    Returns the raw (truncated) text if no digit is found.
    """
    cleaned = text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()
    for tok in cleaned.split():
        t = tok.strip(".,;:()")
        if t in ("0", "1"):
            return t
    if cleaned and cleaned[0] in ("0", "1"):
        return cleaned[0]
    return cleaned[:20]


def _parse_multi_label_pred(text: str, valid_labels: list) -> list:
    """
    Extract a list of present disease names from decoded model output.

    Accepts:
      - "effusion, nodule"
      - "none" / "no finding" / "" → empty list
      - partial/fuzzy matches against valid_labels

    Returns:
      List of matched label strings (lowercase), e.g. ["effusion", "nodule"].
    """
    cleaned = text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()
    cleaned = cleaned.lower().strip().rstrip(".")

    if cleaned in ("none", "no finding", "no findings", "normal", ""):
        return []

    parts = [p.strip() for p in cleaned.replace(";", ",").split(",") if p.strip()]

    found = []
    for part in parts:
        if part in valid_labels:
            if part not in found:
                found.append(part)
        else:
            for vl in valid_labels:
                if vl in part or part in vl:
                    if vl not in found:
                        found.append(vl)
                    break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# AUC helper (binary tasks only)
# ─────────────────────────────────────────────────────────────────────────────

def _get_binary_token_ids(tokenizer, symbol_manager, strategy: str, mode: str = "digit"):
    """
    Vocabulary indices for the negative- and positive-class answer tokens.

    mode:
      "digit" — bare "0" and "1" tokens (colon binary AUC path).
      "yesno" — bare "No" and "Yes" tokens. Use for chest/endo per-class
                binary path — diagnostic confirmed the model emits Yes/No
                at the answer position, not 0/1, so 0/1 logits are noise.

    For strategy == "two_token", `mode` is ignored — we always use the
    symbol tokens currently mapped to "0" and "1" by SymbolManager.

    Skips the SentencePiece leading-space token (id 29871) which LLaMA
    prepends to standalone words — using it gives a biased score.

    Returns (token_id_neg, token_id_pos).
    """
    LLAMA_SPACE_TOKEN = 29871

    def _first_non_space(text: str):
        ids = tokenizer.encode(text, add_special_tokens=False)
        for tid in ids:
            if tid != LLAMA_SPACE_TOKEN:
                return tid
        return ids[0] if ids else None

    if strategy in ("two_token",):
        symbols = symbol_manager.get_current_symbols()
        sym0 = symbols.get("0", "0")
        sym1 = symbols.get("1", "1")
        return _first_non_space(sym0), _first_non_space(sym1)

    if mode == "yesno":
        return _first_non_space("No"), _first_non_space("Yes")
    return _first_non_space("0"), _first_non_space("1")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics: binary
# ─────────────────────────────────────────────────────────────────────────────

def _compute_auc(scores: list, labels: list) -> float:
    """Trapezoidal AUC — no sklearn required."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    tpr_prev = fpr_prev = 0.0
    auc = 0.0
    for _, lbl in pairs:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr
    return auc


def _compute_average_precision(scores: list, labels: list) -> float:
    """
    Average precision (matches sklearn's average_precision_score).
    AP = sum over positive samples of precision_at_rank / n_positives.
    """
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    ap = 0.0
    for rank, (_, lbl) in enumerate(pairs, start=1):
        if lbl == 1:
            tp += 1
            ap += tp / rank
    return ap / n_pos


def _build_per_class_prompt(dataset: str, class_name: str, sym_mappings=None) -> str:
    """
    Yes/no prompt for a single class.

    For regular strategy: asks "Answer Yes or No." — the diagnostic confirmed
    LLaVA naturally emits the Yes/No tokens (logits ~+18) at the answer
    position, while bare '0'/'1' tokens are buried (~+7). Asking for Yes/No
    directly aligns the prompt with the tokens we score against.

    For two_token strategy: substitutes the trained symbols for yes/no.
    """
    hint = {"chest": "chest X-ray", "endo": "endoscopy image"}.get(
        dataset, "medical image"
    )
    nat = class_name.replace("_", " ")
    if sym_mappings and "0" in sym_mappings and "1" in sym_mappings:
        no_sym = sym_mappings["0"]
        yes_sym = sym_mappings["1"]
        return (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            f"Does this {hint} show {nat}? "
            f"Answer with {yes_sym} for yes or {no_sym} for no."
        )
    return (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        f"Does this {hint} show {nat}? Answer Yes or No."
    )


def _compute_binary_metrics(results: list, auc_token_id=None) -> dict:
    """Accuracy, AUC, macro F1, sensitivity, specificity for binary tasks."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    preds = [r["pred"] for r in results]
    gts   = [r["gt"]   for r in results]

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
        "accuracy":    accuracy,
        "macro_f1":    macro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total":       total,
    }

    # AUC from logits
    if auc_token_id is not None:
        valid_pairs = [
            (r["logit_pos"], int(r["gt"]))
            for r in results
            if r.get("logit_pos") is not None and r["gt"] in ("0", "1")
        ]
        if len(valid_pairs) > 10:
            scores_list = [p[0] for p in valid_pairs]
            labels_list = [p[1] for p in valid_pairs]
            metrics["auc"] = _compute_auc(scores_list, labels_list)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics: multi-label
# ─────────────────────────────────────────────────────────────────────────────

def _compute_multi_label_metrics(results: list, valid_labels: list) -> dict:
    """
    Exact-match accuracy, macro F1, and (when per-class scores are present)
    per-class AUC and mAP for multi-label classification.

    Detects score mode via the "scores" field on result records. If present
    (chest/endo per-class binary path), per-class AUC and AP are computed
    using the continuous probability scores; macro_auc and mAP are added.
    Otherwise falls back to set-based exact match + macro F1 only.
    """
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
                auc_j = _compute_auc(col_scores, col_labels)
                ap_j  = _compute_average_precision(col_scores, col_labels)
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

    macro_f1 = f1_sum / n_labels if n_labels > 0 else 0.0

    out = {
        "accuracy":  exact_match,   # exact match (all labels must be correct)
        "macro_f1":  macro_f1,
        "per_label": per_label,
        "total":     total,
    }

    if scores_matrix is not None and n_auc > 0:
        out["macro_auc"] = auc_sum / n_auc
        out["map"]       = ap_sum  / n_ap

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-class binary inference (chest / endo → enables AUC and mAP)
# ─────────────────────────────────────────────────────────────────────────────

def _run_per_class_binary(
    sample, dataset, ground_truth, task_labels, valid_labels_lower,
    sym_mappings, token_id_0, token_id_1,
    tokenizer, model, image_processor, image_folder, sample_index,
    diagnose=False,
):
    """
    For each class, ask the model a yes/no question, capture the logits for
    the 0 and 1 answer tokens at the first generation step, and softmax over
    them to produce a calibrated P(class present) score.

    Score interpretation: P("1") / (P("0") + P("1")) — a true probability in
    [0,1] suitable for AUC and mAP.

    Returns a result dict containing `scores` (list[float] of length K),
    `pred_binary` (list[int] of length K), and parsed GT.

    If `diagnose=True`, prints a per-class table showing:
      - the model's actual top-1 token at step 0 (what it WANTED to say)
      - top-5 candidates with their logits
      - the logits for our '0' and '1' tokens
      - the extracted P(1)
    This is a sanity check to confirm the model emits "0"/"1" tokens (not
    "Yes"/"No" or a leading space) — otherwise our softmax is misranking.
    """
    image_file = sample["image"]
    full_path = os.path.join(image_folder, image_file)
    if not os.path.exists(full_path):
        raise FileNotFoundError(full_path)

    pil_image = Image.open(full_path).convert("RGB")
    image_sizes = [pil_image.size]
    image_tensors = process_images([pil_image], image_processor, model.config)
    if isinstance(image_tensors, list):
        image_tensor = torch.stack(image_tensors, dim=0).half().cuda()
    else:
        image_tensor = image_tensors.half().cuda()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

    softmax = torch.nn.functional.softmax
    class_scores = []
    class_preds  = []
    diag_rows    = [] if diagnose else None

    for cls_name in task_labels:
        prompt_text = _build_per_class_prompt(dataset, cls_name, sym_mappings)
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            gen_out = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        step0_logits = gen_out.scores[0][0]  # [vocab]
        l0 = step0_logits[token_id_0].item()
        l1 = step0_logits[token_id_1].item()
        pair = torch.stack([step0_logits[token_id_0], step0_logits[token_id_1]])
        probs = softmax(pair, dim=0)
        score_1 = probs[1].item()

        class_scores.append(score_1)
        class_preds.append(1 if score_1 >= 0.5 else 0)

        if diagnose:
            top5_vals, top5_idx = torch.topk(step0_logits, 5)
            top5_pairs = [
                (int(idx.item()), tokenizer.decode([int(idx.item())]), float(v.item()))
                for v, idx in zip(top5_vals, top5_idx)
            ]
            diag_rows.append({
                "cls":    cls_name,
                "logit0": l0,
                "logit1": l1,
                "p1":     score_1,
                "top5":   top5_pairs,
            })

    if diagnose and diag_rows:
        print(f"\n--- DIAGNOSTIC sample {sample_index}  image={image_file} ---")
        print(f"GT: {ground_truth}")
        print(f"token_id_0={token_id_0}  token_id_1={token_id_1}  "
              f"('{tokenizer.decode([token_id_0])}', '{tokenizer.decode([token_id_1])}')")
        print(f"  {'class':<24} {'logit0':>7} {'logit1':>7} {'P(1)':>6}  top-5 tokens (id, repr, logit)")
        for r in diag_rows:
            top5_str = " | ".join(
                f"{tid}={tok!r}({lg:+.2f})" for tid, tok, lg in r["top5"]
            )
            print(
                f"  {r['cls']:<24} {r['logit0']:>+7.2f} {r['logit1']:>+7.2f} "
                f"{r['p1']:>6.3f}  {top5_str}"
            )
        print()

    gt_set = set(
        p.strip().lower()
        for p in ground_truth.split(",")
        if p.strip().lower() not in ("", "none")
    )
    gt_binary  = [1 if lbl in gt_set else 0 for lbl in valid_labels_lower]
    pred_parsed = [
        valid_labels_lower[j] for j in range(len(valid_labels_lower)) if class_preds[j] == 1
    ]
    pred_str = ", ".join(pred_parsed) if pred_parsed else "none"

    return {
        "id":          sample.get("id", f"sample_{sample_index}"),
        "image":       image_file,
        "scores":      class_scores,
        "pred_binary": class_preds,
        "pred_parsed": pred_parsed,
        "pred":        pred_str,
        "gt":          ground_truth,
        "gt_parsed":   sorted(gt_set),
        "gt_binary":   gt_binary,
        "correct":     class_preds == gt_binary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def eval_model(args):
    dataset = args.dataset
    cfg = get_dataset_config(dataset)

    is_multi_label = cfg.is_multi_label
    task_labels    = cfg.label_names
    valid_labels_lower = [lbl.lower() for lbl in task_labels]

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading model: {model_name} ...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=model_name,
    )

    # ── Symbol manager ────────────────────────────────────────────────────────
    is_regular = args.strategy in ("regular", "rft")
    symbol_manager = SymbolManager(
        original_labels=task_labels,
        tokenizer=tokenizer,
        dynamic_per_epoch=False,
        symbol_type=args.strategy,
        no_symbols=is_regular,
    )

    if not is_regular:
        mapping_path = args.symbol_mappings or os.path.join(model_path, "symbol_mappings.json")
        if os.path.exists(mapping_path):
            symbol_manager.load_mappings(mapping_path)
            print(f"Loaded symbol mappings: {symbol_manager.get_current_symbols()}")
        else:
            print("WARNING: No symbol_mappings.json found. Auto-generating (pipeline test only).")
            auto_path = os.path.join(model_path, "symbol_mappings_autogen.json")
            symbol_manager.save_mappings(auto_path)
            print(f"Auto-generated symbols saved to: {auto_path}")

    sym_mappings = symbol_manager.get_current_symbols() if not is_regular else None

    # ── Answer-token ids ──────────────────────────────────────────────────────
    # Colon binary path: bare "0"/"1" tokens (or symbols under two_token).
    # Multi-label per-class path: bare "No"/"Yes" tokens — diagnostic confirmed
    # the base LLaVA model emits Yes/No at the answer position, not 0/1.
    use_per_class = is_multi_label and args.icl_shots == 0

    if not is_multi_label:
        token_id_0, token_id_1 = _get_binary_token_ids(
            tokenizer, symbol_manager, args.strategy, mode="digit"
        )
        auc_token_id = token_id_1
    elif use_per_class:
        token_id_0, token_id_1 = _get_binary_token_ids(
            tokenizer, symbol_manager, args.strategy, mode="yesno"
        )
        auc_token_id = None
    else:
        token_id_0, token_id_1 = None, None
        auc_token_id = None

    if use_per_class:
        print(
            f"Multi-label per-class mode: {len(task_labels)} binary queries per image "
            f"→ AUC + mAP will be computed."
        )
        print(
            f"  Answer tokens: neg={token_id_0} '{tokenizer.decode([token_id_0])}'  "
            f"pos={token_id_1} '{tokenizer.decode([token_id_1])}'"
        )
    elif is_multi_label:
        print(
            f"Multi-label text-output mode (icl_shots={args.icl_shots}): "
            f"exact-match + macro F1 only (no AUC/mAP)."
        )

    # ── Load test data ────────────────────────────────────────────────────────
    question_file = os.path.expanduser(args.question_file)
    questions = json.load(open(question_file))
    if args.num_samples > 0:
        questions = questions[: args.num_samples]
    print(f"Loaded {len(questions)} test samples  [{dataset}]")

    # ── ICL setup ─────────────────────────────────────────────────────────────
    example_selector = None
    if args.icl_shots > 0:
        if not args.train_file:
            raise ValueError("--train-file is required when --icl-shots > 0")
        example_selector = ExampleSelector(
            os.path.expanduser(args.train_file), seed=args.seed
        )
        print(f"ICL: {args.icl_shots}-shot from {args.train_file}")

    prompt_builder = LLaVAPromptBuilder()

    # ── Inference loop ────────────────────────────────────────────────────────
    results = []
    desc = f"{dataset} | {args.strategy} | {args.icl_shots}-shot"

    for i, sample in enumerate(tqdm(questions, desc=desc)):
        image_file = sample.get("image", "")
        if not image_file:
            continue

        ground_truth = str(sample["conversations"][1]["value"]).strip()
        raw_instruction = sample["conversations"][0]["value"]
        instruction = raw_instruction.replace(DEFAULT_IMAGE_TOKEN, "").strip()

        try:
            if use_per_class:
                result = _run_per_class_binary(
                    sample=sample,
                    dataset=dataset,
                    ground_truth=ground_truth,
                    task_labels=task_labels,
                    valid_labels_lower=valid_labels_lower,
                    sym_mappings=sym_mappings,
                    token_id_0=token_id_0,
                    token_id_1=token_id_1,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    image_folder=args.image_folder,
                    sample_index=i,
                    diagnose=(i < args.diagnose_samples),
                )
                results.append(result)
                continue

            # Build prompt + image path list
            examples = example_selector.select(args.icl_shots) if example_selector else []
            prompt_str, image_paths = prompt_builder.build(
                instruction=instruction,
                examples=examples,
                test_image_path=image_file,
                symbol_mappings=sym_mappings,
            )

            # Load images
            pil_images = []
            for img_path in image_paths:
                full_path = os.path.join(args.image_folder, img_path)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(full_path)
                pil_images.append(Image.open(full_path).convert("RGB"))

            image_sizes = [img.size for img in pil_images]
            image_tensors = process_images(pil_images, image_processor, model.config)

            if isinstance(image_tensors, list):
                image_tensor = torch.stack(image_tensors, dim=0).half().cuda()
            else:
                image_tensor = image_tensors.half().cuda()
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)

            # Build conversation and tokenize
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], prompt_str)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            # Generate
            with torch.inference_mode():
                if not is_multi_label and auc_token_id is not None:
                    gen_out = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        do_sample=False,
                        max_new_tokens=10,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    output_ids  = gen_out.sequences
                    logit_pos   = gen_out.scores[0][0][auc_token_id].item()
                else:
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        do_sample=False,
                        max_new_tokens=30,
                        use_cache=True,
                    )
                    logit_pos = None

            raw_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # Decode: symbols → original labels
            decoded_text = symbol_manager.convert_symbols_back(raw_text)

            # Build result record
            if is_multi_label:
                pred_parsed = _parse_multi_label_pred(decoded_text, valid_labels_lower)
                gt_parsed   = [
                    p.strip().lower()
                    for p in ground_truth.split(",")
                    if p.strip().lower() not in ("", "none")
                ]
                pred_str = ", ".join(pred_parsed) if pred_parsed else "none"
                result = {
                    "id":           sample.get("id", f"sample_{i}"),
                    "image":        image_file,
                    "raw_output":   raw_text,
                    "decoded":      decoded_text,
                    "pred":         pred_str,
                    "pred_parsed":  pred_parsed,
                    "gt":           ground_truth,
                    "gt_parsed":    gt_parsed,
                    "correct":      set(pred_parsed) == set(gt_parsed),
                }
            else:
                pred = _parse_binary_pred(decoded_text)
                result = {
                    "id":         sample.get("id", f"sample_{i}"),
                    "image":      image_file,
                    "raw_output": raw_text,
                    "decoded":    decoded_text,
                    "pred":       pred,
                    "gt":         ground_truth,
                    "correct":    pred == ground_truth,
                    "logit_pos":  logit_pos,
                }

            results.append(result)

        except FileNotFoundError as e:
            print(f"WARNING: image not found ({e}) — skipping sample {sample.get('id')}")
        except Exception as e:
            print(f"WARNING: error on sample {sample.get('id')}: {e}")
            if "CUDA" in str(e) or "cuDNN" in str(e) or "cudnn" in str(e):
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # ── Compute metrics ───────────────────────────────────────────────────────
    if is_multi_label:
        metrics = _compute_multi_label_metrics(results, valid_labels_lower)
    else:
        metrics = _compute_binary_metrics(results, auc_token_id)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS  dataset={dataset}  strategy={args.strategy}  shots={args.icl_shots}")
    print(f"Samples evaluated : {len(results)}")
    print(f"Accuracy          : {metrics.get('accuracy', 0) * 100:.2f}%"
          f"  {'(exact match)' if is_multi_label else ''}")
    if "auc" in metrics:
        print(f"AUC               : {metrics['auc'] * 100:.2f}%")
    if "macro_auc" in metrics:
        print(f"Macro AUC         : {metrics['macro_auc'] * 100:.2f}%")
    if "map" in metrics:
        print(f"mAP               : {metrics['map'] * 100:.2f}%")
    print(f"Macro F1          : {metrics.get('macro_f1', 0) * 100:.2f}%")
    if not is_multi_label:
        print(f"Sensitivity       : {metrics.get('sensitivity', 0) * 100:.2f}%")
        print(f"Specificity       : {metrics.get('specificity', 0) * 100:.2f}%")
    if is_multi_label and "per_label" in metrics:
        has_auc = any(v.get("auc") is not None for v in metrics["per_label"].values())
        if has_auc:
            print(f"{'Per-label':<24} {'F1':>8} {'AUC':>8} {'AP':>8}")
            for lbl, m in metrics["per_label"].items():
                auc_s = f"{m['auc']*100:.2f}%" if m.get("auc") is not None else "  n/a"
                ap_s  = f"{m['ap']*100:.2f}%"  if m.get("ap")  is not None else "  n/a"
                print(f"  {lbl:<22} {m['f1']*100:>7.2f}% {auc_s:>8} {ap_s:>8}")
        else:
            print("Per-label F1:")
            for lbl, m in metrics["per_label"].items():
                print(f"  {lbl:<20}: {m['f1'] * 100:.2f}%")
    print("=" * 60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    now = datetime.datetime.now()
    out_filename = (
        f"results_{dataset}_{args.strategy}_{args.icl_shots}shot"
        f"_{now.strftime('%Y%m%d_%H%M%S')}.json"
    )

    llava_dir = os.environ.get(
        "LLAVA_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    out_dir = os.path.join(llava_dir, "logs", "json")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)

    output_data = {
        "metadata": {
            "dataset":       dataset,
            "strategy":      args.strategy,
            "icl_shots":     args.icl_shots,
            "model_path":    model_path,
            "model_base":    args.model_base,
            "question_file": args.question_file,
            "train_file":    args.train_file,
            "num_samples":   len(results),
            "timestamp":     now.isoformat(),
            "seed":          args.seed,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved → {out_path}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPRInT VLM Inference & Evaluation")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (required when loading a LoRA checkpoint).")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model or LoRA checkpoint directory.")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Root folder for MedFMC images (MedFMC root).")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to *_test.json generated by medfmc_to_llava.py.")
    parser.add_argument("--dataset", type=str, default="colon",
                        choices=VALID_DATASETS,
                        help="Dataset name — controls evaluation mode (binary vs multi-label).")
    parser.add_argument("--strategy", type=str, default="regular",
                        choices=["regular", "two_token"],
                        help="Inference strategy: 'regular' (0/1 or disease names) or "
                             "'two_token' (symbol → decode to original labels).")
    parser.add_argument("--symbol-mappings", type=str, default=None,
                        help="Path to symbol_mappings.json (two_token strategy). "
                             "Auto-detected from --model-path if omitted.")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of test samples to evaluate. 0 = all.")
    parser.add_argument("--icl-shots", type=int, default=0,
                        help="Number of in-context learning examples per query. 0 = zero-shot.")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Path to *_train_percent100.json (required when --icl-shots > 0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for ICL example selection.")
    parser.add_argument("--diagnose-samples", type=int, default=3,
                        help="For chest/endo per-class binary mode: print the "
                             "model's top-5 first tokens for this many samples, "
                             "so you can verify the model emits '0'/'1' (not "
                             "'Yes'/'No' or a leading space). Set to 0 to disable.")
    args = parser.parse_args()
    eval_model(args)
