import argparse
import torch
import os
import re
import json
import logging
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from models.symbolAdapter.symbol_manager import SymbolManager
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    from models.symbolAdapter.symbol_manager import SymbolManager


# ─────────────────────────────────────────────
# AUC helper — no sklearn required
# ─────────────────────────────────────────────
def _compute_auc(y_true, y_score):
    """
    Binary AUC via trapezoidal rule (sklearn-free).
    y_true : list of int  (0 or 1)
    y_score: list of float (model confidence for class 1)
    Returns AUC in [0, 1].
    """
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    # AUC undefined for single-class ground truth
    if n_pos == 0 or n_neg == 0:
        return float("nan")  
    tp = fp = 0
    auc = 0.0
    fp_prev = 0
    tp_prev = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp + tp_prev) / 2.0  # trapezoid
            tp_prev = tp
        fp_prev = fp
    auc += (tp + tp_prev) / 2.0  # final segment
    return auc / (n_pos * n_neg)


def eval_model(args):
    # ── 1. Model Loading ────────────────────────────────────────────────────
    torch.manual_seed(42)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"Loading model: {model_name}...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=model_name
    )

    # ── 2. Symbol Manager ───────────────────────────────────────────────────
    symbol_manager = SymbolManager(
        original_labels=["0", "1"],
        tokenizer=tokenizer,
        dynamic_per_epoch=False,
        symbol_type=args.strategy
    )

    if args.strategy != "regular":
        if args.symbol_mappings and os.path.exists(args.symbol_mappings):
            symbol_manager.load_mappings(args.symbol_mappings)
            print(f"Loaded symbols from: {args.symbol_mappings}")
            print(f"Synchronized Symbols: {symbol_manager.get_current_symbols()}")
        else:
            mapping_path = os.path.join(model_path, "symbol_mappings.json")
            if os.path.exists(mapping_path):
                symbol_manager.load_mappings(mapping_path)
                print(f"Synchronized Symbols: {symbol_manager.get_current_symbols()}")
            else:
                print(f"WARNING: symbol_mappings.json not found in {model_path}")
                print(f"Auto-generating fresh symbols (zero-shot pipeline test only).")
                auto_save = os.path.join(model_path, "symbol_mappings_autogen.json")
                symbol_manager.save_mappings(auto_save)
                print(f"Auto-generated Symbols: {symbol_manager.get_current_symbols()}")

    # ── 3. Pre-compute AUC token IDs (done once, outside the loop) ──────────
    # AUC needs a soft confidence score for class 1.
    # We extract the first-token logits from model.generate() and compute
    # P(class-1-token) / (P(class-0-token) + P(class-1-token)) per sample.
    # For regular: tokens are the literal characters "0" and "1".
    # For two_token: tokens are the first sub-token of each symbol.
    if args.strategy == "regular":
        # Use [-1] not [0]: Llama tokenizer prepends a space token (29871) to all strings.
        # encode("0") → [29871, 29900]; encode("1") → [29871, 29896].
        # [0] gives 29871 for BOTH classes (same token → AUC is always 0.5).
        # [-1] gives the actual digit token, which differs per class.
        auc_tok_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
        auc_tok_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
    else:
        symbols = symbol_manager.get_current_symbols()  # {"0": "kxzy", "1": "wxab"}
        sym_0 = symbols.get("0", "0")
        sym_1 = symbols.get("1", "1")
        auc_tok_0 = tokenizer.encode(sym_0, add_special_tokens=False)[-1]
        auc_tok_1 = tokenizer.encode(sym_1, add_special_tokens=False)[-1]
    print(f"AUC token IDs — class 0: {auc_tok_0} ('{tokenizer.decode([auc_tok_0])}')  "
          f"class 1: {auc_tok_1} ('{tokenizer.decode([auc_tok_1])}')")

    # ── 4. Load Dataset ──────────────────────────────────────────────────────
    data_path = os.path.expanduser(args.question_file)
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset file not found at {data_path}")
        return

    questions = json.load(open(data_path, "r"))
    print(f"Loaded {len(questions)} samples from {args.question_file}")

    if args.num_samples > 0:
        questions = questions[:args.num_samples]
        print(f"Limited to {args.num_samples} samples (--num-samples flag)")

    results = []
    correct_count = 0

    # ── 5. Inference Loop ────────────────────────────────────────────────────
    print(f"Starting MedFMC Inference (strategy: {args.strategy}, samples: {len(questions)})...")
    for line in tqdm(questions):
        ground_truth = str(line["conversations"][1]["value"]).strip()
        raw_prompt = line["conversations"][0]["value"]

        # ── FIX 1: Explicit image-prompt linking ─────────────────────────────
        # Always ensure <image> token is explicitly present and at the front,
        # regardless of input format changes. Prevents silent image ignoring.

        # 1. Extract pure instruction text (strip <image> if present)
        instruction = raw_prompt.replace(DEFAULT_IMAGE_TOKEN, "").lstrip("\n").strip()

        # 2. Modify instruction text (strategy-aware)
        if args.strategy == "regular":
            if "Answer with the label only" in instruction:
                instruction = instruction[:instruction.rfind("Answer with the label only")].rstrip()
            instruction += "\nAnswer with ONLY a single digit: 0 or 1. Nothing else. No explanation."
        # two_token: keep instruction exactly as-is (matches training distribution)

        # 3. Always re-attach <image> token explicitly at the front
        human_prompt = DEFAULT_IMAGE_TOKEN + "\n" + instruction

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], human_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ── FIX 2: List-based image loading (multi-image ready) ──────────────
        # Support both single "image" (string) and multi "images" (list) for ICL.
        # For now, handles single image; structured for few-shot in future.
        if "images" in line and line["images"]:
            image_files = line["images"]
        elif "image" in line:
            image_files = [line["image"]]
        else:
            raise KeyError(f"Sample {line.get('id', 'UNKNOWN')} missing both 'image' and 'images' keys")

        images_pil = []
        for img_f in image_files:
            img_full_path = os.path.join(args.image_folder, img_f)
            if not os.path.exists(img_full_path):
                print(f"Missing Image: {img_full_path}")
                raise FileNotFoundError(f"Image not found: {img_full_path}")
            images_pil.append(Image.open(img_full_path).convert('RGB'))

        # process_images accepts a list of N PIL images natively
        image_tensors = process_images(images_pil, image_processor, model.config)

        # Handle both stacked tensor (homogeneous shapes) and list (anyres heterogeneous)
        if isinstance(image_tensors, torch.Tensor):
            images_input = image_tensors.half().cuda()
        else:
            images_input = [t.half().cuda() for t in image_tensors]

        image_sizes = [img.size for img in images_pil]

        try:
            input_ids    = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # ── Generation with first-token scores for AUC ───────────────────
            with torch.inference_mode():
                gen_out = model.generate(
                    input_ids,
                    images=images_input,
                    image_sizes=image_sizes,
                    do_sample=(args.temperature > 0),  # use sampling if temperature > 0
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    max_new_tokens=15,
                    use_cache=True,
                    output_scores=True,           # capture per-step logits
                    return_dict_in_generate=True  # return as GenerateOutput dict
                )

            output_ids = gen_out.sequences

            # ── AUC score: extract confidence for class 1 ──
            # Use the first generated token's logits (most likely position)
            if len(gen_out.scores) > 0:
                digit_logits = gen_out.scores[0][0]   # shape: [vocab_size]
                probs = torch.softmax(digit_logits, dim=-1)
                p0 = probs[auc_tok_0].item()
                p1 = probs[auc_tok_1].item()
                # Normalized confidence for class 1
                auc_score = p1 / (p0 + p1) if (p0 + p1) > 1e-9 else 0.5
            else:
                auc_score = 0.5  # neutral confidence if no scores available

            # ── Decode prediction ─────────────────────────────────────────────
            raw_output       = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            processed_output = raw_output.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw_output else raw_output

            if args.strategy == "regular":
                # Try to extract 0 or 1 directly
                match = re.search(r'\b([01])\b', processed_output)
                if match:
                    final_prediction = match.group(1)
                else:
                    # Fallback: convert common outputs to binary
                    lower_output = processed_output.lower()
                    if any(word in lower_output for word in ['yes', 'positive', 'abnormal', 'tumor', 'present', '1']):
                        final_prediction = "1"
                    elif any(word in lower_output for word in ['no', 'negative', 'normal', 'absent', '0']):
                        final_prediction = "0"
                    else:
                        final_prediction = processed_output.strip()
            else:
                # two_token: try symbol conversion first, fallback to regex for base model
                final_prediction = symbol_manager.convert_symbols_back(processed_output).strip()
                if final_prediction not in ["0", "1"]:
                    match = re.search(r'\b([01])\b', processed_output)
                    if match:
                        final_prediction = match.group(1)
                    else:
                        lower_output = processed_output.lower()
                        if any(word in lower_output for word in ['yes', 'positive', 'abnormal', 'tumor', 'present']):
                            final_prediction = "1"
                        elif any(word in lower_output for word in ['no', 'negative', 'normal', 'absent']):
                            final_prediction = "0"
                        else:
                            final_prediction = processed_output.strip()  # keep raw if no match

            # Debug: first 5 samples
            if len(results) < 5:
                print(f"[DEBUG #{len(results)+1}] raw='{raw_output}' | pred='{final_prediction}' "
                      f"| gt='{ground_truth}' | auc_score={auc_score:.4f} "
                      f"| match={final_prediction == ground_truth}")

            is_correct = (final_prediction == ground_truth)
            if is_correct:
                correct_count += 1

            results.append({
                "id":          line.get("id", "N/A"),
                "images":      image_files,  # list of image file paths (can be multi-image for ICL)
                "raw_output":  raw_output,
                "decoded_pred": final_prediction,
                "gt":          ground_truth,
                "correct":     is_correct,
                "auc_score":   round(auc_score, 6)   # soft confidence for AUC
            })

        except Exception as e:
            print(f"Error processing sample {line.get('id')}: {e}")
            continue

    # ── 6. Metrics ───────────────────────────────────────────────────────────
    total    = len(results)
    accuracy = (correct_count / total) * 100 if total > 0 else 0.0

    gt_counts   = Counter(r["gt"]           for r in results)
    pred_counts = Counter(r["decoded_pred"] for r in results)
    labels      = sorted(gt_counts.keys())

    # Per-class precision / recall / F1
    per_class  = {}
    recall_sum = 0.0
    for label in labels:
        tp = sum(1 for r in results if r["gt"] == label and r["decoded_pred"] == label)
        fp = sum(1 for r in results if r["gt"] != label and r["decoded_pred"] == label)
        fn = sum(1 for r in results if r["gt"] == label and r["decoded_pred"] != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        recall_sum += recall
        per_class[label] = {
            "gt_total": gt_counts[label], "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)
        }

    sensitivity  = per_class.get("1", {}).get("recall", 0.0)
    specificity  = per_class.get("0", {}).get("recall", 0.0)
    balanced_acc = recall_sum / len(labels) if labels else 0.0
    macro_f1     = sum(v["f1"] for v in per_class.values()) / len(labels) if labels else 0.0

    # AUC — computed from soft confidence scores (paper metric for ColonPath)
    y_true  = [1 if r["gt"] == "1" else 0 for r in results]
    y_score = [r["auc_score"] for r in results]
    try:
        # Prefer sklearn if available (more numerically stable)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_score) * 100
    except ImportError:
        auc = _compute_auc(y_true, y_score) * 100
    except Exception as e:
        print(f"AUC computation failed: {e}")
        auc = float("nan")

    # ── 7. Print Results ─────────────────────────────────────────────────────
    paper_note = "(paper metric — compare with Table 2 / Table 4)"
    print("\n" + "="*60)
    print(f"FINAL MEDFMC RESULTS  |  STRATEGY: {args.strategy.upper()}")
    print(f"Checkpoint : {os.path.basename(model_path)}")
    print(f"Dataset    : GT {dict(gt_counts)}  |  Pred {dict(pred_counts)}")
    print(f"{'─'*60}")
    print(f"  Accuracy          : {accuracy:.2f}%   {paper_note}")
    print(f"  AUC               : {auc:.2f}%   {paper_note}")
    print(f"{'─'*60}")
    print(f"  Balanced Accuracy : {balanced_acc*100:.2f}%  (fairness — avg recall per class)")
    print(f"  Macro F1          : {macro_f1*100:.2f}%  (fairness — avg F1 per class)")
    print(f"  Sensitivity / TPR : {sensitivity*100:.2f}%  (% of tumours found)")
    print(f"  Specificity / TNR : {specificity*100:.2f}%  (% of normals rejected)")
    print(f"{'─'*60}")
    print(f"  Per-class:")
    for label, s in per_class.items():
        print(f"    Label {label}: {s['tp']}/{s['gt_total']} correct  "
              f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")
    print("="*60 + "\n")

    # ── 8. Save JSON ─────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    date_str     = datetime.now().strftime("%Y%m%d")
    dataset_name = os.path.splitext(os.path.basename(args.question_file))[0].replace("_test", "").replace("_val", "")
    out_path     = os.path.join(results_dir, f"results_{args.strategy}_{dataset_name}_{date_str}.json")

    with open(out_path, "w") as f:
        json.dump({
            # ── Paper metrics (primary comparison targets) ──
            "accuracy":          round(accuracy, 4),
            "auc":               round(auc, 4) if not (auc != auc) else None,  # NaN-safe
            # ── Fairness metrics (bias detection) ──
            "balanced_accuracy": round(balanced_acc * 100, 4),
            "macro_f1":          round(macro_f1 * 100, 4),
            "sensitivity":       round(sensitivity * 100, 4),
            "specificity":       round(specificity * 100, 4),
            # ── Run info ──
            "strategy":          args.strategy,
            "checkpoint":        os.path.basename(model_path),
            "correct":           correct_count,
            "total":             total,
            "gt_distribution":   dict(gt_counts),
            "pred_distribution": dict(pred_counts),
            "per_class":         per_class,
            "details":           results
        }, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base",     type=str, default=None)
    parser.add_argument("--model-path",     type=str, required=True)
    parser.add_argument("--image-folder",   type=str, required=True)
    parser.add_argument("--question-file",  type=str, required=True)
    parser.add_argument("--strategy",       type=str, default="regular", choices=["regular", "two_token"])
    parser.add_argument("--num-samples",    type=int, default=0, help="0 = all samples")
    parser.add_argument("--symbol-mappings", type=str, default=None,
                        help="Path to symbol_mappings.json (two_token only).")
    parser.add_argument("--icl-shots",      type=int, default=0,
                        help="Number of in-context learning (few-shot) examples. 0 = zero-shot inference (default).")
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="Temperature for sampling (0 = greedy decoding, >0 = sampling).")
    args = parser.parse_args()
    eval_model(args)
