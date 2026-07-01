"""
SPRInT VLM Inference + Evaluation — THIN WRAPPER (ICI inference.py style).

This file only: parses CLI, loads the model/checkpoint, resolves WHICH modes to
evaluate, calls the SINGLE shared evaluator in
`models/symbolAdapter/validation.py`, and writes the results JSON. It contains no
scoring or metric code.

Architecture (advisor directive — exactly like ICI):

        Training ──► validation.py ◄── Inference

- 0-shot (every reported MedFMC metric: colon AUC + accuracy, chest/endo AUC+mAP)
  runs through `SPRInTValidationManager.run_inference()`, which calls the SAME
  `_run_colon_auc` / `_run_multilabel_auc_map` methods training uses.
- `--icl-shots > 0` is a LEGACY text-output path, kept clearly separated in this
  wrapper (`_legacy_icl_text_eval`). It is not part of the unified evaluator and
  produces accuracy + macro_f1 only (no AUC/mAP). All reported results are 0-shot.

The results JSON format is unchanged (`_save_multimode_results`) so downstream
scripts (recompute_auc.py, compare_val_vs_infer.py, the paper tables) keep working.

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
from types import SimpleNamespace

import torch

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataload.example_selector import ExampleSelector
from dataload.prompt_builder import LLaVAPromptBuilder
from models.symbolAdapter.symbol_manager import SymbolManager
from models.symbolAdapter.validation import SPRInTValidationManager
from utils.evaluation_utils import compute_binary_metrics, compute_multilabel_metrics
from config.data_config import get_dataset_config, DatasetName

VALID_DATASETS = [e.value for e in DatasetName]

# MedFMC paper save_best / primary metric per task (mirrors sprint_callbacks.py).
_PRIMARY_METRIC = {"chest": "mAP", "endo": "macro_auc"}   # else colon → "accuracy"


# ─────────────────────────────────────────────────────────────────────────────
# Mode resolution (WHICH modes / mappings to evaluate — no metric computation)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_fixed_mapping(symbol_manager):
    """
    Return the checkpoint's trained symbol mapping for 'fixed' mode, robustly.

    two_token → SymbolManager.fixed_mappings (populated at training). ed_ft/id_ft/
    lf_ft store per-epoch symbols in epoch_mappings_history (fixed_mappings stays {}),
    keyed by epoch — and JSON serialises those keys as strings while current_epoch is
    an int, so get_current_symbols() returns {}. Here we pick the highest epoch's
    mapping explicitly (the symbols the converged checkpoint was last trained with).
    Returns {} if no trained symbols exist (→ caller skips 'fixed').
    """
    if symbol_manager.fixed_mappings:
        return dict(symbol_manager.fixed_mappings)
    hist = symbol_manager.epoch_mappings_history or {}
    if hist:
        last_key = max(hist.keys(), key=lambda k: int(k))
        return dict(hist[last_key])
    return {}


def _resolve_eval_modes(args, is_symbol):
    """
    Eval modes to run. --modes overrides; otherwise original/fixed/fresh for the
    symbol strategies (ICI parity) and 'original' only for regular/rft.
    """
    raw = getattr(args, "modes", None)
    if raw:
        req = [m.strip().lower() for m in raw.split(",") if m.strip()]
        valid = [m for m in req if m in ("original", "fixed", "fresh")]
        return valid or ["original"]
    return ["original", "fixed", "fresh"] if is_symbol else ["original"]


def _save_multimode_results(all_modes, dataset, args, model_path):
    """Write ONE combined JSON (all modes) and print a final per-mode comparison."""
    now = datetime.datetime.now()
    _split = getattr(args, "split", "test")
    out_filename = (
        f"results_{dataset}_{args.strategy}_{args.icl_shots}shot_{_split}"
        f"_{now.strftime('%Y%m%d_%H%M%S')}.json"
    )
    llava_dir = os.environ.get(
        "LLAVA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    out_dir = os.path.join(llava_dir, "logs", "json")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)

    primary = "original" if "original" in all_modes else (
        next(iter(all_modes)) if all_modes else None)
    output_data = {
        "metadata": {
            "dataset": dataset, "strategy": args.strategy, "icl_shots": args.icl_shots,
            "model_path": model_path, "model_base": args.model_base,
            "question_file": args.question_file, "train_file": args.train_file,
            "split": _split,
            "timestamp": now.isoformat(), "seed": args.seed,
            "modes": list(all_modes.keys()), "primary_mode": primary,
        },
        # Back-compat: top-level metrics/results mirror the primary (original) mode.
        "metrics": all_modes.get(primary, {}).get("metrics", {}) if primary else {},
        "results": all_modes.get(primary, {}).get("results", []) if primary else [],
        "modes": {
            m: {"metrics": d["metrics"], "mapping": d.get("mapping", {}),
                "results": d["results"]}
            for m, d in all_modes.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS (all modes)  dataset={dataset}  strategy={args.strategy}")
    print("=" * 70)
    for m, d in all_modes.items():
        mt = d["metrics"]
        bits = [f"acc={mt.get('accuracy', 0) * 100:.2f}%",
                f"macroF1={mt.get('macro_f1', 0) * 100:.2f}%"]
        if "auc" in mt:       bits.append(f"AUC={mt['auc'] * 100:.2f}%")
        if "macro_auc" in mt: bits.append(f"macroAUC={mt['macro_auc'] * 100:.2f}%")
        if "map" in mt:       bits.append(f"mAP={mt['map'] * 100:.2f}%")
        print(f"  [{m:<8}] " + "  ".join(bits))
    print(f"Saved → {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY few-shot (icl_shots > 0) text-output path — wrapper-level, NOT the
# unified evaluator. Retained so legacy ICL runs still produce accuracy + macro_f1
# (no AUC/mAP). All reported MedFMC metrics are 0-shot and go through validation.py.
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_icl_text_eval(args, *, model, tokenizer, image_processor, symbol_manager,
                          dataset, is_multi_label, valid_labels_lower,
                          questions, modes, mode_mappings):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from PIL import Image
    from tqdm import tqdm

    if not args.train_file:
        raise ValueError("--train-file is required when --icl-shots > 0")
    example_selector = ExampleSelector(os.path.expanduser(args.train_file), seed=args.seed)
    prompt_builder = LLaVAPromptBuilder()

    def _parse_binary(text):
        t = text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()
        for tok in t.split():
            s = tok.strip(".,;:()")
            if s in ("0", "1"):
                return s
        for tok in t.lower().split():
            s = tok.strip(".,;:()")
            if s == "yes":
                return "1"
            if s == "no":
                return "0"
        return t[:20]

    def _parse_multi(text, labels):
        t = text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()
        t = t.lower().strip().rstrip(".")
        if t in ("none", "no finding", "no findings", "normal", ""):
            return []
        out = []
        for part in [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]:
            if part in labels and part not in out:
                out.append(part)
            elif part not in labels:
                for lbl in labels:
                    if (lbl in part or part in lbl) and lbl not in out:
                        out.append(lbl)
                        break
        return out

    all_modes = {}
    for _mode in modes:
        sym = mode_mappings.get(_mode)
        if _mode in ("fixed", "fresh") and not sym:
            print(f"WARNING: mode '{_mode}' has empty mapping — skipping.")
            continue
        print(f"\n[SPRINT EVAL] LEGACY ICL ({args.icl_shots}-shot) mode={_mode}")
        results = []
        for i, sample in enumerate(tqdm(questions, desc=f"{dataset}|legacy-icl|{_mode}")):
            image_file = sample.get("image", "")
            if not image_file:
                continue
            gt = str(sample["conversations"][1]["value"]).strip()
            instruction = sample["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            try:
                examples = example_selector.select(args.icl_shots)
                prompt_str, image_paths = prompt_builder.build(
                    instruction=instruction, examples=examples,
                    test_image_path=image_file, symbol_mappings=sym,
                )
                pil_images = [Image.open(os.path.join(args.image_folder, p)).convert("RGB")
                              for p in image_paths]
                image_sizes = [im.size for im in pil_images]
                image_tensors = process_images(pil_images, image_processor, model.config)
                if isinstance(image_tensors, list):
                    image_tensor = torch.stack(image_tensors, 0).half().cuda()
                else:
                    image_tensor = image_tensors.half().cuda()
                    if image_tensor.dim() == 3:
                        image_tensor = image_tensor.unsqueeze(0)
                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], prompt_str)
                conv.append_message(conv.roles[1], None)
                input_ids = tokenizer_image_token(
                    conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids, images=image_tensor, image_sizes=image_sizes,
                        do_sample=False, max_new_tokens=30, use_cache=True,
                    )
                raw_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                decoded = symbol_manager.convert_symbols_back(raw_text, mappings=sym) if sym else raw_text
                if is_multi_label:
                    pred_parsed = _parse_multi(decoded, valid_labels_lower)
                    gt_parsed = [p.strip().lower() for p in gt.split(",")
                                 if p.strip().lower() not in ("", "none")]
                    results.append({
                        "id": sample.get("id", f"sample_{i}"), "image": image_file,
                        "raw_output": raw_text, "decoded": decoded,
                        "pred": ", ".join(pred_parsed) if pred_parsed else "none",
                        "pred_parsed": pred_parsed, "gt": gt, "gt_parsed": gt_parsed,
                        "correct": set(pred_parsed) == set(gt_parsed),
                    })
                else:
                    pred = _parse_binary(decoded)
                    results.append({
                        "id": sample.get("id", f"sample_{i}"), "image": image_file,
                        "raw_output": raw_text, "decoded": decoded, "pred": pred,
                        "gt": gt, "correct": pred == gt,
                        "logit_pos": None, "logit_neg": None,
                    })
            except Exception as e:
                print(f"WARNING: legacy-icl error on sample {sample.get('id')}: {e}")
        if is_multi_label:
            metrics = compute_multilabel_metrics(results, valid_labels_lower, tie_header="INFERENCE ")
        else:
            metrics = compute_binary_metrics(results, auc_token_id=None)
        all_modes[_mode] = {"metrics": metrics, "results": results, "mapping": sym or {}}
    return all_modes


# ─────────────────────────────────────────────────────────────────────────────
# Main: load model → call the shared evaluator (validation.py) → save JSON
# ─────────────────────────────────────────────────────────────────────────────

def eval_model(args):
    dataset = args.dataset
    cfg = get_dataset_config(dataset)

    is_multi_label = cfg.is_multi_label
    task_labels    = cfg.label_names
    valid_labels_lower = [lbl.lower() for lbl in task_labels]

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Forward --probe-batch-size to the env var the unified evaluator reads
    # (validation.py:_run_multilabel_auc_map → SPRINT_PROBE_BATCH_SIZE). This keeps
    # the existing orchestrator invocation (which passes --probe-batch-size) driving
    # the SAME batched per-class scoring training uses. 1 = unbatched (default).
    if getattr(args, "probe_batch_size", 1) and args.probe_batch_size > 1:
        os.environ["SPRINT_PROBE_BATCH_SIZE"] = str(args.probe_batch_size)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # builder.py branches on "lora" in model_name to choose LoRA vs standalone loading.
    # Our checkpoint dirs are named llava-{dataset}-{strategy}-... (no "lora") so we
    # inject the suffix whenever model_base is provided — which always means LoRA here.
    if args.model_base is not None and "lora" not in model_name.lower():
        model_name = model_name + "_lora"
    print(f"Loading model: {model_name} ...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=model_name,
        use_flash_attn=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model   : {os.path.basename(args.model_base or model_path)}")
    if args.model_base is not None:
        print(f"  Checkpoint   : {os.path.basename(model_path)}")
    print(f"  Total params : {total_params:,}  ({total_params / 1e9:.2f}B)")

    # ── Symbol manager ────────────────────────────────────────────────────────
    # two_token / ed_ft / id_ft / lf_ft were all trained WITH symbols. At inference
    # we load the trained mapping for ALL of them (ICI parity) so each checkpoint can
    # be evaluated in original/fixed/fresh modes. Only regular/rft has no symbols.
    _is_symbol = args.strategy in ("two_token", "ed_ft", "id_ft", "lf_ft")
    symbol_manager = SymbolManager(
        original_labels=task_labels,
        tokenizer=tokenizer,
        dynamic_per_epoch=False,
        symbol_type="two_token" if _is_symbol else "regular",
        no_symbols=not _is_symbol,
    )

    if _is_symbol:
        mapping_path = args.symbol_mappings or os.path.join(model_path, "symbol_mappings.json")
        if os.path.exists(mapping_path):
            symbol_manager.load_mappings(mapping_path)
            print(f"Loaded symbol mappings from {mapping_path}")
            print(f"  fixed_mappings={bool(symbol_manager.fixed_mappings)}  "
                  f"epoch_history_keys={list((symbol_manager.epoch_mappings_history or {}).keys())}")
        else:
            print("WARNING: No symbol_mappings.json found. Auto-generating (pipeline test only).")
            auto_path = os.path.join(model_path, "symbol_mappings_autogen.json")
            symbol_manager.save_mappings(auto_path)
            print(f"Auto-generated symbols saved to: {auto_path}")

    # ── Resolve eval modes (ICI parity: original / fixed / fresh) ─────────────
    modes = _resolve_eval_modes(args, _is_symbol)
    fixed_map = _resolve_fixed_mapping(symbol_manager) if _is_symbol else {}
    fresh_map = symbol_manager._generate_symbol_mappings() if _is_symbol else {}
    mode_mappings = {"original": None, "fixed": fixed_map, "fresh": fresh_map}
    print(f"Eval modes: {modes}")

    # ── icl_shots > 0 → LEGACY text path (wrapper-level; not the unified evaluator)
    if args.icl_shots > 0:
        question_file = os.path.expanduser(args.question_file)
        print(f"Loading {getattr(args, 'split', 'test')} split for {dataset}: {question_file}")
        questions = json.load(open(question_file))
        if args.num_samples > 0:
            questions = questions[: args.num_samples]
        print(f"[LEGACY ICL] {args.icl_shots}-shot over {len(questions)} samples")
        all_modes = _legacy_icl_text_eval(
            args, model=model, tokenizer=tokenizer, image_processor=image_processor,
            symbol_manager=symbol_manager, dataset=dataset, is_multi_label=is_multi_label,
            valid_labels_lower=valid_labels_lower, questions=questions,
            modes=modes, mode_mappings=mode_mappings,
        )
        _save_multimode_results(all_modes, dataset, args, model_path)
        _primary = "original" if "original" in all_modes else (
            next(iter(all_modes)) if all_modes else None)
        return all_modes[_primary]["metrics"] if _primary else {}

    # ── 0-shot → UNIFIED evaluator (validation.py), the SAME code training uses ─
    # The manager loads the eval split itself (eval_data_path) and uses the full
    # split when max_val_samples == 0. --val-subsample N reproduces the EXACT
    # training validation subset (random.Random(42).sample, == validation.py).
    _max_val = (
        args.val_subsample if (args.val_subsample and args.val_subsample > 0)
        else (args.num_samples if (args.num_samples and args.num_samples > 0) else 0)
    )
    validator = SPRInTValidationManager(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_args=SimpleNamespace(image_folder=args.image_folder),
        symbol_manager=symbol_manager,
        label_names=valid_labels_lower,
        is_multi_label=is_multi_label,
        dataset_name=dataset,
        primary_metric=_PRIMARY_METRIC.get(dataset, "accuracy"),
        max_val_samples=_max_val,
        eval_data_path=os.path.expanduser(args.question_file),
        cfg=cfg,
    )
    all_modes = validator.run_inference(modes=modes, mode_mappings=mode_mappings, strategy=args.strategy)

    _save_multimode_results(all_modes, dataset, args, model_path)

    _primary = "original" if "original" in all_modes else (
        next(iter(all_modes)) if all_modes else None)
    return all_modes[_primary]["metrics"] if _primary else {}


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
                        choices=["regular", "two_token", "ed_ft", "id_ft", "lf_ft"],
                        help="Inference strategy: 'regular'/'ed_ft'/'id_ft'/'lf_ft' use original "
                             "labels; 'two_token' decodes symbols back to original labels.")
    parser.add_argument("--symbol-mappings", type=str, default=None,
                        help="Path to symbol_mappings.json (symbol strategies). "
                             "Auto-detected from --model-path if omitted.")
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated eval modes: original,fixed,fresh "
                             "(ICI parity). Default: all three for symbol strategies "
                             "(two_token/ed_ft/id_ft/lf_ft), 'original' only for "
                             "regular/rft. 'fixed' uses the checkpoint's trained "
                             "symbols; 'fresh' uses newly generated symbols.")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Limit the number of eval samples. 0 = all. For 0-shot this "
                             "maps to the evaluator's max_val_samples (random.Random(42) "
                             "subset, reproducible).")
    parser.add_argument("--val-subsample", type=int, default=0,
                        help="If >0, evaluate this many samples via random.Random(42).sample "
                             "— byte-identical to training validation subsampling "
                             "(validation.py:_load_val_data). Reproduces the EXACT training "
                             "validation subset. 0 = use the full split.")
    parser.add_argument("--split", type=str, default="test",
                        help="Label for logging + output filename only ('test' or 'val'). "
                             "Does NOT change any metric or evaluation logic.")
    parser.add_argument("--icl-shots", type=int, default=0,
                        help="Number of in-context learning examples per query. 0 = zero-shot "
                             "(unified evaluator). >0 = LEGACY text-output path (no AUC/mAP).")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Path to *_train_percent100.json (required when --icl-shots > 0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for ICL example selection.")
    # Back-compat with the existing orchestrator/submit scripts. --probe-batch-size
    # is forwarded to the unified evaluator via SPRINT_PROBE_BATCH_SIZE (batched
    # per-class scoring; sample 0 auto-verified against unbatched). --diagnose-samples
    # / --ablation-samples are accepted for compatibility; the unified evaluator does
    # its own first-sample VAL-PRED / VAL-ABLATION logging.
    parser.add_argument("--probe-batch-size", type=int, default=1,
                        help="Chest/endo per-class probes scored per forward pass "
                             "(shared image, left-padded). 1 = unbatched (default).")
    parser.add_argument("--diagnose-samples", type=int, default=3,
                        help="(compat) Accepted for back-compat with launch scripts; "
                             "the unified evaluator logs first-sample probes itself.")
    parser.add_argument("--ablation-samples", type=int, default=1,
                        help="(compat) Accepted for back-compat with launch scripts; "
                             "the unified evaluator logs the first-sample ablation itself.")
    args = parser.parse_args()
    eval_model(args)
