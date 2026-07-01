"""
SPRInTValidationManager — VLM validation orchestration.

Mirrors ICI's ValidationManager structure.  Delegates pure metric
computation to utils/evaluation_utils.py.  All llava/torch imports
are deferred to method scope to avoid circular imports at load time.

Three validation modes (matching ICI):
  fixed    — symbols frozen from the current training epoch
  original — no symbols; original labels (advisor's primary view)
  fresh    — brand-new random symbols (generalization probe)

Primary metrics (MedFMC benchmark):
  colon        → AUC   (binary; token-"1" logit at step 0, 0 extra passes)
  chest / endo → mAP + macro_AUC  (per-class binary Yes/No queries)

Supplementary metrics (text-generation path, all modes):
  macro_f1   — per-label binary F1 averaged over classes
  accuracy   — exact-match (near 0% for multi-label; expected, not a bug)

Metric definitions and caveats are documented in utils/evaluation_utils.py.
"""

import json
import math
import os
import re
import random
from datetime import datetime

import torch
from PIL import Image

from utils.evaluation_utils import (
    parse_multi_label,
    compute_all_metrics,
    compute_auc_trapz,
    auc_avg_rank,
    compute_ap,
    tie_diagnostics,
    compute_binary_metrics,
    compute_multilabel_metrics,
)
from dataload.medfmc_prompts import build_per_class_prompt, sprint_log, score_probes_pyes


class SPRInTValidationManager:
    """
    Validation orchestrator for SPRInT VLM training.

    Used as a delegate by SPRInTValidationCallback in train.py.
    The callback handles HF Trainer lifecycle; this class handles
    inference, metric computation, logging, and JSON serialisation.
    """

    _MODE_LABELS = {
        "fixed":    "Fixed-Symbols",
        "original": "Original",
        "fresh":    "Fresh-Symbols",
    }

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        data_args,
        symbol_manager,
        label_names: list,
        is_multi_label: bool,
        dataset_name: str,
        primary_metric: str,       # "accuracy" for colon, "mAP" for chest, "macro_auc" for endo
        validation_modes: str = "fixed,original,fresh",
        max_val_samples: int = 100,
        eval_data_path: str = None,
        print_fn=None,
        cfg=None,
    ):
        self.model           = model
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.data_args       = data_args
        self.symbol_manager  = symbol_manager

        self._label_names     = label_names
        self._is_multi_label  = is_multi_label
        self._dataset_name    = dataset_name
        self._primary_metric  = primary_metric
        self._validation_modes = validation_modes
        self._max_val_samples = max_val_samples
        self._eval_data_path  = eval_data_path
        self._val_data        = None            # cached after first load
        self._print           = print_fn or print
        # Full DatasetConfig (original-case label_names + class_definitions +
        # instruction_intro) — used to build the HVB-style per-class mAP/AUC
        # probe so it carries the SAME definition block seen in training.
        self._cfg             = cfg

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_val_data(self) -> list:
        """Load and cache validation JSON; return [] if path not set."""
        if self._val_data is not None:
            return self._val_data
        path = self._eval_data_path
        if not path or not os.path.exists(path):
            return []
        self._print(f"Loading val split for {self._dataset_name}: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        total_in_file = len(data)
        self._print(f"Loaded {total_in_file} examples from {self._dataset_name} val")
        n = self._max_val_samples
        if n > 0 and len(data) > n:
            rng = random.Random(42)
            data = rng.sample(data, n)
            self._print(
                f"[VAL-SUBSAMPLE] random.Random(42).sample → {n} of {total_in_file}"
                f" (reproduces training validation subset)"
            )
        self._print(f"Loaded {self._dataset_name} VAL: {len(data)} samples")
        self._val_data = data
        return data

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image_tensor(self, img_path: str):
        """Open image → RGB → preprocess → [1, C, H, W] on model device."""
        from llava.mm_utils import process_images
        image = Image.open(img_path).convert("RGB")
        tensor = process_images([image], self.image_processor, self.data_args)[0]
        return tensor.unsqueeze(0).to(dtype=self.model.dtype, device='cuda')

    # ── Text-generation validation (one mode) ─────────────────────────────────

    def _run_val_mode(self, symbols: dict) -> dict:
        """
        Run greedy text generation over the cached validation set.

        Returns a dict with macro_f1, accuracy, correct, total,
        valid_samples, invalid_predictions, and per-sample results.
        """
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from tqdm.auto import tqdm as _tqdm

        max_new  = 64 if self._is_multi_label else 10
        val_data = self._load_val_data()
        results  = []

        with torch.no_grad():
            for item in _tqdm(val_data, desc="Evaluating", dynamic_ncols=True):
                img_path = os.path.join(self.data_args.image_folder, item.get("image", ""))
                try:
                    image_tensor = self._load_image_tensor(img_path)
                except Exception:
                    continue

                # Build the prompt from the LIVE config instruction (the current
                # definition block) — NOT the val JSON's baked-in human turn, which
                # may be a stale file from before definitions were added
                # (medfmc_to_llava.py never regenerates *_val_*.json). Only image +
                # ground truth are taken from the JSON. This mirrors the primary
                # mAP/AUC probe, which also builds from cfg, so text-gen and the
                # primary metric stay consistent with training.
                if self._cfg is not None and getattr(self._cfg, "instruction", None):
                    _raw_human = DEFAULT_IMAGE_TOKEN + "\n" + self._cfg.instruction
                else:
                    _raw_human = item["conversations"][0]["value"]
                human_text = _raw_human
                if self.symbol_manager is not None and symbols:
                    human_text = self.symbol_manager.apply_to_text(human_text, symbols)

                conv = conv_templates["vicuna_v1"].copy()
                if DEFAULT_IMAGE_TOKEN not in human_text:
                    human_text = DEFAULT_IMAGE_TOKEN + "\n" + human_text
                conv.append_message(conv.roles[0], human_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()

                # PEFT 0.7.1 + LLaVA inputs_embeds path returns only new tokens.
                # Decode directly — no input prefix in the output tensor.
                output_ids = self.model.generate(
                    inputs=input_ids, images=image_tensor,
                    max_new_tokens=max_new, do_sample=False, use_cache=True,
                )
                pred_text = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()

                gt_orig = item["conversations"][1]["value"].strip()

                if self.symbol_manager is not None and symbols:
                    pred_orig = self.symbol_manager.convert_symbols_back(pred_text, mappings=symbols)
                elif self._is_multi_label:
                    pred_orig = pred_text
                else:
                    m = re.search(r"\b([01])\b", pred_text)
                    pred_orig = m.group(1) if m else pred_text.strip()

                entry = {"gt": gt_orig, "pred": pred_orig, "pred_raw": pred_text}

                if self._is_multi_label:
                    entry["pred_set"] = parse_multi_label(pred_orig, self._label_names)
                    gt_parts = [p.strip().lower() for p in gt_orig.split(",") if p.strip()]
                    entry["gt_set"] = {p for p in gt_parts if p in self._label_names}
                    entry["correct"] = entry["pred_set"] == entry["gt_set"]
                else:
                    entry["correct"] = (pred_orig == gt_orig)

                results.append(entry)

        metrics       = compute_all_metrics(results, self._label_names, self._is_multi_label)
        total         = len(results)
        correct       = sum(r["correct"] for r in results)
        valid_samples = total if self._is_multi_label else sum(
            1 for r in results if r["pred"] in ("0", "1")
        )
        return {
            **metrics,
            "correct":               correct,
            "total":                 total,
            "valid_samples":         valid_samples,
            "invalid_predictions":   total - valid_samples,
            "results":               results,
        }

    # ── AUC inference (colon binary) ──────────────────────────────────────────

    def _run_colon_auc(self, mapping: dict = None, return_details: bool = False):
        """
        Binary AUC for colon: capture token-'1' logit at step 0.
        Zero extra image forward passes beyond the text-gen pass already run.

        mapping:        None (training / 'original' mode) → bare "0"/"1" answer
                        tokens, raw prompt. A dict ('fixed'/'fresh' symbol modes,
                        inference only) → answer tokens are the symbols for "0"/"1"
                        and the human prompt is symbol-substituted, mirroring the
                        old sprint_eval colon symbol path.
        return_details: False (training) → returns the small {AUC, macro_auc, mAP,
                        accuracy_aacc} dict UNCHANGED. True (inference) → returns
                        (full_metrics, per_sample_records) so sprint_eval can write
                        the same results JSON. The scoring is identical either way.
        """
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        def _first_non_space(text):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return next((i for i in ids if i != 29871), ids[0] if ids else None)

        # Answer tokens: bare "0"/"1" (original/training) or the symbols mapped to
        # them (fixed/fresh). Skips the SentencePiece leading-space token (29871).
        tok1_id = _first_non_space(mapping.get("1", "1") if mapping else "1")
        tok0_id = _first_non_space(mapping.get("0", "0") if mapping else "0")

        if return_details:
            print(
                f"  Answer tokens: neg={tok0_id} '{self.tokenizer.decode([tok0_id])}'  "
                f"pos={tok1_id} '{self.tokenizer.decode([tok1_id])}'"
            )

        # Per-sample result records (logit_pos/logit_neg/argmax-pred/gt), then the
        # SHARED canonical aggregator (utils/evaluation_utils.compute_binary_metrics)
        # — the SAME code training and inference use (one evaluation implementation).
        from tqdm.auto import tqdm as _tqdm
        results = []
        with torch.no_grad():
            _val_data_colon = self._load_val_data()
            _desc = f"{self._dataset_name} | original | 0-shot"
            for i, item in enumerate(_tqdm(_val_data_colon, desc=_desc)):
                img_path = os.path.join(self.data_args.image_folder, item.get("image", ""))
                try:
                    image_tensor = self._load_image_tensor(img_path)
                except Exception:
                    continue

                human_text = item["conversations"][0]["value"]
                # Symbol modes substitute the human prompt exactly as training did
                # (no-op when mapping is None → byte-identical to the original path).
                if mapping and self.symbol_manager is not None:
                    human_text = self.symbol_manager.apply_to_text(human_text, mapping)
                conv = conv_templates["vicuna_v1"].copy()
                if DEFAULT_IMAGE_TOKEN not in human_text:
                    human_text = DEFAULT_IMAGE_TOKEN + "\n" + human_text
                conv.append_message(conv.roles[0], human_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()

                if return_details and i < 2:
                    print("=" * 70)
                    print(f"[SPRINT EVAL] sample={i} (logit-AUC mode)")
                    sym_diag = str(mapping) if mapping else "NONE (regular strategy)"
                    print(f"[SPRINT EVAL] ACTIVE SYMBOL MAPPING: {sym_diag}")
                    raw_instr = item["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    print(f"[SPRINT EVAL] HUMAN INSTRUCTION (raw, BEFORE replacement):\n{raw_instr}")
                    print(f"[SPRINT EVAL] FULL PROMPT SENT TO MODEL (AFTER replacement):\n{human_text}")
                    gt_show = item["conversations"][1]["value"].strip()
                    gt_in_mapping = mapping.get(gt_show) if mapping else None
                    print(f"[SPRINT EVAL] GPT ground truth : {gt_show!r}  |  any GT token in sym_mappings: {gt_in_mapping}")
                    print("=" * 70)

                out = self.model.generate(
                    inputs=input_ids, images=image_tensor,
                    max_new_tokens=10, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                )
                l1 = out.scores[0][0][tok1_id].item()
                l0 = out.scores[0][0][tok0_id].item()
                gt = item["conversations"][1]["value"].strip()
                pred = "1" if l1 > l0 else "0"   # argmax(0,1)
                gt_norm = "1" if gt == "1" else "0"
                rec = {
                    "pred":      pred,
                    "gt":        gt_norm,
                    "logit_pos": l1,
                    "logit_neg": l0,
                }
                if return_details:
                    # Decode the generated text only so the inference JSON keeps its
                    # raw_output/decoded fields (not consumed by recompute_auc.py; the
                    # reported accuracy uses logit-argmax via compute_binary_metrics).
                    raw_text = self.tokenizer.batch_decode(
                        out.sequences, skip_special_tokens=True
                    )[0].strip()
                    decoded = (
                        self.symbol_manager.convert_symbols_back(raw_text, mappings=mapping)
                        if (mapping and self.symbol_manager is not None) else raw_text
                    )
                    rec.update({
                        "id":         item.get("id", f"sample_{len(results)}"),
                        "image":      item.get("image", ""),
                        "raw_output": raw_text,
                        "decoded":    decoded,
                        "correct":    pred == gt_norm,
                    })
                results.append(rec)

        m = compute_binary_metrics(results, auc_token_id=1)
        if return_details:
            return m, results
        return {
            "AUC":           round(m.get("auc", m.get("macro_auc", 0.0)), 6),
            "macro_auc":     round(m.get("macro_auc", 0.0), 6),
            "mAP":           round(m.get("map", 0.0), 6),
            "accuracy_aacc": round(m.get("accuracy_aacc", 0.0), 6),
        }

    # ── Influence ablation (proves defs/symbols move P(Yes)) ──────────────────

    def _pyes_val(self, human_text, image_tensor, yes_id, no_id):
        """Score one per-class probe → P(Yes), mirroring the main per-class path."""
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], human_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        out = self.model.generate(
            inputs=input_ids, images=image_tensor,
            max_new_tokens=1, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
        logits = out.scores[0][0]
        l_no, l_yes = logits[no_id].item(), logits[yes_id].item()
        m = max(l_no, l_yes)
        return math.exp(l_yes - m) / (math.exp(l_no - m) + math.exp(l_yes - m))

    def _run_val_influence_ablation(self, val_item, sym_mappings, yes_id, no_id, n_classes=3):
        """
        Counterfactual check (same image, prompt content varied) during validation:
          delta_def    = P(full) - P(no_def)      → definitions' effect on P(Yes)
          delta_symbol = P(full) - P(orig_label)  → symbols' effect (SS-FT only)
        Non-zero deltas prove the def block / symbols actually move the score used
        for AUC/mAP, not merely appear in the prompt.
        """
        from llava.constants import DEFAULT_IMAGE_TOKEN
        if self._cfg is None or not self._cfg.class_definitions:
            return
        img_path = os.path.join(self.data_args.image_folder, val_item.get("image", ""))
        try:
            image_tensor = self._load_image_tensor(img_path)
        except Exception as _e:
            self._print(f"[SPRINT::VAL-ABLATION] skipped (image load: {_e})")
            return
        apply_fn = self.symbol_manager.apply_to_text if self.symbol_manager is not None else None
        sprint_log(
            "VAL-ABLATION", image=val_item.get("image", ""),
            note="P(Yes) under prompt ablations (same image) — proves INFLUENCE, not presence",
            classes=list(self._cfg.label_names[:n_classes]),
        )
        for cls_name in self._cfg.label_names[:n_classes]:
            p_full = build_per_class_prompt(
                self._cfg, self._dataset_name, cls_name,
                sym_mappings=sym_mappings, apply_to_text_fn=apply_fn,
                image_token=DEFAULT_IMAGE_TOKEN,
            )
            p_nodef = build_per_class_prompt(
                self._cfg, self._dataset_name, cls_name,
                sym_mappings=sym_mappings, apply_to_text_fn=apply_fn,
                image_token=DEFAULT_IMAGE_TOKEN, include_definitions=False,
            )
            pf = self._pyes_val(p_full,  image_tensor, yes_id, no_id)
            pn = self._pyes_val(p_nodef, image_tensor, yes_id, no_id)
            row = {"cls": cls_name, "P_yes_full": round(pf, 4),
                   "P_yes_no_def": round(pn, 4), "delta_def": round(pf - pn, 4)}
            if sym_mappings:
                p_orig = build_per_class_prompt(
                    self._cfg, self._dataset_name, cls_name,
                    sym_mappings=None, apply_to_text_fn=apply_fn,
                    image_token=DEFAULT_IMAGE_TOKEN,
                )
                po = self._pyes_val(p_orig, image_tensor, yes_id, no_id)
                row["P_yes_orig_label"] = round(po, 4)
                row["delta_symbol"] = round(pf - po, 4)
            sprint_log("VAL-ABLATION-ROW", **row)

    # ── AUC/mAP inference (chest/endo per-class binary queries) ───────────────

    def _run_multilabel_auc_map(self, sym_mappings: dict = None,
                                return_details: bool = False):
        """
        Per-class binary "Does this show <class>?  Answer Yes or No." queries.

        HVB-faithful: each query carries the SAME "- label: definition" block the
        model saw in training (via dataload.medfmc_prompts.build_per_class_prompt)
        plus a per-class focusing question.  When sym_mappings is provided (SS-FT)
        the whole probe is symbol-substituted exactly as in training.

        For each (sample, class) pair: softmax over [logit_No, logit_Yes] → P(Yes).
        P(Yes) acts as a per-class discriminative score for ranking.
        Macro AUC and mAP are then computed over these scores — UNCHANGED math.

        Compute cost: num_classes × num_val_samples 1-token forward passes.
          chest: 19 × 200 = 3800 passes ≈ 5–8 min/epoch on A100.
          endo:   4 × 200 =  800 passes ≈ 1–2 min/epoch on A100.
        """
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids  = self.tokenizer.encode("No",  add_special_tokens=False)
        yes_id  = next((i for i in yes_ids if i != 29871), yes_ids[0])
        no_id   = next((i for i in no_ids  if i != 29871), no_ids[0])

        # Per-class probe batching (opt-in via env, default 1 = unchanged behaviour).
        # Mirrors sprint_eval.py's --probe-batch-size; shared scorer guarantees the
        # batched P(Yes) matches unbatched (auto-verified on the first val sample).
        try:
            _vbatch = int(os.environ.get("SPRINT_PROBE_BATCH_SIZE", "1") or "1")
        except ValueError:
            _vbatch = 1
        if _vbatch < 1:
            _vbatch = 1

        # Batched probe scoring left-pads the batch; LLaVA's internal multimodal
        # re-pad (llava_arch.py:290) must be LEFT-aligned too, else generate()
        # reads step-0 logits from a pad position for the shorter rows. Training
        # uses right padding (train.py:1233), so set left ONLY for the batched
        # probe and restore below — no train/inference mismatch is introduced.
        # Guarded by _vbatch>1 so the default path is byte-identical (no write).
        _saved_pad_side = getattr(self.model.config, "tokenizer_padding_side", "right")
        if _vbatch > 1:
            self.model.config.tokenizer_padding_side = "left"

        if "chest" in self._dataset_name:
            modality = "chest X-ray"
        elif "endo" in self._dataset_name:
            modality = "endoscopy image"
        else:
            modality = "medical image"

        val_data = self._load_val_data()
        n     = len(val_data)
        n_cls = len(self._label_names)
        all_scores = [[0.0] * n_cls for _ in range(n)]
        all_labels = [[0]   * n_cls for _ in range(n)]


        from tqdm.auto import tqdm as _tqdm
        _desc = f"{self._dataset_name} | per-class AUC/mAP | 0-shot"
        with torch.no_grad():
            for i, item in enumerate(_tqdm(val_data, desc=_desc)):
                img_path = os.path.join(self.data_args.image_folder, item.get("image", ""))
                try:
                    image_tensor = self._load_image_tensor(img_path)
                except Exception:
                    continue

                gt_text  = item["conversations"][1]["value"].strip().lower()
                gt_parts = {p.strip() for p in gt_text.replace(";", ",").split(",") if p.strip()}
                for j, lbl in enumerate(self._label_names):
                    all_labels[i][j] = 1 if lbl in gt_parts else 0

                # Per-sample diagnostic — first 2 samples: show the first-class probe text
                if return_details and i < 2:
                    cls0 = (self._cfg.label_names[0] if self._cfg is not None else self._label_names[0])
                    lbl0 = self._label_names[0]
                    if self._cfg is not None and self._cfg.class_definitions:
                        _p0 = build_per_class_prompt(
                            cfg=self._cfg, dataset=self._dataset_name, class_name=cls0,
                            sym_mappings=sym_mappings,
                            apply_to_text_fn=(self.symbol_manager.apply_to_text
                                              if self.symbol_manager is not None else None),
                            image_token=DEFAULT_IMAGE_TOKEN,
                        )
                    else:
                        _p0 = (DEFAULT_IMAGE_TOKEN + "\n"
                               + f"Does this {modality} show {lbl0.replace('_', ' ')}? Answer Yes or No.")
                    print("=" * 70)
                    print(f"[SPRINT EVAL] sample={i} (per-class binary mode, HVB-style probe)")
                    print(f"[SPRINT EVAL] GT ground truth : {gt_text!r}")
                    print(f"[SPRINT EVAL] First-class probe (class={cls0!r}):\n{_p0}")
                    print("=" * 70)

                # Batched per-class scoring (opt-in: SPRINT_PROBE_BATCH_SIZE>1).
                # Builds every class probe (same prompt logic as below), scores
                # them in left-padded batches via the shared helper, and fills
                # all_scores[i]. P(Yes) is verified == unbatched on the first
                # sample, so AUC/mAP are unchanged.
                if _vbatch > 1:
                    _prompts = []
                    for j, lbl in enumerate(self._label_names):
                        orig_lbl = self._cfg.label_names[j] if self._cfg is not None else lbl
                        if self._cfg is not None and self._cfg.class_definitions:
                            _ht = build_per_class_prompt(
                                cfg=self._cfg, dataset=self._dataset_name, class_name=orig_lbl,
                                sym_mappings=sym_mappings,
                                apply_to_text_fn=(self.symbol_manager.apply_to_text
                                                  if self.symbol_manager is not None else None),
                                image_token=DEFAULT_IMAGE_TOKEN,
                            )
                        else:
                            _ht = (DEFAULT_IMAGE_TOKEN + "\n"
                                   + f"Does this {modality} show {lbl.replace('_', ' ')}? Answer Yes or No.")
                        _prompts.append(_ht)
                    _sc = score_probes_pyes(
                        _prompts, self.tokenizer, self.model, image_tensor, None,
                        no_id, yes_id, batch_size=_vbatch, verify=(i == 0),
                    )
                    for j in range(n_cls):
                        all_scores[i][j] = _sc[j]

                for j, lbl in enumerate([] if _vbatch > 1 else self._label_names):
                    # Original-case label drives the def-block key lookup; the
                    # lowercased self._label_names[j] is kept for GT matching above.
                    orig_lbl = self._cfg.label_names[j] if self._cfg is not None else lbl
                    if self._cfg is not None and self._cfg.class_definitions:
                        human_text = build_per_class_prompt(
                            cfg=self._cfg,
                            dataset=self._dataset_name,
                            class_name=orig_lbl,
                            sym_mappings=sym_mappings,
                            apply_to_text_fn=(
                                self.symbol_manager.apply_to_text
                                if self.symbol_manager is not None else None
                            ),
                            image_token=DEFAULT_IMAGE_TOKEN,
                        )
                    else:
                        # Fallback (no definitions configured): legacy minimal probe.
                        human_text = (
                            DEFAULT_IMAGE_TOKEN + "\n"
                            + f"Does this {modality} show {lbl.replace('_', ' ')}? Answer Yes or No."
                        )
                    conv = conv_templates["vicuna_v1"].copy()
                    conv.append_message(conv.roles[0], human_text)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).cuda()

                    out = self.model.generate(
                        inputs=input_ids, images=image_tensor,
                        max_new_tokens=1, do_sample=False,
                        output_scores=True, return_dict_in_generate=True,
                    )
                    logits = out.scores[0][0]
                    l_no, l_yes = logits[no_id].item(), logits[yes_id].item()
                    del out, logits, input_ids   # free 32K-float score tensor immediately
                    m_val   = max(l_no, l_yes)
                    exp_no  = math.exp(l_no  - m_val)
                    exp_yes = math.exp(l_yes - m_val)
                    all_scores[i][j] = exp_yes / (exp_no + exp_yes)


                # Free image tensor after all 19 (or N) class queries for this sample.
                # Flush fragmented reserved-but-unallocated blocks every 10 samples so
                # the allocator can find contiguous space for subsequent attention buffers.
                del image_tensor
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.empty_cache()

        # Restore the training padding side (no-op when batching was off).
        if _vbatch > 1:
            self.model.config.tokenizer_padding_side = _saved_pad_side

        # Per-sample result records (P(Yes) scores + 0.5-threshold preds + GT),
        # then the SHARED canonical aggregator
        # (utils/evaluation_utils.compute_multilabel_metrics) — the SAME code
        # training and inference use (one evaluation implementation). It also runs
        # the tie diagnostic internally (header "VALIDATION ").
        # scores + gt_binary are the only fields recompute_auc.py reads; the rest
        # (id/image/pred_parsed/gt_parsed/pred/correct) are added under
        # return_details so the inference JSON keeps its existing keys. compute_*
        # only reads scores/gt_binary/pred_binary, so the extra keys never affect
        # the metric values (training output is unchanged).
        results = []
        for i in range(n):
            pred_binary = [1 if all_scores[i][j] >= 0.5 else 0 for j in range(n_cls)]
            rec = {
                "scores":      all_scores[i],
                "gt_binary":   all_labels[i],
                "pred_binary": pred_binary,
            }
            if return_details:
                _item = val_data[i] if i < len(val_data) else {}
                pred_parsed = [self._label_names[j] for j in range(n_cls) if pred_binary[j] == 1]
                gt_parsed   = [self._label_names[j] for j in range(n_cls) if all_labels[i][j] == 1]
                rec.update({
                    "id":          _item.get("id", f"sample_{i}"),
                    "image":       _item.get("image", ""),
                    "pred_parsed": pred_parsed,
                    "gt_parsed":   gt_parsed,
                    "pred":        ", ".join(pred_parsed) if pred_parsed else "none",
                    "correct":     pred_binary == all_labels[i],
                })
                if i < 3:
                    p_yes = {self._label_names[j]: round(all_scores[i][j], 3) for j in range(n_cls)}
                    sprint_log(
                        "INFER-PRED", sample=i, image=_item.get("image", ""),
                        gt_present=sorted(gt_parsed), pred_present=sorted(pred_parsed),
                        per_class_P_yes=p_yes,
                    )
            results.append(rec)
        m = compute_multilabel_metrics(results, self._label_names, tie_header="VALIDATION ")
        if return_details:
            return m, results
        return {
            "macro_auc":     round(m.get("macro_auc", 0.0), 6),
            "mAP":           round(m.get("map", 0.0), 6),
            "accuracy_aacc": round(m.get("accuracy_aacc", 0.0), 6),
        }

    # ── Per-epoch orchestration ───────────────────────────────────────────────

    def run_comprehensive_validation(
        self,
        epoch: int,
        total_epochs: int,
        mode_symbols: dict,
        strategy: str,
        output_dir: str,
        step_losses: list,
        compute_auc_map: bool = True,
    ) -> dict:
        """
        Run all requested validation modes then AUC/mAP.
        Prints all results and saves a per-epoch JSON.

        Returns:
            {epoch, avg_loss, modes: {mode → {macro_f1, accuracy}}, auc_map: {...}}
        """
        dataset_name = self._dataset_name
        self.model.eval()
        epoch_results = {}

        # ── Text-generation validation (one pass per mode) ────────────────────
        for mode_name, symbols in mode_symbols.items():
            mode_label = self._MODE_LABELS.get(mode_name, mode_name.title())

            self._print(f"\n{'='*80}")
            self._print(f"VALIDATION MODE: {mode_label}")
            self._print(f"{'='*80}")

            metrics = self._run_val_mode(symbols)
            epoch_results[mode_name] = metrics

            # Supplementary text-gen metrics — clearly labelled as secondary
            self._print(f"\n{'='*80}")
            if self._is_multi_label:
                self._print(
                    f"Supplementary text-gen metrics for {dataset_name}"
                    f"  [primary metrics: AUC/mAP printed below]"
                )
                self._print(
                    f"  Note: accuracy = exact {len(self._label_names)}-label match;"
                    f" near 0% is expected and normal."
                )
            else:
                self._print(
                    f"Text-gen metrics for {dataset_name}"
                    f"  [primary metric: AUC printed below]"
                )
            self._print(f"{'='*80}")
            self._print(f"  macro_f1:            {metrics['macro_f1']:.4f}")
            self._print(f"  accuracy:            {metrics['accuracy']:.4f}")
            self._print(f"  correct/total:       {metrics['correct']}/{metrics['total']}")
            self._print(f"  valid_samples:       {metrics['valid_samples']}")
            self._print(f"  invalid_predictions: {metrics['invalid_predictions']}")
            self._print(f"  total_samples:       {metrics['total']}")

            self._print(f"\nExample predictions after cleaning:")
            for r in metrics.get("results", [])[:5]:
                self._print(f"Original: {r['pred_raw']}")
                self._print(f"Cleaned:  {r['pred']}")
                self._print(f"True:     {r['gt']}")
                self._print(f"{'--'*25}")

        # ── Primary metrics: AUC / mAP (original mode, logit-based) ──────────
        # Flush PyTorch's reserved-but-unallocated memory pool before the
        # per-class binary pass.  Text-gen above leaves ~350 MiB fragmented;
        # empty_cache() returns it to CUDA so contiguous attention buffers
        # can be allocated without OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        auc_map_results = {}
        if compute_auc_map and "original" in mode_symbols:
            if not self._is_multi_label:
                self._print(f"\n{'='*80}")
                self._print(f"★  PRIMARY METRICS — AUC (MedFMC benchmark, colon binary)")
                self._print(f"{'='*80}")
                auc_map_results = self._run_colon_auc()
                self._print(f"  ★ AUC : {auc_map_results['AUC']:.4f}   ← primary metric")
            else:
                self._print(f"\n{'='*80}")
                self._print(
                    f"★  PRIMARY METRICS — AUC + mAP (MedFMC benchmark, {dataset_name})"
                )
                self._print(
                    f"   Running {len(self._label_names)} binary queries"
                    f" × {len(self._load_val_data())} samples ..."
                )
                self._print(f"{'='*80}")
                # SS-FT carries its fixed symbols into the probe (consistent with
                # training); regular/ed_ft/id_ft/lf_ft evaluate with original
                # labels (matches the final-inference convention in sprint_eval.py).
                probe_syms = (
                    self.symbol_manager.get_current_symbols()
                    if (self.symbol_manager is not None and strategy in ("two_token", "ss_ft"))
                    else {}
                )
                sprint_log(
                    "VAL-AUC-MAP", dataset=self._dataset_name, strategy=strategy,
                    probe_symbols=probe_syms or {},
                )
                auc_map_results = self._run_multilabel_auc_map(sym_mappings=probe_syms)
                self._print(f"  ★ mAP      : {auc_map_results['mAP']:.4f}   ← primary metric")
                self._print(f"  ★ macro_AUC: {auc_map_results['macro_auc']:.4f}   ← primary metric")

        # ── FINAL VALIDATION RESULTS — all metrics in one place ───────────────
        self._print(f"\n{'='*80}")
        self._print(f"FINAL VALIDATION RESULTS — Epoch {epoch}/{total_epochs}")
        self._print(f"{'='*80}")

        if auc_map_results:
            self._print(f"\nPrimary Metrics — MedFMC benchmark (original mode, logit-based)")
            self._print(f"{'-'*80}")
            if "AUC" in auc_map_results:
                self._print(f"  ★ {dataset_name:<20} AUC       : {auc_map_results['AUC']:.4f}")
            if "macro_auc" in auc_map_results:
                self._print(f"  ★ {dataset_name:<20} macro_AUC : {auc_map_results['macro_auc']:.4f}")
            if "mAP" in auc_map_results:
                self._print(f"  ★ {dataset_name:<20} mAP       : {auc_map_results['mAP']:.4f}")

        self._print(f"\nSupplementary Text-Gen Metrics (all modes, generation-based)")
        self._print(f"{'-'*80}")
        for mode_name, metrics in epoch_results.items():
            self._print(
                f"  {mode_name:<10} {dataset_name:<18}:"
                f"  macro_f1={metrics['macro_f1']:.4f}"
                f"  accuracy={metrics['accuracy']:.4f}"
            )
        self._print(f"{'='*80}")

        # ── ICI-style 📊 summary lines ─────────────────────────────────────────
        orig_f1  = epoch_results.get("original", {}).get("macro_f1", 0.0)
        orig_acc = epoch_results.get("original", {}).get("accuracy", 0.0)
        self._print(f"📊 Dataset metrics (original mode): {{{dataset_name}: {orig_f1:.4f}}}  [macro_f1]")
        self._print(f"📊 Combined metric (original mode): {orig_f1:.4f}  [macro_f1]")
        self._print(f"📊 Accuracy        (original mode): {orig_acc:.4f}  [exact-match]")
        self._print(f"📊 Composite string (original mode): {dataset_name}:{orig_f1:.4f}")
        if auc_map_results:
            primary_val = auc_map_results.get(
                self._primary_metric,
                auc_map_results.get("macro_auc", auc_map_results.get("AUC", 0.0))
            )
            self._print(f"📊 {self._primary_metric} (MedFMC primary, original mode): {primary_val:.4f}")
            if "mAP" in auc_map_results:
                self._print(f"📊 mAP      : {auc_map_results['mAP']:.4f}")
            auc_key = "AUC" if "AUC" in auc_map_results else "macro_auc"
            if auc_key in auc_map_results:
                self._print(f"📊 macro_AUC: {auc_map_results[auc_key]:.4f}")

        # ── Avg loss + validation JSON line ───────────────────────────────────
        avg_loss = sum(step_losses) / len(step_losses) if step_losses else float("nan")
        self._print(f"Epoch {epoch} loss: {avg_loss:.6f}")
        val_summary = {m: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]}
                       for m, v in epoch_results.items()}
        if auc_map_results:
            val_summary["auc_map"] = auc_map_results
        self._print(f"Epoch {epoch} validation: {val_summary}")

        # ── Save per-epoch JSON ───────────────────────────────────────────────
        out_path = os.path.join(
            output_dir,
            f"val_epoch{epoch}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "epoch":    epoch,
                "strategy": strategy,
                "avg_loss": avg_loss,
                "modes": {
                    m: {k: v for k, v in mdata.items() if k != "results"}
                    for m, mdata in epoch_results.items()
                },
                "auc_map": auc_map_results,
            }, f, indent=2)
        self._print(f"[SPRInT Val] Saved → {out_path}")

        self.model.train()

        return {
            "epoch":    epoch,
            "avg_loss": avg_loss,
            "modes": {
                m: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"]}
                for m, v in epoch_results.items()
            },
            "auc_map": auc_map_results,
        }

    # ── Inference entry point (thin; mirrors ICI inference.py) ────────────────

    def run_inference(self, *, modes, mode_mappings, strategy=None):
        """
        Single evaluation entry point for standalone inference (0-shot).

        Mirrors ICI's inference.py: sprint_eval.py loads the model/checkpoint,
        builds this manager (with eval_data_path = the test split and
        max_val_samples = 0 for the full split), calls run_inference(), and writes
        the results JSON. ALL scoring runs through the SAME training methods —
        _run_colon_auc (colon) and _run_multilabel_auc_map (chest/endo) — that
        run_comprehensive_validation uses; this method only adds the per-mode loop
        and collects their per-sample records (return_details=True). There is no
        separate inference scorer.

        Returns:
            all_modes = {mode: {"metrics": {...}, "results": [...], "mapping": {...}}}

        consumed verbatim by sprint_eval._save_multimode_results (unchanged JSON).
        """
        self.model.eval()

        # ── Symbol mapping diagnostic (mirrors old sprint_eval1.py) ──────────
        print("=" * 70)
        print(f"[SPRINT EVAL] === SYMBOL MAPPING DIAGNOSTIC ===")
        print(f"[SPRINT EVAL] Dataset   : {self._dataset_name}")
        if strategy:
            print(f"[SPRINT EVAL] Strategy  : {strategy}")
        for _m in modes:
            _mp = mode_mappings.get(_m)
            if _mp:
                print(f"[SPRINT EVAL] mode={_m} sym_mappings : {str(_mp)[:120]}")
            else:
                print(f"[SPRINT EVAL] mode={_m} sym_mappings : NONE (regular / dynamic strategy — original labels used)")
        _infer_mode = "per-class binary probe" if self._is_multi_label else "standard logit-AUC"
        print(f"[SPRINT EVAL] Inference mode: {_infer_mode}")
        print("=" * 70)

        all_modes = {}
        for _mode in modes:
            _map = mode_mappings.get(_mode)
            if _mode in ("fixed", "fresh") and not _map:
                print(f"WARNING: mode '{_mode}' requested but mapping is empty "
                      f"(no trained symbols available) — skipping.")
                continue

            print("\n" + "=" * 70)
            print(f"[SPRINT EVAL] === EVALUATION MODE: {_mode} ===")

            if self._is_multi_label:
                # original → no probe symbols; fixed/fresh → symbol-substituted probe.
                metrics, records = self._run_multilabel_auc_map(
                    sym_mappings=(_map or {}), return_details=True
                )
            else:
                # colon: original → bare 0/1 tokens; fixed/fresh → symbol tokens + prompt.
                metrics, records = self._run_colon_auc(
                    mapping=_map, return_details=True
                )

            all_modes[_mode] = {"metrics": metrics, "results": records,
                                "mapping": _map or {}}

        self.model.train()
        return all_modes

    # ── End-of-training summaries ─────────────────────────────────────────────

    def print_consolidated_summaries(
        self,
        epoch_history: list,
        primary_metric: str,
        best_epoch: int,
        best_score: float,
    ):
        """Print Consolidated Validation Summary and Complete Training Summary."""
        prim = primary_metric
        ds   = self._dataset_name

        # Consolidated Validation Summary — all epochs × modes
        self._print(f"\n{'='*16} Consolidated Validation Summary {'='*16}")
        self._print(
            f"{'Epoch':<12} | {'Mode':<12} | {'macro_f1':<10} | {'accuracy':<10} | {prim}"
        )
        self._print(f"{'-'*70}")
        for entry in epoch_history:
            am       = entry.get("auc_map", {})
            prim_val = am.get(prim, am.get("macro_auc", am.get("AUC", float("nan"))))
            prim_str = (
                f"{prim_val:.4f}"
                if isinstance(prim_val, float) and not math.isnan(prim_val)
                else "  n/a"
            )
            for mode, m in entry["modes"].items():
                auc_col   = prim_str if mode == "original" else "  —"
                best_star = " ★" if entry["epoch"] == best_epoch and mode == "original" else ""
                self._print(
                    f"{entry['epoch']:<12} | {mode:<12} | {m['macro_f1']:<10.4f}"
                    f" | {m['accuracy']:<10.4f} | {auc_col}{best_star}"
                )
        self._print(f"{'='*70}")

        # Complete Training Summary — one row per epoch, original mode
        self._print(f"\n{'='*110}")
        self._print(f"COMPLETE TRAINING SUMMARY - ALL EPOCHS")
        self._print(f"{'='*110}")
        self._print(
            f"{'Epoch':<8} {'Loss':<12} {'macro_f1':<12} {'accuracy':<12} {prim:<12} {'Best?':<6} Mode"
        )
        self._print(f"{'-'*110}")
        for entry in epoch_history:
            f1  = entry["modes"].get("original", {}).get("macro_f1", 0.0)
            acc = entry["modes"].get("original", {}).get("accuracy", 0.0)
            am  = entry.get("auc_map", {})
            prim_val = am.get(prim, am.get("macro_auc", am.get("AUC", float("nan"))))
            prim_str = (
                f"{prim_val:.4f}"
                if isinstance(prim_val, float) and not math.isnan(prim_val)
                else "  n/a"
            )
            best_mark = "★" if entry["epoch"] == best_epoch else " "
            self._print(
                f"{entry['epoch']:<8} {entry['avg_loss']:<12.4f}"
                f" {f1:<12.4f} {acc:<12.4f} {prim_str:<12} {best_mark:<6} original"
            )
        self._print(f"{'='*110}")
        if best_epoch is not None:
            self._print(
                f"Best: epoch {best_epoch}/{len(epoch_history)}"
                f"  {prim}={best_score:.4f}  (original mode)"
            )
        self._print("Training completed successfully")
