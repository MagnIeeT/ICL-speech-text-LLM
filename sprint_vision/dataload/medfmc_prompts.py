"""
Shared MedFMC prompt construction for the per-class mAP/AUC probe.

-----------------------------------------------
In ICI, ONE prompt template (with its "- label: definition" block) feeds
training, validation, AND inference, so class definitions are visible wherever
the model is scored (salmon_processor.format_prompt + symbol_manager applied to
the same batch["prompt"] in symbol_training.py:243 and validation.py:180).

Chest/Endo report mAP/AUC, which needs a per-class probability, so they cannot
score by free generation like HVB.  To stay HVB-faithful we keep the EXACT same
definition block the model trained on and only swap the trailing "list all that
apply" instruction for a per-class focusing question.  The whole probe is then
run through the same symbol substitution used in training (apply_to_text), so:

  - definitions are visible in the probe (HVB property restored), and
  - symbols (SS-FT) are applied to the block AND the question, exactly as in
    train.py:preprocess_v1:504-506.

mAP/AUC math is untouched: the caller still reads P(Yes) from the Yes/No answer
logits.  Only the prompt text changes.

This single builder is imported by BOTH evaluation paths so they cannot drift:
  - sprint_vision/sprint_eval.py            :: _build_per_class_prompt   (inference)
  - sprint_vision/models/symbolAdapter/validation.py :: _run_multilabel_auc_map (training-time val)
"""

import logging
from typing import Callable, Dict, Optional

from config.data_config import render_def_block

# Modality phrase per dataset, used in the focusing question.
MODALITY = {
    "chest": "chest X-ray",
    "endo":  "endoscopy image",
}


def get_modality(dataset: str) -> str:
    """Return the human-readable modality phrase for a dataset name."""
    for key, phrase in MODALITY.items():
        if key in dataset:
            return phrase
    return "medical image"


def build_per_class_prompt(
    cfg,
    dataset: str,
    class_name: str,
    sym_mappings: Optional[Dict[str, str]] = None,
    apply_to_text_fn: Optional[Callable[[str, Dict[str, str]], str]] = None,
    image_token: str = "<image>",
    include_definitions: bool = True,
) -> str:
    """
    Build the HVB-faithful per-class mAP/AUC probe (human turn only).

    Structure (before symbol substitution):

        <image>
        {intro}

        - label_0: definition_0
        ...
        - label_k: definition_k

        Focusing only on {class_name}: does this {modality} show {class_name}? Answer Yes or No.

    The definition block above is byte-identical to the training instruction's
    block (both come from config.render_def_block over the same intro/labels/
    definitions).  When sym_mappings is provided (SS-FT) and apply_to_text_fn is
    given, symbols are substituted over the WHOLE human turn — block AND focusing
    question — so every label token (including the queried class_name) becomes
    its symbol, exactly mirroring training.

    Args:
        cfg:              DatasetConfig with instruction_intro, label_names,
                          class_definitions (chest/endo).
        dataset:          "chest" or "endo" (drives the modality phrase).
        class_name:       Canonical label being queried (e.g. "pleural_effusion").
        sym_mappings:     {label: symbol} for SS-FT, else None/empty for
                          regular/ed_ft/id_ft/lf_ft (original labels at eval).
        apply_to_text_fn: SymbolManager.apply_to_text (or any (text, mapping)->text).
        image_token:      Image placeholder token.
        include_definitions: When False, the "- label: definition" block is
                          omitted (only intro is dropped too) and just the
                          focusing question remains.  Used ONLY by the influence
                          ablation in sprint_eval.py to measure how much the def
                          block moves P(Yes); the real eval path keeps it True.

    Returns:
        The human-turn string (NOT wrapped in the vicuna conversation; the
        caller appends it to conv_templates["vicuna_v1"]).
    """
    modality = get_modality(dataset)
    question = (
        f"Focusing only on {class_name}: "
        f"does this {modality} show {class_name}? Answer Yes or No."
    )
    if include_definitions:
        block = render_def_block(
            cfg.instruction_intro, cfg.label_names, cfg.class_definitions
        )
        human = f"{image_token}\n{block}\n\n{question}"
    else:
        human = f"{image_token}\n{question}"

    if sym_mappings and apply_to_text_fn is not None:
        human = apply_to_text_fn(human, sym_mappings)

    return human


def score_probes_pyes(
    prompts,
    tokenizer,
    model,
    image_tensor,
    image_size,
    token_id_neg,
    token_id_pos,
    batch_size: int = 1,
    verify: bool = False,
    verify_tol: float = 5e-3,
    conv_template: str = "vicuna_v1",
):
    """
    Score P(positive) for a list of per-class probe prompts that all share ONE
    image, via left-padded batched 1-token generation.

    Each entry in `prompts` is the human-turn string (NOT yet wrapped in the
    conversation template). All prompts use the same image (`image_tensor`), so
    the image is passed `batch_size` times per sub-batch.

    `batch_size=1` reproduces the original one-call-per-class behaviour EXACTLY
    (same conv wrap, same tokenizer_image_token, same softmax over [neg, pos]
    logits; an all-ones attention mask on a single full sequence is equivalent to
    passing none). `batch_size>1` groups prompts to cut the number of forward
    passes — the returned P(pos) values are identical, so AUC/mAP math is
    unchanged.

    `verify=True` (use on the first image only) scores both unbatched and
    batched and aborts if any P(pos) diverges by more than `verify_tol`, so a
    padding/masking bug can never silently corrupt a multi-hour run.

    Returns: list[float] of length len(prompts).
    """
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    if batch_size < 1:
        batch_size = 1

    # Wrap each probe in the conversation template and tokenize (image token kept).
    seqs = []
    for human in prompts:
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], human)
        conv.append_message(conv.roles[1], None)
        full = conv.get_prompt()
        ids = tokenizer_image_token(full, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        seqs.append(ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    device = model.device

    def _score(bsz: int):
        out_scores = []
        for start in range(0, len(seqs), bsz):
            chunk = seqs[start:start + bsz]
            n = len(chunk)
            maxlen = max(s.shape[0] for s in chunk)
            input_ids = torch.full((n, maxlen), pad_id, dtype=chunk[0].dtype)
            attn_mask = torch.zeros((n, maxlen), dtype=torch.long)
            for r, s in enumerate(chunk):
                L = s.shape[0]
                input_ids[r, maxlen - L:] = s          # LEFT pad (decoder generation)
                attn_mask[r, maxlen - L:] = 1
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            imgs = image_tensor
            if imgs.shape[0] == 1 and n > 1:
                # .contiguous() so the replicated batch dim isn't a stride-0 view
                # (some cuDNN conv kernels reject stride-0 batch dims). Encoding
                # is deterministic, so all n copies yield identical image features.
                imgs = imgs.expand(n, *imgs.shape[1:]).contiguous()
            gen_kwargs = dict(
                attention_mask=attn_mask,
                images=imgs,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # image_size is optional: sprint_eval passes it (anyres-safe); the
            # training-time validation path originally passed none, so omit it
            # when None to reproduce that call exactly.
            if image_size is not None:
                gen_kwargs["image_sizes"] = [image_size] * n
            with torch.inference_mode():
                # PEFT's generate() rejects positional input_ids → pass as `inputs=`
                # (matches the unbatched path in validation.py / sprint_eval.py).
                out = model.generate(inputs=input_ids, **gen_kwargs)
            step0 = out.scores[0]   # [n, vocab]
            for r in range(n):
                pair = torch.stack([step0[r, token_id_neg], step0[r, token_id_pos]])
                out_scores.append(torch.nn.functional.softmax(pair, dim=0)[1].item())
        return out_scores

    if verify and batch_size > 1:
        ref = _score(1)               # exact, one-probe-per-pass (ground truth)
        bat = _score(batch_size)      # fast, batched
        max_diff = max(abs(a - b) for a, b in zip(ref, bat)) if ref else 0.0
        if max_diff > verify_tol:
            # Drift exceeds tolerance → batched is NOT trustworthy on this sample.
            # We ALREADY computed the exact unbatched scores (ref), so return THOSE
            # instead of raising: the run never crashes, no sample is dropped, and
            # the reported number is exact for this image. (tol=5e-3 is calibrated to
            # the MEASURED batched-vs-exact fp noise on this A100+bf16+LLaVA setup:
            # observed ~0.001 (ed_ft ckpt) up to ~0.0033 (two_token ckpt). 5e-3 sits
            # safely ABOVE that noise floor yet well BELOW a real structural bug
            # ~2e-2 (e.g. the old eager-attention padding error ~0.028), so normal
            # fp noise passes batched/consistent while a genuine break still trips
            # this fallback to exact. A tighter tol (e.g. 2e-3) sits INSIDE the noise
            # band → it would flag normal noise on sample 0 yet still batch the rest,
            # which is inconsistent; avoid that.)
            print(
                f"[BATCH-VERIFY] FALLBACK: batched vs unbatched P(pos) max|Δ|="
                f"{max_diff:.4g} > tol={verify_tol} — using EXACT unbatched scores "
                f"for this sample (no crash, no dropped sample).",
                flush=True,
            )
            return ref
        return bat

    return _score(batch_size)
def score_class_probes_kvcache(
    cfg,
    dataset: str,
    label_names_orig,
    tokenizer,
    model,
    image_tensor,
    image_size,
    token_id_neg,
    token_id_pos,
    sym_mappings: Optional[Dict[str, str]] = None,
    apply_to_text_fn: Optional[Callable[[str, Dict[str, str]], str]] = None,
    image_token: str = "<image>",
    conv_template: str = "vicuna_v1",
    first_image: bool = False,
):
    """
    KV-prefix-cached version of the per-class Yes/No probe.

    Runs ONE forward pass over the shared prefix (image + definitions block)
    with use_cache=True, then reuses that cache for every class's short
    trailing question -- instead of recomputing the whole ~1000-token prefix
    N times. P(Yes) is mathematically identical to score_probes_pyes; only
    HOW it's computed changes.

    Safety: for each class, verifies (via exact token-id comparison) that the
    prefix/tail split matches what full re-tokenization would give. If it
    doesn't match for a given class, falls back to the exact slow recompute
    for THAT class only -- never guesses.
    """
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    device = model.device
    modality = get_modality(dataset)
    block = render_def_block(cfg.instruction_intro, cfg.label_names, cfg.class_definitions)
    prefix_text = f"{image_token}\n{block}\n\n"
    if sym_mappings and apply_to_text_fn is not None:
        prefix_text = apply_to_text_fn(prefix_text, sym_mappings)

    # Wrapped with prefix ONLY (no question) -- used only to find the exact
    # character split point below, never tokenized directly.
    conv_prefix_only = conv_templates[conv_template].copy()
    conv_prefix_only.append_message(conv_prefix_only.roles[0], prefix_text)
    conv_prefix_only.append_message(conv_prefix_only.roles[1], None)
    s_prefix_only = conv_prefix_only.get_prompt()

    def _build_full(class_name):
        question = (
            f"Focusing only on {class_name}: "
            f"does this {modality} show {class_name}? Answer Yes or No."
        )
        if sym_mappings and apply_to_text_fn is not None:
            question = apply_to_text_fn(question, sym_mappings)
        human = prefix_text + question
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], human)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    # Find the character split point using the FIRST class -- it's the same
    # split for every class since prefix_text never changes.
    s_full_first = _build_full(label_names_orig[0])
    split = 0
    max_common = min(len(s_prefix_only), len(s_full_first))
    while split < max_common and s_prefix_only[split] == s_full_first[split]:
        split += 1

    prefix_string = s_full_first[:split]
    prefix_ids = tokenizer_image_token(
        prefix_string, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    n_pref = prefix_ids.shape[0]
    if first_image:
        print(f"[KV-CACHE] prefix_ids : {n_pref}", flush=True)

    # ---- ONE forward pass over the shared prefix (this is the whole point) ----
    with torch.inference_mode():
        prefix_out = model(
            input_ids=prefix_ids.unsqueeze(0).to(device),
            images=image_tensor,
            use_cache=True,
        )
    past = prefix_out.past_key_values

    scores = []
    for cls_idx, class_name in enumerate(label_names_orig):
        s_full = _build_full(class_name)
        full_ids = tokenizer_image_token(
            s_full, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if first_image and cls_idx == 0:
            print(f"[KV-CACHE] full_ids   : {full_ids.shape[0]}", flush=True)

        # Safety check -- if this ever fails, fall back to the slow, exact path
        # for this one class instead of guessing.
        if full_ids.shape[0] <= n_pref or not torch.equal(full_ids[:n_pref], prefix_ids):
            gen_kwargs = dict(
                images=image_tensor, do_sample=False, max_new_tokens=1,
                use_cache=True, return_dict_in_generate=True, output_scores=True,
            )
            if image_size is not None:
                gen_kwargs["image_sizes"] = [image_size]
            with torch.inference_mode():
                out = model.generate(inputs=full_ids.unsqueeze(0).to(device), **gen_kwargs)
            step0 = out.scores[0][0]
            pair = torch.stack([step0[token_id_neg], step0[token_id_pos]])
            scores.append(torch.nn.functional.softmax(pair, dim=0)[1].item())
            continue

        tail_ids = full_ids[n_pref:]
        if first_image and cls_idx == 0:
            print(f"[KV-CACHE] tail_ids   : {tail_ids.shape[0]}", flush=True)
        attn_mask = torch.ones((1, n_pref + tail_ids.shape[0]), dtype=torch.long, device=device)

        with torch.inference_mode():
            tail_out = model(
                input_ids=tail_ids.unsqueeze(0).to(device),
                past_key_values=past,
                attention_mask=attn_mask,
                use_cache=True,
            )
        logits = tail_out.logits[0, -1, :]
        pair = torch.stack([logits[token_id_neg], logits[token_id_pos]])
        scores.append(torch.nn.functional.softmax(pair, dim=0)[1].item())

    return scores


def sprint_log(stage: str, **fields) -> None:
    """
    Greppable structured log: `[SPRINT::<STAGE>] key=value ...`.

    Multi-line values (full prompts) are printed on their own indented lines so
    they remain readable.  Uses print(flush=True) — NOT logging.info — because
    the training/inference processes here run with the root logger at WARNING,
    which silently drops logging.info().  print() always reaches stdout (and the
    teed log file), including from DataLoader worker processes.
    """
    head = f"[SPRINT::{stage}]"
    scalars = {k: v for k, v in fields.items() if "\n" not in str(v)}
    blocks  = {k: v for k, v in fields.items() if "\n" in str(v)}
    if scalars:
        print(f"{head} " + "  ".join(f"{k}={v!r}" for k, v in scalars.items()), flush=True)
    else:
        print(head, flush=True)
    for k, v in blocks.items():
        print(f"{head} {k}:\n{v}", flush=True)
