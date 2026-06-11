"""
Shared MedFMC prompt construction for the per-class mAP/AUC probe.

HVB philosophy (the reason this module exists)
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
