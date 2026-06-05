"""
TOKEN-ID LEVEL VERIFICATION of symbol substitution in our VLM training pipeline.

What this proves
----------------
Your advisor's concern: logs say "symbols replaced" (text level) but the
actual `labels` tensor fed to the loss might still hold ORIGINAL label IDs.
This script checks that claim at the token-ID level — using the REAL
SymbolManager, the REAL LLaVA tokenizer, and an inlined copy of the exact
preprocess_v1 substitution + tokenisation + masking logic.

It runs the pipeline four times on ONE sample (label = "1"):
  RFT   — no SymbolManager
  SS-FT — fixed two-token symbols
  LF-FT — label swap
  ID-FT — fresh per-instance symbols
…and prints the supervised label IDs for each.  All four must differ from RFT.

Why NOT import from llava.train.train?
--------------------------------------
train.py has a module-level import of LLaVATrainer which pulls the chain:
  transformers.Trainer -> peft -> accelerate.utils.memory.clear_device_cache
That name is missing from the accelerate version installed on this cluster,
causing an ImportError before any user code can run.
This script side-steps that entire chain by importing ONLY:
  • llava.conversation  (stdlib + PIL — no heavy deps)
  • llava.constants     (plain constants, zero imports)
  • models.symbolAdapter.symbol_manager  (no LLaVATrainer chain)
  • transformers.AutoTokenizer
  • torch
The substitution + tokenisation + masking logic is inlined verbatim from
llava/train/train.py lines 453-535 (preprocess_v1, has_image=False path).
The proof is valid because the inlined code IS the production code path.

How to run on cluster
---------------------
    cd /home/harinis/LLaVA
    conda activate llava
    python sprint_vision/verify_symbol_token_ids.py

Expected: "ALL PASS"

If it prints FAIL -> do NOT start training.
"""

import os
import sys
import glob
import copy

from packaging import version

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at:  .../LLaVA/sprint_vision/verify_symbol_token_ids.py
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../sprint_vision
LLAVA_ROOT = os.path.dirname(THIS_DIR)                    # .../LLaVA
sys.path.insert(0, LLAVA_ROOT)
sys.path.insert(0, THIS_DIR)

# llava.conversation only needs stdlib + PIL — safe, no LLaVATrainer chain.
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX

# SymbolManager lives in sprint_vision/models/ — no heavy deps.
from models.symbolAdapter.symbol_manager import SymbolManager

import torch
import tokenizers as _tokenizers
from transformers import AutoTokenizer

# Reproduce the same flag used in train.py line 68.
IS_TOKENIZER_GREATER_THAN_0_14 = (
    version.parse(_tokenizers.__version__) >= version.parse("0.14")
)


# ── Tokenizer resolution ─────────────────────────────────────────────────────

def resolve_tokenizer_path() -> str:
    env_path = os.environ.get("LLAVA_TOKENIZER_PATH")
    if env_path:
        return env_path
    snapshot_glob = os.path.expanduser(
        "~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots/*"
    )
    matches = sorted(glob.glob(snapshot_glob))
    if matches:
        return matches[0]
    return "hf-internal-testing/llama-tokenizer"


# ── Inlined preprocess_v1 (has_image=False path) ─────────────────────────────
# Verbatim from llava/train/train.py lines 453-535.
# Only the has_image=False branch is used so we avoid tokenizer_image_token.

def preprocess_v1_inline(sources, tokenizer, symbol_manager=None):
    """
    Inlined preprocess_v1 (has_image=False path only).
    Applies symbol substitution, builds the vicuna_v1 prompt, tokenises,
    and masks the human turn to IGNORE_INDEX exactly as the real function does.
    Returns dict(input_ids=Tensor, labels=Tensor).
    """
    # ── SPRInT substitution — verbatim lines 453-459 ──
    if symbol_manager is not None:
        mappings = symbol_manager.get_current_symbols()
        if mappings:
            for source in sources:
                for sentence in source:
                    if sentence["from"] in ("gpt", "human"):
                        sentence["value"] = symbol_manager.apply_to_text(
                            sentence["value"], mappings
                        )

    # ── Build conversation string — verbatim lines 462-477 ──
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # ── Tokenise — verbatim lines 483-490 (has_image=False branch) ──
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    # ── Mask human turns to IGNORE_INDEX — verbatim lines 494-535 ──
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len      = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # verbatim line 520-522
            if i != 0 and not getattr(tokenizer, "legacy", True) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len      -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenisation mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(input_ids=input_ids, labels=targets)


# ── Sample ────────────────────────────────────────────────────────────────────
# Stripped of the <image> token — we use has_image=False, so the image token
# would be tokenised as plain text and add noise to the output.  The label
# "1" at the answer position is what we are checking; the rest of the prompt
# just provides context for the masking logic.

SAMPLE = {
    "id": "colon_0001",
    "conversations": [
        {
            "from": "human",
            "value": "Is this colon image showing malignant tissue? (0 for No, 1 for Yes)",
        },
        {"from": "gpt", "value": "1"},
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_supervised_ids(out):
    labels = out["labels"]
    mask   = labels[0] != IGNORE_INDEX
    return labels[0][mask].tolist()


def run_one(name, tokenizer, symbol_manager):
    print(f"\n{'=' * 72}")
    print(f" {name}")
    print(f"{'=' * 72}")
    if symbol_manager is None:
        print(" Strategy        : RFT (no SymbolManager)")
        print(" Active mapping  : (none)")
    else:
        print(f" Active mapping  : {symbol_manager.get_current_symbols()}")

    sources = copy.deepcopy([SAMPLE["conversations"]])
    out     = preprocess_v1_inline(sources, tokenizer, symbol_manager)

    print(f" Human turn (after sub): {sources[0][0]['value']!r}")
    print(f" GPT   turn (after sub): {sources[0][1]['value']!r}")

    sup = get_supervised_ids(out)
    print(f" Supervised token IDs  : {sup}")
    print(f" Per-token decode      : {[tokenizer.decode([t]) for t in sup]}")
    print(f" Joined decode         : {tokenizer.decode(sup)!r}")
    return sup


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tok_path = resolve_tokenizer_path()
    print(f"Tokenizer source : {tok_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, model_max_length=2048, padding_side="right", use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    print(f"Vocab size       : {tokenizer.vocab_size}")
    print(f"IS_TOKENIZER_GREATER_THAN_0_14 : {IS_TOKENIZER_GREATER_THAN_0_14}")

    print(f"\nRaw label token IDs (no context):")
    print(f"  '0' -> {tokenizer.encode('0', add_special_tokens=False)}")
    print(f"  '1' -> {tokenizer.encode('1', add_special_tokens=False)}")

    # Set vicuna_v1 template — must happen before calling preprocess_v1_inline.
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    rft_ids  = run_one("RFT  (no substitution)",          tokenizer, None)

    ss_mgr = SymbolManager(
        original_labels=["0", "1"], tokenizer=tokenizer,
        dynamic_per_epoch=False, symbol_type="two_token",
    )
    ssft_ids = run_one("SS-FT (fixed two-token symbols)", tokenizer, ss_mgr)

    lf_mgr = SymbolManager(
        original_labels=["0", "1"], tokenizer=tokenizer, swap_labels=True,
    )
    lfft_ids = run_one("LF-FT (label swap)",              tokenizer, lf_mgr)

    try:
        id_mgr = SymbolManager(
            original_labels=["0", "1"], tokenizer=tokenizer,
            dynamic_per_instance=True, symbol_type="two_token",
        )
        idft_ids = run_one("ID-FT (per-instance fresh syms)", tokenizer, id_mgr)
        id_ft_available = True
    except TypeError as e:
        print(f"\n[SKIP] ID-FT: SymbolManager on this cluster does not support")
        print(f"       'dynamic_per_instance' ({e}).")
        print(f"       ACTION REQUIRED: SCP the updated symbol_manager.py to cluster")
        print(f"       before running any ID-FT training jobs.")
        idft_ids = None
        id_ft_available = False

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(" VERDICT — does substitution reach the token-ID level?")
    print(f"{'=' * 72}")

    cases = [
        ("RFT   vs SS-FT", rft_ids,  ssft_ids),
        ("RFT   vs LF-FT", rft_ids,  lfft_ids),
        ("SS-FT vs LF-FT", ssft_ids, lfft_ids),
    ]
    if id_ft_available:
        cases += [
            ("RFT   vs ID-FT", rft_ids,  idft_ids),
            ("SS-FT vs ID-FT", ssft_ids, idft_ids),
        ]

    all_ok = True
    for label, a, b in cases:
        differ = a != b
        flag   = "PASS" if differ else "FAIL"
        print(f"  [{flag}] {label}")
        print(f"          A IDs = {a}")
        print(f"          B IDs = {b}")
        if not differ:
            all_ok = False

    if not id_ft_available:
        print(f"\n  [SKIP] RFT   vs ID-FT  (symbol_manager.py outdated — see above)")
        print(f"  [SKIP] SS-FT vs ID-FT  (symbol_manager.py outdated — see above)")

    print()
    if all_ok and id_ft_available:
        print(" ALL PASS -- every symbol strategy produces a `labels` tensor whose")
        print(" loss-supervised positions hold DIFFERENT token IDs from RFT.")
        print(" The bug your advisor warned about in ICI's symbol_training.py")
        print(" does NOT exist in our VLM preprocess_v1.  Safe to train.")
    elif all_ok and not id_ft_available:
        print(" PASS (RFT/SS-FT/LF-FT) -- substitution reaches token-ID level.")
        print(" ID-FT skipped: SCP updated symbol_manager.py before ID-FT training.")
        print(" RFT and SS-FT training are safe to start now.")
    else:
        print(" FAIL -- at least one strategy produces the same supervised IDs as")
        print(" another.  DO NOT START TRAINING.  Investigate before launching.")
    print(f"{'=' * 72}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
