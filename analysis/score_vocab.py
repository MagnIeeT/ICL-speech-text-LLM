#!/usr/bin/env python3
"""
Precompute H2 (base log-probability) scores for the full 2-token symbol pool.

Scores every valid 2-token symbol under the base Qwen model (no LoRA, no audio)
using the neutral prompt "The answer is: ". Stores per-symbol:
  - lp_tok1           : log P(tok1)          — high value → H4 collapse risk
  - lp_tok2_given_tok1: log P(tok2 | tok1)
  - lp_total          : lp_tok1 + lp_tok2    — primary H2 quality metric
  - tokens            : [tok1_str, tok2_str]

Output is sorted by lp_total descending (easiest symbols first) so callers can
slice [: easy_k] for easy mode or [-hard_k :] for hard mode.

Run once and commit the output — the pool is deterministic for a fixed tokenizer.

Usage:
    conda run -n qwen python analysis/score_vocab.py [--pool_size N] [--out PATH]
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.symbolAdapter.symbol_manager import generate_fresh_symbols
from transformers import AutoTokenizer, Qwen2AudioForConditionalGeneration

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "Qwen/Qwen2-Audio-7B-Instruct"
NEUTRAL_PROMPT  = "The answer is: "
DEFAULT_POOL    = 3000   # generate more than needed; filter keeps the good ones
DEFAULT_OUT     = os.path.join(os.path.dirname(__file__), "vocab_h2_scores.json")

# Empirical thresholds from embedding_distances.tsv analysis
# ep4_fresh voxceleb (best F1=0.558): avg_base_lp ≈ -33.8
# ep4_fixed voxceleb (worst F1=0.053): avg_base_lp ≈ -42.4
# H4 collapse observed when lp_tok1 > ~-2 (e.g. 'n', 'a', single chars)
REFERENCE = {
    "easy_lp_total_min":  -36.0,   # symbols above this → "easy" mode
    "hard_lp_total_max":  -40.0,   # symbols below this → "hard" mode
    "h4_lp_tok1_max":     -2.0,    # first token too common → collapse risk
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── scoring ───────────────────────────────────────────────────────────────────

def score_symbols(model, tokenizer, symbols: list[str], device: str,
                  batch_size: int = 64) -> dict:
    """
    Returns {symbol: {lp_tok1, lp_tok2_given_tok1, lp_total, h4_risk, tokens}} for every symbol.

    Two-phase batched scoring:
      Phase 1 — single forward pass after neutral prompt → P(tok1) for all symbols at once.
      Phase 2 — chunked batched forward pass: each row is [prompt + tok1_id], all rows
                 same length → one batch gets P(tok2|tok1) for `batch_size` symbols at once.
                 Total passes = ceil(N / batch_size) instead of N.
    """
    prompt_ids = tokenizer(
        NEUTRAL_PROMPT, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)                          # (1, prompt_len)
    prompt_len = prompt_ids.shape[1]

    # ── phase 1: P(tok1) — single forward pass ────────────────────────────────
    log.info("  Phase 1: single forward pass → P(tok1) for all symbols ...")
    with torch.no_grad():
        out = model.language_model(input_ids=prompt_ids)
        lp_step1 = F.log_softmax(out.logits[0, -1, :], dim=-1)  # (vocab,)
    log.info("  Phase 1 done.")

    # ── tokenise all symbols, collect (tok1_id, tok2_id) pairs ───────────────
    valid = []   # [(sym, tok1_id, tok2_id, tok1_str, tok2_str)]
    for sym in symbols:
        ids = tokenizer.encode(sym, add_special_tokens=False)
        if len(ids) < 2:
            continue
        tok1_id, tok2_id = ids[0], ids[1]
        valid.append((
            sym, tok1_id, tok2_id,
            tokenizer.convert_ids_to_tokens(tok1_id),
            tokenizer.convert_ids_to_tokens(tok2_id),
        ))

    # ── phase 2: P(tok2|tok1) — batched forward passes ───────────────────────
    total   = len(valid)
    n_batch = (total + batch_size - 1) // batch_size
    log.info(
        "  Phase 2: %d symbols in %d batches of up to %d ...",
        total, n_batch, batch_size,
    )
    t0 = time.time()

    # Build (1, prompt_len+1) template once; swap last token per batch
    ext_template = torch.cat(
        [prompt_ids, torch.zeros(1, 1, dtype=torch.long, device=device)], dim=1
    )  # (1, prompt_len+1)

    results = {}
    for b_idx in range(n_batch):
        chunk = valid[b_idx * batch_size : (b_idx + 1) * batch_size]
        tok1_ids = [c[1] for c in chunk]
        tok2_ids = [c[2] for c in chunk]

        # Build batch: repeat prompt and fill last position with each tok1
        batch_input = ext_template.expand(len(chunk), -1).clone()   # (B, prompt_len+1)
        batch_input[:, -1] = torch.tensor(tok1_ids, device=device)

        with torch.no_grad():
            out2 = model.language_model(input_ids=batch_input)
            # logits: (B, prompt_len+1, vocab) — take last position
            lp_step2 = F.log_softmax(out2.logits[:, -1, :], dim=-1)  # (B, vocab)

        for j, (sym, tok1_id, tok2_id, tok1_str, tok2_str) in enumerate(chunk):
            lp1 = lp_step1[tok1_id].item()
            lp2 = lp_step2[j, tok2_id].item()
            results[sym] = {
                "lp_tok1":            round(lp1, 4),
                "lp_tok2_given_tok1": round(lp2, 4),
                "lp_total":           round(lp1 + lp2, 4),
                "h4_risk":            lp1 >= REFERENCE["h4_lp_tok1_max"],
                "tokens":             [tok1_str, tok2_str],
                "token_ids":          [tok1_id, tok2_id],
            }

        # Progress log every 5 batches
        done = min((b_idx + 1) * batch_size, total)
        if (b_idx + 1) % 5 == 0 or done == total:
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done)
            log.info(
                "  [%d/%d]  batch %d/%d  %.0fs elapsed  ~%.0fs remaining",
                done, total, b_idx + 1, n_batch, elapsed, eta,
            )

    return results


# ── summary stats ─────────────────────────────────────────────────────────────

def print_summary(scored: dict):
    lp_totals = [v["lp_total"] for v in scored.values()]
    lp_tok1s  = [v["lp_tok1"]  for v in scored.values()]

    easy = sum(1 for v in scored.values()
               if v["lp_total"] >= REFERENCE["easy_lp_total_min"]
               and v["lp_tok1"] < REFERENCE["h4_lp_tok1_max"])
    hard = sum(1 for v in scored.values()
               if v["lp_total"] <= REFERENCE["hard_lp_total_max"]
               and v["lp_tok1"] < REFERENCE["h4_lp_tok1_max"])
    h4_risk = sum(1 for v in scored.values()
                  if v["lp_tok1"] >= REFERENCE["h4_lp_tok1_max"])

    log.info("")
    log.info("── Pool summary ──────────────────────────────────────")
    log.info("  Total scored        : %d", len(scored))
    log.info("  lp_total  range     : [%.2f, %.2f]",
             min(lp_totals), max(lp_totals))
    log.info("  lp_tok1   range     : [%.2f, %.2f]",
             min(lp_tok1s), max(lp_tok1s))
    log.info("  'easy' symbols      : %d  (lp_total >= %.1f, lp_tok1 < %.1f)",
             easy, REFERENCE["easy_lp_total_min"], REFERENCE["h4_lp_tok1_max"])
    log.info("  'hard' symbols      : %d  (lp_total <= %.1f, lp_tok1 < %.1f)",
             hard, REFERENCE["hard_lp_total_max"], REFERENCE["h4_lp_tok1_max"])
    log.info("  H4 collapse-risk    : %d  (lp_tok1 >= %.1f)",
             h4_risk, REFERENCE["h4_lp_tok1_max"])
    log.info("──────────────────────────────────────────────────────")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score symbol pool by H2 base log-probability")
    parser.add_argument("--pool_size", type=int, default=DEFAULT_POOL,
                        help="Number of valid 2-token symbols to generate and score")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output JSON path (default: analysis/vocab_h2_scores.json)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model path or ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    log.info("═" * 60)
    log.info("  Symbol Pool H2 Scorer")
    log.info("  Model  : %s", args.model)
    log.info("  Device : %s", args.device)
    log.info("  Pool   : %d symbols", args.pool_size)
    log.info("  Output : %s", args.out)
    log.info("═" * 60)

    # ── step 1: generate pool ─────────────────────────────────────────────────
    log.info("[1/3] Loading tokenizer and generating %d valid 2-token symbols ...",
             args.pool_size)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    symbols = generate_fresh_symbols(tok, args.pool_size, token_size=2)
    log.info("  Generated %d symbols", len(symbols))

    # ── step 2: load model ────────────────────────────────────────────────────
    log.info("[2/3] Loading base model (float16, no LoRA) ...")
    t_load = time.time()
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("  Model loaded in %.1fs", time.time() - t_load)

    # ── step 3: score ─────────────────────────────────────────────────────────
    log.info("[3/3] Scoring symbols ...")
    scored = score_symbols(model, tok, symbols, args.device)
    log.info("  Scored %d symbols successfully", len(scored))

    del model
    torch.cuda.empty_cache()
    log.info("  Model unloaded (VRAM freed)")

    # ── sort: safe symbols by lp_total desc, H4-risk symbols appended last ───
    # Slicing [:k] = easy+safe, [-k:] (before h4 group) = hard+safe.
    # H4-risk symbols are never in the easy/hard slices.
    safe = {s: v for s, v in scored.items() if not v["h4_risk"]}
    h4   = {s: v for s, v in scored.items() if     v["h4_risk"]}
    sorted_scored = dict(
        sorted(safe.items(), key=lambda kv: kv[1]["lp_total"], reverse=True)
    )
    sorted_scored.update(
        sorted(h4.items(), key=lambda kv: kv[1]["lp_total"], reverse=True)
    )
    log.info("  Safe symbols: %d  |  H4-risk (appended last): %d", len(safe), len(h4))

    print_summary(sorted_scored)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "meta": {
            "model":       args.model,
            "prompt":      NEUTRAL_PROMPT,
            "pool_size":   len(sorted_scored),
            "thresholds":  REFERENCE,
            "note": (
                "Sorted easiest-first (lp_total desc). "
                "Slice [:k] for easy mode, [-k:] for hard mode "
                "after applying h4_lp_tok1_max filter."
            ),
        },
        "symbols": sorted_scored,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    log.info("Saved → %s  (%d symbols)", args.out, len(sorted_scored))


if __name__ == "__main__":
    main()
