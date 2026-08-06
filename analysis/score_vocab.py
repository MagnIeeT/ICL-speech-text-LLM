#!/usr/bin/env python3
"""
Precompute H2 (base log-probability) scores for the full 2-token symbol pool.

Scores every valid 2-token symbol under a base model (no LoRA, no audio) using
the neutral prompt "The answer is: ". Stores per-symbol:
  - lp_tok1           : log P(tok1)          — high value → H4 collapse risk
  - lp_tok2_given_tok1: log P(tok2 | tok1)
  - lp_total          : lp_tok1 + lp_tok2    — primary H2 quality metric
  - tokens            : [tok1_str, tok2_str]

Output is sorted by lp_total descending (easiest symbols first) so callers can
slice [: easy_k] for easy mode or [-hard_k :] for hard mode.

IMPORTANT — scores are MODEL-SPECIFIC. The absolute lp_total scale differs between
models (Qwen2-Audio vs AF3), so easy/hard are selected by PERCENTILE within this
model's own distribution, not by the old Qwen-calibrated absolute cutoffs.

Usage:
    # Qwen (original)
    conda run -n qwen python analysis/score_vocab.py --model_type qwen
    # AF3 — re-score the SAME symbols as the Qwen file to compare (Phase A)
    conda run -n flamingo python analysis/score_vocab.py --model_type flamingo \
        --symbols_from analysis/vocab_h2_scores.json --out analysis/af3_vocab_scores.json
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.symbolAdapter.symbol_manager import generate_fresh_symbols
from transformers import AutoTokenizer

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = {
    "qwen":     "Qwen/Qwen2-Audio-7B-Instruct",
    "flamingo": os.environ.get("FLAMINGO_MODEL_NAME", "nvidia/audio-flamingo-3-hf"),
}
NEUTRAL_PROMPT  = "The answer is: "
DEFAULT_POOL    = 3000   # generate more than needed; filter keeps the good ones
DEFAULT_OUT     = os.path.join(os.path.dirname(__file__), "vocab_h2_scores.json")

# H4 collapse risk (first token too common) — this is a TOKENIZER property, so it
# transfers across Qwen-family models. lp_total easy/hard cutoffs do NOT transfer
# → selected by percentile below.
H4_LP_TOK1_MAX = -2.0
# Legacy Qwen-calibrated absolute cutoffs — kept only for the comparison report.
LEGACY_QWEN = {"easy_lp_total_min": -36.0, "hard_lp_total_max": -40.0}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_type: str, model_path: str, device: str):
    """Load base model + tokenizer for qwen or flamingo. Lazy imports so each
    conda env only needs its own model class."""
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(model_path, trust_remote_code=True).tokenizer

    if model_type == "qwen":
        from transformers import Qwen2AudioForConditionalGeneration as Cls
    elif model_type == "flamingo":
        from transformers import AudioFlamingo3ForConditionalGeneration as Cls
    else:
        raise ValueError(f"unknown model_type {model_type}")

    model = Cls.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def next_token_logits(model, input_ids):
    """Text-only forward → full-sequence logits. Prefers the .language_model
    submodule (Qwen2Audio); falls back to a text-only pass on the full model (AF3)."""
    lm = getattr(model, "language_model", None)
    if lm is not None:
        try:
            return lm(input_ids=input_ids).logits
        except Exception:
            pass
    return model(input_ids=input_ids, use_cache=False).logits


# ── scoring ───────────────────────────────────────────────────────────────────

def score_symbols(model, tokenizer, symbols, device, batch_size=64) -> dict:
    """Returns {symbol: {lp_tok1, lp_tok2_given_tok1, lp_total, h4_risk, tokens}}."""
    prompt_ids = tokenizer(NEUTRAL_PROMPT, return_tensors="pt",
                           add_special_tokens=True).input_ids.to(device)

    log.info("  Phase 1: single forward pass → P(tok1) for all symbols ...")
    with torch.no_grad():
        logits1 = next_token_logits(model, prompt_ids)
        lp_step1 = F.log_softmax(logits1[0, -1, :].float(), dim=-1)
    log.info("  Phase 1 done.")

    valid = []
    for sym in symbols:
        ids = tokenizer.encode(sym, add_special_tokens=False)
        if len(ids) < 2:
            continue
        valid.append((sym, ids[0], ids[1],
                      tokenizer.convert_ids_to_tokens(ids[0]),
                      tokenizer.convert_ids_to_tokens(ids[1])))

    total = len(valid)
    n_batch = (total + batch_size - 1) // batch_size
    log.info("  Phase 2: %d symbols in %d batches of up to %d ...", total, n_batch, batch_size)
    t0 = time.time()
    ext_template = torch.cat(
        [prompt_ids, torch.zeros(1, 1, dtype=torch.long, device=device)], dim=1)

    results = {}
    for b_idx in range(n_batch):
        chunk = valid[b_idx * batch_size:(b_idx + 1) * batch_size]
        batch_input = ext_template.expand(len(chunk), -1).clone()
        batch_input[:, -1] = torch.tensor([c[1] for c in chunk], device=device)
        with torch.no_grad():
            logits2 = next_token_logits(model, batch_input)
            lp_step2 = F.log_softmax(logits2[:, -1, :].float(), dim=-1)
        for j, (sym, tok1_id, tok2_id, tok1_str, tok2_str) in enumerate(chunk):
            lp1 = lp_step1[tok1_id].item()
            lp2 = lp_step2[j, tok2_id].item()
            results[sym] = {
                "lp_tok1": round(lp1, 4),
                "lp_tok2_given_tok1": round(lp2, 4),
                "lp_total": round(lp1 + lp2, 4),
                "h4_risk": lp1 >= H4_LP_TOK1_MAX,
                "tokens": [tok1_str, tok2_str],
                "token_ids": [tok1_id, tok2_id],
            }
        done = min((b_idx + 1) * batch_size, total)
        if (b_idx + 1) % 5 == 0 or done == total:
            el = time.time() - t0
            log.info("  [%d/%d]  %.0fs elapsed  ~%.0fs remaining",
                     done, total, el, el / done * (total - done))
    return results


# ── selection + summary ───────────────────────────────────────────────────────

def percentile_cutoffs(scored, easy_pct, hard_pct):
    """easy = top easy_pct% lp_total among H4-safe; hard = bottom hard_pct%."""
    safe_lp = np.array([v["lp_total"] for v in scored.values() if not v["h4_risk"]])
    easy_min = float(np.percentile(safe_lp, 100 - easy_pct))
    hard_max = float(np.percentile(safe_lp, hard_pct))
    return easy_min, hard_max


def build_pools(scored, easy_min, hard_max):
    """Dedup-by-first-token easy/hard pools (avoids first-token collisions between
    label symbols). Keeps the most extreme symbol per unique first BPE token."""
    def dedup(cands, prefer_high):
        seen = {}
        for s, v in cands:
            ft = v["tokens"][0]
            cur = seen.get(ft)
            if cur is None or (v["lp_total"] > cur[1]["lp_total"]) == prefer_high:
                seen[ft] = (s, v)
        return [s for s, _ in seen.values()]
    safe = [(s, v) for s, v in scored.items() if not v["h4_risk"]]
    easy = [(s, v) for s, v in safe if v["lp_total"] >= easy_min]
    hard = [(s, v) for s, v in safe if v["lp_total"] <= hard_max]
    return dedup(easy, True), dedup(hard, False)


def print_summary(scored, easy_min, hard_max):
    lp_totals = [v["lp_total"] for v in scored.values()]
    lp_tok1s = [v["lp_tok1"] for v in scored.values()]
    safe = [v for v in scored.values() if not v["h4_risk"]]
    easy = sum(1 for v in safe if v["lp_total"] >= easy_min)
    hard = sum(1 for v in safe if v["lp_total"] <= hard_max)
    h4 = sum(1 for v in scored.values() if v["h4_risk"])
    log.info("")
    log.info("── Pool summary (this model) ─────────────────────────")
    log.info("  Total scored     : %d", len(scored))
    log.info("  lp_total range   : [%.2f, %.2f]", min(lp_totals), max(lp_totals))
    log.info("  lp_tok1  range   : [%.2f, %.2f]", min(lp_tok1s), max(lp_tok1s))
    log.info("  easy cutoff      : lp_total >= %.2f  → %d symbols", easy_min, easy)
    log.info("  hard cutoff      : lp_total <= %.2f  → %d symbols", hard_max, hard)
    log.info("  H4 collapse-risk : %d  (lp_tok1 >= %.1f)", h4, H4_LP_TOK1_MAX)
    log.info("──────────────────────────────────────────────────────")


def compare_report(old_path, new_scored):
    """When re-scoring an existing pool under a new model: how well do the old
    (e.g. Qwen) scores transfer? Correlation + where the old hard/easy slices land."""
    try:
        old = json.load(open(old_path)).get("symbols", {})
    except Exception as e:
        log.warning("  compare skipped (%s)", e)
        return None
    common = [s for s in new_scored if s in old]
    if len(common) < 10:
        log.warning("  compare skipped: only %d common symbols", len(common))
        return None
    o = np.array([old[s]["lp_total"] for s in common])
    n = np.array([new_scored[s]["lp_total"] for s in common])
    pearson = float(np.corrcoef(o, n)[0, 1])
    spearman = float(np.corrcoef(np.argsort(np.argsort(o)), np.argsort(np.argsort(n)))[0, 1])
    # percentile of each common symbol's NEW lp_total
    order = np.argsort(np.argsort(n))
    pct = {common[i]: 100.0 * order[i] / (len(common) - 1) for i in range(len(common))}
    old_hard = [s for s in common
                if old[s]["lp_total"] <= LEGACY_QWEN["hard_lp_total_max"] and not old[s]["h4_risk"]]
    old_easy = [s for s in common
                if old[s]["lp_total"] >= LEGACY_QWEN["easy_lp_total_min"] and not old[s]["h4_risk"]]
    log.info("")
    log.info("── Transfer report: OLD (%s) vs THIS model ──", os.path.basename(old_path))
    log.info("  common symbols        : %d", len(common))
    log.info("  Pearson  r(lp_total)  : %+.3f", pearson)
    log.info("  Spearman r(lp_total)  : %+.3f", spearman)
    if old_hard:
        p = [pct[s] for s in old_hard]
        log.info("  OLD-hard slice (n=%d) → THIS model percentile: median %.0f  [%.0f–%.0f]",
                 len(old_hard), np.median(p), min(p), max(p))
        log.info("     (want LOW percentiles if hardness transfers)")
    if old_easy:
        p = [pct[s] for s in old_easy]
        log.info("  OLD-easy slice (n=%d) → THIS model percentile: median %.0f  [%.0f–%.0f]",
                 len(old_easy), np.median(p), min(p), max(p))
        log.info("     (want HIGH percentiles if easiness transfers)")
    log.info("──────────────────────────────────────────────────────")
    return {"pearson": pearson, "spearman": spearman, "n_common": len(common)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Score symbol pool by H2 base log-probability")
    p.add_argument("--model_type", choices=["qwen", "flamingo"], default="qwen")
    p.add_argument("--model", default=None, help="override HF model path/ID")
    p.add_argument("--pool_size", type=int, default=DEFAULT_POOL)
    p.add_argument("--symbols_from", default=None,
                   help="score the symbols from this existing scores JSON (for cross-model compare)")
    p.add_argument("--easy_pct", type=float, default=20.0, help="top %% lp_total → easy")
    p.add_argument("--hard_pct", type=float, default=20.0, help="bottom %% lp_total → hard")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model_path = args.model or DEFAULT_MODEL[args.model_type]
    log.info("═" * 60)
    log.info("  Symbol Pool H2 Scorer  (model_type=%s)", args.model_type)
    log.info("  Model  : %s", model_path)
    log.info("  Device : %s", args.device)
    log.info("  Output : %s", args.out)
    log.info("═" * 60)

    model, tok = load_model_and_tokenizer(args.model_type, model_path, args.device)

    if args.symbols_from:
        symbols = list(json.load(open(args.symbols_from)).get("symbols", {}).keys())
        log.info("[1/3] Scoring %d symbols loaded from %s", len(symbols), args.symbols_from)
    else:
        log.info("[1/3] Generating %d valid 2-token symbols ...", args.pool_size)
        symbols = generate_fresh_symbols(tok, args.pool_size, token_size=2)
    log.info("  %d symbols to score", len(symbols))

    log.info("[2/3] Scoring ...")
    scored = score_symbols(model, tok, symbols, args.device)
    log.info("  Scored %d symbols", len(scored))
    del model
    torch.cuda.empty_cache()

    # sort easiest-first; H4-risk appended last (never in easy/hard slices)
    safe = {s: v for s, v in scored.items() if not v["h4_risk"]}
    h4 = {s: v for s, v in scored.items() if v["h4_risk"]}
    sorted_scored = dict(sorted(safe.items(), key=lambda kv: kv[1]["lp_total"], reverse=True))
    sorted_scored.update(sorted(h4.items(), key=lambda kv: kv[1]["lp_total"], reverse=True))

    easy_min, hard_max = percentile_cutoffs(sorted_scored, args.easy_pct, args.hard_pct)
    print_summary(sorted_scored, easy_min, hard_max)
    easy_pool, hard_pool = build_pools(sorted_scored, easy_min, hard_max)
    log.info("  Dedup pools: easy_pool=%d  hard_pool=%d", len(easy_pool), len(hard_pool))
    cmp = compare_report(args.symbols_from, sorted_scored) if args.symbols_from else None

    thresholds = {"easy_lp_total_min": round(easy_min, 4),
                  "hard_lp_total_max": round(hard_max, 4),
                  "h4_lp_tok1_max": H4_LP_TOK1_MAX}
    payload = {
        "meta": {
            "model": model_path, "model_type": args.model_type, "prompt": NEUTRAL_PROMPT,
            "pool_size": len(sorted_scored),
            "selection": {"easy_pct": args.easy_pct, "hard_pct": args.hard_pct, **thresholds},
            "thresholds": thresholds,   # legacy key consumed by symbol_manager fallback
            "hard_pool_size_dedup": len(hard_pool), "easy_pool_size_dedup": len(easy_pool),
            "compare_vs": args.symbols_from, "compare": cmp,
            "note": "Sorted easiest-first. easy/hard = percentile-based, model-specific. "
                    "easy_pool/hard_pool are first-token-deduplicated.",
        },
        "symbols": sorted_scored,
        "easy_pool": easy_pool,
        "hard_pool": hard_pool,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved → %s  (%d symbols)", args.out, len(sorted_scored))


if __name__ == "__main__":
    main()
