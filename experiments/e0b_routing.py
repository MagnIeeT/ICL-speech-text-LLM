#!/usr/bin/env python3
"""E0b-1: does the definition decide which evidence the model uses?

Chapter 2, step two. Post-hoc only: reads prediction dumps that inference.py
already writes.

Four passes per checkpoint, all on the same items:

    speech_only + orig          -> P_speech   (also the 'neutral' condition)
    text_only   + orig          -> P_text     (what the words imply, per the model)
    speech_only + route_tone    -> answer when the legend says tone decides
    speech_only + route_words   -> answer when the legend says words decide

Everything is speech_only except the reference pass, so the audio carries both
channels and the model must select between them itself. The legend is the only
thing that changes between the two routed conditions.

Two measurements:

1. ROUTING GAP, on the subset where the modalities disagree (P_speech != P_text).
   Those are the only items where routing can show up at all.

       gap = P(answer == P_speech | route_tone) - P(answer == P_speech | route_words)

   Near zero means the definition does not control evidence selection.

2. The CONTROL, on the subset where they agree (P_speech == P_text). Here the
   two legends have nothing to arbitrate, so a well-behaved model should answer
   identically. If the answers diverge, the routing clause is perturbing the
   model generally rather than routing evidence, and measurement 1 is suspect.
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List


def load(path: str, mode: str, dataset: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: predictions file not found: {path}")
    with open(path) as handle:
        records = json.load(handle)
    rows = [r for r in records
            if r.get("validation_mode") == mode and r.get("dataset_type") == dataset]
    if not rows:
        modes = Counter(r.get("validation_mode") for r in records)
        dsets = Counter(r.get("dataset_type") for r in records)
        sys.exit(f"ERROR: nothing with mode={mode!r} dataset={dataset!r} in {path}\n"
                 f"       modes: {dict(modes)}\n       datasets: {dict(dsets)}")
    return rows


def norm(label: Any) -> frozenset:
    """Labels compare as sets, so multi-label tasks work unchanged."""
    if label is None:
        return frozenset()
    return frozenset(p.strip().lower() for p in str(label).split(",") if p.strip())


def check_aligned(named: Dict[str, List[Dict[str, Any]]]) -> None:
    """Position pairing is only valid if every pass saw the same items in order."""
    lengths = {k: len(v) for k, v in named.items()}
    if len(set(lengths.values())) != 1:
        sys.exit(f"ERROR: passes have different lengths: {lengths}\n"
                 "       Re-run them with identical --max_val_samples and --split.")
    ref_name, ref = next(iter(named.items()))
    for name, rows in named.items():
        for i, (a, b) in enumerate(zip(ref, rows)):
            if norm(a.get("true_label")) != norm(b.get("true_label")):
                sys.exit(f"ERROR: gold labels diverge at index {i} between "
                         f"{ref_name!r} and {name!r}:\n"
                         f"       {a.get('true_label')!r} vs {b.get('true_label')!r}\n"
                         "       Records are not aligned; pairing by position is invalid.")


def pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:5.1f}%" if den else "    n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="E0b-1 routing gap analysis.")
    ap.add_argument("--speech_orig", required=True, help="*_p.json: speech_only + orig")
    ap.add_argument("--text_orig", required=True, help="*_p.json: text_only + orig")
    ap.add_argument("--speech_tone", required=True, help="*_p.json: speech_only + route_tone")
    ap.add_argument("--speech_words", required=True, help="*_p.json: speech_only + route_words")
    ap.add_argument("--dataset", default="meld_emotion",
                    help="dataset_type to analyse (default: meld_emotion)")
    ap.add_argument("--mode", default="original", help="validation_mode (default: original)")
    ap.add_argument("--out", default=None, help="write per-item detail here (JSON)")
    args = ap.parse_args()

    passes = {
        "speech_orig": load(args.speech_orig, args.mode, args.dataset),
        "text_orig": load(args.text_orig, args.mode, args.dataset),
        "speech_tone": load(args.speech_tone, args.mode, args.dataset),
        "speech_words": load(args.speech_words, args.mode, args.dataset),
    }
    check_aligned(passes)
    n = len(passes["speech_orig"])

    disagree, agree, detail = [], [], []
    for i in range(n):
        gold = norm(passes["speech_orig"][i].get("true_label"))
        p_sp = norm(passes["speech_orig"][i].get("predicted_label"))
        p_tx = norm(passes["text_orig"][i].get("predicted_label"))
        a_tone = norm(passes["speech_tone"][i].get("predicted_label"))
        a_word = norm(passes["speech_words"][i].get("predicted_label"))
        row = dict(i=i, gold=gold, p_sp=p_sp, p_tx=p_tx, tone=a_tone, words=a_word)
        (disagree if p_sp != p_tx else agree).append(row)
        detail.append({
            "index": i,
            "text": passes["speech_orig"][i].get("text"),
            "true_label": passes["speech_orig"][i].get("true_label"),
            "P_speech": passes["speech_orig"][i].get("predicted_label"),
            "P_text": passes["text_orig"][i].get("predicted_label"),
            "route_tone": passes["speech_tone"][i].get("predicted_label"),
            "route_words": passes["speech_words"][i].get("predicted_label"),
            "modalities_disagree": p_sp != p_tx,
        })

    print(f"\nE0b-1 routing gap   dataset={args.dataset}  mode={args.mode}")
    print(f"  items={n}   modalities disagree on {len(disagree)} ({pct(len(disagree), n)})"
          f"   agree on {len(agree)}\n")

    # ── 1. the measurement, on the disagreement subset ───────────────────────
    d = len(disagree)
    if d == 0:
        print("  No disagreement items: routing cannot be measured on this set.")
    else:
        tone_sp = sum(r["tone"] == r["p_sp"] for r in disagree)
        word_sp = sum(r["words"] == r["p_sp"] for r in disagree)
        tone_tx = sum(r["tone"] == r["p_tx"] for r in disagree)
        word_tx = sum(r["words"] == r["p_tx"] for r in disagree)
        tone_other = sum(r["tone"] != r["p_sp"] and r["tone"] != r["p_tx"] for r in disagree)
        word_other = sum(r["words"] != r["p_sp"] and r["words"] != r["p_tx"] for r in disagree)

        print(f"  ROUTING, on the {d} disagreement items")
        print(f"  {'legend':<14}{'= P_speech':>12}{'= P_text':>12}{'neither':>10}")
        print("  " + "-" * 48)
        print(f"  {'route_tone':<14}{pct(tone_sp, d):>12}{pct(tone_tx, d):>12}{pct(tone_other, d):>10}")
        print(f"  {'route_words':<14}{pct(word_sp, d):>12}{pct(word_tx, d):>12}{pct(word_other, d):>10}")
        gap_sp = 100.0 * (tone_sp - word_sp) / d
        gap_tx = 100.0 * (word_tx - tone_tx) / d
        print(f"\n  routing gap (toward speech) = {gap_sp:+.1f} points")
        print(f"  routing gap (toward text)   = {gap_tx:+.1f} points")
        print("  Both should be positive and similar if the legend controls evidence.")

    # ── 2. the control, on the agreement subset ──────────────────────────────
    a = len(agree)
    if a:
        diverged = sum(r["tone"] != r["words"] for r in agree)
        moved_tone = sum(r["tone"] != r["p_sp"] for r in agree)
        moved_word = sum(r["words"] != r["p_sp"] for r in agree)
        print(f"\n  CONTROL, on the {a} agreement items (nothing to arbitrate)")
        print(f"    route_tone vs route_words differ : {pct(diverged, a)}  <- should be near zero")
        print(f"    route_tone moved off P_speech    : {pct(moved_tone, a)}")
        print(f"    route_words moved off P_speech   : {pct(moved_word, a)}")
        if a and 100.0 * diverged / a > 15.0:
            print("    WARNING: the legends diverge even with nothing to arbitrate, so the")
            print("    routing clause is perturbing the model generally. Treat the routing")
            print("    gap above as unreliable.")

    # ── 3. accuracy against gold, for context ────────────────────────────────
    print(f"\n  ACCURACY vs gold, all {n} items")
    for name, key in [("speech_only + orig", "p_sp"), ("text_only + orig", "p_tx"),
                      ("speech + route_tone", "tone"), ("speech + route_words", "words")]:
        hits = sum(r[key] == r["gold"] for r in disagree + agree)
        print(f"    {name:<22}{pct(hits, n):>8}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(detail, handle, indent=2)
        print(f"\n  Per-item detail -> {args.out}")


if __name__ == "__main__":
    main()
