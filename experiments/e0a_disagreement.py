#!/usr/bin/env python3
"""E0a: measure how often audio-only and transcript-only predictions disagree.

Chapter 2, step one. Post-hoc only: it reads the prediction dumps that
inference.py already writes and touches no pipeline code.

Run the same evaluation twice, changing only the input modality:

    python inference.py --model_type qwen --run_name e0a_meld_speech \\
        --checkpoint_path <ckpt> --dataset_type meld_emotion \\
        --input_mode speech_only --validation_modes original --split test
    python inference.py --model_type qwen --run_name e0a_meld_text \\
        --checkpoint_path <ckpt> --dataset_type meld_emotion \\
        --input_mode text_only   --validation_modes original --split test

then point this script at the two "<run_name>_p.json" files.

Two outputs:
  1. The disagreement rate per dataset, with a 2x2 contingency against the gold
     label. This is the quantitative form of Chapter 1's limitation 3.
  2. The disagreement subset, written to JSON. Those are the items where the two
     modalities lead the model to different answers, and they are the evaluation
     set for E0b, where the routing clause decides which modality wins.

Pairing: the dumps carry no item id, so records are paired by position within a
dataset. That is only valid if both runs saw the same items in the same order,
so the script verifies that the gold-label sequences match and refuses to report
numbers if they do not.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def load_predictions(path: str, mode: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a *_p.json dump, keep one validation mode, group by dataset in order."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: predictions file not found: {path}")
    with open(path) as handle:
        records = json.load(handle)

    modes = Counter(r.get("validation_mode", "?") for r in records)
    if mode not in modes:
        sys.exit(
            f"ERROR: no records with validation_mode={mode!r} in {path}\n"
            f"       modes present: {dict(modes)}"
        )

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("validation_mode") == mode:
            by_dataset[record.get("dataset_type", "?")].append(record)
    return by_dataset


def normalise(label: str) -> frozenset:
    """Labels compare as sets: multi-label tasks (HVB) emit comma-separated lists."""
    if label is None:
        return frozenset()
    parts = (p.strip().lower() for p in str(label).split(","))
    return frozenset(p for p in parts if p)


def check_alignment(ds: str, speech: List[Dict], text: List[Dict]) -> None:
    """Position-pairing is only sound if both runs saw the same items in order."""
    if len(speech) != len(text):
        sys.exit(
            f"ERROR [{ds}]: {len(speech)} speech records vs {len(text)} text records.\n"
            "       The two runs did not evaluate the same items. Re-run both with "
            "identical --max_val_samples and --split."
        )
    for i, (s, t) in enumerate(zip(speech, text)):
        if normalise(s.get("true_label")) != normalise(t.get("true_label")):
            sys.exit(
                f"ERROR [{ds}]: gold labels diverge at index {i}.\n"
                f"       speech run: {s.get('true_label')!r}\n"
                f"       text run:   {t.get('true_label')!r}\n"
                "       Records are not aligned, so pairing by position is invalid."
            )


def compare(ds: str, speech: List[Dict], text: List[Dict]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return summary stats for one dataset plus its disagreement subset."""
    cells = Counter()
    disagreements: List[Dict[str, Any]] = []

    for i, (s, t) in enumerate(zip(speech, text)):
        gold = normalise(s.get("true_label"))
        p_speech = normalise(s.get("predicted_label"))
        p_text = normalise(t.get("predicted_label"))

        speech_ok = p_speech == gold
        text_ok = p_text == gold
        cells[(speech_ok, text_ok)] += 1

        if p_speech != p_text:
            disagreements.append({
                "index": i,
                "dataset_type": ds,
                "text": s.get("text"),
                "true_label": s.get("true_label"),
                "speech_prediction": s.get("predicted_label"),
                "text_prediction": t.get("predicted_label"),
                "speech_correct": speech_ok,
                "text_correct": text_ok,
            })

    n = len(speech)
    stats = {
        "dataset": ds,
        "n": n,
        "speech_acc": cells[(True, True)] + cells[(True, False)],
        "text_acc": cells[(True, True)] + cells[(False, True)],
        "disagreements": len(disagreements),
        "both_correct": cells[(True, True)],
        "speech_only_correct": cells[(True, False)],
        "text_only_correct": cells[(False, True)],
        "both_wrong": cells[(False, False)],
    }
    return stats, disagreements


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:5.1f}%" if total else "    n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E0a: audio-only vs transcript-only prediction disagreement.",
    )
    parser.add_argument("--speech_preds", required=True,
                        help="*_p.json from the --input_mode speech_only run")
    parser.add_argument("--text_preds", required=True,
                        help="*_p.json from the --input_mode text_only run")
    parser.add_argument("--mode", default="original",
                        help="validation_mode to compare (default: original)")
    parser.add_argument("--dataset", default=None,
                        help="restrict to one dataset_type (default: all shared)")
    parser.add_argument("--out", default=None,
                        help="write the disagreement subset here (JSON)")
    args = parser.parse_args()

    speech_by_ds = load_predictions(args.speech_preds, args.mode)
    text_by_ds = load_predictions(args.text_preds, args.mode)

    shared = sorted(set(speech_by_ds) & set(text_by_ds))
    if args.dataset:
        if args.dataset not in shared:
            sys.exit(f"ERROR: {args.dataset!r} not in both runs. Shared: {shared}")
        shared = [args.dataset]
    if not shared:
        sys.exit(
            f"ERROR: the two runs share no dataset.\n"
            f"       speech: {sorted(speech_by_ds)}\n"
            f"       text:   {sorted(text_by_ds)}"
        )

    print(f"\nE0a: audio-only vs transcript-only  (validation_mode={args.mode})")
    print(f"  speech run : {args.speech_preds}")
    print(f"  text run   : {args.text_preds}\n")

    header = (f"{'dataset':<18} {'n':>6} {'speech':>8} {'text':>8} {'disagree':>9}  "
              f"{'both ok':>8} {'speech only':>12} {'text only':>10} {'both wrong':>11}")
    print(header)
    print("-" * len(header))

    all_disagreements: List[Dict[str, Any]] = []
    for ds in shared:
        speech, text = speech_by_ds[ds], text_by_ds[ds]
        check_alignment(ds, speech, text)
        stats, disagreements = compare(ds, speech, text)
        all_disagreements.extend(disagreements)
        n = stats["n"]
        print(f"{ds:<18} {n:>6} {pct(stats['speech_acc'], n):>8} {pct(stats['text_acc'], n):>8} "
              f"{pct(stats['disagreements'], n):>9}  {stats['both_correct']:>8} "
              f"{stats['speech_only_correct']:>12} {stats['text_only_correct']:>10} "
              f"{stats['both_wrong']:>11}")

    print()
    print("Reading the table:")
    print("  disagree    = the two modalities give different answers. This subset is")
    print("                where routing can be tested, and feeds E0b.")
    print("  speech only = audio carries information the transcript does not. On")
    print("                paralinguistic tasks this should dominate 'text only'.")
    print("  text only   = the transcript alone was sufficient and the audio misled,")
    print("                which is the lexical shortcut Chapter 2 targets.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(all_disagreements, handle, indent=2)
        print(f"\nDisagreement subset ({len(all_disagreements)} items) -> {args.out}")
    elif all_disagreements:
        print(f"\n{len(all_disagreements)} disagreements found. Pass --out to save the subset for E0b.")


if __name__ == "__main__":
    main()
