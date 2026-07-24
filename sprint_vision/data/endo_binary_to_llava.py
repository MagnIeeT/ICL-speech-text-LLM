"""
Derive endo_binary (Inflammatory / Mass lesion, multi-label) LLaVA JSON files
from ALREADY-CONVERTED endo_*.json files (medfmc_to_llava.py output).

Mirrors chest_binary_to_llava.py exactly, with endo's 4 -> 2 grouping:

  - Inflammatory = ulcer OR erosion (a depth-graded mucosal-break spectrum)
  - Mass lesion  = polyp OR tumor  (protruding, space-occupying lesions)

Why a separate script, and why derive from endo_*.json rather than the raw
MedFMC .txt label files — same reasoning as chest_binary_to_llava.py:

  - endo_binary is is_multi_label=True with 2 flags OR-reduced from the 4
    original flags. No precedence rule is needed: both output flags can
    independently be 1 (e.g. an ulcerated tumor -> both inflammatory AND
    mass_lesion). Whether that cross-group co-occurrence is common in real
    endo data has NOT yet been measured as of this writing — see
    endo_inflammatory_mass_cooccurrence.py (scratchpad analysis script).

  - medfmc_to_llava.py's generic multi-label parser (_parse_label_from_parts)
    reads the first len(label_names) columns of a raw label line. Pointing
    it at endo's 4-column .txt with a 2-label config would silently read
    only the first 2 raw columns (ulcer, erosion) and drop polyp/tumor
    entirely — the same silent-wrong-data trap as chest. This script avoids
    that by reading the disease NAMES already resolved by medfmc_to_llava.py
    (conversations[1]["value"]) instead of re-parsing raw flag columns, and
    guarantees the exact same image set / split as whichever endo conversion
    already ran.

Usage — run once per endo_*.json split you need (train/val/test, every
shot/exp/percent variant); this script is 1:1 per input file:

    python endo_binary_to_llava.py \
        --input  /path/to/sprint_vision/data/endo_train_percent100.json \
        --output /path/to/sprint_vision/data/endo_binary_train_percent100.json
"""

import argparse
import json
import os
import sys

# sprint_vision/ is one level up from this file (sprint_vision/data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.data_config.endo_binary_config import (
    INFLAMMATORY_LABELS, MASS_LESION_LABELS, ENDO_BINARY_CONFIG,
)


def _group_of(gt_text: str) -> str:
    """
    Map an endo GPT-turn value (e.g. "ulcer, polyp" or "none") to the
    OR-reduced endo_binary GPT-turn value.

    Both groups can be independently present — that's why this stays
    multi-label instead of being forced into a single label.
    """
    findings = {f.strip() for f in gt_text.strip().split(",") if f.strip()}
    findings.discard("none")

    unknown = findings - INFLAMMATORY_LABELS - MASS_LESION_LABELS
    if unknown:
        raise ValueError(
            f"Unrecognized endo label(s) not in the grouping table: {unknown} "
            f"(from gt_text={gt_text!r}). Update INFLAMMATORY_LABELS/"
            f"MASS_LESION_LABELS in endo_binary_config.py."
        )

    present = []
    if findings & INFLAMMATORY_LABELS:
        present.append("inflammatory")
    if findings & MASS_LESION_LABELS:
        present.append("mass_lesion")
    return ", ".join(present) if present else "none"


def convert(input_path: str, output_path: str) -> int:
    with open(input_path, "r") as f:
        endo_data = json.load(f)

    instruction = ENDO_BINARY_CONFIG.instruction
    out_data = []
    group_counts = {"inflammatory_only": 0, "mass_lesion_only": 0, "both": 0, "none": 0}

    for entry in endo_data:
        gt_text = entry["conversations"][1]["value"]
        new_label = _group_of(gt_text)

        if new_label == "none":
            group_counts["none"] += 1
        elif new_label == "inflammatory":
            group_counts["inflammatory_only"] += 1
        elif new_label == "mass_lesion":
            group_counts["mass_lesion_only"] += 1
        else:
            group_counts["both"] += 1

        out_data.append({
            "id": entry["id"].replace("endo_", "endo_binary_", 1),
            "image": entry["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt",   "value": new_label},
            ],
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"  ✅ Converted {len(endo_data)} endo samples -> {output_path}")
    print(f"  \U0001F4CA Label distribution: {group_counts}")
    return len(out_data)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Existing endo_*.json produced by medfmc_to_llava.py "
             "(train/val/test, any shot/exp/percent variant).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output endo_binary_*.json path.",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
