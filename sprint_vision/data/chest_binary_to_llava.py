"""
Derive chest_binary (Pulmonary / Extrapulmonary, multi-label) LLaVA JSON files
from ALREADY-CONVERTED chest_*.json files (medfmc_to_llava.py output).

Why a separate script, and why derive from chest_*.json rather than the raw
MedFMC .txt label files — see memory current_status.md, Session 2026-07-21
"ChestDR Pulmonary/Extrapulmonary" thread:

  - chest_binary is is_multi_label=True with 2 flags OR-reduced from the 19
    original flags. Real cluster data showed 49.4% of images have findings
    from BOTH groups simultaneously (57.5% of images with any finding at
    all) — a single-label split was ruled out; it would discard ground
    truth for the majority of positive images. No precedence rule is
    needed here: both output flags can independently be 1.

  - medfmc_to_llava.py's generic multi-label parser (_parse_label_from_parts)
    reads the first len(label_names) columns of a raw label line. Pointing
    it at chest's 19-column .txt with a 2-label config would silently read
    only the first 2 raw columns (pleural_effusion, nodule) instead of
    grouping all 19 — a real, silent-wrong-data trap. This script avoids
    that class of bug entirely by reading the disease NAMES already
    resolved by medfmc_to_llava.py (conversations[1]["value"], e.g.
    "consolidation, pleural_effusion") instead of re-parsing raw flag
    columns. It also guarantees the exact same image set / sample IDs /
    train-val-test split as whichever chest conversion already ran.

Usage — run once per chest_*.json split you need (train/val/test, every
shot/exp/percent variant); this script is 1:1 per input file:

    python chest_binary_to_llava.py \
        --input  /path/to/sprint_vision/data/chest_train_percent100.json \
        --output /path/to/sprint_vision/data/chest_binary_train_percent100.json

    python chest_binary_to_llava.py \
        --input  /path/to/sprint_vision/data/chest_test.json \
        --output /path/to/sprint_vision/data/chest_binary_test.json
"""

import argparse
import json
import os
import sys

# sprint_vision/ is one level up from this file (sprint_vision/data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.data_config.chest_binary_config import (
    PULMONARY_LABELS, EXTRAPULMONARY_LABELS, CHEST_BINARY_CONFIG,
)


def _group_of(gt_text: str) -> str:
    """
    Map a chest GPT-turn value (e.g. "consolidation, pleural_effusion" or
    "none") to the OR-reduced chest_binary GPT-turn value.

    Both groups can be independently present — that's the whole reason this
    is kept multi-label instead of forced into a single label.
    """
    findings = {f.strip() for f in gt_text.strip().split(",") if f.strip()}
    findings.discard("none")

    unknown = findings - PULMONARY_LABELS - EXTRAPULMONARY_LABELS
    if unknown:
        raise ValueError(
            f"Unrecognized chest label(s) not in the grouping table: {unknown} "
            f"(from gt_text={gt_text!r}). Update PULMONARY_LABELS/"
            f"EXTRAPULMONARY_LABELS in chest_binary_config.py."
        )

    present = []
    if findings & PULMONARY_LABELS:
        present.append("pulmonary")
    if findings & EXTRAPULMONARY_LABELS:
        present.append("extrapulmonary")
    return ", ".join(present) if present else "none"


def convert(input_path: str, output_path: str) -> int:
    with open(input_path, "r") as f:
        chest_data = json.load(f)

    instruction = CHEST_BINARY_CONFIG.instruction
    out_data = []
    group_counts = {"pulmonary_only": 0, "extrapulmonary_only": 0, "both": 0, "none": 0}

    for entry in chest_data:
        gt_text = entry["conversations"][1]["value"]
        new_label = _group_of(gt_text)

        if new_label == "none":
            group_counts["none"] += 1
        elif new_label == "pulmonary":
            group_counts["pulmonary_only"] += 1
        elif new_label == "extrapulmonary":
            group_counts["extrapulmonary_only"] += 1
        else:
            group_counts["both"] += 1

        out_data.append({
            "id": entry["id"].replace("chest_", "chest_binary_", 1),
            "image": entry["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt",   "value": new_label},
            ],
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"  ✅ Converted {len(chest_data)} chest samples -> {output_path}")
    print(f"  \U0001F4CA Label distribution: {group_counts}")
    return len(out_data)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Existing chest_*.json produced by medfmc_to_llava.py "
             "(train/val/test, any shot/exp/percent variant).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output chest_binary_*.json path.",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
