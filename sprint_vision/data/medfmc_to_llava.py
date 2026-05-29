"""
Convert MedFMC dataset label files to LLaVA JSON format.

Usage
-----
Shot-based (fixed few-shot split):
    python medfmc_to_llava.py \
        --medfmc_root /home/harinis/MedFM/data/MedFMC \
        --output_dir  /home/harinis/LLaVA/sprint_vision/data \
        --tasks       colon,chest,endo \
        --shot        20
    → writes: colon_train_shot20.json, colon_test.json, chest_train_shot20.json, ...

Percentage-based training subset (P% of trainval.txt, stratified by label):
    python medfmc_to_llava.py \
        --medfmc_root /home/harinis/MedFM/data/MedFMC \
        --output_dir  /home/harinis/LLaVA/sprint_vision/data \
        --tasks       colon,chest,endo \
        --train_percent 100
    → writes: colon_train_percent100.json, colon_test.json, chest_train_percent100.json, ...

Output naming convention
------------------------
    Training (shot-based):    {task}_train_shot{N}.json
    Training (percent-based): {task}_train_percent{P}.json
    Test (always full):       {task}_test.json

Dataset types
-------------
    colon : Binary     — "filename 0"  (single binary label)
    chest : Multi-label — CONFIRMED (2026-05-26):
              "5B91F7409CCCE2.png 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0"
              → filename + 19 binary values as a SINGLE comma-separated field.
    endo  : Multi-label — CONFIRMED (2026-05-26):
              "13333_2021.11_0003_55200268.png 0 0 0 0"
              → filename + 4 binary values as SPACE-SEPARATED fields.

    Class names from /home/harinis/MedFM/medfmc/datasets/medical_datasets.py
    (Chest19 and Endoscopy classes, confirmed 2026-05-26).

    Ground-truth stored as comma-separated category names:
        "consolidation, effusion"  (categories present in this image)
        "none"                     (no finding)
    The model is prompted to produce the same format.

IMPORTANT DISTINCTION
---------------------
    --shot / --train_percent   →  controls the FINE-TUNING dataset size.
    --icl-shots in sprint_eval.py  →  controls inference-time ICL examples.
    These are completely independent.

The "image" field in each JSON entry is relative to --medfmc_root,
which you pass as --image_folder to train_mem.py and sprint_eval.py.
Example:  "image": "colon/images/abc.png"
Full path at run time: {medfmc_root}/colon/images/abc.png
"""

import json
import math
import os
import random
import argparse
import sys
from typing import Dict, List, Optional

# sprint_vision/ is one level up from this file (sprint_vision/data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.data_config import get_dataset_config, DATASET_CONFIGS, DatasetName

VALID_TASKS = [e.value for e in DatasetName]


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_label_from_parts(parts: List[str], cfg) -> Optional[str]:
    """
    Parse the label field(s) from a split label-file line.

    Args:
        parts: Line split on whitespace, e.g. ["images/img.jpg", "0", "1", "0", "0"]
        cfg:   DatasetConfig for the task

    Returns:
        Label string, or None if the line is malformed.

    Label formats by task:
        colon : single binary column → "0" or "1"
        chest : "filename 0,0,1,...,0"  (19 values, single comma-separated field)
        endo  : "filename 0 0 1 0"      (4 values, space-separated)
    """
    if cfg.is_multi_label:
        n_labels = len(cfg.label_names)

        label_raw = parts[1:]
        if len(label_raw) == 1 and "," in label_raw[0]:
            flag_strs = label_raw[0].split(",")
        else:
            flag_strs = label_raw

        if len(flag_strs) < n_labels:
            return None
        try:
            flags = [int(v.strip()) for v in flag_strs[:n_labels]]
        except ValueError:
            return None
        present = [cfg.label_names[i] for i, f in enumerate(flags) if f == 1]
        return ", ".join(present) if present else "none"
    else:
        if len(parts) < 2:
            return None
        return parts[-1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified subsampling
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_sample(
    samples: List[Dict],
    train_percent: int,
    seed: int,
) -> List[Dict]:
    """
    Return a stratified random subset of train_percent% of samples.

    For binary tasks: stratifies by "0" / "1".
    For multi-label chest: stratifies by exact label string (each unique
    combination of diseases is a stratum).  This keeps the disease
    co-occurrence distribution of the original set.
    """
    rng = random.Random(seed)

    by_label: Dict[str, List[Dict]] = {}
    for entry in samples:
        label = str(entry["conversations"][1]["value"]).strip()
        by_label.setdefault(label, []).append(entry)

    sampled: List[Dict] = []
    for label in sorted(by_label.keys()):
        pool = list(by_label[label])
        rng.shuffle(pool)
        count = max(1, math.ceil(len(pool) * train_percent / 100))
        sampled.extend(pool[:count])

    rng.shuffle(sampled)
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_llava(
    task_name: str,
    task: str,
    input_txt: str,
    img_rel_prefix: str,
    output_json: str,
    train_percent: Optional[int] = None,
    seed: int = 42,
) -> int:
    """
    Convert a single MedFMC label file to LLaVA JSON format.

    Args:
        task_name      : Used for sample IDs, e.g. "colon_train".
        task           : "colon", "chest", or "endo".
        input_txt      : Absolute path to the .txt label file.
        img_rel_prefix : Image path prefix relative to --medfmc_root,
                         e.g. "colon/images".
        output_json    : Absolute path to write the output JSON.
        train_percent  : If set (1–100), keep only that percentage of samples,
                         sampled stratified by label.  None → keep all.
        seed           : RNG seed for stratified subsampling.

    Returns:
        Number of samples written.
    """
    if not os.path.exists(input_txt):
        print(f"  ⚠️  Skipping — label file not found: {input_txt}")
        return 0

    cfg = get_dataset_config(task)
    instruction = cfg.instruction

    llava_data: List[Dict] = []
    skipped = 0

    with open(input_txt, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                skipped += 1
                continue

            img_name = parts[0].strip()
            label = _parse_label_from_parts(parts, cfg)
            if label is None:
                skipped += 1
                continue

            image_rel = os.path.join(img_rel_prefix, img_name)

            entry = {
                "id": f"{task_name}_{os.path.splitext(img_name)[0]}",
                "image": image_rel,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{instruction}"},
                    {"from": "gpt",   "value": label},
                ],
            }
            llava_data.append(entry)

    if skipped:
        print(f"  ⚠️  {skipped} lines skipped (bad format) in {input_txt}")

    if train_percent is not None:
        full_count = len(llava_data)
        llava_data = _stratified_sample(llava_data, train_percent, seed)
        print(
            f"  📊 Stratified subset: {train_percent}% of {full_count}"
            f" → {len(llava_data)} samples  (seed={seed})"
        )

    if cfg.is_multi_label:
        from collections import Counter
        label_counts = Counter(e["conversations"][1]["value"] for e in llava_data)
        none_count = label_counts.get("none", 0)
        with_label_count = sum(v for k, v in label_counts.items() if k != "none")
        print(f"  📊 Label dist: none={none_count}, with_findings={with_label_count}, unique_combos={len(label_counts)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(llava_data, f, indent=2)

    print(f"  ✅ {output_json}  ({len(llava_data)} samples)")
    return len(llava_data)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert MedFMC label files to LLaVA JSON format.\n"
            "\n"
            "Two modes (mutually exclusive):\n"
            "\n"
            "  Shot-based  (--shot N):\n"
            "    Reads train_{N}.txt  — only train_20.txt exists in standard layout.\n"
            "    Output: {task}_train_shot{N}.json\n"
            "\n"
            "  Percent-based  (--train_percent P):\n"
            "    Reads trainval.txt (full training pool) and keeps P%% stratified.\n"
            "    Output: {task}_train_percent{P}.json\n"
            "    Use --base-file to override the source file (default: trainval.txt).\n"
            "\n"
            "  Test split: always reads test_WithLabel.txt → {task}_test.json\n"
            "\n"
            "⚠️  CHEST MULTI-LABEL:\n"
            "    Verify column order before running:\n"
            "    head -5 {medfmc_root}/chest/test_WithLabel.txt\n"
            "    Update CHEST_DISEASE_ORDER in this file if order differs.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--medfmc_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", default="colon,chest,endo")
    parser.add_argument("--shot", default="20")
    parser.add_argument(
        "--train_percent", type=int, default=None, metavar="P",
        help="Percent of trainval.txt to use for fine-tuning (1–100).",
    )
    parser.add_argument(
        "--base-file", default="trainval.txt",
        help="Source file for percent mode (default: trainval.txt).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-file", default="test_WithLabel.txt")
    args = parser.parse_args()

    if args.train_percent is not None and not (1 <= args.train_percent <= 100):
        parser.error("--train_percent must be between 1 and 100.")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"\n📦 Converting MedFMC → LLaVA format")
    print(f"   medfmc_root : {args.medfmc_root}")
    print(f"   output_dir  : {args.output_dir}")
    print(f"   tasks       : {tasks}")
    if args.train_percent is not None:
        print(f"   mode        : percent-based  ({args.train_percent}% of {args.base_file})")
        print(f"   seed        : {args.seed}")
    else:
        print(f"   mode        : shot-based  (reads train_{args.shot}.txt)")

    if "chest" in tasks:
        from config.data_config.chest_config import CHEST_DISEASE_ORDER
        print(f"\n   CHEST column order: {CHEST_DISEASE_ORDER}")
    print()

    total = 0
    for task in tasks:
        if task not in VALID_TASKS:
            print(f"  ⚠️  Unknown task '{task}'. Skipping.")
            continue

        cfg            = get_dataset_config(task)
        task_dir       = os.path.join(args.medfmc_root, task)
        img_rel_prefix = cfg.images_subdir

        print(f"--- {task.upper()} ({'multi-label' if cfg.is_multi_label else 'binary'}) ---")

        if args.train_percent is not None:
            train_json = os.path.join(
                args.output_dir,
                f"{task}_train_percent{args.train_percent}.json",
            )
            base_txt = os.path.join(task_dir, args.base_file)
        else:
            train_json = os.path.join(
                args.output_dir,
                f"{task}_train_shot{args.shot}.json",
            )
            base_txt = os.path.join(task_dir, f"train_{args.shot}.txt")

        n = convert_to_llava(
            task_name=f"{task}_train",
            task=task,
            input_txt=base_txt,
            img_rel_prefix=img_rel_prefix,
            output_json=train_json,
            train_percent=args.train_percent,
            seed=args.seed,
        )
        total += n

        n = convert_to_llava(
            task_name=f"{task}_test",
            task=task,
            input_txt=os.path.join(task_dir, args.test_file),
            img_rel_prefix=img_rel_prefix,
            output_json=os.path.join(args.output_dir, f"{task}_test.json"),
        )
        total += n

    print(f"\n✅ Done. {total} total samples written to {args.output_dir}")


if __name__ == "__main__":
    main()
