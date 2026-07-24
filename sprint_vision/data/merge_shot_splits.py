"""
Merge MedFMC repeated-experiment train splits (exp1..expN) into one combined
training JSON, for the "train on all shots pooled together" experiment.

This is a pure offline pre-processing step. It does NOT change train.py or
LazySupervisedDataset -- it only produces a new JSON file, named to match the
explicit SHOT_EXP=all branch added to run_sprint_finetune.sh's DATA_PATH
derivation. train.py continues to receive a single --data_path string exactly
as before; it never needs to know the file was produced by merging.

Usage
-----
    python merge_shot_splits.py \
        --data_dir /home/harinisrireddykandula/LLaVA/sprint_vision/data \
        --task colon --shot 10
    → reads:  colon_train_shot10_exp1.json ... colon_train_shot10_exp5.json
    → writes: colon_train_shot10_all.json

    Merge only a subset of experiments:
        python merge_shot_splits.py --data_dir ... --task chest --shot 10 --exps 1,2,3

Then train on the merged split:
    TRAINING_SHOTS=10 SHOT_EXP=all DATASET=colon bash run_sprint_finetune.sh

Validation note
----------------
This script only merges TRAIN splits (there is no merged validation set --
validation is meant to stay exactly as in the existing exp1..exp5 runs).
run_sprint_finetune.sh's SHOT_EXP=all branch defaults EVAL_DATA_PATH to exp1's
validation split (VAL_EXP_FOR_ALL=1) automatically; override with
VAL_EXP_FOR_ALL=<K> or EVAL_DATA_PATH=<path> if you want a different one.
"""
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Directory containing {task}_train_shot{N}_exp{K}.json files "
             "(sprint_vision/data)",
    )
    p.add_argument(
        "--task", required=True,
        help="Dataset name matching existing split filenames, e.g. colon, "
             "chest, endo, chest_binary, endo_binary",
    )
    p.add_argument("--shot", required=True, type=int, help="Shot count, e.g. 10")
    p.add_argument(
        "--exps", default="1,2,3,4,5",
        help="Comma-separated experiment indices to merge (default: 1,2,3,4,5)",
    )
    p.add_argument(
        "--out_suffix", default="all",
        help='Suffix for the merged output file: {task}_train_shot{N}_{out_suffix}.json '
             '(default: "all", matching the SHOT_EXP=all branch in run_sprint_finetune.sh)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    exp_ids = [e.strip() for e in args.exps.split(",") if e.strip()]

    merged = []
    seen_ids = set()
    duplicate_count = 0

    for exp in exp_ids:
        in_path = os.path.join(
            args.data_dir, f"{args.task}_train_shot{args.shot}_exp{exp}.json"
        )
        if not os.path.exists(in_path):
            raise FileNotFoundError(
                f"Missing split file: {in_path}\n"
                f"Generate it first, e.g.:\n"
                f"  python medfmc_to_llava.py --tasks {args.task} --shot {args.shot} --exp {exp}"
            )
        with open(in_path, "r") as f:
            data = json.load(f)

        for entry in data:
            entry_id = entry.get("id")
            if entry_id is None:
                continue
            if entry_id in seen_ids:
                duplicate_count += 1
            else:
                seen_ids.add(entry_id)

        merged.extend(data)
        print(f"  exp{exp}: {len(data)} samples  ({in_path})")

    out_path = os.path.join(
        args.data_dir, f"{args.task}_train_shot{args.shot}_{args.out_suffix}.json"
    )
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged {len(exp_ids)} splits -> {len(merged)} total samples")
    if duplicate_count:
        print(
            f"WARNING: {duplicate_count} duplicate sample IDs found across splits "
            f"(same image/id appears in more than one exp) -- kept as-is, no "
            f"dedup applied."
        )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
