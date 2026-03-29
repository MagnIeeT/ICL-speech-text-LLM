"""
Convert MedFMC dataset label files to LLaVA JSON format.

Usage:
    python medfmc_to_llava.py \
        --medfmc_root /home/harinis/MedFM/data/MedFMC \
        --output_dir  /home/harinis/LLaVA/sprint_vision/data \
        --tasks       colon,chest,endo \
        --shot        20

This produces:
    {output_dir}/colon_train.json
    {output_dir}/colon_test.json
    {output_dir}/chest_train.json
    ...

The "image" field in each JSON entry is relative to --medfmc_root,
which is the value you pass as --image_folder to train_mem.py / sprint_eval.py.
For example:  "image": "colon/images/abc.png"
Full path at training time: {medfmc_root}/colon/images/abc.png
"""

import json
import os
import argparse

# ============================================================
# Task-specific classification instructions
# ============================================================
TASK_INSTRUCTIONS = {
    "colon": (
        "Task: Medical Image Classification. Dataset: Colon Histopathology. "
        "Is there a tumor? Answer with the label only (0 for No, 1 for Yes)."
    ),
    "chest": (
        "Task: Medical Image Classification. Dataset: Chest X-Ray. "
        "Is there an abnormality? Answer with the label only (0 for No, 1 for Yes)."
    ),
    "endo": (
        "Task: Medical Image Classification. Dataset: Endoscopy. "
        "Is there a lesion? Answer with the label only (0 for No, 1 for Yes)."
    ),
}


def convert_to_llava(task_name, input_txt, img_rel_prefix, output_json, instruction):
    """
    Convert a single MedFMC label file to LLaVA JSON.

    Args:
        task_name      : used to build the sample id, e.g. "colon_train"
        input_txt      : absolute path to the .txt label file
        img_rel_prefix : prefix for the image path relative to --image_folder
                         e.g. "colon/images" so full relative path = "colon/images/abc.png"
        output_json    : absolute path to write the output JSON
        instruction    : question string shown to the model
    """
    if not os.path.exists(input_txt):
        print(f"  ⚠️  Skipping — label file not found: {input_txt}")
        return 0

    llava_data = []
    skipped = 0

    with open(input_txt, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Label is always the last whitespace-separated token.
            # img_name is everything before it (handles spaces in filenames).
            parts = line.rsplit(None, 1)   # split from right, max 1 split
            if len(parts) != 2:
                skipped += 1
                continue

            img_name, label = parts[0].strip(), parts[1].strip()

            # Relative image path from the MedFMC root (what train_mem.py / sprint_eval.py see)
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

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(llava_data, f, indent=2)

    print(f"  ✅ {output_json}  ({len(llava_data)} samples)")
    return len(llava_data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MedFMC label files to LLaVA JSON format"
    )
    parser.add_argument(
        "--medfmc_root",
        required=True,
        help="Absolute path to the MedFMC root directory, "
             "e.g. /home/harinis/MedFM/data/MedFMC",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where JSON files will be saved, "
             "e.g. /home/harinis/LLaVA/sprint_vision/data",
    )
    parser.add_argument(
        "--tasks",
        default="colon,chest,endo",
        help="Comma-separated list of tasks to convert. Default: colon,chest,endo",
    )
    parser.add_argument(
        "--shot",
        default="20",
        help="Shot setting used in the label filename, e.g. 1, 5, 10, 20. "
             "Expects files named train_{shot}.txt and val_{shot}.txt. Default: 20",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"\n📦 Converting MedFMC → LLaVA format")
    print(f"   medfmc_root : {args.medfmc_root}")
    print(f"   output_dir  : {args.output_dir}")
    print(f"   tasks       : {tasks}")
    print(f"   shot        : {args.shot}\n")

    total = 0
    for task in tasks:
        if task not in TASK_INSTRUCTIONS:
            print(f"  ⚠️  Unknown task '{task}'. Skipping.")
            continue

        task_dir      = os.path.join(args.medfmc_root, task)
        img_rel_prefix = os.path.join(task, "images")   # relative to medfmc_root
        instruction   = TASK_INSTRUCTIONS[task]

        print(f"--- {task.upper()} ---")

        # Train split
        n = convert_to_llava(
            task_name    = f"{task}_train",
            input_txt    = os.path.join(task_dir, f"train_{args.shot}.txt"),
            img_rel_prefix = img_rel_prefix,
            output_json  = os.path.join(args.output_dir, f"{task}_train.json"),
            instruction  = instruction,
        )
        total += n

        # Val/test split
        n = convert_to_llava(
            task_name    = f"{task}_test",
            input_txt    = os.path.join(task_dir, f"val_{args.shot}.txt"),
            img_rel_prefix = img_rel_prefix,
            output_json  = os.path.join(args.output_dir, f"{task}_test.json"),
            instruction  = instruction,
        )
        total += n

    print(f"\n✅ Done. {total} total samples written to {args.output_dir}")


if __name__ == "__main__":
    main()
