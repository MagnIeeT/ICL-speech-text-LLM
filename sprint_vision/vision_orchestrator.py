import glob
import json as _json
import os
import subprocess
import argparse
import time

# ==========================================
# SPRInT VISION MASTER CONFIGURATION
# ==========================================
class Config:
    # --- 1. EXPERIMENT SETTINGS ---
    DATASET  = "colon"    # Options: "colon", "chest", "endo"
    STRATEGY = "regular"  # Options: "regular" (0/1 or disease names) or "two_token" (symbols)
    MODE     = "inference"

    # --- 2. PATHS ---
    LLAVA_DIR   = os.environ.get("LLAVA_DIR",    "/home/harinis/LLaVA")
    PROJECT_DIR = os.environ.get("PROJECT_DIR",  os.path.dirname(os.path.abspath(__file__)))
    IMAGE_ROOT  = os.environ.get("MEDFMC_ROOT",  "/home/harinis/MedFM/data/MedFMC")
    MODEL_BASE  = os.environ.get("MODEL_BASE",   "/home/harinis/.cache/huggingface/hub/llava-v1.5-13b")

    # Fine-tuned checkpoint (LoRA adapter dir). None → base model.
    CHECKPOINT_PATH = None

    # --- 3. HYPERPARAMETERS ---
    BATCH_SIZE = 1
    EPOCHS     = 1
    LR         = "2e-4"


# ==========================================
# EXECUTION LOGIC
# ==========================================

def run_training(dataset: str, strategy: str):
    print(f"Starting TRAINING for {dataset} using {strategy} strategy...")

    data_path  = f"{Config.PROJECT_DIR}/data/{dataset}_train_percent100.json"
    output_dir = f"{Config.PROJECT_DIR}/checkpoints/llava-{dataset}-{strategy}-percent100"

    cmd = [
        "deepspeed", f"{Config.LLAVA_DIR}/llava/train/train_mem.py",
        "--deepspeed",             f"{Config.LLAVA_DIR}/scripts/zero2.json",
        "--lora_enable",           "True",
        "--lora_r",                "128",
        "--lora_alpha",            "256",
        "--sprint_strategy",       strategy,
        "--model_name_or_path",    Config.MODEL_BASE,
        "--version",               "v1",
        "--data_path",             data_path,
        "--image_folder",          Config.IMAGE_ROOT,
        "--vision_tower",          "openai/clip-vit-large-patch14-336",
        "--mm_projector_type",     "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--image_aspect_ratio",    "pad",
        "--bf16",                  "True",
        "--output_dir",            output_dir,
        "--num_train_epochs",      str(Config.EPOCHS),
        "--per_device_train_batch_size", str(Config.BATCH_SIZE),
        "--gradient_accumulation_steps", "8",
        "--save_steps",            "500",
        "--learning_rate",         Config.LR,
        "--weight_decay",          "0.",
        "--warmup_ratio",          "0.03",
        "--lr_scheduler_type",     "cosine",
        "--logging_steps",         "1",
        "--tf32",                  "True",
        "--model_max_length",      "2048",
        "--gradient_checkpointing", "True",
        "--lazy_preprocess",       "True",
        "--report_to",             "wandb",
    ]

    subprocess.run(cmd, cwd=Config.LLAVA_DIR)


def run_inference(dataset: str, strategy: str, num_samples: int = 0, icl_shots: int = 0) -> dict:
    """
    Run sprint_eval.py for one dataset and return the metrics dict.
    Finds the results JSON that sprint_eval.py saves to logs/json/.
    """
    print(f"\n{'='*60}")
    print(f"Running inference: {dataset}  [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"{'='*60}")
    print(f"Num samples : {'ALL' if num_samples == 0 else num_samples}")
    print(f"ICL shots   : {icl_shots}")

    test_json  = f"{Config.PROJECT_DIR}/data/{dataset}_test.json"
    train_json = f"{Config.PROJECT_DIR}/data/{dataset}_train_percent100.json"

    cmd = ["python", f"{Config.PROJECT_DIR}/sprint_eval.py"]

    if Config.CHECKPOINT_PATH:
        print(f"Using fine-tuned checkpoint: {Config.CHECKPOINT_PATH}")
        cmd += ["--model-base", Config.MODEL_BASE, "--model-path", Config.CHECKPOINT_PATH]
    else:
        print(f"No CHECKPOINT_PATH set — running base model: {Config.MODEL_BASE}")
        cmd += ["--model-path", Config.MODEL_BASE]

    cmd += [
        "--image-folder",  Config.IMAGE_ROOT,
        "--question-file", test_json,
        "--dataset",       dataset,
        "--strategy",      strategy,
        "--num-samples",   str(num_samples),
        "--icl-shots",     str(icl_shots),
    ]

    if icl_shots > 0:
        cmd += ["--train-file", train_json]

    # Record time just before run so we can find the JSON written by sprint_eval.py
    t_before = time.time()
    subprocess.run(cmd)

    print(f"\nDone: {dataset} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Find the results JSON just written (most recent file matching the pattern)
    results_dir = os.path.join(Config.LLAVA_DIR, "logs", "json")
    pattern = os.path.join(results_dir, f"results_{dataset}_{strategy}_{icl_shots}shot_*.json")
    matches = [f for f in glob.glob(pattern) if os.path.getmtime(f) >= t_before]
    if not matches:
        # Fallback: just pick the newest file matching the pattern
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)

    if matches:
        newest = max(matches, key=os.path.getmtime)
        data = _json.load(open(newest))
        return data.get("metrics", {})

    return {}


def _print_final_summary(all_metrics: dict, strategy: str, checkpoint: str, icl_shots: int):
    """
    Print ICI-style FINAL VALIDATION RESULTS summary across all datasets.
    """
    SEP  = "=" * 80
    SEP2 = "-" * 80

    print(f"\n{SEP}")
    print("FINAL VALIDATION RESULTS")
    print(SEP)
    print(f"\nCheckpoint : {checkpoint or '(base model)'}")
    print(f"Strategy   : {strategy}")
    print(f"ICL shots  : {icl_shots}")
    print()

    # Primary metrics per dataset
    print(f"Validation Mode: original (fine-tuned checkpoint)")
    print(SEP2)

    primary_scores = {}
    for dataset, m in all_metrics.items():
        # MedFMC primary: colon→AUC, chest/endo→mAP
        if dataset == "colon":
            primary = m.get("auc")
            primary_label = "AUC"
        else:
            primary = m.get("map")
            primary_label = "mAP"

        macro_auc = m.get("macro_auc")
        macro_f1  = m.get("macro_f1", 0)
        accuracy  = m.get("accuracy", 0)

        if primary is not None:
            primary_scores[dataset] = primary
            line = f"{dataset:<20} {primary_label:<10}: {primary:.4f}"
        else:
            line = f"{dataset:<20} {primary_label:<10}: n/a  (insufficient label diversity)"

        if macro_auc is not None and dataset != "colon":
            line += f"   macro_AUC: {macro_auc:.4f}"
        line += f"   macro_F1: {macro_f1:.4f}"
        print(f"  {line}")

    print(SEP)

    # Per-dataset detailed block
    for dataset, m in all_metrics.items():
        print(f"\n{'='*80}")
        print(f"Metrics for {dataset}")
        print(f"{'='*80}")
        if "auc" in m:
            print(f"  AUC            : {m['auc']:.4f}")
        if "macro_auc" in m:
            print(f"  macro_AUC      : {m['macro_auc']:.4f}")
        if "map" in m:
            print(f"  mAP            : {m['map']:.4f}")
        print(f"  macro_F1       : {m.get('macro_f1', 0):.4f}")
        print(f"  accuracy       : {m.get('accuracy', 0):.4f}")
        if "sensitivity" in m:
            print(f"  sensitivity    : {m['sensitivity']:.4f}")
        if "specificity" in m:
            print(f"  specificity    : {m['specificity']:.4f}")

    # Composite summary line (primary metric per dataset)
    print(f"\n{SEP}")
    if primary_scores:
        parts = [f"{ds}:{v:.4f}" for ds, v in primary_scores.items()]
        avg   = sum(primary_scores.values()) / len(primary_scores)
        print(f"📊 per-dataset (MedFMC primary): {' | '.join(parts)}   avg={avg:.4f}")
    else:
        print("📊 per-dataset: no primary metrics (run with full dataset for valid numbers)")
    print(f"{SEP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPRInT Vision Orchestrator")
    parser.add_argument("--mode",            type=str, choices=["train", "inference"],
                        default=Config.MODE)
    parser.add_argument("--datasets",        type=str, default=None,
                        help="Hyphen-separated datasets: 'colon-chest-endo'. "
                             "Overrides --dataset when set.")
    parser.add_argument("--dataset",         type=str, choices=["colon", "chest", "endo"],
                        default=Config.DATASET,
                        help="Single dataset (use --datasets for multiple).")
    parser.add_argument("--strategy",        type=str, choices=["regular", "two_token"],
                        default=Config.STRATEGY,
                        help="SPRInT strategy: 'regular' or 'two_token'.")
    parser.add_argument("--num-samples",     type=int, default=0,
                        help="Samples to evaluate (0 = all).")
    parser.add_argument("--icl-shots",       type=int, default=0,
                        help="In-context learning examples per query (0 = zero-shot).")
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="LoRA adapter checkpoint directory. Empty = base model only.")
    parser.add_argument("--model-base",      type=str, default="",
                        help="Base model path (required when --checkpoint-path is set).")
    args = parser.parse_args()

    if args.checkpoint_path:
        Config.CHECKPOINT_PATH = args.checkpoint_path
    if args.model_base:
        Config.MODEL_BASE = args.model_base

    # Resolve dataset list
    if args.datasets:
        datasets_list = [d.strip() for d in args.datasets.split("-") if d.strip()]
    else:
        datasets_list = [args.dataset]

    if args.mode == "train":
        run_training(dataset=datasets_list[0], strategy=args.strategy)

    elif args.mode == "inference":
        all_metrics = {}
        for dataset in datasets_list:
            metrics = run_inference(
                dataset=dataset,
                strategy=args.strategy,
                num_samples=args.num_samples,
                icl_shots=args.icl_shots,
            )
            all_metrics[dataset] = metrics

        _print_final_summary(
            all_metrics=all_metrics,
            strategy=args.strategy,
            checkpoint=Config.CHECKPOINT_PATH or "",
            icl_shots=args.icl_shots,
        )

        print("==========================================")
        print(f"All datasets complete at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("==========================================")
