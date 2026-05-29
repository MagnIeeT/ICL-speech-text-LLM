import os
import subprocess
import argparse

# ==========================================
# SPRInT VISION MASTER CONFIGURATION
# ==========================================
class Config:
    # --- 1. EXPERIMENT SETTINGS ---
    DATASET  = "chest"    # Options: "colon", "chest", "endo"
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


def run_inference(dataset: str, strategy: str, num_samples: int = 0, icl_shots: int = 0):
    print(f"Starting INFERENCE for {dataset} using {strategy} strategy...")
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

    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPRInT Vision Orchestrator")
    parser.add_argument("--mode",        type=str, choices=["train", "inference"],
                        default=Config.MODE)
    parser.add_argument("--dataset",     type=str, choices=["colon", "chest", "endo"],
                        default=Config.DATASET,
                        help="Dataset to train/evaluate on.")
    parser.add_argument("--strategy",    type=str, choices=["regular", "two_token"],
                        default=Config.STRATEGY,
                        help="SPRInT strategy: 'regular' or 'two_token'.")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Samples to evaluate (0 = all).")
    parser.add_argument("--icl-shots",   type=int, default=0,
                        help="In-context learning examples per query (0 = zero-shot).")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(dataset=args.dataset, strategy=args.strategy)
    elif args.mode == "inference":
        run_inference(
            dataset=args.dataset,
            strategy=args.strategy,
            num_samples=args.num_samples,
            icl_shots=args.icl_shots,
        )
