import os
import subprocess
import argparse

# ==========================================
# ⚙️ SPRInT VISION MASTER CONFIGURATION
# ==========================================
class Config:
    # --- 1. EXPERIMENT SETTINGS 
    
    
    DATASET = "colon"          # Options: "colon", "chest", "endo"
    STRATEGY = "regular"       # Options: "regular" (outputs 0/1) or "two_token" (outputs symbols)
    MODE = "inference"         # Options: "train" or "inference"

    # --- 2. PATHS ---
    # Override any of these via environment variables (e.g. LLAVA_DIR=/my/path bash submit_inference.sh)
    # or edit the defaults here for your cluster.
    LLAVA_DIR   = os.environ.get("LLAVA_DIR",    "/home/harinis/LLaVA")
    PROJECT_DIR = os.environ.get("PROJECT_DIR",  os.path.dirname(os.path.abspath(__file__)))
    IMAGE_ROOT  = os.environ.get("MEDFMC_ROOT",  "/home/harinis/MedFM/data/MedFMC")

    # Base model (HF model ID or local cache path)
    MODEL_BASE  = os.environ.get("MODEL_BASE",   "/home/harinis/.cache/huggingface/hub/llava-v1.5-13b")

    # Fine-tuned checkpoint (LoRA adapter directory).
    # Set to None to run the base model directly (no fine-tuning yet).
    # Once you have a trained checkpoint, set it here, e.g.:
    #   CHECKPOINT_PATH = f"{PROJECT_DIR}/checkpoints/llava-colon-regular"
    CHECKPOINT_PATH = None
    
    # --- 3. HYPERPARAMETERS ---
    BATCH_SIZE = 1
    EPOCHS = 1
    LR = "2e-4"

# ==========================================
# EXECUTION LOGIC
# ==========================================
def run_training():
    print(f"🚀 Starting TRAINING for {Config.DATASET} using {Config.STRATEGY} strategy...")
    
    data_path = f"{Config.PROJECT_DIR}/data/{Config.DATASET}_train.json"
    output_dir = f"{Config.PROJECT_DIR}/checkpoints/llava-{Config.DATASET}-{Config.STRATEGY}"
    
    # 2. Build Command List (ALIGNED WITH MODIFIED train.py)
    cmd = [
        "deepspeed", f"{Config.LLAVA_DIR}/llava/train/train_mem.py",
        "--deepspeed", f"{Config.LLAVA_DIR}/scripts/zero2.json",  # REQUIRED: DeepSpeed config
        "--lora_enable", "True",
        "--lora_r", "128",
        "--lora_alpha", "256",
        "--sprint_strategy", Config.STRATEGY,  # TRIGGERS symbol strategy in train.py
        "--model_name_or_path", Config.MODEL_BASE,
        "--version", "v1",
        "--data_path", data_path,
        "--image_folder", Config.IMAGE_ROOT,
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--image_aspect_ratio", "pad",
        "--bf16", "True",
        "--output_dir", output_dir,
        "--num_train_epochs", str(Config.EPOCHS),
        "--per_device_train_batch_size", str(Config.BATCH_SIZE),
        "--gradient_accumulation_steps", "8",
        "--save_steps", "500",
        "--learning_rate", Config.LR,
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--lazy_preprocess", "True",
        "--report_to", "wandb"
    ]
    
    # 3. Execute
    subprocess.run(cmd, cwd=Config.LLAVA_DIR)

def run_inference(num_samples=0, icl_shots=0):
    print(f"🔍 Starting INFERENCE for {Config.DATASET} using {Config.STRATEGY} strategy...")
    print(f"Num samples: {'ALL' if num_samples == 0 else num_samples}")
    print(f"ICL shots: {icl_shots} (0 = zero-shot)")

    test_json = f"{Config.PROJECT_DIR}/data/{Config.DATASET}_test.json"

    cmd = ["python", f"{Config.PROJECT_DIR}/sprint_eval.py"]

    if Config.CHECKPOINT_PATH:
        # Fine-tuned LoRA checkpoint: pass base model + adapter path
        print(f"Using fine-tuned checkpoint: {Config.CHECKPOINT_PATH}")
        cmd += ["--model-base", Config.MODEL_BASE, "--model-path", Config.CHECKPOINT_PATH]
    else:
        # No checkpoint yet — run base model directly
        print(f"No CHECKPOINT_PATH set — running base model: {Config.MODEL_BASE}")
        cmd += ["--model-path", Config.MODEL_BASE]

    cmd += [
        "--image-folder", Config.IMAGE_ROOT,
        "--question-file", test_json,
        "--strategy", Config.STRATEGY,
        "--num-samples", str(num_samples),
        "--icl-shots", str(icl_shots),
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    # Fix 1: accept --mode and --num-samples from CLI so submit_inference.sh drives execution
    parser = argparse.ArgumentParser(description="SPRInT Vision Orchestrator")
    parser.add_argument("--mode", type=str, choices=["train", "inference"],
                        default=Config.MODE,
                        help="Override Config.MODE at runtime")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--icl-shots", type=int, default=0,
                        help="Number of in-context learning (few-shot) examples. 0 = zero-shot inference (default).")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "inference":
        run_inference(num_samples=args.num_samples, icl_shots=args.icl_shots)