import os
import subprocess

# ==========================================
# ⚙️ SPRInT VISION MASTER CONFIGURATION
# ==========================================
class Config:
    # --- 1. EXPERIMENT SETTINGS ---
    DATASET = "colon"          # Options: "colon", "chest", "endo"
    STRATEGY = "two_token"      # Options: "regular" (outputs 0/1) or "two_token" (outputs symbols)
    MODE = "train"              # Options: "train" or "inference"
    
    # --- 2. PATHS ---
    LLAVA_DIR = "/home/harinis/LLaVA"
    PROJECT_DIR = "/home/harinis/sprint_vision"
    IMAGE_ROOT = "/home/harinis/MedFM/data/MedFMC"
    
    # Ensure this points to your specific v1.5-13b model path
    MODEL_BASE = os.path.expanduser("~/.cache/huggingface/hub/llava-v1.5-13b") 
    
    # --- 3. HYPERPARAMETERS ---
    BATCH_SIZE = 16
    EPOCHS = 1
    LR = "2e-4"

# ==========================================
# EXECUTION LOGIC
# ==========================================
def run_training():
    print(f"🚀 Starting TRAINING for {Config.DATASET} using {Config.STRATEGY} strategy...")
    
    # 1. Set Environment Variable for downstream SymbolManager sync
    os.environ["SPRINT_STRATEGY"] = Config.STRATEGY
    if Config.STRATEGY == "regular":
        os.environ["SPRINT_STRATEGY"] = "original" 
    
    data_path = f"{Config.PROJECT_DIR}/data/{Config.DATASET}_train.json"
    output_dir = f"{Config.PROJECT_DIR}/checkpoints/llava-{Config.DATASET}-{Config.STRATEGY}"
    
    # 2. Build Command List (ALIGNED WITH MODIFIED train.py)
    cmd = [
        "deepspeed", f"{Config.LLAVA_DIR}/llava/train/train_mem.py",
        "--lora_enable", "True", 
        "--lora_r", "128", 
        "--lora_alpha", "256",
        "--sprint_strategy", Config.STRATEGY,  # <--- CRITICAL: TRIGGERS MODIFIED train.py LOGIC
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

def run_inference():
    print(f"🔍 Starting INFERENCE for {Config.DATASET} using {Config.STRATEGY} strategy...")
    
    checkpoint_path = f"{Config.PROJECT_DIR}/checkpoints/llava-{Config.DATASET}-{Config.STRATEGY}"
    test_json = f"{Config.PROJECT_DIR}/data/{Config.DATASET}_test.json" 
    
    # Uses the finalized sprint_eval.py we created
    cmd = [
        "python", f"{Config.PROJECT_DIR}/sprint_eval.py",
        "--model-base", Config.MODEL_BASE,
        "--model-path", checkpoint_path,
        "--image-folder", Config.IMAGE_ROOT,
        "--question-file", test_json,
        "--strategy", Config.STRATEGY
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    if Config.MODE == "train":
        run_training()
    elif Config.MODE == "inference":
        run_inference()