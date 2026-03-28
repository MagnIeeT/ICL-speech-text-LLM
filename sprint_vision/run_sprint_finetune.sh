#!/bin/bash

# ============================================================
# SPRInT-LLaVA Fine-tuning Script (Two-Token Strategy)
# ============================================================

# 1. Environment Setup
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate llava
cd /home/harinis/LLaVA

# 2. Configuration Paths
DATA_PATH="/home/harinis/LLaVA/sprint_vision/data/colon_train.json"
IMAGE_FOLDER="/home/harinis/MedFM/data/MedFMC"
OUTPUT_DIR="./checkpoints/llava-v1.5-13b-sprint-colon-two-token"
MODEL_PATH="/home/harinis/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots/080c95/..." 

# 3. Pre-flight Checks (Researcher Point of View)
echo "------------------------------------------------"
echo "🔍 Validating Environment..."
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Training JSON not found at $DATA_PATH"
    exit 1
fi
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ ERROR: Image folder not found at $IMAGE_FOLDER"
    exit 1
fi
echo "✅ Environment Ready. Starting SPRInT Fine-tuning..."
echo "------------------------------------------------"

# 4. Execution via DeepSpeed
# Note: We added --sprint_strategy here to trigger your new code in train.py
deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --sprint_strategy "two_token" \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb