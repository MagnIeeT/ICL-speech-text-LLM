#!/bin/bash
# ============================================================
# SPRInT-LLaVA Fine-tuning Script
# ============================================================
# Usage:
#   bash run_sprint_finetune.sh                         # trains colon, two_token (default)
#   DATASET=chest  bash run_sprint_finetune.sh
#   DATASET=endo   STRATEGY=regular bash run_sprint_finetune.sh
# ============================================================

# ============================================================
# 1. PATHS  ← UPDATE THESE for your cluster
# ============================================================
LLAVA_DIR="${LLAVA_DIR:-/home/harinis/LLaVA}"
PROJECT_DIR="${LLAVA_DIR}/sprint_vision"
MEDFMC_ROOT="${MEDFMC_ROOT:-/home/harinis/MedFM/data/MedFMC}"

# Local snapshot of the base LLaVA model.
# Find yours with: ls ~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots/
MODEL_CACHE="${HOME}/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE}/$(ls ${MODEL_CACHE} 2>/dev/null | head -1)}"

# ============================================================
# 2. EXPERIMENT SETTINGS  (can override via env vars)
# ============================================================
DATASET="${DATASET:-colon}"          # colon | chest | endo
STRATEGY="${STRATEGY:-two_token}"    # two_token | regular
SHOT="${SHOT:-20}"                   # 1 | 5 | 10 | 20

# ============================================================
# 3. DERIVED PATHS  (do not edit)
# ============================================================
DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train.json"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/llava-${DATASET}-${STRATEGY}-shot${SHOT}"

# ============================================================
# 4. PRE-FLIGHT CHECKS
# ============================================================
echo "=================================================="
echo "  SPRInT-LLaVA Fine-tuning"
echo "=================================================="
echo "  Dataset   : ${DATASET} (${SHOT}-shot)"
echo "  Strategy  : ${STRATEGY}"
echo "  Data file : ${DATA_PATH}"
echo "  Images    : ${MEDFMC_ROOT}"
echo "  Model     : ${MODEL_PATH}"
echo "  Output    : ${OUTPUT_DIR}"
echo "=================================================="

if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Training JSON not found: ${DATA_PATH}"
    echo "   Run medfmc_to_llava.py first:"
    echo "   python ${PROJECT_DIR}/data/medfmc_to_llava.py \\"
    echo "       --medfmc_root ${MEDFMC_ROOT} \\"
    echo "       --output_dir  ${PROJECT_DIR}/data \\"
    echo "       --tasks ${DATASET} --shot ${SHOT}"
    exit 1
fi

if [ ! -d "${MEDFMC_ROOT}" ]; then
    echo "ERROR: Image folder not found: ${MEDFMC_ROOT}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    echo "   Either download liuhaotian/llava-v1.5-13b or set MODEL_PATH env var."
    exit 1
fi

echo "Pre-flight checks passed. Starting training..."

# ============================================================
# 5. ENVIRONMENT
# ============================================================
# Activate the conda environment.
# conda must be available in PATH (run `conda init bash` once on the cluster first).
eval "$(conda shell.bash hook)"
conda activate llava
cd "${LLAVA_DIR}"

# ============================================================
# 6. TRAINING
# ============================================================
deepspeed llava/train/train_mem.py \
    --deepspeed "${LLAVA_DIR}/scripts/zero2.json" \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --sprint_strategy "${STRATEGY}" \
    --model_name_or_path "${MODEL_PATH}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${MEDFMC_ROOT}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

echo ""
echo "=================================================="
echo "Training complete."
echo "   Checkpoint : ${OUTPUT_DIR}"
echo "   Symbols    : ${OUTPUT_DIR}/symbol_mappings.json"
echo "=================================================="
