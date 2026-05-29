#!/bin/bash
# ============================================================
# SPRInT-LLaVA Fine-tuning Script
# ============================================================
#
# IMPORTANT DISTINCTION
# ---------------------
#   TRAINING_SHOTS / TRAIN_PERCENT  →  controls the FINE-TUNING dataset size.
#   --icl-shots in sprint_eval.py   →  controls in-context examples at inference.
#   These are completely separate and independent.
#
# Usage — shot-based training split:
#   STRATEGY=regular    TRAINING_SHOTS=1   bash run_sprint_finetune.sh
#   STRATEGY=regular    TRAINING_SHOTS=5   bash run_sprint_finetune.sh
#   STRATEGY=regular    TRAINING_SHOTS=20  bash run_sprint_finetune.sh
#   STRATEGY=two_token  TRAINING_SHOTS=20  bash run_sprint_finetune.sh
#
# Usage — percentage-based training subset:
#   STRATEGY=regular    TRAIN_PERCENT=20   bash run_sprint_finetune.sh
#   STRATEGY=two_token  TRAIN_PERCENT=50   bash run_sprint_finetune.sh
#
# The correct training JSON must exist before running.
# If it doesn't exist, this script prints the exact command to generate it.
# ============================================================

# ============================================================
# 1. PATHS  ← UPDATE THESE for your cluster
# ============================================================
LLAVA_DIR="${LLAVA_DIR:-/home/harinis/LLaVA}"
PROJECT_DIR="${LLAVA_DIR}/sprint_vision"
MEDFMC_ROOT="${MEDFMC_ROOT:-/home/harinis/MedFM/data/MedFMC}"

MODEL_CACHE="${HOME}/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b/snapshots"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE}/$(ls ${MODEL_CACHE} 2>/dev/null | head -1)}"

# ============================================================
# 2. EXPERIMENT SETTINGS  (override via env vars)
# ============================================================
DATASET="${DATASET:-colon}"
STRATEGY="${STRATEGY:-two_token}"

# ── Training duration ─────────────────────────────────────────────────────────
# ED-FT benefits from multiple epochs so symbols rotate.  Default 1 for all
# other strategies to preserve existing single-epoch behaviour.
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

# ── ICL during training ───────────────────────────────────────────────────────
# ICL_SHOTS: number of in-context examples embedded in each training prompt.
# 0 = standard supervised fine-tuning (no ICL context).
# The ICL pool defaults to DATA_PATH (same training file, self-selection excluded).
# Set ICL_POOL_PATH to a different JSON to use a separate pool.
ICL_SHOTS="${ICL_SHOTS:-0}"
ICL_POOL_PATH="${ICL_POOL_PATH:-}"

# ── Validation (optional) ─────────────────────────────────────────────────────
# Set EVAL_DATA_PATH to a LLaVA-format JSON to enable per-epoch accuracy probes.
# E.g.: EVAL_DATA_PATH="${PROJECT_DIR}/data/colon_test.json"
# MAX_VAL_SAMPLES: 0 = use all samples; any positive int caps the subset.
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"

# ── Set EXACTLY ONE of these to control the fine-tuning dataset size ──────────
#
# TRAINING_SHOTS: use a fixed MedFMC few-shot split.
#   Reads: {DATASET}_train_shot{N}.json
#   Generate: python data/medfmc_to_llava.py --shot N
TRAINING_SHOTS="${TRAINING_SHOTS:-}"

# TRAIN_PERCENT: use a percentage subset of the base shot-20 file.
#   Reads: {DATASET}_train_percent{P}.json
#   Generate: python data/medfmc_to_llava.py --shot 20 --train_percent P
TRAIN_PERCENT="${TRAIN_PERCENT:-}"

# ── Default and mutual-exclusion check ───────────────────────────────────────
if [ -z "${TRAINING_SHOTS}" ] && [ -z "${TRAIN_PERCENT}" ]; then
    TRAINING_SHOTS="20"
fi

if [ -n "${TRAINING_SHOTS}" ] && [ -n "${TRAIN_PERCENT}" ]; then
    echo "ERROR: Set TRAINING_SHOTS or TRAIN_PERCENT — not both."
    echo "  TRAINING_SHOTS  → fixed few-shot split    (e.g. TRAINING_SHOTS=20)"
    echo "  TRAIN_PERCENT   → percentage subset       (e.g. TRAIN_PERCENT=20)"
    exit 1
fi

# ============================================================
# 3. DERIVED PATHS  (do not edit)
# ============================================================
if [ -n "${TRAIN_PERCENT}" ]; then
    # Percentage-based: reads trainval.txt (the full colon training pool), keeps P%.
    DATA_SUFFIX="percent${TRAIN_PERCENT}"
    DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train_percent${TRAIN_PERCENT}.json"
    REGEN_CMD="python ${PROJECT_DIR}/data/medfmc_to_llava.py \\
       --medfmc_root ${MEDFMC_ROOT} \\
       --output_dir  ${PROJECT_DIR}/data \\
       --tasks ${DATASET} --train_percent ${TRAIN_PERCENT}"
else
    # Shot-based: reads train_{N}.txt (only train_20.txt exists in standard layout).
    DATA_SUFFIX="shot${TRAINING_SHOTS}"
    DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train_shot${TRAINING_SHOTS}.json"
    REGEN_CMD="python ${PROJECT_DIR}/data/medfmc_to_llava.py \\
       --medfmc_root ${MEDFMC_ROOT} \\
       --output_dir  ${PROJECT_DIR}/data \\
       --tasks ${DATASET} --shot ${TRAINING_SHOTS}"
fi

# Checkpoint directory name mirrors DATA_SUFFIX — no silent name/data mismatch.
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/llava-${DATASET}-${STRATEGY}-${DATA_SUFFIX}"

# ============================================================
# 4. PRE-FLIGHT CHECKS
# ============================================================
echo "=================================================="
echo "  SPRInT-LLaVA Fine-tuning"
echo "=================================================="
echo "  Dataset         : ${DATASET}"
echo "  Strategy        : ${STRATEGY}"
if [ -n "${TRAIN_PERCENT}" ]; then
    echo "  Fine-tune subset: ${TRAIN_PERCENT}% of base shot-20 training file"
    echo "                    (fine-tuning data size — NOT inference ICL shots)"
else
    echo "  Fine-tune split : ${TRAINING_SHOTS}-shot training file"
    echo "                    (fine-tuning data size — NOT inference ICL shots)"
fi
echo "  Data file       : ${DATA_PATH}"
echo "  Images root     : ${MEDFMC_ROOT}"
echo "  Base model      : ${MODEL_PATH}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Epochs          : ${NUM_TRAIN_EPOCHS}"
echo "  ICL shots       : ${ICL_SHOTS} (0 = no ICL context in training prompts)"
if [ -n "${EVAL_DATA_PATH}" ]; then
    echo "  Val data        : ${EVAL_DATA_PATH} (max ${MAX_VAL_SAMPLES} samples/epoch)"
else
    echo "  Val data        : (none — set EVAL_DATA_PATH to enable)"
fi
echo "=================================================="

if [ ! -f "${DATA_PATH}" ]; then
    echo ""
    echo "ERROR: Training JSON not found:"
    echo "  ${DATA_PATH}"
    echo ""
    echo "Generate it first by running:"
    echo "  ${REGEN_CMD}"
    echo ""
    exit 1
fi

if [ ! -d "${MEDFMC_ROOT}" ]; then
    echo "ERROR: Image folder not found: ${MEDFMC_ROOT}"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Base model path not found: ${MODEL_PATH}"
    echo "   Download liuhaotian/llava-v1.5-13b or set MODEL_PATH env var."
    exit 1
fi

echo "Pre-flight checks passed. Starting training..."

# ============================================================
# 5. ENVIRONMENT
# ============================================================
eval "$(conda shell.bash hook)"
conda activate llava
cd "${LLAVA_DIR}"

# ============================================================
# 6. DERIVED TRAINING SETTINGS
# ============================================================
# ED-FT must use NUM_WORKERS=0 so that the symbol_manager state updated by
# SPRInTSymbolEpochCallback in the main process is visible to __getitem__
# calls without crossing fork boundaries.  Other strategies use 4 workers.
if [ "${STRATEGY}" = "ed_ft" ]; then
    NUM_WORKERS=0
else
    NUM_WORKERS=4
fi

# Optional validation args — only passed when EVAL_DATA_PATH is set.
EVAL_ARGS=""
if [ -n "${EVAL_DATA_PATH}" ]; then
    EVAL_ARGS="--eval_data_path ${EVAL_DATA_PATH} --max_val_samples ${MAX_VAL_SAMPLES}"
fi

# Optional ICL training args — only non-zero icl_shots adds arguments.
ICL_ARGS=""
if [ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null; then
    ICL_ARGS="--icl_shots ${ICL_SHOTS}"
    if [ -n "${ICL_POOL_PATH}" ]; then
        ICL_ARGS="${ICL_ARGS} --icl_pool_path ${ICL_POOL_PATH}"
    fi
fi

# ============================================================
# 7. TRAINING
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
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
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
    --dataloader_num_workers "${NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to wandb \
    ${EVAL_ARGS} \
    ${ICL_ARGS}

echo "=================================================="
echo "Training complete."
echo "   Checkpoint : ${OUTPUT_DIR}"
echo "   Symbols    : ${OUTPUT_DIR}/symbol_mappings.json"
echo "=================================================="
