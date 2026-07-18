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
# Strategies:
#   regular / rft       Standard fine-tuning, original labels (no substitution)
#   two_token / ss_ft   Fixed two-token symbols throughout training
#   ed_ft               Epoch-dynamic: new symbols every epoch (NUM_WORKERS=0 auto)
#   id_ft               Instance-dynamic: fresh symbols every sample
#   lf_ft               Label-flip via derangement (binary: 0↔1; multi-class: shuffled)
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
LLAVA_DIR="${LLAVA_DIR:-/home/harinisrireddykandula/LLaVA}"
PROJECT_DIR="${LLAVA_DIR}/sprint_vision"
MEDFMC_ROOT="${MEDFMC_ROOT:-/home/harinisrireddykandula/MedFM/data/MedFMC}"

MODEL_PATH="${MODEL_PATH:-/home/harinisrireddykandula/llava-v1.5-13b}"

# ============================================================
# 2. EXPERIMENT SETTINGS  (override via env vars)
# ============================================================
DATASET="${DATASET:-colon}"
STRATEGY="${STRATEGY:-regular}"

# ── Training duration ─────────────────────────────────────────────────────────
# ED-FT benefits from multiple epochs so symbols rotate.  Default 1 for all
# other strategies to preserve existing single-epoch behaviour.
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"

# ── ICL during training ───────────────────────────────────────────────────────
# ICL_SHOTS: number of in-context examples embedded in each training prompt.
# 0 = standard supervised fine-tuning (no ICL context).
# The ICL pool defaults to DATA_PATH (same training file, self-selection excluded).
# Set ICL_POOL_PATH to a different JSON to use a separate pool.
ICL_SHOTS="${ICL_SHOTS:-0}"
ICL_POOL_PATH="${ICL_POOL_PATH:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"   # 0 = use all samples

# ── LoRA hyperparameters ──────────────────────────────────────────────────────
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# ── Validation — mirrors ICI: always runs after every epoch ───────────────────
# When TRAINING_SHOTS + SHOT_EXP are both set (MedFMC repeated-experiment protocol),
# default to the official val split (rest of few-shot pool, 789 images for chest).
# Validation subsamples to MAX_VAL_SAMPLES (default 100) per sir's directive — a
# seeded random.Random(42).sample, so it is representative, not the first N.
# Otherwise fall back to the test set. Override with EVAL_DATA_PATH=path or "none".
if [ -z "${EVAL_DATA_PATH:-}" ]; then
    if [ -n "${TRAINING_SHOTS:-}" ] && [ -n "${SHOT_EXP:-}" ]; then
        EVAL_DATA_PATH="${PROJECT_DIR}/data/${DATASET}_val_shot${TRAINING_SHOTS}_exp${SHOT_EXP}.json"
    else
        EVAL_DATA_PATH="${PROJECT_DIR}/data/${DATASET}_test.json"
    fi
fi
[ "${EVAL_DATA_PATH}" = "none" ] && EVAL_DATA_PATH=""
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"   # 0 = use all; 100 = seeded random subsample (sir's directive)
# Comma-separated validation modes: fixed,original,fresh
# For RFT (no symbols), only 'original' ever runs regardless of this setting.
VALIDATION_MODES="${VALIDATION_MODES:-fixed,original,fresh}"
# Whether to compute AUC (colon) or mAP+AUC (chest/endo) during validation.
# Adds ~1-6 extra minutes per epoch. Set to "false" to skip (use macro_F1 only).
COMPUTE_VAL_AUC_MAP="${COMPUTE_VAL_AUC_MAP:-true}"
# Batched per-class validation scoring. >1 = left-padded batched P(Yes) (faster
# val AUC/mAP, logs [BATCH-VERIFY] OK on first sample); 1 = unbatched (default).
# Exported so the deepspeed child + validation.py see it.
SPRINT_PROBE_BATCH_SIZE="${SPRINT_PROBE_BATCH_SIZE:-1}"
export SPRINT_PROBE_BATCH_SIZE

SPRINT_KV_CACHE="${SPRINT_KV_CACHE:-false}"
export SPRINT_KV_CACHE

# ── Set EXACTLY ONE of these to control the fine-tuning dataset size ──────────
#
# TRAINING_SHOTS: fixed MedFMC few-shot split.
#   Without SHOT_EXP: reads train_{N}.txt  → {task}_train_shot{N}.json
#   With    SHOT_EXP: reads {task}_{N}-shot_train_exp{K}.txt → {task}_train_shot{N}_exp{K}.json
#   Generate: python data/medfmc_to_llava.py --shot N [--exp K]
TRAINING_SHOTS="${TRAINING_SHOTS:-}"
SHOT_EXP="${SHOT_EXP:-}"         # experiment index 1-5 for MedFMC repeated splits

# TRAIN_PERCENT: percentage of trainval.txt, stratified by label.
#   Generate: python data/medfmc_to_llava.py --train_percent P
TRAIN_PERCENT="${TRAIN_PERCENT:-}"

# RANDOM_SHOT: random N samples from trainval.txt — one repetition of the
#   few-shot experiment (matches MedFMC paper repeated-random-split protocol).
#   Run 2-3 times with different SHOT_SEED values and average the results.
#   Generate: python data/medfmc_to_llava.py --n_shot N --seed S
RANDOM_SHOT="${RANDOM_SHOT:-}"
SHOT_SEED="${SHOT_SEED:-1}"

# SEED controls training_args.seed (model/RNG: LoRA-A init, sampler shuffle,
# dropout) -- distinct from SHOT_SEED above, which only controls which training
# SAMPLES get selected. Was previously hardcoded to HF's default (42) with no
# way to vary it. For the mean+/-std-over-N-seeds plan, launch with e.g.
# SEED=1 ./run_sprint_finetune.sh, SEED=2 ./run_sprint_finetune.sh, ...
SEED="${SEED:-42}"

# ── Default: TRAIN_PERCENT=100 when nothing is specified ─────────────────────
if [ -z "${TRAINING_SHOTS}" ] && [ -z "${TRAIN_PERCENT}" ] && [ -z "${RANDOM_SHOT}" ]; then
    TRAIN_PERCENT="100"
fi

# ── Mutual-exclusion check ────────────────────────────────────────────────────
_modes_set=0
[ -n "${TRAINING_SHOTS}" ] && _modes_set=$((_modes_set + 1))
[ -n "${TRAIN_PERCENT}" ]  && _modes_set=$((_modes_set + 1))
[ -n "${RANDOM_SHOT}" ]    && _modes_set=$((_modes_set + 1))
if [ "$_modes_set" -gt 1 ]; then
    echo "ERROR: Set exactly one of TRAINING_SHOTS, TRAIN_PERCENT, or RANDOM_SHOT."
    echo "  TRAINING_SHOTS → fixed MedFMC split   e.g. TRAINING_SHOTS=20"
    echo "  TRAIN_PERCENT  → percentage subset     e.g. TRAIN_PERCENT=100"
    echo "  RANDOM_SHOT    → random N-shot         e.g. RANDOM_SHOT=10 SHOT_SEED=1"
    exit 1
fi

# ============================================================
# 3. DERIVED PATHS  (do not edit)
# ============================================================
if [ -n "${TRAIN_PERCENT}" ]; then
    # Percentage-based: reads trainval.txt, keeps P%.
    DATA_SUFFIX="percent${TRAIN_PERCENT}"
    DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train_percent${TRAIN_PERCENT}.json"
    REGEN_CMD="python ${PROJECT_DIR}/data/medfmc_to_llava.py \\
       --medfmc_root ${MEDFMC_ROOT} \\
       --output_dir  ${PROJECT_DIR}/data \\
       --tasks ${DATASET} --train_percent ${TRAIN_PERCENT}"
elif [ -n "${RANDOM_SHOT}" ]; then
    # Random N-shot: reads trainval.txt, samples exactly N (balanced by class).
    DATA_SUFFIX="rshot${RANDOM_SHOT}_seed${SHOT_SEED}"
    DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train_rshot${RANDOM_SHOT}_seed${SHOT_SEED}.json"
    REGEN_CMD="python ${PROJECT_DIR}/data/medfmc_to_llava.py \\
       --medfmc_root ${MEDFMC_ROOT} \\
       --output_dir  ${PROJECT_DIR}/data \\
       --tasks ${DATASET} --n_shot ${RANDOM_SHOT} --seed ${SHOT_SEED}"
else
    # Shot-based: MedFMC pre-defined split.
    if [ -n "${SHOT_EXP}" ]; then
        # Repeated-experiment split: {task}_{N}-shot_train_exp{K}.txt
        DATA_SUFFIX="shot${TRAINING_SHOTS}_exp${SHOT_EXP}"
        DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train_shot${TRAINING_SHOTS}_exp${SHOT_EXP}.json"
        REGEN_CMD="python ${PROJECT_DIR}/data/medfmc_to_llava.py \\
           --medfmc_root ${MEDFMC_ROOT} \\
           --output_dir  ${PROJECT_DIR}/data \\
           --tasks ${DATASET} --shot ${TRAINING_SHOTS} --exp ${SHOT_EXP}"
    else
        # Legacy single split: train_{N}.txt
        DATA_SUFFIX="shot${TRAINING_SHOTS}"
        DATA_PATH="${PROJECT_DIR}/data/${DATASET}_train_shot${TRAINING_SHOTS}.json"
        REGEN_CMD="python ${PROJECT_DIR}/data/medfmc_to_llava.py \\
           --medfmc_root ${MEDFMC_ROOT} \\
           --output_dir  ${PROJECT_DIR}/data \\
           --tasks ${DATASET} --shot ${TRAINING_SHOTS}"
    fi
fi

# Checkpoint directory naming — ICI-style: every launch gets its own UNIQUE folder,
# so re-runs (e.g. 789-val vs 100-val) NEVER overwrite each other.
#   {MMDD_HHMM}_llava-{dataset}-{strategy}-{suffix}[-icl{N}]_val{N}[-{RUN_TAG}]
# Inference does NOT derive this name — point CHECKPOINT_PATH at the folder you want
# (submit_inference.sh), exactly as the ICI orchestrator passes its run folder.
BASE_OUTPUT_DIR="/home/harinisrireddykandula/llava"   # mirrors ICI BASE_OUTPUT_DIR
# RUN_TAG: optional human label appended to the folder (e.g. RUN_TAG=v2). Uniqueness
# is already guaranteed by the timestamp; this is just for readability.
RUN_TAG="${RUN_TAG:-}"
[ -n "${RUN_TAG}" ] && RUN_TAG="-${RUN_TAG}"

# Validation-sample tag — this is the param that silently collided before
# (789 all-data vs 100 subsample). 0 = use all; empty eval = validation off.
if [ -z "${EVAL_DATA_PATH}" ]; then
    VAL_TAG="valoff"
elif [ "${MAX_VAL_SAMPLES}" = "0" ]; then
    VAL_TAG="valall"
else
    VAL_TAG="val${MAX_VAL_SAMPLES}"
fi

# Descriptive run name (mirrors DATA_SUFFIX — no silent name/data mismatch).
# ICL_SHOTS suffix added when > 0 so ICL-during-training runs are self-describing.
RUN_NAME="llava-${DATASET}-${STRATEGY}-${DATA_SUFFIX}"
if [ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null; then
    RUN_NAME="${RUN_NAME}-icl${ICL_SHOTS}"
fi

# OUTPUT_DIR may be set explicitly in the environment (e.g. to resume a specific
# run with SPRINT_RESUME=true); otherwise a fresh timestamped folder is generated.
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/checkpoints/$(date +%m%d_%H%M)_${RUN_NAME}_${VAL_TAG}${RUN_TAG}"
fi

# ============================================================
# 4. PRE-FLIGHT CHECKS
# ============================================================
echo "=================================================="
echo "  SPRInT-LLaVA Fine-tuning"
echo "=================================================="
echo "  Dataset         : ${DATASET}"
echo "  Strategy        : ${STRATEGY}"
if [ -n "${TRAIN_PERCENT}" ]; then
    echo "  Fine-tune subset: ${TRAIN_PERCENT}% of trainval.txt"
    echo "                    (fine-tuning data size -- NOT inference ICL shots)"
elif [ -n "${RANDOM_SHOT}" ]; then
    echo "  Fine-tune mode  : random ${RANDOM_SHOT}-shot  (seed=${SHOT_SEED})"
    echo "                    (one repetition of the few-shot experiment)"
else
    echo "  Fine-tune split : ${TRAINING_SHOTS}-shot training file"
    echo "                    (fine-tuning data size -- NOT inference ICL shots)"
fi
echo "  Data file       : ${DATA_PATH}"
echo "  Images root     : ${MEDFMC_ROOT}"
echo "  Base model      : ${MODEL_PATH}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Epochs          : ${NUM_TRAIN_EPOCHS}"
echo "  ICL shots       : ${ICL_SHOTS} (0 = no ICL context in training prompts)"
echo "  Max samples     : ${MAX_TRAIN_SAMPLES} (0 = all samples)"
echo "  LoRA r          : ${LORA_R}"
echo "  LoRA alpha      : ${LORA_ALPHA}"
if [ -n "${EVAL_DATA_PATH}" ]; then
    echo "  Val data        : ${EVAL_DATA_PATH} (max ${MAX_VAL_SAMPLES} samples/epoch)"
    echo "  Val probe batch : ${SPRINT_PROBE_BATCH_SIZE} (>1 = batched per-class scoring)"
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
    NUM_WORKERS=2
fi

# Optional validation args — only passed when EVAL_DATA_PATH is set.
EVAL_ARGS=""
if [ -n "${EVAL_DATA_PATH}" ]; then
    EVAL_ARGS="--eval_data_path ${EVAL_DATA_PATH} --max_val_samples ${MAX_VAL_SAMPLES} --validation_modes ${VALIDATION_MODES} --compute_val_auc_map ${COMPUTE_VAL_AUC_MAP}"
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
# If CUDA_VISIBLE_DEVICES is set (e.g. cuda_device=2 from the submit script),
# DeepSpeed auto-detection would add --include=localhost:2 while the env var
# simultaneously remaps that GPU to logical index 0 — causing "No slot '2'" error.
# Fix: convert CUDA_VISIBLE_DEVICES to an explicit --include and then unset it
# so DeepSpeed sees all physical GPUs and the --include picks the right one.
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    DS_INCLUDE="--include=localhost:${CUDA_VISIBLE_DEVICES}"
    unset CUDA_VISIBLE_DEVICES
else
    DS_INCLUDE=""
fi

MASTER_PORT=$(( 29500 + RANDOM % 500 ))
deepspeed ${DS_INCLUDE} --master_port ${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed "${LLAVA_DIR}/scripts/zero2.json" \
    --lora_enable True \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --mm_projector_lr 2e-5 \
    --sprint_strategy "${STRATEGY}" \
    --sprint_dataset "${DATASET}" \
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
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --seed "${SEED}" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${NUM_WORKERS}" \
    --lazy_preprocess True \
    --save_strategy epoch \
    --report_to none \
    ${EVAL_ARGS} \
    ${ICL_ARGS} \
    --max_train_samples "${MAX_TRAIN_SAMPLES}"
TRAIN_EXIT=$?

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "=================================================="
    echo "ERROR: Training FAILED (deepspeed exit code ${TRAIN_EXIT})."
    echo "   Check log for traceback above."
    echo "=================================================="
    exit ${TRAIN_EXIT}
fi

echo "=================================================="
echo "Training complete."
echo "   Checkpoint : ${OUTPUT_DIR}"
echo "   Symbols    : ${OUTPUT_DIR}/symbol_mappings.json"
echo "=================================================="
