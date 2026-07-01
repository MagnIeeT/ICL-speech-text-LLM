#!/bin/bash
set -e

# ============================================================
# SPRInT-LLaVA Training - HPC Submit Script
# ============================================================
# HOW TO USE:
#
#   Single run:
#       STRATEGY=regular   DATASET=colon TRAIN_PERCENT=100 bash submit_training.sh
#       STRATEGY=two_token DATASET=colon TRAIN_PERCENT=100 bash submit_training.sh
#
#   ICL during training (K example images embedded in each training prompt):
#       ICL_SHOTS=1 STRATEGY=two_token DATASET=chest TRAINING_SHOTS=10 SHOT_EXP=1 bash submit_training.sh
#       # optional separate example pool:
#       ICL_SHOTS=1 ICL_POOL_PATH=/home/harinisrireddykandula/LLaVA/sprint_vision/data/chest_train_shot10_exp1.json ... bash submit_training.sh
#
#   Chain training jobs:
#       JOB1=$(STRATEGY=regular   DATASET=colon bash submit_training.sh --print-id)
#       JOB2=$(hold_job_id="$JOB1" STRATEGY=two_token DATASET=colon bash submit_training.sh --print-id)
#
#   Chain training → inference:
#       TRAIN_JOB=$(STRATEGY=regular DATASET=colon bash submit_training.sh --print-id)
#       hold_job_id="$TRAIN_JOB" dataset=colon strategy=regular icl_shots=0 bash submit_inference.sh
# ============================================================

# ========================================
# Configuration — edit these values
# ========================================
DATASET="${DATASET:-colon}"      # colon | chest | endo
STRATEGY="${STRATEGY:-regular}"  # regular | two_token | ed_ft | id_ft | lf_ft
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
# ── ICL during training ──────────────────────────────────────────────────────
# ICL_SHOTS: number of in-context example images+labels embedded in each training
#            prompt. 0 = standard supervised fine-tuning (no ICL context).
# ICL_POOL_PATH: optional separate JSON pool to draw examples from. Empty = use
#            the training file itself (self-pool, the sample excludes itself).
# NOTE: with the new class-definition blocks the instruction is long (~450 tok for
#       chest's 19 defs). Each ICL shot repeats it, so high ICL_SHOTS can exceed
#       --model_max_length (2048) and get truncated. Keep ICL_SHOTS small (≈1)
#       for chest unless you also raise MODEL_MAX_LENGTH.
ICL_SHOTS="${ICL_SHOTS:-0}"
ICL_POOL_PATH="${ICL_POOL_PATH:-}"
num_samples="${num_samples:-0}"   # 0 = ALL samples (matches inference script convention)
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Validation modes: comma-separated subset of fixed,original,fresh
# Examples:
#   original                → only text-gen F1, fastest
#   fixed,original,fresh    → all 3 symbol modes (default, mirrors ICI)
VALIDATION_MODES="${VALIDATION_MODES:-fixed,original,fresh}"
# Whether to compute AUC (colon) or mAP+AUC (chest/endo) — MedFMC primary metrics
# true = full MedFMC-aligned eval; false = macro_F1 only (faster)
COMPUTE_VAL_AUC_MAP="${COMPUTE_VAL_AUC_MAP:-true}"
# Batched per-class VALIDATION scoring (chest/endo). >1 enables left-padded
# batched P(Yes) scoring for faster val AUC/mAP; 1 = original unbatched (default,
# byte-identical). Mirrors inference's probe_batch_size; forwarded as
# SPRINT_PROBE_BATCH_SIZE (read by validation.py). When >1, [BATCH-VERIFY] OK is
# logged on the first val sample. Example: probe_batch_size=8 ... bash submit_training.sh
probe_batch_size="${probe_batch_size:-1}"
# Validation subsample cap (mirrors run_sprint_finetune.sh default). Also goes into
# the checkpoint folder name as the val-tag, so 789-val and 100-val never collide.
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"   # 0 = use all; "" via EVAL_DATA_PATH=none = off
# EVAL_DATA_PATH=none disables validation entirely (forwarded so the val-tag is right).
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
# RUN_TAG: optional human label appended to the checkpoint folder (uniqueness is
# already guaranteed by the timestamp; this is just for readability).
RUN_TAG="${RUN_TAG:-}"

# Training data mode — set exactly one:
#   TRAIN_PERCENT               → percentage of trainval.txt (default: 100)
#   TRAINING_SHOTS + SHOT_EXP   → MedFMC repeated-experiment split (recommended for paper)
#   RANDOM_SHOT + SHOT_SEED     → custom random N-shot sampling
TRAINING_SHOTS="${TRAINING_SHOTS:-}"  # e.g. 10  (use with SHOT_EXP for paper protocol)
SHOT_EXP="${SHOT_EXP:-}"              # experiment index 1-5
RANDOM_SHOT="${RANDOM_SHOT:-}"        # empty = not in random-shot mode
SHOT_SEED="${SHOT_SEED:-1}"           # RNG seed for random-shot sampling
if [ -z "${RANDOM_SHOT}" ] && [ -z "${TRAINING_SHOTS}" ]; then
    TRAIN_PERCENT="${TRAIN_PERCENT:-100}"
else
    TRAIN_PERCENT="${TRAIN_PERCENT:-}"
fi

hostname="${hostname:-n10}"
cuda_device="${cuda_device:-0}"
walltime="${walltime:-48:00:00}"

# Set to a job ID (e.g. "12345.cluster") to wait for that job first.
hold_job_id="${hold_job_id:-}"

# Paths
LLAVA_DIR="${LLAVA_DIR:-/home/harinisrireddykandula/LLaVA}"
FINETUNE_SCRIPT="${LLAVA_DIR}/sprint_vision/run_sprint_finetune.sh"
BASE_OUTPUT_DIR="/home/harinisrireddykandula/llava"   # mirrors ICI BASE_OUTPUT_DIR
output_dir="${BASE_OUTPUT_DIR}/logs"

# ========================================
# Auto setup
# ========================================
CURRENT_DATETIME=$(date +"%d%m_%H%M")
TODAY=$(date +"%Y-%m-%d")
if [ -n "${RANDOM_SHOT}" ]; then
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_rshot${RANDOM_SHOT}_s${SHOT_SEED}_${NUM_TRAIN_EPOCHS}ep"
elif [ -n "${TRAINING_SHOTS}" ] && [ -n "${SHOT_EXP}" ]; then
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_shot${TRAINING_SHOTS}_exp${SHOT_EXP}_${NUM_TRAIN_EPOCHS}ep"
elif [ -n "${TRAINING_SHOTS}" ]; then
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_shot${TRAINING_SHOTS}_${NUM_TRAIN_EPOCHS}ep"
else
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_${TRAIN_PERCENT}pct_${NUM_TRAIN_EPOCHS}ep"
fi
# Append ICL context suffix when training uses ICL (ICL_SHOTS=0 is the standard baseline)
[ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null && RUN_NAME="${RUN_NAME}_icl${ICL_SHOTS}"

# ── Checkpoint OUTPUT_DIR (computed HERE so chained inference jobs know the exact
#    path at submit time — the folder doesn't exist yet when JOB_A/JOB_B are queued).
#    Must mirror run_sprint_finetune.sh's DATA_SUFFIX + val-tag + naming exactly;
#    it is exported below so run_sprint_finetune.sh uses it verbatim. ─────────────
if [ -n "${TRAIN_PERCENT}" ]; then
    DATA_SUFFIX="percent${TRAIN_PERCENT}"
elif [ -n "${RANDOM_SHOT}" ]; then
    DATA_SUFFIX="rshot${RANDOM_SHOT}_seed${SHOT_SEED}"
elif [ -n "${SHOT_EXP}" ]; then
    DATA_SUFFIX="shot${TRAINING_SHOTS}_exp${SHOT_EXP}"
else
    DATA_SUFFIX="shot${TRAINING_SHOTS}"
fi
if [ "${EVAL_DATA_PATH}" = "none" ]; then
    VAL_TAG="valoff"
elif [ "${MAX_VAL_SAMPLES}" = "0" ]; then
    VAL_TAG="valall"
else
    VAL_TAG="val${MAX_VAL_SAMPLES}"
fi
CKPT_RUN_NAME="llava-${DATASET}-${STRATEGY}-${DATA_SUFFIX}"
[ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null && CKPT_RUN_NAME="${CKPT_RUN_NAME}-icl${ICL_SHOTS}"
CKPT_RUN_TAG=""
[ -n "${RUN_TAG}" ] && CKPT_RUN_TAG="-${RUN_TAG}"
CKPT_TS=$(date +"%m%d_%H%M")   # MMDD_HHMM — matches run_sprint_finetune.sh naming
OUTPUT_DIR="${BASE_OUTPUT_DIR}/checkpoints/${CKPT_TS}_${CKPT_RUN_NAME}_${VAL_TAG}${CKPT_RUN_TAG}"

LOG_DIR="${output_dir}/${TODAY}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "${LOG_FILE}"

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:${hold_job_id}"
    HOLD_MSG="Waiting for job: ${hold_job_id}"
else
    HOLD_FLAG=""
    HOLD_MSG="Starting immediately (no dependency)"
fi

echo "=========================================="
echo "SPRInT-LLaVA Training Configuration"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Dataset:     ${DATASET}"
echo "Strategy:    ${STRATEGY}"
if [ -n "${RANDOM_SHOT}" ]; then
    echo "Few-shot:    ${RANDOM_SHOT}-shot  (seed=${SHOT_SEED})"
elif [ -n "${TRAINING_SHOTS}" ] && [ -n "${SHOT_EXP}" ]; then
    echo "Few-shot:    ${TRAINING_SHOTS}-shot  exp${SHOT_EXP}  (MedFMC official split)"
elif [ -n "${TRAINING_SHOTS}" ]; then
    echo "Few-shot:    ${TRAINING_SHOTS}-shot  (legacy train_${TRAINING_SHOTS}.txt)"
else
    echo "Train %:     ${TRAIN_PERCENT}%  (of trainval.txt)"
fi
echo "Epochs:      ${NUM_TRAIN_EPOCHS}"
echo "ICL Shots:   ${ICL_SHOTS}"
echo "ICL Pool:    ${ICL_POOL_PATH:-<training file (self-pool)>}"
echo "Num Samples: ${num_samples} (0 = ALL)"
echo "LoRA r:      ${LORA_R}"
echo "LoRA alpha:  ${LORA_ALPHA}"
echo "Val modes:   ${VALIDATION_MODES}"
echo "Val samples: ${MAX_VAL_SAMPLES} (0 = all)"
echo "Val AUC/mAP: ${COMPUTE_VAL_AUC_MAP}"
echo "Val probe batch: ${probe_batch_size} (>1 = batched per-class scoring)"
echo "Checkpoint:  ${OUTPUT_DIR}"
echo "Hostname:    ${hostname}"
echo "Walltime:    ${walltime}"
echo "Dependency:  ${HOLD_MSG}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

# ========================================
# Write job script
# ========================================
TMPJOB=$(mktemp /tmp/llava_train_XXXX.sh)

cat << EOF > ${TMPJOB}
#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=${cuda_device}
exec > >(stdbuf -oL tee -a ${LOG_FILE}) 2>&1

echo "=========================================="
echo "Training job started at: \$(date)"
echo "Running on host: \$(hostname)"
echo "Dataset: ${DATASET}  Strategy: ${STRATEGY}  Epochs: ${NUM_TRAIN_EPOCHS}"
echo "=========================================="

nvidia-smi

DATASET=${DATASET} \
STRATEGY=${STRATEGY} \
TRAIN_PERCENT=${TRAIN_PERCENT} \
TRAINING_SHOTS=${TRAINING_SHOTS} \
SHOT_EXP=${SHOT_EXP} \
RANDOM_SHOT=${RANDOM_SHOT} \
SHOT_SEED=${SHOT_SEED} \
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS} \
ICL_SHOTS=${ICL_SHOTS} \
ICL_POOL_PATH=${ICL_POOL_PATH} \
MAX_TRAIN_SAMPLES=${num_samples} \
LORA_R=${LORA_R} \
LORA_ALPHA=${LORA_ALPHA} \
VALIDATION_MODES=${VALIDATION_MODES} \
COMPUTE_VAL_AUC_MAP=${COMPUTE_VAL_AUC_MAP} \
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES} \
EVAL_DATA_PATH=${EVAL_DATA_PATH} \
SPRINT_PROBE_BATCH_SIZE=${probe_batch_size} \
OUTPUT_DIR=${OUTPUT_DIR} \
LLAVA_DIR=${LLAVA_DIR} \
bash ${FINETUNE_SCRIPT}

echo "=========================================="
echo "Training job completed at: \$(date)"
echo "=========================================="
EOF

chmod +x ${TMPJOB}

# ========================================
# Submit and capture job ID
# ========================================
JOB_ID=$(qsub -q workq \
    $HOLD_FLAG \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=${hostname} \
    -l walltime=${walltime} \
    -o /dev/null \
    -j oe \
    -S /bin/bash \
    ${TMPJOB})

echo ""
echo "=========================================="
echo "Training Job Submitted!"
echo "  Job ID:  ${JOB_ID}"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Status:  qstat | grep harinis"
echo ""
echo "To chain inference (0 / 1 / 5 shots) after this job:"
echo "  # Exact checkpoint path (already created with this name by the training job):"
echo "  CHECKPOINT_PATH=${OUTPUT_DIR}"
echo ""
echo "  JOB_A=\$(hold_job_id=${JOB_ID} CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=0 bash submit_inference.sh --print-id)"
echo "  JOB_B=\$(hold_job_id=\$JOB_A   CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=1 bash submit_inference.sh --print-id)"
echo "         hold_job_id=\$JOB_B   CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=5 bash submit_inference.sh"
echo "=========================================="

if [[ "${1}" == "--print-id" ]]; then
    echo "${JOB_ID}"
fi
