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
#       ICL_SHOTS=1 ICL_POOL_PATH=/home/harinis/LLaVA/sprint_vision/data/chest_train_shot10_exp1.json ... bash submit_training.sh
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
DATASET=chest      # colon | chest | endo
STRATEGY=two_token  # regular | two_token | ed_ft | id_ft | lf_ft
NUM_TRAIN_EPOCHS=5
# ── ICL during training ──────────────────────────────────────────────────────
# ICL_SHOTS: number of in-context example images+labels embedded in each training
#            prompt. 0 = standard supervised fine-tuning (no ICL context).
# ICL_POOL_PATH: optional separate JSON pool to draw examples from. Empty = use
#            the training file itself (self-pool, the sample excludes itself).
# NOTE: with the new class-definition blocks the instruction is long (~450 tok for
#       chest's 19 defs). Each ICL shot repeats it, so high ICL_SHOTS can exceed
#       --model_max_length (2048) and get truncated. Keep ICL_SHOTS small (≈1)
#       for chest unless you also raise MODEL_MAX_LENGTH.
ICL_SHOTS=0
ICL_POOL_PATH="${ICL_POOL_PATH:-}"
num_samples=0   # 0 = ALL samples (matches inference script convention)
LORA_R=8
LORA_ALPHA=32

# Validation modes: comma-separated subset of fixed,original,fresh
# Examples:
#   original                → only text-gen F1, fastest
#   fixed,original,fresh    → all 3 symbol modes (default, mirrors ICI)
VALIDATION_MODES=orginal
# Whether to compute AUC (colon) or mAP+AUC (chest/endo) — MedFMC primary metrics
# true = full MedFMC-aligned eval; false = macro_F1 only (faster)
COMPUTE_VAL_AUC_MAP="${COMPUTE_VAL_AUC_MAP:-true}"

# Training data mode — set exactly one:
#   TRAIN_PERCENT               → percentage of trainval.txt (default: 100)
#   TRAINING_SHOTS + SHOT_EXP   → MedFMC repeated-experiment split (recommended for paper)
#   RANDOM_SHOT + SHOT_SEED     → custom random N-shot sampling
TRAINING_SHOTS=10  # e.g. 10  (use with SHOT_EXP for paper protocol)
SHOT_EXP=2             # experiment index 1-5
RANDOM_SHOT="${RANDOM_SHOT:-}"        # empty = not in random-shot mode
SHOT_SEED="${SHOT_SEED:-1}"           # RNG seed for random-shot sampling
if [ -z "${RANDOM_SHOT}" ] && [ -z "${TRAINING_SHOTS}" ]; then
    TRAIN_PERCENT="${TRAIN_PERCENT:-100}"
else
    TRAIN_PERCENT="${TRAIN_PERCENT:-}"
fi

hostname=n6
cuda_device=2
walltime="${walltime:-48:00:00}"

# Set to a job ID (e.g. "12345.cluster") to wait for that job first.
hold_job_id=12614.eehpc

# Paths
LLAVA_DIR="${LLAVA_DIR:-/home/harinis/LLaVA}"
FINETUNE_SCRIPT="${LLAVA_DIR}/sprint_vision/run_sprint_finetune.sh"
BASE_OUTPUT_DIR="/home/leapers/weights/harinis/llava"   # mirrors ICI BASE_OUTPUT_DIR
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
echo "Val AUC/mAP: ${COMPUTE_VAL_AUC_MAP}"
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
echo "  # Derive checkpoint path first:"
if [ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null; then
    _CKPT_HINT="/home/leapers/weights/harinis/llava/checkpoints/llava-${DATASET}-${STRATEGY}-...-icl${ICL_SHOTS}"
else
    _CKPT_HINT="/home/leapers/weights/harinis/llava/checkpoints/llava-${DATASET}-${STRATEGY}-..."
fi
echo "  CHECKPOINT_PATH=${_CKPT_HINT}"
echo ""
echo "  JOB_A=\$(hold_job_id=${JOB_ID} CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=0 bash submit_inference.sh --print-id)"
echo "  JOB_B=\$(hold_job_id=\$JOB_A   CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=1 bash submit_inference.sh --print-id)"
echo "         hold_job_id=\$JOB_B   CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=5 bash submit_inference.sh"
echo "=========================================="

if [[ "${1}" == "--print-id" ]]; then
    echo "${JOB_ID}"
fi
