#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Symbol Adapter Training - New Cluster Submit Script
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ------------------------------------------------------------
# Job Configuration (edit these)
# ------------------------------------------------------------
MODEL_TYPE="${MODEL_TYPE:-qwen}"                     # qwen | salmonn | flamingo
DATASET_TYPE="${DATASET_TYPE:-voxceleb}"
VAL_DATASET_TYPE="${VAL_DATASET_TYPE:-voxceleb}"

NO_SYMBOLS="${NO_SYMBOLS:-false}"
DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-false}"
DIFF_SYMBOL_ENABLED="${DIFF_SYMBOL_ENABLED:-false}"
SWAP_LABELS="${SWAP_LABELS:-false}"
USE_DPO="${USE_DPO:-false}"

SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_epoch}"
VALIDATION_MODES="${VALIDATION_MODES:-fixed,original,fresh}"

LORA_LR="${LORA_LR:-1e-5}"
LORA_EPOCHS="${LORA_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_WORKERS="${NUM_WORKERS:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

INPUT_MODE="${INPUT_MODE:-speech_only}"
FEWSHOT_MODE="${FEWSHOT_MODE:-text}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-5}"

SYMBOL_DIFFICULTY="${SYMBOL_DIFFICULTY:-random}"
VAL_SYMBOL_DIFFICULTY="${VAL_SYMBOL_DIFFICULTY:-easy}"
NUM_SYMBOL_MAPPINGS="${NUM_SYMBOL_MAPPINGS:-20}"

DSPO_ROUTER_LR="${DSPO_ROUTER_LR:-1e-2}"
DSPO_TAU_ANNEAL_RATE="${DSPO_TAU_ANNEAL_RATE:-0.0001}"
DSPO_SLOT_ONLY="${DSPO_SLOT_ONLY:-false}"
DSPO_PHASE0_EPOCHS="${DSPO_PHASE0_EPOCHS:-0}"
DSPO_PHASE1_PATIENCE="${DSPO_PHASE1_PATIENCE:-0}"
DSPO_PHASE1_EPOCHS="${DSPO_PHASE1_EPOCHS:-0}"
DSPO_NUM_SLOTS="${DSPO_NUM_SLOTS:-25}"
DSPO_SLOT_VOCAB_SIZE="${DSPO_SLOT_VOCAB_SIZE:-100}"
DSPO_ROTATION_INTERVAL="${DSPO_ROTATION_INTERVAL:--1}"
DSPO_PHASE2_ROTATION="${DSPO_PHASE2_ROTATION:--1}"

DPO_BETA="${DPO_BETA:-0.1}"

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
OUTPUT_DIR="${BASE_OUTPUT_DIR}/symbol_training"
CHECKPOINT_DIR="${CHECKPOINT_DIR}"
METRICS_DIR="${METRICS_DIR}"
LOG_DIR="${LOGS_DIR}/$(date +"%Y-%m-%d")"

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${METRICS_DIR}" "${LOG_DIR}"

# ------------------------------------------------------------
# Run name
# ------------------------------------------------------------
MODE="fixed"
[[ "${NO_SYMBOLS}" == "true" ]] && MODE="baseline"
[[ "${DYNAMIC_SYMBOLS}" == "true" ]] && MODE="${SYMBOL_UPDATE_STRATEGY}"
[[ "${DIFF_SYMBOL_ENABLED}" == "true" ]] && MODE="dspo"
[[ "${USE_DPO}" == "true" ]] && MODE="symdpo"
[[ "${SWAP_LABELS}" == "true" ]] && MODE="${MODE}_swap"

RUN_NAME="${RUN_NAME:-$(date +"%d%m_%H%M")_${MODEL_TYPE}_${DATASET_TYPE}_${MODE}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ------------------------------------------------------------
# Conda
# ------------------------------------------------------------
if [[ "${MODEL_TYPE}" == "flamingo" ]]; then
    CONDA_ENV="flamingo"
elif [[ "${MODEL_TYPE}" == "qwen" ]]; then
    CONDA_ENV="qwen"
else
    CONDA_ENV="salmonn2"
fi

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export HF_HOME
export TRANSFORMERS_CACHE

if [[ "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    export _NOHUP_LAUNCHED=1
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    echo "Training launched."
    echo "PID: $!"
    echo "Logs: tail -f ${LOG_FILE}"
    exit 0
fi

echo "=================================================="
echo "Run Name : ${RUN_NAME}"
echo "Model    : ${MODEL_TYPE}"
echo "Dataset  : ${DATASET_TYPE}"
echo "Conda    : ${CONDA_ENV}"
echo "GPU      : ${CUDA_VISIBLE_DEVICES}"
echo "=================================================="

python train.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --val_dataset_type "${VAL_DATASET_TYPE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_examples "${NUM_EXAMPLES}" \
    --val_num_examples "${VAL_NUM_EXAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --lora_lr "${LORA_LR}" \
    --lora_epochs "${LORA_EPOCHS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    --logs_dir "${LOG_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --symbol_update_strategy "${SYMBOL_UPDATE_STRATEGY}" \
    --symbol_difficulty "${SYMBOL_DIFFICULTY}" \
    --val_symbol_difficulty "${VAL_SYMBOL_DIFFICULTY}" \
    --num_symbol_mappings "${NUM_SYMBOL_MAPPINGS}" \
    --input_mode "${INPUT_MODE}" \
    --fewshot_mode "${FEWSHOT_MODE}" \
    --dspo_router_lr "${DSPO_ROUTER_LR}" \
    --dspo_tau_anneal_rate "${DSPO_TAU_ANNEAL_RATE}" \
    --dspo_phase0_epochs "${DSPO_PHASE0_EPOCHS}" \
    --dspo_phase1_patience "${DSPO_PHASE1_PATIENCE}" \
    --dspo_phase1_epochs "${DSPO_PHASE1_EPOCHS}" \
    --dspo_num_slots "${DSPO_NUM_SLOTS}" \
    --dspo_slot_vocab_size "${DSPO_SLOT_VOCAB_SIZE}" \
    --dspo_rotation_interval "${DSPO_ROTATION_INTERVAL}" \
    --dspo_phase2_rotation "${DSPO_PHASE2_ROTATION}" \
    $( [[ "${NO_SYMBOLS}" == "true" ]] && echo --no_symbols ) \
    $( [[ "${DYNAMIC_SYMBOLS}" == "true" ]] && echo --dynamic_symbols ) \
    $( [[ "${DIFF_SYMBOL_ENABLED}" == "true" ]] && echo --diff_symbol_enabled ) \
    $( [[ "${DSPO_SLOT_ONLY}" == "true" ]] && echo --dspo_slot_only ) \
    $( [[ "${SWAP_LABELS}" == "true" ]] && echo --swap_labels ) \
    $( [[ "${USE_DPO}" == "true" ]] && echo --use_dpo ) \
    $( [[ "${USE_DPO}" == "true" ]] && echo --dpo_beta "${DPO_BETA}" ) \
    >> "${LOG_FILE}" 2>&1

echo "Training finished."
