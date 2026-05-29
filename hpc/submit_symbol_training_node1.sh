#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-qwen}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
DATASET_TYPE="${DATASET_TYPE:-voxceleb}"
VAL_DATASET_TYPE="${VAL_DATASET_TYPE:-${DATASET_TYPE}}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_NAME="${RUN_NAME:-$(date +"%Y%m%d_%H%M%S")_${MODEL_TYPE}_${DATASET_TYPE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/training/symbol_training}"
LOG_DIR="${LOG_DIR:-${HOME}/training/logs/$(date +"%Y-%m-%d")}"
LORA_LR="${LORA_LR:-1e-5}"
LORA_EPOCHS="${LORA_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VALIDATION_MODES="${VALIDATION_MODES:-fixed,original,fresh}"
SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_epoch}"
NO_SYMBOLS="${NO_SYMBOLS:-true}"
DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-false}"
DIFF_SYMBOL_ENABLED="${DIFF_SYMBOL_ENABLED:-true}"
SWAP_LABELS="${SWAP_LABELS:-false}"

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi

# Avoid conda activate failing under 'set -u' if MKL_INTERFACE_LAYER is unset
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

conda activate "${CONDA_ENV}"

# Prefer conda's libstdc++ over the system one
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Avoid tokenizers parallelism warnings after fork
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

printf '%s\n' "============================================================"
printf '%s\n' "Starting Symbol Training on node1"
printf '%s\n' "Project Root: ${PROJECT_ROOT}"
printf '%s\n' "Conda Env:    ${CONDA_ENV}"
printf '%s\n' "Model:        ${MODEL_TYPE}"
printf '%s\n' "Dataset:      ${DATASET_TYPE}"
printf '%s\n' "Val Dataset:  ${VAL_DATASET_TYPE}"
printf '%s\n' "Run Name:     ${RUN_NAME}"
printf '%s\n' "Log File:     ${LOG_FILE}"
printf '%s\n' "============================================================"

export CUDA_VISIBLE_DEVICES

python train.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --val_dataset_type "${VAL_DATASET_TYPE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --lora_lr "${LORA_LR}" \
    --lora_epochs "${LORA_EPOCHS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --symbol_update_strategy "${SYMBOL_UPDATE_STRATEGY}" \
    $( [[ "${NO_SYMBOLS}" == "true" ]] && printf '%s' "--no_symbols" ) \
    $( [[ "${DYNAMIC_SYMBOLS}" == "true" ]] && printf '%s' "--dynamic_symbols" ) \
    $( [[ "${DIFF_SYMBOL_ENABLED}" == "true" ]] && printf '%s' "--diff_symbol_enabled" ) \
    $( [[ "${SWAP_LABELS}" == "true" ]] && printf '%s' "--swap_labels" ) \
    2>&1 | tee "${LOG_FILE}"
