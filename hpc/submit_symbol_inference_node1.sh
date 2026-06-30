#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

CONDA_ENV="${CONDA_ENV:-qwen}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
DATASET_TYPE="${DATASET_TYPE:-voxceleb-hvb-voxpopuli-meld_emotion}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"                                            # required: path to .pt checkpoint file
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"                                         # samples per dataset (0 = full val set)
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"                                                  # few-shot examples in prompt
NUM_WORKERS="${NUM_WORKERS:-2}"
VALIDATION_MODES="${VALIDATION_MODES:-original,fixed,fresh}"
SYMBOL_MAP_FILE="${SYMBOL_MAP_FILE:-}"                                            # optional: JSON file to override symbol mappings
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/training/symbol_training}"
LOG_DIR="${LOGS_INFERENCE_DIR:-${HOME}/training/symbol_training/logs_inference}/$(date +"%Y-%m-%d")"

SAMPLES_TAG="${MAX_VAL_SAMPLES}"
[[ "${SAMPLES_TAG}" == "0" ]] && SAMPLES_TAG="all"
RUN_NAME="${RUN_NAME:-$(date +"%H%M%S")_inference_${MODEL_TYPE}_${DATASET_TYPE}_${SAMPLES_TAG}}"

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    export _NOHUP_LAUNCHED=1 RUN_NAME
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    printf 'Inference launched in background (PID: %s)\n' "$!"
    printf 'Follow logs: tail -f "%s"\n' "${LOG_FILE}"
    exit 0
fi

printf '%s\n' "============================================================"
printf '%s\n' "Starting Symbol Inference on node1"
printf '%s\n' "Project Root:    ${PROJECT_ROOT}"
printf '%s\n' "Conda Env:       ${CONDA_ENV}"
printf '%s\n' "Model:           ${MODEL_TYPE}"
printf '%s\n' "Dataset:         ${DATASET_TYPE}"
printf '%s\n' "Checkpoint:      ${CHECKPOINT_PATH}"
printf '%s\n' "Symbol Map:      ${SYMBOL_MAP_FILE:-<from checkpoint>}"
printf '%s\n' "Samples:         ${SAMPLES_TAG}"
printf '%s\n' "Run Name:        ${RUN_NAME}"
printf '%s\n' "Log File:        ${LOG_FILE}"
printf '%s\n' "============================================================"

export CUDA_VISIBLE_DEVICES

python inference.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --device "${DEVICE}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --num_examples "${NUM_EXAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    $( [[ -n "${SYMBOL_MAP_FILE}" ]] && printf '%s %s' "--symbol_map_file" "${SYMBOL_MAP_FILE}" ) \
    >> "${LOG_FILE}" 2>&1
