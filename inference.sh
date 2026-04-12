#!/bin/bash
set -euo pipefail

show_help() {
  cat << 'USAGE'
Usage:
  Environment-variable driven local inference script (no qsub).

  Example:
    MODEL_TYPE=salmonn \
    CHECKPOINT_PATH=/path/to/checkpoint.pt \
    DATASET_TYPE=hvb-voxceleb-voxpopuli-meld_emotion \
    MAX_VAL_SAMPLES=0 \
    NUM_EXAMPLES=0 \
    NO_SYMBOLS=true \
    OUTPUT_DIR=./results \
    ./inference.sh

Available options (env vars):
  MODEL_TYPE                 : salmonn | qwen
  CHECKPOINT_PATH            : path to checkpoint file (required)
  DATASET_TYPE               : hyphen-joined list from {voxceleb,hvb,voxpopuli,meld_emotion}

  MAX_VAL_SAMPLES            : int (default 0 means all)
  NUM_EXAMPLES               : int (few-shot example count for inference pipeline)
  NO_SYMBOLS                 : true | false
  DEVICE                     : cuda:0, cuda:1, cpu, ...

  OUTPUT_DIR                 : output base directory
  RUN_NAME                   : run identifier
  SCRIPT_PATH                : python entrypoint (default: <repo>/inference.py)
  PYTHON_BIN                 : python executable (default: python)

Optional conda activation:
  CONDA_ENV                  : if set, script will try to activate this conda env
  CONDA_SH                   : path to conda.sh (default: /home/leapers/anaconda3/etc/profile.d/conda.sh)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DATASET_TYPE="${DATASET_TYPE:-hvb-voxceleb-voxpopuli-meld_emotion}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"
NO_SYMBOLS="${NO_SYMBOLS:-false}"
DEVICE="${DEVICE:-cuda:0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/inference.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

VALID_DATASETS="voxceleb hvb voxpopuli meld_emotion"

validate_dataset_list() {
  local list="$1"
  IFS='-' read -ra parts <<< "$list"
  for d in "${parts[@]}"; do
    if [[ ! " ${VALID_DATASETS} " =~ " ${d} " ]]; then
      echo "ERROR: Unsupported dataset '${d}' in '${list}'."
      echo "Allowed values: voxceleb, hvb, voxpopuli, meld_emotion"
      exit 1
    fi
  done
}

if [[ "${MODEL_TYPE}" != "salmonn" && "${MODEL_TYPE}" != "qwen" ]]; then
  echo "ERROR: MODEL_TYPE must be one of: salmonn, qwen"
  exit 1
fi

if [[ "${NO_SYMBOLS}" != "true" && "${NO_SYMBOLS}" != "false" && "${NO_SYMBOLS}" != "True" && "${NO_SYMBOLS}" != "False" ]]; then
  echo "ERROR: NO_SYMBOLS must be true or false"
  exit 1
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: CHECKPOINT_PATH is required"
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: inference.py not found at SCRIPT_PATH=${SCRIPT_PATH}"
  exit 1
fi

validate_dataset_list "${DATASET_TYPE}"

TODAY="$(date +"%Y-%m-%d")"
CURRENT_DATETIME="$(date +"%d%m_%H%M")"
RUN_NAME="${RUN_NAME:-${CURRENT_DATETIME}_infer_${MODEL_TYPE}}"
LOG_DIR="${OUTPUT_DIR}/orchestrator_logs/${TODAY}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-/home/leapers/anaconda3/etc/profile.d/conda.sh}"
if [[ -n "${CONDA_ENV}" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda deactivate || true
    conda activate "${CONDA_ENV}"
  else
    echo "WARNING: CONDA_SH not found at ${CONDA_SH}; continuing without conda activation"
  fi
fi

CMD=("${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --model_type "${MODEL_TYPE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --dataset_type "${DATASET_TYPE}" \
  --device "${DEVICE}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --num_examples "${NUM_EXAMPLES}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}")

if [[ "${NO_SYMBOLS}" == "True" || "${NO_SYMBOLS}" == "true" ]]; then
  CMD+=(--no_symbols)
fi

echo "Starting local symbol inference run: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
