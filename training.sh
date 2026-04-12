#!/bin/bash
set -euo pipefail

show_help() {
  cat << 'USAGE'
Usage:
  Environment-variable driven local training script (no qsub).

  Example:
    MODEL_TYPE=salmonn \
    DATASET_TYPE=hvb-meld_emotion \
    VAL_DATASET_TYPE=hvb-meld_emotion \
    NO_SYMBOLS=true \
    OUTPUT_DIR=./results/symbol_training \
    ./training.sh

Available options (env vars):
  MODEL_TYPE                 : salmonn | qwen
  DATASET_TYPE               : hyphen-joined list from {voxceleb,hvb,voxpopuli,meld_emotion}
  VAL_DATASET_TYPE           : same format as DATASET_TYPE (default: DATASET_TYPE)

  NO_SYMBOLS                 : true | false
  DYNAMIC_SYMBOLS            : true | false
  SYMBOL_UPDATE_STRATEGY     : per_epoch | per_instance

  LORA_LR                    : float (default 1e-5)
  LORA_EPOCHS                : int   (default 5)
  BATCH_SIZE                 : int   (default 1)
  GRADIENT_ACCUMULATION_STEPS: int   (default 8)
  MAX_GRAD_NORM              : float (default 1)
  MAX_SAMPLES                : int   (default 0 means all)

  DEVICE                     : cuda:0, cuda:1, cpu, ...
  OUTPUT_DIR                 : output base directory
  RUN_NAME                   : run identifier
  SCRIPT_PATH                : python entrypoint (default: <repo>/train.py)
  PYTHON_BIN                 : python executable (default: python)

Optional conda activation:
  CONDA_ENV                  : if set, script will try to activate this conda env
  CONDA_SH                   : path to conda.sh (default: /home/leapers/anaconda3/etc/profile.d/conda.sh)

Notes:
  1) If NO_SYMBOLS=true, --no_symbols is passed and --dynamic_symbols is ignored.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
DATASET_TYPE="${DATASET_TYPE:-hvb-meld_emotion}"
VAL_DATASET_TYPE="${VAL_DATASET_TYPE:-${DATASET_TYPE}}"
DEVICE="${DEVICE:-cuda:0}"

LORA_LR="${LORA_LR:-1e-5}"
LORA_EPOCHS="${LORA_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

NO_SYMBOLS="${NO_SYMBOLS:-false}"
DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-false}"
SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_epoch}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/train.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/symbol_training}"
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

if [[ "${SYMBOL_UPDATE_STRATEGY}" != "per_epoch" && "${SYMBOL_UPDATE_STRATEGY}" != "per_instance" ]]; then
  echo "ERROR: SYMBOL_UPDATE_STRATEGY must be one of: per_epoch, per_instance"
  exit 1
fi

if [[ "${NO_SYMBOLS}" != "true" && "${NO_SYMBOLS}" != "false" && "${NO_SYMBOLS}" != "True" && "${NO_SYMBOLS}" != "False" ]]; then
  echo "ERROR: NO_SYMBOLS must be true or false"
  exit 1
fi

if [[ "${DYNAMIC_SYMBOLS}" != "true" && "${DYNAMIC_SYMBOLS}" != "false" && "${DYNAMIC_SYMBOLS}" != "True" && "${DYNAMIC_SYMBOLS}" != "False" ]]; then
  echo "ERROR: DYNAMIC_SYMBOLS must be true or false"
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: train.py not found at SCRIPT_PATH=${SCRIPT_PATH}"
  exit 1
fi

validate_dataset_list "${DATASET_TYPE}"
validate_dataset_list "${VAL_DATASET_TYPE}"

CURRENT_DATETIME="$(date +"%d%m_%H%M")"
TODAY="$(date +"%Y-%m-%d")"
RUN_NAME="${RUN_NAME:-${CURRENT_DATETIME}_symbol_lora_${MODEL_TYPE}_${DATASET_TYPE}_${SYMBOL_UPDATE_STRATEGY}}"
LOG_DIR="${OUTPUT_DIR}/logs/${TODAY}"
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
  --dataset_type "${DATASET_TYPE}" \
  --val_dataset_type "${VAL_DATASET_TYPE}" \
  --device "${DEVICE}" \
  --lora_lr "${LORA_LR}" \
  --lora_epochs "${LORA_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --max_samples "${MAX_SAMPLES}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --symbol_update_strategy "${SYMBOL_UPDATE_STRATEGY}")

if [[ "${NO_SYMBOLS}" == "True" || "${NO_SYMBOLS}" == "true" ]]; then
  CMD+=(--no_symbols)
elif [[ "${DYNAMIC_SYMBOLS}" == "True" || "${DYNAMIC_SYMBOLS}" == "true" ]]; then
  CMD+=(--dynamic_symbols)
fi

echo "Starting local symbol training run: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
