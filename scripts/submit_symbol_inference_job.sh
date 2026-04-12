#!/bin/bash
set -euo pipefail

show_help() {
  cat << 'USAGE'
Usage:
  Environment-variable driven inference submit script.

  Example:
    MODEL_TYPE=salmonn \
    CHECKPOINT_PATH=/path/to/checkpoint.pt \
    DATASET_TYPE=hvb-voxceleb-voxpopuli-meld_emotion \
    MAX_VAL_SAMPLES=0 \
    NUM_EXAMPLES=0 \
    NO_SYMBOLS=true \
    SWAP_LABELS=false \
    VALIDATION_MODES=fixed,original \
    OUTPUT_DIR=/path/to/results \
    ./scripts/submit_symbol_inference_job.sh

Available options (env vars):
  MODEL_TYPE               : salmonn | qwen
  CHECKPOINT_PATH          : path to checkpoint file (required)
  DATASET_TYPE             : hyphen-joined list from {voxceleb,hvb,voxpopuli,meld_emotion}
                             examples: voxceleb | hvb-meld_emotion | hvb-voxceleb-voxpopuli-meld_emotion

  MAX_VAL_SAMPLES          : int (default 0 means all)
  NUM_EXAMPLES             : int (few-shot example count for inference pipeline)
  NO_SYMBOLS               : true | false
  SWAP_LABELS              : true | false
  VALIDATION_MODES         : comma-separated: fixed,original,fresh (aliases: both,all,new)
  DEVICE                   : cuda:0, cuda:1, cpu, ...

  OUTPUT_DIR               : output base directory
  RUN_NAME                 : run identifier

HPC / queue settings:
  QUEUE_NAME, HOSTNAME_FILTER, CUDA_DEVICE, WALLTIME, HOLD_JOB_ID

Notes:
  1) This script submits qsub and runs PROJECT_ROOT/inference.py.
  2) Set NO_SYMBOLS=true to bypass symbol replacement entirely.
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
SWAP_LABELS="${SWAP_LABELS:-false}"
VALIDATION_MODES="${VALIDATION_MODES:-fixed,original,fresh}"
DEVICE="${DEVICE:-cuda:0}"

QUEUE_NAME="${QUEUE_NAME:-workq}"
HOSTNAME_FILTER="${HOSTNAME_FILTER:-n10}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WALLTIME="${WALLTIME:-72:00:00}"
HOLD_JOB_ID="${HOLD_JOB_ID:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/inference.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results}"

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

for b in "${NO_SYMBOLS}" "${SWAP_LABELS}"; do
  if [[ "${b}" != "true" && "${b}" != "false" && "${b}" != "True" && "${b}" != "False" ]]; then
    echo "ERROR: Boolean flags (NO_SYMBOLS, SWAP_LABELS) must be true or false"
    exit 1
  fi
done

validate_dataset_list "${DATASET_TYPE}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: CHECKPOINT_PATH is required"
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

TODAY="$(date +"%Y-%m-%d")"
CURRENT_DATETIME="$(date +"%d%m_%H%M")"
RUN_NAME="${RUN_NAME:-${CURRENT_DATETIME}_infer_${MODEL_TYPE}}"
LOG_DIR="${OUTPUT_DIR}/orchestrator_logs/${TODAY}"
mkdir -p "${LOG_DIR}"

if [[ -n "${HOLD_JOB_ID}" ]]; then
  HOLD_FLAG="-W depend=afterok:${HOLD_JOB_ID}"
else
  HOLD_FLAG=""
fi

qsub -q "${QUEUE_NAME}" \
  ${HOLD_FLAG} \
  -l "select=1:num_gpus=1:gpu_mem=48GB:host=${HOSTNAME_FILTER}" \
  -l "walltime=${WALLTIME}" \
  -o /dev/null \
  -j oe \
  -v CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}",\
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log",\
PYTHONUNBUFFERED=1,\
RUN_NAME="${RUN_NAME}",\
SCRIPT_PATH="${SCRIPT_PATH}",\
checkpoint_path="${CHECKPOINT_PATH}",\
dataset_type="${DATASET_TYPE}",\
model_type="${MODEL_TYPE}",\
max_val_samples="${MAX_VAL_SAMPLES}",\
num_examples="${NUM_EXAMPLES}",\
no_symbols="${NO_SYMBOLS}",\
swap_labels="${SWAP_LABELS}",\
validation_modes="${VALIDATION_MODES}",\
device="${DEVICE}",\
output_dir="${OUTPUT_DIR}" \
  -S /bin/bash << 'QSUB_EOF'
#!/bin/bash
set -euo pipefail

CMD=(python "${SCRIPT_PATH}" \
  --model_type "${model_type}" \
  --checkpoint_path "${checkpoint_path}" \
  --dataset_type "${dataset_type}" \
  --device "${device}" \
  --max_val_samples "${max_val_samples}" \
  --num_examples "${num_examples}" \
  --output_dir "${output_dir}" \
  --run_name "${RUN_NAME}" \
  --validation_modes "${validation_modes}")

if [[ "${no_symbols}" == "True" || "${no_symbols}" == "true" ]]; then
  CMD+=(--no_symbols)
fi

if [[ "${swap_labels}" == "True" || "${swap_labels}" == "true" ]]; then
  CMD+=(--swap_labels)
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
QSUB_EOF

echo "Submitted symbol inference job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"
