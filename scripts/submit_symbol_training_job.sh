#!/bin/bash
set -euo pipefail

show_help() {
  cat << 'USAGE'
Usage:
  Environment-variable driven submit script.

  Example:
    MODEL_TYPE=salmonn \
    DATASET_TYPE=hvb-meld_emotion \
    VAL_DATASET_TYPE=hvb-meld_emotion \
    NO_SYMBOLS=true \
    SWAP_LABELS=false \
    VALIDATION_MODES=fixed,original \
    OUTPUT_DIR=/path/to/results/symbol_training \
    ./scripts/submit_symbol_training_job.sh

Available options (env vars):
  MODEL_TYPE               : salmonn | qwen
  DATASET_TYPE             : hyphen-joined list from {voxceleb,hvb,voxpopuli,meld_emotion}
                             examples: voxceleb | hvb-meld_emotion | hvb-voxceleb-voxpopuli-meld_emotion
  VAL_DATASET_TYPE         : same format as DATASET_TYPE

  NO_SYMBOLS               : true | false
                             - true  => disable symbol replacement
                             - false => use symbol-based flow
  SWAP_LABELS              : true | false
                             - true  => swap labels (e.g., positive<->negative)

  DYNAMIC_SYMBOLS          : true | false
                             - true  => pass --dynamic_symbols
                             - false => fixed mappings
  SYMBOL_UPDATE_STRATEGY   : per_epoch | per_instance
  VALIDATION_MODES         : comma-separated: fixed,original,fresh (aliases: both,all,new)

  LORA_LR                  : float (default 1e-5)
  LORA_EPOCHS              : int   (default 5)
  BATCH_SIZE               : int   (default 1)
  GRADIENT_ACCUMULATION_STEPS: int (default 8)
  MAX_GRAD_NORM            : float (default 1)
  MAX_SAMPLES              : int   (default 0 means all)

  DEVICE                   : cuda:0, cuda:1, cpu, ...
  OUTPUT_DIR               : output base directory
  RUN_NAME                 : run identifier

HPC / queue settings:
  QUEUE_NAME, HOSTNAME_FILTER, CUDA_DEVICE, WALLTIME, HOLD_JOB_ID, CONDA_ENV

Notes:
  1) This script submits qsub and runs PROJECT_ROOT/train.py.
  2) If NO_SYMBOLS=true, --no_symbols is passed and --dynamic_symbols is ignored.
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
SWAP_LABELS="${SWAP_LABELS:-false}"
DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-false}"
SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_epoch}"
VALIDATION_MODES="${VALIDATION_MODES:-fixed,original,fresh}"

QUEUE_NAME="${QUEUE_NAME:-workq}"
HOSTNAME_FILTER="${HOSTNAME_FILTER:-n10}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WALLTIME="${WALLTIME:-72:00:00}"
HOLD_JOB_ID="${HOLD_JOB_ID:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/train.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/symbol_training}"

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

for b in "${NO_SYMBOLS}" "${SWAP_LABELS}" "${DYNAMIC_SYMBOLS}"; do
  if [[ "${b}" != "true" && "${b}" != "false" && "${b}" != "True" && "${b}" != "False" ]]; then
    echo "ERROR: Boolean flags (NO_SYMBOLS, SWAP_LABELS, DYNAMIC_SYMBOLS) must be true or false"
    exit 1
  fi
done

validate_dataset_list "${DATASET_TYPE}"
validate_dataset_list "${VAL_DATASET_TYPE}"

CURRENT_DATETIME="$(date +"%d%m_%H%M")"
TODAY="$(date +"%Y-%m-%d")"
RUN_NAME="${RUN_NAME:-${CURRENT_DATETIME}_symbol_lora_${MODEL_TYPE}_${DATASET_TYPE}_${SYMBOL_UPDATE_STRATEGY}}"
LOG_DIR="${OUTPUT_DIR}/logs/${TODAY}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
rm -f "${LOG_DIR}/${RUN_NAME}.log"

if [[ -n "${HOLD_JOB_ID}" ]]; then
  HOLD_FLAG="-W depend=afterok:${HOLD_JOB_ID}"
else
  HOLD_FLAG=""
fi

if [[ "${MODEL_TYPE}" == "qwen" ]]; then
  CONDA_ENV="${CONDA_ENV:-qwen2_new}"
else
  CONDA_ENV="${CONDA_ENV:-salmonn}"
fi

source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda deactivate || true
conda activate "${CONDA_ENV}"

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
model_type="${MODEL_TYPE}",\
dataset_type="${DATASET_TYPE}",\
val_dataset_type="${VAL_DATASET_TYPE}",\
device="${DEVICE}",\
lora_lr="${LORA_LR}",\
lora_epochs="${LORA_EPOCHS}",\
batch_size="${BATCH_SIZE}",\
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}",\
max_grad_norm="${MAX_GRAD_NORM}",\
max_samples="${MAX_SAMPLES}",\
dynamic_symbols="${DYNAMIC_SYMBOLS}",\
no_symbols="${NO_SYMBOLS}",\
swap_labels="${SWAP_LABELS}",\
symbol_update_strategy="${SYMBOL_UPDATE_STRATEGY}",\
validation_modes="${VALIDATION_MODES}",\
OUTPUT_DIR="${OUTPUT_DIR}" \
  -S /bin/bash << 'QSUB_EOF'
#!/bin/bash
set -euo pipefail

export HF_HOME=/home/leapers/common_cache/huggingface
export TRANSFORMERS_CACHE=/home/leapers/common_cache/huggingface
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

CMD=(python "${SCRIPT_PATH}" \
  --model_type "${model_type}" \
  --dataset_type "${dataset_type}" \
  --val_dataset_type "${val_dataset_type}" \
  --device "${device}" \
  --lora_lr "${lora_lr}" \
  --lora_epochs "${lora_epochs}" \
  --batch_size "${batch_size}" \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --max_grad_norm "${max_grad_norm}" \
  --max_samples "${max_samples}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --symbol_update_strategy "${symbol_update_strategy}" \
  --validation_modes "${validation_modes}")

if [[ "${no_symbols}" == "True" || "${no_symbols}" == "true" ]]; then
  CMD+=(--no_symbols)
elif [[ "${dynamic_symbols}" == "True" || "${dynamic_symbols}" == "true" ]]; then
  CMD+=(--dynamic_symbols)
fi

if [[ "${swap_labels}" == "True" || "${swap_labels}" == "true" ]]; then
  CMD+=(--swap_labels)
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
QSUB_EOF

echo "Submitted symbol training job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"



## remove hardcoded value from config hvb path and from slamon beats weight etc. need to add that in readme.