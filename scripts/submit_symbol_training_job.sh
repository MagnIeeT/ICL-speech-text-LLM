#!/bin/bash
set -euo pipefail

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

DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-true}"
SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_epoch}"

QUEUE_NAME="${QUEUE_NAME:-workq}"
HOSTNAME_FILTER="${HOSTNAME_FILTER:-n10}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WALLTIME="${WALLTIME:-72:00:00}"
HOLD_JOB_ID="${HOLD_JOB_ID:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/train.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/symbol_training}"

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
symbol_update_strategy="${SYMBOL_UPDATE_STRATEGY}",\
OUTPUT_DIR="${OUTPUT_DIR}" \
  -S /bin/bash << 'QSUB_EOF'
#!/bin/bash
set -euo pipefail

export HF_HOME=/home/leapers/common_cache/huggingface
export TRANSFORMERS_CACHE=/home/leapers/common_cache/huggingface
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

COMMON_ARGS="--model_type \"${model_type}\" \
  --dataset_type \"${dataset_type}\" \
  --val_dataset_type \"${val_dataset_type}\" \
  --device \"${device}\" \
  --lora_lr ${lora_lr} \
  --lora_epochs ${lora_epochs} \
  --batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --max_grad_norm ${max_grad_norm} \
  --max_samples ${max_samples} \
  --output_dir \"${OUTPUT_DIR}\" \
  --run_name \"${RUN_NAME}\" \
  --symbol_update_strategy \"${symbol_update_strategy}\""

if [[ "${dynamic_symbols}" == "True" || "${dynamic_symbols}" == "true" ]]; then
  COMMON_ARGS="${COMMON_ARGS} --dynamic_symbols"
fi

eval "python ${SCRIPT_PATH} ${COMMON_ARGS} 2>&1 | tee ${LOG_FILE}"
QSUB_EOF

echo "Submitted symbol training job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"
