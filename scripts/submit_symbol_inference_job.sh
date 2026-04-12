#!/bin/bash
set -euo pipefail

MODEL_TYPE="${MODEL_TYPE:-salmonn}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DATASET_TYPE="${DATASET_TYPE:-hvb-voxceleb-voxpopuli-meld_emotion}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"
DEVICE="${DEVICE:-cuda:0}"

QUEUE_NAME="${QUEUE_NAME:-workq}"
HOSTNAME_FILTER="${HOSTNAME_FILTER:-n10}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WALLTIME="${WALLTIME:-72:00:00}"
HOLD_JOB_ID="${HOLD_JOB_ID:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/inference.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results}"

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
device="${DEVICE}",\
output_dir="${OUTPUT_DIR}" \
  -S /bin/bash << 'QSUB_EOF'
#!/bin/bash
set -e
python "${SCRIPT_PATH}" \
  --model_type "${model_type}" \
  --checkpoint_path "${checkpoint_path}" \
  --dataset_type "${dataset_type}" \
  --device "${device}" \
  --max_val_samples "${max_val_samples}" \
  --num_examples "${num_examples}" \
  --output_dir "${output_dir}" \
  --run_name "${RUN_NAME}" 2>&1 | tee "${LOG_FILE}"
QSUB_EOF

echo "Submitted symbol inference job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"
