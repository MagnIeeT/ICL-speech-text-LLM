#!/bin/bash
set -e

# ============================================================
# Symbol Adapter Inference - HPC Submit Script
# ============================================================

# ------------------------------------------------------------
# 1. Environment Setup
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    CHECKPOINT_DIR=$(grep "^CHECKPOINT_DIR=" "${PROJECT_ROOT}/.env" | cut -d'=' -f2- || true)
    BASE_OUTPUT_DIR=$(grep "^BASE_OUTPUT_DIR=" "${PROJECT_ROOT}/.env" | cut -d'=' -f2- || true)
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/results/checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/results/symbol_training}"

# ------------------------------------------------------------
# 2. Job Configuration (Edit these values)
# ------------------------------------------------------------
model_type="qwen"
dataset_type="hvb-voxceleb-voxpopuli-meld_emotion"
CHECKPOINT_PATH="/home/leapers/weights/anmola/ICL-speech-text-LLM/training/checkpoints/0206_2201_qwen_meld_emotion-voxpopuli_per_epoch/lora_epoch2_periodic.pt"
no_symbols=false
swap_labels=false
num_examples=10
max_val_samples=0
validation_modes="original"
device="cuda:0"

# ------------------------------------------------------------
# 3. HPC / Queue Settings
# ------------------------------------------------------------
queue_name="workq"
hostname="n9"
cuda_device=0
walltime="72:00:00"

# ------------------------------------------------------------
# 4. Automatic Setup
# ------------------------------------------------------------
output_dir="${BASE_OUTPUT_DIR}"
SCRIPT_PATH="${PROJECT_ROOT}/inference.py"
CURRENT_DATETIME="$(date +"%d%m_%H%M")"
RUN_NAME="${CURRENT_DATETIME}_infer_${model_type}_${dataset_type}"
LOG_DIR="${output_dir}/logs/$(date +"%Y-%m-%d")"
mkdir -p "${LOG_DIR}"

# Automatically select conda env based on model type
if [[ "${model_type}" == "qwen" ]]; then
    CONDA_ENV="qwen"
elif [[ "${model_type}" == "flamingo" ]]; then
    CONDA_ENV="flamingo"
else
    CONDA_ENV="salmonn"
fi

echo "============================================================"
echo "Submitting Symbol Inference Job: ${RUN_NAME}"
echo "Model:       ${model_type}"
echo "Host:        ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Log File:    ${LOG_DIR}/${RUN_NAME}.log"
echo "============================================================"

qsub -q "$queue_name" \
    -N "$RUN_NAME" \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=$hostname \
    -l walltime=$walltime \
    -o "${LOG_DIR}/${RUN_NAME}.pbs.log" \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},\
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log",\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
PROJECT_ROOT=${PROJECT_ROOT},\
dataset_type=${dataset_type},\
model_type=${model_type},\
max_val_samples=${max_val_samples},\
num_examples=${num_examples},\
device=${device},\
output_dir=${output_dir} \
    -S /bin/bash << EOF
#!/bin/bash
set -e
set -o pipefail

cd \${PROJECT_ROOT}

echo "=== Running on: \$(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES} ==="

source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${cuda_device}
export LD_PRELOAD=""
export LD_LIBRARY_PATH="/home/anmola/.conda/envs/${CONDA_ENV}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

nvidia-smi

python \${SCRIPT_PATH} \
  --model_type "\${model_type}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --dataset_type "\${dataset_type}" \
  --device "\${device}" \
  --max_val_samples \${max_val_samples} \
  --num_examples \${num_examples} \
  --output_dir "\${output_dir}" \
  --run_name "\${RUN_NAME}" \
  --validation_modes "${validation_modes}" \
  $( [ "${no_symbols}" = "true" ] && echo "--no_symbols" || true ) \
  $( [ "${swap_labels}" = "true" ] && echo "--swap_labels" || true ) 2>&1 | tee \${LOG_FILE}

EOF

echo ""
echo "Job Submitted Successfully"
echo "Job Name: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"