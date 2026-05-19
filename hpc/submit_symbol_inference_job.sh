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
    CHECKPOINT_DIR=$(grep "^CHECKPOINT_DIR=" "${PROJECT_ROOT}/.env" | cut -d'=' -f2-)
    BASE_OUTPUT_DIR=$(grep "^BASE_OUTPUT_DIR=" "${PROJECT_ROOT}/.env" | cut -d'=' -f2-)
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/results/checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/results/symbol_training}"

# ------------------------------------------------------------
# 2. Job Configuration (Edit these values)
# ------------------------------------------------------------
model_type="salmonn"
checkpoint_path="${CHECKPOINT_DIR}/0301_1316_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
dataset_type="voxceleb"

no_symbols=true
swap_labels=false
num_examples=3
max_val_samples=10
validation_modes="fixed,original,fresh"
device="cuda:0"

# ------------------------------------------------------------
# 3. HPC / Queue Settings
# ------------------------------------------------------------
queue_name="workq"
hostname="n6"        # Changed to n6 based on your check
cuda_device=0       # Changed to 4 based on exec_host=n6/4
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

if [[ "${model_type}" == "qwen" ]]; then
    CONDA_ENV="qwen2_new"
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

# ✅ FIX: Change -o from /dev/null to a real file so we can see errors
qsub -q $queue_name \
    -N $RUN_NAME \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=$hostname \
    -l walltime=$walltime \
    -o /dev/null \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},\
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log",\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
PROJECT_ROOT=${PROJECT_ROOT},\
checkpoint_path=${checkpoint_path},\
dataset_type=${dataset_type},\
model_type=${model_type},\
max_val_samples=${max_val_samples},\
num_examples=${num_examples},\
device=${device},\
output_dir=${output_dir} \
    -S /bin/bash << EOF
#!/bin/bash
set -e

cd \${PROJECT_ROOT}

# Echo system info for debugging
echo "Job started on: \$(hostname)"
echo "Assigned GPU: \${CUDA_VISIBLE_DEVICES}"

source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python \${SCRIPT_PATH} \
  --model_type "\${model_type}" \
  --checkpoint_path "\${checkpoint_path}" \
  --dataset_type "\${dataset_type}" \
  --device "\${device}" \
  --max_val_samples \${max_val_samples} \
  --num_examples \${num_examples} \
  --output_dir "\${output_dir}" \
  --run_name "\${RUN_NAME}" \
  --validation_modes "${validation_modes}" \
  $( [ "${no_symbols}" = "true" ] && echo "--no_symbols" ) \
  $( [ "${swap_labels}" = "true" ] && echo "--swap_labels" ) 2>&1 | tee \${LOG_FILE}
EOF
