#!/bin/bash
set -e

# ============================================================
# Symbol Adapter Training - HPC Submit Script
# ============================================================

# ------------------------------------------------------------
# 1. Environment Setup
# ------------------------------------------------------------
# Automatically determine PROJECT_ROOT based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# FIX 1: Add '|| true' so grep doesn't kill the script under 'set -e' if the key isn't found
if [ -f "${PROJECT_ROOT}/.env" ]; then
    BASE_OUTPUT_DIR=$(grep "^BASE_OUTPUT_DIR=" "${PROJECT_ROOT}/.env" | cut -d'=' -f2- || true)
fi

# Set defaults
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/results/symbol_training}"

# ------------------------------------------------------------
# 2. Job Configuration (Edit these values)
# ------------------------------------------------------------
model_type="flamingo"           # salmonn | qwen
dataset_type="voxceleb-hvb"        # voxceleb | hvb | voxpopuli | meld_emotion
val_dataset_type="voxceleb-hvb"    # datasets to validate on (hyphen-separated)

no_symbols=false               # true = baseline, false = symbol-based
dynamic_symbols=false          # true = refresh symbols per epoch
swap_labels=false              # true = flip labels (e.g. pos<->neg)
symbol_update_strategy="per_epoch"    # per_epoch | per_instance
validation_modes="fixed,original,fresh" # modes to test each epoch

lora_lr=1e-5
lora_epochs=2
batch_size=1
gradient_accumulation_steps=8
max_samples=10           # 0 for all
device="cuda:0"

# ------------------------------------------------------------
# 3. HPC / Queue Settings
# ------------------------------------------------------------
queue_name="workq"
hostname="n6"
cuda_device=2
walltime="72:00:00"

# ------------------------------------------------------------
# 4. Automatic Setup
# ------------------------------------------------------------
output_dir="${BASE_OUTPUT_DIR}"
SCRIPT_PATH="${PROJECT_ROOT}/train.py"
CURRENT_DATETIME="$(date +"%d%m_%H%M")"
RUN_NAME="${CURRENT_DATETIME}_${model_type}_${dataset_type}_${symbol_update_strategy}"

# FIX 2: Standardized boolean string checks
[ "${no_symbols}" = "true" ] || [ "${no_symbols}" = true ] && RUN_NAME="${RUN_NAME}_baseline"
[ "${swap_labels}" = "true" ] || [ "${swap_labels}" = true ] && RUN_NAME="${RUN_NAME}_swap"

LOG_DIR="${output_dir}/logs/$(date +"%Y-%m-%d")"
mkdir -p "${LOG_DIR}"

if [[ "${model_type}" == "qwen" ]]; then
    CONDA_ENV="qwen"
elif [[ "${model_type}" == "flamingo" ]]; then
    CONDA_ENV="flamingo"
else
    CONDA_ENV="salmonn2"
fi

# FIX 3: Construct boolean arguments cleanly outside the Python call to prevent backslash parsing errors
EXTRA_ARGS=""
[ "${no_symbols}" = "true" ] || [ "${no_symbols}" = true ] && EXTRA_ARGS="${EXTRA_ARGS} --no_symbols"
[ "${dynamic_symbols}" = "true" ] || [ "${dynamic_symbols}" = true ] && EXTRA_ARGS="${EXTRA_ARGS} --dynamic_symbols"
[ "${swap_labels}" = "true" ] || [ "${swap_labels}" = true ] && EXTRA_ARGS="${EXTRA_ARGS} --swap_labels"

echo "============================================================"
echo "Submitting Symbol Training Job: ${RUN_NAME}"
echo "Model:       ${model_type}"
echo "Dataset:     ${dataset_type}"
echo "Conda:       ${CONDA_ENV}"
echo "Queue:       ${queue_name} (Host: ${hostname})"
echo "Project Root: ${PROJECT_ROOT}"
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
model_type=${model_type},\
dataset_type=${dataset_type},\
val_dataset_type=${val_dataset_type},\
lora_lr=${lora_lr},\
lora_epochs=${lora_epochs},\
batch_size=${batch_size},\
gradient_accumulation_steps=${gradient_accumulation_steps},\
max_samples=${max_samples},\
output_dir=${output_dir},\
device=${device} \
    -S /bin/bash << EOF
#!/bin/bash
set -e
set -o pipefail # FIX 4: Ensures the job fails properly if python crashes, instead of tee succeeding.

cd \${PROJECT_ROOT}

source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${cuda_device}
export LD_PRELOAD=""

# FIX 5: Prevents adding a trailing colon if LD_LIBRARY_PATH was empty
export LD_LIBRARY_PATH="/home/anmola/.conda/envs/flamingo/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# FIX 6: Escaped \$(hostname) so it prints the compute node, not the login node
echo "=== Running on: \$(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi

python \${SCRIPT_PATH} \
  --model_type "\${model_type}" \
  --dataset_type "\${dataset_type}" \
  --val_dataset_type "\${val_dataset_type}" \
  --lora_lr \${lora_lr} \
  --lora_epochs \${lora_epochs} \
  --batch_size \${batch_size} \
  --gradient_accumulation_steps \${gradient_accumulation_steps} \
  --max_samples \${max_samples} \
  --output_dir "\${output_dir}" \
  --run_name "\${RUN_NAME}" \
  --validation_modes "${validation_modes}" \
  --symbol_update_strategy "${symbol_update_strategy}" \
  --device "\${device}" \
  ${EXTRA_ARGS} 2>&1 | tee \${LOG_FILE}

EOF

echo ""
echo "Job Submitted Successfully"
echo "Job Name: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"