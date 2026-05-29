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

# Load .env file if it exists into local variables (do NOT export)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    BASE_OUTPUT_DIR=$(grep "^BASE_OUTPUT_DIR=" "${PROJECT_ROOT}/.env" | cut -d'=' -f2-)
fi

# Set defaults
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/results/symbol_training}"

# ------------------------------------------------------------
# 2. Job Configuration (Edit these values)
# ------------------------------------------------------------
model_type="salmonn"           # salmonn | qwen
dataset_type="voxceleb"        # voxceleb | hvb | voxpopuli | meld_emotion
val_dataset_type="voxceleb"    # datasets to validate on (hyphen-separated)

no_symbols=false               # true = baseline, false = symbol-based
dynamic_symbols=false          # true = refresh symbols per epoch
diff_symbol_enabled=false      # true = Differentiable Symbolic Preference Optimization (D-SPO)
swap_labels=false              # true = flip labels (e.g. pos<->neg)
symbol_update_strategy="per_epoch"    # per_epoch | per_instance
validation_modes="fixed,original,fresh" # modes to test each epoch

lora_lr=1e-5
lora_epochs=2
batch_size=1
gradient_accumulation_steps=8
max_samples=10                # 0 for all
device="cuda:0"

# ------------------------------------------------------------
# 3. HPC / Queue Settings
# ------------------------------------------------------------
queue_name="workq"
hostname="n11"
cuda_device=0
walltime="72:00:00"

# ------------------------------------------------------------
# 4. Automatic Setup
# ------------------------------------------------------------
output_dir="${BASE_OUTPUT_DIR}"
SCRIPT_PATH="${PROJECT_ROOT}/train.py"
CURRENT_DATETIME="$(date +"%d%m_%H%M")"
RUN_NAME="${CURRENT_DATETIME}_${model_type}_${dataset_type}_${symbol_update_strategy}"
[ "${no_symbols}" = true ] && RUN_NAME="${RUN_NAME}_baseline"
[ "${diff_symbol_enabled}" = true ] && RUN_NAME="${RUN_NAME}_dspo"
[ "${swap_labels}" = true ] && RUN_NAME="${RUN_NAME}_swap"

LOG_DIR="${output_dir}/logs/$(date +"%Y-%m-%d")"
mkdir -p "${LOG_DIR}"

if [[ "${model_type}" == "qwen" ]]; then
    CONDA_ENV="qwen2_new"
else
    CONDA_ENV="salmonn2"
fi

echo "============================================================"
echo "Submitting Symbol Training Job: ${RUN_NAME}"
echo "Model:       ${model_type}"
echo "Dataset:     ${dataset_type}"
echo "Conda:       ${CONDA_ENV}"
echo "Queue:       ${queue_name} (Host: ${hostname})"
echo "Project Root: ${PROJECT_ROOT}"
echo "============================================================"

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

# ✅ CRITICAL: Move to project root so relative paths in .env work
cd \${PROJECT_ROOT}

source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

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
  $( [ "${no_symbols}" = "true" ] && echo "--no_symbols" ) \
  $( [ "${dynamic_symbols}" = "true" ] && echo "--dynamic_symbols" ) \
  $( [ "${diff_symbol_enabled}" = "true" ] && echo "--diff_symbol_enabled" ) \
  $( [ "${swap_labels}" = "true" ] && echo "--swap_labels" ) 2>&1 | tee \${LOG_FILE}
EOF

echo ""
echo "Job Submitted Successfully"
echo "Job Name: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"
