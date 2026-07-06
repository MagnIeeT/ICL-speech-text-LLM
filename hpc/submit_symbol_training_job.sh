#!/bin/bash
set -e

# ============================================================
# Symbol Adapter Training - HPC Submit Script
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ------------------------------------------------------------
# 1. Load environment from .env for base paths
# ------------------------------------------------------------
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Set fallbacks based on .env structure
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/training}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${BASE_OUTPUT_DIR}/checkpoints}"
METRICS_DIR="${METRICS_DIR:-${BASE_OUTPUT_DIR}/metrics}"
LOGS_DIR="${LOGS_DIR:-${BASE_OUTPUT_DIR}/logs}"

# ------------------------------------------------------------
# 2. Job Configuration (Edit these values)
# ------------------------------------------------------------
model_type="salmonn"               # salmonn | qwen | flamingo
dataset_type="hvb"
val_dataset_type="voxceleb-hvb-voxpopuli-meld_emotion"

# Booleans
no_symbols=true               
dynamic_symbols=false          
diff_symbol_enabled=false      
swap_labels=false              
use_dpo=false

symbol_update_strategy="per_epoch"
validation_modes="fixed,original,fresh"

# Core Training Params
lora_lr=1e-5
lora_epochs=2
batch_size=1
val_batch_size=1
gradient_accumulation_steps=8
max_samples=10           
device="cuda:0"
num_workers=2
warmup_steps=100

# Input & Fewshot Params
input_mode="speech_only"
fewshot_mode="text"
num_examples=0
val_num_examples=0

# Symbol Params
symbol_difficulty="random"
val_symbol_difficulty="random"
num_symbol_mappings=20

# DSPO Params
dspo_router_lr=1e-2
dspo_tau_anneal_rate=0.0001
dspo_slot_only=false
dspo_phase0_epochs=0
dspo_phase1_patience=0
dspo_phase1_epochs=0
dspo_num_slots=25
dspo_slot_vocab_size=100
dspo_rotation_interval=-1
dspo_phase2_rotation=-1

dpo_beta=0.1

# ------------------------------------------------------------
# 3. HPC / Queue Settings
# ------------------------------------------------------------
queue_name="workq"
hostname="n6"
cuda_device=2
walltime="72:00:00"

# ------------------------------------------------------------
# 4. Automatic Setup & Directory Creation
# ------------------------------------------------------------
output_dir="${BASE_OUTPUT_DIR}/symbol_training"
log_date_dir="${LOGS_DIR}/$(date +"%Y-%m-%d")"

mkdir -p "${output_dir}" "${CHECKPOINT_DIR}" "${METRICS_DIR}" "${log_date_dir}"

CURRENT_DATETIME="$(date +"%d%m_%H%M")"
MODE="fixed"
[ "${no_symbols}" = "true" ] && MODE="baseline"
[ "${dynamic_symbols}" = "true" ] && MODE="${symbol_update_strategy}"
[ "${diff_symbol_enabled}" = "true" ] && MODE="dspo"
[ "${use_dpo}" = "true" ] && MODE="symdpo"
[ "${swap_labels}" = "true" ] && MODE="${MODE}_swap"

RUN_NAME="${CURRENT_DATETIME}_${model_type}_${dataset_type}_${MODE}"
LOG_FILE="${log_date_dir}/${RUN_NAME}.log"
SCRIPT_PATH="${PROJECT_ROOT}/train.py"

# Conda Env Selection
if [[ "${model_type}" == "flamingo" ]]; then
    CONDA_ENV="flamingo"
elif [[ "${model_type}" == "qwen" ]]; then
    CONDA_ENV="qwen"
else
    CONDA_ENV="salmonn2"
fi

# Construct boolean arguments cleanly
EXTRA_ARGS=""
[ "${no_symbols}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --no_symbols"
[ "${dynamic_symbols}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --dynamic_symbols"
[ "${diff_symbol_enabled}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --diff_symbol_enabled"
[ "${swap_labels}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --swap_labels"
[ "${use_dpo}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --use_dpo --dpo_beta ${dpo_beta}"
[ "${dspo_slot_only}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --dspo_slot_only"

echo "============================================================"
echo "Submitting Symbol Training Job: ${RUN_NAME}"
echo "Model:       ${model_type}"
echo "Dataset:     ${dataset_type}"
echo "Conda:       ${CONDA_ENV}"
echo "Queue:       ${queue_name} (Host: ${hostname})"
echo "============================================================"

# Export variables so qsub -V can pick them up cleanly
export PROJECT_ROOT CONDA_ENV LOG_FILE RUN_NAME output_dir CHECKPOINT_DIR METRICS_DIR log_date_dir SCRIPT_PATH
export model_type dataset_type val_dataset_type lora_lr lora_epochs batch_size val_batch_size
export gradient_accumulation_steps max_samples num_workers warmup_steps device
export input_mode fewshot_mode num_examples val_num_examples symbol_difficulty val_symbol_difficulty num_symbol_mappings
export validation_modes symbol_update_strategy EXTRA_ARGS
export dspo_router_lr dspo_tau_anneal_rate dspo_phase0_epochs dspo_phase1_patience dspo_phase1_epochs
export dspo_num_slots dspo_slot_vocab_size dspo_rotation_interval dspo_phase2_rotation cuda_device

# Submit with -V to pass all exported environment variables to the compute node
qsub -V -q "$queue_name" \
    -N "$RUN_NAME" \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=$hostname \
    -l walltime=$walltime \
    -o "${log_date_dir}/${RUN_NAME}.pbs.log" \
    -j oe \
    -S /bin/bash << 'EOF'
#!/bin/bash
set -e
set -o pipefail 

cd ${PROJECT_ROOT}

# Ensure .env is loaded on the compute node so paths are available to Python
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Initialize conda
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${cuda_device}
export LD_PRELOAD=""
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

echo "=== Running on: $(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi
export PYTHONPATH="/home/anmola:${PYTHONPATH:-}"
python ${SCRIPT_PATH} \
  --model_type "${model_type}" \
  --dataset_type "${dataset_type}" \
  --val_dataset_type "${val_dataset_type}" \
  --device "${device}" \
  --batch_size ${batch_size} \
  --val_batch_size ${val_batch_size} \
  --max_samples ${max_samples} \
  --num_examples ${num_examples} \
  --val_num_examples ${val_num_examples} \
  --num_workers ${num_workers} \
  --lora_lr ${lora_lr} \
  --lora_epochs ${lora_epochs} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --warmup_steps ${warmup_steps} \
  --output_dir "${output_dir}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --metrics_dir "${METRICS_DIR}" \
  --logs_dir "${log_date_dir}" \
  --run_name "${RUN_NAME}" \
  --validation_modes "${validation_modes}" \
  --symbol_update_strategy "${symbol_update_strategy}" \
  --symbol_difficulty "${symbol_difficulty}" \
  --val_symbol_difficulty "${val_symbol_difficulty}" \
  --num_symbol_mappings ${num_symbol_mappings} \
  --input_mode "${input_mode}" \
  --fewshot_mode "${fewshot_mode}" \
  --dspo_router_lr ${dspo_router_lr} \
  --dspo_tau_anneal_rate ${dspo_tau_anneal_rate} \
  --dspo_phase0_epochs ${dspo_phase0_epochs} \
  --dspo_phase1_patience ${dspo_phase1_patience} \
  --dspo_phase1_epochs ${dspo_phase1_epochs} \
  --dspo_num_slots ${dspo_num_slots} \
  --dspo_slot_vocab_size ${dspo_slot_vocab_size} \
  --dspo_rotation_interval ${dspo_rotation_interval} \
  --dspo_phase2_rotation ${dspo_phase2_rotation} \
  ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}
EOF

echo ""
echo "Job Submitted Successfully"
echo "Job Name: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_FILE}"