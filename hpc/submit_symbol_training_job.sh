#!/bin/bash
set -e

# ============================================================
# Symbol Adapter Training - HPC Submit Script
# ============================================================

# ------------------------------------------------------------
# 1. Environment & Path Setup
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/results/symbol_training}"

# ------------------------------------------------------------
# 2. Job Configuration 
# ------------------------------------------------------------
MODEL_TYPE="salmonn"                                    # qwen | salmonn | flamingo
DATASET_TYPE="hvb"                          # Training dataset(s), dash-separated
VAL_DATASET_TYPE="voxceleb-hvb-voxpopuli-meld_emotion" 

# --- Symbol & DPO Modes ---
USE_DPO=false
DPO_BETA=0.1
NO_SYMBOLS=true
DYNAMIC_SYMBOLS=false
DIFF_SYMBOL_ENABLED=false
SWAP_LABELS=false
SYMBOL_UPDATE_STRATEGY="per_epoch"                   # per_epoch | per_instance
VALIDATION_MODES="fixed,original,fresh"              
SYMBOL_DIFFICULTY="random"
VAL_SYMBOL_DIFFICULTY="random"
NUM_SYMBOL_MAPPINGS=20

# --- D-SPO Settings ---
DSPO_ROUTER_LR=1e-2
DSPO_TAU_ANNEAL_RATE=0.0001
DSPO_SLOT_ONLY=false
DSPO_PHASE0_EPOCHS=0
DSPO_PHASE1_PATIENCE=3
DSPO_PHASE1_EPOCHS=5
DSPO_NUM_SLOTS=20
DSPO_SLOT_VOCAB_SIZE=25
DSPO_ROTATION_INTERVAL=200
DSPO_PHASE2_ROTATION=0

# --- Training Hyperparameters ---
LORA_LR=1e-5
LORA_EPOCHS=2
BATCH_SIZE=1
VAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=100
MAX_SAMPLES=10                                       # 0 for full dataset
NUM_EXAMPLES=0                                       # Few-shot examples in train (0=zero-shot)
VAL_NUM_EXAMPLES=0                                   # Few-shot examples in val
INPUT_MODE="speech_only"
FEWSHOT_MODE="text"
VALIDATE_BEFORE_TRAINING=false
NUM_WORKERS=2

# ------------------------------------------------------------
# 3. HPC / Queue Settings
# ------------------------------------------------------------
QUEUE_NAME="workq"
HOSTNAME="n6"
CUDA_DEVICE=2
WALLTIME="72:00:00"

# ------------------------------------------------------------
# 4. Automatic Setup & Run Name Generation
# ------------------------------------------------------------
if [[ "${MODEL_TYPE}" == "qwen" ]]; then
    CONDA_ENV="qwen"
elif [[ "${MODEL_TYPE}" == "flamingo" ]]; then
    CONDA_ENV="flamingo"
else
    CONDA_ENV="salmonn2"
fi

# Run name mode logic (ported from your old script)
if [[ "${USE_DPO}" == "true" || "${USE_DPO}" == true ]]; then
    _MODE="symdpo"
elif [[ "${DIFF_SYMBOL_ENABLED}" == "true" || "${DIFF_SYMBOL_ENABLED}" == true ]]; then
    _MODE="dspo"
    [[ "${DSPO_SLOT_ONLY}" == "true" || "${DSPO_SLOT_ONLY}" == true ]] && _MODE="dspo_slotonly"
elif [[ "${NO_SYMBOLS}" == "true" || "${NO_SYMBOLS}" == true ]]; then
    _MODE="nosym"
elif [[ "${DYNAMIC_SYMBOLS}" == "true" || "${DYNAMIC_SYMBOLS}" == true ]]; then
    [[ "${SYMBOL_UPDATE_STRATEGY}" == "per_instance" ]] && _MODE="dpi" || _MODE="dpe"
else
    _MODE="fix"
fi
[[ "${SWAP_LABELS}" == "true" || "${SWAP_LABELS}" == true ]] && _MODE="${_MODE}_swap_${SYMBOL_UPDATE_STRATEGY}"

# Short names for log readability
SHORT_MODEL_TYPE="${MODEL_TYPE//qwen/qa}"
SHORT_MODEL_TYPE="${SHORT_MODEL_TYPE//salmonn/sl}"
SHORT_MODEL_TYPE="${SHORT_MODEL_TYPE//flamingo/af}"

SHORT_DATASET_TYPE="${DATASET_TYPE//voxceleb/vb}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//hvb/h}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//meld_emotion/me}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//voxpopuli/vp}"

RUN_NAME="$(date +"%H%M%S")_${SHORT_MODEL_TYPE}_${SHORT_DATASET_TYPE}_${_MODE}"

OUTPUT_DIR="${BASE_OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs/$(date +"%Y-%m-%d")"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/$(date +"%Y-%m-%d")"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# Construct boolean arguments cleanly
EXTRA_ARGS=""
[[ "${NO_SYMBOLS}" == true || "${NO_SYMBOLS}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --no_symbols"
[[ "${DYNAMIC_SYMBOLS}" == true || "${DYNAMIC_SYMBOLS}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --dynamic_symbols"
[[ "${DIFF_SYMBOL_ENABLED}" == true || "${DIFF_SYMBOL_ENABLED}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --diff_symbol_enabled"
[[ "${DSPO_SLOT_ONLY}" == true || "${DSPO_SLOT_ONLY}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --dspo_slot_only"
[[ "${SWAP_LABELS}" == true || "${SWAP_LABELS}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --swap_labels"
[[ "${USE_DPO}" == true || "${USE_DPO}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --use_dpo --dpo_beta ${DPO_BETA}"
[[ "${VALIDATE_BEFORE_TRAINING}" == false || "${VALIDATE_BEFORE_TRAINING}" == "false" ]] && EXTRA_ARGS="${EXTRA_ARGS} --no_validate_before_training"

echo "============================================================"
echo "Submitting Symbol Training Job: ${RUN_NAME}"
echo "Model:       ${MODEL_TYPE}"
echo "Dataset:     ${DATASET_TYPE}"
echo "Conda Env:   ${CONDA_ENV}"
echo "Queue:       ${QUEUE_NAME} (Host: ${HOSTNAME})"
echo "============================================================"

# Using unquoted EOF so variables evaluate before submission, avoiding complex -v passing
qsub -q "$QUEUE_NAME" \
    -N "$RUN_NAME" \
    -l select=1:num_gpus=1:ncpus=4:gpu_mem=48GB:host=${HOSTNAME} \
    -l walltime=${WALLTIME} \
    -o "${LOG_DIR}/${RUN_NAME}.pbs.log" \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
    << EOF
#!/bin/bash
set -e
set -o pipefail 

cd "${PROJECT_ROOT}"

# Initialize Conda
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Dynamically set LD_LIBRARY_PATH based on active conda env
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

echo "=== Running on: \$(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi

python "${PROJECT_ROOT}/train.py" \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --val_dataset_type "${VAL_DATASET_TYPE}" \
    --device "cuda:0" \
    --batch_size ${BATCH_SIZE} \
    --val_batch_size ${VAL_BATCH_SIZE} \
    --max_samples ${MAX_SAMPLES} \
    --num_examples ${NUM_EXAMPLES} \
    --val_num_examples ${VAL_NUM_EXAMPLES} \
    --num_workers ${NUM_WORKERS} \
    --lora_lr ${LORA_LR} \
    --lora_epochs ${LORA_EPOCHS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --symbol_update_strategy "${SYMBOL_UPDATE_STRATEGY}" \
    --dspo_router_lr ${DSPO_ROUTER_LR} \
    --dspo_tau_anneal_rate ${DSPO_TAU_ANNEAL_RATE} \
    --dspo_phase0_epochs ${DSPO_PHASE0_EPOCHS} \
    --dspo_phase1_patience ${DSPO_PHASE1_PATIENCE} \
    --dspo_phase1_epochs ${DSPO_PHASE1_EPOCHS} \
    --dspo_num_slots ${DSPO_NUM_SLOTS} \
    --dspo_slot_vocab_size ${DSPO_SLOT_VOCAB_SIZE} \
    --dspo_rotation_interval ${DSPO_ROTATION_INTERVAL} \
    --dspo_phase2_rotation ${DSPO_PHASE2_ROTATION} \
    --symbol_difficulty "${SYMBOL_DIFFICULTY}" \
    --val_symbol_difficulty "${VAL_SYMBOL_DIFFICULTY}" \
    --num_symbol_mappings ${NUM_SYMBOL_MAPPINGS} \
    --input_mode "${INPUT_MODE}" \
    --fewshot_mode "${FEWSHOT_MODE}" \
    ${EXTRA_ARGS} 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
EOF

echo ""
echo "Job Submitted Successfully"
echo "Job Name: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"