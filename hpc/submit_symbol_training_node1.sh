#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

# --- Model ---
MODEL_TYPE="${MODEL_TYPE:-qwen}"                                                  # model backend: qwen | salmonn | flamingo

# --- Infrastructure ---
# Auto-select conda env based on MODEL_TYPE unless explicitly overridden.
# qwen/salmonn need transformers==4.45.2; flamingo needs transformers>=5.0.
_DEFAULT_CONDA_ENV="qwen"
if [[ "${MODEL_TYPE}" == "flamingo" ]]; then _DEFAULT_CONDA_ENV="flamingo"; fi
CONDA_ENV="${CONDA_ENV:-${_DEFAULT_CONDA_ENV}}"                                   # conda environment (auto: qwen→qwen env, flamingo→flamingo env)
DEVICE="${DEVICE:-cuda:0}"                                                        # torch device for training
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"                                 # which GPU(s) the process can see
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/training/symbol_training}"                      # root output directory
CHECKPOINT_DIR="${CHECKPOINT_BASE:-${HOME}/training/symbol_training/checkpoints}/$(date +"%Y-%m-%d")"  # checkpoint directory (dated subfolder)
LOG_DIR="${LOGS_DIR:-${HOME}/training/logs}/$(date +"%Y-%m-%d")"                 # log directory (dated subfolder)
NUM_WORKERS="${NUM_WORKERS:-2}"                                                   # dataloader worker processes

# --- Data ---
DATASET_TYPE="${DATASET_TYPE:-meld_emotion}"                                      # training dataset(s), dash-separated
VAL_DATASET_TYPE="${VAL_DATASET_TYPE:-voxceleb-hvb-voxpopuli-meld_emotion}"      # validation dataset(s), dash-separated
MAX_SAMPLES="${MAX_SAMPLES:-0}"                                                  # max training samples (0 = full dataset)
INPUT_MODE="${INPUT_MODE:-speech_only}"                                           # query modality: speech_only | text_only
FEWSHOT_MODE="${FEWSHOT_MODE:-text}"                                              # few-shot example modality: text | speech
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"                                                 # few-shot examples in training prompt (0 = zero-shot)
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-0}"                                         # few-shot examples in validation prompt

# --- Training ---
LORA_EPOCHS="${LORA_EPOCHS:-10}"                                                  # number of training epochs
LORA_LR="${LORA_LR:-1e-5}"                                                       # LoRA adapter learning rate
BATCH_SIZE="${BATCH_SIZE:-1}"                                                     # per-step training batch size
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"                                             # per-step validation batch size
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"                  # effective batch = BATCH_SIZE × this
WARMUP_STEPS="${WARMUP_STEPS:-100}"                                               # LR scheduler warmup steps
VALIDATE_BEFORE_TRAINING="${VALIDATE_BEFORE_TRAINING:-false}"                     # true = run validation pass before epoch 1 (baseline score)
VALIDATION_MODES="${VALIDATION_MODES:-original,fixed,fresh}"                     # which symbol modes to validate: original,fixed,fresh

# --- Symbol mode (pick ONE of the blocks below) ---
# SymDPO: DPO loss over chosen/rejected symbol pairs (requires NO_SYMBOLS=false)
USE_DPO="${USE_DPO:-false}"                                                        # true = use SymDPO trainer
DPO_BETA="${DPO_BETA:-0.1}"                                                       # DPO temperature β: higher = stronger preference margin

# D-SPO: differentiable slot routing (set DIFF_SYMBOL_ENABLED=true to activate)
DIFF_SYMBOL_ENABLED="${DIFF_SYMBOL_ENABLED:-true}"                               # true = enable D-SPO differentiable slot routing
DSPO_ROUTER_LR="${DSPO_ROUTER_LR:-1e-2}"                                         # D-SPO router (slot preference matrix) learning rate
DSPO_TAU_ANNEAL_RATE="${DSPO_TAU_ANNEAL_RATE:-0.0001}"                          # D-SPO Gumbel temperature decay rate per step
DSPO_SLOT_ONLY="${DSPO_SLOT_ONLY:-false}"                                         # true = freeze LoRA, train only slot matrix (also needs DIFF_SYMBOL_ENABLED=true)
DSPO_PHASE0_EPOCHS="${DSPO_PHASE0_EPOCHS:-1}"                                    # >0: LoRA-only warmup on original labels before D-SPO starts
DSPO_PHASE1_PATIENCE="${DSPO_PHASE1_PATIENCE:-3}"                                # >0: auto-switch from slot-only Phase 1 to LoRA Phase 2 when conf_mean plateaus for this many epochs
DSPO_PHASE1_EPOCHS="${DSPO_PHASE1_EPOCHS:-5}"                                    # >0: hard cap on Phase 1 epochs — switches to Phase 2 even if patience not triggered
DSPO_NUM_SLOTS="${DSPO_NUM_SLOTS:-20}"                                            # D-SPO total slot pool size (>= num training labels; extra slots used when rotation enabled)
DSPO_SLOT_VOCAB_SIZE="${DSPO_SLOT_VOCAB_SIZE:-25}"                               # D-SPO private vocab tokens per slot (K)
DSPO_ROTATION_INTERVAL="${DSPO_ROTATION_INTERVAL:-200}"                           # Phase 1 slot rotation: -1=fixed, 0=per epoch, >0=every N steps
DSPO_PHASE2_ROTATION="${DSPO_PHASE2_ROTATION:-0}"                               # Phase 2 symbol refresh: -1=fixed, 0=per epoch, 1=per instance, >1=every N steps

# Fixed/dynamic symbols
NO_SYMBOLS="${NO_SYMBOLS:-false}"                                                 # true = disable symbol replacement, use raw labels (must be false for DPO)
DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-false}"                                       # true = regenerate symbol mapping each epoch
SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_instance}"                    # when dynamic symbols refresh: per_epoch | per_instance
SWAP_LABELS="${SWAP_LABELS:-false}"                                               # true = randomly shuffle label↔symbol assignments each epoch

if [[ "${USE_DPO}" == "true" ]]; then
    _MODE="symdpo"
elif [[ "${DIFF_SYMBOL_ENABLED}" == "true" ]]; then
    _MODE="dspo"
    [[ "${DSPO_SLOT_ONLY}" == "true" ]] && _MODE="dspo_slotonly"
elif [[ "${NO_SYMBOLS}" == "true" ]]; then
    _MODE="nosym"
elif [[ "${DYNAMIC_SYMBOLS}" == "true" ]]; then
    _MODE="dyn_${SYMBOL_UPDATE_STRATEGY}"
else
    _MODE="fixed"
fi
[[ "${SWAP_LABELS}" == "true" ]] && _MODE="${_MODE}_swap_${SYMBOL_UPDATE_STRATEGY}"
RUN_NAME="${RUN_NAME:-$(date +"%H%M%S")_${MODEL_TYPE}_${DATASET_TYPE}_${_MODE}}"

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi

# Avoid conda activate failing under 'set -u' if MKL_INTERFACE_LAYER is unset
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

conda activate "${CONDA_ENV}"

# Prefer conda's libstdc++ over the system one
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Avoid tokenizers parallelism warnings after fork
export TOKENIZERS_PARALLELISM="false"
# Force line-buffered output when redirected to a file
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    export _NOHUP_LAUNCHED=1 RUN_NAME
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    printf 'Training launched in background (PID: %s)\n' "$!"
    printf 'Follow logs: tail -f "%s"\n' "${LOG_FILE}"
    exit 0
fi

printf '%s\n' "============================================================"
printf '%s\n' "Starting Symbol Training on node1"
printf '%s\n' "Project Root: ${PROJECT_ROOT}"
printf '%s\n' "Conda Env:    ${CONDA_ENV}"
printf '%s\n' "Model:        ${MODEL_TYPE}"
printf '%s\n' "Dataset:      ${DATASET_TYPE}"
printf '%s\n' "Val Dataset:  ${VAL_DATASET_TYPE}"
printf '%s\n' "Run Name:     ${RUN_NAME}"
printf '%s\n' "Checkpoint:   ${CHECKPOINT_DIR}/${RUN_NAME}"
printf '%s\n' "Log File:     ${LOG_FILE}"
printf '%s\n' "============================================================"

export CUDA_VISIBLE_DEVICES

python train.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --val_dataset_type "${VAL_DATASET_TYPE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_examples "${NUM_EXAMPLES}" \
    --val_num_examples "${VAL_NUM_EXAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --lora_lr "${LORA_LR}" \
    --lora_epochs "${LORA_EPOCHS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --symbol_update_strategy "${SYMBOL_UPDATE_STRATEGY}" \
    $( [[ "${NO_SYMBOLS}" == "true" ]] && printf '%s' "--no_symbols" ) \
    $( [[ "${DYNAMIC_SYMBOLS}" == "true" ]] && printf '%s' "--dynamic_symbols" ) \
    $( [[ "${DIFF_SYMBOL_ENABLED}" == "true" ]] && printf '%s' "--diff_symbol_enabled" ) \
    --dspo_router_lr "${DSPO_ROUTER_LR}" \
    --dspo_tau_anneal_rate "${DSPO_TAU_ANNEAL_RATE}" \
    $( [[ "${DSPO_SLOT_ONLY}" == "true" ]] && printf '%s' "--dspo_slot_only" ) \
    --dspo_phase0_epochs "${DSPO_PHASE0_EPOCHS}" \
    --dspo_phase1_patience "${DSPO_PHASE1_PATIENCE}" \
    --dspo_phase1_epochs "${DSPO_PHASE1_EPOCHS}" \
    --dspo_num_slots "${DSPO_NUM_SLOTS}" \
    --dspo_slot_vocab_size "${DSPO_SLOT_VOCAB_SIZE}" \
    --dspo_rotation_interval "${DSPO_ROTATION_INTERVAL}" \
    --dspo_phase2_rotation "${DSPO_PHASE2_ROTATION}" \
    $( [[ "${SWAP_LABELS}" == "true" ]] && printf '%s' "--swap_labels" ) \
    $( [[ "${USE_DPO}" == "true" ]] && printf '%s' "--use_dpo" ) \
    $( [[ "${USE_DPO}" == "true" ]] && printf '%s' "--dpo_beta ${DPO_BETA}" ) \
    --input_mode "${INPUT_MODE}" \
    --fewshot_mode "${FEWSHOT_MODE}" \
    $( [[ "${VALIDATE_BEFORE_TRAINING}" == "false" ]] && printf '%s' "--no_validate_before_training" ) \
    >> "${LOG_FILE}" 2>&1
