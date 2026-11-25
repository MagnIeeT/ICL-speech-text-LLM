#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"



cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

# Reduce CUDA allocator fragmentation so transient long-sequence batches don't OOM
# at batch_size=2 (card sits ~45/47 GiB; ~2 GiB was reserved-but-unallocated).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Model ---
MODEL_TYPE="${MODEL_TYPE:-qwen}"                                                        # model backend: qwen | salmonn | flamingo

# --- Infrastructure ---
# Auto-select conda env based on MODEL_TYPE unless explicitly overridden.
# qwen needs transformers==4.45.2; flamingo needs transformers>=5.0;
# salmonn needs its own env (transformers==4.28.0, torch cu118) — SALMONN's custom
# modeling code is incompatible with the qwen env's newer stack.
_DEFAULT_CONDA_ENV="qwen"
if [[ "${MODEL_TYPE}" == "flamingo" ]]; then _DEFAULT_CONDA_ENV="flamingo"; fi
if [[ "${MODEL_TYPE}" == "salmonn" ]];  then _DEFAULT_CONDA_ENV="salmonn";  fi
CONDA_ENV="${CONDA_ENV:-${_DEFAULT_CONDA_ENV}}"                                   # conda env (auto: qwen→qwen, flamingo→flamingo, salmonn→salmonn)
DEVICE="${DEVICE:-cuda:0}"                                                        # torch device for training
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"                                 # which GPU(s) the process can see
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/training/symbol_training}"                      # root output directory
CHECKPOINT_DIR="${CHECKPOINT_BASE:-${HOME}/training/symbol_training/checkpoints}/$(date +"%Y-%m-%d")"  # checkpoint directory (dated subfolder)
LOG_DIR="${LOGS_DIR:-${HOME}/training/logs}/$(date +"%Y-%m-%d")"                 # log directory (dated subfolder)
NUM_WORKERS="${NUM_WORKERS:-2}"                                                   # dataloader worker processes

# --- Data ---
# 3-task breadth run (hvb + cremad + minds14_en). Val = full eval battery so we get
# per-epoch held-out curves. IMPORTANT: still select the epoch on TRAINING-original only
# (hvb, cremad, minds14_en) — the held-out sets are logged, never used for selection.
# (voxpopuli excluded — AF3-seen + sits at the floor. Lower VAL_MAX_SAMPLES to speed up.)
#   MODE per job (uncomment ONE, or leave all commented = dpi_ha default):
#     nosym   → NO_SYMBOLS=true
#     fix_ha  → DYNAMIC_SYMBOLS=false
#     dpi_ha  → (defaults, nothing to set)
DATASET_TYPE="${DATASET_TYPE:-hvb-cremad-minds14_en}"                            # training dataset(s), dash-separated
# VAL_DATASET_TYPE="${VAL_DATASET_TYPE:-hvb-cremad-minds14_en-ravdess_song-skit_s2i-speech_commands-minds14_fr-minds14_ko}"  # full eval battery
VAL_DATASET_TYPE="${VAL_DATASET_TYPE:-hvb-cremad-minds14_en}"  # full eval battery
MAX_SAMPLES="${MAX_SAMPLES:-2000}"                                                  # max training samples (0 = full dataset)
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-500}"                                         # max validation samples per dataset (0 = full)
INPUT_MODE="${INPUT_MODE:-speech_only}"                                           # query modality: speech_only | text_only
FEWSHOT_MODE="${FEWSHOT_MODE:-text}"                                              # few-shot example modality: text | speech
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"                                                 # few-shot examples in training prompt (0 = zero-shot; MAX when NUM_EXAMPLES_MIN>0)
NUM_EXAMPLES_MIN="${NUM_EXAMPLES_MIN:-0}"                                        # >=0 = random per-prompt count U[min,NUM_EXAMPLES] during training (min=0 → some zero-shot; Wei uses 2); -1 = fixed. Eval stays fixed
FEWSHOT_PER_CLASS="${FEWSHOT_PER_CLASS:-false}"                                   # true = NUM_EXAMPLES per class (full coverage); false = NUM_EXAMPLES total (class-balanced)
VAL_NUM_EXAMPLES="${VAL_NUM_EXAMPLES:-0}"                                         # few-shot examples in validation prompt
NO_LEGEND="${NO_LEGEND:-false}"                                                   # true = strip label descriptions from prompt (Wei-style, meaning from exemplars/symbols only)

# --- Training ---
LORA_EPOCHS="${LORA_EPOCHS:-3}"                                                  # number of training epochs
LORA_LR="${LORA_LR:-1e-5}"                                                       # LoRA adapter learning rate
BATCH_SIZE="${BATCH_SIZE:-1}"                                                     # per-step training batch size
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"                                             # per-step validation batch size
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"                  # effective batch = BATCH_SIZE × this
WARMUP_STEPS="${WARMUP_STEPS:-100}"                                               # LR scheduler warmup steps
VALIDATE_BEFORE_TRAINING="${VALIDATE_BEFORE_TRAINING:-false}"                     # true = run validation pass before epoch 1 (baseline score)
VALIDATION_MODES="${VALIDATION_MODES:-original}"                     # which symbol modes to validate: original,fixed,fresh

# --- Symbol mode (pick ONE of the blocks below) ---
# SymDPO: DPO loss over chosen/rejected symbol pairs (requires NO_SYMBOLS=false)
USE_DPO="${USE_DPO:-false}"                                                        # true = use SymDPO trainer
DPO_BETA="${DPO_BETA:-0.1}"                                                       # DPO temperature β: higher = stronger preference margin


# Fixed/dynamic symbols
NO_SYMBOLS="${NO_SYMBOLS:-true}"                                                 # true = disable symbol replacement, use raw labels (must be false for DPO)
DYNAMIC_SYMBOLS="${DYNAMIC_SYMBOLS:-true}"                                       # true = regenerate symbol mapping each epoch
SYMBOL_UPDATE_STRATEGY="${SYMBOL_UPDATE_STRATEGY:-per_instance}"                    # when dynamic symbols refresh: per_epoch | per_instance
SWAP_LABELS="${SWAP_LABELS:-true}"                                               # true = randomly shuffle label↔symbol assignments each epoch
NUM_SYMBOL_MAPPINGS="${NUM_SYMBOL_MAPPINGS:-100}"                                  # M pre-generated mappings: per_epoch uses pool[epoch%M], per_instance uses random.choice(pool)
SURROGATE_MIX="${SURROGATE_MIX:-0}"                                                 # per-instance P(flipped regime); 0 = synthetic symbols only
SEED="${SEED:-}"                                                                  # global seed for python/numpy/torch; empty = unseeded

# Surface-form consistency objective. Weight 0 = off (one legend per step).
# Above 0, eligible instances run TWO legends in one step and are penalised for
# answering them differently once mapped back to definitions. Multi-label sets
# (HVB) cannot carry it and take one ordinary pass following SURROGATE_MIX.
# Weight 0 WITH the other two flags set is the control: two views, no coupling.
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0}"                                       # lambda on the divergence term; 0.5-1.0 is the sensible first range
CONSISTENCY_VIEWS="${CONSISTENCY_VIEWS:-sym,flip}"                                  # two of sym|flip|orig: sym,orig adds canonical supervision; sym,sym stays pure surrogate; sym,flip spans both families
CONSISTENCY_DIVERGENCE="${CONSISTENCY_DIVERGENCE:-js}"                          # sym_kl | js | l2
CONSISTENCY_SHUFFLE_LEGEND="${CONSISTENCY_SHUFFLE_LEGEND:-false}"                  # true = random legend order per view and per instance (training only)
CONSISTENCY_CONTROL="${CONSISTENCY_CONTROL:-}"                                     # non-empty = control: two-view path at CONSISTENCY_WEIGHT=0, no coupling
KEEP_ALL_CHECKPOINTS="${KEEP_ALL_CHECKPOINTS:-false}"                              # true = skip the pruner; every epoch's LoRA is kept (diagnostics only, never for selection)

# DG-Trap (definition-trap training, chapter 2). Real label words only; per instance
# orig | goldswap | pairswap naming, harmless word dropout, trap + conditional margins.
# TRAP_ENABLED=true forces SWAP_LABELS=false and SURROGATE_MIX=0 (its own naming).
# A0 (data-only baseline) = TRAP_LAMBDA=0 TRAP_COND_LAMBDA=0.
TRAP_ENABLED="${TRAP_ENABLED:-false}"
TRAP_P_ORIG="${TRAP_P_ORIG:-0.3333}"                                                # naming proportions (normalised in python)
TRAP_P_GOLDSWAP="${TRAP_P_GOLDSWAP:-0.3333}"
TRAP_P_PAIRSWAP="${TRAP_P_PAIRSWAP:-0.3333}"                                        # 0 = ablation A4 (no leak blocker)
TRAP_NN_FRAC="${TRAP_NN_FRAC:-0.5}"                                                 # goldswap partner = nearest-neighbour definition with this prob
TRAP_DROPOUT_MAX="${TRAP_DROPOUT_MAX:-0.2}"                                         # harmless word dropout rate ~ U(0, max)
TRAP_LAMBDA="${TRAP_LAMBDA:-0.5}"                                                   # trap margin weight; 0 = measured only
TRAP_COND_LAMBDA="${TRAP_COND_LAMBDA:-0.5}"                                         # conditional margin weight; 0 = no second pass
TRAP_COND_FRAC="${TRAP_COND_FRAC:-0.3}"                                             # fraction of instances with the conditional term
TRAP_COND_MODE="${TRAP_COND_MODE:-deciding}"                                        # deciding | random (ablation A5)
TRAP_BETA="${TRAP_BETA:-2.0}"
TRAP_MARGIN="${TRAP_MARGIN:-0.5}"
TRAP_COND_MARGIN="${TRAP_COND_MARGIN:-0.5}"
TRAP_DECIDING_WORDS="${TRAP_DECIDING_WORDS:-config/data_config/deciding_words.json}"
if [[ "${TRAP_ENABLED}" == "true" ]]; then
    SWAP_LABELS="false"
    SURROGATE_MIX="0"
fi

if [[ "${USE_DPO}" == "true" ]]; then
    _MODE="symdpo"
elif [[ "${NO_SYMBOLS}" == "true" ]]; then
    _MODE="nosym"
elif [[ "${DYNAMIC_SYMBOLS}" == "true" ]]; then
    [[ "${SYMBOL_UPDATE_STRATEGY}" == "per_instance" ]] && _MODE="dpi" || _MODE="dpe"
else
    _MODE="fix"
fi
[[ "${SWAP_LABELS}" == "true" ]] && _MODE="${_MODE}_swap_${SYMBOL_UPDATE_STRATEGY}"
[[ "${SURROGATE_MIX}" != "0" ]] && _MODE="${_MODE}_mix${SURROGATE_MIX}"
# Consistency tag: _consSO0.5 = views sym,orig at weight 0.5. Weight 0 with views
# set is the control, tagged _consSO0 so it is distinguishable from a plain run.
if [[ -n "${CONSISTENCY_VIEWS}" && ( "${CONSISTENCY_WEIGHT}" != "0" || -n "${CONSISTENCY_CONTROL:-}" ) ]]; then
    _CV="$(printf '%s' "${CONSISTENCY_VIEWS}" | tr -d ',' | tr '[:lower:]' '[:upper:]' | cut -c1,4)"
    [[ "${CONSISTENCY_VIEWS}" == *same* ]] && _CV="Ssame"                          # sym,same would otherwise collide with sym,sym ("SS")
    _MODE="${_MODE}_cons${_CV}${CONSISTENCY_WEIGHT}"
    [[ "${CONSISTENCY_DIVERGENCE}" != "js" ]] && _MODE="${_MODE}${CONSISTENCY_DIVERGENCE}"
    [[ "${CONSISTENCY_SHUFFLE_LEGEND}" == "true" ]] && _MODE="${_MODE}_shuf"
fi
if [[ "${TRAP_ENABLED}" == "true" ]]; then
    # e.g. _trap_lt0.5_lc0.5 ; non-default knobs appended so ablations stay distinguishable
    _MODE="trap_lt${TRAP_LAMBDA}_lc${TRAP_COND_LAMBDA}"
    awk "BEGIN{exit !(${TRAP_P_PAIRSWAP}+0==0)}" && _MODE="${_MODE}_nopair"     # numeric: 0, 0.0, 0.00
    [[ "${TRAP_COND_MODE}" != "deciding" ]] && _MODE="${_MODE}_cond${TRAP_COND_MODE}"
    [[ "${TRAP_DROPOUT_MAX}" != "0.2" ]] && _MODE="${_MODE}_do${TRAP_DROPOUT_MAX}"
    [[ "${TRAP_NN_FRAC}" != "0.5" ]] && _MODE="${_MODE}_nn${TRAP_NN_FRAC}"
fi
[[ "${NO_LEGEND}" == "true" ]] && _MODE="${_MODE}_nl"                              # no-legend tag
[[ "${NUM_EXAMPLES}" != "0" ]] && _MODE="${_MODE}_fs${NUM_EXAMPLES}${FEWSHOT_MODE:0:1}"   # few-shot tag (_fs4s = 4 speech, _fs4t = 4 text)

SHORT_MODEL_TYPE="${MODEL_TYPE//qwen/qa}"
SHORT_MODEL_TYPE="${SHORT_MODEL_TYPE//salmonn/sl}"
SHORT_MODEL_TYPE="${SHORT_MODEL_TYPE//flamingo/af}"

SHORT_DATASET_TYPE="${DATASET_TYPE//voxceleb/vb}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//hvb/h}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//meld_emotion/me}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//voxpopuli/vp}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//cremad/cr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//ravdess_song/rs}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//skit_s2i/sk}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_en/m14en}"

RUN_NAME="${RUN_NAME:-$(date +"%H%M%S")_${SHORT_MODEL_TYPE}_${SHORT_DATASET_TYPE}_${_MODE}}"

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

NSYS_PROFILE="${NSYS_PROFILE:-false}"                                             # true = wrap python with nsys profile
TORCH_PROFILE="${TORCH_PROFILE:-false}"                                           # true = enable PyTorch built-in profiler on epoch 1

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

_PYTHON_CMD="python"
if [[ "${NSYS_PROFILE}" == "true" ]]; then
    NSYS_OUT="${OUTPUT_DIR}/${RUN_NAME}_nsys"
    _PYTHON_CMD="nsys profile --output=${NSYS_OUT} --trace=cuda,osrt --force-overwrite=true --wait=primary python"
    printf 'nsys profiling enabled → %s.nsys-rep\n' "${NSYS_OUT}"
fi

${_PYTHON_CMD} train.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --val_dataset_type "${VAL_DATASET_TYPE}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --val_max_samples "${VAL_MAX_SAMPLES}" \
    --num_examples "${NUM_EXAMPLES}" \
    --num_examples_min "${NUM_EXAMPLES_MIN}" \
    $( [[ "${FEWSHOT_PER_CLASS}" == "true" ]] && printf '%s' "--fewshot_per_class" ) \
    --val_num_examples "${VAL_NUM_EXAMPLES}" \
    --num_workers "${NUM_WORKERS}" \
    --lora_lr "${LORA_LR}" \
    --lora_epochs "${LORA_EPOCHS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --run_name "${RUN_NAME}" \
    $( [[ -n "${SEED}" ]] && printf '%s %s' "--seed" "${SEED}" ) \
    --validation_modes "${VALIDATION_MODES}" \
    --symbol_update_strategy "${SYMBOL_UPDATE_STRATEGY}" \
    $( [[ "${NO_SYMBOLS}" == "true" ]] && printf '%s' "--no_symbols" ) \
    $( [[ "${DYNAMIC_SYMBOLS}" == "true" ]] && printf '%s' "--dynamic_symbols" ) \
    $( [[ "${SWAP_LABELS}" == "true" ]] && printf '%s' "--swap_labels" ) \
    --num_symbol_mappings "${NUM_SYMBOL_MAPPINGS}" \
    --surrogate_mix "${SURROGATE_MIX}" \
    --consistency_weight "${CONSISTENCY_WEIGHT}" \
    --consistency_views "${CONSISTENCY_VIEWS}" \
    --consistency_divergence "${CONSISTENCY_DIVERGENCE}" \
    $( [[ "${CONSISTENCY_SHUFFLE_LEGEND}" == "true" ]] && printf '%s' "--consistency_shuffle_legend" ) \
    $( [[ -n "${CONSISTENCY_CONTROL}" ]] && printf '%s' "--consistency_control" ) \
    $( [[ "${TRAP_ENABLED}" == "true" ]] && printf '%s ' "--trap_enabled" \
        "--trap_p_orig" "${TRAP_P_ORIG}" "--trap_p_goldswap" "${TRAP_P_GOLDSWAP}" \
        "--trap_p_pairswap" "${TRAP_P_PAIRSWAP}" "--trap_nn_frac" "${TRAP_NN_FRAC}" \
        "--trap_dropout_max" "${TRAP_DROPOUT_MAX}" "--trap_lambda" "${TRAP_LAMBDA}" \
        "--trap_cond_lambda" "${TRAP_COND_LAMBDA}" "--trap_cond_frac" "${TRAP_COND_FRAC}" \
        "--trap_cond_mode" "${TRAP_COND_MODE}" "--trap_beta" "${TRAP_BETA}" \
        "--trap_margin" "${TRAP_MARGIN}" "--trap_cond_margin" "${TRAP_COND_MARGIN}" \
        "--trap_deciding_words" "${TRAP_DECIDING_WORDS}" ) \
    $( [[ "${USE_DPO}" == "true" ]] && printf '%s' "--use_dpo" ) \
    $( [[ "${USE_DPO}" == "true" ]] && printf '%s' "--dpo_beta ${DPO_BETA}" ) \
    --input_mode "${INPUT_MODE}" \
    --fewshot_mode "${FEWSHOT_MODE}" \
    $( [[ "${NO_LEGEND}" == "true" ]] && printf '%s' "--no_legend" ) \
    $( [[ "${VALIDATE_BEFORE_TRAINING}" == "false" ]] && printf '%s' "--no_validate_before_training" ) \
    $( [[ "${TORCH_PROFILE}" == "true" ]] && printf '%s' "--torch_profile" ) \
    >> "${LOG_FILE}" 2>&1

# --- Prune non-best checkpoints (keeps best epoch per validation mode on train tasks) ---
# Safe no-op if training didn't complete or the dir is empty; never fails the run.
if [[ "${KEEP_ALL_CHECKPOINTS}" == "true" ]]; then
    printf '%s\n' "KEEP_ALL_CHECKPOINTS=true: pruner skipped, every epoch kept for ${RUN_NAME}" >> "${LOG_FILE}" 2>&1
else
    printf '%s\n' "Pruning non-best checkpoints for ${RUN_NAME}" >> "${LOG_FILE}" 2>&1
    python hpc/prune_checkpoints.py "${CHECKPOINT_DIR}/${RUN_NAME}" --apply >> "${LOG_FILE}" 2>&1 || true
fi