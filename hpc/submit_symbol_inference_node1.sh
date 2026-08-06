#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

MODEL_TYPE="${MODEL_TYPE:-flamingo}"
_DEFAULT_CONDA_ENV="qwen"
if [[ "${MODEL_TYPE}" == "flamingo" ]]; then _DEFAULT_CONDA_ENV="flamingo"; fi
CONDA_ENV="${CONDA_ENV:-${_DEFAULT_CONDA_ENV}}"
# Eval battery — 8 datasets, evaluated original,fresh in one run.
#   trained tasks (reference): hvb, cremad   |   held-out: ravdess_song, skit_s2i, speech_commands, minds14_en/fr/ko
# NOTE: speech_commands is a capability probe with no legend → its `fresh` column is
# meaningless (ignore it); only its `original` (transcription) number matters.
# voxpopuli is intentionally excluded (AF3-seen + sits at the floor).
DATASET_TYPE="${DATASET_TYPE:-hvb-cremad-ravdess_song-skit_s2i-speech_commands-minds14_en-minds14_fr-minds14_ko}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# ══════════════════════════════════════════════════════════════════════════════
#  hvb+cremad CHECKPOINT PRESETS — best training-original epoch per seed.
#  Uncomment EXACTLY ONE group (the 3 assignments on its line), then run the script.
#  Each preset is one model-instance; leave all commented to use the default below.
# ──────────────────────────────────────────────────────────────────────────────
#  nosym (n=2)
# CKPT_DATE=2026-07-30 TRAIN_RUN=201011_af_h-cremad_nosym  EPOCH=3   # nosym  seed1
# CKPT_DATE=2026-07-30 TRAIN_RUN=201026_af_h-cremad_nosym  EPOCH=2   # nosym  seed2
#  fix_ha (n=2)
# CKPT_DATE=2026-07-30 TRAIN_RUN=201053_af_h-cremad_fix_ha EPOCH=5   # fix_ha seed1
# CKPT_DATE=2026-07-30 TRAIN_RUN=202904_af_h-cremad_fix_ha EPOCH=5   # fix_ha seed2
#  dpi_ha (n=5)
# CKPT_DATE=2026-07-30 TRAIN_RUN=203511_af_h-cremad_dpi_ha EPOCH=4   # dpi    seed1
# CKPT_DATE=2026-07-30 TRAIN_RUN=203518_af_h-cremad_dpi_ha EPOCH=3   # dpi    seed2
# CKPT_DATE=2026-07-31 TRAIN_RUN=090352_af_h-cremad_dpi_ha EPOCH=2   # dpi    seed3
# CKPT_DATE=2026-07-31 TRAIN_RUN=090402_af_h-cremad_dpi_ha EPOCH=5   # dpi    seed4
# CKPT_DATE=2026-07-31 TRAIN_RUN=090409_af_h-cremad_dpi_ha EPOCH=3   # dpi    seed5
#  untrained base AF3 (n=1) — no checkpoint/LoRA
# UNTRAINED=true
# ──────────────────────────────────────────────────────────────────────────────
#  SINGLE-TASK cremad (1-task point) — reused clean checkpoints, best epoch by
#  cremad_original (inference-selected, see metrics/cremad_trained_comparison.csv).
#  Run these 4 (2 nosym + 2 fix_ha) over the full battery; dpi trained separately.
# CKPT_DATE=2026-07-23 TRAIN_RUN=193750_af_cremad_nosym  EPOCH=4    # cr nosym  seed1
# CKPT_DATE=2026-07-22 TRAIN_RUN=180547_af_cremad_nosym  EPOCH=2    # cr nosym  seed2
# CKPT_DATE=2026-07-23 TRAIN_RUN=193915_af_cremad_fix_ha EPOCH=10   # cr fix_ha seed1
# CKPT_DATE=2026-07-23 TRAIN_RUN=193901_af_cremad_fix_ha EPOCH=8    # cr fix_ha seed2
# ──────────────────────────────────────────────────────────────────────────────
#  3-TASK (hvb+cremad+minds14_en) — FLIPPED-label steerability eval.
#  Run each with:  VALIDATION_MODES=original,flipped   (real-label flip → symbol pool irrelevant)
# CKPT_DATE=2026-08-02 TRAIN_RUN=112859_af_h-cr-m14en_dpi_ha EPOCH=4  # dpi   seed1
# CKPT_DATE=2026-08-02 TRAIN_RUN=112912_af_h-cr-m14en_dpi_ha EPOCH=5  # dpi   seed2
# CKPT_DATE=2026-08-02 TRAIN_RUN=113131_af_h-cr-m14en_dpi_ha EPOCH=1  # dpi   seed3
# CKPT_DATE=2026-08-02 TRAIN_RUN=112544_af_h-cr-m14en_nosym  EPOCH=4  # nosym seed1
CKPT_DATE=2026-08-02 TRAIN_RUN=112555_af_h-cr-m14en_nosym  EPOCH=5  # nosym seed2
#   (+ untrained: uncomment UNTRAINED=true above)
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_RUN="${TRAIN_RUN:-203511_af_h-cremad_dpi_ha}"
EPOCH="${EPOCH:-4}"
CKPT_DATE="${CKPT_DATE:-2026-07-30}"
UNTRAINED="${UNTRAINED:-false}"                                                   # true = run base AF3 with NO checkpoint/LoRA
if [[ "${UNTRAINED}" == "true" ]]; then
    CHECKPOINT_PATH=""
elif [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    CHECKPOINT_PATH=$(ls "${HOME}/training/symbol_training/checkpoints/${CKPT_DATE}/${TRAIN_RUN}/lora_epoch${EPOCH}"*.pt 2>/dev/null | head -1)
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        printf 'ERROR: no checkpoint found for TRAIN_RUN=%s EPOCH=%s CKPT_DATE=%s\n' "${TRAIN_RUN}" "${EPOCH}" "${CKPT_DATE}" >&2
        exit 1
    fi
fi
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500}"                                         # samples per dataset (0 = full val set)
NUM_EXAMPLES="${NUM_EXAMPLES:-1}"                                                  # few-shot examples in eval prompt (0 = zero-shot)
FEWSHOT_MODE="${FEWSHOT_MODE:-text}"                                               # few-shot exemplar modality: text | speech
FEWSHOT_PER_CLASS="${FEWSHOT_PER_CLASS:-true}"                                    # true = NUM_EXAMPLES per class (full coverage); false = total (class-balanced)
NUM_WORKERS="${NUM_WORKERS:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
VALIDATION_MODES="${VALIDATION_MODES:-fresh}"                                           # original,fixed,fresh,flipped
VAL_FLIP_SEED="${VAL_FLIP_SEED:-0}"                                                       # 'flipped' mode real-label derangement seed (same seed = same flip across models)
NO_LEGEND="${NO_LEGEND:-false}"                                                           # true = strip label descriptions at eval (ablation: does the description carry the work?)
SPLIT="${SPLIT:-validation}"                                                            # test | validation

# Symbol map probe — swap to any of:
#   analysis/symbol_maps/ep3_fixed.json   (voxceleb F1=0.028, meld F1=0.142)
#   analysis/symbol_maps/ep3_fresh.json   (voxceleb F1=0.024, meld F1=0.334)
#   analysis/symbol_maps/ep4_fixed.json   (voxceleb F1=0.053, meld F1=0.275)
#   analysis/symbol_maps/ep4_fresh.json   (voxceleb F1=0.425, meld F1=0.251) ← best
# ${PROJECT_ROOT}/analysis/symbol_maps/ep4_fresh.json
SYMBOL_MAP_FILE="${SYMBOL_MAP_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/training/symbol_training}"
METRICS_DIR="${METRICS_BASE:-${HOME}/training/symbol_training/metrics}/$(date +"%Y-%m-%d")"  # metrics output (dated subfolder)
LOG_DIR="${LOGS_INFERENCE_DIR:-${HOME}/training/symbol_training/logs_inference}/$(date +"%Y-%m-%d")"

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
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//speech_commands/sc}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_en/m14en}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_fr/m14fr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_ko/m14ko}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//sprsound/spr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//heysquad/hsq}"

# Extract epoch number from checkpoint filename (e.g. lora_epoch1_phase0.pt → 1)
if [[ "${CHECKPOINT_PATH}" =~ epoch([0-9]+) ]]; then
    EPOCH_NUM="${BASH_REMATCH[1]}"
else
    EPOCH_NUM="X"
fi

# Extract training dataset from checkpoint dir name (3rd _-separated field)
# e.g. "173148_qa_me_dspo" → "me"
CHECKPOINT_DIR_NAME=$(basename "$(dirname "${CHECKPOINT_PATH}")")
TRAIN_DATA_RAW=$(echo "${CHECKPOINT_DIR_NAME}" | cut -d'_' -f3)
TRAIN_DATA="${TRAIN_DATA_RAW//voxceleb/vb}"
TRAIN_DATA="${TRAIN_DATA//hvb/h}"
TRAIN_DATA="${TRAIN_DATA//meld_emotion/me}"
TRAIN_DATA="${TRAIN_DATA//voxpopuli/vp}"
[[ -z "${TRAIN_DATA}" ]] && TRAIN_DATA="unk"

[[ "${MAX_VAL_SAMPLES}" == "0" ]] && SAMPLES_TAG="" || SAMPLES_TAG="_${MAX_VAL_SAMPLES}"
[[ -n "${CHECKPOINT_PATH}" ]] && _CKPT_TAG="_tr${TRAIN_DATA}_ep${EPOCH_NUM}" || _CKPT_TAG="_untrained"
_SPLIT_TAG="${SPLIT:0:3}"
# few-shot tag: _sh0 = zero-shot; _fs4s = 4 speech, _fs4t = 4 text (+pc if per-class)
if [[ "${NUM_EXAMPLES}" == "0" ]]; then
    _FS_TAG="_sh0"
else
    _FS_TAG="_fs${NUM_EXAMPLES}${FEWSHOT_MODE:0:1}$( [[ "${FEWSHOT_PER_CLASS}" == "true" ]] && printf 'pc' )"
fi
[[ "${NO_LEGEND}" == "true" ]] && _FS_TAG="${_FS_TAG}_nl"   # distinguish no-legend from with-legend runs
RUN_NAME="${RUN_NAME:-$(date +"%H%M%S")_i_${SHORT_MODEL_TYPE}_${SHORT_DATASET_TYPE}_${_SPLIT_TAG}${SAMPLES_TAG}${_CKPT_TAG}${_FS_TAG}}"

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${METRICS_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    export _NOHUP_LAUNCHED=1 RUN_NAME
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    printf 'Inference launched in background (PID: %s)\n' "$!"
    printf 'Follow logs: tail -f "%s"\n' "${LOG_FILE}"
    exit 0
fi

printf '%s\n' "============================================================"
printf '%s\n' "Starting Symbol Inference on node1"
printf '%s\n' "Project Root:    ${PROJECT_ROOT}"
printf '%s\n' "Conda Env:       ${CONDA_ENV}"
printf '%s\n' "Model:           ${MODEL_TYPE}"
printf '%s\n' "Dataset:         ${DATASET_TYPE}"
printf '%s\n' "Checkpoint:      ${CHECKPOINT_PATH}"
printf '%s\n' "Symbol Map:      ${SYMBOL_MAP_FILE:-<from checkpoint>}"
printf '%s\n' "Split:           ${SPLIT}"
printf '%s\n' "Samples:         ${SAMPLES_TAG}"
printf '%s\n' "Metrics Dir:     ${METRICS_DIR}/${RUN_NAME}"
printf '%s\n' "Run Name:        ${RUN_NAME}"
printf '%s\n' "Log File:        ${LOG_FILE}"
printf '%s\n' "============================================================"

export CUDA_VISIBLE_DEVICES

python inference.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --device "${DEVICE}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --num_examples "${NUM_EXAMPLES}" \
    --fewshot_mode "${FEWSHOT_MODE}" \
    $( [[ "${FEWSHOT_PER_CLASS}" == "true" ]] && printf '%s' "--fewshot_per_class" ) \
    --num_workers "${NUM_WORKERS}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --val_flip_seed "${VAL_FLIP_SEED}" \
    $( [[ "${NO_LEGEND}" == "true" ]] && printf '%s' "--no_legend" ) \
    --split "${SPLIT}" \
    --metrics_dir "${METRICS_DIR}" \
    $( [[ -n "${SYMBOL_MAP_FILE}" ]] && printf '%s %s' "--symbol_map_file" "${SYMBOL_MAP_FILE}" ) \
    >> "${LOG_FILE}" 2>&1
