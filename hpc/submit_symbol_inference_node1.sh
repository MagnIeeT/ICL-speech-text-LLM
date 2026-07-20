#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

CONDA_ENV="${CONDA_ENV:-qwen}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
DATASET_TYPE="${DATASET_TYPE:-voxceleb-hvb-voxpopuli-meld_emotion}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Checkpoint from 041203_qwen_meld_emotion_dspo run (epoch 1, Phase0-LoRA)
# Swap to any other .pt file to test a different checkpoint
#${HOME}/training/symbol_training/checkpoints/173148_qwen_meld_emotion_dspo/lora_epoch1_phase0.pt
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"                                         # samples per dataset (0 = full val set)
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"                                                  # few-shot examples in prompt
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
VALIDATION_MODES="${VALIDATION_MODES:-fixed}"
SPLIT="${SPLIT:-test}"                                                            # test | validation

# Symbol map probe — swap to any of:
#   analysis/symbol_maps/ep3_fixed.json   (voxceleb F1=0.028, meld F1=0.142)
#   analysis/symbol_maps/ep3_fresh.json   (voxceleb F1=0.024, meld F1=0.334)
#   analysis/symbol_maps/ep4_fixed.json   (voxceleb F1=0.053, meld F1=0.275)
#   analysis/symbol_maps/ep4_fresh.json   (voxceleb F1=0.425, meld F1=0.251) ← best
# ${PROJECT_ROOT}/analysis/symbol_maps/ep4_fresh.json
SYMBOL_MAP_FILE="${SYMBOL_MAP_FILE:-${PROJECT_ROOT}/analysis/symbol_maps/ep4_fresh.json}"
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
[[ -n "${CHECKPOINT_PATH}" ]] && _CKPT_TAG="_tr${TRAIN_DATA}_ep${EPOCH_NUM}" || _CKPT_TAG=""
RUN_NAME="${RUN_NAME:-$(date +"%H%M%S")_i_${SHORT_MODEL_TYPE}_${SHORT_DATASET_TYPE}${SAMPLES_TAG}${_CKPT_TAG}_sh${NUM_EXAMPLES}}"

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
    --num_workers "${NUM_WORKERS}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --split "${SPLIT}" \
    --metrics_dir "${METRICS_DIR}" \
    $( [[ -n "${SYMBOL_MAP_FILE}" ]] && printf '%s %s' "--symbol_map_file" "${SYMBOL_MAP_FILE}" ) \
    >> "${LOG_FILE}" 2>&1
