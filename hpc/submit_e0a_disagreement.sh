#!/usr/bin/env bash
#
# E0a: where do audio-only and transcript-only predictions diverge?
#
# Runs submit_symbol_inference_node1.sh twice on the SAME checkpoint and the
# SAME items, changing only --input_mode, then compares the two prediction
# dumps. Produces:
#   1. the disagreement rate per dataset, with a 2x2 against the gold label
#   2. the disagreement subset, which is the evaluation set for E0b
#
# Edit the config block below, then:   ./hpc/submit_e0a_disagreement.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_TYPE="${MODEL_TYPE:-qwen}"                        # qwen | flamingo | salmonn
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"       # pick an idle GPU: ssh node1 nvidia-smi

# Which Chapter 1 checkpoint to probe. Set UNTRAINED=true for the pre-trained baseline.
CKPT_DATE="${CKPT_DATE:-2026-08-23}"
TRAIN_RUN="${TRAIN_RUN:-003002_qa_h-cr-m14en_dpi_ha}"   # DG-ICL-Sym, qwen
EPOCH="${EPOCH:-4}"
UNTRAINED="${UNTRAINED:-false}"

# MELD is the signal: conversational, never trained on, transcripts carry real
# emotion. CREMA-D is the control: its sentences are emotionally neutral, so
# text-only should collapse toward chance.
DATASET_TYPE="${DATASET_TYPE:-meld_emotion-cremad}"
SPLIT="${SPLIT:-test}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
SEED="${SEED:-1}"

TAG="${TAG:-sym}"                                       # short name for this checkpoint
E0A_DIR="${E0A_DIR:-${HOME}/training/e0a}"
# ─────────────────────────────────────────────────────────────────────────────

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_SPEECH="e0a_${TAG}_speech_${STAMP}"
RUN_TEXT="e0a_${TAG}_text_${STAMP}"
LOG_DIR="${E0A_DIR}/logs"
LOG_FILE="${LOG_DIR}/e0a_${TAG}_${STAMP}.log"
mkdir -p "${LOG_DIR}" "${E0A_DIR}/metrics"

# Background on first entry so the terminal is free, same pattern as the other scripts.
if [[ "${_E0A_LAUNCHED:-0}" != "1" ]]; then
    export _E0A_LAUNCHED=1
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    printf 'E0a launched in background (PID: %s)\n' "$!"
    printf 'Follow: tail -f "%s"\n' "${LOG_FILE}"
    exit 0
fi

# Resolve the checkpoint here and pass it explicitly. The inner script contains a
# block of preset CKPT_DATE/TRAIN_RUN/EPOCH lines; if one is left uncommented it
# overrides exported values. Passing CHECKPOINT_PATH bypasses that entirely.
if [[ "${UNTRAINED}" == "true" ]]; then
    CHECKPOINT_PATH=""
else
    CHECKPOINT_PATH="$(ls "${HOME}/training/symbol_training/checkpoints/${CKPT_DATE}/${TRAIN_RUN}/lora_epoch${EPOCH}"*.pt 2>/dev/null | head -1)"
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        printf 'ERROR: no checkpoint at %s/%s/lora_epoch%s*.pt\n' "${CKPT_DATE}" "${TRAIN_RUN}" "${EPOCH}"
        exit 1
    fi
fi

printf '%s\n' "============================================================"
printf '%s\n' "E0a disagreement probe"
printf '%s\n' "Model:      ${MODEL_TYPE}"
printf '%s\n' "Checkpoint: ${CHECKPOINT_PATH:-<untrained base model>}"
printf '%s\n' "Datasets:   ${DATASET_TYPE}   split=${SPLIT}  max=${MAX_VAL_SAMPLES}"
printf '%s\n' "GPU:        ${CUDA_VISIBLE_DEVICES}"
printf '%s\n' "============================================================"

# _NOHUP_LAUNCHED=1 makes the inner script run in the foreground instead of
# backgrounding itself, so the two modes run sequentially on one GPU.
run_one_mode () {
    local mode="$1" run_name="$2"
    printf '\n>>> %s  (input_mode=%s)\n' "${run_name}" "${mode}"
    _NOHUP_LAUNCHED=1 \
    RUN_NAME="${run_name}" \
    INPUT_MODE="${mode}" \
    MODEL_TYPE="${MODEL_TYPE}" \
    DATASET_TYPE="${DATASET_TYPE}" \
    SPLIT="${SPLIT}" \
    MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    VALIDATION_MODES="original" \
    NUM_EXAMPLES="0" \
    NO_LEGEND="false" \
    CKPT_DATE="${CKPT_DATE}" \
    TRAIN_RUN="${TRAIN_RUN}" \
    EPOCH="${EPOCH}" \
    UNTRAINED="${UNTRAINED}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    SEED="${SEED}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    METRICS_BASE="${E0A_DIR}/metrics" \
    OUTPUT_DIR="${E0A_DIR}" \
        bash "${SCRIPT_DIR}/submit_symbol_inference_node1.sh"
}

run_one_mode "speech_only" "${RUN_SPEECH}"
run_one_mode "text_only"   "${RUN_TEXT}"

# The inner script writes to ${METRICS_BASE}/<date>/<run_name>/<run_name>_p.json.
# Locate by name rather than rebuilding the path, so a midnight rollover between
# the two runs cannot break this.
find_preds () {
    find "${E0A_DIR}/metrics" -name "$1_p.json" -type f 2>/dev/null | head -1
}
PRED_SPEECH="$(find_preds "${RUN_SPEECH}")"
PRED_TEXT="$(find_preds "${RUN_TEXT}")"

if [[ -z "${PRED_SPEECH}" || -z "${PRED_TEXT}" ]]; then
    printf '\nERROR: prediction dumps not found.\n'
    printf '  speech: %s\n' "${PRED_SPEECH:-<missing>}"
    printf '  text:   %s\n' "${PRED_TEXT:-<missing>}"
    printf 'Check the inference logs above for a failure.\n'
    exit 1
fi

printf '\n============================================================\n'
printf 'E0a analysis\n'
printf '============================================================\n'
python experiments/e0a_disagreement.py \
    --speech_preds "${PRED_SPEECH}" \
    --text_preds "${PRED_TEXT}" \
    --mode original \
    --out "${E0A_DIR}/disagreement_${TAG}_${STAMP}.json"

printf '\nDone. Subset: %s\n' "${E0A_DIR}/disagreement_${TAG}_${STAMP}.json"
printf 'Log:         %s\n' "${LOG_FILE}"
