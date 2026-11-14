#!/usr/bin/env bash
#
# E0b-1: does the definition decide which evidence the model uses?
#
# Four passes on the same items, then the routing-gap analysis:
#
#   1. speech_only + orig         -> P_speech, and the 'neutral' condition
#   2. text_only   + orig         -> P_text, what the words imply per the model
#   3. speech_only + route_tone   -> legend says the TONE decides
#   4. speech_only + route_words  -> legend says the WORDS decide
#
# Everything except pass 2 is speech_only: the audio already carries both
# channels, so the model must select between them itself and the legend is the
# only thing that differs between passes 3 and 4.
#
# Edit the config block below, then:   ./hpc/submit_e0b_routing.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

# ─── Config ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"       # idle GPU: ssh node1 nvidia-smi

# ─── Which checkpoint to probe: uncomment exactly ONE line ───────────────────
# The ${CONDITION:-...} form means the uncommented line sets the default while
# an exported CONDITION still wins, so a batch launch needs no file edit.
CONDITION="${CONDITION:-sym}"       # DG-ICL-Sym   (Qwen)
# CONDITION="${CONDITION:-it}"      # Instruction Tuning (Qwen)
# CONDITION="${CONDITION:-flip}"    # DG-ICL-Flip  (Qwen)
# CONDITION="${CONDITION:-base}"    # untrained Qwen2-Audio
# CONDITION="${CONDITION:-af_sym}"  # DG-ICL-Sym   (AF3)
# CONDITION="${CONDITION:-af_it}"   # Instruction Tuning (AF3)
# CONDITION="${CONDITION:-af_flip}" # DG-ICL-Flip  (AF3)
# CONDITION="${CONDITION:-af_base}" # untrained Audio Flamingo 3

case "${CONDITION}" in
    sym)     TAG=sym;     MODEL_TYPE=qwen;     CKPT_DATE=2026-08-23; TRAIN_RUN=003002_qa_h-cr-m14en_dpi_ha;                  EPOCH=4; UNTRAINED=false ;;
    it)      TAG=it;      MODEL_TYPE=qwen;     CKPT_DATE=2026-08-23; TRAIN_RUN=004932_qa_h-cr-m14en_nosym;                   EPOCH=3; UNTRAINED=false ;;
    flip)    TAG=flip;    MODEL_TYPE=qwen;     CKPT_DATE=2026-09-03; TRAIN_RUN=014804_qa_h-cr-m14en_nosym_swap_per_instance; EPOCH=4; UNTRAINED=false ;;
    base)    TAG=base;    MODEL_TYPE=qwen;     CKPT_DATE=;           TRAIN_RUN=;                                             EPOCH=;  UNTRAINED=true  ;;
    af_sym)  TAG=af_sym;  MODEL_TYPE=flamingo; CKPT_DATE=2026-08-02; TRAIN_RUN=112859_af_h-cr-m14en_dpi_ha;                  EPOCH=4; UNTRAINED=false ;;
    af_it)   TAG=af_it;   MODEL_TYPE=flamingo; CKPT_DATE=2026-08-02; TRAIN_RUN=112544_af_h-cr-m14en_nosym;                   EPOCH=4; UNTRAINED=false ;;
    af_flip) TAG=af_flip; MODEL_TYPE=flamingo; CKPT_DATE=2026-08-29; TRAIN_RUN=003524_af_h-cr-m14en_nosym_swap_per_instance; EPOCH=4; UNTRAINED=false ;;
    af_base) TAG=af_base; MODEL_TYPE=flamingo; CKPT_DATE=;           TRAIN_RUN=;                                             EPOCH=;  UNTRAINED=true  ;;
    *)       printf 'ERROR: unknown CONDITION=%s\n' "${CONDITION}"
             printf 'Expected: sym|it|flip|base|af_sym|af_it|af_flip|af_base\n'; exit 1 ;;
esac

# MELD is the measurement: held out from every checkpoint, conversational, so its
# transcripts carry real emotion. CREMA-D is the control and is IN the training
# mixture, and its sentences are emotionally neutral, so route_words there should
# degrade sharply if the model is really following the routing clause.
DATASET_TYPE="${DATASET_TYPE:-meld_emotion-cremad}"
ANALYSE_DATASET="${ANALYSE_DATASET:-meld_emotion}"
SPLIT="${SPLIT:-test}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
SEED="${SEED:-1}"
E0B_DIR="${E0B_DIR:-${HOME}/training/e0b}"
# ─────────────────────────────────────────────────────────────────────────────

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${E0B_DIR}/logs"
LOG_FILE="${LOG_DIR}/e0b_${TAG}_${STAMP}.log"
mkdir -p "${LOG_DIR}" "${E0B_DIR}/metrics"

if [[ "${_E0B_LAUNCHED:-0}" != "1" ]]; then
    export _E0B_LAUNCHED=1
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    printf 'E0b launched in background (PID: %s)\n' "$!"
    printf 'Follow: tail -f "%s"\n' "${LOG_FILE}"
    exit 0
fi

# Resolve the checkpoint here and pass it explicitly, so a stray preset in the
# inner script cannot redirect the run.
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
printf '%s\n' "E0b-1 routing probe"
printf '%s\n' "Condition:  ${CONDITION}  (tag=${TAG})"
printf '%s\n' "Model:      ${MODEL_TYPE}"
printf '%s\n' "Checkpoint: ${CHECKPOINT_PATH:-<untrained base model>}"
printf '%s\n' "Datasets:   ${DATASET_TYPE}   split=${SPLIT}  max=${MAX_VAL_SAMPLES}"
printf '%s\n' "Analyse:    ${ANALYSE_DATASET}"
printf '%s\n' "GPU:        ${CUDA_VISIBLE_DEVICES}"
printf '%s\n' "============================================================"

run_pass () {
    local run_name="$1" input_mode="$2" variant="$3"
    printf '\n>>> %s  (input_mode=%s, legend=%s)\n' "${run_name}" "${input_mode}" "${variant}"
    _NOHUP_LAUNCHED=1 \
    RUN_NAME="${run_name}" \
    INPUT_MODE="${input_mode}" \
    DEFINITION_VARIANT="${variant}" \
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
    METRICS_BASE="${E0B_DIR}/metrics" \
    OUTPUT_DIR="${E0B_DIR}" \
        bash "${SCRIPT_DIR}/submit_symbol_inference_node1.sh"
}

R_SP="e0b_${TAG}_sp_orig_${STAMP}"
R_TX="e0b_${TAG}_tx_orig_${STAMP}"
R_TONE="e0b_${TAG}_sp_tone_${STAMP}"
R_WORDS="e0b_${TAG}_sp_words_${STAMP}"

run_pass "${R_SP}"    "speech_only" "orig"
run_pass "${R_TX}"    "text_only"   "orig"
run_pass "${R_TONE}"  "speech_only" "route_tone"
run_pass "${R_WORDS}" "speech_only" "route_words"

find_preds () { find "${E0B_DIR}/metrics" -name "$1_p.json" -type f 2>/dev/null | head -1; }
P_SP="$(find_preds "${R_SP}")"
P_TX="$(find_preds "${R_TX}")"
P_TONE="$(find_preds "${R_TONE}")"
P_WORDS="$(find_preds "${R_WORDS}")"

for pair in "speech+orig:${P_SP}" "text+orig:${P_TX}" "route_tone:${P_TONE}" "route_words:${P_WORDS}"; do
    if [[ -z "${pair#*:}" ]]; then
        printf '\nERROR: missing prediction dump for %s. Check the inference logs above.\n' "${pair%%:*}"
        exit 1
    fi
done

printf '\n============================================================\n'
printf 'E0b-1 analysis\n'
printf '============================================================\n'
# The inner script activates conda in its own subshell, so resolve a python here.
PY_BIN="${HOME}/miniconda3/envs/${CONDA_ENV:-qwen}/bin/python"
[[ -x "${PY_BIN}" ]] || PY_BIN="$(command -v python3 || true)"
if [[ -z "${PY_BIN}" ]]; then
    printf 'ERROR: no python found. Dumps are intact; run experiments/e0b_routing.py by hand.\n'
    exit 1
fi

"${PY_BIN}" experiments/e0b_routing.py \
    --speech_orig "${P_SP}" \
    --text_orig "${P_TX}" \
    --speech_tone "${P_TONE}" \
    --speech_words "${P_WORDS}" \
    --dataset "${ANALYSE_DATASET}" \
    --out "${E0B_DIR}/routing_${TAG}_${ANALYSE_DATASET}_${STAMP}.json"

printf '\nDone. Detail: %s\n' "${E0B_DIR}/routing_${TAG}_${ANALYSE_DATASET}_${STAMP}.json"
printf 'Log:          %s\n' "${LOG_FILE}"
