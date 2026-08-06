#!/usr/bin/env bash
# Run inference for all epoch checkpoints of ONE run, then print a summary table.
#
# Usage:
#   RUN=155314_af_h_nosym CUDA_VISIBLE_DEVICES=2 ./hpc/run_inference_all_checkpoints.sh
#   (self-nohups into the background; SPLIT defaults to validation)
#
# ── VALIDATION re-runs needed (7 runs: 5 nosym + 2 fix_ha; all under 2026-07-14) ──
# Uncomment/copy one line at a time, each on a free GPU, to run in parallel:
#
#   RUN=155314_af_h_nosym  CUDA_VISIBLE_DEVICES=1 ./hpc/run_inference_all_checkpoints.sh
#   RUN=155523_af_h_nosym  CUDA_VISIBLE_DEVICES=2 ./hpc/run_inference_all_checkpoints.sh
#   RUN=160053_af_h_nosym  CUDA_VISIBLE_DEVICES=3 ./hpc/run_inference_all_checkpoints.sh
#   RUN=160847_af_h_nosym  CUDA_VISIBLE_DEVICES=4 ./hpc/run_inference_all_checkpoints.sh
#   RUN=161154_af_h_nosym  CUDA_VISIBLE_DEVICES=5 ./hpc/run_inference_all_checkpoints.sh
#   RUN=162702_af_h_fix_ha CUDA_VISIBLE_DEVICES=6 ./hpc/run_inference_all_checkpoints.sh
#   RUN=172937_af_h_fix_ha CUDA_VISIBLE_DEVICES=7 ./hpc/run_inference_all_checkpoints.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN="${RUN:-172937_af_h_fix_ha}"
CHECKPOINT_BASE="${HOME}/training/symbol_training/checkpoints/2026-07-14/${RUN}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
DATASET_TYPE="${DATASET_TYPE:-hvb-voxpopuli-cremad-ravdess_song}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500}"
SPLIT="${SPLIT:-validation}"
VALIDATION_MODES="${VALIDATION_MODES:-original,fixed}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"

SHORT_DATASET_TYPE="${DATASET_TYPE//voxceleb/vb}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//hvb/h}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//meld_emotion/me}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//voxpopuli/vp}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//cremad/cr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//ravdess_song/rs}"

TRAIN_RUN_PREFIX=$(echo "${RUN}" | cut -d'_' -f1)
TRAIN_DATA_RAW=$(echo "${RUN}" | cut -d'_' -f3)
TRAIN_DATA="${TRAIN_DATA_RAW//voxceleb/vb}"
TRAIN_DATA="${TRAIN_DATA//hvb/h}"
TRAIN_DATA="${TRAIN_DATA//meld_emotion/me}"
TRAIN_DATA="${TRAIN_DATA//voxpopuli/vp}"
[[ -z "${TRAIN_DATA}" ]] && TRAIN_DATA="unk"

[[ "${MAX_VAL_SAMPLES}" == "0" ]] && SAMPLES_TAG="" || SAMPLES_TAG="_${MAX_VAL_SAMPLES}"
_SPLIT_TAG="${SPLIT:0:3}"

TODAY="$(date +"%Y-%m-%d")"
LOG_DIR="${HOME}/training/symbol_training/logs_inference/${TODAY}"
METRICS_DIR="${HOME}/training/symbol_training/metrics/${TODAY}"
OUTPUT_DIR="${HOME}/training/symbol_training"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${OUTPUT_DIR}"

if [[ "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    OUTER_LOG="${LOG_DIR}/${RUN}_batch_infer.log"
    export _NOHUP_LAUNCHED=1 RUN
    nohup "$0" >> "${OUTER_LOG}" 2>&1 &
    printf 'Batch inference launched in background (PID: %s)\n' "$!"
    printf 'Follow logs: tail -f "%s"\n' "${OUTER_LOG}"
    exit 0
fi

# Activate conda
if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate flamingo
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

checkpoints=( $(ls "${CHECKPOINT_BASE}"/lora_epoch[0-9]*.pt 2>/dev/null | sort -V) )
total=${#checkpoints[@]}

echo "============================================================"
echo "Run:     ${RUN}"
echo "GPU:     ${CUDA_VISIBLE_DEVICES}  Split: ${SPLIT}  Samples/dataset: ${MAX_VAL_SAMPLES}"
echo "Epochs:  ${total}"
echo "============================================================"

# Track metric files and epoch numbers for the summary table
MANIFEST_FILE="${METRICS_DIR}/${RUN}_manifest.txt"
> "${MANIFEST_FILE}"   # clear/create

count=0
for ckpt in "${checkpoints[@]}"; do
    count=$((count + 1))
    if [[ "${ckpt}" =~ epoch([0-9]+) ]]; then EPOCH_NUM="${BASH_REMATCH[1]}"; else EPOCH_NUM="${count}"; fi

    RUN_NAME="$(date +"%H%M%S")_i_af_${SHORT_DATASET_TYPE}_${_SPLIT_TAG}${SAMPLES_TAG}_tr${TRAIN_RUN_PREFIX}${TRAIN_DATA}_ep${EPOCH_NUM}_sh${NUM_EXAMPLES}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    METRIC_FILE="${METRICS_DIR}/${RUN_NAME}/${RUN_NAME}_m.json"

    echo ""
    echo "[${count}/${total}] epoch ${EPOCH_NUM} → ${LOG_FILE}"

    python inference.py \
        --model_type flamingo \
        --dataset_type "${DATASET_TYPE}" \
        --device cuda:0 \
        --checkpoint_path "${ckpt}" \
        --max_val_samples "${MAX_VAL_SAMPLES}" \
        --num_examples 0 \
        --num_workers "${NUM_WORKERS}" \
        --val_batch_size "${VAL_BATCH_SIZE}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "${RUN_NAME}" \
        --validation_modes "${VALIDATION_MODES}" \
        --split "${SPLIT}" \
        --metrics_dir "${METRICS_DIR}" \
        >> "${LOG_FILE}" 2>&1

    # Record epoch → metric file mapping
    echo "${EPOCH_NUM} ${METRIC_FILE}" >> "${MANIFEST_FILE}"
    echo "    done"
done

# ── Summary table ─────────────────────────────────────────────────────────────
SUMMARY_FILE="${METRICS_DIR}/${RUN}_summary.txt"
echo ""
echo "============================================================"
echo "Building summary table → ${SUMMARY_FILE}"
echo "============================================================"

python3 <<PYEOF
import json, os

manifest_path = "${MANIFEST_FILE}"
summary_path  = "${SUMMARY_FILE}"
modes         = [m.strip() for m in "${VALIDATION_MODES}".split(",")]
datasets      = ["hvb", "voxpopuli", "cremad", "ravdess_song"]

entries = []
with open(manifest_path) as f:
    for line in f:
        ep, mfile = line.strip().split(" ", 1)
        entries.append((int(ep), mfile))
entries.sort(key=lambda x: x[0])

header  = f"{'Epoch':<8} | {'Mode':<12} | {'hvb':<16} | {'voxpopuli':<16} | {'cremad':<16} | {'ravdess_song':<16} | Avg"
divider = "-" * 112

rows = [header, divider]
for ep, mfile in entries:
    if not os.path.exists(mfile):
        rows.append(f"{ep:<8} | MISSING metrics file")
        rows.append(divider)
        continue
    with open(mfile) as f:
        m = json.load(f)
    for mode in modes:
        scores = [m.get(f"{mode}_{ds}", {}).get("macro_f1_with_invalid", 0.0) for ds in datasets]
        avg = sum(scores) / len(scores)
        row = f"{ep:<8} | {mode:<12} | " + " | ".join(f"{s:<16.6f}" for s in scores) + f" | {avg:.6f}"
        rows.append(row)
    rows.append(divider)

output = "\n".join(rows)
print(output)
with open(summary_path, "w") as f:
    f.write(output + "\n")
print(f"\nSaved → {summary_path}")
PYEOF
