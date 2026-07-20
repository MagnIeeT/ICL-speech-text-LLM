#!/usr/bin/env bash
# Run inference sequentially on all complete checkpoint runs from 2026-07-14.
# Calls python inference.py directly (no nohup) so jobs run one at a time.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 nohup ./hpc/run_inference_all_checkpoints.sh > ~/batch_inference.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 RUNS="155314_af_h_nosym 162702_af_h_fix_ha" ./hpc/run_inference_all_checkpoints.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_BASE="${HOME}/training/symbol_training/checkpoints/2026-07-14"

ALL_RUNS="155314_af_h_nosym 155523_af_h_nosym 160053_af_h_nosym 160847_af_h_nosym 161154_af_h_nosym 162702_af_h_fix_ha 172937_af_h_fix_ha"
RUNS="${RUNS:-${ALL_RUNS}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DATASET_TYPE="${DATASET_TYPE:-hvb-voxpopuli-cremad-ravdess_song}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-500}"
SPLIT="${SPLIT:-validation}"
VALIDATION_MODES="${VALIDATION_MODES:-original}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LOG_DIR="${HOME}/training/symbol_training/logs_inference/$(date +"%Y-%m-%d")"
METRICS_DIR="${HOME}/training/symbol_training/metrics/$(date +"%Y-%m-%d")"
OUTPUT_DIR="${HOME}/training/symbol_training"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${OUTPUT_DIR}"

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

total=0
for run in ${RUNS}; do
    for ckpt in "${CHECKPOINT_BASE}/${run}"/lora_epoch[0-9]*.pt; do
        [[ -f "${ckpt}" ]] && total=$((total + 1))
    done
done
echo "============================================================"
echo "Batch inference: ${total} checkpoints across $(echo ${RUNS} | wc -w) runs"
echo "GPU: ${CUDA_VISIBLE_DEVICES}  Split: ${SPLIT}  Samples: ${MAX_VAL_SAMPLES}"
echo "============================================================"

count=0
for run in ${RUNS}; do
    for ckpt in $(ls "${CHECKPOINT_BASE}/${run}"/lora_epoch[0-9]*.pt 2>/dev/null | sort -V); do
        count=$((count + 1))

        # Build run name: timestamp_i_af_<datasets>_<samples>_tr<run>_ep<N>
        if [[ "${ckpt}" =~ epoch([0-9]+) ]]; then EPOCH_NUM="${BASH_REMATCH[1]}"; else EPOCH_NUM="X"; fi
        RUN_NAME="$(date +"%H%M%S")_i_af_${SPLIT}_${MAX_VAL_SAMPLES}_tr${run}_ep${EPOCH_NUM}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

        echo ""
        echo "[${count}/${total}] ${run} / epoch ${EPOCH_NUM} → ${LOG_FILE}"

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

        echo "    done — $(grep '📊' ${LOG_FILE} | tail -4)"
    done
done

echo ""
echo "============================================================"
echo "All ${total} checkpoints evaluated. Metrics in ${METRICS_DIR}"
echo "============================================================"
