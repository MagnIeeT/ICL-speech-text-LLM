#!/bin/bash
# ============================================================
# Submit inference jobs for all 3 datasets: colon, chest, endo
#
# Uses ONE checkpoint (trained on one dataset) and runs inference
# on all 3 datasets — for cross-dataset generalization testing.
#
# Usage:
#   bash submit_all_inference_jobs.sh               # defaults (regular, 0-shot)
#   bash submit_all_inference_jobs.sh --dry-run     # print commands only, no qsub
#
#   CHECKPOINT_PATH=/path/to/ckpt bash submit_all_inference_jobs.sh
#   strategy=two_token icl_shots=5 bash submit_all_inference_jobs.sh
#
# To run all shot counts:
#   for SHOTS in 0 1 5; do
#       icl_shots=$SHOTS bash submit_all_inference_jobs.sh
#   done
# ============================================================

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ============================================================
# Configuration — edit these defaults
# ============================================================
strategy="${strategy:-regular}"        # regular | two_token
icl_shots="${icl_shots:-0}"            # 0 | 1 | 5
model_type="${model_type:-llava-v1.5-13b}"
num_samples="${num_samples:-0}"        # 0 = ALL

LLAVA_DIR="${LLAVA_DIR:-/home/harinisrireddykandula/LLaVA}"
SPRINT_DIR="${LLAVA_DIR}/sprint_vision"
SUBMIT="${SPRINT_DIR}/submit_inference.sh"

# Single checkpoint used for ALL 3 datasets (train on one, infer on all)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/harinisrireddykandula/llava/checkpoints/llava-chest-regular-shot10_exp1}"

# Datasets to run (order matters for node/CUDA assignment below)
DATASETS=(colon chest endo)

# Node/GPU assignment — one entry per dataset in DATASETS order.
# Adjust to match available cluster resources (check: qstat -n | grep harinisrireddykandula).
HOSTNAMES=(n6    n6    n6  )
CUDA_DEVS=(0     1     2   )

# ============================================================
# Submit loop
# ============================================================
SUBMITTED=()

echo "=========================================="
echo "LLaVA All-Dataset Inference Submit"
echo "=========================================="
echo "Strategy   : ${strategy}"
echo "ICL Shots  : ${icl_shots}"
echo "Checkpoint : ${CHECKPOINT_PATH}"
echo "Datasets   : ${DATASETS[*]}"
echo "=========================================="
echo ""

for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    NODE="${HOSTNAMES[$i]}"
    CUDA="${CUDA_DEVS[$i]}"

    CMD="hostname=${NODE} \
cuda_device=${CUDA} \
dataset=${DS} \
strategy=${strategy} \
icl_shots=${icl_shots} \
num_samples=${num_samples} \
model_type=${model_type} \
CHECKPOINT_PATH=${CHECKPOINT_PATH} \
bash ${SUBMIT}"

    echo "Job $((i + 1))/${#DATASETS[@]} : ${DS}  strategy=${strategy}  icl_shots=${icl_shots}  [${NODE}:gpu${CUDA}]"
    echo "  Checkpoint: ${CHECKPOINT_PATH}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  DRY-RUN cmd:"
        echo "    ${CMD}"
        echo ""
    else
        JOB_ID=$(eval "${CMD} --print-id" 2>&1 | grep -E "^[0-9]+\." | head -1 || true)
        if [ -n "${JOB_ID}" ]; then
            SUBMITTED+=("${JOB_ID}  ${DS}_${strategy}_icl${icl_shots}")
            echo "  Submitted: ${JOB_ID}"
        else
            echo "  WARNING: could not capture job ID (check output above)"
            SUBMITTED+=("UNKNOWN  ${DS}_${strategy}_icl${icl_shots}")
        fi
        echo ""
    fi
done

# ============================================================
# Summary
# ============================================================
if [ "${DRY_RUN}" -eq 0 ]; then
    echo "=========================================="
    echo "All ${#SUBMITTED[@]} jobs submitted:"
    for ENTRY in "${SUBMITTED[@]}"; do
        echo "  ${ENTRY}"
    done
    echo ""
    echo "Monitor with:"
    echo "  qstat | grep harinisrireddykandula"
    echo "  watch -n 30 'qstat | grep harinisrireddykandula'"
    echo ""
    echo "Logs in: /home/harinisrireddykandula/llava/logs//$(date +%Y-%m-%d)/"
    echo "=========================================="
fi
