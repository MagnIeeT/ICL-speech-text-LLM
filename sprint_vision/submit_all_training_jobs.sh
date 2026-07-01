#!/bin/bash
# ============================================================
# Submit all 20 10-shot training jobs for the SPRInT VLM paper.
#
# Datasets : colon, chest
# Strategies: regular, two_token
# Exps     : 1–5  (MedFMC official repeated-experiment splits)
# = 2 datasets × 2 strategies × 5 exps = 20 jobs
#
# Pre-requisites (run these first if not done):
#   1. SCP run_sprint_finetune.sh from local to cluster
#   2. Generate colon exp4/exp5 JSONs:
#       cd /home/harinisrireddykandula/LLaVA
#       for EXP in 4 5; do
#         python sprint_vision/data/medfmc_to_llava.py \
#           --medfmc_root /home/harinisrireddykandula/MedFM/data/MedFMC \
#           --output_dir  sprint_vision/data \
#           --tasks colon --shot 10 --exp ${EXP}
#       done
#   3. Verify chest exp1–5 JSONs exist:
#       ls /home/harinisrireddykandula/LLaVA/sprint_vision/data/chest_train_shot10_exp*.json
#
# Usage:
#   bash submit_all_training_jobs.sh
#   bash submit_all_training_jobs.sh --dry-run   # print commands only, no qsub
# ============================================================

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

LLAVA_DIR="${LLAVA_DIR:-/home/harinisrireddykandula/LLaVA}"
SPRINT_DIR="${LLAVA_DIR}/sprint_vision"
SUBMIT="${SPRINT_DIR}/submit_training.sh"

# Node assignment — distribute 20 jobs across two nodes (10 each).
# Adjust if n10/n11 are unavailable; check with: qstat -n | grep harinisrireddykandula
NODES=(n10 n10 n10 n10 n10 n11 n11 n11 n11 n11 n10 n10 n10 n10 n10 n11 n11 n11 n11 n11)
CUDA_DEVS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

# ── Verify prerequisite JSON files ───────────────────────────────────────────
echo "Checking prerequisite JSON files..."
MISSING=0
for DATASET in colon chest; do
    for EXP in 1 2 3 4 5; do
        JSON="${SPRINT_DIR}/data/${DATASET}_train_shot10_exp${EXP}.json"
        if [ ! -f "${JSON}" ]; then
            echo "  MISSING: ${JSON}"
            MISSING=$((MISSING + 1))
        else
            COUNT=$(wc -l < "${JSON}" || echo "?")
            echo "  OK     : ${JSON}  (~${COUNT} lines)"
        fi
    done
done

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "ERROR: ${MISSING} required JSON file(s) missing."
    echo "Generate them first (see header comment)."
    exit 1
fi
echo "All 10 JSON files present."
echo ""

# ── Submit loop ───────────────────────────────────────────────────────────────
JOB_IDX=0
SUBMITTED=()

for DATASET in colon chest; do
    for STRATEGY in regular two_token; do
        for EXP in 1 2 3 4 5; do
            NODE="${NODES[$JOB_IDX]}"
            CUDA="${CUDA_DEVS[$JOB_IDX]}"

            CMD="hostname=${NODE} cuda_device=${CUDA} \
DATASET=${DATASET} \
STRATEGY=${STRATEGY} \
TRAINING_SHOTS=10 \
SHOT_EXP=${EXP} \
NUM_TRAIN_EPOCHS=5 \
bash ${SUBMIT}"

            echo "Job $((JOB_IDX + 1))/20 : ${DATASET} ${STRATEGY} exp${EXP}  [${NODE}]"
            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "  DRY-RUN: ${CMD}"
                echo ""
            else
                JOB_ID=$(eval "${CMD} --print-id" 2>&1 | grep -E "^[0-9]+\." | head -1 || true)
                if [ -n "${JOB_ID}" ]; then
                    SUBMITTED+=("${JOB_ID}  ${DATASET}_${STRATEGY}_exp${EXP}")
                    echo "  Submitted: ${JOB_ID}"
                else
                    echo "  WARNING: could not capture job ID (check output above)"
                    SUBMITTED+=("UNKNOWN  ${DATASET}_${STRATEGY}_exp${EXP}")
                fi
                echo ""
            fi

            JOB_IDX=$((JOB_IDX + 1))
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
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
    echo "Checkpoints in: //home/harinisrireddykandula/llava/checkpoints"
    echo "=========================================="
fi
