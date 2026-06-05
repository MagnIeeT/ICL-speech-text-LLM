#!/bin/bash
set -e

# ============================================================
# LLaVA Inference Submit Script
# ============================================================
# Submits ONE job that runs all specified datasets sequentially
# into a single log file.
#
# Usage:
#   bash submit_inference.sh             (submit job)
#   bash submit_inference.sh --dry-run   (preview only)
#   bash submit_inference.sh --print-id  (submit + print job ID)
# ============================================================

# ========================================
# Configuration — edit these values
# ========================================
datasets=colon-chest-endo          # hyphen-separated: "chest" or "colon-chest-endo"
strategy=regular                 # regular | two_token
model_type="${model_type:-llava-v1.5-13b}"

num_samples=0              # 0 = ALL samples
icl_shots=0

hostname=n10
cuda_device=0

# Fine-tuned checkpoint (LoRA adapter dir) — same checkpoint used for all datasets
CHECKPOINT_PATH=/home/leapers/weights/harinis/llava/checkpoints/llava-chest-regular-shot10_exp2

# Set to a job ID (e.g. "12093.eehpc") to wait for that job first.
hold_job_id=

# ========================================
# Paths
# ========================================
LLAVA_DIR="${LLAVA_DIR:-/home/harinis/LLaVA}"
MODEL_BASE="${MODEL_BASE:-/home/harinis/.cache/huggingface/hub/llava-v1.5-13b}"
output_dir="${LLAVA_DIR}/logs"
orchestrator_path="${LLAVA_DIR}/sprint_vision/vision_orchestrator.py"

# convert hyphens to spaces for the loop inside the job script
datasets_loop=$(echo "${datasets}" | tr '-' ' ')

# ========================================
# Auto setup
# ========================================
CURRENT_DATETIME=$(date +"%d%m_%H%M")
TODAY=$(date +"%Y-%m-%d")
RUN_NAME="${CURRENT_DATETIME}_infer_${strategy}_${icl_shots}shot_${model_type}"

LOG_DIR="${output_dir}/${TODAY}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "${LOG_FILE}"

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:${hold_job_id}"
    HOLD_MSG="Waiting for job: ${hold_job_id}"
else
    HOLD_FLAG=""
    HOLD_MSG="Starting immediately (no dependency)"
fi

echo "=========================================="
echo "LLaVA Inference Configuration"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Datasets:    ${datasets}"
echo "Strategy:    ${strategy}"
echo "Model:       ${model_type}"
echo "Shots:       ${icl_shots}"
echo "Samples:     ${num_samples} (0 = ALL)"
echo "Hostname:    ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Checkpoint:  ${CHECKPOINT_PATH}"
echo "Dependency:  ${HOLD_MSG}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

if [[ "${1}" == "--dry-run" ]]; then
    echo "DRY-RUN — no job submitted"
    exit 0
fi

# ========================================
# Write job script
# ========================================
TMPJOB=$(mktemp /tmp/llava_infer_XXXX.sh)

cat << EOF > ${TMPJOB}
#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=${cuda_device}
exec > >(stdbuf -oL tee -a ${LOG_FILE}) 2>&1

echo "=========================================="
echo "Job started at: \$(date)"
echo "Running on host: \$(hostname)"
echo "Datasets: ${datasets}  Strategy: ${strategy}  Shots: ${icl_shots}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "=========================================="

eval "\$(conda shell.bash hook)"
conda activate llava
cd ${LLAVA_DIR}
echo "Working directory: \$(pwd)"

echo "=========================================="
echo "GPU Status:"
echo "=========================================="
nvidia-smi

stdbuf -oL -eL python -u "${orchestrator_path}" \\
    --mode inference \\
    --datasets "${datasets}" \\
    --strategy "${strategy}" \\
    --num-samples "${num_samples}" \\
    --icl-shots "${icl_shots}" \\
    --checkpoint-path "${CHECKPOINT_PATH}" \\
    --model-base "${MODEL_BASE}"
EOF

chmod +x ${TMPJOB}

# ========================================
# Submit and capture job ID
# ========================================
JOB_ID=$(qsub -q workq \
    $HOLD_FLAG \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=${hostname} \
    -l walltime=24:00:00 \
    -o /dev/null \
    -j oe \
    -S /bin/bash \
    ${TMPJOB})

echo ""
echo "=========================================="
echo "Job Submitted!"
echo "  Job ID:  ${JOB_ID}"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Status:  qstat | grep harinis"
echo ""
echo "To chain another job after this one:"
echo "  hold_job_id=${JOB_ID} bash submit_inference.sh"
echo "=========================================="

if [[ "${1}" == "--print-id" ]]; then
    echo "${JOB_ID}"
fi
