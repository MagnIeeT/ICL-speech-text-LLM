#!/bin/bash
set -e

# ============================================================
# LLaVA Inference Submit Script
# ============================================================
# HOW TO USE:
#
#   Single run:
#       dataset=colon strategy=regular icl_shots=0 bash submit_inference.sh
#       dataset=chest  strategy=regular icl_shots=0 bash submit_inference.sh
#       dataset=endo   strategy=regular icl_shots=0 bash submit_inference.sh
#
#   Chain 0/1/5-shot jobs (each waits for previous):
#       JOB1=$(dataset=colon icl_shots=0 bash submit_inference.sh --print-id)
#       JOB2=$(hold_job_id="$JOB1" dataset=colon icl_shots=1 bash submit_inference.sh --print-id)
#       JOB3=$(hold_job_id="$JOB2" dataset=colon icl_shots=5 bash submit_inference.sh --print-id)
#
#   Run multiple datasets:
#       for ds in colon chest endo; do
#           dataset=$ds icl_shots=0 bash submit_inference.sh
#       done
# ============================================================

# ========================================
# Configuration — edit these values
# ========================================
dataset=endo                       # colon | chest | endo
strategy=regular                  # regular | two_token
model_type="llava-v1.5-13b"

num_samples=0      # 0 = ALL samples
icl_shots=0

hostname="n10"
cuda_device=0

# Set to a job ID (e.g. "12345.cluster") to wait for that job first.
hold_job_id=11787.eehpc

# Paths
LLAVA_DIR="${LLAVA_DIR:-/home/harinis/LLaVA}"
output_dir="${LLAVA_DIR}/logs"
orchestrator_path="${LLAVA_DIR}/sprint_vision/vision_orchestrator.py"

# ========================================
# Auto setup
# ========================================
CURRENT_DATETIME=$(date +"%d%m_%H%M")
TODAY=$(date +"%Y-%m-%d")
RUN_NAME="${CURRENT_DATETIME}_infer_${dataset}_${strategy}_${icl_shots}shot_${model_type}"

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
echo "Dataset:     ${dataset}"
echo "Strategy:    ${strategy}"
echo "Model:       ${model_type}"
echo "Shots:       ${icl_shots}"
echo "Samples:     ${num_samples} (0 = ALL)"
echo "Hostname:    ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Dependency:  ${HOLD_MSG}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

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
echo "Dataset: ${dataset}  Strategy: ${strategy}  Shots: ${icl_shots}"
echo "=========================================="

eval "\$(conda shell.bash hook)"
conda activate llava
cd ${LLAVA_DIR}
echo "Working directory: \$(pwd)"

echo "=========================================="
echo "GPU Status:"
echo "=========================================="
nvidia-smi

echo "=========================================="
echo "Running LLaVA Inference via vision_orchestrator.py..."
echo "=========================================="

stdbuf -oL -eL python -u "${orchestrator_path}" \\
    --mode inference \\
    --dataset "${dataset}" \\
    --strategy "${strategy}" \\
    --num-samples "${num_samples}" \\
    --icl-shots "${icl_shots}"

echo "=========================================="
echo "Job completed at: \$(date)"
echo "=========================================="
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
echo "  hold_job_id=${JOB_ID} dataset=${dataset} icl_shots=5 bash submit_inference.sh"
echo "=========================================="

if [[ "${1}" == "--print-id" ]]; then
    echo "${JOB_ID}"
fi
