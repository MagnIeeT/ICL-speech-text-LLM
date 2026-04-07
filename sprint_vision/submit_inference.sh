#!/bin/bash
# ============================================================
# LLaVA General Inference Script
# Strategy, dataset, and paths are set in vision_orchestrator.py.
# Edit ONLY the job settings below (num_samples, hostname, etc.).
# ============================================================

# ========================================
# Configuration - Edit these values
# ========================================
model_type="llava-v1.5-13b"

num_samples=0                # Number of samples to infer (0 = ALL) — passed to orchestrator
batch_size=1                    # Inference batch size
gradient_accumulation_steps=8   # Kept for template consistency

hostname="n6"
cuda_device=2

# Paths — override via env vars or edit here
LLAVA_DIR="${LLAVA_DIR:-/home/harinis/LLaVA}"
output_dir="${LLAVA_DIR}/logs"
orchestrator_path="${LLAVA_DIR}/sprint_vision/vision_orchestrator.py"
hold_job_id=""

# ========================================
# Auto Setup & Logging Validation
# ========================================
eval "$(conda shell.bash hook)"
conda activate llava

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:$hold_job_id"
else
    HOLD_FLAG=""
fi

# Dynamic Run Naming
CURRENT_DATETIME=$(date +"%d%m_%H%M")
TODAY=$(date +"%Y-%m-%d")

RUN_NAME="${CURRENT_DATETIME}_infer_${model_type}_samples${num_samples}"
LOG_DIR="${output_dir}/${TODAY}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "${LOG_FILE}"

echo "=========================================="
echo "LLaVA Inference Configuration"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Model:       ${model_type}"
echo "Samples:     ${num_samples} (0 = ALL)"
echo "Batch Size:  ${batch_size}"
echo "Hostname:    ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Log File:    ${LOG_FILE}"
echo "NOTE: Strategy/dataset/paths are set in vision_orchestrator.py"
echo "=========================================="

# ========================================
# Write job script to temp file
# ========================================
TMPJOB=$(mktemp /tmp/llava_infer_XXXX.sh)

cat << EOF > ${TMPJOB}
#!/bin/bash
set -e

# Real-time unbuffered logging setup
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=${cuda_device}
exec > >(stdbuf -oL tee -a ${LOG_FILE}) 2>&1

echo "=========================================="
echo "Job started at: \$(date)"
echo "Running on host: \$(hostname)"
echo "CUDA devices: ${cuda_device}"
echo "=========================================="

# Environment Activation
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
    --num-samples "${num_samples}"

echo "=========================================="
echo "Job completed at: \$(date)"
echo "=========================================="
EOF

chmod +x ${TMPJOB}

# ========================================
# Submit Job via qsub
# ========================================
qsub -q workq \
    $HOLD_FLAG \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=${hostname} \
    -l walltime=24:00:00 \
    -o /dev/null \
    -j oe \
    -v num_samples=${num_samples},cuda_device=${cuda_device},LOG_FILE=${LOG_FILE} \
    -S /bin/bash \
    ${TMPJOB}

echo ""
echo "=========================================="
echo "Job Submitted!"
echo "Monitor with:"
echo "  tail -f ${LOG_FILE}"
echo "  qstat | grep harinis"
echo "=========================================="
