#!/bin/bash
# ============================================================
# LLaVA General Inference Script
# ============================================================

# ========================================
# Configuration - Edit these values
# ========================================
model_type="llava-v1.5-13b"
checkpoint_path="liuhaotian/llava-v1.5-13b" 
dataset_type="colon"              

num_samples=10               # Number of samples to infer (0 = ALL)
batch_size=1                    # Inference batch size
gradient_accumulation_steps=8   # Kept for template consistency

hostname="n6"
cuda_device=2
output_dir="/home/harinis/LLaVA/logs"
script_path="/home/harinis/LLaVA/sprint_vision/sprint_eval.py" 
hold_job_id=""

# --- SPRInT MODIFICATION: ADDED NECESSARY PATHS ---
strategy="regular"               # "regular" or "two_token"
image_folder="/home/harinis/MedFM/data/MedFMC"
question_file="/home/harinis/LLaVA/sprint_vision/data/colon_test.json"
# --------------------------------------------------

# ========================================
# Auto Setup & Logging Validation
# ========================================
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate llava

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:$hold_job_id"
else
    HOLD_FLAG=""
fi

# Dynamic Run Naming
CURRENT_DATETIME=$(date +"%d%m_%H%M")
TODAY=$(date +"%Y-%m-%d")
CHECKPOINT_NAME=$(basename "$checkpoint_path")
CLEAN_DATASET=$(echo $dataset_type | tr '_' '-')

RUN_NAME="${CURRENT_DATETIME}_infer_${model_type}_${CHECKPOINT_NAME}_samples${num_samples}"
LOG_DIR="${output_dir}/${TODAY}"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "${LOG_FILE}"

echo "=========================================="
echo "LLaVA Inference Configuration"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Model:       ${model_type}"
echo "Dataset:     ${dataset_type}"
echo "Samples:     ${num_samples} (0 = ALL)"
echo "Batch Size:  ${batch_size}"
echo "Grad Accum:  ${gradient_accumulation_steps}"
echo "Hostname:    ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

# ========================================
# Write job script to temp file
# ========================================
TMPJOB=$(mktemp /tmp/llava_infer_XXXX.sh)

# FIX: Use EOF without quotes to allow variable expansion in the temp file
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
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate llava
cd /home/harinis/LLaVA
echo "Working directory: \$(pwd)"

echo "=========================================="
echo "GPU Status:"
echo "=========================================="
nvidia-smi

echo "=========================================="
echo "Running LLaVA Inference Python Script..."
echo "=========================================="

# stdbuf ensures output is written to the log immediately, not held in memory
# MODIFIED: Passing specific arguments to sprint_eval.py
stdbuf -oL -eL python -u "${script_path}" \\
    --model-path "${checkpoint_path}" \\
    --image-folder "${image_folder}" \\
    --question-file "${question_file}" \\
    --strategy "${strategy}"

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