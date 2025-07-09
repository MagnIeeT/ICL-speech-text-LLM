#!/bin/bash -l
#SBATCH --job-name=inference_orchestrator
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/%x_%j.out
#SBATCH --error=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/%x_%j.err
#SBATCH --export=ALL

# ========================================
# Configuration - Edit these values
# ========================================

# HVB - VoxCeleb 

# Regular FT Similarity
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0407_0329_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch2_periodic.pt"
# Regular FT Random
checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0407_0341_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch2_periodic.pt"
# Symbol FT Similarity
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0407_2300_orchestrator_bypass_mlp_sym_1c_10le_1me_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch2_periodic.pt"
# Symbol FT Random
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0607_2216_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch2_periodic.pt"

# HVB - Meld-Emotion
# Regular FT Random
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0607_2157_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_salmonn_hvb_meld_emotion/lora_step0_cycle0_epoch2_periodic.pt"
# Regular FT Similarity
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0507_2024_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_salmonn_hvb_meld_emotion/lora_step0_cycle0_epoch2_periodic.pt"

dataset_type="hvb"
max_val_samples=0
device="cuda:0"
output_dir="/home/sriramg/aneeraj/storage/results/model_ICL"

# ========================== Conda Setup ==========================
export CONDA_ENV="salmon"
echo "Set conda environment to: $CONDA_ENV"
source /home/sriramg/aneeraj/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate "$CONDA_ENV"
echo "Activated conda environment: $CONDA_ENV"

# ===== CUDA Info (Optional Debug) =====
export CUDA_HOME=$HOME/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "Checking CUDA and NVIDIA driver versions..."
nvcc --version
nvidia-smi

# ========================================
# Validation
# ========================================
if [ ! -f "$checkpoint_path" ]; then
    echo "❌ ERROR: Checkpoint not found at $checkpoint_path"
    exit 1
fi

SCRIPT_PATH="/home/sriramg/aneeraj/code/ICL-speech-text-LLM/models/symbolAdapter/orchestrator_inference.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ ERROR: Inference script not found at $SCRIPT_PATH"
    exit 1
fi

# ========================================
# Metadata and Logging
# ========================================
CLEAN_DATASET_TYPE=$(echo "$dataset_type" | tr ',' '-' | tr -d ' ')
CURRENT_DATETIME=$(date +"%d%m_%H%M")
CHECKPOINT_NAME=$(basename "$checkpoint_path" | sed 's/\.pt$//' | sed 's/\.pth$//')
RUN_NAME="${CURRENT_DATETIME}_inference_${CHECKPOINT_NAME}_${CLEAN_DATASET_TYPE}"

TODAY=$(date +"%Y-%m-%d")
LOG_DIR="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/${TODAY}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
rm -f "$LOG_PATH"

# ========================================
# Display Configuration
# ========================================
echo "=========================================="
echo "Orchestrator Inference Job Configuration"
echo "=========================================="
echo "Run Name:        ${RUN_NAME}"
echo "Checkpoint:      ${checkpoint_path}"
echo "Dataset:         ${dataset_type}"
echo "Device:          ${device}"
echo "Max Samples:     ${max_val_samples}"
echo "Script Path:     ${SCRIPT_PATH}"
echo "Output Directory:${output_dir}"
echo "Log File:        ${LOG_PATH}"
echo "=========================================="

# ========================================
# Run Inference
# ========================================
echo "Running Orchestrator Inference..."
echo "Started at: $(date)"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

python "${SCRIPT_PATH}" \
    --checkpoint_path "${checkpoint_path}" \
    --dataset_type "${dataset_type}" \
    --device "${device}" \
    --max_val_samples ${max_val_samples} \
    --output_dir "${output_dir}" > "${LOG_PATH}" 2>&1

EXIT_CODE=$?

echo "=========================================="
echo "Inference job completed at: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Orchestrator inference completed successfully."
else
    echo "❌ Inference failed. Check log: ${LOG_PATH}"
fi

exit $EXIT_CODE
