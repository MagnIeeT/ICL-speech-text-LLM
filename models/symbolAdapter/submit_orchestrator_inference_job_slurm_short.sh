#!/bin/bash -l
#SBATCH --job-name=inference_orchestrator
#SBATCH --partition=short
#SBATCH --nodelist=cn2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --gres=gpu:A5000:1
#SBATCH --output=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/slurm_logs/%x_%j.out
#SBATCH --error=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/slurm_logs/%x_%j.err
#SBATCH --export=ALL

# ========================================
# Configuration - Edit these values
# ========================================

# HVB - VoxCeleb 
# checkpoint_path="/home/sriramg/aneeraj/storage/salmonn_v1.pth"
# Regular FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1208_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-voxceleb/lora_step0_cycle0_epoch4_periodic.pt"
# Symbol FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1215_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# Random Regular FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1107_0900_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# Random Symbol FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1207_1132_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-voxceleb/lora_step0_cycle0_epoch1_periodic.pt"

# HVB - Meld-Emotion
# Regular FT 
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1211_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-meld_emotion/lora_step0_cycle0_epoch3_periodic.pt"
# Symbol FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1007_0908_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-meld_emotion/lora_step0_cycle0_epoch1_periodic.pt"
# Random Regular FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1107_0903_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-meld_emotion/lora_step0_cycle0_epoch1_periodic.pt"
# Random Symbol FT
checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1207_1134_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-meld_emotion/lora_step0_cycle0_epoch2_periodic.pt"

# Voxpopuli - Voxceleb  
# Regular FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1213_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxceleb-voxpopuli/lora_step0_cycle0_epoch4_periodic.pt"
# Symbol FT 
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1007_0234_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxceleb-voxpopuli/lora_step0_cycle0_epoch1_periodic.pt"
# Random Regular FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1207_0107_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxpopuli-voxceleb/lora_step0_cycle0_epoch2_periodic.pt"
# Random Symbol FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1207_1137_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxceleb-voxpopuli/lora_step0_cycle0_epoch2_periodic.pt"

# ALL DATASET TRAIN
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1007_1834_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxceleb-meld_emotion-voxpopuli-hvb/lora_step0_cycle0_epoch4_periodic.pt"

dataset_type="voxceleb-hvb-voxpopuli-meld_emotion"  # Comma-separated list of dataset types
max_val_samples=0
device="cuda:0"
output_dir="/home/sriramg/aneeraj/storage/results/model_ICL"
num_examples=0

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

GPU_LOG_PATH="${LOG_DIR}/${RUN_NAME}_gpu_usage.csv"
CPU_LOG_PATH="${LOG_DIR}/${RUN_NAME}_cpu_mem_usage.log"
FINAL_TOP_PATH="${LOG_DIR}/${RUN_NAME}_final_top.log"
FINAL_GPU_PATH="${LOG_DIR}/${RUN_NAME}_final_gpu.log"

# Start GPU usage logging
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
           --format=csv -l 5 > "$GPU_LOG_PATH" &
GPU_PID=$!

# Start CPU/mem logging
(
  echo "timestamp,CPU %,MEM %,Load Avg,Used RAM (MB),Free RAM (MB)"
  while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    CPU_USAGE=$(top -b -n1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    MEM_USAGE=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
    LOAD_AVG=$(uptime | awk -F'load average:' '{ print $2 }' | sed 's/^[ \t]*//')
    MEM_USED=$(free -m | awk '/Mem:/ {print $3}')
    MEM_FREE=$(free -m | awk '/Mem:/ {print $4}')
    echo "${TIMESTAMP},${CPU_USAGE},${MEM_USAGE},${LOAD_AVG},${MEM_USED},${MEM_FREE}"
    sleep 5
  done
) >> "$CPU_LOG_PATH" &
CPU_PID=$!

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
    --num_examples ${num_examples} \
    --output_dir "${output_dir}" > "${LOG_PATH}" 2>&1

EXIT_CODE=$?

# ========================================
# Final Logs and Cleanup
# ========================================
kill $GPU_PID
kill $CPU_PID
top -b -n1 -u $USER > "$FINAL_TOP_PATH"
nvidia-smi > "$FINAL_GPU_PATH"

echo "=========================================="
echo "Inference job completed at: $(date)"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Orchestrator inference completed successfully."
else
    echo "❌ Inference failed. Check log: ${LOG_PATH}"
fi

exit $EXIT_CODE
