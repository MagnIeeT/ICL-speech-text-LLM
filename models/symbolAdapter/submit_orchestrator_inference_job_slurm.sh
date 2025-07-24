#!/bin/bash
#SBATCH --job-name=inference_orchestrator
<<<<<<< HEAD
#SBATCH --partition=long
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/slurm_logs/%x_%j.out
#SBATCH --error=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_inference/logs/slurm_logs/%x_%j.err
=======
#SBATCH --chdir=/home/sriramg/chandnia
#SBATCH --output=/home/sriramg/chandnia/slurm_logs/qwen_inference/%x_%j.out
#SBATCH --error=/home/sriramg/chandnia/slurm_logs/qwen_inference/%x_%j.err
#SBATCH --partition=ada
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH --gres=gpu:1
>>>>>>> f1678dd (Updated models and data/model_processors.py)
#SBATCH --export=ALL

SLURM_LOG_DIR="/home/sriramg/chandnia/slurm_logs/qwen_inference"
mkdir -p "$SLURM_LOG_DIR"

# ========================================
# Configuration - Edit these values
# ========================================

# Regular FT - All Four
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1107_1951_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_voxceleb-hvb-meld_emotion-voxpopuli_reg_ft/lora_step0_cycle0_epoch9_periodic.pt"

# Dynamic per epoch symbol change
# hvb-vox
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1107_1420_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_sym_qwen_hvb-voxceleb/lora_step0_cycle0_epoch4_periodic.pt"
# hvb-meld_emotion
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1507_0122_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_sym_qwen_hvb-meld_emotion/lora_step0_cycle0_epoch4_periodic.pt"
# voxpopuli-vox
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1707_0641_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_sym_qwen_voxceleb-voxpopuli/lora_step0_cycle0_epoch6_periodic.pt"


# HVB - VoxCeleb 
# Regular FT
<<<<<<< HEAD
checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1208_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-voxceleb/lora_step0_cycle0_epoch4_periodic.pt"
# Symbol FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1215_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-voxceleb/lora_step0_cycle0_epoch1_periodic.pt"

# HVB - Meld-Emotion
# Regular FT 
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1211_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-meld_emotion/lora_step0_cycle0_epoch3_periodic.pt"
# Symbol FT
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1007_0908_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_hvb-meld_emotion/lora_step0_cycle0_epoch1_periodic.pt"

# Voxpopuli - Voxceleb
# Regular FT
checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/0907_1213_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxceleb-voxpopuli/lora_step0_cycle0_epoch4_periodic.pt"
# Symbol FT 
# checkpoint_path="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/checkpoints/1007_0234_orchestrator_bypass_mlp_sym_1c_5le_1me_bypass_mlp_org_salmonn_voxceleb-voxpopuli/lora_step0_cycle0_epoch1_periodic.pt"

dataset_type="voxceleb-hvb-voxpopuli-meld_emotion"  # Comma-separated list of dataset types
max_val_samples=0
device="cuda:0"
output_dir="/home/sriramg/aneeraj/storage/results/model_ICL"
num_examples=2
# ========================== Conda Setup ==========================
export CONDA_ENV="salmon"
=======
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/0807_1026_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-voxceleb_reg_ft/lora_step0_cycle0_epoch2_periodic.pt"
# Symbol FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/0807_1036_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-voxceleb_sym_ft/lora_step0_cycle0_epoch3_periodic.pt"
# Random Regular FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1207_0651_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-voxceleb/lora_step0_cycle0_epoch4_periodic.pt"
# Random Symbol FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1207_0653_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-voxceleb/lora_step0_cycle0_epoch5_periodic.pt"

# HVB - Meld-Emotion
# Regular FT 
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/0907_1739_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-meld_emotion_reg_ft/lora_step0_cycle0_epoch2_periodic.pt"
# Symbol FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/0907_1740_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-meld_emotion_sym_ft/lora_step0_cycle0_epoch4_periodic.pt"
# Random Regular FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1207_0654_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-meld_emotion/lora_step0_cycle0_epoch6_periodic.pt"
# Random Symbol FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1207_0655_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_hvb-meld_emotion/lora_step0_cycle0_epoch2_periodic.pt"

# Voxpopuli - Voxceleb
# Regular FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1107_1949_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_voxceleb-voxpopuli_reg_ft/lora_step0_cycle0_epoch10_periodic.pt"
# Symbol FT 
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/1107_1947_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_voxceleb-voxpopuli_sym_ft/lora_step0_cycle0_epoch7_periodic.pt"
# Random Regular FT
# checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/2307_1958_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_voxceleb-voxpopuli_swap_reg/lora_step0_cycle0_epoch10_periodic.pt"
# Random Symbol FT
checkpoint_path="/home/sriramg/chandnia/results/orchestrator_training/checkpoints/2307_1949_orchestrator_bypass_mlp_org_1c_10le_1me_bypass_mlp_org_qwen_voxceleb-voxpopuli_swap_sym/lora_step0_cycle0_epoch10_periodic.pt"



model_type="qwen"    # Options: salmonn, qwen, flamingo2
dataset_type="voxceleb-hvb-voxpopuli-meld_emotion"  
device="cuda:0"
output_dir="/home/sriramg/chandnia/results/orchestrator_inference"
max_val_samples=0

num_examples=3


# ===== Set Conda Environment =====
if [ "$model_type" == "salmonn" ]; then
    export CONDA_ENV="salmon"
elif [ "$model_type" == "qwen" ]; then
    export CONDA_ENV="qwen2"
else
    echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
    exit 1
fi

>>>>>>> f1678dd (Updated models and data/model_processors.py)
echo "Set conda environment to: $CONDA_ENV"
source /home/sriramg/chandnia/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $CONDA_ENV
echo "Activated conda environment: $CONDA_ENV"

# ========================================
# Validation
# ========================================

# Check if checkpoint exists
if [ ! -f "$checkpoint_path" ]; then
    echo "ERROR: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

# Clean dataset type for file names
CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')

# Get current date and time
CURRENT_DATETIME=$(date +"%d%m_%H%M")


# Extract training timestamp, dataset, and epoch from checkpoint path
CHECKPOINT_DIR=$(basename "$(dirname "$checkpoint_path")")
TRAINING_TIMESTAMP=$(echo "$CHECKPOINT_DIR" | cut -d'_' -f1-2)
EPOCH_NUM=$(basename "$checkpoint_path" | sed -n 's/.*epoch\([0-9]\+\).*/\1/p')

# ✅ IMPROVED: Extract everything after "salmonn_" to get full training dataset
TRAINING_DATASET=$(echo "$CHECKPOINT_DIR" | sed 's/.*salmonn_//')

clean_dataset_name() {
    local dataset=$1
    dataset=$(echo "$dataset" | sed 's/voxceleb/vox/g')
    dataset=$(echo "$dataset" | sed 's/voxpopuli/vop/g') 
    dataset=$(echo "$dataset" | sed 's/meld_emotion/meld/g')
    dataset=$(echo "$dataset" | sed 's/_/-/g')  # Replace underscores with hyphens
    echo "$dataset"
}

TRAINING_DATASET_CLEAN=$(clean_dataset_name "$TRAINING_DATASET")
CLEAN_DATASET_TYPE_COMPACT=$(clean_dataset_name "$CLEAN_DATASET_TYPE")
# Create compact checkpoint identifier with training dataset
CHECKPOINT_NAME="${TRAINING_TIMESTAMP}_ep${EPOCH_NUM}_${TRAINING_DATASET_CLEAN}"

# Update RUN_NAME
RUN_NAME="${CURRENT_DATETIME}_infer_${CHECKPOINT_NAME}_on_${CLEAN_DATASET_TYPE_COMPACT}_${num_examples}ex"

# Set script path
SCRIPT_PATH="/home/sriramg/chandnia/code/ICL-speech-text-LLM/models/symbolAdapter/orchestrator_inference.py"

TODAY=$(date +"%Y-%m-%d")
LOG_DIR="/home/sriramg/chandnia/results/orchestrator_inference/logs/${TODAY}"
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
    --run_name "${RUN_NAME}" \
    --checkpoint_path "${checkpoint_path}" \
    --dataset_type "${dataset_type}" \
    --device "${device}" \
    --max_val_samples ${max_val_samples} \
    --num_examples ${num_examples} \
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
