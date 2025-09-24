#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/scripts/submit_unified_training_job.sh

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
dataset_type="voxceleb"  # Dataset type(s) to use
device="cuda:0"  # GPU device

# Training parameters
lora_lr=1e-5
mlp_lr=1e-4
lora_epochs=1
lora_final_epochs=1 
mlp_epochs=1

total_cycles=1

# MLP Architecture parameters (NEW SECTION)
use_output_mlp=False  # Enable/disable output MLP
bypass_mlp=True 


hidden_dim=8
batch_size=1
gradient_accumulation_steps=8
max_grad_norm=1.0
max_samples=10  # ✅ Set reasonable default instead of 0

# Set conda environment
export CONDA_ENV="salmonn"
echo "Set conda environment to: $CONDA_ENV"
source /home/leapers/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV

# Clean dataset type for filenames
if [[ $dataset_type == *"-"* ]]; then
    CLEAN_DATASET_TYPE=$(echo $dataset_type | tr '-' '_')
else
    CLEAN_DATASET_TYPE=$dataset_type
fi

# Get current date and time
CURRENT_DATETIME=$(date +"%d%m_%H%M")

# Generate descriptive run name (UPDATED TO INCLUDE MLP INFO)
# Generate descriptive run name (UPDATED TO INCLUDE MLP INFO)
if [ "$bypass_mlp" = "True" ] || [ "$bypass_mlp" = "true" ]; then
    MLP_SUFFIX="bypass_mlp"
else
    if [ "$use_output_mlp" = "True" ] || [ "$use_output_mlp" = "true" ]; then
        MLP_SUFFIX="io_mlp"  # Input + Output MLP
    else
        MLP_SUFFIX="i_mlp"   # Input MLP only
    fi
fi

RUN_NAME="${CURRENT_DATETIME}_unified_${total_cycles}c_${lora_epochs}le_${mlp_epochs}me_${MLP_SUFFIX}_${model_type}_${CLEAN_DATASET_TYPE}"
# Set script path
SCRIPT_PATH="/home/neeraja/code/ICL/models/unified_symbol_training.py"
TODAY=$(date +"%Y-%m-%d")

# Directory setup
OUTPUT_DIR="/home/leapers/weights/neeraja/ICL/unified_training"  
LOG_DIR="/home/neeraja/results/ICL/logs/unified_training/${TODAY}"

# Create directories
for dir in "$LOG_DIR" "$OUTPUT_DIR"; do
    if ! mkdir -p "$dir"; then
        echo "Error: Cannot create directory $dir"
        exit 1
    fi
done

# Remove old log file if it exists
rm -f "${LOG_DIR}/${RUN_NAME}.log"

# Print configuration (UPDATED WITH MLP INFO)
echo "=========================================="
echo "Unified Symbol Training Job Configuration"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "Dataset: ${dataset_type}"
echo "Device: ${device}"
echo "Cycles: ${total_cycles}"
echo "LoRA Epochs/Cycle: ${lora_epochs}"
echo "MLP Epochs/Cycle: ${mlp_epochs}"
echo "LoRA LR: ${lora_lr}"
echo "MLP LR: ${mlp_lr}"
echo "Use Output MLP: ${use_output_mlp}"  # NEW
echo "Hidden Dim: ${hidden_dim}"          # NEW
echo "Max Samples: ${max_samples}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_DIR}/${RUN_NAME}.log"
echo "=========================================="

# Submit job (UPDATED WITH NEW PARAMETERS)
qsub -q long-gpu.q -V -cwd \
    -l hostname=compute-0-9 \
    -l h_rt=72:00:00 \
    -o "${LOG_DIR}/${RUN_NAME}.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=0,\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
dataset_type=${dataset_type},\
model_type=${model_type},\
device=${device},\
lora_lr=${lora_lr},\
mlp_lr=${mlp_lr},\
lora_epochs=${lora_epochs},\
mlp_epochs=${mlp_epochs},\
total_cycles=${total_cycles},\
lora_final_epochs=${lora_final_epochs},\
use_output_mlp=${use_output_mlp},\
hidden_dim=${hidden_dim},\
batch_size=${batch_size},\
gradient_accumulation_steps=${gradient_accumulation_steps},\
max_grad_norm=${max_grad_norm},\
max_samples=${max_samples},\
bypass_mlp=${bypass_mlp},\
OUTPUT_DIR=${OUTPUT_DIR} \
    -S /bin/bash /home/neeraja/code/ICL/scripts/unified_training.sh

echo "Submitted unified symbol training job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"