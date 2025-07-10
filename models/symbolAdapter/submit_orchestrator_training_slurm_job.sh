#!/bin/bash -l
#SBATCH --job-name=train_orchestrator
#SBATCH --output=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/logs/slurm_logs/%x_%j.out
#SBATCH --error=/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/logs/slurm_logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=long
#SBATCH --gres=gpu:A6000:1
#SBATCH --export=ALL

SLURM_LOG_DIR="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/logs/${TODAY}"
mkdir -p "$SLURM_LOG_DIR"

# ==== Configuration - Edit these values as needed ====
model_type="salmonn"    # Options: "salmonn" or "qwen"
dataset_type="voxceleb-meld_emotion-voxpopuli-hvb"   # Dataset type(s) to use

# Training parameters
mlp_lr=1e-5
lora_lr=1e-5
mlp_epochs=1
lora_epochs=2 # change to 5 
lora_final_epochs=1 
total_cycles=1
dynamic_symbols_per_epoch=False  # Generate new symbols each epoch
batch_size=1
gradient_accumulation_steps=8
max_grad_norm=1.0
max_samples=10 # Set reasonable default
num_examples=5
# Orchestrator-specific parameters (get_default_config() method in traning_configs.py)
schedule_type="bypass_mlp_sym"    # Options: "lora_first", "mlp_first", "joint_training","lora_mlp_joint"

# MLP Architecture parameters
use_output_mlp=False   # Enable/disable output MLP
bypass_mlp=True 
hidden_dim=32


# Set conda environment based on model type
if [ "$model_type" == "salmonn" ]; then
    export CONDA_ENV="salmon"
elif [ "$model_type" == "qwen" ]; then
    export CONDA_ENV="qwen2"
else
    echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
    exit 1
fi

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

# Clean dataset type for filenames
if [[ $dataset_type == *","* ]]; then
    CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')
else
    CLEAN_DATASET_TYPE=$dataset_type
fi


CURRENT_DATETIME=$(date +"%d%m_%H%M")

if [ "$bypass_mlp" = "True" ] || [ "$bypass_mlp" = "true" ]; then
    if [ "$dynamic_symbols_per_epoch" = "True" ] || [ "$dynamic_symbols_per_epoch" = "true" ]; then
        MLP_SUFFIX="bypass_mlp_sym"
    else
        MLP_SUFFIX="bypass_mlp_org"
    fi
else
    if [ "$use_output_mlp" = "True" ] || [ "$use_output_mlp" = "true" ]; then
        MLP_SUFFIX="io_mlp"     # Input + Output MLP
    else
        MLP_SUFFIX="i_mlp"      # Input MLP only
    fi
fi


# Calculate effective batch size
effective_batch_size=$((batch_size * gradient_accumulation_steps))


RUN_NAME="${CURRENT_DATETIME}_orchestrator_${schedule_type}_${total_cycles}c_${lora_epochs}le_${mlp_epochs}me_${MLP_SUFFIX}_${model_type}_${CLEAN_DATASET_TYPE}"

# Directory setup
SCRIPT_PATH="/home/sriramg/aneeraj/code/ICL-speech-text-LLM/models/symbolAdapter/orchestrator_training.py"
TODAY=$(date +"%Y-%m-%d")

# Directory setup
OUTPUT_DIR="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training"
LOG_DIR="/home/sriramg/aneeraj/storage/results/model_ICL/orchestrator_training/logs/${TODAY}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "$LOG_FILE"



echo "=========================================="
echo "Orchestrator Symbol Training Job Configuration"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "Dataset: ${dataset_type}"
echo "Schedule Type: ${schedule_type}"
echo "Cycles: ${total_cycles}"
echo "LoRA Epochs/Cycle: ${lora_epochs}"
echo "MLP Epochs/Cycle: ${mlp_epochs}"
echo "LoRA LR: ${lora_lr}"
echo "MLP LR: ${mlp_lr}"
echo "Use Output MLP: ${use_output_mlp}"
echo "Bypass MLP: ${bypass_mlp}"
echo "Dynamic Symbols: ${dynamic_symbols_per_epoch}"
echo "Hidden Dim: ${hidden_dim}"
echo "Max Samples: ${max_samples}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_FILE}"
echo "=========================================="


# ======= Build and Run Python Command =======
CMD="python \"$SCRIPT_PATH\" \
    --run_name \"$RUN_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --model_type \"$model_type\" \
    --dataset_type \"$dataset_type\" \
    --lora_lr $lora_lr \
    --mlp_lr $mlp_lr \
    --lora_epochs $lora_epochs \
    --mlp_epochs $mlp_epochs \
    --total_cycles $total_cycles \
    --lora_final_epochs $lora_final_epochs \
    --hidden_dim $hidden_dim \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_grad_norm $max_grad_norm \
    --max_samples $max_samples \
    --num_examples $num_examples \
    --schedule_type \"$schedule_type\""


# Append boolean flags only if true
if [ "$use_output_mlp" = "True" ] || [ "$use_output_mlp" = "true" ]; then
    CMD="$CMD --use_output_mlp"
fi

if [ "$bypass_mlp" = "True" ] || [ "$bypass_mlp" = "true" ]; then
    CMD="$CMD --bypass_mlp"
fi

if [ "$dynamic_symbols_per_epoch" = "True" ] || [ "$dynamic_symbols_per_epoch" = "true" ]; then
    CMD="$CMD --dynamic_symbols_per_epoch"
fi

if [ "$only_original" = "True" ] || [ "$only_original" = "true" ]; then
    CMD="$CMD --only_original"
fi


# Run the command
echo "Executing training script..."
eval "$CMD" > "$LOG_FILE" 2>&1

echo "Submitted orchestrator symbol training job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_FILE}"