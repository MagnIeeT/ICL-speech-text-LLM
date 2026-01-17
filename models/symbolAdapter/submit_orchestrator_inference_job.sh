#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/models/symbolAdapter/submit_orchestrator_inference_job.sh

# ========================================
# Configuration - Edit these values as needed
# ========================================

# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# unset LD_LIBRARY_PATH
# python -c "import torch; print(torch.cuda.is_available())"

# HVB-Vox
# random
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0301_1316_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
# change every epoch
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0301_1318_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# change every step
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0101_2146_orchestrator_10e_20sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# original
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0901_1452_orchestrator_5e_1sce_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
# interspeech
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1001_1521_orchestrator_5e_1sce_bypass_mlp_org_salmonn_hvb_swap_voxceleb_swap/lora_step0_cycle0_epoch1_periodic.pt"

#meld-vop
#random
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0401_1209_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_voxpopuli_meld_emotion/lora_step0_cycle0_epoch5_periodic.pt"
# change every step
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0401_1210_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_voxpopuli_meld_emotion/lora_step0_cycle0_epoch3_periodic.pt"
#change every epoch
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0501_0241_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_voxpopuli_meld_emotion/lora_step0_cycle0_epoch5_periodic.pt"
#original 
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1101_1357_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_meld_emotion/lora_step0_cycle0_epoch3_periodic.pt"
#swap interspeech
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1101_2301_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_swap_meld_emotion_swap/lora_step0_cycle0_epoch1_periodic.pt"
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1501_2325_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_swap_meld_emotion_swap/lora_step0_cycle0_epoch5_periodic.pt"

#vox vop
#random
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0501_2331_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_voxpopuli_voxceleb/lora_step0_cycle0_epoch3_periodic.pt"
#every step
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0601_1443_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_voxpopuli_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
#every epoch
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0601_1448_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_voxpopuli_voxceleb/lora_step0_cycle0_epoch2_periodic.pt"
#original
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1201_1327_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_voxceleb/lora_step0_cycle0_epoch3_periodic.pt"
#swap interspeech
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1201_2125_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_swap_voxceleb_swap/lora_step0_cycle0_epoch5_periodic.pt"
# checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1501_2327_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_swap_voxceleb_swap/lora_step0_cycle0_epoch3_periodic.pt"


#hvb meld
#interspeech swap
checkpoint_path="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1701_0021_orchestrator_5e_10sce_bypass_mlp_org_salmonn_hvb_swap_meld_emotion_swap/lora_step0_cycle0_epoch4_periodic.pt"


dataset_type="hvb-voxceleb-voxpopuli-meld_emotion"  # Dataset type to evaluate on

# dataset_type="voxpopuli"  # Dataset type to evaluate on
max_val_samples=0         # 0 = use all samples

num_examples=0

# Optional parameters
device="cuda:0"
output_dir="/home/neeraja/results/ICL-speech-text-LLM/"


hostname="n8"
cuda_device=1

hold_job_id="4209.eehpc"
# ========================================
# Validation and Setup
# ========================================

# Set conda environment
export CONDA_ENV="salmonn"
echo "Set conda environment to: $CONDA_ENV"
source /home/leapers/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV   


if [ -n "$hold_job_id" ]; then
    echo "Job will wait for completion of job: $hold_job_id"
    HOLD_FLAG="-W depend=afterok:$hold_job_id"
else
    HOLD_FLAG=""
fi




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
SCRIPT_PATH="/home/neeraja/code/ICL-speech-text-LLM/models/symbolAdapter/orchestrator_inference.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories
LOG_DIR="/home/neeraja/results/ICL-speech-text-LLM/orchestrator_inference/logs/${TODAY}"

mkdir -p "$LOG_DIR"

# Remove old log file if exists
rm -f "${LOG_DIR}/${RUN_NAME}.log"

# ========================================
# Display Configuration
# ========================================
echo "=========================================="
echo "Orchestrator Inference Job Configuration"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "Checkpoint: ${checkpoint_path}"
echo "Dataset: ${dataset_type}"
echo "Max Samples: ${max_val_samples}"
echo "Device: ${device}"
echo "Output Dir: ${output_dir}"
echo "Log File: ${LOG_DIR}/${RUN_NAME}.log"
echo "Queue: ${queue_name}"
echo "Hostname: ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "=========================================="

# ========================================
# Submit Job
# ========================================
qsub -q workq \
    $HOLD_FLAG \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=$hostname \
    -l walltime=24:00:00 \
    -o /dev/null \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},\
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log",\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
checkpoint_path=${checkpoint_path},\
dataset_type=${dataset_type},\
max_val_samples=${max_val_samples},\
num_examples=${num_examples},\
device=${device},\
output_dir=${output_dir} \
    -S /bin/bash << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "Starting Orchestrator Inference Job"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Python path: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""



# Run inference with detailed logging
python ${SCRIPT_PATH} \
    --checkpoint_path "${checkpoint_path}" \
    --dataset_type "${dataset_type}" \
    --device "${device}" \
    --max_val_samples ${max_val_samples} \
    --num_examples ${num_examples} \
    --output_dir "${output_dir}"\
    --run_name "${RUN_NAME}" 2>&1 | tee ${LOG_FILE}

EXIT_CODE=$?

exit ${EXIT_CODE}
EOF


# ========================================
# Report Job Submission
# ========================================
JOB_ID_NUM=$(echo $JOB_ID | cut -d' ' -f3)

echo ""
echo "=========================================="
echo "Job Submitted Successfully"
echo "=========================================="
echo "Job Name: ${RUN_NAME}"
echo "Job ID: ${JOB_ID_NUM}"
echo "Host: ${hostname}"
echo ""
echo "Monitor commands:"
echo "  tail -f ${LOG_DIR}/${RUN_NAME}.log"
echo "  qstat | grep ${JOB_ID_NUM}"
echo "  qstat -j ${JOB_ID_NUM}"
echo ""
echo "Results will be saved to:"
echo "  ${output_dir}/orchestrator_metrics/"
echo "  ${output_dir}/orchestrator_logs/"
echo "=========================================="


