#!/bin/bash
# ============================================================
# Unified ASR Evaluation Submit Script
# Supports: Qwen baseline, Qwen LoRA, SALMONN baseline, SALMONN LoRA
# CHANGE ONLY the "Configuration" block below
# ============================================================

# ========================================
# Configuration - Edit these values
# ========================================
hostname="n6"
cuda_device=1
device="cuda:0"
# set to 0 or empty for full dataset
num_samples=0

# Determine project directories dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SALMONN_ROOT="$(cd "$PROJECT_ROOT/../SALMONN" && pwd)"

output_dir="${SCRIPT_DIR}"

# ... (omitted LoRA checkpoints) ...

# ========================================
# SALMONN fixed paths — do not change
# ========================================
# Load paths from .env if available
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

salmonn_ckpt="${SALMONN_CKPT_PATH:-$PROJECT_ROOT/weights/SALMONN/salmonn_v1.pth}"
beats_path="${BEATS_CKPT_PATH:-$PROJECT_ROOT/weights/SALMONN/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}"
llama_path="${LLAMA_MODEL_NAME:-lmsys/vicuna-13b-v1.1}"
whisper_path="${WHISPER_MODEL_NAME:-openai/whisper-large-v2}"

# ========================================
# Auto Setup - No need to edit below
# ========================================
# source /home/leapers/anaconda3/etc/profile.d/conda.sh
# Unset this to use local conda if available
# source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null

if [ "$model" == "qwen" ]; then
    CONDA_ENV="qwen2_new"
    SCRIPT_PATH="${SCRIPT_DIR}/evaluate_asr.py"
    if [ -n "$lora_checkpoint" ]; then
        RUN_LABEL="qwen_lora"
        qwen_checkpoint="${QWEN_MODEL_NAME:-Qwen/Qwen2-Audio-7B-Instruct}"
    else
        RUN_LABEL="qwen_baseline"
        qwen_checkpoint="Qwen/Qwen2-Audio-7B"
    fi
else
    CONDA_ENV="salmonn"
    SCRIPT_PATH="${SCRIPT_DIR}/evaluate_asr_salmonn.py"
    if [ -n "$lora_checkpoint" ]; then
        RUN_LABEL="salmonn_lora"
    else
        RUN_LABEL="salmonn_baseline"
    fi
fi

conda activate $CONDA_ENV

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:$hold_job_id"
else
    HOLD_FLAG=""
fi

CURRENT_DATETIME=$(date +"%d%m_%H%M")
RUN_NAME="${CURRENT_DATETIME}_${RUN_LABEL}_${dataset}"

TODAY=$(date +"%Y-%m-%d")
LOG_DIR="${output_dir}/logs/${TODAY}"
mkdir -p "$LOG_DIR"
rm -f "${LOG_DIR}/${RUN_NAME}.log"

echo "=========================================="
echo "ASR Evaluation Job Configuration"
echo "=========================================="
echo "Model:       ${model}"
echo "Run Name:    ${RUN_NAME}"
echo "Script:      ${SCRIPT_PATH}"
echo "Dataset:     ${dataset}"
echo "LoRA:        ${lora_checkpoint:-None (baseline)}"
echo "LoRA rank:   ${lora_rank}  alpha: ${lora_alpha}"
echo "Conda Env:   ${CONDA_ENV}"
echo "Hostname:    ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Log File:    ${LOG_DIR}/${RUN_NAME}.log"
echo "=========================================="

qsub -q workq \
    $HOLD_FLAG \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=${hostname} \
    -l walltime=24:00:00 \
    -o /dev/null \
    -j oe \
    -v num_samples=${num_samples},\
CUDA_VISIBLE_DEVICES=${cuda_device},\
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log",\
PYTHONUNBUFFERED=1,\
model=${model},\
SCRIPT_PATH=${SCRIPT_PATH},\
dataset=${dataset},\
device=${device},\
qwen_checkpoint=${qwen_checkpoint},\
lora_checkpoint=${lora_checkpoint},\
lora_rank=${lora_rank},\
lora_alpha=${lora_alpha},\
lora_dropout=${lora_dropout},\
salmonn_ckpt=${salmonn_ckpt},\
beats_path=${beats_path},\
llama_path=${llama_path},\
whisper_path=${whisper_path} \
    -S /bin/bash << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "Starting ASR Evaluation Job"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

source /home/leapers/anaconda3/etc/profile.d/conda.sh
unset BNB_CUDA_VERSION

if [ "$model" == "qwen" ]; then
    conda activate qwen2_new
else
    conda activate salmonn
fi

export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM + 29000))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export PYTHONPATH=${SALMONN_ROOT}:${PROJECT_ROOT}:$PYTHONPATH

echo "Python: $(which python)"
cd ${SCRIPT_DIR}
echo "Working directory: $(pwd)"
nvidia-smi

if [ "$model" == "qwen" ]; then
    if [ -n "$lora_checkpoint" ]; then
        echo "Running Qwen LoRA evaluation..."
        python ${SCRIPT_PATH} \
            --checkpoint "${qwen_checkpoint}" \
            --dataset "${dataset}" \
            --lora-checkpoint "${lora_checkpoint}" \
            --lora-rank ${lora_rank} \
            --lora-alpha ${lora_alpha} \
            --lora-dropout ${lora_dropout} \
            --batch-size 1 \
            --num-workers 2 2>&1 | tee ${LOG_FILE}
    else
        echo "Running Qwen baseline evaluation..."
        python ${SCRIPT_PATH} \
            --checkpoint "${qwen_checkpoint}" \
            --dataset "${dataset}" \
            --batch-size 1 \
            --num-workers 2 2>&1 | tee ${LOG_FILE}
    fi
else
    if [ -n "$lora_checkpoint" ]; then
        echo "Running SALMONN LoRA evaluation..."
        python ${SCRIPT_PATH} \
            --cfg-path "${SALMONN_ROOT}/configs/decode_config.yaml" \
            --dataset "${dataset}" \
            --device "${device}" \
            --lora-checkpoint "${lora_checkpoint}" \
            --lora-rank ${lora_rank} \
            --lora-alpha ${lora_alpha} \
            --lora-dropout ${lora_dropout} 2>&1 | tee ${LOG_FILE}
    else
        echo "Running SALMONN baseline evaluation..."
        python ${SCRIPT_PATH} \
            --cfg-path "${SALMONN_ROOT}/configs/decode_config.yaml" \
            --dataset "${dataset}" \
            --device "${device}" 2>&1 | tee ${LOG_FILE}
    fi
fi

EXIT_CODE=$?
exit ${EXIT_CODE}
EOF

echo ""
echo "=========================================="
echo "Job Submitted Successfully"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "Hostname: ${hostname}"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/${RUN_NAME}.log"
echo "  qstat | grep harinis"
echo "=========================================="