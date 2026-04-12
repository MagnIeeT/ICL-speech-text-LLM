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
output_dir="/home/harinis/ICL_qwen_run/eval_tasks"

# Hold until another job finishes (leave empty to run immediately)
hold_job_id=""

# ----------------------------------------
# MODEL SELECTION: "qwen" or "salmonn"
# ----------------------------------------
model="salmonn"

# CHANGE ONLY THIS LINE
# Options: "librispeech" or "vocalsound" or "covost2" or "librispeech_other"
# ========================================
dataset="librispeech_other"

# ========================================
# LoRA Checkpoint
# For BASELINE: leave lora_checkpoint=""
# For LoRA run: uncomment one checkpoint
# Works for BOTH qwen and salmonn
# ========================================

# --- QWEN checkpoints ---
# sym:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0502_1528_orchestrator_5e_10sce_bypass_mlp_sym_qwen_voxpopuli_meld_emotion/lora_step0_cycle0_epoch4_periodic.pt"
# epoch:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0302_2320_orchestrator_5e_1sce_bypass_mlp_sym_qwen_voxpopuli_meld_emotion/lora_step0_cycle0_epoch4_periodic.pt"
# mixed:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0702_1739_orchestrator_5e_10sce_bypass_mlp_sym_qwen_voxpopuli_meld_emotion/lora_step0_cycle0_epoch5_periodic.pt"
# regular:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0502_2252_orchestrator_5e_10sce_bypass_mlp_org_qwen_voxpopuli_meld_emotion/lora_step0_cycle0_epoch5_periodic.pt"
# qwen meld+vop low lora rank=8, alpha =32
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2401_1102_orchestrator_5e_1sce_bypass_mlp_sym_qwen_voxpopuli_meld_emotion/lora_step0_cycle0_epoch5_periodic.pt"

# --- SALMONN checkpoints ---
# sym:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2801_1141_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch4_periodic.pt"
# epoch:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2801_2141_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch4_periodic.pt"
# mixed:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0103_2158_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# regular:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0103_1818_orchestrator_5e_10sce_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"

# salmonn vox+vop lora=64, alpha=128
# sym(overfitting)
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/1902_1229_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_voxceleb_voxpopuli/lora_step0_cycle0_epoch2_periodic.pt"
#epoch:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2002_1115_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_voxceleb_voxpopuli/lora_step0_cycle0_epoch3_periodic.pt"
# mixed:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2302_1655_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_voxceleb_voxpopuli/lora_step0_cycle0_epoch4_periodic.pt"
#rft:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2402_1226_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxceleb_voxpopuli/lora_step0_cycle0_epoch5_periodic.pt"
# lf ft(flipped):
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2502_1531_orchestrator_5e_10sce_bypass_mlp_org_salmonn_voxpopuli_swap_voxceleb_swap/lora_step0_cycle0_epoch2_periodic.pt"

# salmonn hvb+vox lora=64, alpha=128
# sym(overfitting):
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2602_1609_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
#epoch:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2602_1614_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
#fliped lf ft:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/2702_1635_orchestrator_5e_1sce_bypass_mlp_org_salmonn_hvb_swap_voxceleb_swap/lora_step0_cycle0_epoch1_periodic.pt"
#rft(regular):
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0103_1818_orchestrator_5e_10sce_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
#mixed:
#lora_checkpoint="/home/leapers/weights/harinis/ICL-speech-text-LLM/orchestrator_training/checkpoints/0103_2158_orchestrator_5e_10sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"

# HVB-Vox lora rank=8, alpha=32
# random
 #lora_checkpoint="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0301_1316_orchestrator_5e_20sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
# change every epoch
 #lora_checkpoint="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0301_1318_orchestrator_5e_1sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# change every step
 #lora_checkpoint="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0101_2146_orchestrator_10e_20sce_bypass_mlp_sym_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch1_periodic.pt"
# original
lora_checkpoint="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/0901_1452_orchestrator_5e_1sce_bypass_mlp_org_salmonn_hvb_voxceleb/lora_step0_cycle0_epoch5_periodic.pt"
# interspeech
# lora_checkpoint="/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training/checkpoints/1001_1521_orchestrator_5e_1sce_bypass_mlp_org_salmonn_hvb_swap_voxceleb_swap/lora_step0_cycle0_epoch1_periodic.pt"

#lora_checkpoint=""

# LoRA rank/alpha — change here for both qwen and salmonn
lora_rank=8
lora_alpha=32
lora_dropout=0.1


# ========================================
# SALMONN fixed paths — do not change
# ========================================
salmonn_ckpt="/home/leapers/weights/SALMONN/salmonn_v1.pth"
beats_path="/home/leapers/weights/SALMONN/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
llama_path="lmsys/vicuna-13b-v1.1"
whisper_path="openai/whisper-large-v2"

# ========================================
# Auto Setup - No need to edit below
# ========================================
source /home/leapers/anaconda3/etc/profile.d/conda.sh
unset BNB_CUDA_VERSION

if [ "$model" == "qwen" ]; then
    CONDA_ENV="qwen2_new"
    SCRIPT_PATH="/home/harinis/ICL_qwen_run/eval_tasks/evaluate_asr.py"
    if [ -n "$lora_checkpoint" ]; then
        RUN_LABEL="qwen_lora"
        qwen_checkpoint="Qwen/Qwen2-Audio-7B-Instruct"
    else
        RUN_LABEL="qwen_baseline"
        qwen_checkpoint="Qwen/Qwen2-Audio-7B"
    fi
else
    CONDA_ENV="salmonn"
    SCRIPT_PATH="/home/harinis/ICL_qwen_run/eval_tasks/evaluate_asr_salmonn.py"
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
export PYTHONPATH=/home/harinis/ICL_qwen_run/SALMONN:$PYTHONPATH

echo "Python: $(which python)"
cd /home/harinis/ICL_qwen_run/eval_tasks
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
            --num-samples ${num_samples} \
            --num-workers 2 2>&1 | tee ${LOG_FILE}
    else
        echo "Running Qwen baseline evaluation..."
        python ${SCRIPT_PATH} \
            --checkpoint "${qwen_checkpoint}" \
            --dataset "${dataset}" \
            --batch-size 1 \
            --num-samples ${num_samples} \
            --num-workers 2 2>&1 | tee ${LOG_FILE}
    fi
else
    if [ -n "$lora_checkpoint" ]; then
        echo "Running SALMONN LoRA evaluation..."
        python ${SCRIPT_PATH} \
            --cfg-path "/home/harinis/ICL_qwen_run/SALMONN/configs/decode_config.yaml" \
            --dataset "${dataset}" \
            --device "${device}" \
            --lora-checkpoint "${lora_checkpoint}" \
            --lora-rank ${lora_rank} \
            --lora-alpha ${lora_alpha} \
            --lora-dropout ${lora_dropout} \
            --num-samples ${num_samples} 2>&1 | tee ${LOG_FILE}
    else
        echo "Running SALMONN baseline evaluation..."
        python ${SCRIPT_PATH} \
            --cfg-path "/home/harinis/ICL_qwen_run/SALMONN/configs/decode_config.yaml" \
            --dataset "${dataset}" \
            --device "${device}" \
            --num-samples ${num_samples} 2>&1 | tee ${LOG_FILE}
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