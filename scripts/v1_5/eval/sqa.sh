#!/bin/bash
# ============================================================
# LLaVA ScienceQA Evaluation Script
# ============================================================

# ========================================
# Configuration - Edit these values
# ========================================
hostname="n8"
cuda_device=1
num_samples=0
output_dir="/home/harinis/LLaVA/logs"
hold_job_id=""

# ========================================
# Auto Setup
# ========================================
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate llava

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:$hold_job_id"
else
    HOLD_FLAG=""
fi

CURRENT_DATETIME=$(date +"%d%m_%H%M")
RUN_NAME="${CURRENT_DATETIME}_llava_scienceqa_samples${num_samples}"
TODAY=$(date +"%Y-%m-%d")
LOG_DIR="${output_dir}/${TODAY}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "${LOG_FILE}"

echo "=========================================="
echo "LLaVA ScienceQA Evaluation"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Num Samples: ${num_samples} (0 = ALL)"
echo "Hostname:    ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

# Write job script to temp file
TMPJOB=$(mktemp /tmp/llava_job_XXXX.sh)
cat > ${TMPJOB} << ENDJOB
#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=${cuda_device}
exec > >(stdbuf -oL tee -a ${LOG_FILE}) 2>&1
echo "=========================================="
echo "Job started at: \$(date)"
echo "Running on host: \$(hostname)"
echo "CUDA devices: ${cuda_device}"
echo "=========================================="
source /home/leapers/anaconda3/etc/profile.d/conda.sh
conda activate llava
cd /home/harinis/LLaVA
echo "Working directory: \$(pwd)"
echo "=========================================="
echo "GPU Status:"
echo "=========================================="
nvidia-smi
echo "=========================================="
echo "Preparing dataset..."
echo "=========================================="
python3 -u -c "
import json, sys
num = int('${num_samples}')
data = json.load(open('./playground/data/eval/scienceqa/llava_test_CQM-A.json'))
print(f'Total available samples: {len(data)}', flush=True)
if num == 0:
    subset = data
    print(f'Running on ALL {len(subset)} samples', flush=True)
else:
    subset = data[:num]
    print(f'Running on FIRST {num} samples', flush=True)
with open('./playground/data/eval/scienceqa/llava_test_subset.json', 'w') as f:
    json.dump(subset, f)
print('Subset file created!', flush=True)
sys.stdout.flush()
"
echo "=========================================="
echo "Running LLaVA inference..."
echo "Started at: \$(date)"
echo "=========================================="
stdbuf -oL -eL python -u -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/scienceqa/llava_test_subset.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-subset.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
echo "=========================================="
echo "Inference done at: \$(date)"
echo "Running accuracy evaluation..."
echo "=========================================="
stdbuf -oL -eL python -u llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-subset.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-subset_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b-subset_result.json
echo "=========================================="
echo "Job completed at: \$(date)"
echo "=========================================="
ENDJOB

chmod +x ${TMPJOB}

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
