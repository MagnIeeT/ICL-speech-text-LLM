#!/bin/bash
hostname="n6"
cuda_device=1
device="cuda:0"

source /home/leapers/anaconda3/etc/profile.d/conda.sh
unset BNB_CUDA_VERSION

TODAY=$(date +"%Y-%m-%d")
LOG_DIR="/home/harinis/ICL_qwen_run/eval_qwen_and_salmonn/logs/${TODAY}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/test_loading.log"

echo "Submitting test_loading job..."
echo "Log: ${LOG_FILE}"

qsub -q workq \
    -l select=1:num_gpus=1:gpu_mem=48GB:host=${hostname} \
    -l walltime=01:00:00 \
    -o /dev/null \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},\
LOG_FILE="${LOG_FILE}",\
PYTHONUNBUFFERED=1 \
    -S /bin/bash << 'EOF'
#!/bin/bash
source /home/leapers/anaconda3/etc/profile.d/conda.sh
unset BNB_CUDA_VERSION
conda activate salmonn
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM + 29000))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export PYTHONPATH=/home/harinis/ICL_qwen_run/SALMONN:$PYTHONPATH
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
nvidia-smi
cd /home/harinis/ICL_qwen_run/eval_qwen_and_salmonn
python test_loading.py 2>&1 | tee ${LOG_FILE}
EOF

echo "Monitor: tail -f ${LOG_FILE}"
echo "qstat | grep harinis"
OUTEREOF