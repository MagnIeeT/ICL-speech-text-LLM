#!/bin/bash
set -e

# Force IST timezone labelling everywhere.
export TZ='Asia/Kolkata'

# The LOGIN node clock runs ~5.5h FAST vs the COMPUTE node (where jobs actually
# run). Filenames were stamped with login time and drifted from the in-job time.
# Fix: grab the compute node's epoch ONCE (correct clock) and format every
# timestamp from it. Falls back to the login clock if the node is unreachable.
_TS_EPOCH=$(ssh -o BatchMode=yes -o ConnectTimeout=5 compute date +%s 2>/dev/null || true)
if [ -n "${_TS_EPOCH}" ]; then
    _ts() { date -d "@${_TS_EPOCH}" +"$1"; }
else
    echo "WARNING: could not reach 'compute' for time; using login clock (may be ~5.5h fast)." >&2
    _ts() { date +"$1"; }
fi

# ============================================================
# LLaVA Inference - HPC Submit Script  —  EE CLUSTER (IISc)
# ============================================================
# EE variant, modelled on ICI/hpc/submit_symbol_inference_job.sh (the script
# sir made for EE) but wired to the VLM components (vision_orchestrator.py).
#   * PBS/Torque (qsub) with `-v` inline env passing + heredoc | tee
#   * queue = "compute", NO host pin (the partition picks the node)
#   * conda: source <CONDA_BASE>/etc/profile.d/conda.sh; conda activate llava
#   * LD_PRELOAD / LD_LIBRARY_PATH exports (env-lib safety), like ICI
#   * Paths under /home/harinisrireddykandula/ (EE home)
#
# Submits ONE job that runs all specified datasets sequentially into a single
# log file (via vision_orchestrator.py).
#
# Usage:
#   bash submit_inference_compute.sh             (submit job)
#   bash submit_inference_compute.sh --dry-run   (preview only)
#   bash submit_inference_compute.sh --print-id  (submit + print job ID)
#
#   datasets=chest strategy=two_token \
#     CHECKPOINT_PATH=/home/harinisrireddykandula/llava/checkpoints/<ckpt> \
#     bash submit_inference_compute.sh
#
#   probe_batch_size=8 diagnose_samples=0 datasets=chest strategy=two_token \
#     bash submit_inference_compute.sh        # batched scoring + [BATCH-VERIFY]
#
#   eval_modes=original,fixed,fresh strategy=two_token datasets=chest \
#     bash submit_inference_compute.sh        # explicit symbol eval modes
# ============================================================

# ------------------------------------------------------------
# 1. Job Configuration (edit / override via env)
# ------------------------------------------------------------
datasets=colon-chest-endo  # hyphen-separated: "chest" or "colon-chest-endo"
strategy=two_token            # regular | two_token | ed_ft | id_ft | lf_ft
model_type="${model_type:-llava-v1.5-13b}"

num_samples="${num_samples:-0}"            # 0 = ALL samples
icl_shots=0

# Inference scoring (opt-in; defaults reproduce the old unbatched path).
probe_batch_size=1  # >1 enables batched per-class scoring (chest/endo)
diagnose_samples=0  # set 0 so [BATCH-VERIFY] fires on sample 0 when batching

# Validation-set evaluation (additive; default OFF = test JSON, unchanged).
#   USE_VALIDATION=true  → eval on {dataset}_val_shot{val_shot}_exp{val_exp}.json
#   MAX_VAL_SAMPLES>0    → reproduce training seeded subset (random.Random(42)); 0 = full val
#   Example: USE_VALIDATION=true MAX_VAL_SAMPLES=0 datasets=chest strategy=ed_ft \
#            CHECKPOINT_PATH=<ckpt>/checkpoint-best bash submit_inference_compute.sh
USE_VALIDATION=true
MAX_VAL_SAMPLES=100
val_shot=10
val_exp=2

# Eval modes (comma-separated subset of original,fixed,fresh). Empty = sprint_eval
# default (symbol strategies → original,fixed,fresh ; regular/rft → original only).
eval_modes=original
# Fine-tuned checkpoint (LoRA adapter dir) — same checkpoint used for all datasets
CHECKPOINT_PATH=
# Set to a job ID (e.g. "12093.eehpc") to wait for that job first.
hold_job_id=

# ------------------------------------------------------------
# 2. EE PBS / Queue / Conda settings
# ------------------------------------------------------------
queue_name="${queue_name:-GPU_only}"        # EE GPU queue (others: workq, CPU_only)
hostname=compute                   # EMPTY = let 'compute' pick the node; set to pin
# GPU SELECTION: the 'compute' node has 8 GPUs (0-7). Pick a FREE one (check with
# `nvidia-smi` on the node) and pass it, e.g.  cuda_device=3 bash submit_inference_compute.sh
cuda_device=3
walltime="${walltime:-24:00:00}"
# EE has NO GPU PBS resource (ngpus/num_gpus undefined) — GPUs live on the single
# 'compute' node reached via the GPU_only queue; pick the GPU with CUDA_VISIBLE_DEVICES.
# 'mem' is system RAM (node has 262GB), NOT GPU memory.
ncpus="${ncpus:-8}"
mem="${mem:-64gb}"

# >>> CONFIRM ON EE <<<  conda base dir that holds etc/profile.d/conda.sh
CONDA_BASE="${CONDA_BASE:-/home/harinisrireddykandula/miniconda3}"
CONDA_ENV="${conda_env:-llava}"

# ------------------------------------------------------------
# 3. Paths
# ------------------------------------------------------------
LLAVA_DIR="${LLAVA_DIR:-/home/harinisrireddykandula/LLaVA}"
MODEL_BASE="${MODEL_BASE:-/home/harinisrireddykandula/llava-v1.5-13b}"
output_dir="${LLAVA_DIR}/logs"
orchestrator_path="${LLAVA_DIR}/sprint_vision/vision_orchestrator.py"

# ------------------------------------------------------------
# 4. Automatic setup
# ------------------------------------------------------------
CURRENT_DATETIME=$(_ts "%d%m_%H%M%S")   # seconds → two same-minute submits never collide
TODAY=$(_ts "%Y-%m-%d")
RUN_NAME="${CURRENT_DATETIME}_infer_${strategy}_${icl_shots}shot_${model_type}"

LOG_DIR="${output_dir}/${TODAY}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
rm -f "${LOG_FILE}"

# PBS resource string — host pin only if requested.
RESOURCE="select=1:ncpus=${ncpus}:mem=${mem}"
if [ -n "${hostname}" ]; then
    RESOURCE="${RESOURCE}:host=${hostname}"
    HOST_MSG="${hostname} (pinned)"
else
    HOST_MSG="<any compute node>"
fi

if [ -n "$hold_job_id" ]; then
    HOLD_FLAG="-W depend=afterok:${hold_job_id}"
    HOLD_MSG="Waiting for job: ${hold_job_id}"
else
    HOLD_FLAG=""
    HOLD_MSG="Starting immediately (no dependency)"
fi

echo "=========================================="
echo "LLaVA Inference Configuration  (EE / ${queue_name})"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Datasets:    ${datasets}"
echo "Strategy:    ${strategy}"
echo "Model:       ${model_type}"
echo "Shots:       ${icl_shots}"
echo "Samples:     ${num_samples} (0 = ALL)"
echo "Queue:       ${queue_name}"
echo "Host:        ${HOST_MSG}"
echo "Resource:    ${RESOURCE}"
echo "CUDA Device: ${cuda_device}"
echo "Conda:       ${CONDA_BASE}  (env: ${CONDA_ENV})"
echo "Checkpoint:  ${CHECKPOINT_PATH}"
echo "Probe batch: ${probe_batch_size}  Diagnose: ${diagnose_samples}"
echo "Eval modes:  ${eval_modes:-<default>}"
echo "Use validation: ${USE_VALIDATION}  (val shot${val_shot} exp${val_exp}, max_val_samples=${MAX_VAL_SAMPLES})"
echo "Dependency:  ${HOLD_MSG}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

if [[ "${1}" == "--dry-run" ]]; then
    echo "DRY-RUN — no job submitted"
    exit 0
fi

# ------------------------------------------------------------
# 5. Submit (qsub -v inline env passing + heredoc, ICI-style)
#    Comma-containing values (eval_modes) are inlined at submit time, NOT via -v.
# ------------------------------------------------------------
JOB_ID=$(qsub -q "${queue_name}" \
    $HOLD_FLAG \
    -N "${RUN_NAME}" \
    -l ${RESOURCE} \
    -l walltime=${walltime} \
    -o /dev/null \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},\
PYTHONUNBUFFERED=1,\
PYTHONIOENCODING=utf-8,\
LOG_FILE=${LOG_FILE},\
RUN_NAME=${RUN_NAME},\
LLAVA_DIR=${LLAVA_DIR},\
orchestrator_path=${orchestrator_path},\
datasets=${datasets},\
strategy=${strategy},\
num_samples=${num_samples},\
icl_shots=${icl_shots},\
CHECKPOINT_PATH=${CHECKPOINT_PATH},\
MODEL_BASE=${MODEL_BASE},\
PROBE_BATCH_SIZE=${probe_batch_size},\
DIAGNOSE_SAMPLES=${diagnose_samples},\
USE_VALIDATION=${USE_VALIDATION},\
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES},\
VAL_SHOT=${val_shot},\
VAL_EXP=${val_exp},\
TZ=Asia/Kolkata \
    -S /bin/bash << EOF
#!/bin/bash
set -e
set -o pipefail

cd \${LLAVA_DIR}
export TZ='Asia/Kolkata'
# Login-node clock is ~5.5h fast → stale .pyc can shadow freshly-copied .py edits.
# Never cache bytecode, so Python always runs the current source.
export PYTHONDONTWRITEBYTECODE=1

echo "=== Running on: \$(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES} ==="
echo "Datasets: \${datasets}  Strategy: \${strategy}  Shots: \${icl_shots}"
echo "Checkpoint: \${CHECKPOINT_PATH}"
echo "Probe batch size: \${PROBE_BATCH_SIZE}  Diagnose samples: \${DIAGNOSE_SAMPLES}"
echo "Eval modes: ${eval_modes:-<default>}"
echo "Use validation: \${USE_VALIDATION}  (val shot\${VAL_SHOT} exp\${VAL_EXP}, max_val_samples=\${MAX_VAL_SAMPLES})"

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Pin the chosen GPU (PBS doesn't schedule GPUs here, so set it explicitly).
export CUDA_VISIBLE_DEVICES=${cuda_device}

export LD_PRELOAD=""
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

# Eval-mode choice — read by vision_orchestrator.py Config.EVAL_MODES → --modes.
export EVAL_MODES="${eval_modes}"

nvidia-smi

stdbuf -oL -eL python -u "\${orchestrator_path}" \
    --mode inference \
    --datasets "\${datasets}" \
    --strategy "\${strategy}" \
    --num-samples "\${num_samples}" \
    --icl-shots "\${icl_shots}" \
    --checkpoint-path "\${CHECKPOINT_PATH}" \
    --model-base "\${MODEL_BASE}" 2>&1 | tee \${LOG_FILE}
EOF
)

echo ""
echo "=========================================="
echo "Job Submitted!"
echo "  Job ID:  ${JOB_ID}"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Status:  qstat -u \$(whoami)"
echo ""
echo "To chain another job after this one:"
echo "  hold_job_id=${JOB_ID} bash submit_inference_compute.sh"
echo "=========================================="

if [[ "${1}" == "--print-id" ]]; then
    echo "${JOB_ID}"
fi
