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
# SPRInT-LLaVA Training - HPC Submit Script  —  EE CLUSTER (IISc)
# ============================================================
# EE variant, modelled on ICI/hpc/submit_symbol_training_job.sh (the script sir
# made for EE) but wired to the VLM components (run_sprint_finetune.sh → deepspeed).
#   * PBS/Torque (qsub) + heredoc | tee
#   * queue = "compute", NO host pin (the partition picks the node)
#   * conda: source <CONDA_BASE>/etc/profile.d/conda.sh; conda activate llava
#   * LD_PRELOAD / LD_LIBRARY_PATH exports (env-lib safety), like ICI
#   * Paths under /home/harinisrireddykandula/ (EE home)
#
# HOW TO USE:
#   Single run:
#       STRATEGY=regular   DATASET=colon TRAIN_PERCENT=100 bash submit_training_compute.sh
#       STRATEGY=two_token DATASET=colon TRAIN_PERCENT=100 bash submit_training_compute.sh
#
#   MedFMC official 10-shot split (paper protocol):
#       STRATEGY=two_token DATASET=chest TRAINING_SHOTS=10 SHOT_EXP=1 bash submit_training_compute.sh
#
#   ICL during training (K example images embedded in each training prompt):
#       ICL_SHOTS=1 STRATEGY=two_token DATASET=chest TRAINING_SHOTS=10 SHOT_EXP=1 bash submit_training_compute.sh
#
#   Chain training → inference:
#       TRAIN_JOB=$(STRATEGY=regular DATASET=colon bash submit_training_compute.sh --print-id)
#       hold_job_id="$TRAIN_JOB" dataset=colon strategy=regular icl_shots=0 bash submit_inference_compute.sh
# ============================================================

# ------------------------------------------------------------
# 1. Job Configuration (edit / override via env)
# ------------------------------------------------------------
DATASET=endo_binary    # colon | chest | endo | endo_binary | chest_binary | colon_binary
STRATEGY=regular     # regular | two_token | ed_ft | id_ft | lf_ft
NUM_TRAIN_EPOCHS=5
# ── ICL during training ──────────────────────────────────────────────────────
# ICL_SHOTS: in-context example images+labels embedded in each training prompt.
#            0 = standard supervised fine-tuning (no ICL context).
# ICL_POOL_PATH: optional separate JSON pool. Empty = use the training file itself.
# NOTE: chest's 19 class-definitions make the instruction long (~450 tok); each ICL
#       shot repeats it, so high ICL_SHOTS can exceed --model_max_length (2048).
#       Keep ICL_SHOTS small (≈1) for chest unless you also raise MODEL_MAX_LENGTH.
ICL_SHOTS=0
ICL_POOL_PATH="${ICL_POOL_PATH:-}"
num_samples=5   # 0 = ALL samples
LORA_R=8
LORA_ALPHA=32

# Validation modes: comma-separated subset of fixed,original,fresh
#   original              → only text-gen F1, fastest
#   fixed,original,fresh  → all 3 symbol modes (default, mirrors ICI)
VALIDATION_MODES=original
# true = full MedFMC-aligned eval (colon AUC; chest/endo mAP+AUC); false = macro_F1 only
COMPUTE_VAL_AUC_MAP="${COMPUTE_VAL_AUC_MAP:-true}"
# Batched per-class VALIDATION scoring (chest/endo). >1 enables left-padded batched
# P(Yes) scoring; 1 = original unbatched (default, byte-identical). Forwarded as
# SPRINT_PROBE_BATCH_SIZE (read by validation.py). >1 logs [BATCH-VERIFY] OK.
probe_batch_size=1

kv_cache=true
# Validation subsample cap. Also forms the checkpoint folder val-tag.
MAX_VAL_SAMPLES=5  # 0 = use all
# EVAL_DATA_PATH=none disables validation entirely (forwarded so the val-tag is right).
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
# RUN_TAG: optional human label appended to the checkpoint folder (readability only).
RUN_TAG="${RUN_TAG:-}"

# Training data mode — set exactly one:
#   TRAIN_PERCENT               → percentage of trainval.txt (default: 100)
#   TRAINING_SHOTS + SHOT_EXP   → MedFMC repeated-experiment split (paper protocol)
#   RANDOM_SHOT + SHOT_SEED     → custom random N-shot sampling
TRAINING_SHOTS=10 # e.g. 10  (use with SHOT_EXP for paper protocol)
SHOT_EXP=1             # experiment index 1-5
RANDOM_SHOT="${RANDOM_SHOT:-}"        # empty = not in random-shot mode
SHOT_SEED="${SHOT_SEED:-1}"           # RNG seed for random-shot sampling
if [ -z "${RANDOM_SHOT}" ] && [ -z "${TRAINING_SHOTS}" ]; then
    TRAIN_PERCENT="${TRAIN_PERCENT:-100}"
else
    TRAIN_PERCENT="${TRAIN_PERCENT:-}"
fi

# Set to a job ID (e.g. "12345.eehpc") to wait for that job first.
hold_job_id=

# ------------------------------------------------------------
# 2. EE PBS / Queue / Conda settings
# ------------------------------------------------------------
queue_name="${queue_name:-GPU_only}"        # EE GPU queue (others: workq, CPU_only)
hostname=compute                    # EMPTY = let 'compute' pick the node; set to pin
# GPU SELECTION: the 'compute' node has 8 GPUs (0-7). Pick a FREE one (check with
# `nvidia-smi` on the node) and pass it, e.g.  cuda_device=3 bash submit_training_compute.sh
cuda_device=0
walltime="${walltime:-48:00:00}"
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
FINETUNE_SCRIPT="${LLAVA_DIR}/sprint_vision/run_sprint_finetune.sh"
BASE_OUTPUT_DIR="/home/harinisrireddykandula/llava"   # mirrors ICI BASE_OUTPUT_DIR
output_dir="${BASE_OUTPUT_DIR}/logs"

# ------------------------------------------------------------
# 4. Automatic setup
# ------------------------------------------------------------
CURRENT_DATETIME=$(_ts "%d%m_%H%M")
TODAY=$(_ts "%Y-%m-%d")
if [ -n "${RANDOM_SHOT}" ]; then
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_rshot${RANDOM_SHOT}_s${SHOT_SEED}_${NUM_TRAIN_EPOCHS}ep"
elif [ -n "${TRAINING_SHOTS}" ] && [ -n "${SHOT_EXP}" ]; then
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_shot${TRAINING_SHOTS}_exp${SHOT_EXP}_${NUM_TRAIN_EPOCHS}ep"
elif [ -n "${TRAINING_SHOTS}" ]; then
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_shot${TRAINING_SHOTS}_${NUM_TRAIN_EPOCHS}ep"
else
    RUN_NAME="${CURRENT_DATETIME}_train_${DATASET}_${STRATEGY}_${TRAIN_PERCENT}pct_${NUM_TRAIN_EPOCHS}ep"
fi
# Append ICL context suffix when training uses ICL (ICL_SHOTS=0 is the standard baseline)
[ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null && RUN_NAME="${RUN_NAME}_icl${ICL_SHOTS}"

# ── Checkpoint OUTPUT_DIR (computed HERE so chained inference jobs know the exact
#    path at submit time). Mirrors run_sprint_finetune.sh's DATA_SUFFIX + val-tag
#    naming; exported below so run_sprint_finetune.sh uses it verbatim. ──────────
if [ -n "${TRAIN_PERCENT}" ]; then
    DATA_SUFFIX="percent${TRAIN_PERCENT}"
elif [ -n "${RANDOM_SHOT}" ]; then
    DATA_SUFFIX="rshot${RANDOM_SHOT}_seed${SHOT_SEED}"
elif [ -n "${SHOT_EXP}" ]; then
    DATA_SUFFIX="shot${TRAINING_SHOTS}_exp${SHOT_EXP}"
else
    DATA_SUFFIX="shot${TRAINING_SHOTS}"
fi
if [ "${EVAL_DATA_PATH}" = "none" ]; then
    VAL_TAG="valoff"
elif [ "${MAX_VAL_SAMPLES}" = "0" ]; then
    VAL_TAG="valall"
else
    VAL_TAG="val${MAX_VAL_SAMPLES}"
fi
CKPT_RUN_NAME="llava-${DATASET}-${STRATEGY}-${DATA_SUFFIX}"
[ "${ICL_SHOTS}" -gt 0 ] 2>/dev/null && CKPT_RUN_NAME="${CKPT_RUN_NAME}-icl${ICL_SHOTS}"
CKPT_RUN_TAG=""
[ -n "${RUN_TAG}" ] && CKPT_RUN_TAG="-${RUN_TAG}"
CKPT_TS=$(_ts "%m%d_%H%M")   # MMDD_HHMM — matches run_sprint_finetune.sh naming
# "checkpoints" (no suffix) is a symlink to /data2/.../llava/checkpoints, which
# is NOT mounted on the "compute" node the training job actually runs on (only
# on "master") -- writing there crashes with FileNotFoundError at trainer init.
# "checkpoints_local" is a real directory on /home, reachable from compute.
# submit_train_and_relocate.sh moves the finished run to /data2 afterward, from
# master (which can see both paths) -- see that script for the automated version.
OUTPUT_DIR="${BASE_OUTPUT_DIR}/checkpoints_local/${CKPT_TS}_${CKPT_RUN_NAME}_${VAL_TAG}${CKPT_RUN_TAG}"

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
echo "SPRInT-LLaVA Training Configuration  (EE / ${queue_name})"
echo "=========================================="
echo "Run Name:    ${RUN_NAME}"
echo "Dataset:     ${DATASET}"
echo "Strategy:    ${STRATEGY}"
if [ -n "${RANDOM_SHOT}" ]; then
    echo "Few-shot:    ${RANDOM_SHOT}-shot  (seed=${SHOT_SEED})"
elif [ -n "${TRAINING_SHOTS}" ] && [ -n "${SHOT_EXP}" ]; then
    echo "Few-shot:    ${TRAINING_SHOTS}-shot  exp${SHOT_EXP}  (MedFMC official split)"
elif [ -n "${TRAINING_SHOTS}" ]; then
    echo "Few-shot:    ${TRAINING_SHOTS}-shot  (legacy train_${TRAINING_SHOTS}.txt)"
else
    echo "Train %:     ${TRAIN_PERCENT}%  (of trainval.txt)"
fi
echo "Epochs:      ${NUM_TRAIN_EPOCHS}"
echo "ICL Shots:   ${ICL_SHOTS}"
echo "ICL Pool:    ${ICL_POOL_PATH:-<training file (self-pool)>}"
echo "Num Samples: ${num_samples} (0 = ALL)"
echo "LoRA r:      ${LORA_R}   alpha: ${LORA_ALPHA}"
echo "Val modes:   ${VALIDATION_MODES}"
echo "Val samples: ${MAX_VAL_SAMPLES} (0 = all)"
echo "Val AUC/mAP: ${COMPUTE_VAL_AUC_MAP}"
echo "Val probe batch: ${probe_batch_size} (>1 = batched per-class scoring)"
echo "KV-cache:    ${kv_cache} (true = fast prefix-cached chest/endo validation)"
echo "Checkpoint:  ${OUTPUT_DIR}"
echo "Queue:       ${queue_name}"
echo "Host:        ${HOST_MSG}"
echo "Resource:    ${RESOURCE}"
echo "GPU (CUDA):  ${cuda_device}   (compute node has GPUs 0-7; choose a free one)"
echo "Conda:       ${CONDA_BASE}  (env: ${CONDA_ENV})"
echo "Walltime:    ${walltime}"
echo "Dependency:  ${HOLD_MSG}"
echo "Log File:    ${LOG_FILE}"
echo "=========================================="

if [[ "${1}" == "--dry-run" ]]; then
    echo "DRY-RUN — no job submitted"
    exit 0
fi

# ------------------------------------------------------------
# 5. Submit (heredoc | tee, ICI-style). All run_sprint_finetune.sh env vars are
#    inlined at submit time (VALIDATION_MODES has commas → unsafe via qsub -v).
# ------------------------------------------------------------
JOB_ID=$(qsub -q "${queue_name}" \
    $HOLD_FLAG \
    -N "${RUN_NAME}" \
    -l ${RESOURCE} \
    -l walltime=${walltime} \
    -o /dev/null \
    -j oe \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},PYTHONUNBUFFERED=1,PYTHONIOENCODING=utf-8,LOG_FILE=${LOG_FILE},TZ=Asia/Kolkata \
    -S /bin/bash << EOF
#!/bin/bash
set -e
set -o pipefail

cd ${LLAVA_DIR}
export TZ='Asia/Kolkata'
# Login-node clock is ~5.5h fast → stale .pyc can shadow freshly-copied .py edits.
# Never cache bytecode, so Python always runs the current source.
export PYTHONDONTWRITEBYTECODE=1

echo "=== Training job started at: \$(date) ==="
echo "=== Running on: \$(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES} ==="
echo "Dataset: ${DATASET}  Strategy: ${STRATEGY}  Epochs: ${NUM_TRAIN_EPOCHS}"

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Pin the chosen GPU (PBS doesn't schedule GPUs here, so set it explicitly).
export CUDA_VISIBLE_DEVICES=${cuda_device}

export LD_PRELOAD=""
export LD_LIBRARY_PATH="${CONDA_BASE}/envs/${CONDA_ENV}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"

nvidia-smi

DATASET=${DATASET} \
STRATEGY=${STRATEGY} \
TRAIN_PERCENT=${TRAIN_PERCENT} \
TRAINING_SHOTS=${TRAINING_SHOTS} \
SHOT_EXP=${SHOT_EXP} \
RANDOM_SHOT=${RANDOM_SHOT} \
SHOT_SEED=${SHOT_SEED} \
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS} \
ICL_SHOTS=${ICL_SHOTS} \
ICL_POOL_PATH=${ICL_POOL_PATH} \
MAX_TRAIN_SAMPLES=${num_samples} \
LORA_R=${LORA_R} \
LORA_ALPHA=${LORA_ALPHA} \
VALIDATION_MODES=${VALIDATION_MODES} \
COMPUTE_VAL_AUC_MAP=${COMPUTE_VAL_AUC_MAP} \
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES} \
EVAL_DATA_PATH=${EVAL_DATA_PATH} \
SPRINT_PROBE_BATCH_SIZE=${probe_batch_size} \
SPRINT_KV_CACHE=${kv_cache} \
OUTPUT_DIR=${OUTPUT_DIR} \
LLAVA_DIR=${LLAVA_DIR} \
bash ${FINETUNE_SCRIPT} 2>&1 | tee \${LOG_FILE}

echo "=== Training job completed at: \$(date) ==="
EOF
)

echo ""
echo "=========================================="
echo "Training Job Submitted!"
echo "  Job ID:  ${JOB_ID}"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Status:  qstat -u \$(whoami)"
echo ""
echo "To chain inference (0 / 1 / 5 shots) after this job:"
echo "  CHECKPOINT_PATH=${OUTPUT_DIR}"
echo ""
echo "  JOB_A=\$(hold_job_id=${JOB_ID} CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=0 bash submit_inference_compute.sh --print-id)"
echo "  JOB_B=\$(hold_job_id=\$JOB_A   CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=1 bash submit_inference_compute.sh --print-id)"
echo "         hold_job_id=\$JOB_B   CHECKPOINT_PATH=\$CHECKPOINT_PATH dataset=${DATASET} strategy=${STRATEGY} icl_shots=5 bash submit_inference_compute.sh"
echo "=========================================="

# ------------------------------------------------------------
# 6. Auto-relocate watcher (background, detached).
#
# "checkpoints_local" (see OUTPUT_DIR above) is a real directory on /home,
# reachable from "compute" -- where training actually runs. The PERMANENT
# home for finished checkpoints is /data2/.../llava/checkpoints, which is
# mounted on "master" (this script's own host) but NOT on "compute"
# (confirmed via direct ssh: mount/df/ls all empty there, mkdir /data2 fails
# with Permission denied) -- so training can't write there directly.
#
# This background task polls qstat (from master, which CAN reach /data2)
# until the job leaves the queue, then moves the finished checkpoint over.
# It's spawned detached (disown + /dev/null stdin) so it survives this
# script exiting and keeps running even if this shell session ends, and it
# does NOT block this script's own return -- --print-id / hold_job_id
# chaining below is unaffected, same as before this was added.
# ------------------------------------------------------------
DATA2_CKPT_ROOT="/data2/harinisrireddykandula/llava/checkpoints"
RELOCATE_LOG="${LOG_DIR}/${RUN_NAME}_relocate.log"
(
    {
        echo "[relocate] watching job ${JOB_ID} -> will move ${OUTPUT_DIR}"
        while qstat -u "$(whoami)" 2>/dev/null | grep -q "${JOB_ID}"; do
            sleep 60
        done
        echo "[relocate] job ${JOB_ID} left the queue at $(date)"
        if [ -d "${OUTPUT_DIR}" ]; then
            mkdir -p "${DATA2_CKPT_ROOT}"
            DEST="${DATA2_CKPT_ROOT}/$(basename "${OUTPUT_DIR}")"
            echo "[relocate] moving ${OUTPUT_DIR} -> ${DEST}"
            mv "${OUTPUT_DIR}" "${DEST}"
            echo "[relocate] done: ${DEST}"
        else
            echo "[relocate] WARNING: ${OUTPUT_DIR} does not exist -- job likely failed before creating a checkpoint. Nothing to move."
        fi
    } >> "${RELOCATE_LOG}" 2>&1
) </dev/null &
disown
echo "Auto-relocate watcher started (background, detached) -- log: ${RELOCATE_LOG}"
echo "  (moves the finished checkpoint to ${DATA2_CKPT_ROOT}/ once the job completes)"
echo "=========================================="

if [[ "${1}" == "--print-id" ]]; then
    echo "${JOB_ID}"
fi
