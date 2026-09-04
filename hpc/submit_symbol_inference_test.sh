#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  TEST-SPLIT INFERENCE — FINAL PAPER NUMBERS
#  Copy of submit_symbol_inference_node1.sh with every final checkpoint listed
#  below, grouped by table/experiment. Uncomment EXACTLY ONE preset line
#  (CKPT_DATE + TRAIN_RUN + EPOCH), set the per-section run vars shown in each
#  header, then run this script. All presets default to SPLIT=test.
#  Epochs = best training-original epoch per seed (or the only saved epoch).
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

MODEL_TYPE="${MODEL_TYPE:-flamingo}"  # flamingo | qwen | salmonn
_DEFAULT_CONDA_ENV="qwen"
if [[ "${MODEL_TYPE}" == "flamingo" ]]; then _DEFAULT_CONDA_ENV="flamingo"; fi
if [[ "${MODEL_TYPE}" == "salmonn" ]];  then _DEFAULT_CONDA_ENV="salmonn";  fi
CONDA_ENV="${CONDA_ENV:-${_DEFAULT_CONDA_ENV}}"

# Eval battery — trained (reference): hvb, cremad, minds14_en | held-out: skit_s2i,
# ravdess_song, minds14_fr, minds14_ko (+ speech_commands probe: only its `original`
# column is meaningful). heysquad/sprsound optional (QA / shelved-medical).
DATASET_TYPE="${DATASET_TYPE:-hvb-cremad-minds14_en-ravdess_song-skit_s2i-minds14_fr-minds14_ko-speech_commands-sprsound-heysquad}"
# Clustering eval (9 = fr/ko/skit × {meaningful,neutral,symbol}):
# DATASET_TYPE="${DATASET_TYPE:-minds14_fr_cat_mean-minds14_fr_cat_neu-minds14_fr_cat_sym-minds14_ko_cat_mean-minds14_ko_cat_neu-minds14_ko_cat_sym-skit_cat_mean-skit_cat_neu-skit_cat_sym}"
DEVICE="${DEVICE:-cuda:0}"

# ══════════════════════════════════════════════════════════════════════════════
#  (1) MAIN TABLE — AF3, 3-task (hvb+cremad+minds14_en), eval-A, TEST split.
#      RUN EACH WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
# ──────────────────────────────────────────────────────────────────────────────
#  Untrained (n=1):  UNTRAINED=true MODEL_TYPE=flamingo
#  Instruction Tuning / nosym (n=3)
# CKPT_DATE=2026-08-02 TRAIN_RUN=112544_af_h-cr-m14en_nosym EPOCH=4  # AF3 IT   seed1
# CKPT_DATE=2026-08-02 TRAIN_RUN=112555_af_h-cr-m14en_nosym EPOCH=5  # AF3 IT   seed2
# CKPT_DATE=2026-08-28 TRAIN_RUN=235526_af_h-cr-m14en_nosym EPOCH=5  # AF3 IT   seed3 (NEW)
#  DGST / dpi_ha (n=3)
# CKPT_DATE=2026-08-02 TRAIN_RUN=112859_af_h-cr-m14en_dpi_ha EPOCH=4  # AF3 DGST seed1
# CKPT_DATE=2026-08-02 TRAIN_RUN=112912_af_h-cr-m14en_dpi_ha EPOCH=5  # AF3 DGST seed2
# CKPT_DATE=2026-08-02 TRAIN_RUN=113131_af_h-cr-m14en_dpi_ha EPOCH=1  # AF3 DGST seed3
#  Symbol Tuning (Wei) = dpi_ha_nl_fs1t, evaluated eval-A (n=3)
# CKPT_DATE=2026-08-06 TRAIN_RUN=185105_af_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=4  # AF3 Wei seed1
# CKPT_DATE=2026-08-06 TRAIN_RUN=185112_af_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=5  # AF3 Wei seed2
# CKPT_DATE=2026-08-06 TRAIN_RUN=185119_af_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=5  # AF3 Wei seed3
#  Flipped-label Tuning (Agrawal) = nosym_swap (n=3)  [== section (4); main-table row]
# CKPT_DATE=2026-08-29 TRAIN_RUN=003509_af_h-cr-m14en_nosym_swap_per_instance EPOCH=2  # AF3 flip-train seed1
# CKPT_DATE=2026-08-29 TRAIN_RUN=003524_af_h-cr-m14en_nosym_swap_per_instance EPOCH=4  # AF3 flip-train seed2
CKPT_DATE=2026-08-29 TRAIN_RUN=003530_af_h-cr-m14en_nosym_swap_per_instance EPOCH=3  # AF3 flip-train seed3
# ══════════════════════════════════════════════════════════════════════════════
#  (2) MAIN TABLE — Qwen2-Audio, 3-task, eval-A, TEST split.
#      RUN EACH WITH:  MODEL_TYPE=qwen INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
# ──────────────────────────────────────────────────────────────────────────────
#  Untrained (n=1):  UNTRAINED=true MODEL_TYPE=qwen
#  Instruction Tuning / nosym (n=3)
# CKPT_DATE=2026-08-23 TRAIN_RUN=004932_qa_h-cr-m14en_nosym EPOCH=3  # Qwen IT   seed1
# CKPT_DATE=2026-08-23 TRAIN_RUN=005022_qa_h-cr-m14en_nosym EPOCH=4  # Qwen IT   seed2
# CKPT_DATE=2026-08-28 TRAIN_RUN=235131_qa_h-cr-m14en_nosym EPOCH=3  # Qwen IT   seed3 (NEW)
#  DGST / dpi_ha (n=3)
# CKPT_DATE=2026-08-23 TRAIN_RUN=003002_qa_h-cr-m14en_dpi_ha EPOCH=3  # Qwen DGST seed1
# CKPT_DATE=2026-08-23 TRAIN_RUN=003016_qa_h-cr-m14en_dpi_ha EPOCH=2  # Qwen DGST seed2
# CKPT_DATE=2026-08-23 TRAIN_RUN=003227_qa_h-cr-m14en_dpi_ha EPOCH=3  # Qwen DGST seed3
#  Symbol Tuning (Wei) = dpi_ha_nl_fs1t, evaluated eval-A (n=3)
# CKPT_DATE=2026-08-23 TRAIN_RUN=004014_qa_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=4  # Qwen Wei seed1
# CKPT_DATE=2026-08-23 TRAIN_RUN=004029_qa_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=3  # Qwen Wei seed2
# CKPT_DATE=2026-08-28 TRAIN_RUN=235404_qa_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=3  # Qwen Wei seed3 (NEW)
#  Flipped-label Tuning (Agrawal) = nosym_swap (n=3)  [main-table row]
# CKPT_DATE=2026-09-03 TRAIN_RUN=014617_qa_h-cr-m14en_nosym_swap_per_instance EPOCH=2  # Qwen flip-train seed1
# CKPT_DATE=2026-09-03 TRAIN_RUN=014636_qa_h-cr-m14en_nosym_swap_per_instance EPOCH=3  # Qwen flip-train seed2
# CKPT_DATE=2026-09-03 TRAIN_RUN=014804_qa_h-cr-m14en_nosym_swap_per_instance EPOCH=4  # Qwen flip-train seed3
# ══════════════════════════════════════════════════════════════════════════════
#  (3) STEERABILITY — AF3 FLIPPED eval (deranged label↔definition).
#      RUN EACH WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,flipped NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
#      (reuses the main-table AF3 DGST / IT ckpts + untrained)
# ──────────────────────────────────────────────────────────────────────────────
#   DGST: 112859 / 112912 / 113131  · IT: 112544 / 112555 / 235526  · Untrained
# ══════════════════════════════════════════════════════════════════════════════
#  (4) FLIPPED-TRAINING baseline (Agrawal-style) — AF3, eval-A, TEST split (n=3).
#      Trained on deranged real labels + definitions (NO_SYMBOLS + SWAP_LABELS).
#      RUN EACH WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
# ──────────────────────────────────────────────────────────────────────────────
# CKPT_DATE=2026-08-29 TRAIN_RUN=003509_af_h-cr-m14en_nosym_swap_per_instance EPOCH=2  # AF3 flip-train seed1
# CKPT_DATE=2026-08-29 TRAIN_RUN=003524_af_h-cr-m14en_nosym_swap_per_instance EPOCH=4  # AF3 flip-train seed2
# CKPT_DATE=2026-08-29 TRAIN_RUN=003530_af_h-cr-m14en_nosym_swap_per_instance EPOCH=3  # AF3 flip-train seed3
# ══════════════════════════════════════════════════════════════════════════════
#  (4b) STATIC SYMBOL TUNING (fix_ha) — AF3 3-task, dynamic-vs-static ablation (n=2).
#      Fixed label→symbol map (DYNAMIC_SYMBOLS=false). eval-A, TEST split.
#      RUN EACH WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
# ──────────────────────────────────────────────────────────────────────────────
# CKPT_DATE=2026-08-02 TRAIN_RUN=112629_af_h-cr-m14en_fix_ha EPOCH=3  # AF3 static seed1
# CKPT_DATE=2026-08-02 TRAIN_RUN=112738_af_h-cr-m14en_fix_ha EPOCH=3  # AF3 static seed2
#  NOTE: breadth curve (1/2-task) + full static ablation stay on VAL (many runs;
#  ablation, not headline). Only run these on test if you want the static row to
#  match the main-table test numbers.
# ══════════════════════════════════════════════════════════════════════════════
#  (5) IT-VARIANT ABLATION — AF3 (real labels; ± demonstrations, ± definitions).
#      Plain IT = section (1) nosym. Variants below, all eval-A, TEST split.
#      RUN EACH WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
# ──────────────────────────────────────────────────────────────────────────────
#  IT + demonstrations  (nosym_fs1t, real labels + legend + 1 demo/class)
# CKPT_DATE=2026-08-07 TRAIN_RUN=100621_af_h-cr-m14en_nosym_fs1t EPOCH=5  # IT+demos seed1
# CKPT_DATE=2026-08-07 TRAIN_RUN=100738_af_h-cr-m14en_nosym_fs1t EPOCH=5  # IT+demos seed2
# CKPT_DATE=2026-08-07 TRAIN_RUN=100829_af_h-cr-m14en_nosym_fs1t EPOCH=5  # IT+demos seed3
#  IT + demonstrations, no definitions  (nosym_nl_fs1t)
# CKPT_DATE=2026-08-06 TRAIN_RUN=185143_af_h-cr-m14en_nosym_nl_fs1t EPOCH=5  # IT+demos,nodef seed1
# CKPT_DATE=2026-08-07 TRAIN_RUN=100544_af_h-cr-m14en_nosym_nl_fs1t EPOCH=4  # IT+demos,nodef seed2
# CKPT_DATE=2026-08-07 TRAIN_RUN=100549_af_h-cr-m14en_nosym_nl_fs1t EPOCH=5  # IT+demos,nodef seed3
# ══════════════════════════════════════════════════════════════════════════════
#  (6) DEFINITIONS vs DEMONSTRATIONS  (tab:def-vs-demo, held-out SYMBOLIC = fresh col).
#      Baselines shown "demonstrations only, no definitions"; DGST shown across all 3
#      prompt-ingredient settings on the SAME model (defs-only / demos-only / both).
#      Def.=legend on (NO_LEGEND=false); Demo.=1 example/class (NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true).
#   ── Baselines "demonstrations only" (NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=true):  DONE
# CKPT_DATE=2026-08-06 TRAIN_RUN=185105_af_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=4  # Wei demos-only seed1 (done)
# CKPT_DATE=2026-08-06 TRAIN_RUN=185112_af_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=5  # Wei demos-only seed2 (done)
# CKPT_DATE=2026-08-06 TRAIN_RUN=185119_af_h-cr-m14en_dpi_ha_nl_fs1t EPOCH=5  # Wei demos-only seed3 (done)
#   ── DGST "demonstrations only"  RUN WITH: MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#        VALIDATION_MODES=original,fresh NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=true SPLIT=test
# CKPT_DATE=2026-08-02 TRAIN_RUN=112859_af_h-cr-m14en_dpi_ha EPOCH=4  # DGST demos-only seed1
# CKPT_DATE=2026-08-02 TRAIN_RUN=112912_af_h-cr-m14en_dpi_ha EPOCH=5  # DGST demos-only seed2
# CKPT_DATE=2026-08-02 TRAIN_RUN=113131_af_h-cr-m14en_dpi_ha EPOCH=1  # DGST demos-only seed3
#   ── DGST "definitions + demonstrations"  RUN WITH: MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#        VALIDATION_MODES=original,fresh NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=false SPLIT=test
# CKPT_DATE=2026-08-02 TRAIN_RUN=112859_af_h-cr-m14en_dpi_ha EPOCH=4  # DGST def+demos seed1
# CKPT_DATE=2026-08-02 TRAIN_RUN=112912_af_h-cr-m14en_dpi_ha EPOCH=5  # DGST def+demos seed2
# CKPT_DATE=2026-08-02 TRAIN_RUN=113131_af_h-cr-m14en_dpi_ha EPOCH=1  # DGST def+demos seed3
#  (DGST "definitions only" = main table section (1), 0.663.)
#  NOTE: full 3x3 train×eval grid (DGST-B dpi_ha_fs1t 184309/319/454, etc.) = VAL-only
#  → appendix app:grid; do NOT re-run on test.
# ══════════════════════════════════════════════════════════════════════════════
#  (7) k-SWEEP (does IT catch DGST with more demos?) — IT ckpts ONLY, held-out SYMBOLIC.
#      IT ckpts 112544/112555/235526. DGST 0.663 = flat reference, NOT swept. For each k:
#        MODEL_TYPE=flamingo INPUT_MODE=speech_only VALIDATION_MODES=fresh \
#        NUM_EXAMPLES=<k> FEWSHOT_PER_CLASS=true NO_LEGEND=false FEWSHOT_MODE=text SPLIT=test
#      k=0 = main-table IT (done).  k=1 = IT eval-B _fs1tpc (DONE, all 3 seeds — same config).
#      ONLY k=2 and k=3 remain → 6 runs.
# ══════════════════════════════════════════════════════════════════════════════
#  (8) CLUSTERING — user-defined categories. Uncomment the clustering DATASET_TYPE line above
#      (and comment the main battery). All 9 style-datasets (fr/ko/skit × mean/neu/sym) run in
#      ONE pass per ckpt; the _m.json auto-gives per-style numbers. Metric = accuracy.
#      RUN EACH CKPT TWICE (audio + text):
#        MODEL_TYPE=flamingo VALIDATION_MODES=original NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test \
#          INPUT_MODE=speech_only     # audio
#          INPUT_MODE=text_only       # text (repeat)
#      13 ckpts/modality: DGST 112859/112912/113131 · IT 112544/112555/235526 ·
#             Wei 185105/112/119 · Flip 003509/003524/003530 · Untrained (UNTRAINED=true).
#      × 2 modalities (audio speech_only + text text_only) = 26 runs.
# ══════════════════════════════════════════════════════════════════════════════
#  (9) BREADTH CURVE — DGST trained on 1 / 2 / 3 tasks (nested), TEST split.
#      Read ONLY the held-out-4 fresh column {skit_s2i,ravdess_song,minds14_fr,minds14_ko}
#      — the SAME fixed probe at every breadth point (never "remaining N tasks").
#      3-task point = main-table DGST (112859/112912/113131), already on test.
#      RUN EACH WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only \
#                      VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test
# ──────────────────────────────────────────────────────────────────────────────
#  -- DGST (dpi_ha) --
#  1-task = cremad (n=3)
# CKPT_DATE=2026-08-03 TRAIN_RUN=092228_af_cr_dpi_ha EPOCH=5  # DGST 1-task seed1
# CKPT_DATE=2026-08-03 TRAIN_RUN=092309_af_cr_dpi_ha EPOCH=5  # DGST 1-task seed2
# CKPT_DATE=2026-08-03 TRAIN_RUN=092446_af_cr_dpi_ha EPOCH=4  # DGST 1-task seed3
#  2-task = hvb+cremad (n=3)   [best trained-original epoch per seed]
# CKPT_DATE=2026-07-31 TRAIN_RUN=090352_af_h-cremad_dpi_ha EPOCH=2  # DGST 2-task seed1
# CKPT_DATE=2026-07-31 TRAIN_RUN=090402_af_h-cremad_dpi_ha EPOCH=5  # DGST 2-task seed2
# CKPT_DATE=2026-07-31 TRAIN_RUN=090409_af_h-cremad_dpi_ha EPOCH=3  # DGST 2-task seed3
#  3-task = hvb+cremad+minds14_en = section (1) main-table DGST (no re-run needed).
#  -- IT (nosym) comparison line --  (only best epoch is saved per seed → use it)
#  1-task = cremad (n=3)
# CKPT_DATE=2026-07-22 TRAIN_RUN=180126_af_cremad_nosym EPOCH=2  # IT 1-task seed1
# CKPT_DATE=2026-07-22 TRAIN_RUN=180214_af_cremad_nosym EPOCH=3  # IT 1-task seed2
# CKPT_DATE=2026-07-22 TRAIN_RUN=180547_af_cremad_nosym EPOCH=2  # IT 1-task seed3
#  2-task = hvb+cremad (n=2 only — no more seeds on disk)
# CKPT_DATE=2026-07-30 TRAIN_RUN=201011_af_h-cremad_nosym EPOCH=3  # IT 2-task seed1
# CKPT_DATE=2026-07-30 TRAIN_RUN=201026_af_h-cremad_nosym EPOCH=2  # IT 2-task seed2
#  3-task IT = section (1) main-table nosym 112544/112555/235526 (no re-run needed).
# ══════════════════════════════════════════════════════════════════════════════
#  (10) FAITHFUL AGRAWAL — Flip ckpts, TOTAL-k demonstrations + definitions (NOT per-class).
#      Agrawal (our prior, arXiv:2505.07731) = definitions + total-k text demos, few-shot.
#      Held-out SYMBOLIC (fresh). k=0 point = Flip def-only 0.534 (main table, no re-run).
#      Shows Agrawal(Flip) rising with k vs DGST flat zero-shot 0.663.
#      RUN EACH k WITH:  MODEL_TYPE=flamingo INPUT_MODE=speech_only VALIDATION_MODES=fresh \
#        NUM_EXAMPLES=<k> FEWSHOT_PER_CLASS=false NO_LEGEND=false FEWSHOT_MODE=text SPLIT=test
#      (FEWSHOT_PER_CLASS=false => total-k, the Agrawal protocol; run tag ...fs<k>t (no 'pc'))
#      k in {1, 5, 10} — spans Agrawal's own 0-5 range and beyond; k=0 = Flip def-only 0.534.
# ──────────────────────────────────────────────────────────────────────────────
# CKPT_DATE=2026-08-29 TRAIN_RUN=003509_af_h-cr-m14en_nosym_swap_per_instance EPOCH=2  # Agrawal/Flip seed1
# CKPT_DATE=2026-08-29 TRAIN_RUN=003524_af_h-cr-m14en_nosym_swap_per_instance EPOCH=4  # Agrawal/Flip seed2
# CKPT_DATE=2026-08-29 TRAIN_RUN=003530_af_h-cr-m14en_nosym_swap_per_instance EPOCH=3  # Agrawal/Flip seed3
#  Run all 3 seeds at each k in {1,5,10} = 9 runs.
# ══════════════════════════════════════════════════════════════════════════════
#  (11) DEFINITION-SENSITIVITY ABLATION  [MECHANISM WIRED — just add variant text to configs]
#      Q: does performance depend on definition WORDING? Vary the legend text (e.g.
#      paraphrase / terse) with labels & task fixed; held-out symbolic; expect ~stable
#      => robustness (plugs reviewer "def quality?" hole).
#      Run for BOTH our variants: DGST 112859/112912/113131 AND Flip 003509/003524/003530.
#      ONE-TIME SETUP: in the configs you want to test, add
#        prompt_variants={"paraphrase": _PROMPT_PARA, "terse": _PROMPT_TERSE}
#        (reword ONLY text after each "- label:"; keep label names = symbol-swap anchors).
#      THEN RUN (per variant, per ckpt):
#        MODEL_TYPE=flamingo INPUT_MODE=speech_only VALIDATION_MODES=original,fresh \
#          NUM_EXAMPLES=0 NO_LEGEND=false SPLIT=test DEFINITION_VARIANT=paraphrase
#      Run name gets a _dv<variant> tag so variants don't collide with orig runs.
#      (clustering's mean/neu/sym tests NAME sensitivity — complementary, NOT def wording.)
# ══════════════════════════════════════════════════════════════════════════════

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
TRAIN_RUN="${TRAIN_RUN:-112859_af_h-cr-m14en_dpi_ha}"
EPOCH="${EPOCH:-4}"
CKPT_DATE="${CKPT_DATE:-2026-08-02}"
UNTRAINED="${UNTRAINED:-false}"                                                  # true = run base model with NO checkpoint/LoRA
if [[ "${UNTRAINED}" == "true" ]]; then
    CHECKPOINT_PATH=""
elif [[ -z "${CHECKPOINT_PATH:-}" ]]; then
    CHECKPOINT_PATH=$(ls "${HOME}/training/symbol_training/checkpoints/${CKPT_DATE}/${TRAIN_RUN}/lora_epoch${EPOCH}"*.pt 2>/dev/null | head -1)
    if [[ -z "${CHECKPOINT_PATH}" ]]; then
        printf 'ERROR: no checkpoint found for TRAIN_RUN=%s EPOCH=%s CKPT_DATE=%s\n' "${TRAIN_RUN}" "${EPOCH}" "${CKPT_DATE}" >&2
        exit 1
    fi
fi
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"                                           # 0 = FULL test set (final numbers)
NUM_EXAMPLES="${NUM_EXAMPLES:-0}"                                                  # few-shot examples in eval prompt (0 = zero-shot)
FEWSHOT_MODE="${FEWSHOT_MODE:-text}"                                               # few-shot exemplar modality: text | speech
FEWSHOT_PER_CLASS="${FEWSHOT_PER_CLASS:-true}"
NUM_WORKERS="${NUM_WORKERS:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
VALIDATION_MODES="${VALIDATION_MODES:-original,fresh}"                            # original,fixed,fresh,flipped
VAL_FLIP_SEED="${VAL_FLIP_SEED:-0}"
NO_LEGEND="${NO_LEGEND:-false}"
DEFINITION_VARIANT="${DEFINITION_VARIANT:-verbose}"                                  # def-sensitivity ablation: orig | paraphrase | terse | verbose (set per run; keys, NOT the _PROMPT_* constant names)
INPUT_MODE="${INPUT_MODE:-speech_only}"                                           # speech_only (audio, main table) | text_only
SPLIT="${SPLIT:-test}"                                                            # TEST for final paper numbers

SYMBOL_MAP_FILE="${SYMBOL_MAP_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/training/symbol_training}"
METRICS_DIR="${METRICS_BASE:-${HOME}/training/symbol_training/metrics}/$(date +"%Y-%m-%d")"
LOG_DIR="${LOGS_INFERENCE_DIR:-${HOME}/training/symbol_training/logs_inference}/$(date +"%Y-%m-%d")"

SHORT_MODEL_TYPE="${MODEL_TYPE//qwen/qa}"
SHORT_MODEL_TYPE="${SHORT_MODEL_TYPE//salmonn/sl}"
SHORT_MODEL_TYPE="${SHORT_MODEL_TYPE//flamingo/af}"

SHORT_DATASET_TYPE="${DATASET_TYPE//voxceleb/vb}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//hvb/h}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//meld_emotion/me}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//voxpopuli/vp}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//cremad/cr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//ravdess_song/rs}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//skit_s2i/sk}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//speech_commands/sc}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_en/m14en}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_fr/m14fr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//minds14_ko/m14ko}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//sprsound/spr}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//heysquad/hsq}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//_cat_mean/Cme}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//_cat_neu/Cne}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//_cat_sym/Csy}"
SHORT_DATASET_TYPE="${SHORT_DATASET_TYPE//skit/sk}"

if [[ "${CHECKPOINT_PATH}" =~ epoch([0-9]+) ]]; then
    EPOCH_NUM="${BASH_REMATCH[1]}"
else
    EPOCH_NUM="X"
fi

CHECKPOINT_DIR_NAME=$(basename "$(dirname "${CHECKPOINT_PATH}")")
TRAIN_DATA_RAW=$(echo "${CHECKPOINT_DIR_NAME}" | cut -d'_' -f3)
TRAIN_DATA="${TRAIN_DATA_RAW//voxceleb/vb}"
TRAIN_DATA="${TRAIN_DATA//hvb/h}"
TRAIN_DATA="${TRAIN_DATA//meld_emotion/me}"
TRAIN_DATA="${TRAIN_DATA//voxpopuli/vp}"
[[ -z "${TRAIN_DATA}" ]] && TRAIN_DATA="unk"

[[ "${MAX_VAL_SAMPLES}" == "0" ]] && SAMPLES_TAG="" || SAMPLES_TAG="_${MAX_VAL_SAMPLES}"
[[ -n "${CHECKPOINT_PATH}" ]] && _CKPT_TAG="_tr${TRAIN_DATA}_ep${EPOCH_NUM}" || _CKPT_TAG="_untrained"
_SPLIT_TAG="${SPLIT:0:3}"
if [[ "${NUM_EXAMPLES}" == "0" ]]; then
    _FS_TAG="_sh0"
else
    _FS_TAG="_fs${NUM_EXAMPLES}${FEWSHOT_MODE:0:1}$( [[ "${FEWSHOT_PER_CLASS}" == "true" ]] && printf 'pc' || true )"
fi
[[ "${NO_LEGEND}" == "true" ]] && _FS_TAG="${_FS_TAG}_nl"
[[ "${DEFINITION_VARIANT}" != "orig" ]] && _FS_TAG="${_FS_TAG}_dv${DEFINITION_VARIANT}"
RUN_NAME="${RUN_NAME:-$(date +"%H%M%S")_i_${SHORT_MODEL_TYPE}_${SHORT_DATASET_TYPE}_${_SPLIT_TAG}${SAMPLES_TAG}${_CKPT_TAG}${_FS_TAG}}"

if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
else
    eval "$(/usr/bin/conda shell.bash hook)"
fi

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${METRICS_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

if [[ "${_NOHUP_LAUNCHED:-0}" != "1" ]]; then
    export _NOHUP_LAUNCHED=1 RUN_NAME
    nohup "$0" >> "${LOG_FILE}" 2>&1 &
    printf 'Inference launched in background (PID: %s)\n' "$!"
    printf 'Follow logs: tail -f "%s"\n' "${LOG_FILE}"
    exit 0
fi

printf '%s\n' "============================================================"
printf '%s\n' "Starting Symbol Inference on node1 (TEST split)"
printf '%s\n' "Project Root:    ${PROJECT_ROOT}"
printf '%s\n' "Conda Env:       ${CONDA_ENV}"
printf '%s\n' "Model:           ${MODEL_TYPE}"
printf '%s\n' "Dataset:         ${DATASET_TYPE}"
printf '%s\n' "Checkpoint:      ${CHECKPOINT_PATH}"
printf '%s\n' "Symbol Map:      ${SYMBOL_MAP_FILE:-<from checkpoint>}"
printf '%s\n' "Split:           ${SPLIT}"
printf '%s\n' "Samples:         ${SAMPLES_TAG:-<full>}"
printf '%s\n' "Metrics Dir:     ${METRICS_DIR}/${RUN_NAME}"
printf '%s\n' "Run Name:        ${RUN_NAME}"
printf '%s\n' "Log File:        ${LOG_FILE}"
printf '%s\n' "============================================================"

export CUDA_VISIBLE_DEVICES

python inference.py \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --device "${DEVICE}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --num_examples "${NUM_EXAMPLES}" \
    --fewshot_mode "${FEWSHOT_MODE}" \
    $( [[ "${FEWSHOT_PER_CLASS}" == "true" ]] && printf '%s' "--fewshot_per_class" ) \
    --num_workers "${NUM_WORKERS}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --validation_modes "${VALIDATION_MODES}" \
    --val_flip_seed "${VAL_FLIP_SEED}" \
    $( [[ "${NO_LEGEND}" == "true" ]] && printf '%s' "--no_legend" ) \
    --split "${SPLIT}" \
    --input_mode "${INPUT_MODE}" \
    --definition_variant "${DEFINITION_VARIANT}" \
    --metrics_dir "${METRICS_DIR}" \
    $( [[ -n "${SYMBOL_MAP_FILE}" ]] && printf '%s %s' "--symbol_map_file" "${SYMBOL_MAP_FILE}" ) \
    >> "${LOG_FILE}" 2>&1




# ┌─────┬───────────────────────────────────────────┬──────────────────────────────────┬───────┐
# │  #  │                    Run                    │              Config              │ Count │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ A   │ AF3 static seed2 (112629)                 │ eval-A original,fresh    (done)         │ 1     │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ B   │ AF3 flip-train steerability               │ original,flipped   (done)              │ 3     │
# │     │ (003509/524/530)                          │                                  │       │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ C   │ Wei eval-C (185105/112/119)   + DGST   +IT         │ fresh NUM_EXAMPLES=1   (done)          │ 6     │
# │     │                                           │ NO_LEGEND=true                   │       │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ C2  │ DGST def-vs-demo (112859/112912/113131):  │ NUM_EXAMPLES=1 FEWSHOT_PER_CLASS │ 9     │
# │     │  demos-only NO_LEGEND=true  +             │ =true original,fresh split=test  │       │
# │     │  def+demos  NO_LEGEND=false  (eval-B)     │ (fills tab:def-vs-demo)   (done)        │       │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ D   │ IT-variants §5 (nosym_fs1t×3 +            │ eval-A original,fresh            │ 6     │
# │     │ nosym_nl_fs1t×3)                          │                          (done)         │       │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ E   │ Clustering (DGST/IT/Wei/Flip ×3 + Untr    │ original NUM_EXAMPLES=0,         │ 26    │
# │     │ = 13/modality × 2 modalities)             │ audio+text        (done)             │       │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ F   │ k-sweep IT (112544/112555/235526 ×        │ fresh NUM_EXAMPLES=k, held-out   │ 6  (done)   │
# │     │ k=1,2,3)                                  │ symbolic                         │       │
# ├─────┼───────────────────────────────────────────┼──────────────────────────────────┼───────┤
# │ G   │ Qwen flip-train main (after training ×3)  │ eval-A original,fresh            │ 3  (done)   │
# H 1 task, 2 task 3 task.   11
# I  Faithful-Agrawal Flip TOTAL-k (§10): 003509/003524/003530 × k=1,5,10, fresh,
#    NUM_EXAMPLES=k FEWSHOT_PER_CLASS=false NO_LEGEND=false = 9 runs
# J  Definition-sensitivity (§11) [PLANNED, needs code]: DGST + Flip × def-wording
#    variants (orig/paraphrase/terse), fresh held-out. Robustness check.
# K  AGRAWAL-TRAINED (flip+demos, NEW ckpts AF3×3 + Qwen×3) — after training:
#    (K1) MAIN-TABLE row, ZERO-SHOT eval-A: 6 runs (AF3×3 + Qwen×3),
#         MODEL_TYPE=<af|qa> VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false
#    (K2) tab:agrawal FEW-SHOT (AF3 demo-trained only) × m in {1,5,10}: 9 runs (3 seeds×3 k),
#         MODEL_TYPE=flamingo VALIDATION_MODES=original,fresh NUM_EXAMPLES=m \
#         FEWSHOT_PER_CLASS=false NO_LEGEND=false FEWSHOT_MODE=text
#    Total = 15 (excl. SALMONN). m=0 row of tab:agrawal = our def-only Flip 0.534 (have).
# └─────┴───────────────────────────────────────────┴──────────────────────────────────┴───────┘