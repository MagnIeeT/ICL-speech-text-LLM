# Inference checklist — AF3 3×3 (trained-regime × eval-regime) + medical + QA
Model: flamingo. Split: validation (flip `SPLIT=test` for the held-out final number).
Battery `DATASET_TYPE` = default (hvb,cremad,m14en [trained] + ravdess,skit,speech_commands,m14fr,m14ko [held-out]).
**speech_commands: use ORIGINAL only — ignore its fresh column.**  All battery runs: `VALIDATION_MODES=original,fresh`.

Three eval conditions:
- **eval-A zero-shot+legend** → `NUM_EXAMPLES=0 NO_LEGEND=false`
- **eval-B fewshot+legend**   → `NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=false`
- **eval-C fewshot-only**     → `NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=true`

STATUS: A-trained row is DONE (eval-A/B/C all have inference; eval-A-fresh from logs). B- & C-trained rows have only their diagonal (from logs). The blocks below run the **remaining off-diagonals for B- and C-trained only = 24 runs** (12 ckpts × 2 non-native conditions each). Epochs = best-train-original.

--------------------------------------------------------------------------------
## Checkpoint registry (9 dpi + 8 nosym) — "CKPT_DATE TRAIN_RUN EPOCH"
```bash
# dpi
DPI_A=( "2026-08-02 112859_af_h-cr-m14en_dpi_ha 4" "2026-08-02 112912_af_h-cr-m14en_dpi_ha 5" "2026-08-02 113131_af_h-cr-m14en_dpi_ha 1" )        # A legend-only
DPI_B=( "2026-08-06 184309_af_h-cr-m14en_dpi_ha_fs1t 5" "2026-08-06 184319_af_h-cr-m14en_dpi_ha_fs1t 5" "2026-08-06 184454_af_h-cr-m14en_dpi_ha_fs1t 5" )   # B legend+fs
DPI_C=( "2026-08-06 185105_af_h-cr-m14en_dpi_ha_nl_fs1t 4" "2026-08-06 185112_af_h-cr-m14en_dpi_ha_nl_fs1t 5" "2026-08-06 185119_af_h-cr-m14en_dpi_ha_nl_fs1t 5" )  # C fewshot-only
# nosym
NSY_A=( "2026-08-02 112544_af_h-cr-m14en_nosym 4" "2026-08-02 112555_af_h-cr-m14en_nosym 5" )                                                    # A legend-only
NSY_B=( "2026-08-07 100621_af_h-cr-m14en_nosym_fs1t 5" "2026-08-07 100738_af_h-cr-m14en_nosym_fs1t 5" "2026-08-07 100829_af_h-cr-m14en_nosym_fs1t 5" )       # B legend+fs
NSY_C=( "2026-08-06 185143_af_h-cr-m14en_nosym_nl_fs1t 5" "2026-08-07 100544_af_h-cr-m14en_nosym_nl_fs1t 4" "2026-08-07 100549_af_h-cr-m14en_nosym_nl_fs1t 5" )  # C fewshot-only
```
Paste the registry once per shell, then run the eval blocks below.

--------------------------------------------------------------------------------
## EVAL-A  (zero-shot + legend)   B- & C-trained   [12 runs]
```bash
for E in "${DPI_B[@]}" "${DPI_C[@]}" "${NSY_B[@]}" "${NSY_C[@]}"; do set -- $E
  MODEL_TYPE=flamingo CKPT_DATE=$1 TRAIN_RUN=$2 EPOCH=$3 \
  VALIDATION_MODES=original,fresh NUM_EXAMPLES=0 NO_LEGEND=false \
  ./hpc/submit_symbol_inference_node1.sh; done
```

## EVAL-B  (fewshot + legend)   C-trained only   [6 runs]
```bash
for E in "${DPI_C[@]}" "${NSY_C[@]}"; do set -- $E
  MODEL_TYPE=flamingo CKPT_DATE=$1 TRAIN_RUN=$2 EPOCH=$3 \
  VALIDATION_MODES=original,fresh NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=false \
  ./hpc/submit_symbol_inference_node1.sh; done
```

## EVAL-C  (fewshot only)   B-trained only   [6 runs]
```bash
for E in "${DPI_B[@]}" "${NSY_B[@]}"; do set -- $E
  MODEL_TYPE=flamingo CKPT_DATE=$1 TRAIN_RUN=$2 EPOCH=$3 \
  VALIDATION_MODES=original,fresh NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=true \
  ./hpc/submit_symbol_inference_node1.sh; done
```

### (optional) failed-seed patch — dpi 113131 eval-A Original didn't finish
```bash
MODEL_TYPE=flamingo CKPT_DATE=2026-08-02 TRAIN_RUN=113131_af_h-cr-m14en_dpi_ha EPOCH=1 VALIDATION_MODES=original NUM_EXAMPLES=0 NO_LEGEND=false ./hpc/submit_symbol_inference_node1.sh
```

--------------------------------------------------------------------------------
## 2. MEDICAL — SPRSound (DATASET_TYPE=sprsound; opaque labels cas/das). Not trained on → ALL cells need inference.
Full matrix = reuse the registry + EVAL-A/B/C blocks with `DATASET_TYPE=sprsound` prepended. Lighter dpi-vs-nosym slice (1 seed/cell, 3 conditions):
```bash
for FLAGS in "NUM_EXAMPLES=0 NO_LEGEND=false" "NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=false" "NUM_EXAMPLES=1 FEWSHOT_PER_CLASS=true NO_LEGEND=true"; do
  for E in "2026-08-06 184309_af_h-cr-m14en_dpi_ha_fs1t 5" "2026-08-07 100621_af_h-cr-m14en_nosym_fs1t 5"; do set -- $E
    MODEL_TYPE=flamingo DATASET_TYPE=sprsound CKPT_DATE=$1 TRAIN_RUN=$2 EPOCH=$3 VALIDATION_MODES=original,fresh $FLAGS ./hpc/submit_symbol_inference_node1.sh; done; done
```

## 3. QA — HeySQuAD (DATASET_TYPE=heysquad) — ORIGINAL only, zero-shot+legend. Metric EM / token_f1 / format_compliance.
```bash
MODEL_TYPE=flamingo DATASET_TYPE=heysquad CKPT_DATE=2026-08-06 TRAIN_RUN=184309_af_h-cr-m14en_dpi_ha_fs1t EPOCH=5 VALIDATION_MODES=original SPLIT=validation NUM_EXAMPLES=0 NO_LEGEND=false ./hpc/submit_symbol_inference_node1.sh
MODEL_TYPE=flamingo DATASET_TYPE=heysquad CKPT_DATE=2026-08-07 TRAIN_RUN=100621_af_h-cr-m14en_nosym_fs1t EPOCH=5 VALIDATION_MODES=original SPLIT=validation NUM_EXAMPLES=0 NO_LEGEND=false ./hpc/submit_symbol_inference_node1.sh
```

--------------------------------------------------------------------------------
## Diagonal = HAVE from training-log validation (held-out-4 fresh, mean±std):
dpi  A 0.627±.018 · B 0.581±.057 · C 0.522±.031     nosym A 0.512±.038 · B 0.473±.032 · C 0.240±.033
Battery remaining off-diagonal (B/C-trained only) = 24 runs. Then QA + medical (full matrix). A-trained row already complete via inference+logs.
