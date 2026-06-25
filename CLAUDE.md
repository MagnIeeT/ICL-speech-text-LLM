# CLAUDE.md — ICL-speech-text-LLM

Speech emotion/intent classification using in-context learning with audio LLMs (Qwen2-Audio, Flamingo, SALMONN). Supports LoRA fine-tuning with optional differentiable symbol replacement (D-SPO).

---

## Project layout

```
train.py                          # training entry point
inference.py                      # inference entry point
models/
  custom_qwen.py                  # Qwen2-Audio-7B-Instruct wrapper (PRIMARY)
  custom_flamingo.py              # Audio Flamingo 3 wrapper
  custom_salmonn.py               # SALMONN wrapper
  symbolAdapter/
    symbol_training.py            # SymbolTrainingOrchestrator — main training loop, all D-SPO phases
    symbol_manager.py             # SymbolManager — symbol generation, per-dataset maps, batch replacement
    validation.py                 # ValidationManager — multi-mode validation runner
    symbol_router.py              # SymbolRouter — slot→vocab Gumbel-softmax routing matrix
    dspo_module.py                # DspoModule — injects soft slot embeddings during training
    vocab_filter.py               # VocabFilter — generates fresh symbol token pools
    symbol_training_dpo.py        # SymDPO trainer (DPO-based, separate from D-SPO)
dataload/
  multi_task_dataset.py           # MultiTaskDataset — handles multiple datasets, few-shot, audio loading
  flamingo_processor.py           # FlamingoProcessor — audio precompute in DataLoader workers (FAST PATH)
  qwen_processor.py               # QwenProcessor
  salmon_processor.py             # SalmonProcessor
  data_utils.py                   # create_combined_dataloader()
config/
  train_config/training_configs.py  # TrainingConfig, LoRAConfig, DifferentiableSymbolConfig, CLI args
  data_config/master_config.py      # DATASET_CONFIGS — all dataset metadata and valid_labels
  data_config/*_config.py           # per-dataset configs (voxceleb, hvb, voxpopuli, meld_emotion)
hpc/
  submit_symbol_training_node1.sh   # launch training on node1 (nohup, all env vars, mode tagging)
utils/
  training_utils.py
  evaluation_utils.py
```

---

## How to run training

```bash
# On the login node (not node1 directly):
./hpc/submit_symbol_training_node1.sh
```

Key env vars (override before running):
```bash
MODEL_TYPE=qwen              # qwen | flamingo | salmonn
DATASET_TYPE=meld_emotion    # training dataset(s), dash-separated for multi
VAL_DATASET_TYPE=voxceleb-hvb-voxpopuli-meld_emotion
MAX_SAMPLES=0                # 0 = full dataset
LORA_EPOCHS=10
DIFF_SYMBOL_ENABLED=true     # enables D-SPO
DSPO_NUM_SLOTS=25
DSPO_SLOT_VOCAB_SIZE=25      # K vocab tokens per slot
DSPO_ROTATION_INTERVAL=200   # optimizer steps between slot rotations
DSPO_PHASE0_EPOCHS=1         # LoRA warmup before D-SPO starts
DSPO_PHASE1_PATIENCE=3       # patience for Phase 1 → Phase 2 transition
DSPO_PHASE1_EPOCHS=5         # hard cap on Phase 1
WARMUP_STEPS=100
CUDA_VISIBLE_DEVICES=1       # which GPU (default GPU 1 on node1)
```

Logs land at: `~/training/symbol_training/logs/YYYY-MM-DD/<RUN_NAME>.log`
Checkpoints: `~/training/symbol_training/<RUN_NAME>/`

To follow a run:
```bash
tail -f ~/training/symbol_training/logs/$(date +%Y-%m-%d)/<RUN_NAME>.log
```

To check GPU / system load on node1:
```bash
ssh node1 "nvidia-smi && uptime"
```

---

## Training pipeline

```
train.py
  → TrainingConfig (from CLI args)
  → SymbolManager(labels_per_dataset=...)   # per-dataset symbol maps
  → SymbolTrainingOrchestrator.run_complete_training()
      ├── Phase 0 (phase0_epochs > 0)
      │     self.router = None, no_symbols = True
      │     LoRA trains on original labels
      ├── Phase 1 (slot_only mode, router trains)
      │     Gumbel-softmax slot routing → soft symbol embeddings in forward()
      │     slot rotation every DSPO_ROTATION_INTERVAL optimizer steps
      │     exits when conf_mean plateaus (patience) or phase1_epochs hit
      └── Phase 2 (LoRA + fixed decoded symbols)
            router frozen, symbols decoded once per refresh
            symbol refresh: 0=per epoch, -1=fixed, N=every N steps
```

Per batch:
```
_apply_symbol_replacement(raw_batch, epoch, batch_idx)
  → get_dataset_info(batch)           # ds_name_str, relevant_labels
  → replace_symbols_in_batch()        # text substitution (no-op if no_symbols)
  → tokenize_batch_with_audio()       # injects _precomputed into prompts → fast path
```

---

## Symbol system

### Per-dataset symbol maps
`symbol_manager._pure_symbol_mappings` and `fixed_mappings` are `Dict[str, Dict[str, str]]`:
```
{ds_name: {label: symbol}}   # e.g. {"meld_emotion": {"neutral": "taok"}, "voxceleb": {"neutral": "vpfp"}}
```
Overlapping labels (e.g. "neutral" in both meld and voxceleb) get INDEPENDENT symbols — never shared across datasets. Generated in one call with globally unique symbols, then sliced per dataset.

Key method: `get_flat_symbols_for_ds(ds_name, epoch)` → flat `{label: symbol}` for one dataset.

### Cross-task validation extension
Training datasets → use their learned symbols.
Cross-task datasets (not in training) → each gets ALL its labels randomly sampled from the symbol pool independently.

### _build_current_symbol_map always returns `{ds_name: {label: symbol}}`
Three paths: Phase 2 (`_phase2_label_map`), Phase 1 router decode, non-D-SPO (`get_symbols_for_epoch`).

---

## Flamingo audio fast path (IMPORTANT)

**Do not break this or training slows 8x.**

`flamingo_processor.process_inputs()` runs in DataLoader workers (num_workers=2), precomputes:
```python
{"input_features": tensor, "input_features_mask": tensor, "num_audio_tokens": int}
```
stored as `batch["_precomputed"]` (list of dicts after collation).

`tokenize_batch_with_audio()` injects `_precomputed` into each prompt dict before calling `processor.tokenize_batch()`.

`_tokenize_one()` fast path: replaces audio item with `<sound>*N` text → `apply_chat_template` skips feature extraction → injects precomputed features afterward.

If `batch["_precomputed"]` is None or missing → slow path (~5s/sample vs ~0.7s fast path).

---

## Datasets

| Dataset | Task | Labels | Notes |
|---|---|---|---|
| meld_emotion | emotion | neutral, joy, sadness, anger, fear, disgust, surprise | 9988 train samples |
| voxceleb | sentiment | positive, negative, neutral | "disagreement" is OOV — skipped from metrics |
| hvb | dialogue acts | ~34 labels | |
| voxpopuli | intent | several | |

Dataset configs: `config/data_config/master_config.py` → `DATASET_CONFIGS: Dict[DatasetType, DatasetConfig]`

---

## Validation modes

Three modes run every epoch (configured via `--validation_modes original,fixed,fresh`):
- `original` — model uses original label names (no symbols)
- `fixed` — model uses the current epoch's symbol map
- `fresh` — model uses newly generated symbols (tests generalization)

Primary metric: `macro_f1_with_invalid` — invalid predictions (out-of-vocab) count as wrong.

OOV **true** labels (e.g. "disagreement" in VoxCeleb) are excluded from both numerator and denominator.

---

## Metrics

`avg_score` in epoch history = average across datasets for the PRIMARY (first) validation mode.

Validation scores logged as:
```
📊 [original] per-dataset: {'voxceleb': 0.56, 'hvb': 0.16, ...}  avg=0.29
```

---

## Models

### Qwen2-Audio-7B-Instruct (PRIMARY, `model_type=qwen`)
- Conda env: `qwen`
- LoRA applied to attention layers
- `custom_qwen.py`: `forward()` for training loss, `generate_output()` for inference

### Audio Flamingo 3 (nvidia/audio-flamingo-3-hf, `model_type=flamingo`)
- Conda env: `flamingo`
- Requires `transformers>=5.0`
- Audio tokens processed via cross-attention in every transformer layer
- Pre-computes audio features in DataLoader workers (see fast path above)
- `CUDA_VISIBLE_DEVICES=1` by default, auto-selects `flamingo` conda env

### SALMONN (`model_type=salmonn`)
- Secondary, some known bugs (see Known issues)

---

## D-SPO technical notes

- **Slot placeholders**: `<slot_i_0>`, `<slot_i_1>` (two-token per slot) — must be added to tokenizer via `setup_dspo_tokenizer()` before model init
- **Router**: `slot_vocab_indices [num_slots, K, token_size]` — K candidate vocab tokens per slot position
- **Tau annealing**: decays per optimizer step at rate `dspo_tau_anneal_rate=0.0001`
- **Rotation interval**: counts optimizer steps (increments after `optimizer.step()`)
- **Phase 1 → Phase 2 transition**: triggered by `conf_mean` plateau (patience) or hard `phase1_epochs` cap
- **D-SPO inference**: hard token swap in `input_ids` via `slot_replacement={placeholder_id: vocab_id}` — NOT soft embeddings (training-only)

---

## Known issues / gotchas

- `custom_salmonn.py:generate_output()` missing `slot_replacement` param — crashes if Salmonn + D-SPO
- `salmon_processor.py` has the same prompt-length truncation bug as old qwen_processor (not fixed)
- `inference.py` restores D-SPO router state from checkpoint but requires `diff_symbol_enabled` flag
- `data_utils.py` dataset cache key doesn't include `max_samples` — restart process if changing it
- GitHub push needs token in remote URL: `git remote set-url origin https://magnieet:<TOKEN>@github.com/...`
- `nvidia-smi` must be run via `ssh node1` — not available on login node
- node1 is a shared machine; CPU load of 50+ kills DataLoader worker throughput (check before launching)

---

## Research direction

Staged plan (see `RESEARCH_APPROACH.md`):
1. SFT + fixed symbols (baseline)
2. D-SPO — current focus
3. SymDPO-style DPO with fixed/dynamic symbols
4. D-SPO + SymDPO combined

Primary model: Qwen2-Audio-7B-Instruct. Flamingo is a secondary experiment.
