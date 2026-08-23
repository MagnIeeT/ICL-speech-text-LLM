# ICL Speech-Text LLM

In-Context Learning pipeline for Speech-Text Large Language Models. Supports LoRA fine-tuning with a Symbol Adapter — a training strategy that replaces class labels with random tokens to prevent the model from exploiting pre-trained label knowledge. Includes Differentiable Symbolic Preference Optimization (D-SPO) for end-to-end differentiable symbol learning.

## Supported Models

| Model | Backend | Notes |
|---|---|---|
| Qwen2-Audio-7B-Instruct | `custom_qwen.py` | Primary, fully tested |
| SALMONN | `custom_salmonn.py` | LLaMA + Whisper + BEATs |
| Audio Flamingo 3 | `custom_flamingo.py` | nvidia/audio-flamingo-3-hf |

## Project Structure

```
ICL-speech-text-LLM/
├── train.py                          # Training entrypoint
├── inference.py                      # Inference entrypoint
├── config/
│   ├── train_config/training_configs.py   # TrainingConfig + all sub-configs
│   └── data_config/                       # Per-dataset configs (voxceleb, hvb, ...)
├── dataload/
│   ├── qwen_processor.py             # Qwen tokenization + audio features
│   ├── flamingo_processor.py         # Flamingo processor
│   ├── salmon_processor.py           # SALMONN processor
│   ├── multi_task_dataset.py         # Multi-dataset sampling
│   └── data_utils.py                 # Dataset loading helpers
├── models/
│   ├── backends/                     # custom_qwen.py, custom_flamingo.py, custom_salmonn.py
│   └── symbolAdapter/
│       ├── symbol_training.py        # Training orchestrator
│       ├── symbol_manager.py         # Label→symbol mapping + swap logic
│       ├── validation.py             # Validation manager (multi-mode)
│       ├── symbol_router.py          # D-SPO slot matrix (Gumbel-Softmax)
│       ├── dspo_module.py            # D-SPO soft embedding injection
│       └── vocab_filter.py           # Symbol pool generation
├── utils/
│   ├── training_utils.py             # Checkpoint load/save
│   └── evaluation_utils.py           # macro_f1, macro_f1_with_invalid
├── eval_asr_tasks/                   # ASR/ST evaluation scripts
├── requirements/                     # Conda environment files
└── hpc/                              # SLURM submit scripts
```

## Setup

**1. Environment:**
```bash
conda env create -f requirements/environment.yml
conda activate qwen          # or salmonn
```

**2. Configuration (.env):**
```bash
cp .env.example .env
```
Key variables:
- `QWEN_MODEL_NAME` — e.g. `Qwen/Qwen2-Audio-7B-Instruct`
- `FLAMINGO_MODEL_NAME` — e.g. `nvidia/audio-flamingo-3-hf`
- `SALMONN_CKPT_PATH`, `BEATS_CKPT_PATH`, `WHISPER_MODEL_NAME`, `LLAMA_MODEL_NAME`
- `VOXCELEB_TRAIN_PATH`, `HVB_TRAIN_PATH`, etc. — dataset split paths
- `BASE_OUTPUT_DIR`, `CHECKPOINT_DIR`, `LOGS_DIR`

See `.env.example` for the full list of keys (copy it to `.env` and fill in real paths).

**3. SALMONN weights** (only for `--model_type salmonn`):
- `salmonn_v1.pth` → `SALMONN_CKPT_PATH` — from [tsinghua-ee/SALMONN](https://huggingface.co/tsinghua-ee/SALMONN)
- BEATs `cpt2` → `BEATS_CKPT_PATH` — from [WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2](https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2)
- Whisper (`openai/whisper-large-v2`) and Vicuna (`lmsys/vicuna-13b-v1.1`) auto-download from their HF ids on first run.

SALMONN runs in the `qwen` conda env (shares `transformers==4.45.2`); no separate env is required.

## Training

### HPC (SLURM)
```bash
# Edit the config block at the top of the script, then:
./hpc/submit_symbol_training_node1.sh
```

### Local
```bash
python train.py \
  --model_type qwen \
  --dataset_type voxceleb \
  --run_name my_run \
  --lora_epochs 10 \
  --max_samples 0 \
  --num_examples 5
```

## Configuration Reference

### Core
| Argument | Description |
|---|---|
| `--model_type` | `qwen`, `salmonn`, `flamingo` |
| `--dataset_type` | `voxceleb`, `hvb`, `voxpopuli`, `meld_emotion` (comma-separated for multi) |
| `--val_dataset_type` | Validation dataset (defaults to `dataset_type`) |
| `--run_name` | Unique run identifier for logging and checkpoints |
| `--device` | e.g. `cuda:0` |
| `--max_samples` | Training samples per dataset (0 = all) |
| `--num_examples` | Few-shot examples per prompt |

### LoRA
| Argument | Default | Description |
|---|---|---|
| `--lora_lr` | `1e-5` | LoRA learning rate |
| `--lora_epochs` | `5` | Training epochs |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation |
| `--max_grad_norm` | `1.0` | Gradient clipping |

### Symbol Adapter Strategy
| Argument | Description |
|---|---|
| `--no_symbols` | Baseline — use original label names, no replacement |
| `--dynamic_symbols` | Generate new symbol mappings each epoch |
| `--symbol_update_strategy` | `per_epoch` (default) or `per_instance` |
| `--swap_labels` | Shuffle label→symbol assignments (see Swap Mode below) |
| `--num_symbol_mappings` | Pool of pre-generated label→symbol maps; `per_instance` samples one per example |
| `--no_legend` | Strip label definitions from the prompt (exemplar-only, Wei-style). Default keeps definitions (definition-guided) |
| `--validation_modes` | Comma-separated: `fixed`, `original`, `fresh`, `flipped` (default: all) |

### Differentiable Symbolic Preference Optimization (D-SPO)
| Argument | Default | Description |
|---|---|---|
| `--diff_symbol_enabled` | — | Enable D-SPO end-to-end differentiable symbol learning |
| `--dspo_slot_vocab_size` | `10` | Private candidate tokens per slot |
| `--dspo_rotation_interval` | `0` | Rotate slot assignments every N global steps (0 = per epoch) |

**Architecture:** D-SPO maintains a learnable Slot Matrix Π of shape `[num_slots, K]` where `K = slot_vocab_size`. Each slot owns a **private non-overlapping vocabulary of K tokens** — slots cannot converge on the same token by construction. The total token pool required is `num_slots × K` (default: 20 × 10 = 200 tokens), filtered to ASCII alphabetic single-token candidates to ensure well-formed embeddings.

**Training:** Slot→label assignments are held fixed for a rotation window (per epoch by default, or every N global steps via `--dspo_rotation_interval`). Within each window, consistent gradient signal flows to each slot's router. Gumbel-Softmax (hard=True, straight-through estimator) keeps the forward pass discrete while allowing gradients to flow back through the preference matrix.

**Validation:** The `K` most confident slots (highest `max(softmax(preferences[i]))`) are selected for each dataset's label set, ensuring the most learned slots are used for evaluation. Slot assignments are fixed for the entire validation run so scores are comparable across epochs.

**Cross-task transfer:** After training, each slot's router has converged on a token that functions as a reliable label placeholder. At inference on a new task with `K` classes, select the top-K most confident slots — the model has learned to treat their tokens as in-context label anchors regardless of the specific class semantics.

## Symbol Adapter Modes

### No Symbols (`--no_symbols`)
Model sees original label names. Baseline for comparison.

### Fixed Symbols (default)
Labels replaced with random 4–5 character tokens (e.g. `neutral → tepj`). Same mapping throughout training. Forces the model to use in-context examples rather than pre-trained label knowledge.

### Dynamic Symbols (`--dynamic_symbols`)
New random symbol set generated each epoch (`per_epoch`) or each batch (`per_instance --symbol_update_strategy per_instance`).

### Swap Mode (`--swap_labels`)
Shuffles which symbol is assigned to which label. Scope is always within the current dataset's label set — never crosses dataset boundaries.

- With `--no_symbols`: shuffles original label names (e.g. `neutral↔positive`)
- Without `--no_symbols`: generates symbols first, then shuffles the label→symbol assignment
- Frequency: controlled by `--symbol_update_strategy` (`per_epoch` or `per_instance`)

### D-SPO (`--diff_symbol_enabled`)
Learns the optimal symbol-to-label mapping end-to-end via a differentiable slot matrix.

- Each of `num_slots` slots owns a private pool of `K` candidate tokens (non-overlapping — no two slots compete for the same token)
- Slot→label assignments rotate on a configurable schedule (per epoch or every N steps) so all slots receive training signal over time
- At validation, the top-K most converged slots are selected automatically by confidence score
- Enables **cross-task transfer**: learned slot tokens can be reused as label placeholders on unseen tasks without retraining the router

## Validation Modes

Three modes run sequentially each epoch (configurable via `--validation_modes`):

| Mode | Description |
|---|---|
| `original` | Original label names — tests whether symbol training hurt base capability |
| `fixed` | Same symbols used during training — tests how well the model learned the mapping |
| `fresh` | Brand new symbols never seen in training — tests whether the model follows the in-context definition for a novel label |
| `flipped` | Real labels paired with deranged (contradictory) definitions — tests steerability: does the model follow the definition or revert to its label prior |

**Primary metric:** `macro_f1_with_invalid` — samples with out-of-vocabulary true labels (e.g. `disagreement` in VoxCeleb) are excluded from both metrics. Invalid model predictions (outputs not in the valid label set) are counted as wrong. `avg_score` in epoch summaries reflects the primary (first configured) validation mode, averaged across datasets.

## Active Datasets

| Dataset | Task | Role |
|---|---|---|
| HVB | Dialogue acts (~34) | Trained |
| CREMA-D | Speech emotion | Trained |
| MInDS-14 (en) | Banking intent | Trained |
| MInDS-14 (fr / ko) | Banking intent | Held-out |
| Skit-S2I | Speech-to-intent | Held-out |
| RAVDESS-Song | Song emotion | Held-out |
| Speech Commands | Keyword spotting | Held-out |
| SPRSound | Respiratory-sound classification | Held-out |
| HeySQuAD | Spoken question answering | Held-out (generative) |
| VoxCeleb / VoxPopuli / MELD / RAVDESS / ESD | Sentiment / intent / emotion | Additional (configured) |

Dataset configs in `config/data_config/`. Register new datasets in `config/data_config/master_config.py`. User-defined 3-cluster taxonomies for the clustering eval are built by `utils/build_custom_taxonomy.py`.

## Checkpoints

Checkpoints saved to `$CHECKPOINT_DIR/<run_name>/`. Format:
```
lora_epoch{N}_periodic.pt   # saved every checkpoint_frequency epochs
lora_epoch{N}_final.pt      # saved at end of training
```

Checkpoint keys: `model_state` (LoRA weights), `router_state` (D-SPO router), `optimizer_state`, `config`, `symbol_mappings`.
