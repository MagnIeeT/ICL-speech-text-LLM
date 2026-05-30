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
| `--validation_modes` | Comma-separated: `fixed`, `original`, `fresh` (default: all three) |

### Differentiable Symbolic Preference Optimization (D-SPO)
| Argument | Description |
|---|---|
| `--diff_symbol_enabled` | Enable D-SPO end-to-end differentiable symbol learning |

D-SPO maintains a learnable slot matrix (Preference Matrix Π). During training, soft embeddings are injected into the LLM's input space via Gumbel-Softmax (hard=True, straight-through estimator). During validation, hard argmax token IDs replace slot placeholders. Slot→label assignments are fixed per dataset per validation run for comparable scores across epochs.

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
Learns the optimal symbol-to-label mapping end-to-end. The router randomly samples `k` slots per training batch so all slots get exposure over time. During validation, slot assignments are fixed per dataset.

## Validation Modes

Three modes run sequentially each epoch (configurable via `--validation_modes`):

| Mode | Description |
|---|---|
| `original` | Original label names — tests whether symbol training hurt base capability |
| `fixed` | Same symbols used during training — tests how well the model learned the mapping |
| `fresh` | Brand new symbols never seen in training — tests whether the model learned to use in-context mappings generally |

**Primary metric:** `macro_f1_with_invalid` — samples with out-of-vocabulary true labels (e.g. `disagreement` in VoxCeleb) are excluded from both metrics. Invalid model predictions (outputs not in the valid label set) are counted as wrong. `avg_score` in epoch summaries reflects the primary (first configured) validation mode, averaged across datasets.

## Active Datasets

| Dataset | Task | Labels |
|---|---|---|
| VoxCeleb | Sentiment | positive, negative, neutral |
| HVB | Dialogue Acts | ~34 act types |
| VoxPopuli | Named Entity | Multi-label |
| MELD-Emotion | Emotion | 7 emotion classes |

Dataset configs in `config/data_config/`. Register new datasets in `config/data_config/master_config.py`.

## Checkpoints

Checkpoints saved to `$CHECKPOINT_DIR/<run_name>/`. Format:
```
lora_epoch{N}_periodic.pt   # saved every checkpoint_frequency epochs
lora_epoch{N}_final.pt      # saved at end of training
```

Checkpoint keys: `model_state` (LoRA weights), `router_state` (D-SPO router), `optimizer_state`, `config`, `symbol_mappings`.
