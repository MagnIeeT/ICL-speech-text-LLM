# ICL Speech-Text LLM (Strict Active Pipeline)

This repository now uses a strict active structure centered on:
- top-level training/inference entrypoints
- a LoRA-only symbol-adapter training flow
- four active datasets
- minimal active utility surface

Legacy code is retained under `archive/` and is not part of the active runtime.

## Active Project Structure

```text
ICL-speech-text-LLM/
├── training.sh                      # Local training runner (no qsub)
├── inference.sh                     # Local inference runner (no qsub)
├── train.py                         # Active training entrypoint
├── inference.py                     # Active inference entrypoint
├── config/
│   ├── train_config/
│   │   └── training_configs.py      # CLI + dataclass training config
│   └── data_config/
│       ├── master_config.py         # DatasetType, DatasetSplit, registry
│       ├── voxceleb_config.py       # Dataset paths/prompts/labels
│       ├── hvb_config.py
│       ├── voxpopuli_config.py
│       └── meld_emotion_config.py
├── dataload/
│   ├── data_utils.py                # Active dataset loading + cache
│   ├── multi_task_dataset.py        # Unified dataset pipeline
│   ├── model_processors.py          # Processor base + get_processor()
│   ├── qwen_processor.py            # Qwen-specific processor
│   └── salmon_processor.py          # SALMONN-specific processor
├── models/
│   ├── backends/
│   │   ├── custom_qwen.py
│   │   └── custom_salmonn.py
│   └── symbolAdapter/
│       ├── symbol_manager.py
│       ├── symbol_training.py
│       └── validation.py
├── scripts/
│   ├── submit_symbol_training_job.sh
│   └── submit_symbol_inference_job.sh
├── utils/
│   ├── generate_fewshots.py         # One-time few-shot dataset generation utility
│   ├── evaluation_utils.py
│   ├── reprocess_metrics.py
│   └── training_utils.py            # Minimal checkpoint utility
├── eval_tasks/                      # Renamed from eval_qwen_and_salmonn
├── requirements/
│   ├── environment.yml
│   └── environment_qwen.yml
└── archive/                         # Legacy code and references
```

## Active Datasets

Current active dataset types:
- `voxceleb`
- `hvb`
- `voxpopuli`
- `meld_emotion`

These are defined in `config/data_config/master_config.py`.

## Where To Update Config

### 1) Training/inference runtime config
Edit defaults and argument behavior in:
- `config/train_config/training_configs.py`

Important fields:
- `DataConfig.dataset_type`
- `DataConfig.val_dataset_type`
- `DataConfig.batch_size`, `DataConfig.max_samples`
- `LoRAConfig.learning_rate`, `LoRAConfig.epochs`
- `SymbolConfig.dynamic_symbols`, `SymbolConfig.update_strategy`

### 2) Dataset paths/prompts/labels
Edit per-dataset files:
- `config/data_config/voxceleb_config.py`
- `config/data_config/hvb_config.py`
- `config/data_config/voxpopuli_config.py`
- `config/data_config/meld_emotion_config.py`

Update these keys as needed:
- `paths` (train/validation/test)
- `audio_lookup_paths`
- `prompt_template`
- `valid_labels`
- `text_key`, `completion_key`

## Environment Setup

Use one of:
- `requirements/environment.yml` (SALMONN-focused)
- `requirements/environment_qwen.yml` (Qwen-focused)

Example:

```bash
conda env create -f requirements/environment.yml
conda activate salmonn
```

Or for Qwen:

```bash
conda env create -f requirements/environment_qwen.yml
conda activate qwen2_new
```

## How To Generate Few-Shot Datasets (One-Time)

Use:
- `utils/generate_fewshots.py`

Before running, update inside that file:
- `DATASET_CONFIG`
- `target_split`
- `source_splits`
- `top_k`
- `PROCESSED_BASE_PATH`

Then run:

```bash
python utils/generate_fewshots.py
```

Outputs:
- augmented dataset with `few_shot_examples`
- audio lookup dataset for speech few-shot mode

## Local Run Commands

### Training

```bash
python train.py \
  --model_type salmonn \
  --dataset_type hvb-meld_emotion \
  --val_dataset_type hvb-meld_emotion \
  --device cuda:0 \
  --batch_size 1 \
  --max_samples 0 \
  --lora_lr 1e-5 \
  --lora_epochs 5 \
  --gradient_accumulation_steps 8 \
  --max_grad_norm 1.0 \
  --dynamic_symbols \
  --symbol_update_strategy per_epoch \
  --output_dir ./results/symbol_training \
  --run_name my_train_run
```

### Inference

```bash
python inference.py \
  --model_type salmonn \
  --checkpoint_path /path/to/checkpoint.pt \
  --dataset_type hvb-voxceleb-voxpopuli-meld_emotion \
  --device cuda:0 \
  --max_val_samples 0 \
  --num_examples 0 \
  --output_dir ./results \
  --run_name my_infer_run
```

## Local Runner Scripts (No qsub)

### Training (`training.sh`)

```bash
MODEL_TYPE=salmonn \
DATASET_TYPE=hvb-meld_emotion \
VAL_DATASET_TYPE=hvb-meld_emotion \
NO_SYMBOLS=false \
DYNAMIC_SYMBOLS=true \
OUTPUT_DIR=./results/symbol_training \
./training.sh
```

### Inference (`inference.sh`)

```bash
MODEL_TYPE=salmonn \
CHECKPOINT_PATH=/path/to/checkpoint.pt \
DATASET_TYPE=hvb-voxceleb-voxpopuli-meld_emotion \
NO_SYMBOLS=true \
OUTPUT_DIR=./results \
./inference.sh
```

Notes:
- These wrappers do not submit to queue; they run python directly on the current machine.
- Use `./training.sh --help` and `./inference.sh --help` to view all env options.

## HPC Submit Scripts

### Submit training

```bash
MODEL_TYPE=salmonn \
DATASET_TYPE=hvb-meld_emotion \
VAL_DATASET_TYPE=hvb-meld_emotion \
OUTPUT_DIR=/path/to/results/symbol_training \
./scripts/submit_symbol_training_job.sh
```

### Submit inference

```bash
MODEL_TYPE=salmonn \
CHECKPOINT_PATH=/path/to/checkpoint.pt \
DATASET_TYPE=hvb-voxceleb-voxpopuli-meld_emotion \
OUTPUT_DIR=/path/to/results \
./scripts/submit_symbol_inference_job.sh
```

## Outputs

Training writes under:
- `<output_dir>/checkpoints/<run_name>/...`
- `<output_dir>/logs/<date>/<run_name>.log`

Inference writes under:
- `<output_dir>/orchestrator_metrics/<date>/<run_name>_metrics.json`
- `<output_dir>/orchestrator_metrics/<date>/<run_name>_predictions.json`
- `<output_dir>/orchestrator_logs/<date>/<run_name>.log`

## Evaluation Folder

`eval_qwen_and_salmonn/` has been renamed to `eval_tasks/`.

We can work through and simplify files in `eval_tasks/` in a separate pass.
