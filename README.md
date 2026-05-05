# ICL Speech-Text LLM (Strict Active Pipeline)

This repository implements an In-Context Learning (ICL) pipeline for Speech-Text Large Language Models, specifically focused on a LoRA-only symbol-adapter training flow.

## Project Structure

```text
ICL-speech-text-LLM/
├── train.py                         # Active training entrypoint
├── inference.py                     # Active inference entrypoint
├── config/                          # Training and data configurations
├── dataload/                        # Dataset and model processing logic
├── models/                          # Model backends and symbol-adapter logic
├── utils/                           # Shared utilities (environment, few-shots, etc.)
├── requirements/                    # Environment setup files
├── .env.example                     # Template for environment variables
└── hpc/                             # HPC/Cluster submit scripts (gitignored)
```

## Setup

1.  **Environment:**
    Create a conda environment using the provided `.yml` files.
    ```bash
    conda env create -f requirements/environment.yml
    conda activate salmonn
    ```

2.  **Configuration (.env):**
    Copy `.env.example` to `.env` and configure your local paths. This is **required** as the code uses these variables to avoid hardcoded paths.
    ```bash
    cp .env.example .env
    ```
    Key variables to set:
    - `SALMONN_CKPT_PATH`: Path to the `salmonn_v1.pth` file.
    - `BEATS_CKPT_PATH`: Path to the `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` file.
    - `WHISPER_MODEL_NAME`: e.g., `openai/whisper-large-v2`.
    - `LLAMA_MODEL_NAME`: e.g., `lmsys/vicuna-13b-v1.1`.
    - `VOXCELEB_TRAIN_PATH`, etc.: Paths to your local dataset splits.

## Usage

### Training (HPC)
The HPC scripts are now self-contained. You can edit the "Job Configuration" block at the top of the script and run it without passing external environment variables.
```bash
# 1. Edit the config block in hpc/submit_symbol_training_job.sh
# 2. Run the script
./hpc/submit_symbol_training_job.sh
```

### Inference (HPC)
Similarly for inference:
```bash
# 1. Edit CHECKPOINT_PATH and other values in hpc/submit_symbol_inference_job.sh
# 2. Run the script
./hpc/submit_symbol_inference_job.sh
```

### ASR Evaluation (HPC)
For ASR tasks:
```bash
./eval_asr_tasks/submit_eval.sh
```

### Local Execution
You can still run the scripts directly. They will automatically load paths from your `.env` file.
```bash
python train.py --model_type salmonn --dataset_type hvb --run_name my_test
```

## Configuration Parameters

The pipeline is highly configurable via CLI arguments or the `.env` file. Below are the key parameters:

### Core Settings
- `--model_type`: Choose between `salmonn` or `qwen` (default: `salmonn`).
- `--device`: Target device, e.g., `cuda:0` or `cpu`.
- `--run_name`: A unique identifier for the run (used for logging and checkpoints).

### Symbol Adapter Strategy
- `--no_symbols`: (Boolean) If set, disables symbol replacement and uses original labels (Baseline mode).
- `--dynamic_symbols`: (Boolean) If set, generates new symbol-to-label mappings for every epoch.
- `--symbol_update_strategy`: 
  - `per_epoch`: Mappings are fixed for the entire epoch.
  - `per_instance`: Each training sample gets a unique, randomized symbol mapping.
- `--swap_labels`: (Boolean) For binary/categorical tasks, flips the label semantics (e.g., positive becomes negative) to test robustness.
- `--validation_modes`: Comma-separated list of modes to test during validation:
  - `fixed`: Uses the same symbol mappings as the training set.
  - `original`: Uses the base model's original natural language labels.
  - `fresh`: Generates brand new symbol mappings never seen during training.

### Data & In-Context Learning (ICL)
- `--dataset_type`: Hyphen-joined list of datasets (e.g., `hvb-voxceleb`). Active: `voxceleb`, `hvb`, `voxpopuli`, `meld_emotion`.
- `--max_samples`: Total training samples per dataset (0 = all).
- `--num_examples`: Number of few-shot examples to include in the prompt.
- `--num_workers`: Number of data loading threads (increase for faster audio processing).

### LoRA Hyperparameters
- `--lora_lr`: Learning rate for LoRA weights (default: `1e-5`).
- `--lora_epochs`: Number of training epochs.
- `--gradient_accumulation_steps`: Steps to accumulate gradients before an optimizer update.

## Active Datasets

Current active dataset types: `voxceleb`, `hvb`, `voxpopuli`, `meld_emotion`. These are defined and registered in `config/data_config/master_config.py`.

## Data Preparation

To generate augmented few-shot datasets, use:
```bash
python utils/generate_fewshots.py
```
Ensure you have set `BASE_DATA_DIR` in your `.env` file.
