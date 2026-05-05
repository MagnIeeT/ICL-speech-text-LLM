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

2.  **Configuration:**
    Copy the `.env.example` file to `.env` and fill in your local paths and model identifiers.
    ```bash
    cp .env.example .env
    # Edit .env to set your data, checkpoint, and base model paths
    ```

## Base Models

The pipeline supports the following base models by default (configurable in `.env`):
- **SALMONN:** Uses `lmsys/vicuna-13b-v1.1` as the LLM and `openai/whisper-large-v2` for audio features.
- **Qwen2-Audio:** Uses `Qwen/Qwen2-Audio-7B-Instruct`.

Ensure you have the necessary permissions to access these models on HuggingFace if you are downloading them for the first time.

## Usage

### Training

```bash
python train.py \
  --model_type salmonn \
  --dataset_type voxceleb \
  --output_dir ./results/symbol_training \
  --run_name my_train_run
```

### Inference

```bash
python inference.py \
  --model_type salmonn \
  --checkpoint_path /path/to/checkpoint.pt \
  --dataset_type voxceleb \
  --run_name my_infer_run
```

## Active Datasets

Current active dataset types: `voxceleb`, `hvb`, `voxpopuli`, `meld_emotion`. These are defined and registered in `config/data_config/master_config.py`.

## Data Preparation

To generate augmented few-shot datasets, use:
```bash
python utils/generate_fewshots.py
```
Ensure you have set `BASE_DATA_DIR` in your `.env` file.
