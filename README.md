# SPRInT Vision (VLM)

Symbol Prompt Training pipeline for Vision-Language Models on medical imaging. Supports LoRA
fine-tuning of LLaVA-v1.5-13B with a **Symbol Adapter** — a training strategy that replaces
class labels with meaningless tokens so the model learns the label mapping from prompt context
rather than exploiting pre-trained label semantics. This is the vision counterpart of the ICL
Speech-Text pipeline; it uses the same five SPRInT strategies and the same three-mode validation,
adapted to the MedFMC medical-image benchmark.

## Supported Model

| Model | Backend | Notes |
|-------|---------|-------|
| LLaVA-v1.5-13B | `llava/` (CLIP-ViT-L/14-336 + Vicuna-13B) | LoRA fine-tuning; `mlp2x_gelu` projector |

## Project Structure

```
LLaVA Code/
├── llava/                               # LLaVA model + training core
│   ├── train/
│   │   ├── train.py                 # Training entrypoint; preprocess_v1() does symbol substitution
│   │   ├── train_mem.py             # flash-attn wrapper → calls train()
│   │   └── llava_trainer.py         # LLaVATrainer (HF Trainer subclass)
│   └── model/
│       ├── builder.py               # load_pretrained_model() — LoRA load (kept unmerged)
│       ├── llava_arch.py            # Multimodal forward (image-token embedding merge)
│       └── language_model/          # LlavaLlamaForCausalLM
│
└── sprint_vision/                       # SPRInT strategy + eval layer
    ├── sprint_eval.py               # Inference entrypoint (thin wrapper)
    ├── vision_orchestrator.py       # Train/inference launcher (subprocess wrapper)
    ├── run_sprint_finetune.sh       # deepspeed training launch (edit env block at top)
    ├── submit_training.sh           # qsub/PBS training submit
    ├── submit_inference.sh          # qsub/PBS inference submit
    ├── config/
    │   └── data_config/
    │       ├── master_config.py     # DatasetConfig registry + get_dataset_config()
    │       ├── colon_config.py      # Binary, 2 labels
    │       ├── chest_config.py      # Multi-label, 19 classes
    │       └── endo_config.py       # Multi-label, 4 classes
    ├── data/
    │   └── medfmc_to_llava.py       # MedFMC .txt → LLaVA JSON (train / val / test)
    ├── dataload/
    │   ├── medfmc_prompts.py        # Per-class probe builder + P(Yes) scorer
    │   ├── example_selector.py      # Stratified-random ICL example selection
    │   └── prompt_builder.py        # Multi-image prompt builder (legacy ICL path)
    ├── models/
    │   └── symbolAdapter/
    │       ├── symbol_manager.py    # Label→symbol mapping + swap logic (5 strategies)
    │       ├── validation.py        # SPRInTValidationManager — THE single evaluator
    │       └── sprint_callbacks.py  # HF Trainer callbacks (validation, ED-FT rotation, progress)
    └── utils/
        └── evaluation_utils.py      # macro_f1, AUC, mAP, tie-correct references (shared metrics)
```

The pipeline spans **two folders**: `llava/` holds the model and the training entrypoint
(`train.py`), while `sprint_vision/` holds the SPRInT strategy, launchers, and the shared evaluator.
Training runs `deepspeed → llava/train/train_mem.py → train.py`; inference runs `sprint_vision/sprint_eval.py`.

## Setup

**1. Environment:**

```bash
conda activate llava
```
Cluster packages: `peft=0.7.1`, `transformers=4.37.2`, `accelerate=0.21.0`, `torch=2.1.2`,
`flash-attn` (both training and inference run `flash_attention_2`).

**2. Paths (env vars, read by `run_sprint_finetune.sh` / `vision_orchestrator.py`):**

| Variable | Example |
|----------|---------|
| `LLAVA_DIR` | `/home/harinisrireddykandula/LLaVA` |
| `MODEL_BASE` / `MODEL_PATH` | base `llava-v1.5-13b` weights |
| `MEDFMC_ROOT` | `/home/harinisrireddykandula/MedFM/data/MedFMC` |

## Data Preparation

Convert MedFMC `.txt` label files into LLaVA conversation JSON:

```bash
python data/medfmc_to_llava.py \
  --medfmc_root $MEDFMC_ROOT \
  --output_dir  ./data \
  --tasks colon,chest,endo \
  --shot 10 --exp 1          # MedFMC repeated few-shot protocol (exp 1..5)
```

Produces, per task: `{task}_train_shot{N}_exp{K}.json`, `{task}_val_shot{N}_exp{K}.json`,
`{task}_test.json`. The val split is the rest of the few-shot pool; validation subsamples a
**seeded** `random.Random(42)` subset of it (default 100 samples).

## Training

**HPC (PBS/qsub):**

```bash
# Edit the config block at the top, then:
STRATEGY=regular DATASET=chest TRAINING_SHOTS=10 SHOT_EXP=1 bash submit_training.sh
```

**Local / direct:**

```bash
STRATEGY=two_token DATASET=chest TRAINING_SHOTS=10 SHOT_EXP=1 bash run_sprint_finetune.sh
```

This launches `deepspeed` (ZeRO-2) → `llava/train/train_mem.py` → `train.py`. Symbol substitution
happens in `train.py::preprocess_v1()` (both the human turn and the GPT answer). After every epoch,
`SPRInTValidationCallback` runs the three-mode validation and tracks the best epoch.

## Configuration Reference

**Core (env vars for `run_sprint_finetune.sh`)**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `colon` | `colon`, `chest`, `endo` |
| `STRATEGY` | `regular` | `regular`, `two_token`, `ed_ft`, `id_ft`, `lf_ft` |
| `TRAINING_SHOTS` + `SHOT_EXP` | — | Fixed MedFMC few-shot split (N shots, experiment K) |
| `TRAIN_PERCENT` | — | Alternative: percentage split instead of shots |
| `MAX_TRAIN_SAMPLES` | `0` | Cap training samples (0 = all) |

**LoRA / Trainer**

| Argument | Default | Description |
|----------|---------|-------------|
| `LORA_R` | `8` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |
| `NUM_TRAIN_EPOCHS` | `5` | Training epochs |
| `per_device_train_batch_size` | `1` | Micro-batch |
| `gradient_accumulation_steps` | `8` | Effective batch = 8 |
| `--bf16` / `--tf32` | `True` | Precision (inference matches: bf16 + TF32) |

**Validation**

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_DATA_PATH` | `{ds}_val_shot{N}_exp{K}.json` | Val split; `none` disables validation |
| `MAX_VAL_SAMPLES` | `100` | Seeded `random.Random(42)` subsample (0 = full split) |
| `VALIDATION_MODES` | `fixed,original,fresh` | Which modes to run each epoch |
| `COMPUTE_VAL_AUC_MAP` | `true` | Compute the per-class AUC/mAP probe |
| `SPRINT_PROBE_BATCH_SIZE` | `1` | Per-class probes scored per forward pass (>1 = batched) |

## SPRInT Strategy Modes

All strategies use the **same prompt template** — only the label→symbol mapping changes. The mapping
must appear in **both** the human prompt and the GPT answer so the model learns it from context.

| Strategy | `STRATEGY=` | What changes |
|----------|-------------|--------------|
| **RFT** (Regular) | `regular` | Nothing — original label names (baseline) |
| **SS-FT** (Static Symbol) | `two_token` | Fixed meaningless tokens, same mapping all run |
| **ED-FT** (Epoch-Dynamic) | `ed_ft` | New symbol set generated each epoch |
| **ID-FT** (Instance-Dynamic) | `id_ft` | New symbol set generated each sample |
| **LF-FT** (Label-Flip) | `lf_ft` | Label→symbol assignments shuffled |

Symbol logic lives in `models/symbolAdapter/symbol_manager.py`. ED-FT rotation is driven by
`SPRInTSymbolEpochCallback` at `on_epoch_begin` (`NUM_WORKERS=0` enforced so symbol state is shared
in-process, not across DataLoader fork boundaries).

## Inference & Evaluation

Inference calls the **same** `SPRInTValidationManager` used during training — there is exactly one
evaluator, so training-validation and inference compute every metric with identical code.

```bash
# via the orchestrator (recommended)
CHECKPOINT_PATH=/path/to/llava-chest-regular-shot10_exp1 \
  strategy=regular datasets=chest bash submit_inference.sh

# or directly
python sprint_eval.py \
  --model-base   $MODEL_BASE \
  --model-path   /path/to/checkpoint \
  --image-folder $MEDFMC_ROOT \
  --question-file ./data/chest_test.json \
  --dataset chest --strategy regular \
  --modes original,fixed,fresh
```

**Reproducing the training validation set at inference** (to compare like-for-like): set
`USE_VALIDATION=true`, `VAL_SHOT`/`VAL_EXP` to match the trained run, and `MAX_VAL_SAMPLES=100`.
The orchestrator then passes `--val-subsample 100` so inference scores the **exact same** seeded
100 samples that training validated on.

Results JSON → `{LLAVA_DIR}/logs/json/results_{dataset}_{strategy}_{shots}shot_{split}_{timestamp}.json`.

## Validation Modes

Three modes run sequentially each epoch (configurable via `VALIDATION_MODES`):

| Mode | Description |
|------|-------------|
| `original` | Original label names — tests whether symbol training hurt base capability (**advisor's primary view**) |
| `fixed` | Same symbols used during training — tests how well the model learned the mapping |
| `fresh` | Brand-new symbols never seen in training — tests general in-context mapping ability |

Symbol strategies evaluate all three; `regular` runs `original` only.

## Metrics

MedFMC-official primary metric per task, plus supplementary text-generation metrics:

| Dataset | Type | Primary metric | Extra |
|---------|------|----------------|-------|
| Colon | Binary | AUC (token-`1` logit at step 0) + accuracy | macro_f1 |
| Chest | Multi-label (19) | **mAP** (per-class AP → mean) | macro_auc, macro_f1 |
| Endo | Multi-label (4) | **macro_auc** (per-class AUC → mean) | mAP, macro_f1 |

Multi-label AUC/mAP come from per-class binary "Does this show *class*? Answer Yes/No." probes:
for each (image, class) a softmax over `[No, Yes]` logits gives P(Yes), used as the per-class
ranking score. All metric math lives in `utils/evaluation_utils.py` (shared by both pipelines).

**Note on multi-label accuracy:** exact all-label match is near 0% and is expected, not a bug —
the reported cross-dataset accuracy is per-class average accuracy (`accuracy_aacc`).

## Active Datasets

| Dataset | Task | Type | Classes | trainval / test |
|---------|------|------|---------|-----------------|
| ColonPath | Colorectal lesion (WSI patch) | Binary | 2 | 2358 / 3296 |
| ChestDR | Thoracic abnormalities | Multi-label | 19 | 979 / 1161 |
| Endo | Colorectal lesion (endoscopy) | Multi-label | 4 | 929 / 881 |

Dataset configs in `config/data_config/`. Register new datasets in `master_config.py`.

## Checkpoints

Saved to `{output_dir}/checkpoint-{step}/` (LoRA adapter + `config.json`), with the best epoch
copied to `checkpoint-best/`. Best epoch is selected on the MedFMC primary metric (chest→mAP,
endo→macro_auc, colon→accuracy_aacc, AUC as tiebreaker). Trained symbol mappings are written to
`symbol_mappings.json` alongside the checkpoint.

## Training ↔ Inference Parity

The two pipelines are kept numerically aligned so a checkpoint's training-validation score
reproduces at inference on the same samples:

- **Same evaluator & metrics** — both run `SPRInTValidationManager` + `evaluation_utils.py`.
- **Same validation subset** — one seeded sampler, `random.Random(42).sample(data, 100)`
  (`validation.py`), independent of any global RNG. Deterministic given the same val JSON file.
- **Same precision** — bf16 model, vision tower bf16, `flash_attention_2`, and TF32 enabled in
  both (`sprint_eval.py` sets `allow_tf32=True` to match training's `--tf32 True`).
- **LoRA kept unmerged at inference** (`llava/model/builder.py`) so op order matches training-time
  validation (merging shifts bf16 AUC/mAP by 1–4%).
- **Same image preprocessing** — `image_aspect_ratio='pad'` (expand2square) propagated to inference.

Residual differences under ~1% on AUC/mAP are the **sampling floor** of ranking metrics on 100
samples (bf16/flash-attention non-determinism amplified by tie-sensitive ranking), not a bug — for
a stable number, evaluate the full split rather than a 100-sample subset.
