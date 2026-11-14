# DG-ICL: Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding

Reference implementation for *Definition-Guided In-Context Learning for Zero-Shot
Spoken Language Understanding* (Neeraj Agrawal, Sriram Ganapathy; Department of
Electrical Engineering, Indian Institute of Science).

Adapting an SLU system to a new or revised label taxonomy normally means new
labeled data and another round of training. DG-ICL instead fine-tunes a
speech-text LLM so that a category's meaning is carried by its natural-language
*definition* rather than its label name. During training every class is renamed
on the fly, either to a synthetic symbol (**DG-ICL-Sym**) or to a permuted real
label (**DG-ICL-Flip**), while its definition stays fixed. This removes the lexical
shortcut and forces the model to ground audio in the definition it was given. A
new category can then be introduced at inference just by defining it in the
prompt, zero-shot.

The repository state at submission is tagged `v1.0-paper-submission`.

## Supported models

| Model | Params | Backend | `--model_type` | Conda env |
|---|---|---|---|---|
| Audio Flamingo 3 (`nvidia/audio-flamingo-3-hf`) | ~8B | `models/backends/custom_flamingo.py` | `flamingo` | `flamingo` |
| Qwen2-Audio-7B-Instruct | ~8B | `models/backends/custom_qwen.py` | `qwen` | `qwen` |
| SALMONN (Whisper + BEATs + Vicuna-13B) | ~13B | `models/backends/custom_salmonn.py` | `salmonn` | `salmonn2` |

Audio encoders and LLM weights stay frozen; adaptation is LoRA on the attention
projections (`q,k,v,o` for AF3 and Qwen2-Audio; `q,v` plus the speech Q-Former
and projection for SALMONN).

## Setup

**1. Environment**, one per backend:

```bash
conda env create -f requirements/environment_qwen.yml       # qwen
conda env create -f requirements/environment_flamingo.yml   # flamingo (needs transformers>=5.0)
conda env create -f requirements/environment_salmonn2.yml   # salmonn
conda activate qwen
```

**2. Paths**. Copy the template and fill in real paths:

```bash
cp .env.example .env
```

`.env` holds the HF model ids, the output/checkpoint/metrics/log directories and
one `<NAME>_{TRAIN,VAL,TEST}_PATH` per dataset. It is gitignored.

**3. SALMONN weights** (only for `--model_type salmonn`):

- `salmonn_v1.pth` → `SALMONN_CKPT_PATH`, from [tsinghua-ee/SALMONN](https://huggingface.co/tsinghua-ee/SALMONN)
- BEATs `cpt2` → `BEATS_CKPT_PATH`, from [WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2](https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2)
- Whisper (`openai/whisper-large-v2`) and Vicuna (`lmsys/vicuna-13b-v1.1`) download from their HF ids on first run.

## Datasets

All audio is resampled to 16 kHz and all splits follow the official partitions.
`utils/process_*.py` convert each raw release into the `{audio, label, text}`
format the loaders expect; each takes `--raw_dir` and `--out_base` and prints the
`.env` keys to set. Register a new dataset in
`config/data_config/master_config.py`.

| Dataset | Task | Classes | Role |
|---|---|---|---|
| HVB | Dialogue acts (multi-label) | 18 | Fine-tuning |
| CREMA-D | Speech emotion | 6 | Fine-tuning |
| MInDS-14 en | Banking intent | 14 | Fine-tuning |
| Skit-S2I | Indian-accented banking intent | 14 | Held-out |
| RAVDESS-Song | Sung emotion | 6 | Held-out |
| MInDS-14 fr / ko | Cross-lingual intent | 14 | Held-out |
| Google Speech Commands | Keyword spotting | 10 | Capability probe |
| HeySQuAD | Spoken QA (generative) | n/a | Capability probe |

SLUE VoxCeleb and VoxPopuli are deliberately excluded: both appear in the
pre-training corpora of the foundation models under test.

For the zero-shot clustering evaluation, `utils/build_custom_taxonomy.py` groups
the fine-grained Skit-S2I and MInDS-14 fr/ko intents into three coarse
categories under three naming styles: *meaningful*, *neutral* and *symbolic*
(`skit_cat_mean`, `minds14_fr_cat_neu`, `minds14_ko_cat_sym`, …).

## Training

The four conditions in the paper differ only in how the label surface form is
chosen; the prompt template, definitions and hyperparameters are identical.

```bash
# DG-ICL-Sym: surrogate symbols, one of K=100 bijective maps per instance
python train.py --model_type qwen --run_name dgicl_sym_seed1 --seed 1 \
  --dataset_type hvb-cremad-minds14_en --val_dataset_type hvb-cremad-minds14_en \
  --dynamic_symbols --symbol_update_strategy per_instance --num_symbol_mappings 100 \
  --num_examples 0 --lora_epochs 5 --max_samples 0

# DG-ICL-Flip: real labels permuted per instance, definitions unchanged
python train.py --model_type qwen --run_name dgicl_flip_seed1 --seed 1 \
  --dataset_type hvb-cremad-minds14_en \
  --no_symbols --swap_labels --symbol_update_strategy per_instance \
  --num_examples 0 --lora_epochs 5 --max_samples 0

# Instruction Tuning (IT) baseline: canonical labels, definition-guided prompt
python train.py --model_type qwen --run_name it_seed1 --seed 1 \
  --dataset_type hvb-cremad-minds14_en \
  --no_symbols --num_examples 0 --lora_epochs 5 --max_samples 0

# Symbol Tuning (Wei et al.): symbols, NO definitions, meaning from demonstrations
python train.py --model_type qwen --run_name symtune_seed1 --seed 1 \
  --dataset_type hvb-cremad-minds14_en \
  --dynamic_symbols --symbol_update_strategy per_instance --num_symbol_mappings 100 \
  --no_legend --num_examples 5 --num_examples_min 2 --fewshot_per_class \
  --lora_epochs 5 --max_samples 0

# Flipped-Label Tuning (Agrawal et al.): permuted labels + definitions + 0-5 demos
python train.py --model_type qwen --run_name fliptune_seed1 --seed 1 \
  --dataset_type hvb-cremad-minds14_en \
  --no_symbols --swap_labels --symbol_update_strategy per_instance \
  --num_examples 5 --num_examples_min 0 --lora_epochs 5 --max_samples 0
```

Hyperparameters for the reported runs come from the `LoRAConfig` defaults in
`config/train_config/training_configs.py`: rank `8`, `alpha=32`, dropout `0.1`,
AdamW at `1e-5` with a cosine schedule and 100 warmup steps; effective batch size 8
(AF3 `--batch_size 2 --gradient_accumulation_steps 4`; Qwen2-Audio and SALMONN
`--batch_size 1 --gradient_accumulation_steps 8`). AF3 and Qwen2-Audio train up
to 5 epochs over three seeds; SALMONN runs 3 epochs at `--max_samples 2000` from
a single seed as a scale reference. Checkpoints are selected on mean zero-shot
macro-F1 over the fine-tuning tasks' validation splits with original labels
(`--validation_modes original`).

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--model_type` | `salmonn` | `qwen`, `flamingo`, `salmonn` |
| `--dataset_type` | `hvb-cremad-minds14_en` | Training dataset(s), dash-separated |
| `--val_dataset_type` | = `--dataset_type` | Validation datasets |
| `--run_name` | *required* | Identifies logs and checkpoints |
| `--seed` | unset | Seeds Python/NumPy/torch |
| `--max_samples` | `100` | Training samples per dataset (0 = all) |
| `--num_examples` | `5` | Demonstrations per prompt (0 = zero-shot, as in DG-ICL) |
| `--num_examples_min` | `-1` | If ≥0, draw a per-prompt count from U[min, num_examples] |
| `--lora_epochs` / `--lora_lr` | `5` / `1e-5` | Epochs, LoRA learning rate |
| `--batch_size` / `--gradient_accumulation_steps` | `1` / `8` | Effective batch = product |
| `--no_symbols` | off | Keep canonical label names (IT and Flip conditions) |
| `--dynamic_symbols` | off | Draw surrogate symbols from the pre-computed pool |
| `--symbol_update_strategy` | `per_epoch` | `per_epoch` or `per_instance` (the paper uses per-instance) |
| `--num_symbol_mappings` | `100` | K, the pool of bijective label→symbol maps |
| `--swap_labels` | off | Permute label assignments (with `--no_symbols` → DG-ICL-Flip) |
| `--no_legend` | off | Strip the definitions from the prompt (Symbol Tuning baseline) |
| `--definition_variant` | `orig` | Alternate-wording legend for the definition-sensitivity ablation |
| `--input_mode` | `speech_only` | `speech_only` or `text_only` (transcript queries) |
| `--validation_modes` | `fixed,original,fresh` | See below |

Checkpoints land in `$CHECKPOINT_DIR/<run_name>/` as
`lora_epoch{N}_periodic.pt` and `lora_epoch{N}_final.pt`, each holding
`model_state` (trainable weights only), `config` and `symbol_mappings`.

## Evaluation

```bash
python inference.py --model_type qwen --run_name eval_sym \
  --checkpoint_path $CHECKPOINT_DIR/dgicl_sym_seed1/lora_epoch5_final.pt \
  --dataset_type hvb-cremad-skit_s2i-ravdess_song-minds14_en-minds14_fr-minds14_ko \
  --validation_modes original,fresh --num_examples 0 --split test --seed 1
```

Label modes (`--validation_modes`, comma-separated) keep the prompt template
fixed and change only the label surface form:

| Mode | Description |
|---|---|
| `original` | Canonical semantic names. Does surrogate tuning cost anything on familiar labels? |
| `fresh` | Novel symbols never seen in training. Tests prior-free definition following |
| `fixed` | The symbols seen during training. Did the mapping itself get learned? |
| `flipped` | Real labels deranged against their definitions. Tests counterfactual steerability |

`--val_flip_seed` fixes the derangement so the `flipped` legend is identical
across models. Classification is scored with macro-F1 where generations are
string-matched against the candidate labels and invalid generations count as
misclassifications (`macro_f1_with_invalid` in `utils/evaluation_utils.py`);
HeySQuAD reports EM and token-F1; Speech Commands reports accuracy. Decoding is
greedy throughout.

## Layout

```
train.py                              # training entrypoint
inference.py                          # evaluation entrypoint
config/
  train_config/training_configs.py    # TrainingConfig + CLI
  data_config/master_config.py        # dataset registry (DATASET_CONFIGS)
  data_config/*_config.py             # per-dataset labels, definitions, prompt variants
dataload/
  multi_task_dataset.py               # multi-task sampling, few-shot assembly, legend construction
  {qwen,flamingo,salmon}_processor.py # per-backend tokenization + audio features
models/
  backends/custom_{qwen,flamingo,salmonn}.py
  symbolAdapter/
    symbol_manager.py                 # symbol generation, per-dataset maps, batch substitution
    symbol_training.py                # training loop
    validation.py                     # multi-mode validation runner
utils/
  build_custom_taxonomy.py            # 3-cluster taxonomies for the clustering eval
  process_*.py                        # dataset preprocessing
  evaluation_utils.py                 # macro-F1, QA metrics
```

Symbols are random lowercase strings kept only if they tokenize to exactly two
subword units in every prompt position (standalone, leading space, trailing
comma), so target length is uniform across classes. See
`generate_fresh_symbols()` in `models/symbolAdapter/symbol_manager.py`. Labels
shared across datasets (e.g. `neutral` in both CREMA-D and RAVDESS-Song) always
receive independent symbols.

## Citation

```bibtex
@article{agrawal2026dgicl,
  title   = {Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding},
  author  = {Agrawal, Neeraj and Ganapathy, Sriram},
  year    = {2026}
}
```
