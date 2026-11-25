"""Training configuration for Symbol Adapter (LoRA + optional dynamic symbols)."""

import argparse
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SymbolUpdateStrategy(Enum):
    """How dynamic symbols are refreshed."""

    PER_EPOCH = "per_epoch"
    PER_INSTANCE = "per_instance"


class ValidationSymbolMode(Enum):
    """Validation symbol modes."""

    FIXED = "fixed"
    ORIGINAL = "original"
    FRESH = "fresh"
    FLIPPED = "flipped"
    INDEX = "index"      # generic names label1..labelK (sorted label order); a format never trained on


class ModelType(Enum):
    """Supported model types."""

    SALMONN = "salmonn"
    LLAMA = "llama"
    QWEN = "qwen"
    FLAMINGO = "flamingo"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 5
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100


@dataclass
class SymbolConfig:
    dynamic_symbols: bool = False
    no_symbols: bool = False
    update_strategy: SymbolUpdateStrategy = SymbolUpdateStrategy.PER_EPOCH
    symbol_token_size: int = 2
    validation_modes: str = "fixed,original,fresh"
    swap_labels: bool = False
    # Per-instance surrogate mixture. 0 keeps the single-regime behaviour. Above 0,
    # each instance draws the flipped (permuted real label) regime with this
    # probability and the synthetic-symbol regime otherwise. Dataset size and step
    # count are unchanged; only the surrogate vocabulary varies per instance.
    surrogate_mix: float = 0.0
    # --- Surface-form consistency objective (see models/symbolAdapter/consistency_loss.py) ---
    # Weight on the divergence between the two legends' definition distributions.
    # 0 disables the objective entirely and restores the single-legend path.
    # Set > 0 and each eligible instance runs two legends in one step:
    #     L = NLL(view_a) + NLL(view_b) + weight * D(p_a, p_b)
    # weight=0 with the objective otherwise configured is the control run: two
    # legends supervised, no coupling, which isolates what the divergence adds.
    consistency_weight: float = 0.0
    # The two legends to couple, comma-separated, from sym | flip | orig.
    #   sym,orig  couples synthetic symbols with canonical labels (adds canonical
    #             supervision the surrogate regimes never had)
    #   sym,sym   two independent symbol drawings; pure surrogate tuning, leaves
    #             canonical as a genuinely unseen surface at evaluation
    #   sym,flip  spans both surface families without ever pairing a word with
    #             its own definition
    consistency_views: str = "sym,flip"
    # Which divergence measures the disagreement: sym_kl | js | l2.
    consistency_divergence: str = "js"
    # Shuffle the legend bullet order independently in each view (and in the
    # single pass of ineligible datasets), so position cannot stand in for the
    # definition. Validation and test keep the fixed paper order.
    consistency_shuffle_legend: bool = False
    # Control run: the two-view path at weight 0 (divergence logged, not trained).
    consistency_control: bool = False
    # --- DG-Trap (see models/symbolAdapter/trap_loss.py, docs/chapter2_plan.md 13a) ---
    # Real label words only; each instance is shown as orig | goldswap | pairswap.
    trap_enabled: bool = False
    trap_p_orig: float = 1 / 3
    trap_p_goldswap: float = 1 / 3
    trap_p_pairswap: float = 1 / 3
    # symbol naming: every label of the task gets a fresh random 2-token string for this
    # instance (copy practice; names carry no meaning). 0 = off, old runs unchanged.
    trap_p_symbol: float = 0.0
    # comma-separated symbol-map JSON files whose strings are never drawn in training
    # (keeps the test symbol draws unseen)
    trap_symbol_exclude: str = ""
    # tokens scored by the conditional margin: "all" = whole completion incl. other gold
    # labels and EOS (A0-A3 behaviour); "gold" = only the damaged gold label's tokens
    trap_cond_span: str = "all"
    trap_nn_frac: float = 0.5         # goldswap partner = gold's nearest-neighbour definition with this prob, else random non-gold
    trap_dropout_max: float = 0.2      # per-instance harmless (non-deciding) word dropout rate ~ U(0, max)
    trap_lambda: float = 0.5           # weight on the trap margin (goldswap instances)
    trap_cond_lambda: float = 0.5      # weight on the conditional margin; 0 = no second pass
    trap_cond_frac: float = 0.3        # fraction of instances that get the conditional term
    trap_cond_mode: str = "deciding"   # deciding | random  (ablation A5: remove as many random words)
    trap_beta: float = 2.0
    trap_margin: float = 0.5
    trap_cond_margin: float = 0.5
    trap_deciding_words: str = "config/data_config/deciding_words.json"
    # Size of the pre-computed pool of bijective label->symbol mappings (K in the paper).
    # Per epoch: pool[epoch % K], per instance: random.choice(pool)
    num_symbol_mappings: int = 100
    # Seed for the 'flipped' validation mode's real-label derangement.
    # Same seed → identical flipped legend across models (fair comparison).
    val_flip_seed: int = 0


@dataclass
class DataConfig:
    dataset_type: str = "hvb-cremad-minds14_en"
    batch_size: int = 1
    max_samples: int = 10
    split: str = "test"
    val_batch_size: Optional[int] = 1
    val_max_samples: int = 500
    val_frequency: int = 1
    val_dataset_type: Optional[str] = None
    num_examples: int = 5
    num_examples_min: int = -1         # >=0 → random per-prompt count U[min, num_examples] during TRAINING (min=0 allows zero-shot prompts); -1 = fixed count
    val_num_examples: int = 5
    num_workers: int = 2
    input_mode: str = "speech_only"   # speech_only | text_only
    fewshot_mode: str = "text"        # text | speech
    no_legend: bool = False           # strip label descriptions from the prompt (Wei-style: meaning from exemplars/symbols only)
    fewshot_per_class: bool = False   # few-shot: True → num_examples PER class (full coverage); False → num_examples TOTAL (class-balanced)
    definition_variant: str = "orig"  # definition-sensitivity ablation: pick an alternate-wording legend from a dataset's prompt_variants; "orig" = default template
    strip_gold_definition: bool = False  # counterfactual eval: drop only the gold class's definition (label kept)
    strip_extra_definitions: int = 0     # with strip_gold_definition: also drop N random non-gold definitions


from utils.environment import get_env_path


class _LegacyDSPOConfig:
    """Unpickling shim for checkpoints written before D-SPO was removed from main.

    Checkpoints at or before v1.0-paper-submission pickled a TrainingConfig whose
    `diff_symbol_config` field held a DifferentiableSymbolConfig instance. Pickle
    resolves that class by name against this module at load time, so without a
    binding under the historical name torch.load raises

        Can't get attribute 'DifferentiableSymbolConfig' on <module
        'config.train_config.training_configs'>

    and every existing checkpoint becomes unloadable. D-SPO itself lives on
    chapter-2; the restored attribute is inert and nothing reads it.
    Do not add fields here or use this class for anything else.
    """


# Historical name, kept bound for pickle compatibility only.
DifferentiableSymbolConfig = _LegacyDSPOConfig


@dataclass
class TrainingConfig:
    model_type: ModelType = ModelType.SALMONN
    salmonn_tokenizer_name: str = field(default_factory=lambda: get_env_path("SALMONN_TOKENIZER_NAME"))
    qwen_model_name: str = field(default_factory=lambda: get_env_path("QWEN_MODEL_NAME"))
    flamingo_model_name: str = field(default_factory=lambda: get_env_path("FLAMINGO_MODEL_NAME"))

    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    symbol_config: SymbolConfig = field(default_factory=SymbolConfig)
    data_config: DataConfig = field(default_factory=DataConfig)

    output_dir: str = field(default_factory=lambda: get_env_path("BASE_OUTPUT_DIR"))
    checkpoint_dir: str = field(default_factory=lambda: get_env_path("CHECKPOINT_DIR"))
    metrics_dir: str = field(default_factory=lambda: get_env_path("METRICS_DIR"))
    logs_dir: str = field(default_factory=lambda: get_env_path("LOGS_DIR"))
    
    run_name: str = "symbol_training_run"
    seed: Optional[int] = None
    checkpoint_frequency: int = 1
    validate_before_training: bool = True
    torch_profile: bool = False

    device: str = "cuda:0"

    inference_mode: bool = False

    def __post_init__(self):
        if not self.device.startswith(("cuda", "cpu")):
            raise ValueError(f"Invalid device: {self.device}")
        if self.data_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.data_config.val_batch_size is None:
            self.data_config.val_batch_size = self.data_config.batch_size
            
        # Fallback to output_dir if granular paths are not set
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if not self.metrics_dir:
            self.metrics_dir = os.path.join(self.output_dir, "metrics")
        if not self.logs_dir:
            self.logs_dir = os.path.join(self.output_dir, "logs")

    def get_training_output_dir(self) -> str:
        """Legacy helper, now points to checkpoint subdir."""
        return os.path.join(self.checkpoint_dir, self.run_name)
        
    def get_checkpoint_dir(self) -> str:
        return os.path.join(self.checkpoint_dir, self.run_name)

    def get_metrics_dir(self) -> str:
        return os.path.join(self.metrics_dir, self.run_name)

    def get_logs_dir(self) -> str:
        return os.path.join(self.logs_dir, self.run_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "salmonn_tokenizer_name": self.salmonn_tokenizer_name,
            "qwen_model_name": self.qwen_model_name,
            "flamingo_model_name": self.flamingo_model_name,
            "lora_config": {
                "rank": self.lora_config.rank,
                "alpha": self.lora_config.alpha,
                "dropout": self.lora_config.dropout,
                "learning_rate": self.lora_config.learning_rate,
                "weight_decay": self.lora_config.weight_decay,
                "epochs": self.lora_config.epochs,
                "gradient_accumulation_steps": self.lora_config.gradient_accumulation_steps,
                "max_grad_norm": self.lora_config.max_grad_norm,
            },
            "symbol_config": {
                "dynamic_symbols": self.symbol_config.dynamic_symbols,
                "no_symbols": self.symbol_config.no_symbols,
                "update_strategy": self.symbol_config.update_strategy.value,
                "symbol_token_size": self.symbol_config.symbol_token_size,
                  "validation_modes": self.symbol_config.validation_modes,
                  "swap_labels": self.symbol_config.swap_labels,
                "trap": {k: getattr(self.symbol_config, k) for k in self.symbol_config.__dataclass_fields__
                         if k.startswith("trap_")},
            },
            "data_config": {
                "dataset_type": self.data_config.dataset_type,
                "val_dataset_type": self.data_config.val_dataset_type,
                "batch_size": self.data_config.batch_size,
                "val_batch_size": self.data_config.val_batch_size,
                "max_samples": self.data_config.max_samples,
                "val_max_samples": self.data_config.val_max_samples,
                "num_workers": self.data_config.num_workers,
            },
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "metrics_dir": self.metrics_dir,
            "logs_dir": self.logs_dir,
            "run_name": self.run_name,
            "device": self.device,
        }

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        lora_config = LoRAConfig(
            learning_rate=args.lora_lr,
            epochs=args.lora_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=getattr(args, "warmup_steps", 100),
        )

        symbol_config = SymbolConfig(
            dynamic_symbols=getattr(args, "dynamic_symbols", False),
            no_symbols=getattr(args, "no_symbols", False),
            update_strategy=SymbolUpdateStrategy(getattr(args, "symbol_update_strategy", "per_epoch")),
            symbol_token_size=2,
            validation_modes=getattr(args, "validation_modes", "fixed,original,fresh"),
            swap_labels=getattr(args, "swap_labels", False),
            num_symbol_mappings=getattr(args, "num_symbol_mappings", 100),
            surrogate_mix=getattr(args, "surrogate_mix", 0.0),
            consistency_weight=getattr(args, "consistency_weight", 0.0),
            consistency_views=getattr(args, "consistency_views", "sym,flip"),
            consistency_divergence=getattr(args, "consistency_divergence", "js"),
            consistency_shuffle_legend=getattr(args, "consistency_shuffle_legend", False),
            consistency_control=getattr(args, "consistency_control", False),
            val_flip_seed=getattr(args, "val_flip_seed", 0),
            trap_enabled=getattr(args, "trap_enabled", False),
            trap_p_orig=getattr(args, "trap_p_orig", 1 / 3),
            trap_p_goldswap=getattr(args, "trap_p_goldswap", 1 / 3),
            trap_p_pairswap=getattr(args, "trap_p_pairswap", 1 / 3),
            trap_p_symbol=getattr(args, "trap_p_symbol", 0.0),
            trap_symbol_exclude=getattr(args, "trap_symbol_exclude", ""),
            trap_cond_span=getattr(args, "trap_cond_span", "all"),
            trap_nn_frac=getattr(args, "trap_nn_frac", 0.5),
            trap_dropout_max=getattr(args, "trap_dropout_max", 0.2),
            trap_lambda=getattr(args, "trap_lambda", 0.5),
            trap_cond_lambda=getattr(args, "trap_cond_lambda", 0.5),
            trap_cond_frac=getattr(args, "trap_cond_frac", 0.3),
            trap_cond_mode=getattr(args, "trap_cond_mode", "deciding"),
            trap_beta=getattr(args, "trap_beta", 2.0),
            trap_margin=getattr(args, "trap_margin", 0.5),
            trap_cond_margin=getattr(args, "trap_cond_margin", 0.5),
            trap_deciding_words=getattr(args, "trap_deciding_words",
                                        "config/data_config/deciding_words.json"),
        )
        if symbol_config.no_symbols:
            symbol_config.dynamic_symbols = False

        data_config = DataConfig(
            dataset_type=args.dataset_type,
            val_dataset_type=args.val_dataset_type if args.val_dataset_type else args.dataset_type,
            batch_size=args.batch_size,
            val_batch_size=getattr(args, "val_batch_size", None),
            max_samples=args.max_samples,
            val_max_samples=(
                min(getattr(args, "val_max_samples", 500), args.max_samples)
                if args.max_samples > 0
                else getattr(args, "val_max_samples", 500)
            ),
            split=getattr(args, "split", "test"),
            num_workers=getattr(args, "num_workers", 2),
            num_examples=getattr(args, "num_examples", 5),
            num_examples_min=getattr(args, "num_examples_min", -1),
            val_num_examples=getattr(args, "val_num_examples", 5),
            input_mode=getattr(args, "input_mode", "speech_only"),
            fewshot_mode=getattr(args, "fewshot_mode", "text"),
            no_legend=getattr(args, "no_legend", False),
            fewshot_per_class=getattr(args, "fewshot_per_class", False),
            definition_variant=getattr(args, "definition_variant", "orig"),
        )

        return cls(
            validate_before_training=not getattr(args, "no_validate_before_training", False),
            torch_profile=getattr(args, "torch_profile", False),
            model_type=ModelType(args.model_type),
            salmonn_tokenizer_name=getattr(args, "salmonn_tokenizer_name", "lmsys/vicuna-13b-v1.1"),
            qwen_model_name=getattr(args, "qwen_model_name", "Qwen/Qwen2-Audio-7B-Instruct"),
            flamingo_model_name=getattr(args, "flamingo_model_name", "nvidia/audio-flamingo-3-hf"),
            lora_config=lora_config,
            symbol_config=symbol_config,
            data_config=data_config,
            # argparse defines these with default None, so getattr always finds the
            # attribute -- fall back to the .env value only when the flag is unset.
            output_dir=getattr(args, "output_dir", None) or get_env_path("BASE_OUTPUT_DIR"),
            checkpoint_dir=getattr(args, "checkpoint_dir", None) or get_env_path("CHECKPOINT_DIR"),
            metrics_dir=getattr(args, "metrics_dir", None) or get_env_path("METRICS_DIR"),
            logs_dir=getattr(args, "logs_dir", None) or get_env_path("LOGS_DIR"),
            run_name=args.run_name,
            seed=getattr(args, "seed", None),
            device=args.device,
        )


def create_training_config(**kwargs) -> TrainingConfig:
    return TrainingConfig(**kwargs)


def get_default_config(dynamic_symbols: bool = False, symbol_update_strategy: str = "per_epoch") -> TrainingConfig:
    config = TrainingConfig()
    config.symbol_config.dynamic_symbols = dynamic_symbols
    config.symbol_config.update_strategy = SymbolUpdateStrategy(symbol_update_strategy)
    return config


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbol Adapter Training (LoRA-only)")

    parser.add_argument("--model_type", type=str, default="salmonn", choices=["salmonn", "llama", "qwen", "flamingo"])
    parser.add_argument("--salmonn_tokenizer_name", type=str, default="lmsys/vicuna-13b-v1.1")
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--flamingo_model_name", type=str, default="nvidia/audio-flamingo-3-hf")
    parser.add_argument("--dataset_type", type=str, default="hvb-cremad-minds14_en",
                        help="Training dataset(s), dash-separated. Default = the paper's fine-tuning mixture")
    parser.add_argument("--val_dataset_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=None, help="Validation batch size (default: same as batch_size)")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--val_max_samples", type=int, default=500, help="Max validation samples per dataset (0 = full)")
    parser.add_argument("--num_examples", type=int, default=5, help="Few-shot examples per training prompt (fixed count, or the MAX when --num_examples_min > 0)")
    parser.add_argument("--num_examples_min", type=int, default=-1, help="If >=0, random per-prompt count U[min, num_examples] during training (min=0 → some prompts zero-shot; Wei-style uses 2). -1 = fixed count. Eval stays fixed at num_examples")
    parser.add_argument("--val_num_examples", type=int, default=5, help="Few-shot examples per validation prompt")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--dynamic_symbols", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR scheduler warmup steps")
    parser.add_argument("--no_symbols", action="store_true")
    parser.add_argument("--surrogate_mix", type=float, default=0.0,
                        help="Probability per instance of using the flipped (permuted real label) "
                             "regime instead of synthetic symbols. 0 = off. No extra data: the "
                             "surrogate vocabulary varies per instance within the same budget")
    parser.add_argument("--consistency_weight", type=float, default=0.0,
                        help="Weight on the surface-form consistency term. 0 = off (single "
                             "legend per step, current behaviour). Above 0, each eligible "
                             "instance runs two legends in one step and is penalised for "
                             "answering them differently once mapped back to definitions. "
                             "Pass 0 together with the other consistency flags for the "
                             "two-view-no-coupling control run")
    parser.add_argument("--consistency_views", type=str, default="sym,flip",
                        help="The two legends to couple, from sym|flip|orig. 'sym,orig' adds "
                             "canonical supervision; 'sym,sym' stays pure surrogate tuning and "
                             "leaves canonical unseen; 'sym,flip' spans both surface families "
                             "without pairing any word with its own definition. 'sym,same' "
                             "reuses view A's symbols in view B; with "
                             "--consistency_shuffle_legend the views differ only in line order")
    parser.add_argument("--consistency_divergence", type=str, default="js",
                        choices=["sym_kl", "js", "l2"],
                        help="Divergence for the consistency term. sym_kl is unbounded with the "
                             "strongest gradient; js is bounded by log 2; l2 is mildest")
    parser.add_argument("--consistency_shuffle_legend", action="store_true",
                        help="Shuffle the legend bullet order independently per view and per "
                             "instance during consistency training, so the shared invariant is "
                             "the definition, not the line position. Validation is unaffected")
    parser.add_argument("--consistency_control", action="store_true",
                        help="Control run: take the two-view path even at --consistency_weight 0, "
                             "so it matches the weighted run in data and passes but has no "
                             "coupling. The divergence is logged, not trained")
    # DG-Trap (definition-trap training); see models/symbolAdapter/trap_loss.py
    parser.add_argument("--trap_enabled", action="store_true",
                        help="DG-Trap: per instance orig|goldswap|pairswap naming over real label words, "
                             "harmless word dropout, trap margin and conditional margin. Needs batch_size 1; "
                             "incompatible with --swap_labels, --surrogate_mix and the consistency objective")
    parser.add_argument("--trap_p_orig", type=float, default=1 / 3)
    parser.add_argument("--trap_p_goldswap", type=float, default=1 / 3)
    parser.add_argument("--trap_p_pairswap", type=float, default=1 / 3)
    parser.add_argument("--trap_p_symbol", type=float, default=0.0,
                        help="symbol naming: all labels renamed to fresh random 2-token strings; 0 = off")
    parser.add_argument("--trap_symbol_exclude", type=str, default="",
                        help="comma-separated symbol-map JSON files whose strings training never draws")
    parser.add_argument("--trap_cond_span", type=str, default="all", choices=["all", "gold"],
                        help="conditional margin scores the whole completion (all) or only the "
                             "damaged gold label's tokens (gold)")
    parser.add_argument("--trap_nn_frac", type=float, default=0.5,
                        help="goldswap partner is the gold's nearest-neighbour definition with this probability")
    parser.add_argument("--trap_dropout_max", type=float, default=0.2,
                        help="harmless word dropout rate per instance ~ U(0, max); 0 = off")
    parser.add_argument("--trap_lambda", type=float, default=0.5,
                        help="weight on the trap margin; 0 = measured and logged, not trained")
    parser.add_argument("--trap_cond_lambda", type=float, default=0.5,
                        help="weight on the conditional margin; 0 = off (no second pass)")
    parser.add_argument("--trap_cond_frac", type=float, default=0.3)
    parser.add_argument("--trap_cond_mode", type=str, default="deciding", choices=["deciding", "random"])
    parser.add_argument("--trap_beta", type=float, default=2.0)
    parser.add_argument("--trap_margin", type=float, default=0.5)
    parser.add_argument("--trap_cond_margin", type=float, default=0.5)
    parser.add_argument("--trap_deciding_words", type=str, default="config/data_config/deciding_words.json")
    parser.add_argument("--swap_labels", action="store_true", help="Swap labels (e.g., positive<->negative) during prompt/completion rewriting")
    parser.add_argument(
        "--validation_modes",
        type=str,
        default="fixed,original,fresh",
        help="Comma-separated validation symbol modes: fixed,original,fresh",
    )
    parser.add_argument(
        "--symbol_update_strategy",
        type=str,
        default="per_epoch",
        choices=["per_epoch", "per_instance"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for Python/NumPy/torch. The paper reports mean +/- std over three seeds; omit for an unseeded run",
    )
    parser.add_argument(
        "--val_flip_seed",
        type=int,
        default=0,
        help="Seed for the 'flipped' validation mode real-label derangement (same seed = same flip across models)",
    )
    parser.add_argument(
        "--num_symbol_mappings",
        type=int,
        default=100,
        help="Size of the pre-computed pool of label->symbol mappings (K in the paper; per_epoch: pool[epoch%%K], per_instance: random.choice(pool))",
    )

    parser.add_argument("--input_mode", type=str, default="speech_only", choices=["speech_only", "text_only", "speech_text"], help="Query input modality; speech_text presents audio and a labelled transcript line together")
    parser.add_argument("--fewshot_mode", type=str, default="text", choices=["text", "speech"], help="Few-shot example modality")
    parser.add_argument("--no_legend", action="store_true", help="Strip label descriptions from the prompt legend (meaning conveyed only by exemplars/symbols; Wei-style)")
    parser.add_argument("--fewshot_per_class", action="store_true", help="Few-shot: num_examples PER class (full coverage) instead of TOTAL (class-balanced)")
    parser.add_argument("--definition_variant", type=str, default="orig", help="Definition-sensitivity ablation: select an alternate-wording legend from a dataset's prompt_variants (e.g. paraphrase, terse); 'orig' uses the default template")
    parser.add_argument("--use_dpo", action="store_true", help="Use the SymDPO preference trainer instead of cross-entropy (Chapter 2)")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta: log-ratio margin temperature")
    parser.add_argument("--no_validate_before_training", action="store_true", help="Skip baseline validation before epoch 1")
    parser.add_argument("--torch_profile", action="store_true", help="Enable PyTorch profiler on epoch 1")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--metrics_dir", type=str, default=None)
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()
    if args.val_dataset_type is None:
        args.val_dataset_type = args.dataset_type
    return args
