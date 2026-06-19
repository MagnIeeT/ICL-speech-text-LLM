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


@dataclass
class SymbolConfig:
    dynamic_symbols: bool = False
    no_symbols: bool = False
    update_strategy: SymbolUpdateStrategy = SymbolUpdateStrategy.PER_EPOCH
    symbol_type: str = "two_token"
    seed: Optional[int] = None
    validation_modes: str = "fixed,original,fresh"
    swap_labels: bool = False


@dataclass
class DataConfig:
    dataset_type: str = "voxceleb"
    batch_size: int = 1
    max_samples: int = 10
    split: str = "test"
    val_batch_size: Optional[int] = 1
    val_max_samples: int = 200
    val_frequency: int = 1
    val_dataset_type: Optional[str] = None
    num_examples: int = 5
    val_num_examples: int = 5
    num_workers: int = 2
    input_mode: str = "speech_only"   # speech_only | text_only
    fewshot_mode: str = "text"        # text | speech


from utils.environment import get_env_path

@dataclass
class DifferentiableSymbolConfig:
    enabled: bool = False
    num_slots: int = 25
    slot_vocab_size: int = 100     # K: private vocab tokens per slot (non-overlapping)
    slot_only: bool = False        # freeze LoRA, train router only
    tau: float = 1.0
    tau_min: float = 0.1
    tau_anneal_rate: float = 0.0001
    router_lr: float = 1e-2
    rotation_interval: int = 0      # 0 = rotate slot assignments per epoch; >0 = every N global steps
    integrity_alpha: float = 1.0
    integrity_beta: float = 1.0
    phase1_patience: int = 0    # >0: slot-only Phase 1 → auto-switch to LoRA when conf_mean plateaus

    @property
    def pool_size(self) -> int:
        """Total tokens needed: one private vocab per slot."""
        return self.num_slots * self.slot_vocab_size


@dataclass
class TrainingConfig:
    model_type: ModelType = ModelType.SALMONN
    salmonn_tokenizer_name: str = field(default_factory=lambda: get_env_path("SALMONN_TOKENIZER_NAME"))
    qwen_model_name: str = field(default_factory=lambda: get_env_path("QWEN_MODEL_NAME"))
    flamingo_model_name: str = field(default_factory=lambda: get_env_path("FLAMINGO_MODEL_NAME"))

    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    symbol_config: SymbolConfig = field(default_factory=SymbolConfig)
    diff_symbol_config: DifferentiableSymbolConfig = field(default_factory=DifferentiableSymbolConfig)
    data_config: DataConfig = field(default_factory=DataConfig)

    output_dir: str = field(default_factory=lambda: get_env_path("BASE_OUTPUT_DIR"))
    checkpoint_dir: str = field(default_factory=lambda: get_env_path("CHECKPOINT_DIR"))
    metrics_dir: str = field(default_factory=lambda: get_env_path("METRICS_DIR"))
    logs_dir: str = field(default_factory=lambda: get_env_path("LOGS_DIR"))
    
    run_name: str = "symbol_training_run"
    checkpoint_frequency: int = 1
    validate_before_training: bool = True

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
                "symbol_type": self.symbol_config.symbol_type,
                  "validation_modes": self.symbol_config.validation_modes,
                  "swap_labels": self.symbol_config.swap_labels,
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
        )

        symbol_config = SymbolConfig(
            dynamic_symbols=getattr(args, "dynamic_symbols", False),
            no_symbols=getattr(args, "no_symbols", False),
            update_strategy=SymbolUpdateStrategy(getattr(args, "symbol_update_strategy", "per_epoch")),
            symbol_type="two_token",
              validation_modes=getattr(args, "validation_modes", "fixed,original,fresh"),
              swap_labels=getattr(args, "swap_labels", False),
        )
        if symbol_config.no_symbols:
            symbol_config.dynamic_symbols = False

        diff_symbol_config = DifferentiableSymbolConfig(
            enabled=getattr(args, "diff_symbol_enabled", False),
            num_slots=getattr(args, "dspo_num_slots", 25),
            slot_vocab_size=getattr(args, "dspo_slot_vocab_size", 100),
            rotation_interval=getattr(args, "dspo_rotation_interval", 0),
            router_lr=getattr(args, "dspo_router_lr", 1e-3),
            tau_anneal_rate=getattr(args, "dspo_tau_anneal_rate", 0.0001),
            slot_only=getattr(args, "dspo_slot_only", False),
            phase1_patience=getattr(args, "dspo_phase1_patience", 0),
        )

        data_config = DataConfig(
            dataset_type=args.dataset_type,
            val_dataset_type=args.val_dataset_type if args.val_dataset_type else args.dataset_type,
            batch_size=args.batch_size,
            val_batch_size=getattr(args, "val_batch_size", None),
            max_samples=args.max_samples,
            val_max_samples=200 if args.max_samples == 0 else min(200, args.max_samples),
            split=getattr(args, "split", "test"),
            num_workers=getattr(args, "num_workers", 2),
            num_examples=getattr(args, "num_examples", 5),
            val_num_examples=getattr(args, "val_num_examples", 5),
            input_mode=getattr(args, "input_mode", "speech_only"),
            fewshot_mode=getattr(args, "fewshot_mode", "text"),
        )

        return cls(
            validate_before_training=not getattr(args, "no_validate_before_training", False),
            model_type=ModelType(args.model_type),
            salmonn_tokenizer_name=getattr(args, "salmonn_tokenizer_name", "lmsys/vicuna-13b-v1.1"),
            qwen_model_name=getattr(args, "qwen_model_name", "Qwen/Qwen2-Audio-7B-Instruct"),
            flamingo_model_name=getattr(args, "flamingo_model_name", "nvidia/audio-flamingo-3-hf"),
            lora_config=lora_config,
            symbol_config=symbol_config,
            diff_symbol_config=diff_symbol_config,
            data_config=data_config,
            output_dir=getattr(args, "output_dir", get_env_path("BASE_OUTPUT_DIR")),
            checkpoint_dir=getattr(args, "checkpoint_dir", get_env_path("CHECKPOINT_DIR")),
            metrics_dir=getattr(args, "metrics_dir", get_env_path("METRICS_DIR")),
            logs_dir=getattr(args, "logs_dir", get_env_path("LOGS_DIR")),
            run_name=args.run_name,
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
    parser.add_argument("--dataset_type", type=str, default="voxceleb")
    parser.add_argument("--val_dataset_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=None, help="Validation batch size (default: same as batch_size)")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--num_examples", type=int, default=5, help="Few-shot examples per training prompt")
    parser.add_argument("--val_num_examples", type=int, default=5, help="Few-shot examples per validation prompt")
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--dynamic_symbols", action="store_true")
    parser.add_argument("--diff_symbol_enabled", action="store_true", help="Enable Differentiable Symbolic Preference Optimization (D-SPO)")
    parser.add_argument("--dspo_num_slots", type=int, default=25, help="D-SPO: total number of slots (must be >= sum of all training dataset label counts)")
    parser.add_argument("--dspo_slot_vocab_size", type=int, default=100, help="D-SPO: private candidate tokens per slot")
    parser.add_argument("--dspo_slot_only", action="store_true", help="D-SPO: freeze LoRA, train router/slot matrix only")
    parser.add_argument("--dspo_phase1_patience", type=int, default=0, help="D-SPO: epochs without conf_mean improvement before switching from slot-only Phase 1 to joint LoRA Phase 2 (0=disabled)")
    parser.add_argument("--dspo_rotation_interval", type=int, default=0, help="D-SPO: rotate slot assignments every N global steps (0 = per epoch)")
    parser.add_argument("--dspo_router_lr", type=float, default=1e-3, help="D-SPO: router learning rate")
    parser.add_argument("--dspo_tau_anneal_rate", type=float, default=0.0001, help="D-SPO: tau annealing rate per step")
    parser.add_argument("--no_symbols", action="store_true")
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

    parser.add_argument("--input_mode", type=str, default="speech_only", choices=["speech_only", "text_only"], help="Query input modality")
    parser.add_argument("--fewshot_mode", type=str, default="text", choices=["text", "speech"], help="Few-shot example modality")
    parser.add_argument("--use_dpo", action="store_true", help="Use SymDPO trainer instead of cross-entropy")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta (preference margin temperature)")
    parser.add_argument("--no_validate_before_training", action="store_true", help="Skip baseline validation before epoch 1")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--metrics_dir", type=str, default=None)
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()
    if args.val_dataset_type is None:
        args.val_dataset_type = args.dataset_type
    return args
