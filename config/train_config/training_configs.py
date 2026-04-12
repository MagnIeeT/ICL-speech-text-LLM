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


class ModelType(Enum):
    """Supported model types."""
    SALMONN = "salmonn"
    LLAMA = "llama"
    QWEN = "qwen"


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 5
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0


@dataclass
class SymbolConfig:
    dynamic_symbols: bool = False
    update_strategy: SymbolUpdateStrategy = SymbolUpdateStrategy.PER_EPOCH
    symbol_type: str = "two_token"
    seed: Optional[int] = None


@dataclass
class DataConfig:
    dataset_type: str = "voxceleb"
    batch_size: int = 1
    max_samples: int = 10
    split: str = "test"
    val_batch_size: Optional[int] = 1
    val_max_samples: int = 200
    val_frequency: int = 1
    val_dataset_type: str = "voxceleb-hvb-meld_emotion-voxpopuli"
    num_examples: int = 0


@dataclass
class TrainingConfig:
    model_type: ModelType = ModelType.SALMONN
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    symbol_config: SymbolConfig = field(default_factory=SymbolConfig)
    data_config: DataConfig = field(default_factory=DataConfig)

    output_dir: str = "/home/leapers/weights/neeraja/ICL-speech-text-LLM/orchestrator_training"
    run_name: str = "symbol_training_run"
    checkpoint_frequency: int = 1
    save_every_n_steps: int = 500

    device: str = "cuda:0"
    mixed_precision: bool = False
    compile_model: bool = False
    log_level: str = "INFO"
    log_frequency: int = 10

    inference_mode: bool = False

    def __post_init__(self):
        if not self.device.startswith(("cuda", "cpu")):
            raise ValueError(f"Invalid device: {self.device}")
        if self.data_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.data_config.val_batch_size is None:
            self.data_config.val_batch_size = self.data_config.batch_size

    def get_training_output_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints", self.run_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
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
                "update_strategy": self.symbol_config.update_strategy.value,
                "symbol_type": self.symbol_config.symbol_type,
            },
            "data_config": {
                "dataset_type": self.data_config.dataset_type,
                "val_dataset_type": self.data_config.val_dataset_type,
                "batch_size": self.data_config.batch_size,
                "val_batch_size": self.data_config.val_batch_size,
                "max_samples": self.data_config.max_samples,
                "val_max_samples": self.data_config.val_max_samples,
            },
            "output_dir": self.output_dir,
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
            update_strategy=SymbolUpdateStrategy(getattr(args, "symbol_update_strategy", "per_epoch")),
            symbol_type="two_token",
        )

        data_config = DataConfig(
            dataset_type=args.dataset_type,
            val_dataset_type=getattr(args, "val_dataset_type", args.dataset_type),
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            val_max_samples=200 if args.max_samples == 0 else min(200, args.max_samples),
            split=getattr(args, "split", "test"),
        )

        return cls(
            model_type=ModelType(args.model_type),
            lora_config=lora_config,
            symbol_config=symbol_config,
            data_config=data_config,
            output_dir=args.output_dir,
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

    parser.add_argument("--model_type", type=str, default="salmonn", choices=["salmonn", "llama", "qwen"])
    parser.add_argument("--dataset_type", type=str, default="voxceleb")
    parser.add_argument("--val_dataset_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=100)

    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--dynamic_symbols", action="store_true")
    parser.add_argument(
        "--symbol_update_strategy",
        type=str,
        default="per_epoch",
        choices=["per_epoch", "per_instance"],
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()
    if args.val_dataset_type is None:
        args.val_dataset_type = args.dataset_type
    return args
