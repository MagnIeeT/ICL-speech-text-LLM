"""
Training Configuration Classes for Symbol Adapter
Centralized configuration management for different training modes
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class TrainingMode(Enum):
    """Training mode options"""
    LORA_FIRST = "lora_first"          # Original alternating LoRA-MLP cycles
    MLP_FIRST = "mlp_first"            # MLP first, then alternating cycles
    JOINT_TRAINING = "joint_training"  # Train MLP and LoRA simultaneously
    BYPASS_MLP_SYM = "bypass_mlp_sym"  # Pure LoRA training with dynamic symbols
    BYPASS_MLP_ORG = "bypass_mlp_org"  # Pure LoRA training (no symbols)
    LORA_MLP_JOINT = "lora_mlp_joint"  # ✅ ADD: LoRA only → MLP only → Joint

class SymbolMode(Enum):
    """Symbol handling modes"""
    FIXED = "fixed"                     # Same symbols throughout training
    DYNAMIC_PER_EPOCH = "dynamic_per_epoch"  # New symbols each epoch
    DYNAMIC_PER_CYCLE = "dynamic_per_cycle"  # New symbols each cycle
    NO_SYMBOLS = "no_symbols"          # No symbol replacement

class ModelType(Enum):
    """Supported model types"""
    SALMONN = "salmonn"
    LLAMA = "llama"
    QWEN = "qwen"

@dataclass
class MLPConfig:
    """MLP-specific configuration"""
    use_input_mlp: bool = True
    use_output_mlp: bool = False
    hidden_dim: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    activation: str = "relu"
    
    # Training parameters
    epochs: int = 3
    initial_epochs: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    scheduler: str = "linear"  # ✅ ADD THIS
    warmup_steps: float = 100  # ✅ ADD THIS
    
    def __post_init__(self):
        if not self.use_input_mlp and not self.use_output_mlp:
            logging.info("MLPConfig: Both input and output MLPs disabled (bypass_mlp mode)")
        elif self.use_input_mlp and self.use_output_mlp:
            logging.info("MLPConfig: Both input and output MLPs enabled")
        elif self.use_input_mlp:
            logging.info("MLPConfig: Only input MLP enabled")
        else:
            logging.info("MLPConfig: Only output MLP enabled")

@dataclass
class LoRAConfig:
    """LoRA-specific configuration"""
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    
    # Training parameters
    epochs: int = 1
    final_epochs: int = 1
    initial_epochs: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"  # ✅ ADD THIS

                 
    
    warmup_per_epoch: bool = True        # Restart warmup each epoch
    warmup_steps_per_epoch: int = 300     # Steps per epoch if warmup_per_epoch=True
    warmup_ratio: float = 0               # Percentage of total training
    warmup_steps: int = 100               # Absolute number of steps

@dataclass
class SymbolConfig:
    """Symbol-specific configuration"""
    mode: SymbolMode = SymbolMode.FIXED
    symbol_type: str = "two_token"
    
    # Dynamic symbol parameters
    regenerate_frequency: int = 1  # How often to regenerate (epochs or cycles)
    seed: Optional[int] = None     # For reproducible symbol generation

@dataclass
class DataConfig:
    """Data-specific configuration"""
    dataset_type: str = 'voceleb'  # Default dataset type
    batch_size: int = 1
    max_samples: int = 10
    split: str = "test"
    
    # Validation parameters
    val_batch_size: Optional[int] = 1
    val_max_samples: int = 200  # Default validation samples
    val_frequency: int = 1  # Validate every N epochs
    val_dataset_type: str = "voxceleb-hvb-meld_emotion-voxpopuli"  # Default validation dataset type


@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Core configuration
    mode: TrainingMode = TrainingMode.LORA_FIRST
    model_type: ModelType = ModelType.SALMONN
    
    # Sub-configurations
    mlp_config: MLPConfig = field(default_factory=MLPConfig)
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    symbol_config: SymbolConfig = field(default_factory=SymbolConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Training schedule
    total_cycles: int = 2
    
    # I/O parameters
    output_dir: str = "/home/leapers/weights/neeraja/ICL/orchestrator_training"

    # output_dir: str = "/data1/chandnia/neeraja/results/model_ICL"  # Default output directory
    run_name: str = "symbol_training_run"
    checkpoint_frequency: int = 1  # Save checkpoint every N epochs
    
    # Device and performance
    device: str = "cuda:0"
    mixed_precision: bool = False
    compile_model: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_frequency: int = 1  # Log every N steps
    
    # NEW: Inference configuration
    inference_mode: bool = False
    
    only_original: bool = False  # Only use original labels without symbols

    scheduler: str = "cosine"  # ✅ ADD GLOBAL SCHEDULER
    warmup_steps: float = 100  # ✅ ADD GLOBAL WARMUP

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._set_derived_values()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate training mode compatibility
        if self.mode == TrainingMode.BYPASS_MLP_SYM:
            if self.symbol_config.mode == SymbolMode.NO_SYMBOLS:
                raise ValueError("BYPASS_MLP_SYM mode requires symbol replacement")
        
        if self.mode == TrainingMode.BYPASS_MLP_ORG:
            if self.symbol_config.mode != SymbolMode.NO_SYMBOLS:
                logging.warning("BYPASS_MLP_ORG mode typically doesn't use symbols")
        
        # Validate device
        if not self.device.startswith(("cuda", "cpu")):
            raise ValueError(f"Invalid device: {self.device}")
        
        # Validate batch size
        if self.data_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
    
    def _set_derived_values(self):
        """Set derived configuration values"""
        # Set validation batch size if not specified
        if self.data_config.val_batch_size is None:
            self.data_config.val_batch_size = self.data_config.batch_size
        
        # Set symbol mode based on training mode
        if self.mode == TrainingMode.BYPASS_MLP_SYM and self.symbol_config.mode == SymbolMode.FIXED:
            logging.info("Setting symbol mode to DYNAMIC_PER_EPOCH for BYPASS_MLP_SYM training")
            self.symbol_config.mode = SymbolMode.DYNAMIC_PER_EPOCH
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get training schedule information"""
        if self.mode == TrainingMode.LORA_FIRST:
            total_steps = 1 + (self.total_cycles * 2) + 1  # initial + cycles + final
        elif self.mode == TrainingMode.MLP_FIRST:
            total_steps = 1 + (self.total_cycles * 2) + 1  # initial_mlp + cycles + final
        elif self.mode == TrainingMode.JOINT_TRAINING:
            total_steps = self.total_cycles
        elif self.mode == TrainingMode.BYPASS_MLP_SYM or self.mode == TrainingMode.BYPASS_MLP_ORG:
            # Bypass MLP training does not have MLP cycles
            # Total steps is just the number of cycles
            total_steps = self.total_cycles 
        else:
            total_steps = self.total_cycles
        
        return {
            "mode": self.mode.value,
            "total_cycles": self.total_cycles,
            "total_steps": total_steps,
            "mlp_epochs_per_cycle": self.mlp_config.epochs,
            "lora_epochs_per_cycle": self.lora_config.epochs,
            "lora_final_epochs": self.lora_config.final_epochs
        }
    
    def get_model_suffix(self) -> str:
        """Generate descriptive suffix for model naming"""
        # MLP configuration
        if self.mode == TrainingMode.BYPASS_MLP_SYM:
            mlp_suffix = "bypass_mlp_sym"
        elif self.mlp_config.use_input_mlp and self.mlp_config.use_output_mlp:
            mlp_suffix = "io_mlp"
        elif self.mlp_config.use_input_mlp:
            mlp_suffix = "i_mlp"
        elif self.mlp_config.use_output_mlp:
            mlp_suffix = "o_mlp"
        else:
            mlp_suffix = "no_mlp_org"
        
        # Training mode
        mode_suffix = self.mode.value
        
        # Symbol mode
        symbol_suffix = self.symbol_config.mode.value
        
        return f"{mode_suffix}_{mlp_suffix}_{symbol_suffix}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mode": self.mode.value,
            "model_type": self.model_type.value,
            "mlp_config": {
                "use_input_mlp": self.mlp_config.use_input_mlp,
                "use_output_mlp": self.mlp_config.use_output_mlp,
                "hidden_dim": self.mlp_config.hidden_dim,
                "learning_rate": self.mlp_config.learning_rate,
                "epochs": self.mlp_config.epochs,
                "initial_epochs": self.mlp_config.initial_epochs
            },
            "lora_config": {
                "rank": self.lora_config.rank,
                "alpha": self.lora_config.alpha,
                "learning_rate": self.lora_config.learning_rate,
                "epochs": self.lora_config.epochs,
                "initial_epochs": self.lora_config.initial_epochs,
                "final_epochs": self.lora_config.final_epochs,
            },
            "symbol_config": {
                "mode": self.symbol_config.mode.value,
                "symbol_type": self.symbol_config.symbol_type,
            },
            "data_config": {
                "dataset_type": self.data_config.dataset_type,
                "batch_size": self.data_config.batch_size,
                "max_samples": self.data_config.max_samples,
            },
            "total_cycles": self.total_cycles,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "device": self.device,
        }
    
    def get_training_output_dir(self) -> str:
        """Get training output directory with run_name structure"""
        return os.path.join(self.output_dir, "checkpoints", self.run_name)
    
    
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        """Create configuration from command line arguments"""
        # Create sub-configurations
        mlp_config = MLPConfig(
            use_input_mlp=not args.bypass_mlp,
            use_output_mlp=False if args.bypass_mlp else getattr(args, 'use_output_mlp', False),
            hidden_dim=args.hidden_dim,
            learning_rate=args.mlp_lr,
            epochs=args.mlp_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        lora_config = LoRAConfig(
            learning_rate=args.lora_lr,
            epochs=args.lora_epochs,
            final_epochs=args.lora_final_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
        
        # Determine symbol mode
        # if args.swap or args.regular:
        #     self.only_original = True

        
        if getattr(args, 'dynamic_symbols_per_epoch', False):
            symbol_mode = SymbolMode.DYNAMIC_PER_EPOCH
        else:
            symbol_mode = SymbolMode.FIXED
        
        symbol_config = SymbolConfig(
            mode=symbol_mode,
            symbol_type="two_token",
        )
        
        data_config = DataConfig(
            dataset_type=args.dataset_type,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            val_max_samples = 200 if args.max_samples ==0 else min(200, args.max_samples),
            split=getattr(args, 'split', 'test'),
        )
        
        # Determine training mode
        if args.bypass_mlp:
            if args.dynamic_symbols_per_epoch:
                training_mode = TrainingMode.BYPASS_MLP_SYM
            else:
                training_mode = TrainingMode.BYPASS_MLP_ORG
        elif getattr(args, 'schedule_type','joint_training' )== 'joint_training':
            training_mode = TrainingMode.JOINT_TRAINING
        elif getattr(args, 'schedule_type','joint_training') == 'mlp_first':
            training_mode = TrainingMode.MLP_FIRST
        elif getattr(args, 'schedule_type','joint_training') == 'lora_mlp_joint':  # ✅ ADD THIS
            training_mode = TrainingMode.LORA_MLP_JOINT  # ✅ ADD THIS
        else:
            training_mode = TrainingMode.LORA_FIRST
        
        return cls(
            mode=training_mode,
            model_type=ModelType(args.model_type),
            mlp_config=mlp_config,
            lora_config=lora_config,
            symbol_config=symbol_config,
            data_config=data_config,
            total_cycles=args.total_cycles,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device
        )


def create_training_config(**kwargs) -> TrainingConfig:
    """
    Convenience function to create training configuration
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(**kwargs)


def get_default_config(schedule_type: str = "lora_first") -> TrainingConfig:
    """
    Get default configuration for a specific training mode
    
    Args:
        mode: Training mode ("lora_first", "mlp_first", "joint_training", "bypass_mlp")
        
    Returns:
        Default TrainingConfig for the specified mode
    """
    base_config = TrainingConfig()
    
    if schedule_type == "lora_first":
        base_config.mode = TrainingMode.LORA_FIRST
        base_config.symbol_config.mode = SymbolMode.FIXED
        
    elif schedule_type == "mlp_first":
        base_config.mode = TrainingMode.MLP_FIRST
        base_config.symbol_config.mode = SymbolMode.FIXED
        
    elif schedule_type == "joint_training":
        base_config.mode = TrainingMode.JOINT_TRAINING
        base_config.symbol_config.mode = SymbolMode.DYNAMIC_PER_EPOCH
        base_config.mlp_config.use_input_mlp = True
        base_config.mlp_config.use_output_mlp = False
        base_config.total_cycles = 1  # More cycles for joint training
        
    elif schedule_type == "bypass_mlp_sym":
        base_config.mode = TrainingMode.BYPASS_MLP_SYM
        # Bypass MLP training with dynamic symbols
        base_config.symbol_config.mode = SymbolMode.DYNAMIC_PER_EPOCH
        base_config.mlp_config.use_input_mlp = False
        base_config.mlp_config.use_output_mlp = False

    elif schedule_type == "bypass_mlp_org":
        base_config.mode = TrainingMode.BYPASS_MLP_ORG
        # Pure LoRA training without symbols
        base_config.symbol_config.mode = SymbolMode.NO_SYMBOLS
        base_config.mlp_config.use_input_mlp = False
        base_config.mlp_config.use_output_mlp = False
        
    else:
        raise ValueError(f"Unknown training mode: {schedule_type}")
    
    return base_config


# Backwards compatibility function
def parse_training_args() -> argparse.Namespace:
    """
    Parse training arguments with backwards compatibility
    This maintains compatibility with existing shell scripts
    """
    parser = argparse.ArgumentParser(description="Symbol Adapter Training")
    
    # Core arguments
    parser.add_argument("--model_type", type=str, default="salmonn", choices=["salmonn", "llama", "qwen"])
    parser.add_argument("--dataset_type", type=str, default="voxceleb")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=100)
    
    # MLP arguments
    parser.add_argument("--use_output_mlp", action="store_true", help="Use output MLP")
    parser.add_argument("--bypass_mlp", action="store_true", help="Bypass MLP training")
    
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--mlp_lr", type=float, default=1e-5)
    parser.add_argument("--mlp_epochs", type=int, default=1)
    parser.add_argument("--mlp_initial_epochs", type=int, default=1)
    
    # LoRA arguments
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--lora_epochs", type=int, default=5)
    parser.add_argument("--lora_initial_epochs", type=int, default=1)
    parser.add_argument("--lora_final_epochs", type=int, default=1)
    
    # Training arguments
    parser.add_argument("--total_cycles", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # New arguments
    parser.add_argument("--dynamic_symbols_per_epoch", action="store_true", 
                       help="Generate new symbols each epoch")
    parser.add_argument("--schedule_type", type=str, default="lora_first",
                       choices=["lora_first", "mlp_first", "joint_training","bypass_mlp_sym", "bypass_mlp_org","lora_mlp_joint",""])
    
    # I/O arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    

    
    return parser.parse_args()