"""
Symbol Training Orchestrator
Main training pipeline that coordinates MLP, LoRA, and Joint trainers
with symbol management and comprehensive validation
"""

import logging
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from datetime import datetime

from ..symbol_manager import SymbolManager
from ..configs.training_configs import TrainingConfig, SymbolMode
from .validation import ValidationManager
from .schedulers import TrainingScheduler, TrainingStep
from .unified_trainer import UnifiedTrainer


class SymbolTrainingOrchestrator:
    """
    Main orchestrator for Symbol Adapter training pipeline
    Manages the complete training process with multiple cycles and phases
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer=None,
        symbol_manager=None  # NEW: Accept existing symbol_manager
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        # Use provided symbol_manager or create new one
        self.symbol_manager = symbol_manager
        logging.info("Using provided SymbolManager")

        
        # Rest of initialization...
        self.scheduler = TrainingScheduler(config)
        self.trainer = UnifiedTrainer(
            config=config,
            symbol_manager=symbol_manager,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            tokenizer=self.tokenizer
        )
        
        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            max_val_samples=config.data_config.val_max_samples
        )
        
        # Training state
        self.current_cycle = 0
        self.current_step = 0
        self.training_history = []
        # self.best_scores = {}
        
        # Add summary tracking
        self.training_summary = []
        
        # Setup logging and output directories
        self._setup_training_environment()
    
    
    def _setup_training_environment(self):
        """Setup logging and output directories"""
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)

        
        # Save configuration
        train_dir = self.config.get_training_output_dir()
        os.makedirs(train_dir, exist_ok=True)

        config_path = os.path.join(train_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logging.info(f"Training environment setup complete")
        logging.info(f"Output directory: {self.config.output_dir}")
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logging.info("🚀 Starting Complete Symbol Training Pipeline")
        
        # ✅ GIVE TRAINER ACCESS TO ORCHESTRATOR FOR EPOCH TRACKING
        self.trainer.orchestrator = self
        
        # Get training schedule
        training_steps = self.scheduler.generate_schedule()
        
        current_cycle = -1
        
        for step_idx, step in enumerate(training_steps):
            # Log cycle summary when cycle changes
            if step.cycle != current_cycle and current_cycle >= 0:
                self._log_cycle_summary(current_cycle)
            current_cycle = step.cycle
            
            # Execute training step with unified trainer
            validation_scores = self.trainer.train_step(step)
        
        # Log final summaries
        self._log_cycle_summary(current_cycle)
        self._log_final_summary()
        
        return None
    
    def _execute_training_step(self, step: TrainingStep) -> Dict[str, Any]:
        """Execute a single training step"""
        
        # Update symbol manager for this step
        if step.cycle != self.current_cycle:
            self.current_cycle = step.cycle
            logging.info(f"Starting new cycle {step.cycle}")
        
        # Route to appropriate trainer based on phase
        if step.phase == "mlp":
            logging.info("🔧 Executing MLP Training Step")
            return self.mlp_trainer.train_mlp_step(step)
        
        elif step.phase == "lora":
            logging.info("🔗 Executing LoRA Training Step")
            return self.lora_trainer.train_lora_step(step)
        
        elif step.phase == "bypass_mlp_sym":
            logging.info("🔄 Executing Dynamic LoRA Training Step (Bypass MLP)")
            return self.lora_trainer.train_lora_step(step)
        
        elif step.phase == "bypass_mlp_org":
            logging.info("🚫 Executing LoRA Training Step (No Symbols)")
            step.use_symbols = False
            return self.lora_trainer.train_lora_step(step)
        
        elif step.phase == "joint":
            logging.info("🤝 Executing Joint MLP+LoRA Training Step")
            return self.joint_trainer.train_joint_step(step)
        
        else:
            raise ValueError(f"Unknown training phase: {step.phase}")
    
    def _log_training_configuration(self):
        """Log the complete training configuration"""
        logging.info("TRAINING CONFIGURATION:")
        logging.info(f"  Dataset: {self.config.data_config.dataset_type}")
        logging.info(f"  Device: {self.config.device}")
        logging.info(f"  Cycles: {self.config.total_cycles}")
        logging.info(f"  Symbol Mode: {self.config.symbol_config.mode}")
        logging.info(f"  Output Dir: {self.config.output_dir}")
        
        # Model configuration
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"  Model Parameters: {total_params:,}")
        
        # Data configuration
        logging.info(f"  Train Batches: {len(self.train_dataloader)}")
        logging.info(f"  Val Batches: {len(self.val_dataloader)}")
        
        logging.info("=" * 80)
    
    
    def _log_final_summary(self):
        """Log complete training summary - Enhanced with MLP indicator and compact format"""
        logging.info("=" * 180)
        logging.info("COMPLETE TRAINING SUMMARY - ALL EPOCHS")
        logging.info("=" * 180)
        
        if not self.training_summary:
            logging.info("No training data to summarize")
            logging.info("=" * 180)
            return
        
        # ✅ Get dataset context for better display (same as cycle summary)
        dataset_type_str = getattr(self.config.data_config, 'dataset_type', '')
        dataset_names_train = set(dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str])
        
        val_dataset_type_str = getattr(self.config.data_config, 'val_dataset_type', dataset_type_str)
        dataset_names_val = set(val_dataset_type_str.split('-') if '-' in val_dataset_type_str else [val_dataset_type_str])
        val_only_datasets = dataset_names_val - dataset_names_train
        
        # ✅ Determine all mode keys
        all_keys = set()
        for entry in self.training_summary:
            for key in entry.keys():
                if key not in ['phase', 'cycle', 'step', 'epoch', 'epochs_total']:
                    all_keys.add(key)
        
        # Filter out loss and composite keys for main display
        mode_keys = [k for k in sorted(all_keys) if not k.endswith('_loss') and not k.endswith('_composite')]
        
        if mode_keys:
            # ✅ Get all unique datasets from all validation results
            all_datasets = set()
            for entry in self.training_summary:
                for mode in mode_keys:
                    value = entry.get(mode, "")
                    if isinstance(value, str) and "|" in value:
                        for pair in value.split("|"):
                            if ":" in pair:
                                dataset, _ = pair.split(":", 1)
                                all_datasets.add(dataset)
            
            if all_datasets:
                # ✅ Create dataset info with context (consistent with cycle summary)
                dataset_info = {}
                for dataset in sorted(all_datasets):
                    is_trained = dataset in dataset_names_train
                    if dataset == 'voxceleb':
                        abbrev = 'VOX'
                    elif dataset == 'hvb':
                        abbrev = 'HVB'
                    elif dataset == 'meld_emotion':
                        abbrev = 'MEL'
                    elif dataset == 'voxpopuli':
                        abbrev = 'VXP'
                    else:
                        abbrev = dataset[:3].upper()
                    
                    dataset_info[dataset] = {
                        'abbrev': abbrev,
                        'is_trained': is_trained,
                        'context': 'TRN' if is_trained else 'VAL'
                    }
                
                # ✅ ENHANCED HEADER: Compact Cycle-Epoch + MLP Column + Mode
                header_parts = ["Phase", "C-E", "Stp", "MLP"]  # ✅ Combined cycle-epoch, added MLP column
                for dataset in sorted(all_datasets):
                    info = dataset_info[dataset]
                    header_parts.append(f"{info['abbrev']}({info['context']})")
                header_parts.append("Mode")
                
                header = "  ".join(f"{h:<8}" for h in header_parts)  # ✅ Reduced spacing
                logging.info(header)
                logging.info("-" * len(header))
                
                # ✅ ENHANCED DATA ROWS with MLP indicator
                for entry in self.training_summary:
                    phase = entry['phase'][:5]
                    cycle_epoch = f"{entry['cycle']}-{entry['epoch']}"  # ✅ Combined format
                    step_num = str(entry['step'])
                    
                    # Show each validation mode as a separate row
                    for mode in mode_keys:
                        value = entry.get(mode, "N/A")
                        
                        # ✅ PARSE MLP STATUS from mode name
                        if 'no_mlp_' in mode.lower():
                            mlp_status = "0"  # No MLP
                            mode_clean = mode.replace('no_mlp_', '').replace('_', '+').title()
                        elif 'mlp_' in mode.lower():
                            mlp_status = "1"  # MLP enabled
                            mode_clean = mode.replace('mlp_', '').replace('_', '+').title()
                        else:
                            mlp_status = "?"  # Unknown
                            mode_clean = mode.replace('_', '+').title()
                        
                        # Create row for this mode
                        row_parts = [phase, cycle_epoch, step_num, mlp_status]
                        
                        # Add dataset scores
                        if isinstance(value, str) and "|" in value:
                            datasets = {}
                            for pair in value.split("|"):
                                if ":" in pair:
                                    dataset, score = pair.split(":", 1)
                                    try:
                                        datasets[dataset] = float(score)
                                    except ValueError:
                                        datasets[dataset] = 0.0
                            
                            for dataset in sorted(all_datasets):
                                if dataset in datasets:
                                    score = datasets[dataset]
                                    row_parts.append(f"{score:.3f}")
                                else:
                                    # ✅ Context-aware missing values (same logic as cycle summary)
                                    if dataset in val_only_datasets and 'symbol' in mode.lower() and 'fresh' not in mode.lower():
                                        row_parts.append("SKIP")  # Expected skip for val-only in symbol mode
                                    else:
                                        row_parts.append("N/A")   # Unexpected missing
                        else:
                            # Special case values
                            for dataset in sorted(all_datasets):
                                if value == "val_only_skip:0.000000":
                                    row_parts.append("SKIP")
                                else:
                                    row_parts.append("N/A")
                        
                        # Add clean mode name
                        row_parts.append(mode_clean[:6])
                        
                        row = "  ".join(f"{r:<8}" for r in row_parts)
                        logging.info(row)
                    
                    # Add separator between different entries for readability
                    logging.info("-" * len(header))
            
            else:
                # ✅ Fallback for no dataset results with MLP column
                header_parts = ["Phase", "C-E", "Step", "MLP"] + [mode.replace('no_mlp_', '').replace('mlp_', '').title()[:8] for mode in mode_keys]
                header = "  ".join(f"{h:<12}" for h in header_parts)
                logging.info(header)
                logging.info("-" * len(header))
                
                for entry in self.training_summary:
                    cycle_epoch = f"{entry['cycle']}-{entry['epoch']}"
                    
                    for mode in mode_keys:
                        value = entry.get(mode, "N/A")
                        
                        # ✅ Parse MLP status
                        if 'no_mlp_' in mode.lower():
                            mlp_status = "0"
                            mode_clean = mode.replace('no_mlp_', '').replace('_', '+').title()
                        elif 'mlp_' in mode.lower():
                            mlp_status = "1"
                            mode_clean = mode.replace('mlp_', '').replace('_', '+').title()
                        else:
                            mlp_status = "?"
                            mode_clean = mode.replace('_', '+').title()
                        
                        row_parts = [
                            entry['phase'][:5], 
                            cycle_epoch,
                            str(entry['step']),
                            mlp_status
                        ]
                        
                        if value == "val_only_skip:0.000000":
                            row_parts.append("SKIPPED")
                        elif isinstance(value, float):
                            row_parts.append(f"{value:.4f}")
                        else:
                            display_value = str(value)[:10] if len(str(value)) > 10 else str(value)
                            row_parts.append(display_value)
                        
                        row = "  ".join(f"{r:<12}" for r in row_parts)
                        logging.info(row)
        
        else:
            logging.info("No validation modes found in training summary")
        
        logging.info("=" * 180)
        logging.info("✅ Training completed successfully!")
 
    def _log_cycle_summary(self, cycle: int):
        """Enhanced cycle summary with MLP indicator and compact format"""
        cycle_entries = [e for e in self.training_summary if e['cycle'] == cycle]
        
        if not cycle_entries:
            return
            
        logging.info("=" * 160)
        logging.info(f"CYCLE {cycle} SUMMARY")
        logging.info("=" * 160)
        
        # Get dataset context
        dataset_type_str = getattr(self.config.data_config, 'dataset_type', '')
        dataset_names_train = set(dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str])
        
        val_dataset_type_str = getattr(self.config.data_config, 'val_dataset_type', dataset_type_str)
        dataset_names_val = set(val_dataset_type_str.split('-') if '-' in val_dataset_type_str else [val_dataset_type_str])
        val_only_datasets = dataset_names_val - dataset_names_train
        
        # Get all mode keys and parse them
        all_mode_keys = set()
        for entry in cycle_entries:
            for key in entry.keys():
                if key not in ['phase', 'cycle', 'step', 'epoch', 'epochs_total']:
                    all_mode_keys.add(key)
        
        mode_keys = [k for k in sorted(all_mode_keys) if not k.endswith('_loss') and not k.endswith('_composite')]
        
        if mode_keys:
            # Get all unique datasets
            all_datasets = set()
            for entry in cycle_entries:
                for mode in mode_keys:
                    value = entry.get(mode, "")
                    if isinstance(value, str) and "|" in value:
                        for pair in value.split("|"):
                            if ":" in pair:
                                dataset, _ = pair.split(":", 1)
                                all_datasets.add(dataset)
            
            if all_datasets:
                # Create dataset info with context
                dataset_info = {}
                for dataset in sorted(all_datasets):
                    is_trained = dataset in dataset_names_train
                    if dataset == 'voxceleb':
                        abbrev = 'VOX'
                    elif dataset == 'hvb':
                        abbrev = 'HVB'
                    elif dataset == 'meld_emotion':
                        abbrev = 'MEL'
                    elif dataset == 'voxpopuli':
                        abbrev = 'VXP'
                    else:
                        abbrev = dataset[:3].upper()
                    
                    dataset_info[dataset] = {
                        'abbrev': abbrev,
                        'is_trained': is_trained,
                        'context': 'TRN' if is_trained else 'VAL'
                    }
                
                # ✅ ENHANCED HEADER: Compact Cycle-Epoch + MLP Column + Mode
                header_parts = ["Phase", "C-E", "MLP"]  # ✅ Combined cycle-epoch, added MLP column
                for dataset in sorted(all_datasets):
                    info = dataset_info[dataset]
                    header_parts.append(f"{info['abbrev']}({info['context']})")
                header_parts.append("Mode")
                
                header = "  ".join(f"{h:<8}" for h in header_parts)  # ✅ Reduced spacing
                logging.info(header)
                logging.info("-" * len(header))
                
                # ✅ ENHANCED DATA ROWS with MLP indicator
                for entry in cycle_entries:
                    phase = entry['phase'][:5]
                    cycle_epoch = f"{entry['cycle']}-{entry['epoch']}"  # ✅ Combined format
                    
                    for mode in mode_keys:
                        value = entry.get(mode, "N/A")
                        
                        # ✅ PARSE MLP STATUS from mode name
                        if 'no_mlp_' in mode.lower():
                            mlp_status = "0"  # No MLP
                            mode_clean = mode.replace('no_mlp_', '').replace('_', '+').title()
                        elif 'mlp_' in mode.lower():
                            mlp_status = "1"  # MLP enabled
                            mode_clean = mode.replace('mlp_', '').replace('_', '+').title()
                        else:
                            mlp_status = "?"  # Unknown
                            mode_clean = mode.replace('_', '+').title()
                        
                        # Create row
                        row_parts = [phase, cycle_epoch, mlp_status]
                        
                        # Add dataset scores
                        if isinstance(value, str) and "|" in value:
                            datasets = {}
                            for pair in value.split("|"):
                                if ":" in pair:
                                    dataset, score = pair.split(":", 1)
                                    try:
                                        datasets[dataset] = float(score)
                                    except ValueError:
                                        datasets[dataset] = 0.0
                            
                            for dataset in sorted(all_datasets):
                                if dataset in datasets:
                                    score = datasets[dataset]
                                    row_parts.append(f"{score:.3f}")
                                else:
                                    # Context-aware missing values
                                    if dataset in val_only_datasets and 'symbol' in mode.lower() and 'fresh' not in mode.lower():
                                        row_parts.append("SKIP")
                                    else:
                                        row_parts.append("N/A")
                        else:
                            # Special case values
                            for dataset in sorted(all_datasets):
                                if value == "val_only_skip:0.000000":
                                    row_parts.append("SKIP")
                                else:
                                    row_parts.append("N/A")
                        
                        # Add clean mode name
                        row_parts.append(mode_clean[:6])
                        
                        row = "  ".join(f"{r:<8}" for r in row_parts)
                        logging.info(row)
                    
                    # Add separator between entries
                    if len(cycle_entries) > 1:
                        logging.info("-" * len(header))
        
        logging.info("=" * 160)

    def _track_epoch_summary(self, step: TrainingStep, epoch: int, validation_scores: dict):
        """Track summary data for each EPOCH - Using composite strings"""
        summary_entry = {
            'phase': step.phase.upper(),
            'cycle': step.cycle,
            'step': step.step_id,
            'epoch': epoch,
            'epochs_total': step.epochs,
        }
        
        # ✅ SIMPLE: Store composite strings directly
        for key, value in validation_scores.items():
            # Store as-is - could be composite string or single value  
            summary_entry[key] = str(value) if value is not None else "N/A"
        
        self.training_summary.append(summary_entry)
