"""
Training Schedule Generators for Symbol Adapter
Creates different training schedules based on configuration
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..configs.training_configs import TrainingConfig, TrainingMode

@dataclass
class TrainingStep:
    """Represents a single training step in the schedule"""
    phase: str                    # "lora", "mlp", "joint", "lora_initial","lora_final", "mlp_initial"
    epochs: int                   # Number of epochs for this step
    cycle: int                    # Which cycle this step belongs to (0-indexed)
    step_id: int                  # Step number in overall schedule (0-indexed)
    description: str              # Human-readable description
    
    # Training parameters
    learning_rate: Optional[float] = None
    gradient_accumulation_steps: Optional[int] = None
    max_grad_norm: Optional[float] = None
    
    # Flags
    freeze_mlp: bool = True       # Whether to freeze MLP weights
    freeze_lora: bool = True      # Whether to freeze LoRA weights
    use_symbols: bool = True      # Whether to use symbol replacement
    dynamic_symbols: bool = False # Whether to generate new symbols this step
    bypass_mlp: bool = False      # ✅ ADD: Whether to bypass MLP layers completely
    
    def __post_init__(self):
        """Set training flags based on phase"""
        if self.phase in ["mlp"]:
            self.freeze_mlp = False
            self.freeze_lora = True
        elif self.phase in ["lora"]:
            self.freeze_mlp = True
            self.freeze_lora = False
        elif self.phase == "joint":
            self.freeze_mlp = False
            self.freeze_lora = False
            self.dynamic_symbols = True
        else:
            self.freeze_mlp = True
            self.freeze_lora = True
            self.use_symbols = False
            self.dynamic_symbols = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "phase": self.phase,
            "epochs": self.epochs,
            "cycle": self.cycle,
            "step_id": self.step_id,
            "description": self.description,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "freeze_mlp": self.freeze_mlp,
            "freeze_lora": self.freeze_lora,
            "use_symbols": self.use_symbols,
            "dynamic_symbols": self.dynamic_symbols,
            "bypass_mlp": self.bypass_mlp,  # ✅ ADD THIS LINE
        }


class TrainingScheduler:
    """Generates training schedules based on configuration"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.schedule: List[TrainingStep] = []
    
    def generate_schedule(self) -> List[TrainingStep]:
        """Generate training schedule based on configuration mode"""
        if self.config.mode == TrainingMode.LORA_FIRST:
            self.schedule = self._generate_lora_first_schedule()
        elif self.config.mode == TrainingMode.MLP_FIRST:
            self.schedule = self._generate_mlp_first_schedule()
        elif self.config.mode == TrainingMode.JOINT_TRAINING:
            self.schedule = self._generate_joint_training_schedule()
        elif self.config.mode == TrainingMode.BYPASS_MLP_SYM:
            self.schedule = self._generate_bypass_mlp_sym_schedule()
        elif self.config.mode == TrainingMode.BYPASS_MLP_ORG:
            self.schedule = self._generate_bypass_mlp_org_schedule()
        elif self.config.mode == TrainingMode.LORA_MLP_JOINT:  # ✅ ADD THIS
            self.schedule = self._generate_lora_mlp_joint_schedule()  # ✅ ADD THIS
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
        
        # Log the generated schedule
        self._log_schedule()
        
        return self.schedule
    
    def _generate_lora_first_schedule(self) -> List[TrainingStep]:
        """
        Generate LoRA-first schedule:
        Initial LoRA → [LoRA-MLP cycles] → Final LoRA
        """
        schedule = []
        step_id = 0
        
        # Phase 1: Initial LoRA training
        schedule.append(TrainingStep(
            phase="lora",
            epochs=self.config.lora_config.initial_epochs,
            cycle=0,
            step_id=step_id,
            description="Initial LoRA training - task learning",
            learning_rate=self.config.lora_config.learning_rate,
            gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
            max_grad_norm=self.config.lora_config.max_grad_norm,
        ))
        step_id += 1
        
        # Phase 2: Alternating LoRA-MLP cycles
        for cycle in range(self.config.total_cycles):
            # MLP phase
            schedule.append(TrainingStep(
                phase="mlp",
                epochs=self.config.mlp_config.epochs,
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} MLP training - learn symbols",
                learning_rate=self.config.mlp_config.learning_rate,
                gradient_accumulation_steps=self.config.mlp_config.gradient_accumulation_steps,
                max_grad_norm=self.config.mlp_config.max_grad_norm,
            ))
            step_id += 1
            
            # LoRA phase
            schedule.append(TrainingStep(
                phase="lora",
                epochs=self.config.lora_config.epochs,
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} LoRA training - task adaptation",
                learning_rate=self.config.lora_config.learning_rate,
                gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
                max_grad_norm=self.config.lora_config.max_grad_norm,
            ))
            step_id += 1
        
        # Phase 3: Final LoRA training
        schedule.append(TrainingStep(
            phase="lora",
            epochs=self.config.lora_config.final_epochs,
            cycle=self.config.total_cycles,
            step_id=step_id,
            description="Final LoRA training - task optimization",
            learning_rate=self.config.lora_config.learning_rate,
            gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
            max_grad_norm=self.config.lora_config.max_grad_norm,
        ))
        
        return schedule
    
    def _generate_mlp_first_schedule(self) -> List[TrainingStep]:
        """
        Generate MLP-first schedule:
        Initial MLP → [LoRA-MLP cycles] → Final LoRA
        """
        schedule = []
        step_id = 0
        
        # Phase 1: Initial MLP training
        schedule.append(TrainingStep(
            phase="mlp",
            epochs=self.config.mlp_config.initial_epochs,
            cycle=0,
            step_id=step_id,
            description="Initial MLP training - learn symbol representations",
            learning_rate=self.config.mlp_config.learning_rate,
            gradient_accumulation_steps=self.config.mlp_config.gradient_accumulation_steps,
            max_grad_norm=self.config.mlp_config.max_grad_norm,
        ))
        step_id += 1
        
        # Phase 2: Alternating LoRA-MLP cycles
        for cycle in range(self.config.total_cycles):
            # LoRA phase
            schedule.append(TrainingStep(
                phase="lora",
                epochs=self.config.lora_config.epochs,
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} LoRA training - task adaptation",
                learning_rate=self.config.lora_config.learning_rate,
                gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
                max_grad_norm=self.config.lora_config.max_grad_norm,
            ))
            step_id += 1
            
            # MLP phase
            schedule.append(TrainingStep(
                phase="mlp",
                epochs=self.config.mlp_config.epochs,
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} MLP training - refine symbols",
                learning_rate=self.config.mlp_config.learning_rate,
                gradient_accumulation_steps=self.config.mlp_config.gradient_accumulation_steps,
                max_grad_norm=self.config.mlp_config.max_grad_norm,
            ))
            step_id += 1
        
        # Phase 3: Final LoRA training
        schedule.append(TrainingStep(
            phase="lora",
            epochs=self.config.lora_config.final_epochs,
            cycle=self.config.total_cycles,
            step_id=step_id,
            description="Final LoRA training - task optimization",
            learning_rate=self.config.lora_config.learning_rate,
            gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
            max_grad_norm=self.config.lora_config.max_grad_norm,
        ))
        
        return schedule
    
    def _generate_joint_training_schedule(self) -> List[TrainingStep]:
        """
        Generate joint training schedule:
        Train MLP and LoRA simultaneously
        """
        schedule = []
        step_id = 0
        
        # Joint training cycles
        for cycle in range(self.config.total_cycles):
            schedule.append(TrainingStep(
                phase="joint",
                epochs=max(self.config.mlp_config.epochs, self.config.lora_config.epochs),
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} Joint MLP+LoRA training",
                learning_rate=None,  # Will use separate optimizers
                gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
                max_grad_norm=self.config.lora_config.max_grad_norm,
            ))
            step_id += 1
        
        return schedule

    def _generate_bypass_mlp_sym_schedule(self) -> List[TrainingStep]:
        """
        Generate bypass MLP schedule:
        Pure LoRA training with dynamic symbols
        """
        schedule = []
        step_id = 0
        
        # Dynamic LoRA training cycles
        for cycle in range(self.config.total_cycles):
            schedule.append(TrainingStep(
                phase="lora",
                epochs=self.config.lora_config.epochs,
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} LoRA training - dynamic symbols",
                learning_rate=self.config.lora_config.learning_rate,
                gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
                max_grad_norm=self.config.lora_config.max_grad_norm,
                freeze_mlp=True,  # MLPs don't exist anyway
                freeze_lora=False,
                use_symbols=True,
                dynamic_symbols=True,
                bypass_mlp=True,
            ))
            step_id += 1
        
        return schedule
    
    def _generate_bypass_mlp_org_schedule(self) -> List[TrainingStep]:
        """
        Generate bypass MLP schedule:
        Pure LoRA training with dynamic symbols
        """
        schedule = []
        step_id = 0
        
        # Dynamic LoRA training cycles
        for cycle in range(self.config.total_cycles):
            schedule.append(TrainingStep(
                phase="lora",
                epochs=self.config.lora_config.epochs,
                cycle=cycle,
                step_id=step_id,
                description=f"Cycle {cycle+1} LoRA training - dynamic symbols",
                learning_rate=self.config.lora_config.learning_rate,
                gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
                max_grad_norm=self.config.lora_config.max_grad_norm,
                freeze_mlp=True,  # MLPs don't exist anyway
                freeze_lora=False,
                use_symbols=False,
                dynamic_symbols=False,
                bypass_mlp=True,
            ))
            step_id += 1
        
        return schedule

    def _generate_lora_mlp_joint_schedule(self) -> List[TrainingStep]:
        """
        Generate LoRA-MLP-Joint schedule:
        LoRA only → MLP only → Joint training (no cycles)
        """
        schedule = []
        step_id = 0
        
        # Phase 1: Pure LoRA training (bypass MLP)
        schedule.append(TrainingStep(
            phase="lora",
            epochs=self.config.lora_config.epochs,  # e.g., 5-8 epochs
            cycle=0,  # ✅ Keep cycle for consistency
            step_id=step_id,
            description="Initial LoRA training - bypass MLP completely",
            learning_rate=self.config.lora_config.learning_rate,
            gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
            max_grad_norm=self.config.lora_config.max_grad_norm,
            bypass_mlp=True,  # ✅ No MLP interference
        ))
        step_id += 1
        
        # Phase 2: Pure MLP training (freeze LoRA)
        schedule.append(TrainingStep(
            phase="mlp", 
            epochs=self.config.mlp_config.epochs,  # e.g., 2-3 epochs
            cycle=0,
            step_id=step_id,
            description="MLP training - LoRA frozen, build on stable foundation",
            learning_rate=self.config.mlp_config.learning_rate,
            gradient_accumulation_steps=self.config.mlp_config.gradient_accumulation_steps,
            max_grad_norm=self.config.mlp_config.max_grad_norm,
            bypass_mlp=False,  # ✅ MLP layers active
        ))
        step_id += 1
        
        # Phase 3: Joint training (both active)
        schedule.append(TrainingStep(
            phase="joint",
            epochs=self.config.lora_config.final_epochs,  # e.g., 1-2 epochs
            cycle=0,
            step_id=step_id,
            description="Joint training - fine-tune both LoRA and MLP together",
            learning_rate=min(self.config.lora_config.learning_rate, self.config.mlp_config.learning_rate) / 2,
            gradient_accumulation_steps=self.config.lora_config.gradient_accumulation_steps,
            max_grad_norm=self.config.lora_config.max_grad_norm,
            bypass_mlp=False,  # ✅ Both active
        ))
        
        return schedule

    
    
    
    def _log_schedule(self):
        """Log the generated training schedule"""
        logging.info(f"Generated {self.config.mode.value.upper()} training schedule with {len(self.schedule)} steps:")
        logging.info("=" * 80)
        
        for step in self.schedule:
            freeze_info = []
            if step.freeze_mlp:
                freeze_info.append("MLP frozen")
            else:
                freeze_info.append("MLP active")
            
            if step.freeze_lora:
                freeze_info.append("LoRA frozen")
            else:
                freeze_info.append("LoRA active")
            
            symbol_info = ""
            if step.use_symbols:
                if step.dynamic_symbols:
                    symbol_info = " (dynamic symbols)"
                else:
                    symbol_info = " (fixed symbols)"
            else:
                symbol_info = " (no symbols)"
            
            logging.info(f"  Step {step.step_id+1}: {step.phase} ({step.epochs} epochs) - {step.description}")
            logging.info(f"    └─ {', '.join(freeze_info)}{symbol_info}")
        
        logging.info("=" * 80)
        
        # Log training flow summary
        logging.info("Training Flow Summary:")
        phase_counts = {}
        for step in self.schedule:
            phase_counts[step.phase] = phase_counts.get(step.phase, 0) + 1
        
        for phase, count in phase_counts.items():
            logging.info(f"  {phase}: {count} steps")
        
        total_epochs = sum(step.epochs for step in self.schedule)
        logging.info(f"  Total epochs: {total_epochs}")
        logging.info("=" * 80)
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get schedule summary statistics"""
        if not self.schedule:
            return {}
        
        phase_counts = {}
        phase_epochs = {}
        for step in self.schedule:
            phase_counts[step.phase] = phase_counts.get(step.phase, 0) + 1
            phase_epochs[step.phase] = phase_epochs.get(step.phase, 0) + step.epochs
        
        return {
            "mode": self.config.mode.value,
            "total_steps": len(self.schedule),
            "total_epochs": sum(step.epochs for step in self.schedule),
            "total_cycles": self.config.total_cycles,
            "phase_counts": phase_counts,
            "phase_epochs": phase_epochs,
            "steps": [step.to_dict() for step in self.schedule]
        }
    
    def save_schedule(self, filepath: str):
        """Save schedule to file"""
        import json
        
        schedule_data = self.get_schedule_summary()
        
        with open(filepath, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        logging.info(f"Saved training schedule to {filepath}")
    
    def load_schedule(self, filepath: str):
        """Load schedule from file"""
        import json
        
        with open(filepath, 'r') as f:
            schedule_data = json.load(f)
        
        # Reconstruct schedule from data
        self.schedule = []
        for step_data in schedule_data["steps"]:
            step = TrainingStep(
                phase=step_data["phase"],
                epochs=step_data["epochs"],
                cycle=step_data["cycle"],
                step_id=step_data["step_id"],
                description=step_data["description"],
                learning_rate=step_data.get("learning_rate"),
                gradient_accumulation_steps=step_data.get("gradient_accumulation_steps"),
                max_grad_norm=step_data.get("max_grad_norm"),
            )
            # Manually set flags since __post_init__ was already called
            step.freeze_mlp = step_data["freeze_mlp"]
            step.freeze_lora = step_data["freeze_lora"]
            step.use_symbols = step_data["use_symbols"]
            step.dynamic_symbols = step_data["dynamic_symbols"]
            step.bypass_mlp = step_data["bypass_mlp"]  # ✅ ADD THIS LINE
            
            self.schedule.append(step)
        
        logging.info(f"Loaded training schedule from {filepath}")

