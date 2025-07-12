import logging
import os
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from ..configs.training_configs import TrainingConfig
from ..symbol_manager import SymbolManager
from .validation import ValidationManager
from .schedulers import TrainingStep

from transformers import get_scheduler


class UnifiedTrainer:
    """
    Unified trainer that can handle all training phases: LoRA, MLP, Joint
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        symbol_manager: SymbolManager,
        tokenizer=None
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        
        # Training state
        self.current_epoch = 0
        self.optimizer = None
        self.scheduler = None
        
        # Initialize validator
        self.validator = ValidationManager(
            config=config,
            symbol_manager=symbol_manager,
            tokenizer=tokenizer,
            max_val_samples=getattr(config.data_config, 'val_max_samples', 200)
        )
        
        logging.info("UnifiedTrainer initialized")
    
    def train_step(self, step: TrainingStep) -> dict:
        """Universal training method for any phase"""
        logging.info(f"🚀 Starting {step.phase.upper()} Training: {step.description}")
        
        # Setup parameters and optimizer based on phase
        self._setup_training_phase(step)
        
        validation_scores = {}
        # Train for specified epochs
        for epoch in range(step.epochs):
            self.current_epoch = epoch
            logging.info(f"--- {step.phase.upper()} Epoch {epoch+1}/{step.epochs} ---")
            
            # Training epoch
            epoch_loss = self._train_epoch(step, epoch)
            # step_metrics["train_loss"].append(epoch_loss)
            
            logging.info(f"{step.phase.upper()} Epoch {epoch+1} Loss: {epoch_loss:.6f}")
            
            # Validation after each epoch
            if getattr(self.config, 'validate_every_epoch', True):
                epoch_validation = self._validate_epoch(step, epoch)
                validation_scores.update(epoch_validation)
                # step_metrics["validation_scores"][epoch] = epoch_validation
                
                # ✅ TRACK EACH EPOCH INDIVIDUALLY
                if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, '_track_epoch_summary'):
                    self.orchestrator._track_epoch_summary(step, epoch, epoch_validation)
                
                # Log epoch summary
                self._log_epoch_summary(step, epoch + 1, epoch_loss, epoch_validation)
                
            # Save periodic checkpoint
            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(step, epoch, "periodic")
        

        return validation_scores
    
    def _setup_training_phase(self, step: TrainingStep):
        """Setup model parameters and optimizer based on training phase"""
        
        if step.phase == "lora":
            self._setup_lora_training(step)
        else:
            raise ValueError(f"Unknown training phase: {step.phase}")
    
    def _setup_lora_training(self, step: TrainingStep):
        """Setup LoRA parameters for training"""
        logging.info(f"Setting up LoRA training for {step.description}")

        # Log parameter status
        self._log_parameter_status("LoRA")
        
        # Setup optimizer for LoRA parameters only
        self._setup_lora_optimizer(step)
        
        # Set model to training mode
        self.model.train()
        
        logging.info("✓ LoRA training setup complete")
    
    
    def _log_parameter_status(self, training_mode: str):
        """Log current parameter training status with breakdown"""
        # Count different parameter types
        mlp_params = sum(p.numel() for name, p in self.model.named_parameters() 
                        if p.requires_grad and ('mlp' in name.lower() or 'symbol' in name.lower()) and 'lora' not in name.lower())
        
        lora_params = sum(p.numel() for name, p in self.model.named_parameters() 
                         if p.requires_grad and 'lora' in name.lower())
        
        speech_params = 0
        if hasattr(self.model, 'salmonn'):
            speech_params = sum(p.numel() for name, p in self.model.named_parameters() 
                              if p.requires_grad and ('speech_Qformer' in name or 'speech_query_tokens' in name or 'speech_llama_proj' in name))
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        
        logging.info("=" * 60)
        logging.info(f"{training_mode.upper()} TRAINING PARAMETER STATUS:")
        logging.info(f"  MLP Parameters:     {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
        logging.info(f"  LoRA Parameters:    {lora_params:,} ({lora_params/total_params*100:.1f}%)")
        if speech_params > 0:
            logging.info(f"  SALMONN Parameters: {speech_params:,} ({speech_params/total_params*100:.1f}%)")
        logging.info(f"  Total Trainable:    {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logging.info(f"  Frozen Parameters:  {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        logging.info(f"  Total Parameters:   {total_params:,}")
        logging.info("=" * 60)
    
    def _setup_lora_optimizer(self, step: TrainingStep):
        """Setup optimizer for Symbol LoRA parameters ONLY"""
        
        # ✅ COMMENTED OUT: Original LoRA + SALMONN parameter collection
        # Get trainable LoRA parameters
        lora_params = []
        salmonn_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif ('speech_Qformer' in name or 'speech_query_tokens' in name or 'speech_llama_proj' in name):
                    salmonn_params.append(param)
        
        if not lora_params and not salmonn_params:
            raise ValueError("No trainable LoRA or SALMONN parameters found!")
        
        # Create parameter groups with different learning rates if needed
        param_groups = []
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': step.learning_rate or self.config.lora_config.learning_rate,
                'name': 'lora'
            })
        
        if salmonn_params:
            param_groups.append({
                'params': salmonn_params,
                'lr': (step.learning_rate or self.config.lora_config.learning_rate),
                'name': 'salmonn'
            })
        
        
        logging.info(f"Created AdamW optimizer with {len(param_groups)} parameter groups:")
        for group in param_groups:
            logging.info(f"  {group['name']}: {len(group['params'])} layers, LR={group['lr']}")
 
        

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lora_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.lora_config.weight_decay,
        )
        
        # Create learning rate scheduler
        total_steps = len(self.train_dataloader) * step.epochs // step.gradient_accumulation_steps
 

        if self.config.lora_config.warmup_per_epoch:
        # Use per-epoch warmup
            warmup_steps = self.config.lora_config.warmup_steps_per_epoch
            logging.info(f"Using per-epoch warmup: {warmup_steps} steps per epoch")
        elif self.config.lora_config.warmup_ratio > 0:
            # Use ratio-based warmup
            warmup_steps = int(total_steps * self.config.lora_config.warmup_ratio)
            logging.info(f"Using ratio-based warmup: {warmup_steps} steps ({self.config.lora_config.warmup_ratio*100}% of {total_steps})")
        else:
            # Use absolute warmup steps
            warmup_steps = self.config.lora_config.warmup_steps
            logging.info(f"Using absolute warmup: {warmup_steps} steps")

        if self.config.lora_config.warmup_per_epoch:
            # Create custom scheduler that restarts warmup each epoch
            self.scheduler = self._create_per_epoch_scheduler(warmup_steps, total_steps)
        else:
            # Standard scheduler
            self.scheduler = get_scheduler(
                name=self.config.lora_config.scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
        logging.info(f"Scheduler: {self.config.lora_config.scheduler}, Warmup: {warmup_steps}, Total: {total_steps}")


    def _create_per_epoch_scheduler(self, warmup_steps_per_epoch: int, total_steps: int):
        """Create scheduler that restarts warmup at each epoch"""
    
        def lr_lambda(step):
            epoch_length = len(self.train_dataloader) // self.config.lora_config.gradient_accumulation_steps
            current_epoch = step // epoch_length
            step_in_epoch = step % epoch_length
            
            # Warmup phase within epoch
            if step_in_epoch < warmup_steps_per_epoch:
                return step_in_epoch / warmup_steps_per_epoch
            
            # Cosine decay phase within epoch
            progress = (step_in_epoch - warmup_steps_per_epoch) / (epoch_length - warmup_steps_per_epoch)
            import math
            return 0.5 * (1 + math.cos(math.pi * progress))
            # return 0.8 * 0.2 (1 + math.cos(math.pi * progress))

            # return 1.0
        
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(self.optimizer, lr_lambda)
 
    def _train_epoch(self, step: TrainingStep, epoch: int) -> float:
        """Train for one epoch with scheduler and gradient accumulation"""
        
        # ✅ NEW: Reset scheduler if using per-epoch warmup
        if self.config.lora_config.warmup_per_epoch and epoch > 0:
            logging.info(f"Restarting warmup for epoch {epoch}")
            # Scheduler automatically handles this with our custom lr_lambda
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get symbols for this epoch
        epoch_symbols = self.symbol_manager.get_symbols_for_epoch(epoch)
        
        # Create progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"{step.phase.upper()} Epoch {epoch+1}",
            leave=False
        )
        
        # ✅ KEEP: Gradient accumulation steps (NOT REMOVED)
        accumulation_steps = step.gradient_accumulation_steps or self.config.lora_config.gradient_accumulation_steps

        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # DETAILED BATCH LOGGING FOR FIRST BATCH

                    if batch_idx > 0 and batch_idx % 50 == 0:
                        logging.info(f"Clearing memory at step {batch_idx}")
                        gc.collect()
                        torch.cuda.empty_cache()

                    # force_new_symbols = (batch_idx % (100 * accumulation_steps) == 0)
                    force_new_symbols = False
                    random_mask = True
                    # Apply symbol replacement
                    if getattr(step, 'use_symbols', True):
                        updated_batch = self.symbol_manager.replace_symbols_in_batch(
                            batch, epoch=epoch, random_mask=random_mask, force_new_symbols=force_new_symbols
                        )
                    else:
                        updated_batch = batch

                    if batch_idx == 0 or force_new_symbols:
                        logging.info(f"=== {step.phase.upper()} EPOCH BATCH 1 CONTENT ===")
                        if "prompt" in batch:
                            logging.info("ORIGINAL PROMPTS:")
                            for i, prompt in enumerate(batch["prompt"][:2]):
                                logging.info(f"  [{i}] {prompt}")
                        if "completion" in batch:
                            logging.info("ORIGINAL COMPLETIONS:")
                            for i, completion in enumerate(batch["completion"][:2]):
                                logging.info(f"  [{i}] {completion}")
                        
                        if getattr(step, 'use_symbols', True):
                            current_symbols = self.symbol_manager.get_symbols_for_epoch(epoch)
                            logging.info(f"EPOCH {epoch} SYMBOL MAPPINGS: {current_symbols}")
                        else:
                            logging.info("NO SYMBOL REPLACEMENT")
                    
                    # Calculate if random_mask should be True
                    
                    
                    # DETAILED BATCH LOGGING AFTER REPLACEMENT
                    if batch_idx == 0 or force_new_symbols:
                        if getattr(step, 'use_symbols', True):
                            logging.info("UPDATED AFTER SYMBOL REPLACEMENT:")
                            if "prompt" in updated_batch:
                                logging.info("UPDATED PROMPTS:")
                                for i, prompt in enumerate(updated_batch["prompt"][:2]):
                                    logging.info(f"  [{i}] {prompt}")
                            if "completion" in updated_batch:
                                logging.info("UPDATED COMPLETIONS:")
                                for i, completion in enumerate(updated_batch["completion"][:2]):
                                    logging.info(f"  [{i}] {completion}")
                        else:
                            logging.info("NO CHANGES (symbols not used)")
                        logging.info(f"=== END {step.phase.upper()} EPOCH BATCH 1 CONTENT ===")
                    
                    # Move to device
                    updated_batch = self._move_batch_to_device(updated_batch)
                    
                    # Forward pass
                    outputs = self.model(updated_batch)
                    loss = outputs.get("loss")
                    
                    if loss is None:
                        logging.warning(f"No loss returned for batch {batch_idx}")
                        continue
                    
                    # ✅ KEEP: Scale loss for gradient accumulation (NOT REMOVED)
                    loss = loss / accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # ✅ KEEP: Gradient accumulation with scheduler (NOT REMOVED)
                    if (batch_idx + 1) % accumulation_steps == 0:
 
                        max_grad_norm =  self.config.lora_config.max_grad_norm

                        
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        # ✅ CRITICAL: Update optimizer AND scheduler
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        if hasattr(self, 'scheduler') and self.scheduler:
                            self.scheduler.step()  # ✅ ADD THIS LINE

                    
                    # Update metrics
                    total_loss += loss.item() * accumulation_steps
                    num_batches += 1
                    
                    # ✅ ADD: Show current learning rate in progress bar
                    current_lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * accumulation_steps:.6f}",
                        'avg_loss': f"{total_loss / num_batches:.6f}",
                        'lr': f"{current_lr:.2e}"  # ✅ Show LR changes
                    })
                    
                    # Log frequently during first epoch
                    if epoch == 0 and batch_idx % getattr(self.config, 'log_frequency', 10) == 0:
                        logging.info(f"Batch {batch_idx}: loss = {loss.item() * accumulation_steps:.6f}, lr = {current_lr:.2e}")
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            logging.info(f"{step.phase.upper()} training interrupted by user")
        
        finally:
            progress_bar.close()
        
        return total_loss / max(num_batches, 1)
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the correct device"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _validate_epoch(self, step: TrainingStep, epoch: int, is_final: bool = False) -> dict:
        """Universal validation - Simplified, no mapping"""
        logging.info(f"Validating {step.phase.upper()} (Epoch {epoch+1})")
        
        # ✅ Direct return - no mapping needed
        return self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=epoch,
            phase=step.phase,
            cycle=step.cycle,
            step=step
        )

    def _log_epoch_summary(self, step: TrainingStep, epoch: int, epoch_loss: float, validation_scores: dict):
        """Log summary after each epoch - Works with any keys"""
        logging.info("=" * 120)
        logging.info(f"EPOCH SUMMARY - {step.phase.upper()} Cycle {step.cycle} Epoch {epoch}")
        logging.info("=" * 120)
        logging.info(f"Phase: {step.phase.upper():<6} | Cycle: {step.cycle:<2} | Epoch: {epoch:<2} | Loss: {epoch_loss:.4f}")
        
        # ✅ FIX: Proper if-else structure for each validation mode
        for mode_name, composite_score in validation_scores.items():
            if isinstance(composite_score, str) and "|" in composite_score:
                # Parse composite string
                datasets = {}
                for pair in composite_score.split("|"):
                    if ":" in pair:
                        dataset, score = pair.split(":", 1)
                        datasets[dataset] = score
                
                dataset_parts = [f"{dataset}={score}" for dataset, score in datasets.items()]
                logging.info(f"  {mode_name:<18}: {' | '.join(dataset_parts)}")
            else:
                # ✅ FIX: This should be inside the loop, not attached to it
                logging.info(f"  {mode_name:<18}: {composite_score}")

        logging.info("=" * 120)
    
    
    def _save_checkpoint(self, step: TrainingStep, epoch: int, checkpoint_type: str):
        """Save model checkpoint with only trainable parameters"""
        checkpoint_dir = self.config.get_training_output_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_name = f"{step.phase}_step{step.step_id}_cycle{step.cycle}_epoch{epoch+1}_{checkpoint_type}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Save only trainable parameters
        trainable_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.clone()
        
        checkpoint_data = {
            'step_info': {
                'phase': step.phase,
                'step_id': step.step_id,
                'cycle': step.cycle,
                'epoch': epoch + 1,
                'description': step.description
            },
            'model_state': trainable_state,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'symbol_mappings': {
                'current_epoch_mappings': self.symbol_manager.get_symbols_for_epoch(epoch),
                'original_labels': self.symbol_manager.original_labels,
                'symbol_type': self.symbol_manager.symbol_type,
                'dynamic_per_epoch': self.symbol_manager.dynamic_per_epoch
            }
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore trainable parameters"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        # Load trainable parameters
        model_state = checkpoint['model_state']
        for name, param in self.model.named_parameters():
            if name in model_state:
                param.data.copy_(model_state[name])
                logging.info(f"Loaded parameter: {name}")
        
        # Load optimizer state if available
        if checkpoint.get('optimizer_state') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        step_info = checkpoint.get('step_info', {})
        logging.info(f"Loaded checkpoint from {step_info.get('phase', 'unknown')} "
                   f"step {step_info.get('step_id', 'unknown')} "
                   f"epoch {step_info.get('epoch', 'unknown')}")
        
        return checkpoint
