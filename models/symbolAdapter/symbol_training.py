"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import os
import random
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.train_config.training_configs import TrainingConfig, SymbolUpdateStrategy
from config.data_config.master_config import DATASET_CONFIGS
from .symbol_manager import SymbolManager
from .validation import ValidationManager
from .symbol_router import SymbolRouter
from .vocab_filter import VocabFilter
from .dspo_module import DspoModule


class SymbolTrainingOrchestrator:
    """Single-path trainer with optional dynamic symbol replacement."""

    def __init__(
        self,
        config: TrainingConfig,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer=None,
        symbol_manager: SymbolManager = None,
        train_dataset_names: set = None,
        processor=None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.symbol_manager = symbol_manager
        self.train_dataset_names = train_dataset_names or set()
        self.processor = processor
        self.optimizer = None
        self.global_step = 0
        self._logged_sample = False

        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            max_val_samples=config.data_config.val_max_samples,
            processor=self.processor,
        )

        # D-SPO Components
        self.router = None
        self.dspo_module = None
        if self.config.diff_symbol_config.enabled:
            vocab_filter = VocabFilter(tokenizer)
            symbol_pool = vocab_filter.generate_symbol_pool(
                pool_size=self.config.diff_symbol_config.pool_size,
                exclude_labels=self.symbol_manager.original_labels if self.symbol_manager else None
            )
            self.router = SymbolRouter(
                num_slots=self.config.diff_symbol_config.num_slots,
                pool_size=len(symbol_pool),
                symbol_pool_indices=symbol_pool,
                initial_tau=self.config.diff_symbol_config.tau,
                tau_min=self.config.diff_symbol_config.tau_min,
            ).to(self.config.device)
            
            slot_tokens = [f"<slot_{i}>" for i in range(self.config.diff_symbol_config.num_slots)]
            slot_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in slot_tokens]
            self.dspo_module = DspoModule(slot_token_ids).to(self.config.device)
            
            logging.info("D-SPO initialized: %d slots, %d symbols in pool", self.config.diff_symbol_config.num_slots, len(symbol_pool))

        self._setup_training_environment()

    def _setup_training_environment(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.metrics_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)

    def _setup_lora_optimizer(self):
        trainable_params = []
        lora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                lora_params.append(param)
        if lora_params:
            trainable_params.append({"params": lora_params, "lr": self.config.lora_config.learning_rate})
        if self.router is not None:
            trainable_params.append({"params": self.router.parameters(), "lr": self.config.diff_symbol_config.router_lr})
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.lora_config.learning_rate)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else:
                device_batch[key] = value
        return device_batch

    def _apply_symbol_replacement(self, batch: Dict[str, Any], epoch: int, batch_idx: int) -> Dict[str, Any]:
        """Coordination logic: identify labels, bind slots, and Use MODULAR Tokenization."""
        dataset_types = batch.get("dataset_type", [])
        ds_name = dataset_types[0] if isinstance(dataset_types, list) and len(dataset_types) > 0 else "unknown"
        ds_name_str = ds_name.value if hasattr(ds_name, "value") else str(ds_name)
        
        relevant_labels = []
        for dt_enum, ds_cfg in DATASET_CONFIGS.items():
            if dt_enum.value == ds_name_str:
                relevant_labels = sorted(list(ds_cfg.valid_labels))
                break
        if not relevant_labels: relevant_labels = self.symbol_manager.original_labels

        p_mappings, c_mappings = None, None
        if self.router is not None:
            num_labels = len(relevant_labels)
            num_slots_available = self.config.diff_symbol_config.num_slots
            batch_slot_indices = random.sample(list(range(num_slots_available)), k=min(num_labels, num_slots_available))
            vocab_indices, _ = self.router.get_slot_mappings(batch_slot_indices, hard=True)
            p_mappings = {label: f"<slot_{slot_idx}>" for label, slot_idx in zip(relevant_labels, batch_slot_indices)}
            clean_symbols = [self.tokenizer.decode([idx]).strip() for idx in vocab_indices]
            c_mappings = {label: symbol for label, symbol in zip(relevant_labels, clean_symbols)}
        else:
            force_new = (self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE) or (batch_idx == 0)
            p_mappings = c_mappings = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)

        # 1. Rewrite text
        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_mappings, completion_mappings=c_mappings)

        # 2. MODULAR TOKENIZATION: Delegate to the processor
        if self.processor is not None and "prompt" in updated_batch:
            tokenized_data = self.processor.tokenize_batch(updated_batch["prompt"], updated_batch["completion"])
            updated_batch.update(tokenized_data)

        return updated_batch

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, num_batches = 0.0, 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            try:
                updated_batch = self._apply_symbol_replacement(batch, epoch, batch_idx)
                if not self._logged_sample:
                    logging.info("Decoded Check (Batch 0): %s", self.tokenizer.decode(updated_batch["input_ids"][0]))
                    self._logged_sample = True
                updated_batch = self._move_batch_to_device(updated_batch)
                outputs = self.model(updated_batch, router=self.router, dspo_module=self.dspo_module)
                loss = outputs.get("loss")
                if loss is None or torch.isnan(loss): continue
                (loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                total_loss += loss.item() * accumulation_steps
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}"})
            except Exception as exc: logging.error(f"Batch {batch_idx} failed: {exc}")
        return total_loss / max(num_batches, 1)

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        return self.validator.run_comprehensive_validation(model=self.model, val_dataloader=self.val_dataloader, epoch=epoch)

    def run_complete_training(self) -> Dict[str, Any]:
        self._setup_lora_optimizer()
        history = []
        for epoch in range(self.config.lora_config.epochs):
            history.append({"epoch": epoch + 1, "train_loss": self._train_one_epoch(epoch), "validation": self._run_validation(epoch)})
        return {"history": history}
