"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import os
import random
from typing import Any, Dict, List

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
        processor,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer=None,
        symbol_manager: SymbolManager = None,
        train_dataset_names: set = None,
    ):
        self.config = config
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.symbol_manager = symbol_manager
        self.train_dataset_names = train_dataset_names or set()
        self.optimizer = None
        self.global_step = 0
        self._logged_sample = False

        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            processor=self.processor,
            max_val_samples=config.data_config.val_max_samples,
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
        checkpoint_dir = self.config.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logging.info("Training environment setup complete")

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
        if not trainable_params: raise ValueError("No trainable parameters found")
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.lora_config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.config.lora_config.weight_decay)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor): device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else: device_batch[key] = value
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
            if batch_idx == 0:
                logging.info(f"D-SPO Mapping for Batch 0 ({ds_name_str}):")
                for label, slot_marker in p_mappings.items(): logging.info(f"  {label} -> {slot_marker} (Target: {c_mappings[label]})")
        else:
            force_new = (self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE) or (batch_idx == 0)
            p_mappings = c_mappings = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)

        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_mappings, completion_mappings=c_mappings)
        if self.processor is not None and "prompt" in updated_batch:
            tokenized_data = self.processor.tokenize_batch(updated_batch["prompt"], updated_batch["completion"])
            updated_batch.update(tokenized_data)
        return updated_batch

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, num_batches = 0.0, 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, raw_batch in enumerate(progress_bar):
            try:
                if "completion" not in raw_batch or not raw_batch["completion"]:
                    for key in ["label", "true_label", "target"]:
                        if key in raw_batch and raw_batch[key]: raw_batch["completion"] = raw_batch[key]; break

                updated_batch = self._apply_symbol_replacement(raw_batch, epoch, batch_idx)
                if not self._logged_sample:
                    logging.info("--- Training Batch 0 Debug ---")
                    logging.info("Decoded Check: %s", self.tokenizer.decode(updated_batch["input_ids"][0][-30:]))
                    self._logged_sample = True

                updated_batch = self._move_batch_to_device(updated_batch)
                outputs = self.model(updated_batch, router=self.router, dspo_module=self.dspo_module)
                
                loss = outputs.get("loss")
                if loss is None or torch.isnan(loss): continue
                (loss / accumulation_steps).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.router is not None and self.global_step % 10 == 0:
                        params = [p for p in self.router.parameters() if p.grad is not None]
                        if params:
                            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                            logging.info(f"D-SPO Step {self.global_step}: Router Grad Norm = {grad_norm:.4f}")
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.router is not None:
                        self.router.update_tau(self.config.diff_symbol_config.tau_anneal_rate)

                total_loss += loss.item() * accumulation_steps
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}"})
            except Exception as exc: logging.error(f"Batch {batch_idx} failed: {exc}")
        
        # Accumulation Cleanup (From remote branch)
        if num_batches > 0 and (num_batches % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        progress_bar.close()
        return total_loss / max(num_batches, 1)

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        return self.validator.run_comprehensive_validation(model=self.model, val_dataloader=self.val_dataloader, epoch=epoch)

    def _save_checkpoint(self, epoch: int, checkpoint_type: str):
        checkpoint_dir = self.config.get_training_output_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"lora_epoch{epoch + 1}_{checkpoint_type}.pt")
        trainable_state = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        checkpoint_data = {
            "model_state": trainable_state,
            "router_state": self.router.state_dict() if self.router else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
            "symbol_mappings": {
                "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch) if self.symbol_manager else {},
                "original_labels": self.symbol_manager.original_labels if self.symbol_manager else [],
                "symbol_type": self.symbol_manager.symbol_type if self.symbol_manager else "",
                "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch if self.symbol_manager else False,
            }
        }
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")

    def run_complete_training(self) -> Dict[str, Any]:
        self._setup_lora_optimizer()
        history = []
        for epoch in range(self.config.lora_config.epochs):
            self._logged_sample = False
            logging.info(f"Epoch {epoch + 1}/{self.config.lora_config.epochs}")
            epoch_loss = self._train_one_epoch(epoch)
            validation_scores = self._run_validation(epoch)
            history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "validation": validation_scores})
            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "periodic")
        
        # 1. Consolidated Validation Summary Table (Our Big Table)
        logging.info("=" * 30 + " Consolidated Validation Summary " + "=" * 30)
        all_ds_names, all_mode_names = [], []
        for entry in history:
            modes = entry.get("validation", {}).get("all_modes", {})
            for mode, datasets in modes.items():
                if mode not in all_mode_names: all_mode_names.append(mode)
                for ds in datasets.keys():
                    if ds not in all_ds_names: all_ds_names.append(ds)
        
        if all_ds_names:
            train_ds = [n for n in all_ds_names if n in self.train_dataset_names]
            val_ds = [n for n in all_ds_names if n not in self.train_dataset_names]
            header = f"{'Epoch':<8} | {'Mode':<12}"
            for ds in all_ds_names: header += f" | {ds:<15}"
            if len(train_ds) > 1: header += " | Avg (train)"
            if len(val_ds) > 1: header += " | Avg (val)"
            logging.info(header + "\n" + "-" * len(header))
            for i, entry in enumerate(history):
                modes = entry.get("validation", {}).get("all_modes", {})
                for mode in all_mode_names:
                    datasets = modes.get(mode, {})
                    row = f"{entry['epoch']:<8} | {mode:<12}"
                    t_scores, v_scores = [], []
                    for ds in all_ds_names:
                        s = datasets.get(ds, {}).get("score")
                        row += f" | {s:<15.6f}" if s is not None else " | -"
                        if s is not None: (t_scores if ds in train_ds else v_scores).append(s)
                    if len(train_ds) > 1: row += f" | {sum(t_scores)/len(t_scores):<12.6f}" if t_scores else " | -"
                    if len(val_ds) > 1: row += f" | {sum(v_scores)/len(v_scores):<12.6f}" if v_scores else " | -"
                    logging.info(row)
                if i < len(history) - 1: logging.info("-" * len(header))
        logging.info("=" * 80)

        # 2. COMPLETE TRAINING SUMMARY (Remote branch's Small Table)
        logging.info("=" * 100 + "\nCOMPLETE TRAINING SUMMARY\n" + "=" * 100)
        header = f"{'Epoch':<8} {'Loss':<12} {'Avg Score':<12} {'Mode'}"
        logging.info(header + "\n" + "-" * 100)
        for e in history:
            val = e["validation"]
            modes = val.get("all_modes", {})
            mode_key = "fixed" if "fixed" in modes else (list(modes.keys())[0] if modes else "N/A")
            logging.info(f"{e['epoch']:<8} {e['train_loss']:<12.4f} {val.get('avg_score', 0.0):<12.4f} {mode_key}")
        logging.info("=" * 100)
        
        self._save_checkpoint(self.config.lora_config.epochs - 1, "final")
        return {"history": history}
