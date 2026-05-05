"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.train_config.training_configs import TrainingConfig, SymbolUpdateStrategy
from .symbol_manager import SymbolManager
from .validation import ValidationManager


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
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.symbol_manager = symbol_manager
        self.optimizer = None
        self.global_step = 0

        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            max_val_samples=config.data_config.val_max_samples,
        )

        self._setup_training_environment()

    def _setup_training_environment(self):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.metrics_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)

        checkpoint_dir = self.config.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logging.info("Training environment setup complete")
        logging.info(f"Checkpoint directory: {checkpoint_dir}")
        logging.info(f"Metrics directory: {self.config.get_metrics_dir()}")
        logging.info(f"Logs directory: {self.config.get_logs_dir()}")

    def _setup_lora_optimizer(self):
        lora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                lora_params.append(param)

        if not lora_params:
            raise ValueError("No trainable LoRA parameters found")

        self.optimizer = torch.optim.AdamW(
            [{"params": lora_params, "lr": self.config.lora_config.learning_rate}],
            lr=self.config.lora_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.lora_config.weight_decay,
        )

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
        if not self.config.symbol_config.dynamic_symbols:
            return self.symbol_manager.replace_symbols_in_batch(batch, epoch=epoch, force_new_symbols=False)

        if self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE:
            force_new_symbols = True
        else:
            force_new_symbols = batch_idx == 0

        return self.symbol_manager.replace_symbols_in_batch(
            batch,
            epoch=epoch,
            force_new_symbols=force_new_symbols,
        )

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                updated_batch = self._apply_symbol_replacement(batch, epoch, batch_idx)
                updated_batch = self._move_batch_to_device(updated_batch)

                outputs = self.model(updated_batch)
                loss = outputs.get("loss")
                if loss is None:
                    continue

                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.config.lora_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.lora_config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                total_loss += loss.item() * accumulation_steps
                num_batches += 1

                if num_batches > 0:
                    progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}"})
            except Exception as exc:
                logging.error(f"Batch {batch_idx} failed: {exc}")

        progress_bar.close()
        return total_loss / max(num_batches, 1)
    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        return self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=epoch,
        )

    def _save_checkpoint(self, epoch: int, checkpoint_type: str):
        checkpoint_dir = self.config.get_training_output_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_name = f"lora_epoch{epoch + 1}_{checkpoint_type}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        trainable_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.clone()

        checkpoint_data = {
            "step_info": {
                "phase": "lora",
                "epoch": epoch + 1,
                "description": "LoRA epoch training",
            },
            "model_state": trainable_state,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
            "symbol_mappings": {
                "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch),
                "original_labels": self.symbol_manager.original_labels,
                "symbol_type": self.symbol_manager.symbol_type,
                "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch,
            },
        }

        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_name}")

    def run_complete_training(self) -> Dict[str, Any]:
        logging.info("Starting simple LoRA training (no phases, no cycles, no steps)")

        self._setup_lora_optimizer()
        self.optimizer.zero_grad(set_to_none=True)

        history = []

        for epoch in range(self.config.lora_config.epochs):
            logging.info(f"Epoch {epoch + 1}/{self.config.lora_config.epochs}")

            epoch_loss = self._train_one_epoch(epoch)
            validation_scores = self._run_validation(epoch)

            logging.info(f"Epoch {epoch + 1} loss: {epoch_loss:.6f}")
            logging.info(f"Epoch {epoch + 1} validation: {validation_scores}")

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "validation": validation_scores,
                }
            )

            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "periodic")

        self._save_checkpoint(self.config.lora_config.epochs - 1, "final")
        return {"history": history}
