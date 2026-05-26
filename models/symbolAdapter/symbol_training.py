"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import os
from typing import Any, Dict, List

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

    def _tokenize_raw_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        prompts = raw_batch.get("prompt", [])
        completions = raw_batch.get("completion", [])
        audios = raw_batch.get("audio", [])
        texts = raw_batch.get("text", [])
        dataset_types = raw_batch.get("dataset_type", [])
        input_modes = raw_batch.get("input_mode", [])
        fewshot_modes = raw_batch.get("fewshot_mode", [])
        examples_audios = raw_batch.get("examples_audio", [])
        is_training_flags = raw_batch.get("is_training", [])

        n = len(prompts) if isinstance(prompts, (list, tuple)) else 1
        if n == 0:
            return raw_batch

        def _ensure_len(val, default=None):
            if isinstance(val, (list, tuple)) and len(val) == n:
                return list(val)
            if n == 1 and not isinstance(val, (list, tuple)) and val is not None:
                return [val]
            return [default] * n

        prompts = _ensure_len(prompts, None)
        completions = _ensure_len(completions, "")
        audios = _ensure_len(audios, None)
        texts = _ensure_len(texts, "")
        dataset_types = _ensure_len(dataset_types, None)
        input_modes = _ensure_len(input_modes, None)
        fewshot_modes = _ensure_len(fewshot_modes, None)
        examples_audios = _ensure_len(examples_audios, None)
        is_training_flags = _ensure_len(is_training_flags, True)

        processed_items: List[Dict[str, Any]] = []
        for i in range(n):
            if prompts[i] is None:
                continue
            item = {
                "prompt": prompts[i],
                "completion": completions[i],
                "audio": audios[i],
                "text": texts[i],
                "dataset_type": dataset_types[i],
                "input_mode": input_modes[i],
                "fewshot_mode": fewshot_modes[i],
                "examples_audio": examples_audios[i],
            }
            inputs = self.processor.process_inputs(item, is_training=bool(is_training_flags[i]))
            merged = dict(item)
            merged.update(inputs)
            processed_items.append(merged)

        if not processed_items:
            return raw_batch

        tokenized_batch = self.processor.collate_batch(processed_items)
        tokenized_batch["prompt"] = prompts
        tokenized_batch["completion"] = completions
        tokenized_batch["text"] = texts
        tokenized_batch["dataset_type"] = dataset_types

        return tokenized_batch

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, raw_batch in enumerate(progress_bar):
            try:
                # Ensure completion field exists
                if "completion" not in raw_batch or not raw_batch["completion"]:
                    for key in ["label", "true_label", "target", "answer"]:
                        if key in raw_batch and raw_batch[key] is not None:
                            raw_batch["completion"] = raw_batch[key]
                            break

                # 1) Apply symbol replacement on RAW batch
                updated_raw = self._apply_symbol_replacement(raw_batch, epoch, batch_idx)

                # Log one sample per training run
                if not self._logged_sample:
                    prompts = updated_raw.get("prompt")
                    labels = updated_raw.get("completion")
                    if isinstance(prompts, list) and isinstance(labels, list) and prompts and labels:
                        logging.info("Sample prompt (epoch %d): %s", epoch + 1, prompts[0])
                        logging.info("Sample label (epoch %d): %s", epoch + 1, labels[0])
                        self._logged_sample = True

                # 2) Tokenize AFTER replacement
                updated_batch = self._tokenize_raw_batch(updated_raw)

                # 3) Move tensors to device
                updated_batch = self._move_batch_to_device(updated_batch)

                outputs = self.model(updated_batch)
                loss = outputs.get("loss")
                if loss is None:
                    continue

                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.config.lora_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.lora_config.max_grad_norm
                        )
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

        if num_batches > 0 and (num_batches % accumulation_steps) != 0:
            if self.config.lora_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.lora_config.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

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
                "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch) if self.symbol_manager else {},
                "original_labels": self.symbol_manager.original_labels if self.symbol_manager else [],
                "symbol_type": self.symbol_manager.symbol_type if self.symbol_manager else "",
                "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch if self.symbol_manager else False,
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

        # Consolidated validation summary
        logging.info("================ Consolidated Validation Summary ================")
        all_dataset_names = []
        all_mode_names = []
        for entry in history:
            validation = entry.get("validation", {})
            all_modes = validation.get("all_modes", {})
            for mode, datasets in all_modes.items():
                if mode not in all_mode_names:
                    all_mode_names.append(mode)
                for ds_name in datasets.keys():
                    if ds_name not in all_dataset_names:
                        all_dataset_names.append(ds_name)

        if all_dataset_names:
            dataset_display = {
                name: f"{name} (train)" if name in self.train_dataset_names else f"{name} (val)"
                for name in all_dataset_names
            }
            header_cols = ["Epoch", "Mode"] + [dataset_display[name] for name in all_dataset_names]
            header = " | ".join(f"{col:<12}" for col in header_cols)
            logging.info(header)
            logging.info("-" * len(header))

            for entry in history:
                epoch_num = entry["epoch"]
                validation = entry.get("validation", {})
                all_modes = validation.get("all_modes", {})
                for mode in all_mode_names:
                    datasets = all_modes.get(mode, {})
                    row_values = [f"{epoch_num:<12}", f"{mode:<12}"]
                    for ds_name in all_dataset_names:
                        score = datasets.get(ds_name, {}).get("score")
                        row_values.append(f"{score:.6f}" if score is not None else "-")
                    logging.info(" | ".join(row_values))
        else:
            logging.info("No validation metrics available to summarize.")
        logging.info("=================================================================")

        self._save_checkpoint(self.config.lora_config.epochs - 1, "final")

        # Final summary table
        logging.info("=" * 100)
        logging.info("COMPLETE TRAINING SUMMARY - ALL EPOCHS")
        logging.info("=" * 100)
        header = f"{'Epoch':<8} {'Loss':<12} {'Avg Score':<12}"
        if history:
            first_modes = history[0]["validation"].get("all_modes", {})
            mode_key = (
                "original" if "original" in first_modes
                else ("fixed" if "fixed" in first_modes
                else (list(first_modes.keys())[0] if first_modes else None))
            )
            if mode_key:
                for ds_name in first_modes[mode_key].keys():
                    header += f" {ds_name:<12}"
            header += f" {'Mode'}"
        logging.info(header)
        logging.info("-" * 100)
        for entry in history:
            ep = entry["epoch"]
            loss = entry["train_loss"]
            val = entry["validation"]
            avg = val.get("avg_score", 0.0)
            modes = val.get("all_modes", {})
            mode_key = (
                "original" if "original" in modes
                else ("fixed" if "fixed" in modes
                else (list(modes.keys())[0] if modes else None))
            )
            row = f"{ep:<8} {loss:<12.4f} {avg:<12.4f}"
            if mode_key:
                for ds_val in modes[mode_key].values():
                    score = ds_val.get("score", 0.0) if isinstance(ds_val, dict) else 0.0
                    row += f" {score:<12.4f}"
                row += f" {mode_key}"
            logging.info(row)
        logging.info("=" * 100)

        return {"history": history}