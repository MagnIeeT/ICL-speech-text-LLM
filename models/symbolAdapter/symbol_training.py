"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import random
import os
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config.train_config.training_configs import TrainingConfig, SymbolUpdateStrategy
from .symbol_manager import SymbolManager, get_dataset_info, tokenize_batch_with_audio
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
        self._epoch_log_count = 0  # logs first 2 batches per epoch
        self._swap_cache: dict = {}       # per-dataset swap mapping; reset each epoch

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
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logging.info("Training environment setup complete")

    def _setup_lora_optimizer(self, remaining_epochs: int = None, lr_override: float = None):
        base_lr = lr_override if lr_override is not None else self.config.lora_config.learning_rate
        # Collect ALL unfrozen model params, not just LoRA-named ones. SALMONN unfreezes
        # the speech Q-Former + speech_llama_proj (freeze_*=False in custom_salmonn.py);
        # a name filter on "lora" would silently drop them from the optimizer so they
        # never train. For qwen/flamingo only LoRA is unfrozen, so this is equivalent.
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if not lora_params:
            raise ValueError("No trainable parameters found")
        n_model = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad)
        n_lora = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad and "lora" in n.lower())
        logging.info("Optimizer: model_params=%d (of which lora=%d)  lr=%.2e", n_model, n_lora, base_lr)
        self.optimizer = torch.optim.AdamW([{"params": lora_params, "lr": base_lr}], lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.config.lora_config.weight_decay)
        t_max = remaining_epochs if remaining_epochs is not None else self.config.lora_config.epochs
        steps_per_epoch = max(len(self.train_dataloader) // max(self.config.lora_config.gradient_accumulation_steps, 1), 1)
        total_steps = max(t_max, 1) * steps_per_epoch
        warmup = self.config.lora_config.warmup_steps
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
        logging.info("CosineWithWarmup: warmup=%d  total_steps=%d  lr=%.2e", warmup, total_steps, base_lr)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor): device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else: device_batch[key] = value
        return device_batch

    def _get_swap_mappings(self, ds_name_str: str, relevant_labels: list, epoch: int, batch_idx: int,
                           force_real_labels: bool = False) -> dict:
        """Return label->surface mapping for swap mode, cached per epoch or fresh per instance.

        force_real_labels=True permutes the REAL label words even when symbols are
        enabled. The surrogate mixture needs this: its flipped regime must supply
        real vocabulary, which is the whole reason it preserves canonical ability.
        Without it, no_symbols=False would permute the synthetic symbols instead and
        both regimes would be synthetic.
        """
        per_instance = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE
        if per_instance or ds_name_str not in self._swap_cache:
            base_mapping = None
            if not self.config.symbol_config.no_symbols and not force_real_labels:
                full_base = self.symbol_manager._pure_symbol_mappings.get(ds_name_str, {})
                base_mapping = {l: full_base[l] for l in relevant_labels if l in full_base}
            mapping = self.symbol_manager.generate_swap_mapping_for_labels(
                relevant_labels,
                base_symbol_mapping=base_mapping,
                epoch=epoch if not per_instance else None,
            )
            if not per_instance:
                self._swap_cache[ds_name_str] = mapping
        else:
            mapping = self._swap_cache[ds_name_str]
        if batch_idx < 2:
            mode = "symbol-swap" if not self.config.symbol_config.no_symbols else "label-swap"
            logging.info("Swap mapping [%s] (epoch=%d batch=%d dataset=%s): %s",
                         mode, epoch + 1, batch_idx, ds_name_str, mapping)
        return mapping

    def _set_epoch_symbol_maps_on_dataset(self, epoch: int) -> None:
        """Push symbol maps onto the dataset before the DataLoader iterator forks workers.
        Workers inherit the maps at fork time — no IPC needed after this point.
        Only used for non-swap modes (fixed, per_epoch, per_instance, no_symbols).
        """
        dataset = getattr(self.train_dataloader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'set_epoch_symbol_maps'):
            return

        no_symbols = self.config.symbol_config.no_symbols
        per_instance = (self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE)

        if no_symbols:
            dataset.set_epoch_symbol_maps({}, no_symbols=True)
        elif per_instance:
            pool_list = self.symbol_manager._pregenerated_mappings or [self.symbol_manager.fixed_mappings]
            ds_pool: Dict[str, List] = {}
            for full_map in pool_list:
                for ds_name, label_map in full_map.items():
                    ds_pool.setdefault(ds_name, []).append(label_map)
            dataset.set_epoch_symbol_maps({}, no_symbols=False, per_instance=True, pool=ds_pool)
        else:
            maps = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=False)
            dataset.set_epoch_symbol_maps(maps, no_symbols=False, per_instance=False)

        logging.info(
            "Symbol maps pushed to dataset workers (epoch=%d no_symbols=%s per_instance=%s)",
            epoch + 1, no_symbols, per_instance,
        )

    def _apply_symbol_replacement(self, batch: Dict[str, Any], epoch: int, batch_idx: int) -> Dict[str, Any]:
        ds_name_str, relevant_labels = get_dataset_info(batch, self.symbol_manager.original_labels)

        # Surrogate regime for this instance. Both regimes are surrogate tuning: the
        # flipped one assigns permuted REAL label words, never the canonical label, so
        # no instruction-tuning targets enter training.
        _mix = getattr(self.config.symbol_config, "surrogate_mix", 0.0)
        _use_flip = self.config.symbol_config.swap_labels or (_mix > 0 and random.random() < _mix)

        if _use_flip:
            # From the mixture, the flipped regime must use permuted REAL words.
            _from_mix = not self.config.symbol_config.swap_labels
            p_mappings = c_mappings = self._get_swap_mappings(
                ds_name_str, relevant_labels, epoch, batch_idx, force_real_labels=_from_mix)
        else:
            force_new = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE
            all_ds_mappings = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)
            p_mappings = c_mappings = all_ds_mappings.get(ds_name_str) or all_ds_mappings.get("") or {}
            if batch_idx < 2:
                logging.info("Symbol mapping (epoch=%d batch=%d dataset=%s): %s", epoch + 1, batch_idx, ds_name_str, p_mappings)

        replaced = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_mappings, completion_mappings=c_mappings)
        return tokenize_batch_with_audio(replaced, self.processor, replaced.get("completion"))

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._swap_cache = {}  # reset per-epoch swap assignments
        total_loss, num_batches = 0.0, 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps

        # For non-swap modes: push maps onto dataset so workers do substitution.
        # Must happen before progress_bar is created — that's when workers fork.
        # Mixing decides the regime per instance, so the maps cannot be pushed to the
        # dataloader workers ahead of time; substitute in the main thread as swap mode does.
        _use_worker_symbols = not (self.config.symbol_config.swap_labels
                                   or getattr(self.config.symbol_config, "surrogate_mix", 0.0) > 0)
        if _use_worker_symbols:
            self._set_epoch_symbol_maps_on_dataset(epoch)

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        _PROFILE = epoch == 0  # only profile first epoch
        _t_batch_start = time.time()

        _torch_prof = None
        if _PROFILE and getattr(self.config, "torch_profile", False):
            import torch.profiler as _tp
            _prof_out = os.path.join(self.config.output_dir, f"{self.config.run_name}_torchprof")
            _torch_prof = _tp.profile(
                activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
                on_trace_ready=_tp.tensorboard_trace_handler(_prof_out),
                schedule=_tp.schedule(wait=2, warmup=2, active=20, repeat=1),
            )
            _torch_prof.start()
            logging.info("PyTorch profiler started → %s", _prof_out)

        for batch_idx, raw_batch in enumerate(progress_bar):
            try:
                _t0 = time.time()
                t_dataloader = _t0 - _t_batch_start  # time waiting for DataLoader

                if "completion" not in raw_batch or not raw_batch["completion"]:
                    for key in ["label", "true_label", "target"]:
                        if key in raw_batch and raw_batch[key]: raw_batch["completion"] = raw_batch[key]; break

                _t1 = time.time()
                if _use_worker_symbols:
                    if "input_ids" in raw_batch:
                        # Flamingo (File 3): worker tokenized — main thread is no-op.
                        updated_batch = raw_batch
                    else:
                        # Qwen (worker tokenization not yet applied): tokenize in main thread.
                        updated_batch = tokenize_batch_with_audio(raw_batch, self.processor, raw_batch.get("completion"))
                else:
                    # swap_labels: substitution + tokenization in main thread.
                    updated_batch = self._apply_symbol_replacement(raw_batch, epoch, batch_idx)
                _t2 = time.time()
                t_tokenize = _t2 - _t1

                if self._epoch_log_count < 2:
                    ids   = updated_batch["input_ids"][0]
                    p_len = int(updated_batch["prompt_length"][0])
                    full_prompt  = self.tokenizer.decode(ids[:p_len],  skip_special_tokens=False)
                    completion_ids = [t for t in ids[p_len:].tolist() if t != -100 and t != self.tokenizer.pad_token_id]
                    completion_txt = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                    logging.info("=" * 60)
                    logging.info("TRAIN epoch=%d batch=%d  prompt_len=%d  seq_len=%d",
                                 epoch + 1, batch_idx, p_len, len(ids))
                    logging.info("--- Full prompt ---\n%s", full_prompt)
                    logging.info("--- Completion text : %r", completion_txt)
                    logging.info("--- Completion IDs  : %s", completion_ids)
                    logging.info("=" * 60)
                    self._epoch_log_count += 1

                updated_batch = self._move_batch_to_device(updated_batch)
                torch.cuda.synchronize()
                _t3 = time.time()
                t_to_device = _t3 - _t2

                outputs = self.model(updated_batch)
                torch.cuda.synchronize()
                _t4 = time.time()
                t_forward = _t4 - _t3

                loss = outputs.get("loss")
                if loss is None or torch.isnan(loss): continue
                (loss / accumulation_steps).backward()
                torch.cuda.synchronize()
                _t5 = time.time()
                t_backward = _t5 - _t4

                if _PROFILE and batch_idx < 50:
                    seq_len = updated_batch["input_ids"].shape[-1]
                    logging.info(
                        "TIMING batch=%d seq_len=%d | dataloader=%.3fs tokenize=%.3fs "
                        "to_device=%.3fs forward=%.3fs backward=%.3fs total=%.3fs",
                        batch_idx, seq_len,
                        t_dataloader, t_tokenize, t_to_device, t_forward, t_backward,
                        time.time() - _t_batch_start,
                    )

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}"})
                _t_batch_start = time.time()
                if _torch_prof is not None:
                    _torch_prof.step()
            except Exception as exc: logging.error(f"Batch {batch_idx} failed: {exc}")
        
        # Accumulation Cleanup
        if num_batches > 0 and (num_batches % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        if _torch_prof is not None:
            _torch_prof.stop()
            logging.info("PyTorch profiler saved.")
        progress_bar.close()
        return total_loss / max(num_batches, 1)

    def _build_current_symbol_map(self, epoch: int) -> Dict[str, Dict[str, str]]:
        """Build {ds_name: {label: symbol}} for 'fixed' validation — always reflects current training state."""
        return self.symbol_manager.get_symbols_for_epoch(epoch)

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        # symbol_manager covers every label of every configured dataset, so the map
        # needs no extension for cross-task validation datasets.
        symbol_map = self._build_current_symbol_map(epoch)
        return self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=epoch,
            symbol_map=symbol_map,
        )

    def _save_checkpoint(self, epoch: int, checkpoint_type: str):
        checkpoint_dir = self.config.get_training_output_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"lora_epoch{epoch + 1}_{checkpoint_type}.pt")
        trainable_state = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        checkpoint_data = {
            "model_state": trainable_state,
            # optimizer_state intentionally NOT saved — it's never loaded (no resume path)
            # and unused at inference; storing it ~tripled checkpoint size.
            "optimizer_state": None,
            "config": self.config,
            "symbol_mappings": {
                "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch) if self.symbol_manager else {},
                "original_labels": self.symbol_manager.original_labels if self.symbol_manager else [],
                "symbol_token_size": self.symbol_manager.token_size if self.symbol_manager else 2,
                "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch if self.symbol_manager else False,
            },
        }
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")

    def run_complete_training(self) -> Dict[str, Any]:
        total_epochs = self.config.lora_config.epochs
        history = []

        if self.config.validate_before_training:
            logging.info("Running baseline validation before training (epoch 0)")
            baseline_scores = self._run_validation(epoch=-1)
            logging.info(f"Baseline validation (pre-training): {baseline_scores}")
            history.append({"epoch": 0, "train_loss": None, "validation": baseline_scores})

        self._setup_lora_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        logging.info("Starting LoRA training for %d epochs", total_epochs)

        for epoch in range(total_epochs):
            self._epoch_log_count = 0
            logging.info(f"Epoch {epoch + 1}/{total_epochs}")
            epoch_loss = self._train_one_epoch(epoch)
            validation_scores = self._run_validation(epoch)
            logging.info(f"Epoch {epoch + 1} loss: {epoch_loss:.6f}")
            logging.info(f"Epoch {epoch + 1} validation: {validation_scores}")

            history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "validation": validation_scores})
            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "periodic")

        # 1. Consolidated Validation Summary Table
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
            for ds in all_ds_names:
                col = f"{ds}(T)" if ds in train_ds else ds
                header += f" | {col:<15}"
            if len(train_ds) > 1: header += " | Avg (train)"
            if len(val_ds) > 1: header += " | Avg (val)"
            logging.info(header)
            logging.info("-" * len(header))
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

        self._save_checkpoint(total_epochs - 1, "final")
        return {"history": history}
