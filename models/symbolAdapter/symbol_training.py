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
        self._epoch_log_count = 0  # logs first 2 batches per epoch
        self._swap_cache: dict = {}       # per-dataset swap mapping; reset each epoch
        self._slot_assignments: dict = {} # ds_name → list[slot_idx]; reset per rotation
        self._assignment_steps: dict = {} # ds_name → global_step when last assigned

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
            num_slots = self.config.diff_symbol_config.num_slots
            K = self.config.diff_symbol_config.slot_vocab_size
            required_pool = num_slots * K

            vocab_filter = VocabFilter(tokenizer)
            symbol_pool = vocab_filter.generate_symbol_pool(
                pool_size=required_pool,
                exclude_labels=self.symbol_manager.original_labels if self.symbol_manager else None,
            )
            if len(symbol_pool) < required_pool:
                logging.warning(
                    "D-SPO: pool has %d tokens but need %d (%d slots × %d). "
                    "Reducing slot_vocab_size to fit.",
                    len(symbol_pool), required_pool, num_slots, K,
                )
                K = len(symbol_pool) // num_slots
                symbol_pool = symbol_pool[:num_slots * K]

            # Partition pool into non-overlapping groups of K, one per slot
            random.shuffle(symbol_pool)
            slot_vocab_indices = [symbol_pool[i * K:(i + 1) * K] for i in range(num_slots)]
            logging.info(
                "D-SPO: %d slots × %d private tokens each (pool=%d)",
                num_slots, K, len(symbol_pool),
            )

            self.router = SymbolRouter(
                num_slots=num_slots,
                slot_vocab_size=K,
                slot_vocab_indices=slot_vocab_indices,
                initial_tau=self.config.diff_symbol_config.tau,
                tau_min=self.config.diff_symbol_config.tau_min,
            ).to(self.config.device)

            slot_tokens = [f"<slot_{i}>" for i in range(num_slots)]
            slot_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in slot_tokens]
            unk_id = tokenizer.unk_token_id
            bad_tokens = [t for t, tid in zip(slot_tokens, slot_token_ids) if tid is None or tid == unk_id]
            if bad_tokens:
                raise ValueError(
                    f"D-SPO: slot tokens not in tokenizer vocabulary: {bad_tokens}. "
                    "Ensure setup_dspo_tokenizer() was called before model init."
                )
            logging.info(
                "D-SPO slot token IDs validated (first 10 of %d): %s",
                len(slot_tokens),
                dict(list(zip(slot_tokens, slot_token_ids))[:10]),
            )
            self.dspo_module = DspoModule(slot_token_ids).to(self.config.device)
            logging.info("D-SPO initialized: %d slots, %d tokens/slot", num_slots, K)

            # Attach to model so validation.py can find them via getattr(model, "router")
            self.model.router = self.router
            self.model.dspo_module = self.dspo_module

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
        slot_only = self.config.diff_symbol_config.slot_only
        if not slot_only:
            lora_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    lora_params.append(param)
            if lora_params:
                trainable_params.append({"params": lora_params, "lr": self.config.lora_config.learning_rate})
        if self.router is not None:
            trainable_params.append({"params": self.router.parameters(), "lr": self.config.diff_symbol_config.router_lr})
        if not trainable_params: raise ValueError("No trainable parameters found")
        if slot_only:
            logging.info("slot_only=True: LoRA frozen, training router only (%d params)", sum(p.numel() for p in self.router.parameters()))
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
        _dspo_probs = None  # pre-computed probs to pass to injection (same gumbel sample as targets)
        if self.router is not None:
            # Fixed slot assignment: sorted labels → slot_0, slot_1, ... (set once, never rotated)
            # Matches validation which uses the same alphabetical ordering.
            num_labels = len(relevant_labels)
            if ds_name_str not in self._slot_assignments:
                self._slot_assignments[ds_name_str] = list(range(num_labels))
                logging.info(
                    "D-SPO fixed slot assignment (dataset=%s): %s → slots %s",
                    ds_name_str, relevant_labels, self._slot_assignments[ds_name_str],
                )
            batch_slot_indices = self._slot_assignments[ds_name_str]

            # D-SPO Mode — slot assignments rotate on a schedule instead of every batch.
            # rotation_interval=0 → per-epoch (reset at epoch start, assign on first batch).
            # rotation_interval>0 → every N global steps.
            # num_labels = len(relevant_labels)
            # num_slots_available = self.config.diff_symbol_config.num_slots
            # rotation_interval = self.config.diff_symbol_config.rotation_interval
            # steps_per_epoch = len(self.train_dataloader)
            # effective_interval = rotation_interval if rotation_interval > 0 else steps_per_epoch
            # last_step = self._assignment_steps.get(ds_name_str, -(effective_interval + 1))
            # needs_rotation = (
            #     ds_name_str not in self._slot_assignments
            #     or (self.global_step - last_step) >= effective_interval
            # )
            # if needs_rotation:
            #     batch_slot_indices = random.sample(
            #         list(range(num_slots_available)),
            #         k=min(num_labels, num_slots_available),
            #     )
            #     self._slot_assignments[ds_name_str] = batch_slot_indices
            #     self._assignment_steps[ds_name_str] = self.global_step
            #     logging.info(
            #         "D-SPO slot rotation (step=%d interval=%d dataset=%s): %s → slots %s",
            #         self.global_step, effective_interval, ds_name_str,
            #         relevant_labels, batch_slot_indices,
            #     )
            # else:
            #     batch_slot_indices = self._slot_assignments[ds_name_str]

            # Always use deterministic (argmax) for training targets — stable cross-batch.
            # Tau still anneals softmax temperature for gradient quality, but never
            # re-introduces gumbel noise that would flip argmax and corrupt the training signal.
            all_slot_indices = list(range(self.router.num_slots))
            deterministic = True
            all_vocab_indices, _dspo_probs = self.router.get_slot_mappings(all_slot_indices, hard=True, deterministic=deterministic)
            vocab_indices = all_vocab_indices[batch_slot_indices]
            p_mappings = {label: f"<slot_{slot_idx}>" for label, slot_idx in zip(relevant_labels, batch_slot_indices)}
            clean_symbols = [
                self.tokenizer.decode([idx]).strip() or f"<tok_{idx.item()}>"
                for idx in vocab_indices
            ]
            c_mappings = {label: symbol for label, symbol in zip(relevant_labels, clean_symbols)}
            if batch_idx < 2:
                logging.info("D-SPO mapping (epoch=%d batch=%d dataset=%s):", epoch + 1, batch_idx, ds_name_str)
                mismatches = []
                for j, (label, slot_idx) in enumerate(zip(relevant_labels, batch_slot_indices)):
                    # Verify: what _dspo_probs would inject == completion target
                    k_idx = torch.argmax(_dspo_probs[slot_idx]).item()
                    inject_vocab_id = self.router.slot_vocab_indices[slot_idx][k_idx].item()
                    inject_token = self.tokenizer.decode([inject_vocab_id]).strip()
                    target_token = clean_symbols[j]
                    match_str = "OK" if inject_token == target_token else f"MISMATCH inject={inject_token!r}"
                    logging.info("  %-20s -> prompt: %-12s  completion: %-12s  inject_check: %s",
                                 label, p_mappings[label], target_token, match_str)
                    if inject_token != target_token:
                        mismatches.append(slot_idx)
                if mismatches:
                    logging.warning("D-SPO inject/target mismatch on slots %s", mismatches)
                else:
                    logging.info("D-SPO inject/target alignment: ALL OK (mode=%s tau=%.3f)",
                                 "deterministic" if deterministic else "gumbel", self.router.tau)
        elif self.config.symbol_config.swap_labels:
            per_instance = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE
            # per_instance: new shuffle every batch
            # per_epoch: one shuffle per dataset per epoch (cached after first batch)
            if per_instance or ds_name_str not in self._swap_cache:
                base_mapping = None
                if not self.config.symbol_config.no_symbols:
                    # Use _pure_symbol_mappings — always label→symbol, never affected by
                    # swap_labels. Filter to this dataset's labels only.
                    full_base = self.symbol_manager._pure_symbol_mappings
                    base_mapping = {l: full_base[l] for l in relevant_labels if l in full_base}
                # per_epoch: pass epoch so shuffle is seeded → guaranteed different each epoch
                # per_instance: pass None so global random state → varies each batch
                mapping = self.symbol_manager.generate_swap_mapping_for_labels(
                    relevant_labels,
                    base_symbol_mapping=base_mapping,
                    epoch=epoch if not per_instance else None,
                )
                if not per_instance:
                    self._swap_cache[ds_name_str] = mapping
            else:
                mapping = self._swap_cache[ds_name_str]
            p_mappings = c_mappings = mapping
            if batch_idx < 2:
                mode = "symbol-swap" if not self.config.symbol_config.no_symbols else "label-swap"
                logging.info("Swap mapping [%s] (epoch=%d batch=%d dataset=%s): %s",
                             mode, epoch + 1, batch_idx, ds_name_str, p_mappings)
        else:
            force_new = (self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE) or (batch_idx == 0)
            p_mappings = c_mappings = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)
            if batch_idx < 2:
                logging.info("Symbol mapping (epoch=%d batch=%d dataset=%s): %s", epoch + 1, batch_idx, ds_name_str, p_mappings)

        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_mappings, completion_mappings=c_mappings)
        if self.processor is not None and "prompt" in updated_batch:
            tokenized_data = self.processor.tokenize_batch(updated_batch["prompt"], updated_batch["completion"])
            updated_batch.update(tokenized_data)
        if _dspo_probs is not None:
            updated_batch["dspo_probs"] = _dspo_probs  # moved to device by _move_batch_to_device
        return updated_batch

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._swap_cache = {}  # reset per-epoch swap assignments
        # if self.router is not None and self.config.diff_symbol_config.rotation_interval == 0:
        #     # Per-epoch rotation: clear assignments so first batch of each epoch rotates
        #     self._slot_assignments = {}
        #     self._assignment_steps = {}
        total_loss, num_batches = 0.0, 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, raw_batch in enumerate(progress_bar):
            try:
                if "completion" not in raw_batch or not raw_batch["completion"]:
                    for key in ["label", "true_label", "target"]:
                        if key in raw_batch and raw_batch[key]: raw_batch["completion"] = raw_batch[key]; break

                updated_batch = self._apply_symbol_replacement(raw_batch, epoch, batch_idx)
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
                outputs = self.model(updated_batch, router=self.router, dspo_module=self.dspo_module)
                
                loss = outputs.get("loss")
                if loss is None or torch.isnan(loss): continue
                (loss / accumulation_steps).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.router is not None:
                        log_this_step = (self.global_step % 50) < 3
                        if log_this_step:
                            params = [p for p in self.router.parameters() if p.grad is not None]
                            if params:
                                router_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in params) ** 0.5
                                entropy = self.router.get_safety_scores().item()
                                logging.info("D-SPO step=%d  router_grad_norm=%.4f  slot_entropy=%.4f  tau=%.4f",
                                             self.global_step, router_grad_norm, entropy, self.router.tau)
                                # Check if preferences are actually changing (direct weight diagnostic)
                                prefs = self.router.preferences  # [num_slots, K]
                                pref_norm = prefs.norm().item()
                                pref_max_diff = (prefs.max(dim=-1).values - prefs.min(dim=-1).values).mean().item()
                                conf_scores = self.router.get_confidence_scores()
                                conf_max = conf_scores.max().item()
                                conf_mean = conf_scores.mean().item()
                                logging.info(
                                    "D-SPO step=%d  prefs_norm=%.4f  pref_spread=%.4f  "
                                    "conf_max=%.3f  conf_mean=%.3f  tau=%.4f",
                                    self.global_step, pref_norm, pref_max_diff,
                                    conf_max, conf_mean, self.router.tau,
                                )
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.router is not None:
                        self.router.update_tau(self.config.diff_symbol_config.tau_anneal_rate)

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.6f}"})
            except Exception as exc: logging.error(f"Batch {batch_idx} failed: {exc}")
        
        # Accumulation Cleanup
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
        logging.info("Starting simple LoRA training (no phases, no cycles, no steps)")
        self._setup_lora_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        history = []

        if self.config.validate_before_training:
            logging.info("Running baseline validation before training (epoch 0)")
            baseline_scores = self._run_validation(epoch=-1)
            logging.info(f"Baseline validation (pre-training): {baseline_scores}")
            history.append({"epoch": 0, "train_loss": None, "validation": baseline_scores})

        for epoch in range(self.config.lora_config.epochs):
            self._epoch_log_count = 0
            logging.info(f"Epoch {epoch + 1}/{self.config.lora_config.epochs}")
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

        self._save_checkpoint(self.config.lora_config.epochs - 1, "final")
        return {"history": history}
