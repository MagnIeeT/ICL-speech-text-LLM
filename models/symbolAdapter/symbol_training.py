"""Simple Symbol Adapter trainer: LoRA-only, epoch-based, symbol strategy driven."""

import json
import logging
import os
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config.train_config.training_configs import TrainingConfig, SymbolUpdateStrategy
from config.data_config.master_config import DATASET_CONFIGS, DatasetType
from .symbol_manager import SymbolManager, get_dataset_info, compute_slot_offsets, tokenize_batch_with_audio
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
        self._slot_labels: dict = {}      # ds_name → ordered label list (matches slot_assignments)
        self._assignment_steps: dict = {} # ds_name → global_step when last assigned
        self._slot_offsets: dict = compute_slot_offsets(self.train_dataset_names)
        self._in_phase2: bool = False
        self._full_router = None             # always references the router even when Phase 0 temporarily sets self.router = None
        self._slot_symbol_map: dict = None   # slot_idx → symbol text; built from all trained slots at Phase 1→2 transition
        self._phase2_label_map: dict = {}    # ds_name → {label → symbol}; per-dataset for multi-dataset D-SPO

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
            if self.config.diff_symbol_config.use_fresh_symbols:
                symbol_pool = vocab_filter.generate_fresh_symbol_pool(
                    pool_size=required_pool,
                    token_size=self.config.diff_symbol_config.symbol_token_size,
                )
            else:
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

            token_size = self.config.diff_symbol_config.symbol_token_size
            unk_id = tokenizer.unk_token_id
            slot_placeholder_ids = []
            bad_tokens = []
            for i in range(num_slots):
                position_ids = []
                for p in range(token_size):
                    tok = f"<slot_{i}_{p}>"
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if tid is None or tid == unk_id:
                        bad_tokens.append(tok)
                    position_ids.append(tid)
                slot_placeholder_ids.append(position_ids)
            if bad_tokens:
                raise ValueError(
                    f"D-SPO: slot tokens not in tokenizer vocabulary: {bad_tokens}. "
                    "Ensure setup_dspo_tokenizer() was called before model init."
                )
            logging.info(
                "D-SPO slot placeholder IDs validated (%d slots × %d positions, first 3): %s",
                num_slots, token_size,
                {f"slot_{i}": slot_placeholder_ids[i] for i in range(min(3, num_slots))},
            )
            self.dspo_module = DspoModule(slot_placeholder_ids).to(self.config.device)
            logging.info("D-SPO initialized: %d slots, %d tokens/slot", num_slots, K)
            self._full_router = self.router  # permanent ref; self.router becomes None during Phase 0

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
        trainable_params = []
        slot_only = self.config.diff_symbol_config.slot_only
        if not slot_only:
            lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora" in n.lower()]
            if lora_params:
                trainable_params.append({"params": lora_params, "lr": base_lr})
        if self.router is not None:
            # filter by requires_grad so a frozen router (Phase 2) is automatically excluded
            router_params = [p for p in self.router.parameters() if p.requires_grad]
            if router_params:
                trainable_params.append({"params": router_params, "lr": self.config.diff_symbol_config.router_lr})
        if not trainable_params:
            raise ValueError("No trainable parameters found")
        n_lora = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad and "lora" in n.lower())
        n_router = sum(p.numel() for p in self.router.parameters() if p.requires_grad) if self.router else 0
        logging.info("Optimizer: slot_only=%s  lora_params=%d  router_params=%d  lr=%.2e", slot_only, n_lora, n_router, base_lr)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.config.lora_config.weight_decay)
        t_max = remaining_epochs if remaining_epochs is not None else self.config.lora_config.epochs
        steps_per_epoch = max(len(self.train_dataloader) // max(self.config.lora_config.gradient_accumulation_steps, 1), 1)
        total_steps = max(t_max, 1) * steps_per_epoch
        # router-only phase: no warmup; LoRA phases: warmup from config
        warmup = 0 if slot_only else self.config.lora_config.warmup_steps
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

    def _get_phase2_mappings(self, ds_name_str: str, relevant_labels: list, batch_idx: int):
        """Return (p_map, c_map) for Phase 2 — refresh per dataset per phase2_rotation schedule."""
        phase2_rotation = self.config.diff_symbol_config.phase2_rotation
        available_slots = list(self._slot_symbol_map.keys())
        ds_label_map = self._phase2_label_map.get(ds_name_str)
        needs_refresh = (
            ds_label_map is None
            or (phase2_rotation == 0 and batch_idx == 0)
            or (phase2_rotation > 0 and self.global_step % phase2_rotation == 0)
        )
        if needs_refresh:
            if phase2_rotation == -1 and self.config.diff_symbol_config.rotation_interval == -1:
                chosen = self._slot_assignments.get(ds_name_str, available_slots[:len(relevant_labels)])
            else:
                chosen = random.sample(available_slots, k=min(len(relevant_labels), len(available_slots)))
            ds_label_map = {label: self._slot_symbol_map[s] for label, s in zip(relevant_labels, chosen)}
            self._phase2_label_map[ds_name_str] = ds_label_map
            logging.info("Phase 2 symbol refresh (step=%d phase2_rotation=%d dataset=%s): %s",
                         self.global_step, phase2_rotation, ds_name_str, ds_label_map)
        p_map = {l: ds_label_map[l] for l in relevant_labels if l in ds_label_map}
        if batch_idx < 2:
            logging.info("Phase 2 text map [%s]: %s", ds_name_str, p_map)
        return p_map, p_map

    def _get_swap_mappings(self, ds_name_str: str, relevant_labels: list, epoch: int, batch_idx: int) -> dict:
        """Return label→symbol mapping for swap mode — cached per epoch or fresh per instance."""
        per_instance = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE
        if per_instance or ds_name_str not in self._swap_cache:
            base_mapping = None
            if not self.config.symbol_config.no_symbols:
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

    def _compute_phase1_mappings(self, ds_name_str: str, relevant_labels: list,
                                  batch_slot_indices: list, epoch: int, batch_idx: int):
        """Gumbel forward pass → (p_mappings, c_mappings, dspo_probs) for Phase 1."""
        all_slot_indices = list(range(self.router.num_slots))
        all_vocab_indices, dspo_probs = self.router.get_slot_mappings(all_slot_indices, hard=True, deterministic=False)
        vocab_indices = all_vocab_indices[batch_slot_indices]
        token_size = self.router.slot_vocab_indices.shape[2]

        p_mappings = {
            label: "".join(f"<slot_{slot_idx}_{p}>" for p in range(token_size))
            for label, slot_idx in zip(relevant_labels, batch_slot_indices)
        }
        clean_symbols = [
            self.tokenizer.decode(vocab_indices[j].tolist()).strip() or f"<tok_{vocab_indices[j].tolist()}>"
            for j in range(len(relevant_labels))
        ]
        c_mappings = dict(zip(relevant_labels, clean_symbols))

        if batch_idx < 2:
            logging.info("D-SPO mapping (epoch=%d batch=%d dataset=%s):", epoch + 1, batch_idx, ds_name_str)
            mismatches = []
            for j, (label, slot_idx) in enumerate(zip(relevant_labels, batch_slot_indices)):
                k_idx = torch.argmax(dspo_probs[slot_idx]).item()
                inject_token = self.tokenizer.decode(self.router.slot_vocab_indices[slot_idx][k_idx].tolist()).strip()
                match_str = "OK" if inject_token == clean_symbols[j] else f"MISMATCH inject={inject_token!r}"
                logging.info("  %-20s -> prompt: %-12s  completion: %-12s  inject_check: %s",
                             label, p_mappings[label], clean_symbols[j], match_str)
                if inject_token != clean_symbols[j]:
                    mismatches.append(slot_idx)
            if mismatches:
                logging.warning("D-SPO inject/target mismatch on slots %s", mismatches)
            else:
                logging.info("D-SPO inject/target alignment: ALL OK (mode=gumbel tau=%.3f)", self.router.tau)

        return p_mappings, c_mappings, dspo_probs

    def _get_phase1_slot_indices(self, ds_name_str: str, relevant_labels: list) -> list:
        """Return slot indices for Phase 1 — fixed assignment or rotation-based."""
        num_labels = len(relevant_labels)
        rotation_interval = self.config.diff_symbol_config.rotation_interval

        if rotation_interval == -1:
            if ds_name_str not in self._slot_assignments:
                offset = self._slot_offsets.get(ds_name_str, 0)
                self._slot_assignments[ds_name_str] = list(range(offset, offset + num_labels))
                self._slot_labels[ds_name_str] = list(relevant_labels)
                logging.info("D-SPO fixed slot assignment (dataset=%s): %s → slots %s",
                             ds_name_str, relevant_labels, self._slot_assignments[ds_name_str])
            return self._slot_assignments[ds_name_str]

        steps_per_epoch = len(self.train_dataloader)
        effective_interval = rotation_interval if rotation_interval > 0 else steps_per_epoch
        last_step = self._assignment_steps.get(ds_name_str, -(effective_interval + 1))
        needs_rotation = (
            ds_name_str not in self._slot_assignments
            or (self.global_step - last_step) >= effective_interval
        )
        if needs_rotation:
            indices = random.sample(list(range(self.router.num_slots)), k=min(num_labels, self.router.num_slots))
            self._slot_assignments[ds_name_str] = indices
            self._slot_labels[ds_name_str] = list(relevant_labels)
            self._assignment_steps[ds_name_str] = self.global_step
            logging.info("D-SPO slot rotation (step=%d interval=%d dataset=%s): %s → slots %s",
                         self.global_step, effective_interval, ds_name_str, relevant_labels, indices)
        return self._slot_assignments[ds_name_str]

    def _apply_symbol_replacement(self, batch: Dict[str, Any], epoch: int, batch_idx: int) -> Dict[str, Any]:
        ds_name_str, relevant_labels = get_dataset_info(batch, self.symbol_manager.original_labels)

        p_mappings, c_mappings = None, None
        _dspo_probs = None
        if self.router is not None:
            if self._slot_symbol_map is not None:
                # Phase 2: bypass router — decode symbols from slot map
                p_map, c_map = self._get_phase2_mappings(ds_name_str, relevant_labels, batch_idx)
                replaced = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_map, completion_mappings=c_map)
                return tokenize_batch_with_audio(replaced, self.processor, replaced.get("completion"))

            # Phase 1: resolve slot indices, then Gumbel forward
            batch_slot_indices = self._get_phase1_slot_indices(ds_name_str, relevant_labels)
            p_mappings, c_mappings, _dspo_probs = self._compute_phase1_mappings(
                ds_name_str, relevant_labels, batch_slot_indices, epoch, batch_idx
            )
        elif self.config.symbol_config.swap_labels:
            p_mappings = c_mappings = self._get_swap_mappings(ds_name_str, relevant_labels, epoch, batch_idx)
        else:
            force_new = (self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE) or (batch_idx == 0)
            all_ds_mappings = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)
            p_mappings = c_mappings = all_ds_mappings.get(ds_name_str) or all_ds_mappings.get("") or {}
            if batch_idx < 2:
                logging.info("Symbol mapping (epoch=%d batch=%d dataset=%s): %s", epoch + 1, batch_idx, ds_name_str, p_mappings)

        replaced = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_mappings, completion_mappings=c_mappings)
        updated_batch = tokenize_batch_with_audio(replaced, self.processor, replaced.get("completion"))
        if _dspo_probs is not None:
            updated_batch["dspo_probs"] = _dspo_probs
        return updated_batch

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._swap_cache = {}  # reset per-epoch swap assignments
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
                fwd_router = None if self._in_phase2 else self.router
                fwd_dspo   = None if self._in_phase2 else self.dspo_module
                outputs = self.model(updated_batch, router=fwd_router, dspo_module=fwd_dspo)
                
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
                    self.scheduler.step()
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
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        progress_bar.close()
        return total_loss / max(num_batches, 1)

    def _build_current_symbol_map(self, epoch: int) -> Dict[str, Dict[str, str]]:
        """Build {ds_name: {label: symbol}} for 'fixed' validation — always reflects current training state."""
        # Phase 2: _phase2_label_map is already {ds_name: {label: symbol}}
        if self._phase2_label_map:
            return dict(self._phase2_label_map)

        # D-SPO Phase 0/1: decode per-dataset from router
        active_router = self._full_router or self.router
        if active_router is not None:
            current_map: Dict[str, Dict[str, str]] = {}
            for ds_name in self.train_dataset_names:
                if ds_name in self._slot_assignments:
                    slot_indices = self._slot_assignments[ds_name]
                    labels = self._slot_labels.get(ds_name, [])
                else:
                    # Phase 0: no assignments yet — use offset-based slots + config labels
                    try:
                        labels = list(DATASET_CONFIGS[DatasetType(ds_name)].valid_labels)
                    except (ValueError, KeyError):
                        continue
                    offset = self._slot_offsets.get(ds_name, 0)
                    slot_indices = list(range(offset, offset + len(labels)))
                if not labels or not slot_indices:
                    continue
                vocab_indices, _ = active_router.get_slot_mappings(slot_indices, hard=True, deterministic=True)
                current_map[ds_name] = {
                    label: self.tokenizer.decode(idx.tolist()).strip() or f"<tok_{idx.tolist()}>"
                    for label, idx in zip(labels, vocab_indices)
                }
            return current_map

        # Non-D-SPO: symbol_manager already returns {ds_name: {label: symbol}}
        return self.symbol_manager.get_symbols_for_epoch(epoch)

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        symbol_map = self._build_current_symbol_map(epoch)

        # D-SPO: extend symbol_map to cross-task val datasets.
        # rotation=-1: only n fixed trained slots → pool = those n symbols.
        # rotation>=0: all N slots trained over rotations → decode all N for larger pool.
        # Non-D-SPO: symbol_manager covers all labels already, unknown is always empty.
        active_router = self._full_router or self.router
        if active_router is not None:
            rotation_interval = self.config.diff_symbol_config.rotation_interval
            if rotation_interval == -1:
                # pool = unique training symbols across all training datasets
                pool = list({s for ds_map in symbol_map.values() for s in ds_map.values()})
            else:
                all_indices = list(range(active_router.num_slots))
                vocab_indices, _ = active_router.get_slot_mappings(all_indices, hard=True, deterministic=True)
                pool = [self.tokenizer.decode(v.tolist()).strip() or f"<tok_{v.tolist()}>" for v in vocab_indices]

            per_ds_map = {}
            for ds_type, ds_cfg in DATASET_CONFIGS.items():
                ds_name = ds_type.value
                labels = list(ds_cfg.valid_labels or [])
                if not labels:
                    continue
                if ds_name in self.train_dataset_names:
                    per_ds_map[ds_name] = dict(symbol_map.get(ds_name, {}))
                else:
                    chosen = random.sample(pool, k=min(len(labels), len(pool)))
                    per_ds_map[ds_name] = dict(zip(labels, chosen))
            symbol_map = per_ds_map

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
            "router_state": (self._full_router or self.router).state_dict() if (self._full_router or self.router) else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
            "symbol_mappings": {
                "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch) if self.symbol_manager else {},
                "original_labels": self.symbol_manager.original_labels if self.symbol_manager else [],
                "symbol_token_size": self.symbol_manager.token_size if self.symbol_manager else 2,
                "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch if self.symbol_manager else False,
            },
            "dspo_slot_symbol_map": self._slot_symbol_map,
            "dspo_phase2_label_map": self._phase2_label_map,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")

    def run_complete_training(self) -> Dict[str, Any]:
        phase0_epochs = self.config.diff_symbol_config.phase0_epochs
        phase1_patience = self.config.diff_symbol_config.phase1_patience
        phase1_epochs = self.config.diff_symbol_config.phase1_epochs
        total_epochs = phase0_epochs + self.config.lora_config.epochs
        in_phase2 = False
        best_conf_mean = -1.0
        best_router_state = None
        patience_counter = 0
        phase1_epoch_count = 0
        history = []

        if self.config.validate_before_training:
            logging.info("Running baseline validation before training (epoch 0)")
            baseline_scores = self._run_validation(epoch=-1)
            logging.info(f"Baseline validation (pre-training): {baseline_scores}")
            history.append({"epoch": 0, "phase": "pre", "train_loss": None, "validation": baseline_scores})

        # ── Phase 0: LoRA warmup on original labels ───────────────────────────
        # Temporarily hide the router and set no_symbols so _apply_symbol_replacement
        # uses original labels with no D-SPO injection — no changes to inner methods needed.
        if phase0_epochs > 0:
            saved_router = self.router
            self.router = None
            self.config.symbol_config.no_symbols = True
            self.config.diff_symbol_config.slot_only = False
            self._setup_lora_optimizer(remaining_epochs=phase0_epochs)
            self.optimizer.zero_grad(set_to_none=True)
            logging.info("Starting Phase 0 — LoRA warmup on original labels for %d epochs", phase0_epochs)

            for epoch in range(phase0_epochs):
                self._epoch_log_count = 0
                logging.info(f"Epoch {epoch + 1}/{total_epochs} [Phase0-LoRA]")
                epoch_loss = self._train_one_epoch(epoch)
                validation_scores = self._run_validation(epoch)
                logging.info(f"Epoch {epoch + 1} loss: {epoch_loss:.6f}")
                logging.info(f"Epoch {epoch + 1} validation: {validation_scores}")
                history.append({"epoch": epoch + 1, "phase": "phase0", "train_loss": epoch_loss, "validation": validation_scores})
                if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(epoch, "phase0")

            # Restore router and no_symbols before Phase 1
            self.router = saved_router
            self.config.symbol_config.no_symbols = False

        # ── Phase 1 + Phase 2 ─────────────────────────────────────────────────
        if phase1_patience > 0 and self.router is not None:
            self.config.diff_symbol_config.slot_only = True
            logging.info("phase1_patience=%d: forcing slot_only=True for Phase 1. LoRA unlocks when conf_mean plateaus.", phase1_patience)

        logging.info("Starting Phase 1 — slot_only=%s  phase1_patience=%d  epochs=%d",
                     self.config.diff_symbol_config.slot_only, phase1_patience, self.config.lora_config.epochs)
        self._setup_lora_optimizer()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.config.lora_config.epochs):
            self._epoch_log_count = 0
            abs_epoch = phase0_epochs + epoch
            current_phase = "phase2" if in_phase2 else "phase1"
            phase_tag = " [Phase2-LoRA]" if in_phase2 else (" [Phase1-Slots]" if phase1_patience > 0 else "")
            logging.info(f"Epoch {abs_epoch + 1}/{total_epochs}{phase_tag}")
            epoch_loss = self._train_one_epoch(abs_epoch)
            validation_scores = self._run_validation(abs_epoch)
            logging.info(f"Epoch {abs_epoch + 1} loss: {epoch_loss:.6f}")
            logging.info(f"Epoch {abs_epoch + 1} validation: {validation_scores}")

            history.append({"epoch": abs_epoch + 1, "phase": current_phase, "train_loss": epoch_loss, "validation": validation_scores})
            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(abs_epoch, "periodic")

            # Phase 1 convergence check: track best conf_mean, switch to Phase 2 when patience exceeded
            if (phase1_patience > 0 or phase1_epochs > 0) and not in_phase2 and self.router is not None:
                current_conf_mean = self.router.get_confidence_scores().mean().item()
                phase1_epoch_count += 1
                if current_conf_mean > best_conf_mean:
                    best_conf_mean = current_conf_mean
                    best_router_state = {k: v.detach().cpu().clone() for k, v in self.router.state_dict().items()}
                    patience_counter = 0
                    logging.info("Phase 1: new best conf_mean=%.4f at epoch %d (saved to CPU RAM)", best_conf_mean, abs_epoch + 1)
                else:
                    patience_counter += 1
                    logging.info("Phase 1: conf_mean=%.4f, no improvement %d/%d", current_conf_mean, patience_counter, phase1_patience)

                patience_triggered = phase1_patience > 0 and patience_counter >= phase1_patience
                cap_triggered = phase1_epochs > 0 and phase1_epoch_count >= phase1_epochs
                if cap_triggered:
                    logging.info("Phase 1: hard cap reached (%d/%d epochs)", phase1_epoch_count, phase1_epochs)
                remaining = self.config.lora_config.epochs - (epoch + 1)
                if (patience_triggered or cap_triggered) and remaining > 0:
                    logging.info("=" * 60)
                    logging.info("PHASE 2: Slots converged (best conf_mean=%.4f). Switching to LoRA text-symbol training.", best_conf_mean)
                    logging.info("%d epochs remaining.", remaining)
                    logging.info("=" * 60)
                    device = next(self.router.parameters()).device
                    self.router.load_state_dict({k: v.to(device) for k, v in best_router_state.items()})
                    for p in self.router.parameters():
                        p.requires_grad_(False)
                    self.config.diff_symbol_config.slot_only = False

                    # Decode all trained slots → slot_idx: symbol_text (permanent lookup, no router needed after this)
                    # rotation_interval=-1: only the n fixed slots were trained → decode those only
                    # rotation_interval>=0: all N slots were trained via rotation → decode all N
                    rotation_interval = self.config.diff_symbol_config.rotation_interval
                    if rotation_interval == -1:
                        slots_to_decode = sorted({s for indices in self._slot_assignments.values() for s in indices})
                    else:
                        slots_to_decode = list(range(self.router.num_slots))
                    vocab_indices, _ = self.router.get_slot_mappings(slots_to_decode, hard=True, deterministic=True)
                    self._slot_symbol_map = {
                        slot_idx: self.tokenizer.decode(v_idx.tolist()).strip() or f"<tok_{v_idx.tolist()}>"
                        for slot_idx, v_idx in zip(slots_to_decode, vocab_indices)
                    }
                    logging.info("Phase 2 slot→symbol map (%d slots): %s", len(self._slot_symbol_map), self._slot_symbol_map)
                    self._in_phase2 = True

                    # Lower LR avoids peak-LR shock — LoRA was last trained in Phase 0
                    self._setup_lora_optimizer(remaining_epochs=remaining, lr_override=self.config.lora_config.learning_rate * 0.1)
                    self.optimizer.zero_grad(set_to_none=True)
                    in_phase2 = True

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
            header = f"{'Epoch':<8} | {'Phase':<8} | {'Mode':<12}"
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
                    row = f"{entry['epoch']:<8} | {entry.get('phase', ''):<8} | {mode:<12}"
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
