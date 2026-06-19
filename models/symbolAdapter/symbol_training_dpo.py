"""SymDPO-style DPO trainer for symbol-based speech ICL.

Chosen response: correct symbol for the query audio given symbolic demonstration context.
Rejected response: a randomly sampled wrong symbol from the same label set.

Reference model = base model with LoRA disabled (no extra memory overhead).
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.train_config.training_configs import TrainingConfig, SymbolUpdateStrategy
from config.data_config.master_config import DATASET_CONFIGS
from .symbol_manager import SymbolManager, get_dataset_info
from .validation import ValidationManager


class SymbolDPOOrchestrator:
    """DPO trainer that enforces audio grounding via symbolic chosen/rejected preference pairs."""

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
        beta: float = 0.1,
    ):
        self.config = config
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.symbol_manager = symbol_manager
        self.train_dataset_names = train_dataset_names or set()
        self.beta = beta
        self.optimizer = None
        self.global_step = 0
        self._epoch_log_count = 0
        self._swap_cache: dict = {}

        if config.symbol_config.no_symbols and not config.symbol_config.swap_labels:
            logging.warning(
                "SymbolDPO: no_symbols=True means all completions use original label names. "
                "Rejected sampling will always fail — every batch will be skipped. "
                "DPO requires at least two distinct symbols to form preference pairs."
            )

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
        logging.info("DPO training environment setup complete (beta=%.3f)", self.beta)

    def _setup_lora_optimizer(self):
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora" in n.lower()]
        if not lora_params:
            raise ValueError("No trainable LoRA parameters found")
        self.optimizer = torch.optim.AdamW(
            [{"params": lora_params, "lr": self.config.lora_config.learning_rate}],
            lr=self.config.lora_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.lora_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.config.lora_config.epochs, 1), eta_min=0.0
        )
        logging.info("DPO optimizer: %d LoRA params, CosineAnnealingLR T_max=%d lr=%.2e",
                     sum(p.numel() for p in lora_params), self.config.lora_config.epochs, self.config.lora_config.learning_rate)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else:
                device_batch[key] = value
        return device_batch

    def _determine_mappings(self, batch: Dict[str, Any], epoch: int, batch_idx: int) -> Tuple[Dict, Dict]:
        """Return (prompt_mappings, completion_mappings) for this batch."""
        ds_name_str, relevant_labels = get_dataset_info(batch, self.symbol_manager.original_labels if self.symbol_manager else [])
        per_instance = self.config.symbol_config.update_strategy == SymbolUpdateStrategy.PER_INSTANCE

        if self.config.symbol_config.swap_labels:
            if per_instance or ds_name_str not in self._swap_cache:
                base_mapping = None
                if not self.config.symbol_config.no_symbols:
                    full_base = self.symbol_manager._pure_symbol_mappings
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
            return mapping, mapping
        else:
            # DPO requires consistent symbols throughout all training — never regenerate.
            mapping = self.symbol_manager.get_symbols_for_epoch(0, force_new_symbols=False)
            return mapping, mapping

    def _sample_rejected_symbol(self, chosen_symbol: str, c_mappings: Dict[str, str]) -> Optional[str]:
        """Sample a rejected completion with the same number of labels as chosen.

        For multi-label completions (e.g. "Xk7, mA"), samples the same count of
        wrong symbols so the comparison is length-fair. Falls back to however many
        wrong symbols are available if fewer than needed.
        """
        chosen_parts = [s.strip() for s in chosen_symbol.split(",")]
        chosen_set = set(chosen_parts)
        num_needed = len(chosen_parts)

        alternatives = [s for s in c_mappings.values() if s not in chosen_set]
        if not alternatives:
            return None

        sampled = random.sample(alternatives, min(num_needed, len(alternatives)))
        return ", ".join(sampled)

    def _merge_chosen_rejected(self, chosen_batch: Dict[str, Any], rejected_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Stack chosen and rejected into a single batch of size 2.

        input_ids and attention_mask differ (different completions) → cat along batch dim.
        input_features and feature_attention_mask are identical → duplicate once.
        Everything else is shared metadata → keep as-is.
        """
        token_keys = {"input_ids", "attention_mask", "prompt_length"}
        merged = {}
        for key, c_val in chosen_batch.items():
            if not isinstance(c_val, torch.Tensor):
                merged[key] = c_val
            elif key in token_keys:
                # Token sequences differ between chosen and rejected (different completions)
                merged[key] = torch.cat([c_val, rejected_batch[key]], dim=0)
            else:
                # Audio features and all other tensors are identical (same prompt/audio)
                # Duplicate chosen's value to match the merged batch size of 2×B
                merged[key] = torch.cat([c_val, c_val], dim=0)
        return merged

    def _get_completion_logps(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return mean log prob of completion tokens for every example in the batch. Shape: [B]

        Autoregressive shift: logit at position t predicts token at t+1, so completion
        tokens [prompt_len .. T-1] are predicted by logits [prompt_len-1 .. T-2].
        """
        outputs = self.model(batch)
        logits = outputs["logits"]  # [B, T, V]

        if logits is None:
            B = batch["input_ids"].shape[0]
            return torch.zeros(B, device=self.config.device, requires_grad=True)

        input_ids = batch["input_ids"]       # [B, T]
        prompt_lengths = batch["prompt_length"]
        B, T = input_ids.shape
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        vocab_size = logits.shape[-1]

        result = []
        for i in range(B):
            prompt_len = int(prompt_lengths[i])
            ids = input_ids[i]                           # [T]
            lgt = logits[i]                              # [T, V]

            comp_ids    = ids[prompt_len:]               # [comp_len]
            comp_logits = lgt[prompt_len - 1: T - 1, :] # [comp_len, V]

            if comp_ids.shape[0] == 0 or comp_logits.shape[0] == 0:
                result.append(torch.tensor(0.0, device=self.config.device))
                continue

            log_probs = F.log_softmax(comp_logits.float(), dim=-1)
            valid_mask = (comp_ids != pad_id) & (comp_ids != -100)

            if not valid_mask.any():
                result.append(torch.tensor(0.0, device=self.config.device))
                continue

            safe_ids = comp_ids.clamp(0, vocab_size - 1)
            token_log_probs = log_probs.gather(1, safe_ids.unsqueeze(1)).squeeze(1)
            result.append(token_log_probs[valid_mask].mean())

        return torch.stack(result)  # [B]

    def _compute_dpo_loss(self, chosen_batch: Dict[str, Any], rejected_batch: Dict[str, Any]) -> torch.Tensor:
        """2 forward passes (down from 4) by batching chosen + rejected together.

        Pass 1 — policy (LoRA ON):  batch[0]=chosen, batch[1]=rejected → both logps with grad.
        Pass 2 — reference (LoRA OFF): same merged batch → both ref logps, no grad.
        """
        merged = self._merge_chosen_rejected(chosen_batch, rejected_batch)

        policy_logps = self._get_completion_logps(merged)           # [2], grad ON
        policy_chosen_logp, policy_rejected_logp = policy_logps[0], policy_logps[1]

        self.model.model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps = self._get_completion_logps(merged).detach() # [2], no grad
        self.model.model.enable_adapter_layers()
        ref_chosen_logp, ref_rejected_logp = ref_logps[0], ref_logps[1]

        chosen_ratio  = policy_chosen_logp  - ref_chosen_logp
        rejected_ratio = policy_rejected_logp - ref_rejected_logp
        return -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio))

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self._swap_cache = {}
        total_loss, num_batches, skipped = 0.0, 0, 0
        accumulation_steps = self.config.lora_config.gradient_accumulation_steps
        progress_bar = tqdm(self.train_dataloader, desc=f"DPO Epoch {epoch + 1}", leave=False)

        for batch_idx, raw_batch in enumerate(progress_bar):
            try:
                if "completion" not in raw_batch or not raw_batch["completion"]:
                    for key in ["label", "true_label", "target"]:
                        if key in raw_batch and raw_batch[key]:
                            raw_batch["completion"] = raw_batch[key]
                            break

                p_mappings, c_mappings = self._determine_mappings(raw_batch, epoch, batch_idx)

                if not c_mappings:
                    skipped += 1
                    continue

                # Symbol-replace prompt and completion text (no tokenization yet)
                text_batch = self.symbol_manager.replace_symbols_in_batch(
                    raw_batch, prompt_mappings=p_mappings, completion_mappings=c_mappings
                )

                chosen_completion = text_batch["completion"]  # list[str], correct symbol(s)
                chosen_sym = chosen_completion[0] if chosen_completion else ""

                rejected_sym = self._sample_rejected_symbol(chosen_sym, c_mappings)
                if rejected_sym is None:
                    skipped += 1
                    continue

                rejected_completion = [
                    self._sample_rejected_symbol(sym, c_mappings) or sym
                    for sym in chosen_completion
                ]

                if self._epoch_log_count < 2:
                    original_label = raw_batch["completion"][0] if raw_batch.get("completion") else "?"
                    logged_rejected = rejected_completion[0] if rejected_completion else ""
                    logging.info("=" * 60)
                    logging.info(
                        "DPO train epoch=%d batch=%d  label=%r  chosen=%r  rejected=%r  mapping=%s",
                        epoch + 1, batch_idx, original_label, chosen_sym, logged_rejected, c_mappings,
                    )
                    logging.info("=" * 60)
                    self._epoch_log_count += 1

                # Tokenize separately: same prompt, different completion
                # For Flamingo: embed audio into prompt dicts so tokenize_batch can access it
                if text_batch.get("audio"):
                    for p, a in zip(text_batch["prompt"], text_batch["audio"]):
                        if isinstance(p, dict): p["_audio"] = a
                chosen_tok   = self.processor.tokenize_batch(text_batch["prompt"], chosen_completion)
                rejected_tok  = self.processor.tokenize_batch(text_batch["prompt"], rejected_completion)

                if self._epoch_log_count < 2:
                    c_ids_raw = self.tokenizer.encode(chosen_sym, add_special_tokens=False)
                    r_ids_raw = self.tokenizer.encode(rejected_sym, add_special_tokens=False)
                    c_toks = [self.tokenizer.decode([t]) for t in c_ids_raw]
                    r_toks = [self.tokenizer.decode([t]) for t in r_ids_raw]
                    logging.info(
                        "TOK epoch=%d batch=%d  chosen=%r→%s(n=%d)  rejected=%r→%s(n=%d)  "
                        "chosen_seq_len=%d  rejected_seq_len=%d",
                        epoch + 1, batch_idx,
                        chosen_sym, c_toks, len(c_ids_raw),
                        rejected_sym, r_toks, len(r_ids_raw),
                        chosen_tok["input_ids"].shape[-1],
                        rejected_tok["input_ids"].shape[-1],
                    )

                # Merge: audio features from text_batch, token ids from each tok
                chosen_batch   = {**text_batch, **chosen_tok}
                rejected_batch  = {**text_batch, **rejected_tok}

                chosen_batch   = self._move_batch_to_device(chosen_batch)
                rejected_batch  = self._move_batch_to_device(rejected_batch)

                loss = self._compute_dpo_loss(chosen_batch, rejected_batch)

                if loss is None or torch.isnan(loss):
                    skipped += 1
                    continue

                (loss / accumulation_steps).backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.lora_config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"dpo_loss": f"{total_loss / num_batches:.6f}", "skip": skipped})

            except Exception as exc:
                logging.error("DPO batch %d failed: %s", batch_idx, exc, exc_info=True)

        # Flush remaining accumulated gradients
        if num_batches > 0 and (num_batches % accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.lora_config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        progress_bar.close()
        avg = total_loss / max(num_batches, 1)
        logging.info(
            "DPO epoch=%d  avg_loss=%.6f  batches=%d  skipped=%d",
            epoch + 1, avg, num_batches, skipped,
        )
        return avg

    def _run_validation(self, epoch: int) -> Dict[str, Any]:
        return self.validator.run_comprehensive_validation(
            model=self.model, val_dataloader=self.val_dataloader, epoch=epoch
        )

    def _save_checkpoint(self, epoch: int, checkpoint_type: str):
        checkpoint_dir = self.config.get_training_output_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"lora_epoch{epoch + 1}_{checkpoint_type}.pt")
        trainable_state = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        torch.save(
            {
                "model_state": trainable_state,
                "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                "config": self.config,
                "symbol_mappings": {
                    "current_epoch_mappings": self.symbol_manager.get_symbols_for_epoch(epoch) if self.symbol_manager else {},
                    "original_labels": self.symbol_manager.original_labels if self.symbol_manager else [],
                    "symbol_type": self.symbol_manager.symbol_type if self.symbol_manager else "",
                    "dynamic_per_epoch": self.symbol_manager.dynamic_per_epoch if self.symbol_manager else False,
                },
                "dpo_config": {"beta": self.beta},
            },
            checkpoint_path,
        )
        logging.info("Saved DPO checkpoint: %s", os.path.basename(checkpoint_path))

    def run_complete_training(self) -> Dict[str, Any]:
        logging.info("Starting SymDPO training  beta=%.3f  epochs=%d", self.beta, self.config.lora_config.epochs)
        self._setup_lora_optimizer()
        self.optimizer.zero_grad(set_to_none=True)
        history = []

        if self.config.validate_before_training:
            logging.info("Baseline validation before DPO training (epoch 0)")
            baseline_scores = self._run_validation(epoch=-1)
            logging.info("Baseline: %s", baseline_scores)
            history.append({"epoch": 0, "train_loss": None, "validation": baseline_scores})

        for epoch in range(self.config.lora_config.epochs):
            self._epoch_log_count = 0
            logging.info("DPO Epoch %d/%d", epoch + 1, self.config.lora_config.epochs)
            epoch_loss = self._train_one_epoch(epoch)
            validation_scores = self._run_validation(epoch)
            self.scheduler.step()
            logging.info("Epoch %d  dpo_loss=%.6f", epoch + 1, epoch_loss)
            logging.info("Epoch %d  validation=%s", epoch + 1, validation_scores)
            history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "validation": validation_scores})

            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "periodic")

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
