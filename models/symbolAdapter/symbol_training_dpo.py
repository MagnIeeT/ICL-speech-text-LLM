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
from .symbol_manager import SymbolManager
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

        # Detect Flamingo processor once — same duck-typed check as orchestrator.
        # FlamingoProcessor.tokenize_batch accepts original_audios; Qwen does not.
        self._is_flamingo_processor = self._detect_flamingo_processor()

        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            processor=self.processor,
            max_val_samples=config.data_config.val_max_samples,
        )
        self._setup_training_environment()

    # ------------------------------------------------------------------
    # Flamingo detection + audio recovery (mirrors SymbolTrainingOrchestrator)
    # ------------------------------------------------------------------

    def _detect_flamingo_processor(self) -> bool:
        if self.processor is None:
            return False
        import inspect
        sig = inspect.signature(self.processor.tokenize_batch)
        return "original_audios" in sig.parameters

    def _extract_audio_from_batch(self, batch: Dict[str, Any]) -> List[Any]:
        """
        Build a per-item list of raw audio arrays for use as the
        `original_audios` fallback in FlamingoProcessor.tokenize_batch().
        """
        if "audio" in batch and isinstance(batch["audio"], (list, tuple)):
            audios = list(batch["audio"])
            if any(a is not None for a in audios):
                return audios
        prompts = batch.get("prompt", [])
        return [
            p.get("audio") if isinstance(p, dict) else None
            for p in prompts
        ]

    def _tokenize_for_flamingo_or_qwen(
        self,
        prompts: List[Any],
        completions: List[str],
        original_audios: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Unified tokenize_batch call that passes original_audios for Flamingo
        and omits it for Qwen (which ignores it via **kwargs anyway).
        """
        if self._is_flamingo_processor:
            return self.processor.tokenize_batch(
                prompts,
                completions,
                original_audios=original_audios,
            )
        return self.processor.tokenize_batch(prompts, completions, original_audios=original_audios)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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

    def _get_relevant_labels(self, batch: Dict[str, Any]) -> List[str]:
        dataset_types = batch.get("dataset_type", [])
        ds_name = dataset_types[0] if isinstance(dataset_types, list) and dataset_types else "unknown"
        ds_name_str = ds_name.value if hasattr(ds_name, "value") else str(ds_name)
        for dt_enum, ds_cfg in DATASET_CONFIGS.items():
            if dt_enum.value == ds_name_str:
                return sorted(list(ds_cfg.valid_labels))
        return self.symbol_manager.original_labels if self.symbol_manager else []

    def _determine_mappings(self, batch: Dict[str, Any], epoch: int, batch_idx: int) -> Tuple[Dict, Dict]:
        """Return (prompt_mappings, completion_mappings) for this batch."""
        relevant_labels = self._get_relevant_labels(batch)
        dataset_types = batch.get("dataset_type", [])
        ds_name = dataset_types[0] if isinstance(dataset_types, list) and dataset_types else "unknown"
        ds_name_str = ds_name.value if hasattr(ds_name, "value") else str(ds_name)
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
            force_new = per_instance or batch_idx == 0
            mapping = self.symbol_manager.get_symbols_for_epoch(epoch, force_new_symbols=force_new)
            return mapping, mapping

    def _sample_rejected_symbol(self, chosen_symbol: str, c_mappings: Dict[str, str]) -> Optional[str]:
        """Pick a random symbol from the current mapping that is not the chosen symbol."""
        alternatives = [s for s in c_mappings.values() if s != chosen_symbol]
        if not alternatives:
            return None
        return random.choice(alternatives)

    def _get_completion_logp(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the sum of log probs of completion tokens.

        Uses the autoregressive shift: logit at position t predicts token at t+1,
        so completion tokens [prompt_len .. T-1] are predicted by logits [prompt_len-1 .. T-2].
        """
        outputs = self.model(batch)
        logits = outputs["logits"]  # [B, T, V]

        if logits is None:
            return torch.tensor(0.0, device=self.config.device, requires_grad=True)

        input_ids = batch["input_ids"]  # [B, T]
        prompt_len = int(batch["prompt_length"][0])
        ids = input_ids[0]   # [T]
        lgt = logits[0]      # [T, V]
        T = ids.shape[0]

        comp_ids    = ids[prompt_len:]
        comp_logits = lgt[prompt_len - 1: T - 1, :]

        if comp_ids.shape[0] == 0 or comp_logits.shape[0] == 0:
            return torch.tensor(0.0, device=self.config.device, requires_grad=True)

        log_probs = F.log_softmax(comp_logits.float(), dim=-1)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        valid_mask = (comp_ids != pad_id) & (comp_ids != -100)

        if not valid_mask.any():
            return torch.tensor(0.0, device=self.config.device, requires_grad=True)

        vocab_size = log_probs.shape[-1]
        safe_ids = comp_ids.clone()
        safe_ids[~valid_mask] = 0
        safe_ids = safe_ids.clamp(0, vocab_size - 1)

        token_log_probs = log_probs.gather(1, safe_ids.unsqueeze(1)).squeeze(1)
        return token_log_probs[valid_mask].sum()

    def _compute_dpo_loss(self, chosen_batch: Dict[str, Any], rejected_batch: Dict[str, Any]) -> torch.Tensor:
        """
        4 forward passes:
          policy (LoRA on)  × chosen   → policy_chosen_logp   [grad]
          policy (LoRA on)  × rejected → policy_rejected_logp  [grad]
          ref    (LoRA off) × chosen   → ref_chosen_logp       [no grad]
          ref    (LoRA off) × rejected → ref_rejected_logp     [no grad]
        """
        policy_chosen_logp   = self._get_completion_logp(chosen_batch)
        policy_rejected_logp = self._get_completion_logp(rejected_batch)

        self.model.model.disable_adapter_layers()
        with torch.no_grad():
            ref_chosen_logp   = self._get_completion_logp(chosen_batch).detach()
            ref_rejected_logp = self._get_completion_logp(rejected_batch).detach()
        self.model.model.enable_adapter_layers()

        chosen_ratio   = policy_chosen_logp   - ref_chosen_logp
        rejected_ratio = policy_rejected_logp - ref_rejected_logp
        return -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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

                # Symbol-replace prompt and completion text (no tokenization yet).
                text_batch = self.symbol_manager.replace_symbols_in_batch(
                    raw_batch, prompt_mappings=p_mappings, completion_mappings=c_mappings
                )

                chosen_completion = text_batch["completion"]  # list[str], correct symbol(s)
                chosen_sym = chosen_completion[0] if chosen_completion else ""

                rejected_sym = self._sample_rejected_symbol(chosen_sym, c_mappings)
                if rejected_sym is None:
                    skipped += 1
                    continue

                rejected_completion = [rejected_sym] * len(chosen_completion)

                if self._epoch_log_count < 2:
                    original_label = raw_batch["completion"][0] if raw_batch.get("completion") else "?"
                    logging.info("=" * 60)
                    logging.info(
                        "DPO train epoch=%d batch=%d  label=%r  chosen=%r  rejected=%r  mapping=%s",
                        epoch + 1, batch_idx, original_label, chosen_sym, rejected_sym, c_mappings,
                    )
                    logging.info("=" * 60)
                    self._epoch_log_count += 1

                # ----------------------------------------------------------
                # Tokenization — Flamingo needs original_audios fallback.
                # Both chosen and rejected share the same prompt (and thus
                # the same audio), so we extract audio once and reuse it.
                # ----------------------------------------------------------
                original_audios = self._extract_audio_from_batch(raw_batch) if self._is_flamingo_processor else None

                chosen_tok   = self._tokenize_for_flamingo_or_qwen(
                    text_batch["prompt"], chosen_completion, original_audios
                )
                rejected_tok = self._tokenize_for_flamingo_or_qwen(
                    text_batch["prompt"], rejected_completion, original_audios
                )

                # Merge: audio features from text_batch, token ids from each tok
                chosen_batch   = {**text_batch, **chosen_tok}
                rejected_batch = {**text_batch, **rejected_tok}

                chosen_batch   = self._move_batch_to_device(chosen_batch)
                rejected_batch = self._move_batch_to_device(rejected_batch)

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

    # ------------------------------------------------------------------
    # Validation / checkpointing
    # ------------------------------------------------------------------

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
            logging.info("Epoch %d  dpo_loss=%.6f", epoch + 1, epoch_loss)
            logging.info("Epoch %d  validation=%s", epoch + 1, validation_scores)
            history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "validation": validation_scores})

            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, "periodic")

        self._save_checkpoint(self.config.lora_config.epochs - 1, "final")
        return {"history": history} 