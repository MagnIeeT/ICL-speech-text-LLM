"""Validation logic for Symbol Adapter training and inference."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from config.data_config.master_config import DatasetType, DATASET_CONFIGS
from config.train_config.training_configs import TrainingConfig, ValidationSymbolMode
from utils.evaluation_utils import evaluate_predictions

from .symbol_manager import SymbolManager

logger = logging.getLogger(__name__)


class ValidationManager:
    """Manages validation for training and inference with configurable symbol modes."""

    def __init__(self, config: TrainingConfig, symbol_manager: SymbolManager, tokenizer, max_val_samples: int = 200, processor=None):
        self.config = config
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        self.max_val_samples = max_val_samples
        self.processor = processor
        self.is_inference_mode = getattr(config, "inference_mode", False)

    def _resolve_validation_modes(self) -> List[Tuple[str, bool, bool]]:
        raw = getattr(self.config.symbol_config, "validation_modes", "fixed,original,fresh")
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
        mode_map = {ValidationSymbolMode.FIXED.value: ("fixed", False, False), ValidationSymbolMode.ORIGINAL.value: ("original", True, False), ValidationSymbolMode.FRESH.value: ("fresh", False, True)}
        return [mode_map[t] for t in tokens if t in mode_map]

    def validate_model(self, model, val_dataloader, epoch: int, use_original_labels: bool = False, use_dynamic_symbols: bool = False, symbol_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        model.eval()
        p_mappings, c_mappings = None, None
        mode_name = "Original"

        if not use_original_labels:
            if use_dynamic_symbols:
                p_mappings = c_mappings = self.symbol_manager._generate_symbol_mappings()
                mode_name = "Fresh-Symbols"
            else:
                p_mappings = c_mappings = symbol_mappings or self.symbol_manager.get_symbols_for_epoch(epoch)
                mode_name = "Fixed-Symbols"

        logger.info(f"--- Validation Mode: {mode_name} (Epoch {epoch + 1}) ---")
        try:
            with torch.no_grad():
                return self._run_validation_with_utils(model, val_dataloader, p_mappings, c_mappings, use_original_labels)
        except Exception as exc:
            import traceback
            logger.error(f"Validation failed for {mode_name}: {exc}\n{traceback.format_exc()}")
            return {}
        finally:
            model.train()

    def _run_validation_with_utils(self, model, val_dataloader, p_mappings, c_mappings, use_original=False) -> Dict[str, Any]:
        all_results = {}
        progress_bar = tqdm(val_dataloader, desc="Evaluating", total=len(val_dataloader), leave=False)
        router = getattr(model, "router", None)
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                dataset_types = batch.get("dataset_type", [])
                ds_name = dataset_types[0] if isinstance(dataset_types, list) and len(dataset_types) > 0 else "unknown"
                ds_name_str = ds_name.value if hasattr(ds_name, "value") else str(ds_name)
                
                relevant_labels = []
                for dt_enum, ds_cfg in DATASET_CONFIGS.items():
                    if dt_enum.value == ds_name_str:
                        relevant_labels = sorted(list(ds_cfg.valid_labels))
                        break
                if not relevant_labels: relevant_labels = self.symbol_manager.original_labels

                if use_original:
                    batch_p_map, batch_c_map = {}, {}
                elif router is not None and not use_dynamic_symbols:
                    import random
                    rng = random.Random(batch_idx)
                    slots = rng.sample(list(range(self.config.diff_symbol_config.num_slots)), k=min(len(relevant_labels), self.config.diff_symbol_config.num_slots))
                    vocab_indices, _ = router.get_slot_mappings(slots, hard=True)
                    batch_p_map = {label: f"<slot_{s}>" for label, s in zip(relevant_labels, slots)}
                    symbols = [self.tokenizer.decode([idx]).strip() for idx in vocab_indices]
                    batch_c_map = {label: sym for label, sym in zip(relevant_labels, symbols)}
                else:
                    batch_p_map, batch_c_map = p_mappings, c_mappings

                # 1. Rewrite text (Prompt part only)
                updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=batch_p_map, completion_mappings=batch_c_map)
                
                # 2. MODULAR TOKENIZATION: Call without completions for Validation/Inference
                if self.processor is not None:
                    # WE PASS completions=None so the model has to generate the answer!
                    tokenized_data = self.processor.tokenize_batch(updated_batch["prompt"], completions=None)
                    updated_batch.update(tokenized_data)
                
                # 3. Move tensors and generate
                updated_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in updated_batch.items()}
                
                dspo_module = getattr(model, "dspo_module", None)
                predictions = model.generate_output(updated_batch, router=router, dspo_module=dspo_module)

                for i, pred in enumerate(predictions):
                    dt_val = batch["dataset_type"][i]
                    dt_key = dt_val.value if hasattr(dt_val, "value") else str(dt_val)
                    all_results.setdefault(dt_key, [])
                    true_label = batch["completion"][i]
                    
                    conv_pred = self.symbol_manager.convert_symbols_back(pred, mappings=batch_c_map) if not use_original and batch_c_map else pred
                    all_results[dt_key].append({"text": batch["text"][i], "true_label": true_label, "predicted_label": conv_pred.strip(), "dataset_type": dt_key})
        finally:
            progress_bar.close()

        final = {}
        for ds, res in all_results.items():
            m = evaluate_predictions(res, DatasetType(ds))
            final[ds] = {"score": m.get("macro_f1_with_invalid", 0.0)}
        return final

    def run_comprehensive_validation(self, model, val_dataloader, epoch: int, symbol_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        results = {}
        for mode, use_orig, use_dyn in self._resolve_validation_modes():
            results[mode] = self.validate_model(model, val_dataloader, epoch, use_orig, use_dyn, symbol_mappings)
        self.log_validation_summary(results, epoch)
        return {"all_modes": results}

    def log_validation_summary(self, results, epoch):
        logger.info("=" * 60 + f"\n EPOCH {epoch + 1} VALIDATION SUMMARY\n" + "=" * 60)
        for mode, datasets in results.items():
            for ds, m in datasets.items(): logger.info(f"{ds:<15} | {mode:<10} | {m['score']:.4f}")
        logger.info("=" * 60)
