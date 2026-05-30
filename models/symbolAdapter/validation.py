"""Validation logic for Symbol Adapter training and inference."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from config.data_config.master_config import DatasetType, DATASET_CONFIGS
from config.train_config.training_configs import (
    TrainingConfig,
    ValidationSymbolMode,
)

from utils.evaluation_utils import evaluate_predictions
from .symbol_manager import SymbolManager

logger = logging.getLogger(__name__)


class ValidationManager:
    """Manages validation for training and inference with robust remote logic."""

    def __init__(
        self,
        config: TrainingConfig,
        symbol_manager: SymbolManager,
        tokenizer,
        processor=None,
        max_val_samples: int = 200,
    ):
        self.config = config
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_val_samples = max_val_samples
        self.is_inference_mode = getattr(config, "inference_mode", False)

    def _resolve_validation_modes(self):
        """Remote version of mode resolution."""
        raw = getattr(self.config.symbol_config, "validation_modes", "fixed,original,fresh")
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
        valid = {ValidationSymbolMode.FIXED.value, ValidationSymbolMode.ORIGINAL.value, ValidationSymbolMode.FRESH.value}
        ordered_unique = []
        for token in tokens:
            if token in valid and token not in ordered_unique:
                ordered_unique.append(token)
        if not ordered_unique:
            ordered_unique = [ValidationSymbolMode.FIXED.value]
        mode_map = {
            ValidationSymbolMode.FIXED.value: ("fixed", False, False),
            ValidationSymbolMode.ORIGINAL.value: ("original", True, False),
            ValidationSymbolMode.FRESH.value: ("fresh", False, True),
        }
        return [mode_map[token] for token in ordered_unique]

    def validate_model(
        self, model, val_dataloader, epoch: int,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False,
        symbol_mappings: Optional[Dict[str, str]] = None,
    ):
        model.eval()
        if use_original_labels:
            symbol_mappings_to_use = {}
            mode_name = "Original"
        elif use_dynamic_symbols:
            symbol_mappings_to_use = self.symbol_manager._generate_symbol_mappings()
            mode_name = "Fresh-Symbols"
        else:
            symbol_mappings_to_use = symbol_mappings if symbol_mappings is not None else self.symbol_manager.get_symbols_for_epoch(epoch)
            mode_name = "Fixed-Symbols"

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"VALIDATION MODE: {mode_name}")
        logger.info("=" * 80)

        try:
            with torch.no_grad():
                metrics_by_dataset = self._run_validation_with_utils(
                    model=model,
                    val_dataloader=val_dataloader,
                    symbol_mappings=symbol_mappings_to_use,
                    use_original_labels=use_original_labels,
                )
            return metrics_by_dataset
        except Exception as exc:
            import traceback
            logger.error(f"Validation failed for {mode_name}: {exc}\n{traceback.format_exc()}")
            return {}
        finally:
            model.train()

    def _run_validation_with_utils(self, model, val_dataloader, symbol_mappings: Dict[str, str], use_original_labels: bool = False):
        all_results = {}
        progress_bar = tqdm(val_dataloader, desc="Evaluating", total=len(val_dataloader), leave=False)
        router = getattr(model, "router", None)
        
        try:
            for batch in progress_bar:
                dataset_types = batch.get("dataset_type", [])
                ds_name = dataset_types[0] if isinstance(dataset_types, list) and len(dataset_types) > 0 else "unknown"
                ds_name_str = ds_name.value if hasattr(ds_name, "value") else str(ds_name)
                
                relevant_labels = []
                for dt_enum, ds_cfg in DATASET_CONFIGS.items():
                    if dt_enum.value == ds_name_str:
                        relevant_labels = sorted(list(ds_cfg.valid_labels))
                        break
                if not relevant_labels: relevant_labels = self.symbol_manager.original_labels

                # Determine D-SPO Mappings
                if use_original_labels:
                    p_map, c_map = {}, {}
                elif router is not None and not use_original_labels:
                    import random
                    # Seed with a deterministic value for this sample to ensure consistent slots
                    sample_seed = sum(batch.get("input_ids", [torch.tensor([0])])[0].tolist()) if "input_ids" in batch else 0
                    rng = random.Random(sample_seed)
                    slots = rng.sample(list(range(self.config.diff_symbol_config.num_slots)), k=min(len(relevant_labels), self.config.diff_symbol_config.num_slots))
                    vocab_indices, _ = router.get_slot_mappings(slots, hard=True)
                    p_map = {label: f"<slot_{s}>" for label, s in zip(relevant_labels, slots)}
                    symbols = [self.tokenizer.decode([idx]).strip() for idx in vocab_indices]
                    c_map = {label: sym for label, sym in zip(relevant_labels, symbols)}
                else:
                    p_map = c_map = symbol_mappings

                # 1. Text Replacement
                updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, prompt_mappings=p_map, completion_mappings=c_map)

                # 2. MODULAR TOKENIZATION (Our Fix)
                if self.processor is not None:
                    tokenized_data = self.processor.tokenize_batch(updated_batch["prompt"], completions=None)
                    updated_batch.update(tokenized_data)

                # 3. Move and Generate (D-SPO Aware)
                updated_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in updated_batch.items()}
                dspo_module = getattr(model, "dspo_module", None)
                predictions = model.generate_output(updated_batch, router=router, dspo_module=dspo_module)

                for i, pred in enumerate(predictions):
                    dt_val = batch["dataset_type"][i]
                    dt_key = dt_val.value if hasattr(dt_val, "value") else str(dt_val)
                    all_results.setdefault(dt_key, [])
                    true_label = batch["completion"][i]
                    conv_pred = self.symbol_manager.convert_symbols_back(pred, mappings=c_map) if not use_original_labels and c_map else pred
                    all_results[dt_key].append({"text": batch["text"][i], "true_label": true_label, "predicted_label": conv_pred.strip(), "dataset_type": dt_key})
        finally:
            progress_bar.close()

        final_metrics = {}
        for ds_name, dt_results in all_results.items():
            try:
                dataset_type = DatasetType(ds_name)
                dt_metrics = evaluate_predictions(dt_results, dataset_type)
                
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"Metrics for {ds_name}")
                logger.info("=" * 80)
                for mk, mv in dt_metrics.items(): logger.info(f"  {mk}: {mv}")
                
                logger.info("")
                logger.info("Example predictions after cleaning:")
                for sample in dt_results[:5]:
                    logger.info(f"Original: {sample['predicted_label']}")
                    logger.info(f"True: {sample['true_label']}")
                    logger.info("-" * 50)
                
                score = dt_metrics.get("macro_f1_with_invalid", dt_metrics.get("accuracy", 0.0))
                final_metrics[ds_name] = {"score": score, "detailed": dt_metrics, "predictions": (dt_results if self.is_inference_mode else None)}
            except Exception as exc:
                logger.error(f"Error evaluating {ds_name}: {exc}")
                final_metrics[ds_name] = {"score": 0.0}
        return final_metrics

    def run_comprehensive_validation(self, model, val_dataloader, epoch: int, symbol_mappings: Optional[Dict[str, str]] = None):
        mode_defs = self._resolve_validation_modes()
        comprehensive_results = {}
        for mode_suffix, use_original, use_dynamic in mode_defs:
            metrics_by_dataset = self.validate_model(model, val_dataloader, epoch, use_original, use_dynamic, symbol_mappings)
            comprehensive_results[mode_suffix] = metrics_by_dataset

        self.log_validation_summary(comprehensive_results, epoch)
        
        fixed_mode_results = comprehensive_results.get("fixed", {})
        dataset_metrics = {ds_name: metrics.get("score", 0.0) for ds_name, metrics in fixed_mode_results.items()}
        combined_metric = sum(dataset_metrics.values()) / len(dataset_metrics) if dataset_metrics else 0.0
        metric_string = "|".join([f"{k}:{v:.4f}" for k, v in dataset_metrics.items()])

        logger.info(f"📊 Dataset metrics (fixed mode): {dataset_metrics}")
        logger.info(f"📊 Combined metric (fixed mode): {combined_metric:.4f}")
        logger.info(f"📊 Composite string (fixed mode): {metric_string}")

        return {"avg_score": combined_metric, "all_modes": comprehensive_results}

    def log_validation_summary(self, comprehensive_results, epoch):
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL VALIDATION RESULTS")
        logger.info("=" * 80)
        for mode, datasets in comprehensive_results.items():
            logger.info("")
            logger.info(f"Validation Mode: {mode}")
            logger.info("-" * 80)
            for ds_name, metrics in datasets.items():
                score = metrics.get("score", 0.0)
                logger.info(f"{ds_name:<20} : {score:.4f}")
        logger.info("=" * 80)
