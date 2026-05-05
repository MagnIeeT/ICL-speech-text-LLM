"""Validation logic for Symbol Adapter training and inference."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from config.data_config.master_config import DatasetType
from config.train_config.training_configs import TrainingConfig, ValidationSymbolMode
from utils.evaluation_utils import clean_prediction, evaluate_predictions

from .symbol_manager import SymbolManager

logger = logging.getLogger(__name__)


class ValidationManager:
    """Manages validation for training and inference with configurable symbol modes."""

    def __init__(
        self,
        config: TrainingConfig,
        symbol_manager: SymbolManager,
        tokenizer,
        max_val_samples: int = 200,
    ):
        self.config = config
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        self.max_val_samples = max_val_samples
        self.is_inference_mode = getattr(config, "inference_mode", False)

    def _resolve_validation_modes(self) -> List[Tuple[str, bool, bool]]:
        """
        Resolve validation modes from config.
        Returns list of tuples: (mode_key_suffix, use_original_labels, use_dynamic_symbols)
        """
        raw = getattr(self.config.symbol_config, "validation_modes", "fixed,original,fresh")
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]

        valid = {
            ValidationSymbolMode.FIXED.value,
            ValidationSymbolMode.ORIGINAL.value,
            ValidationSymbolMode.FRESH.value,
        }

        is_baseline = getattr(self.config.symbol_config, "no_symbols", False) and not getattr(self.config.symbol_config, "swap_labels", False)

        ordered_unique: List[str] = []
        for token in tokens:
            if token in valid:
                effective_token = token
                # For baseline models, 'fixed' (training mappings) is the same as 'original' (no mappings)
                if is_baseline and token == ValidationSymbolMode.FIXED.value:
                    effective_token = ValidationSymbolMode.ORIGINAL.value
                
                if effective_token not in ordered_unique:
                    ordered_unique.append(effective_token)

        if not ordered_unique:
            ordered_unique = [ValidationSymbolMode.FIXED.value]

        mode_map = {
            ValidationSymbolMode.FIXED.value: ("fixed", False, False),
            ValidationSymbolMode.ORIGINAL.value: ("original", True, False),
            ValidationSymbolMode.FRESH.value: ("fresh", False, True),
        }
        return [mode_map[token] for token in ordered_unique]

    def validate_model(
        self,
        model,
        val_dataloader,
        epoch: int,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False,
        symbol_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run one validation mode and return metrics."""
        model.eval()

        if use_original_labels:
            symbol_mappings_to_use = {}
            mode_name = "Original"
        elif use_dynamic_symbols:
            symbol_mappings_to_use = self.symbol_manager._generate_symbol_mappings()
            mode_name = "Fresh-Symbols"
        else:
            if self.is_inference_mode and symbol_mappings is not None:
                symbol_mappings_to_use = symbol_mappings
            else:
                symbol_mappings_to_use = self.symbol_manager.get_symbols_for_epoch(epoch)
            mode_name = "Fixed-Symbols"

        logger.info(f"--- Validation Mode: {mode_name} (Epoch {epoch + 1}) ---")

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
            logger.error(f"Validation failed for {mode_name}: {exc}")
            return {}
        finally:
            model.train()

    def _run_validation_with_utils(
        self,
        model,
        val_dataloader,
        symbol_mappings: Dict[str, str],
        use_original_labels: bool = False,
    ) -> Dict[str, Any]:
        """Run validation using evaluation helpers and return per-dataset metrics."""
        all_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Determine datasets to validate
        if not self.is_inference_mode:
            val_dataset_type_str = self.config.data_config.val_dataset_type
            dataset_names_val = val_dataset_type_str.split("-") if "-" in val_dataset_type_str else [val_dataset_type_str]
        else:
            # In inference mode, use whatever is in the dataloader's items
            dataset_names_val = [] # Will be populated dynamically

        progress_bar = tqdm(val_dataloader, desc="Evaluating", total=len(val_dataloader), leave=False)

        try:
            for batch_idx, batch in enumerate(progress_bar):
                if use_original_labels:
                    updated_batch = batch
                else:
                    updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, mappings=symbol_mappings)

                # Expect model backend to implement generate_output
                predictions = model.generate_output(updated_batch)

                for i, pred in enumerate(predictions):
                    dt = batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"]
                    dt_key = dt.value if hasattr(dt, "value") else str(dt)
                    
                    if dt_key not in all_results:
                        all_results[dt_key] = []
                    
                    true_label = batch["completion"][i] if isinstance(batch["completion"], list) else batch["completion"]

                    converted_pred = pred
                    if not use_original_labels and symbol_mappings:
                        converted_pred = self.symbol_manager.convert_symbols_back(pred, mappings=symbol_mappings)

                    result = {
                        "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                        "true_label": true_label,
                        "predicted_label": converted_pred.strip(),
                        "dataset_type": dt_key,
                    }
                    all_results[dt_key].append(result)
        finally:
            progress_bar.close()

        # Compute metrics per dataset
        final_metrics = {}
        for dataset_name, dt_results in all_results.items():
            try:
                dataset_type = DatasetType(dataset_name)
                dt_metrics = evaluate_predictions(dt_results, dataset_type)
                
                # Prioritize macro_f1_with_invalid as requested
                score = dt_metrics.get("macro_f1_with_invalid")
                if score is None:
                    score = dt_metrics.get("accuracy", 0.0)
                
                final_metrics[dataset_name] = {
                    "score": score,
                    "detailed": dt_metrics,
                    "predictions": dt_results if self.is_inference_mode else None
                }
            except Exception as exc:
                logger.error(f"Error evaluating {dataset_name}: {exc}")
                final_metrics[dataset_name] = {"score": 0.0}

        return final_metrics

    def run_comprehensive_validation(
        self,
        model,
        val_dataloader,
        epoch: int,
        symbol_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run all configured validation modes."""
        mode_defs = self._resolve_validation_modes()
        comprehensive_results = {}

        for mode_suffix, use_original, use_dynamic in mode_defs:
            metrics_by_dataset = self.validate_model(
                model=model,
                val_dataloader=val_dataloader,
                epoch=epoch,
                use_original_labels=use_original,
                use_dynamic_symbols=use_dynamic,
                symbol_mappings=symbol_mappings,
            )
            comprehensive_results[mode_suffix] = metrics_by_dataset

        self.log_validation_summary(comprehensive_results, epoch)
        
        # For training history compatibility, return a simple summary score
        # Typically the average of the primary mode's scores
        primary_mode = mode_defs[0][0]
        primary_scores = [m["score"] for m in comprehensive_results.get(primary_mode, {}).values()]
        avg_score = sum(primary_scores) / len(primary_scores) if primary_scores else 0.0
        
        return {
            "avg_score": avg_score,
            "all_modes": comprehensive_results
        }

    def log_validation_summary(self, comprehensive_results: Dict[str, Any], epoch: int):
        """Simple, readable logging of results."""
        logger.info("=" * 60)
        logger.info(f" EPOCH {epoch + 1} VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        header = f"{'Dataset':<15} | {'Mode':<10} | {'Score (F1/Acc)':<12}"
        logger.info(header)
        logger.info("-" * 60)
        
        for mode, datasets in comprehensive_results.items():
            for ds_name, metrics in datasets.items():
                score = metrics.get("score", 0.0)
                logger.info(f"{ds_name:<15} | {mode:<10} | {score:.4f}")
        
        logger.info("=" * 60)
