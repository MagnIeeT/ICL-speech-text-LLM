"""Validation logic for Symbol Adapter training and inference."""

import logging
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from config.data_config.master_config import DatasetType, get_dataset_config
from config.train_config.training_configs import (
    TrainingConfig,
    ValidationSymbolMode,
)

from utils.evaluation_utils import evaluate_predictions
from .symbol_manager import SymbolManager, get_dataset_info, tokenize_batch_with_audio

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
        symbol_map: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        model.eval()
        if use_original_labels:
            symbol_mappings_to_use = {}
            mode_name = "Original"
        elif use_dynamic_symbols:
            diff = self.symbol_manager.val_symbol_difficulty
            symbol_mappings_to_use = self.symbol_manager._generate_symbol_mappings(
                force=True, difficulty=diff
            )
            mode_name = f"Fresh-Symbols({diff})"
        else:
            # symbol_map comes pre-built from trainer (_build_current_symbol_map); fallback for non-trainer callers
            symbol_mappings_to_use = symbol_map if symbol_map is not None else self.symbol_manager.get_symbols_for_epoch(epoch)
            mode_name = "Fixed-Symbols"

        # Push symbol maps to val dataset so workers apply substitution + tokenize
        # (same pattern as training's _set_epoch_symbol_maps_on_dataset).
        # persistent_workers=False means each enumerate() forks fresh workers that
        # inherit the updated maps — no IPC needed.
        dataset = getattr(val_dataloader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'set_epoch_symbol_maps'):
            dataset.set_epoch_symbol_maps(symbol_mappings_to_use, no_symbols=use_original_labels)

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

    def _run_validation_with_utils(self, model, val_dataloader, symbol_mappings: Dict[str, Dict[str, str]], use_original_labels: bool = False):
        all_results = {}
        progress_bar = tqdm(val_dataloader, desc="Evaluating", total=len(val_dataloader), leave=False)
        logged_datasets: set = set()
        logged_prompts: set = set()

        try:
            for batch in progress_bar:
                ds_name_str, relevant_labels = get_dataset_info(batch, self.symbol_manager.original_labels)

                if use_original_labels:
                    c_map = {}
                else:
                    ds_map = symbol_mappings.get(ds_name_str, {})
                    c_map = {l: ds_map[l] for l in relevant_labels if l in ds_map}

                if ds_name_str not in logged_datasets:
                    logged_datasets.add(ds_name_str)
                    if use_original_labels:
                        logger.info("VAL [%s] original labels (no replacement)", ds_name_str)
                    elif c_map:
                        logger.info("VAL [%s] symbol mapping: %s", ds_name_str, c_map)
                    else:
                        logger.info("VAL [%s] no matching symbols — falling back to original labels", ds_name_str)

                # Workers applied symbol substitution + tokenization (left-pad via collate_fn).
                # Fall back to main-thread tokenization for processors without worker tokenization (e.g. Qwen).
                if "input_ids" not in batch:
                    updated_batch = tokenize_batch_with_audio(batch, self.processor, completions=None, padding_side="left")
                else:
                    updated_batch = batch

                # Log one full prompt per dataset per validation mode
                if ds_name_str not in logged_prompts and "input_ids" in updated_batch:
                    logged_prompts.add(ds_name_str)
                    full_prompt = self.tokenizer.decode(
                        updated_batch["input_ids"][0], skip_special_tokens=False
                    )
                    logger.info("VAL prompt [%s]:\n%s", ds_name_str, full_prompt)

                # Move to device and generate
                updated_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in updated_batch.items()}
                try:
                    ds_cfg = get_dataset_config(DatasetType(ds_name_str))
                    raw = model.generate_output(updated_batch, max_new_tokens=ds_cfg.max_new_tokens)
                except Exception as gen_exc:
                    logger.warning("Skipping batch (generate_output failed): %s", gen_exc)
                    continue
                # Qwen returns (texts, token_log_probs); Flamingo/Salmonn return List[str]
                if isinstance(raw, tuple):
                    predictions, token_log_probs = raw
                else:
                    predictions, token_log_probs = raw, [None] * len(raw)

                for i, pred in enumerate(predictions):
                    dt_val = batch["dataset_type"][i]
                    dt_key = dt_val.value if hasattr(dt_val, "value") else str(dt_val)
                    all_results.setdefault(dt_key, [])
                    true_label = batch["completion"][i]
                    if not use_original_labels and c_map:
                        true_label = self.symbol_manager.convert_symbols_back(true_label, mappings=c_map)
                    conv_pred = self.symbol_manager.convert_symbols_back(pred, mappings=c_map) if not use_original_labels and c_map else pred
                    lps = token_log_probs[i] if token_log_probs[i] is not None else []
                    all_results[dt_key].append({
                        "text": batch["text"][i],
                        "true_label": true_label,
                        "raw_prediction": pred,
                        "token_log_probs": lps,
                        "symbol_log_prob": round(sum(lps), 4) if lps else None,
                        "predicted_label": conv_pred.strip(),
                        "dataset_type": dt_key,
                    })
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

    def run_comprehensive_validation(self, model, val_dataloader, epoch: int, symbol_map: Optional[Dict[str, Dict[str, str]]] = None):
        mode_defs = self._resolve_validation_modes()
        comprehensive_results = {}
        for mode_suffix, use_original, use_dynamic in mode_defs:
            metrics_by_dataset = self.validate_model(model, val_dataloader, epoch, use_original, use_dynamic, symbol_map)
            comprehensive_results[mode_suffix] = metrics_by_dataset

        self.log_validation_summary(comprehensive_results, epoch)
        
        # Per-mode breakdown log + compute avg_score from the primary (first) mode.
        # avg_score = average across datasets for the primary mode so that with
        # a single dataset it equals that dataset's score directly.
        primary_mode = mode_defs[0][0] if mode_defs else None
        combined_metric = 0.0
        metric_string = ""
        for mode_suffix, mode_results in comprehensive_results.items():
            mode_scores = {ds: m.get("score", 0.0) for ds, m in mode_results.items()}
            mode_avg = sum(mode_scores.values()) / len(mode_scores) if mode_scores else 0.0
            logger.info("📊 [%s] per-dataset: %s  avg=%.4f", mode_suffix, mode_scores, mode_avg)
            if mode_suffix == primary_mode:
                combined_metric = mode_avg
                metric_string = "|".join([f"{k}:{v:.4f}" for k, v in mode_scores.items()])

        logger.info("📊 Primary mode (%s) avg: %.4f", primary_mode, combined_metric)
        logger.info("📊 Composite string: %s", metric_string)

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
