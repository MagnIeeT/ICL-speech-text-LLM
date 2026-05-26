"""Validation logic for Symbol Adapter training and inference."""

import logging
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from config.data_config.master_config import DatasetType
from config.train_config.training_configs import (
    TrainingConfig,
    ValidationSymbolMode,
)

from utils.evaluation_utils import evaluate_predictions
from .symbol_manager import SymbolManager

logger = logging.getLogger(__name__)


class ValidationManager:
    """Manages validation for training and inference."""

    def __init__(
        self,
        config: TrainingConfig,
        symbol_manager: SymbolManager,
        tokenizer,
        processor=None,  # ✅ FIX 1: Add processor to __init__ so we can tokenize
        max_val_samples: int = 200,
    ):

        self.config = config
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_val_samples = max_val_samples

        self.is_inference_mode = getattr(
            config,
            "inference_mode",
            False,
        )

    def _resolve_validation_modes(self):

        raw = getattr(
            self.config.symbol_config,
            "validation_modes",
            "fixed,original,fresh",
        )

        tokens = [
            token.strip().lower()
            for token in raw.split(",")
            if token.strip()
        ]

        valid = {
            ValidationSymbolMode.FIXED.value,
            ValidationSymbolMode.ORIGINAL.value,
            ValidationSymbolMode.FRESH.value,
        }

        ordered_unique = []

        for token in tokens:
            if token in valid and token not in ordered_unique:
                ordered_unique.append(token)

        if not ordered_unique:
            ordered_unique = [ValidationSymbolMode.FIXED.value]

        mode_map = {
            ValidationSymbolMode.FIXED.value: (
                "fixed",
                False,
                False,
            ),
            ValidationSymbolMode.ORIGINAL.value: (
                "original",
                True,
                False,
            ),
            ValidationSymbolMode.FRESH.value: (
                "fresh",
                False,
                True,
            ),
        }

        return [mode_map[token] for token in ordered_unique]

    # ✅ FIX 2: Add tokenization logic to convert the raw strings/dicts into tensors for inference
    def _tokenize_raw_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        prompts = raw_batch.get("prompt", [])
        completions = raw_batch.get("completion", [])
        audios = raw_batch.get("audio", [])
        texts = raw_batch.get("text", [])
        dataset_types = raw_batch.get("dataset_type", [])
        input_modes = raw_batch.get("input_mode", [])
        fewshot_modes = raw_batch.get("fewshot_mode", [])
        examples_audios = raw_batch.get("examples_audio", [])

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
            # Crucially: is_training=False for validation!
            inputs = self.processor.process_inputs(item, is_training=False)
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

    def validate_model(
        self,
        model,
        val_dataloader,
        epoch: int,
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
            symbol_mappings_to_use = (
                symbol_mappings
                if symbol_mappings is not None
                else self.symbol_manager.get_symbols_for_epoch(epoch)
            )
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
    ):

        all_results = {}

        progress_bar = tqdm(
            val_dataloader,
            desc="Evaluating",
            total=len(val_dataloader),
            leave=False,
        )

        try:
            for batch in progress_bar:

                if use_original_labels:
                    updated_batch = batch
                else:
                    updated_batch = self.symbol_manager.replace_symbols_in_batch(
                        batch,
                        mappings=symbol_mappings,
                    )

                # ✅ FIX 3: Tokenize the batch BEFORE passing it to generate_output
                tokenized_batch = self._tokenize_raw_batch(updated_batch)

                predictions = model.generate_output(tokenized_batch)

                for i, pred in enumerate(predictions):

                    dt = (
                        batch["dataset_type"][i]
                        if isinstance(batch["dataset_type"], list)
                        else batch["dataset_type"]
                    )

                    dt_key = dt.value if hasattr(dt, "value") else str(dt)

                    if dt_key not in all_results:
                        all_results[dt_key] = []

                    true_label = (
                        batch["completion"][i]
                        if isinstance(batch["completion"], list)
                        else batch["completion"]
                    )

                    converted_pred = pred

                    if not use_original_labels and symbol_mappings:
                        converted_pred = self.symbol_manager.convert_symbols_back(
                            pred,
                            mappings=symbol_mappings,
                        )

                    result = {
                        "text": (
                            batch["text"][i]
                            if isinstance(batch["text"], list)
                            else batch["text"]
                        ),
                        "true_label": true_label,
                        "predicted_label": converted_pred.strip(),
                        "dataset_type": dt_key,
                    }

                    all_results[dt_key].append(result)

        finally:
            progress_bar.close()

        final_metrics = {}

        for dataset_name, dt_results in all_results.items():

            try:
                dataset_type = DatasetType(dataset_name)

                dt_metrics = evaluate_predictions(
                    dt_results,
                    dataset_type,
                )

                logger.info("")
                logger.info("=" * 80)
                logger.info(f"Metrics for {dataset_name}")
                logger.info("=" * 80)

                for metric_key, metric_value in dt_metrics.items():
                    logger.info(f"  {metric_key}: {metric_value}")

                logger.info("")
                logger.info("Example predictions after cleaning:")

                for sample in dt_results[:5]:
                    logger.info(f"Original: {sample['predicted_label']}")
                    logger.info(f"Cleaned: {sample['predicted_label']}")
                    logger.info(f"True: {sample['true_label']}")
                    logger.info("-" * 50)

                score = dt_metrics.get("macro_f1_with_invalid")
                if score is None:
                    score = dt_metrics.get("accuracy", 0.0)

                final_metrics[dataset_name] = {
                    "score": score,
                    "detailed": dt_metrics,
                    "predictions": (dt_results if self.is_inference_mode else None),
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
    ):

        mode_defs = self._resolve_validation_modes()
        comprehensive_results: Dict[str, Any] = {}

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

        self.log_validation_summary(
            comprehensive_results,
            epoch,
        )

        fixed_mode_results = comprehensive_results.get("fixed", {})
        dataset_metrics = {
            ds_name: metrics.get("score", 0.0)
            for ds_name, metrics in fixed_mode_results.items()
        }

        combined_metric = (
            sum(dataset_metrics.values()) / len(dataset_metrics)
            if dataset_metrics else 0.0
        )

        metric_string = "|".join(
            [f"{k}:{v:.4f}" for k, v in dataset_metrics.items()]
        )

        logger.info(f"📊 Dataset metrics (fixed mode): {dataset_metrics}")
        logger.info(f"📊 Combined metric (fixed mode): {combined_metric:.4f}")
        logger.info(f"📊 Composite string (fixed mode): {metric_string}")

        return {
            "avg_score": combined_metric,
            "all_modes": comprehensive_results,
        }

    def log_validation_summary(
        self,
        comprehensive_results: Dict[str, Any],
        epoch: int,
    ):

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