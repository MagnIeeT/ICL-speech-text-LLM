"""Validation logic for Symbol Adapter training and inference."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from config.data_config.master_config import DatasetType
from config.train_config.training_configs import TrainingConfig, ValidationSymbolMode
from utils.evaluation_utils import clean_prediction, evaluate_predictions

from .symbol_manager import SymbolManager


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
        if getattr(self.config.symbol_config, "no_symbols", False) and not getattr(self.config.symbol_config, "swap_labels", False):
            return [("original", True, False)]

        raw = getattr(self.config.symbol_config, "validation_modes", "fixed,original,fresh")
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]

        aliases = {
            "both": ["fixed", "original"],
            "all": ["fixed", "original", "fresh"],
            "new": ["fresh"],
            "symbols": ["fixed"],
        }

        expanded: List[str] = []
        for token in tokens:
            expanded.extend(aliases.get(token, [token]))

        valid = {
            ValidationSymbolMode.FIXED.value,
            ValidationSymbolMode.ORIGINAL.value,
            ValidationSymbolMode.FRESH.value,
        }

        ordered_unique: List[str] = []
        for token in expanded:
            if token in valid and token not in ordered_unique:
                ordered_unique.append(token)

        if not ordered_unique:
            ordered_unique = [ValidationSymbolMode.FIXED.value]

        mode_map = {
            ValidationSymbolMode.FIXED.value: ("symbols", False, False),
            ValidationSymbolMode.ORIGINAL.value: ("original", True, False),
            ValidationSymbolMode.FRESH.value: ("fresh", False, True),
        }
        return [mode_map[token] for token in ordered_unique]

    def validate_model(
        self,
        model,
        val_dataloader,
        epoch: int,
        phase: str,
        cycle: int = 0,
        bypass_mlp: bool = False,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False,
        symbol_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
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

        mlp_mode = "NoMLP" if bypass_mlp else "MLP"
        full_mode_name = f"{mlp_mode}+{mode_name}"
        logging.info("=== Validation: %s (Epoch %s, %s) ===", full_mode_name, epoch, phase.upper())

        try:
            with torch.no_grad():
                metrics = self._run_validation_with_utils(
                    model=model,
                    val_dataloader=val_dataloader,
                    symbol_mappings=symbol_mappings_to_use,
                    mode_name=full_mode_name,
                    epoch=epoch,
                    phase=phase,
                    use_original_labels=use_original_labels,
                    use_dynamic_symbols=use_dynamic_symbols,
                )

            logging.info("✓ %s Validation Score: %.4f", full_mode_name, metrics["accuracy"])
            return metrics
        except Exception as exc:
            logging.error("Validation failed for %s: %s", full_mode_name, exc)
            return {"accuracy": 0.0, "loss": float("inf"), "total_samples": 0}
        finally:
            model.train()

    def _run_validation_with_utils(
        self,
        model,
        val_dataloader,
        symbol_mappings: Dict[str, str],
        mode_name: str,
        epoch: int,
        phase: str,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False,
    ) -> Dict[str, float]:
        """Run validation using evaluation helpers."""
        all_results: Dict[str, List[Dict[str, Any]]] = {}
        processed_samples = 0

        dataset_type_str = self.config.data_config.dataset_type
        dataset_names_train = set(dataset_type_str.split("-") if "-" in dataset_type_str else [dataset_type_str])

        if not self.is_inference_mode:
            val_dataset_type_str = self.config.data_config.val_dataset_type
            dataset_names_val = val_dataset_type_str.split("-") if "-" in val_dataset_type_str else [val_dataset_type_str]
        else:
            dataset_names_val = list(dataset_names_train)

        val_only_datasets = set(dataset_names_val) - dataset_names_train
        if val_only_datasets and not use_original_labels and not use_dynamic_symbols:
            logging.info("Training datasets: %s", list(dataset_names_train))
            logging.info("Val-only datasets (skip in fixed-symbol mode): %s", list(val_only_datasets))

        for dataset_name in dataset_names_val:
            all_results[dataset_name] = []

        progress_bar = tqdm(val_dataloader, desc=f"Val {mode_name}", total=len(val_dataloader))

        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    if use_original_labels:
                        updated_batch = batch
                    else:
                        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, mappings=symbol_mappings)

                    predictions = model.generate_output(updated_batch)

                    for i, pred in enumerate(predictions):
                        dt = batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"]
                        dt_key = dt.value if hasattr(dt, "value") else str(dt)
                        true_label = batch["completion"][i] if isinstance(batch["completion"], list) else batch["completion"]

                        if dt_key in val_only_datasets and not use_original_labels and not use_dynamic_symbols:
                            continue

                        converted_pred = pred
                        if not use_original_labels and symbol_mappings:
                            converted_pred = self.symbol_manager.convert_symbols_back(pred, mappings=symbol_mappings)

                        try:
                            dataset_type = DatasetType(dt_key)
                            cleaned_pred = clean_prediction(converted_pred, dataset_type)
                        except Exception:
                            cleaned_pred = converted_pred.strip()

                        result = {
                            "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                            "true_label": true_label,
                            "predicted_label": str(cleaned_pred).strip(),
                            "dataset_type": dt_key,
                        }

                        if dt_key in all_results:
                            all_results[dt_key].append(result)

                        processed_samples += 1

                    progress_bar.set_postfix({"samples": processed_samples})
                except Exception as exc:
                    logging.error("Error during validation batch %s: %s", batch_idx, exc)
                    continue
        finally:
            progress_bar.close()

        dataset_metric_values: Dict[str, float] = {}
        computed_detailed_metrics: Dict[str, Dict[str, Any]] = {}

        for dataset_name in dataset_names_val:
            dt_results = all_results.get(dataset_name, [])
            if not dt_results:
                if dataset_name in val_only_datasets and not use_original_labels and not use_dynamic_symbols:
                    logging.info("%s skipped in fixed-symbol mode (expected)", dataset_name)
                else:
                    logging.warning("No results for dataset %s", dataset_name)
                    dataset_metric_values[dataset_name] = 0.0
                continue

            try:
                dataset_type = DatasetType(dataset_name)
                dt_metrics = evaluate_predictions(dt_results, dataset_type)
                computed_detailed_metrics[dataset_name] = dt_metrics

                if dataset_name.lower() in {"voxceleb", "meld_emotion"}:
                    dataset_metric_values[dataset_name] = dt_metrics.get("macro_f1_with_invalid", 0.0)
                else:
                    dataset_metric_values[dataset_name] = dt_metrics.get("macro_f1", 0.0)
            except Exception as exc:
                logging.error("Error evaluating predictions for %s: %s", dataset_name, exc)
                dataset_metric_values[dataset_name] = 0.0

        if dataset_metric_values:
            composite_metric_str = "|".join([f"{dataset}:{score:.4f}" for dataset, score in dataset_metric_values.items()])
            main_metric_value = sum(dataset_metric_values.values()) / len(dataset_metric_values)
        else:
            composite_metric_str = "no_data:0.000000"
            main_metric_value = 0.0

        if self.is_inference_mode:
            self.all_results = all_results
            self.computed_detailed_metrics = computed_detailed_metrics

        return {
            "accuracy": main_metric_value,
            "composite_accuracy": composite_metric_str,
            "loss": 0.0,
            "total_samples": sum(len(all_results.get(name, [])) for name in dataset_names_val if name in dataset_metric_values),
        }

    def run_comprehensive_validation(
        self,
        model,
        val_dataloader,
        epoch: int,
        phase: str,
        cycle: int = 0,
        step: Optional[Any] = None,
        symbol_mappings: Optional[Dict[str, str]] = None,
    ) -> Union[Dict[str, float], Dict[str, Any]]:
        """Run configured validation modes and aggregate outputs."""
        self.is_inference_mode = getattr(self.config, "inference_mode", False)

        bypass_mlp = getattr(model, "bypass_mlp", False)
        use_symbols = step.use_symbols if step else True

        mode_defs = self._resolve_validation_modes()
        if not use_symbols:
            mode_defs = [mode for mode in mode_defs if mode[0] == "original"]
            if not mode_defs:
                mode_defs = [("original", True, False)]

        prefix = "no_mlp" if bypass_mlp else "mlp"
        validation_results: Dict[str, Any] = {}

        if self.is_inference_mode:
            accumulated_detailed_metrics: Dict[str, Any] = {}
            accumulated_predictions: List[Dict[str, Any]] = []

        logging.info("Validation modes for %s: %s", phase.upper(), [m[0] for m in mode_defs])

        for mode_suffix, use_original, use_dynamic in mode_defs:
            mode_key = f"{prefix}_{mode_suffix}"
            try:
                metrics = self.validate_model(
                    model=model,
                    val_dataloader=val_dataloader,
                    epoch=epoch,
                    phase=phase,
                    cycle=cycle,
                    bypass_mlp=bypass_mlp,
                    use_original_labels=use_original,
                    use_dynamic_symbols=use_dynamic,
                    symbol_mappings=symbol_mappings,
                )
                validation_results[mode_key] = metrics.get("composite_accuracy", "no_data:0.000000")
                validation_results[f"{mode_key}_loss"] = metrics.get("loss", 0.0)

                if self.is_inference_mode:
                    if hasattr(self, "computed_detailed_metrics"):
                        for dataset_name, dataset_metrics in self.computed_detailed_metrics.items():
                            accumulated_detailed_metrics[f"{dataset_name}_{mode_key}"] = dataset_metrics
                    if hasattr(self, "all_results"):
                        for dataset_name, results in self.all_results.items():
                            for result in results:
                                result["validation_mode"] = mode_key
                            accumulated_predictions.extend(results)
            except Exception as exc:
                logging.error("Validation mode %s failed: %s", mode_key, exc)
                validation_results[mode_key] = "error:0.000000"
                validation_results[f"{mode_key}_loss"] = float("inf")

        if self.is_inference_mode:
            return {
                "validation_scores": validation_results,
                "detailed_metrics": accumulated_detailed_metrics,
                "all_predictions": accumulated_predictions,
                "inference_metadata": {
                    "epoch": epoch,
                    "phase": phase,
                    "cycle": cycle,
                    "total_samples": len(accumulated_predictions),
                },
            }
        return validation_results

    def log_validation_summary(self, validation_results: Dict[str, float], epoch: int, phase: str, cycle: int = 0):
        """Enhanced logging with contextual missing dataset handling."""
        logging.info("=" * 140)
        logging.info(f"{phase.upper()} CYCLE {cycle} EPOCH {epoch} VALIDATION SUMMARY:")
        logging.info("=" * 140)

        dataset_type_str = self.config.data_config.dataset_type
        dataset_names_train = set(dataset_type_str.split("-") if "-" in dataset_type_str else [dataset_type_str])

        if not getattr(self, "is_inference_mode", False):
            val_dataset_type_str = self.config.data_config.val_dataset_type
            dataset_names_val = set(val_dataset_type_str.split("-") if "-" in val_dataset_type_str else [val_dataset_type_str])
            val_only_datasets = dataset_names_val - dataset_names_train
        else:
            val_only_datasets = set()

        composite_results = {k: v for k, v in validation_results.items() if not k.endswith("_loss") and isinstance(v, str)}

        if composite_results:
            all_datasets = set()
            for composite_str in composite_results.values():
                if "|" in composite_str:
                    dataset_scores = parse_composite_metric(composite_str)
                    all_datasets.update(dataset_scores.keys())

            if all_datasets:
                dataset_info = {}
                for dataset in sorted(all_datasets):
                    abbrev = dataset[:3].upper()
                    is_trained = dataset in dataset_names_train
                    dataset_info[dataset] = {
                        "abbrev": abbrev,
                        "is_trained": is_trained,
                        "context": "TRN" if is_trained else "VAL",
                    }

                header_parts = ["Mode"]
                for dataset in sorted(all_datasets):
                    info = dataset_info[dataset]
                    header_parts.append(f"{info['abbrev']}({info['context']})")
                header_parts.append("Avg")

                header = "  ".join(f"{h:<12}" for h in header_parts)
                logging.info(header)
                logging.info("-" * len(header))

                for mode, composite_str in composite_results.items():
                    if "|" in composite_str:
                        dataset_scores = parse_composite_metric(composite_str)
                        mode_short = mode.replace("no_mlp_", "").replace("mlp_", "").replace("_", "+")
                        row_parts = [mode_short[:8]]

                        for dataset in sorted(all_datasets):
                            if dataset in dataset_scores:
                                row_parts.append(f"{dataset_scores[dataset]:.3f}")
                            else:
                                if dataset in val_only_datasets and "symbol" in mode.lower() and "fresh" not in mode.lower():
                                    row_parts.append("SKIP")
                                else:
                                    row_parts.append("N/A")

                        if dataset_scores:
                            avg_score = sum(dataset_scores.values()) / len(dataset_scores)
                            row_parts.append(f"{avg_score:.3f}")
                        else:
                            row_parts.append("N/A")

                        row = "  ".join(f"{r:<12}" for r in row_parts)
                        logging.info(row)
                    else:
                        mode_short = mode.replace("no_mlp_", "").replace("mlp_", "").replace("_", "+")
                        logging.info(f"{mode_short:<12}  {composite_str}")

        logging.info("=" * 140)


def parse_composite_metric(composite_str: str) -> Dict[str, float]:
    """Parse composite metric string back to dictionary."""
    if not composite_str or composite_str == "no_data:0.000000":
        return {}

    result = {}
    for pair in composite_str.split("|"):
        if ":" in pair:
            dataset, score = pair.split(":", 1)
            try:
                result[dataset] = float(score)
            except ValueError:
                result[dataset] = 0.0
    return result


def create_composite_metric(dataset_metrics: Dict[str, float]) -> str:
    """Create composite metric string from dictionary."""
    if not dataset_metrics:
        return "no_data:0.000000"
    return "|".join([f"{dataset}:{score:.6f}" for dataset, score in dataset_metrics.items()])


def get_best_dataset_metric(composite_str: str) -> Tuple[str, float]:
    """Get the best performing dataset from composite string."""
    metrics = parse_composite_metric(composite_str)
    if not metrics:
        return "none", 0.0

    best_dataset = max(metrics, key=metrics.get)
    best_score = metrics[best_dataset]
    return best_dataset, best_score
