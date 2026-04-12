"""
Validation Logic for Symbol Adapter Training
Handles validation with different symbol modes and configurations
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm
import numpy as np
import re

from .symbol_manager import SymbolManager
from config.train_config.training_configs import TrainingConfig
from utils.evaluation_utils import evaluate_predictions, clean_prediction
from config.data_config.master_config import DatasetType


class ValidationManager:
    """Manages validation for different training modes and symbol configurations"""
    
    def __init__(
        self,
        config: TrainingConfig,
        symbol_manager: SymbolManager,
        tokenizer,
        max_val_samples: int = 200
    ):
        self.config = config
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        
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
        symbol_mappings: Optional[Dict[str, str]] = None  # ✅ NEW PARAMETER
    ) -> Dict[str, float]:
        """
        Comprehensive validation function that handles all training modes
        Uses utils.evaluation_utils for evaluation
        """
        model.eval()

        # Get symbols for validation using SymbolManager methods
        if use_original_labels:
            symbol_mappings_to_use = {}  # No symbol replacement
            mode_name = "Original"
        elif use_dynamic_symbols:
            # Generate fresh symbols for this validation
            symbol_mappings_to_use = self.symbol_manager._generate_symbol_mappings()
            mode_name = "Fresh-Symbols"
        else:
            # ✅ Use provided mappings if in inference mode, otherwise use epoch mappings
            if self.is_inference_mode and symbol_mappings is not None:
                symbol_mappings_to_use = symbol_mappings
                mode_name = "Fixed-Symbols"
            else:
                # Use symbols from training epoch
                symbol_mappings_to_use = self.symbol_manager.get_symbols_for_epoch(epoch)
                mode_name = "Fixed-Symbols"
        
        # Configure validation mode name
        mlp_mode = "NoMLP" if bypass_mlp else "MLP"
        full_mode_name = f"{mlp_mode}+{mode_name}"
        
        logging.info(f"=== Validation: {full_mode_name} (Epoch {epoch}, {phase.upper()}) ===")
        if symbol_mappings_to_use:
            logging.info(f"Using symbols: {symbol_mappings_to_use}")
        
        # Run validation using utils.evaluation_utils
        try:
            with torch.no_grad():
                metrics = self._run_validation_with_utils(
                    model=model,
                    val_dataloader=val_dataloader,
                    symbol_mappings=symbol_mappings_to_use,  # ✅ USE THE DETERMINED MAPPINGS
                    mode_name=full_mode_name,
                    epoch=epoch,
                    phase=phase,
                    use_original_labels=use_original_labels,
                    use_dynamic_symbols=use_dynamic_symbols
                )
            
            logging.info(f"✓ {full_mode_name} Validation Score: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Validation failed for {full_mode_name}: {str(e)}")
            return {"accuracy": 0.0, "loss": float('inf'), "total_samples": 0}
        
        finally:
            model.train()  # Reset to training mode

    def _run_validation_with_utils(
        self,
        model,
        val_dataloader,
        symbol_mappings: Dict[str, str],
        mode_name: str,
        epoch: int,
        phase: str,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False
    ) -> Dict[str, float]:
        """Run validation using utils.evaluation_utils functions"""
        
        all_results = {}
        processed_samples = 0

        # ✅ Get training datasets
        dataset_type_str = self.config.data_config.dataset_type
        dataset_names_train = set(dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str])
        
        # ✅ Get validation datasets
        if not self.is_inference_mode:
            val_dataset_type_str = self.config.data_config.val_dataset_type
            dataset_names_val = val_dataset_type_str.split('-') if '-' in val_dataset_type_str else [val_dataset_type_str]
        else:
            dataset_names_val = list(dataset_names_train)
        
        # ✅ Identify val-only datasets that should be skipped in symbol modes
        val_only_datasets = set(dataset_names_val) - dataset_names_train
        
        # ✅ Log what we're doing
        if val_only_datasets and not use_original_labels and not use_dynamic_symbols:
            logging.info(f"📊 Training datasets: {list(dataset_names_train)}")
            logging.info(f"📊 Val-only datasets (will skip in symbol mode): {list(val_only_datasets)}")
        
        dataset_names = dataset_names_val

        for dataset_name in dataset_names:
            all_results[dataset_name] = []
        
        # Create progress bar
        progress_bar = tqdm(
            val_dataloader, 
            desc=f"Val {mode_name}",
            total=len(val_dataloader),
        )
        
        # ✅ Flag to log first validation prompt
        log_first_prompt = True
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Apply symbol replacement using SymbolManager methods
                    if use_original_labels:
                        # Use original batch without symbol replacement
                        updated_batch = batch
                    # elif use_dynamic_symbols:
                    #     # For dynamic symbols, use the fresh mappings directly
                    #     updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, mappings=symbol_mappings)
                    # else:
                    #     # For fixed symbols, replace using SymbolManager's method with the epoch
                    #     updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, mappings=symbol_mappings)
                    else:
                        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, mappings=symbol_mappings)

                    # ✅ Log first validation prompt
                    if log_first_prompt:
                        logging.info("=" * 80)
                        logging.info(f"FIRST VALIDATION PROMPT - {mode_name} (Epoch {epoch})")
                        logging.info("=" * 80)
                        
                        first_sample_org = batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']
                        first_sample = updated_batch['prompt'][0] if isinstance(updated_batch['prompt'], list) else updated_batch['prompt']
                        true_label = batch['completion'][0] if isinstance(batch['completion'], list) else batch['completion']
                        
                        logging.info(f"Validation Prompt Org: {str(first_sample_org)}")
                        logging.info(f"Validation Prompt: {str(first_sample)}")
                        logging.info(f"True Label: {true_label}")
                        
                        if not use_original_labels:
                            if use_dynamic_symbols:
                                logging.info(f"Using Fresh Symbols: {symbol_mappings}")
                            else:
                                logging.info(f"Using Fixed Symbols: {symbol_mappings}")
                        else:
                            logging.info("Using Original Labels (no symbols)")
                        
                        logging.info("=" * 80)
                        log_first_prompt = False
                    
                    # Generate outputs using the model's generate_output method
                    predictions = model.generate_output(updated_batch)
                    
                    # Process results using SymbolManager and evaluation_utils
                    for i, pred in enumerate(predictions):
                        dt = batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"]
                        dt_key = dt.value if hasattr(dt, 'value') else str(dt)
                        true_label = batch["completion"][i] if isinstance(batch["completion"], list) else batch["completion"]
                        
                        # ✅ CORRECT FILTERING: Skip val-only datasets in symbol modes
                        if (dt_key in val_only_datasets and 
                            not use_original_labels and 
                            not use_dynamic_symbols):
                            # Skip this sample - it's a val-only dataset in fixed symbol mode
                            continue
                        
                        # Convert symbols back to original labels using SymbolManager
                        converted_pred = pred
                        if not use_original_labels and symbol_mappings:
                            converted_pred = self.symbol_manager.convert_symbols_back(pred, mappings=symbol_mappings)
                            # if use_dynamic_symbols:
                            #     converted_pred = self.symbol_manager.convert_symbols_back(pred, mappings=symbol_mappings)
                            # else:
                            #     converted_pred = self.symbol_manager.convert_symbols_back(pred, mappings=symbol_mappings)
                        
                        # Clean the prediction using utils.evaluation_utils
                        try:
                            dataset_type = DatasetType(dt_key)
                            cleaned_pred = clean_prediction(converted_pred, dataset_type)
                        except:
                            cleaned_pred = converted_pred.strip()
                        
                        result = {
                            "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                            "true_label": true_label,
                            "predicted_label": str(cleaned_pred).strip(),
                            "dataset_type": dt_key
                        }
                        
                        # Add to results
                        if dt_key in all_results:
                            all_results[dt_key].append(result)
                        
                        processed_samples += 1
                        
                        # ✅ Log first prediction conversion
                        if batch_idx < 5 and i == 0:
                            logging.info("=" * 60)
                            logging.info("FIRST VALIDATION PREDICTION CONVERSION:")
                            logging.info(f"Raw Prediction: {pred}")
                            logging.info(f"Converted Prediction: {converted_pred}")
                            logging.info(f"Cleaned Prediction: {cleaned_pred}")
                            logging.info(f"True Label: {true_label}")
                            logging.info(f"Dataset Type: {dt_key}")
                            logging.info("=" * 60)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'samples': processed_samples
                    })
                        
                except Exception as e:
                    logging.error(f"Error during validation batch {batch_idx}: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            logging.info(f"Validation interrupted at {processed_samples} samples")
        
        finally:
            progress_bar.close()
        
        # Calculate metrics using utils.evaluation_utils.evaluate_predictions
        dataset_metric_values = {}
        computed_detailed_metrics = {}
        
        for dataset_name in dataset_names:
            dt_results = all_results.get(dataset_name, [])
            
            if dt_results:
                try:
                    dataset_type = DatasetType(dataset_name)
                    dt_metrics = evaluate_predictions(dt_results, dataset_type)
                    
                    computed_detailed_metrics[dataset_name] = dt_metrics
                    
                    logging.info(f"Metrics for {dataset_name} ({mode_name}):")
                    for metric, value in dt_metrics.items():
                        if isinstance(value, (float, int)):
                            logging.info(f"  {metric}: {value:.4f}")
                        else:
                            logging.info(f"  {metric}: {value}")
                    
                    # Extract metric for each dataset
                    if dataset_name.lower() == 'voxceleb':
                        dataset_metric_values[dataset_name] = dt_metrics.get('macro_f1_with_invalid', 0.0)
                    elif dataset_name.lower() == 'hvb':
                        dataset_metric_values[dataset_name] = dt_metrics.get('macro_f1', 0.0)
                    elif dataset_name.lower() == 'meld_emotion':
                        dataset_metric_values[dataset_name] = dt_metrics.get('macro_f1_with_invalid', 0.0)
                    else:
                        dataset_metric_values[dataset_name] = dt_metrics.get('macro_f1', 0.0)
                        
                except Exception as e:
                    logging.error(f"Error evaluating predictions for {dataset_name}: {str(e)}")
                    dataset_metric_values[dataset_name] = 0.0
            else:
                # ✅ Handle empty results properly
                if dataset_name in val_only_datasets and not use_original_labels and not use_dynamic_symbols:
                    # Expected - val-only dataset in symbol mode
                    logging.info(f"📊 {dataset_name} skipped in symbol mode (expected)")
                else:
                    # Unexpected - trained dataset with no results
                    logging.warning(f"⚠️ No results for dataset {dataset_name}")
                    dataset_metric_values[dataset_name] = 0.0
        
        # ✅ Create composite metric string
        if dataset_metric_values:
            composite_metric_str = "|".join([f"{dataset}:{score:.4f}" for dataset, score in dataset_metric_values.items()])
            main_metric_value = sum(dataset_metric_values.values()) / len(dataset_metric_values)
            
            logging.info(f"📊 Dataset metrics: {dataset_metric_values}")
            logging.info(f"📊 Combined metric: {main_metric_value:.4f}")
            logging.info(f"📊 Composite string: {composite_metric_str}")
        else:
            composite_metric_str = "no_data:0.000000"
            main_metric_value = 0.0
            logging.info(f"📊 No validation data available")
        
        # Store results for inference mode
        if self.is_inference_mode:
            self.all_results = all_results
            self.computed_detailed_metrics = computed_detailed_metrics
            self.dataset_metric_values = dataset_metric_values
            self.composite_metric_str = composite_metric_str
        
        return {
            "accuracy": main_metric_value,
            "composite_accuracy": composite_metric_str,
            "loss": 0.0,
            "total_samples": sum(len(all_results.get(name, [])) for name in dataset_names if name in dataset_metric_values)
        }

    # Keep all the existing methods unchanged
    def run_comprehensive_validation(
        self,
        model,
        val_dataloader,
        epoch: int,
        phase: str,  # This comes from TrainingStep.phase: "lora", "mlp", "joint"
        cycle: int = 0,
        step: Optional[Any] = None,  # Add step parameter
        symbol_mappings: Optional[Dict[str, str]] = None  # ✅ NEW PARAMETER
    ) -> Union[Dict[str, float], Dict[str, Any]]:
        """
        Run comprehensive validation with all modes (MLP+Symbols, NoMLP+Symbols, etc.)
        """
        validation_results = {}
        
        # Check model state
        bypass_mlp = getattr(model, 'bypass_mlp', False)
        use_symbols = step.use_symbols if step else True  # Default to True if no step provided

        dynamic_symbols_enabled = self.config.symbol_config.dynamic_symbols
        is_inference_mode = getattr(self.config, 'inference_mode', False)
        self.is_inference_mode = is_inference_mode  # Store for later use

        # self.only_original = getattr(self.config, 'only_original', False)
        self.only_original = False


        if self.is_inference_mode:
            accumulated_detailed_metrics = {}
            accumulated_predictions = []

        # Determine validation modes based on phase + model state + step config
        # if phase == "lora":
            # if bypass_mlp and use_symbols:
                # LoRA training with bypass_mlp=True and symbols

        
        
        modes = [
            ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols (from training)
            ("no_mlp_fresh", True, False, True),      # NoMLP + Fresh Symbols
            ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
        ]
   
        # modes = [
        #     ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols (from training)
        #     ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
        # ]

        logging.info(f"Validation modes for {phase.upper()} (bypass_mlp={bypass_mlp}, use_symbols={use_symbols}):")

    
        # Run each validation mode
        for mode_key, bypass_mlp_val, use_original, use_dynamic in modes:

            if self.is_inference_mode:
                # Skip fixed symbol modes (use_dynamic=False) when dynamic symbols are enabled
                if  use_dynamic:
                    logging.info(f"⏭️ Skipping {mode_key} (fixed symbols) - dynamic symbols enabled in inference")
                    continue

            if self.only_original:
                # If only original labels are used, skip all symbol modes
                if not use_original:
                    logging.info(f"⏭️ Skipping {mode_key} - only original labels are used")
                    continue

            # if self.only_original:
            #     # If only original labels are used, skip all symbol modes
            #     if use_original:
            #         logging.info(f"⏭️ Skipping {mode_key} -  original labels are not used")
            #         continue

            try:
                metrics = self.validate_model(
                    model=model,
                    val_dataloader=val_dataloader,
                    epoch=epoch,
                    phase=phase,
                    cycle=cycle,
                    bypass_mlp=bypass_mlp_val,
                    use_original_labels=use_original,
                    use_dynamic_symbols=use_dynamic,
                    symbol_mappings=symbol_mappings  # ✅ PASS PARAMETER
                )
                composite_metric = metrics.get("composite_accuracy", "no_data:0.000000")
                # validation_results[mode_key] = metrics["accuracy"]
                validation_results[mode_key] =  composite_metric
                validation_results[f"{mode_key}_loss"] = metrics["loss"]
                # validation_results[f"{mode_key}_composite"] = composite_metric
                                                          
                # logging.info(f"Accumulated detailed metrics for {mode_key}: {composite_metric}")
                
                if self.is_inference_mode:
                    if hasattr(self, 'computed_detailed_metrics'):
                        for dataset_name, dataset_metrics in self.computed_detailed_metrics.items():
                            key = f"{dataset_name}_{mode_key}"  # e.g., "voxceleb_mlp_symbols"
                            accumulated_detailed_metrics[key] = dataset_metrics

                    
                    # Collect predictions with mode information
                    if hasattr(self, 'all_results'):
                        for dataset_name, results in self.all_results.items():
                            for result in results:
                                result['validation_mode'] = mode_key  # ✅ TAG WITH MODE
                            accumulated_predictions.extend(results)
                
            except Exception as e:
                logging.error(f"Validation mode {mode_key} failed: {str(e)}")
                validation_results[mode_key] = 0.0
                # validation_results[f"{mode_key}_composite"] = "error:0.000000"
                validation_results[f"{mode_key}_loss"] = float('inf')


        
        # Return based on mode
        if self.is_inference_mode:
            logging.info("🔍 Running in inference mode - collecting detailed results")
            
            # Return accumulated results (NO additional computation)
            return {
                'validation_scores': validation_results,
                'detailed_metrics': accumulated_detailed_metrics,  # ✅ PRE-COMPUTED
                'all_predictions': accumulated_predictions,        # ✅ PRE-COLLECTED
                'inference_metadata': {
                    'epoch': epoch,
                    'phase': phase,
                    'cycle': cycle,
                    'total_samples': len(accumulated_predictions)
                }
            }
        else:
            # Training mode: return only validation scores (as before)
            return validation_results


    def log_validation_summary(self, validation_results: Dict[str, float], epoch: int, phase: str, cycle: int = 0):
        """Enhanced logging with contextual missing dataset handling"""
        logging.info("=" * 140)
        logging.info(f"{phase.upper()} CYCLE {cycle} EPOCH {epoch} VALIDATION SUMMARY:")
        logging.info("=" * 140)
        
        # Get training vs validation datasets for context
        dataset_type_str = self.config.data_config.dataset_type
        dataset_names_train = set(dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str])
        
        if not getattr(self, 'is_inference_mode', False):
            val_dataset_type_str = self.config.data_config.val_dataset_type
            dataset_names_val = set(val_dataset_type_str.split('-') if '-' in val_dataset_type_str else [val_dataset_type_str])
            val_only_datasets = dataset_names_val - dataset_names_train
        else:
            val_only_datasets = set()
        
        composite_results = {k: v for k, v in validation_results.items() 
                            if not k.endswith('_loss') and isinstance(v, str)}
        
        if composite_results:
            # Get all datasets and create display mapping
            all_datasets = set()
            for composite_str in composite_results.values():
                if '|' in composite_str:
                    dataset_scores = parse_composite_metric(composite_str)
                    all_datasets.update(dataset_scores.keys())
            
            if all_datasets:
                # Create abbreviated names and context info
                dataset_info = {}
                for dataset in sorted(all_datasets):
                    abbrev = dataset[:3].upper()
                    is_trained = dataset in dataset_names_train
                    dataset_info[dataset] = {
                        'abbrev': abbrev,
                        'is_trained': is_trained,
                        'context': 'TRN' if is_trained else 'VAL'
                    }
                
                # Header with context
                header_parts = ["Mode"]
                for dataset in sorted(all_datasets):
                    info = dataset_info[dataset]
                    header_parts.append(f"{info['abbrev']}({info['context']})")
                header_parts.append("Avg")
                
                header = "  ".join(f"{h:<12}" for h in header_parts)
                logging.info(header)
                logging.info("-" * len(header))
                
                # Data rows
                for mode, composite_str in composite_results.items():
                    if '|' in composite_str:
                        dataset_scores = parse_composite_metric(composite_str)
                        mode_short = mode.replace('no_mlp_', '').replace('mlp_', '').replace('_', '+')
                        row_parts = [mode_short[:8]]
                        
                        for dataset in sorted(all_datasets):
                            info = dataset_info[dataset]
                            
                            if dataset in dataset_scores:
                                score = dataset_scores[dataset]
                                row_parts.append(f"{score:.3f}")
                            else:
                                # ✅ Context-aware missing values
                                if dataset in val_only_datasets and 'symbol' in mode.lower() and 'fresh' not in mode.lower():
                                    row_parts.append("SKIP")  # Expected skip for val-only in symbol mode
                                else:
                                    row_parts.append("N/A")   # Unexpected missing
                        
                        # Average
                        if dataset_scores:
                            avg_score = sum(dataset_scores.values()) / len(dataset_scores)
                            row_parts.append(f"{avg_score:.3f}")
                        else:
                            row_parts.append("N/A")
                        
                        row = "  ".join(f"{r:<12}" for r in row_parts)
                        logging.info(row)
                    else:
                        mode_short = mode.replace('no_mlp_', '').replace('mlp_', '').replace('_', '+')
                        logging.info(f"{mode_short:<12}  {composite_str}")
        
        logging.info("=" * 140)


def parse_composite_metric(composite_str: str) -> Dict[str, float]:
    """Parse composite metric string back to dictionary"""
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
    """Create composite metric string from dictionary"""
    if not dataset_metrics:
        return "no_data:0.000000"
    return "|".join([f"{dataset}:{score:.6f}" for dataset, score in dataset_metrics.items()])

def get_best_dataset_metric(composite_str: str) -> Tuple[str, float]:
    """Get the best performing dataset from composite string"""
    metrics = parse_composite_metric(composite_str)
    if not metrics:
        return "none", 0.0
    
    best_dataset = max(metrics, key=metrics.get)
    best_score = metrics[best_dataset]
    return best_dataset, best_score


