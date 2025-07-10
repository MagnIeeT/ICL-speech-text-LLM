import pandas as pd
import numpy as np
import logging
import json
import os
import re
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data.master_config import get_dataset_config, get_swap_config, DatasetType

logger = logging.getLogger(__name__)

def evaluate_predictions(predictions: List[Dict[str, Any]], dataset_type: DatasetType) -> Dict[str, Any]:
    """
    Evaluate model predictions based on dataset type.
    
    Args:
        predictions: List of prediction dictionaries with true_label and predicted_sentiment
        dataset_type: Type of dataset being evaluated
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not predictions:
        logger.warning("Empty predictions list provided for evaluation")
        return {"error": "Empty predictions list", "accuracy": 0.0}
    
    try:
        # Get the appropriate config based on dataset type
        if dataset_type in [DatasetType.VOXCELEB_SWAP, DatasetType.HVB_SWAP, DatasetType.VOXPOPULI_SWAP]:
            config = get_swap_config(dataset_type)
        else:
            config = get_dataset_config(dataset_type)
        
        

        if not config:
            logger.warning(f"No config found for dataset type: {dataset_type}")
            return {"error": "Invalid dataset type"}
        
        # Extract and clean labels
        true_labels = [p.get("true_label", "") for p in predictions]
        # Clean the predicted labels
        pred_labels = [clean_prediction(p.get("predicted_label", ""), dataset_type) for p in predictions]
        

        
        # Log some example predictions for debugging
        logger.info("\nExample predictions after cleaning:")
        for i in range(min(5, len(predictions))):
            logger.info(f"Original: {predictions[i].get('predicted_label', '')}")
            logger.info(f"Cleaned: {pred_labels[i]}")
            logger.info(f"True: {true_labels[i]}")
            logger.info("-" * 50)
        
        # Create DataFrame for evaluation
        df = pd.DataFrame({
            "text": [p.get("text", "") for p in predictions],
            "gt": true_labels,
            "pd": pred_labels
        })
        
        # Get valid labels for this dataset type
        valid_labels = None
        if hasattr(config, 'valid_labels') and config.valid_labels is not None:
            valid_labels = [label.lower() for label in config.valid_labels]
        
        # Calculate metrics based on dataset type
        if dataset_type in [
            DatasetType.VOXCELEB, 
            DatasetType.VOXCELEB_SWAP, 
            DatasetType.VOXCELEB_GREEK,
            DatasetType.MELD,
            DatasetType.MELD_GREEK,
            DatasetType.MELD_EMOTION,
            DatasetType.MELD_EMOTION_GREEK
        ]:
            metrics = evaluate_voxceleb(df, valid_labels)
        elif dataset_type in [DatasetType.HVB, DatasetType.HVB_SWAP, DatasetType.HVB_GREEK]:
            metrics = evaluate_hvb(df, valid_labels)
        elif dataset_type in [DatasetType.VOXPOPULI, DatasetType.VOXPOPULI_SWAP, DatasetType.VOXPOPULI_GREEK]:
            metrics = evaluate_voxpopuli(df, valid_labels)
        elif dataset_type == DatasetType.VOXPOPULI_NEL:
            metrics = evaluate_vp_nel(df, valid_labels)
        # elif dataset_type == DatasetType.SQQ:
        #     print(f"Calling evaluate_sqq with {len(df)} samples")
        #     metrics = evaluate_sqq(df)
        elif dataset_type == DatasetType.SQA:
            print(f"Calling evaluate_sqa with {len(df)} samples")
            metrics = evaluate_sqa(df)
        else:
            logger.warning(f"Unsupported dataset type for evaluation: {dataset_type}")
            return {"accuracy": 0.0}
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in evaluate_predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e), "accuracy": 0.0}

def evaluate_voxceleb(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate VoxCeleb predictions (single-label classification)"""
    total_samples = len(df)
    
    # Convert to lowercase for consistent comparison
    df['gt'] = df['gt'].str.lower()
    df['pd'] = df['pd'].str.lower()
    
    # Filter for valid ground truth labels
    df = df[df['gt'].isin(valid_classes)]
    after_gt_filter = len(df)
    
    # Calculate metrics with invalid class
    df_with_invalid = df.copy()
    df_with_invalid['pd'] = df_with_invalid['pd'].apply(
        lambda x: x if x in valid_classes else 'invalid'
    )
    
    macro_f1_with_invalid = f1_score(
        df_with_invalid['gt'].values,
        df_with_invalid['pd'].values,
        average="macro",
        labels=valid_classes,
        zero_division=0
    )
    
    # Calculate standard metrics (filtered)
    df_filtered = df[df['pd'].isin(valid_classes)]
    after_pred_filter = len(df_filtered)
    
    if after_pred_filter == 0:
        logger.warning("No valid predictions found for evaluation")
        return {
            'macro_f1_filtered': 0.0,
            'macro_f1_with_invalid': 0.0,
            'invalid_predictions': len(df[~df['pd'].isin(valid_classes)]),
            'total_samples': total_samples,
            'valid_gt_samples': after_gt_filter,
            'valid_samples': 0
        }
    
    matrix_filtered = confusion_matrix(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        labels=valid_classes
    )
    
    class_accuracy_filtered = matrix_filtered.diagonal()/matrix_filtered.sum(axis=1)
    class_samples_filtered = matrix_filtered.sum(axis=1)
    
    macro_f1_filtered = f1_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average="macro",
        labels=valid_classes,
        zero_division=0
    )
    
    # Calculate per-class metrics
    class_precision = precision_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average=None,
        labels=valid_classes,
        zero_division=0
    )
    
    class_recall = recall_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average=None,
        labels=valid_classes,
        zero_division=0
    )
    
    class_f1 = f1_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average=None,
        labels=valid_classes,
        zero_division=0
    )
    
    # Calculate overall accuracy
    accuracy = accuracy_score(df_filtered['gt'].values, df_filtered['pd'].values)
    
    return {
        # Standard (filtered) metrics
        'accuracy': accuracy,
        'macro_f1_filtered': macro_f1_filtered,
        'class_accuracy_filtered': class_accuracy_filtered.tolist(),
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'confusion_matrix_filtered': matrix_filtered.tolist(),
        'valid_samples': after_pred_filter,
        
        # Metrics including invalid predictions
        'macro_f1_with_invalid': macro_f1_with_invalid,
        'invalid_predictions': len(df[~df['pd'].isin(valid_classes)]),
        
        # General statistics
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter,
        'valid_classes': valid_classes
    }

def evaluate_hvb(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate HVB predictions (multi-label classification)"""
    total_samples = len(df)

    # Convert string labels to lists if needed
    df['gt'] = df['gt'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    df['pd'] = df['pd'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    
    # Convert to lowercase for consistent comparison
    df['gt'] = df['gt'].apply(lambda labels: [label.lower() for label in labels])
    df['pd'] = df['pd'].apply(lambda labels: [label.lower() for label in labels])
    
    # Filter for samples with at least one valid ground truth label
    df = df[df['gt'].apply(lambda labels: any(label in valid_classes for label in labels))]
    after_gt_filter = len(df)
    
    # Count samples with no valid predictions
    invalid_samples = sum(1 for pred_labels in df['pd'] 
                         if not any(label in valid_classes for label in pred_labels))
    
    # Convert to binary matrix format with invalid handling
    def to_binary_vector(labels: List[str], valid_classes: List[str]) -> np.ndarray:
        # Check if any valid label exists
        has_valid_label = any(label in valid_classes for label in labels)
        if not has_valid_label:
            # Return zero vector for invalid predictions
            return np.zeros(len(valid_classes))
        return np.array([1 if label in labels else 0 for label in valid_classes])
    
    y_true = np.array([to_binary_vector(labels, valid_classes) for labels in df['gt']])
    y_pred = np.array([to_binary_vector(labels, valid_classes) for labels in df['pd']])
    
    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate support for each class
    support = y_true.sum(axis=0)
    
    # Calculate exact match accuracy
    exact_match = sum(np.array_equal(t, p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    
    return {
        'exact_match': exact_match,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'support': support.tolist(),
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter,
        'invalid_samples': invalid_samples,
        'valid_classes': valid_classes
    }

def evaluate_voxpopuli(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate VoxPopuli entity type predictions"""
    total_samples = len(df)
    
    # Add 'none' to valid classes if not already present
    all_valid_classes = valid_classes + ['none'] if 'none' not in valid_classes else valid_classes

    # Convert string labels to lists if needed and clean them
    df['gt'] = df['gt'].apply(lambda x: [label.strip().lower() for label in x.split(',')] if isinstance(x, str) else x)
    df['pd'] = df['pd'].apply(lambda x: [label.strip().lower() for label in x.split(',')] if isinstance(x, str) else x)
    
    # Filter for samples with at least one valid ground truth label
    df = df[df['gt'].apply(lambda labels: any(label in all_valid_classes for label in labels))]
    after_gt_filter = len(df)
    
    # Count invalid predictions
    invalid_samples = sum(1 for pred_labels in df['pd'] 
                         if not any(label in all_valid_classes for label in pred_labels))
    
    # Convert to binary matrix format with invalid handling
    def to_binary_vector(labels: List[str], valid_classes: List[str]) -> np.ndarray:
        # Check if any valid label exists
        has_valid_label = any(label in valid_classes for label in labels)
        if not has_valid_label:
            # Return zero vector for invalid predictions
            return np.zeros(len(valid_classes))
        return np.array([1 if label in labels else 0 for label in valid_classes])
    
    # Use all_valid_classes (including 'none') for binary vectors
    y_true = np.array([to_binary_vector(labels, all_valid_classes) for labels in df['gt']])
    y_pred = np.array([to_binary_vector(labels, all_valid_classes) for labels in df['pd']])
    
    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate support for each class
    support = y_true.sum(axis=0)
    
    # Calculate exact match accuracy
    exact_match = sum(np.array_equal(t, p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    
    return {
        'exact_match': exact_match,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'support': support.tolist(),
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter,
        'invalid_samples': invalid_samples,
        'valid_classes': valid_classes
    }

def parse_entities(entity_string):
    """Helper function to parse entity string format into list of tuples"""
    parsed_entities = []
    if not entity_string or entity_string.strip() == "":
        return parsed_entities
        
    for entity in entity_string.split(';'):
        if entity.strip():
            try:
                entity_type, times = entity.strip().split(':')
                start, end = map(float, times.strip().split())
                parsed_entities.append((entity_type.strip(), start, end))
            except Exception as e:
                logger.warning(f"Error parsing entity: {entity}, Error: {e}")
                continue
    return parsed_entities

def evaluate_vp_nel(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate VoxPopuli Named Entity Linking predictions with time alignments"""
    total_samples = len(df)


    # Convert predictions and ground truth to lowercase
    df['gt'] = df['gt'].str.lower()
    df['pd'] = df['pd'].str.lower()
    
    # Parse predictions and ground truth
    parsed_gt = {idx: parse_entities(row['gt']) for idx, row in df.iterrows()}
    parsed_pred = {idx: parse_entities(row['pd']) for idx, row in df.iterrows()}
    
    # Calculate word-level metrics with different tolerances
    word_metrics = {}
    tolerances = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    for tolerance in tolerances:
        total_correct = 0
        total_pred = 0
        total_gt = 0
        
        for idx in parsed_gt.keys():
            gt_entities = parsed_gt[idx]
            pred_entities = parsed_pred.get(idx, [])
            
            total_gt += len(gt_entities)
            total_pred += len(pred_entities)
            
            # Track matched ground truth entities
            matched_gt = set()
            
            # Check each prediction against ground truth
            for pred_type, pred_start, pred_end in pred_entities:
                best_overlap = 0
                best_match_idx = None
                
                # Find best matching ground truth entity of the same type
                for gt_idx, (gt_type, gt_start, gt_end) in enumerate(gt_entities):
                    if gt_idx in matched_gt:
                        continue
                    
                    if pred_type.upper() == gt_type.upper():
                        overlap_start = max(pred_start, gt_start)
                        overlap_end = min(pred_end, gt_end)
                        if overlap_end > overlap_start:
                            overlap = (overlap_end - overlap_start) / (gt_end - gt_start)
                            if overlap >= tolerance and overlap > best_overlap:
                                best_overlap = overlap
                                best_match_idx = gt_idx
                
                if best_match_idx is not None:
                    total_correct += 1
                    matched_gt.add(best_match_idx)
        
        precision = total_correct / max(total_pred, 1)
        recall = total_correct / max(total_gt, 1)
        f1 = 2 * (precision * recall) / max((precision + recall), 1e-6)
        
        word_metrics[str(tolerance)] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate frame-level metrics
    total_pred_frames = 0
    total_gt_frames = 0
    total_correct_frames = 0
    
    for idx in parsed_gt.keys():
        gt_entities = parsed_gt[idx]
        pred_entities = parsed_pred.get(idx, [])
        
        # Convert time ranges to frame counts for predictions
        for pred_type, pred_start, pred_end in pred_entities:
            pred_frames = int((pred_end - pred_start) * 100)  # Convert to centiseconds
            total_pred_frames += pred_frames
            
            # Check overlap with ground truth of same type
            for gt_type, gt_start, gt_end in gt_entities:
                if pred_type.upper() == gt_type.upper():
                    overlap_start = max(pred_start, gt_start)
                    overlap_end = min(pred_end, gt_end)
                    if overlap_end > overlap_start:
                        correct_frames = int((overlap_end - overlap_start) * 100)
                        total_correct_frames += correct_frames
        
        # Count ground truth frames
        for _, gt_start, gt_end in gt_entities:
            gt_frames = int((gt_end - gt_start) * 100)
            total_gt_frames += gt_frames
    
    frame_precision = total_correct_frames / max(total_pred_frames, 1)
    frame_recall = total_correct_frames / max(total_gt_frames, 1)
    frame_f1 = 2 * (frame_precision * frame_recall) / max((frame_precision + frame_recall), 1e-6)
    
    return {
        'word_metrics': word_metrics,
        'frame_metrics': {
            'precision': frame_precision,
            'recall': frame_recall,
            'f1': frame_f1
        },
        'total_samples': total_samples,
        'total_gt_entities': sum(len(entities) for entities in parsed_gt.values()),
        'total_pred_entities': sum(len(entities) for entities in parsed_pred.values()),
        'total_frames': {
            'gt': total_gt_frames,
            'pred': total_pred_frames,
            'correct': total_correct_frames
        }
    }

def clean_prediction(prediction: str, dataset_type: DatasetType = None) -> str:
    """
    Clean prediction based on dataset type and expected format.
    Uses config-driven approach instead of hardcoded labels.
    """
    # First remove basic escape characters and normalize whitespace
    cleaned = prediction.replace('\\', '')
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
    
    # Handle newlines - take first line only
    if '\n' in cleaned:
        cleaned = cleaned.split('\n')[0]

    cleaned = re.sub(r',\s*,', ',', cleaned)  # Replace multiple commas with single comma
    cleaned = re.sub(r',\s*$', '', cleaned)   # Remove trailing comma
    cleaned = re.sub(r'^\s*,', '', cleaned)   # Remove leading comma
    
    # Get valid labels from config for any dataset type
    valid_labels = None
    try:
        if dataset_type:
            config = get_dataset_config(dataset_type)
            
            if config and hasattr(config, 'valid_labels') and config.valid_labels:
                valid_labels = set([label.lower() for label in config.valid_labels])
    except Exception as e:
        logger.warning(f"Error getting config for {dataset_type}: {e}")
        valid_labels = None
    
    # Dataset-specific processing based on expected format
    if dataset_type in [
        DatasetType.VOXCELEB, 
        DatasetType.VOXCELEB_GREEK, 
        DatasetType.MELD_EMOTION,
        DatasetType.MELD_EMOTION_GREEK
    ]:
        # Single-label classification: take first valid word
        words = re.split(r'[^a-zA-Z]', cleaned)
        words = [w.strip().lower() for w in words]
        words = [w for w in words if w]  # Remove empty strings
        
        if valid_labels and words:
            # Return first valid label found
            for word in words:
                if word in valid_labels:
                    return word
            # If no valid label found, return first word as fallback
            return words[0]
        elif words:
            return words[0]
        return cleaned.lower()
    
    elif dataset_type in [
        DatasetType.HVB, 
        DatasetType.HVB_GREEK
    ]:

        # Multi-label classification: keep all valid labels
        labels = [l.strip().lower() for l in cleaned.split(',')]
        # Filter out empty strings and partial/incomplete labels
        labels = [l for l in labels if l and '(' not in l and l.strip()]
        
        if valid_labels:
            # Keep only valid labels from config
            valid_found = [label for label in labels if label in valid_labels]
            if valid_found:
                return ', '.join(valid_found)
            # If no valid labels found, return cleaned original as fallback
            return cleaned
        else:
            # No config available, return cleaned labels
            return ', '.join(labels) if labels else cleaned
    
    elif dataset_type in [
        DatasetType.VOXPOPULI, 
        DatasetType.VOXPOPULI_GREEK
    ]:
        # Multi-label classification with 'none' option
        if cleaned.lower().strip() == 'none':
            return 'none'
        
        labels = [l.strip().lower() for l in cleaned.split(',')]
        labels = [l for l in labels if l and '(' not in l and l.strip()]
        
        if valid_labels:
            # Add 'none' to valid labels if not present
            extended_valid = valid_labels.copy()
            extended_valid.add('none')
            
            valid_found = [label for label in labels if label in extended_valid]
            if valid_found:
                return ', '.join(valid_found)
            return cleaned
        else:
            return ', '.join(labels) if labels else cleaned
    
    elif dataset_type == DatasetType.SQA:
        # For SQA, expect "start_time end_time"
        cleaned = cleaned.strip()
        try:
            start, end = map(float, cleaned.split())
            return f"{start:.2f} {end:.2f}"
        except:
            return cleaned
            
    elif dataset_type == DatasetType.VOXPOPULI_NEL:
        # For VP_NEL, expect "TYPE: start end" format


        if cleaned.lower() == 'none':
            return 'none'
        
        try:
            spans = cleaned.split(';')
            cleaned_spans = []
            for span in spans:
                span = span.strip()
                if ':' in span:
                    entity_type, times = span.split(':', 1)
                    try:
                        start, end = map(float, times.strip().split())
                        cleaned_spans.append(f"{entity_type.strip()}: {start:.2f} {end:.2f}")
                    except:
                        cleaned_spans.append(span)
            return '; '.join(cleaned_spans)
        except:
            return cleaned
    # Default cleaning for unknown dataset types
    return cleaned.lower().strip()

def analyze_errors(true_labels: List[Any], pred_labels: List[Any], dataset_type: DatasetType) -> Dict[str, Any]:
    """
    Analyze prediction errors to identify patterns.
    
    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        dataset_type: Type of dataset
        
    Returns:
        Dictionary with error analysis results
    """
    try:
        # For multi-label datasets, convert to sets for comparison
        if dataset_type in [DatasetType.HVB, DatasetType.HVB_SWAP, DatasetType.HVB_GREEK]:
            errors = []
            for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
                true_set = set(true) if isinstance(true, list) else set([true])
                pred_set = set(pred) if isinstance(pred, list) else set([pred])
                
                if true_set != pred_set:
                    errors.append({
                        "index": i,
                        "true": true,
                        "pred": pred,
                        "missing": list(true_set - pred_set),
                        "extra": list(pred_set - true_set)
                    })
            
            # Analyze common missing and extra labels
            missing_counts = Counter()
            extra_counts = Counter()
            
            for error in errors:
                for label in error["missing"]:
                    missing_counts[label] += 1
                for label in error["extra"]:
                    extra_counts[label] += 1
            
            return {
                "num_errors": len(errors),
                "error_rate": len(errors) / len(true_labels),
                "common_missing_labels": dict(missing_counts.most_common(5)),
                "common_extra_labels": dict(extra_counts.most_common(5)),
                "example_errors": errors[:5] if errors else []  # Include first 5 errors as examples
            }
        else:
            # For single-label classification
            errors = []
            for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
                if true != pred:
                    errors.append({
                        "index": i,
                        "true": true,
                        "pred": pred
                    })
            
            # Count confusion pairs
            confusion_pairs = Counter()
            for error in errors:
                confusion_pairs[(error["true"], error["pred"])] += 1
            
            # Get most common confusion pairs
            common_confusions = {}
            for (true, pred), count in confusion_pairs.most_common(5):
                common_confusions[f"{true} → {pred}"] = count
            
            return {
                "num_errors": len(errors),
                "error_rate": len(errors) / len(true_labels),
                "common_confusions": common_confusions,
                "example_errors": errors[:5] if errors else []  # Include first 5 errors as examples
            }
    
    except Exception as e:
        logger.error(f"Error in analyze_errors: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "num_errors": 0,
            "error_rate": 0.0
        }

def save_evaluation_results(metrics: Dict, output_dir: str, filename: str) -> None:
    """
    Save evaluation metrics to a file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
        filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Convert numpy values to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    metrics_converted = convert_numpy(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_converted, f, indent=2)
    
    logger.info(f"Saved evaluation results to {output_path}")

def evaluate_sqq(df: pd.DataFrame, valid_classes: List[str] = None) -> Dict:
    """Evaluate SQQ predictions with time alignments (start, end timestamps only)"""
    total_samples = len(df)
    
    def parse_timestamps(time_string):
        """Parse timestamp string 'start end' into tuple"""
        if not time_string or time_string.strip() == "":
            return []
        try:
            start, end = map(float, time_string.strip().split())
            return [(start, end)]
        except Exception as e:
            logger.warning(f"Error parsing timestamps: {time_string}, Error: {e}")
            return []
    
    # Parse predictions and ground truth
    parsed_gt = {idx: parse_timestamps(row['gt']) for idx, row in df.iterrows()}
    parsed_pred = {idx: parse_timestamps(row['pd']) for idx, row in df.iterrows()}
    
    # Calculate word-level metrics with different tolerances
    word_metrics = {}
    tolerances = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    for tolerance in tolerances:
        total_correct = 0
        total_pred = 0
        total_gt = 0
        
        for idx in parsed_gt.keys():
            gt_timestamps = parsed_gt[idx]
            pred_timestamps = parsed_pred.get(idx, [])
            
            total_gt += len(gt_timestamps)
            total_pred += len(pred_timestamps)
            
            # Track matched ground truth timestamps
            matched_gt = set()
            
            # Check each prediction against ground truth
            for pred_start, pred_end in pred_timestamps:
                best_overlap = 0
                best_match_idx = None
                
                # Find best matching ground truth timestamp
                for gt_idx, (gt_start, gt_end) in enumerate(gt_timestamps):
                    if gt_idx in matched_gt:
                        continue
                    
                    overlap_start = max(pred_start, gt_start)
                    overlap_end = min(pred_end, gt_end)
                    if overlap_end > overlap_start:
                        overlap = (overlap_end - overlap_start) / (gt_end - gt_start)
                        if overlap >= tolerance and overlap > best_overlap:
                            best_overlap = overlap
                            best_match_idx = gt_idx
                
                if best_match_idx is not None:
                    total_correct += 1
                    matched_gt.add(best_match_idx)
        
        precision = total_correct / max(total_pred, 1)
        recall = total_correct / max(total_gt, 1)
        f1 = 2 * (precision * recall) / max((precision + recall), 1e-6)
        
        word_metrics[str(tolerance)] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate frame-level metrics
    total_pred_frames = 0
    total_gt_frames = 0
    total_correct_frames = 0
    
    for idx in parsed_gt.keys():
        gt_timestamps = parsed_gt[idx]
        pred_timestamps = parsed_pred.get(idx, [])
        
        # Convert time ranges to frame counts for predictions
        for pred_start, pred_end in pred_timestamps:
            pred_frames = int((pred_end - pred_start) * 100)  # Convert to centiseconds
            total_pred_frames += pred_frames
            
            # Check overlap with ground truth
            for gt_start, gt_end in gt_timestamps:
                overlap_start = max(pred_start, gt_start)
                overlap_end = min(pred_end, gt_end)
                if overlap_end > overlap_start:
                    correct_frames = int((overlap_end - overlap_start) * 100)
                    total_correct_frames += correct_frames
        
        # Count ground truth frames
        for gt_start, gt_end in gt_timestamps:
            gt_frames = int((gt_end - gt_start) * 100)
            total_gt_frames += gt_frames
    
    frame_precision = total_correct_frames / max(total_pred_frames, 1)
    frame_recall = total_correct_frames / max(total_gt_frames, 1)
    frame_f1 = 2 * (frame_precision * frame_recall) / max((frame_precision + frame_recall), 1e-6)
    
    return {
        'word_metrics': word_metrics,
        'frame_metrics': {
            'precision': frame_precision,
            'recall': frame_recall,
            'f1': frame_f1
        },
        'total_samples': total_samples,
        'total_gt_segments': sum(len(timestamps) for timestamps in parsed_gt.values()),
        'total_pred_segments': sum(len(timestamps) for timestamps in parsed_pred.values()),
        'total_frames': {
            'gt': total_gt_frames,
            'pred': total_pred_frames,
            'correct': total_correct_frames
        }
    }

def evaluate_sqa(df: pd.DataFrame, valid_classes: List[str] = None) -> Dict:
    """
    Evaluate Question-Answering predictions with improved error handling.
    """
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    # Make sure nltk has necessary data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    total_samples = len(df)
    
    # Add debug logging for first few rows
    logger.info(f"SQA evaluation - processing {total_samples} samples")
    if total_samples > 0:
        sample_rows = df.head(3)
        for i, row in sample_rows.iterrows():
            logger.info(f"Sample {i}: GT='{row.get('gt', 'None')}', Pred='{row.get('pd', 'None')}'")
    
    # Process ground truth and predicted answers
    def normalize_answer(text):
        """Normalize answer by lowercasing, removing punctuation and extra spaces"""
        if text is None:
            return ""
            
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
    def get_tokens(text):
        """Get tokens from normalized text"""
        normalized = normalize_answer(text)
        if not normalized:
            return []
        return normalized.split()
    
    # Calculate metrics
    exact_matches = 0
    f1_scores = []
    bleu_scores = []
    
    # Initialize smoother for BLEU score calculation
    smoother = SmoothingFunction().method1
    
    try:
        for _, row in df.iterrows():
            # Get ground truth and prediction with fallback to empty string
            gt = row.get('gt', "")
            pred = row.get('pd', "")
            
            # Handle None values
            if gt is None: gt = ""
            if pred is None: pred = ""
            
            # Exact match
            exact_match = 1 if normalize_answer(gt) == normalize_answer(pred) else 0
            exact_matches += exact_match
            
            # F1 Score (token level)
            gt_tokens = get_tokens(gt)
            pred_tokens = get_tokens(pred)
            
            if not gt_tokens and not pred_tokens:
                f1 = 1.0  # Both empty means perfect match
            elif not gt_tokens or not pred_tokens:
                f1 = 0.0  # One empty means no match
            else:
                # Calculate token overlap
                common_tokens = Counter(gt_tokens) & Counter(pred_tokens)
                num_common = sum(common_tokens.values())
                
                # Calculate precision and recall
                precision = num_common / max(len(pred_tokens), 1)
                recall = num_common / max(len(gt_tokens), 1)
                
                # Calculate F1
                f1 = 2 * (precision * recall) / max((precision + recall), 1e-6)
            
            f1_scores.append(f1)
            
            # BLEU Score - with safety checks
            try:
                if gt_tokens:
                    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)
                else:
                    bleu = 0.0 if pred_tokens else 1.0
            except Exception as e:
                logger.warning(f"Error calculating BLEU score: {e}")
                bleu = 0.0
                
            bleu_scores.append(bleu)
        
        # Calculate averages
        em_score = exact_matches / max(total_samples, 1)
        avg_f1 = sum(f1_scores) / max(len(f1_scores), 1)
        avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
        
        # Create results dictionary
        results = {
            'exact_match': em_score,
            'f1_score': avg_f1,
            'bleu_score': avg_bleu,
            'total_samples': total_samples,
            'samples_evaluated': len(f1_scores),
            'sample_metrics': {
                'exact_match': [1 if f == 1.0 else 0 for f in f1_scores],
                'f1_scores': f1_scores,
                'bleu_scores': bleu_scores
            }
        }
        
        return results
    except Exception as e:
        logger.error(f"Error in evaluate_sqa: {str(e)}")
        traceback.print_exc()
        return {
            'exact_match': 0.0,
            'f1_score': 0.0,
            'bleu_score': 0.0,
            'total_samples': total_samples,
            'samples_evaluated': 0,
            'error': str(e)
        }
