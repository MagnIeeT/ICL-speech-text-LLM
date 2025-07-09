import os
import logging
import json
import torch
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Any, Optional, List, Union, Callable
import random
import traceback
from datasets import load_from_disk

from data.master_config import DatasetType, DatasetSplit

logger = logging.getLogger(__name__)

# Global cache for datasets to avoid reloading
_DATASET_CACHE = {}
_AUDIO_LOOKUP_CACHE = {}

def load_dataset(dataset_type: DatasetType, split: str = "train", use_cache: bool = True):
    """
    Load a dataset based on its type and split.
    
    Args:
        dataset_type: Type of dataset to load
        split: Dataset split to load ("train", "val", or "test")
        use_cache: Whether to use the in-memory cache
        
    Returns:
        The loaded dataset
        
    Raises:
        ValueError: If the dataset type is unsupported
        FileNotFoundError: If the dataset file doesn't exist
    """
    # Convert string split to enum
    if isinstance(split, str):
        split = DatasetSplit(split)
    
    # Check cache first
    cache_key = f"{dataset_type.value}_{split.value}"
    if use_cache and cache_key in _DATASET_CACHE:
        logger.debug(f"Using cached dataset for {dataset_type} {split}")
        return _DATASET_CACHE[cache_key]
    
    try:
        # Get dataset config
        from data.master_config import get_dataset_config
        # Map variant types to base type for dataset loading
        base_type = dataset_type
        if dataset_type in [DatasetType.VOXCELEB_GREEK, DatasetType.VOXCELEB_SWAP]:
            base_type = DatasetType.VOXCELEB
        elif dataset_type in [DatasetType.HVB_GREEK, DatasetType.HVB_SWAP]:
            base_type = DatasetType.HVB
        elif dataset_type in [DatasetType.VOXPOPULI_GREEK, DatasetType.VOXPOPULI_SWAP]:
            base_type = DatasetType.VOXPOPULI
        elif dataset_type == DatasetType.SQA:
            base_type = DatasetType.SQA
        elif dataset_type == DatasetType.VP_NEL:
            base_type = DatasetType.VP_NEL
        elif dataset_type in [DatasetType.MELD_EMOTION_GREEK, DatasetType.MELD_EMOTION_SWAP]:
            base_type = DatasetType.MELD_EMOTION
        
        config = get_dataset_config(base_type)
        
        # Get dataset path
        dataset_path = config.get_path(split)
        
        logger.info(f"Loading dataset {dataset_type} {split} from {dataset_path}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load dataset
        start_time = time.time()
        
        # Use load_from_disk for Hugging Face datasets
        data = load_from_disk(dataset_path)
        
        logger.info(f"Loaded {len(data)} examples from {dataset_type} {split} in {time.time() - start_time:.2f}s")
        
        # Cache the dataset
        if use_cache:
            _DATASET_CACHE[cache_key] = data
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_type} {split}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise



def clear_dataset_cache():
    """
    Clear the dataset cache to free memory.
    
    Returns:
        Number of items cleared from cache
    """
    global _DATASET_CACHE, _AUDIO_LOOKUP_CACHE
    dataset_count = len(_DATASET_CACHE)
    audio_lookup_count = len(_AUDIO_LOOKUP_CACHE)
    
    _DATASET_CACHE.clear()
    _AUDIO_LOOKUP_CACHE.clear()
    
    logger.info(f"Dataset cache cleared: {dataset_count} datasets, {audio_lookup_count} audio lookups")
    return dataset_count + audio_lookup_count

def get_dataset_sample(dataset_type: DatasetType, split: str = "train", n_samples: int = 5, seed: Optional[int] = None):
    """
    Get a random sample from a dataset for inspection.
    
    Args:
        dataset_type: Type of dataset
        split: Dataset split
        n_samples: Number of samples to return
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled dataset items
    """
    try:
        # Load dataset
        data = load_dataset(dataset_type, split)
        
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Sample items
        if len(data) <= n_samples:
            return data
        
        return random.sample(data, n_samples)
    
    except Exception as e:
        logger.error(f"Error getting dataset sample: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

@lru_cache(maxsize=32)
def get_dataset_stats(dataset_type: DatasetType, split: str = "train"):
    """
    Get statistics about a dataset.
    
    Args:
        dataset_type: Type of dataset
        split: Dataset split
        
    Returns:
        Dictionary with dataset statistics
    """
    try:
        # Load dataset
        data = load_dataset(dataset_type, split)
        
        # Compute statistics
        stats = {
            "dataset_type": dataset_type.value,
            "split": split,
            "num_examples": len(data),
        }
        
        # Get label distribution if available
        from data.master_config import get_dataset_config
        config = get_dataset_config(dataset_type)
        
        if hasattr(config, "completion_key") and config.completion_key:
            label_counts = {}
            for item in data:
                if config.completion_key in item:
                    label = item[config.completion_key]
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            stats["label_distribution"] = label_counts
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

def validate_dataset(dataset_type: DatasetType, split: str = "train"):
    """
    Validate a dataset for common issues.
    
    Args:
        dataset_type: Type of dataset
        split: Dataset split
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Load dataset
        data = load_dataset(dataset_type, split)
        
        # Get dataset config
        from data.master_config import get_dataset_config
        config = get_dataset_config(dataset_type)
        
        # Check for required fields
        missing_fields = {}
        required_fields = [config.completion_key, config.text_key]
        
        for item_idx, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    if field not in missing_fields:
                        missing_fields[field] = []
                    missing_fields[field].append(item_idx)
        
        # Check for audio paths if using audio
        missing_audio = []
        if config.audio_lookup_paths:
            for item_idx, item in enumerate(data):
                if "audio_path" not in item:
                    missing_audio.append(item_idx)
        
        return {
            "dataset_type": dataset_type.value,
            "split": split,
            "num_examples": len(data),
            "missing_fields": missing_fields,
            "missing_audio": missing_audio,
            "is_valid": len(missing_fields) == 0 and len(missing_audio) == 0
        }
    
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e), "is_valid": False}
