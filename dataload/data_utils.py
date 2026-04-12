import logging
from typing import Dict

from datasets import load_from_disk

from config.data_config.master_config import DatasetSplit, DatasetType, get_dataset_config

logger = logging.getLogger(__name__)

_DATASET_CACHE: Dict[str, object] = {}


def load_dataset(dataset_type: DatasetType, split: str = "train", use_cache: bool = True):
    """Load a configured dataset split from disk with optional in-memory caching."""
    split_enum = DatasetSplit(split) if isinstance(split, str) else split
    cache_key = f"{dataset_type.value}_{split_enum.value}"

    if use_cache and cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    config = get_dataset_config(dataset_type)
    dataset_path = config.get_path(split_enum)
    data = load_from_disk(dataset_path)

    logger.info("Loaded %d examples from %s %s", len(data), dataset_type.value, split_enum.value)
    if use_cache:
        _DATASET_CACHE[cache_key] = data
    return data


def clear_dataset_cache() -> int:
    """Clear dataset cache and return number of cached entries removed."""
    count = len(_DATASET_CACHE)
    _DATASET_CACHE.clear()
    return count
