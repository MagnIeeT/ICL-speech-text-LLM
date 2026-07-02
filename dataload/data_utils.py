import logging
from typing import Dict

from datasets import load_from_disk
from torch.utils.data import DataLoader

from config.data_config.master_config import DatasetSplit, DatasetType, get_dataset_config
from config.train_config.training_configs import TrainingConfig
from dataload.multi_task_dataset import BaseMultiTaskDataset, MultiTaskDataset, TrainingBaseDataset

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


def load_datasets_for_config(config: TrainingConfig, inference_mode: bool = False):
    """Load train/validation (or test) datasets from config."""
    dataset_type_str = config.data_config.dataset_type
    dataset_names = dataset_type_str.split("-") if "-" in dataset_type_str else [dataset_type_str]

    train_datasets = {}
    val_datasets = {}

    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            if inference_mode:
                logger.info("Loading test split for %s", dataset_name)
                full_test_dataset = load_dataset(dataset_type, split="test")
                if config.data_config.max_samples > 0:
                    test_samples = min(config.data_config.max_samples, len(full_test_dataset))
                    val_datasets[dataset_type] = full_test_dataset.select(range(test_samples))
                else:
                    val_datasets[dataset_type] = full_test_dataset
                train_datasets[dataset_type] = None
                logger.info("Loaded %s TEST: %d samples", dataset_name, len(val_datasets[dataset_type]))
            else:
                logger.info("Loading train split for %s", dataset_name)
                full_train_dataset = load_dataset(dataset_type, split="train")
                if config.data_config.max_samples > 0:
                    train_datasets[dataset_type] = full_train_dataset.select(range(config.data_config.max_samples))
                else:
                    train_datasets[dataset_type] = full_train_dataset
                logger.info("Loaded %s TRAIN: %d samples", dataset_name, len(train_datasets[dataset_type]))
        except Exception as exc:
            logger.error("Failed to load dataset %s: %s", dataset_name, exc)
            continue

    if not inference_mode:
        val_dataset_str = config.data_config.val_dataset_type or ""
        val_dataset_names = val_dataset_str.split("-") if "-" in val_dataset_str else ([val_dataset_str] if val_dataset_str else [])
    else:
        val_dataset_names = []

    for dataset_name in val_dataset_names:
        dataset_type = DatasetType(dataset_name)
        if not inference_mode:
            try:
                full_val_dataset = load_dataset(dataset_type, split="validation")
                if config.data_config.val_max_samples > 0:
                    val_samples = min(config.data_config.val_max_samples, len(full_val_dataset))
                    val_datasets[dataset_type] = full_val_dataset.select(range(val_samples))
                else:
                    val_datasets[dataset_type] = full_val_dataset
            except Exception as exc:
                logger.error("Failed to load validation dataset %s: %s", dataset_name, exc)
                continue

    return train_datasets, val_datasets


def create_combined_dataloader(
    datasets,
    processor,
    config: TrainingConfig,
    num_examples=5,
    is_training: bool = False,
    shuffle: bool = None,
    interleave: bool = None,
):
    """Create a combined dataloader from per-task datasets."""
    if shuffle is None:
        shuffle = is_training
    if interleave is None:
        interleave = is_training

    per_task_datasets = {}
    for dataset_type, task_dataset in datasets.items():
        dataset_class = TrainingBaseDataset if shuffle else BaseMultiTaskDataset
        per_task_datasets[dataset_type] = dataset_class(
            dataset_type=dataset_type,
            dataset=task_dataset,
            processor=processor,
            input_mode=config.data_config.input_mode,
            fewshot_mode=config.data_config.fewshot_mode,
            num_examples=num_examples if num_examples is not None else config.data_config.num_examples,
            random_examples=False,
            split=DatasetSplit.TRAIN if is_training else DatasetSplit.TEST,
            model_type=config.model_type.value,
            run_name=config.run_name,
            randomize_swap=False,
            is_training=is_training,
        )

    combined_dataset = MultiTaskDataset(
        datasets=per_task_datasets,
        balance_datasets=False,
        interleave=interleave,
    )
    batch_size = config.data_config.batch_size if is_training else config.data_config.val_batch_size

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,
        num_workers=config.data_config.num_workers,
        pin_memory=False,
        drop_last=shuffle,  # drop only for training, keep all samples for validation/inference
    )
    return dataloader


def clear_dataset_cache() -> int:
    """Clear dataset cache and return number of cached entries removed."""
    count = len(_DATASET_CACHE)
    _DATASET_CACHE.clear()
    return count