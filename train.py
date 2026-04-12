#!/usr/bin/env python3
"""Top-level training pipeline entrypoint."""

import logging
import traceback
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer

from config.data_config.master_config import DatasetSplit, DatasetType, get_dataset_config
from config.train_config.training_configs import TrainingConfig, parse_training_args
from dataload.model_processors import get_processor
from dataload.multi_task_dataset import BaseMultiTaskDataset, MultiTaskInferenceDataset, MultiTaskTrainingDataset
from models.symbolAdapter.symbol_manager import SymbolManager
from models.symbolAdapter.no_symbol_manager import NoSymbolManager
from models.symbolAdapter.symbol_training import SymbolTrainingOrchestrator
from dataload.data_utils import load_dataset

try:
    from transformers import Qwen2AudioProcessor
except ImportError:
    Qwen2AudioProcessor = None


def setup_tokenizer(model_type: str = "salmonn"):
    """Legacy tokenizer helper."""
    llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
    llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llama_tokenizer.padding_side = "right"
    return llama_tokenizer


def setup_tokenizer_and_processor(config):
    """Build tokenizer + processor for the selected backend."""
    model_type = config.model_type.value
    logging.info("Setting up tokenizer and processor for model type: %s", model_type)

    if model_type == "salmonn":
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "right"
        processor = get_processor(config.model_type.value, tokenizer=tokenizer)
        return tokenizer, processor

    if model_type == "qwen":
        input_processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
        processor = get_processor(config.model_type.value, processor=input_processor)
        tokenizer = input_processor.tokenizer
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "right"
        return tokenizer, processor

    raise ValueError(f"Unsupported model type: {model_type}")


def load_datasets_for_config(config: TrainingConfig, inference_mode: bool = False):
    """Load datasets based on train/validation/test settings."""
    dataset_type_str = config.data_config.dataset_type
    dataset_names = dataset_type_str.split("-") if "-" in dataset_type_str else [dataset_type_str]

    train_datasets = {}
    val_datasets = {}

    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            if inference_mode:
                logging.info("Loading test split for %s", dataset_name)
                full_test_dataset = load_dataset(dataset_type, split="test")
                if config.data_config.max_samples > 0:
                    test_samples = min(config.data_config.max_samples, len(full_test_dataset))
                    val_datasets[dataset_type] = full_test_dataset.select(range(test_samples))
                else:
                    val_datasets[dataset_type] = full_test_dataset
                train_datasets[dataset_type] = None
                logging.info("Loaded %s TEST: %d samples", dataset_name, len(val_datasets[dataset_type]))
            else:
                logging.info("Loading train split for %s", dataset_name)
                full_train_dataset = load_dataset(dataset_type, split="train")
                if config.data_config.max_samples > 0:
                    train_datasets[dataset_type] = full_train_dataset.select(range(config.data_config.max_samples))
                else:
                    train_datasets[dataset_type] = full_train_dataset
                logging.info("Loaded %s TRAIN: %d samples", dataset_name, len(train_datasets[dataset_type]))
        except Exception as exc:
            logging.error("Failed to load dataset %s: %s", dataset_name, exc)
            continue

    val_dataset_str = config.data_config.val_dataset_type
    val_dataset_names = val_dataset_str.split("-") if "-" in val_dataset_str else [val_dataset_str]

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
                logging.error("Failed to load validation dataset %s: %s", dataset_name, exc)
                continue

    return train_datasets, val_datasets


def create_combined_dataloader(datasets, processor, config: TrainingConfig, num_examples=5, shuffle=False):
    """Create combined dataloader from datasets using direct multi-task dataset wiring."""
    per_task_datasets = {}
    for dataset_type, task_dataset in datasets.items():
        per_task_datasets[dataset_type] = BaseMultiTaskDataset(
            dataset_type=dataset_type,
            dataset=task_dataset,
            processor=processor,
            input_mode="speech_only",
            fewshot_mode="text",
            num_examples=num_examples if num_examples is not None else config.data_config.num_examples,
            random_examples=False,
            split=DatasetSplit.TRAIN if shuffle else DatasetSplit.TEST,
            model_type=config.model_type.value,
            run_name=config.run_name,
            randomize_swap=False,
        )

    if shuffle:
        combined_dataset = MultiTaskTrainingDataset(
            datasets=per_task_datasets,
            processor=processor,
            balance_datasets=False,
            interleave=False,
        )
        batch_size = config.data_config.batch_size
    else:
        combined_dataset = MultiTaskInferenceDataset(
            datasets=per_task_datasets,
            processor=processor,
            balance_datasets=False,
            interleave=False,
        )
        batch_size = config.data_config.val_batch_size

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def extract_dataset_labels(config: TrainingConfig) -> List[str]:
    """Extract merged label vocabulary from validation datasets."""
    dataset_type_str = config.data_config.val_dataset_type
    dataset_names = dataset_type_str.split("-") if "-" in dataset_type_str else [dataset_type_str]

    all_valid_labels = set()
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            dataset_config = get_dataset_config(dataset_type)
            all_valid_labels.update(dataset_config.valid_labels)
        except Exception as exc:
            logging.warning("Could not get labels for %s: %s", dataset_name, exc)

    dataset_labels = sorted(list(all_valid_labels))
    logging.info("Extracted dataset labels: %s", dataset_labels)
    return dataset_labels


def extract_dataset_labels_dict(config: TrainingConfig) -> Dict[str, List[str]]:
    """Extract per-dataset label vocabulary."""
    dataset_type_str = config.data_config.val_dataset_type
    dataset_names = dataset_type_str.split("-") if "-" in dataset_type_str else [dataset_type_str]

    dataset_labels_dict = {}
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            dataset_config = get_dataset_config(dataset_type)
            dataset_labels_dict[dataset_name] = sorted(list(dataset_config.valid_labels))
        except Exception as exc:
            logging.warning("Could not get labels for %s: %s", dataset_name, exc)
            dataset_labels_dict[dataset_name] = []

    logging.info("Extracted dataset labels: %s", dataset_labels_dict)
    return dataset_labels_dict


def initialize_model(config: TrainingConfig, tokenizer, symbol_manager) -> torch.nn.Module:
    """Initialize backend model with LoRA parameters."""
    initial_symbol_mappings = symbol_manager.get_symbols_for_epoch(0)
    logging.info("Initial symbol mappings: %s", initial_symbol_mappings)

    model_type = config.model_type.value

    if model_type == "salmonn":
        from models.backends.custom_salmonn import CustomSalmonn

        return CustomSalmonn(
            device=config.device,
            lora=True,
            lora_rank=config.lora_config.rank,
            lora_alpha=config.lora_config.alpha,
            lora_dropout=config.lora_config.dropout,
            low_resource=False,
        )

    if model_type == "qwen":
        from models.backends.custom_qwen import CustomQwen

        return CustomQwen(
            model_path="Qwen/Qwen2-Audio-7B-Instruct",
            device=config.device,
            lora=True,
            lora_rank=config.lora_config.rank,
            lora_alpha=config.lora_config.alpha,
            lora_dropout=config.lora_config.dropout,
        )

    raise ValueError(f"Unknown model_type: {model_type}. Supported types are 'salmonn' and 'qwen'")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    try:
        setup_logging()
        logging.info("Starting orchestrator training")
        args = parse_training_args()

        config = TrainingConfig.from_args(args)
        logging.info("Training configuration: %s", config.to_dict())

        tokenizer, processor = setup_tokenizer_and_processor(config)
        dataset_labels = extract_dataset_labels(config)

        if config.symbol_config.no_symbols:
            symbol_manager = NoSymbolManager(original_labels=dataset_labels, tokenizer=tokenizer)
        else:
            symbol_manager = SymbolManager(
                original_labels=dataset_labels,
                tokenizer=tokenizer,
                dynamic_per_epoch=config.symbol_config.dynamic_symbols,
                symbol_type=config.symbol_config.symbol_type,
            )

        train_datasets, val_datasets = load_datasets_for_config(config)
        train_dataloader = create_combined_dataloader(train_datasets, processor, config, shuffle=True)
        val_dataloader = create_combined_dataloader(val_datasets, processor, config, shuffle=False)

        model = initialize_model(config, tokenizer, symbol_manager)

        orchestrator = SymbolTrainingOrchestrator(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            symbol_manager=symbol_manager,
        )
        orchestrator.run_complete_training()
        logging.info("Training completed successfully")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as exc:
        logging.error("Training failed: %s", exc)
        logging.error("Traceback: %s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
