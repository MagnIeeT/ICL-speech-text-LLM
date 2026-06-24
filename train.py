#!/usr/bin/env python3
"""Top-level training pipeline entrypoint."""

import logging
import traceback
from typing import Dict, List

import torch
from transformers import LlamaTokenizer

from utils.environment import setup_environment
setup_environment()

from config.data_config.master_config import DatasetType, get_dataset_config
from config.train_config.training_configs import TrainingConfig, parse_training_args
from dataload.model_processors import get_processor
from models.symbolAdapter.symbol_manager import SymbolManager
from models.symbolAdapter.symbol_training import SymbolTrainingOrchestrator
from models.symbolAdapter.symbol_training_dpo import SymbolDPOOrchestrator
from models.symbolAdapter.dspo_module import setup_dspo_tokenizer, setup_dspo_model
from dataload.data_utils import create_combined_dataloader, load_datasets_for_config

def setup_tokenizer_and_processor(config):
    """Build tokenizer + processor for the selected backend."""
    model_type = config.model_type.value
    logging.info("Setting up tokenizer and processor for model type: %s", model_type)

    if model_type == "salmonn":
        tokenizer = LlamaTokenizer.from_pretrained(config.salmonn_tokenizer_name, use_fast=False)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer = setup_dspo_tokenizer(tokenizer, config)
        tokenizer.padding_side = "right"
        processor = get_processor(config.model_type.value, tokenizer=tokenizer)
        return tokenizer, processor

    if model_type == "qwen":
        try:
            from transformers import Qwen2AudioProcessor
        except (ImportError, OSError) as e:
            raise ImportError(f"Qwen2AudioProcessor is not available in this environment: {e}") from e
        input_processor = Qwen2AudioProcessor.from_pretrained(config.qwen_model_name, trust_remote_code=True)
        tokenizer = input_processor.tokenizer
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer = setup_dspo_tokenizer(tokenizer, config)
        tokenizer.padding_side = "right"
        processor = get_processor(config.model_type.value, processor=input_processor, tokenizer=tokenizer)
        return tokenizer, processor

    if model_type == "flamingo":
        from transformers import AutoProcessor as _AutoProcessor
        input_processor = _AutoProcessor.from_pretrained(config.flamingo_model_name, trust_remote_code=True)
        processor = get_processor(config.model_type.value, processor=input_processor)
        tokenizer = input_processor.tokenizer
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer = setup_dspo_tokenizer(tokenizer, config)
        tokenizer.padding_side = "right"
        return tokenizer, processor

    raise ValueError(f"Unsupported model type: {model_type}")


def extract_dataset_labels(config: TrainingConfig) -> List[str]:
    """Extract merged label vocabulary from ALL registered datasets."""
    from config.data_config.master_config import DATASET_CONFIGS
    all_valid_labels = set()
    for dataset_config in DATASET_CONFIGS.values():
        if dataset_config.valid_labels:
            all_valid_labels.update(dataset_config.valid_labels)
    return sorted(list(all_valid_labels))


def extract_dataset_labels_dict(config: TrainingConfig) -> Dict[str, List[str]]:
    """Extract per-dataset label vocabulary for all registered datasets."""
    from config.data_config.master_config import DATASET_CONFIGS
    dataset_labels_dict = {}
    for name, dataset_config in DATASET_CONFIGS.items():
        if dataset_config.valid_labels:
            dataset_labels_dict[name.value] = sorted(list(dataset_config.valid_labels))
    return dataset_labels_dict


def initialize_model(config: TrainingConfig, tokenizer, symbol_manager, raw_processor) -> torch.nn.Module:
    """Initialize backend model with LoRA parameters."""
    model_type = config.model_type.value
    if model_type == "salmonn":
        from models.backends.custom_salmonn import CustomSalmonn
        model = CustomSalmonn(device=config.device, lora=True, lora_rank=config.lora_config.rank,
                              lora_alpha=config.lora_config.alpha, lora_dropout=config.lora_config.dropout)
        return setup_dspo_model(model, tokenizer, config)
    
    if model_type == "qwen":
        from models.backends.custom_qwen import CustomQwen
        model = CustomQwen(model_path=config.qwen_model_name, input_processor=raw_processor, 
                           device=config.device, lora=True, lora_rank=config.lora_config.rank, 
                           lora_alpha=config.lora_config.alpha, lora_dropout=config.lora_config.dropout)
        return setup_dspo_model(model, tokenizer, config)

    if model_type == "flamingo":
        from models.backends.custom_flamingo import CustomFlamingo
        model = CustomFlamingo(model_path=config.flamingo_model_name, device=config.device, lora=True,
                               lora_rank=config.lora_config.rank, lora_alpha=config.lora_config.alpha,
                               lora_dropout=config.lora_config.dropout)
        return setup_dspo_model(model, tokenizer, config)

    raise ValueError(f"Unknown model_type: {model_type}")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()], force=True)


def main():
    try:
        setup_logging()
        args = parse_training_args()
        config = TrainingConfig.from_args(args)
        logging.info("Starting training with config: %s", config.run_name)

        if config.diff_symbol_config.enabled:
            from config.data_config.master_config import DATASET_CONFIGS
            ds_names = set(config.data_config.dataset_type.split("-"))
            total_training_labels = sum(
                len(ds_cfg.valid_labels)
                for dt, ds_cfg in DATASET_CONFIGS.items()
                if dt.value in ds_names
            )
            if config.diff_symbol_config.num_slots < total_training_labels:
                config.diff_symbol_config.num_slots = total_training_labels
                logging.info("D-SPO: num_slots raised to %d (minimum = total training labels)", total_training_labels)
            else:
                logging.info("D-SPO: num_slots=%d (pool), %d training labels", config.diff_symbol_config.num_slots, total_training_labels)

        tokenizer, processor = setup_tokenizer_and_processor(config)
        
        dataset_labels = extract_dataset_labels(config)
        symbol_manager = SymbolManager(
            original_labels=dataset_labels,
            tokenizer=tokenizer,
            dynamic_per_epoch=config.symbol_config.dynamic_symbols,
            token_size=config.symbol_config.symbol_token_size,
            no_symbols=config.symbol_config.no_symbols,
            swap_labels=config.symbol_config.swap_labels,
        )

        train_datasets, val_datasets = load_datasets_for_config(config)
        train_dataset_names = {dt.value if hasattr(dt, "value") else str(dt) for dt, ds in train_datasets.items() if ds is not None}

        train_dataloader = create_combined_dataloader(train_datasets, processor, config, num_examples=config.data_config.num_examples, is_training=True)
        val_dataloader = create_combined_dataloader(val_datasets, processor, config, num_examples=config.data_config.val_num_examples, is_training=False)

        # Retrieve raw_processor for backend initialization
        raw_processor = processor.processor if hasattr(processor, "processor") else None
        model = initialize_model(config, tokenizer, symbol_manager, raw_processor)

        if getattr(args, "use_dpo", False):
            orchestrator = SymbolDPOOrchestrator(
                config=config, model=model, processor=processor,
                train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                tokenizer=tokenizer, symbol_manager=symbol_manager,
                train_dataset_names=train_dataset_names,
                beta=getattr(args, "dpo_beta", 0.1),
            )
        else:
            orchestrator = SymbolTrainingOrchestrator(
                config=config, model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                tokenizer=tokenizer, symbol_manager=symbol_manager, train_dataset_names=train_dataset_names,
                processor=processor
            )
        orchestrator.run_complete_training()
        logging.info("Training completed successfully")

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as exc:
        logging.error("Training failed: %s", exc)
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
