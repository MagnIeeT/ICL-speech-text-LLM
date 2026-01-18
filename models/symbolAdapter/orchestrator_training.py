#!/usr/bin/env python3
"""
Orchestrator Training Script
Hybrid Version: Intern's Structure + Qwen Fix + Crash Safety
"""

import os
import sys
import traceback
import logging
import torch
from datetime import datetime
from typing import List, Dict

# Add parent directory to path for imports
ICL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ICL_ROOT)

# Import the orchestrator and configurations
from models.symbolAdapter.training.symbol_training import SymbolTrainingOrchestrator
from models.symbolAdapter.configs.training_configs import TrainingConfig, parse_training_args, TrainingMode

# --- MODIFIED: Commented out top-level import to prevent crash in Qwen env ---
# from models.mlp_salmonn import MLPSalmonn
# -----------------------------------------------------------------------------

from models.symbolAdapter.symbol_manager import SymbolManager

# Import data utilities
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType, get_dataset_config
from data.model_processors import get_processor
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, Qwen2AudioProcessor

# --- NEW: Import CustomQwen to prevent crashes ---
#from models.custom_qwen import CustomQwen
# -------------------------------------------------

def setup_tokenizer(model_type: str = "salmonn"):
    """Setup tokenizer (Legacy helper)"""
    llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
    llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.padding_side = "right"
    return llama_tokenizer

def setup_tokenizer_and_processor(config):
    """
    Returns:
        tokenizer: Tokenizer to be used for SymbolManager
        processor: Processor to be used for DatasetFactory
    """
    model_type = config.model_type.value
    logging.info(f"Setting up tokenizer and processor for model type: {model_type}")

    if model_type == "salmonn":
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = "right"
        
        processor = get_processor(config.model_type.value, tokenizer=tokenizer)
        return tokenizer, processor

    elif model_type == "qwen":
        input_processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
        processor = get_processor(config.model_type.value, processor=input_processor)
        
        # For Qwen, tokenizer comes from processor
        tokenizer = input_processor.tokenizer
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = "right"
        return tokenizer, processor

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_datasets_for_config(config: TrainingConfig, inference_mode: bool = False):
    """Load datasets based on configuration"""
    dataset_type_str = config.data_config.dataset_type
    dataset_names = dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str]
    
    train_datasets = {}
    val_datasets = {}
    
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            
            if inference_mode:
                # ✅ INFERENCE MODE: Load only test split
                logging.info(f"🔍 Loading test split for {dataset_name}")
                full_test_dataset = load_dataset(dataset_type, split="test")
                
                if config.data_config.max_samples > 0:
                    test_samples = min(config.data_config.max_samples, len(full_test_dataset))
                    val_datasets[dataset_type] = full_test_dataset.select(range(test_samples))
                else:
                    val_datasets[dataset_type] = full_test_dataset
                
                train_datasets[dataset_type] = None 
                logging.info(f"✓ Loaded {dataset_name} TEST: {len(val_datasets[dataset_type])} samples")
                
            else:
                # ✅ TRAINING MODE: Load train and validation splits
                logging.info(f"📚 Loading train/val splits for {dataset_name}")
                full_train_dataset = load_dataset(dataset_type, split="train")
                
                if config.data_config.max_samples > 0:
                    train_datasets[dataset_type] = full_train_dataset.select(range(config.data_config.max_samples))
                else:
                    train_datasets[dataset_type] = full_train_dataset
                logging.info(f"✓ Loaded {dataset_name} Train: {len(train_datasets[dataset_type])} samples")
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            continue
    
    
    val_dataset_str = config.data_config.val_dataset_type
    val_dataset_names = val_dataset_str.split('-') if '-' in val_dataset_str else [val_dataset_str]

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
            except Exception as e:
                logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
                continue

    return train_datasets, val_datasets 
 

def create_combined_dataloader(datasets, processor, config: TrainingConfig, num_examples=5, shuffle=False):
    """Create combined dataloader from datasets"""
    dataset_types = list(datasets.keys())
    
    # --- SAFETY FIX: Ensure SALMONN behaves EXACTLY like before ---
    if config.model_type.value == "qwen":
        # Qwen needs text instructions + audio
        input_mode_setting = "speech_and_text"
    else:
        # SALMONN (Your Sir's Model) stays strictly "speech_only"
        # This guarantees your results will NOT vary.
        input_mode_setting = "speech_only"
    # -------------------------------------------------------------
    
    combined_dataset = DatasetFactory.create_dataset(
        dataset_type=dataset_types,
        dataset=datasets,
        processor=processor,
        is_training=shuffle,
        input_mode=input_mode_setting,  # <--- Uses the safe setting logic above
        fewshot_mode="text",
        num_examples=num_examples if num_examples is not None else config.data_config.num_examples,
        random_examples=False,
        model_type=config.model_type.value,
        run_name=config.run_name,
        randomize_swap=False,
        balance_datasets=False,
        interleave=False
    )
    
    batch_size = config.data_config.batch_size if not shuffle else config.data_config.val_batch_size
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def extract_dataset_labels(config: TrainingConfig) -> List[str]:
    """Extract dataset labels once - centralized function"""
    dataset_type_str = config.data_config.val_dataset_type
    dataset_names = dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str]
    
    all_valid_labels = set()
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            dataset_config = get_dataset_config(dataset_type)
            all_valid_labels.update(dataset_config.valid_labels)
        except Exception as e:
            logging.warning(f"Could not get labels for {dataset_name}: {e}")
            
    dataset_labels = sorted(list(all_valid_labels))
    logging.info(f"Extracted dataset labels: {dataset_labels}")
    return dataset_labels

def extract_dataset_labels_dict(config: TrainingConfig) -> Dict[str, List[str]]:
    """Extract dataset labels once - centralized function"""
    dataset_type_str = config.data_config.val_dataset_type
    dataset_names = dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str]
    
    dataset_labels_dict = {}
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            dataset_config = get_dataset_config(dataset_type)
            dataset_labels_dict[dataset_name] = sorted(list(dataset_config.valid_labels))
        except Exception as e:
            logging.warning(f"Could not get labels for {dataset_name}: {e}")
            dataset_labels_dict[dataset_name] = []
    
    logging.info(f"Extracted dataset labels: {dataset_labels_dict}")
    return dataset_labels_dict

def initialize_model(config: TrainingConfig, tokenizer, symbol_manager) -> torch.nn.Module:
    """Initialize the model based on model_type"""
    
    initial_symbol_mappings = symbol_manager.get_symbols_for_epoch(0)
    logging.info(f"Initial symbol mappings: {initial_symbol_mappings}")
    
    model_type = config.model_type.value

    if model_type == "salmonn":
        # --- MODIFIED: Import moved here (Lazy Import) ---
        from models.mlp_salmonn import MLPSalmonn
        # -------------------------------------------------
        
        model = MLPSalmonn(
            device=config.device,
            lora=True,
            lora_rank=config.lora_config.rank,
            lora_alpha=config.lora_config.alpha,
            lora_dropout=config.lora_config.dropout,
            low_resource=False,
        )
        return model

    elif model_type == "qwen":
        from models.custom_qwen import CustomQwen

        model = CustomQwen(
            model_path="Qwen/Qwen2-Audio-7B-Instruct",
            device=config.device,
            lora=True,
            lora_rank=config.lora_config.rank,
            lora_alpha=config.lora_config.alpha,
            lora_dropout=config.lora_config.dropout,
        )
        return model

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported types are 'salmonn' and 'qwen'")

def setup_logging() -> str: 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    ) 
    return None 


def main():
    try:
        setup_logging()
        logging.info("🚀 Starting orchestrator training...") 
        logging.info("📋 Parsing arguments...")
        args = parse_training_args()

        config = TrainingConfig.from_args(args)
        logging.info(f"Training configuration: {config.to_dict()}")

        tokenizer, processor = setup_tokenizer_and_processor(config)
        logging.info("✓ Tokenizer and Processor initialized")

        dataset_labels = extract_dataset_labels(config)
         
        # --- CRASH PREVENTION ---
        # Arguments 'original_labels_dict' and 'only_original' are commented out
        # because your SymbolManager file does not support them yet.
        
        symbol_manager = SymbolManager(
            original_labels=dataset_labels,
            tokenizer=tokenizer,
            dynamic_per_epoch=(config.symbol_config.mode.value == "dynamic_per_epoch"),
            symbol_type=config.symbol_config.symbol_type,
            # original_labels_dict=extract_dataset_labels_dict(config),  <-- COMMENTED OUT FOR SAFETY
            # only_original=config.symbol_config.only_original           <-- COMMENTED OUT FOR SAFETY
        )
        logging.info("✓ SymbolManager initialized")
        
        train_datasets, val_datasets = load_datasets_for_config(config) 
        
        train_dataloader = create_combined_dataloader(train_datasets, processor, config, shuffle=True)
        val_dataloader = create_combined_dataloader(val_datasets, processor, config, shuffle=False)
        
        model = initialize_model(config, tokenizer, symbol_manager) 
        logging.info("✓ Model initialized")
        
        logging.info("🚀 Starting Symbol Training Orchestrator...")  
        
        try: 
            orchestrator = SymbolTrainingOrchestrator( 
                config=config,
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                symbol_manager=symbol_manager
            )
            
            orchestrator.run_complete_training()
            logging.info("✅ Training completed successfully!")  
            
        except KeyboardInterrupt:
            logging.info("❌ Training interrupted by user")
                
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()