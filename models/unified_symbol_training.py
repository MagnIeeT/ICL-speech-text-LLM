#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import torch
import json
from datetime import datetime
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn, generate_one_word_two_token_symbols, create_label_mapping
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType, get_dataset_config
from utils.training_utils import setup_logging, save_checkpoint
from torch.utils.data import DataLoader
from data.model_processors import get_processor
from utils.evaluation_utils import evaluate_predictions, clean_prediction

def parse_args():
    parser = argparse.ArgumentParser(description="Unified MLP and LoRA Training")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="salmonn")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    
    # FIXED: Change from expecting arguments to using store_true
    parser.add_argument("--use_output_mlp", action="store_true", help="Use output MLP in architecture")
    parser.add_argument("--bypass_mlp", action="store_true", help="Bypass MLP entirely - pure LoRA training")
    
    # Training hyperparameters
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--mlp_lr", type=float, default=1e-4)
    parser.add_argument("--lora_epochs", type=int, default=2)
    parser.add_argument("--mlp_epochs", type=int, default=2)
    parser.add_argument("--total_cycles", type=int, default=3)
    parser.add_argument("--lora_final_epochs", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb_greek")
    parser.add_argument("--max_samples", type=int, default=64)
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/home/leapers/weights/neeraja/ICL/unified_training")
    parser.add_argument("--run_name", type=str, default="")
    
    # NEW: Schedule type parameter
    parser.add_argument("--schedule_type", type=str, default="bypass_mlp", 
                       choices=["simplified", "mlp_first", "bypass_mlp"],
                       help="Training schedule type")
    
    return parser.parse_args()

def setup_unified_logging(args):
    """Setup logging with timestamp"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not args.run_name:
        args.run_name = f"simplified_{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Arguments: {args}")
    return args

def train_mlp_phase_simplified(model, dataloader, args, cycle, num_epochs, current_mappings, val_dataloader=None):
    """Simplified MLP training with validation"""
    
    logging.info(f"=== MLP Training: {num_epochs} epochs ===")
    
    model.train()
    model.set_mlp_training_mode()
    
    # Track metrics for each epoch
    epoch_metrics = []
    
    # Get MLP parameters correctly
    mlp_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ('input_mlp' in name or 'output_mlp' in name):
            mlp_params.append(param)
    
    optimizer = torch.optim.Adam(mlp_params, lr=args.mlp_lr, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        accumulated_loss = 0
        
        logging.info(f"MLP Cycle {cycle} Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"MLP C{cycle} E{epoch+1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                model.step_counter = step + epoch * len(dataloader)
                
                if step == 0:
                    logging.info("=== STEP 1 BATCH CONTENT ===")
                    if "prompt" in batch:
                        logging.info("ORIGINAL PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"][:2]):
                            logging.info(f"  [{i}] {prompt}")
                    if "completion" in batch:
                        logging.info("ORIGINAL COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"][:2]):
                            logging.info(f"  [{i}] {completion}")
                    logging.info(f"FIXED SYMBOL MAPPINGS: {current_mappings}")
                
                batch = replace_symbols_in_batch(batch, current_mappings)
                
                if step == 0:
                    logging.info("UPDATED AFTER SYMBOL REPLACEMENT:")
                    if "prompt" in batch:
                        logging.info("UPDATED PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"][:2]):
                            logging.info(f"  [{i}] {prompt}")
                    if "completion" in batch:
                        logging.info("UPDATED COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"][:2]):
                            logging.info(f"  [{i}] {completion}")
                    logging.info("=== END STEP 1 BATCH CONTENT ===")
                
                if step % 3 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(batch)
                loss = outputs["loss"]
                
                if torch.isnan(loss).any() or loss.item() == 0.0:
                    logging.error(f"Invalid loss at step {step}: {loss.item()} - skipping batch")
                    optimizer.zero_grad()
                    continue
                
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                
                accumulated_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    has_nan_grad = False
                    total_norm = 0
                    for param in mlp_params:  # FIXED: Use mlp_params list
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2)
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logging.error(f"NaN/Inf gradient detected")
                                has_nan_grad = True
                            else:
                                total_norm += grad_norm.item() ** 2
                    
                    total_norm = total_norm ** 0.5
                    
                    if has_nan_grad:
                        logging.warning(f"NaN gradient detected - zeroing gradients")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    torch.nn.utils.clip_grad_norm_(mlp_params, args.max_grad_norm)  # FIXED
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    avg_accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
                    progress_bar.set_postfix({"acc_loss": f"{avg_accumulated_loss:.6f}"})
                    
                    total_loss += accumulated_loss
                    valid_batches += 1
                    accumulated_loss = 0
                
            except Exception as e:
                logging.error(f"Error in MLP batch {step}: {e}")
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # FIXED: Handle remaining gradients
        if accumulated_loss > 0:
            torch.nn.utils.clip_grad_norm_(mlp_params, args.max_grad_norm)  # FIXED
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            valid_batches += 1
        
        progress_bar.close()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            logging.info(f"✓ MLP Epoch {epoch+1} completed: Average loss = {epoch_loss:.6f}")
            
            # Run validation and collect metrics
            if val_dataloader is not None:
                epoch_result = {}
                
                # Test WITH MLP + SYMBOLS
                val_score_mlp_symbols = validate_model(model, val_dataloader, args, current_mappings, "MLP", epoch+1, bypass_mlp=False, use_original_labels=False)
                epoch_result['mlp_symbols'] = val_score_mlp_symbols
                
                # Test WITHOUT MLP + SYMBOLS
                val_score_no_mlp_symbols = validate_model(model, val_dataloader, args, current_mappings, "MLP", epoch+1, bypass_mlp=True, use_original_labels=False)
                epoch_result['no_mlp_symbols'] = val_score_no_mlp_symbols
                
                # Test WITH MLP + ORIGINAL LABELS
                val_score_mlp_original = validate_model(model, val_dataloader, args, current_mappings, "MLP", epoch+1, bypass_mlp=False, use_original_labels=True)
                epoch_result['mlp_original'] = val_score_mlp_original
                
                # Test WITHOUT MLP + ORIGINAL LABELS
                val_score_no_mlp_original = validate_model(model, val_dataloader, args, current_mappings, "MLP", epoch+1, bypass_mlp=True, use_original_labels=True)
                epoch_result['no_mlp_original'] = val_score_no_mlp_original
                
                # Store epoch results
                epoch_result['epoch'] = epoch + 1
                epoch_result['cycle'] = cycle
                epoch_result['phase'] = 'MLP'
                epoch_result['loss'] = epoch_loss
                epoch_metrics.append(epoch_result)
                
                # ✅ NEW: Log individual epoch summary immediately
                logging.info("=" * 60)
                logging.info(f"MLP CYCLE {cycle} EPOCH {epoch+1} SUMMARY:")
                logging.info("=" * 60)
                logging.info(f"Training Loss: {epoch_loss:.6f}")
                logging.info(f"Validation Scores:")
                logging.info(f"  └─ Symbols: With MLP {val_score_mlp_symbols:.4f}, Without MLP {val_score_no_mlp_symbols:.4f}")
                logging.info(f"  └─ Original: With MLP {val_score_mlp_original:.4f}, Without MLP {val_score_no_mlp_original:.4f}")
                logging.info("=" * 60)
    
    # ✅ ENHANCED: Log complete cycle summary at the end
    if epoch_metrics:
        logging.info("=" * 80)
        logging.info(f"MLP CYCLE {cycle} COMPLETE SUMMARY:")
        logging.info("=" * 80)
        logging.info(f"{'Epoch':<5} {'Loss':<10} {'MLP+Sym':<9} {'NoMLP+Sym':<10} {'MLP+Orig':<9} {'NoMLP+Orig':<10}")
        logging.info("-" * 80)
        for result in epoch_metrics:
            logging.info(f"{result['epoch']:<5} {result['loss']:<10.4f} "
                        f"{result['mlp_symbols']:<9.4f} {result['no_mlp_symbols']:<10.4f} "
                        f"{result['mlp_original']:<9.4f} {result['no_mlp_original']:<10.4f}")
        
        # Find best epoch in this cycle
        best_epoch = max(epoch_metrics, key=lambda x: x['mlp_symbols'])
        logging.info("-" * 80)
        logging.info(f"Best MLP+Symbols: Epoch {best_epoch['epoch']} = {best_epoch['mlp_symbols']:.4f}")
        logging.info("=" * 80)
    
    return current_mappings, epoch_metrics

def train_lora_phase_unified(model, dataloader, args, cycle, current_mappings, num_epochs, val_dataloader=None, 
                            dynamic_symbols=False, llama_tokenizer=None, dataset_labels=None):
    """Unified LoRA training with optional dynamic symbol generation per epoch"""
    
    if dynamic_symbols:
        logging.info(f"=== LoRA Training with Dynamic Symbols: {num_epochs} epochs ===")
        if not llama_tokenizer or not dataset_labels:
            raise ValueError("dynamic_symbols=True requires llama_tokenizer and dataset_labels")
    else:
        logging.info(f"=== LoRA Training with Fixed Symbols: {num_epochs} epochs ===")
    
    model.train()
    model.set_lora_training_mode()
    
    # Track metrics for each epoch
    epoch_metrics = []
    
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.lora_lr,
        weight_decay=1e-5
    )
    
    for epoch in range(num_epochs):
        # Handle symbol generation based on mode
        if dynamic_symbols:
            # Generate fresh symbols for each epoch
            logging.info(f"=== Generating NEW symbols for Epoch {epoch+1} ===")
            new_random_symbols = generate_one_word_two_token_symbols(len(dataset_labels), llama_tokenizer)
            epoch_mappings = create_label_mapping(dataset_labels, new_random_symbols)
            
            logging.info(f"Epoch {epoch+1} Symbol Mappings: {epoch_mappings}")
            
            # Update model with new symbols (no MLP retraining needed since bypassed)
            model.update_label_tokens(epoch_mappings)
        else:
            # Use provided fixed mappings
            epoch_mappings = current_mappings
        
        model.train()
        total_loss = 0
        valid_batches = 0
        accumulated_loss = 0
        
        # Create appropriate progress bar description
        if dynamic_symbols:
            desc = f"LoRA C{cycle} E{epoch+1} Fresh-Sym"
        else:
            desc = f"LoRA C{cycle} E{epoch+1}"
        
        logging.info(f"LoRA Cycle {cycle} Epoch {epoch+1}/{num_epochs} {'(Fresh Symbols)' if dynamic_symbols else '(Fixed Symbols)'}")
        progress_bar = tqdm(dataloader, desc=desc)
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                model.step_counter = step + epoch * len(dataloader)
                
                if step == 0:
                    if dynamic_symbols:
                        logging.info("=== STEP 1 BATCH CONTENT (FRESH SYMBOLS) ===")
                    else:
                        logging.info("=== STEP 1 BATCH CONTENT (FIXED SYMBOLS) ===")
                    
                    if "prompt" in batch:
                        logging.info("ORIGINAL PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"][:2]):
                            logging.info(f"  [{i}] {prompt}")
                    if "completion" in batch:
                        logging.info("ORIGINAL COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"][:2]):
                            logging.info(f"  [{i}] {completion}")
                    
                    if dynamic_symbols:
                        logging.info(f"FRESH SYMBOL MAPPINGS: {epoch_mappings}")
                    else:
                        logging.info(f"FIXED SYMBOL MAPPINGS: {epoch_mappings}")
                
                # Apply symbol replacements (same logic for both modes)
                batch = replace_symbols_in_batch(batch, epoch_mappings)
                
                if step == 0:
                    if dynamic_symbols:
                        logging.info("UPDATED WITH FRESH SYMBOLS:")
                    else:
                        logging.info("UPDATED WITH FIXED SYMBOLS:")
                    
                    if "prompt" in batch:
                        logging.info("UPDATED PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"][:2]):
                            logging.info(f"  [{i}] {prompt}")
                    if "completion" in batch:
                        logging.info("UPDATED COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"][:2]):
                            logging.info(f"  [{i}] {completion}")
                    logging.info("=== END STEP 1 BATCH CONTENT ===")
                
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(batch)
                loss = outputs["loss"]
                
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                
                accumulated_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    avg_accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
                    
                    if dynamic_symbols:
                        progress_bar.set_postfix({"acc_loss": f"{avg_accumulated_loss:.4f}", "symbols": f"epoch{epoch+1}"})
                    else:
                        progress_bar.set_postfix({"acc_loss": f"{avg_accumulated_loss:.4f}"})
                    
                    total_loss += accumulated_loss
                    valid_batches += 1
                    accumulated_loss = 0
                
            except Exception as e:
                logging.error(f"Error in LoRA batch {step}: {e}")
                continue
        
        if accumulated_loss > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            valid_batches += 1
        
        progress_bar.close()
        
        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            
            if dynamic_symbols:
                logging.info(f"✓ LoRA Epoch {epoch+1} (Fresh Symbols) completed: Average loss = {epoch_loss:.4f}")
            else:
                logging.info(f"✓ LoRA Epoch {epoch+1} (Fixed Symbols) completed: Average loss = {epoch_loss:.4f}")
            
            # Run validation with current epoch's symbols
            if val_dataloader is not None:
                epoch_result = {}
                
                if dynamic_symbols:
                    # Dynamic mode: Test fresh symbols vs original labels
                    val_score_symbols = validate_model(model, val_dataloader, args, epoch_mappings, "LoRA-Dynamic", epoch+1, bypass_mlp=True, use_original_labels=False)
                    val_score_original = validate_model(model, val_dataloader, args, epoch_mappings, "LoRA-Dynamic", epoch+1, bypass_mlp=True, use_original_labels=True)
                    
                    epoch_result['fresh_symbols'] = val_score_symbols
                    epoch_result['original_labels'] = val_score_original
                    epoch_result['phase'] = 'LoRA-Dynamic'
                    epoch_result['symbol_mappings'] = epoch_mappings.copy()
                    
                    # ✅ NEW: Log individual epoch summary immediately
                    logging.info("=" * 60)
                    logging.info(f"LoRA DYNAMIC CYCLE {cycle} EPOCH {epoch+1} SUMMARY:")
                    logging.info("=" * 60)
                    logging.info(f"Training Loss: {epoch_loss:.4f}")
                    logging.info(f"Validation Scores:")
                    logging.info(f"  └─ Fresh Symbols: {val_score_symbols:.4f}")
                    logging.info(f"  └─ Original Labels: {val_score_original:.4f}")
                    logging.info(f"  └─ Symbols Used: {epoch_mappings}")
                    logging.info("=" * 60)
                    
                else:
                    # Fixed mode: Test all four combinations
                    val_score_mlp_symbols = validate_model(model, val_dataloader, args, current_mappings, "LoRA", epoch+1, bypass_mlp=False, use_original_labels=False)
                    val_score_no_mlp_symbols = validate_model(model, val_dataloader, args, current_mappings, "LoRA", epoch+1, bypass_mlp=True, use_original_labels=False)
                    val_score_mlp_original = validate_model(model, val_dataloader, args, current_mappings, "LoRA", epoch+1, bypass_mlp=False, use_original_labels=True)
                    val_score_no_mlp_original = validate_model(model, val_dataloader, args, current_mappings, "LoRA", epoch+1, bypass_mlp=True, use_original_labels=True)
                    
                    epoch_result['mlp_symbols'] = val_score_mlp_symbols
                    epoch_result['no_mlp_symbols'] = val_score_no_mlp_symbols
                    epoch_result['mlp_original'] = val_score_mlp_original
                    epoch_result['no_mlp_original'] = val_score_no_mlp_original
                    epoch_result['phase'] = 'LoRA'
                    
                    # ✅ NEW: Log individual epoch summary immediately
                    logging.info("=" * 60)
                    logging.info(f"LoRA FIXED CYCLE {cycle} EPOCH {epoch+1} SUMMARY:")
                    logging.info("=" * 60)
                    logging.info(f"Training Loss: {epoch_loss:.4f}")
                    logging.info(f"Validation Scores:")
                    logging.info(f"  └─ Symbols: With MLP {val_score_mlp_symbols:.4f}, Without MLP {val_score_no_mlp_symbols:.4f}")
                    logging.info(f"  └─ Original: With MLP {val_score_mlp_original:.4f}, Without MLP {val_score_no_mlp_original:.4f}")
                    logging.info("=" * 60)
                
                # Common epoch result fields
                epoch_result['epoch'] = epoch + 1
                epoch_result['cycle'] = cycle
                epoch_result['loss'] = epoch_loss
                epoch_metrics.append(epoch_result)
    
    # ✅ ENHANCED: Log complete cycle summary at the end (keep existing logic)
    if epoch_metrics:
        logging.info("=" * 80)
        if dynamic_symbols:
            logging.info(f"LoRA DYNAMIC SYMBOLS CYCLE {cycle} COMPLETE SUMMARY:")
            logging.info("=" * 80)
            logging.info(f"{'Epoch':<5} {'Loss':<10} {'Fresh-Sym':<10} {'Original':<10}")
            logging.info("-" * 80)
            for result in epoch_metrics:
                logging.info(f"{result['epoch']:<5} {result['loss']:<10.4f} "
                            f"{result['fresh_symbols']:<10.4f} {result['original_labels']:<10.4f}")
                logging.info(f"      Symbols: {result['symbol_mappings']}")
        else:
            logging.info(f"LoRA FIXED SYMBOLS CYCLE {cycle} COMPLETE SUMMARY:")
            logging.info("=" * 80)
            logging.info(f"{'Epoch':<5} {'Loss':<10} {'MLP+Sym':<9} {'NoMLP+Sym':<10} {'MLP+Orig':<9} {'NoMLP+Orig':<10}")
            logging.info("-" * 80)
            for result in epoch_metrics:
                logging.info(f"{result['epoch']:<5} {result['loss']:<10.4f} "
                            f"{result['mlp_symbols']:<9.4f} {result['no_mlp_symbols']:<10.4f} "
                            f"{result['mlp_original']:<9.4f} {result['no_mlp_original']:<10.4f}")
        logging.info("=" * 80)
    
    # Return appropriate mappings
    if dynamic_symbols:
        return epoch_mappings, epoch_metrics
    else:
        return current_mappings, epoch_metrics

def replace_symbols_in_batch(batch, symbol_mappings):
    """Replace symbols in batch prompts and completions"""
    if not symbol_mappings:
        return batch
    
    updated_batch = batch.copy()
    
    # Replace in prompts
    if "prompt" in batch:
        updated_prompts = []
        for prompt in batch["prompt"]:
            updated_prompt = prompt
            for original, discovered in symbol_mappings.items():
                updated_prompt = updated_prompt.replace(original, discovered)
            updated_prompts.append(updated_prompt)
        updated_batch["prompt"] = updated_prompts
    
    # Replace in completions
    if "completion" in batch:
        updated_completions = []
        for completion in batch["completion"]:
            updated_completion = completion
            for original, discovered in symbol_mappings.items():
                updated_completion = updated_completion.replace(original, discovered)
            updated_completions.append(updated_completion)
        updated_batch["completion"] = updated_completions
    
    return updated_batch

def load_datasets(args, datasets):
    """Load training and validation datasets"""
    train_datasets = {}
    val_datasets = {}
    
    for dataset_name in datasets:
        try:
            dataset_type = DatasetType(dataset_name)
            
            # Load both train and validation
            full_train_dataset = load_dataset(dataset_type, split="train")
            full_val_dataset = load_dataset(dataset_type, split="validation")
            
            if args.max_samples > 0:
                train_datasets[dataset_type] = full_train_dataset.select(range(args.max_samples))
                # Use smaller validation set
                val_samples = min(args.max_samples // 2, len(full_val_dataset))
                val_datasets[dataset_type] = full_val_dataset.select(range(val_samples))
            else:
                train_datasets[dataset_type] = full_train_dataset
                val_datasets[dataset_type] = full_val_dataset
            
            logging.info(f"✓ Loaded {dataset_name}: {len(train_datasets[dataset_type])} train, {len(val_datasets[dataset_type])} val samples")
            
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            continue
    
    return train_datasets, val_datasets

def create_combined_dataloader(datasets, processor, args, shuffle=False):
    """Create combined dataloader"""
    dataset_types = list(datasets.keys())
    
    combined_dataset = DatasetFactory.create_dataset(
        dataset_type=dataset_types,
        dataset=datasets,
        processor=processor,
        is_training=shuffle,
        input_mode="speech_only",
        fewshot_mode="text",
        num_examples=5,
        random_examples=False,
        model_type=args.model_type,
        run_name=args.run_name,
        randomize_swap=False,
        balance_datasets=False,
        interleave=False
    )
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def generate_simplified_schedule(args):
    """Generate simplified training schedule"""
    logging.info(f"Generating simplified schedule: {args.total_cycles} cycles, {args.lora_epochs} LoRA epochs, {args.mlp_epochs} MLP epochs")
    
    schedule = [
        {"phase": "lora_initial", "epochs": 1, "description": "Initial task learning"}
    ]
    
    for cycle in range(args.total_cycles):
        schedule.extend([
            {"phase": "mlp", "epochs": args.mlp_epochs, "description": f"Cycle {cycle+1} MLP training"},
            {"phase": "lora", "epochs": args.lora_epochs, "description": f"Cycle {cycle+1} LoRA training"}
        ])
    
    schedule.append({
        "phase": "lora_final", 
        "epochs": args.lora_final_epochs, 
        "description": "Final task optimization"
    })
    
    logging.info(f"Generated training schedule with {len(schedule)} steps:")
    for i, step in enumerate(schedule):
        logging.info(f"  Step {i+1}: {step['phase']} ({step['epochs']} epochs) - {step['description']}")
    
    return schedule

def generate_bypass_mlp_schedule(args):
    """Generate schedule for bypass MLP mode (LoRA only with dynamic symbols)"""
    logging.info(f"Generating BYPASS MLP schedule: {args.total_cycles} cycles, {args.lora_epochs} LoRA epochs each")
    
    schedule = []
    
    for cycle in range(args.total_cycles):
        schedule.append({
            "phase": "lora_dynamic", 
            "epochs": args.lora_epochs, 
            "description": f"Cycle {cycle+1} LoRA with dynamic symbols"
        })
    
    # Final phase with more epochs
    schedule.append({
        "phase": "lora_dynamic_final", 
        "epochs": args.lora_final_epochs, 
        "description": "Final LoRA with dynamic symbols"
    })
    
    logging.info(f"Generated BYPASS MLP training schedule with {len(schedule)} steps:")
    for i, step in enumerate(schedule):
        logging.info(f"  Step {i+1}: {step['phase']} ({step['epochs']} epochs) - {step['description']}")
    
    return schedule

def generate_mlp_first_schedule(args):
    """Generate MLP-first training schedule: Initial MLP → Alternating LoRA-MLP cycles → Final LoRA"""
    logging.info(f"Generating MLP-FIRST schedule: Initial MLP → {args.total_cycles} LoRA-MLP cycles → Final LoRA")
    
    schedule = []
    
    # PHASE 1: Initial MLP training (learn symbol representations first)
    schedule.append({
        "phase": "mlp_initial", 
        "epochs": args.lora_final_epochs, 
        "description": "Initial MLP training - learn symbol representations"
    })
    
    # PHASE 2: Alternating LoRA-MLP cycles
    for cycle in range(args.total_cycles):
        # LoRA phase: Fine-tune task adaptation
        schedule.append({
            "phase": "lora", 
            "epochs": args.lora_epochs, 
            "description": f"Cycle {cycle+1} LoRA training - task adaptation"
        })
        
        # MLP phase: Refine symbol representations
        schedule.append({
            "phase": "mlp", 
            "epochs": args.mlp_epochs, 
            "description": f"Cycle {cycle+1} MLP training - refine symbols"
        })
    
    # PHASE 3: Final LoRA training (final task optimization)
    schedule.append({
        "phase": "lora_final", 
        "epochs": args.lora_final_epochs, 
        "description": "Final LoRA training - task optimization"
    })
    
    logging.info(f"Generated MLP-FIRST training schedule with {len(schedule)} steps:")
    for i, step in enumerate(schedule):
        logging.info(f"  Step {i+1}: {step['phase']} ({step['epochs']} epochs) - {step['description']}")
    
    # Log training flow
    logging.info("Training Flow:")
    logging.info("  1. Initial MLP    → Learn symbol representations")
    logging.info("  2. LoRA-MLP cycles → Alternate task adaptation & symbol refinement")
    logging.info("  3. Final LoRA     → Final task optimization")
    
    return schedule

def generate_advanced_schedule(args, schedule_type="simplified"):
    """Generate different training schedules based on type"""
    
    if schedule_type == "simplified":
        return generate_simplified_schedule(args)
    elif schedule_type == "mlp_first":
        return generate_mlp_first_schedule(args)
    elif schedule_type == "bypass_mlp":
        return generate_bypass_mlp_schedule(args)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def save_trainable_checkpoint(model, path, epoch=0, loss=0.0, phase_info=None):
    """Save only trainable parameters to reduce checkpoint size"""
    full_state_dict = model.state_dict()
    trainable_state_dict = {}
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Save if parameter requires grad OR if it's MLP (always save MLPs)
        if param.requires_grad or 'input_mlp' in name or 'output_mlp' in name:
            trainable_state_dict[name] = full_state_dict[name]
            trainable_params += param.numel()
    
    checkpoint = {
        "trainable_state_dict": trainable_state_dict,
        "epoch": epoch,
        "loss": loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "phase_info": phase_info or {},  # NEW: Include phase information
        "timestamp": datetime.now().isoformat()  # NEW: Add timestamp
    }
    
    torch.save(checkpoint, path)
    logging.info(f"✓ Saved {trainable_params:,} trainable params (of {total_params:,} total) to {path}")
    
    if phase_info:
        logging.info(f"  Phase: {phase_info.get('phase', 'unknown')}, Description: {phase_info.get('description', 'N/A')}")

def validate_model(model, val_dataloader, args, current_mappings, phase_name, epoch, bypass_mlp=False, use_original_labels=False):
    """Run validation using the model's generate_output method and calculate metrics"""
    
    # Set bypass flag manually for this validation run
    original_bypass_state = getattr(model, 'bypass_mlp_for_inference', False)
    model.bypass_mlp_for_inference = bypass_mlp
    
    # Create info string for logging
    mlp_info = "WITH MLP" if not bypass_mlp else "WITHOUT MLP (BYPASSED)"
    label_info = "ORIGINAL LABELS" if use_original_labels else "SYMBOL LABELS"
    bypass_info = f"{mlp_info} + {label_info}"
    
    logging.info(f"=== Validation for {phase_name} Epoch {epoch} ({bypass_info}) ===")
    
    model.eval()
    all_results = {dt.value: [] for dt in [DatasetType(name) for name in args.dataset_type.split('-')]}
    
    # Create reverse mappings for symbol conversion back to original labels
    reverse_mappings = {}
    if not use_original_labels:  # Only create reverse mappings if using symbols
        for original_label, random_symbol in current_mappings.items():
            reverse_mappings[random_symbol.lower()] = original_label
            reverse_mappings[random_symbol] = original_label
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"{phase_name} Validation ({bypass_info})")):
            try:
                # Apply symbol replacements ONLY if not using original labels
                if use_original_labels:
                    # Use original batch without symbol replacement
                    updated_batch = batch
                    logging.info(f"Using ORIGINAL labels - no symbol replacement")
                else:
                    # Apply symbol replacements as usual
                    updated_batch = replace_symbols_in_batch(batch, current_mappings)
                
                # Generate outputs using the model's generate_output method
                predictions = model.generate_output(updated_batch)
                
                # Process results (EXACT SAME LOGIC AS TRAIN.PY)
                for i, pred in enumerate(predictions):
                    dt = batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"]
                    dt_key = dt.value if hasattr(dt, 'value') else str(dt)
                    true_label = batch["completion"][i] if isinstance(batch["completion"], list) else batch["completion"]
                    
                    # Convert symbols back to original labels ONLY if we used symbols
                    converted_pred = pred
                    if not use_original_labels and reverse_mappings:
                        for random_symbol, original_label in reverse_mappings.items():
                            if random_symbol in converted_pred:
                                converted_pred = converted_pred.replace(random_symbol, original_label)
                            elif random_symbol.lower() in converted_pred.lower():
                                import re
                                pattern = re.compile(re.escape(random_symbol), re.IGNORECASE)
                                if pattern.search(converted_pred):
                                    converted_pred = pattern.sub(original_label, converted_pred)
                    
                    # Clean the prediction
                    try:
                        cleaned_pred = clean_prediction(converted_pred, dt)
                    except:
                        cleaned_pred = converted_pred.strip()
                    
                    result = {
                        "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                        "true_label": true_label,
                        "predicted_label": str(cleaned_pred).strip(),
                        "dataset_type": dt_key
                    }
                    all_results[dt_key].append(result)
                    
                    # Log like train.py (first few samples only)
                    
                    logging.info(f"Org_pred: {pred}")
                    logging.info(f"Predicted: {cleaned_pred}")
                    logging.info(f"True label: {true_label}")
                    logging.info(f"Dataset type: {dt_key}")
                    logging.info("=" * 50)
                
                if batch_idx > 100:
                    break
            
                    
            except Exception as e:
                logging.error(f"Error during validation batch: {str(e)}")
                continue
    
    # Restore original bypass state
    model.bypass_mlp_for_inference = original_bypass_state
    
    # Calculate metrics for each dataset type (EXACT SAME AS BEFORE)
    metrics = {}
    dataset_names = args.dataset_type.split('-')
    
    # NEW: Extract the main metric based on dataset type
    main_metric_value = 0.0
    
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            dt_key = dataset_type.value
            dt_results = all_results.get(dt_key, [])
            
            if dt_results:
                try:
                    dt_metrics = evaluate_predictions(dt_results, dataset_type)
                    metrics[dt_key] = dt_metrics
                    
                    # Log metrics (SAME AS BEFORE)
                    logging.info(f"Metrics for {dataset_name} ({bypass_info}):")
                    for metric, value in dt_metrics.items():
                        if isinstance(value, (float, int)):
                            logging.info(f"  {metric}: {value:.4f}")
                        else:
                            logging.info(f"  {metric}: {value}")
                    
                    # NEW: Extract main metric for return value
                    if dataset_name.lower() == 'voxceleb':
                        main_metric_value = dt_metrics.get('macro_f1_with_invalid', 0.0)
                    elif dataset_name.lower() == 'hvb':
                        main_metric_value = dt_metrics.get('macro_f1', 0.0)
                    elif dataset_name.lower() == 'meld_emotion':
                        # Default to macro_f1_filtered for other datasets
                        main_metric_value = dt_metrics.get('macro_f1_filtered', 0.0)
                    elif dataset_name.lower() == 'voxceleb':
                        main_metric_value = dt_metrics.get('macro_f1', 0.0)
                    else:
                        main_metric_value = dt_metrics.get('macro_f1', 0.0)
                        
                except Exception as e:
                    logging.error(f"Error evaluating predictions for {dataset_name}: {str(e)}")
                    metrics[dt_key] = {"error": str(e)}
                    
        except Exception as e:
            logging.error(f"Failed to process dataset {dataset_name}: {e}")
            continue
    
    # NEW: Return the main metric value
    return main_metric_value

def main():
    args = parse_args()
    setup_unified_logging(args)
    
    # Log schedule type
    logging.info(f"=== Training Schedule: {args.schedule_type.upper()} ===")
    
    if args.bypass_mlp:
        logging.info("=== BYPASS MLP Mode: Pure LoRA Training with Dynamic Symbols ===")
        training_schedule = generate_bypass_mlp_schedule(args)
    else:
        # Use the selected schedule type
        training_schedule = generate_advanced_schedule(args, args.schedule_type)
        
        if args.schedule_type == "mlp_first":
            logging.info("=== MLP-FIRST Mode: Initial MLP → LoRA-MLP Cycles → Final LoRA ===")
        else:
            logging.info("=== SIMPLIFIED Mode: Initial LoRA → MLP-LoRA Cycles → Final LoRA ===")
    
    if args.bypass_mlp:
        logging.info("=== BYPASS MLP Mode: Pure LoRA Training with Dynamic Symbols ===")
        logging.info(f"Training configuration:")
        logging.info(f"  Total cycles: {args.total_cycles}")
        logging.info(f"  LoRA epochs per cycle: {args.lora_epochs}")
        logging.info(f"  Final LoRA epochs: {args.lora_final_epochs}")
        logging.info(f"  MLP bypassed: True (no MLP training or inference)")
        logging.info(f"  Dynamic symbols: Fresh symbols generated each epoch")
    else:
        logging.info("=== Normal Mode: MLP and LoRA Training ===")
        logging.info(f"Training configuration:")
        logging.info(f"  Total cycles: {args.total_cycles}")
        logging.info(f"  LoRA epochs per cycle: {args.lora_epochs}")
        logging.info(f"  MLP epochs per cycle: {args.mlp_epochs}")
        logging.info(f"  Final LoRA epochs: {args.lora_final_epochs}")
        logging.info(f"  Use output MLP: {args.use_output_mlp}")
    
    # Setup tokenizer and datasets
    from transformers import LlamaTokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
    llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.padding_side = "right"
    
    # Load datasets
    train_datasets, val_datasets = load_datasets(args, args.dataset_type.split('-'))
    processor = get_processor(args.model_type, tokenizer=llama_tokenizer)
    
    # Create dataloaders
    train_dataloader = create_combined_dataloader(train_datasets, processor, args, shuffle=True)
    val_dataloader = create_combined_dataloader(val_datasets, processor, args, shuffle=False)
    
    logging.info(f"✓ Created train dataloader: {len(train_dataloader)} batches")
    logging.info(f"✓ Created validation dataloader: {len(val_dataloader)} batches")
    
    # Generate initial symbols
    dataset_names = args.dataset_type.split('-')
    all_valid_labels = set()
    for dataset_name in dataset_names:
        dataset_type = DatasetType(dataset_name)
        config = get_dataset_config(dataset_type)
        all_valid_labels.update(config.valid_labels)
    
    dataset_labels = sorted(list(all_valid_labels))
    random_symbols = generate_one_word_two_token_symbols(len(dataset_labels), llama_tokenizer)
    current_symbol_mappings = create_label_mapping(dataset_labels, random_symbols)
    
    # Initialize model
    model = MLPSalmonn(
        device=args.device,
        label_tokens=list(current_symbol_mappings.values()),
        hidden_dim=args.hidden_dim,
        dropout=0.1,
        freeze_base=True,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        low_resource=True,
        use_output_mlp=args.use_output_mlp  # Use the parameter
    )
    
    # Apply initial symbol mappings
    logging.info(f"Initial symbol mappings: {current_symbol_mappings}")
    model.update_label_tokens(current_symbol_mappings)
    
    # Generate training schedule
    training_schedule = generate_advanced_schedule(args, args.schedule_type)
    
    # NEW: Track all training metrics
    all_training_metrics = []
    
    cycle = 0
    for step, schedule in enumerate(training_schedule):
        logging.info(f"=== Step {step+1}/{len(training_schedule)}: {schedule['description']} ({schedule['epochs']} epochs) ===")
        
        if args.bypass_mlp:
            # BYPASS MLP MODE: Only dynamic LoRA training
            if schedule["phase"] in ["lora_dynamic", "lora_dynamic_final"]:
                current_symbol_mappings, epoch_metrics = train_lora_phase_unified(
                    model=model, 
                    dataloader=train_dataloader, 
                    args=args, 
                    cycle=cycle, 
                    current_mappings=current_symbol_mappings,  # Not used in dynamic mode but required for signature
                    num_epochs=schedule["epochs"], 
                    val_dataloader=val_dataloader,
                    dynamic_symbols=True,  # KEY: Enable dynamic symbols
                    llama_tokenizer=llama_tokenizer, 
                    dataset_labels=dataset_labels
                )
                
                all_training_metrics.extend(epoch_metrics)
                cycle += 1
                
                checkpoint_name = f"{args.run_name}_bypass_cycle_{cycle}_lora_{schedule['epochs']}epochs"
        else:
            # Handle different phases
            if schedule["phase"] in ["lora_initial", "lora", "lora_final"]:
                # LoRA training
                current_symbol_mappings, epoch_metrics = train_lora_phase_unified(
                    model=model, 
                    dataloader=train_dataloader, 
                    args=args, 
                    cycle=cycle, 
                    current_mappings=current_symbol_mappings, 
                    num_epochs=schedule["epochs"], 
                    val_dataloader=val_dataloader,
                    dynamic_symbols=False
                )
                
                all_training_metrics.extend(epoch_metrics)
                
                # Checkpoint naming
                if schedule["phase"] == "lora_initial":
                    checkpoint_name = f"{args.run_name}_initial_lora_epoch_{schedule['epochs']}"
                elif schedule["phase"] == "lora_final":
                    checkpoint_name = f"{args.run_name}_final_lora_epoch_{schedule['epochs']}"
                else:
                    checkpoint_name = f"{args.run_name}_cycle_{cycle}_lora_epoch_{schedule['epochs']}"
            
            elif schedule["phase"] in ["mlp", "mlp_initial"]:
                # MLP training
                current_symbol_mappings, epoch_metrics = train_mlp_phase_simplified(
                    model, val_dataloader, args, cycle, schedule["epochs"], current_symbol_mappings,
                    val_dataloader=train_dataloader
                )
                
                all_training_metrics.extend(epoch_metrics)
                
                # Checkpoint naming
                if schedule["phase"] == "mlp_initial":
                    checkpoint_name = f"{args.run_name}_initial_mlp_epoch_{schedule['epochs']}"
                else:
                    checkpoint_name = f"{args.run_name}_cycle_{cycle}_mlp_epoch_{schedule['epochs']}"
                    
                # Increment cycle after MLP phase (in mlp_first mode)
                if args.schedule_type == "mlp_first" and schedule["phase"] == "mlp":
                    cycle += 1
            
            # For simplified mode, increment cycle after LoRA phase
            if args.schedule_type == "simplified" and schedule["phase"] == "lora":
                cycle += 1
        
        # Save checkpoint code stays the same...
        logging.info(f"Saving checkpoint: {checkpoint_name}")
        checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_trainable_checkpoint(
            model=model,
            path=os.path.join(checkpoint_dir, "model.pt"),
            epoch=step,
            loss=0.0,
            phase_info={
                "phase": schedule["phase"],
                "cycle": cycle if schedule["phase"] == "mlp" else cycle,
                "epochs": schedule["epochs"],
                "step": step,
                "description": schedule["description"]
            }
        )
    
    # NEW: Final comprehensive summary
    if all_training_metrics:
        logging.info("\n" + "=" * 100)
        if args.bypass_mlp:
            logging.info("BYPASS MLP TRAINING SUMMARY - DYNAMIC SYMBOLS")
            logging.info("=" * 100)
            logging.info(f"{'Phase':<15} {'Cycle':<5} {'Epoch':<5} {'Loss':<8} {'Fresh-Sym':<10} {'Original':<10}")
            logging.info("-" * 100)
            
            for result in all_training_metrics:
                logging.info(f"{result['phase']:<15} {result['cycle']:<5} {result['epoch']:<5} "
                            f"{result['loss']:<8.4f} {result['fresh_symbols']:<10.4f} {result['original_labels']:<10.4f}")
        else:
            logging.info("COMPLETE TRAINING SUMMARY - ALL EPOCHS")
            logging.info("=" * 100)
            logging.info(f"{'Phase':<6} {'Cycle':<5} {'Epoch':<5} {'Loss':<8} {'MLP+Sym':<9} {'NoMLP+Sym':<10} {'MLP+Orig':<9} {'NoMLP+Orig':<10}")
            logging.info("-" * 100)
            
            for result in all_training_metrics:
                logging.info(f"{result['phase']:<6} {result['cycle']:<5} {result['epoch']:<5} "
                            f"{result['loss']:<8.4f} {result['mlp_symbols']:<9.4f} {result['no_mlp_symbols']:<10.4f} "
                            f"{result['mlp_original']:<9.4f} {result['no_mlp_original']:<10.4f}")
        
        logging.info("=" * 100)
        
        # Find best performing configurations
        best_mlp_sym = max(all_training_metrics, key=lambda x: x['mlp_symbols'])
        best_no_mlp_sym = max(all_training_metrics, key=lambda x: x['no_mlp_symbols'])
        best_mlp_orig = max(all_training_metrics, key=lambda x: x['mlp_original'])
        best_no_mlp_orig = max(all_training_metrics, key=lambda x: x['no_mlp_original'])
        
        logging.info("BEST PERFORMING CONFIGURATIONS:")
        logging.info(f"Best MLP+Symbols: {best_mlp_sym['phase']} Cycle {best_mlp_sym['cycle']} Epoch {best_mlp_sym['epoch']} = {best_mlp_sym['mlp_symbols']:.4f}")
        logging.info(f"Best NoMLP+Symbols: {best_no_mlp_sym['phase']} Cycle {best_no_mlp_sym['cycle']} Epoch {best_no_mlp_sym['epoch']} = {best_no_mlp_sym['no_mlp_symbols']:.4f}")
        logging.info(f"Best MLP+Original: {best_mlp_orig['phase']} Cycle {best_mlp_orig['cycle']} Epoch {best_mlp_orig['epoch']} = {best_mlp_orig['mlp_original']:.4f}")
        logging.info(f"Best NoMLP+Original: {best_no_mlp_orig['phase']} Cycle {best_no_mlp_orig['cycle']} Epoch {best_no_mlp_orig['epoch']} = {best_no_mlp_orig['no_mlp_original']:.4f}")
        logging.info("=" * 100)
    
    # Save final model with descriptive name
    final_checkpoint_name = f"{args.run_name}_final_model"
    logging.info(f"Saving final model: {final_checkpoint_name}")
    final_checkpoint_dir = os.path.join(args.output_dir, final_checkpoint_name)
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    save_trainable_checkpoint(
        model=model,
        path=os.path.join(final_checkpoint_dir, "model.pt"),
        epoch=-1,
        loss=0.0,
        phase_info={
            "phase": "final",
            "total_cycles": args.total_cycles,
            "total_steps": len(training_schedule),
            "description": "Final trained model"
        }
    )
    
    final_mappings_file = os.path.join(args.output_dir, f"{args.run_name}_final_symbol_mappings.json")
    with open(final_mappings_file, 'w') as f:
        json.dump({
            "final_mappings": current_symbol_mappings,
            "training_schedule": training_schedule,
            "args": vars(args)
        }, f, indent=2)
    
    logging.info(f"Training complete! Files saved with prefix: {args.run_name}")
    logging.info(f"Final model: {final_checkpoint_dir}/model.pt")
    logging.info(f"Final mappings: {final_mappings_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)