# finetune_sentiment_debug.py
import torch
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from tqdm import tqdm
import gc
import sys
import os
from custom_salmon import CustomSALMONN
from transformers import WhisperFeatureExtractor
from torch.nn.utils.rnn import pad_sequence
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from dataset_config import DatasetType, DATASET_CONFIGS, DatasetSplit, get_swapped_config, apply_label_mapping
from salmon_datasets import FinetuneDataset, collate_fn  # Add this import
from peft import LoraConfig, get_peft_model, TaskType




parser = argparse.ArgumentParser()
# Model paths
parser.add_argument("--model_path", type=str, default="lmsys/vicuna-13b-v1.1", help="Path to LLaMA model")
parser.add_argument("--whisper_path", type=str, default="openai/whisper-large-v2", help="Path to Whisper model")
parser.add_argument("--beats_path", type=str, default="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", help="Path to BEATs model")
parser.add_argument("--salmonn_checkpoint", type=str, default="/home/sriramg/aneeraj/storage/salmonn_v1.pth", help="Path to SALMONN checkpoint")
parser.add_argument("--speech_llama_proj_model", type=str, default="", help="Path to QFormer projection weights")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument('--log_file', type=str, default='finetune_llamma2.log', help='Name of the log file')
# Training settings

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_steps", type=int, default=100)

# LoRA settings
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)

# Resource management
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--device_8bit", type=int, default=0)

# Input mode
parser.add_argument("--input_mode", type=str, default='speech_only', 
                   choices=['speech_only', 'speech_and_text', 'text_only'],
                   help="Whether to use speech only, text only, or both speech and text")

# New arguments
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before backward pass")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine"], help="Type of learning rate scheduler")
parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to checkpoint to resume from")
parser.add_argument("--dataset_type", type=str, 
                   choices=["voxceleb", "voxceleb_swap", "hvb", "hvb_greek", "hvb_swap"],  # Updated choices
                   default="voxceleb", help="Type of dataset to use")
parser.add_argument("--fewshot_mode", type=str, default='text', 
                   choices=['text', 'speech'],
                   help="Whether to use text or speech for few-shot examples")
parser.add_argument("--model_type", type=str, 
                   choices=["salmonn", "qwen2"],
                   default="salmonn", 
                   help="Type of model to use")

args = parser.parse_args()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logging.info("Starting script...")

class MemoryTracker:
    def __init__(self, device, threshold_mb=20000):
        self.device = device
        self.threshold_mb = threshold_mb
        self.peak_memory = 0
        
    def check_memory(self, batch_idx):
        if not torch.cuda.is_available():
            return
        current_memory = torch.cuda.memory_allocated(self.device) / 1024**2
        self.peak_memory = max(self.peak_memory, current_memory)
        print(f"\nCurrent Memory: {current_memory:.2f} MB")
        torch.cuda.empty_cache()
        gc.collect()
        return current_memory

def calculate_weighted_loss(outputs, batch, class_weights):
    loss = outputs["loss"]
    # Apply class weights to loss
    label = batch["completion"][0].split('</s>')[0]  # Extract label from completion
    weight = class_weights[label]
    return loss * weight

def evaluate_random_samples(model, processor, train_dataloader, device, num_samples=10):
    model.eval()
    
    with torch.no_grad():
        dataloader_iter = iter(train_dataloader)
        for i in range(num_samples):
            try:
                # Get a single batch
                batch = next(dataloader_iter)
                
                # Move batch to device
                batch = {k: v.to(device).to(torch.float16) if k == "input_features" 
                        else v.to(device) if isinstance(v, torch.Tensor) 
                        else v for k, v in batch.items()}
                
                # Generate
                outputs = model.generate(
                    input_ids=batch["input_ids"][:, :batch["prompt_length"][0]],  # Only use up to prompt length
                    attention_mask=batch["attention_mask"][:, :batch["prompt_length"][0]],  # Match prompt length
                    input_features=batch["input_features"],
                    feature_attention_mask=batch["feature_attention_mask"],
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.1,
                )
                
                # Decode and log results
                prompt_length = batch["prompt_length"][0]  # Since batch_size is 1
                label = processor.tokenizer.decode(batch["input_ids"][0][prompt_length:])
                generated = processor.tokenizer.decode(outputs[0][prompt_length:])
                logging.info(f"\nSample {i+1}/{num_samples}:")
                logging.info(f"Label: {label}")
                logging.info(f"Generated: {generated}")
                
            except StopIteration:
                logging.info("Reached end of dataset during evaluation")
                break
    
    model.train()

def train(model, processor, train_dataloader, optimizer, scheduler, num_epochs, device, class_weights=None):
    scaler = torch.cuda.amp.GradScaler()
    
    # Debug model device
    print(f"Model device: {next(model.parameters()).device}")
    
    model.train()
    best_loss = float('inf')
    memory_tracker = MemoryTracker(device)
    start_epoch = 0
    
    # Load checkpoint if exists
    
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logging.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = 15 + 1
        logging.info(f"Resuming from epoch {start_epoch}")
    else:
        logging.info("Starting training from scratch")
        start_epoch = 0
    
    # Add at start of train function
    logging.info(f"Output directory from args: {args.output_dir}")
    
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    logging.info(f"Creating checkpoint directory at: {checkpoint_dir}")
    
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Successfully created checkpoint directory")
    except Exception as e:
        logging.error(f"Error creating checkpoint directory: {str(e)}")
        logging.error(f"Current working directory: {os.getcwd()}")
        raise
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                    
                if args.model_type == "qwen2":
                    # Debug tokenization for first batch of first epoch only
                    if epoch == 0 and batch_idx == 0:
                        logging.info("\n=== Debug Tokenization ===")
                        decoded_full = processor.tokenizer.decode(batch["input_ids"][0])
                        decoded_prompt = processor.tokenizer.decode(
                            batch["input_ids"][0][:batch["prompt_length"][0]]
                        )
                        decoded_completion = processor.tokenizer.decode(
                            batch["input_ids"][0][batch["prompt_length"][0]:]
                        )
                        
                        logging.info(f"Full sequence: {decoded_full}")
                        logging.info(f"Prompt part: {decoded_prompt}")
                        logging.info(f"Completion part: {decoded_completion}")
                        logging.info(f"Prompt length: {batch['prompt_length'][0]}")
                        logging.info(f"Input ids shape: {batch['input_ids'].shape}")
                        logging.info("========================\n")
                    
                    # Move all batch tensors to device and convert to float16
                    batch = {k: v.to(device).to(torch.float16) if k == "input_features" 
                            else v.to(device) if isinstance(v, torch.Tensor) 
                            else v for k, v in batch.items()}
                    
                    # Create labels tensor - set to -100 for prompt tokens
                    labels = torch.full_like(batch["input_ids"], -100, device=device)
                    for i, prompt_len in enumerate(batch["prompt_length"]):
                        labels[i, prompt_len:] = batch["input_ids"][i, prompt_len:]
                    
                    # Debug tokenization for first batch of first epoch
                    if epoch == 0 and batch_idx == 0:
                        logging.info("=== First Batch Token Debug ===")
                        logging.info(f"Last 20 input_ids: {batch['input_ids'][0][-20:].tolist()}")
                        logging.info(f"Last 20 labels: {labels[0][-20:].tolist()}")
                        logging.info(f"Last 20 attention_mask: {batch['attention_mask'][0][-20:].tolist()}")
                        logging.info(f"Last 20 feature_attention_mask: {batch['feature_attention_mask'][0][-20:].tolist()}")
                        
                        # Add input features debug info
                        logging.info(f"Input features shape: {batch['input_features'].shape}")
                        logging.info(f"Last 20 input features (first dimension): {batch['input_features'][0, -20:, 0].tolist()}")
                        logging.info(f"Input features min: {batch['input_features'].min().item()}")
                        logging.info(f"Input features max: {batch['input_features'].max().item()}")
                        logging.info(f"Input features mean: {batch['input_features'].mean().item()}")
                        
                        decoded_last_input = processor.tokenizer.decode(batch['input_ids'][0][-20:])
                        decoded_last_labels = processor.tokenizer.decode([x for x in labels[0][-20:] if x != -100])
                        logging.info(f"Last 20 input tokens decoded: {decoded_last_input}")
                        logging.info(f"Last 20 labels decoded: {decoded_last_labels}")
                        logging.info("==============================")
                    
                    # Forward pass with consistent dtype
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            input_features=batch["input_features"],
                            feature_attention_mask=batch["feature_attention_mask"],
                            labels=labels,
                            return_dict=True
                        )
                        loss = outputs.loss / args.gradient_accumulation_steps

                    torch.cuda.empty_cache()
                else:
                    # Existing SALMONN code - update to pass input_mode
                    if "raw_wav" in batch and batch["raw_wav"] is not None:
                        batch["raw_wav"] = batch["raw_wav"].to(device)
                    if "padding_mask" in batch and batch["padding_mask"] is not None:
                        batch["padding_mask"] = batch["padding_mask"].to(device)
                    
                    # Log input_mode for first batch of first epoch
                    if epoch == 0 and batch_idx == 0:
                        input_mode = batch.get("input_mode", args.input_mode)
                        logging.info(f"\n=== Processing batch with input_mode: {input_mode} ===")
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(batch, input_mode=args.input_mode)
                        loss = outputs["loss"] / args.gradient_accumulation_steps
                
                # Debug loss device
                # logging.info(f"\nLoss device before scaling: {loss.device}")
                
                # Ensure loss is on CUDA
                if not loss.is_cuda:
                    logging.warning("Warning: Loss is not on CUDA device!")
                    loss = loss.to(device)
                    logging.info(f"Loss device after moving: {loss.device}")
                
                # Handle NaN loss
                if torch.isnan(loss):
                    logging.warning(f"\nNaN loss detected at batch {batch_idx}")
                    continue
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logging.warning(f"NaN gradient in {name}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    optimizer.zero_grad()
                    continue
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Step optimizer and scaler
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Add evaluation every 50 batches - only for Qwen2
                if args.model_type == "qwen2" and batch_idx % 500 == 0:
                    # Force flush the progress bar
                    
                    logging.info("\n=== Running inference on random samples ===")
                    sys.stdout.flush()  # Flush stdout buffer
                    
                    evaluate_random_samples(model, processor, train_dataloader, device)
                    
                    logging.info("=======================================\n")
                    sys.stdout.flush()  # Flush stdout buffer again
                    

                
                # Memory tracking for all models
                if batch_idx % 50 == 0:
                    current_memory = memory_tracker.check_memory(batch_idx)
                    torch.cuda.empty_cache()
                
                # Update progress
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })
                
            except torch.cuda.OutOfMemoryError:
                logging.error(f"OOM at batch {batch_idx}, clearing cache and skipping batch")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

        # Save checkpoint
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if avg_epoch_loss < best_loss:
            # Create checkpoint directory inside the output_dir
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}')
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Get trainable parameter mapping
            param_grad_dic = {
                k: v.requires_grad for (k, v) in model.named_parameters()
            }
            
            # Get full state dict and filter out non-trainable parameters
            state_dict = model.state_dict()
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    del state_dict[k]
            
            torch.save({
                "model": state_dict,  # Using same key as runner.py
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),  # Save scaler state
                "epoch": epoch,
                "loss": avg_epoch_loss,
            }, os.path.join(checkpoint_path, 'model.pt'))
            
            # Log the size of saved checkpoint
            full_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB
            trainable_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 2)  # MB
            logging.info(f'Saved checkpoint with loss {avg_epoch_loss:.4f}')
            logging.info(f'Checkpoint size reduced from {full_size:.2f} MB to {trainable_size:.2f} MB')

def get_model(args):
    if args.model_type == "qwen2":
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16  # Add explicit float16
        )
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Add LoRA config
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=args.lora_dropout,
            inference_mode=False, 
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Convert to PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            trust_remote_code=True
        )
        return model, processor
    else:
        # Existing SALMONN model initialization
        config = {
            "llama_path": args.model_path,
            "whisper_path": args.whisper_path,
            "beats_path": args.beats_path,
            "use_speech_Qformer": True,
            "freeze_whisper": True,
            "freeze_beats": True,
            "freeze_speech_QFormer":False,
            "num_speech_query_token": 1,
            "window_level_Qformer": True,
            "second_per_window": 0.333333,
            "second_stride": 0.333333,
            "speech_llama_proj_model": args.speech_llama_proj_model,
            "freeze_speech_llama_proj": False,
            "lora": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "low_resource": args.load_in_8bit,
            "device_8bit": args.device_8bit,
            "ckpt": args.salmonn_checkpoint
        }
        model = CustomSALMONN.from_config(config)
        wav_processor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)
        return model, wav_processor

def main():
    # Add these debug statements at the start
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA current device:", torch.cuda.current_device())

    # Set environment variable for CUDA error debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model, processor = get_model(args)
        print("Model loaded successfully")
        print("Model type:", type(model))
        
        # Move model to device
        model = model.to(device)
        print(f"Model device after moving: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        raise
    
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset with error handling
    logging.info("Loading dataset...")
    dataset_type = DatasetType(args.dataset_type)
    dataset_config = DATASET_CONFIGS[dataset_type]
    train_path = dataset_config.get_path(DatasetSplit.TRAIN)
    
    logging.info(f"Loading dataset from: {train_path}")
    if not os.path.exists(train_path):
        raise ValueError(f"Dataset path does not exist: {train_path}")
        
    dataset = load_from_disk(train_path)
    logging.info(f"Dataset loaded with {len(dataset)} examples")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")
    
    # Use FinetuneDataset instead of SpeechSentimentDataset
    train_dataset = FinetuneDataset(
        dataset_type=dataset_type,
        dataset=dataset, 
        wav_processor=processor, 
        input_mode=args.input_mode,
        balance_strategy=None,
        fewshot_mode=args.fewshot_mode,
        model_type=args.model_type
    )
    
    # Create dataloader using the collate_fn from salmon_datasets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_dataset.get_sampler(),  # Use the built-in sampler method
        shuffle=train_dataset.get_sampler() is None,  # Only shuffle if not using sampler
        collate_fn=collate_fn
    )
    
    if args.fewshot_mode == 'speech':
        audio_lookup_path = dataset_config.get_audio_lookup_path(DatasetSplit.TRAIN)
        logging.info(f"Using audio lookup from: {audio_lookup_path}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    
    # Setup scheduler based on type
    num_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    
    if args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=1e-6
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    logging.info(f"Using {args.scheduler_type} scheduler with {num_training_steps} total steps")
    logging.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logging.info(f"Max gradient norm: {args.max_grad_norm}")
    
    # Start training
    logging.info("Starting training...")
    train(model, processor,train_dataloader, optimizer, scheduler, 
          args.num_epochs, model.device, train_dataset.class_weights)
    
    # Save final model - Only save trainable parameters
    final_checkpoint_path = os.path.join(args.output_dir, 'final_model.pt')
    
    # Get trainable parameter mapping
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    
    # Get full state dict and filter out non-trainable parameters
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    
    torch.save({
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": args.num_epochs,
    }, final_checkpoint_path)
    
    # Log the size reduction
    full_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB
    trainable_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 2)  # MB
    logging.info(f"Full model size: {full_size:.2f} MB")
    logging.info(f"Trainable parameters size: {trainable_size:.2f} MB")
    logging.info(f"Training completed! Model saved to {final_checkpoint_path}")
    
if __name__ == "__main__":
    main()




