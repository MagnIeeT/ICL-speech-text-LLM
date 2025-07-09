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
parser.add_argument("--num_epochs", type=int, default=15)
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
                   choices=['speech_only', 'speech_and_text'],
                   help="Whether to use speech only or both speech and text")

# New arguments
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before backward pass")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine"], help="Type of learning rate scheduler")
parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to checkpoint to resume from")

args = parser.parse_args()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{args.log_file}',mode='w'),
        logging.StreamHandler()
    ]
)

class SpeechSentimentDataset(Dataset):
    def __init__(self, dataset, wav_processor, input_mode='speech_and_text', balance_strategy='weighted'):
        valid_labels = ["Positive", "Negative", "Neutral"]
        filtered_indices = [i for i in range(len(dataset)) if dataset[i]['sentiment'] in valid_labels]
        self.dataset = dataset.select(filtered_indices)
        self.wav_processor = wav_processor
        self.input_mode = input_mode
        self.balance_strategy = balance_strategy
        
        # Calculate class weights and sampling weights
        label_counts = {}
        self.indices_by_label = {label: [] for label in valid_labels}
        
        for idx, item in enumerate(self.dataset):
            label = item['sentiment']
            label_counts[label] = label_counts.get(label, 0) + 1
            self.indices_by_label[label].append(idx)
            
        # Calculate class weights (inverse frequency)
        total_samples = sum(label_counts.values())
        self.class_weights = {
            label: total_samples / (len(valid_labels) * count)
            for label, count in label_counts.items()
        }
        
        logging.info(f"Label distribution: {label_counts}")
        logging.info(f"Class weights: {self.class_weights}")
        
        # Create weighted sampling if needed
        if balance_strategy == 'weighted':
            self.sample_weights = [self.class_weights[self.dataset[i]['sentiment']] for i in range(len(self.dataset))]
        elif balance_strategy == 'oversample':
            # Oversample minority classes to match majority class
            max_samples = max(label_counts.values())
            oversampled_indices = []
            for label in valid_labels:
                label_indices = self.indices_by_label[label]
                # Repeat indices to match majority class size
                oversampled_indices.extend(np.random.choice(
                    label_indices,
                    size=max_samples,
                    replace=True
                ))
            self.dataset = self.dataset.select(oversampled_indices)
            logging.info(f"Dataset size after oversampling: {len(self.dataset)}")
        
        # Store a fixed example for debugging - store the full processed item
        logging.info("Initializing debug example...")
        try:
            self.debug_example = self.__getitem__(0)
            logging.info(f"Debug example initialized with sentiment: {self.debug_example['completion']}")
        except Exception as e:
            logging.error(f"Failed to initialize debug example: {str(e)}")
            self.debug_example = None

    def get_sampler(self):
        if self.balance_strategy == 'weighted':
            return torch.utils.data.WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.dataset),
                replacement=True
            )
        return None

    def __len__(self):
        return len(self.dataset)
    
    def _create_prompt(self, item):
        base_prompt = """You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: Positive, Negative, or Neutral.

Guidelines:
- Choose Positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose Negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose Neutral ONLY IF the statement is purely factual with zero emotional content"""

        max_available = min(5, len(item['few_shot_examples']))
        num_examples = np.random.randint(0, max_available)
        
        if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0 and num_examples > 0:
            selected_examples = item['few_shot_examples'][:num_examples]
            examples_text = "\n\n".join([
                f"Text: {example['text']}\n"
                f"Output: {example['label']}"
                for example in selected_examples
            ])
            
            if self.input_mode == 'speech_only':
                input_section = "<Speech><SpeechHere></Speech>"
            else:  # speech_and_text
                input_section = f"<Speech><SpeechHere></Speech>\nTranscript: {item['normalized_text']}"
            
            prompt = f"""{base_prompt}

Here are few examples to learn from:
{examples_text}

Now analyze this input:
{input_section}
Output:"""
        else:
            if self.input_mode == 'speech_only':
                input_section = "<Speech><SpeechHere></Speech>"
            else:  # speech_and_text
                input_section = f"<Speech><SpeechHere></Speech>\nTranscript: {item['normalized_text']}"
            
            prompt = f"""{base_prompt}

Now analyze this input:
{input_section}
Output:"""
            
        return {"prompt": prompt, "completion": f"{item['sentiment']}</s>"}

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process audio features
        raw_wav = torch.from_numpy(np.array(item["audio"]['array']))
        spectrogram = self.wav_processor(raw_wav, sampling_rate=16000, return_tensors="pt")["input_features"]
        
        # Create prompt data
        prompt_data = self._create_prompt(item)
        
        return {
            "spectrogram": spectrogram.squeeze(0),
            "raw_wav": raw_wav,
            "wav_length": len(raw_wav),
            "text": item["normalized_text"],
            "prompt": prompt_data["prompt"],
            "completion": prompt_data["completion"],
            "few_shot_examples": item.get('few_shot_examples', [])
        }

    def get_debug_example(self):
        """Get a fixed example for debugging generation"""
        if self.debug_example is None:
            logging.error("Debug example was not properly initialized")
            raise ValueError("Debug example not initialized")
            
        logging.info("Preparing debug example for generation...")
        try:
            debug_dict = {
                "spectrogram": self.debug_example["spectrogram"].unsqueeze(0),  # Add batch dimension
                "raw_wav": self.debug_example["raw_wav"].unsqueeze(0),  # Add batch dimension
                "wav_length": self.debug_example["wav_length"],
                "prompt": [self.debug_example["prompt"]],  # Make it a list
                "completion": [self.debug_example["completion"]],  # Make it a list
                "text": [self.debug_example["text"]]  # Make it a list
            }
            logging.info(f"Debug example prepared with shape: {debug_dict['spectrogram'].shape}")
            return debug_dict
        except Exception as e:
            logging.error(f"Error preparing debug example: {str(e)}")
            raise

def collate_fn(batch):
    # Get lengths and pad raw_wav
    wav_lengths = torch.tensor([item['wav_length'] for item in batch])
    raw_wavs = [item['raw_wav'] for item in batch]
    raw_wavs_padded = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
    
    # Create padding mask
    padding_mask = torch.arange(raw_wavs_padded.size(1)).unsqueeze(0) >= wav_lengths.unsqueeze(1)
    
    # Stack spectrograms
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    
    return {
        "spectrogram": spectrograms,
        "raw_wav": raw_wavs_padded,
        "padding_mask": padding_mask,
        "prompt": [item['prompt'] for item in batch],
        "completion": [item['completion'] for item in batch]
    }

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

def train(model, train_dataloader, optimizer, scheduler, num_epochs, device, class_weights, train_dataset):
    model.train()
    best_loss = float('inf')
    memory_tracker = MemoryTracker(device)
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    # Load checkpoint if exists
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logging.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])  # Load scaler state
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        logging.info(f"Resuming from epoch {start_epoch} with loss {best_loss}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move inputs to device
                batch["spectrogram"] = batch["spectrogram"].to(device)
                if batch["raw_wav"] is not None:
                    batch["raw_wav"] = batch["raw_wav"].to(device)
                if batch["padding_mask"] is not None:
                    batch["padding_mask"] = batch["padding_mask"].to(device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    loss = calculate_weighted_loss(outputs, batch, class_weights)
                    loss = loss / args.gradient_accumulation_steps
                    # loss = outputs["loss"] / args.gradient_accumulation_steps
                
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
                
                # Memory tracking
                if batch_idx % 50 == 0:
                    current_memory = memory_tracker.check_memory(batch_idx)
                    torch.cuda.empty_cache()
                
                # Update progress
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
                })
                
                # Debug generation every 50 batches
                # if batch_idx % 50 == 0:
                #     model.eval()
                #     try:
                #         logging.info("Getting debug example...")
                #         debug_sample = train_dataset.get_debug_example()
                #         if debug_sample is None:
                #             logging.error("Got None debug sample")
                #             continue
                            
                #         logging.info(f"Debug sample keys: {debug_sample.keys()}")
                        
                #         # Move to device
                #         debug_sample = {
                #             k: v.to(device) if torch.is_tensor(v) else v 
                #             for k, v in debug_sample.items()
                #         }
                        
                #         with torch.no_grad():
                #             prediction = model.generate_sentiment(debug_sample)
                #             if not prediction:
                #                 logging.error("Model returned empty prediction")
                #                 continue
                        
                #         logging.info("\nDebug Generation Test:")
                #         logging.info(f"Input text: {debug_sample['text']}")
                #         true_sentiment = debug_sample['completion'][0].split('</s>')[0] if debug_sample['completion'] else "Unknown"
                #         logging.info(f"True sentiment: {true_sentiment}")
                #         logging.info(f"Predicted sentiment: {prediction[0] if prediction else 'No prediction'}")
                #     except Exception as e:
                #         logging.error(f"Error in debug generation: {str(e)}")
                #         import traceback
                #         logging.error(traceback.format_exc())
                #     finally:
                #         model.train()
                
            except torch.cuda.OutOfMemoryError:
                logging.error(f"OOM at batch {batch_idx}, clearing cache and skipping batch")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
        
        # Save checkpoint
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if avg_epoch_loss < best_loss:
            checkpoint_path = f'{args.output_dir}/checkpoints/epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}'
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
            }, f'{checkpoint_path}/model.pt')
            
            # Log the size of saved checkpoint
            full_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB
            trainable_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 2)  # MB
            logging.info(f'Saved checkpoint with loss {avg_epoch_loss:.4f}')
            logging.info(f'Checkpoint size reduced from {full_size:.2f} MB to {trainable_size:.2f} MB')

def main():
    # Add device check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SALMONN model
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

    logging.info("Initializing SALMONN model...")
    model = CustomSALMONN.from_config(config)
    model.to(device)
    
    # Initialize wav_processor
    logging.info("Initializing WhisperFeatureExtractor...")
    wav_processor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = load_from_disk("/data2/neeraja/neeraja/data/voxceleb_train_fewshots")
    train_dataset = SpeechSentimentDataset(
        dataset, 
        wav_processor, 
        input_mode=args.input_mode,
        balance_strategy= 'weighted' #'weighted'  # or 'oversample'
    )
    
    # Create sampler for weighted sampling
    sampler = train_dataset.get_sampler()
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler if available
        shuffle=sampler is None,  # Only shuffle if not using sampler
        collate_fn=collate_fn
    )
    
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
    train(model, train_dataloader, optimizer, scheduler, 
          args.num_epochs, model.device, train_dataset.class_weights,
          train_dataset)
    
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




