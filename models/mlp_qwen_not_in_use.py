import sys
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout, 
    force=True)


class MLPQwen(nn.Module):
    """Clean Qwen2 Audio model with Input/Output MLP Architecture for compatibility with orchestrator"""

    def __init__(
        self, 
        model_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
        lora: bool = True,
        low_resource: bool = False,
        lora_rank: int = 8,
        Lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        prompt_template: str = "",
        max_txt_len: int = 512,
        ckpt_path: str = None,
        device=None, 
        use_fp16: bool = True):

        """Initialize the MLPQwen model"""
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16

        logging.info("=" * 80)
        logging.info("🔧 INITIALIZING MLP-QWEN MODEL")
        logging.info("=" * 80)
        logging.info(f"Device: {self.device}")
        logging.info(f"LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

        logging.info(f"🔄 Loading base Qwen2 Audio model from {model_path}")
        start_time = time.time()

        sys.stdout.flush()  # ✅ Force flush stdout
        logging.getLogger().handlers[0].flush() 

        # Load Qwen2 model, processor and 
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        self.input_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = self.input_processor.tokenizer 

        logging.info(f"Loaded Qwen2 Audio model successfully in {time.time() - start_time:.2f} seconds")
        logging.info(f"Loaded tokenizer with vocab size: {self.tokenizer.vocab_size}")

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        logging.info("Frozen all model parameters")

        # Move model to device
        logging.info(f"Moving model to device: {self.device}")
        self.model.to(self.device)
        sys.stdout.flush() 

        # Add LoRA if specified
        self.lora = lora

        # if lora:
        #     lora_config = LoraConfig(
        #         r=lora_rank,
        #         lora_alpha=lora_alpha,
        #         target_modules=["q_proj", "k_proj"],
        #         lora_dropout=lora_dropout,
        #         inference_mode=False,
        #         task_type=TaskType.CAUSAL_LM,
        #     )
        #     self.model = get_peft_model(self.model, lora_config)
        #     logging.info("Applied LoRA to Qwen2 Audio model")

       # Load checkpoint if provided
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            logging.info(f"Loaded checkpoint from {ckpt_path}")

        self.batch_counter = 0

        # Get embedding module
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'model'):
                self.embed_module = self.model.model.model.embed_tokens
            else:
                self.embed_module = self.model.model.embed_tokens
        else:
            self.embed_module = self.model.embed_tokens

    def forward(self, samples):
        """
        Forward pass for training.
        """
        start_time = time.time()
        logging.debug(f"Starting forward pass for batch {self.batch_counter}")
        
        # Extract and convert inputs with proper types
        input_ids = samples["input_ids"].to(self.device).long()  # Ensure long type for embeddings
        attention_mask = samples["attention_mask"].to(self.device)
        input_features = samples.get("input_features")
        feature_attention_mask = samples.get("feature_attention_mask")
        
        # Create labels tensor - set to -100 for prompt tokens
        # Create labels tensor with matching dtype
        labels = torch.full_like(input_ids, -100, device=self.device, dtype=torch.long)
        for i, prompt_len in enumerate(samples["prompt_length"]):
            labels[i, prompt_len:] = input_ids[i, prompt_len:].clone()
        
        # Debug logging for first batch
        if self.batch_counter == 0:
            logging.info("=== First Batch Token Debug ===")
            logging.info(f"Last 20 input_ids: {input_ids[0][-20:].tolist()}")
            logging.info(f"Last 20 labels: {labels[0][-20:].tolist()}")
            logging.info(f"Last 20 attention_mask: {attention_mask[0][-20:].tolist()}")
            logging.info(f"Last 20 feature_attention_mask: {feature_attention_mask[0][-20:].tolist()}")
            
            # Add input features debug info
            logging.info(f"Input features shape: {input_features.shape}")
            logging.info(f"Last 20 input features (first dimension): {input_features[0, -20:, 0].tolist()}")
            logging.info(f"Input features min: {input_features.min().item()}")
            logging.info(f"Input features max: {input_features.max().item()}")
            logging.info(f"Input features mean: {input_features.mean().item()}")
            
            decoded_last_input = self.input_processor.tokenizer.decode(input_ids[0][-20:])
            decoded_last_labels = self.input_processor.tokenizer.decode([x for x in labels[0][-20:] if x != -100])
            logging.info(f"Last 20 input tokens decoded: {decoded_last_input}")
            logging.info(f"Last 20 labels decoded: {decoded_last_labels}")
            logging.info("==============================")
        
        # Move input_features to device and convert to float16 if present
        if input_features is not None:
            input_features = input_features.to(self.device)
            if self.use_fp16:
                input_features = input_features.half()  # Use .half() instead of to(torch.float16)
        
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(self.device)
          
        # Forward pass with consistent dtype
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
                return_dict=True
            )
        
        logging.debug(f"Forward pass completed in {time.time() - start_time:.2f} seconds")
        
        # Log loss for first few batches
        if self.batch_counter < 5:
            logging.info(f"Batch {self.batch_counter} loss: {outputs.loss.item():.4f}")
        
        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}

    def generate_output(self, batch):
        """
        Generate predictions following inference_llama2_salmon_final.py
        """
        # Convert input tensors to proper types
        input_ids = batch["input_ids"].to(self.device).long()  # Ensure long type
        attention_mask = batch["attention_mask"].to(self.device)
        input_features = batch.get("input_features")
        feature_attention_mask = batch.get("feature_attention_mask")


        # Check empty feature tensors 
        if input_features is not None and input_features.shape[0] == 0:
            input_features = None
        if feature_attention_mask is not None and feature_attention_mask.shape[0] == 0:
            feature_attention_mask = None
            

        # Handle input features with consistent dtype
        if input_features is not None:
            input_features = input_features.to(self.device)
            if self.use_fp16:
                input_features = input_features.half()  # Use .half() instead of to(torch.float16)
        
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(self.device)

        # Generate with consistent dtype
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=10,
            )
            
            if generated_ids.size(1) <= input_ids.size(1):
                logging.error(f"Generated sequence ({generated_ids.size(1)}) not longer than input ({input_ids.size(1)})")
                return [""] * input_ids.size(0)
                
            generated_ids = generated_ids[:, input_ids.size(1):]
            
            outputs = self.input_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return outputs


def generate_one_word_two_token_symbols(num_symbols, tokenizer):
    """Generate 2-token symbols"""
    import random
    import string

    chars = string.ascii_lowercase
    two_token_words = []
    used_words = set()

    attempts = 0
    max_attempts = 10000

    while (len(two_token_words) < num_symbols) and (attempts < max_attempts):
        attempts += 1
        word_length = random.choice([4, 5])
        word = ''.join(random.choice(chars) for _ in range(word_length))

        if word in used_words:
            continue

        used_words.add(word)

        try:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 2:
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                if decoded.lower() == word.lower():
                    two_token_words.append(word)
        except Exception:
            continue

    return two_token_words[:num_symbols]

def create_label_mapping(original_labels, random_symbols):
    """Create simple label to symbol mapping"""
    return {orig: rand for orig, rand in zip(original_labels, random_symbols)}
    

