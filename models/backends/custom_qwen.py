import sys
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from utils.training_utils import load_checkpoint
from utils.environment import get_env_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomQwen(nn.Module):
    """
    Custom implementation of Qwen2 Audio model.
    Provides a standardized interface for training and inference.
    """
    def __init__(self, 
                 model_path: str = None,
                 input_processor = None,
                 lora: bool = True,
                 low_resource: bool = True,
                 lora_rank: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 prompt_template: str = "",
                 max_txt_len: int = 512,
                 ckpt_path: str = None,
                 device=None, 
                 use_fp16: bool = True):
        """
        Initialize the CustomQwen model.
        """
        super().__init__()
        
        model_path = model_path or get_env_path("QWEN_MODEL_NAME")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        
        logging.info(f"Loading Qwen2 Audio model from {model_path}")
        start_time = time.time()
        
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        
        # LOGICAL FIX: Use the provided input_processor to ensure unified tokenization
        if input_processor is not None:
            self.input_processor = input_processor
            logging.info("CustomQwen: Using unified input_processor with slot tokens.")
        else:
            self.input_processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        
        logging.info(f"Loaded Qwen2 Audio model in {time.time() - start_time:.2f} seconds")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        if lora:
            logging.info(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj"],
                lora_dropout=lora_dropout,
                inference_mode=False, 
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            logging.info("Applied LoRA to Qwen2 Audio model")
            self.print_trainable_parameters()
        
        self.model.to(self.device)
        
        if ckpt_path:
            checkpoint = load_checkpoint(ckpt_path, map_location=device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            
        self.prompt_template = prompt_template
        self.max_txt_len = max_txt_len
        self.lora = lora
        self.batch_counter = 0
        
    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logging.info(f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)")

    def get_speech_embeddings(self, samples):
        return None, None, None, None

    def forward(self, samples, router=None, dspo_module=None):
        """
        Forward pass for training. Supports D-SPO differentiable symbol injection.
        """
        start_time = time.time()
        logging.debug(f"Starting forward pass for batch {self.batch_counter}")
        
        input_ids = samples["input_ids"].to(self.device).long()
        attention_mask = samples["attention_mask"].to(self.device)
        input_features = samples.get("input_features")
        feature_attention_mask = samples.get("feature_attention_mask")
        
        labels = torch.full_like(input_ids, -100, device=self.device, dtype=torch.long)
        for i, prompt_len in enumerate(samples["prompt_length"]):
            labels[i, prompt_len:] = input_ids[i, prompt_len:].clone()
        
        if self.batch_counter == 0:
            logging.info("=== First Batch Token Debug ===")
            # Use the unified tokenizer for accurate decoding
            decoded_last_input = self.input_processor.tokenizer.decode(input_ids[0][-30:])
            decoded_last_labels = self.input_processor.tokenizer.decode([x for x in labels[0][-30:] if x != -100])
            logging.info(f"Last 30 input tokens decoded: {decoded_last_input}")
            logging.info(f"Last 30 labels decoded: {decoded_last_labels}")
            
            if router is not None:
                logging.info("D-SPO: Router detected in forward pass. Enabling differentiable injection.")
            logging.info("==============================")
        
        if input_features is not None:
            input_features = input_features.to(self.device)
            if self.use_fp16:
                input_features = input_features.half()
        
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(self.device)
        
        # Handle D-SPO Injection
        inputs_embeds = None
        if router is not None and dspo_module is not None:
            embedding_layer = self.model.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
            
            inputs_embeds = dspo_module.inject_differentiable_symbols(
                input_ids=input_ids,
                input_embeds=inputs_embeds,
                router=router,
                embedding_layer=embedding_layer
            )
            
            if self.batch_counter == 0:
                logging.info(f"D-SPO: Injected soft embeddings. Shape: {inputs_embeds.shape}")
            
            input_ids_to_pass = None
        else:
            input_ids_to_pass = input_ids

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            outputs = self.model(
                input_ids=input_ids_to_pass,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
                return_dict=True
            )
        
        if self.batch_counter < 5:
            logging.info(f"Batch {self.batch_counter} loss: {outputs.loss.item():.4f}")
        
        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}

    def generate_output(self, batch):
        input_ids = batch["input_ids"].to(self.device).long()
        attention_mask = batch["attention_mask"].to(self.device)
        input_features = batch.get("input_features")
        feature_attention_mask = batch.get("feature_attention_mask")

        if input_features is not None and input_features.shape[0] == 0:
            input_features = None
        if feature_attention_mask is not None and feature_attention_mask.shape[0] == 0:
            feature_attention_mask = None
            
        if input_features is not None:
            input_features = input_features.to(self.device)
            if self.use_fp16:
                input_features = input_features.half()
        
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                max_new_tokens=10,
            )
            
            if generated_ids.size(1) <= input_ids.size(1):
                return [""] * input_ids.size(0)
                
            generated_ids = generated_ids[:, input_ids.size(1):]
            outputs = self.input_processor.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return outputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)
