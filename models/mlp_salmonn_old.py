import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import nullcontext

# Import PEFT for LoRA
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperFeatureExtractor

# Path setup for SALMONN import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SALMONN.models.salmonn_org import SALMONN

class MLPSalmonn(nn.Module):
    """SIMPLIFIED SALMONN with MLP - compatible with unified_symbol_training.py"""
    
    def __init__(
        self,
        llama_path="lmsys/vicuna-13b-v1.1",
        whisper_path="openai/whisper-large-v2",
        beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        label_tokens=None,
        hidden_dim=None,
        dropout=0.1,
        freeze_base=True,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        device=None,
        low_resource=False
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # SIMPLIFIED SALMONN config
        salmonn_config = {
            "llama_path": llama_path,
            "whisper_path": whisper_path,
            "beats_path": beats_path,
            "lora": True,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "low_resource": False,
            "use_speech_Qformer": True,
            "freeze_whisper": True,
            "freeze_beats": True,
            "freeze_speech_QFormer": False,
            "num_speech_query_token": 1,
            "window_level_Qformer": True,
            "second_per_window": 0.333333,
            "second_stride": 0.333333,
            "speech_llama_proj_model": "",
            "freeze_speech_llama_proj": False,
            "ckpt": "/home/sriramg/aneeraj/storage/salmonn_v1.pth"
        }

        # Initialize SALMONN
        logging.info("Loading base SALMONN model...")
        self.salmonn = SALMONN.from_config(salmonn_config)
        logging.info("Base SALMONN model loaded successfully")
        self.salmonn.to(self.device)
        
        # SIMPLIFIED attributes
        self.label_tokens = label_tokens or []
        self.label_token_ids = []
        self.batch_counter = 0
        self.speech_placeholder = "<SpeechHere>"
        self.bypass_mlp_during_lora = False
        
        # ✅ FIX: Store original label mapping for discovery
        self.original_to_random_mapping = {}  # Will store {"alpha": "duhl"}
        
        # Get model components
        self.llama_model = self.salmonn.llama_model
        self.llama_tokenizer = self.salmonn.llama_tokenizer
        
        self.current_cycle = 0
        # Get embedding module
        if hasattr(self.llama_model, 'model'):
            if hasattr(self.llama_model.model, 'model'):
                self.embed_module = self.llama_model.model.model.embed_tokens
            else:
                self.embed_module = self.llama_model.model.embed_tokens
        else:
            self.embed_module = self.llama_model.embed_tokens
        
        self.embed_dim = self.embed_module.weight.shape[1]
        self.hidden_dim = hidden_dim or self.embed_dim
        
        # SIMPLIFIED MLP
        self.position_wise_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # ✅ Add layer norm for stability
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # Add this after your existing position_wise_mlp (around line 100)

        # INPUT MLP (existing - rename for clarity)
        self.input_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # OUTPUT MLP (new - symmetric to input MLP)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # Get lm_head for final projection
        if hasattr(self.llama_model, 'lm_head'):
            self.lm_head = self.llama_model.lm_head
        elif hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'lm_head'):
            self.lm_head = self.llama_model.model.lm_head
        else:
            # Create lm_head if not found
            self.lm_head = nn.Linear(self.embed_dim, self.llama_tokenizer.vocab_size, bias=False)

        # SIMPLIFIED initialization
        with torch.no_grad():
            for layer in self.position_wise_mlp:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=2.0)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    torch.nn.init.constant_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)
        
        self.position_wise_mlp.to(self.device)
        self.input_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        
        # Convert label tokens to token IDs
        if self.label_tokens:
            self.label_token_ids = self._get_label_token_ids()
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.salmonn.parameters():
                param.requires_grad = False
            for param in self.position_wise_mlp.parameters():
                param.requires_grad = True
        
        # SIMPLIFIED attributes for compatibility
        self.original_embed_matrix = self.embed_module.weight.clone().detach()
        self.original_embed_matrix.requires_grad = False
        self.symbol_mappings = {}
    
    
    def apply_mlp_to_embeddings(self, embeddings, token_ids, temperature=0.1, hard_quantization=False):
        # ✅ BYPASS MLP during LoRA training
        if hasattr(self, 'mlp_training_mode') and not self.mlp_training_mode:
            logging.info("LoRA mode: BYPASSING MLP transformation")
            return embeddings  # No MLP transformation
        
        # Only apply MLP during MLP training
        logging.info("MLP mode: APPLYING MLP transformation")
        
        # ✅ EMERGENCY FIX: Force dtype consistency
        target_dtype = embeddings.dtype
        target_device = embeddings.device
        
        # Ensure MLP matches input dtype
        logging.info(f"Converting MLP to device {target_device} with dtype {target_dtype}")
        self.position_wise_mlp = self.position_wise_mlp.to(dtype=target_dtype)
        
        # ✅ Add shape debugging
        logging.info(f"MLP input shape: {embeddings.shape}")
        logging.info(f"Token IDs shape: {token_ids.shape}")
        
        # ✅ FIX: Flatten 3D tensors to 2D if needed
        original_shape = embeddings.shape
        if len(embeddings.shape) == 3:
            # Reshape [batch, seq_len, hidden] -> [batch*seq_len, hidden]
            embeddings = embeddings.view(-1, embeddings.shape[-1])
            if len(token_ids.shape) == 2:
                token_ids = token_ids.view(-1)
            logging.info(f"Reshaped from {original_shape} to {embeddings.shape}")
        
        # ✅ Ensure proper tensor shapes before MLP
        if len(embeddings.shape) != 2:
            logging.error(f"Invalid embedding shape: {embeddings.shape}, expected 2D")
            return embeddings.view(original_shape) if len(original_shape) == 3 else embeddings
        
        # Ensure all computations use the same dtype
        positions_to_transform = []
        for label_token_id in self.label_token_ids:
            positions = (token_ids == label_token_id).nonzero(as_tuple=True)[0]
            positions_to_transform.extend(positions.tolist())
        
        if not positions_to_transform:
            return embeddings

        positions_tensor = torch.tensor(positions_to_transform, device=target_device)
        to_transform = embeddings[positions_tensor].to(dtype=target_dtype)
        
        # Apply MLP with error handling
        try:
            with torch.cuda.amp.autocast(enabled=False):
                mlp_output = self.input_mlp(to_transform)
                
                adaptive_scale = 1
                transformed = to_transform + adaptive_scale * mlp_output  #X_hat <- X +alph* MLP(X)
                
        except Exception as e:
            logging.error(f"MLP transformation failed: {e}")
            return embeddings
        
        try:
            # Normalize transformed embeddings
            transformed_norm = F.normalize(transformed, p=2, dim=-1)
            
            # Get vocabulary embeddings (normalized)
            vocab_embeds = self.embed_module.weight #70K * 5128
            vocab_norm = F.normalize(vocab_embeds, p=2, dim=-1)
            
            # Compute cosine similarities
            similarities = torch.mm(transformed_norm, vocab_norm.t()) # S_hat
            
            # ✅ CHOOSE QUANTIZATION TYPE BASED ON FLAG
            if hard_quantization:
                # Hard quantization for targets
                hard_indices = torch.argmax(similarities, dim=-1)
                final_embeddings = self.embed_module.weight[hard_indices]
            else:
                # Soft quantization for training
                soft_weights = F.softmax(similarities / temperature, dim=-1)
                final_embeddings = torch.mm(soft_weights, vocab_embeds)
            
            # ✅ COLLECT DISCOVERIES (works for both hard and soft)
            if (hasattr(self, 'store_discoveries') and self.store_discoveries):
                logging.info("✓ MLP Discovery collection is ACTIVE")
                if not hasattr(self, 'soft_embeddings_by_token_id'):
                    self.soft_embeddings_by_token_id = {}
                
                # ✅ ADD TOP-K LOGGING HERE
                top_k_similarities, top_k_indices = torch.topk(similarities, k=5, dim=-1)
                
                hard_indices = torch.argmax(similarities, dim=-1)  # Keep existing discovery logic
                
                hard_embeddings = self.embed_module.weight[hard_indices]
                hard_norm = F.normalize(hard_embeddings, p=2, dim=-1)
                soft_norm = F.normalize(final_embeddings, p=2, dim=-1)
                hard_soft_sim = F.cosine_similarity(hard_norm, soft_norm, dim=-1)


                # Initialize storage (existing code)
                if not hasattr(self, 'discovery_similarities'):
                    self.discovery_similarities = {}
                if not hasattr(self, 'discovered_mappings'):
                    self.discovered_mappings = {}
                
                for i, pos_idx in enumerate(positions_to_transform):
                    orig_token_id = token_ids[positions_tensor[i]].item()
                    discovered_token_id = hard_indices[i].item()
                    similarity_score = similarities[i, discovered_token_id].item()
                    

                    # Store existing mappings (keep this)
                    self.discovered_mappings[orig_token_id] = discovered_token_id
                    self.soft_embeddings_by_token_id[orig_token_id] = final_embeddings[i].clone().detach()

                    random_token_text = self.llama_tokenizer.decode([orig_token_id], skip_special_tokens=True)
                    discovered_token_text = self.llama_tokenizer.decode([discovered_token_id], skip_special_tokens=True)
                    
                    similarity_key = f"{orig_token_id}->{discovered_token_id}"
                    self.discovery_similarities[similarity_key] = {
                        'similarity': similarity_score,
                        'random_token': orig_token_id,
                        'random_text': random_token_text,
                        'discovered_token': discovered_token_id,
                        'discovered_text': discovered_token_text
                    }

                    # ✅ NEW: Log top-5 alternatives (only for first 2 tokens to avoid spam)
                    if i < 2:
                        logging.info(f"=== Token {orig_token_id}('{random_token_text}') Top 5 Alternatives ===")
                        for k in range(5):
                            alt_token_id = top_k_indices[i, k].item()
                            alt_similarity = top_k_similarities[i, k].item()
                            alt_text = self.llama_tokenizer.decode([alt_token_id], skip_special_tokens=True)
                            is_original = "🔸" if alt_token_id == orig_token_id else "  "
                            logging.info(f"  {is_original}{k+1}. {alt_token_id}('{alt_text}') [sim: {alt_similarity:.4f}]")
                        
                        logging.info(f"Stored soft embedding for token {orig_token_id}")
                        
                        h_s_similarity = hard_soft_sim[i].item()
                        hard_token_id = hard_indices[i].item()

                        orig_text = self.llama_tokenizer.decode([orig_token_id], skip_special_tokens=True)
                        hard_text = self.llama_tokenizer.decode([hard_token_id], skip_special_tokens=True)
                        
                        logging.info(f"Position {i}: '{orig_text}' -> hard '{hard_text}' [cos_sim: {similarity_score:.4f}]")
                        logging.info(f"  Hard-Soft embedding similarity: {h_s_similarity:.4f} (temp: {temperature})")
                        
                        # Log temperature effect
                        max_soft_weight = torch.max(soft_weights[i]).item()
                        logging.info(f"  Max soft weight: {max_soft_weight:.4f} (should be close to 1.0 for low temp)")
                        # Keep existing discovery logging (only first few)
                    if i < 3:
                        logging.info(f"Discovery: token {orig_token_id}('{random_token_text}') -> token {discovered_token_id}('{discovered_token_text}') [sim: {similarity_score:.4f}]")
            
            # Update embeddings
            output_embeds = embeddings.clone()
            if len(embeddings.shape) == 2:
                output_embeds[positions_tensor] = final_embeddings
            else:
                batch_indices = torch.zeros_like(positions_tensor)
                output_embeds[batch_indices, positions_tensor] = final_embeddings
            
            # ✅ Reshape back to original if needed
            if len(original_shape) == 3 and final_embeddings is not None:
                final_embeddings = final_embeddings.view(original_shape)
            
            return output_embeds
            
        
        except Exception as e:
            logging.error(f"Error in MLP transformation: {e}")
            return embeddings


    def compute_mlp_loss(self, wrapped_embeds, wrapped_atts, target_tokens):
        """MLP loss using extracted soft embeddings and mapped token IDs"""
        
        logging.info("=== MLP Loss (Using Soft Embeddings + Mapped IDs) ===")
        logging.info(f"wrapped_embeds shape: {wrapped_embeds.shape}")
        logging.info(f"target_tokens.input_ids shape: {target_tokens.input_ids.shape}")
        
        try:
            # ✅ SINGLE FUNCTION CALL FOR BOTH EMBEDDINGS AND IDS
            target_embeds_soft, quantized_target_ids = self.extract_target_embeddings_and_ids(
                wrapped_embeds, target_tokens
            )
            
            # Combine: soft prompt + soft targets
            combined_embeds = torch.cat([wrapped_embeds, target_embeds_soft], dim=1)
            combined_attention_mask = torch.cat([wrapped_atts, target_tokens.attention_mask], dim=1)
            
            # Labels: use quantized (mapped) token IDs
            prompt_length = wrapped_embeds.size(1)
            batch_size = wrapped_embeds.size(0)
            target_length = quantized_target_ids.size(1)
            
            labels = torch.full(
                (batch_size, prompt_length + target_length),
                fill_value=-100,
                dtype=torch.long,
                device=wrapped_embeds.device
            )
            labels[:, prompt_length:] = quantized_target_ids
            
            # Forward pass with consistent soft embeddings
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    labels=labels,
                    return_dict=True,
                    use_cache=False
                )
            
            base_loss = outputs.loss
            loss_scale = 1.0  # No scaling for now
            scaled_loss = base_loss * loss_scale
            
            logging.info(f"Soft embedding loss: {scaled_loss:.6f}")
            
            # ✅ Check for NaN loss
            if torch.isnan(scaled_loss).any():
                logging.error("NaN loss detected - returning zero loss")
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
            
            return {
                "loss": scaled_loss,
                "logits": outputs.logits,
                "quantized_labels": quantized_target_ids
            }
            
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"CUDA OOM in MLP loss: {e}")
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
            
        except Exception as e:
            logging.error(f"MLP loss computation failed: {e}")
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

    def compute_standard_loss(self, wrapped_embeds, wrapped_atts, target_tokens):
        """Standard loss function for LoRA training"""
        # Get target embeddings (no MLP transformation)
        target_embeds = self.embed_module(target_tokens.input_ids)
        
        # Create labels for standard training
        prompt_length = wrapped_embeds.size(1)
        labels = torch.full(
            [wrapped_atts.shape[0], wrapped_atts.shape[1] + target_tokens.input_ids.size(1)],
            fill_value=-100,
            dtype=torch.long,
            device=wrapped_embeds.device
        )
        labels[:, prompt_length:] = target_tokens.input_ids
        
        attention_mask = torch.cat([wrapped_atts, target_tokens.attention_mask], dim=1)
        labels[attention_mask == 0] = -100
        
        # Standard LLaMA forward
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=torch.cat([wrapped_embeds, target_embeds], dim=1),
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}
    
    def forward(self, samples):
        """Forward pass with mode-specific loss functions"""
        
        # Process speech and wrap prompts (same as before)
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        
        if self.batch_counter == 0:
            logging.info(f"Prompt example:\n{samples['prompt'][0]}")

        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            num_examples, example_embeds, example_atts
        )
        
        # Process target text
        target_tokens = self.llama_tokenizer(
            samples["completion"],
            padding='longest',
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        ).to(wrapped_embeds.device)
        
        self.batch_counter += 1

        # ✅ Use different loss functions based on training mode
        if hasattr(self, 'mlp_training_mode') and self.mlp_training_mode:
            logging.info("Using MLP loss")
            return self.compute_mlp_loss(wrapped_embeds, wrapped_atts, target_tokens)
        else:
            logging.info("Using standard token-level loss")
            return self.compute_standard_loss(wrapped_embeds, wrapped_atts, target_tokens)

    def get_speech_embeddings(self, samples):
        """
        Processes speech inputs to generate embeddings.
        Implementation from CustomSALMONN.
        """
        start_time = time.time()

        # Check if this is SQA dataset with question and document audio
        is_sqa = (
            "question_spectrogram" in samples and 
            "document_spectrogram" in samples
        )

        if is_sqa:
            # Process question and document audio
            has_main_speech = (
                samples["question_spectrogram"] is not None and 
                samples["document_spectrogram"] is not None
            )
            has_examples = (
                "example_question_spectrograms" in samples and 
                "example_document_spectrograms" in samples and 
                samples["example_question_spectrograms"] is not None and 
                samples["example_document_spectrograms"] is not None
            )
        else:
            # Original logic for other datasets
            has_main_speech = "spectrogram" in samples and samples["spectrogram"] is not None
            has_examples = "example_spectrograms" in samples and samples["example_spectrograms"] is not None

        # Add detailed logging for first batch
        if self.batch_counter == 0 and not is_sqa:
            if has_main_speech:
                logging.info("=== Input Data Debug (First 5 values) ===")
                logging.info(f"Spectrogram dtype: {samples['spectrogram'].dtype}")
                logging.info("Spectrogram first 5 values:")
                spec_flat = samples["spectrogram"][0].flatten()  # First batch item
                logging.info(f"{spec_flat[:10].tolist()}")
                logging.info(f"{spec_flat[-10:].tolist()}")
                
                if samples.get("raw_wav") is not None:
                    logging.info(f"\nRaw WAV dtype: {samples['raw_wav'].dtype}")
                    logging.info("Raw WAV first 5 values:")
                    wav_flat = samples["raw_wav"][0].flatten()  # First batch item
                    logging.info(f"{wav_flat[:10].tolist()}")
                    logging.info(f"{wav_flat[-10:].tolist()}")

                if "padding_mask" in samples:
                    logging.info("=== Padding Mask Debug ===")
                    padding_mask = samples["padding_mask"]
                    logging.info(f"Padding mask dtype: {padding_mask.dtype}")
                    logging.info(f"Padding mask shape: {padding_mask.shape}")
                    logging.info(f"Padding mask first 5 values: {padding_mask[0, :5].tolist()}")
                    logging.info(f"Non-padded length: {padding_mask.sum().item()}")
                    logging.info(f"Padding percentage: {(1 - padding_mask.float().mean()) * 100:.2f}%")

        if self.batch_counter == 0:
            if has_main_speech:
                logging.info("Valid main speech data detected")
            else:
                logging.info("No valid main speech data detected")
            
            if has_examples:
                logging.info("Valid speech examples detected")
            else:
                logging.info("No valid speech examples detected")
        
        def to_device_and_dtype(tensor):
            if tensor is None:
                return None
            tensor = tensor.to(device=self.device)
            if hasattr(self, 'use_fp16') and self.use_fp16 and tensor.dtype == torch.float32:
                tensor = tensor.to(dtype=torch.float16)
            return tensor

        # Process main inputs
        speech_embeds = speech_atts = None
        if has_main_speech:

            # Standard speech processing
            spectrograms = to_device_and_dtype(samples["spectrogram"])
            padding_mask = to_device_and_dtype(samples.get("padding_mask"))
            raw_wav = to_device_and_dtype(samples.get("raw_wav"))
            
            speech_embeds, speech_atts = self.encode_speech(spectrograms, raw_wav, padding_mask)
    
        # Process examples
        example_embeds = example_atts = None
        if has_examples:
            # Standard example processing
            example_specs = to_device_and_dtype(samples["example_spectrograms"])
            example_padding = to_device_and_dtype(samples.get("example_padding_masks"))
            example_raw = to_device_and_dtype(samples.get("example_raw_wavs"))
            
            # Initialize lists to store example embeddings
            batch_example_embeds = []
            batch_example_atts = []
            
            # Process each batch
            for b in range(len(example_specs)):
                example_count = len(example_specs[b])
                batch_embeds = []
                batch_atts = []
                
                # Process each example in this batch
                for i in range(example_count):
                    spec = example_specs[b][i]
                    
                    # Get padding mask and raw wav if available
                    pad = example_padding[b][i] if example_padding is not None else None
                    raw = example_raw[b][i] if example_raw is not None else None
                    
                    # Encode example speech
                    embed, att = self.encode_speech(spec.unsqueeze(0), raw, pad)
                    
                    # Add to batch
                    batch_embeds.append(embed.squeeze(0))
                    batch_atts.append(att.squeeze(0))
                
                batch_example_embeds.append(batch_embeds)
                batch_example_atts.append(batch_atts)
                
                example_embeds = batch_example_embeds
                example_atts = batch_example_atts
        
        logging.info(f"Speech embedding generation took {time.time() - start_time:.2f} seconds")
        
        return speech_embeds, speech_atts, example_embeds, example_atts
        
    def custom_prompt_wrap(self, embeds, atts, prompts, num_examples=None, example_embeds=None, example_atts=None):
        """Wraps speech embeddings with text prompts and examples - reduced logging"""
        # Infer device from model parameters
        device = next(self.parameters()).device
        
        # Add flag for SQA dataset - only check embeds
        is_sqa = isinstance(embeds, tuple)
        
        batch_size = len(prompts)
        max_examples = num_examples.max().item() if num_examples is not None else 0
        batch_embeds, batch_atts = [], []
        
        for b in range(batch_size):
            parts = []
            suffix = prompts[b]
            if max_examples > 0 and example_embeds is not None:

                # Original example handling
                for i in range(max_examples):
                    example_marker = f"<Example{i}>"
                    if example_marker in suffix:
                        before_example, after_example = suffix.split(example_marker, 1)
                        parts.append(before_example)
                        suffix = after_example
                    else:
                        parts.append("")

            if self.speech_placeholder in suffix:
                before_speech, after_speech = suffix.split(self.speech_placeholder)
                parts.append(before_speech)
                suffix = after_speech
            else:
                # For text-only mode, we don't split on speech placeholder
                parts.append(suffix)
                suffix = ""
            parts.append(suffix)
            
            # Process text parts - Ensure everything is on the correct device and dtype
            tokenized_parts = [
                self.llama_tokenizer(part, padding="longest", return_tensors="pt", add_special_tokens=False) 
                for part in parts
            ]
            # Move to device and ensure input_ids are Long
            tokenized_parts = [
                {k: v.to(device=device, dtype=torch.long if k == 'input_ids' else v.dtype) 
                 for k, v in tokens.items()}
                for tokens in tokenized_parts
            ]
            
            # Get embeddings from the original embedding layer
            part_embeds = []
            for tokens in tokenized_parts:
                part_embed = self.embed_module(tokens['input_ids'].squeeze(0))

                # Only log for first few batches
                if self.batch_counter < 3:
                    part_text = self.llama_tokenizer.decode(tokens['input_ids'].squeeze(0), skip_special_tokens=True)
                    logging.info(f"Processing prompt part: '{part_text[:50]}...' ({len(tokens['input_ids'].squeeze(0))} tokens)")
                
                # Apply MLP if training and not bypassed
                if self.label_token_ids:
                    original_shape = part_embed.shape
                    part_embed = self.apply_mlp_to_embeddings(
                        part_embed, 
                        tokens['input_ids'].squeeze(0)
                    )
                    if self.batch_counter < 3:
                        logging.info(f"Prompt part processed: shape {original_shape} -> {part_embed.shape}")
                
                part_embeds.append(part_embed)
                
            part_atts = [tokens['attention_mask'].squeeze(0) for tokens in tokenized_parts]
            
            # Combine embeddings
            combined_embeds, combined_atts = [], []
            
            

            for i in range(len(part_embeds) - 2):
                combined_embeds.append(part_embeds[i])
                combined_atts.append(part_atts[i])
                
                if i < max_examples and example_embeds is not None:
                    # Add example embeddings if available for this batch and example index
                    if b < len(example_embeds) and i < len(example_embeds[b]):
                        combined_embeds.append(example_embeds[b][i])
                        combined_atts.append(example_atts[b][i])
            


            # Add final parts
            if embeds is not None:  # Speech mode
                combined_embeds.extend([part_embeds[-2], embeds[b], part_embeds[-1]])
                combined_atts.extend([part_atts[-2], atts[b], part_atts[-1]])
            else:  # Text-only mode
                # For text-only, concatenate the final parts without speech embeddings
                if len(part_embeds) >= 2:
                    combined_embeds.extend([part_embeds[-2], part_embeds[-1]])
                    combined_atts.extend([part_atts[-2], part_atts[-1]])
                else:
                    # If there's only one part (e.g., no speech placeholder)
                    combined_embeds.append(part_embeds[-1])
                    combined_atts.append(part_atts[-1])
            
            # Concatenate all parts
            batch_embeds.append(torch.cat(combined_embeds, dim=0))
            batch_atts.append(torch.cat(combined_atts, dim=0))
        
        return torch.stack(batch_embeds, dim=0), torch.stack(batch_atts, dim=0)
    
    # SIMPLIFIED supporting functions
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        """Encode speech inputs using SALMONN"""
        return self.salmonn.encode_speech(    
            spectrogram=spectrogram,
            raw_wav=raw_wav,
            audio_padding_mask=audio_padding_mask
        )
    
    def _get_label_token_ids(self):
        """SIMPLIFIED: Convert label tokens to token IDs"""
        token_ids = []
        self.label_to_token_ids = {}
        
        for label in self.label_tokens:
            if isinstance(label, int):
                token_ids.append(label)
                self.label_to_token_ids[label] = [label]
            else:
                tokenized = self.llama_tokenizer.encode(label, add_special_tokens=False)
                if len(tokenized) > 0:
                    token_ids.extend(tokenized)
                    self.label_to_token_ids[label] = tokenized
                    logging.info(f"Label '{label}' -> {tokenized}")
        
        return token_ids
    
    def maybe_autocast(self):
        """Context manager for autocast"""
        if self.device != "cpu" and hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp.autocast(enabled=True)
        else:
            return nullcontext()
    
    
    def freeze_mlp_weights(self):
        """Freeze MLP weights"""
        for param in self.position_wise_mlp.parameters():
            param.requires_grad = False

    def unfreeze_mlp_weights(self):
        """Unfreeze MLP weights"""
        for param in self.position_wise_mlp.parameters():
            param.requires_grad = True

    def freeze_lora_weights(self):
        """Freeze LoRA weights AND QFormer for MLP training"""
        frozen_count = 0
        
        # Freeze LoRA weights
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
                frozen_count += 1
        
        # Freeze QFormer components (speech_Qformer and speech_query_tokens)
        if hasattr(self.salmonn, 'speech_Qformer'):
            for name, param in self.salmonn.speech_Qformer.named_parameters():
                param.requires_grad = False
                frozen_count += 1
        
        if hasattr(self.salmonn, 'speech_query_tokens'):
            self.salmonn.speech_query_tokens.requires_grad = False
            frozen_count += 1
        
        # Freeze speech_llama_proj if it should be trainable during LoRA
        if hasattr(self.salmonn, 'speech_llama_proj'):
            for name, param in self.salmonn.speech_llama_proj.named_parameters():
                param.requires_grad = False
                frozen_count += 1
        
        logging.info(f"Frozen LoRA and QFormer weights ({frozen_count} parameters)")

    def unfreeze_lora_weights(self):
        """Unfreeze LoRA weights AND QFormer for LoRA training"""
        unfrozen_count = 0
        
        # Unfreeze LoRA weights
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                unfrozen_count += 1
        
        # Unfreeze QFormer components for training (since freeze_speech_QFormer=False in your config)
        if hasattr(self.salmonn, 'speech_Qformer'):
            for name, param in self.salmonn.speech_Qformer.named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
        
        if hasattr(self.salmonn, 'speech_query_tokens'):
            self.salmonn.speech_query_tokens.requires_grad = True
            unfrozen_count += 1
        
        # Unfreeze speech_llama_proj if it should be trainable during LoRA
        # Check your SALMONN config - if freeze_speech_llama_proj=False, then unfreeze it
        if hasattr(self.salmonn, 'speech_llama_proj'):
            for name, param in self.salmonn.speech_llama_proj.named_parameters():
                param.requires_grad = True  # Set to True if freeze_speech_llama_proj=False
                unfrozen_count += 1
        
        logging.info(f"Unfrozen LoRA and QFormer weights ({unfrozen_count} parameters)")

    def get_trainable_parameters(self):
        """Get trainable parameters"""
        trainable = []
        frozen = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        
        return trainable, frozen

    def set_mlp_training_mode(self):
        """Set model to MLP training mode"""
        self.mlp_training_mode = True
        self.freeze_lora_weights()
        self.unfreeze_mlp_weights()
        logging.info("✓ Set to MLP training mode - MLP unfrozen, LoRA frozen")

    def set_lora_training_mode(self):
        """Set model to LoRA training mode"""
        self.mlp_training_mode = False
        self.freeze_mlp_weights()
        self.unfreeze_lora_weights()
        logging.info("✓ Set to LoRA training mode - LoRA unfrozen, MLP frozen")

    def set_inference_mode(self):
        """Set model to inference mode"""
        self.mlp_training_mode = False
        self.freeze_mlp_weights()
        self.freeze_lora_weights()
        self.eval()  # Set PyTorch eval mode
        logging.info("✓ Set to inference mode - All weights frozen")

    def set_joint_training_mode(self):
        """Set model to joint MLP+LoRA training mode (if needed)"""
        self.mlp_training_mode = True  # Can be either True or False depending on loss function
        self.unfreeze_mlp_weights()
        self.unfreeze_lora_weights()
        logging.info("✓ Set to joint training mode - Both MLP and LoRA unfrozen")


    def get_training_mode_status(self):
        """Get current training mode status"""
        mlp_trainable = any(p.requires_grad for p in self.position_wise_mlp.parameters())
        lora_trainable = any(p.requires_grad for name, p in self.named_parameters() if 'lora' in name.lower())
        mlp_mode = getattr(self, 'mlp_training_mode', False)
        
        status = {
            'mlp_training_mode': mlp_mode,
            'mlp_weights_trainable': mlp_trainable,
            'lora_weights_trainable': lora_trainable,
            'pytorch_training_mode': self.training
        }
        
        # Determine current mode
        if mlp_trainable and not lora_trainable and mlp_mode:
            current_mode = "MLP_TRAINING"
        elif not mlp_trainable and lora_trainable and not mlp_mode:
            current_mode = "LORA_TRAINING"
        elif mlp_trainable and lora_trainable:
            current_mode = "JOINT_TRAINING"
        elif not mlp_trainable and not lora_trainable:
            current_mode = "INFERENCE"
        else:
            current_mode = "INCONSISTENT"
        
        status['current_mode'] = current_mode
        return status

    def update_label_tokens(self, symbol_mappings):
        """Update tracked label tokens from symbol mappings with empty string handling"""
        
        new_label_tokens = []
        valid_token_ids = []
        
        for label, symbol in symbol_mappings.items():
            if symbol and symbol.strip():  # Check if symbol is not empty
                try:
                    tokens = self.llama_tokenizer.encode(symbol, add_special_tokens=False)
                    if tokens:  # Check if tokenization succeeded
                        new_label_tokens.append(symbol)
                        valid_token_ids.extend(tokens)
                        logging.info(f"Symbol '{symbol}' -> tokens {tokens}")
                    else:
                        logging.warning(f"Symbol '{symbol}' tokenized to empty list - skipping")
                except Exception as e:
                    logging.error(f"Error tokenizing symbol '{symbol}': {e} - skipping")
            else:
                logging.warning(f"Empty symbol for label '{label}' - skipping")
        
        if not valid_token_ids:
            logging.error("No valid tokens found - keeping previous tokens")
            return
        
        self.label_tokens = new_label_tokens
        self.label_token_ids = list(set(valid_token_ids))
        
        logging.info(f"Updated to track {len(valid_token_ids)} tokens from {len(new_label_tokens)} symbols")

    def convert_token_mappings_to_text(self, token_mappings):
        """SIMPLIFIED: Convert token mappings to text mappings"""
        text_mappings = {}
        
        logging.info("=== Final Discovered Symbol Mappings ===")
        
        if hasattr(self, 'original_to_random_mapping') and self.original_to_random_mapping:
            for original_label, random_symbol in self.original_to_random_mapping.items():
                random_token_ids = self.llama_tokenizer.encode(random_symbol, add_special_tokens=False)
                
                discovered_texts = []
                for token_id in random_token_ids:
                    if token_id in token_mappings:
                        discovered_token_id = token_mappings[token_id]
                        # ✅ USE ALREADY-STORED TEXT
                        similarity_key = f"{token_id}->{discovered_token_id}"
                        if hasattr(self, 'discovery_similarities') and similarity_key in self.discovery_similarities:
                            discovered_text = self.discovery_similarities[similarity_key]['discovered_text']
                        else:
                            discovered_text = self.llama_tokenizer.decode([discovered_token_id], skip_special_tokens=True)
                        discovered_texts.append(discovered_text)
                    else:
                        # Keep original token text
                        original_text = self.llama_tokenizer.decode([token_id], skip_special_tokens=True)
                        discovered_texts.append(original_text)
                
                final_discovered = ''.join(discovered_texts).strip()
                text_mappings[original_label] = final_discovered
                
                logging.info(f"'{original_label}' -> '{random_symbol}' -> '{final_discovered}'")
        else:
            logging.warning("original_to_random_mapping is empty - no conversions possible")
        
        return text_mappings

    def generate_output(self, samples):
        """
        Generate predictions for speech or text input using MLP-transformed symbols.
        Based on custom_salmon.py but with MLP symbol transformation.
        """
        start_time = time.time()
        
        # Process speech embeddings (same as custom_salmon)
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        if self.batch_counter == 0:
            if speech_embeds is not None:
                if isinstance(speech_embeds, tuple):
                    # For SQA dataset with question and document embeddings
                    q_embeds, d_embeds = speech_embeds
                    logging.info("SQA Speech data detected and processed")
                    logging.info(f"Question embeddings device: {q_embeds.device}")
                    logging.info(f"Question embeddings range: {q_embeds.min():.3f} to {q_embeds.max():.3f}")
                    logging.info(f"Document embeddings device: {d_embeds.device}")
                    logging.info(f"Document embeddings range: {d_embeds.min():.3f} to {d_embeds.max():.3f}")
                else:
                    # Original logging for single speech embedding
                    logging.info("Speech data detected and processed")
                    logging.info(f"Speech embeddings device: {speech_embeds.device}")
                    logging.info(f"Speech embeddings range: {speech_embeds.min():.3f} to {speech_embeds.max():.3f}")
            else:
                logging.info("No speech data detected, using text-only mode")
        
        if self.batch_counter == 0:
            logging.info(f"Prompt example:\n{samples['prompt'][0]}")
        
        # Get number of examples per sample
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        
        # ✅ USE MLPSalmonn's custom_prompt_wrap (which includes MLP transformation)
        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            num_examples, example_embeds, example_atts
        )
        
        if self.batch_counter == 0:
            logging.info(f"Wrapped embeddings shape: {wrapped_embeds.shape}")
            logging.info(f"Wrapped attention mask shape: {wrapped_atts.shape}")
            
            # Add device information logging
            logging.info(f"Device details:")
            logging.info(f"Wrapped embeddings device: {wrapped_embeds.device}")
            logging.info(f"Wrapped attention mask device: {wrapped_atts.device}")
            logging.info(f"LLaMA model device: {next(self.llama_model.parameters()).device}")
            logging.info(f"Model main device: {self.device}")
            
            # Check if tensors are contiguous
            logging.info(f"Wrapped embeddings contiguous: {wrapped_embeds.is_contiguous()}")
            logging.info(f"Wrapped attention mask contiguous: {wrapped_atts.is_contiguous()}")
            
            # Force contiguous if needed
            if not wrapped_embeds.is_contiguous():
                wrapped_embeds = wrapped_embeds.contiguous()
                logging.info(f"Forced wrapped_embeds to be contiguous")
        
        gen_start_time = time.time()
        
        # Generate with LLaMA (same as custom_salmon)
        with torch.inference_mode():
            outputs = self.llama_model.generate(
                inputs_embeds=wrapped_embeds,
                attention_mask=wrapped_atts,
                max_new_tokens=samples.get("max_new_tokens", 10),
                num_beams=samples.get("num_beams", 1),
                do_sample=samples.get("do_sample", False),
                min_length=samples.get("min_length", 1),
                top_p=samples.get("top_p", 0.9),
                repetition_penalty=samples.get("repetition_penalty", 1.0),
                length_penalty=samples.get("length_penalty", 1.0),
                temperature=samples.get("temperature", 0.8),
                pad_token_id=self.llama_tokenizer.pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                return_dict_in_generate=False,  # Faster, since we don't need extra outputs
                output_scores=False,  
            )
        
        gen_time = time.time() - gen_start_time
        
        if self.batch_counter == 0:
            logging.info(f"Raw output tokens: {outputs[0].tolist()}")
            logging.info(f"Tokenizer vocab size: {self.llama_tokenizer.vocab_size}")
            first_pred = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"First raw prediction: {first_pred}")
        
        # Decode predictions
        predictions = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        total_time = time.time() - start_time
        if self.batch_counter == 0:
            logging.info(f"Time breakdown - Generation: {gen_time:.2f}s, Total: {total_time:.2f}s")
        
        self.batch_counter += 1
        logging.info(f"Generation took {time.time() - start_time:.2f} seconds")
        return predictions

    def get_final_discovered_symbols(self):
        """Extract final discovered symbol mappings after MLP training"""
        if not hasattr(self, 'discovered_mappings') or not self.discovered_mappings:
            logging.warning("No discovered mappings found!")
            return {}
        
        # Convert token mappings to text mappings
        final_mappings = self.convert_token_mappings_to_text(self.discovered_mappings)
        
        logging.info("=== Final Discovered Symbol Mappings ===")
        for original, discovered in final_mappings.items():
            logging.info(f"'{original}' -> '{discovered}'")
        
        return final_mappings

    def update_to_discovered_symbols(self):
        """Update model to use discovered symbols instead of random ones"""
        discovered_mappings = self.get_final_discovered_symbols()
        
        if discovered_mappings:
            # Update the symbol mappings
            self.update_label_tokens(discovered_mappings)
            logging.info("✓ Updated model to use discovered symbols for LoRA training")
            return discovered_mappings
        else:
            logging.warning("No discovered symbols found, keeping random symbols")
            return self.original_to_random_mapping

    def extract_target_embeddings_and_ids(self, wrapped_embeds, target_tokens):
        """Extract both soft embeddings and mapped token IDs with proper batch handling"""
        
        batch_size = wrapped_embeds.size(0)
        target_length = target_tokens.input_ids.size(1)
        embed_dim = wrapped_embeds.size(2)
        
        # Initialize outputs
        extracted_embeds = torch.zeros(
            batch_size, target_length, embed_dim,
            device=wrapped_embeds.device,
            dtype=wrapped_embeds.dtype
        )
        
        # Initialize quantized IDs tensor
        quantized_target_ids = torch.zeros_like(target_tokens.input_ids)
        
        logging.info(f"Processing batch_size={batch_size}, target_length={target_length}")
        
        # ✅ PROPER BATCH LOOP
        for b in range(batch_size):
            for t in range(target_length):
                orig_id = target_tokens.input_ids[b, t].item()
                
                # ✅ TOKEN MAPPING LOGIC
                if hasattr(self, 'discovered_mappings') and orig_id in self.discovered_mappings:
                    mapped_id = self.discovered_mappings[orig_id]
                    if mapped_id == 0:
                        logging.warning(f"Batch {b}, Token {t}: Mapped token {orig_id} -> 0 (empty), using original")
                        quantized_target_ids[b, t] = orig_id
                        final_token_id = orig_id
                    else:
                        quantized_target_ids[b, t] = mapped_id
                        final_token_id = mapped_id
                        if b == 0 and t < 2:  # Log only first batch, first few tokens
                            logging.info(f"Batch {b}, Token {t}: Mapped token {orig_id} -> {mapped_id}")
                else:
                    quantized_target_ids[b, t] = orig_id
                    final_token_id = orig_id
                    if b == 0 and t < 2:
                        logging.info(f"Batch {b}, Token {t}: No mapping for token {orig_id}, keeping original")
                
                # ✅ EMBEDDING EXTRACTION LOGIC
                # Try to get soft embedding for the ORIGINAL token (before mapping)
                if (hasattr(self, 'soft_embeddings_by_token_id') and 
                    orig_id in self.soft_embeddings_by_token_id):
                    
                    extracted_embeds[b, t] = self.soft_embeddings_by_token_id[orig_id]
                    if b == 0 and t < 2:  # Log only first batch, first few tokens
                        logging.info(f"Batch {b}, Token {t}: Using stored soft embedding for token {orig_id}")
                else:
                    # Fallback: use hard embedding for the MAPPED token
                    extracted_embeds[b, t] = self.embed_module.weight[final_token_id]
                    if b == 0 and t < 2:
                        logging.info(f"Batch {b}, Token {t}: Using hard embedding for token {final_token_id}")
        
        logging.info(f"Original target tokens (first batch): {target_tokens.input_ids[0].flatten().tolist()}")
        logging.info(f"Quantized target tokens (first batch): {quantized_target_ids[0].flatten().tolist()}")
        logging.info(f"Extracted target embeddings shape: {extracted_embeds.shape}")
        
        return extracted_embeds, quantized_target_ids

    def compute_mlp_loss_v2(self, wrapped_embeds, wrapped_atts, target_tokens):
        """NEW: Input MLP → LLaMA → Output MLP → Cross-entropy"""
        
        try:
            # 1. Apply Input MLP to target embeddings
            target_embeds = self.embed_module(target_tokens.input_ids)
            
            if hasattr(self, 'mlp_training_mode') and self.mlp_training_mode:
                # Apply input MLP transformation
                transformed_target_embeds = self.apply_input_mlp_transformation(target_embeds)
            else:
                transformed_target_embeds = target_embeds
            
            # 2. Combine prompt + transformed target embeddings
            combined_embeds = torch.cat([wrapped_embeds, transformed_target_embeds], dim=1)
            combined_attention_mask = torch.cat([wrapped_atts, target_tokens.attention_mask], dim=1)
            
            # 3. Forward through LLaMA (get hidden states)
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )
            
            # 4. Extract target portion of output hidden states
            prompt_length = wrapped_embeds.size(1)
            target_output_hiddens = outputs.hidden_states[-1][:, prompt_length:, :]
            
            # 5. Apply Output MLP transformation
            reconstructed_embeds = self.output_mlp(target_output_hiddens)
            
            # 6. Project to vocab space using lm_head
            logits = self.lm_head(reconstructed_embeds)
            
            # 7. Cross-entropy loss with original target token IDs
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tokens.input_ids.view(-1),
                ignore_index=-100
            )
            
            return {"loss": loss, "logits": logits}
            
        except Exception as e:
            logging.error(f"MLP v2 loss computation failed: {e}")
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

# SIMPLIFIED symbol generation functions
def generate_one_word_two_token_symbols(num_symbols, tokenizer):
    """SIMPLIFIED: Generate 2-token symbols"""
    import random
    import string
    
    chars = string.ascii_lowercase
    two_token_words = []
    used_words = set()
    
    logging.info("Searching for 2-token words...")
    
    attempts = 0
    max_attempts = 10000
    
    while len(two_token_words) < num_symbols and attempts < max_attempts:
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
                    logging.info(f"Found 2-token word #{len(two_token_words)}: '{word}' -> {token_ids}")
                    
                    token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
                    logging.info(f"  Tokens: {token_texts}")
        except:
            continue
    
    logging.info(f"Generated {len(two_token_words)} two-token words from {attempts} attempts")
    
    
    return two_token_words[:num_symbols]

def create_label_mapping(original_labels, random_symbols):
    """Create simple label to symbol mapping"""
    return {orig: rand for orig, rand in zip(original_labels, random_symbols)}
