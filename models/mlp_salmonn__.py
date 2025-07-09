import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import nullcontext
import sys

# Import PEFT for LoRA
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperFeatureExtractor

# Path setup for SALMONN import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SALMONN.models.salmonn_org import SALMONN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout, force=True)

class MLPSalmonn(nn.Module):
    """CLEAN SALMONN with Input/Output MLP Architecture"""
    
    def __init__(
        self,
        llama_path="lmsys/vicuna-13b-v1.1",
        whisper_path="openai/whisper-large-v2",
        beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        label_tokens=None,
        hidden_dim=None,
        dropout=0.1,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        device=None,
        low_resource=False,
        use_output_mlp=True,  # Control output MLP usage
        bypass_mlp=False      # NEW: If True, don't create any MLP layers
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_output_mlp = use_output_mlp
        self.bypass_mlp = bypass_mlp  # NEW: Store bypass flag
        
        # SALMONN config
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
        logging.info("=" * 80)
        logging.info("🔧 INITIALIZING MLP-SALMONN MODEL")
        logging.info("=" * 80)
        logging.info(f"Device: {self.device}")
        logging.info(f"Bypass MLP: {self.bypass_mlp}")
        logging.info(f"Use Output MLP: {self.use_output_mlp}")
        logging.info(f"LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

        logging.info("🔄 Loading base SALMONN model...")
        
        sys.stdout.flush()  # ✅ Force flush stdout
        logging.getLogger().handlers[0].flush() 
        # Initialize SALMONN
        logging.info("Loading base SALMONN model...")
        self.salmonn = SALMONN.from_config(salmonn_config)
        logging.info("Base SALMONN model loaded successfully")
        self.salmonn.to(self.device)
        sys.stdout.flush() 
        
        # ✅ ADD: Symbol LoRA adapter on top of SALMONN
        # if hasattr(self.salmonn, 'llama_model'):
        #     symbol_lora_config = LoraConfig(
        #         task_type="CAUSAL_LM",
        #         r=4,                   # ✅ Higher rank for symbol learning
        #         lora_alpha=8,          # ✅ Strong adaptation
        #         lora_dropout=0.1,
        #         target_modules=[        
        #             "q_proj", "k_proj", "v_proj", "o_proj",    
        #             "gate_proj", "up_proj", "down_proj",  
        #         ],
        #         bias="none"
        #     )
            
        #     # ✅ Apply Symbol LoRA to LLaMA model (which already has SALMONN's LoRA)
        #     self.symbol_lora_model = get_peft_model(
        #         self.salmonn.llama_model, 
        #         symbol_lora_config
                
        #     )
            
        #     logging.info("✅ Added Symbol LoRA adapter with rank 32")
        #     sys.stdout.flush() 
        # else:
        #     logging.warning("⚠️ Could not add Symbol LoRA - llama_model not found")
        #     sys.stdout.flush() 
        #     self.symbol_lora_model = None
        
        # Essential attributes only
        self.label_tokens = label_tokens or []
        self.label_token_ids = []
        self.batch_counter = 0
        self.speech_placeholder = "<SpeechHere>"
        
        # Get model components
        self.llama_model = self.salmonn.llama_model
        self.llama_tokenizer = self.salmonn.llama_tokenizer
        
        # Get embedding module
        if hasattr(self.llama_model, 'model'):
            if hasattr(self.llama_model.model, 'model'):
                self.embed_module = self.llama_model.model.model.embed_tokens
            else:
                self.embed_module = self.llama_model.model.embed_tokens
        else:
            self.embed_module = self.llama_model.embed_tokens
        
        # self.embed_dim = self.embed_module.weight.shape[1]
        # self.hidden_dim = hidden_dim or self.embed_dim
        
        # CREATE MLP LAYERS ONLY IF NOT BYPASSED
        
        # if not self.bypass_mlp:
        #     dropout =  0.2
        #     # INPUT MLP (for transforming input embeddings)
        #     self.input_mlp = nn.Sequential(
        #         nn.Linear(self.embed_dim, self.hidden_dim),
        #         nn.LayerNorm(self.hidden_dim),
        #         nn.GELU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(self.hidden_dim, self.embed_dim),
        #         nn.LayerNorm(self.embed_dim),
        #         nn.Dropout(dropout)
        #     )

        #     # OUTPUT MLP (for transforming LLaMA output embeddings) - optional
        #     if self.use_output_mlp:
        #         self.output_mlp = nn.Sequential(
        #             nn.Linear(self.embed_dim, self.hidden_dim),
        #             nn.LayerNorm(self.hidden_dim),
        #             nn.GELU(),
        #             nn.Linear(self.hidden_dim, self.embed_dim),
        #             nn.LayerNorm(self.embed_dim)  
        #         )
        #     else:
        #         self.output_mlp = None
        #         logging.info("Output MLP disabled")

        #     # Xavier initialization for MLPs
        #     self._initialize_mlp_weights(self.input_mlp)
        #     if self.output_mlp is not None:
        #         self._initialize_mlp_weights(self.output_mlp)
            
        #     self.input_mlp.to(self.device)
        #     if self.output_mlp is not None:
        #         self.output_mlp.to(self.device)
                
        #     logging.info(f"MLPs created: Input MLP=True, Output MLP={self.use_output_mlp}")
        # else:
        #     # NO MLP LAYERS CREATED
        #     self.input_mlp = None
        #     self.output_mlp = None
        #     logging.info("🚫 BYPASS_MLP=True: No MLP layers created")
        
        # Get lm_head - only needed for MLP architecture
        # if not self.bypass_mlp:
        #     # Get lm_head - only needed for MLP architecture
        #     if hasattr(self.llama_model, 'lm_head'):
        #         self.lm_head = self.llama_model.lm_head
        #         logging.info("Found lm_head at llama_model.lm_head")
        #     elif hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'lm_head'):
        #         self.lm_head = self.llama_model.model.lm_head
        #         logging.info("Found lm_head at llama_model.model.lm_head")
        #     else:
        #         raise RuntimeError("lm_head not found in LLaMA model! Cannot proceed without output projection layer.")
        # else:
        #     # For bypass mode, we don't extract lm_head separately
        #     # The model will use the built-in lm_head through standard forward calls
        #     self.lm_head = None
        #     logging.info("🚫 BYPASS_MLP=True: Not extracting lm_head separately")

        # self.debug_salmonn_lora_targets()
    
    # def _initialize_mlp_weights(self, mlp):
    #     """Xavier initialization for MLP layers"""
    #     with torch.no_grad():
    #         for layer in mlp:
    #             if isinstance(layer, nn.Linear):
    #                 torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
    #                 if layer.bias is not None:
    #                     torch.nn.init.zeros_(layer.bias)
    #             elif isinstance(layer, nn.LayerNorm):
    #                 torch.nn.init.constant_(layer.weight, 1.0)
    #                 torch.nn.init.constant_(layer.bias, 0.0)


    def _initialize_mlp_weights(self, mlp):
        """Initialize MLP to be identity transformation initially"""
        with torch.no_grad():
            for i, layer in enumerate(mlp):
                if isinstance(layer, nn.Linear):
                    if i == len(mlp) - 2:  # Last Linear layer before final LayerNorm
                        # Initialize to output zeros -> residual connection gives identity
                        torch.nn.init.zeros_(layer.weight)
                        torch.nn.init.zeros_(layer.bias)
                        logging.info("Initialized final MLP layer to zeros (identity residual)")
                    else:
                        # Small random initialization for other layers
                        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        if layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
                            
                elif isinstance(layer, nn.LayerNorm):
                    torch.nn.init.constant_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)

    def apply_input_mlp_transformation(self, embeddings, token_ids):
        """Apply Input MLP to specific token positions (can be bypassed)"""

        bypass_inference = hasattr(self, 'bypass_mlp_for_inference') and self.bypass_mlp_for_inference
    
        # ✅ Use logging instead of print
        # logging.info(f"🔍 MLP TRANSFORM DEBUG:")
        # logging.info(f"  bypass_mlp: {self.bypass_mlp}")
        # logging.info(f"  bypass_mlp_for_inference: {bypass_inference}")
        # logging.info(f"  input_mlp exists: {self.input_mlp is not None}")
        # logging.info(f"  token ids: {self.label_token_ids}")
        # logging.info(f"  label_token_ids: {len(self.label_token_ids)} tokens")
        sys.stdout.flush() 
        
        # NEW: Return unchanged if MLP is bypassed
        if self.bypass_mlp or self.input_mlp is None:
            if self.batch_counter == 0:  # Log only for first batch
                logging.info("BYPASSING Input MLP transformation (bypass_mlp=True)")
            return embeddings
        
        # NEW: Check if MLP should be bypassed for inference
        if hasattr(self, 'bypass_mlp_for_inference') and self.bypass_mlp_for_inference:
            if self.batch_counter == 0:  # Log only for first batch
                logging.info("BYPASSING Input MLP transformation for inference")
                sys.stdout.flush()
            return embeddings
        
        # MLPs are ALWAYS applied during training (but may be frozen)
        if not self.label_token_ids:
            return embeddings
        
        # Handle 3D tensors
        original_shape = embeddings.shape
        if len(embeddings.shape) == 3:
            embeddings = embeddings.view(-1, embeddings.shape[-1])
            token_ids = token_ids.view(-1)
        
        # Find positions to transform
        positions_to_transform = []
        for label_token_id in self.label_token_ids:
            positions = (token_ids == label_token_id).nonzero(as_tuple=True)[0]
            positions_to_transform.extend(positions.tolist())
        
        if not positions_to_transform:
            return embeddings.view(original_shape) if len(original_shape) == 3 else embeddings
        
        positions_tensor = torch.tensor(positions_to_transform, device=embeddings.device)
        to_transform = embeddings[positions_tensor]
        
        # Apply Input MLP transformation
        try:
            # FIXED: Ensure dtype compatibility
            to_transform = to_transform.to(dtype=self.input_mlp[0].weight.dtype)
            
            # Before MLP
            
            
            mlp_output = self.input_mlp(to_transform)

            orig_norm = torch.norm(to_transform, dim=-1).mean()
            mlp_norm = torch.norm(mlp_output, dim=-1).mean()

            scale_factor = 0.001
            transformed =  (scale_factor * mlp_output) + to_transform
            
            # After MLP  
            final_norm = torch.norm(transformed, dim=-1).mean()
            
            logging.info(f"MLP Transform - Orig: {orig_norm:.4f}, MLP: {mlp_norm:.4f}, Scale: {scale_factor:.4f}, Final: {final_norm:.4f}")
            sys.stdout.flush()
            
            # CONTINUOUS EMBEDDINGS: Use MLP output directly (not quantized)
            transformed = mlp_output + to_transform # Direct MLP output for training
            
            # IMPROVED: More flexible logging conditions
            should_log = False
            
            # Log if it's the very first batch of training
            if self.batch_counter == 0:
                should_log = True
                log_reason = "first batch"
            
            # Log every 100 steps during any phase
            elif hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
                should_log = True
                log_reason = f"step {self.step_counter}"
            
            # Log if this is an MLP training phase and it's the end of epoch
            elif hasattr(self, 'is_mlp_training') and self.is_mlp_training and hasattr(self, 'is_epoch_end') and self.is_epoch_end:
                should_log = True
                log_reason = "end of MLP epoch"
            
            # Log periodically during MLP training (every 50 steps)
            elif hasattr(self, 'is_mlp_training') and self.is_mlp_training and hasattr(self, 'step_counter') and self.step_counter % 50 == 0:
                should_log = True
                log_reason = f"MLP training step {self.step_counter}"
            
            # HARD QUANTIZATION FOR LOGGING ONLY
            if should_log and len(positions_to_transform) > 0:
                # Hard quantization just for logging symbol changes
                mlp_norm = F.normalize(mlp_output, p=2, dim=-1)
                
                # FIXED: Ensure vocab embeddings have same dtype
                vocab_embeddings = self.embed_module.weight.to(dtype=mlp_norm.dtype)
                vocab_norm = F.normalize(vocab_embeddings, p=2, dim=-1)
                similarities = torch.mm(mlp_norm, vocab_norm.t())
                hard_indices = torch.argmax(similarities, dim=-1)
                
                logging.info(f"=== Input MLP Transformation Log ({log_reason}) ===")
                for i in range(min(3, len(positions_to_transform))):
                    orig_id = token_ids[positions_tensor[i]].item()
                    new_id = hard_indices[i].item()
                    similarity = similarities[i, new_id].item()
                    
                    orig_text = self.llama_tokenizer.decode([orig_id], skip_special_tokens=True)
                    new_text = self.llama_tokenizer.decode([new_id], skip_special_tokens=True)
                    
                    logging.info(f"  '{orig_text}' (id:{orig_id}) -> '{new_text}' (id:{new_id}) [sim: {similarity:.4f}]")
            
            # Update embeddings with CONTINUOUS transformed embeddings
            output_embeds = embeddings.clone()
            # FIXED: Ensure transformed embeddings match original dtype
            transformed = transformed.to(dtype=output_embeds.dtype)
            output_embeds[positions_tensor] = transformed
            
            # Reshape back if needed
            if len(original_shape) == 3:
                output_embeds = output_embeds.view(original_shape)
            
            return output_embeds
            
        except Exception as e:
            logging.error(f"Input MLP transformation failed: {e}")
            logging.error(f"Embeddings dtype: {embeddings.dtype}, MLP weight dtype: {self.input_mlp[0].weight.dtype}")
            return embeddings.view(original_shape) if len(original_shape) == 3 else embeddings

    def _ensure_dtype_compatibility(self, tensor, target_module):
        """Helper function to ensure tensor has compatible dtype with target module"""
        if target_module is None:
            return tensor
        
        # Get the target dtype from the first parameter of the module
        target_dtype = next(target_module.parameters()).dtype
        
        # Convert if necessary
        if tensor.dtype != target_dtype:
            logging.debug(f"Converting tensor from {tensor.dtype} to {target_dtype}")
            tensor = tensor.to(dtype=target_dtype)
        
        return tensor

    # def compute_mlp_loss(self, wrapped_embeds, wrapped_atts, target_tokens):
    #     """UNIFIED: Input MLP → Symbol LoRA Model → [Optional Output MLP] → Cross-entropy"""
        
    #     # NEW: If MLP is bypassed, use standard loss
    #     if self.bypass_mlp:
    #         logging.info("BYPASS_MLP=True: Using standard loss function")
    #         return self.compute_standard_loss(wrapped_embeds, wrapped_atts, target_tokens)
        
    #     logging.info(f"=== MLP Architecture Loss (Input MLP → Symbol LoRA → {'Output MLP → ' if self.use_output_mlp else ''}Cross-entropy) ===")
        
    #     try:
    #         # 1. Apply Input MLP to target embeddings (ALWAYS applied when not bypassed)
    #         target_embeds = self.embed_module(target_tokens.input_ids)
    #         transformed_target_embeds = self.apply_input_mlp_transformation(
    #             target_embeds, target_tokens.input_ids
    #         )
            
    #         # 2. Combine prompt + transformed target embeddings
    #         combined_embeds = torch.cat([wrapped_embeds, transformed_target_embeds], dim=1)
    #         combined_attention_mask = torch.cat([wrapped_atts, target_tokens.attention_mask], dim=1)
            
    #         # ✅ 3. Forward through Symbol LoRA Model (NOT base llama_model)
    #         model_to_use = self.symbol_lora_model if self.symbol_lora_model is not None else self.llama_model
            
    #         if self.use_output_mlp:
    #             # Get hidden states for output MLP
    #             with self.maybe_autocast():
    #                 outputs = model_to_use(  # ✅ Use Symbol LoRA model
    #                     inputs_embeds=combined_embeds,
    #                     attention_mask=combined_attention_mask,
    #                     output_hidden_states=True,
    #                     return_dict=True,
    #                     use_cache=False
    #                 )
                
    #             # 4. Extract target portion of output hidden states
    #             prompt_length = wrapped_embeds.size(1)
    #             target_output_hiddens = outputs.hidden_states[-1][:, prompt_length:, :]
                
    #             # 5. Apply Output MLP transformation with dtype fix
    #             target_output_hiddens = self._ensure_dtype_compatibility(target_output_hiddens, self.output_mlp)
    #             reconstructed_embeds = self.output_mlp(target_output_hiddens)
                
    #             # 6. Project to vocab space using lm_head with dtype fix
    #             reconstructed_embeds = self._ensure_dtype_compatibility(reconstructed_embeds, self.lm_head)
    #             logits = self.lm_head(reconstructed_embeds)
                
    #         else:
    #             # Direct forward without output MLP
    #             # Create labels for standard forward
    #             prompt_length = wrapped_embeds.size(1)
    #             labels = torch.full(
    #                 (combined_embeds.size(0), combined_embeds.size(1)),
    #                 fill_value=-100,
    #                 dtype=torch.long,
    #                 device=combined_embeds.device
    #             )
    #             labels[:, prompt_length:] = target_tokens.input_ids
                
    #             with self.maybe_autocast():
    #                 outputs = model_to_use(  # ✅ Use Symbol LoRA model
    #                     inputs_embeds=combined_embeds,
    #                     attention_mask=combined_attention_mask,
    #                     labels=labels,
    #                     return_dict=True,
    #                     use_cache=False
    #                 )
                
    #             logits = outputs.logits[:, prompt_length:, :]  # Extract target portion
            
    #         # 7. Cross-entropy loss with original target token IDs
    #         loss = F.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             target_tokens.input_ids.view(-1),
    #             ignore_index=-100
    #         )
            
    #         logging.info(f"Symbol LoRA Model loss: {loss:.6f}")
            
    #         return {"loss": loss, "logits": logits}
            
    #     except Exception as e:
    #         logging.error(f"Symbol LoRA loss computation failed: {e}")
    #         logging.error(f"Error details: {type(e).__name__}: {str(e)}")
    #         return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

    def compute_standard_loss(self, wrapped_embeds, wrapped_atts, target_tokens):
        """Standard loss function for LoRA training - Use Symbol LoRA Model"""
        # Get target embeddings (no MLP transformation)
        target_embeds = self.embed_module(target_tokens.input_ids)


        # if not self.bypass_mlp and self.input_mlp is not None and self.label_token_ids:
        #     target_embeds = self.apply_input_mlp_transformation(
        #         target_embeds, target_tokens.input_ids
        #     )
            # logging.info("✅ Applied MLP transformation to target embeddings")
        
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
        
        # ✅ Use Symbol LoRA Model (NOT base llama_model)
        # model_to_use = self.symbol_lora_model if self.symbol_lora_model is not None else self.llama_model
        model_to_use = self.llama_model
        
        # Standard forward with Symbol LoRA
        with self.maybe_autocast():
            outputs = model_to_use(  # ✅ Use Symbol LoRA model
                inputs_embeds=torch.cat([wrapped_embeds, target_embeds], dim=1),
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}
    
    def forward(self, samples):
        """Forward pass - ALWAYS use MLP architecture"""
        
        # Process speech and wrap prompts
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

        # ALWAYS use MLP loss (since MLPs are always in architecture)
        return self.compute_standard_loss(wrapped_embeds, wrapped_atts, target_tokens)

    def get_speech_embeddings(self, samples):
        """Process speech inputs to generate embeddings - KEPT FROM ORIGINAL"""
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
                spec_flat = samples["spectrogram"][0].flatten()
                logging.info(f"{spec_flat[:10].tolist()}")
                logging.info(f"{spec_flat[-10:].tolist()}")
                
                if samples.get("raw_wav") is not None:
                    logging.info(f"\nRaw WAV dtype: {samples['raw_wav'].dtype}")
                    logging.info("Raw WAV first 5 values:")
                    wav_flat = samples["raw_wav"][0].flatten()
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
        """Wrap speech embeddings with text prompts and examples - KEPT FROM ORIGINAL"""
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
                # if self.label_token_ids:
                #     original_shape = part_embed.shape
                #     part_embed = self.apply_input_mlp_transformation(
                #         part_embed, 
                #         tokens['input_ids'].squeeze(0)
                #     )
                #     if self.batch_counter < 3:
                #         logging.info(f"Prompt part processed: shape {original_shape} -> {part_embed.shape}")
                
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
    
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        """Encode speech inputs using SALMONN"""
        return self.salmonn.encode_speech(    
            spectrogram=spectrogram,
            raw_wav=raw_wav,
            audio_padding_mask=audio_padding_mask
        )
    
    
    def maybe_autocast(self):
        """Context manager for autocast"""
        if self.device != "cpu" and hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp.autocast(enabled=True)
        else:
            return nullcontext()

    def update_label_tokens(self, symbol_mappings):
        """Update tracked label tokens from symbol mappings"""
        new_label_tokens = []
        valid_token_ids = []
        
        for label, symbol in symbol_mappings.items():
            if symbol and symbol.strip():
                try:
                    tokens = self.llama_tokenizer.encode(symbol, add_special_tokens=False)
                    if tokens:
                        new_label_tokens.append(symbol)
                        valid_token_ids.extend(tokens)
                        logging.info(f"Symbol '{symbol}' -> tokens {tokens}")
                except Exception as e:
                    logging.error(f"Error tokenizing symbol '{symbol}': {e}")
        
        if valid_token_ids:
            self.label_tokens = new_label_tokens
            self.label_token_ids = list(set(valid_token_ids))
            logging.info(f"Updated to track {len(valid_token_ids)} tokens from {len(new_label_tokens)} symbols")

    def set_bypass_mlp(self, bypass=True):
        """Set whether to bypass MLP transformations"""
        self.bypass_mlp_for_inference = bypass
        if bypass:
            logging.info("✓ MLP transformations will be BYPASSED")
        else:
            logging.info("✓ MLP transformations will be APPLIED")

    def generate_output(self, samples):
        """Generate predictions with Symbol LoRA Model"""
        start_time = time.time()
        
        # Process speech embeddings
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        if self.batch_counter == 0:
            logging.info(f"Prompt example:\n{samples['prompt'][0]}")
            sys.stdout.flush() 
        
        # Get number of examples per sample
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        
        # Wrap embeddings with prompts (Input MLP is applied here in custom_prompt_wrap)
        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            num_examples, example_embeds, example_atts
        )
        
        # Check if MLPs should be bypassed
        # bypass_mlp = (
        #     self.bypass_mlp or  # Architecture-level bypass
        #     (hasattr(self, 'bypass_mlp_for_inference') and self.bypass_mlp_for_inference)
        # )
        
        # Extract generation parameters
        generation_kwargs = {
            "max_new_tokens": samples.get("max_new_tokens", 10),
            "num_beams": samples.get("num_beams", 1),
            "do_sample": samples.get("do_sample", False),
            "min_length": samples.get("min_length", 1),
            "top_p": samples.get("top_p", 0.9),
            "repetition_penalty": samples.get("repetition_penalty", 1.0),
            "length_penalty": samples.get("length_penalty", 1.0),
            "temperature": samples.get("temperature", 0.8),
            "pad_token_id": self.llama_tokenizer.pad_token_id,
            "eos_token_id": self.llama_tokenizer.eos_token_id,
            "return_dict_in_generate": False,
            "output_scores": False,
        }
        
        # ✅ Use Symbol LoRA Model for generation
        # model_to_use = self.symbol_lora_model if self.symbol_lora_model is not None else self.llama_model
        model_to_use =  self.llama_model
        
        # if bypass_mlp or not self.use_output_mlp or self.output_mlp is None:
        #     logging.info("Using Symbol LoRA Model - bypassing Output MLP")
        #     sys.stdout.flush() 
            
            # Standard generation with Symbol LoRA Model
        with torch.inference_mode():
            outputs = model_to_use.generate(  # ✅ Use Symbol LoRA model
                inputs_embeds=wrapped_embeds,
                attention_mask=wrapped_atts,
                **generation_kwargs
            )
        
        # else:
        #     logging.info("Using Symbol LoRA Model - applying Output MLP during generation")
            
        #     # Custom generation loop with Symbol LoRA Model + Output MLP
        #     outputs = self._generate_with_output_mlp(
        #         wrapped_embeds, 
        #         wrapped_atts, 
        #         generation_kwargs,
        #         model_to_use  # ✅ Pass Symbol LoRA model
        #     )
        
        # Decode predictions
        predictions = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        self.batch_counter += 1
        logging.info(f"Generation with Symbol LoRA took {time.time() - start_time:.2f} seconds")
        return predictions

    # def _generate_with_output_mlp(self, input_embeds, attention_mask, generation_kwargs, model_to_use=None):
    #     """Custom generation loop using Symbol LoRA Model + Output MLP"""
        
    #     # ✅ Use provided model or default to Symbol LoRA model
    #     if model_to_use is None:
    #         model_to_use = self.symbol_lora_model if self.symbol_lora_model is not None else self.llama_model
        
    #     # Safety check
    #     if self.bypass_mlp or self.output_mlp is None:
    #         logging.warning("Output MLP generation called but MLP is bypassed or None, falling back to standard")
    #         with torch.inference_mode():
    #             return model_to_use.generate(  # ✅ Use Symbol LoRA model
    #                 inputs_embeds=input_embeds,
    #                 attention_mask=attention_mask,
    #                 **generation_kwargs
    #             )
        
    #     batch_size = input_embeds.shape[0]
        
    #     # Extract parameters (SAME AS STANDARD GENERATION)
    #     max_new_tokens = generation_kwargs.get("max_new_tokens", 10)
    #     temperature = generation_kwargs.get("temperature", 0.8)
    #     top_p = generation_kwargs.get("top_p", 0.9)
    #     do_sample = generation_kwargs.get("do_sample", False)
    #     repetition_penalty = generation_kwargs.get("repetition_penalty", 1.0)
    #     num_beams = generation_kwargs.get("num_beams", 1)
    #     pad_token_id = generation_kwargs.get("pad_token_id", self.llama_tokenizer.pad_token_id)
    #     eos_token_id = generation_kwargs.get("eos_token_id", self.llama_tokenizer.eos_token_id)
        
    #     # For beam search, we'd need more complex logic, so for now only support greedy/sampling
    #     if num_beams > 1:
    #         logging.warning("Beam search not supported with Output MLP, falling back to greedy/sampling")
        
    #     # Initialize generation
    #     generated_tokens = []
    #     current_embeds = input_embeds
    #     current_attention = attention_mask
    #     finished = torch.zeros(batch_size, dtype=torch.bool, device=input_embeds.device)
        
    #     with torch.inference_mode():
    #         for step in range(max_new_tokens):
    #             try:
    #                 # ✅ Forward pass through Symbol LoRA Model
    #                 outputs = model_to_use(  # ✅ Use Symbol LoRA model
    #                     inputs_embeds=current_embeds,
    #                     attention_mask=current_attention,
    #                     output_hidden_states=True,
    #                     return_dict=True,
    #                     use_cache=False
    #                 )
                    
    #                 # Get last hidden state (batch_size, seq_len, hidden_size)
    #                 last_hidden_state = outputs.hidden_states[-1]
                    
    #                 # Get the last token's hidden state for each sample in batch
    #                 last_token_hidden = last_hidden_state[:, -1:, :]  # (batch_size, 1, hidden_size)
                    
    #                 # Apply Output MLP transformation with dtype fix
    #                 if self.output_mlp is not None:
    #                     # FIXED: Ensure dtype compatibility before Output MLP
    #                     last_token_hidden = self._ensure_dtype_compatibility(last_token_hidden, self.output_mlp)
    #                     transformed_hidden = self.output_mlp(last_token_hidden)
    #                 else:
    #                     transformed_hidden = last_token_hidden
                    
    #                 # Apply lm_head to get vocabulary logits with dtype fix
    #                 # FIXED: Ensure dtype compatibility before lm_head
    #                 transformed_hidden = self._ensure_dtype_compatibility(transformed_hidden, self.lm_head)
    #                 logits = self.lm_head(transformed_hidden)  # (batch_size, 1, vocab_size)
    #                 logits = logits.squeeze(1)  # (batch_size, vocab_size)
                    
    #                 # Apply repetition penalty (SAME AS STANDARD)
    #                 if repetition_penalty != 1.0 and step > 0:
    #                     # Simple repetition penalty on previously generated tokens
    #                     for i in range(batch_size):
    #                         if len(generated_tokens) > 0:
    #                             prev_tokens = generated_tokens[i] if len(generated_tokens) > i else []
    #                             for token_id in prev_tokens:
    #                                 if logits[i, token_id] < 0:
    #                                     logits[i, token_id] *= repetition_penalty
    #                                 else:
    #                                     logits[i, token_id] /= repetition_penalty
                    
    #                 # Apply temperature scaling (SAME AS STANDARD)
    #                 if temperature != 1.0:
    #                     logits = logits / temperature
                    
    #                 # Sample next token (SAME LOGIC AS STANDARD)
    #                 if do_sample:
    #                     # Top-p sampling (SAME AS STANDARD)
    #                     if top_p < 1.0:
    #                         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #                         cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            
    #                         # Remove tokens with cumulative probability above the threshold
    #                         sorted_indices_to_remove = cumulative_probs > top_p
    #                         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #                         sorted_indices_to_remove[..., 0] = 0
                            
    #                         indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    #                         logits = logits.masked_fill(indices_to_remove, float('-inf'))
                        
    #                     # Sample from the filtered distribution
    #                     probs = torch.softmax(logits, dim=-1)
    #                     next_token = torch.multinomial(probs, num_samples=1)
    #                 else:
    #                     # Greedy sampling (SAME AS STANDARD)
    #                     next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
    #                 # Mask finished sequences
    #                 next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_token_id)
                    
    #                 # Store generated token
    #                 if step == 0:
    #                     generated_tokens = next_token
    #                 else:
    #                     generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                    
    #                 # Check for EOS token
    #                 finished = finished | (next_token.squeeze(-1) == eos_token_id)
    #                 if torch.all(finished):
    #                     break
                    
    #                 # Prepare for next iteration
    #                 # Convert next_token to embeddings
    #                 next_token_embeds = self.embed_module(next_token)  # (batch_size, 1, hidden_size)
                    
    #                 # Apply Input MLP to next token embeddings if they are label tokens
    #                 next_token_embeds = self.apply_input_mlp_transformation(
    #                     next_token_embeds, 
    #                     next_token.squeeze(-1)
    #                 )
                    
    #                 # Concatenate to current embeddings
    #                 current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
                    
    #                 # Update attention mask
    #                 next_attention = torch.ones(batch_size, 1, dtype=current_attention.dtype, device=current_attention.device)
    #                 current_attention = torch.cat([current_attention, next_attention], dim=1)
                    
    #             except Exception as e:
    #                 logging.error(f"Error in generation step {step}: {e}")
    #                 logging.error(f"Error details: {type(e).__name__}: {str(e)}")
    #                 # Return what we have so far
    #                 if step == 0:
    #                     # If first step fails, return empty tokens
    #                     return torch.zeros((batch_size, 1), dtype=torch.long, device=input_embeds.device)
    #                 else:
    #                     break
        
    #     return generated_tokens

    def debug_salmonn_lora_targets(self):
        """Debug what SALMONN's LoRA actually targets"""
        logging.info("=" * 60)
        logging.info("SALMONN LoRA TARGET ANALYSIS")
        logging.info("=" * 60)
        
        if hasattr(self.salmonn, 'llama_model') and hasattr(self.salmonn.llama_model, 'peft_config'):
            peft_config = self.salmonn.llama_model.peft_config
            
            for adapter_name, config in peft_config.items():
                logging.info(f"SALMONN Adapter '{adapter_name}':")
                logging.info(f"  Target modules: {config.target_modules}")
                logging.info(f"  Rank: {config.r}")
                logging.info(f"  Alpha: {config.lora_alpha}")
                logging.info(f"  Dropout: {config.lora_dropout}")
        
        # Check actual parameter names
        lora_layers = set()
        for name, param in self.salmonn.llama_model.named_parameters():
            if 'lora' in name.lower():
                # Extract layer type from parameter name
                # e.g., "model.layers.0.self_attn.q_proj.lora_A.default.weight"
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                        lora_layers.add(part)
                        break
        
        logging.info(f"SALMONN LoRA actually targets: {sorted(lora_layers)}")
        logging.info("=" * 60)

# SIMPLIFIED helper functions
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
        except:
            continue
    
    return two_token_words[:num_symbols]

def create_label_mapping(original_labels, random_symbols):
    """Create simple label to symbol mapping"""
    return {orig: rand for orig, rand in zip(original_labels, random_symbols)}

# code/ICL/models/
# ├── mlp_salmonn.py                     # ✅ EXISTING (keep as is)
# ├── unified_symbol_training.py         # ✅ EXISTING (keep as is)  
# ├── unified_inference.py               # ✅ EXISTING (keep as is)
# └── symbolAdapter/                     # 🆕 NEW FOLDER
#     ├── __init__.py                    # 🆕 Empty init
#     ├── symbol_manager.py              # 🆕 START HERE
#     ├── training/
#     │   ├── __init__.py
#     │   ├── symbol_training.py         # Main training entry point
#     │   ├── mlp_trainer.py            
#     │   ├── lora_trainer.py           
#     │   ├── joint_trainer.py          
#     │   ├── validation.py             
#     │   └── schedulers.py             
#     ├── utils/
#     │   ├── __init__.py
#     │   ├── data_utils.py             
#     │   └── checkpoint_utils.py       
#     └── configs/
#         └── training_configs.py