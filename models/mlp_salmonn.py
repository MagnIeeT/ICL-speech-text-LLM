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
        beats_path="/home/leapers/weights/SALMONN/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        device=None,
        low_resource=False,
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7.0
        self.use_fp16 = False
        # SALMONN config
        salmonn_config = {
            "llama_path": llama_path,
            "whisper_path": whisper_path,
            "beats_path": beats_path,
            "lora": True,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "low_resource": low_resource,
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
            "ckpt": "/home/leapers/weights/SALMONN/salmonn_v1.pth"
        }
        logging.info("=" * 80)
        logging.info("🔧 INITIALIZING MLP-SALMONN MODEL")
        logging.info("=" * 80)
        logging.info(f"Device: {self.device}")
        logging.info(f"LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

        logging.info("🔄 Loading base SALMONN model...")
        
        sys.stdout.flush()  # ✅ Force flush stdout
        logging.getLogger().handlers[0].flush() 
        # Initialize SALMONN
        logging.info("Loading base SALMONN model...")
        
        self.salmonn = SALMONN.from_config(salmonn_config)
        self.salmonn.to(self.device) 

        # ✅ FIX: Cast internal SALMONN components to match LLaMA's precision
        # We avoid calling .to() on self.salmonn because it contains the LLaMA meta-tensors
        # target_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # for name, module in self.salmonn.named_children():
        #     if name != 'llama_model':  # Skip LLaMA as it's already handled by device_map
        #         module.to(device=self.device, dtype=target_dtype)
        #         logging.info(f"Moved {name} to {self.device} with dtype {target_dtype}")

        logging.info("Base SALMONN model loaded successfully")
        sys.stdout.flush() 

        self.batch_counter = 0
        self.speech_placeholder = "<SpeechHere>"

        self.lora = lora
        
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

    def forward(self, samples):
        """
        Forward pass for training.
        """
        start_time = time.time()
        
        # Process speech embeddings
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        if self.batch_counter == 0:
            logging.info("\n=== Embeddings Status ===")
            logging.info(f"Main speech embeddings: {'Present' if speech_embeds is not None else 'None'}")
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
            logging.info(f"Output:\n{samples['completion'][0]}")
        
        # Get number of examples per sample
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        
        # Wrap embeddings with prompts
        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            num_examples, example_embeds, example_atts
        )
        
        if self.batch_counter == 0:
            logging.info(f"Wrapped embeddings shape: {wrapped_embeds.shape}")
            logging.info(f"Wrapped attention mask shape: {wrapped_atts.shape}")
        
        gen_start_time = time.time()
        
        # Process target text
        target_tokens = self.llama_tokenizer(
            samples["completion"],
            padding='longest',
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        ).to(wrapped_embeds.device)
        
        target_embeds = self.llama_model.model.embed_tokens(target_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(target_tokens.input_ids)
        prompt_length = wrapped_embeds.size(1)
        
        # Create labels tensor
        labels = torch.full(
            [wrapped_atts.shape[0], wrapped_atts.shape[1] + target_tokens.input_ids.size(1)],
            fill_value=-100,
            dtype=torch.long,
            device=wrapped_embeds.device
        )
        labels[:, prompt_length:] = target_tokens.input_ids
        
        # Mask padding tokens
        attention_mask = torch.cat([wrapped_atts, target_tokens.attention_mask], dim=1)
        labels[attention_mask == 0] = -100
        
        # Forward through LLaMA
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=torch.cat([wrapped_embeds, target_embeds], dim=1),
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        logging.info(f"Forward pass took {time.time() - gen_start_time:.2f} seconds")
        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}

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
                    non_padded_count = (padding_mask == False).sum().item()
                    padding_percent = padding_mask.float().mean().item() * 100
                    
                    logging.info(f"Non-padded length (real data): {non_padded_count}")
                    logging.info(f"Padding percentage: {padding_percent:.2f}%")

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
        """Context manager for mixed precision training"""
        return torch.cuda.amp.autocast() if self.use_fp16 else nullcontext()

    def generate_output(self, samples):
        """
        Generate predictions for speech or text input.
        """
        start_time = time.time()
        
        # Process speech embeddings
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
        
        # Wrap embeddings with prompts
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
        
        # Generate with LLaMA
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

