import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import nullcontext
from utils.environment import get_env_path

# Import PEFT for LoRA
from SALMONN.models.salmonn import SALMONN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout, force=True)

class CustomSalmonn(nn.Module):
    """CLEAN SALMONN with LoRA Symbol Adapter Architecture and D-SPO Support"""
    
    def __init__(
        self,
        llama_path=None,
        whisper_path=None,
        beats_path=None,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        device=None,
        low_resource=False,
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        llama_path = llama_path or get_env_path("LLAMA_MODEL_NAME")
        whisper_path = whisper_path or get_env_path("WHISPER_MODEL_NAME")
        beats_path = beats_path or get_env_path("BEATS_CKPT_PATH")
        salmonn_ckpt = get_env_path("SALMONN_CKPT_PATH")
        self.use_fp16 = False
        
        salmonn_config = {
            "llama_path": llama_path,
            "whisper_path": whisper_path,
            "beats_path": get_env_path("BEATS_CKPT_PATH"),
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
            "ckpt": get_env_path("SALMONN_CKPT_PATH")
        }
        logging.info("=" * 80)
        logging.info("🔧 INITIALIZING CUSTOM-SALMONN MODEL")
        logging.info("=" * 80)
        logging.info(f"Device: {self.device}")
        logging.info(f"LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")

        logging.info("🔄 Loading base SALMONN model...")
        sys.stdout.flush()
        
        self.salmonn = SALMONN.from_config(salmonn_config)
        self.salmonn.to(self.device) 

        logging.info("Base SALMONN model loaded successfully")
        sys.stdout.flush() 

        self.batch_counter = 0
        self.speech_placeholder = "<SpeechHere>"
        self.lora = lora
        
        self.llama_model = self.salmonn.llama_model
        self.llama_tokenizer = self.salmonn.llama_tokenizer
        
        if hasattr(self.llama_model, 'model'):
            if hasattr(self.llama_model.model, 'model'):
                self.embed_module = self.llama_model.model.model.embed_tokens
            else:
                self.embed_module = self.llama_model.model.embed_tokens
        else:
            self.embed_module = self.llama_model.embed_tokens

    def forward(self, samples, router=None, dspo_module=None):
        """
        Forward pass for training. Supports D-SPO injection.
        """
        start_time = time.time()
        
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        if self.batch_counter == 0:
            logging.info("\n=== Embeddings Status ===")
            logging.info(f"Main speech embeddings: {'Present' if speech_embeds is not None else 'None'}")
            if speech_embeds is not None:
                if isinstance(speech_embeds, tuple):
                    logging.info("SQA Speech data detected and processed")
                else:
                    logging.info("Speech data detected and processed")
            
            logging.info(f"Prompt example:\n{samples['prompt'][0]}")
            logging.info(f"Output:\n{samples['completion'][0]}")
            
            if router is not None:
                logging.info("D-SPO: Router detected in forward pass. Enabling differentiable injection.")

        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long)),
            example_embeds, example_atts,
            router=router, dspo_module=dspo_module
        )
        
        if self.batch_counter == 0:
            logging.info(f"Wrapped embeddings shape: {wrapped_embeds.shape}")
            logging.info(f"Wrapped attention mask shape: {wrapped_atts.shape}")
        
        gen_start_time = time.time()
        
        target_tokens = self.llama_tokenizer(
            samples["completion"],
            padding='longest',
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        ).to(wrapped_embeds.device)
        
        target_embeds = self.embed_module(target_tokens.input_ids)
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
        
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=torch.cat([wrapped_embeds, target_embeds], dim=1),
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        logging.debug(f"Forward pass took {time.time() - gen_start_time:.2f} seconds")
        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}

    def get_speech_embeddings(self, samples):
        """Process speech inputs to generate embeddings"""
        start_time = time.time()
        is_sqa = ("question_spectrogram" in samples and "document_spectrogram" in samples)

        if is_sqa:
            has_main_speech = (samples["question_spectrogram"] is not None and samples["document_spectrogram"] is not None)
            has_examples = ("example_question_spectrograms" in samples and "example_document_spectrograms" in samples)
        else:
            has_main_speech = "spectrogram" in samples and samples["spectrogram"] is not None
            has_examples = "example_spectrograms" in samples and samples["example_spectrograms"] is not None

        if self.batch_counter == 0 and not is_sqa and has_main_speech:
            logging.info("=== Input Data Debug ===")
            spec_flat = samples["spectrogram"][0].flatten()
            logging.info(f"Spectrogram stats - min: {spec_flat.min():.3f}, max: {spec_flat.max():.3f}")

        def to_device_and_dtype(tensor):
            if tensor is None: return None
            tensor = tensor.to(device=self.device)
            if hasattr(self, 'use_fp16') and self.use_fp16 and tensor.dtype == torch.float32:
                tensor = tensor.to(dtype=torch.float16)
            return tensor

        speech_embeds = speech_atts = None
        if has_main_speech:
            spectrograms = to_device_and_dtype(samples["spectrogram"])
            padding_mask = to_device_and_dtype(samples.get("padding_mask"))
            raw_wav = to_device_and_dtype(samples.get("raw_wav"))
            speech_embeds, speech_atts = self.encode_speech(spectrograms, raw_wav, padding_mask)
    
        example_embeds = example_atts = None
        if has_examples:
            example_specs = to_device_and_dtype(samples["example_spectrograms"])
            example_padding = to_device_and_dtype(samples.get("example_padding_masks"))
            example_raw = to_device_and_dtype(samples.get("example_raw_wavs"))
            
            batch_example_embeds, batch_example_atts = [], []
            for b in range(len(example_specs)):
                example_count = len(example_specs[b])
                batch_embeds, batch_atts = [], []
                for i in range(example_count):
                    spec = example_specs[b][i]
                    pad = example_padding[b][i] if example_padding is not None else None
                    raw = example_raw[b][i] if example_raw is not None else None
                    embed, att = self.encode_speech(spec.unsqueeze(0), raw, pad)
                    batch_embeds.append(embed.squeeze(0))
                    batch_atts.append(att.squeeze(0))
                batch_example_embeds.append(batch_embeds)
                batch_example_atts.append(batch_atts)
            example_embeds = batch_example_embeds
            example_atts = batch_example_atts
        
        return speech_embeds, speech_atts, example_embeds, example_atts
        
    def custom_prompt_wrap(self, embeds, atts, prompts, num_examples=None, example_embeds=None, example_atts=None, router=None, dspo_module=None):
        """Wrap speech embeddings with text prompts and examples - supports D-SPO."""
        device = next(self.parameters()).device
        batch_size = len(prompts)
        max_examples = num_examples.max().item() if num_examples is not None else 0
        batch_embeds, batch_atts = [], []
        
        for b in range(batch_size):
            parts = []
            suffix = prompts[b]
            if max_examples > 0 and example_embeds is not None:
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
                parts.append(suffix)
                suffix = ""
            parts.append(suffix)
            
            tokenized_parts = [
                self.llama_tokenizer(part, padding="longest", return_tensors="pt", add_special_tokens=False) 
                for part in parts
            ]
            tokenized_parts = [
                {k: v.to(device=device, dtype=torch.long if k == 'input_ids' else v.dtype) 
                 for k, v in tokens.items()}
                for tokens in tokenized_parts
            ]
            
            part_embeds = []
            for tokens in tokenized_parts:
                input_ids = tokens['input_ids']
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                # Get base embeddings
                embeds_base = self.embed_module(input_ids)
                
                # D-SPO Injection if available
                if router is not None and dspo_module is not None:
                    embeds_base = dspo_module.inject_differentiable_symbols(
                        input_ids=input_ids,
                        input_embeds=embeds_base,
                        router=router,
                        embedding_layer=self.embed_module
                    )
                
                if self.batch_counter < 3:
                    part_text = self.llama_tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                    logging.info(f"Processing prompt part: '{part_text[:50]}...' ({input_ids.size(1)} tokens)")
                    if router is not None:
                        logging.debug("D-SPO: Injected soft embeddings for prompt part.")

                part_embeds.append(embeds_base.squeeze(0))
                
            part_atts = [tokens['attention_mask'].squeeze(0) for tokens in tokenized_parts]
            
            combined_embeds, combined_atts = [], []
            for i in range(len(part_embeds) - 2):
                combined_embeds.append(part_embeds[i])
                combined_atts.append(part_atts[i])
                if i < max_examples and example_embeds is not None:
                    if b < len(example_embeds) and i < len(example_embeds[b]):
                        combined_embeds.append(example_embeds[b][i])
                        combined_atts.append(example_atts[b][i])

            if embeds is not None:
                combined_embeds.extend([part_embeds[-2], embeds[b], part_embeds[-1]])
                combined_atts.extend([part_atts[-2], atts[b], part_atts[-1]])
            else:
                if len(part_embeds) >= 2:
                    combined_embeds.extend([part_embeds[-2], part_embeds[-1]])
                    combined_atts.extend([part_atts[-2], part_atts[-1]])
                else:
                    combined_embeds.append(part_embeds[-1])
                    combined_atts.append(part_atts[-1])
            
            batch_embeds.append(torch.cat(combined_embeds, dim=0))
            batch_atts.append(torch.cat(combined_atts, dim=0))
        
        return torch.stack(batch_embeds, dim=0), torch.stack(batch_atts, dim=0)
    
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        return self.salmonn.encode_speech(spectrogram=spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)
    
    def maybe_autocast(self):
        return torch.cuda.amp.autocast() if self.use_fp16 else nullcontext()

    def generate_output(self, samples, max_new_tokens: int = 10):
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(speech_embeds, speech_atts, samples["prompt"], num_examples, example_embeds, example_atts)
        
        with torch.inference_mode():
            outputs = self.llama_model.generate(
                inputs_embeds=wrapped_embeds, attention_mask=wrapped_atts,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
            )
        return self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
