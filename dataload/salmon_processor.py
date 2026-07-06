import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from config.data_config.master_config import DatasetType
from utils.environment import get_env_path
from .model_processors import ModelProcessor

logger = logging.getLogger(__name__)


class SalmonProcessor(ModelProcessor):
    """Processor for SALMONN model. Handles audio features and modular tokenization."""

    def __init__(self, tokenizer, max_length: int = 512, symbol_manager=None):
        super().__init__(symbol_manager=symbol_manager)
        from transformers import WhisperFeatureExtractor

        whisper_path = get_env_path("WHISPER_MODEL_NAME", "openai/whisper-large-v2")
        logger.info("Initializing WhisperFeatureExtractor from %s", whisper_path)
        self.processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Returns audio features and raw text strings. Skips tokenization."""
        prompt = data.get("prompt", "")
        audio = data.get("audio")
        examples_audio = data.get("examples_audio", [])
        completion = data.get("completion", "")

        spectrogram = None
        raw_wav = None
        wav_length = 0

        if audio is not None:
            raw_wav = torch.tensor(audio)
            spectrogram = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
            wav_length = len(raw_wav)

        examples_speech = []
        if examples_audio:
            for example_audio in examples_audio:
                example_raw_wav = torch.tensor(example_audio)
                example_spectrogram = self.processor(example_audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
                examples_speech.append({
                    "raw_wav": example_raw_wav,
                    "spectrogram": example_spectrogram,
                    "wav_length": len(example_raw_wav),
                })

        return {
            "prompt": prompt,
            "completion": completion,
            "spectrogram": spectrogram,
            "raw_wav": raw_wav,
            "wav_length": wav_length,
            "examples_speech": examples_speech,
            "num_examples": len(examples_speech),
        }

    def tokenize_batch(self, prompts: List[str], completions: Optional[List[str]] = None, padding_side: str = "right") -> Dict[str, torch.Tensor]:
        """
        Unified tokenization for Salmonn (Llama-based). 
        """
        new_ids, new_masks, prompt_lens = [], [], []
        
        for i, p_text in enumerate(prompts):
            if completions is not None:
                # Training: Prompt + Completion
                full_text = f"{p_text}{completions[i]}"
                tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False, add_special_tokens=True)
                p_ids = self.tokenizer(p_text, add_special_tokens=True).input_ids
                prompt_lens.append(len(p_ids))
            else:
                # Validation: Prompt only
                tokenized = self.tokenizer(p_text, truncation=True, max_length=self.max_length, padding=False, add_special_tokens=True)
                prompt_lens.append(len(tokenized.input_ids))
                
            new_ids.append(torch.tensor(tokenized.input_ids))
            new_masks.append(torch.tensor(tokenized.attention_mask))
            
        return {
            "input_ids": pad_sequence(new_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence(new_masks, batch_first=True, padding_value=0),
            "prompt_length": torch.tensor(prompt_lens)
        }

    def format_prompt(
        self, template: str, text: str, examples: Optional[List[Dict]] = None,
        input_mode: str = "speech_only", fewshot_mode: str = "text",
        dataset_type: Optional[DatasetType] = None, **kwargs,
    ) -> str:
        examples_text = ""
        if examples and len(examples) > 0:
            if fewshot_mode == "speech":
                examples_text = "\n\n".join([f"<Speech><Example{i}></Speech>\nOutput: {example.get('label', '')}" for i, example in enumerate(examples)])
            else:
                examples_text = "\n\n".join([f"Text: {example.get('text', '')}\nOutput: {example.get('label', '')}" for example in examples])
            examples_text = f"\nHere are few examples to learn from:\n{examples_text}\n\n"

        input_section = f"Text: {text}" if input_mode == "text_only" else "<Speech><SpeechHere></Speech>"
        return f"{template}\n{examples_text}Now analyze this input:\n{input_section}\nOutput:"

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Flexible collate for raw items and tokenized items."""
        if not batch_items: return {}
        
        # If not tokenized, collect lists
        if "input_ids" not in batch_items[0]:
            batch: Dict[str, Any] = {}
            has_valid_speech = all(item.get("spectrogram") is not None for item in batch_items)
            if has_valid_speech:
                batch["wav_lengths"] = torch.tensor([item["wav_length"] for item in batch_items])
                raw_wavs = [item["raw_wav"] for item in batch_items]
                batch["raw_wav"] = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
                batch["padding_mask"] = torch.arange(batch["raw_wav"].size(1)).unsqueeze(0) >= batch["wav_lengths"].unsqueeze(1)
                batch["spectrogram"] = torch.stack([item["spectrogram"] for item in batch_items])
            skip_keys = {"spectrogram", "raw_wav", "padding_mask", "wav_length", "wav_lengths", "num_examples"}
            for key in batch_items[0].keys():
                if key in skip_keys:
                    continue
                batch[key] = [item.get(key) for item in batch_items]
            batch["num_examples"] = torch.tensor([item.get("num_examples", 0) for item in batch_items])
            return batch

        # If tokenized, pad and stack
        batch: Dict[str, Any] = {}
        batch["input_ids"] = torch.stack([item["input_ids"] for item in batch_items])
        batch["attention_mask"] = torch.stack([item["attention_mask"] for item in batch_items])

        has_valid_speech = all(item.get("spectrogram") is not None for item in batch_items)
        if has_valid_speech:
            batch["wav_lengths"] = torch.tensor([item["wav_length"] for item in batch_items])
            raw_wavs = [item["raw_wav"] for item in batch_items]
            batch["raw_wav"] = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
            batch["padding_mask"] = torch.arange(batch["raw_wav"].size(1)).unsqueeze(0) >= batch["wav_lengths"].unsqueeze(1)
            batch["spectrogram"] = torch.stack([item["spectrogram"] for item in batch_items])

        batch["num_examples"] = torch.tensor([item["num_examples"] for item in batch_items])
        for key in ["prompt", "completion", "text", "dataset_type"]:
            if key in batch_items[0]: batch[key] = [item[key] for item in batch_items]
        return batch
