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
    """Processor for SALMONN model."""

    def __init__(self, tokenizer, max_length: int = 128, symbol_manager=None):
        super().__init__(symbol_manager=symbol_manager)
        from transformers import WhisperFeatureExtractor

        whisper_path = get_env_path("WHISPER_MODEL_NAME", "openai/whisper-large-v2")
        logger.info("Initializing WhisperFeatureExtractor from %s", whisper_path)
        self.processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        prompt = data.get("prompt", "")
        
        # GLOBAL FIX: Apply Symbol Replacements BEFORE Tokenization
        mappings = data.get("prompt_mappings") or data.get("mappings")
        if self.symbol_manager is not None and mappings is not None:
            prompt = self.symbol_manager._apply_mapping_safe(prompt, mappings)

        completion = data.get("completion", "")
        c_mappings = data.get("completion_mappings") or data.get("mappings")
        if self.symbol_manager is not None and c_mappings is not None:
            completion = self.symbol_manager._apply_mapping_safe(completion, c_mappings)

        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        spectrogram = None
        raw_wav = None
        wav_length = 0

        if audio := data.get("audio"):
            if data.get("input_mode", "speech_only") != "text_only":
                raw_wav = torch.tensor(audio)
                spectrogram = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
                wav_length = len(raw_wav)

        examples_speech = []
        if examples_audio := data.get("examples_audio", []):
            for example_audio in examples_audio:
                example_raw_wav = torch.tensor(example_audio)
                example_spectrogram = self.processor(
                    example_audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.squeeze(0)
                examples_speech.append(
                    {
                        "raw_wav": example_raw_wav,
                        "spectrogram": example_spectrogram,
                        "wav_length": len(example_raw_wav),
                    }
                )

        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "spectrogram": spectrogram,
            "raw_wav": raw_wav,
            "wav_length": wav_length,
            "examples_speech": examples_speech,
            "num_examples": len(examples_speech),
            "completion": completion,
        }

    def format_prompt(
        self,
        template: str,
        text: str,
        examples: Optional[List[Dict]] = None,
        input_mode: str = "speech_only",
        fewshot_mode: str = "text",
        dataset_type: Optional[DatasetType] = None,
        **kwargs,
    ) -> str:
        examples_text = ""
        if examples and len(examples) > 0:
            if fewshot_mode == "speech":
                examples_text = "\n\n".join(
                    [
                        f"<Speech><Example{i}></Speech>\nOutput: {example.get('label', '')}"
                        for i, example in enumerate(examples)
                    ]
                )
            else:
                examples_text = "\n\n".join(
                    [f"Text: {example.get('text', '')}\nOutput: {example.get('label', '')}" for example in examples]
                )
            examples_text = f"\nHere are few examples to learn from:\n{examples_text}\n\n"

        input_section = f"Text: {text}" if input_mode == "text_only" else "<Speech><SpeechHere></Speech>"
        return f"{template}\n{examples_text}Now analyze this input:\n{input_section}\nOutput:"

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        batch["input_ids"] = torch.stack([item["input_ids"] for item in batch_items])
        batch["attention_mask"] = torch.stack([item["attention_mask"] for item in batch_items])

        has_valid_speech = all(item.get("spectrogram") is not None for item in batch_items)
        if has_valid_speech:
            wav_lengths = torch.tensor([item["wav_length"] for item in batch_items])
            raw_wavs = [item["raw_wav"] for item in batch_items]
            raw_wavs_padded = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
            padding_mask = torch.arange(raw_wavs_padded.size(1)).unsqueeze(0) >= wav_lengths.unsqueeze(1)
            spectrograms = torch.stack([item["spectrogram"] for item in batch_items])
            batch["wav_lengths"] = wav_lengths
            batch["raw_wav"] = raw_wavs_padded
            batch["padding_mask"] = padding_mask
            batch["spectrogram"] = spectrograms

        max_examples = max(item["num_examples"] for item in batch_items)
        if max_examples > 0:
            has_speech_examples = any(
                "examples_speech" in item
                and len(item["examples_speech"]) > 0
                and item["examples_speech"][0]["spectrogram"] is not None
                for item in batch_items
            )

            if has_speech_examples:
                max_length = max(
                    example["wav_length"]
                    for item in batch_items
                    if "examples_speech" in item
                    for example in item["examples_speech"][: item["num_examples"]]
                )

                example_specs = []
                example_wavs = []
                example_masks = []

                for item in batch_items:
                    examples = item.get("examples_speech", [])[: item["num_examples"]]
                    batch_specs = []
                    batch_wavs = []
                    batch_masks = []

                    for example in examples:
                        spec = example["spectrogram"]
                        wav = example["raw_wav"]
                        wav_length = example["wav_length"]
                        batch_specs.append(spec)
                        if wav.size(0) < max_length:
                            wav = F.pad(wav, (0, max_length - wav.size(0)), value=0)
                        batch_wavs.append(wav)
                        mask = torch.arange(max_length, device=wav.device) >= wav_length
                        batch_masks.append(mask)

                    while len(batch_specs) < max_examples:
                        pad_spec = torch.zeros_like(batch_specs[0]) if batch_specs else torch.zeros((80, 3000))
                        batch_specs.append(pad_spec)
                        batch_wavs.append(torch.zeros(max_length, device=wav.device))
                        batch_masks.append(torch.ones(max_length, device=wav.device, dtype=torch.bool))

                    example_specs.append(torch.stack(batch_specs))
                    example_wavs.append(torch.stack(batch_wavs))
                    example_masks.append(torch.stack(batch_masks))

                batch["example_spectrograms"] = torch.stack(example_specs)
                batch["example_wavs"] = torch.stack(example_wavs)
                batch["example_padding_masks"] = torch.stack(example_masks)

        batch["num_examples"] = torch.tensor([item["num_examples"] for item in batch_items])
        for key in ["prompt", "completion", "text", "dataset_type"]:
            if key in batch_items[0]:
                batch[key] = [item[key] for item in batch_items]

        return batch
