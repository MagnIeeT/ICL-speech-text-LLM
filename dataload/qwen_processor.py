from typing import Any, Dict, List, Optional

import torch
import numpy as np

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor


def _pad_sequence(tensors: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(t.shape[-1] for t in tensors)
    out = []
    for t in tensors:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        pad_size = max_len - t.shape[-1]
        if pad_size > 0:
            t = torch.nn.functional.pad(t, (0, pad_size), value=pad_value)
        out.append(t)
    return torch.stack(out)


class QwenProcessor(ModelProcessor):
    """Processor for Qwen2 audio model. Handles audio features and modular tokenization."""

    def __init__(self, processor, tokenizer=None, max_length: int = 512, symbol_manager=None):
        super().__init__(symbol_manager=symbol_manager)
        self.processor = processor
        # FIX: Ensure the internal processor uses the same updated tokenizer object
        if tokenizer is not None:
            self.processor.tokenizer = tokenizer
        self.tokenizer = self.processor.tokenizer
        self.max_length = max_length

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Returns audio features and raw text strings. Skips tokenization."""
        return {
            "prompt": data.get("prompt", ""),
            "completion": data.get("completion", ""),
            "input_features": self._process_audio(data.get("audio"), data.get("examples_audio")),
        }

    def _process_audio(self, audio, examples_audio):
        audios = []
        if examples_audio:
            if isinstance(examples_audio, (list, tuple)):
                audios.extend(examples_audio)
            else:
                audios.append(examples_audio)
        if audio:
            audios.append(audio)
            
        if not audios:
            return None
        
        # We use the processor just to get the audio features
        inputs = self.processor(text=" ", audios=audios, return_tensors="pt", sampling_rate=16000)
        return inputs.input_features.squeeze(0).to(torch.float16)

    def tokenize_batch(self, prompts: List[str], completions: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Unified tokenization for Qwen. 
        - If completions provided: [Prompt] + [Completion] + [EOS] (Training)
        - If completions None: [Prompt] only (Inference/Validation)
        """
        new_ids, new_masks, prompt_lens = [], [], []
        
        for i, p_text in enumerate(prompts):
            if completions is not None:
                # Training Path: Full sequence withEOS
                c_text = completions[i]
                full_text = f"{p_text}{c_text}{self.tokenizer.eos_token}"
                tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False, add_special_tokens=True)
                
                # Calculate prompt length for loss masking (everything before completion)
                p_ids = self.tokenizer(p_text, add_special_tokens=True).input_ids
                prompt_lens.append(len(p_ids))
            else:
                # Inference/Validation Path (Prompt only)
                tokenized = self.tokenizer(p_text, truncation=True, max_length=self.max_length, padding=False, add_special_tokens=True)
                prompt_lens.append(len(tokenized.input_ids))
                
            new_ids.append(torch.tensor(tokenized.input_ids))
            new_masks.append(torch.tensor(tokenized.attention_mask))
            
        return {
            "input_ids": _pad_sequence(new_ids, pad_value=self.tokenizer.pad_token_id),
            "attention_mask": _pad_sequence(new_masks, pad_value=0),
            "prompt_length": torch.tensor(prompt_lens)
        }

    def format_prompt(
        self, template: str, text: str, examples: Optional[List[Dict]] = None,
        input_mode: str = "speech_only", fewshot_mode: str = "text",
        dataset_type: Optional[DatasetType] = None, **kwargs,
    ) -> str:
        conversation = [{"role": "system", "content": template}]
        user_content = []

        if examples and len(examples) > 0:
            user_content.append({"type": "text", "text": "Here are few examples to learn from:\n"})
            for example in examples:
                example_text = example.get("text", "")
                example_label = example.get("label", "")
                if fewshot_mode == "speech":
                    user_content.extend([
                        {"type": "audio", "audio_url": "dummy_url"},
                        {"type": "text", "text": f"Label: {example_label}\n"}
                    ])
                else:
                    user_content.extend([
                        {"type": "text", "text": f"Text: {example_text}\n"},
                        {"type": "text", "text": f"Label: {example_label}\n"}
                    ])

        user_content.append({"type": "text", "text": "\nNow analyze this input:\n"})
        if input_mode == "text_only":
            user_content.append({"type": "text", "text": text})
        else:
            user_content.append({"type": "audio", "audio_url": "dummy_url"})

        conversation.append({"role": "user", "content": user_content})
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of items.
        Handles both raw items (no input_ids) and tokenized items.
        """
        if not batch_items:
            return {}

        # If not tokenized yet, just collect into lists
        if "input_ids" not in batch_items[0]:
            batch: Dict[str, Any] = {}
            for key in batch_items[0].keys():
                batch[key] = [item.get(key) for item in batch_items]
            return batch

        # If tokenized, pad and stack
        batch: Dict[str, Any] = {}
        passthrough = {"prompt", "text", "completion", "dataset_type"}
        
        for key in batch_items[0].keys():
            values = [item[key] for item in batch_items if key in item]
            if not values: continue

            if key in passthrough:
                batch[key] = values
            elif key == "input_ids":
                batch[key] = _pad_sequence(values, pad_value=self.tokenizer.pad_token_id)
            elif key == "attention_mask":
                batch[key] = _pad_sequence(values, pad_value=0)
            elif key == "input_features":
                batch[key] = torch.stack(values) if values[0].dim() == 3 else torch.cat(values, dim=0)
            elif key == "prompt_length":
                batch[key] = torch.tensor([int(v) for v in values])
            else:
                try: batch[key] = torch.stack(values)
                except: batch[key] = values
        return batch
