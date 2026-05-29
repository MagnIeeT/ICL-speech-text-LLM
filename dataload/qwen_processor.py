from typing import Any, Dict, List, Optional

import torch

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor


class QwenProcessor(ModelProcessor):
    """Processor for Qwen2 audio model. Handles audio features and modular tokenization."""

    def __init__(self, processor, tokenizer=None, max_length: int = 512, symbol_manager=None):
        super().__init__(symbol_manager=symbol_manager)
        self.processor = processor
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
        if examples_audio: audios.extend(examples_audio)
        if audio: audios.append(audio)
        if not audios: return None
        
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
                # 1. Training Path
                c_text = completions[i]
                full_text = f"{p_text}{c_text}{self.tokenizer.eos_token}"
                tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding=False, add_special_tokens=True)
                
                # Calculate prompt length for loss masking
                p_ids = self.tokenizer(p_text, add_special_tokens=True).input_ids
                prompt_lens.append(len(p_ids))
            else:
                # 2. Inference/Validation Path (Prompt only)
                tokenized = self.tokenizer(p_text, truncation=True, max_length=self.max_length, padding=False, add_special_tokens=True)
                prompt_lens.append(len(tokenized.input_ids))
                
            new_ids.append(torch.tensor(tokenized.input_ids))
            new_masks.append(torch.tensor(tokenized.attention_mask))
            
        from torch.nn.utils.rnn import pad_sequence
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
        conversation = [{"role": "system", "content": template}]
        user_content = []
        if examples and len(examples) > 0:
            user_content.append({"type": "text", "text": "Here are few examples to learn from:\n"})
            for example in examples:
                user_content.extend([{"type": "text", "text": f"Text: {example.get('text', '')}\nLabel: {example.get('label', '')}\n"}])
        user_content.append({"type": "text", "text": "\nNow analyze this input:\n"})
        user_content.append({"type": "audio", "audio_url": "dummy_url"})
        conversation.append({"role": "user", "content": user_content})
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        for key in batch_items[0].keys():
            if key == "input_features":
                if all(item.get("input_features") is not None for item in batch_items):
                    batch[key] = torch.stack([item["input_features"] for item in batch_items])
            elif key in ["prompt", "text", "completion", "dataset_type"]:
                batch[key] = [item[key] for item in batch_items if key in item]
        return batch
