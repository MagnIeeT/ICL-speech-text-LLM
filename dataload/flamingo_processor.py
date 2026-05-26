"""
FlamingoProcessor
=================
Processor for nvidia/audio-flamingo-3-hf using AutoProcessor.apply_chat_template.

Design:
- Dataset returns RAW fields.
- Symbol replacement happens on RAW batch.
- Tokenization happens in orchestrator/validator by calling process_inputs per item.
- collate_batch:
    RAW items  -> dict-of-lists
    tokenized  -> pad/stack tensors
"""

from typing import Any, Dict, List, Optional
import copy

import numpy as np
import torch

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor


class FlamingoProcessor(ModelProcessor):
    """Processor for nvidia/audio-flamingo-3-hf."""

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor  
        self.max_length = max_length

    def format_prompt(
        self,
        template: str,
        text: str,
        examples: Optional[List[Dict]] = None,
        input_mode: str = "speech_only",
        fewshot_mode: str = "text",
        dataset_type: Optional[DatasetType] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        conversation = [{"role": "system", "content": template}]
        user_content: List[Dict] = []

        if examples:
            user_content.append({"type": "text", "text": "Here are a few examples:\n"})
            for example in examples:
                example_text = example.get("text", "")
                example_label = example.get("label", "")
                if fewshot_mode == "speech" and example.get("audio") is not None:
                    user_content.extend(
                        [
                            {"type": "audio", "audio": _ensure_numpy(example["audio"])},
                            {"type": "text", "text": f"Label: {example_label}\n"},
                        ]
                    )
                else:
                    user_content.extend(
                        [
                            {"type": "text", "text": f"Text: {example_text}\n"},
                            {"type": "text", "text": f"Label: {example_label}\n"},
                        ]
                    )

        user_content.append({"type": "text", "text": "\nNow analyse this input:\n"})

        if input_mode == "text_only":
            user_content.append({"type": "text", "text": text})
        else:
            user_content.append({"type": "text", "text": "__AUDIO_PLACEHOLDER__"})

        conversation.append({"role": "user", "content": user_content})
        return {"conversation": conversation, "input_mode": input_mode}

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False) -> Dict[str, Any]:
        prompt_obj = data.get("prompt")
        if not isinstance(prompt_obj, dict) or "conversation" not in prompt_obj:
            raise ValueError(
                "FlamingoProcessor expects data['prompt'] to be a dict returned by format_prompt() "
                "with key 'conversation'."
            )

        conversation = copy.deepcopy(prompt_obj["conversation"])

        audio = data.get("audio")
        completion: str = data.get("completion", "")
        input_mode: str = data.get("input_mode", prompt_obj.get("input_mode", "speech_only"))

        for msg in conversation:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            new_content: List[Dict] = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and "__AUDIO_PLACEHOLDER__" in part.get("text", "")
                ):
                    cleaned = part.get("text", "").replace("__AUDIO_PLACEHOLDER__", "").strip()
                    if cleaned:
                        new_content.append({"type": "text", "text": cleaned})
                    if input_mode != "text_only" and audio is not None:
                        new_content.append({"type": "audio", "audio": _ensure_numpy(audio)})
                else:
                    new_content.append(part)
            msg["content"] = new_content

        prompt_only_conv = [c for c in conversation if c.get("role") != "assistant"]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_only_conv,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        prompt_ids = prompt_inputs["input_ids"]
        if prompt_ids.dim() > 1 and prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.squeeze(0)

        prompt_mask = prompt_inputs["attention_mask"]
        if prompt_mask.dim() > 1 and prompt_mask.shape[0] == 1:
            prompt_mask = prompt_mask.squeeze(0)

        if is_training and completion:
            eos = self.processor.tokenizer.eos_token or ""
            completion_text = f"{completion}{eos}"

            completion_inputs = self.processor.tokenizer(
                completion_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            completion_ids = completion_inputs["input_ids"].squeeze(0)
            completion_mask = completion_inputs["attention_mask"].squeeze(0)

            full_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
            full_mask = torch.cat([prompt_mask, completion_mask], dim=-1)

            full_inputs = dict(prompt_inputs)
            full_inputs["input_ids"] = full_ids
            full_inputs["attention_mask"] = full_mask
            
            prompt_length = int(prompt_ids.shape[-1])
            
        else:
            full_inputs = dict(prompt_inputs)
            full_inputs["input_ids"] = prompt_ids
            full_inputs["attention_mask"] = prompt_mask
            
            prompt_length = int(prompt_ids.shape[-1])

        seq_len = int(full_inputs["input_ids"].shape[-1])
        
        if is_training and prompt_length >= seq_len:
            raise ValueError(
                f"prompt_length ({prompt_length}) >= seq_len ({seq_len}). "
                "This usually means assistant completion wasn't included in tokenization."
            )

        result: Dict[str, Any] = {"prompt_length": prompt_length}
        for key, value in full_inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dim() > 1 and value.shape[0] == 1:
                    value = value.squeeze(0)
                result[key] = value
            else:
                result[key] = value
        return result

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch_items:
            return {}

        if "input_ids" not in batch_items[0]:
            batch: Dict[str, Any] = {}
            keys = set().union(*[item.keys() for item in batch_items])
            for key in keys:
                batch[key] = [item.get(key) for item in batch_items]
            return batch

        batch: Dict[str, Any] = {}
        keys = set().union(*[item.keys() for item in batch_items])

        passthrough = {
            "prompt", "text", "true_label", "dataset_type", 
            "completion", "audio", "examples_audio", 
            "input_mode", "fewshot_mode", "is_training",
        }

        for key in keys:
            values = [item[key] for item in batch_items if key in item]
            if not values:
                continue

            if key in passthrough:
                batch[key] = values
                continue

            if key == "prompt_length":
                batch[key] = torch.tensor([int(item.get("prompt_length", 0)) for item in batch_items])
                continue

            if not isinstance(values[0], torch.Tensor):
                batch[key] = values
                continue

            if key == "input_ids":
                batch[key] = _pad_sequence(values, pad_value=self.processor.tokenizer.pad_token_id or 0)
            elif key == "attention_mask":
                batch[key] = _pad_sequence(values, pad_value=0)
            else:
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values

        return batch


def _ensure_numpy(audio) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    if isinstance(audio, torch.Tensor):
        return audio.float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


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