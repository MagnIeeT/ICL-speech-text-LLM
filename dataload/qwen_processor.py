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
    """Processor for Qwen2 audio model."""

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
    ) -> str:
        """Returns a plain string — Qwen renders the chat template at format time."""
        conversation = [{"role": "system", "content": template}]
        user_content = []

        if examples and len(examples) > 0:
            user_content.append({"type": "text", "text": "Here are few examples to learn from:\n"})
            for example in examples:
                example_text = example.get("text", "")
                example_label = example.get("label", "")
                if fewshot_mode == "speech":
                    user_content.extend(
                        [
                            {"type": "audio", "audio_url": "dummy_url"},
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

        user_content.append({"type": "text", "text": "\nNow analyze this input:\n"})
        if input_mode == "text_only":
            user_content.append({"type": "text", "text": text})
        else:
            user_content.append({"type": "audio", "audio_url": "dummy_url"})

        conversation.append({"role": "user", "content": user_content})
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False) -> Dict[str, Any]:
        """
        Tokenize a single item.
        data["prompt"] is a plain string for Qwen.
        If is_training=True, appends completion tokens after prompt tokens.
        """
        text = data.get("prompt", "")
        audio = data.get("audio")
        examples_audio = data.get("examples_audio")
        completion = data.get("completion", "")
        input_mode = data.get("input_mode", "speech_only")

        # Build audio list
        audios = []
        if examples_audio is not None:
            if isinstance(examples_audio, (list, tuple)):
                audios.extend(examples_audio)
            else:
                audios.append(examples_audio)
        if audio is not None:
            audios.append(audio)

        # Tokenize prompt only first to get prompt_length
        if input_mode == "text_only":
            prompt_inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
        else:
            prompt_inputs = self.processor(
                text=text,
                audio=audios if len(audios) > 0 else None,
                return_tensors="pt",
                sampling_rate=16000,
            )

        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        prompt_mask = prompt_inputs["attention_mask"].squeeze(0)
        prompt_length = int(prompt_ids.shape[-1])

        if is_training and completion:
            # Append completion tokens
            eos = self.processor.tokenizer.eos_token or ""
            completion_text = f"{completion}{eos}"
            completion_inputs = self.processor.tokenizer(
                completion_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            completion_ids = completion_inputs["input_ids"].squeeze(0)
            completion_mask = completion_inputs["attention_mask"].squeeze(0)

            full_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
            full_mask = torch.cat([prompt_mask, completion_mask], dim=-1)
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        result = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "prompt_length": prompt_length,
        }

        # Add audio features if present
        if input_mode != "text_only":
            if hasattr(prompt_inputs, "input_features") and prompt_inputs.input_features is not None:
                result["input_features"] = prompt_inputs.input_features.squeeze(0).to(torch.float16)
            if hasattr(prompt_inputs, "feature_attention_mask") and prompt_inputs.feature_attention_mask is not None:
                result["feature_attention_mask"] = prompt_inputs.feature_attention_mask.squeeze(0)

        return result

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of items.
        Handles two cases:
        - Raw items (no input_ids yet) — just collects into lists
        - Tokenized items (has input_ids) — pads and stacks tensors
        """
        if not batch_items:
            return {}

        # RAW path — items not yet tokenized
        if "input_ids" not in batch_items[0]:
            batch: Dict[str, Any] = {}
            keys = set().union(*[item.keys() for item in batch_items])
            for key in keys:
                batch[key] = [item.get(key) for item in batch_items]
            return batch

        # TOKENIZED path — pad and stack
        batch: Dict[str, Any] = {}

        passthrough = {
            "prompt", "text", "true_label", "dataset_type",
            "completion", "audio", "examples_audio",
            "input_mode", "fewshot_mode", "is_training",
        }

        keys = set().union(*[item.keys() for item in batch_items])

        for key in keys:
            values = [item[key] for item in batch_items if key in item]
            if not values:
                continue

            if key in passthrough:
                batch[key] = values
                continue

            if key == "prompt_length":
                batch[key] = torch.tensor(
                    [int(item.get("prompt_length", 0)) for item in batch_items]
                )
                continue

            if not isinstance(values[0], torch.Tensor):
                batch[key] = values
                continue

            if key == "input_ids":
                batch[key] = _pad_sequence(
                    values,
                    pad_value=self.processor.tokenizer.pad_token_id or 0,
                )
            elif key == "attention_mask":
                batch[key] = _pad_sequence(values, pad_value=0)
            elif key == "input_features":
                # input_features can be 2D or 3D depending on number of audios
                try:
                    if values[0].dim() == 3:
                        batch[key] = torch.cat(values, dim=0)
                    else:
                        batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
            elif key == "feature_attention_mask":
                try:
                    if values[0].dim() == 2:
                        batch[key] = torch.cat(values, dim=0)
                    else:
                        batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values
            else:
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError:
                    batch[key] = values

        return batch