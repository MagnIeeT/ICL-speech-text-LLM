from typing import Any, Dict, List, Optional

import torch

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor


class QwenProcessor(ModelProcessor):
    """Processor for Qwen2 audio model."""

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        text = data.get("prompt", "")
        audio = data.get("audio")
        examples_audio = data.get("examples_audio")
        completion = data.get("completion", "")
        input_mode = data.get("input_mode", "speech_only")

        audios = []
        if examples_audio is not None:
            audios.extend(examples_audio)
        if audio is not None:
            audios.append(audio)

        input_text = text
        if is_training:
            completion_with_eos = f"{completion}{self.processor.tokenizer.eos_token}"
            input_text = f"{text}{completion_with_eos}"

        prompt_tokens = self.processor.tokenizer(text, return_tensors="pt").input_ids
        prompt_length = prompt_tokens.size(1)

        if input_mode == "text_only":
            inputs = self.processor.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "prompt_length": prompt_length,
            }

        inputs = self.processor(
            text=input_text,
            audios=audios if len(audios) > 0 else None,
            return_tensors="pt",
            sampling_rate=16000,
        )

        if hasattr(inputs, "input_features") and inputs.input_features is not None:
            inputs.input_features = inputs.input_features.to(torch.float16)

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "input_features": inputs.input_features.squeeze(0) if hasattr(inputs, "input_features") else None,
            "feature_attention_mask": inputs.feature_attention_mask.squeeze(0)
            if hasattr(inputs, "feature_attention_mask")
            else None,
            "prompt_length": prompt_length,
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
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        keys = batch_items[0].keys()

        for key in keys:
            if key in ["input_ids", "attention_mask"]:
                batch[key] = torch.stack([item[key] for item in batch_items if key in item])
            elif key == "input_features":
                if all(item.get("input_features") is not None for item in batch_items):
                    if len(batch_items[0]["input_features"].shape) == 3:
                        batch[key] = torch.cat([item["input_features"] for item in batch_items])
                    else:
                        batch[key] = torch.stack([item["input_features"] for item in batch_items])
            elif key == "feature_attention_mask":
                if all(item.get("feature_attention_mask") is not None for item in batch_items):
                    if len(batch_items[0]["feature_attention_mask"].shape) == 2:
                        batch[key] = torch.cat([item["feature_attention_mask"] for item in batch_items])
                    else:
                        batch[key] = torch.stack([item["feature_attention_mask"] for item in batch_items])
            elif key == "prompt_length":
                batch[key] = torch.tensor([item["prompt_length"] for item in batch_items])
            elif key in ["prompt", "text", "true_label", "dataset_type", "completion"]:
                batch[key] = [item[key] for item in batch_items if key in item]

        return batch
