"""
FlamingoProcessor
=================
Processor for nvidia/audio-flamingo-3-hf using AutoProcessor.apply_chat_template.

Design (pipeline contract)
--------------------------
- Dataset returns RAW fields (prompt object + completion + audio).
- Symbol replacement happens on RAW batch (prompt/completion rewritten before tokenization).
- Tokenization happens in orchestrator/validator by calling tokenize_batch() AFTER symbol replacement.
- collate_batch:
    RAW items  -> dict-of-lists
    tokenized  -> pad/stack input_ids + attention_mask; stack other tensors only if shapes match.

Key difference from Qwen
------------------------
Flamingo's apply_chat_template injects exactly N <sound> tokens based on audio duration.
If audio is missing at tokenization time, processor emits wrong number of audio tokens → crash:
    "Audio features and audio tokens do not match, tokens: 1, features: N"

Fix:
- audio is embedded in the prompt dict by format_prompt() at the TOP LEVEL: prompt["audio"]
- tokenize_batch() accepts original_audios fallback for recovery if prompt lost audio key.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor

logger = logging.getLogger(__name__)


def _ensure_numpy(audio) -> np.ndarray:
    if audio is None:
        return None
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    if isinstance(audio, torch.Tensor):
        return audio.float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _pad_1d(
    tensors: List[torch.Tensor],
    pad_value: int = 0,
    padding_side: str = "right",
) -> torch.Tensor:
    """Pad 1D token sequences (input_ids / attention_mask) along last dim."""
    max_len = max(int(t.shape[-1]) for t in tensors)
    out = []
    for t in tensors:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        pad_size = max_len - int(t.shape[-1])
        if pad_size > 0:
            pad_args = (pad_size, 0) if padding_side == "left" else (0, pad_size)
            t = torch.nn.functional.pad(t, pad_args, value=pad_value)
        out.append(t)
    return torch.stack(out)


class FlamingoProcessor(ModelProcessor):
    """Processor for nvidia/audio-flamingo-3-hf."""

    def __init__(self, processor, max_length: int = 512, symbol_manager=None):
        super().__init__(symbol_manager=symbol_manager)
        self.processor = processor
        self.max_length = max_length

    # ------------------------------------------------------------------
    # format_prompt
    # ------------------------------------------------------------------
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
        user_content: List[Dict[str, Any]] = []

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
            # Insert audio at tokenize time to guarantee apply_chat_template sees the audio array.
            user_content.append({"type": "text", "text": "__AUDIO_PLACEHOLDER__"})

        conversation.append({"role": "user", "content": user_content})

        prompt_dict: Dict[str, Any] = {"conversation": conversation, "input_mode": input_mode}
        if kwargs.get("audio") is not None:
            prompt_dict["audio"] = kwargs["audio"]

        return prompt_dict

    # ------------------------------------------------------------------
    # process_inputs (single item) — NO tokenization
    # ------------------------------------------------------------------
    def process_inputs(self, data: Dict[str, Any], is_training: bool = False) -> Dict[str, Any]:
        prompt_obj = data.get("prompt")
        if not isinstance(prompt_obj, dict) or "conversation" not in prompt_obj:
            raise ValueError("FlamingoProcessor expects data['prompt'] dict with key 'conversation'.")

        audio = data.get("audio")
        if audio is None:
            audio = prompt_obj.get("audio")

        input_mode = data.get("input_mode", prompt_obj.get("input_mode", "speech_only"))
        if audio is None and input_mode != "text_only":
            logger.warning(
                "Flamingo process_inputs: missing audio for non-text_only item. "
                "tokenize_batch must get original_audios fallback or this will crash."
            )

        return {
            "prompt": prompt_obj,
            "completion": data.get("completion", ""),
            "audio": audio,
            "input_mode": input_mode,
        }

    # ------------------------------------------------------------------
    # _tokenize_single
    # ------------------------------------------------------------------
    def _tokenize_single(
        self,
        prompt_obj: Dict[str, Any],
        completion: str,
        audio: Any,
        is_training: bool,
    ) -> Dict[str, Any]:
        input_mode: str = prompt_obj.get("input_mode", "speech_only")
        conversation = copy.deepcopy(prompt_obj["conversation"])

        # Replace placeholder with actual audio block.
        for msg in conversation:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            new_content: List[Dict[str, Any]] = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and "__AUDIO_PLACEHOLDER__" in part.get("text", "")
                ):
                    cleaned = part.get("text", "").replace("__AUDIO_PLACEHOLDER__", "").strip()
                    if cleaned:
                        new_content.append({"type": "text", "text": cleaned})

                    if input_mode != "text_only":
                        if audio is None:
                            raise ValueError("Flamingo tokenization requires audio but audio is None.")
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

        # Normalize [1, L] -> [L]
        prompt_ids = prompt_inputs["input_ids"]
        if isinstance(prompt_ids, torch.Tensor) and prompt_ids.dim() > 1 and prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.squeeze(0)

        prompt_mask = prompt_inputs.get("attention_mask")
        if isinstance(prompt_mask, torch.Tensor) and prompt_mask.dim() > 1 and prompt_mask.shape[0] == 1:
            prompt_mask = prompt_mask.squeeze(0)

        prompt_length = int(prompt_ids.shape[-1])

        if is_training:
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
            full_mask = torch.cat([prompt_mask, completion_mask], dim=-1) if prompt_mask is not None else None
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        out: Dict[str, Any] = {"prompt_length": prompt_length}

        # Keep all keys returned by apply_chat_template (may include audio-related tensors)
        for k, v in prompt_inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dim() > 1 and v.shape[0] == 1:
                    v = v.squeeze(0)
                out[k] = v
            else:
                out[k] = v

        out["input_ids"] = full_ids
        if full_mask is not None:
            out["attention_mask"] = full_mask

        return out

    # ------------------------------------------------------------------
    # tokenize_batch
    # ------------------------------------------------------------------
    def tokenize_batch(
        self,
        prompts: List[Any],
        completions: Optional[List[str]] = None,
        padding_side: Optional[str] = None,
        original_audios: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        is_training = completions is not None
        batch_items: List[Dict[str, Any]] = []

        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, dict):
                raise ValueError("FlamingoProcessor expects each prompt to be a dict with 'conversation'.")

            completion = completions[i] if (completions is not None and i < len(completions)) else ""

            audio = prompt.get("audio")
            if audio is None and original_audios is not None and i < len(original_audios):
                audio = original_audios[i]

            input_mode = prompt.get("input_mode", "speech_only")
            if audio is None and input_mode != "text_only":
                raise ValueError(
                    f"Flamingo tokenize_batch: item {i} missing audio with input_mode={input_mode}. "
                    "Pass original_audios= or ensure dataset passes audio into format_prompt()."
                )

            item = self._tokenize_single(
                prompt_obj=prompt,
                completion=completion,
                audio=audio,
                is_training=is_training,
            )
            batch_items.append(item)

        return self.collate_batch(batch_items, padding_side=padding_side)

    # ------------------------------------------------------------------
    # collate_batch
    # ------------------------------------------------------------------
    def collate_batch(
        self,
        batch_items: List[Dict[str, Any]],
        padding_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not batch_items:
            return {}

        if padding_side is None:
            padding_side = "right" if any(bool(item.get("completion")) for item in batch_items) else "left"

        keys = set().union(*[item.keys() for item in batch_items])
        batch: Dict[str, Any] = {}

        passthrough = {
            "prompt",
            "text",
            "true_label",
            "dataset_type",
            "completion",
            "audio",
            "examples_audio",
            "input_mode",
            "fewshot_mode",
            "is_training",
        }

        for key in keys:
            values = [item.get(key) for item in batch_items]

            if key in passthrough:
                batch[key] = values
                continue

            if key == "prompt_length":
                batch[key] = torch.tensor([int(v or 0) for v in values])
                continue

            if key == "input_ids":
                tensors = [v for v in values if isinstance(v, torch.Tensor)]
                batch[key] = _pad_1d(
                    tensors,
                    pad_value=self.processor.tokenizer.pad_token_id or 0,
                    padding_side=padding_side,
                )
                continue

            if key == "attention_mask":
                tensors = [v for v in values if isinstance(v, torch.Tensor)]
                batch[key] = _pad_1d(tensors, pad_value=0, padding_side=padding_side)
                continue

            # For any other tensors returned by the processor (including audio features/masks):
            # - stack if shapes match
            # - otherwise keep as list (safe; avoids incorrect padding assumptions)
            if isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except Exception:
                    batch[key] = values
            else:
                batch[key] = values

        return batch