"""
FlamingoProcessor for nvidia/audio-flamingo-3-hf.

Key design constraint vs Qwen:
  Qwen inserts exactly 1 <|AUDIO|> token per clip regardless of duration.
  Flamingo inserts N <sound> tokens per clip where N = f(audio_duration / 30s).
  Therefore we CANNOT use tokenize=False (text string) because the token count
  would be wrong. We must call apply_chat_template(tokenize=True) with the actual
  audio array so the processor can determine the correct N internally.

Data flow:
  process_inputs()  → stores raw prompt dict + raw audio numpy array in the batch
                       (does NOT tokenize — symbol replacement happens first)
  tokenize_batch()  → receives prompt dicts with _audio injected by training loop,
                       calls apply_chat_template(tokenize=True, return_dict=True)
                       which handles <sound> expansion and audio feature extraction
                       in one shot, then appends completion tokens for training.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor


def _pad_sequence(tensors: List[torch.Tensor], pad_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    max_len = max(t.shape[-1] for t in tensors)
    out = []
    for t in tensors:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        pad_size = max_len - t.shape[-1]
        if pad_size > 0:
            pad_args = (pad_size, 0) if padding_side == "left" else (0, pad_size)
            t = torch.nn.functional.pad(t, pad_args, value=pad_value)
        out.append(t)
    return torch.stack(out)


def _pad_features(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad audio feature tensors along dim=1 (time) to the longest in the batch."""
    if not tensors or tensors[0].dim() < 2:
        return torch.stack(tensors)
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        pad = max_len - t.shape[1]
        if pad > 0:
            pad_shape = list(t.shape)
            pad_shape[1] = pad
            t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=1)
        padded.append(t)
    return torch.stack(padded)


class FlamingoProcessor(ModelProcessor):
    """Processor for nvidia/audio-flamingo-3-hf."""

    SAMPLING_RATE = 16000

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self._log_count = 0

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
        """Returns a conversation dict. Audio arrays go directly into the conversation."""
        conversation = [{"role": "system", "content": template}]
        user_content: List[Dict] = []

        if examples:
            user_content.append({"type": "text", "text": "Here are a few examples:\n"})
            for example in examples:
                if fewshot_mode == "speech" and example.get("audio") is not None:
                    user_content.extend([
                        {"type": "audio", "audio": _ensure_numpy(example["audio"]), "sampling_rate": self.SAMPLING_RATE},
                        {"type": "text", "text": f"Label: {example.get('label', '')}\n"},
                    ])
                else:
                    user_content.extend([
                        {"type": "text", "text": f"Text: {example.get('text', '')}\n"},
                        {"type": "text", "text": f"Label: {example.get('label', '')}\n"},
                    ])

        user_content.append({"type": "text", "text": "\nNow analyse this input:\n"})
        if input_mode == "text_only":
            user_content.append({"type": "text", "text": text})
        else:
            # Placeholder filled by tokenize_batch via _audio injected by training/val loop
            user_content.append({"type": "audio", "audio": None, "sampling_rate": self.SAMPLING_RATE})

        conversation.append({"role": "user", "content": user_content})
        return {"conversation": conversation, "input_mode": input_mode}

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False) -> Dict[str, Any]:
        """Pre-compute audio AND tokenize in the DataLoader worker.

        By tokenizing here (in the worker, parallel to GPU forward/backward), the main
        thread only needs to move tensors to device and call forward() — eliminating
        apply_chat_template from the critical path and pushing GPU utilization from ~23%
        toward ~80%+.

        Falls back to returning raw fields if tokenization fails, so the main thread can
        still tokenize via the legacy path.
        """
        prompt = data.get("prompt")
        completion = data.get("completion", "")
        audio = data.get("audio")
        audio_np = _ensure_numpy(audio) if audio is not None else None

        precomputed = None
        if audio_np is not None:
            try:
                audio_inputs, _ = self.processor._process_audio(
                    [audio_np],
                    sampling_rate=self.SAMPLING_RATE,
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                precomputed = {
                    "input_features": audio_inputs["input_features"],
                    "input_features_mask": audio_inputs["input_features_mask"],
                    "num_audio_tokens": int(audio_inputs["num_audio_tokens"][0].item()),
                }
            except Exception as exc:
                logging.warning("FlamingoProcessor: audio pre-compute failed (%s), falling back to slow path", exc)

        try:
            tokenized = self._tokenize_one(prompt, audio_np, completion if is_training else "", precomputed=precomputed)
        except Exception as exc:
            logging.warning("FlamingoProcessor: worker tokenization failed (%s), returning raw for main-thread fallback", exc)
            return {
                "prompt": prompt,
                "completion": completion,
                "audio": audio_np,
                "_precomputed": precomputed,
            }

        # Keep raw fields alongside tokenized tensors so debug logging and passthrough
        # keys in collate_batch still work.
        tokenized["prompt"] = prompt
        tokenized["completion"] = completion
        tokenized["audio"] = audio_np
        tokenized["_precomputed"] = precomputed
        return tokenized

    def tokenize_batch(
        self,
        prompts: List[Any],
        completions: Optional[List[str]] = None,
        padding_side: str = "right",
    ) -> Dict[str, Any]:
        """
        Full tokenization after symbol replacement.
        Expects prompt dicts with _audio and _precomputed already injected:
            for p, a, pre in zip(batch["prompt"], batch["audio"], batch["_precomputed"]):
                if isinstance(p, dict):
                    p["_audio"] = a
                    if pre is not None: p["_precomputed"] = pre
        """
        items = []
        for i, prompt in enumerate(prompts):
            completion = completions[i] if completions is not None else ""
            audio = prompt.get("_audio") if isinstance(prompt, dict) else None
            precomputed = prompt.get("_precomputed") if isinstance(prompt, dict) else None
            items.append(self._tokenize_one(prompt, audio, completion, precomputed=precomputed))
        return self.collate_batch(items, padding_side=padding_side)

    def _tokenize_one(self, prompt_obj: Dict, audio, completion: str, precomputed=None) -> Dict[str, Any]:
        if not isinstance(prompt_obj, dict) or "conversation" not in prompt_obj:
            raise ValueError("FlamingoProcessor expects prompt to be a dict with 'conversation' key.")

        conversation = copy.deepcopy(prompt_obj["conversation"])
        input_mode = prompt_obj.get("input_mode", "speech_only")
        has_audio = input_mode != "text_only" and audio is not None

        if precomputed is not None and has_audio:
            # Fast path: substitute <sound>*N as text so apply_chat_template skips the feature
            # extractor (already computed in DataLoader workers). The Flamingo chat template
            # renders audio items as text anyway — so token sequences are identical.
            N = precomputed["num_audio_tokens"]
            for msg in conversation:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                msg["content"] = [
                    {"type": "text", "text": self.processor.audio_token * N}
                    if (isinstance(p, dict) and p.get("type") == "audio" and p.get("audio") is None)
                    else p
                    for p in content
                ]

            prompt_only_conv = [c for c in conversation if c.get("role") != "assistant"]
            prompt_inputs = self.processor.apply_chat_template(
                prompt_only_conv,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Inject pre-computed features (not returned since no audio items in conv)
            prompt_inputs["input_features"] = precomputed["input_features"]
            prompt_inputs["input_features_mask"] = precomputed["input_features_mask"]
        else:
            # Slow path (fallback): embed audio array directly, let apply_chat_template run
            # the feature extractor. Used when pre-computation failed or text_only mode.
            for msg in conversation:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                if has_audio:
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "audio" and part.get("audio") is None:
                            part["audio"] = _ensure_numpy(audio)
                else:
                    msg["content"] = [
                        p for p in content
                        if not (isinstance(p, dict) and p.get("type") == "audio" and p.get("audio") is None)
                    ]

            prompt_only_conv = [c for c in conversation if c.get("role") != "assistant"]
            prompt_inputs = self.processor.apply_chat_template(
                prompt_only_conv,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        prompt_ids = prompt_inputs["input_ids"].squeeze(0)
        prompt_mask = prompt_inputs["attention_mask"].squeeze(0)
        prompt_length = int(prompt_ids.shape[-1])

        if completion:
            eos = self.tokenizer.eos_token or ""
            comp_out = self.tokenizer(
                f"{completion}{eos}",
                add_special_tokens=False,
                return_tensors="pt",
            )
            comp_ids = comp_out["input_ids"].squeeze(0)
            comp_mask = comp_out["attention_mask"].squeeze(0)
            full_ids = torch.cat([prompt_ids, comp_ids], dim=-1)
            full_mask = torch.cat([prompt_mask, comp_mask], dim=-1)
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        if self._log_count < 3:
            audio_token_id = getattr(self.processor, "audio_token_id", None)
            n_sound = int((prompt_ids == audio_token_id).sum().item()) if audio_token_id is not None else -1
            dur_s = round(len(audio) / self.SAMPLING_RATE, 2) if audio is not None else None
            feat_shape = list(prompt_inputs["input_features"].shape) if "input_features" in prompt_inputs else None
            logging.info(
                "FlamingoProcessor [call %d]: audio_dur=%ss  <sound>_tokens=%d  input_features=%s  prompt_len=%d  total_len=%d",
                self._log_count, dur_s, n_sound, feat_shape, prompt_length, int(full_ids.shape[-1]),
            )
            self._log_count += 1

        result: Dict[str, Any] = {
            "prompt_length": prompt_length,
            "input_ids": full_ids,
            "attention_mask": full_mask,
        }
        for key, value in prompt_inputs.items():
            if key in ("input_ids", "attention_mask"):
                continue
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0) if value.dim() > 1 and value.shape[0] == 1 else value
            else:
                result[key] = value
        return result

    def collate_batch(self, batch_items: List[Dict[str, Any]], padding_side: str = "right") -> Dict[str, Any]:
        if not batch_items:
            return {}

        # Raw (un-tokenized) items — just group by key
        if "input_ids" not in batch_items[0]:
            batch: Dict[str, Any] = {}
            for key in set().union(*[item.keys() for item in batch_items]):
                batch[key] = [item.get(key) for item in batch_items]
            return batch

        passthrough = {"prompt", "text", "true_label", "dataset_type", "completion", "audio", "input_mode"}
        batch: Dict[str, Any] = {}

        for key in set().union(*[item.keys() for item in batch_items]):
            values = [item[key] for item in batch_items if key in item]
            if not values:
                continue
            if key in passthrough:
                batch[key] = values
            elif key == "prompt_length":
                batch[key] = torch.tensor([int(v) for v in values])
            elif not isinstance(values[0], torch.Tensor):
                batch[key] = values
            elif key == "input_ids":
                batch[key] = _pad_sequence(values, pad_value=self.tokenizer.pad_token_id or 0, padding_side=padding_side)
            elif key == "attention_mask":
                batch[key] = _pad_sequence(values, pad_value=0, padding_side=padding_side)
            elif key in ("input_features", "input_features_mask"):
                try:
                    batch[key] = _pad_features(values)
                except Exception:
                    batch[key] = values
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
