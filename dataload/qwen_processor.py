import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config.data_config.master_config import DatasetType
from .model_processors import ModelProcessor


def _pad_sequence(
    tensors: List[torch.Tensor],
    pad_value: int = 0,
    padding_side: str = "right",
) -> torch.Tensor:
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


class QwenProcessor(ModelProcessor):
    """Processor for Qwen2 audio model. Handles audio features and modular tokenization."""

    def __init__(self, processor, tokenizer=None, max_length: int = 512, symbol_manager=None):
        super().__init__(symbol_manager=symbol_manager)
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.max_length = max_length
        self._audio_logged = False
        self._audio_stats_count = 0

    def process_inputs(self, data: Dict[str, Any], is_training: bool = False):
        """Returns audio features and raw text strings. Skips tokenization."""
        input_features, feature_attention_mask = self._process_audio(
            data.get("audio"),
            data.get("examples_audio"),
        )
        result = {
            "prompt": data.get("prompt", ""),
            "completion": data.get("completion", ""),
            "input_features": input_features,
            "audio": data.get("audio"),
        }
        if feature_attention_mask is not None:
            result["feature_attention_mask"] = feature_attention_mask
        return result

    def _process_audio(self, audio, examples_audio):
        audios = []
        if examples_audio:
            if isinstance(examples_audio, (list, tuple)):
                audios.extend(examples_audio)
            else:
                audios.append(examples_audio)
        if audio is not None:
            audios.append(audio)

        if not audios:
            logging.warning(
                "_process_audio: no audio provided, returning None "
                "(prompt may have <|AUDIO|> but no features)"
            )
            return None, None

        valid_audios = []
        for i, a in enumerate(audios):
            arr = np.array(a)
            if arr.ndim == 0 or arr.size == 0:
                logging.warning(
                    "_process_audio: audio[%d] is empty (shape=%s), skipping.",
                    i,
                    arr.shape,
                )
                continue
            valid_audios.append(a)

        if not valid_audios:
            logging.warning(
                "_process_audio: all %d audio arrays were empty/invalid, returning None.",
                len(audios),
            )
            return None, None

        audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        placeholder_text = " ".join([audio_placeholder] * len(valid_audios))
        inputs = self.processor(
            text=placeholder_text,
            audio=valid_audios,
            return_tensors="pt",
            sampling_rate=16000,
        )

        try:
            raw_features = inputs["input_features"]
        except (KeyError, AttributeError):
            logging.warning(
                "_process_audio: input_features missing from processor output for %d audio clip(s). "
                "Audio may be too short or malformed. Returning None.",
                len(valid_audios),
            )
            return None, None

        if raw_features is None:
            logging.warning(
                "_process_audio: input_features is None for %d audio clip(s). Returning None.",
                len(valid_audios),
            )
            return None, None

        features = raw_features.squeeze(0).to(torch.float16)
        feat_mask = inputs.feature_attention_mask.squeeze(0) if hasattr(inputs, "feature_attention_mask") else None

        if self._audio_stats_count < 5:
            for i, a in enumerate(valid_audios):
                arr = np.array(a)
                logging.info(
                    "Audio stats [item %d, clip %d/%d]: length=%.2fs  samples=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                    self._audio_stats_count,
                    i + 1,
                    len(valid_audios),
                    len(arr) / 16000,
                    len(arr),
                    arr.mean(),
                    arr.std(),
                    arr.min(),
                    arr.max(),
                )
            logging.info(
                "Audio features shape=%s  dtype=%s  mask=%s",
                list(features.shape),
                str(features.dtype),
                list(feat_mask.shape) if feat_mask is not None else None,
            )
            self._audio_stats_count += 1

        return features, feat_mask

    def tokenize_batch(
        self,
        prompts: List[str],
        completions: Optional[List[str]] = None,
        padding_side: str = "right",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch. For validation/inference, if audio exists we use the full Qwen processor
        to ensure <|AUDIO|> is expanded into the correct number of audio tokens.
        """
        original_audios = kwargs.get("original_audios", None)
        new_ids: List[torch.Tensor] = []
        new_masks: List[torch.Tensor] = []
        prompt_lens: List[int] = []

        AUDIO_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<|AUDIO|>")

        for i, p_text in enumerate(prompts):
            if completions is not None:
                # --------------------------
                # Training path
                # --------------------------
                c_text = completions[i]
                c_ids = self.tokenizer(
                    f"{c_text}{self.tokenizer.eos_token}",
                    add_special_tokens=False,
                ).input_ids

                prompt_budget = self.max_length - len(c_ids)
                p_tokenized = self.tokenizer(
                    p_text,
                    truncation=True,
                    max_length=max(prompt_budget, 1),
                    padding=False,
                    add_special_tokens=True,
                )
                prompt_len = len(p_tokenized.input_ids)

                # Guard: audio token must survive prompt truncation.
                # Use continue (not raise) so one bad item skips without killing
                # the whole batch — the orchestrator's except catches raises anyway,
                # but continue is cleaner and keeps other items in the batch alive.
                if AUDIO_TOKEN_ID not in p_tokenized.input_ids:
                    logging.warning(
                        "tokenize_batch: <|AUDIO|> token (id=%d) lost after prompt truncation "
                        "(prompt_budget=%d). Skipping item %d.",
                        AUDIO_TOKEN_ID,
                        prompt_budget,
                        i,
                    )
                    continue

                combined_ids = p_tokenized.input_ids + c_ids
                combined_mask = p_tokenized.attention_mask + [1] * len(c_ids)

                if i == 0:
                    p_ids_full = self.tokenizer(p_text, add_special_tokens=True).input_ids
                    was_truncated = len(p_ids_full) > prompt_len
                    if was_truncated:
                        logging.warning(
                            "Prompt truncated from %d to %d tokens to fit completion.",
                            len(p_ids_full),
                            prompt_len,
                        )

                prompt_lens.append(prompt_len)
                new_ids.append(torch.tensor(combined_ids))
                new_masks.append(torch.tensor(combined_mask))
                continue

            # --------------------------
            # Validation / inference path
            # --------------------------
            audio = None
            if original_audios is not None and i < len(original_audios):
                audio = original_audios[i]

            if i < 3:
                logging.info(
                    "Qwen validate tokenize: item=%d  n_audio_tokens_in_text=%d",
                    i,
                    p_text.count("<|AUDIO|>"),
                )

            if audio is not None:
                arr = np.array(audio)
                if arr.size > 0:
                    n_audio = p_text.count("<|AUDIO|>")
                    if n_audio <= 0:
                        n_audio = 1

                    audio_list = [audio] * n_audio

                    proc_out = self.processor(
                        text=p_text,
                        audio=audio_list,
                        return_tensors="pt",
                        sampling_rate=16000,
                    )
                    tokenized_ids = proc_out["input_ids"].squeeze(0).tolist()
                    tokenized_mask = proc_out["attention_mask"].squeeze(0).tolist()

                    prompt_lens.append(len(tokenized_ids))
                    new_ids.append(torch.tensor(tokenized_ids))
                    new_masks.append(torch.tensor(tokenized_mask))
                    continue

            # Fallback: no audio, tokenizer only
            tokenized = self.tokenizer(
                p_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
            )
            prompt_lens.append(len(tokenized.input_ids))
            new_ids.append(torch.tensor(tokenized.input_ids))
            new_masks.append(torch.tensor(tokenized.attention_mask))

        # Guard: if every item was skipped (all lost their audio token after
        # truncation), raise explicitly so the orchestrator logs a clear error
        # rather than crashing inside _pad_sequence with an empty list.
        if not new_ids:
            raise ValueError(
                "tokenize_batch: all items in batch were skipped (audio token lost in all). "
                "Check prompt truncation budget or audio pipeline."
            )

        return {
            "input_ids": _pad_sequence(
                new_ids,
                pad_value=self.tokenizer.pad_token_id,
                padding_side=padding_side,
            ),
            "attention_mask": _pad_sequence(
                new_masks,
                pad_value=0,
                padding_side=padding_side,
            ),
            "prompt_length": torch.tensor(prompt_lens),
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

    def collate_batch(self, batch_items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Collate a list of items.
        **kwargs absorbs Flamingo-specific params harmlessly.

        IMPORTANT: For audio feature tensors, we must keep batch alignment.
        If any item has input_features=None, we raise (fail-fast) so you learn
        immediately that the dataset has missing/invalid audio in speech mode.
        """
        if not batch_items:
            return {}

        # RAW items branch (before tokenization)
        if "input_ids" not in batch_items[0]:
            batch: Dict[str, Any] = {}
            for key in batch_items[0].keys():
                values = [item.get(key) for item in batch_items]

                if key in ("input_features", "feature_attention_mask"):
                    if any(v is None for v in values):
                        logging.warning(
                            "collate_batch: some items missing %s (None present). "
                            "This will cause audio/token mismatches for Qwen. Failing fast.",
                            key,
                        )
                        raise ValueError(f"Missing {key} in batch; cannot stack.")
                    batch[key] = torch.stack(values)
                    continue

                batch[key] = values

            if batch.get("input_features") is None:
                logging.warning("Qwen collate_batch: input_features is None for this batch.")

            if "input_features" in batch and not self._audio_logged:
                feat = batch["input_features"]
                logging.info(
                    "Audio [first batch]: input_features shape=%s  dtype=%s",
                    list(feat.shape),
                    str(feat.dtype),
                )
                if "feature_attention_mask" in batch:
                    mask = batch["feature_attention_mask"]
                    logging.info(
                        "Audio [first batch]: feature_attention_mask shape=%s  active_frames=%s",
                        list(mask.shape),
                        mask.sum(dim=-1).tolist(),
                    )
                self._audio_logged = True

            return batch

        # TOKENIZED branch (after tokenization)
        batch: Dict[str, Any] = {}
        passthrough = {"prompt", "text", "completion", "dataset_type", "audio"}

        for key in batch_items[0].keys():
            values = [item[key] for item in batch_items if key in item]
            if not values:
                continue

            if key in passthrough:
                batch[key] = values
            elif key == "input_ids":
                batch[key] = _pad_sequence(values, pad_value=self.tokenizer.pad_token_id)
            elif key == "attention_mask":
                batch[key] = _pad_sequence(values, pad_value=0)
            elif key == "input_features":
                batch[key] = torch.stack(values)
            elif key == "prompt_length":
                batch[key] = torch.tensor([int(v) for v in values])
            else:
                try:
                    batch[key] = torch.stack(values)
                except Exception:
                    batch[key] = values

        return batch