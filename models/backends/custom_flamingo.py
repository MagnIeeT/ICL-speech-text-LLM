import logging
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration

from utils.environment import get_env_path
from utils.training_utils import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class CustomFlamingo(nn.Module):
    """
    Backend wrapper for nvidia/audio-flamingo-3-hf.

    Notes:
    - Uses torch.autocast(device_type="cuda", dtype=torch.bfloat16) for bf16.
    - Supports D-SPO (router + dspo_module) injection.
    - Defensive fix for multi-audio (few-shot) batches:
        When a prompt contains multiple audio clips, apply_chat_template stacks them,
        producing input_features of shape [B, num_audios, 128, T] (4D).
        HF AudioFlamingo3's conv1d expects [total_audios, 128, T] (3D).
        Fix: flatten dims 0 and 1 before the model call.
    """

    def __init__(
        self,
        model_path: str = None,
        lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        prompt_template: str = "",
        max_txt_len: int = 512,
        ckpt_path: str = None,
        device=None,
        use_bf16: bool = True,
    ):
        super().__init__()

        model_path = model_path or get_env_path("FLAMINGO_MODEL_NAME")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = use_bf16
        self.torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

        logging.info("Loading Audio Flamingo 3 model from %s", model_path)
        start_time = time.time()

        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )

        self.input_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        logging.info("Loaded Audio Flamingo 3 in %.2f seconds", time.time() - start_time)

        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        if lora:
            logging.info("Applying LoRA: rank=%d, alpha=%d", lora_rank, lora_alpha)
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=lora_dropout,
                bias="none",
                inference_mode=False,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            logging.info("LoRA applied to Audio Flamingo 3")
            self.print_trainable_parameters()

        logging.info("Moving model to device: %s", self.device)
        self.model.to(self.device)

        if ckpt_path:
            checkpoint = load_checkpoint(ckpt_path, map_location=self.device)
            state = checkpoint.get("model_state", checkpoint.get("model", {}))
            self.model.load_state_dict(state, strict=False)

        self.prompt_template = prompt_template
        self.max_txt_len = max_txt_len
        self.lora = lora
        self.batch_counter = 0

        logging.info("Initialized CustomFlamingo from %s", model_path)
        logging.info("Precision: %s", "BF16" if use_bf16 else "FP32")

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logging.info(
            "Trainable params: %d (%.2f%% of %d total)",
            trainable,
            100.0 * trainable / total,
            total,
        )

    _SKIP_KEYS = frozenset(["prompt_length", "prompt", "text", "true_label", "dataset_type", "completion"])

    def _prepare_model_inputs(self, samples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        model_inputs: Dict[str, torch.Tensor] = {}
        for key, value in samples.items():
            if key in self._SKIP_KEYS:
                continue
            if not isinstance(value, torch.Tensor):
                continue

            value = value.to(self.device)

            if value.is_floating_point():
                value = value.to(self.torch_dtype)

            model_inputs[key] = value

        return model_inputs

    def _autocast_ctx(self):
        from contextlib import nullcontext

        if str(self.device).startswith("cuda") and self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _fix_audio_shapes(model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Flatten multi-audio batch dimensions before passing to HF AudioFlamingo3.

        apply_chat_template stacks multiple audio clips per item, producing:
          input_features:      [B, num_audios, mel_bins, time]  (4D)
          input_features_mask: [B, num_audios, time]            (3D)

        HF conv1d expects:
          input_features:      [B * num_audios, mel_bins, time] (3D)
          input_features_mask: [B * num_audios, time]           (2D)

        Single-audio batches are already 3D/2D and pass through unchanged.
        """
        if "input_features" in model_inputs:
            x = model_inputs["input_features"]
            if x.dim() == 4:
                # [B, num_audios, mel, time] -> [B*num_audios, mel, time]
                logging.debug(
                    "Flattening input_features: %s -> %s",
                    tuple(x.shape),
                    (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
                )
                model_inputs["input_features"] = x.flatten(0, 1).contiguous()

        if "input_features_mask" in model_inputs:
            m = model_inputs["input_features_mask"]
            if m.dim() == 3:
                # [B, num_audios, time] -> [B*num_audios, time]
                model_inputs["input_features_mask"] = m.flatten(0, 1).contiguous()
            elif m.dim() == 4:
                # [B, num_audios, 1, time] or similar -> [B*num_audios, time]
                model_inputs["input_features_mask"] = m.flatten(0, 1).squeeze(1).contiguous()

        return model_inputs

    def forward(self, samples: Dict[str, Any], router=None, dspo_module=None) -> Dict[str, torch.Tensor]:
        input_ids = samples["input_ids"].to(self.device).long()
        attention_mask = samples["attention_mask"].to(self.device)

        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)

        for i, prompt_len in enumerate(samples["prompt_length"]):
            pl = int(prompt_len)
            pl = max(0, min(pl, labels.size(1)))
            labels[i, :pl] = -100

        # Avoid NaN loss by providing a dummy_loss with a valid grad_fn
        if (labels != -100).sum().item() == 0:
            logging.error(
                "All labels are -100 (no supervised tokens). Skipping batch to avoid NaN. "
                "prompt_length=%s seq_len=%d",
                samples.get("prompt_length"),
                input_ids.size(1),
            )
            self.batch_counter += 1
            dummy_loss = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0.0
            return {"loss": dummy_loss, "logits": None, "labels": labels}

        model_inputs = self._prepare_model_inputs(samples)
        model_inputs["attention_mask"] = attention_mask
        model_inputs["labels"] = labels

        # Flatten [B, num_audios, mel, time] -> [B*num_audios, mel, time] for HF conv1d
        model_inputs = self._fix_audio_shapes(model_inputs)

        # Log shapes for first few batches to confirm correctness
        if self.batch_counter < 3:
            if "input_features" in model_inputs:
                x = model_inputs["input_features"]
                logging.info("AF3 forward: input_features shape=%s dtype=%s", tuple(x.shape), x.dtype)
            if "input_features_mask" in model_inputs:
                m = model_inputs["input_features_mask"]
                logging.info("AF3 forward: input_features_mask shape=%s dtype=%s", tuple(m.shape), m.dtype)

        # D-SPO injection
        if router is not None and dspo_module is not None:
            embedding_layer = self.model.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
            inputs_embeds = dspo_module.inject_differentiable_symbols(
                input_ids=input_ids,
                input_embeds=inputs_embeds,
                router=router,
                embedding_layer=embedding_layer,
            )
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs.pop("input_ids", None)
        else:
            model_inputs["input_ids"] = input_ids

        with self._autocast_ctx():
            outputs = self.model(**model_inputs, return_dict=True)

        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}

    def generate_output(self, batch: Dict[str, Any], slot_replacement=None) -> List[str]:
        input_ids = batch["input_ids"].to(self.device).long()
        attention_mask = batch["attention_mask"].to(self.device)

        # D-SPO inference: swap placeholder token IDs with hard vocab tokens
        if slot_replacement:
            input_ids = input_ids.clone()
            for placeholder_id, hard_vocab_id in slot_replacement.items():
                input_ids[input_ids == placeholder_id] = hard_vocab_id

        model_inputs = self._prepare_model_inputs(batch)
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask

        # Flatten [B, num_audios, mel, time] -> [B*num_audios, mel, time] for HF conv1d
        model_inputs = self._fix_audio_shapes(model_inputs)

        with self._autocast_ctx():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=20,
            )

        if generated_ids.size(1) <= input_ids.size(1):
            logging.error(
                "Generated sequence (%d) not longer than input (%d)",
                generated_ids.size(1),
                input_ids.size(1),
            )
            return [""] * input_ids.size(0)

        generated_ids = generated_ids[:, input_ids.size(1):]

        outputs = self.input_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CustomFlamingo":
        logging.info("Creating CustomFlamingo from config: %s", config)
        return cls(**config)