# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    you may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import shutil
import copy
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

# --- SPRInT MODIFICATION: IMPORT ---
import sys
# Path points to your main sprint_vision folder inside llava
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sprint_vision")))
try:
    from models.symbolAdapter.symbol_manager import SymbolManager
    from models.symbolAdapter.sprint_callbacks import (
        SPRInTSymbolEpochCallback,
        SPRInTProgressCallback,
        SPRInTEpochLogCallback,
        SPRInTValidationCallback,
    )
    from dataload.example_selector import ExampleSelector
    from dataload.prompt_builder import LLaVAPromptBuilder
except ImportError:
    logging.warning("SPRInT modules not found. Check sprint_vision/ path.")
    SymbolManager = None
    SPRInTSymbolEpochCallback = None
    SPRInTProgressCallback = None
    SPRInTEpochLogCallback = None
    SPRInTValidationCallback = None
    ExampleSelector = None
    LLaVAPromptBuilder = None
# -----------------------------------

# SPRInT logging state.
# _SPRINT_LOGGED_MAPPINGS: keyed by mapping snapshot → logs the first sample of
#   each unique mapping (covers static SS-FT once, and each ED-FT epoch change).
# _SPRINT_INSTANCE_LOG_COUNT: global emission counter for ID-FT, which has a fresh
#   mapping per sample — a key-based throttle would log every sample and grow the
#   dict unbounded, so we throttle by instance index instead (first 5, then /500).
# _SPRINT_TOKLEN_LOGGED: one-time token-length log (def blocks lengthen the prompt).
_SPRINT_LOGGED_MAPPINGS: dict = {}   # mapping_snapshot_str -> log_count
_SPRINT_INSTANCE_LOG_COUNT: int = 0
_SPRINT_TOKLEN_LOGGED: bool = False
_SPRINT_ICL_LOGGED: bool = False     # one-time log of the assembled ICL training prompt
_SPRINT_TRAINPROMPT_LOGGED: bool = False  # one-time first training prompt for regular (no-symbol) strategy


def _sprint_is_log_proc() -> bool:
    """
    True only in the main process or DataLoader worker 0, on local rank 0.

    preprocess_v1 runs inside DataLoader workers (num_workers>1), each a separate
    process with its own log-throttle counters — so without this gate the SPRInT
    training logs print once PER WORKER (the 4x duplication).  Gating on
    get_worker_info().id keeps 4 workers for speed while printing one clean copy.
    Note: all workers inherit local_rank=0, so a local_rank check alone would NOT
    de-duplicate; the worker-id check is what does it.
    """
    try:
        _wi = torch.utils.data.get_worker_info()
    except Exception:
        _wi = None
    if not (_wi is None or _wi.id == 0):
        return False
    return os.environ.get("LOCAL_RANK", "0") in ("0", "")

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset

# Compatibility shim: older accelerate lacks clear_device_cache (added ~0.26).
# Must be patched before peft is imported (peft.utils.loftq_utils imports it).
try:
    from accelerate.utils.memory import clear_device_cache as _cdc  # noqa: F401
except ImportError:
    import accelerate.utils.memory as _acc_mem
    import torch as _torch
    def _clear_device_cache():
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    _acc_mem.clear_device_cache = _clear_device_cache

from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, process_images

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    # --- SPRInT MODIFICATION: ICL TRAINING ---
    icl_shots: int = field(default=0,
                           metadata={"help": "ICL examples per training prompt. 0 = no ICL."})
    icl_pool_path: Optional[str] = field(default=None,
                           metadata={"help": "JSON pool for ICL example selection. Defaults to data_path."})
    icl_seed: int = field(default=42,
                           metadata={"help": "Base seed for per-item ICL selection (seed + item_index used per sample)."})
    max_train_samples: int = field(default=0,
                           metadata={"help": "Cap training to first N samples. 0 = use all samples."})
    # -----------------------------------------


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # --- SPRInT MODIFICATION: STRATEGY FLAG ---
    sprint_strategy: str = field(default="regular")
    sprint_dataset: str = field(default="colon",
                                metadata={"help": "Dataset name (colon|chest|endo). Drives the label vocabulary for SymbolManager."})
    # --- SPRInT MODIFICATION: VALIDATION FIELDS ---
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation JSON (LLaVA format). If set, runs symbol-aware accuracy after each epoch."}
    )
    max_val_samples: int = field(
        default=100,
        metadata={"help": "Max validation samples per epoch. 0 = use all."}
    )
    validation_modes: str = field(
        default="fixed,original,fresh",
        metadata={"help": "Comma-separated modes to run each epoch: fixed,original,fresh. "
                           "For RFT (no symbol_manager), only 'original' ever runs regardless."}
    )
    compute_val_auc_map: bool = field(
        default=True,
        metadata={"help": "Compute AUC (colon) or mAP+AUC (chest/endo) during validation. "
                           "Adds ~1-6 extra minutes per epoch. Set False to skip."}
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    symbol_manager: Optional[SymbolManager] = None # --- SPRInT MODIFICATION ---
) -> Dict:
    # --- SPRInT MODIFICATION: SYMBOLIC SWAP ---
    # Use apply_to_text (two-pass safe replacement) for BOTH human and GPT turns.
    # Same mapping is used across both turns of one sample (called once outside loop)
    # so the model sees consistent prompt-hint ↔ target pairing.
    # Works for:
    #   - Binary colon GPT turn  "0"                        → "kxzy"
    #   - Multi-label chest turn "pleural_effusion, nodule" → "abcd, efgh"
    #   - Human prompt           "(0 for No, 1 for Yes)"    → "(kxzy for No, wxab for Yes)"
    if symbol_manager is not None:
        mappings = symbol_manager.get_current_symbols()
        if mappings:
            global _SPRINT_LOGGED_MAPPINGS, _SPRINT_INSTANCE_LOG_COUNT
            is_per_instance = getattr(symbol_manager, "dynamic_per_instance", False)
            if is_per_instance:
                # ID-FT: fresh mapping per sample → throttle by global instance
                # index (first 5, then every 500th) to avoid unbounded log + dict.
                n = _SPRINT_INSTANCE_LOG_COUNT
                do_log = (n < 5) or (n % 500 == 0)
                _SPRINT_INSTANCE_LOG_COUNT = n + 1
                log_tag = f"ID-FT instance #{n}"
                mapping_key = None
            else:
                # SS-FT (one static mapping) / ED-FT (one mapping per epoch):
                # log the first sample of each unique mapping snapshot.
                mapping_key = str(sorted(mappings.items()))
                do_log = _SPRINT_LOGGED_MAPPINGS.get(mapping_key, 0) < 1
                log_tag = "static/epoch mapping"
            # Print one clean copy: only main process / worker 0 on rank 0.
            do_log = do_log and _sprint_is_log_proc()
            for source in sources:
                for sentence in source:
                    if sentence["from"] in ("gpt", "human"):
                        before_val = sentence["value"]
                        sentence["value"] = symbol_manager.apply_to_text(sentence["value"], mappings)
                        if do_log:
                            turn_label = "HUMAN INSTRUCTION" if sentence["from"] == "human" else "GPT TARGET"
                            # print() not logging.info() — root logger is at WARNING here, which
                            # drops logging.info(); print reaches stdout even from DataLoader workers.
                            print("=" * 70, flush=True)
                            print(f"[SPRINT::TRAIN] ({log_tag}) ACTIVE SYMBOL MAPPING: {mappings}", flush=True)
                            print(f"[SPRINT::TRAIN] turn={sentence['from']}  {turn_label} BEFORE replacement:\n{before_val}", flush=True)
                            print(f"[SPRINT::TRAIN] turn={sentence['from']}  {turn_label} AFTER  replacement:\n{sentence['value']}", flush=True)
                            print("=" * 70, flush=True)
            if do_log and mapping_key is not None:
                _SPRINT_LOGGED_MAPPINGS[mapping_key] = 1
    # ------------------------------------------

    # Regular strategy (no symbol_manager): the symbol BEFORE/AFTER block above
    # never runs, so log the FIRST training prompt once here — so the def-block
    # instruction + target are visible at the START of epoch 1 (not only later at
    # validation). Symbol strategies already show this via the block above.
    if symbol_manager is None:
        global _SPRINT_TRAINPROMPT_LOGGED
        if (not _SPRINT_TRAINPROMPT_LOGGED) and _sprint_is_log_proc():
            _SPRINT_TRAINPROMPT_LOGGED = True
            for _src in sources:
                for _sent in _src:
                    if _sent["from"] in ("human", "gpt"):
                        _tl = "HUMAN INSTRUCTION" if _sent["from"] == "human" else "GPT TARGET"
                        print("=" * 70, flush=True)
                        print(f"[SPRINT::TRAIN] (regular / no symbols) {_tl}:\n{_sent['value']}", flush=True)
                        print("=" * 70, flush=True)

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # SPRInT one-time token-length log — class-definition blocks lengthen the
    # prompt; confirm the def block is not being truncated at model_max_length.
    global _SPRINT_TOKLEN_LOGGED
    if (not _SPRINT_TOKLEN_LOGGED) and _sprint_is_log_proc():
        _SPRINT_TOKLEN_LOGGED = True
        _maxlen = tokenizer.model_max_length
        _n = int(input_ids.shape[1])
        _trunc = _n >= _maxlen
        print(
            f"[SPRINT::TRAIN] token length: n_input_ids={_n}  "
            f"model_max_length={_maxlen}  truncated={_trunc}",
            flush=True,
        )
        if _trunc:
            print("!" * 70, flush=True)
            print(
                f"[SPRINT::TRAIN] *** WARNING: prompt ({_n} tok) EXCEEDS model_max_length "
                f"({_maxlen}). The collator truncates the END, which cuts off the answer "
                f"tokens → loss=nan and CORRUPTED training. Fix: lower ICL_SHOTS (chest fits "
                f"≈1 with definitions), or raise --model_max_length (e.g. 4096), or set "
                f"ICL_SHOTS=0.",
                flush=True,
            )
            print("!" * 70, flush=True)

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    symbol_manager: Optional[SymbolManager] = None # --- SPRInT MODIFICATION ---
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, symbol_manager=symbol_manager)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    
    # Default LLaVA logic
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def _expand2square(pil_img, background_color):
    """Pad a PIL image to a square by centering on a solid background."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 symbol_manager: Optional[SymbolManager] = None): # --- SPRInT MODIFICATION ---
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        max_n = getattr(data_args, "max_train_samples", 0)
        if max_n > 0:
            list_data_dict = list_data_dict[:max_n]
            rank0_print(f"[SPRInT] max_train_samples={max_n}: using {len(list_data_dict)} samples")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.symbol_manager = symbol_manager # --- SPRInT MODIFICATION ---

        # --- SPRInT MODIFICATION: ICL TRAINING SETUP ---
        self.icl_shots = getattr(data_args, "icl_shots", 0)
        self.icl_seed  = getattr(data_args, "icl_seed", 42)
        if self.icl_shots > 0 and ExampleSelector is not None and LLaVAPromptBuilder is not None:
            pool_path = getattr(data_args, "icl_pool_path", None) or data_path
            self.example_selector = ExampleSelector(pool_path, seed=self.icl_seed)
            self.prompt_builder   = LLaVAPromptBuilder()
            rank0_print(f"[ICL Training] icl_shots={self.icl_shots}, pool={pool_path}")
        else:
            self.example_selector = None
            self.prompt_builder   = None
        # -------------------------------------------------

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        # 128 tokens per image is the estimate used in lengths; multiply by (1 + icl_shots)
        # so group_by_modality_length bins ICL samples with other long multimodal samples.
        extra_img_tokens = self.icl_shots * 128
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len + extra_img_tokens if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _load_image(self, image_file: str) -> torch.Tensor:
        """Load, pad, and preprocess a single image to a [C, H, W] tensor."""
        processor = self.data_args.image_processor
        image = Image.open(
            os.path.join(self.data_args.image_folder, image_file)
        ).convert("RGB")
        if self.data_args.image_aspect_ratio == "pad":
            image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        return processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # ── ICL training path ─────────────────────────────────────────────────
        # When icl_shots > 0 we embed K example images + labels into the human
        # turn, mirroring exactly what LLaVAPromptBuilder produces at inference.
        # preprocess_multimodal is SKIPPED because it collapses multiple <image>
        # tokens to one; tokenizer_image_token handles them correctly already.
        if self.icl_shots > 0 and self.example_selector is not None:
            item = self.list_data_dict[i]

            # Extract plain instruction (strip leading <image>\n from human turn)
            raw_human = item["conversations"][0]["value"]
            instruction = raw_human.replace(DEFAULT_IMAGE_TOKEN, "").lstrip("\n").strip()

            # Select K balanced examples; exclude this sample by ID
            examples = self.example_selector.select(
                n_shots=self.icl_shots,
                exclude_id=item.get("id"),
                seed=self.icl_seed + i,       # deterministic per item, varied across items
            )

            # Build human turn with K+1 <image> tokens (original labels, no symbol
            # substitution here — preprocess_v1 applies symbols uniformly below)
            new_human_val, image_paths = self.prompt_builder.build(
                instruction=instruction,
                examples=examples,
                test_image_path=item.get("image", ""),
                symbol_mappings=None,
            )

            # One-time COMPACT log of the ICL examples + structure, before epochs.
            # We deliberately do NOT dump the full K+1 prompt (the instruction is
            # identical in every block, so it would repeat the def block K+1 times).
            # Setting _SPRINT_TRAINPROMPT_LOGGED here suppresses the regular-path
            # prompt dump below so the same thing isn't printed twice.
            global _SPRINT_ICL_LOGGED, _SPRINT_TRAINPROMPT_LOGGED
            if (not _SPRINT_ICL_LOGGED) and _sprint_is_log_proc():
                _SPRINT_ICL_LOGGED = True
                _SPRINT_TRAINPROMPT_LOGGED = True
                n_blocks = len(examples) + 1
                print("=" * 70, flush=True)
                print(f"[SPRINT::ICL-TRAIN] icl_shots={self.icl_shots}  test sample id={item.get('id')!r}", flush=True)
                print(f"[SPRINT::ICL-TRAIN] selected {len(examples)} in-context example(s):", flush=True)
                for _k, _ex in enumerate(examples):
                    print(f"[SPRINT::ICL-TRAIN]   shot {_k}: id={_ex.get('id')!r}  image={_ex.get('image')!r}  label={_ex.get('label')!r}", flush=True)
                print(f"[SPRINT::ICL-TRAIN] image order (examples..., then test) = {image_paths}", flush=True)
                print(f"[SPRINT::ICL-TRAIN] prompt structure (ICI-style, instruction ONCE): "
                      f"instruction/definitions, then 'Here are few examples...', then "
                      f"{len(examples)} example pair(s) [<image> + 'Answer: <label>'], then "
                      f"'Now analyze this image:' + the query <image> "
                      f"(model predicts its label). {n_blocks} images total.", flush=True)
                print(f"[SPRINT::ICL-TRAIN] instruction/definition preamble (appears ONCE):\n{instruction}", flush=True)
                print(f"[SPRINT::ICL-TRAIN] test GPT target = {item['conversations'][1]['value']!r}", flush=True)
                print("=" * 70, flush=True)

            icl_sources = [[
                {"from": "human", "value": new_human_val},
                {"from": "gpt",   "value": item["conversations"][1]["value"]},
            ]]

            data_dict = preprocess(
                copy.deepcopy(icl_sources),
                self.tokenizer,
                has_image=True,
                symbol_manager=self.symbol_manager,
            )
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

            # Store K+1 images as a list of [C,H,W] tensors.
            # The collator keeps them as a list → LLaVA's prepare_inputs_labels_for_multimodal
            # takes the list path (type is list) and assigns one feature set per <image> token.
            data_dict["image"] = [self._load_image(p) for p in image_paths]
            return data_dict
        # ── End ICL path ──────────────────────────────────────────────────────

        # ── Standard single-image path (unchanged) ────────────────────────────
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image = self._load_image(image_file)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # --- SPRInT MODIFICATION: PASS SYMBOL MANAGER ---
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            symbol_manager=self.symbol_manager)
        # ------------------------------------------------

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if isinstance(images[0], list):
                # ICL multi-image path: each sample holds a list of [C,H,W] tensors.
                # Flatten to a single list so LLaVA's prepare_inputs_labels_for_multimodal
                # receives type(images) == list and assigns one feature set per <image> token.
                flat: List[torch.Tensor] = []
                for img_list in images:
                    flat.extend(img_list)
                batch['images'] = flat
            elif all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                symbol_manager: Optional[SymbolManager] = None) -> Dict: # --- SPRInT MODIFICATION ---
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                symbol_manager=symbol_manager) # --- SPRInT MODIFICATION ---
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# Shared mutable cell: LLaVATrainer.compute_loss writes the latest microbatch
# loss here so SPRInTProgressCallback.on_substep_end can read it and keep the
# tqdm postfix fresh on every raw batch, not just every optimizer step.
_sprint_microbatch_loss: list = [None]

def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # --- SPRInT MODIFICATION: INITIALIZE SYMBOL MANAGER ---
    sprint_manager = None
    _strategy = training_args.sprint_strategy

    # Resolve label vocabulary from dataset config — supports colon/chest/endo
    try:
        from config.data_config.master_config import get_dataset_config as _get_ds_cfg
        _original_labels = _get_ds_cfg(training_args.sprint_dataset).label_names
        rank0_print(f"SPRInT: dataset='{training_args.sprint_dataset}', labels={_original_labels}")
    except Exception as _e:
        rank0_print(f"SPRInT: dataset config load failed ({_e}), falling back to ['0','1']")
        _original_labels = ["0", "1"]

    if SymbolManager is not None and _strategy not in ("regular", "rft"):
        if _strategy == "lf_ft":
            sprint_manager = SymbolManager(
                original_labels=_original_labels,
                tokenizer=tokenizer,
                swap_labels=True,
            )
        elif _strategy == "ed_ft":
            sprint_manager = SymbolManager(
                original_labels=_original_labels,
                tokenizer=tokenizer,
                dynamic_per_epoch=True,
                symbol_type="two_token",
            )
            # HuggingFace Trainer preprocesses data at __getitem__ time, not inside
            # a per-batch training loop.  Without this call, epoch_mappings_history
            # is empty when the dataset is first accessed, get_current_symbols()
            # returns {}, and no symbol substitution happens — ED-FT silently trains
            # as RFT with no error.
            sprint_manager.get_symbols_for_epoch(0, force_new_symbols=True)
        elif _strategy == "id_ft":
            # Per-instance dynamic: fresh symbols are generated inside
            # preprocess_v1 each time get_current_symbols() is called.
            # Each sample's prompt + GPT turn share the same fresh mapping
            # because preprocess_v1 reads mappings ONCE per sample.
            sprint_manager = SymbolManager(
                original_labels=_original_labels,
                tokenizer=tokenizer,
                dynamic_per_instance=True,
                symbol_type="two_token",
            )
        else:
            # ss_ft / two_token: static fixed symbols, same mapping for the full run.
            sprint_manager = SymbolManager(
                original_labels=_original_labels,
                tokenizer=tokenizer,
                dynamic_per_epoch=False,
                symbol_type="two_token",
            )
        rank0_print(f"SPRInT SymbolManager Initialized (strategy={_strategy}). Mappings: {sprint_manager.get_current_symbols()}")
    else:
        rank0_print(f"SPRInT: strategy='{_strategy}' — using original labels, no symbol replacement.")
    # -----------------------------------------------------

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              symbol_manager=sprint_manager) # --- SPRInT MODIFICATION ---

    # --- SPRInT MODIFICATION: BUILD CALLBACKS ---
    sprint_callbacks = [SPRInTEpochLogCallback(total_epochs=int(training_args.num_train_epochs))]
    rank0_print(f"[SPRInT] Registered SPRInTEpochLogCallback for {training_args.num_train_epochs} epochs.")
    if sprint_manager is not None and _strategy == "ed_ft":
        sprint_callbacks.append(SPRInTSymbolEpochCallback(sprint_manager))
        rank0_print("[SPRInT] Registered SPRInTSymbolEpochCallback for ED-FT epoch rotation.")

    if training_args.eval_data_path:
        sprint_callbacks.append(SPRInTValidationCallback(
            model=model,
            tokenizer=tokenizer,
            image_processor=data_args.image_processor if hasattr(data_args, "image_processor") else None,
            data_args=data_args,
            training_args=training_args,
            symbol_manager=sprint_manager,
        ))
        rank0_print(f"[SPRInT] Registered SPRInTValidationCallback — eval_data_path={training_args.eval_data_path}")
    # --------------------------------------------

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=sprint_callbacks if sprint_callbacks else None,
                    **data_module)

    # Replace HF's ProgressCallback with ICI-style SPRInTProgressCallback.
    # This gives per-batch 'Batch N loss: X.XXXX' lines and epoch/loss in the tqdm bar.
    trainer.remove_callback(transformers.ProgressCallback)
    _train_ds_len = len(data_module.get('train_dataset', [])) if data_module else 0
    trainer.add_callback(SPRInTProgressCallback(train_dataset_len=_train_ds_len))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # --- SPRInT MODIFICATION: SAVE SYMBOL MAPPINGS ---
    if sprint_manager is not None and (training_args.local_rank == 0 or training_args.local_rank == -1):
        mappings_path = os.path.join(training_args.output_dir, "symbol_mappings.json")
        sprint_manager.save_mappings(mappings_path)
        rank0_print(f"✅ SPRInT: Saved symbol mappings to {mappings_path}")
    # --------------------------------------------------

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            # Sync non_lora_trainables.bin + config.json + symbol_mappings.json into
            # every checkpoint-N and checkpoint-best so any epoch can be used directly
            # for inference without manual file copying.
            _ckpt_dirs = [
                os.path.join(training_args.output_dir, d)
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-") and
                   os.path.isdir(os.path.join(training_args.output_dir, d))
            ]
            for _ckpt_dir in _ckpt_dirs:
                for _fname in ('non_lora_trainables.bin', 'config.json', 'symbol_mappings.json'):
                    _src = os.path.join(training_args.output_dir, _fname)
                    if os.path.isfile(_src):
                        shutil.copy2(_src, os.path.join(_ckpt_dir, _fname))
            rank0_print(
                f"[SPRInT] Synced inference files into {len(_ckpt_dirs)} checkpoint(s): "
                + ", ".join(os.path.basename(d) for d in sorted(_ckpt_dirs))
            )
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()