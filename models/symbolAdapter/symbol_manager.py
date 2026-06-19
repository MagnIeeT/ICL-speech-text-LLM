"""
Symbol Manager for dynamic/fixed symbol handling and optional label swapping.
"""

import logging
import random
import re
import string
from typing import Any, Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer

from config.data_config.master_config import DATASET_CONFIGS


def get_dataset_info(batch: Dict, fallback_labels: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Extract dataset name string and sorted label list from a batch."""
    ds = batch.get("dataset_type", [])
    ds = ds[0] if isinstance(ds, list) and ds else ds
    ds_name_str = ds.value if hasattr(ds, "value") else str(ds)
    for dt_enum, ds_cfg in DATASET_CONFIGS.items():
        if dt_enum.value == ds_name_str:
            return ds_name_str, sorted(list(ds_cfg.valid_labels))
    return ds_name_str, (fallback_labels or [])


def compute_slot_offsets(training_ds_names: set) -> Dict[str, int]:
    """Return {ds_name: slot_offset} for each training dataset, sorted alphabetically.
    Guarantees non-overlapping slot ranges across datasets."""
    offsets, offset = {}, 0
    for dt_enum, ds_cfg in sorted(DATASET_CONFIGS.items(), key=lambda x: x[0].value):
        if dt_enum.value in training_ds_names:
            offsets[dt_enum.value] = offset
            offset += len(ds_cfg.valid_labels)
    return offsets


class SymbolManager:
    """Manages mappings used to rewrite labels in prompt/completion text."""

    def __init__(
        self,
        original_labels: List[str],
        tokenizer: PreTrainedTokenizer,
        dynamic_per_epoch: bool = False,
        symbol_type: str = "two_token",
        no_symbols: bool = False,
        swap_labels: bool = False,
    ):
        self.original_labels = sorted(list(set(original_labels)))
        self.tokenizer = tokenizer
        self.dynamic_per_epoch = dynamic_per_epoch
        self.symbol_type = symbol_type
        self.no_symbols = no_symbols
        self.swap_labels = swap_labels

        self.fixed_mappings: Dict[str, str] = {}
        self.epoch_mappings_history: Dict[int, Dict[str, str]] = {}
        self.current_epoch = 0

        # Pure symbol mappings: label→symbol, generated once at init, never affected by
        # swap_labels. Used as the base pool when doing symbol-level swap.
        self._pure_symbol_mappings: Dict[str, str] = {}
        if not self.no_symbols:
            if self.symbol_type == "two_token":
                syms = self._generate_two_token_symbols(len(self.original_labels))
            else:
                syms = ["".join(random.choices(string.ascii_lowercase, k=4)) for _ in self.original_labels]
            self._pure_symbol_mappings = dict(zip(self.original_labels, syms))

        if self.no_symbols and not self.swap_labels:
            logging.info("No-symbol mode enabled - symbol replacement is disabled")
        elif self.swap_labels:
            logging.info("Swap mode enabled (update_strategy controls per-epoch vs per-instance)")
            if self._pure_symbol_mappings:
                logging.info("Base symbol pool for swap: %s", self._pure_symbol_mappings)
        elif not self.dynamic_per_epoch:
            self.fixed_mappings = self._pure_symbol_mappings.copy()
            logging.info("Generated fixed mappings: %s", self.fixed_mappings)
        else:
            logging.info("Dynamic mode enabled - mappings will be generated per epoch")

    def get_symbols_for_epoch(self, epoch: int, force_new_symbols: bool = False) -> Dict[str, str]:
        if self.no_symbols and not self.swap_labels: return {}
        if not self.dynamic_per_epoch: return self.fixed_mappings
        if force_new_symbols or epoch not in self.epoch_mappings_history:
            self.epoch_mappings_history[epoch] = self._generate_symbol_mappings()
        self.current_epoch = epoch
        return self.epoch_mappings_history[epoch]

    def get_current_symbols(self) -> Dict[str, str]:
        if not self.dynamic_per_epoch: return self.fixed_mappings
        return self.epoch_mappings_history.get(self.current_epoch, {})

    def get_reverse_mappings(self, epoch: Optional[int] = None, mappings: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if mappings is not None: active = mappings
        elif epoch is not None: active = self.get_symbols_for_epoch(epoch)
        else: active = self.get_current_symbols()
        return {v.lower(): k for k, v in active.items()}

    def _generate_symbol_mappings(self, force: bool = False) -> Dict[str, str]:
        if self.no_symbols and not force: return {}
        if self.symbol_type == "two_token": symbols = self._generate_two_token_symbols(len(self.original_labels))
        else: symbols = ["".join(random.choices(string.ascii_lowercase, k=4)) for _ in self.original_labels]
        return dict(zip(self.original_labels, symbols))

    def generate_swap_mapping_for_labels(
        self,
        relevant_labels: List[str],
        base_symbol_mapping: Optional[Dict[str, str]] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Generate a swap mapping scoped to the given dataset's label set.

        base_symbol_mapping: if provided (self.no_symbols=False), shuffles symbols across
                             labels (swap at symbol level). If None (self.no_symbols=True),
                             shuffles the original label names within this set.
        epoch: if provided (per_epoch mode), seeds the shuffle from (epoch, labels) so
               the same epoch always produces the same mapping. If None (per_instance
               mode), uses global random state for batch-level variety.
        """
        labels = sorted(list(set(relevant_labels)))
        if len(labels) <= 1:
            return {l: l for l in labels}

        if epoch is not None:
            rng = random.Random(hash((epoch, tuple(labels))) & 0x7FFFFFFF)
            shuffle_fn = rng.shuffle
        else:
            shuffle_fn = random.shuffle

        if self.no_symbols or not base_symbol_mapping:
            shuffled = labels[:]
            shuffle_fn(shuffled)
            return dict(zip(labels, shuffled))
        else:
            symbols = [base_symbol_mapping.get(l, l) for l in labels]
            shuffle_fn(symbols)
            return dict(zip(labels, symbols))

    def _generate_two_token_symbols(self, num_symbols: int) -> List[str]:
        words, used, attempts = [], set(), 0
        while len(words) < num_symbols and attempts < 10000:
            attempts += 1
            w = "".join(random.choice(string.ascii_lowercase) for _ in range(random.choice([4, 5])))
            if w in used: continue
            used.add(w)
            try:
                w_ids = self.tokenizer.encode(w, add_special_tokens=False)
                w_comma_ids = self.tokenizer.encode(f"{w},", add_special_tokens=False)
                cs_w_ids = self.tokenizer.encode(f", {w}", add_special_tokens=False)
                if (
                    len(w_ids) == 2
                    and len(w_comma_ids) == 3 and w_comma_ids[:2] == w_ids
                    and len(cs_w_ids) == 3
                ):
                    words.append(w)
            except: continue
        return words[:num_symbols]

    def _apply_mapping_safe(self, text: str, mapping: Dict[str, str]) -> str:
        if not mapping: return text

        # Protect description text in bullet lines from label replacement.
        # "- label: DESCRIPTION" — DESCRIPTION may contain the label word as a regular
        # English word (e.g. "other: Actions that don't fit other categories") and
        # must not be replaced. Only the label name (before the colon) is a target.
        desc_store = {}
        def _protect_desc(m):
            key = f"__DESCPROTECT_{len(desc_store):04d}__"
            desc_store[key] = m.group(2)
            return m.group(1) + key
        protected = re.sub(r'(?m)^([ \t]*-[^:\n]+: )(.+)$', _protect_desc, text)

        placeholders, transformed = {}, protected
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        for idx, src in enumerate(sorted_keys):
            placeholder = f"__SWAP_PLACEHOLDER_{idx}__"
            placeholders[placeholder] = mapping[src]
            transformed = re.compile(r'\b' + re.escape(src) + r'\b', re.IGNORECASE).sub(placeholder, transformed)
        for p, d in placeholders.items():
            transformed = transformed.replace(p, d)

        for key, val in desc_store.items():
            transformed = transformed.replace(key, val)
        return transformed

    def _apply_mapping_to_prompt_obj(self, prompt_obj: Any, mapping: Dict[str, str]) -> Any:
        if isinstance(prompt_obj, str): return self._apply_mapping_safe(prompt_obj, mapping)
        if isinstance(prompt_obj, dict) and "conversation" in prompt_obj:
            new_obj = dict(prompt_obj)
            new_conv = []
            for msg in prompt_obj.get("conversation", []):
                new_msg = dict(msg)
                if isinstance(new_msg.get("content"), str): new_msg["content"] = self._apply_mapping_safe(new_msg["content"], mapping)
                elif isinstance(new_msg.get("content"), list):
                    new_msg["content"] = [{"type": p["type"], "text": self._apply_mapping_safe(p["text"], mapping)} if p.get("type") == "text" else p for p in new_msg["content"]]
                new_conv.append(new_msg)
            new_obj["conversation"] = new_conv
            return new_obj
        return prompt_obj

    def replace_symbols_in_batch(self, batch, prompt_mappings, completion_mappings=None):
        p_map = prompt_mappings
        c_map = completion_mappings or prompt_mappings
        if not p_map: return batch
        updated = batch.copy()
        if "prompt" in batch: updated["prompt"] = [self._apply_mapping_to_prompt_obj(p, p_map) for p in batch["prompt"]]
        if "completion" in batch: updated["completion"] = [self._apply_mapping_safe(c, c_map) for c in batch["completion"]]
        return updated

    def convert_symbols_back(self, text: str, epoch: Optional[int] = None, mappings: Optional[Dict[str, str]] = None) -> str:
        if mappings is None and epoch is None and self.no_symbols and not self.swap_labels: return text
        rev = self.get_reverse_mappings(epoch=epoch, mappings=mappings)
        for s, l in rev.items():
            if s in text.lower(): text = re.compile(re.escape(s), re.IGNORECASE).sub(l, text)
        return text
