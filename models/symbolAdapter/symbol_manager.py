"""
Symbol Manager for dynamic/fixed symbol handling and optional label swapping.
"""

import logging
import random
import re
import string
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizer


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

    def _generate_symbol_mappings(self) -> Dict[str, str]:
        if self.swap_labels: return self._generate_swap_mappings()
        if self.no_symbols: return {}
        if self.symbol_type == "two_token": symbols = self._generate_two_token_symbols(len(self.original_labels))
        else: symbols = ["".join(random.choices(string.ascii_lowercase, k=4)) for _ in self.original_labels]
        return dict(zip(self.original_labels, symbols))

    def _generate_swap_mappings(self) -> Dict[str, str]:
        labels = list(self.original_labels)
        if len(labels) <= 1: return {l: l for l in labels}
        shuffled = labels[:]
        random.shuffle(shuffled)
        return dict(zip(labels, shuffled))

    def generate_swap_mapping_for_labels(
        self,
        relevant_labels: List[str],
        base_symbol_mapping: Optional[Dict[str, str]] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Generate a swap mapping scoped to the given dataset's label set.

        no_symbols=True  → shuffle original label names within this set
        no_symbols=False → keep the same symbols but shuffle which symbol is
                           assigned to which label (swap at symbol level)

        epoch: when provided (per_epoch mode), seed the shuffle from (epoch, labels)
               so each epoch is guaranteed a different mapping. When None
               (per_instance mode), uses global random state for batch-level variety.
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
                if len(self.tokenizer.encode(w, add_special_tokens=False)) == 2: words.append(w)
            except: continue
        return words[:num_symbols]

    def _apply_mapping_safe(self, text: str, mapping: Dict[str, str], active_keys: Optional[set] = None) -> str:
        if not mapping: return text
        if active_keys is None: active_keys = set(mapping.keys())
        placeholders, transformed = {}, text
        sorted_keys = sorted([k for k in mapping.keys() if k in active_keys], key=len, reverse=True)
        for idx, src in enumerate(sorted_keys):
            placeholder = f"__SWAP_PLACEHOLDER_{idx}__"
            placeholders[placeholder] = mapping[src]
            transformed = re.compile(r'\b' + re.escape(src) + r'\b', re.IGNORECASE).sub(placeholder, transformed)
        for p, d in placeholders.items(): transformed = transformed.replace(p, d)
        return transformed

    def _apply_mapping_to_prompt_obj(self, prompt_obj: Any, mapping: Dict[str, str], active_keys: set) -> Any:
        if isinstance(prompt_obj, str): return self._apply_mapping_safe(prompt_obj, mapping, active_keys)
        if isinstance(prompt_obj, dict) and "conversation" in prompt_obj:
            new_obj = dict(prompt_obj)
            new_conv = []
            for msg in prompt_obj.get("conversation", []):
                new_msg = dict(msg)
                if isinstance(new_msg.get("content"), str): new_msg["content"] = self._apply_mapping_safe(new_msg["content"], mapping, active_keys)
                elif isinstance(new_msg.get("content"), list):
                    new_msg["content"] = [{"type": p["type"], "text": self._apply_mapping_safe(p["text"], mapping, active_keys)} if p.get("type") == "text" else p for p in new_msg["content"]]
                new_conv.append(new_msg)
            new_obj["conversation"] = new_conv
            return new_obj
        return prompt_obj

    def replace_symbols_in_batch(self, batch, epoch=None, prompt_mappings=None, completion_mappings=None, random_mask=False, force_new_symbols=False):
        if prompt_mappings is not None: p_map, c_map = prompt_mappings, (completion_mappings or prompt_mappings)
        elif epoch is not None: p_map = c_map = self.get_symbols_for_epoch(epoch, force_new_symbols=force_new_symbols)
        else: p_map = c_map = self.get_current_symbols()
        if not p_map: return batch
        updated = batch.copy()
        mask = set(random.sample(list(p_map.keys()), max(1, len(p_map)//8))) if random_mask else set(p_map.keys())
        if "prompt" in batch: updated["prompt"] = [self._apply_mapping_to_prompt_obj(p, p_map, mask) for p in batch["prompt"]]
        if "completion" in batch: updated["completion"] = [self._apply_mapping_safe(c, c_map, set(c_map.keys())) for c in batch["completion"]]
        return updated

    def convert_symbols_back(self, text: str, epoch: Optional[int] = None, mappings: Optional[Dict[str, str]] = None) -> str:
        if mappings is None and epoch is None and self.no_symbols and not self.swap_labels: return text
        rev = self.get_reverse_mappings(epoch=epoch, mappings=mappings)
        for s, l in rev.items():
            if s in text.lower(): text = re.compile(re.escape(s), re.IGNORECASE).sub(l, text)
        return text
