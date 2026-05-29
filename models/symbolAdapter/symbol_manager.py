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

        if self.no_symbols and not self.swap_labels:
            logging.info("No-symbol mode enabled - symbol replacement is disabled")
        elif not self.dynamic_per_epoch:
            self.fixed_mappings = self._generate_symbol_mappings()
            logging.info("Generated fixed mappings: %s", self.fixed_mappings)
        else:
            logging.info("Dynamic mode enabled - mappings will be generated per epoch")

    def get_symbols_for_epoch(self, epoch: int, force_new_symbols: bool = False) -> Dict[str, str]:
        if self.no_symbols and not self.swap_labels: return {}
        if not self.dynamic_per_epoch: return self.fixed_mappings
        if force_new_symbols or epoch not in self.epoch_mappings_history:
            self.epoch_mappings_history[epoch] = self._generate_symbol_mappings()
        return self.epoch_mappings_history[epoch]

    def get_current_symbols(self) -> Dict[str, str]:
        return self.fixed_mappings if not self.dynamic_per_epoch else self.epoch_mappings_history.get(self.current_epoch, {})

    def get_reverse_mappings(self, epoch: Optional[int] = None, mappings: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        active = mappings or (self.get_symbols_for_epoch(epoch) if epoch is not None else self.get_current_symbols())
        return {v.lower(): k for k, v in active.items()}

    def _generate_symbol_mappings(self) -> Dict[str, str]:
        if self.swap_labels: return self._generate_swap_mappings()
        if self.no_symbols: return {}
        symbols = self._generate_two_token_symbols(len(self.original_labels)) if self.symbol_type == "two_token" else ["".join(random.choices(string.ascii_lowercase, k=4)) for _ in self.original_labels]
        return dict(zip(self.original_labels, symbols))

    def _generate_swap_mappings(self) -> Dict[str, str]:
        labels = list(self.original_labels)
        if len(labels) <= 1: return {l: l for l in labels}
        shuffled = labels[:]
        random.shuffle(shuffled)
        return dict(zip(labels, shuffled))

    def _generate_two_token_symbols(self, num_symbols: int) -> List[str]:
        words = []
        while len(words) < num_symbols:
            w = "".join(random.choices(string.ascii_lowercase, k=5))
            try:
                if len(self.tokenizer.encode(w, add_special_tokens=False)) == 2: words.append(w)
            except: pass
        return words

    def _apply_mapping_safe(self, text: str, mapping: Dict[str, str]) -> str:
        """Apply mapping with case-insensitivity and global replacement."""
        if not text or not mapping: return text
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        transformed = text
        for src in sorted_keys:
            dst = mapping[src]
            # More aggressive regex: no word boundaries to ensure we catch 'Label:neutral'
            pattern = re.compile(re.escape(src), re.IGNORECASE)
            transformed = pattern.sub(dst, transformed)
        return transformed

    def replace_symbols_in_batch(
        self, batch: Dict[str, Any], epoch: Optional[int] = None,
        prompt_mappings: Optional[Dict[str, str]] = None,
        completion_mappings: Optional[Dict[str, str]] = None,
        force_new_symbols: bool = False
    ) -> Dict[str, Any]:
        p_map = prompt_mappings or self.get_symbols_for_epoch(epoch or 0, force_new_symbols=force_new_symbols)
        c_map = completion_mappings or p_map
        if not p_map: return batch
        updated = batch.copy()
        if "prompt" in batch:
            updated["prompt"] = [self._apply_mapping_safe(p, p_map) for p in batch["prompt"]]
        if "completion" in batch:
            updated["completion"] = [self._apply_mapping_safe(c, c_map) for c in batch["completion"]]
        return updated

    def convert_symbols_back(self, text: str, epoch: Optional[int] = None, mappings: Optional[Dict[str, str]] = None) -> str:
        rev = self.get_reverse_mappings(epoch=epoch, mappings=mappings)
        for s, l in rev.items():
            if s in text.lower(): text = re.compile(re.escape(s), re.IGNORECASE).sub(l, text)
        return text
