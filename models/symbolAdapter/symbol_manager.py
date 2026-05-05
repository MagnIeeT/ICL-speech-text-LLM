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
        self.original_labels = original_labels
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
        if self.no_symbols and not self.swap_labels:
            return {}

        if not self.dynamic_per_epoch:
            return self.fixed_mappings

        if force_new_symbols:
            new_mappings = self._generate_symbol_mappings()
            self.epoch_mappings_history[epoch] = new_mappings
            logging.info("Epoch %s mappings: %s", epoch, new_mappings)
        elif epoch not in self.epoch_mappings_history:
            prev_epoch = epoch - 1
            if prev_epoch in self.epoch_mappings_history:
                self.epoch_mappings_history[epoch] = self.epoch_mappings_history[prev_epoch]
            else:
                self.epoch_mappings_history[epoch] = self._generate_symbol_mappings()
                logging.info("Epoch %s mappings: %s", epoch, self.epoch_mappings_history[epoch])

        self.current_epoch = epoch
        return self.epoch_mappings_history[epoch]

    def get_current_symbols(self) -> Dict[str, str]:
        if not self.dynamic_per_epoch:
            return self.fixed_mappings
        return self.epoch_mappings_history.get(self.current_epoch, {})

    def get_reverse_mappings(
        self,
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        if mappings is not None:
            active_mappings = mappings
        elif epoch is not None:
            active_mappings = self.get_symbols_for_epoch(epoch)
        else:
            active_mappings = self.get_current_symbols()

        reverse_mappings: Dict[str, str] = {}
        for original_label, symbol in active_mappings.items():
            reverse_mappings[symbol.lower()] = original_label
            reverse_mappings[symbol] = original_label
        return reverse_mappings

    def _generate_symbol_mappings(self) -> Dict[str, str]:
        if self.swap_labels:
            return self._generate_swap_mappings()
        if self.no_symbols:
            return {}

        if self.symbol_type == "two_token":
            symbols = self._generate_two_token_symbols(len(self.original_labels))
        else:
            raise ValueError(f"Unsupported symbol type: {self.symbol_type}")

        return dict(zip(self.original_labels, symbols))

    def _generate_swap_mappings(self) -> Dict[str, str]:
        labels = list(self.original_labels)
        n = len(labels)

        if n <= 1:
            return {label: label for label in labels}

        if n == 2:
            return {labels[0]: labels[1], labels[1]: labels[0]}

        shuffled = labels[:]
        for _ in range(100):
            random.shuffle(shuffled)
            if all(src != dst for src, dst in zip(labels, shuffled)):
                return dict(zip(labels, shuffled))

        # Fallback deterministic derangement by rotation.
        rotated = labels[1:] + labels[:1]
        return dict(zip(labels, rotated))

    def _generate_two_token_symbols(self, num_symbols: int) -> List[str]:
        chars = string.ascii_lowercase
        two_token_words: List[str] = []
        used_words = set()

        attempts = 0
        max_attempts = 10000

        while len(two_token_words) < num_symbols and attempts < max_attempts:
            attempts += 1
            word_length = random.choice([4, 5])
            word = "".join(random.choice(chars) for _ in range(word_length))

            if word in used_words:
                continue
            used_words.add(word)

            try:
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 2:
                    decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    if decoded.lower() == word.lower():
                        two_token_words.append(word)
            except Exception:
                continue

        if len(two_token_words) < num_symbols:
            logging.warning("Could only generate %d symbols, needed %d", len(two_token_words), num_symbols)

        return two_token_words[:num_symbols]

    def _apply_mapping_safe(self, text: str, mapping: Dict[str, str], active_keys: Optional[set] = None) -> str:
        """Apply mapping in two passes so overlapping keys/values do not cascade."""
        if not mapping:
            return text

        if active_keys is None:
            active_keys = set(mapping.keys())

        placeholders: Dict[str, str] = {}
        transformed = text

        for idx, src in enumerate(mapping.keys()):
            if src not in active_keys:
                continue
            placeholder = f"__SWAP_PLACEHOLDER_{idx}__"
            placeholders[placeholder] = mapping[src]
            transformed = transformed.replace(src, placeholder)

        for placeholder, dst in placeholders.items():
            transformed = transformed.replace(placeholder, dst)

        return transformed

    def replace_symbols_in_batch(
        self,
        batch: Dict[str, Any],
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, str]] = None,
        random_mask: bool = False,
        force_new_symbols: bool = False,
    ) -> Dict[str, Any]:
        # If no specific mapping/epoch requested AND manager is in no_symbols mode, return original batch
        if mappings is None and epoch is None and self.no_symbols and not self.swap_labels:
            return batch

        if mappings is not None:
            symbol_mappings = mappings
        elif epoch is not None:
            symbol_mappings = self.get_symbols_for_epoch(epoch, force_new_symbols=force_new_symbols)
        else:
            symbol_mappings = self.get_current_symbols()

        # If we still have no mappings (e.g. baseline requested current symbols), return batch
        if not symbol_mappings:
            return batch

        updated_batch = batch.copy()

        if random_mask:
            num_to_mask = max(1, len(symbol_mappings) // 8)
            masked_labels = set(random.sample(list(symbol_mappings.keys()), num_to_mask))
        else:
            masked_labels = set(symbol_mappings.keys())

        if "prompt" in batch:
            updated_batch["prompt"] = [
                self._apply_mapping_safe(prompt, symbol_mappings, masked_labels) for prompt in batch["prompt"]
            ]

        if "completion" in batch:
            updated_batch["completion"] = [
                self._apply_mapping_safe(completion, symbol_mappings, masked_labels)
                for completion in batch["completion"]
            ]

        return updated_batch

    def convert_symbols_back(
        self,
        text: str,
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, str]] = None,
    ) -> str:
        # If no specific mapping/epoch provided AND manager is in no_symbols mode, return original text
        if mappings is None and epoch is None and self.no_symbols and not self.swap_labels:
            return text

        if mappings is not None:
            reverse_mappings = self.get_reverse_mappings(mappings=mappings)
        elif epoch is not None:
            reverse_mappings = self.get_reverse_mappings(epoch)
        else:
            reverse_mappings = self.get_reverse_mappings()

        if not reverse_mappings:
            return text

        converted = text
        for symbol, original_label in reverse_mappings.items():
            if symbol in converted:
                converted = converted.replace(symbol, original_label)
            elif symbol.lower() in converted.lower():
                pattern = re.compile(re.escape(symbol), re.IGNORECASE)
                if pattern.search(converted):
                    converted = pattern.sub(original_label, converted)

        return converted
