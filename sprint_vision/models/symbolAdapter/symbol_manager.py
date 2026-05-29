"""
SymbolManager for dynamic/fixed symbol handling and optional label swapping.

Ported from ICI/models/symbolAdapter/symbol_manager.py and extended with
vlm_ICL-specific features (save/load JSON mappings, mixed symbol sets for
step-level selection during training).

Key additions vs. the old vlm_ICL version (to match ICI exactly):
  - no_symbols  param  → disables all symbol substitution (RFT baseline)
  - swap_labels param  → derangement permutation instead of random symbols (LF-FT)
  - _generate_swap_mappings()  → produces label-flip mapping
  - _apply_mapping_safe()      → two-pass replacement preventing cascade corruption
  - replace_symbols_in_batch() now uses _apply_mapping_safe() (was plain .replace())
  - apply_to_text()            → convenience for external callers (prompt_builder)
"""

import json
import logging
import random
import re
import string
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizer


class SymbolManager:
    """
    Manages symbol mappings for training and inference.

    Supports:
      - Fixed symbols  (SS-FT)  : same mapping for all epochs
      - Dynamic symbols (ED-FT) : new mapping generated each epoch
      - No symbols     (RFT)    : identity — labels unchanged
      - Label swap     (LF-FT)  : derangement permutation of original labels
    """

    def __init__(
        self,
        original_labels: List[str],
        tokenizer: PreTrainedTokenizer,
        dynamic_per_epoch: bool = False,
        symbol_type: str = "two_token",
        no_symbols: bool = False,
        swap_labels: bool = False,
    ):
        """
        Args:
            original_labels:   e.g. ["0", "1"]
            tokenizer:         Used to validate two-token symbols.
            dynamic_per_epoch: True → regenerate symbols each epoch (ED-FT).
            symbol_type:       "two_token" or "regular" (identity, kept for
                               backward-compat — prefer no_symbols=True for RFT).
            no_symbols:        True → disable all symbol replacement (RFT).
            swap_labels:       True → swap labels via derangement (LF-FT).
        """
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
            logging.info("No-symbol mode enabled — symbol replacement disabled (RFT)")
        elif not self.dynamic_per_epoch:
            self.fixed_mappings = self._generate_symbol_mappings()
            self.list_of_symbols = list(self.fixed_mappings.values())
            logging.info("Generated fixed symbol mappings: %s", self.fixed_mappings)
        else:
            logging.info("Dynamic symbol mode — mappings will be generated per epoch")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_symbols_for_epoch(self, epoch: int, force_new_symbols: bool = False) -> Dict[str, str]:
        """Return symbol mappings for the given epoch."""
        if self.no_symbols and not self.swap_labels:
            return {}

        if not self.dynamic_per_epoch:
            return self.fixed_mappings

        if force_new_symbols:
            new_mappings = self._generate_symbol_mappings()
            self.epoch_mappings_history[epoch] = new_mappings
            logging.info("Epoch %d symbol mappings (new): %s", epoch, new_mappings)
        elif epoch not in self.epoch_mappings_history:
            prev_epoch = epoch - 1
            if prev_epoch in self.epoch_mappings_history:
                self.epoch_mappings_history[epoch] = self.epoch_mappings_history[prev_epoch]
                logging.info("Epoch %d: reusing symbols from epoch %d", epoch, prev_epoch)
            else:
                self.epoch_mappings_history[epoch] = self._generate_symbol_mappings()
                logging.info("Epoch %d symbol mappings: %s", epoch, self.epoch_mappings_history[epoch])

        self.current_epoch = epoch
        return self.epoch_mappings_history[epoch]

    def get_current_symbols(self) -> Dict[str, str]:
        """Return the currently active symbol mappings."""
        if self.no_symbols and not self.swap_labels:
            return {}
        if not self.dynamic_per_epoch:
            return self.fixed_mappings
        return self.epoch_mappings_history.get(self.current_epoch, {})

    def get_reverse_mappings(
        self,
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Return symbol→original_label mapping for decoding model output."""
        if mappings is not None:
            active = mappings
        elif epoch is not None:
            active = self.get_symbols_for_epoch(epoch)
        else:
            active = self.get_current_symbols()

        reverse: Dict[str, str] = {}
        for original_label, symbol in active.items():
            reverse[symbol.lower()] = original_label
            reverse[symbol] = original_label
        return reverse

    def apply_to_text(self, text: str, mappings: Optional[Dict[str, str]] = None) -> str:
        """
        Apply current (or provided) symbol mappings to a single text string.

        Convenience wrapper around _apply_mapping_safe() for external callers
        such as prompt_builder.  Uses two-pass replacement.
        """
        active = mappings if mappings is not None else self.get_current_symbols()
        return self._apply_mapping_safe(text, active)

    def replace_symbols_in_batch(
        self,
        batch: Dict[str, Any],
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, str]] = None,
        random_mask: bool = False,
        force_new_symbols: bool = False,
    ) -> Dict[str, Any]:
        """
        Replace labels with symbols in batch["prompt"] and batch["completion"].

        Uses _apply_mapping_safe() (two-pass) to prevent cascade corruption,
        matching ICI's replace_symbols_in_batch() exactly.
        """
        if mappings is None and epoch is None and self.no_symbols and not self.swap_labels:
            return batch

        if mappings is not None:
            symbol_mappings = mappings
        elif epoch is not None:
            symbol_mappings = self.get_symbols_for_epoch(epoch, force_new_symbols=force_new_symbols)
        else:
            symbol_mappings = self.get_current_symbols()

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
                self._apply_mapping_safe(p, symbol_mappings, masked_labels)
                for p in batch["prompt"]
            ]

        if "completion" in batch:
            updated_batch["completion"] = [
                self._apply_mapping_safe(c, symbol_mappings, masked_labels)
                for c in batch["completion"]
            ]

        return updated_batch

    def convert_symbols_back(
        self,
        text: str,
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, str]] = None,
    ) -> str:
        """Convert symbol tokens in model output back to original labels."""
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

    def get_symbol_tokens(self, epoch: Optional[int] = None) -> List[str]:
        """Return list of symbol strings (for model tracking)."""
        mappings = self.get_symbols_for_epoch(epoch) if epoch is not None else self.get_current_symbols()
        return list(mappings.values())

    def save_mappings(self, filepath: str) -> None:
        """Persist all mappings to a JSON file."""
        data = {
            "original_labels": self.original_labels,
            "dynamic_per_epoch": self.dynamic_per_epoch,
            "symbol_type": self.symbol_type,
            "no_symbols": self.no_symbols,
            "swap_labels": self.swap_labels,
            "fixed_mappings": self.fixed_mappings,
            "epoch_mappings_history": self.epoch_mappings_history,
            "current_epoch": self.current_epoch,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logging.info("Saved symbol mappings to %s", filepath)

    def load_mappings(self, filepath: str) -> None:
        """Restore mappings from a JSON file saved by save_mappings()."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.original_labels = data["original_labels"]
        self.dynamic_per_epoch = data["dynamic_per_epoch"]
        self.symbol_type = data["symbol_type"]
        self.no_symbols = data.get("no_symbols", False)
        self.swap_labels = data.get("swap_labels", False)
        self.fixed_mappings = data["fixed_mappings"]
        self.epoch_mappings_history = data["epoch_mappings_history"]
        self.current_epoch = data["current_epoch"]
        logging.info("Loaded symbol mappings from %s", filepath)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_symbol_mappings(self) -> Dict[str, str]:
        """Dispatch to the appropriate mapping generator."""
        if self.swap_labels:
            return self._generate_swap_mappings()
        if self.no_symbols:
            return {}
        if self.symbol_type == "regular":
            # Identity mapping — kept for backward compatibility.
            # Prefer no_symbols=True for RFT going forward.
            return dict(zip(self.original_labels, self.original_labels))
        if self.symbol_type == "two_token":
            symbols = self._generate_two_token_symbols(len(self.original_labels))
            return dict(zip(self.original_labels, symbols))
        raise ValueError(f"Unsupported symbol_type: {self.symbol_type!r}")

    def _generate_swap_mappings(self) -> Dict[str, str]:
        """
        Generate a derangement (no fixed points) of the original labels.

        Mirrors ICI's _generate_swap_mappings() exactly.
        For binary labels ["0", "1"] this simply swaps them.
        """
        labels = list(self.original_labels)
        n = len(labels)

        if n <= 1:
            return {label: label for label in labels}
        if n == 2:
            return {labels[0]: labels[1], labels[1]: labels[0]}

        # Try random derangement
        shuffled = labels[:]
        for _ in range(100):
            random.shuffle(shuffled)
            if all(src != dst for src, dst in zip(labels, shuffled)):
                return dict(zip(labels, shuffled))

        # Deterministic fallback: rotation by one
        rotated = labels[1:] + labels[:1]
        return dict(zip(labels, rotated))

    def _generate_two_token_symbols(self, num_symbols: int) -> List[str]:
        """
        Generate random lowercase words that tokenize to exactly 2 sub-tokens.

        Mirrors ICI's _generate_two_token_symbols() exactly.
        """
        chars = string.ascii_lowercase
        two_token_words: List[str] = []
        used_words: set = set()
        attempts = 0
        max_attempts = 10_000

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
            logging.warning(
                "Could only generate %d two-token symbols, needed %d",
                len(two_token_words),
                num_symbols,
            )

        return two_token_words[:num_symbols]

    def _apply_mapping_safe(
        self,
        text: str,
        mapping: Dict[str, str],
        active_keys: Optional[set] = None,
    ) -> str:
        """
        Two-pass replacement to prevent cascade corruption.

        Mirrors ICI's SymbolManager._apply_mapping_safe() exactly.

        Pass 1: replace each active source label with a unique placeholder.
        Pass 2: replace each placeholder with its target symbol.

        This ensures that if label A maps to a value that contains label B,
        label B is not inadvertently replaced in the second step.

        Args:
            text:        Input string.
            mapping:     Full {original → symbol} dict.
            active_keys: Subset of mapping.keys() to apply
                         (supports random_mask in replace_symbols_in_batch).
                         None → apply all keys.
        """
        if not mapping:
            return text

        if active_keys is None:
            active_keys = set(mapping.keys())

        placeholders: Dict[str, str] = {}
        transformed = text

        # Pass 1
        for idx, src in enumerate(mapping.keys()):
            if src not in active_keys:
                continue
            placeholder = f"__SWAP_PLACEHOLDER_{idx}__"
            placeholders[placeholder] = mapping[src]
            transformed = transformed.replace(src, placeholder)

        # Pass 2
        for placeholder, dst in placeholders.items():
            transformed = transformed.replace(placeholder, dst)

        return transformed

    # ── Mixed-symbol training helpers (vlm_ICL step-level selection) ──────────

    def setup_mixed_symbol_sets(self, epoch: int, num_symbol_sets: int = 2) -> None:
        """
        Prepare multiple symbol sets for mixed-symbol training.
        Call once at the start of each epoch.
        """
        self._mixed_symbol_sets = [self.get_symbols_for_epoch(epoch)]
        self._mixed_symbol_sets_epoch = epoch

        for _ in range(1, num_symbol_sets):
            fresh = self._generate_two_token_symbols(len(self.original_labels))
            self._mixed_symbol_sets.append(dict(zip(self.original_labels, fresh)))

        logging.info("Setup %d symbol sets for epoch %d:", num_symbol_sets, epoch)
        for i, s in enumerate(self._mixed_symbol_sets):
            logging.info("  Set %d: %s", i, s)

    def get_symbol_set_for_step(self, batch_idx: int, grad_accum_steps: int = 1) -> Dict[str, str]:
        """
        Select a symbol set for the given batch index.
        Deterministic per effective update step; uses a seeded local RNG.
        """
        if not hasattr(self, "_mixed_symbol_sets") or not self._mixed_symbol_sets:
            raise ValueError("Call setup_mixed_symbol_sets() before get_symbol_set_for_step().")
        effective_idx = batch_idx // grad_accum_steps
        seed = effective_idx + getattr(self, "_mixed_symbol_sets_epoch", 0)
        return random.Random(seed).choice(self._mixed_symbol_sets)

    def replace_symbols_with_specific_set(
        self, batch: Dict[str, Any], symbol_set: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply a specific symbol set to batch (used in mixed-symbol training)."""
        updated = batch.copy()
        if "prompt" in batch:
            updated["prompt"] = [
                self._apply_mapping_safe(p, symbol_set) for p in batch["prompt"]
            ]
        if "completion" in batch:
            updated["completion"] = [
                self._apply_mapping_safe(c, symbol_set) for c in batch["completion"]
            ]
        return updated

    def __str__(self) -> str:
        mode = "Dynamic" if self.dynamic_per_epoch else "Fixed"
        if self.no_symbols:
            mode = "NoSymbols"
        elif self.swap_labels:
            mode = "SwapLabels"
        return (
            f"SymbolManager({mode}, {len(self.get_current_symbols())} mappings, "
            f"epoch={self.current_epoch})"
        )
