"""
Symbol Manager for Dynamic and Fixed Symbol Handling
Extracted from unified_symbol_training.py for modular design
"""

import logging
import torch
import random
import string
from typing import Dict, List, Optional, Any, Union
from transformers import PreTrainedTokenizer

class SymbolManager:
    """
    Manages symbol mappings for training and inference
    Supports both fixed symbols (same throughout training) and dynamic symbols (new per epoch)
    """
    
    def __init__( 
        self, 
        original_labels: List[str],
        original_labels_dict: Optional[Dict[str,List[str]]],
        tokenizer: PreTrainedTokenizer,
        dynamic_per_epoch: bool = False,
        symbol_type: str = "two_token",
        only_original: bool = False # If True, return original labels as symbols without generation
    ):
        """
        Initialize Symbol Manager
        
        Args:
            original_labels: List of original dataset labels
            tokenizer: Tokenizer for symbol validation
            dynamic_per_epoch: If True, generate new symbols each epoch
            symbol_type: Type of symbols to generate ("two_token", "random_words", etc.)
        """
        self.original_labels = original_labels
        self.original_labels_dict = original_labels_dict
        self.tokenizer = tokenizer
        self.dynamic_per_epoch = dynamic_per_epoch
        self.symbol_type = symbol_type
        self.only_original = only_original
        
        logging.info(f"SymbolManager init: only_original={self.only_original}, dynamic_per_epoch={self.dynamic_per_epoch}, symbol_type={self.symbol_type}")

        # Fixed symbols (used when dynamic_per_epoch=False)
        self.fixed_mappings = {}
        
        # Dynamic symbols tracking (used when dynamic_per_epoch=True)
        self.epoch_mappings_history = {}  # epoch -> mappings
        self.current_epoch = 0
        
        # Generate initial symbols
        if not self.dynamic_per_epoch:
            self.fixed_mappings = self._generate_symbol_mappings()
            logging.info(f"Generated fixed symbol mappings: {self.fixed_mappings}")
        else:
            logging.info("Dynamic symbol mode - symbols will be generated per epoch")
    
    def get_symbols_for_epoch(self, epoch: int) -> Dict[str, str]:
        """
        Get symbol mappings for a specific epoch
        
        Args:
            epoch: Epoch number (0-indexed)
            
        Returns:
            Dictionary mapping original labels to symbols
        """
        if not self.dynamic_per_epoch:
            # Return fixed symbols
            return self.fixed_mappings
        
        # Dynamic mode: generate new symbols for this epoch
        if epoch not in self.epoch_mappings_history:
            logging.info(f"Generating NEW symbols for epoch {epoch}")
            new_mappings = self._generate_symbol_mappings()
            self.epoch_mappings_history[epoch] = new_mappings
            
            # Log the new mappings
            logging.info(f"Epoch {epoch} symbol mappings:")
            for orig, symbol in new_mappings.items():
                logging.info(f"  '{orig}' -> '{symbol}'")
        
        self.current_epoch = epoch
        return self.epoch_mappings_history[epoch]
    
    def get_current_symbols(self) -> Dict[str, str]:
        """Get current symbol mappings"""
        if not self.dynamic_per_epoch:
            return self.fixed_mappings
        else:
            return self.epoch_mappings_history.get(self.current_epoch, {})
    
    def get_reverse_mappings_dict(
        self,
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, str]:
        """
        Get reverse mappings (symbol → original label) across all datasets.

        Args:
            epoch: Epoch to retrieve symbols from (if not provided, use current).
            mappings: Optional dataset-wise mapping: {dataset: {label: symbol}}

        Returns:
            Flattened dictionary mapping symbols to original labels (symbol → label).
        """
        if mappings:
            label_to_symbol_by_dataset = mappings
        elif epoch is not None:
            label_to_symbol_by_dataset = self.get_symbols_for_epoch(epoch)
        else:
            label_to_symbol_by_dataset = self.get_current_symbols()

        reverse_mappings = {}
        for dataset, label_to_symbol in label_to_symbol_by_dataset.items():
            for label, symbol in label_to_symbol.items():
                reverse_mappings[symbol] = label
                reverse_mappings[symbol.lower()] = label  # for case-insensitive matching

        return reverse_mappings

    def get_reverse_mappings(self, epoch: Optional[int] = None,mappings: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get reverse mappings (symbol -> original label) for symbol conversion
        
        Args:
            epoch: Specific epoch (if None, uses current)
            
        Returns:
            Dictionary mapping symbols to original labels
        """
        if mappings:
            mappings = mappings
        elif epoch is not None:
            mappings = self.get_symbols_for_epoch(epoch)
        else:
            mappings = self.get_current_symbols()
        
        reverse_mappings = {}
        for original_label, symbol in mappings.items():
            reverse_mappings[symbol.lower()] = original_label
            reverse_mappings[symbol] = original_label
        
        return reverse_mappings
    
    def _generate_symbol_mappings(self) -> Dict[str, Dict[str, str]]:
        """Generate symbol mappings based on symbol_type, uniquely per dataset."""

        logging.info(f"original_labels_dict keys: {self.original_labels_dict.keys()}")
        mapped_dict = {}

        if self.only_original:
            for dataset in self.original_labels_dict:
                mapped_dict[dataset] = dict(zip(
                    self.original_labels_dict[dataset],
                    self.original_labels_dict[dataset]
                ))
            return mapped_dict

        for dataset, labels in self.original_labels_dict.items():
            logging.info(f"🎯 Generating symbols for dataset: {dataset} | {len(labels)} labels")

            if self.symbol_type == "two_token":
                symbols = self._generate_two_token_symbols(len(labels))
            else:
                raise ValueError(f"Unsupported symbol type: {self.symbol_type}")

            mapping = dict(zip(labels, symbols))
            mapped_dict[dataset] = mapping

            logging.info(f"✅ {dataset} Symbol mapping: {mapping}")

        return mapped_dict
        
    def _generate_two_token_symbols(self, num_symbols: int) -> List[str]:
        """
        Generate 2-token symbols (extracted from existing code)
        """
        chars = string.ascii_lowercase
        two_token_words = []
        used_words = set()
        
        attempts = 0
        max_attempts = 10000
        
        while len(two_token_words) < num_symbols and attempts < max_attempts:
            attempts += 1
            word_length = random.choice([4, 5])
            word = ''.join(random.choice(chars) for _ in range(word_length))
            
            if word in used_words:
                continue
            
            used_words.add(word)
            
            try:
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 2:
                    decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    if decoded.lower() == word.lower():
                        two_token_words.append(word)
            except:
                continue
        
        if len(two_token_words) < num_symbols:
            logging.warning(f"Could only generate {len(two_token_words)} symbols, needed {num_symbols}")
        
        return two_token_words[:num_symbols]

    def replace_symbols_in_batch(
        self,
        batch: Dict[str, Any],
        epoch: Optional[int] = None,
        mappings: Optional[Dict[str, Dict[str, str]]] = None,
        swap_symbols: bool = False,
        label_permutations: Optional[Dict[str, Dict[str, str]]] = None,
        original_labels: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Replace symbols in a batch, optionally swapping symbols using dataset-specific permutations.
        """

        # Get dataset-level label-to-symbol mapping
        if mappings is not None:
            label_to_symbol_by_dataset = mappings
        elif epoch is not None:
            label_to_symbol_by_dataset = self.get_symbols_for_epoch(epoch)
        else:
            label_to_symbol_by_dataset = self.get_current_symbols()

        if not label_to_symbol_by_dataset:
            logging.warning("✘ No symbol mappings found. Returning original batch.")
            return batch

        symbol_mappings = {}

        for dataset, label_to_symbol in label_to_symbol_by_dataset.items():
            # logging.info(f"\n📦 Dataset: {dataset}")
            # logging.info(f"  Label → Symbol: {label_to_symbol}")

            if swap_symbols and label_permutations and dataset in label_permutations:
                perm_map = label_permutations[dataset]  # {original_label: permuted_label}
                # logging.info(f"  🔁 Permutation Map: {perm_map}")
                try:
                    for orig_label, permuted_label in perm_map.items():
                        to_symbol = label_to_symbol[permuted_label]
                        symbol_mappings[orig_label] = to_symbol
                        # logging.info(f"    🔄 {orig_label} → {to_symbol}")
                except KeyError as e:
                    raise KeyError(f"❌ Missing label in label_to_symbol for dataset '{dataset}': {e}")
            else:
                logging.info(f"  ⏩ No permutation applied.")
                for orig_label, symbol in label_to_symbol.items():
                    symbol_mappings[orig_label] = symbol
                    # logging.info(f"    ➕ {orig_label} → {symbol}")


        updated_batch = batch.copy()

        # logging.info("\n🔄 Replacing symbols in batch...")

        logging.info(f"SYMBOL MAPPINGS {symbol_mappings}")

        # --- 🛠️ Two-pass non-destructive replacement ---
        def non_destructive_replace(text: str, mapping: Dict[str, str]) -> str:
            placeholder_map = {}
            temp_text = text

            # First pass: Replace original labels with placeholders
            for i, (label, _) in enumerate(mapping.items()):
                placeholder = f"<<SYM{i}>>"
                placeholder_map[placeholder] = label
                temp_text = temp_text.replace(label, placeholder)

            # Second pass: Replace placeholders with final symbols
            for placeholder, label in placeholder_map.items():
                temp_text = temp_text.replace(placeholder, mapping[label])

            return temp_text

        if "prompt" in batch:
            updated_batch["prompt"] = [
                non_destructive_replace(p, symbol_mappings) for p in batch["prompt"]
            ]
            # logging.info("✅ Updated Prompts (First 2):")
            # for i, p in enumerate(updated_batch["prompt"][:2]):
            #     logging.info(f"  [{i}] {p}")

        if "completion" in batch:
            updated_batch["completion"] = [
                non_destructive_replace(c, symbol_mappings) for c in batch["completion"]
            ]
            # logging.info("✅ Updated Completions (First 2):")
            # for i, c in enumerate(updated_batch["completion"][:2]):
            #     logging.info(f"  [{i}] {c}")

        logging.info("Symbol replacement completed.\n")
        
        return updated_batch

    def convert_symbols_back(self, text: str, epoch: Optional[int] = None, mappings: Optional[Dict[str, str]] = None) -> str:
        """
        Convert symbols back to original labels in text
        
        Args:
            text: Text with symbols to convert
            epoch: Specific epoch mappings to use (if None, uses current)
            mappings: Custom mappings to use (if provided, overrides epoch-based mappings)
            
        Returns:
            Text with symbols converted back to original labels
        """
        # Use custom mappings if provided, otherwise get epoch-based mappings
        if mappings is not None:
            # Create reverse mappings from custom mappings
            reverse_mappings = self.get_reverse_mappings_dict(mappings=mappings)
        elif epoch is not None:
            reverse_mappings = self.get_reverse_mappings_dict(epoch)
        else:
            reverse_mappings = self.get_reverse_mappings_dict()
        
        if not reverse_mappings:
            return text
        
        converted = text
        for symbol, original_label in reverse_mappings.items():
            if symbol in converted:
                converted = converted.replace(symbol, original_label)
            elif symbol.lower() in converted.lower():
                import re
                pattern = re.compile(re.escape(symbol), re.IGNORECASE)
                if pattern.search(converted):
                    converted = pattern.sub(original_label, converted)
        
        return converted
    
    def get_symbol_tokens(self, epoch: Optional[int] = None) -> List[str]:
        """
        Get list of symbol tokens for model tracking
        
        Args:
            epoch: Specific epoch (if None, uses current)
            
        Returns:
            List of symbol strings
        """
        mappings = self.get_symbols_for_epoch(epoch) if epoch is not None else self.get_current_symbols()
        return list(mappings.values())
    
    def save_mappings(self, filepath: str) -> None:
        """Save symbol mappings to file"""
        import json
        
        data = {
            "original_labels": self.original_labels,
            "dynamic_per_epoch": self.dynamic_per_epoch,
            "symbol_type": self.symbol_type,
            "fixed_mappings": self.fixed_mappings,
            "epoch_mappings_history": self.epoch_mappings_history,
            "current_epoch": self.current_epoch
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Saved symbol mappings to {filepath}")
    
    def load_mappings(self, filepath: str) -> None:
        """Load symbol mappings from file"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.original_labels = data["original_labels"]
        self.dynamic_per_epoch = data["dynamic_per_epoch"]
        self.symbol_type = data["symbol_type"]
        self.fixed_mappings = data["fixed_mappings"]
        self.epoch_mappings_history = data["epoch_mappings_history"]
        self.current_epoch = data["current_epoch"]
        
        logging.info(f"Loaded symbol mappings from {filepath}")
    
    def __str__(self) -> str:
        """String representation"""
        mode = "Dynamic" if self.dynamic_per_epoch else "Fixed"
        current_mappings = self.get_current_symbols()
        return f"SymbolManager({mode}, {len(current_mappings)} mappings, epoch={self.current_epoch})"


# Backwards compatibility functions (to keep existing code working)
def generate_one_word_two_token_symbols(num_symbols: int, tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Backwards compatibility function for existing code
    """
    manager = SymbolManager(
        original_labels=[f"label_{i}" for i in range(num_symbols)],
        tokenizer=tokenizer,
        dynamic_per_epoch=False,
        symbol_type="two_token"
    )
    return manager._generate_two_token_symbols(num_symbols)

def create_label_mapping(original_labels: List[str], symbols: List[str]) -> Dict[str, str]:
    """
    Backwards compatibility function for existing code
    """
    return dict(zip(original_labels, symbols))