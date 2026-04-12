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
        tokenizer: PreTrainedTokenizer,
        dynamic_per_epoch: bool = False,
        symbol_type: str = "two_token",
        no_symbols: bool = False
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
        self.tokenizer = tokenizer
        self.dynamic_per_epoch = dynamic_per_epoch
        self.symbol_type = symbol_type
        self.no_symbols = no_symbols

        # Fixed symbols (used when dynamic_per_epoch=False)
        self.fixed_mappings = {}
        
        # Dynamic symbols tracking (used when dynamic_per_epoch=True)
        self.epoch_mappings_history = {}  # epoch -> mappings
        self.current_epoch = 0
        
        # Generate initial symbols
        if self.no_symbols:
            self.list_of_symbols = []
            logging.info("No-symbol mode enabled - symbol replacement is disabled")
        elif not self.dynamic_per_epoch:
            self.fixed_mappings = self._generate_symbol_mappings()
            self.list_of_symbols = list(self.fixed_mappings.values())
            logging.info(f"Generated fixed symbol mappings: {self.fixed_mappings}")
        else:
            logging.info("Dynamic symbol mode - symbols will be generated per epoch")
    
    def get_symbols_for_epoch(self, epoch: int, force_new_symbols: bool = False) -> Dict[str, str]:
        """
        Get symbol mappings for a specific epoch, with option to force new symbols

        Args:
            epoch: Epoch number (0-indexed)
            force_new_symbols: If True, generate new symbols even if already present

        Returns:
            Dictionary mapping original labels to symbols
        """
        if self.no_symbols:
            return {}

        if not self.dynamic_per_epoch:
            # Return fixed symbols
            return self.fixed_mappings

        # Dynamic mode: generate new symbols only when forced, otherwise reuse previous
        if force_new_symbols:
            logging.info(f"Generating NEW symbols for epoch {epoch} (force_new_symbols={force_new_symbols})")
            new_mappings = self._generate_symbol_mappings()
            self.epoch_mappings_history[epoch] = new_mappings

            # Log the new mappings
            logging.info(f"Epoch {epoch} symbol mappings:")
            for orig, symbol in new_mappings.items():
                logging.info(f"  '{orig}' -> '{symbol}'")

        elif epoch not in self.epoch_mappings_history:
            # If not forced and epoch not in history, reuse previous epoch's symbols
            prev_epoch = epoch - 1
            if prev_epoch in self.epoch_mappings_history:
                self.epoch_mappings_history[epoch] = self.epoch_mappings_history[prev_epoch]
                logging.info(f"Reusing symbols from epoch {prev_epoch} for epoch {epoch}")
            else:
                # First epoch: generate symbols
                logging.info(f"Generating initial symbols for epoch {epoch}")
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
    
    def _generate_symbol_mappings(self) -> Dict[str, str]:
        """Generate symbol mappings based on symbol_type"""
        if self.no_symbols:
            return {}

        # return dict(zip(self.original_labels, self.original_labels))
        if self.symbol_type == "two_token":
            symbols = self._generate_two_token_symbols(len(self.original_labels))
        else:
            raise ValueError(f"Unsupported symbol type: {self.symbol_type}")
        
        return dict(zip(self.original_labels, symbols))
        
    
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
        mappings: Optional[Dict[str, str]] = None,
        random_mask: bool = False,
        force_new_symbols: bool = False
    ) -> Dict[str, Any]:
        """
        Replace symbols in batch data, with optional random masking
        
        Args:
            batch: Batch dictionary with 'prompt' and 'completion' keys
            epoch: Specific epoch mappings to use (if None, uses current)
            mappings: Custom mappings to use (if provided, overrides epoch-based mappings)
            random_mask: If True, only replace a random subset (~1/4) of labels with symbols
            
        Returns:
            Updated batch with symbols replaced
        """
        if self.no_symbols:
            return batch

        # Use custom mappings if provided, otherwise get epoch-based mappings
        if mappings is not None:
            symbol_mappings = mappings
        elif epoch is not None:
            symbol_mappings = self.get_symbols_for_epoch(epoch, force_new_symbols=force_new_symbols)
        else:
            symbol_mappings = self.get_current_symbols()
        
        if not symbol_mappings:
            return batch
        
        updated_batch = batch.copy()
        
        # Determine which labels to mask if random_mask is True
        if random_mask:
            num_to_mask = max(1, len(symbol_mappings) // 8)
            masked_labels = set(random.sample(list(symbol_mappings.keys()), num_to_mask))
        else:
            masked_labels = set(symbol_mappings.keys())
        
        # Replace in prompts
        if "prompt" in batch:
            updated_prompts = []
            for prompt in batch["prompt"]:
                updated_prompt = prompt
                for original, symbol in symbol_mappings.items():
                    if original in masked_labels:
                        updated_prompt = updated_prompt.replace(original, symbol)
                updated_prompts.append(updated_prompt)
            updated_batch["prompt"] = updated_prompts
        
        # Replace in completions
        if "completion" in batch:
            updated_completions = []
            for completion in batch["completion"]:
                updated_completion = completion
                for original, symbol in symbol_mappings.items():
                    if original in masked_labels:
                        updated_completion = updated_completion.replace(original, symbol)
                updated_completions.append(updated_completion)
            updated_batch["completion"] = updated_completions
        
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
        if self.no_symbols:
            return text

        # Use custom mappings if provided, otherwise get epoch-based mappings
        if mappings is not None:
            # Create reverse mappings from custom mappings
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


    def setup_mixed_symbol_sets(self, epoch: int, num_symbol_sets: int = 2):
        """
        Setup multiple symbol sets for mixed symbol training.
        Call this once at the start of each epoch.
        
        Args:
            epoch: Current epoch number
            num_symbol_sets: Number of different symbol sets to create
        """
        self._mixed_symbol_sets = []
        self._mixed_symbol_sets_epoch = epoch
        
        # Set 0: Use the regular epoch symbols
        self._mixed_symbol_sets.append(self.get_symbols_for_epoch(epoch))
        
        # Generate additional fresh symbol sets using existing method
        for i in range(1, num_symbol_sets):
            # Use existing _generate_two_token_symbols method
            fresh_symbols_list = self._generate_two_token_symbols(len(self.original_labels))
            fresh_mappings = dict(zip(self.original_labels, fresh_symbols_list))
            self._mixed_symbol_sets.append(fresh_mappings)
        
        logging.info(f"Setup {num_symbol_sets} symbol sets for mixed training (epoch {epoch}):")
        for i, symbols in enumerate(self._mixed_symbol_sets):
            logging.info(f"  Set {i}: {symbols}")
    

    def get_symbol_set_for_step(self, batch_idx: int, grad_accum_steps: int = 1) -> Dict[str, str]:
        """
        Get the symbol set for a specific batch index.
        Ensures consistency across gradient accumulation steps.
        """
        if not hasattr(self, '_mixed_symbol_sets') or not self._mixed_symbol_sets:
            raise ValueError("Mixed symbol sets not initialized. Call setup_mixed_symbol_sets() first.")
        
        # 1. Calculate the 'Update Step' (Effective Index)
        # If grad_accum=4: batch 0,1,2,3 all result in effective_idx 0
        effective_idx = batch_idx // grad_accum_steps
        
        # 2. Create a local random generator seeded by this index
        # This makes the choice RANDOM but REPEATABLE for the same update step
        seed = effective_idx + getattr(self, '_mixed_symbol_sets_epoch', 0)
        rng = random.Random(seed)
        
        # 3. Pick a random set
        return rng.choice(self._mixed_symbol_sets)

    def replace_symbols_with_specific_set(self, batch: Dict[str, Any], symbol_set: Dict[str, str]) -> Dict[str, Any]:
        """
        Replace symbols in batch using a specific symbol set.
        
        Args:
            batch: Input batch with 'prompt' and 'completion' keys
            symbol_set: Specific symbol mapping to use
            
        Returns:
            Updated batch with symbol replacements
        """
        updated_batch = batch.copy()
        
        # Replace in prompts
        if 'prompt' in batch:
            updated_prompts = []
            for prompt in batch['prompt']:
                updated_prompt = prompt
                for original, symbol in symbol_set.items():
                    updated_prompt = updated_prompt.replace(original, symbol)
                updated_prompts.append(updated_prompt)
            updated_batch['prompt'] = updated_prompts
        
        # Replace in completions
        if 'completion' in batch:
            updated_completions = []
            for completion in batch['completion']:
                updated_completion = completion
                for original, symbol in symbol_set.items():
                    updated_completion = updated_completion.replace(original, symbol)
                updated_completions.append(updated_completion)
            updated_batch['completion'] = updated_completions
        
        return updated_batch