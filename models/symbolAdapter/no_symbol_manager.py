"""No-op symbol manager used when no-symbol mode is enabled."""

from typing import Dict, List, Optional


class NoSymbolManager:
    def __init__(self, original_labels: List[str], tokenizer=None):
        self.original_labels = original_labels
        self.tokenizer = tokenizer
        self.dynamic_per_epoch = False
        self.symbol_type = "none"

    def replace_symbols_in_batch(self, batch, *args, **kwargs):
        return batch

    def convert_symbols_back(self, text: str, mappings: Optional[Dict[str, str]] = None) -> str:
        return text

    def get_symbols_for_epoch(self, epoch: int) -> Dict[str, str]:
        return {}

    def _generate_symbol_mappings(self) -> Dict[str, str]:
        return {}
