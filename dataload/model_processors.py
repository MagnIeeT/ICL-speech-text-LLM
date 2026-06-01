import abc
from typing import Any, Dict, List, Optional

import torch


class ModelProcessor(abc.ABC):
    """Abstract base class for model-specific processing."""

    def __init__(self, symbol_manager=None):
        self.symbol_manager = symbol_manager

    @abc.abstractmethod
    def process_inputs(self, data: Dict[str, Any], is_training: bool = False) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def format_prompt(
        self,
        template: str,
        text: str,
        examples: Optional[List[Dict]] = None,
        input_mode: str = "speech_only",
        fewshot_mode: str = "text",
        dataset_type: Optional[Any] = None,
    ) -> Any:
        """
        Return type is model-dependent:
        - Qwen/Salmonn: typically a rendered string prompt
        - Flamingo: a structured prompt object (dict with "conversation")
        """
        pass

    @abc.abstractmethod
    def tokenize_batch(self, prompts: List[str], completions: Optional[List[str]] = None, padding_side: str = "right") -> Dict[str, torch.Tensor]:
        """Unified tokenization method for training and validation."""
        pass

    @abc.abstractmethod
    def collate_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


def get_processor(model_type: str, processor=None, tokenizer=None, symbol_manager=None) -> ModelProcessor:
    """Return a model-specific processor instance."""
    model_type = model_type.lower()

    if model_type == "salmonn":
        from .salmon_processor import SalmonProcessor
        return SalmonProcessor(tokenizer, symbol_manager=symbol_manager)

    if model_type in ["qwen", "qwen2"]:
        from .qwen_processor import QwenProcessor
        return QwenProcessor(processor, tokenizer=tokenizer, symbol_manager=symbol_manager)

    if model_type in ["flamingo", "audioflamingo", "audioflamingo3"]:
        from .flamingo_processor import FlamingoProcessor
        # Keep Flamingo identical to remote version
        return FlamingoProcessor(processor)

    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = ["ModelProcessor", "get_processor"]
