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
        **kwargs,
    ) -> Any:
        """
        Return type is model-dependent:
        - Qwen/Salmonn: a rendered string prompt
        - Flamingo: a structured prompt dict {"conversation": ..., "input_mode": ..., "audio": ...}

        **kwargs allows callers to pass model-specific extras without breaking
        processors that don't need them. In particular:
          audio=<np.ndarray>  — passed by BaseMultiTaskDataset for Flamingo so
                                the audio array is embedded in the prompt dict
                                and survives symbol replacement intact.
                                Qwen/Salmonn processors safely ignore this kwarg.
        """
        pass

    @abc.abstractmethod
    def tokenize_batch(
        self,
        prompts: List[Any],
        completions: Optional[List[str]] = None,
        padding_side: str = "right",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified tokenization for training and validation.

        **kwargs allows processor-specific parameters without breaking the
        shared call sites. In particular:
          original_audios=List[Any]  — Flamingo only. A fallback list of raw
                                       audio arrays used when audio has been
                                       stripped from prompt dicts during symbol
                                       replacement. Qwen/Salmonn ignore this.
        """
        pass

    @abc.abstractmethod
    def collate_batch(self, batch_items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
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
        return FlamingoProcessor(processor)

    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = ["ModelProcessor", "get_processor"]