"""
Vocabulary Filtering for D-SPO.
Filters LLM vocabulary to identify 'Atomic Placeholders' (Neutral, Single-Token, Shielded).
"""

import logging
import torch
from typing import List, Optional, Set
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class VocabFilter:
    """
    Implements section 3.1: Vocabulary Auditing & Symbol Pool Generation.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def generate_symbol_pool(
        self, 
        pool_size: int = 500,
        exclude_labels: Optional[List[str]] = None,
        model_embeddings: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Generates a pool of token IDs following D-SPO constraints.
        """
        logger.info("Generating D-SPO Verified Symbol Pool...")
        
        vocab = self.tokenizer.get_vocab()
        candidate_ids = []

        # 1. Atomic Consistency: Only single-subword tokens
        # 2. Semantic Neutrality: (Simplified here, can be extended with frequency/similarity)
        
        # Identify all token IDs that compose the original labels to prevent 'Semantic Leakage'
        excluded_token_ids = set()
        if exclude_labels:
            for label in exclude_labels:
                # Tokenize the label to get all its constituent pieces
                label_tokens = self.tokenizer.encode(label, add_special_tokens=False)
                excluded_token_ids.update(label_tokens)
                # Also exclude case variations
                label_tokens_lower = self.tokenizer.encode(label.lower(), add_special_tokens=False)
                excluded_token_ids.update(label_tokens_lower)
        
        # Also exclude all special tokens (including our new <slot_n> placeholders)
        if hasattr(self.tokenizer, "all_special_ids"):
            excluded_token_ids.update(self.tokenizer.all_special_ids)

        for token, token_id in vocab.items():
            # Basic sanity checks
            if len(token) < 2 or len(token) > 10:
                continue
            
            # Hard Blacklist check (Labels + Special Tokens)
            if token_id in excluded_token_ids:
                continue

            # Check if it's an 'atomic' token (decodes back to itself cleanly)
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if not decoded:
                continue

            # Check if it's single token for the decoded string
            re_encoded = self.tokenizer.encode(decoded, add_special_tokens=False)
            if len(re_encoded) != 1:
                continue

            # Semantic Neutrality: Exclude common punctuation, numbers, non-ASCII.
            if any(char.isdigit() for char in decoded):
                continue
            if not decoded.isalpha():
                continue
            if not decoded.isascii():
                continue

            candidate_ids.append(token_id)

        logger.info(f"Found {len(candidate_ids)} atomic candidates.")

        # Frequency Proxy: In many BPE tokenizers (like Llama/Qwen), tokens added later
        # (higher IDs) are statistically less frequent in general corpora.
        # We sort by ID descending to prioritize 'rarer' tokens.
        candidate_ids = sorted(candidate_ids, reverse=True)

        if len(candidate_ids) < pool_size:
            logger.warning(f"Could only find {len(candidate_ids)} candidates, but requested {pool_size}.")
            return candidate_ids
        
        # Take the top N rarest (highest ID) candidates
        return candidate_ids[:pool_size]
