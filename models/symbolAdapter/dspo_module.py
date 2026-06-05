"""
Differentiable Symbolic Preference Optimization (D-SPO) Module.
Handles the injection of 'soft' embeddings into the LLM's computational graph.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class DspoModule(nn.Module):
    """
    Modular engine to inject differentiable symbol embeddings into LLM input.
    Ensures the computational graph remains connected from the loss to the Slot Matrix.
    """

    def __init__(self, slot_token_ids: List[int]):
        super().__init__()
        # These are the IDs of placeholder tokens added to the tokenizer (e.g., <slot_0>, <slot_1>)
        self.register_buffer("slot_token_ids", torch.tensor(slot_token_ids, dtype=torch.long))

    _inject_log_count: int = 0  # class-level counter; logs first 5 injections ever

    def inject_differentiable_symbols(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        router: nn.Module,
        embedding_layer: nn.Module
    ) -> torch.Tensor:
        """
        Replaces placeholder embeddings with differentiable symbol embeddings.

        Each slot has a private vocab of K tokens. For slot i:
          probs[i]      : [K] Gumbel-Softmax weights (sums to 1, over K only)
          slot_embeds[i]: [K, hidden] embeddings of slot i's K candidate tokens
          soft_embed[i] = probs[i] @ slot_embeds[i]   — [hidden]

        Args:
            input_ids:       [Batch, Seq] original token IDs
            input_embeds:    [Batch, Seq, Hidden] embeddings from the base model
            router:          SymbolRouter instance
            embedding_layer: base model embed_tokens layer

        Returns:
            modified_embeds: [Batch, Seq, Hidden] with soft symbol embeddings injected
        """
        import logging
        device = input_embeds.device

        # Get Gumbel-Softmax probs for all slots at once: [num_slots, K]
        all_slots = list(range(router.num_slots))
        _, probs = router.get_slot_mappings(all_slots, hard=True)  # [num_slots, K]

        modified_embeds = input_embeds.clone()
        injection_count = 0

        for i, slot_id in enumerate(self.slot_token_ids):
            if i >= router.num_slots:
                break

            mask = (input_ids == slot_id)
            if not mask.any():
                continue

            # Slot i's private K token embeddings: [K, hidden]
            slot_token_ids_i = router.slot_vocab_indices[i].to(device)  # [K]
            slot_embeds = embedding_layer(slot_token_ids_i)              # [K, hidden]

            # Soft embedding: weighted sum over this slot's K tokens only
            soft_embed = torch.matmul(probs[i].to(slot_embeds.dtype), slot_embeds)  # [hidden]

            if DspoModule._inject_log_count < 5:
                logging.info(
                    "D-SPO: Injecting Slot %d at %d positions. "
                    "Soft Embed Stats: mean=%.4f, std=%.4f",
                    i, mask.sum().item(),
                    soft_embed.mean().item(), soft_embed.std().item(),
                )
                DspoModule._inject_log_count += 1

            modified_embeds[mask] = soft_embed.to(modified_embeds.dtype)
            injection_count += 1

        return modified_embeds


def setup_dspo_tokenizer(tokenizer, config):
    """Adds slot placeholder tokens to the tokenizer if D-SPO is enabled."""
    if config.diff_symbol_config.enabled:
        slot_tokens = [f"<slot_{i}>" for i in range(config.diff_symbol_config.num_slots)]
        # Use a set to avoid duplicates if called multiple times
        tokenizer.add_special_tokens({"additional_special_tokens": slot_tokens})
        import logging
        logging.info("D-SPO: Added %d slot tokens to tokenizer", len(slot_tokens))
    return tokenizer


def setup_dspo_model(model, tokenizer, config):
    """Resizes model embeddings to match the new tokenizer size if D-SPO is enabled."""
    if config.diff_symbol_config.enabled:
        import logging
        new_size = len(tokenizer)

        # 1. Check if the wrapped model has a direct resize method (Standard HF / AudioFlamingo3)
        if hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
            model.model.resize_token_embeddings(new_size)
            logging.info("D-SPO: Resized base model embeddings to %d", new_size)
            
        # 2. Salmonn style fallback
        elif hasattr(model, "model") and hasattr(model.model, "llama_model"):
            model.model.llama_model.resize_token_embeddings(new_size)
            logging.info("D-SPO: Resized Salmonn (Llama) embeddings to %d", new_size)
            
        # 3. Explicit fallback for Flamingo variants where language model is nested under lang_encoder
        elif hasattr(model, "model") and hasattr(model.model, "lang_encoder") and hasattr(model.model.lang_encoder, "resize_token_embeddings"):
            model.model.lang_encoder.resize_token_embeddings(new_size)
            logging.info("D-SPO: Resized Flamingo lang_encoder embeddings to %d", new_size)
            
        # 4. Final safety fallback to prevent silent failures
        else:
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(new_size)
                logging.info("D-SPO: Resized outer wrapper embeddings to %d", new_size)
            else:
                logging.error("D-SPO: CRITICAL ERROR - Could not find a valid resize_token_embeddings method. Model will crash.")
                raise AttributeError("The current model architecture does not support token embedding resizing.")
                
    return model
