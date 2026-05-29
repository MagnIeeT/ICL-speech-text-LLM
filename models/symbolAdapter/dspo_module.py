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

    def inject_differentiable_symbols(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        router: nn.Module,
        embedding_layer: nn.Module
    ) -> torch.Tensor:
        """
        Replaces placeholder embeddings with differentiable symbol embeddings.
        
        Args:
            input_ids: [Batch, Seq] original token IDs.
            input_embeds: [Batch, Seq, Hidden] embeddings from the base model.
            router: The SymbolRouter instance.
            embedding_layer: The base model's embedding layer (e.g., model.model.embed_tokens).
            
        Returns:
            modified_embeds: [Batch, Seq, Hidden] with differentiable symbols injected.
        """
        device = input_embeds.device
        
        # 1. Get the Full Vocabulary Pool Embeddings [Pool_Size, Hidden]
        # We look up the embeddings for every token in our 'Verified Pool'.
        pool_indices = router.symbol_pool_indices.to(device)
        pool_embeds = embedding_layer(pool_indices)

        # 2. Get Differentiable Probabilities from the Router [Num_Slots, Pool_Size]
        # We use all slots to prepare the full mapping matrix.
        all_slots = list(range(router.num_slots))
        _, probs = router.get_slot_mappings(all_slots, hard=True)

        # 3. Calculate "Soft Embeddings" for all slots [Num_Slots, Hidden]
        # This matmul is the 'magic': it connects the graph.
        # soft_slot_embeds[i] = probs[i] @ pool_embeds
        soft_slot_embeds = torch.matmul(probs, pool_embeds.to(probs.dtype))

        # 4. Inject into the sequence
        # We clone to ensure we don't interfere with the base model's original cache
        modified_embeds = input_embeds.clone()

        # Iterate through the slots we have placeholders for
        # Note: self.slot_token_ids should match the order of indices 0...M in the Router
        injection_count = 0
        for i, slot_id in enumerate(self.slot_token_ids):
            if i >= router.num_slots:
                break
                
            # Find positions where this placeholder token exists
            mask = (input_ids == slot_id)
            if mask.any():
                # Get stats for verification (only for first few injections to avoid log flooding)
                if injection_count < 5:
                    soft_val = soft_slot_embeds[i]
                    import logging
                    logging.info(f"D-SPO: Injecting Slot {i} at {mask.sum().item()} positions. "
                                 f"Soft Embed Stats: mean={soft_val.mean().item():.4f}, std={soft_val.std().item():.4f}")
                
                # Replace the static placeholder embedding with the differentiable one
                modified_embeds[mask] = soft_slot_embeds[i].to(modified_embeds.dtype)
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

        # Determine the actual model object based on the wrapper type
        if hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
            # Standard transformers model (like Qwen)
            model.model.resize_token_embeddings(new_size)
            logging.info("D-SPO: Resized base model embeddings to %d", new_size)
        elif hasattr(model, "model") and hasattr(model.model, "llama_model"):
            # Salmonn style
            model.model.llama_model.resize_token_embeddings(new_size)
            logging.info("D-SPO: Resized Salmonn (Llama) embeddings to %d", new_size)
        else:
            logging.warning("D-SPO: Could not find standard resize_token_embeddings method on model.")
    return model
