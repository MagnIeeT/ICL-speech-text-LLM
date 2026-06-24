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

    slot_placeholder_ids: [num_slots, token_size] — tokenizer IDs of <slot_i_p> placeholders.
    For each slot i and each position p, the placeholder <slot_i_p> is replaced with a
    soft embedding computed as probs[i] @ slot_embeds_p[i], where probs[i] is shared
    across all positions (bound pair — same K-way choice drives all token_size positions).
    """

    def __init__(self, slot_placeholder_ids: List[List[int]]):
        super().__init__()
        # [num_slots, token_size] — placeholder token IDs for each slot position
        self.register_buffer(
            "slot_placeholder_ids",
            torch.tensor(slot_placeholder_ids, dtype=torch.long),
        )

    _inject_log_count: int = 0  # class-level counter; logs first 5 injections ever

    def inject_differentiable_symbols(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        router: nn.Module,
        embedding_layer: nn.Module,
        pre_computed_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Replaces placeholder embeddings with differentiable symbol embeddings.

        For slot i and position p:
          probs[i]         : [K] — shared weights over K candidate symbols (bound pair)
          slot_embeds_p[i] : [K, hidden] — embeddings of the p-th token of each candidate
          soft_embed_p     = probs[i] @ slot_embeds_p[i]   — [hidden]

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

        if pre_computed_probs is not None:
            probs = pre_computed_probs  # [num_slots, K]
            if DspoModule._inject_log_count < 5:
                logging.info("D-SPO injection: using pre-computed probs (inject == target guaranteed)")
        else:
            all_slots = list(range(router.num_slots))
            _, probs = router.get_slot_mappings(all_slots, hard=True)  # [num_slots, K]
            if DspoModule._inject_log_count < 5:
                logging.info("D-SPO injection: sampling fresh probs (validation mode)")

        modified_embeds = input_embeds.clone()
        token_size = self.slot_placeholder_ids.shape[1]

        for i in range(min(router.num_slots, self.slot_placeholder_ids.shape[0])):
            for p in range(token_size):
                placeholder_id = self.slot_placeholder_ids[i, p]
                mask = (input_ids == placeholder_id)
                if not mask.any():
                    continue

                # p-th token of each of slot i's K candidate symbols: [K, hidden]
                slot_tok_ids_p = router.slot_vocab_indices[i, :, p].to(device)  # [K]
                slot_embeds_p = embedding_layer(slot_tok_ids_p)                  # [K, hidden]

                # Soft embedding: same probs[i] drives all positions (bound pair)
                soft_embed = torch.matmul(probs[i].to(slot_embeds_p.dtype), slot_embeds_p)  # [hidden]

                if DspoModule._inject_log_count < 5:
                    k_idx = torch.argmax(probs[i]).item()
                    hard_token_id = slot_tok_ids_p[k_idx]
                    hard_embed = embedding_layer(hard_token_id.unsqueeze(0)).squeeze(0).to(soft_embed.dtype)
                    max_diff = (soft_embed - hard_embed).abs().max().item()
                    mean_diff = (soft_embed - hard_embed).abs().mean().item()
                    logging.info(
                        "D-SPO: Slot %d pos %d — injecting at %d positions. "
                        "Soft Embed Stats: mean=%.4f, std=%.4f | "
                        "Hard token id=%d | soft==hard? max_diff=%.2e mean_diff=%.2e",
                        i, p, mask.sum().item(),
                        soft_embed.mean().item(), soft_embed.std().item(),
                        hard_token_id.item(), max_diff, mean_diff,
                    )
                    DspoModule._inject_log_count += 1

                modified_embeds[mask] = soft_embed.to(modified_embeds.dtype)

        return modified_embeds


def setup_dspo_tokenizer(tokenizer, config):
    """Adds slot placeholder tokens to the tokenizer if D-SPO is enabled."""
    if config.diff_symbol_config.enabled:
        num_slots = config.diff_symbol_config.num_slots
        token_size = config.diff_symbol_config.symbol_token_size
        slot_tokens = [
            f"<slot_{i}_{p}>"
            for i in range(num_slots)
            for p in range(token_size)
        ]
        tokenizer.add_special_tokens({"additional_special_tokens": slot_tokens})
        import logging
        logging.info(
            "D-SPO: Added %d slot tokens to tokenizer (%d slots × %d positions)",
            len(slot_tokens), num_slots, token_size,
        )
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
