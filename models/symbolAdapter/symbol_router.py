"""
Differentiable Symbol Router for D-SPO.
Each slot has a private non-overlapping vocabulary of K tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SymbolRouter(nn.Module):
    """
    Slot Matrix where every slot owns a private set of K candidate tokens.
    Non-overlapping private vocabs prevent slots from converging on the same token.
    preferences[i] has shape [K] — softmax is over K, never over full vocab.
    """

    def __init__(
        self,
        num_slots: int,
        slot_vocab_size: int,
        slot_vocab_indices: List,  # [num_slots, K, token_size] — token IDs per symbol position
        initial_tau: float = 1.0,
        tau_min: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_vocab_size = slot_vocab_size

        # [num_slots, K, token_size] — each slot's private symbol token IDs, not learnable
        self.register_buffer(
            "slot_vocab_indices",
            torch.tensor(slot_vocab_indices, dtype=torch.long),
        )

        # [num_slots, K] — learned weights; zero init → uniform Gumbel start
        self.preferences = nn.Parameter(torch.zeros(num_slots, slot_vocab_size))

        self.tau = initial_tau
        self.tau_min = tau_min

    def get_slot_mappings(
        self,
        slot_indices: List[int],
        hard: bool = True,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each requested slot, sample one token from its private K-token vocab.

        Args:
            deterministic: if True, use argmax(preferences) instead of gumbel sampling.
                Probs use a straight-through softmax estimator so gradients still flow.
                This gives stable targets across batches — the selected token only
                changes when the router actually learns to prefer a different token.
                If False (default), gumbel noise is added each call (original behaviour).

        Returns:
            vocab_indices: [S, token_size] — chosen token IDs per position for each slot
            probs:         [S, K] — weights used for soft embedding (grad-connected)
        """
        slot_prefs = self.preferences[slot_indices]                           # [S, K]

        if deterministic:
            # Straight-through: hard argmax forward, softmax gradient backward
            discrete_indices = torch.argmax(slot_prefs, dim=-1)              # [S]
            soft_probs = F.softmax(slot_prefs / self.tau, dim=-1)            # [S, K]
            hard_one_hot = F.one_hot(discrete_indices, num_classes=slot_prefs.size(-1)).to(soft_probs.dtype)
            probs = hard_one_hot + (soft_probs - soft_probs.detach())        # straight-through
        else:
            probs = F.gumbel_softmax(slot_prefs, tau=self.tau, hard=hard, dim=-1)  # [S, K]
            discrete_indices = torch.argmax(probs, dim=-1)                   # [S]

        # discrete_indices[i] indexes into slot_vocab_indices[slot_indices[i], :, :]
        slot_vocab = self.slot_vocab_indices[slot_indices]                    # [S, K, token_size]
        vocab_indices = slot_vocab[
            torch.arange(len(slot_indices), device=slot_vocab.device),
            discrete_indices,
        ]  # [S, token_size] — token IDs for each position of the chosen symbol

        return vocab_indices, probs

    def get_confidence_scores(self) -> torch.Tensor:
        """
        Per-slot confidence: max softmax prob over the slot's K candidates.
        Higher = router has converged on a single token for this slot.
        Shape: [num_slots]
        """
        probs = F.softmax(self.preferences, dim=-1)
        return probs.max(dim=-1).values

    def update_tau(self, anneal_rate: float):
        self.tau = max(self.tau_min, self.tau * (1.0 - anneal_rate))

    def get_safety_scores(self) -> torch.Tensor:
        """Mean entropy across all slots (lower = more peaked)."""
        probs = F.softmax(self.preferences, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()
