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
        slot_vocab_indices: List[List[int]],  # [num_slots][K] — token IDs, non-overlapping
        initial_tau: float = 1.0,
        tau_min: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_vocab_size = slot_vocab_size

        # [num_slots, K] — each slot's private token IDs, not learnable
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each requested slot, sample one token from its private K-token vocab.

        Returns:
            vocab_indices: [S] — the chosen real vocab token ID for each slot
            probs:         [S, K] — Gumbel-Softmax weights (used for soft embedding)
        """
        slot_prefs = self.preferences[slot_indices]                           # [S, K]
        probs = F.gumbel_softmax(slot_prefs, tau=self.tau, hard=hard, dim=-1) # [S, K]
        discrete_indices = torch.argmax(probs, dim=-1)                        # [S]

        # discrete_indices[i] is an index into slot_vocab_indices[slot_indices[i]]
        slot_vocab = self.slot_vocab_indices[slot_indices]                    # [S, K]
        vocab_indices = slot_vocab[
            torch.arange(len(slot_indices), device=slot_vocab.device),
            discrete_indices,
        ]  # [S] — actual token IDs in the model's vocabulary

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
        self.tau = max(self.tau_min, self.tau - anneal_rate)

    def get_safety_scores(self) -> torch.Tensor:
        """Mean entropy across all slots (lower = more peaked)."""
        probs = F.softmax(self.preferences, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()
