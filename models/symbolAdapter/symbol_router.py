"""
Differentiable Symbol Router for D-SPO.
Implements the Slot Matrix and Gumbel-Softmax routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SymbolRouter(nn.Module):
    """
    Manages the differentiable 'Slot Matrix' (Π) and performs Gumbel-Softmax routing.
    """

    def __init__(
        self,
        num_slots: int,
        pool_size: int,
        symbol_pool_indices: List[int],
        initial_tau: float = 1.0,
        tau_min: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.pool_size = pool_size
        self.symbol_pool_indices = torch.tensor(symbol_pool_indices, dtype=torch.long)
        
        # Preference Matrix (Π) - initialized randomly
        # Shape: [M, N] where M=Slots, N=Pool Size
        self.preferences = nn.Parameter(torch.randn(num_slots, pool_size))
        
        self.tau = initial_tau
        self.tau_min = tau_min

    def get_slot_mappings(
        self, 
        slot_indices: List[int], 
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples symbols for the requested slots.
        """
        # Select preference vectors for the requested slots [K, N]
        slot_prefs = self.preferences[slot_indices]
        
        # Gumbel-Softmax sampling [K, N]
        probs = F.gumbel_softmax(slot_prefs, tau=self.tau, hard=hard, dim=-1)
        
        discrete_indices = torch.argmax(probs, dim=-1)
        vocab_indices = self.symbol_pool_indices.to(probs.device)[discrete_indices]
        
        # Log selection for debugging (only if in training/verbose mode)
        import logging
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            for i, slot_idx in enumerate(slot_indices):
                logging.debug(f"Router: Slot {slot_idx} selected Token ID {vocab_indices[i].item()}")
        
        return vocab_indices, probs

    def update_tau(self, anneal_rate: float):
        """Anneal the temperature."""
        self.tau = max(self.tau_min, self.tau - anneal_rate)

    def get_safety_scores(self) -> torch.Tensor:
        """
        Returns a measure of how 'peaked' the distributions are.
        Lower entropy = more confident/learned mappings.
        """
        probs = F.softmax(self.preferences, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean()
