"""OpenVLA-style action tokenizer: continuous actions → 256 bins per dimension."""

import torch


class ActionTokenizer:
    """Discretize continuous 7-DoF actions into 256 bins per dimension.

    MimicGen actions are already in [-1, 1] range.
    Bin width = 2/256 ≈ 0.0078.

    Args:
        n_bins: Number of discrete bins per dimension.
        min_val: Minimum action value.
        max_val: Maximum action value.
    """

    def __init__(self, n_bins: int = 256, min_val: float = -1.0, max_val: float = 1.0):
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode continuous actions to discrete bin indices.

        Args:
            actions: (B, T, 7) float tensor in [min_val, max_val].

        Returns:
            (B, T, 7) long tensor with values in [0, n_bins-1].
        """
        norm = (actions - self.min_val) / (self.max_val - self.min_val)
        return (norm * (self.n_bins - 1)).clamp(0, self.n_bins - 1).long()

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete bin indices to continuous actions.

        Args:
            tokens: (B, T, 7) long tensor with values in [0, n_bins-1].

        Returns:
            (B, T, 7) float tensor in [min_val, max_val].
        """
        return (
            (tokens.float() + 0.5) / self.n_bins * (self.max_val - self.min_val)
            + self.min_val
        )
