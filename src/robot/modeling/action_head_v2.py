"""View-agnostic action head for DA3 action tokens.

Supports mean pooling (default) or concat across views.
Mean pooling allows variable number of views at inference time.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPResNetBlock(nn.Module):
    """LayerNorm -> Linear -> ReLU residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.linear(self.norm(x)))


class ActionHeadV2(nn.Module):
    """Predict continuous actions from per-view DA3 action tokens.

    Args:
        input_dim: Dimension of each action token (e.g. 1536).
        n_views: Number of camera views (used only for concat mode).
        hidden_dim: Hidden dimension of MLP.
        n_dims: Action space dimensionality (e.g. 7 for 6DoF + gripper).
        chunk_size: Number of sub-actions per timestep.
        num_blocks: Number of MLP-ResNet blocks.
        pool_mode: 'mean' for view-agnostic mean pooling,
                   'concat' for fixed-view concatenation (legacy).
        chunk_position_encoding: 'none' keeps the legacy single-vector output
                   projection; 'learned' gives each sub-action a learned order
                   embedding before the MLP blocks.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        n_views: int = 2,
        hidden_dim: int = 1536,
        n_dims: int = 7,
        chunk_size: int = 1,
        num_blocks: int = 2,
        pool_mode: str = "mean",
        chunk_position_encoding: str = "none",
    ):
        super().__init__()
        self.n_views = n_views
        self.n_dims = n_dims
        self.chunk_size = chunk_size
        self.pool_mode = pool_mode
        self.chunk_position_encoding = str(chunk_position_encoding or "none").lower()
        if self.chunk_position_encoding not in ("none", "learned"):
            raise ValueError(
                "chunk_position_encoding must be 'none' or 'learned', "
                f"got {chunk_position_encoding!r}."
            )

        if pool_mode == "concat":
            proj_in = input_dim * n_views
        else:
            proj_in = input_dim

        self.input_proj = nn.Linear(proj_in, hidden_dim)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        if self.chunk_position_encoding == "learned":
            self.chunk_pos_embed = nn.Parameter(torch.empty(chunk_size, hidden_dim))
            nn.init.normal_(self.chunk_pos_embed, mean=0.0, std=0.02)
            self.output = nn.Linear(hidden_dim, n_dims)
        else:
            self.register_parameter("chunk_pos_embed", None)
            self.output = nn.Linear(hidden_dim, n_dims * chunk_size)

    def _masked_view_mean(
        self,
        action_tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if view_mask is None:
            return action_tokens.mean(dim=2)
        if view_mask.shape != action_tokens.shape[:3]:
            raise ValueError(
                "view_mask must match action token leading shape "
                f"{tuple(action_tokens.shape[:3])}, got {tuple(view_mask.shape)}."
            )
        mask = view_mask.to(device=action_tokens.device, dtype=action_tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=2).clamp_min(1.0)
        return (action_tokens * mask).sum(dim=2) / denom

    def _reshape_input(
        self,
        action_tokens: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Reshape action tokens to (B, T, D) depending on pool_mode.

        Input: (B, S, D) where S = T * n_views, or (B, T, V, D).
        Output: (B, T, proj_in_dim).
        """
        if action_tokens.ndim == 4:
            # (B, T, V, D)
            batch_size, timesteps, n_views, dim = action_tokens.shape
            if self.pool_mode == "concat":
                if view_mask is not None and not bool(view_mask.to(dtype=torch.bool).all()):
                    raise ValueError("Masked variable views require pool_mode='mean'.")
                return action_tokens.reshape(batch_size, timesteps, n_views * dim), batch_size, timesteps
            else:
                return self._masked_view_mean(action_tokens, view_mask), batch_size, timesteps

        if action_tokens.ndim == 3:
            batch_size, seq_len, dim = action_tokens.shape
            if seq_len % self.n_views != 0:
                raise ValueError(
                    f"Sequence length {seq_len} not divisible by n_views={self.n_views}."
                )
            timesteps = seq_len // self.n_views
            tokens_4d = action_tokens.reshape(batch_size, timesteps, self.n_views, dim)
            if view_mask is not None and view_mask.shape != (batch_size, timesteps, self.n_views):
                raise ValueError(
                    "view_mask must be shaped "
                    f"{(batch_size, timesteps, self.n_views)} for flattened action tokens, "
                    f"got {tuple(view_mask.shape)}."
                )
            if self.pool_mode == "concat":
                if view_mask is not None and not bool(view_mask.to(dtype=torch.bool).all()):
                    raise ValueError("Masked variable views require pool_mode='mean'.")
                return tokens_4d.reshape(batch_size, timesteps, self.n_views * dim), batch_size, timesteps
            else:
                return self._masked_view_mean(tokens_4d, view_mask), batch_size, timesteps

        raise ValueError(f"Expected 3 or 4 dims, got {action_tokens.shape}.")

    def forward(self, action_tokens: torch.Tensor, view_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, batch_size, timesteps = self._reshape_input(action_tokens, view_mask=view_mask)
        proj_dtype = self.input_proj.weight.dtype
        if x.dtype != proj_dtype:
            x = x.to(proj_dtype)
        x = F.relu(self.input_proj(x))
        if self.chunk_position_encoding == "learned":
            chunk_pos = self.chunk_pos_embed.to(dtype=x.dtype)
            x = x.unsqueeze(2) + chunk_pos.view(1, 1, self.chunk_size, -1)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.output(x)
        if self.chunk_size == 1:
            return x.reshape(batch_size, timesteps, self.n_dims)
        if self.chunk_position_encoding == "learned":
            return x.reshape(batch_size, timesteps, self.chunk_size, self.n_dims)
        return x.reshape(batch_size, timesteps, self.chunk_size, self.n_dims)
