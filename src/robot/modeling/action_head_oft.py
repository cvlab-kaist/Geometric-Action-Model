"""OpenVLA-OFT L1 regression action head for DA3 Stage 1 fine-tuning.

Ports ``MLPResNetBlock``, ``MLPResNet`` and ``L1RegressionActionHead`` from
openvla-oft (https://github.com/moojink/openvla-oft,
``prismatic/models/action_heads.py``; MIT licensed, Copyright (c) 2025 Moo Jin
Kim, Chelsea Finn, Percy Liang). The original head expects per-action-dim
hidden states from a VLM of shape ``(B, chunk_len * action_dim, hidden_dim)``.
DA3 emits one action hidden state per view (``(B, T, V, D)``), so we bridge
with a learnable per-action-dim embedding that expands each pooled timestep
token into ``action_dim`` per-dim tokens before the ported head runs.

This mirrors the interface of ``ActionHeadV2.forward`` so the two heads are
drop-in swappable under ``action_head.type`` in config.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Verbatim port of openvla-oft MLPResNet building blocks.
# Source: openvla-oft/prismatic/models/action_heads.py  (MIT license).
# ---------------------------------------------------------------------------


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection (openvla-oft)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks (openvla-oft)."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """openvla-oft L1 regression head.

    Input  : ``(B, chunk_len * action_dim, hidden_dim)``
    Reshape: ``(B, chunk_len, action_dim * hidden_dim)``
    Output : ``(B, chunk_len, action_dim)`` predicted actions (L1 loss upstream).

    ``chunk_len`` is inferred at call time from the input shape instead of
    being hard-coded via a module-level constant, so the same head works
    across configs with different timestep counts (openvla-oft originally
    read NUM_ACTIONS_CHUNK from ``prismatic.vla.constants``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = actions_hidden_states.shape
        if seq_len % self.action_dim != 0:
            raise ValueError(
                f"seq_len={seq_len} not divisible by action_dim={self.action_dim}."
            )
        chunk_len = seq_len // self.action_dim
        rearranged = actions_hidden_states.reshape(batch_size, chunk_len, -1)
        return self.model(rearranged)


# ---------------------------------------------------------------------------
# DA3-facing adapter that mirrors ActionHeadV2's interface.
# ---------------------------------------------------------------------------


class OFTL1RegressionHead(nn.Module):
    """DA3-compatible wrapper around openvla-oft's ``L1RegressionActionHead``.

    Drop-in replacement for ``ActionHeadV2``. Accepts per-view DA3 action
    tokens in either ``(B, T, V, D)`` or ``(B, T * V, D)`` layout and returns
    predicted delta-actions in the same ``(B, T, n_dims)`` (or
    ``(B, T, chunk_size, n_dims)``) shape as the existing head.

    Because the upstream openvla-oft head was designed for per-action-dim
    hidden states emitted by a VLM, and DA3 only emits per-view action tokens,
    we expand each pooled timestep hidden state to ``n_dims`` per-dim slots via
    a learnable ``dim_embed`` bias (one vector per action dim). This matches
    the paper's "regression transformer over learnable action queries" idea
    while keeping the head's internal computation verbatim to openvla-oft.

    Args:
        input_dim: Dim of each DA3 action token (1536 for DA3-Giant).
        hidden_dim: MLP-ResNet hidden size.
        n_views: Cameras per timestep (for ``(B, S, D)`` → ``(B, T, V, D)`` reshape).
        n_dims: Action space dim (7 for 6DoF + gripper).
        chunk_size: Sub-actions per timestep (must be 1 in this head).
        num_blocks: Depth of the ported MLPResNet (openvla-oft default: 2).
        pool_mode: "mean" (view-agnostic) or "concat" (fixed-view).
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 1536,
        n_views: int = 2,
        n_dims: int = 7,
        chunk_size: int = 1,
        num_blocks: int = 2,
        pool_mode: str = "mean",
    ):
        super().__init__()
        if chunk_size != 1:
            raise NotImplementedError(
                "OFTL1RegressionHead currently supports chunk_size=1 only "
                "(DA3 main configs use chunk_size=1). Please raise a spec to "
                "extend this if a chunked run is actually needed."
            )
        self.n_views = int(n_views)
        self.n_dims = int(n_dims)
        self.chunk_size = int(chunk_size)
        self.pool_mode = str(pool_mode)

        if self.pool_mode == "concat":
            proj_in = input_dim * n_views
        else:
            proj_in = input_dim
        # Optional projection from raw DA3 hidden to head hidden size.
        if proj_in != hidden_dim:
            self.input_proj: nn.Module = nn.Linear(proj_in, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Per-action-dim learnable bias. Added to the pooled per-timestep
        # hidden so openvla-oft's head sees N=T*action_dim tokens.
        self.dim_embed = nn.Parameter(torch.randn(self.n_dims, hidden_dim) * 0.02)

        # Verbatim openvla-oft head. chunk_len is inferred from input shape.
        self.head = L1RegressionActionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_dim=self.n_dims,
            num_blocks=int(num_blocks),
        )

    def _pool_views(self, action_tokens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if action_tokens.ndim == 4:
            batch_size, timesteps, n_views, _ = action_tokens.shape
            if self.pool_mode == "concat":
                pooled = action_tokens.reshape(batch_size, timesteps, -1)
            else:
                pooled = action_tokens.mean(dim=2)
            return pooled, batch_size, timesteps
        if action_tokens.ndim == 3:
            batch_size, seq_len, dim = action_tokens.shape
            if seq_len % self.n_views != 0:
                raise ValueError(
                    f"Sequence length {seq_len} not divisible by n_views={self.n_views}."
                )
            timesteps = seq_len // self.n_views
            tokens_4d = action_tokens.reshape(batch_size, timesteps, self.n_views, dim)
            if self.pool_mode == "concat":
                pooled = tokens_4d.reshape(batch_size, timesteps, self.n_views * dim)
            else:
                pooled = tokens_4d.mean(dim=2)
            return pooled, batch_size, timesteps
        raise ValueError(f"Expected 3 or 4 dims, got {action_tokens.shape}.")

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        pooled, batch_size, timesteps = self._pool_views(action_tokens)
        # Project to head hidden size if needed.
        pooled = self.input_proj(pooled)  # (B, T, hidden)
        # Expand per-action-dim: (B, T, 1, H) + (1, 1, n_dims, H) → (B, T, n_dims, H)
        dim_embed = self.dim_embed.to(dtype=pooled.dtype)
        per_dim = pooled.unsqueeze(2) + dim_embed.unsqueeze(0).unsqueeze(0)
        per_dim_flat = per_dim.reshape(batch_size, timesteps * self.n_dims, -1)
        actions = self.head.predict_action(per_dim_flat)  # (B, T, n_dims)
        return actions
