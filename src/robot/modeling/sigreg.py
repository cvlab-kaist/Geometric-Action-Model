"""Sketched Isotropic Gaussian Regularizer (SIGReg).

Port from LeJEPA / LeWorldModel (arXiv:2511.08544, arXiv:2603.19312,
github.com/lucas-maes/le-wm, MIT license).

Key properties (matched to LeWM's `module.py:11-37`):
- Integration knots `t = linspace(0, max_knot, knots)` on the positive axis.
- Gaussian envelope `exp(-t^2 / 2)` : the TARGET CF's modulus.
- Trapezoidal weights over the integration.
- Random unit directions are re-drawn every forward : Cramer-Wold randomization.
- No whitening inside the loss (the upstream BN-MLP projector does it).
- Scale statistic by sample count (makes the numeric scale O(1) regardless of batch).
- Outer loss coefficient (commonly λ=0.09 in LeWM) is applied at the call site.

Usage:
    sigreg = SIGReg(d_model=1024, n_projections=1024, knots=17, max_knot=3.0)
    loss = sigreg(z)   # z: (B, d_model) -> scalar
"""

from typing import Optional

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """LeWM-compatible SIGReg loss.

    Args:
        d_model: embedding dimensionality (last axis of `z`).
        n_projections: number of random unit directions (M). LeWM default 1024.
        knots: integration-grid resolution. LeWM default 17.
        max_knot: upper bound of the integration grid (t in [0, max_knot]).
            LeWM uses 3.0 (beyond 3 standard deviations the Gaussian
            envelope is numerically zero).
        redraw: if True (recommended), sample new random directions every
            forward. If False, use a fixed reproducible buffer.
    """

    def __init__(
        self,
        d_model: int,
        n_projections: int = 1024,
        knots: int = 17,
        max_knot: float = 3.0,
        redraw: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_projections = int(n_projections)
        self.knots = int(knots)
        self.max_knot = float(max_knot)
        self.redraw = bool(redraw)

        # Integration grid and trapezoidal weights times target-CF envelope.
        t = torch.linspace(0.0, self.max_knot, self.knots)
        # Trapezoidal weights: endpoints get dt/2, interior dt.
        dt = t[1] - t[0] if self.knots > 1 else torch.tensor(1.0)
        trap = torch.full((self.knots,), float(dt))
        trap[0] = trap[-1] = 0.5 * float(dt)
        envelope = (-0.5 * t * t).exp()
        weights = trap * envelope
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

        if not self.redraw:
            directions = torch.randn(self.n_projections, self.d_model)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.register_buffer("fixed_directions", directions, persistent=False)
        else:
            self.fixed_directions = None  # type: ignore[assignment]

    def _sample_directions(self, device, dtype) -> torch.Tensor:
        if self.redraw:
            d = torch.randn(self.n_projections, self.d_model, device=device, dtype=dtype)
            return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.fixed_directions.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            z: (N, d_model) OR (B, N, d_model); leading dims are flattened.
                **The caller supplies z from the BN-MLP projector**.
            mask: optional (N,) bool tensor selecting rows (True = keep).

        Returns:
            Scalar SIGReg loss in [0, ~O(max_knot)].
        """
        if z.dim() < 2 or z.shape[-1] != self.d_model:
            raise ValueError(
                f"SIGReg expects last dim = {self.d_model}, got shape {tuple(z.shape)}."
            )

        # Trig kernels are numerically fragile in bf16 once projected phases grow.
        # Keep the regularizer in fp32; the caller applies the outer loss weight.
        flat = z.reshape(-1, self.d_model).float()
        if mask is not None:
            flat = flat[mask.reshape(-1).to(flat.device)]
        n = flat.shape[0]
        if n < 2:
            return flat.new_zeros(())

        directions = self._sample_directions(device=flat.device, dtype=torch.float32)

        # Project: (M, D) @ (D, N) -> (M, N)
        proj = directions @ flat.t()

        # Empirical CF phi_N(t) = (1/N) Σ exp(i t x_i).
        # phase: (M, N, K)
        t = self.t.to(device=flat.device, dtype=flat.dtype)     # (K,)
        w = self.weights.to(device=flat.device, dtype=flat.dtype)  # (K,)

        phase = proj.unsqueeze(-1) * t.view(1, 1, -1)            # (M, N, K)
        cos_part = phase.cos().mean(dim=1)                       # (M, K)
        sin_part = phase.sin().mean(dim=1)                       # (M, K)

        # Target CF phi_0(t) = exp(-t^2 / 2), real.
        phi0 = (-0.5 * t * t).exp()                              # (K,)

        diff_sq = (cos_part - phi0.view(1, -1)) ** 2 + sin_part ** 2  # (M, K)

        # Weighted integral over t (trapezoidal * envelope), then mean across directions.
        # LeWM additionally scales by sample count to make the loss roughly scale-invariant
        # with batch size.
        weighted = (diff_sq * w.view(1, -1)).sum(dim=-1)          # (M,)
        stat = weighted.mean() * float(n)
        return stat
