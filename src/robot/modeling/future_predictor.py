"""GAMFuturePredictor : modernized dense block-autoregressive predictor.

Revision 2026-04-21 (commit: feature/gam-arch-modernize)
-------------------------------------------------------------
Drop-in rewrite of the predictor used by the unified Stage 1 training path.

Architectural changes vs the previous implementation:

- LayerNorm + AdaLN-Zero removed. Blocks are now plain pre-norm transformers
  with RMSNorm + SwiGLU + LayerScale (ViT-22B / DINOv3 / DiT-3 convention).
  AdaLN had nothing to modulate once proprio was already present as an
  in-sequence token (Path 1) : the mean-collapsed `cond_proj` path (Path 2)
  was redundant and temporally lossy, so it was dropped entirely.
- Q/K RMSNorm inside attention ("QK-norm"). Eliminates the bf16 logit
  explosion that the old code papered over with `attention_fp32=True`
  autocast-disable. All attention now runs in the surrounding autocast dtype.
- Learned absolute position embeddings (`t_embed`, `v_embed`, `slot_type_embed`,
  ...) replaced by axial RoPE. Visual patches get factorized 4D RoPE on
  (t, v, y, x); proprio / prev-action tokens get 1D RoPE on the timestep axis
  only. DA3-Giant itself is a RoPE backbone, so predictor position signal now
  lives in the same coordinate system. `max_timesteps` is no longer a hard
  OOB limit.
- Block-causal `L×L` dense boolean mask replaced by `flex_attention`
  BlockMask built from a small `mask_mod` closure. Fallback: plain SDPA with
  a lazily-built mask if flex_attention is unavailable. Memory for the mask
  itself drops from O(L²) to effectively free.
- Output heads replaced from bare `Linear d_model → 1536` to `RMSNorm + Linear`
  (pre-norm head) for every (visual / action / proprio) target.
- The legacy `FuturePredictor` v1/v2 (Level-0 with CLIP prepend + dense
  future-slot expansion) is gone. Only `GAMFuturePredictor` remains.

Checkpoints from the old predictor are **not loadable**. Parameter layout is
different everywhere (no modulation MLPs, no abs-position tables, different
attention submodule surface). Running gam experiments will not
resume across this commit.

Contract with the unified training path (`src/robot/losses/unified_loss.py`,
`src/train_robot.py`):
  - forward signature unchanged at the kwargs level.
  - output dict keys unchanged: `predicted_next_visual_tokens`,
    `predicted_next_proprio`, `predicted_action_tokens`,
    `encoded_prev_action_tokens`, `encoded_proprio_tokens`, `sigreg_loss`.
  - `self.language_len` attribute preserved (train_robot.py reads it to pad
    CLIP token output to matching length).
  - `view_valid_mask` is optional and parameter-free. It masks padded camera
    slots without changing checkpoint compatibility.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# -----------------------------------------------------------------------------
# flex_attention availability
# -----------------------------------------------------------------------------


def _patch_triton_compiled_kernel_hooks() -> None:
    """Backfill Triton hooks expected by the CSCS PyTorch Inductor build."""

    try:
        from triton.compiler.compiler import CompiledKernel
    except Exception:
        return

    for hook_name in ("launch_enter_hook", "launch_exit_hook"):
        if not hasattr(CompiledKernel, hook_name):
            setattr(CompiledKernel, hook_name, None)


try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention_raw,
        create_block_mask as _create_block_mask,
    )
    _patch_triton_compiled_kernel_hooks()
    # flex_attention must itself be compiled to use the fused BlockMask kernel.
    # A model-level torch.compile can graph-break before this call and silently
    # fall back to the dense-score eager path, which is both slow and memory hot.
    _flex_attention = torch.compile(_flex_attention_raw)
    _HAS_FLEX = True
except Exception:
    _flex_attention = None
    _create_block_mask = None
    _HAS_FLEX = False


# -----------------------------------------------------------------------------
# Norm layers
# -----------------------------------------------------------------------------


def _make_rmsnorm(d_model: int, eps: float = 1e-6) -> nn.Module:
    """Use torch's native RMSNorm when available (≥ 2.4); else tiny fallback."""
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model, eps=eps)

    class _RMSNormFallback(nn.Module):
        def __init__(self, d: int, eps_: float):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps_

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_dtype = x.dtype
            x32 = x.float()
            rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return (x32 * rms).to(dtype=orig_dtype) * self.weight

    return _RMSNormFallback(d_model, eps)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU FFN (Llama / Gemma / DiT-3)."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class LayerScale(nn.Module):
    """Zero-initialized per-channel residual gate (ViT-22B / DINOv3 / DiT-3).

    Replaces AdaLN-Zero's "gate = 0 at init" property without a modulation MLP.
    The residual sublayer contributes nothing at step 0 and ramps in as the
    parameter learns away from zero.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class LanguageFiLM(nn.Module):
    """Language-conditioned feature-wise modulation for visual predictor tokens.

    OpenVLA-OFT+ modulates visual hidden units with scale/shift vectors derived
    from the average language embedding. We mirror that contract at the
    predictor boundary: one pooled language vector produces one global
    per-hidden-unit affine transform, shared across all visual tokens in a
    block. Zero init makes the module an identity map at step 0.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        film_cond: Optional[torch.Tensor],
        visual_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if film_cond is None:
            return x
        mod = self.proj(film_cond.to(device=x.device, dtype=self.proj.weight.dtype)).to(dtype=x.dtype)
        scale, shift = mod.chunk(2, dim=-1)
        modulated = x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        if visual_token_mask is None:
            return modulated
        if visual_token_mask.shape[0] != x.shape[1]:
            raise ValueError(
                f"visual_token_mask length {visual_token_mask.shape[0]} "
                f"mismatches sequence length {x.shape[1]}."
            )
        mask = visual_token_mask.to(device=x.device, dtype=torch.bool).view(1, -1, 1)
        return torch.where(mask, modulated, x)


# -----------------------------------------------------------------------------
# RoPE : 4D axial for visual patches, 1D timestep for single-slot tokens
# -----------------------------------------------------------------------------


def _split_rope_dims(head_dim: int) -> Tuple[int, int, int, int]:
    """LTA-style (t, v, y, x) dimension split for 4D axial RoPE.

    V and T axes get small, low-frequency allocations. Y/X split the rest.
    All entries even (RoPE requires sin/cos pairs).
    """
    dim_v = max(2, head_dim // 24)
    dim_t = max(4, head_dim // 12)
    # Ensure even
    if dim_v % 2:
        dim_v += 1
    if dim_t % 2:
        dim_t += 1
    dim_spatial = head_dim - dim_v - dim_t
    dim_y = dim_spatial // 2
    if dim_y % 2:
        dim_y -= 1
    dim_x = head_dim - dim_v - dim_t - dim_y
    assert dim_x % 2 == 0, (dim_v, dim_t, dim_y, dim_x, head_dim)
    return dim_v, dim_t, dim_y, dim_x


class ShallowRoPE(nn.Module):
    """Axial RoPE for the gam AR predictor.

    Coordinates per token: (t, v, y, x).
      - Visual patches in view v: t=step, v=cam_id, y=patch_y, x=patch_x
      - Visual CLS / register in view v: t=step, v=cam_id, y=0, x=0
      - Proprio token at step t: t=step, v=V (reserved), y=0, x=0
      - Prev-action token at step t: t=step, v=V+1, y=0, x=0

    The `v` axis covers camera ids plus two reserved "virtual views" so that
    the proprio and action-history tokens have distinct v-axis RoPE offsets
    from the real cameras. Same (t, v, y, x) across tokens is allowed : RoPE
    only encodes *relative* position; identity is still carried by the
    token's learned transform path.
    """

    def __init__(
        self,
        head_dim: int,
        grid_h: int,
        grid_w: int,
        theta_base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.dim_split = _split_rope_dims(head_dim)
        # Per-axis base multiplier (LTA): low freq on v/t, high on y/x.
        base_mults = (10.0, 5.0, 1.0, 1.0)
        for axis, (dim, base_mult) in enumerate(zip(self.dim_split, base_mults)):
            inv_freq = 1.0 / (
                (theta_base * base_mult)
                ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            self.register_buffer(f"inv_freq_{axis}", inv_freq, persistent=False)
        # Cache by (tuple of position signature, device, dtype)
        self._cache: dict = {}

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _axis_cos_sin(
        self,
        axis: int,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"inv_freq_{axis}").to(device=positions.device, dtype=dtype)
        freqs = torch.outer(positions.to(dtype=dtype), inv_freq)
        return freqs.cos(), freqs.sin()

    def build_positions(
        self,
        H: int,
        V: int,
        num_patches: int,
        num_prefix_visual: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return integer positions for every token in the sequence.

        Sequence order per timestep block (same as predictor.forward):
          [ view_0(prefix + patches) ... view_{V-1}(prefix + patches) |
            proprio_slot | prev_action_slot ]

        Args:
            num_prefix_visual: number of non-spatial prefix tokens per view
                (CLS + registers). They get (t, v, 0, 0).
            num_patches: spatial patches per view = grid_h * grid_w.

        Returns:
            IntTensor (L, 4) with columns (t, v, y, x).
        """
        tokens_per_view = num_prefix_visual + num_patches
        tokens_per_step = V * tokens_per_view + 2  # +proprio +prev_action
        L = H * tokens_per_step

        pos = torch.empty(L, 4, device=device, dtype=torch.long)
        # Build per-step block
        # patch y/x grid
        ys = torch.arange(self.grid_h, device=device)
        xs = torch.arange(self.grid_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_y = grid_y.flatten()   # (num_patches,)
        patch_x = grid_x.flatten()   # (num_patches,)
        assert patch_y.numel() == num_patches, (patch_y.numel(), num_patches)

        idx = 0
        for t in range(H):
            for v in range(V):
                # Prefix visual: (t, v, 0, 0)
                for _ in range(num_prefix_visual):
                    pos[idx, 0] = t
                    pos[idx, 1] = v
                    pos[idx, 2] = 0
                    pos[idx, 3] = 0
                    idx += 1
                # Patches
                pos[idx : idx + num_patches, 0] = t
                pos[idx : idx + num_patches, 1] = v
                pos[idx : idx + num_patches, 2] = patch_y
                pos[idx : idx + num_patches, 3] = patch_x
                idx += num_patches
            # Proprio slot: (t, V, 0, 0)
            pos[idx, 0] = t
            pos[idx, 1] = V
            pos[idx, 2] = 0
            pos[idx, 3] = 0
            idx += 1
            # Prev-action slot: (t, V+1, 0, 0)
            pos[idx, 0] = t
            pos[idx, 1] = V + 1
            pos[idx, 2] = 0
            pos[idx, 3] = 0
            idx += 1
        assert idx == L
        return pos

    def _get_cos_sin(
        self,
        positions: torch.Tensor,    # (L, 4) int64
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build per-token cos/sin tables by concatenating per-axis tables.

        Cache key uses the full positions tensor's data_ptr + shape + device
        + dtype. Since `build_positions` always returns the same tensor
        instance for a given (H, V, grid) call, the data_ptr changes only
        when the caller rebuilds positions : which is exactly when we want
        a cache miss.
        """
        key = (positions.data_ptr(), positions.shape[0], positions.device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cos_parts, sin_parts = [], []
        for axis in range(4):
            cos_axis, sin_axis = self._axis_cos_sin(
                axis, positions[:, axis].to(torch.float32), dtype
            )
            cos_parts.append(cos_axis)
            sin_parts.append(sin_axis)
        cos = torch.cat(cos_parts, dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, L, D//2)
        sin = torch.cat(sin_parts, dim=-1).unsqueeze(0).unsqueeze(0)
        self._cache[key] = (cos, sin)
        return cos, sin

    def apply_rope(
        self,
        q: torch.Tensor,        # (B, H, L, D_head)
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply axial RoPE. `positions` must match query's L."""
        cos, sin = self._get_cos_sin(positions, q.dtype)
        q_out = self._apply_rotary(q, cos, sin)
        k_out = self._apply_rotary(k, cos, sin)
        return q_out, k_out

    def apply(self, fn, *args, **kwargs):
        """Preserve nn.Module.apply for DeepSpeed's module traversal."""
        if callable(fn) and not args and not kwargs:
            return super().apply(fn)
        return self.apply_rope(fn, *args, **kwargs)


# -----------------------------------------------------------------------------
# Attention with QK-norm
# -----------------------------------------------------------------------------


def _make_block_causal_mask_mod(tokens_per_step: int) -> Callable:
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx // tokens_per_step) >= (kv_idx // tokens_per_step)
    return mask_mod


def _make_concat_lang_block_causal_mask_mod(
    tokens_per_step: int,
    lang_len: int,
) -> Callable:
    """Block-causal mask with a language prefix of length `lang_len`.

    Sequence layout: [lang(0..lang_len-1) | step_0 | step_1 | ... ].
    Lang queries attend only to lang keys (bidirectional inside the prefix);
    step queries attend to ALL lang keys + step keys with the usual
    block-causal ordering. This keeps the language representation pure (not
    polluted by per-step state) while making step tokens see the full
    instruction.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_is_lang = q_idx < lang_len
        kv_is_lang = kv_idx < lang_len
        # Lang query: attend only to lang keys.
        # Step query: attend to lang keys (always) or step keys with q_step >= kv_step.
        q_step = (q_idx - lang_len) // tokens_per_step
        kv_step = (kv_idx - lang_len) // tokens_per_step
        step_to_step = (~q_is_lang) & (~kv_is_lang) & (q_step >= kv_step)
        step_to_lang = (~q_is_lang) & kv_is_lang
        lang_to_lang = q_is_lang & kv_is_lang
        return step_to_step | step_to_lang | lang_to_lang
    return mask_mod


class QKNormAttention(nn.Module):
    """Multi-head self-attention with RMSNorm(q), RMSNorm(k) pre-softmax.

    QK-norm stabilizes bf16 attention at long sequences and removes the need
    for the `attention_fp32` autocast-disable workaround. Follows DiT-3 /
    SD3 / ViT-22B convention : learnable scale on Q and K after the dot-
    product projection, before SDPA.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim * num_heads != d_model:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Per-head RMSNorm on q and k (applied to the last dim = head_dim).
        self.q_norm = _make_rmsnorm(self.head_dim)
        self.k_norm = _make_rmsnorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[ShallowRoPE] = None,
        rope_positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        flex_block_mask=None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_new_kv: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Self-attention with optional KV cache for AR rollout.

        When `past_kv=(K_past, V_past)` is provided, `x` is treated as NEW
        tokens only; Q/K/V are computed on new tokens, then K/V are concatenated
        with cached (K_past, V_past) along the sequence dim. SDPA sees the full
        K/V with Q only for new tokens: complexity drops from O(L_total^2) to
        O(L_new × L_total). Because the new timestep block is the LATEST in
        block-causal order, it attends to all past + all within-block tokens,
        so `attn_mask`/`flex_block_mask` are ignored in cache mode.

        Shapes (cache mode): K_past, V_past = (B, num_heads, L_past, head_dim).
        `rope_positions` must describe only the NEW tokens with absolute t.
        """
        b, l, d = x.shape
        qkv = self.qkv_proj(x).reshape(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # QK-norm (on last dim = head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary : applied to NEW q,k only; past K already has RoPE applied.
        if rope is not None and rope_positions is not None:
            q, k = rope.apply_rope(q, k, rope_positions)

        # RMSNorm parameters are fp32, so Q/K can be promoted under bf16
        # autocast. SDPA/flex_attention requires matching Q/K/V dtypes.
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)

        if past_kv is not None:
            k_past, v_past = past_kv
            k_full = torch.cat([k_past, k], dim=2)
            v_full = torch.cat([v_past, v], dim=2)
            # Cache mode: new block attends to past + self fully : no mask.
            out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None)
            new_kv = (k_full, v_full)
        else:
            # Attention backend: flex if given, else SDPA
            if flex_block_mask is not None and _HAS_FLEX:
                out = _flex_attention(q, k, v, block_mask=flex_block_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            new_kv = (k, v) if return_new_kv else None

        out = out.transpose(1, 2).reshape(b, l, d)
        out = self.dropout(self.out_proj(out))
        if return_new_kv:
            return out, new_kv
        return out


class QKNormCrossAttention(nn.Module):
    """Cross-attention: query from sequence, K/V from text context.

    Keep-mask over text tokens (True = attend). Short KV length (≤ 77 for
    CLIP-L), so plain SDPA with an explicit mask is fine.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim * num_heads != d_model:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = _make_rmsnorm(self.head_dim)
        self.k_norm = _make_rmsnorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,                  # (B, Lq, D)
        context: torch.Tensor,            # (B, Lk, D)
        keep_mask: Optional[torch.Tensor] = None,   # (B, Lk) bool
    ) -> torch.Tensor:
        b, lq, d = x.shape
        lk = context.shape[1]
        q = self.q_proj(x).reshape(b, lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(b, lk, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
        attn_mask = None
        if keep_mask is not None:
            attn_mask = keep_mask[:, None, None, :].to(dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, lq, d)
        return self.dropout(self.out_proj(out))


# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------


class PlainBlock(nn.Module):
    """Pre-norm transformer block with QK-norm attention + LayerScale.

    Used by `GAMFuturePredictor(condition_mode="concat")`. Language
    enters as a prepended sub-sequence, so this block has no cross-attention
    submodule. KV cache is supported on the self-attention path so the AR
    `forward_incremental` rollout works the same way as in cross_attn mode.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = _make_rmsnorm(d_model)
        self.attn = QKNormAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = _make_rmsnorm(d_model)
        self.ffn = SwiGLU(d_model, int(d_model * ffn_ratio), dropout=dropout)
        self.ls1 = LayerScale(d_model)
        self.ls2 = LayerScale(d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[ShallowRoPE] = None,
        rope_positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        flex_block_mask=None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_new_kv: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if return_new_kv or past_kv is not None:
            attn_out, new_kv = self.attn(
                self.norm1(x),
                rope=rope,
                rope_positions=rope_positions,
                attn_mask=attn_mask,
                flex_block_mask=flex_block_mask,
                past_kv=past_kv,
                return_new_kv=True,
            )
        else:
            attn_out = self.attn(
                self.norm1(x),
                rope=rope,
                rope_positions=rope_positions,
                attn_mask=attn_mask,
                flex_block_mask=flex_block_mask,
            )
            new_kv = None
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.ffn(self.norm2(x)))
        if return_new_kv:
            return x, new_kv
        return x


class PlainFiLMBlock(nn.Module):
    """Plain transformer block with OpenVLA-OFT-style language FiLM.

    The language vector modulates only visual tokens, after self-attention and
    before the FFN. Proprio and previous-action tokens remain unmodulated so
    their numeric semantics stay carried by their own token projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = _make_rmsnorm(d_model)
        self.attn = QKNormAttention(d_model, num_heads, dropout=dropout)
        self.lang_film = LanguageFiLM(d_model)
        self.norm2 = _make_rmsnorm(d_model)
        self.ffn = SwiGLU(d_model, int(d_model * ffn_ratio), dropout=dropout)
        self.ls1 = LayerScale(d_model)
        self.ls2 = LayerScale(d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[ShallowRoPE] = None,
        rope_positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        flex_block_mask=None,
        film_cond: Optional[torch.Tensor] = None,
        visual_token_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_new_kv: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if return_new_kv or past_kv is not None:
            attn_out, new_kv = self.attn(
                self.norm1(x),
                rope=rope,
                rope_positions=rope_positions,
                attn_mask=attn_mask,
                flex_block_mask=flex_block_mask,
                past_kv=past_kv,
                return_new_kv=True,
            )
        else:
            attn_out = self.attn(
                self.norm1(x),
                rope=rope,
                rope_positions=rope_positions,
                attn_mask=attn_mask,
                flex_block_mask=flex_block_mask,
            )
            new_kv = None
        x = x + self.ls1(attn_out)
        x = self.lang_film(x, film_cond=film_cond, visual_token_mask=visual_token_mask)
        x = x + self.ls2(self.ffn(self.norm2(x)))
        if return_new_kv:
            return x, new_kv
        return x


class PlainTextCrossBlock(nn.Module):
    """Self-attn + optional text cross-attn + FFN, each with LayerScale."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_self = _make_rmsnorm(d_model)
        self.self_attn = QKNormAttention(d_model, num_heads, dropout=dropout)
        self.ls_self = LayerScale(d_model)

        self.norm_cross = _make_rmsnorm(d_model)
        self.cross_attn = QKNormCrossAttention(d_model, num_heads, dropout=dropout)
        self.ls_cross = LayerScale(d_model)

        self.norm_ffn = _make_rmsnorm(d_model)
        self.ffn = SwiGLU(d_model, int(d_model * ffn_ratio), dropout=dropout)
        self.ls_ffn = LayerScale(d_model)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[ShallowRoPE] = None,
        rope_positions: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        flex_block_mask=None,
        text_context: Optional[torch.Tensor] = None,
        text_keep_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_new_kv: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Self-attn (cache path when past_kv or return_new_kv set)
        if return_new_kv or past_kv is not None:
            attn_out, new_kv = self.self_attn(
                self.norm_self(x),
                rope=rope,
                rope_positions=rope_positions,
                attn_mask=self_attn_mask,
                flex_block_mask=flex_block_mask,
                past_kv=past_kv,
                return_new_kv=True,
            )
        else:
            attn_out = self.self_attn(
                self.norm_self(x),
                rope=rope,
                rope_positions=rope_positions,
                attn_mask=self_attn_mask,
                flex_block_mask=flex_block_mask,
            )
            new_kv = None
        x = x + self.ls_self(attn_out)
        if text_context is not None and text_context.shape[1] > 0:
            x = x + self.ls_cross(
                self.cross_attn(
                    self.norm_cross(x),
                    context=text_context,
                    keep_mask=text_keep_mask,
                )
            )
        x = x + self.ls_ffn(self.ffn(self.norm_ffn(x)))
        if return_new_kv:
            return x, new_kv
        return x


# -----------------------------------------------------------------------------
# Input projector : unchanged from prior revision (BN for SIGReg compatibility).
# -----------------------------------------------------------------------------


class BNMLPProjector(nn.Module):
    """BatchNorm1d + 2-layer MLP from DA3 space (d_in) to predictor space (d_model).

    LeWM (arXiv 2603.19312 §3.1) is explicit that LayerNorm on the encoder's
    last layer breaks SIGReg's anti-collapse property. BN aggregates per-
    channel statistics over all tokens in the batch and keeps that property.

    Trade-off: BN has a *systematic* train/eval mismatch : `model.train()`
    normalizes by batch statistics, `model.eval()` by running stats. With
    BF16 forward and a closed-loop rollout that runs single-env (effective
    batch=1), this mismatch is one of the channels through which BF16
    rounding errors compound into trajectory divergence. Use `LNMLPProjector`
    (the default in current configs) when SIGReg is disabled.
    """

    def __init__(self, d_in: int, d_model: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or d_model
        self.bn = nn.BatchNorm1d(d_in, momentum=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _apply(self, fn):
        super()._apply(fn)
        # Keep BN stats/affine in fp32 even if the parent model is cast to bf16.
        self.bn.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        out_dtype = x.dtype
        if self.bn.running_mean.dtype != torch.float32:
            self.bn.float()
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            flat = x.reshape(b * l, d).float()
            flat = self.bn(flat)
        mlp_dtype = next(self.mlp.parameters()).dtype
        flat = self.mlp(flat.to(dtype=mlp_dtype))
        return flat.reshape(b, l, -1).to(dtype=out_dtype)


class LNMLPProjector(nn.Module):
    """LayerNorm + 2-layer MLP from DA3 space (d_in) to predictor space (d_model).

    Mode-invariant alternative to `BNMLPProjector`. Per-token normalization
    over the channel dim : no train/eval mode mismatch, no batch-size
    sensitivity, no running stats to manage. Default in current configs.

    Don't pair with non-zero `lambda_sigreg`: per LeWM (arXiv 2603.19312 §3.1),
    LayerNorm on the encoder's pre-output breaks SIGReg's anti-collapse
    property. SIGReg is disabled in gam configs, so this is fine.
    """

    def __init__(self, d_in: int, d_model: int, hidden: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        hidden = hidden or d_model
        self.norm = nn.LayerNorm(d_in, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm is dtype-aware via PyTorch's autocast : no manual cast
        # required because the operation reduces over the channel dim and
        # LayerNorm parameters take the surrounding autocast dtype.
        return self.mlp(self.norm(x))


# -----------------------------------------------------------------------------
# Main predictor
# -----------------------------------------------------------------------------


class GAMFuturePredictor(nn.Module):
    """Dense block-autoregressive predictor at DA3-Giant's pre-global layer 12
    boundary.

    Sequence layout per timestep `t` (in order):
      [ view_0(CLS + R registers + P patches)  ...  view_{V-1}(...) |
        proprio_token_t | prev_action_token_{t-1} ]

    Full sequence = concat over t = 0 .. H-1. Attention is block-causal
    across timesteps (each step attends itself + all earlier steps), full
    within a step.

    For every observed step t in [0, H), the block produces:
      - predicted next visual tokens `o_{t+1}` (shape V × (1+R+P) × d_da3)
      - predicted next proprio `s_{t+1}` (shape proprio_dim)
      - predicted DA3-deep action token `a_t` (shape V × d_da3; one per view,
        read from the CLS position of the predicted visual block for that view)

    Text can be fed as cross-attention memory, a prepended sequence prefix, or
    OpenVLA-OFT-style FiLM visual-token modulation.
    """

    SUPPORTED_CONDITION_MODES = ("cross_attn", "concat", "film")

    def __init__(
        self,
        d_da3: int = 1536,
        d_model: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        num_patches_per_view: int = 256,
        num_register_tokens: int = 4,
        use_language: bool = True,
        language_dim: int = 768,
        language_len: int = 77,
        proprio_dim: int = 7,
        action_dim: int = 7,
        action_chunk_size: int = 1,
        sigreg: Optional[nn.Module] = None,
        sigreg_proj_dim: int = 256,
        sigreg_pool_mode: str = "cls",
        condition_mode: str = "cross_attn",
        input_proj_norm: str = "ln",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_da3 = int(d_da3)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.ffn_ratio = float(ffn_ratio)
        self.num_patches_per_view = int(num_patches_per_view)
        self.num_register_tokens = int(num_register_tokens)
        self.visual_tokens_per_view = 1 + self.num_register_tokens + self.num_patches_per_view
        self.tokens_per_view = self.visual_tokens_per_view
        # num_prefix_visual = CLS (1) + registers (R). Patches follow with grid
        # (y, x). RoPE uses this split to assign (y=0, x=0) to prefix.
        self.num_prefix_visual = 1 + self.num_register_tokens
        # Grid side inferred from num_patches_per_view (expect square).
        side = int(round(math.sqrt(self.num_patches_per_view)))
        if side * side != self.num_patches_per_view:
            raise ValueError(
                f"num_patches_per_view={self.num_patches_per_view} must be square; "
                "ShallowRoPE requires a square (H, W) patch grid."
            )
        self.grid_h = side
        self.grid_w = side

        self.use_language = bool(use_language)
        self.language_dim = int(language_dim)
        self.language_len = int(language_len)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        self.action_history_dim = self.action_dim * self.action_chunk_size
        self.use_gradient_checkpointing = bool(gradient_checkpointing)

        condition_mode = str(condition_mode).strip().lower()
        if condition_mode not in self.SUPPORTED_CONDITION_MODES:
            raise ValueError(
                f"GAMFuturePredictor.condition_mode={condition_mode!r} "
                f"not in {self.SUPPORTED_CONDITION_MODES}."
            )
        self.condition_mode = condition_mode

        # --- Input projection ---
        # Input projector: BN (legacy, train/eval mode-asymmetric, SIGReg-friendly)
        # vs LN (default, mode-invariant). LN drops the BF16 train/eval
        # mismatch channel from BatchNorm's running_stats vs batch_stats path
        # and removes the fp32 cast hack inside the forward. Pair with
        # lambda_sigreg=0 (default in gam configs).
        input_proj_norm = str(input_proj_norm).strip().lower()
        if input_proj_norm == "ln" or input_proj_norm == "layernorm":
            self.in_proj = LNMLPProjector(d_da3, d_model)
        elif input_proj_norm == "bn" or input_proj_norm == "batchnorm":
            self.in_proj = BNMLPProjector(d_da3, d_model)
        else:
            raise ValueError(
                f"predictor.input_proj_norm={input_proj_norm!r} not in {{'ln', 'bn'}}."
            )
        self.input_proj_norm = input_proj_norm

        # --- RoPE (axial 4D; reused for both visual and non-spatial tokens) ---
        self.rope = ShallowRoPE(
            head_dim=self.d_model // self.num_heads,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
        )

        # --- Token identity: learned modality/type offsets ---
        # RoPE gives relative position; identity of visual-patch vs CLS vs
        # proprio vs prev-action is still needed. We keep a tiny per-token-
        # type additive embedding (no temporal / view component : RoPE handles
        # that axis).
        # Slot types: 0 = CLS, 1 = register, 2 = patch, 3 = proprio, 4 = prev_action
        self.type_embed = nn.Parameter(torch.zeros(5, d_model))
        nn.init.normal_(self.type_embed, std=0.02)

        # --- Language embedding ---
        # cross_attn/concat use lang_proj + lang_pos. concat additionally adds
        # an in-sequence type-embedding offset and prepends the projected lang
        # tokens; cross_attn feeds them as K/V via PlainTextCrossBlock. film
        # pools projected language tokens without positional offsets and uses
        # per-block FiLM projectors to modulate visual predictor tokens.
        if self.use_language:
            self.lang_proj = nn.Linear(language_dim, d_model, bias=True)
            self.lang_pos = nn.Parameter(torch.zeros(self.language_len, d_model))
            nn.init.normal_(self.lang_pos, std=0.02)
            if self.condition_mode == "concat":
                # Slot type 5: language token. Allocated only when concat is
                # enabled so cross_attn checkpoints are bit-identical.
                self.lang_type_embed = nn.Parameter(torch.zeros(d_model))
                nn.init.normal_(self.lang_type_embed, std=0.02)
            else:
                self.lang_type_embed = None
        else:
            self.lang_proj = None
            self.lang_pos = None
            self.lang_type_embed = None

        # --- Single-slot token projections ---
        self.proprio_token_proj = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_history_proj = nn.Sequential(
            nn.Linear(self.action_history_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Transformer stack ---
        # cross_attn: per-block self-attn + cross-attn-to-language + FFN.
        # concat:     per-block self-attn + FFN. Language is prepended to the
        #             token sequence and seen via self-attention only.
        # film:       per-block self-attn + language FiLM on visual tokens + FFN.
        if self.condition_mode == "cross_attn":
            block_cls = PlainTextCrossBlock
        elif self.condition_mode == "film":
            block_cls = PlainFiLMBlock
        else:
            block_cls = PlainBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # --- Pre-norm output heads ---
        self.out_visual_norm = _make_rmsnorm(d_model)
        self.out_action_norm = _make_rmsnorm(d_model)
        self.out_proprio_norm = _make_rmsnorm(d_model)
        self.future_visual_proj = nn.Linear(d_model, d_da3, bias=True)
        self.action_proj = nn.Linear(d_model, d_da3, bias=True)
        self.future_proprio_proj = nn.Linear(d_model, proprio_dim, bias=True)

        # Small-std init so outputs are non-degenerate from step 0; residuals
        # are already zero-inert via LayerScale, so the head needs nonzero
        # values for gradient flow.
        nn.init.normal_(self.future_visual_proj.weight, std=0.02)
        nn.init.zeros_(self.future_visual_proj.bias)
        nn.init.normal_(self.action_proj.weight, std=0.02)
        nn.init.zeros_(self.action_proj.bias)
        nn.init.normal_(self.future_proprio_proj.weight, std=0.02)
        nn.init.zeros_(self.future_proprio_proj.bias)

        # --- Optional SIGReg ---
        self.sigreg = sigreg
        self.sigreg_proj_dim = int(sigreg_proj_dim)
        self.sigreg_pool_mode = str(sigreg_pool_mode)
        if sigreg is not None:
            self.sigreg_proj = nn.Linear(d_model, self.sigreg_proj_dim, bias=True)
        else:
            self.sigreg_proj = None

        # Cache for flex_attention BlockMask (keyed by sequence length).
        self._flex_mask_cache: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_language(self, lang_feats: torch.Tensor) -> torch.Tensor:
        if self.lang_proj is None:
            raise RuntimeError("use_language=False : can't embed language")
        x = self.lang_proj(lang_feats)
        x = x + self.lang_pos.unsqueeze(0).to(dtype=x.dtype)
        return x

    def _language_keep_mask(
        self,
        lang_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        lang_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if lang_len <= 0:
            return None
        if lang_padding_mask is None:
            return torch.ones(batch_size, lang_len, device=device, dtype=torch.bool)
        keep = lang_padding_mask.to(device=device).bool()
        if keep.shape[1] > lang_len:
            keep = keep[:, :lang_len]
        elif keep.shape[1] < lang_len:
            pad = torch.zeros(
                keep.shape[0],
                lang_len - keep.shape[1],
                device=device,
                dtype=torch.bool,
            )
            keep = torch.cat([keep, pad], dim=1)
        if keep.shape[0] != batch_size:
            raise ValueError(
                f"Language padding mask batch {keep.shape[0]} mismatches predictor batch {batch_size}."
            )
        empty_rows = ~keep.any(dim=1)
        if empty_rows.any():
            keep = keep.clone()
            keep[empty_rows, 0] = True
        return keep

    def _pool_language_for_film(
        self,
        lang_feats: torch.Tensor,
        lang_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Masked-mean language condition for FiLM, matching OpenVLA-OFT+.

        Unlike concat/cross-attn, this path intentionally skips learned
        language position embeddings: FiLM should get one task-level language
        vector rather than a sequence-position-specific token representation.
        """
        if self.lang_proj is None:
            raise RuntimeError("use_language=False : can't pool language for FiLM")
        lang_tokens = self.lang_proj(
            lang_feats.to(device=device, dtype=self.lang_proj.weight.dtype)
        ).to(dtype=dtype)
        keep = self._language_keep_mask(
            lang_padding_mask=lang_padding_mask,
            batch_size=batch_size,
            lang_len=lang_tokens.shape[1],
            device=device,
        )
        if keep is None:
            return lang_tokens.mean(dim=1)
        weights = keep.to(dtype=dtype, device=device).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (lang_tokens * weights).sum(dim=1) / denom

    def _embed_proprio_history(
        self,
        proprio: Optional[torch.Tensor],
        proprio_history: Optional[torch.Tensor],
        H: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if proprio_history is None:
            extra = ""
            if proprio is not None:
                extra = f" Received proprio shape {tuple(proprio.shape)} instead."
            raise ValueError(
                "GAMFuturePredictor requires explicit proprio_history with shape (B,H,D)."
                + extra
            )
        if proprio_history.ndim != 3:
            raise ValueError(f"Expected proprio_history as (B,H,D), got {tuple(proprio_history.shape)}.")
        if proprio_history.shape[0] != batch_size:
            raise ValueError(
                f"proprio_history batch {proprio_history.shape[0]} mismatches predictor batch {batch_size}."
            )
        if proprio_history.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio_history dim {proprio_history.shape[-1]} mismatches proprio_dim={self.proprio_dim}."
            )
        if proprio_history.shape[1] > H:
            proprio_history = proprio_history[:, -H:]
        elif proprio_history.shape[1] < H:
            pad_len = H - proprio_history.shape[1]
            pad = proprio_history[:, :1].expand(-1, pad_len, -1)
            proprio_history = torch.cat([pad, proprio_history], dim=1)

        proj_dtype = next(self.proprio_token_proj.parameters()).dtype
        tokens = self.proprio_token_proj(
            proprio_history.to(device=device, dtype=proj_dtype)
        ).to(dtype=dtype)
        return tokens + self.type_embed[3].view(1, 1, -1).to(dtype=dtype)

    def _embed_action_history(
        self,
        past_action_history: Optional[torch.Tensor],
        H: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if past_action_history is None:
            raise ValueError("GAMFuturePredictor requires aligned past_action_history.")
        if past_action_history.shape[0] != batch_size:
            raise ValueError(
                f"past_action_history batch {past_action_history.shape[0]} mismatches predictor batch {batch_size}."
            )
        if past_action_history.ndim < 3:
            raise ValueError(
                "Expected past_action_history with shape (B, H, ...action...), "
                f"got {tuple(past_action_history.shape)}."
            )
        if past_action_history.shape[1] > H:
            past_action_history = past_action_history[:, -H:]
        elif past_action_history.shape[1] < H:
            pad_shape = (batch_size, H - past_action_history.shape[1], *past_action_history.shape[2:])
            pad = torch.zeros(
                pad_shape,
                device=past_action_history.device,
                dtype=past_action_history.dtype,
            )
            past_action_history = torch.cat([pad, past_action_history], dim=1)
        flat = past_action_history.reshape(batch_size, H, -1)
        if flat.shape[-1] != self.action_history_dim:
            raise ValueError(
                f"past_action_history flattened dim {flat.shape[-1]} mismatches "
                f"action_history_dim={self.action_history_dim}."
            )
        proj_dtype = next(self.action_history_proj.parameters()).dtype
        tokens = self.action_history_proj(flat.to(device=device, dtype=proj_dtype)).to(dtype=dtype)
        return tokens + self.type_embed[4].view(1, 1, -1).to(dtype=dtype)

    def _visual_type_embed(
        self,
        H: int,
        V: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return additive embedding for a (H * V * P_view) flat token stream.

        Layout per view: [CLS | registers | patches]. Types: 0 CLS, 1 reg, 2 patch.
        """
        types = torch.zeros(self.visual_tokens_per_view, dtype=torch.long, device=device)
        types[0] = 0                         # CLS
        types[1 : 1 + self.num_register_tokens] = 1    # registers
        types[1 + self.num_register_tokens :] = 2      # patches
        emb = self.type_embed[types].to(dtype=dtype)   # (P_view, d_model)
        # Broadcast over H, V
        return emb.view(1, 1, 1, self.visual_tokens_per_view, self.d_model).expand(
            1, H, V, self.visual_tokens_per_view, self.d_model
        ).reshape(1, H * V * self.visual_tokens_per_view, self.d_model)

    def _build_visual_token_mask(self, H: int, V: int, device: torch.device) -> torch.Tensor:
        """Mask visual-token positions inside flattened per-step predictor input."""
        tokens_per_step = V * self.visual_tokens_per_view + 2
        visual_per_step = V * self.visual_tokens_per_view
        idx = torch.arange(H * tokens_per_step, device=device)
        return (idx % tokens_per_step) < visual_per_step

    def _build_step_keep_mask(
        self,
        view_valid_mask: torch.Tensor,
        H: int,
        V: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a flattened predictor token keep-mask from per-view validity.

        Visual tokens from padded camera slots are False; proprio and previous
        action tokens stay True because they are per-timestep, not per-view.
        """
        if view_valid_mask.shape != (view_valid_mask.shape[0], H, V):
            raise ValueError(
                "view_valid_mask must have shape "
                f"(B,{H},{V}), got {tuple(view_valid_mask.shape)}."
            )
        keep = view_valid_mask.to(device=device, dtype=torch.bool)
        visual = keep[:, :, :, None].expand(
            -1, -1, -1, self.visual_tokens_per_view
        ).reshape(keep.shape[0], H, V * self.visual_tokens_per_view)
        slots = torch.ones(keep.shape[0], H, 2, device=device, dtype=torch.bool)
        return torch.cat([visual, slots], dim=2).reshape(keep.shape[0], -1)

    @staticmethod
    def _apply_sequence_keep(x: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if keep_mask is None:
            return x
        return x * keep_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

    @staticmethod
    def _merge_key_keep_mask(
        attn_mask: Optional[torch.Tensor],
        keep_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if keep_mask is None or bool(keep_mask.all().item()):
            return attn_mask
        key_keep = keep_mask[:, None, None, :].to(dtype=torch.bool)
        if torch.equal(keep_mask, keep_mask[:1].expand_as(keep_mask)):
            key_keep = key_keep[:1]
        if attn_mask is None:
            q_len = keep_mask.shape[1]
            query_keep = torch.ones(
                key_keep.shape[0],
                1,
                q_len,
                1,
                device=keep_mask.device,
                dtype=torch.bool,
            )
            return query_keep & key_keep
        return attn_mask.to(device=keep_mask.device, dtype=torch.bool) & key_keep

    def _build_concat_lang_positions(
        self,
        lang_len: int,
        V: int,
        device: torch.device,
    ) -> torch.Tensor:
        """RoPE positions for the prepended language prefix.

        Each lang token i gets `(t = -1, v = V + 2, y = i, x = 0)`. The
        negative time axis keeps lang strictly "before" all step blocks under
        relative-position attention; the dedicated `v = V + 2` virtual view
        is distinct from real cameras (`v < V`), proprio (`v = V`), and
        prev-action (`v = V + 1`); the y axis encodes within-prompt order.
        """
        pos = torch.empty(lang_len, 4, device=device, dtype=torch.long)
        pos[:, 0] = -1
        pos[:, 1] = V + 2
        pos[:, 2] = torch.arange(lang_len, device=device, dtype=torch.long)
        pos[:, 3] = 0
        return pos

    def _build_concat_dense_mask(
        self,
        H: int,
        V: int,
        lang_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Dense mask for concat conditioning.

        Layout: [lang(L_lang) | step_0 | step_1 | ... | step_{H-1}].
        Rules:
          - lang queries attend only to lang keys (bidirectional inside the
            language prefix; lang representation stays unpolluted by step
            tokens).
          - step queries attend to ALL lang keys + step keys with
            block-causal ordering (q_step >= kv_step).
        """
        tokens_per_step = V * self.visual_tokens_per_view + 2
        L_steps = H * tokens_per_step
        L_total = lang_len + L_steps

        mask = torch.zeros(L_total, L_total, dtype=torch.bool, device=device)
        # Lang block (rows + cols 0..lang_len)
        if lang_len > 0:
            mask[:lang_len, :lang_len] = True
        # Step queries attend to all lang
        if lang_len > 0:
            mask[lang_len:, :lang_len] = True
        # Step block-causal among themselves
        step_ids = torch.arange(H, device=device, dtype=torch.long).repeat_interleave(tokens_per_step)
        step_block = step_ids[:, None] >= step_ids[None, :]
        mask[lang_len:, lang_len:] = step_block
        return mask.unsqueeze(0).unsqueeze(0)

    def _get_flex_block_mask(self, H: int, V: int, device: torch.device, lang_len: int = 0):
        if not _HAS_FLEX:
            return None
        tokens_per_step = V * self.visual_tokens_per_view + 2
        L_steps = H * tokens_per_step
        L = L_steps + int(lang_len)
        # Cache key includes lang_len so a concat-mode mask never gets reused
        # by a cross_attn-mode call (and vice versa).
        key = (L, tokens_per_step, int(lang_len), device)
        cached = self._flex_mask_cache.get(key)
        if cached is not None:
            return cached
        try:
            if lang_len <= 0:
                mask_mod = _make_block_causal_mask_mod(tokens_per_step)
            else:
                mask_mod = _make_concat_lang_block_causal_mask_mod(
                    tokens_per_step=tokens_per_step,
                    lang_len=int(lang_len),
                )
            block_mask = _create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device
            )
        except Exception as exc:
            # Warn ONCE per session so the user notices dense-mask fallback
            # (which is O(L²) memory and can OOM at large H).
            if not getattr(self, "_flex_warned", False):
                import logging
                logging.getLogger(__name__).warning(
                    "flex_attention.create_block_mask failed (%s: %s); "
                    "falling back to dense block-causal mask. "
                    "This costs O(L^2) memory at L=%d.",
                    type(exc).__name__, exc, L,
                )
                self._flex_warned = True
            return None
        self._flex_mask_cache[key] = block_mask
        return block_mask

    def _build_dense_block_causal_mask(
        self,
        H: int,
        V: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokens_per_step = V * self.visual_tokens_per_view + 2
        step_ids = torch.arange(H, device=device, dtype=torch.long).repeat_interleave(tokens_per_step)
        m = step_ids[:, None] >= step_ids[None, :]
        return m.unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        past_visual_tokens: torch.Tensor,       # (B, H, V, 1+R+P, d_da3)
        proprio: Optional[torch.Tensor] = None,
        proprio_history: Optional[torch.Tensor] = None,     # (B, H, proprio_dim)
        past_action_history: Optional[torch.Tensor] = None, # (B, H, chunk, action_dim)
        lang_feats: Optional[torch.Tensor] = None,
        lang_padding_mask: Optional[torch.Tensor] = None,
        context_valid_mask: Optional[torch.Tensor] = None,  # (B, H) bool
        view_valid_mask: Optional[torch.Tensor] = None,     # (B, H, V) bool
    ) -> dict:
        b, H, V, P, d = past_visual_tokens.shape
        if P != self.visual_tokens_per_view or d != self.d_da3:
            raise ValueError(
                f"past_visual_tokens shape {tuple(past_visual_tokens.shape)} mismatches "
                f"(B, H, V, {self.visual_tokens_per_view}, {self.d_da3})."
            )
        device = past_visual_tokens.device
        view_keep = None
        step_token_keep_mask = None
        if view_valid_mask is not None:
            view_keep = view_valid_mask.to(device=device, dtype=torch.bool)
            if view_keep.shape != (b, H, V):
                raise ValueError(
                    f"view_valid_mask shape {tuple(view_keep.shape)} mismatches "
                    f"(B,H,V)=({b},{H},{V})."
                )
            if not bool(view_keep.any(dim=2).all().item()):
                raise ValueError("Each timestep must keep at least one valid view.")
            step_token_keep_mask = self._build_step_keep_mask(view_keep, H, V, device)

        # 1. Project visual context into predictor space.
        flat_context = past_visual_tokens.reshape(b, H * V * P, d)
        context = self.in_proj(flat_context)
        context = context + self._visual_type_embed(H, V, device, context.dtype)
        context = context.reshape(b, H, V, P, self.d_model)
        if view_keep is not None:
            context = context * view_keep[:, :, :, None, None].to(dtype=context.dtype)
        context = context.reshape(b, H, V * P, self.d_model)

        # 2. Embed proprio + action-history tokens (one each per timestep).
        proprio_tokens = self._embed_proprio_history(
            proprio=proprio,
            proprio_history=proprio_history,
            H=H,
            batch_size=b,
            device=device,
            dtype=context.dtype,
        ).unsqueeze(2)   # (B, H, 1, d_model)
        action_history_tokens = self._embed_action_history(
            past_action_history=past_action_history,
            H=H,
            batch_size=b,
            device=device,
            dtype=context.dtype,
        ).unsqueeze(2)   # (B, H, 1, d_model)

        # 3. Assemble per-step blocks and flatten across steps.
        step_blocks = torch.cat([context, proprio_tokens, action_history_tokens], dim=2)
        if context_valid_mask is not None:
            valid = context_valid_mask.to(device=device, dtype=torch.bool)
            if valid.shape != (b, H):
                raise ValueError(
                    f"context_valid_mask shape {tuple(valid.shape)} mismatches (B,H)=({b},{H})."
                )
            step_blocks = step_blocks * valid[:, :, None, None].to(dtype=step_blocks.dtype)
        x = step_blocks.reshape(b, -1, self.d_model)
        x = self._apply_sequence_keep(x, step_token_keep_mask)
        L_total = x.shape[1]

        # 4. Positions for RoPE (once per H/V) : step token positions.
        step_rope_positions = self.rope.build_positions(
            H=H, V=V,
            num_patches=self.num_patches_per_view,
            num_prefix_visual=self.num_prefix_visual,
            device=device,
        )

        # 5. Resolve language conditioning. Three paths:
        #    - cross_attn: lang feeds blocks as cross-attention KV; the
        #      step token sequence stays unchanged.
        #    - concat:    lang is prepended to the step token sequence;
        #      blocks are PlainBlock (no cross_attn submodule).
        #    - film:      lang is masked-mean pooled and modulates visual
        #      predictor tokens in each PlainFiLMBlock.
        lang_emb = None                  # cross_attn payload (KV)
        lang_keep_mask = None             # cross_attn mask
        film_cond = None                  # film payload (B, d_model)
        visual_token_mask = None           # film target positions in x
        prepended_lang_len = 0            # concat: how many tokens are prefixed
        if self.use_language and lang_feats is not None:
            if self.condition_mode == "cross_attn":
                lang_emb_raw = self._embed_language(lang_feats.to(context.dtype)).to(context.dtype)
                lang_emb = lang_emb_raw
                lang_keep_mask = self._language_keep_mask(
                    lang_padding_mask=lang_padding_mask,
                    batch_size=b,
                    lang_len=lang_emb_raw.shape[1],
                    device=device,
                )
            elif self.condition_mode == "concat":
                lang_emb_raw = self._embed_language(lang_feats.to(context.dtype)).to(context.dtype)
                # concat: type-tag and prepend.
                lang_tokens = lang_emb_raw + self.lang_type_embed.view(1, 1, -1).to(dtype=context.dtype)
                # Zero out padded positions so they don't influence self-attn.
                if lang_padding_mask is not None:
                    keep = lang_padding_mask.to(dtype=lang_tokens.dtype, device=lang_tokens.device)
                    lang_tokens = lang_tokens * keep.unsqueeze(-1)
                prepended_lang_len = lang_tokens.shape[1]
                x = torch.cat([lang_tokens, x], dim=1)
            elif self.condition_mode == "film":
                film_cond = self._pool_language_for_film(
                    lang_feats=lang_feats,
                    lang_padding_mask=lang_padding_mask,
                    batch_size=b,
                    device=device,
                    dtype=context.dtype,
                )
                visual_token_mask = self._build_visual_token_mask(H, V, device)

        # 6. Attention mask + RoPE positions, possibly extended for concat lang prefix.
        flex_block_mask = self._get_flex_block_mask(H, V, device, lang_len=prepended_lang_len)
        if flex_block_mask is not None:
            dense_mask = None
        elif prepended_lang_len > 0:
            dense_mask = self._build_concat_dense_mask(H, V, prepended_lang_len, device)
        else:
            dense_mask = self._build_dense_block_causal_mask(H, V, device)

        if prepended_lang_len > 0:
            lang_rope_positions = self._build_concat_lang_positions(
                lang_len=prepended_lang_len, V=V, device=device,
            )
            rope_positions = torch.cat([lang_rope_positions, step_rope_positions], dim=0)
            if step_token_keep_mask is not None:
                lang_keep = torch.ones(
                    b,
                    prepended_lang_len,
                    device=device,
                    dtype=torch.bool,
                )
                sequence_keep_mask = torch.cat([lang_keep, step_token_keep_mask], dim=1)
            else:
                sequence_keep_mask = None
        else:
            rope_positions = step_rope_positions
            sequence_keep_mask = step_token_keep_mask

        if sequence_keep_mask is not None and not bool(sequence_keep_mask.all().item()):
            # Batch-specific padded-view masks require SDPA with an explicit key
            # mask because the cached flex BlockMask is shape-only.
            flex_block_mask = None
            dense_mask = self._merge_key_keep_mask(dense_mask, sequence_keep_mask)

        # 7. Transformer stack.
        use_checkpoint = (
            bool(self.use_gradient_checkpointing)
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        )
        for block in self.blocks:
            if self.condition_mode == "cross_attn":
                if use_checkpoint:
                    x = torch_checkpoint(
                        lambda x_in, block=block: block(
                            x_in,
                            rope=self.rope,
                            rope_positions=rope_positions,
                            self_attn_mask=dense_mask,
                            flex_block_mask=flex_block_mask,
                            text_context=lang_emb,
                            text_keep_mask=lang_keep_mask,
                        ),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        rope=self.rope,
                        rope_positions=rope_positions,
                        self_attn_mask=dense_mask,
                        flex_block_mask=flex_block_mask,
                        text_context=lang_emb,
                        text_keep_mask=lang_keep_mask,
                    )
                x = self._apply_sequence_keep(x, sequence_keep_mask)
            elif self.condition_mode == "film":
                if use_checkpoint:
                    x = torch_checkpoint(
                        lambda x_in, block=block: block(
                            x_in,
                            rope=self.rope,
                            rope_positions=rope_positions,
                            attn_mask=dense_mask,
                            flex_block_mask=flex_block_mask,
                            film_cond=film_cond,
                            visual_token_mask=visual_token_mask,
                        ),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        rope=self.rope,
                        rope_positions=rope_positions,
                        attn_mask=dense_mask,
                        flex_block_mask=flex_block_mask,
                        film_cond=film_cond,
                        visual_token_mask=visual_token_mask,
                    )
                x = self._apply_sequence_keep(x, sequence_keep_mask)
            else:
                # PlainBlock signature: (x, rope, rope_positions, attn_mask, flex_block_mask)
                if use_checkpoint:
                    x = torch_checkpoint(
                        lambda x_in, block=block: block(
                            x_in,
                            rope=self.rope,
                            rope_positions=rope_positions,
                            attn_mask=dense_mask,
                            flex_block_mask=flex_block_mask,
                        ),
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x,
                        rope=self.rope,
                        rope_positions=rope_positions,
                        attn_mask=dense_mask,
                        flex_block_mask=flex_block_mask,
                    )
                x = self._apply_sequence_keep(x, sequence_keep_mask)

        # Strip language prefix before per-step reshape.
        if prepended_lang_len > 0:
            x = x[:, prepended_lang_len:]

        # 8. Split per-step outputs.
        tokens_per_step = V * P + 2
        x = x.reshape(b, H, tokens_per_step, self.d_model)
        visual_h_raw = x[:, :, : V * P].reshape(b, H, V, P, self.d_model)
        proprio_h = x[:, :, V * P]
        action_history_h = x[:, :, V * P + 1]

        # 9. Pre-norm output heads. Action token comes from the *dedicated
        #    action slot* hidden (action_history_h), separate from visual CLS.
        #    DA3
        #    blocks 13+ expect a single per-timestep action token, so we pre-
        #    norm once per timestep and then repeat across views to seed DA3's
        #    per-view action-token insertion contract.
        visual_h = self.out_visual_norm(visual_h_raw)
        action_h = self.out_action_norm(action_history_h)
        proprio_h_norm = self.out_proprio_norm(proprio_h)

        z_next_all = self.future_visual_proj(visual_h).to(past_visual_tokens.dtype)
        action_token_per_step = self.action_proj(action_h)                     # (B, H, d_da3)
        action_token_all = (
            action_token_per_step.unsqueeze(2)
            .expand(-1, -1, V, -1)
            .contiguous()
            .to(past_visual_tokens.dtype)
        )                                                                       # (B, H, V, d_da3)
        s_next_all = self.future_proprio_proj(proprio_h_norm).to(past_visual_tokens.dtype)
        if view_keep is not None:
            z_next_all = z_next_all * view_keep[:, :, :, None, None].to(dtype=z_next_all.dtype)
            action_token_all = action_token_all * view_keep[:, :, :, None].to(
                dtype=action_token_all.dtype
            )

        # 10. Optional SIGReg on pooled predicted-future-visual.
        sigreg_loss = None
        if self.sigreg is not None:
            fut_hidden = visual_h.reshape(b * H, V, P, self.d_model)
            if self.sigreg_pool_mode == "cls":
                pooled = fut_hidden[:, :, 0, :]
            elif self.sigreg_pool_mode == "patch_mean":
                pooled = fut_hidden[:, :, 1 + self.num_register_tokens :, :].mean(dim=2)
            elif self.sigreg_pool_mode == "all_mean":
                pooled = fut_hidden.mean(dim=2)
            else:
                raise ValueError(f"Unknown sigreg_pool_mode {self.sigreg_pool_mode}")
            pooled_small = self.sigreg_proj(pooled)
            sigreg_loss = self.sigreg(pooled_small.reshape(-1, self.sigreg_proj_dim))

        return {
            "predicted_next_visual_tokens": z_next_all,
            "predicted_next_proprio": s_next_all,
            "predicted_action_tokens": action_token_all,
            "encoded_prev_action_tokens": action_history_h,
            "encoded_proprio_tokens": proprio_h,
            "sigreg_loss": sigreg_loss,
        }

    # ------------------------------------------------------------------
    # Incremental AR forward with KV cache
    # ------------------------------------------------------------------

    def _rope_positions_single_step(
        self,
        t_abs: int,
        V: int,
        device: torch.device,
    ) -> torch.Tensor:
        """RoPE positions for ONE timestep block at absolute t=t_abs.

        Same layout as `ShallowRoPE.build_positions` but for H=1 with t forced
        to `t_abs` (not 0). Returns (tokens_per_step, 4).
        """
        tokens_per_view = self.num_prefix_visual + self.num_patches_per_view
        tokens_per_step = V * tokens_per_view + 2
        pos = torch.empty(tokens_per_step, 4, device=device, dtype=torch.long)
        ys = torch.arange(self.grid_h, device=device)
        xs = torch.arange(self.grid_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_y = grid_y.flatten()
        patch_x = grid_x.flatten()
        idx = 0
        for v in range(V):
            for _ in range(self.num_prefix_visual):
                pos[idx, 0] = t_abs
                pos[idx, 1] = v
                pos[idx, 2] = 0
                pos[idx, 3] = 0
                idx += 1
            pos[idx : idx + self.num_patches_per_view, 0] = t_abs
            pos[idx : idx + self.num_patches_per_view, 1] = v
            pos[idx : idx + self.num_patches_per_view, 2] = patch_y
            pos[idx : idx + self.num_patches_per_view, 3] = patch_x
            idx += self.num_patches_per_view
        # proprio slot
        pos[idx, 0] = t_abs
        pos[idx, 1] = V
        pos[idx, 2] = 0
        pos[idx, 3] = 0
        idx += 1
        # prev_action slot
        pos[idx, 0] = t_abs
        pos[idx, 1] = V + 1
        pos[idx, 2] = 0
        pos[idx, 3] = 0
        idx += 1
        assert idx == tokens_per_step
        return pos

    @torch.no_grad()
    def forward_incremental(
        self,
        new_visual_tokens: torch.Tensor,     # (B, 1, V, P, d_da3) one new timestep
        new_proprio: torch.Tensor,           # (B, 1, proprio_dim)
        new_prev_action: torch.Tensor,       # (B, 1, chunk, action_dim)
        past_length: int,
        past_kvs: Optional[list] = None,
        lang_feats: Optional[torch.Tensor] = None,
        lang_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """AR-step forward using KV cache.

        Drop-in replacement for a single iteration of the main AR loop. On the
        first call pass `past_length=observed_length-1, past_kvs=None` to warm
        the cache with the final observed step (everything before is already in
        cache from a previous call chain; simplest: call with past_length=0 and
        past_kvs=None for iteration 0 where x contains all observed frames,
        then subsequent calls feed ONE new step each).

        Returns dict mirroring `forward`'s outputs PLUS `new_past_kvs` : a list
        of (K, V) tuples, one per transformer layer, to feed back on next call.
        """
        b, H_new, V, P, d = new_visual_tokens.shape
        if H_new != 1:
            raise ValueError(f"forward_incremental expects H=1 new step, got H={H_new}")
        if P != self.visual_tokens_per_view or d != self.d_da3:
            raise ValueError(
                f"new_visual_tokens shape {tuple(new_visual_tokens.shape)} != "
                f"(B,1,V,{self.visual_tokens_per_view},{self.d_da3})"
            )
        device = new_visual_tokens.device

        # 1. Visual context projection + type embed
        flat_context = new_visual_tokens.reshape(b, V * P, d)
        context = self.in_proj(flat_context)
        context = context + self._visual_type_embed(1, V, device, context.dtype).squeeze(0)

        # 2. Proprio + action slots
        proprio_tokens = self._embed_proprio_history(
            proprio=None,
            proprio_history=new_proprio,
            H=1,
            batch_size=b,
            device=device,
            dtype=context.dtype,
        )  # (B, 1, d_model)
        action_history_tokens = self._embed_action_history(
            past_action_history=new_prev_action,
            H=1,
            batch_size=b,
            device=device,
            dtype=context.dtype,
        )  # (B, 1, d_model)

        # 3. Concat into single-step block
        x = torch.cat([context, proprio_tokens, action_history_tokens], dim=1)
        tokens_per_step = V * P + 2
        assert x.shape[1] == tokens_per_step

        # 4. RoPE positions for absolute t = past_length (step-only).
        step_rope_positions = self._rope_positions_single_step(
            t_abs=past_length, V=V, device=device,
        )

        # 5. Language conditioning. Cross-attn: pass as KV memory each step.
        # Concat: prepend lang to x ONLY on the first call (when past_kvs is
        # None) so it gets baked into every block's KV cache; subsequent calls
        # see lang via cached K/V from prior layers. FiLM: pass the pooled
        # language vector every step and modulate only the new visual tokens.
        lang_emb = None
        lang_keep_mask = None
        film_cond = None
        visual_token_mask = None
        rope_positions = step_rope_positions
        cache_warm = past_kvs is not None
        prepended_lang_len = 0
        if self.use_language and lang_feats is not None:
            if self.condition_mode == "cross_attn":
                lang_emb_raw = self._embed_language(lang_feats.to(context.dtype)).to(context.dtype)
                lang_emb = lang_emb_raw
                lang_keep_mask = self._language_keep_mask(
                    lang_padding_mask=lang_padding_mask,
                    batch_size=b,
                    lang_len=lang_emb_raw.shape[1],
                    device=device,
                )
            elif self.condition_mode == "concat":
                lang_emb_raw = self._embed_language(lang_feats.to(context.dtype)).to(context.dtype)
                # concat: prepend on first call only.
                if not cache_warm:
                    lang_tokens = lang_emb_raw + self.lang_type_embed.view(1, 1, -1).to(dtype=context.dtype)
                    if lang_padding_mask is not None:
                        keep = lang_padding_mask.to(dtype=lang_tokens.dtype, device=lang_tokens.device)
                        lang_tokens = lang_tokens * keep.unsqueeze(-1)
                    prepended_lang_len = lang_tokens.shape[1]
                    x = torch.cat([lang_tokens, x], dim=1)
                    lang_rope_positions = self._build_concat_lang_positions(
                        lang_len=prepended_lang_len, V=V, device=device,
                    )
                    rope_positions = torch.cat([lang_rope_positions, step_rope_positions], dim=0)
            elif self.condition_mode == "film":
                film_cond = self._pool_language_for_film(
                    lang_feats=lang_feats,
                    lang_padding_mask=lang_padding_mask,
                    batch_size=b,
                    device=device,
                    dtype=context.dtype,
                )
                visual_token_mask = self._build_visual_token_mask(1, V, device)

        # 6. Transformer stack with KV cache.
        new_past_kvs = []
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            if self.condition_mode == "cross_attn":
                x, kv = block(
                    x,
                    rope=self.rope,
                    rope_positions=rope_positions,
                    self_attn_mask=None,
                    flex_block_mask=None,
                    text_context=lang_emb,
                    text_keep_mask=lang_keep_mask,
                    past_kv=past_kvs[i],
                    return_new_kv=True,
                )
            elif self.condition_mode == "film":
                x, kv = block(
                    x,
                    rope=self.rope,
                    rope_positions=rope_positions,
                    attn_mask=None,
                    flex_block_mask=None,
                    film_cond=film_cond,
                    visual_token_mask=visual_token_mask,
                    past_kv=past_kvs[i],
                    return_new_kv=True,
                )
            else:
                x, kv = block(
                    x,
                    rope=self.rope,
                    rope_positions=rope_positions,
                    attn_mask=None,
                    flex_block_mask=None,
                    past_kv=past_kvs[i],
                    return_new_kv=True,
                )
            new_past_kvs.append(kv)

        # Strip prepended lang prefix before reshaping into per-step blocks.
        if prepended_lang_len > 0:
            x = x[:, prepended_lang_len:]

        # 7. Split per-step outputs (H=1)
        x = x.reshape(b, 1, tokens_per_step, self.d_model)
        visual_h_raw = x[:, :, : V * P].reshape(b, 1, V, P, self.d_model)
        proprio_h = x[:, :, V * P]
        action_history_h = x[:, :, V * P + 1]

        visual_h = self.out_visual_norm(visual_h_raw)
        action_h = self.out_action_norm(action_history_h)
        proprio_h_norm = self.out_proprio_norm(proprio_h)

        z_next = self.future_visual_proj(visual_h).to(new_visual_tokens.dtype)
        action_token_per_step = self.action_proj(action_h)
        action_token = (
            action_token_per_step.unsqueeze(2)
            .expand(-1, -1, V, -1)
            .contiguous()
            .to(new_visual_tokens.dtype)
        )
        s_next = self.future_proprio_proj(proprio_h_norm).to(new_visual_tokens.dtype)

        return {
            "predicted_next_visual_tokens": z_next,
            "predicted_next_proprio": s_next,
            "predicted_action_tokens": action_token,
            "new_past_kvs": new_past_kvs,
        }


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_future_predictor(cfg: dict, sigreg: Optional[nn.Module] = None) -> nn.Module:
    """Instantiate the predictor from a config dict.

    Only `predictor.type: gam` is supported. Legacy config keys like
    `action_seed_mode` / `use_action_history` are explicitly rejected to
    surface stale configs.
    """
    pred_type = str(cfg.get("type", cfg.get("architecture", "gam"))).lower()
    allowed = {"gam"}
    legacy_aliases = {"level0", "v1", "v2"}
    if pred_type in legacy_aliases:
        raise ValueError(
            f"predictor.type={pred_type!r} is no longer supported. The v1/v2/"
            "level0 FuturePredictor was removed in the 2026-04-21 arch-modernize "
            "refactor. Use predictor.type: gam."
        )
    if pred_type not in allowed:
        raise ValueError(
            f"Unknown predictor.type={pred_type!r}. Must be one of {sorted(allowed)}."
        )

    # Reject legacy keys that no longer apply.
    removed_keys = [
        key
        for key in (
            "action_seed_mode",
            "use_action_history",
            "predict_future_proprio",
            "deep_context_steps",
            "deep_context_full_prob",
            "attention_fp32",
            "max_timesteps",
            "max_views",
        )
        if key in cfg
    ]
    if removed_keys:
        raise ValueError(
            "Legacy predictor keys no longer supported after 2026-04-21 refactor: "
            f"{removed_keys}. Remove them from the config. "
            "(`max_timesteps` / `max_views`: position scaling is now RoPE-based; "
            "`attention_fp32`: replaced by QK-norm in attention.)"
        )

    return GAMFuturePredictor(
        d_da3=int(cfg.get("d_da3", 1536)),
        d_model=int(cfg.get("d_model", 1024)),
        depth=int(cfg.get("depth", 12)),
        num_heads=int(cfg.get("num_heads", 16)),
        ffn_ratio=float(cfg.get("ffn_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.0)),
        num_patches_per_view=int(cfg.get("num_patches_per_view", 256)),
        num_register_tokens=int(cfg.get("num_register_tokens", 4)),
        use_language=bool(cfg.get("use_language", True)),
        language_dim=int(cfg.get("language_dim", 768)),
        language_len=int(cfg.get("language_len", 77)),
        proprio_dim=int(cfg.get("proprio_dim", 7)),
        action_dim=int(cfg.get("action_dim", 7)),
        action_chunk_size=int(cfg.get("action_chunk_size", 1)),
        sigreg=sigreg,
        sigreg_proj_dim=int(cfg.get("sigreg_proj_dim", 256)),
        sigreg_pool_mode=str(cfg.get("sigreg_pool_mode", "cls")),
        condition_mode=str(cfg.get("condition_mode", "cross_attn")),
        input_proj_norm=str(cfg.get("input_proj_norm", "ln")),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )
