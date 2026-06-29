"""Unified future-predictor forward + loss helpers.

Slots the FuturePredictor into the existing Stage 1 training loop without
duplicating the whole training function. The entry point is
`compute_unified_forward_loss(...)` which runs:

  1. Student DA3 on H past frames (per-slot AE for strict past, noact at boundary).
  2. Teacher DA3 on H past frames (past-only, no future leakage) for past feature-reg.
  3. Frozen Teacher DA3 on ALL T frames (for future L2 targets).
  4. Predictor: past tokens + future mask slots + language -> predictions.
  5. Losses: action L1 after DA3 propagation (F+1 actions), feature L2
     future, feature reg past, optional SIGReg anti-collapse, optional
     DA3-style future depth.

This is a pure utility : no Dataset / DataLoader / optimizer logic here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cuda_profile_mark(profile: Optional[Dict[str, object]], name: str) -> None:
    if profile is None or not torch.cuda.is_available():
        return
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    profile.setdefault("_cuda_marks", []).append((name, event))


def _profile_interval_key(mark_name: str) -> Optional[str]:
    if mark_name == "deep_done":
        return "deep_return_overhead_ms"
    if mark_name.endswith("_done"):
        return f"{mark_name[:-5]}_ms"
    return None


def _finalize_cuda_profile(profile: Optional[Dict[str, object]]) -> None:
    if profile is None or not torch.cuda.is_available():
        return
    marks = profile.pop("_cuda_marks", [])
    if len(marks) < 2:
        return
    torch.cuda.synchronize()
    for (prev_name, prev_event), (name, event) in zip(marks[:-1], marks[1:]):
        key = _profile_interval_key(name)
        if key is None:
            continue
        profile[key] = float(profile.get(key, 0.0)) + float(prev_event.elapsed_time(event))
    profile["fwd_profile_total_ms"] = float(marks[0][1].elapsed_time(marks[-1][1]))
    named = {name: event for name, event in marks}
    if "deep_start" in named and "deep_done" in named:
        profile["deep_propagate_total_ms"] = float(named["deep_start"].elapsed_time(named["deep_done"]))


# -----------------------------------------------------------------------------
# Slot assembly: DA3 raw output -> (B, T_slots, V, slots, D) in [action, CLS, patches] order
# -----------------------------------------------------------------------------


def extract_level0_slots(
    raw_level0: torch.Tensor,        # (B, T*V, total_tokens, 2*embed_dim) from encode_with_actions
    current_norm_action: torch.Tensor,   # kept for legacy call-site compatibility
    T: int,
    V: int,
    embed_dim: int = 1536,
    num_register_tokens: int = 4,
) -> torch.Tensor:
    """Assemble Level-0 slot tensor in [action, CLS, patches] order.

    DA3 layout in raw tokens (post-action-injection):
        pos 0:                        camera / CLS token
        pos 1:                        action token
        pos 2 .. 1+num_register:      register tokens
        pos 2+num_register .. end:    patch tokens (256 for 224/14)

    Our predictor expects, per slot:
        pos 0: action
        pos 1: CLS
        pos 2..257: 256 patches

    Args:
        raw_level0: (B, T*V, total_tokens, 2*embed_dim) : the `raw` output
            from DA3's `_run_backbone` at Level 0 (layer 19). First half is
            the pre-global `local_x` cache from the previous local block,
            matching Robot-GLD's Level-0 latent convention.
        current_norm_action: kept for call-site compatibility. The predictor
            slot action seed is read from Level-0 `local_x` position 1.
        T, V: timesteps and views.

    Returns:
        (B, T, V, slots, D) tensor.
    """
    b = raw_level0.shape[0]
    tokens_per_view = raw_level0.shape[2]
    _ = current_norm_action
    full = raw_level0[..., :embed_dim]            # (B, T*V, tokens_per_view, D)

    # Derive slot indices.
    cls_idx = 0
    reg_start = 2
    reg_end = reg_start + num_register_tokens
    patches_start = reg_end
    num_patches = tokens_per_view - patches_start

    cls_tok = full[:, :, cls_idx:cls_idx + 1, :]                       # (B, T*V, 1, d)
    patches = full[:, :, patches_start:, :]                             # (B, T*V, 256, d)
    action = full[:, :, 1:2, :]                                          # (B, T*V, 1, d)

    assembled = torch.cat([action, cls_tok, patches], dim=2)            # (B, T*V, 258, d)
    return assembled.reshape(b, T, V, 1 + 1 + num_patches, -1)


def extract_level0_slots_no_action(
    raw_level0: torch.Tensor,        # (B, T*V, total_tokens, 2*embed_dim) from encode_all_levels_raw
    T: int,
    V: int,
    embed_dim: int = 1536,
    num_register_tokens: int = 4,
) -> torch.Tensor:
    """Assemble frozen-teacher Level-0 slots in [dummy_action, CLS, patches] order.

    The frozen teacher has no action tokens. Future feature supervision uses
    only patch positions, but returning the same slot shape keeps monitor and
    debug paths aligned with predictor outputs.
    """
    b = raw_level0.shape[0]
    full = raw_level0[..., :embed_dim]            # (B, T*V, tokens_per_view, D)

    cls_idx = 0
    patches_start = 1 + num_register_tokens
    cls_tok = full[:, :, cls_idx:cls_idx + 1, :]
    patches = full[:, :, patches_start:, :]
    action = torch.zeros_like(cls_tok)

    assembled = torch.cat([action, cls_tok, patches], dim=2)
    return assembled.reshape(b, T, V, 1 + 1 + patches.shape[2], -1)


# -----------------------------------------------------------------------------
# Forward + loss
# -----------------------------------------------------------------------------


def _slots_to_da3_propagation_inputs(
    slots: torch.Tensor,
    embed_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert `[action, cls, patches]` slots to DA3 propagation inputs.

    The D-d slot is the Robot-GLD-style Level-0 pre-global `local_x` seed.
    DA3 propagation initializes the current stream from the same seed, replays
    the Level-0 global block, then continues through the remaining blocks.
    """
    b, steps, v_count, n_tokens, dim = slots.shape
    flat = slots.reshape(b * steps * v_count, n_tokens, dim)
    action = flat[:, 0, :]
    cls = flat[:, 1, :]
    patches = flat[:, 2:, :]
    expected = int(embed_dim) * 2
    if patches.shape[-1] not in (int(embed_dim), expected):
        raise ValueError(
            f"Cannot decode future slots with dim={patches.shape[-1]}; expected {embed_dim} or {expected}."
        )
    return patches, cls, action


def _slots_to_da3_decode_inputs(
    slots: torch.Tensor,
    embed_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert `[action, cls, patches]` slots to DPT-only DA3 inputs."""
    patches, cls, _action = _slots_to_da3_propagation_inputs(slots, embed_dim)
    return patches, cls


def _resize_depth_target(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tuple(target.shape[-2:]) == tuple(size):
        return target, mask
    mask_bool = mask.bool()
    target = torch.where(
        mask_bool & torch.isfinite(target),
        target.float(),
        torch.zeros_like(target, dtype=torch.float32),
    )
    flat_target = target.reshape(-1, 1, *target.shape[-2:])
    flat_mask = mask_bool.reshape(-1, 1, *mask.shape[-2:]).float()
    resized_target = F.interpolate(flat_target * flat_mask, size=size, mode="bilinear", align_corners=False)
    resized_mask = F.interpolate(flat_mask, size=size, mode="bilinear", align_corners=False)
    resized_target = resized_target / resized_mask.clamp_min(1e-6)
    target_out = resized_target.reshape(*target.shape[:-2], *size)
    mask_out = (resized_mask.reshape(*mask.shape[:-2], *size) > 0.5)
    target_out = torch.where(mask_out, target_out, torch.zeros_like(target_out))
    return target_out, mask_out


def _resize_bool_mask(
    mask: torch.Tensor,
    size: Tuple[int, int],
    *,
    strict: bool = False,
) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask.bool()
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    resized = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    threshold = 0.999 if strict else 0.5
    return (resized.reshape(*mask.shape[:-2], *size) > threshold)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mask = mask.bool() & torch.isfinite(values)
    values = torch.where(mask, values, torch.zeros_like(values))
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def _depth_gradient_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    dx_mask = mask[..., :, 1:] & mask[..., :, :-1]
    if dx_mask.any():
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        losses.append(_masked_mean((pred_dx - target_dx).abs(), dx_mask))
    dy_mask = mask[..., 1:, :] & mask[..., :-1, :]
    if dy_mask.any():
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        losses.append(_masked_mean((pred_dy - target_dy).abs(), dy_mask))
    if not losses:
        return pred.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(losses).sum()


# -----------------------------------------------------------------------------
# DA3 paper-style helpers (arXiv 2511.10647 Section 3.3). Kept grouped with the
# depth loss so both the subset and the full variant live in one place.
# -----------------------------------------------------------------------------


def _softplus_confidence(raw: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Map raw DPT confidence logits to strictly-positive confidence weights.

    Guarantees ``D_c > 0`` so ``log D_c`` never hits ``-inf`` and the paper's
    log-penalty regularizer is numerically stable.
    """
    return F.softplus(raw.float()) + float(floor)


def _confidence_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    lambda_conf: float,
) -> torch.Tensor:
    """DA3 Eq L_D: ``(1/|Ω|) Σ m_p (D_c · |D̂ − D| − λ_c · log D_c)``.

    ``conf`` must already be strictly positive (use ``_softplus_confidence``).
    The reduction averages over the valid-mask support to keep scale invariant
    to valid-pixel count.
    """
    conf = conf.float()
    abs_err = (pred.float() - target.float()).abs()
    per_pixel = conf * abs_err - float(lambda_conf) * torch.log(conf)
    return _masked_mean(per_pixel, mask)


def _pixel_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the homogeneous pixel-centre grid ``[u+0.5, v+0.5, 1]`` with shape (H, W, 3)."""
    vs = torch.arange(H, device=device, dtype=dtype) + 0.5
    us = torch.arange(W, device=device, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    ones = torch.ones_like(grid_u)
    return torch.stack([grid_u, grid_v, ones], dim=-1)  # (H, W, 3)


def _compute_gt_ray_map(
    intrinsics: torch.Tensor,
    extrinsics_c2w: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Build the DA3 ray map ``R ∈ R^{H×W×6} = concat(origin_world, dir_world)``.

    Args:
        intrinsics: ``(..., 3, 3)`` camera intrinsics for the target resolution.
        extrinsics_c2w: ``(..., 4, 4)`` camera-to-world transform.
        H, W: ray-map spatial resolution.

    Returns:
        ``(..., H, W, 6)`` tensor. ``origin_world`` is broadcast to every pixel,
        ``dir_world`` is L2-normalized.
    """
    K = intrinsics.float()
    E = extrinsics_c2w.float()
    device = K.device
    dtype = K.dtype
    pix = _pixel_grid(H, W, device=device, dtype=dtype)  # (H, W, 3)
    # K^-1 on each batch element : shape broadcasting via `torch.linalg.inv`.
    K_inv = torch.linalg.inv(K)  # (..., 3, 3)
    # dir_cam = K_inv @ [u, v, 1]^T, then rotate by R_c2w into world
    # pix has shape (H, W, 3); unsqueeze for matmul.
    pix_flat = pix.reshape(-1, 3).transpose(0, 1)  # (3, H*W)
    dir_cam = K_inv @ pix_flat                     # (..., 3, H*W)
    R_c2w = E[..., :3, :3]                         # (..., 3, 3)
    dir_world = R_c2w @ dir_cam                    # (..., 3, H*W)
    dir_world = dir_world / dir_world.norm(dim=-2, keepdim=True).clamp_min(1e-8)
    dir_world = dir_world.transpose(-1, -2).reshape(*E.shape[:-2], H, W, 3)
    origin = E[..., :3, 3]                         # (..., 3)
    origin_map = origin.unsqueeze(-2).unsqueeze(-2).expand(*E.shape[:-2], H, W, 3)
    return torch.cat([origin_map, dir_world], dim=-1)  # (..., H, W, 6)


def _unproject_depth_to_pointmap(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_c2w: torch.Tensor,
) -> torch.Tensor:
    """Unproject a depth map to a world-frame 3D point map.

    Args:
        depth: ``(..., H, W)`` metric depth (meters).
        intrinsics: ``(..., 3, 3)`` for the same H, W.
        extrinsics_c2w: ``(..., 4, 4)``.

    Returns:
        ``(..., H, W, 3)`` world-frame 3D points.
    """
    depth = depth.float()
    K = intrinsics.float()
    E = extrinsics_c2w.float()
    H, W = depth.shape[-2], depth.shape[-1]
    pix = _pixel_grid(H, W, device=depth.device, dtype=depth.dtype)  # (H, W, 3)
    K_inv = torch.linalg.inv(K)
    pix_flat = pix.reshape(-1, 3).transpose(0, 1)         # (3, H*W)
    dir_cam = K_inv @ pix_flat                            # (..., 3, H*W)
    dir_cam = dir_cam.transpose(-1, -2).reshape(*K.shape[:-2], H, W, 3)
    pts_cam = dir_cam * depth.unsqueeze(-1)               # (..., H, W, 3)
    R_c2w = E[..., :3, :3]
    t_c2w = E[..., :3, 3]
    # pts_world = R_c2w @ pts_cam^T + t_c2w, per pixel
    pts_flat = pts_cam.reshape(*pts_cam.shape[:-3], -1, 3)      # (..., H*W, 3)
    pts_world = pts_flat @ R_c2w.transpose(-1, -2)              # rotate
    pts_world = pts_world + t_c2w.unsqueeze(-2)                 # translate
    return pts_world.reshape(*pts_cam.shape)


def da3_full_depth_loss(
    pred_depth: torch.Tensor,
    pred_depth_conf: torch.Tensor,
    pred_ray: torch.Tensor,
    pred_ray_conf: torch.Tensor,
    target_depth_da3: torch.Tensor,
    target_mask: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_extrinsics_c2w: torch.Tensor,
    scene_scale: torch.Tensor,
    *,
    lambda_grad: float = 1.0,
    lambda_conf: float = 0.2,
    lambda_ray: float = 1.0,
    lambda_point: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full DA3 student depth loss from arXiv 2511.10647 Section 3.3.

    Implements ``L = L_D + L_M + L_P + α·L_grad`` with α=1 fixed (matches
    paper) and per-term knobs for the confidence log-penalty weight, the ray
    weight, and the point-map weight. ``L_C`` (camera regression) is computed
    separately by the caller because it depends on the encoder's ``cam_dec``
    output rather than the DPT head.

    Shapes (let ``F = flat batch-time-view``, ``H, W`` = depth resolution,
    ``Hr, Wr`` = ray resolution):
        pred_depth          (F, H, W)
        pred_depth_conf     (F, H, W)
        pred_ray            (F, Hr, Wr, 6)
        pred_ray_conf       (F, Hr, Wr)
        target_depth_da3    (F, H, W)           : already normalized by scene_scale
        target_mask         (F, H, W)           : bool
        target_intrinsics   (F, 3, 3)           : in pixel units of the 224x224 crop
        target_extrinsics_c2w (F, 4, 4)
        scene_scale         (F,) or scalar tensor
    """
    assert pred_depth.shape == target_depth_da3.shape, (
        f"pred_depth {tuple(pred_depth.shape)} vs target_depth_da3 "
        f"{tuple(target_depth_da3.shape)} must match."
    )

    device = pred_depth.device
    pred_depth = pred_depth.float()
    target_depth_da3 = target_depth_da3.float().to(device)
    target_mask = target_mask.bool().to(device)
    H_d, W_d = pred_depth.shape[-2:]

    # --- L_D (confidence-weighted depth L1 + log penalty) ---
    valid_depth = (
        target_mask
        & torch.isfinite(pred_depth)
        & torch.isfinite(target_depth_da3)
        & (target_depth_da3 > 0)
    )
    depth_conf = _softplus_confidence(pred_depth_conf.to(device))
    loss_depth = _confidence_weighted_l1(
        pred_depth, target_depth_da3, depth_conf, valid_depth, lambda_conf,
    )

    # --- L_grad (finite-diff gradient L1) ---
    loss_grad = _depth_gradient_l1(pred_depth, target_depth_da3, valid_depth)

    # --- L_M (ray map: confidence-weighted L1 + log penalty) ---
    # DPT ray output is downsampled (128x128 for 224 inputs). Build GT at that
    # resolution by transforming intrinsics to the ray resolution.
    Hr, Wr = pred_ray.shape[-3:-1]
    K = target_intrinsics.float().to(device)
    E = target_extrinsics_c2w.float().to(device)
    # Scale intrinsics from depth resolution to ray resolution.
    scale_u = float(Wr) / float(W_d)
    scale_v = float(Hr) / float(H_d)
    K_ray = K.clone()
    K_ray[..., 0, 0] = K[..., 0, 0] * scale_u
    K_ray[..., 0, 2] = K[..., 0, 2] * scale_u
    K_ray[..., 1, 1] = K[..., 1, 1] * scale_v
    K_ray[..., 1, 2] = K[..., 1, 2] * scale_v
    gt_ray = _compute_gt_ray_map(K_ray, E, Hr, Wr)  # (F, Hr, Wr, 6)
    ray_conf = _softplus_confidence(pred_ray_conf.to(device))
    ray_valid = torch.ones_like(ray_conf, dtype=torch.bool)
    # L1 per-pixel across the 6 channels, averaged.
    ray_abs = (pred_ray.float() - gt_ray).abs().mean(dim=-1)
    per_pixel_ray = ray_conf * ray_abs - float(lambda_conf) * torch.log(ray_conf)
    loss_ray = _masked_mean(per_pixel_ray, ray_valid)

    # --- L_P (point-map L1 in world meters) ---
    # Restore predicted depth to meters (student is trained in DA3-normalized
    # units so multiply back by the per-sample scene_scale).
    if scene_scale.ndim == 0:
        ss = scene_scale.to(device).float().expand(pred_depth.shape[0])
    else:
        ss = scene_scale.to(device).float()
    ss_view = ss.view(-1, *([1] * (pred_depth.ndim - 1)))
    pred_depth_m = pred_depth * ss_view
    target_depth_m = target_depth_da3 * ss_view
    pred_points = _unproject_depth_to_pointmap(pred_depth_m, K, E)
    target_points = _unproject_depth_to_pointmap(target_depth_m, K, E)
    point_err = (pred_points - target_points).abs().mean(dim=-1)  # (F, H, W)
    loss_point = _masked_mean(point_err, valid_depth)

    # --- Combine with paper's α=1 for gradient; per-term lambdas for the rest ---
    loss_total = (
        loss_depth
        + float(lambda_grad) * loss_grad
        + float(lambda_ray) * loss_ray
        + float(lambda_point) * loss_point
    )
    metrics = {
        "depth_l1": loss_depth.detach(),
        "depth_grad_l1": loss_grad.detach(),
        "depth_valid_ratio": valid_depth.float().mean().detach(),
        "ray_l1": loss_ray.detach(),
        "ray_conf_mean": ray_conf.float().mean().detach(),
        "depth_conf_mean": depth_conf.float().mean().detach(),
        "point_l1": loss_point.detach(),
    }
    return loss_total, metrics


def da3_style_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    grad_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Depth + finite-difference gradient loss on DA3-normalized targets."""
    if pred_depth.shape[-2:] != target_depth.shape[-2:]:
        target_depth, target_mask = _resize_depth_target(
            target_depth,
            target_mask,
            pred_depth.shape[-2:],
        )
    valid = (
        target_mask.bool()
        & torch.isfinite(pred_depth)
        & torch.isfinite(target_depth)
        & (target_depth > 0)
        & (pred_depth > 0)
    )
    depth_l1 = _masked_mean((pred_depth.float() - target_depth.float()).abs(), valid)
    grad_l1 = _depth_gradient_l1(pred_depth.float(), target_depth.float(), valid)
    loss = depth_l1 + float(grad_weight) * grad_l1
    metrics = {
        "depth_l1": depth_l1.detach(),
        "depth_grad_l1": grad_l1.detach(),
        "depth_valid_ratio": valid.float().mean().detach(),
    }
    return loss, metrics


def compute_unified_forward_loss(
    student_da3: nn.Module,                    # trainable DA3GiantEncoder
    teacher_da3: nn.Module,                    # frozen DA3GiantEncoder (no grads)
    future_predictor: nn.Module,               # FuturePredictor
    text_conditioner: Optional[nn.Module],     # TextConditioner or None
    action_head: nn.Module,                    # ActionHeadV2 (or OFT)
    regularizer,                               # FeatureRegularizer for past feat reg
    all_views_norm: torch.Tensor,              # (B, T*V, 3, H_img, W_img) normalized images
    gt_actions: torch.Tensor,                  # (B, T, 7) normalized GT actions
    proprio: torch.Tensor,                     # (B, proprio_dim)
    language_texts: Optional[List[str]],       # list[B] or None
    T: int,
    V: int,
    H: int,
    lambda_action: float,
    lambda_feat_future: float,
    lambda_sigreg: float,
    use_bf16: bool,
    target_all_views_norm: Optional[torch.Tensor] = None,  # geometry-only teacher target images
    lambda_depth: float = 0.0,
    gt_depth_da3: Optional[torch.Tensor] = None,
    gt_depth_mask: Optional[torch.Tensor] = None,
    depth_grad_weight: float = 1.0,
    depth_decode_chunk_size: int = 1,
    depth_future_steps: int = 0,
    depth_decode_context: str = "full_sequence",
    num_register_tokens: int = 4,
    embed_dim: int = 1536,
) -> Dict[str, torch.Tensor]:
    """One step of unified forward + loss computation.

    Returns dict with keys:
      - `loss_total`: scalar total loss (requires grad).
      - `loss_action`, `loss_feat_future`, `loss_feat_past`, `loss_sigreg`,
        `loss_depth`, `loss_camera`: scalar components.
      - `action_pred`: (B, F+1, 7) predicted future actions (including boundary).
      - `gt_actions_future`: (B, F+1, 7) GT for loss/metrics.
      - `z_future_student`: (B, F, V, 258, 1536) predicted future features.
      - `z_future_teacher`: detached teacher targets.
      - `student_raw`: list of raw levels (for existing reg/monitoring hooks).
      - `teacher_past_raw`: past-only teacher raw (for logging).
    """
    device = all_views_norm.device
    b = all_views_norm.shape[0]
    F_ = T - H
    target_views_norm = all_views_norm if target_all_views_norm is None else target_all_views_norm
    # Channel layout inside the flat view dim.
    # all_views_norm is laid out as (B, T*V, 3, H, W) with block-t-major (t outer, v inner).
    # First H*V entries are the past.
    past_views = all_views_norm[:, : H * V]
    target_past_views = target_views_norm[:, : H * V]
    assert past_views.shape[1] == H * V, (
        f"past slice: {past_views.shape[1]} vs H*V={H * V}"
    )

    # Per-slot AE mask: strict-past slots (< H-1) get AE=True; current boundary + future noact=False.
    # Student only sees H past slots; its AE mask has length H.
    student_ae = torch.zeros(H, dtype=torch.bool, device=device)
    if H > 1:
        student_ae[: H - 1] = True

    # -------- Student DA3 pass on past H frames --------
    past_gt_actions = gt_actions[:, :H]   # (B, H, 7)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        student_feats, student_action_tokens, student_raw = student_da3.encode_with_actions(
            past_views,
            action_input=past_gt_actions,
            override_n_steps=H,
            per_slot_ae_mask=student_ae,
        )
    # student_feats: list of (patches, cls) per level
    # student_action_tokens: (B, H*V, 1536) : final normalized action tokens.
    # student_raw: list of raw tensors

    # -------- Teacher / target passes (no grad) --------
    # Two frozen-teacher targets:
    #   (a) past feature_reg target: teacher on past-only frames, so no future
    #       information leaks into the past regularizer.
    #   (b) future feature_l2 target: teacher on all T frames; this target is
    #       fully frozen and matches the past-reg reference model.
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        teacher_past_raw = teacher_da3.encode_all_levels_raw(target_past_views)
        teacher_full_raw = teacher_da3.encode_all_levels_raw(target_views_norm)

    # Slice teacher_past_raw to match Student's level shapes. The existing
    # FeatureRegularizer handles a list of levels. Its internal indexing assumes
    # student_raw tokens include the action slot, while teacher_past_raw from
    # encode_all_levels_raw follows the current Stage 1 convention.

    # -------- Assemble past tokens for the predictor --------
    # We need the Level-0 slot tensor for the student: shape (B, H, V, 258, 1536).
    # Student's `raw_levels[0]` has shape (B, H*V, total_tokens, 2*embed_dim).
    past_slots = extract_level0_slots(
        raw_level0=student_raw[0],
        current_norm_action=student_action_tokens,
        T=H, V=V,
        embed_dim=embed_dim,
        num_register_tokens=num_register_tokens,
    )  # (B, H, V, 258, 1536)

    # -------- Teacher future target slots --------
    # From frozen teacher raw Level 0: (B, T*V, total_tokens, 2*embed_dim).
    # The teacher has no action slot; we add a dummy action position for shape
    # compatibility and supervise patches only below.
    with torch.no_grad():
        teacher_slots_all = extract_level0_slots_no_action(
            raw_level0=teacher_full_raw[0],
            T=T, V=V,
            embed_dim=embed_dim,
            num_register_tokens=num_register_tokens,
        )   # (B, T, V, 258, 1536)
        teacher_future_slots = teacher_slots_all[:, H:, :, :, :].detach()   # (B, F, V, 258, 1536)

    # -------- Language tokens --------
    lang_feats = None
    lang_pad_mask = None
    if text_conditioner is not None and language_texts is not None:
        tok_out = text_conditioner.encode_tokens(language_texts, pad_to=future_predictor.language_len)
        lang_feats = tok_out["last_hidden_state"]       # (B, L, 768)
        lang_pad_mask = tok_out["attention_mask"]       # (B, L) bool

    # Datasets provide proprio as a per-timestep sequence (B, T, 7). The
    # predictor's AdaLN condition is a single boundary state: last observed H.
    proprio_cond = proprio[:, H - 1, :] if proprio.ndim == 3 else proprio

    # -------- Predictor forward --------
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        pred_out = future_predictor(
            past_tokens=past_slots,
            proprio=proprio_cond,
            F_=F_,
            lang_feats=lang_feats,
            lang_padding_mask=lang_pad_mask,
        )
    z_future = pred_out["z_future"]                    # (B, F, V, 258, 1536)
    sigreg_loss = pred_out.get("sigreg_loss", None)

    # -------- Action predictions --------
    # Boundary action (slot H-1) already comes from the student DA3 final action
    # token. Future action slots are Level-0 seeds, so they must be propagated
    # through DA3 before ActionHeadV2 sees them.
    student_action_4d = student_action_tokens.reshape(b, H, V, -1)
    boundary_action_tokens = student_action_4d[:, -1:, :, :]   # (B, 1, V, 1536)
    future_action_tokens = None
    if F_ > 0:
        action_decode_slots = torch.cat([past_slots.detach(), z_future], dim=1)
        patches_for_action, cls_for_action, action_for_action = _slots_to_da3_propagation_inputs(
            action_decode_slots,
            embed_dim,
        )
        propagated = student_da3.propagate_and_predict_grad(
            patches_for_action,
            cls_for_action,
            action_for_action,
            total_view=(H + F_) * V,
            action_head=None,
            cond_num=H * V,
            decode_visuals=False,
        )
        if "action_tokens" not in propagated:
            raise RuntimeError("DA3 propagation did not return action tokens.")
        propagated_actions = propagated["action_tokens"].reshape(b, H + F_, V, -1)
        future_action_tokens = propagated_actions[:, H:, :, :].contiguous()

    if future_action_tokens is not None:
        all_future_action_tokens = torch.cat([boundary_action_tokens, future_action_tokens], dim=1)
    else:
        all_future_action_tokens = boundary_action_tokens
    # ActionHeadV2 expects (B, T, V, D) or (B, S, D). Feed 4D.
    action_pred = action_head(all_future_action_tokens)   # (B, F+1, 7)
    gt_actions_future = gt_actions[:, H - 1:]              # (B, F+1, 7)

    loss_action = F.l1_loss(action_pred.float(), gt_actions_future.float())

    # -------- Future feature L2 --------
    # Student predictor output vs teacher future target. Compare patches only (skip action/CLS
    # because those are supervised via action_L1 and feature-reg respectively).
    student_future_patches = z_future[:, :, :, 2:, :]      # (B, F, V, 256, 1536)
    teacher_future_patches = teacher_future_slots[:, :, :, 2:, :]
    loss_feat_future = F.mse_loss(
        student_future_patches.float(),
        teacher_future_patches.float(),
    )

    # -------- Past feature reg (vs past-only teacher) --------
    loss_feat_past = regularizer.feature_reg_loss(
        [feat.float() for feat in student_raw],
        [feat.float() for feat in teacher_past_raw],
    )

    # -------- SIGReg --------
    if sigreg_loss is None:
        sigreg_loss = past_slots.new_tensor(0.0, dtype=torch.float32)
    else:
        sigreg_loss = sigreg_loss.float()

    # -------- Optional DA3-style future depth target --------
    loss_depth = past_slots.new_tensor(0.0, dtype=torch.float32)
    depth_metrics: Dict[str, torch.Tensor] = {}
    if float(lambda_depth) > 0.0:
        if gt_depth_da3 is None or gt_depth_mask is None:
            raise ValueError("lambda_depth > 0 requires gt_depth_da3 and gt_depth_mask in the batch.")
        target_depth = gt_depth_da3[:, H:, :, :, :].to(device=device, dtype=torch.float32)
        target_mask = gt_depth_mask[:, H:, :, :, :].to(device=device).bool()
        if target_depth.shape[:3] != z_future.shape[:3]:
            raise ValueError(
                "GT depth target shape mismatches predicted future slots: "
                f"target={tuple(target_depth.shape[:3])} z_future={tuple(z_future.shape[:3])}"
            )
        z_depth = z_future
        max_depth_steps = int(depth_future_steps)
        if max_depth_steps > 0 and z_depth.shape[1] > max_depth_steps:
            z_depth = z_depth[:, :max_depth_steps]
            target_depth = target_depth[:, :max_depth_steps]
            target_mask = target_mask[:, :max_depth_steps]

        depth_context = str(depth_decode_context)
        if depth_context == "future_only":
            decode_slots = z_depth
            decode_t = z_depth.shape[1]
        elif depth_context == "full_sequence":
            decode_slots = torch.cat([past_slots.detach(), z_depth], dim=1)
            decode_t = H + z_depth.shape[1]
        else:
            raise ValueError(f"Unsupported depth_decode_context={depth_decode_context!r}")

        patches_for_depth, cls_for_depth = _slots_to_da3_decode_inputs(decode_slots, embed_dim)
        decoded = student_da3.propagate_and_decode_grad(
            patches_for_depth,
            cls_for_depth,
            total_view=decode_t * V,
            dpt_chunk_size=depth_decode_chunk_size,
        )
        if "depth" not in decoded:
            raise RuntimeError("DA3 depth head did not return a depth map.")
        pred_depth = decoded["depth"]
        if pred_depth.ndim == 4 and pred_depth.shape[1] == 1:
            pred_depth = pred_depth[:, 0]
        pred_depth = pred_depth.reshape(b, decode_t, V, *pred_depth.shape[-2:])
        if depth_context == "full_sequence":
            pred_depth = pred_depth[:, H:]
        loss_depth, depth_metrics = da3_style_depth_loss(
            pred_depth=pred_depth,
            target_depth=target_depth,
            target_mask=target_mask,
            grad_weight=depth_grad_weight,
        )
        depth_metrics["depth_supervised_steps"] = pred_depth.new_tensor(float(z_depth.shape[1])).detach()
        depth_metrics["depth_decode_slots"] = pred_depth.new_tensor(float(decode_t * V)).detach()
    loss_camera = past_slots.new_tensor(0.0, dtype=torch.float32)

    loss_total = (
        lambda_action * loss_action
        + lambda_feat_future * loss_feat_future
        + regularizer.lambda_feat * loss_feat_past
        + lambda_sigreg * sigreg_loss
        + float(lambda_depth) * loss_depth
    )

    return {
        "loss_total": loss_total,
        "loss_action": loss_action,
        "loss_feat_future": loss_feat_future,
        "loss_feat_past": loss_feat_past,
        "loss_sigreg": sigreg_loss,
        "loss_depth": loss_depth,
        "loss_camera": loss_camera,
        "depth_metrics": depth_metrics,
        "action_pred": action_pred,
        "gt_actions_future": gt_actions_future,
        "z_future_student": z_future,
        "z_future_teacher": teacher_future_slots,
        "past_slots": past_slots,
        "student_feats": student_feats,
        "student_raw": student_raw,
        "teacher_past_raw": teacher_past_raw,
        "boundary_action_tokens": boundary_action_tokens,
        "future_action_tokens": future_action_tokens,
        "H": H, "F_": F_, "T": T, "V": V,
    }


def compute_gam_forward_loss(
    student_da3: nn.Module,
    teacher_da3: nn.Module,
    future_predictor: nn.Module,
    text_conditioner: Optional[nn.Module],
    action_head: nn.Module,
    proprio_head: Optional[nn.Module],
    regularizer,
    all_views_norm: torch.Tensor,
    gt_actions: torch.Tensor,
    proprio: torch.Tensor,
    language_texts: Optional[List[str]],
    T: int,
    V: int,
    H: int,
    lambda_action: float,
    lambda_feat_future: float,
    lambda_proprio_future: float,
    lambda_sigreg: float,
    use_bf16: bool,
    teacher_views_norm: Optional[torch.Tensor] = None,
    teacher_depth_valid_mask: Optional[torch.Tensor] = None,
    lambda_action_direct: float = 1.0,
    lambda_action_refine: float = 1.0,
    lambda_depth: float = 0.0,
    gt_depth_da3: Optional[torch.Tensor] = None,
    gt_depth_mask: Optional[torch.Tensor] = None,
    depth_grad_weight: float = 1.0,
    depth_decode_chunk_size: int = 1,
    teacher_depth_fallback: bool = False,
    skip_depth_if_no_gt: bool = False,
    lambda_path_b_deep_feat_reg: float = 0.0,
    deep_feat_reg_layer_weight_min: float = 0.5,
    lambda_feat_current: float = 1.0,
    shallow_layer: Optional[int] = None,
    embed_dim: int = 1536,
    deep_gradient_checkpointing: bool = False,
    deep_temporal_causal_mask: bool = False,
    feature_channel_stats: Optional[Dict[str, torch.Tensor]] = None,
    feature_loss_norm: str = "none",
    feature_loss_norm_eps: float = 1e-6,
    feature_target_mode: str = "future",
    feature_loss_type: str = "l2",
    # --- DA3 paper full depth-loss knobs (arXiv 2511.10647 Section 3.3) ---
    depth_loss_type: str = "da3_style",
    lambda_depth_conf: float = 0.2,
    lambda_ray: float = 0.0,
    lambda_point: float = 0.0,
    gt_camera_intrinsics: Optional[torch.Tensor] = None,
    gt_camera_extrinsics_c2w: Optional[torch.Tensor] = None,
    gt_depth_scene_scale: Optional[torch.Tensor] = None,
    prev_action_mask_rate: float = 0.0,
    prev_action_mask_include_t0: bool = False,
    action_loss_mask: Optional[torch.Tensor] = None,
    transition_loss_mask: Optional[torch.Tensor] = None,
    context_valid_mask: Optional[torch.Tensor] = None,
    view_valid_mask: Optional[torch.Tensor] = None,
    past_action_history: Optional[torch.Tensor] = None,
    forward_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, torch.Tensor]:
    """Unified gam AR loss.

    Observed pre-`alt_start` visual features are consumed by the predictor as
    dense timestep blocks `[o_t, s_t, a_{t-1}]`. The predictor emits the next-step
    block `[o_{t+1}, s_{t+1}, a_t]` for every observed timestep, and the selected
    DA3 deep stack consumes the full predicted sequence directly.

    `prev_action_mask_rate` (default 0): scheduled-sampling-style masking on
    past_action_history during training to mitigate exposure bias. With prob p
    each (b, t>=1) chunk slot is zeroed (matching the rollout case where the
    model's predicted previous action may differ from the training GT). Dataset
    loaders may provide episode-aligned previous actions; in that case only
    true pre-episode slots are zero.
    """
    if float(lambda_feat_current) != 0.0:
        raise ValueError(
            "gam keeps the current visual state observed; "
            "set predictor.lambda_feat_current=0.0."
        )
    if H >= T:
        raise ValueError(f"gam requires H < T so t+1 exists; got H={H}, T={T}.")
    device = all_views_norm.device
    b = all_views_norm.shape[0]
    teacher_views_norm = all_views_norm if teacher_views_norm is None else teacher_views_norm
    if teacher_depth_valid_mask is not None:
        teacher_depth_valid_mask = teacher_depth_valid_mask.to(device=device, dtype=torch.bool)
    past_views = all_views_norm[:, : H * V]
    context_valid_mask_used = None
    if context_valid_mask is not None:
        context_valid_mask_used = context_valid_mask[:, :H].to(device=device, dtype=torch.bool)
    action_loss_mask_used = None
    if action_loss_mask is not None:
        action_loss_mask_used = action_loss_mask[:, :H].to(device=device, dtype=torch.bool)
    transition_loss_mask_used = None
    if transition_loss_mask is not None:
        transition_loss_mask_used = transition_loss_mask[:, 1 : H + 1].to(device=device, dtype=torch.bool)
    view_valid_all = None
    past_view_valid = None
    future_view_valid = None
    current_view_valid = None
    if view_valid_mask is not None:
        view_valid_all = view_valid_mask[:, :T].to(device=device, dtype=torch.bool)
        if view_valid_all.shape[:3] != (b, T, V):
            raise ValueError(
                "view_valid_mask must be shaped "
                f"{(b, T, V)}, got {tuple(view_valid_all.shape)}."
            )
        past_view_valid = view_valid_all[:, :H]
        future_view_valid = view_valid_all[:, 1 : H + 1]
        current_view_valid = view_valid_all[:, H - 1 : H]

    def _combine_step_view_mask(
        step_mask: Optional[torch.Tensor],
        view_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if view_mask is None:
            return step_mask
        mask = view_mask
        if step_mask is not None:
            step = step_mask.to(device=mask.device, dtype=torch.bool)
            while step.ndim > 2:
                step = step.any(dim=-1)
            step = step[:, : mask.shape[1]]
            mask = mask & step.unsqueeze(-1)
        return mask

    feature_future_mask = _combine_step_view_mask(transition_loss_mask_used, future_view_valid)

    def _masked_reduce(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        value = value.float()
        if mask is None:
            return value.mean()
        mask_b = mask.to(device=value.device, dtype=torch.bool)
        while mask_b.ndim < value.ndim:
            mask_b = mask_b.unsqueeze(-1)
        mask_b = mask_b.expand_as(value)
        mask_f = mask_b.to(dtype=value.dtype)
        denom = mask_f.sum().clamp_min(1.0)
        value = torch.where(mask_b, value, torch.zeros_like(value))
        return value.sum() / denom

    def _masked_l1_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return _masked_reduce((pred.float() - target.float()).abs(), mask)

    def _masked_mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return _masked_reduce((pred.float() - target.float()).square(), mask)

    def _run_action_head(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return action_head(tokens)
        try:
            return action_head(tokens, view_mask=mask)
        except TypeError as exc:
            if bool(mask.to(dtype=torch.bool).all()):
                return action_head(tokens)
            raise TypeError(
                "Masked variable views require an action head that accepts view_mask "
                "(ActionHeadV2 pool_mode='mean')."
            ) from exc
    if shallow_layer is None:
        shallow_layer = int(getattr(student_da3, "shallow_target_layer", 12))
    else:
        shallow_layer = int(shallow_layer)

    _cuda_profile_mark(forward_profile, "fwd_start")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        student_shallow = student_da3.encode_shallow_visual_slots(
            past_views,
            T=H,
            V=V,
            target_layer=shallow_layer,
        )
    _cuda_profile_mark(forward_profile, "student_shallow_done")
    past_visual = student_shallow["visual_tokens"].detach()
    if context_valid_mask_used is not None:
        past_visual = past_visual * context_valid_mask_used[:, :, None, None, None].to(
            device=past_visual.device,
            dtype=past_visual.dtype,
        )
    if past_view_valid is not None:
        past_visual = past_visual * past_view_valid[:, :, :, None, None].to(
            device=past_visual.device,
            dtype=past_visual.dtype,
        )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        teacher_shallow = teacher_da3.encode_shallow_visual_slots(
            teacher_views_norm,
            T=T,
            V=V,
            target_layer=shallow_layer,
        )
    _cuda_profile_mark(forward_profile, "teacher_shallow_done")
    teacher_visual_all = teacher_shallow["visual_tokens"].detach()
    if view_valid_all is not None:
        teacher_visual_all = teacher_visual_all * view_valid_all[:, :, :, None, None].to(
            device=teacher_visual_all.device,
            dtype=teacher_visual_all.dtype,
        )
    current_target = teacher_visual_all[:, H - 1 : H]
    future_target = teacher_visual_all[:, H : H + 1]
    future_targets_all = teacher_visual_all[:, 1 : H + 1]

    lang_feats = None
    lang_pad_mask = None
    if text_conditioner is not None and language_texts is not None:
        tok_out = text_conditioner.encode_tokens(language_texts, pad_to=future_predictor.language_len)
        lang_feats = tok_out["last_hidden_state"]
        lang_pad_mask = tok_out["attention_mask"]
    _cuda_profile_mark(forward_profile, "text_encode_done")

    if proprio.ndim == 3:
        proprio_history = proprio[:, :H, :]
        proprio_cond = proprio[:, H - 1, :]
    else:
        proprio_history = proprio[:, None, :].expand(-1, H, -1)
        proprio_cond = proprio
    if past_action_history is not None:
        past_action_history = past_action_history[:, :H].to(device=gt_actions.device, dtype=gt_actions.dtype)
    elif H > 1:
        zero_prev_action = torch.zeros_like(gt_actions[:, :1])
        past_action_history = torch.cat([zero_prev_action, gt_actions[:, : H - 1]], dim=1)
    else:
        past_action_history = torch.zeros_like(gt_actions[:, :1])
    if context_valid_mask_used is not None:
        context_mask = context_valid_mask_used.to(device=proprio_history.device, dtype=proprio_history.dtype)
        proprio_history = proprio_history * context_mask.unsqueeze(-1)
        proprio_cond = proprio_cond * context_mask[:, -1:].to(
            device=proprio_cond.device,
            dtype=proprio_cond.dtype,
        )
        action_mask = context_valid_mask_used.to(
            device=past_action_history.device,
            dtype=past_action_history.dtype,
        )
        while action_mask.ndim < past_action_history.ndim:
            action_mask = action_mask.unsqueeze(-1)
        past_action_history = past_action_history * action_mask

    # Scheduled-sampling-style masking on past actions (training only; rollout
    # always uses model's own predicted prev actions). With prob p, zero out
    # the t>=1 slots so the model learns surrounding context alongside
    # prev_action input. Reduces train/rollout distribution
    # gap while preserving any episode-start zero token.
    if float(prev_action_mask_rate) > 0.0 and (H > 1 or bool(prev_action_mask_include_t0)):
        # Per-(batch, timestep) mask, broadcast over chunk and action dims.
        keep_dims = (b, H) + (1,) * (past_action_history.ndim - 2)
        mask = (torch.rand(keep_dims, device=past_action_history.device) < float(prev_action_mask_rate))
        if not bool(prev_action_mask_include_t0):
            # Legacy: keep t=0 unmasked because episode-aligned dataset history
            # already populates it. Set prev_action_mask_include_t0=true to
            # also expose the model to zero prev_action at t=0, matching the
            # rollout-first-step distribution.
            mask[:, 0] = False
        past_action_history = torch.where(
            mask,
            torch.zeros_like(past_action_history),
            past_action_history,
        )
    _cuda_profile_mark(forward_profile, "condition_prep_done")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        pred_out = future_predictor(
            past_visual_tokens=past_visual,
            proprio=proprio_cond,
            proprio_history=proprio_history,
            past_action_history=past_action_history,
            lang_feats=lang_feats,
            lang_padding_mask=lang_pad_mask,
            context_valid_mask=context_valid_mask_used,
            view_valid_mask=past_view_valid,
        )
    _cuda_profile_mark(forward_profile, "predictor_done")
    current_observed = past_visual[:, H - 1 : H]
    predicted_next_visual_tokens = pred_out.get("predicted_next_visual_tokens", None)
    if predicted_next_visual_tokens is None:
        raise RuntimeError("gam AR predictor must return predicted_next_visual_tokens.")
    last_predicted_visual_tokens = predicted_next_visual_tokens[:, H - 1 : H]
    predicted_next_proprio = pred_out.get("predicted_next_proprio", None)
    last_predicted_proprio = predicted_next_proprio[:, H - 1 : H] if predicted_next_proprio is not None else None
    predicted_action_tokens = pred_out.get("predicted_action_tokens", None)
    sigreg_loss = pred_out.get("sigreg_loss", None)
    if future_view_valid is not None:
        predicted_next_visual_tokens = predicted_next_visual_tokens * future_view_valid[:, :, :, None, None].to(
            device=predicted_next_visual_tokens.device,
            dtype=predicted_next_visual_tokens.dtype,
        )

    if predicted_action_tokens is None:
        raise RuntimeError("gam AR predictor must return predicted_action_tokens.")
    predicted_action_tokens = predicted_action_tokens.to(past_visual.dtype)
    direct_action_view_mask = past_view_valid
    if direct_action_view_mask is not None:
        predicted_action_tokens = predicted_action_tokens * direct_action_view_mask[:, :, :, None].to(
            device=predicted_action_tokens.device,
            dtype=predicted_action_tokens.dtype,
        )
    predicted_actions_direct = None
    loss_action_direct = None
    loss_action_direct_for_total = past_visual.new_tensor(0.0, dtype=torch.float32)
    if float(lambda_action_direct) > 0.0:
        predicted_actions_direct = _run_action_head(predicted_action_tokens, direct_action_view_mask)
        loss_action_direct = _masked_l1_loss(
            predicted_actions_direct,
            gt_actions[:, :H],
            action_loss_mask_used,
        )
        loss_action_direct_for_total = loss_action_direct
    _cuda_profile_mark(forward_profile, "action_direct_done")

    predicted_sequence_start_timestep = 1
    deep_visual = predicted_next_visual_tokens
    deep_steps = int(deep_visual.shape[1])
    deep_actions = predicted_action_tokens
    deep_step_valid_mask = None
    if action_loss_mask_used is not None:
        deep_step_valid_mask = action_loss_mask_used
        while deep_step_valid_mask.ndim > 2:
            deep_step_valid_mask = deep_step_valid_mask.any(dim=-1)
    last_predicted_action_tokens = deep_actions[:, -1:]
    has_depth_target = gt_depth_da3 is not None and gt_depth_mask is not None
    decode_depth = float(lambda_depth) > 0.0 and (
        has_depth_target or bool(teacher_depth_fallback) or not bool(skip_depth_if_no_gt)
    )
    want_deep_feat_reg = float(lambda_path_b_deep_feat_reg) > 0.0
    deep_action_view_mask = None if future_view_valid is None else future_view_valid[:, :deep_steps]
    propagate_kwargs = {
        "decode_visuals": decode_depth,
        "dpt_chunk_size": depth_decode_chunk_size,
        "gradient_checkpointing": deep_gradient_checkpointing,
        "return_multi_level": want_deep_feat_reg,
        "step_valid_mask": deep_step_valid_mask,
        "deep_temporal_causal_mask": deep_temporal_causal_mask,
        "profile": forward_profile,
    }
    deep_out = student_da3.propagate_shallow_with_actions_grad(
        deep_visual,
        deep_actions,
        **propagate_kwargs,
    )
    _cuda_profile_mark(forward_profile, "deep_done")
    if "action_tokens" not in deep_out:
        raise RuntimeError("gam DA3 propagation did not return action tokens.")
    action_tokens = deep_out["action_tokens"].reshape(b, deep_steps, V, embed_dim)
    if deep_action_view_mask is not None:
        action_tokens = action_tokens * deep_action_view_mask[:, :, :, None].to(
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )
    predicted_actions_deep = _run_action_head(action_tokens, deep_action_view_mask)
    action_pred = predicted_actions_deep
    target_actions = gt_actions[:, :H]
    loss_action_refine = _masked_l1_loss(action_pred, target_actions, action_loss_mask_used)
    loss_action = (
        float(lambda_action_direct) * loss_action_direct_for_total
        + float(lambda_action_refine) * loss_action_refine
    )
    _cuda_profile_mark(forward_profile, "action_refine_loss_done")
    target_proprio_all = proprio[:, 1 : H + 1].float() if proprio.ndim == 3 else None
    target_last_proprio = target_proprio_all[:, H - 1 : H] if target_proprio_all is not None else None
    loss_proprio_future_direct = past_visual.new_tensor(0.0, dtype=torch.float32)
    loss_proprio_future_head = past_visual.new_tensor(0.0, dtype=torch.float32)
    proprio_predicted_from_deep = None
    if predicted_next_proprio is not None and target_proprio_all is not None:
        loss_proprio_future_direct = _masked_l1_loss(
            predicted_next_proprio,
            target_proprio_all,
            transition_loss_mask_used,
        )
    if proprio_head is not None and target_proprio_all is not None:
        proprio_pred_all = proprio_head(action_tokens)
        proprio_predicted_from_deep = proprio_pred_all[:, -1:]
        loss_proprio_future_head = _masked_l1_loss(
            proprio_pred_all,
            target_proprio_all,
            transition_loss_mask_used,
        )
    loss_proprio_future = loss_proprio_future_direct + loss_proprio_future_head
    _cuda_profile_mark(forward_profile, "proprio_loss_done")

    def _channel_norm(x: torch.Tensor) -> torch.Tensor:
        if feature_channel_stats is None:
            return x.float()
        mean = feature_channel_stats["mean"].to(device=x.device, dtype=torch.float32)
        std = feature_channel_stats["std"].to(device=x.device, dtype=torch.float32)
        view_shape = (1,) * (x.ndim - 1) + (mean.numel(),)
        return (x.float() - mean.reshape(view_shape)) / std.clamp_min(feature_loss_norm_eps).reshape(view_shape)

    feature_loss_type = str(feature_loss_type).lower()
    if feature_loss_type not in {"l2", "mse", "l1", "mae"}:
        raise ValueError(f"Unknown feature_loss_type={feature_loss_type!r}.")

    def _feature_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feature_loss_type in {"l1", "mae"}:
            return _masked_l1_loss(x, y, mask)
        return _masked_mse_loss(x, y, mask)

    feature_loss_norm = str(feature_loss_norm).lower()
    if feature_loss_norm in {"", "none", "raw"}:
        use_feature_channel_norm = False
    elif feature_loss_norm in {"channel", "channel_stats", "stats"}:
        if feature_channel_stats is None:
            raise ValueError("feature_loss_norm='channel_stats' requires feature_channel_stats.")
        use_feature_channel_norm = True
    else:
        raise ValueError(f"Unknown feature_loss_norm={feature_loss_norm!r}.")

    feature_target_mode = str(feature_target_mode).lower()
    if feature_target_mode not in {"future", "delta"}:
        raise ValueError(f"Unknown feature_target_mode={feature_target_mode!r}.")

    copy_current_future = past_visual
    if feature_target_mode == "delta":
        future_pred_target = predicted_next_visual_tokens.float() - copy_current_future.float()
        future_target_ref = future_targets_all.float() - copy_current_future.float()
        future_baseline_target = torch.zeros_like(future_target_ref)
    else:
        future_pred_target = predicted_next_visual_tokens.float()
        future_target_ref = future_targets_all.float()
        future_baseline_target = copy_current_future.float()

    loss_feat_current_raw = _feature_loss(current_observed.float(), current_target.float(), current_view_valid)
    loss_feat_future_raw = _feature_loss(future_pred_target, future_target_ref, feature_future_mask)
    loss_feat_future_copy_raw = _feature_loss(future_baseline_target, future_target_ref, feature_future_mask)
    loss_feat_current_norm = _feature_loss(_channel_norm(current_observed), _channel_norm(current_target), current_view_valid)
    loss_feat_future_norm = _feature_loss(
        _channel_norm(future_pred_target),
        _channel_norm(future_target_ref),
        feature_future_mask,
    )
    loss_feat_future_copy_norm = _feature_loss(
        _channel_norm(future_baseline_target),
        _channel_norm(future_target_ref),
        feature_future_mask,
    )
    if use_feature_channel_norm:
        loss_feat_future = loss_feat_future_norm
    else:
        loss_feat_future = loss_feat_future_raw
    loss_feat_current = current_target.new_tensor(0.0, dtype=torch.float32)
    loss_feat_past = past_visual.new_tensor(0.0, dtype=torch.float32)
    if sigreg_loss is None:
        sigreg_loss = past_visual.new_tensor(0.0, dtype=torch.float32)
    else:
        sigreg_loss = sigreg_loss.float()
    _cuda_profile_mark(forward_profile, "feature_loss_done")

    loss_depth = past_visual.new_tensor(0.0, dtype=torch.float32)
    loss_ray = past_visual.new_tensor(0.0, dtype=torch.float32)
    loss_point = past_visual.new_tensor(0.0, dtype=torch.float32)
    depth_metrics: Dict[str, torch.Tensor] = {}
    depth_target_for_viz: Optional[torch.Tensor] = None
    depth_target_mask_for_viz: Optional[torch.Tensor] = None
    depth_target_label = "GT depth"
    if decode_depth:
        teacher_depth_all = None
        used_teacher_depth = False

        def _teacher_depth_all() -> torch.Tensor:
            nonlocal teacher_depth_all
            if teacher_depth_all is not None:
                return teacher_depth_all
            _cuda_profile_mark(forward_profile, "teacher_depth_decode_start")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                teacher_levels_for_depth = teacher_da3.encode_all_levels(teacher_views_norm)
                if isinstance(teacher_levels_for_depth, dict):
                    teacher_levels_for_depth = [
                        teacher_levels_for_depth[level]
                        for level in sorted(teacher_levels_for_depth.keys())
                    ]
                teacher_decode_kwargs = {
                    "batch_size": b,
                    "views_per_sequence": teacher_views_norm.shape[1],
                    "frames_chunk_size": max(1, int(depth_decode_chunk_size)),
                }
                teacher_depth = teacher_da3.decode_depth(
                    teacher_levels_for_depth,
                    **teacher_decode_kwargs,
                )
            if teacher_depth.ndim == 4 and teacher_depth.shape[1] == 1:
                teacher_depth = teacher_depth[:, 0]
            T_total = teacher_views_norm.shape[1] // V
            teacher_depth_all = teacher_depth.reshape(
                b, T_total, V, *teacher_depth.shape[-2:]
            ).detach()
            _cuda_profile_mark(forward_profile, "teacher_depth_decode_done")
            return teacher_depth_all

        if (gt_depth_da3 is None or gt_depth_mask is None) and bool(teacher_depth_fallback):
            used_teacher_depth = True
            gt_depth_da3 = _teacher_depth_all()
            gt_depth_mask = torch.isfinite(gt_depth_da3) & (gt_depth_da3 > 0)
            if teacher_depth_valid_mask is not None:
                teacher_valid = _resize_bool_mask(
                    teacher_depth_valid_mask,
                    gt_depth_mask.shape[-2:],
                    strict=True,
                )
                gt_depth_mask = gt_depth_mask & teacher_valid
        if gt_depth_da3 is None or gt_depth_mask is None:
            raise ValueError("lambda_depth > 0 requires gt_depth_da3 and gt_depth_mask in the batch.")
        if "depth" not in deep_out:
            raise RuntimeError("Stage 1 depth head did not return a depth map.")
        pred_depth = deep_out["depth"]
        if pred_depth.ndim == 4 and pred_depth.shape[1] == 1:
            pred_depth = pred_depth[:, 0]
        pred_depth = pred_depth.reshape(b, deep_steps, V, *pred_depth.shape[-2:])
        target_depth = gt_depth_da3[:, 1 : H + 1].to(device=device, dtype=torch.float32)
        target_mask = gt_depth_mask[:, 1 : H + 1].to(device=device).bool()
        if future_view_valid is not None:
            depth_view_mask = future_view_valid[:, : target_mask.shape[1]]
            while depth_view_mask.ndim < target_mask.ndim:
                depth_view_mask = depth_view_mask.unsqueeze(-1)
            target_mask = target_mask & depth_view_mask
        gt_frame_mask = target_mask.flatten(3).any(dim=-1)
        teacher_fallback_frames = pred_depth.new_tensor(0.0)
        if bool(teacher_depth_fallback):
            used_teacher_depth = True
            teacher_target = _teacher_depth_all()[:, 1 : H + 1].to(
                device=device, dtype=torch.float32
            )
            teacher_target_mask = torch.isfinite(teacher_target) & (teacher_target > 0)
            if teacher_depth_valid_mask is not None:
                teacher_valid = _resize_bool_mask(
                    teacher_depth_valid_mask[:, 1 : H + 1],
                    teacher_target_mask.shape[-2:],
                    strict=True,
                )
                teacher_target_mask = teacher_target_mask & teacher_valid
            if future_view_valid is not None:
                teacher_view_mask = future_view_valid[:, : teacher_target_mask.shape[1]]
                while teacher_view_mask.ndim < teacher_target_mask.ndim:
                    teacher_view_mask = teacher_view_mask.unsqueeze(-1)
                teacher_target_mask = teacher_target_mask & teacher_view_mask
            fallback_frame_mask = ~gt_frame_mask
            if future_view_valid is not None:
                fallback_frame_mask = fallback_frame_mask & future_view_valid[:, : fallback_frame_mask.shape[1]]
            teacher_fallback_frames = fallback_frame_mask.to(dtype=torch.float32).sum().detach()
            while fallback_frame_mask.ndim < target_depth.ndim:
                fallback_frame_mask = fallback_frame_mask.unsqueeze(-1)
            target_depth = torch.where(fallback_frame_mask, teacher_target, target_depth)
            target_mask = torch.where(fallback_frame_mask, teacher_target_mask, target_mask)
        if transition_loss_mask_used is not None:
            depth_step_mask = transition_loss_mask_used
            while depth_step_mask.ndim < target_mask.ndim:
                depth_step_mask = depth_step_mask.unsqueeze(-1)
            target_mask = target_mask & depth_step_mask
        depth_target_for_viz = target_depth.detach()
        depth_target_mask_for_viz = target_mask.detach()
        if used_teacher_depth:
            depth_target_label = "target depth (GT/teacher)"
        depth_loss_mode = str(depth_loss_type).lower()
        if depth_loss_mode == "da3_full":
            for key in ("depth_conf", "ray", "ray_conf"):
                if key not in deep_out:
                    raise RuntimeError(
                        f"depth_loss_type='da3_full' requires DPT head output '{key}'. "
                        "Ensure propagate_shallow_with_actions_grad forwards all DPT keys."
                    )
            if gt_camera_intrinsics is None or gt_camera_extrinsics_c2w is None:
                raise ValueError(
                    "depth_loss_type='da3_full' requires gt_camera_intrinsics and "
                    "gt_camera_extrinsics_c2w in the batch (from LiberoHDF5SequenceDataset)."
                )
            if gt_depth_scene_scale is None:
                raise ValueError(
                    "depth_loss_type='da3_full' requires gt_depth_scene_scale in the batch."
                )
            pred_depth_conf = deep_out["depth_conf"]
            pred_ray = deep_out["ray"]
            pred_ray_conf = deep_out["ray_conf"]

            def _reshape_leading(x, spatial_dims):
                x = x.reshape(-1, *x.shape[-spatial_dims:])
                return x.reshape(b, deep_steps, V, *x.shape[-spatial_dims:])

            pred_depth_conf = _reshape_leading(pred_depth_conf, 2)
            pred_ray = _reshape_leading(pred_ray, 3)
            pred_ray_conf = _reshape_leading(pred_ray_conf, 2)
            K_future = gt_camera_intrinsics[:, 1 : H + 1].to(device).float()
            E_future = gt_camera_extrinsics_c2w[:, 1 : H + 1].to(device).float()
            ss = gt_depth_scene_scale.to(device).float()
            ss_per_sample = ss.expand(b) if ss.ndim == 0 else ss
            F_ = b * deep_steps * V
            pred_depth_flat = pred_depth.reshape(F_, *pred_depth.shape[-2:])
            pred_depth_conf_flat = pred_depth_conf.reshape(F_, *pred_depth_conf.shape[-2:])
            pred_ray_flat = pred_ray.reshape(F_, *pred_ray.shape[-3:])
            pred_ray_conf_flat = pred_ray_conf.reshape(F_, *pred_ray_conf.shape[-2:])
            target_depth_flat = target_depth.reshape(F_, *target_depth.shape[-2:])
            target_mask_flat = target_mask.reshape(F_, *target_mask.shape[-2:])
            K_flat = K_future.reshape(F_, 3, 3)
            E_flat = E_future.reshape(F_, 4, 4)
            ss_flat = ss_per_sample.reshape(b, 1, 1).expand(b, deep_steps, V).reshape(F_)

            loss_total_depth, depth_metrics_full = da3_full_depth_loss(
                pred_depth=pred_depth_flat,
                pred_depth_conf=pred_depth_conf_flat,
                pred_ray=pred_ray_flat,
                pred_ray_conf=pred_ray_conf_flat,
                target_depth_da3=target_depth_flat,
                target_mask=target_mask_flat,
                target_intrinsics=K_flat,
                target_extrinsics_c2w=E_flat,
                scene_scale=ss_flat,
                lambda_grad=float(depth_grad_weight),
                lambda_conf=float(lambda_depth_conf),
                lambda_ray=float(lambda_ray),
                lambda_point=float(lambda_point),
            )
            loss_depth = loss_total_depth
            loss_ray = depth_metrics_full["ray_l1"].float()
            loss_point = depth_metrics_full["point_l1"].float()
            depth_metrics.update(depth_metrics_full)
        elif depth_loss_mode == "da3_style":
            loss_depth, depth_metrics = da3_style_depth_loss(
                pred_depth=pred_depth,
                target_depth=target_depth,
                target_mask=target_mask,
                grad_weight=depth_grad_weight,
            )
        else:
            raise ValueError(
                f"Unknown depth_loss_type={depth_loss_type!r}. "
                "Expected 'da3_style' or 'da3_full'."
            )
        depth_metrics["depth_supervised_steps"] = pred_depth.new_tensor(float(deep_steps)).detach()
        if future_view_valid is not None:
            depth_slots = float(future_view_valid[:, :deep_steps].to(dtype=torch.float32).sum().detach().item())
        else:
            depth_slots = float(deep_steps * V)
        depth_metrics["depth_decode_slots"] = pred_depth.new_tensor(depth_slots).detach()
        depth_metrics["depth_gt_frames"] = gt_frame_mask.to(dtype=torch.float32).sum().detach()
        depth_metrics["depth_teacher_fallback_frames"] = teacher_fallback_frames
        depth_loss_mode_code = {
            "da3_style": 0.0,
            "da3_full": 1.0,
        }[depth_loss_mode]
        depth_metrics["depth_loss_type"] = pred_depth.new_tensor(depth_loss_mode_code).detach()
    elif float(lambda_depth) > 0.0 and bool(skip_depth_if_no_gt):
        depth_metrics["depth_skipped_no_gt"] = past_visual.new_tensor(1.0, dtype=torch.float32)
    _cuda_profile_mark(forward_profile, "depth_target_loss_done")

    loss_camera = past_visual.new_tensor(0.0, dtype=torch.float32)

    # Path B deep multi-level feature distillation (optional, off by default).
    # Compares student deep features (output of `propagate_shallow_with_actions_grad`
    # at OUT_LAYERS=[19,27,33,39], driven by predictor outputs) with teacher
    # full-forward features at the same layers (driven by GT observed images).
    # Layer-weighted, deepest layer = 1.0, shallower scaled to
    # `deep_feat_reg_layer_weight_min`. Student level_feats are channel-doubled
    # (local || current); we compare the current half so dims align with
    # teacher's `encode_all_levels` output (single embed_dim).
    loss_deep_feat_reg = past_visual.new_tensor(0.0, dtype=torch.float32)
    if want_deep_feat_reg:
        student_levels = deep_out.get("level_feats")
        if not student_levels:
            raise RuntimeError(
                "lambda_path_b_deep_feat_reg > 0 requires "
                "propagate_shallow_with_actions_grad(return_multi_level=True) "
                "but deep_out has no 'level_feats' key."
            )
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            teacher_full = teacher_da3.encode_all_levels(teacher_views_norm)
            teacher_levels = [teacher_full[i] for i in range(len(teacher_full))]
        if len(student_levels) != len(teacher_levels):
            raise RuntimeError(
                f"deep feat reg: student levels={len(student_levels)} vs "
                f"teacher levels={len(teacher_levels)}"
            )
        T_total = teacher_views_norm.shape[1] // V
        N_levels = len(student_levels)
        total_w = 0.0
        deep_feat_mask = None
        deep_feat_view_mask = _combine_step_view_mask(
            None if transition_loss_mask_used is None else transition_loss_mask_used[:, :deep_steps],
            None if future_view_valid is None else future_view_valid[:, :deep_steps],
        )
        if deep_feat_view_mask is not None:
            if deep_feat_view_mask.ndim == 2:
                deep_feat_view_mask = deep_feat_view_mask.unsqueeze(-1).expand(b, deep_steps, V)
            deep_feat_mask = deep_feat_view_mask.reshape(b, deep_steps * V)
        for lvl_idx, ((s_p, s_c), (t_p_flat, t_c_flat)) in enumerate(
            zip(student_levels, teacher_levels)
        ):
            # Both student and teacher come back as cat_token features (local
            # || current concat), so the channel dim already matches; no slice
            # needed. Student grad flows; teacher is detached for distillation.
            s_p_f = s_p.float()  # (b, deep_steps*V, N, 2C)
            s_c_f = s_c.float()  # (b, deep_steps*V, 2C)
            t_p = t_p_flat.reshape(b, T_total, V, t_p_flat.shape[-2], t_p_flat.shape[-1])
            t_c = t_c_flat.reshape(b, T_total, V, t_c_flat.shape[-1])
            t_start = predicted_sequence_start_timestep
            t_end = t_start + deep_steps
            t_p_win = t_p[:, t_start:t_end].reshape(b, deep_steps * V, *t_p.shape[-2:]).float().detach()
            t_c_win = t_c[:, t_start:t_end].reshape(b, deep_steps * V, t_c.shape[-1]).float().detach()
            if s_p_f.shape != t_p_win.shape or s_c_f.shape != t_c_win.shape:
                raise RuntimeError(
                    f"deep feat reg shape mismatch at level {lvl_idx}: "
                    f"student patch {tuple(s_p_f.shape)} vs teacher {tuple(t_p_win.shape)}; "
                    f"student cls {tuple(s_c_f.shape)} vs teacher {tuple(t_c_win.shape)}"
                )
            if N_levels > 1:
                w = float(deep_feat_reg_layer_weight_min) + (
                    1.0 - float(deep_feat_reg_layer_weight_min)
                ) * lvl_idx / (N_levels - 1)
            else:
                w = 1.0
            loss_lvl = _masked_mse_loss(s_p_f, t_p_win, deep_feat_mask) + _masked_mse_loss(
                s_c_f,
                t_c_win,
                deep_feat_mask,
            )
            loss_deep_feat_reg = loss_deep_feat_reg + w * loss_lvl
            total_w += w
        loss_deep_feat_reg = loss_deep_feat_reg / max(total_w, 1e-8)
    _cuda_profile_mark(forward_profile, "deep_feat_reg_done")

    loss_total = (
        float(lambda_action) * loss_action
        + float(lambda_feat_current) * loss_feat_current
        + float(lambda_feat_future) * loss_feat_future
        + float(lambda_proprio_future) * loss_proprio_future
        + float(lambda_sigreg) * sigreg_loss
        + float(lambda_depth) * loss_depth
        + float(lambda_path_b_deep_feat_reg) * loss_deep_feat_reg
    )
    _cuda_profile_mark(forward_profile, "loss_total_done")
    _finalize_cuda_profile(forward_profile)

    return {
        "loss_total": loss_total,
        "loss_action": loss_action,
        "loss_action_direct": loss_action_direct,
        "loss_action_refine": loss_action_refine,
        "lambda_action_direct": float(lambda_action_direct),
        "lambda_action_refine": float(lambda_action_refine),
        "loss_feat_future": loss_feat_future,
        "loss_feat_current": loss_feat_current,
        "loss_feat_future_raw": loss_feat_future_raw,
        "loss_feat_current_raw": loss_feat_current_raw,
        "loss_feat_future_norm": loss_feat_future_norm,
        "loss_feat_current_norm": loss_feat_current_norm,
        "loss_feat_future_copy_raw": loss_feat_future_copy_raw,
        "loss_feat_future_copy_norm": loss_feat_future_copy_norm,
        "feature_loss_norm": feature_loss_norm,
        "feature_loss_type": feature_loss_type,
        "feature_target_mode": feature_target_mode,
        "loss_feat_past": loss_feat_past,
        "loss_deep_feat_reg": loss_deep_feat_reg,
        "loss_proprio_future": loss_proprio_future,
        "loss_proprio_future_direct": loss_proprio_future_direct,
        "loss_proprio_future_head": loss_proprio_future_head,
        "loss_sigreg": sigreg_loss,
        "loss_depth": loss_depth,
        "loss_ray": loss_ray,
        "loss_point": loss_point,
        "loss_camera": loss_camera,
        "depth_metrics": depth_metrics,
        "depth_target": depth_target_for_viz,
        "depth_target_mask": depth_target_mask_for_viz,
        "depth_target_label": depth_target_label,
        "action_pred": action_pred,
        "predicted_actions_deep": predicted_actions_deep,
        "predicted_actions_direct": predicted_actions_direct,
        "target_actions": target_actions,
        "observed_current_visual_tokens": current_observed,
        "target_current_visual_tokens": current_target,
        "predicted_last_visual_tokens": last_predicted_visual_tokens,
        "predicted_next_visual_tokens": predicted_next_visual_tokens,
        "predicted_last_proprio": last_predicted_proprio,
        "predicted_next_proprio": predicted_next_proprio,
        "target_last_visual_tokens": future_target,
        "target_next_visual_tokens": future_targets_all,
        "target_last_proprio": target_last_proprio,
        "target_next_proprio": target_proprio_all,
        "action_loss_mask_used": action_loss_mask_used,
        "transition_loss_mask_used": transition_loss_mask_used,
        "context_valid_mask_used": context_valid_mask_used,
        "view_valid_mask_used": view_valid_all,
        "view_valid_ratio": (
            past_visual.new_tensor(1.0, dtype=torch.float32)
            if view_valid_all is None
            else view_valid_all.to(dtype=torch.float32).mean().detach()
        ),
        "proprio_predicted_from_deep": proprio_predicted_from_deep,
        "past_visual_tokens": past_visual,
        "student_feats": None,
        "student_raw": [],
        "teacher_past_raw": [],
        "last_predicted_action_tokens": last_predicted_action_tokens,
        "deep_visual_tokens": deep_visual.detach(),
        "deep_action_tokens": deep_actions.detach(),
        "deep_gradient_checkpointing": bool(deep_gradient_checkpointing),
        "deep_temporal_causal_mask": bool(deep_temporal_causal_mask),
        "predicted_sequence_start_timestep": predicted_sequence_start_timestep,
        "predicted_sequence_steps": deep_steps,
        "deep_current_timestep": None,
        "H": H,
        "T": T,
        "V": V,
        "predictor_type": "gam",
        "forward_profile": forward_profile,
    }


# -----------------------------------------------------------------------------
# Config-gated H sampler
# -----------------------------------------------------------------------------


def sample_H(H_choices: List[int], H_weights: Optional[List[float]] = None) -> int:
    """Variable-H curriculum: pick H per batch from choices."""
    import random
    if H_weights is None:
        return random.choice(H_choices)
    return random.choices(H_choices, weights=H_weights, k=1)[0]
