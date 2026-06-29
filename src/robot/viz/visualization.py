"""DA3 depth visualization for WandB: GT vs Predicted.

Uses encoder.propagate_and_decode() to get depth from:
  - GT Level 0 features (from DA3 encoding of real images)
  - Predicted Level 0 features (from DiT output)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Sequence

ACTION_DIM_NAMES = ("dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper")


def log_training_input_images(
    batch: Dict,
    step: int,
    *,
    wandb_run=None,
    log_dict: Optional[Dict] = None,
    prefix: str = "inputs",
    num_samples: int = 1,
) -> None:
    """Log the *exact* RGB images that go into the encoder (post-augmentation).

    Designed for sanity checks: confirms the training pipeline is delivering
    the expected camera frames (orientation, crop, color jitter, JPEG noise)
    and the expected per-task text prompt.

    For each of the first ``num_samples`` batch rows, emits one wandb.Image
    per camera (named after the resolved camera key) plus the task description
    string under ``f"{prefix}/sample_{i}/task"``.

    No-op if wandb_run is None and log_dict is None.
    """
    if wandb_run is None and log_dict is None:
        return
    if "all_view_images" not in batch:
        return

    try:
        import wandb
    except ImportError:
        return

    target_log = log_dict if log_dict is not None else {}

    imgs = batch["all_view_images"]
    if not isinstance(imgs, torch.Tensor):
        return
    # (B, T, V, 3, H, W) : pre-encoder normalization (raw [0, 1] RGB after augmentation).
    if imgs.ndim != 6:
        return
    B = int(imgs.shape[0])
    T = int(imgs.shape[1])
    V = int(imgs.shape[2])
    n = max(1, min(int(num_samples), B))

    camera_keys_batch = batch.get("camera_keys", None)
    task_descs = batch.get("task_description", None)
    dataset_names = batch.get("dataset_name", None)
    action_stats_keys = batch.get("action_stats_key", None)

    for i in range(n):
        if camera_keys_batch is not None and i < len(camera_keys_batch):
            camera_keys = list(camera_keys_batch[i])
        else:
            camera_keys = [f"cam{v}" for v in range(V)]
        task = task_descs[i] if isinstance(task_descs, (list, tuple)) and i < len(task_descs) else None
        ds_name = dataset_names[i] if isinstance(dataset_names, (list, tuple)) and i < len(dataset_names) else None
        stats_key = (
            action_stats_keys[i]
            if isinstance(action_stats_keys, (list, tuple)) and i < len(action_stats_keys)
            else None
        )

        for v in range(V):
            cam_name = camera_keys[v] if v < len(camera_keys) else f"cam{v}"
            cam_short = str(cam_name).split(".")[-1].replace(" ", "_") or f"cam{v}"
            # Log the *current* timestep frame (the predictor's observed input) +
            # the next-step frame (the distillation target) when available.
            for t_label, t_idx in (("now", 0), ("next", min(1, T - 1))):
                if t_idx >= T:
                    continue
                img = imgs[i, t_idx, v].detach()
                if img.dtype.is_floating_point:
                    arr = img.permute(1, 2, 0).cpu().to(torch.float32).numpy()
                    if float(np.nanmax(arr)) <= 1.5:  # [0, 1]
                        arr = np.clip(arr, 0.0, 1.0) * 255.0
                    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                else:
                    arr = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                caption_bits: List[str] = [
                    f"sample={i}",
                    f"t={t_label}",
                    cam_name,
                ]
                if ds_name:
                    caption_bits.append(f"ds={ds_name}")
                if stats_key and stats_key != ds_name:
                    caption_bits.append(f"stats={stats_key}")
                if task:
                    caption_bits.append(f'task="{task}"')
                target_log[f"{prefix}/sample{i}/{cam_short}_{t_label}"] = wandb.Image(
                    arr, caption=_join_caption(caption_bits)
                )

        if task:
            target_log[f"{prefix}/sample{i}/task"] = task
        if ds_name:
            target_log[f"{prefix}/sample{i}/dataset"] = ds_name
        if stats_key:
            target_log[f"{prefix}/sample{i}/action_stats_key"] = stats_key

    if log_dict is None and wandb_run is not None:
        wandb_run.log(target_log, step=int(step))


def _join_caption(parts: Sequence[Optional[str]]) -> str:
    return " | ".join(str(part) for part in parts if part)


def depth_to_colormap(depth: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    import matplotlib.cm as cm
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return np.zeros((*depth.shape[-2:], 3), dtype=np.uint8)
    d_min, d_max = depth[valid].min(), depth[valid].max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth)
    depth_norm = np.nan_to_num(depth_norm, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = (cm.turbo(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def depth_to_pointcloud_image(depth, rgb, fx=200.0, fy=200.0, max_points=10000):
    H, W = depth.shape
    cx, cy = W / 2, H / 2
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth > 0) & np.isfinite(depth)
    z = depth[valid]
    if len(z) == 0:
        return np.ones((512, 512, 3), dtype=np.uint8) * 30
    x3d = (u[valid] - cx) * z / fx
    rgb_v = rgb[valid]
    if len(z) > max_points:
        idx = np.random.choice(len(z), max_points, replace=False)
        x3d, z, rgb_v = x3d[idx], z[idx], rgb_v[idx]
    canvas = np.ones((512, 512, 3), dtype=np.uint8) * 30
    xr = x3d.max() - x3d.min()
    zr = z.max() - z.min()
    if xr > 1e-6 and zr > 1e-6:
        px = ((x3d - x3d.min()) / xr * 472 + 20).astype(int).clip(0, 511)
        py = ((z - z.min()) / zr * 472 + 20).astype(int).clip(0, 511)
        canvas[py, px] = rgb_v
    return canvas


def _decode_depth(encoder, patches, cls_token, total_view):
    """Helper: propagate L0 features → DPT → depth."""
    result = encoder.propagate_and_decode(patches, cls_token, total_view=total_view)
    if "depth" not in result:
        return None
    depth = result["depth"]
    # Normalize shape to (BV, H, W)
    if depth.ndim == 4:
        if depth.shape[0] == 1 and depth.shape[1] > 1:
            depth = depth.squeeze(0)
        elif depth.shape[1] == 1:
            depth = depth.squeeze(1)
        else:
            depth = depth.reshape(-1, *depth.shape[-2:])
    return depth


def _make_depth_row(images, depth, b=0, n_views=2):
    """Make RGB|Depth side-by-side panels for n_views."""
    panels = []
    for v in range(min(n_views, depth.shape[0])):
        if images is not None and v < images.shape[1]:
            rgb = (images[b, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            rgb = np.zeros((*depth.shape[-2:], 3), dtype=np.uint8)
        d = depth[v].cpu().float().numpy()
        panels.append(np.concatenate([rgb, depth_to_colormap(d)], axis=1))
    return np.concatenate(panels, axis=0) if panels else None


@torch.no_grad()
def log_da3_visualizations(
    encoder, batch, device, step, wandb_run=None,
    pred_features=None, pred_cls=None,
):
    """Log GT and predicted depth/pointcloud visualizations.

    Args:
        encoder: DA3GiantEncoder with propagate_and_decode.
        batch: Data batch.
        device: GPU.
        step: Training step.
        wandb_run: WandB run.
        pred_features: (BV, N, 3072) DiT-predicted Level 0 features (optional).
        pred_cls: (BV, 3072) DiT-predicted CLS tokens (optional).
    """
    if wandb_run is None:
        return

    import wandb

    current_imgs = batch["current_images"].to(device)
    future_imgs = batch["future_images"].to(device)
    B, T = future_imgs.shape[:2]
    b = 0

    # Normalize images
    current_norm = encoder.normalize_images(current_imgs)
    future_flat = future_imgs.reshape(B, T * 2, 3, *future_imgs.shape[-2:])
    future_norm = encoder.normalize_images(future_flat)
    all_views = torch.cat([current_norm, future_norm], dim=1)

    log_dict = {}

    # ========== GT Depth ==========
    # Encode GT Level 0 features before latent normalization.
    ref_patches, ref_cls = encoder.encode_single(current_norm, level=0, return_cls=True)
    all_patches, all_cls = encoder.encode_single(all_views, level=0, return_cls=True)

    # GT depth: reference views
    gt_ref_depth = _decode_depth(encoder, ref_patches, ref_cls, total_view=2)
    if gt_ref_depth is not None:
        panel = _make_depth_row(current_imgs, gt_ref_depth, b=b, n_views=2)
        if panel is not None:
            log_dict["da3_gt/ref_rgb_depth"] = wandb.Image(panel, caption=f"GT Ref: RGB|Depth")

        # GT pointcloud (ref ext)
        rgb_np = (current_imgs[b, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pc = depth_to_pointcloud_image(gt_ref_depth[0].cpu().float().numpy(), rgb_np)
        log_dict["da3_gt/pointcloud_ref"] = wandb.Image(pc, caption="GT Pointcloud (ref ext)")

    # GT depth: all 14 views grid
    gt_all_depth = _decode_depth(encoder, all_patches, all_cls, total_view=14)
    if gt_all_depth is not None:
        n = min(14, gt_all_depth.shape[0])
        panels = [depth_to_colormap(gt_all_depth[v].cpu().float().numpy()) for v in range(n)]
        while len(panels) < 14:
            panels.append(np.zeros_like(panels[0]))
        grid = np.concatenate([
            np.concatenate(panels[:7], axis=1),
            np.concatenate(panels[7:14], axis=1),
        ], axis=0)
        log_dict["da3_gt/all_views_depth"] = wandb.Image(grid, caption="GT All 14 views depth")

    # ========== Predicted Depth (from DiT output) ==========
    if pred_features is not None:
        pred_patches = pred_features
        pred_cls_tok = pred_cls

        pred_all_depth = _decode_depth(encoder, pred_patches, pred_cls_tok, total_view=14)
        if pred_all_depth is not None:
            # Pred depth: ref views
            if pred_all_depth.shape[0] >= 2:
                pred_ref_depth = pred_all_depth[:2]
                panel = _make_depth_row(current_imgs, pred_ref_depth, b=b, n_views=2)
                if panel is not None:
                    log_dict["da3_pred/ref_rgb_depth"] = wandb.Image(panel, caption="Pred Ref: RGB|Depth")

                rgb_np = (current_imgs[b, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pc = depth_to_pointcloud_image(pred_ref_depth[0].cpu().float().numpy(), rgb_np)
                log_dict["da3_pred/pointcloud_ref"] = wandb.Image(pc, caption="Pred Pointcloud (ref ext)")

            # Pred depth: all 14 views grid
            n = min(14, pred_all_depth.shape[0])
            panels = [depth_to_colormap(pred_all_depth[v].cpu().float().numpy()) for v in range(n)]
            while len(panels) < 14:
                panels.append(np.zeros_like(panels[0]))
            grid = np.concatenate([
                np.concatenate(panels[:7], axis=1),
                np.concatenate(panels[7:14], axis=1),
            ], axis=0)
            log_dict["da3_pred/all_views_depth"] = wandb.Image(grid, caption="Pred All 14 views depth")

    if log_dict:
        wandb.log(log_dict, step=step)


def _resize_uint8_image(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == hw:
        return image
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((hw[1], hw[0]), resample=Image.BILINEAR))


def _render_rgb_depth_timeline(
    depth: torch.Tensor,
    images: Optional[torch.Tensor],
    *,
    n_steps: int,
    n_views: int,
    title: str,
    gt_depth: Optional[torch.Tensor] = None,
    gt_mask: Optional[torch.Tensor] = None,
    target_label: str = "target depth",
    pred_label: str = "prop depth",
    start_timestep: int = 0,
    current_timestep: Optional[int] = None,
) -> np.ndarray:
    """Render depth `(S*V,H,W)` as RGB|target|Pred cells over timestep/view."""
    depth_cpu = depth.detach().float().cpu()
    gt_cpu = gt_depth.detach().float().cpu() if gt_depth is not None else None
    mask_cpu = gt_mask.detach().bool().cpu() if gt_mask is not None else None
    rows = []
    for t in range(n_steps):
        cells = []
        for v in range(n_views):
            idx = t * n_views + v
            if idx >= depth_cpu.shape[0]:
                continue
            depth_img = depth_to_colormap(depth_cpu[idx].numpy())
            cell_parts = []
            if images is not None:
                rgb = _to_uint8_image(images[t, v])
                rgb = _resize_uint8_image(rgb, depth_img.shape[:2])
                cell_parts.append(rgb)
            if gt_cpu is not None and t < gt_cpu.shape[0] and v < gt_cpu.shape[1]:
                gt_frame = gt_cpu[t, v].numpy()
                gt_valid = None
                if mask_cpu is not None and t < mask_cpu.shape[0] and v < mask_cpu.shape[1]:
                    gt_valid = mask_cpu[t, v].numpy()
                gt_img = depth_to_colormap(gt_frame, gt_valid)
                gt_img = _resize_uint8_image(gt_img, depth_img.shape[:2])
                cell_parts.append(gt_img)
            cell_parts.append(depth_img)
            cell = np.concatenate(cell_parts, axis=1)
            layout = " | ".join(
                (["RGB"] if images is not None else [])
                + ([target_label] if gt_cpu is not None else [])
                + [pred_label]
            )
            cells.append(
                _annotate_uint8_image(
                    cell,
                    [
                        title,
                        _format_timeline_label(t, v, start_timestep, current_timestep),
                        layout,
                    ],
                )
            )
        if cells:
            rows.append(np.concatenate(cells, axis=1))
    if not rows:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return np.concatenate(rows, axis=0)


def _format_timeline_label(
    slot: int,
    view: int,
    start_timestep: int,
    current_timestep: Optional[int],
) -> str:
    obs_t = int(start_timestep) + int(slot)
    role = "slot"
    if current_timestep is not None:
        cur_t = int(current_timestep)
        if obs_t < cur_t:
            role = "context"
        elif obs_t == cur_t:
            role = "current"
        else:
            role = f"future+{obs_t - cur_t}"
    return f"{role} obs_t={obs_t} view={view}"


def _decode_slots_for_monitor(encoder, slots: torch.Tensor):
    """Decode `[action, cls, patches]` slots through DA3 for monitor-only depth.

    FuturePredictor predicts the 1536-d Robot-GLD-style Level-0 pre-global
    `local_x` tokens. `propagate_and_decode()` initializes the current stream
    from that seed, replays the Level-0 global block, then continues through
    the remaining local/global blocks. Future depth remains a monitor proxy
    because the first decoded level starts from a predicted latent rather than
    observed RGB.
    """
    was_training = encoder.training
    try:
        encoder.eval()
        slots = torch.nan_to_num(slots.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        b, n_steps, n_views, n_tokens, dim = slots.shape
        flat = slots.contiguous().reshape(b * n_steps * n_views, n_tokens, dim)
        cls = flat[:, 1, :]
        patches = flat[:, 2:, :]
        expected_dim = int(getattr(encoder, "embed_dim", dim)) * 2
        if patches.shape[-1] not in (int(getattr(encoder, "embed_dim", dim)), expected_dim):
            raise ValueError(
                f"Cannot decode slots with dim={patches.shape[-1]}; expected "
                f"{getattr(encoder, 'embed_dim', dim)} or {expected_dim}."
            )
        result = _decode_depth(encoder, patches, cls, total_view=n_steps * n_views)
        return result
    finally:
        encoder.train(was_training)


def _decode_direct_depth_for_monitor(encoder, features_per_level):
    """Decode observed conditioning features through the normal DA3 DPT path."""
    if features_per_level is None:
        return None
    was_training = encoder.training
    try:
        encoder.eval()
        detached = [(p.detach(), c.detach()) for p, c in features_per_level]
        depth = encoder.decode_depth(detached)
        if depth is None:
            return None
        if depth.ndim == 4:
            if depth.shape[0] == 1 and depth.shape[1] > 1:
                depth = depth.squeeze(0)
            elif depth.shape[1] == 1:
                depth = depth.squeeze(1)
            else:
                depth = depth.reshape(-1, *depth.shape[-2:])
        return depth.detach().cpu()
    finally:
        encoder.train(was_training)


def _normalize_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    """Normalize DA3 depth outputs to flattened `(N, H, W)` CPU tensors."""
    depth = depth.detach()
    if depth.ndim == 4:
        if depth.shape[0] == 1 and depth.shape[1] > 1:
            depth = depth.squeeze(0)
        elif depth.shape[1] == 1:
            depth = depth.squeeze(1)
        else:
            depth = depth.reshape(-1, *depth.shape[-2:])
    return depth.detach().cpu()


def _splice_direct_conditioning_depth(
    proxy_depth: Optional[torch.Tensor],
    direct_depth: Optional[torch.Tensor],
    n_conditioning_views: int,
) -> Optional[torch.Tensor]:
    """Replace proxy conditioning depth with direct DA3 depth when shapes allow."""
    if proxy_depth is None:
        return direct_depth
    if direct_depth is None or n_conditioning_views <= 0:
        return proxy_depth

    proxy = proxy_depth.detach().cpu()
    direct = direct_depth.detach().cpu()
    if proxy.ndim == 4:
        if proxy.shape[0] == 1 and proxy.shape[1] > 1:
            proxy = proxy.squeeze(0)
        elif proxy.shape[1] == 1:
            proxy = proxy.squeeze(1)
        else:
            proxy = proxy.reshape(-1, *proxy.shape[-2:])
    if direct.ndim == 4:
        if direct.shape[0] == 1 and direct.shape[1] > 1:
            direct = direct.squeeze(0)
        elif direct.shape[1] == 1:
            direct = direct.squeeze(1)
        else:
            direct = direct.reshape(-1, *direct.shape[-2:])
    if proxy.ndim != 3 or direct.ndim != 3:
        return proxy_depth
    if proxy.shape[-2:] != direct.shape[-2:]:
        return proxy_depth

    n = min(int(n_conditioning_views), proxy.shape[0], direct.shape[0])
    if n <= 0:
        return proxy
    merged = proxy.clone()
    merged[:n] = direct[:n].to(dtype=merged.dtype)
    return merged


@torch.no_grad()
def log_gam_future_visualizations(
    encoder,
    batch,
    device,
    step: int,
    wandb_run=None,
    visual_tokens: Optional[torch.Tensor] = None,
    action_tokens: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    target_label: Optional[str] = None,
    start_timestep: int = 0,
    current_timestep: Optional[int] = None,
    prefix: str = "unified",
    log_dict: Optional[Dict] = None,
):
    """Log gam DA3-deep depth monitor panels to WandB.

    `visual_tokens` are block-12 visual states in `[CLS, registers, patches]`
    order and `action_tokens` are the block-13 action seeds. This path matches
    the gam action decode path instead of the legacy Level-0 slot proxy.
    """
    if wandb_run is None or visual_tokens is None or action_tokens is None:
        return

    import wandb

    was_training = encoder.training
    try:
        encoder.eval()
        visual = torch.nan_to_num(
            visual_tokens.detach().to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        actions = torch.nan_to_num(
            action_tokens.detach().to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        result = encoder.propagate_shallow_with_actions(
            visual,
            actions,
            decode_visuals=True,
        )
    finally:
        encoder.train(was_training)

    if "depth" not in result:
        return

    n_steps = int(visual_tokens.shape[1])
    n_views = int(visual_tokens.shape[2])
    depth = _normalize_depth_tensor(result["depth"])

    images = None
    gt_depth = None
    gt_mask = None
    depth_label = target_label or "GT depth"
    if "all_view_images" in batch:
        all_images = batch["all_view_images"].to(device)
        start = max(int(start_timestep), 0)
        end = min(start + n_steps, all_images.shape[1])
        if end > start:
            images = all_images[0, start:end, :n_views].detach().cpu()
            n_steps = images.shape[0]
            depth = depth[: n_steps * n_views]
            if target_depth is not None:
                gt_depth = target_depth[0, :n_steps, :n_views].detach().cpu()
            elif "gt_depth_da3" in batch:
                gt_depth = batch["gt_depth_da3"][0, start:end, :n_views].detach().cpu()
            if target_mask is not None:
                gt_mask = target_mask[0, :n_steps, :n_views].detach().cpu()
            elif "gt_depth_mask" in batch:
                gt_mask = batch["gt_depth_mask"][0, start:end, :n_views].detach().cpu()

    panel = _render_rgb_depth_timeline(
        depth,
        images,
        n_steps=n_steps,
        n_views=n_views,
        title="gam deep",
        gt_depth=gt_depth,
        gt_mask=gt_mask,
        target_label=depth_label,
        pred_label="prop depth",
        start_timestep=start,
        current_timestep=current_timestep,
    )
    target_log = log_dict if log_dict is not None else {}
    target_log[f"{prefix}/depth_current_future"] = wandb.Image(
        panel,
        caption=(
            f"RGB | {depth_label} | DA3 blocks 13+ propagated depth from gam deep window "
            f"start_t={int(start_timestep)} steps={n_steps}"
        ),
    )
    if log_dict is None:
        wandb.log(target_log, step=step)


@torch.no_grad()
def log_unified_future_visualizations(
    encoder,
    batch,
    device,
    step: int,
    wandb_run=None,
    pred_slots: Optional[torch.Tensor] = None,
    direct_current_features=None,
    H: int = 1,
    start_timestep: Optional[int] = None,
    target_depth: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    target_label: Optional[str] = None,
    prefix: str = "unified",
    log_dict: Optional[Dict] = None,
):
    """Log unified depth monitor panels to WandB."""
    if wandb_run is None or pred_slots is None:
        return

    import wandb

    n_steps = int(pred_slots.shape[1])
    n_views = int(pred_slots.shape[2])
    depth = _decode_slots_for_monitor(encoder, pred_slots.detach())
    if depth is None:
        return
    direct_depth = _decode_direct_depth_for_monitor(encoder, direct_current_features)
    if direct_depth is not None:
        depth = _splice_direct_conditioning_depth(
            depth,
            direct_depth,
            n_conditioning_views=max(0, int(H) * n_views),
        )

    images = None
    gt_depth = None
    gt_mask = None
    depth_label = target_label or "GT depth"
    if "all_view_images" in batch:
        all_images = batch["all_view_images"].to(device)
        start = max(int(H) - 1, 0) if start_timestep is None else max(int(start_timestep), 0)
        end = min(start + n_steps, all_images.shape[1])
        if end > start:
            images = all_images[0, start:end, :n_views].detach().cpu()
            n_steps = images.shape[0]
            depth = depth[: n_steps * n_views]
            if target_depth is not None:
                gt_depth = target_depth[0, :n_steps, :n_views].detach().cpu()
            elif "gt_depth_da3" in batch:
                gt_depth = batch["gt_depth_da3"][0, start:end, :n_views].detach().cpu()
            if target_mask is not None:
                gt_mask = target_mask[0, :n_steps, :n_views].detach().cpu()
            elif "gt_depth_mask" in batch:
                gt_mask = batch["gt_depth_mask"][0, start:end, :n_views].detach().cpu()

    panel = _render_rgb_depth_timeline(
        depth,
        images,
        n_steps=n_steps,
        n_views=n_views,
        title="current+future",
        gt_depth=gt_depth,
        gt_mask=gt_mask,
        target_label=depth_label,
        pred_label="proxy depth",
        start_timestep=start,
        current_timestep=int(H) - 1,
    )
    target_log = log_dict if log_dict is not None else {}
    target_log[f"{prefix}/depth_current_future"] = wandb.Image(
        panel,
        caption=(
            f"RGB | {depth_label} | direct/proxy DA3 depth; row labels mark "
            "context/current/future timesteps"
        ),
    )
    if log_dict is None:
        wandb.log(target_log, step=step)


# ============================================================
# 3D Action Trajectory Visualization: GT vs Predicted
# ============================================================

def _actions_to_trajectory(actions: np.ndarray) -> np.ndarray:
    """Convert delta actions (T, 7) to absolute EEF positions (T+1, 3).

    actions[:, :3] are delta xyz, actions[:, 3:6] are delta rotation (ignored for position),
    actions[:, 6] is gripper.
    """
    positions = np.zeros((len(actions) + 1, 3), dtype=np.float64)
    for t in range(len(actions)):
        positions[t + 1] = positions[t] + actions[t, :3]
    return positions


def _flatten_chunked_actions(actions: np.ndarray) -> np.ndarray:
    """Convert action arrays to a simple temporal sequence of shape (N, D).

    Supported inputs:
      - (T, D): standard action sequence
      - (T, C, D): chunked action sequence, flattened to (T*C, D)
    """
    if actions.ndim == 2:
        return actions
    if actions.ndim == 3:
        return actions.reshape(-1, actions.shape[-1])
    raise ValueError(f"Expected actions with 2 or 3 dims, got shape {actions.shape}.")


def _collapse_chunked_actions_to_tokens(actions: np.ndarray) -> np.ndarray:
    """Collapse chunked actions to one diagnostic action per action token."""
    if actions.ndim != 3:
        return actions
    collapsed = actions.sum(axis=1)
    if collapsed.shape[-1] >= 7:
        collapsed[:, -1] = actions[:, -1, -1]
    return collapsed


def _to_uint8_image(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float().clamp(0, 1)
    if img.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(img.shape)}")
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _annotate_uint8_image(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    text_lines = [str(line) for line in lines if line]
    if not text_lines:
        return np.asarray(pil_img)

    line_height = 14
    pad = 4
    box_height = pad * 2 + line_height * len(text_lines)
    box_width = min(pil_img.width, max(120, max(len(line) for line in text_lines) * 7 + pad * 2))
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0, 160))
    y = pad
    for line in text_lines:
        draw.text((pad, y), line, fill=(255, 255, 255, 255))
        y += line_height
    return np.asarray(pil_img)


def _render_image_timeline(
    all_view_images: torch.Tensor,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    camera_keys: Optional[Sequence[str]] = None,
    dataset_name: Optional[str] = None,
    episode_ref: Optional[str] = None,
) -> np.ndarray:
    """Render (T, V, C, H, W) as a timestep-by-view image grid."""
    rows = []
    for t in range(all_view_images.shape[0]):
        row = []
        for v in range(all_view_images.shape[1]):
            frame_id = None if frame_indices is None or t >= len(frame_indices) else int(frame_indices[t])
            cam_name = None if camera_keys is None or v >= len(camera_keys) else str(camera_keys[v]).split(".")[-1]
            lines = [
                dataset_name,
                episode_ref,
                f"frame={frame_id} t={t} v={v}" if frame_id is not None else f"t={t} v={v}",
                cam_name,
            ]
            row.append(_annotate_uint8_image(_to_uint8_image(all_view_images[t, v]), lines))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _render_heatmap(matrix: np.ndarray, title: str, xlabels: list[str]) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = max(6, 0.8 * len(xlabels))
    fig_h = max(3, 0.3 * matrix.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Step")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


@torch.no_grad()
def log_robot_debug_batch(
    batch,
    gt_actions: torch.Tensor,
    pred_actions: Optional[torch.Tensor],
    step: int,
    wandb_run=None,
    batch_idx: int = 0,
    prefix: str = "debug",
):
    """Log rich batch-level debug visuals for images, actions, and proprio."""
    if wandb_run is None:
        return

    import wandb

    log_dict = {}
    dataset_name = None
    if "dataset_name" in batch and len(batch["dataset_name"]) > batch_idx:
        dataset_name = batch["dataset_name"][batch_idx]
    episode_ref = None
    if "episode_index" in batch:
        episode_ref = f"episode={int(batch['episode_index'][batch_idx])}"
    elif "episode_id" in batch and len(batch["episode_id"]) > batch_idx:
        episode_ref = f"episode={batch['episode_id'][batch_idx]}"
    start_t = None
    if "start_t" in batch:
        start_t = int(batch["start_t"][batch_idx])
    task_description = None
    if "task_description" in batch and len(batch["task_description"]) > batch_idx:
        task_description = batch["task_description"][batch_idx]
    camera_keys = None
    if "camera_keys" in batch and len(batch["camera_keys"]) > batch_idx:
        camera_keys = batch["camera_keys"][batch_idx]
    frame_indices = None
    if "frame_indices" in batch:
        frame_indices = batch["frame_indices"][batch_idx].tolist()

    if "all_view_images" in batch:
        timeline = _render_image_timeline(
            batch["all_view_images"][batch_idx],
            frame_indices=frame_indices,
            camera_keys=camera_keys,
            dataset_name=dataset_name,
            episode_ref=episode_ref,
        )
        caption = _join_caption(
            [
                dataset_name or "dataset",
                episode_ref or "episode=?",
                f"start_t={start_t}" if start_t is not None else None,
                task_description or "task",
            ]
        )
        log_dict[f"{prefix}/views_timeline"] = wandb.Image(timeline, caption=caption)
        all_view_images = batch["all_view_images"][batch_idx]
        for t in range(all_view_images.shape[0]):
            frame_id = None if frame_indices is None or t >= len(frame_indices) else int(frame_indices[t])
            for v in range(all_view_images.shape[1]):
                cam_name = None if camera_keys is None or v >= len(camera_keys) else str(camera_keys[v]).split(".")[-1]
                frame_key = f"{prefix}/frames/t{t:02d}_v{v:02d}"
                frame_caption = _join_caption(
                    [
                        dataset_name or "dataset",
                        episode_ref or "episode=?",
                        f"start_t={start_t}" if start_t is not None else None,
                        f"frame={frame_id}" if frame_id is not None else f"t={t}",
                        f"view={v}",
                        cam_name,
                        task_description or "task",
                    ]
                )
                frame_img = _annotate_uint8_image(
                    _to_uint8_image(all_view_images[t, v]),
                    [
                        dataset_name,
                        episode_ref,
                        f"start={start_t} frame={frame_id}" if frame_id is not None and start_t is not None else None,
                        f"t={t} v={v}",
                        cam_name,
                    ],
                )
                log_dict[frame_key] = wandb.Image(frame_img, caption=frame_caption)

    gt_seq = _flatten_chunked_actions(gt_actions[batch_idx].detach().to(dtype=torch.float32).cpu().numpy())
    log_dict[f"{prefix}/actions_gt_heatmap"] = wandb.Image(
        _render_heatmap(gt_seq, "GT Actions", list(ACTION_DIM_NAMES))
    )

    if pred_actions is not None:
        pred_seq = _flatten_chunked_actions(
            pred_actions[batch_idx].detach().to(dtype=torch.float32).cpu().numpy()
        )
        log_dict[f"{prefix}/actions_pred_heatmap"] = wandb.Image(
            _render_heatmap(pred_seq, "Pred Actions", list(ACTION_DIM_NAMES))
        )
        diff_seq = pred_seq - gt_seq
        log_dict[f"{prefix}/actions_error_heatmap"] = wandb.Image(
            _render_heatmap(diff_seq, "Pred - GT Actions", list(ACTION_DIM_NAMES))
        )

    if "proprioception" in batch:
        proprio = batch["proprioception"][batch_idx].detach().to(dtype=torch.float32).cpu().numpy()
        proprio_labels = [f"p{i}" for i in range(proprio.shape[-1])]
        log_dict[f"{prefix}/proprio_heatmap"] = wandb.Image(
            _render_heatmap(proprio, "Proprioception", proprio_labels)
        )

    if log_dict:
        wandb.log(log_dict, step=step)


def _render_trajectory_3views(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    gt_gripper: np.ndarray,
    pred_gripper: np.ndarray,
) -> np.ndarray:
    """Render GT (green) vs Pred (red) 3D trajectories from XY, XZ, YZ views.

    Returns an (H, W, 3) uint8 image with 3 subplots side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    view_configs = [
        ("X", "Y", 0, 1),
        ("X", "Z", 0, 2),
        ("Y", "Z", 1, 2),
    ]

    for ax, (xlabel, ylabel, ix, iy) in zip(axes, view_configs):
        # GT trajectory
        ax.plot(gt_positions[:, ix], gt_positions[:, iy],
                "o-", color="green", markersize=4, linewidth=2, label="GT", alpha=0.8)
        # Pred trajectory
        ax.plot(pred_positions[:, ix], pred_positions[:, iy],
                "s--", color="red", markersize=4, linewidth=2, label="Pred", alpha=0.8)
        # Start point
        ax.plot(gt_positions[0, ix], gt_positions[0, iy],
                "*", color="blue", markersize=12, zorder=5)

        # Gripper state as marker size (open=big, closed=small)
        for t in range(len(gt_gripper)):
            size = 8 if gt_gripper[t] > 0 else 3
            ax.plot(gt_positions[t + 1, ix], gt_positions[t + 1, iy],
                    "o", color="green", markersize=size, alpha=0.5)
        for t in range(len(pred_gripper)):
            size = 8 if pred_gripper[t] > 0 else 3
            ax.plot(pred_positions[t + 1, ix], pred_positions[t + 1, iy],
                    "s", color="red", markersize=size, alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Action Trajectory: GT (green) vs Pred (red)", fontsize=12)
    fig.tight_layout()

    # Render to numpy
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _render_trajectory_3d(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    gt_gripper: np.ndarray,
    pred_gripper: np.ndarray,
    step: int,
):
    """Create interactive 3D trajectory plot for WandB using plotly."""
    import plotly.graph_objects as go
    import wandb

    fig = go.Figure()

    # GT trajectory (green)
    fig.add_trace(go.Scatter3d(
        x=gt_positions[:, 0], y=gt_positions[:, 1], z=gt_positions[:, 2],
        mode="lines+markers",
        marker=dict(size=4, color="green", opacity=0.8),
        line=dict(color="green", width=3),
        name="GT",
    ))

    # Pred trajectory (red)
    fig.add_trace(go.Scatter3d(
        x=pred_positions[:, 0], y=pred_positions[:, 1], z=pred_positions[:, 2],
        mode="lines+markers",
        marker=dict(size=4, color="red", opacity=0.8),
        line=dict(color="red", width=3, dash="dash"),
        name="Pred",
    ))

    # Start point (blue)
    fig.add_trace(go.Scatter3d(
        x=[gt_positions[0, 0]], y=[gt_positions[0, 1]], z=[gt_positions[0, 2]],
        mode="markers",
        marker=dict(size=8, color="blue", symbol="diamond"),
        name="Start",
    ))

    # GT endpoint
    fig.add_trace(go.Scatter3d(
        x=[gt_positions[-1, 0]], y=[gt_positions[-1, 1]], z=[gt_positions[-1, 2]],
        mode="markers",
        marker=dict(size=6, color="darkgreen", symbol="x"),
        name="GT end",
    ))

    # Pred endpoint
    fig.add_trace(go.Scatter3d(
        x=[pred_positions[-1, 0]], y=[pred_positions[-1, 1]], z=[pred_positions[-1, 2]],
        mode="markers",
        marker=dict(size=6, color="darkred", symbol="x"),
        name="Pred end",
    ))

    fig.update_layout(
        title=f"Step {step}: GT (green) vs Pred (red)",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        width=700, height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return wandb.Plotly(fig)


def _render_per_dim_actions(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
) -> np.ndarray:
    """Render per-dimension action comparison (T, 7) as line plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    n_dims = gt_actions.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(n_dims * 2.5, 3))
    if n_dims == 1:
        axes = [axes]

    timesteps = np.arange(len(gt_actions))
    for d, ax in enumerate(axes):
        name = dim_names[d] if d < len(dim_names) else f"dim{d}"
        ax.plot(timesteps, gt_actions[:, d], "o-", color="green", markersize=3, label="GT")
        ax.plot(timesteps, pred_actions[:, d], "s--", color="red", markersize=3, label="Pred")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Per-dim Actions: GT vs Pred", fontsize=11)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


@torch.no_grad()
def log_action_trajectory(
    gt_actions: torch.Tensor,
    pred_actions: torch.Tensor,
    step: int,
    wandb_run=None,
    batch_idx: int = 0,
    prefix: str = "action_vis",
    caption_note: str = "",
    log_dict: Optional[Dict] = None,
):
    """Log 3D action trajectory visualization to WandB.

    GT and Pred are both OSC commands in the same 7-dim space, so cumsum
    gives directly comparable trajectories (same coordinate system & scale).

    Args:
        gt_actions: (B, T, n_dims) ground truth OSC commands.
        pred_actions: (B, T, n_dims) predicted OSC commands.
        step: Training step.
        wandb_run: WandB run object.
        batch_idx: Which sample in the batch to visualize.
        prefix: WandB key prefix (use different prefixes for GT-prop vs Pred-prop
                to avoid key collisions).
    """
    if wandb_run is None:
        return

    import wandb

    gt_raw = gt_actions[batch_idx].detach().to(dtype=torch.float32).cpu().numpy()
    pred_raw = pred_actions[batch_idx].detach().to(dtype=torch.float32).cpu().numpy()
    chunk_size = int(gt_raw.shape[1]) if gt_raw.ndim == 3 else 1
    token_horizon = int(gt_raw.shape[0]) if gt_raw.ndim >= 2 else 0
    gt = _flatten_chunked_actions(gt_raw)
    pred = _flatten_chunked_actions(pred_raw)
    horizon = min(len(gt), len(pred))
    if horizon <= 0:
        return
    gt = gt[:horizon]
    pred = pred[:horizon]

    target_log = log_dict if log_dict is not None else {}

    # 3D trajectory: cumsum of OSC commands (GT & Pred in same space)
    gt_pos = _actions_to_trajectory(gt)
    pred_pos = _actions_to_trajectory(pred)
    gt_gripper = gt[:, -1] if gt.shape[1] >= 7 else np.zeros(len(gt))
    pred_gripper = pred[:, -1] if pred.shape[1] >= 7 else np.zeros(len(pred))

    # 2D projections (XY, XZ, YZ)
    traj_img = _render_trajectory_3views(gt_pos, pred_pos, gt_gripper, pred_gripper)
    target_log[f"{prefix}/trajectory_3view"] = wandb.Image(
        traj_img,
        caption=(
            f"Step {step}: GT(green) vs Pred(red), "
            f"flattened action horizon={horizon}, chunk_size={chunk_size}. "
            f"{caption_note}"
        ),
    )

    if chunk_size > 1:
        gt_token = _collapse_chunked_actions_to_tokens(gt_raw)
        pred_token = _collapse_chunked_actions_to_tokens(pred_raw)
        token_horizon_cmp = min(len(gt_token), len(pred_token))
        if token_horizon_cmp > 0:
            gt_token = gt_token[:token_horizon_cmp]
            pred_token = pred_token[:token_horizon_cmp]
            gt_token_pos = _actions_to_trajectory(gt_token)
            pred_token_pos = _actions_to_trajectory(pred_token)
            gt_token_gripper = gt_token[:, -1] if gt_token.shape[1] >= 7 else np.zeros(len(gt_token))
            pred_token_gripper = pred_token[:, -1] if pred_token.shape[1] >= 7 else np.zeros(len(pred_token))
            token_img = _render_trajectory_3views(
                gt_token_pos,
                pred_token_pos,
                gt_token_gripper,
                pred_token_gripper,
            )
            target_log[f"{prefix}/trajectory_token_3view"] = wandb.Image(
                token_img,
                caption=(
                    f"Step {step}: token-level diagnostic trajectory, "
                    f"token horizon={token_horizon_cmp}, chunk_size={chunk_size}. "
                    "XYZ deltas are summed inside each chunk; gripper uses chunk tail. "
                    f"{caption_note}"
                ),
            )

    # 3D interactive plot
    try:
        traj_3d = _render_trajectory_3d(gt_pos, pred_pos, gt_gripper, pred_gripper, step)
        target_log[f"{prefix}/trajectory_3d"] = traj_3d
    except Exception:
        pass

    # Per-dimension comparison
    dim_img = _render_per_dim_actions(gt, pred)
    target_log[f"{prefix}/per_dim"] = wandb.Image(
        dim_img, caption=f"Step {step}: Per-dim action comparison"
    )

    # Scalar metrics
    l1_xyz = np.abs(gt[:, :3] - pred[:, :3]).mean()
    l1_all = np.abs(gt - pred).mean()
    target_log[f"{prefix}/l1_xyz"] = l1_xyz
    target_log[f"{prefix}/l1_all"] = l1_all
    target_log[f"{prefix}/horizon_flat_actions"] = int(horizon)
    target_log[f"{prefix}/chunk_size"] = int(chunk_size)
    target_log[f"{prefix}/token_horizon"] = int(token_horizon)

    if log_dict is None:
        wandb.log(target_log, step=step)


def _quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert (N, 4) xyzw quaternion to (N, 3, 3) rotation matrix."""
    i, j, k, r = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    R = np.zeros((len(quat_xyzw), 3, 3))
    R[:, 0, 0] = 1 - 2 * (j * j + k * k)
    R[:, 0, 1] = 2 * (i * j - k * r)
    R[:, 0, 2] = 2 * (i * k + j * r)
    R[:, 1, 0] = 2 * (i * j + k * r)
    R[:, 1, 1] = 1 - 2 * (i * i + k * k)
    R[:, 1, 2] = 2 * (j * k - i * r)
    R[:, 2, 0] = 2 * (i * k - j * r)
    R[:, 2, 1] = 2 * (j * k + i * r)
    R[:, 2, 2] = 1 - 2 * (i * i + j * j)
    return R


def _quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular distance in degrees between (N,4) quaternion pairs."""
    dot = np.abs(np.sum(q1 * q2, axis=-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def _draw_frustum_2d(ax, pos, forward, right, color, label, scale=0.15):
    """Draw a camera frustum in 2D (trapezoid + direction)."""
    tip = pos + forward * scale * 1.5
    left_corner = pos + (forward - right * 0.6) * scale
    right_corner = pos + (forward + right * 0.6) * scale
    # Frustum body
    verts = np.array([pos, left_corner, tip, right_corner, pos])
    ax.fill(verts[:, 0], verts[:, 1], alpha=0.15, color=color)
    ax.plot(verts[:, 0], verts[:, 1], color=color, linewidth=1.5, alpha=0.8)
    ax.plot(*pos, "o", color=color, markersize=6, label=label)


def log_camera_visualization(
    teacher_pose_enc: torch.Tensor,
    student_pose_enc: torch.Tensor,
    step: int,
    wandb_run=None,
    n_views: int = 2,
    image_hw: tuple = (224, 224),
    prefix: str = "camera_vis",
):
    """Log teacher vs student camera pose visualization to wandb.

    Layout: 4-panel figure
      [Top-down BEV (XZ)] [Side view (XY)] [Per-view params] [Per-view error]
    Plus per-view scalar metrics with angular quaternion distance.

    Args:
        teacher_pose_enc: (B*V, 9) teacher pose encoding [t(3), qvec(4), fov(2)]
        student_pose_enc: (B*V, 9) student pose encoding
        step: training step
        wandb_run: wandb run object
        n_views: number of views per sample
        image_hw: (H, W) for intrinsic reconstruction
    """
    if wandb_run is None:
        return
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teacher = teacher_pose_enc.detach().float().cpu()
    student = student_pose_enc.detach().float().cpu()

    t_pose = teacher[:n_views]  # (V, 9)
    s_pose = student[:n_views]

    t_trans = t_pose[:, :3].numpy()
    s_trans = s_pose[:, :3].numpy()
    t_quat = t_pose[:, 3:7].numpy()
    s_quat = s_pose[:, 3:7].numpy()
    t_fov = t_pose[:, 7:9].numpy()
    s_fov = s_pose[:, 7:9].numpy()

    t_R = _quat_to_rotmat(t_quat)  # (V, 3, 3)
    s_R = _quat_to_rotmat(s_quat)

    # Forward = R @ [0, 0, -1], Right = R @ [1, 0, 0]
    t_fwd = -t_R[:, :, 2]  # (V, 3)
    s_fwd = -s_R[:, :, 2]
    t_right = t_R[:, :, 0]
    s_right = s_R[:, :, 0]

    view_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_views]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Top-down BEV (X-Z plane) ---
    ax_bev = axes[0, 0]
    for v in range(n_views):
        c = view_colors[v]
        # Teacher frustum
        _draw_frustum_2d(
            ax_bev,
            t_trans[v, [0, 2]], t_fwd[v, [0, 2]], t_right[v, [0, 2]],
            color=c, label=f"T-V{v}" if v < 2 else None, scale=0.12,
        )
        # Student frustum (dashed outline)
        s_tip = s_trans[v, [0, 2]] + s_fwd[v, [0, 2]] * 0.12 * 1.5
        s_lc = s_trans[v, [0, 2]] + (s_fwd[v, [0, 2]] - s_right[v, [0, 2]] * 0.6) * 0.12
        s_rc = s_trans[v, [0, 2]] + (s_fwd[v, [0, 2]] + s_right[v, [0, 2]] * 0.6) * 0.12
        sv = np.array([s_trans[v, [0, 2]], s_lc, s_tip, s_rc, s_trans[v, [0, 2]]])
        ax_bev.plot(sv[:, 0], sv[:, 1], "--", color=c, linewidth=1.5, alpha=0.6)
        ax_bev.plot(*s_trans[v, [0, 2]], "^", color=c, markersize=6,
                    label=f"S-V{v}" if v < 2 else None)
        # Error line
        ax_bev.plot(
            [t_trans[v, 0], s_trans[v, 0]], [t_trans[v, 2], s_trans[v, 2]],
            ":", color="gray", linewidth=0.8, alpha=0.5,
        )
    ax_bev.set_xlabel("X")
    ax_bev.set_ylabel("Z")
    ax_bev.set_title("Top-Down (BEV, XZ)")
    ax_bev.legend(fontsize=7, loc="upper right")
    ax_bev.set_aspect("equal", adjustable="datalim")
    ax_bev.grid(True, alpha=0.3)

    # --- Panel 2: Side view (X-Y plane) ---
    ax_side = axes[0, 1]
    for v in range(n_views):
        c = view_colors[v]
        _draw_frustum_2d(
            ax_side,
            t_trans[v, [0, 1]], t_fwd[v, [0, 1]], t_right[v, [0, 1]],
            color=c, label=f"T-V{v}" if v < 2 else None, scale=0.12,
        )
        s_tip = s_trans[v, [0, 1]] + s_fwd[v, [0, 1]] * 0.12 * 1.5
        s_lc = s_trans[v, [0, 1]] + (s_fwd[v, [0, 1]] - s_right[v, [0, 1]] * 0.6) * 0.12
        s_rc = s_trans[v, [0, 1]] + (s_fwd[v, [0, 1]] + s_right[v, [0, 1]] * 0.6) * 0.12
        sv = np.array([s_trans[v, [0, 1]], s_lc, s_tip, s_rc, s_trans[v, [0, 1]]])
        ax_side.plot(sv[:, 0], sv[:, 1], "--", color=c, linewidth=1.5, alpha=0.6)
        ax_side.plot(*s_trans[v, [0, 1]], "^", color=c, markersize=6,
                     label=f"S-V{v}" if v < 2 else None)
        ax_side.plot(
            [t_trans[v, 0], s_trans[v, 0]], [t_trans[v, 1], s_trans[v, 1]],
            ":", color="gray", linewidth=0.8, alpha=0.5,
        )
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Y")
    ax_side.set_title("Side View (XY)")
    ax_side.legend(fontsize=7, loc="upper right")
    ax_side.set_aspect("equal", adjustable="datalim")
    ax_side.grid(True, alpha=0.3)

    # --- Panel 3: Per-view parameter comparison (grouped bar) ---
    ax_params = axes[1, 0]
    param_labels = ["tx", "ty", "tz", "qx", "qy", "qz", "qw", "fov_h", "fov_w"]
    x = np.arange(len(param_labels))
    bar_w = 0.8 / (n_views * 2)
    for v in range(n_views):
        offset_t = (v * 2) * bar_w - 0.4 + bar_w / 2
        offset_s = (v * 2 + 1) * bar_w - 0.4 + bar_w / 2
        ax_params.bar(x + offset_t, t_pose[v].numpy(), bar_w,
                      color=view_colors[v], alpha=0.8, label=f"T-V{v}")
        ax_params.bar(x + offset_s, s_pose[v].numpy(), bar_w,
                      color=view_colors[v], alpha=0.35, label=f"S-V{v}",
                      edgecolor=view_colors[v], linewidth=1)
    ax_params.set_xticks(x)
    ax_params.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=8)
    ax_params.set_title("Pose Parameters (all views)")
    ax_params.legend(fontsize=6, ncol=n_views, loc="upper right")
    ax_params.grid(True, axis="y", alpha=0.3)

    # --- Panel 4: Per-view error breakdown ---
    ax_err = axes[1, 1]
    group_labels = ["trans (mm)", "rot (deg)", "fov (rad)"]
    gx = np.arange(len(group_labels))
    bar_w_err = 0.25
    for v in range(n_views):
        trans_err_v = np.linalg.norm(t_trans[v] - s_trans[v])
        rot_err_v = _quat_angular_distance(t_quat[v:v+1], s_quat[v:v+1])[0]
        fov_err_v = np.linalg.norm(t_fov[v] - s_fov[v])
        vals = [trans_err_v * 1000, rot_err_v, fov_err_v]
        offset = (v - (n_views - 1) / 2) * bar_w_err
        bars = ax_err.bar(gx + offset, vals, bar_w_err, color=view_colors[v],
                          alpha=0.8, label=f"V{v}")
        for bar, val in zip(bars, vals):
            ax_err.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax_err.set_xticks(gx)
    ax_err.set_xticklabels(group_labels, fontsize=9)
    ax_err.set_title("Error by Component")
    ax_err.legend(fontsize=7)
    ax_err.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Camera Pose: Teacher (solid) vs Student (dashed) : step {step}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    wandb.log({f"{prefix}/camera_poses": wandb.Image(fig)}, step=step)
    plt.close(fig)

    # --- Scalar metrics (per-view + mean) ---
    metrics = {}
    trans_errs, rot_errs, fov_errs = [], [], []
    for v in range(n_views):
        te = np.linalg.norm(t_trans[v] - s_trans[v])
        re = _quat_angular_distance(t_quat[v:v+1], s_quat[v:v+1])[0]
        fe = np.linalg.norm(t_fov[v] - s_fov[v])
        metrics[f"{prefix}/v{v}_trans_err"] = te
        metrics[f"{prefix}/v{v}_rot_err_deg"] = re
        metrics[f"{prefix}/v{v}_fov_err"] = fe
        trans_errs.append(te)
        rot_errs.append(re)
        fov_errs.append(fe)
    metrics[f"{prefix}/translation_error"] = np.mean(trans_errs)
    metrics[f"{prefix}/rotation_error_deg"] = np.mean(rot_errs)
    metrics[f"{prefix}/fov_error"] = np.mean(fov_errs)
    wandb.log(metrics, step=step)
