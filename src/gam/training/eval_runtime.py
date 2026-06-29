"""Closed-loop eval runtime helpers used by training."""

from __future__ import annotations

import logging
import math
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from .distributed import _plain_config_container


def normalize_closed_loop_eval_profiles(training_cfg):
    """Return enabled closed-loop eval profiles plus legacy-prefix mode."""
    profiles_raw = _plain_config_container(training_cfg.get("closed_loop_evals", None))
    legacy_mode = profiles_raw is None
    if profiles_raw is None:
        legacy_raw = _plain_config_container(training_cfg.get("closed_loop_eval", {}))
        legacy_cfg = dict(legacy_raw) if legacy_raw else {}
        profiles = [legacy_cfg] if bool(legacy_cfg.get("enabled", False)) else []
    else:
        if not isinstance(profiles_raw, (list, tuple)):
            raise TypeError("training.closed_loop_evals must be a list of profile dictionaries.")
        profiles = []
        for idx, item in enumerate(profiles_raw):
            item = _plain_config_container(item)
            if not item:
                continue
            profile = dict(item)
            profile.setdefault("name", f"profile{idx}")
            if bool(profile.get("enabled", False)):
                profiles.append(profile)
    for idx, profile in enumerate(profiles):
        name = str(profile.get("name") or f"profile{idx}")
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or f"profile{idx}"
        profile["name"] = safe_name
        profile["_wandb_prefix"] = "rollout" if legacy_mode else f"rollout/{safe_name}"
    return profiles, legacy_mode


def prepare_rollout_video_frames(frames, macro_block_size=16):
    """Convert variable-sized rollout debug frames into a single mp4-safe stack."""
    arrays = []
    max_h = 0
    max_w = 0
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            continue
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(arr)
        arrays.append(arr)
        max_h = max(max_h, int(arr.shape[0]))
        max_w = max(max_w, int(arr.shape[1]))
    if not arrays:
        return []

    block = max(1, int(macro_block_size))
    target_h = int(math.ceil(max_h / block) * block)
    target_w = int(math.ceil(max_w / block) * block)
    packed = []
    for arr in arrays:
        h, w = int(arr.shape[0]), int(arr.shape[1])
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:h, :w, :] = arr
        packed.append(canvas)
    return packed


def closed_loop_video_dir(
    *,
    profile_cfg: dict,
    experiment_dir: str,
    profile_name: str,
    train_steps: int,
    benchmark: str,
) -> str:
    explicit = profile_cfg.get("_video_dir") or profile_cfg.get("video_dir")
    if explicit:
        return str(explicit)

    return os.path.join(
        experiment_dir,
        "rollout_videos",
        profile_name,
        f"step_{int(train_steps):07d}",
    )


def shutdown_train_loader_workers_for_closed_loop_eval(
    loader: DataLoader,
    data_iter,
    *,
    logger: logging.Logger,
    rank: int,
    step: int,
):
    """Stop DataLoader workers before long simulator eval."""
    try:
        n_workers = int(getattr(loader, "num_workers", 0) or 0)
    except Exception:
        n_workers = 0
    if n_workers <= 0:
        return data_iter

    candidates = []
    if data_iter is not None:
        candidates.append(data_iter)
    persistent_iter = getattr(loader, "_iterator", None)
    if persistent_iter is not None and all(persistent_iter is not item for item in candidates):
        candidates.append(persistent_iter)

    shutdown_count = 0
    for iterator in candidates:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if not callable(shutdown):
            continue
        if bool(getattr(iterator, "_shutdown", False)):
            continue
        try:
            shutdown()
            shutdown_count += 1
        except Exception as exc:
            if rank == 0:
                logger.warning(
                    "[step=%07d] DataLoader worker shutdown before closed-loop eval failed: %s",
                    int(step),
                    exc,
                )
    if hasattr(loader, "_iterator"):
        try:
            loader._iterator = None
        except Exception:
            pass
    if rank == 0 and (shutdown_count > 0 or persistent_iter is not None):
        logger.info(
            "[step=%07d] shut down %d DataLoader iterator(s) before closed-loop eval; "
            "workers will be recreated after eval",
            int(step),
            int(shutdown_count),
        )
    return None


def gather_eval_tensors(preds_list, gts_list, keys_list, world_size, rank, device):
    """All-gather eval predictions, targets, and keys across ranks."""
    local_pred = torch.cat(preds_list, dim=0) if preds_list else torch.empty(0, device=device)
    local_gt = torch.cat(gts_list, dim=0) if gts_list else torch.empty(0, device=device)
    if world_size <= 1:
        return local_pred, local_gt, keys_list, local_pred.shape[0]
    import torch.distributed as _dist

    local_n = torch.tensor([local_pred.shape[0]], device=device)
    n_list = [torch.zeros_like(local_n) for _ in range(world_size)]
    _dist.all_gather(n_list, local_n)
    max_n = int(max(x.item() for x in n_list))
    if max_n == 0:
        return local_pred, local_gt, [], 0
    if local_pred.shape[0] < max_n:
        pad = max_n - local_pred.shape[0]
        if local_pred.shape[0] > 0:
            local_pred = torch.cat([local_pred, local_pred[:1].expand(pad, *local_pred.shape[1:])], dim=0)
            local_gt = torch.cat([local_gt, local_gt[:1].expand(pad, *local_gt.shape[1:])], dim=0)
        else:
            local_pred = torch.zeros((max_n, *local_pred.shape[1:]) if local_pred.dim() > 1 else (max_n,), device=device)
            local_gt = torch.zeros_like(local_pred)
    pred_shards = [torch.zeros_like(local_pred) for _ in range(world_size)]
    gt_shards = [torch.zeros_like(local_gt) for _ in range(world_size)]
    _dist.all_gather(pred_shards, local_pred)
    _dist.all_gather(gt_shards, local_gt)
    gathered_keys = [None for _ in range(world_size)]
    _dist.all_gather_object(gathered_keys, keys_list)
    if rank == 0:
        preds = torch.cat([pred_shards[r][:int(n_list[r].item())] for r in range(world_size)], dim=0)
        gts = torch.cat([gt_shards[r][:int(n_list[r].item())] for r in range(world_size)], dim=0)
        keys = [k for rk in gathered_keys for k in (rk or [])]
        count = sum(int(x.item()) for x in n_list)
        return preds, gts, keys, count
    return local_pred, local_gt, [], 0


def gather_variable_eval_tensor(tensors_list, world_size, rank, device, *, dtype=None):
    """All-gather a variable-length tensor list along dim 0."""
    local = torch.cat(tensors_list, dim=0) if tensors_list else None
    local_n_value = 0 if local is None else int(local.shape[0])
    local_n = torch.tensor([local_n_value], device=device, dtype=torch.long)
    if dtype is None and local is not None:
        dtype = local.dtype
    if dtype is None:
        dtype = torch.float32

    if world_size <= 1:
        if local is None:
            return torch.empty((0,), device=device, dtype=dtype), 0
        return local.to(device=device, dtype=dtype), local_n_value

    import torch.distributed as _dist

    trailing_shape = tuple(local.shape[1:]) if local is not None else None
    shape_list = [None for _ in range(world_size)]
    _dist.all_gather_object(shape_list, trailing_shape)
    resolved_shape = next((shape for shape in shape_list if shape is not None), ())
    if local is None:
        local = torch.empty((0, *resolved_shape), device=device, dtype=dtype)
    else:
        local = local.to(device=device, dtype=dtype)

    n_list = [torch.zeros_like(local_n) for _ in range(world_size)]
    _dist.all_gather(n_list, local_n)
    max_n = int(max(x.item() for x in n_list))
    if max_n == 0:
        return local, 0
    if local.shape[0] < max_n:
        pad = max_n - local.shape[0]
        if local.shape[0] > 0:
            pad_rows = local[:1].expand(pad, *local.shape[1:])
        else:
            pad_rows = torch.zeros((pad, *resolved_shape), device=device, dtype=local.dtype)
        local = torch.cat([local, pad_rows], dim=0)
    shards = [torch.zeros_like(local) for _ in range(world_size)]
    _dist.all_gather(shards, local)
    if rank == 0:
        gathered = torch.cat([shards[r][: int(n_list[r].item())] for r in range(world_size)], dim=0)
        return gathered, sum(int(x.item()) for x in n_list)
    return local, 0
