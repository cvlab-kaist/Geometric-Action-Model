"""Action metrics and stats logging helpers for Stage 1 training.

Extracted from ``train_robot.py`` to keep the training entrypoint smaller.
These functions are behavior-preserving and self-contained: they depend only
on stdlib/torch plus ``robot.data.dataset`` statistics helpers.
"""

from __future__ import annotations

import torch

from robot.data.dataset import (
    compute_action_statistics,
    summarize_action_statistics,
)


ACTION_DIM_NAMES = ("dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper")


def _masked_mean(value, mask=None):
    if mask is None:
        return value.mean()
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def compute_action_metrics(action_pred, gt_actions, action_loss_mask=None):
    """Compute action loss and metrics in normalized [-1, 1] space.

    Accuracy thresholds (fraction of predictions within tolerance):
      acc_5pct: |error| < 0.1  (5% of [-1,1] range)
      acc_10pct: |error| < 0.2 (10% of [-1,1] range)
    """
    abs_err = (action_pred - gt_actions).abs()
    loss_action = _masked_mean(abs_err, action_loss_mask)
    with torch.no_grad():
        sq_err = (action_pred - gt_actions) ** 2
        action_mse = _masked_mean(sq_err, action_loss_mask)
        acc_1pct = _masked_mean((abs_err < 0.02).float(), action_loss_mask)
        acc_5pct = _masked_mean((abs_err < 0.1).float(), action_loss_mask)
        acc_10pct = _masked_mean((abs_err < 0.2).float(), action_loss_mask)
    return loss_action, action_mse, acc_1pct, acc_5pct, acc_10pct


def reshape_action_sequence(actions):
    """Normalize action tensors to (B, S, D), flattening chunk dimension if present."""
    if actions.ndim == 4:
        batch_size, timesteps, chunk_size, dim = actions.shape
        return actions.reshape(batch_size, timesteps * chunk_size, dim)
    if actions.ndim == 3:
        return actions
    raise ValueError(f"Expected action tensor with 3 or 4 dims, got {actions.shape}.")


def reshape_action_mask(action_mask, actions):
    if action_mask is None:
        return None
    if actions.ndim == 4:
        batch_size, timesteps, chunk_size, _ = actions.shape
        if action_mask.ndim == 2:
            action_mask = action_mask.unsqueeze(-1).expand(batch_size, timesteps, chunk_size)
        elif action_mask.ndim == 3 and action_mask.shape[-1] == 1 and chunk_size > 1:
            action_mask = action_mask.expand(batch_size, timesteps, chunk_size)
        elif action_mask.ndim == 4:
            return action_mask.reshape(batch_size, timesteps * chunk_size, action_mask.shape[-1])
        return action_mask.reshape(batch_size, timesteps * chunk_size)
    if actions.ndim == 3:
        if action_mask.ndim == 3 and action_mask.shape[-1] == 1:
            action_mask = action_mask.squeeze(-1)
        if action_mask.ndim not in (2, 3):
            raise ValueError(f"Expected action mask with 2 dims for action tensor {actions.shape}, got {action_mask.shape}.")
        return action_mask
    raise ValueError(f"Expected action tensor with 3 or 4 dims, got {actions.shape}.")


def compute_action_detail_metrics(action_pred, gt_actions, action_loss_mask=None):
    """Compute chunk-aware regression metrics in the given action space."""
    with torch.no_grad():
        pred_seq = reshape_action_sequence(action_pred.detach())
        gt_seq = reshape_action_sequence(gt_actions.detach())
        mask_seq = reshape_action_mask(action_loss_mask.detach(), action_pred.detach()) if action_loss_mask is not None else None
        if mask_seq is not None:
            mask_seq = mask_seq.to(device=pred_seq.device, dtype=torch.bool)
            if mask_seq.ndim == pred_seq.ndim:
                valid_bool = mask_seq
                slot_mask = valid_bool.any(dim=-1)
            else:
                slot_mask = mask_seq
                valid_bool = mask_seq.unsqueeze(-1).expand_as(pred_seq)
            valid = valid_bool.to(dtype=pred_seq.dtype)
            slot_mask_f = slot_mask.to(dtype=pred_seq.dtype)
        else:
            valid_bool = None
            valid = None
            slot_mask = None
            slot_mask_f = None
        diff = pred_seq - gt_seq
        abs_err = diff.abs()
        sq_err = diff.square()

        if valid is None:
            per_dim_mae = abs_err.mean(dim=(0, 1))
            per_dim_mse = sq_err.mean(dim=(0, 1))
            timestep_mae = abs_err.mean(dim=(0, 2))
            timestep_translation_mae = abs_err[:, :, :3].mean(dim=(0, 2))
            timestep_rotation_mae = abs_err[:, :, 3:6].mean(dim=(0, 2))
            timestep_gripper_mae = abs_err[:, :, 6].mean(dim=0)
            seq_acc_5pct = (abs_err < 0.1).all(dim=-1).float().mean()
            seq_acc_10pct = (abs_err < 0.2).all(dim=-1).float().mean()
            rel_acc_1pct = (abs_err < 0.02).float().mean()
            rel_acc_5pct = (abs_err < 0.1).float().mean()
            rel_acc_10pct = (abs_err < 0.2).float().mean()
            centered_gt = gt_seq - gt_seq.mean(dim=(0, 1), keepdim=True)
            per_dim_var = centered_gt.square().mean(dim=(0, 1))
            overall_mse = sq_err.mean()
            overall_var = centered_gt.square().mean()
        else:
            per_dim_den = valid.sum(dim=(0, 1)).clamp_min(1.0)
            per_dim_mae = (abs_err * valid).sum(dim=(0, 1)) / per_dim_den
            per_dim_mse = (sq_err * valid).sum(dim=(0, 1)) / per_dim_den
            timestep_den = valid.sum(dim=(0, 2)).clamp_min(1.0)
            timestep_mae = (abs_err * valid).sum(dim=(0, 2)) / timestep_den
            valid_trans = valid[:, :, :3]
            valid_rot = valid[:, :, 3:6]
            valid_grip = valid[:, :, 6]
            trans_den = valid_trans.sum(dim=(0, 2)).clamp_min(1.0)
            rot_den = valid_rot.sum(dim=(0, 2)).clamp_min(1.0)
            grip_den = valid_grip.sum(dim=0).clamp_min(1.0)
            timestep_translation_mae = (abs_err[:, :, :3] * valid_trans).sum(dim=(0, 2)) / trans_den
            timestep_rotation_mae = (abs_err[:, :, 3:6] * valid_rot).sum(dim=(0, 2)) / rot_den
            timestep_gripper_mae = (abs_err[:, :, 6] * valid_grip).sum(dim=0) / grip_den
            seq_den = slot_mask_f.sum().clamp_min(1.0)
            seq_acc_5pct = (((abs_err < 0.1) | ~valid_bool).all(dim=-1).float() * slot_mask_f).sum() / seq_den
            seq_acc_10pct = (((abs_err < 0.2) | ~valid_bool).all(dim=-1).float() * slot_mask_f).sum() / seq_den
            rel_acc_1pct = ((abs_err < 0.02).float() * valid).sum() / valid.sum().clamp_min(1.0)
            rel_acc_5pct = ((abs_err < 0.1).float() * valid).sum() / valid.sum().clamp_min(1.0)
            rel_acc_10pct = ((abs_err < 0.2).float() * valid).sum() / valid.sum().clamp_min(1.0)
            gt_mean = (gt_seq * valid).sum(dim=(0, 1), keepdim=True) / per_dim_den.view(1, 1, -1)
            centered_gt = gt_seq - gt_mean
            per_dim_var = (centered_gt.square() * valid).sum(dim=(0, 1)) / per_dim_den
            overall_mse = (sq_err * valid).sum() / valid.sum().clamp_min(1.0)
            overall_var = (centered_gt.square() * valid).sum() / valid.sum().clamp_min(1.0)

        if valid is None:
            trans_vec_err = torch.linalg.vector_norm(diff[:, :, :3], dim=-1)
            rot_vec_err = torch.linalg.vector_norm(diff[:, :, 3:6], dim=-1)
            overall_l1 = abs_err.mean()
            translation_mae = abs_err[:, :, :3].mean()
            rotation_mae = abs_err[:, :, 3:6].mean()
            gripper_mae = abs_err[:, :, 6].mean()
            trans_vec_mae = trans_vec_err.mean()
            rot_vec_mae = rot_vec_err.mean()
        else:
            elem_den = valid.sum().clamp_min(1.0)
            valid_trans = valid[:, :, :3]
            valid_rot = valid[:, :, 3:6]
            valid_grip = valid[:, :, 6]
            trans_slot_mask = valid_trans.any(dim=-1).to(dtype=pred_seq.dtype)
            rot_slot_mask = valid_rot.any(dim=-1).to(dtype=pred_seq.dtype)
            trans_vec_err = torch.linalg.vector_norm(diff[:, :, :3] * valid_trans, dim=-1)
            rot_vec_err = torch.linalg.vector_norm(diff[:, :, 3:6] * valid_rot, dim=-1)
            overall_l1 = (abs_err * valid).sum() / elem_den
            translation_mae = (abs_err[:, :, :3] * valid_trans).sum() / valid_trans.sum().clamp_min(1.0)
            rotation_mae = (abs_err[:, :, 3:6] * valid_rot).sum() / valid_rot.sum().clamp_min(1.0)
            gripper_mae = (abs_err[:, :, 6] * valid_grip).sum() / valid_grip.sum().clamp_min(1.0)
            trans_vec_mae = (trans_vec_err * trans_slot_mask).sum() / trans_slot_mask.sum().clamp_min(1.0)
            rot_vec_mae = (rot_vec_err * rot_slot_mask).sum() / rot_slot_mask.sum().clamp_min(1.0)
        per_dim_r2 = torch.where(
            per_dim_var > 1e-12,
            1.0 - per_dim_mse / (per_dim_var + 1e-12),
            torch.zeros_like(per_dim_mse),
        )
        if float(overall_var.item()) > 1e-12:
            overall_r2 = 1.0 - overall_mse / (overall_var + 1e-12)
        else:
            overall_r2 = torch.zeros((), dtype=overall_mse.dtype, device=overall_mse.device)

    return {
        "l1": float(overall_l1.item()),
        "mse": float(overall_mse.item()),
        "r2": float(overall_r2.item()),
        "per_dim_mae": per_dim_mae.cpu().tolist(),
        "per_dim_mse": per_dim_mse.cpu().tolist(),
        "per_dim_r2": per_dim_r2.cpu().tolist(),
        "translation_mae": float(translation_mae.item()),
        "rotation_mae": float(rotation_mae.item()),
        "gripper_mae": float(gripper_mae.item()),
        "trans_vec_mae": float(trans_vec_mae.item()),
        "rot_vec_mae": float(rot_vec_mae.item()),
        "rel_acc_1pct": float(rel_acc_1pct.item()),
        "rel_acc_5pct": float(rel_acc_5pct.item()),
        "rel_acc_10pct": float(rel_acc_10pct.item()),
        "seq_acc_5pct": float(seq_acc_5pct.item()),
        "seq_acc_10pct": float(seq_acc_10pct.item()),
        "timestep_mae": timestep_mae.cpu().tolist(),
        "timestep_translation_mae": timestep_translation_mae.cpu().tolist(),
        "timestep_rotation_mae": timestep_rotation_mae.cpu().tolist(),
        "timestep_gripper_mae": timestep_gripper_mae.cpu().tolist(),
    }


def add_named_metrics(log_dict, prefix, names, values):
    for name, value in zip(names, values):
        log_dict[f"{prefix}/{name}"] = float(value)


def add_indexed_metrics(log_dict, prefix, values):
    for idx, value in enumerate(values):
        log_dict[f"{prefix}/{idx:02d}"] = float(value)


def compute_raw_action_metrics(pred_norm, gt_norm, normalizer, stats_keys=None, action_loss_mask=None):
    """Compute denormalized regression metrics in canonical raw action space."""
    with torch.no_grad():
        pred_raw = normalizer.denormalize(pred_norm.detach(), stats_keys=stats_keys)
        gt_raw = normalizer.denormalize(gt_norm.detach(), stats_keys=stats_keys)
    return compute_action_detail_metrics(pred_raw, gt_raw, action_loss_mask=action_loss_mask)


def compute_per_dataset_action_metrics(pred_norm, gt_norm, normalizer, stats_keys, action_loss_mask=None):
    """Aggregate normalized and denormalized metrics per dataset key."""
    if stats_keys is None:
        return {}

    with torch.no_grad():
        pred_norm_det = pred_norm.detach().cpu()
        gt_norm_det = gt_norm.detach().cpu()
        pred_raw_det = normalizer.denormalize(pred_norm.detach(), stats_keys=stats_keys).cpu()
        gt_raw_det = normalizer.denormalize(gt_norm.detach(), stats_keys=stats_keys).cpu()
        mask_det = action_loss_mask.detach().cpu() if action_loss_mask is not None else None

    metrics_by_dataset = {}
    for key in sorted(set(stats_keys)):
        batch_idx = [i for i, current in enumerate(stats_keys) if current == key]
        if not batch_idx:
            continue
        dataset_mask = mask_det[batch_idx] if mask_det is not None else None
        metrics_by_dataset[key] = {
            "count": len(batch_idx),
            "norm": compute_action_detail_metrics(
                pred_norm_det[batch_idx],
                gt_norm_det[batch_idx],
                action_loss_mask=dataset_mask,
            ),
            "raw": compute_action_detail_metrics(
                pred_raw_det[batch_idx],
                gt_raw_det[batch_idx],
                action_loss_mask=dataset_mask,
            ),
        }
    return metrics_by_dataset


def add_per_dataset_metrics(log_dict, prefix, metrics_by_dataset):
    for dataset_name, metrics in metrics_by_dataset.items():
        ds_prefix = f"{prefix}/{dataset_name}"
        log_dict[f"{ds_prefix}/count"] = int(metrics["count"])
        for space_name in ("norm", "raw"):
            space_metrics = metrics[space_name]
            base = f"{ds_prefix}/{space_name}"
            log_dict[f"{base}/l1"] = float(space_metrics["l1"])
            log_dict[f"{base}/mse"] = float(space_metrics["mse"])
            log_dict[f"{base}/r2"] = float(space_metrics["r2"])
            log_dict[f"{base}/translation_l1"] = float(space_metrics["translation_mae"])
            log_dict[f"{base}/rotation_l1"] = float(space_metrics["rotation_mae"])
            log_dict[f"{base}/gripper_l1"] = float(space_metrics["gripper_mae"])
            add_named_metrics(log_dict, f"{base}/per_dim_l1", ACTION_DIM_NAMES, space_metrics["per_dim_mae"])
            add_named_metrics(log_dict, f"{base}/per_dim_mse", ACTION_DIM_NAMES, space_metrics["per_dim_mse"])
            add_named_metrics(log_dict, f"{base}/per_dim_r2", ACTION_DIM_NAMES, space_metrics["per_dim_r2"])


def maybe_log_action_stats(dataset, dataset_cfg, logger, wandb_run):
    max_samples = int(dataset_cfg.get("action_stats_samples", 0))
    if max_samples <= 0:
        return
    stats = compute_action_statistics(dataset, max_samples=max_samples)
    logger.info("Action stats: %s", summarize_action_statistics(stats))
    if wandb_run is not None:
        wandb_run.summary["action/stats_keys"] = sorted(stats.keys())
        for key, item in stats.items():
            wandb_run.summary[f"action/{key}/mean"] = item["mean"].tolist()
            wandb_run.summary[f"action/{key}/std"] = item["std"].tolist()
            wandb_run.summary[f"action/{key}/min"] = item["min"].tolist()
            wandb_run.summary[f"action/{key}/max"] = item["max"].tolist()
