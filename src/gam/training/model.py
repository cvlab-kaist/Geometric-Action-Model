"""Model and batch helpers for GAM Stage 1 training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DA3FineTuneModel(nn.Module):
    """Trainable DA3 student + action head + optional GAM future predictor."""

    def __init__(
        self,
        student_da3,
        action_head,
        proprio_head=None,
        proprio_conditioner=None,
        future_predictor=None,
        text_conditioner=None,
    ):
        super().__init__()
        self.student_da3 = student_da3
        self.action_head = action_head
        self.proprio_head = proprio_head
        self.proprio_conditioner = proprio_conditioner
        self.future_predictor = future_predictor
        self.text_conditioner = text_conditioner

    def forward(self, images, proprio=None, action_input=None, force_action_input=False):
        features_per_level, action_tokens, raw_levels = self.student_da3.encode_with_actions(
            images,
            action_input=action_input,
            force_action_input=force_action_input,
        )
        action_pred = self.action_head(action_tokens)
        return action_pred, raw_levels, features_per_level


def unwrap_train_model(model):
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    return raw_model


def strip_compile_prefix_state_dict(state):
    if state is None:
        return None
    return {str(k).replace("_orig_mod.", ""): v for k, v in state.items()}


def resolve_backbone_action_input_dim(action_head_cfg, embed_dim, *, logger=None, label="action_head.input_dim"):
    raw = action_head_cfg.get("input_dim", None)
    if raw is None or str(raw).lower() in {"auto", "da3", "backbone", "embed_dim"}:
        return int(embed_dim)
    value = int(raw)
    if value == int(embed_dim):
        return value
    if value == 1536:
        if logger is not None:
            logger.warning(
                "%s=%d mismatches selected Stage 1 embed_dim=%d; using embed_dim. "
                "Set input_dim: auto or omit it when switching backbones.",
                label,
                value,
                int(embed_dim),
            )
        return int(embed_dim)
    raise ValueError(f"{label}={value} must match selected Stage 1 embed_dim={int(embed_dim)}.")


def is_auto_value(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "infer", "from_camera_keys"}


def dataset_pad_view_max(dataset_cfg: dict) -> Optional[int]:
    mode = str(dataset_cfg.get("view_mode", dataset_cfg.get("view_policy", "")) or "").strip().lower()
    enabled = bool(dataset_cfg.get("pad_views_to_max", mode in {"pad_to_max", "padded", "pad", "variable", "variable_pad"}))
    if not enabled:
        return None
    raw = dataset_cfg.get("max_views", dataset_cfg.get("view_max_views"))
    if raw is None or is_auto_value(raw):
        raise ValueError("dataset.view_mode=pad_to_max requires dataset.max_views.")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"dataset.max_views must be positive, got {raw!r}.")
    return value


def sequence_length(value) -> Optional[int]:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def resolve_da3_n_views(dataset_cfg: dict, da3_ft_cfg: dict) -> int:
    """Resolve train-time view count from explicit config or camera lists."""
    candidates: list[tuple[str, int]] = []
    pad_view_max = dataset_pad_view_max(dataset_cfg)

    def add_candidate(label: str, raw_value) -> None:
        if raw_value is None or is_auto_value(raw_value):
            return
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {raw_value!r}.")
        candidates.append((label, value))

    add_candidate("da3_finetune.n_views", da3_ft_cfg.get("n_views", None))
    add_candidate("dataset.n_views", dataset_cfg.get("n_views", None))
    if pad_view_max is not None:
        candidates.append(("dataset.max_views", int(pad_view_max)))

    camera_len = sequence_length(dataset_cfg.get("camera_keys"))
    if camera_len is not None:
        if camera_len <= 0:
            raise ValueError("dataset.camera_keys must be non-empty when provided.")
        if pad_view_max is not None:
            if int(camera_len) > int(pad_view_max):
                raise ValueError(
                    "dataset.camera_keys length exceeds dataset.max_views: "
                    f"len={camera_len}, max_views={pad_view_max}."
                )
        else:
            candidates.append(("len(dataset.camera_keys)", int(camera_len)))

    if not candidates:
        return 2

    resolved = candidates[0][1]
    mismatches = [(label, value) for label, value in candidates if value != resolved]
    if mismatches:
        details = ", ".join(f"{label}={value}" for label, value in candidates)
        raise ValueError(f"View-count config mismatch: {details}.")

    rollout_len = sequence_length(dataset_cfg.get("rollout_camera_keys"))
    if rollout_len is not None and pad_view_max is not None and rollout_len > resolved:
        raise ValueError(
            "dataset.rollout_camera_keys length must be <= resolved max_views: "
            f"len={rollout_len}, max_views={resolved}."
        )
    if rollout_len is not None and pad_view_max is None and rollout_len != resolved:
        raise ValueError(
            "dataset.rollout_camera_keys length must match resolved n_views: "
            f"len={rollout_len}, n_views={resolved}."
        )

    return int(resolved)


def sync_da3_view_count_to_cfg(cfg, dataset_cfg: dict, da3_ft_cfg: dict, n_views: int) -> None:
    dataset_cfg["n_views"] = int(n_views)
    da3_ft_cfg["n_views"] = int(n_views)
    if OmegaConf.is_config(cfg):
        OmegaConf.update(cfg, "dataset.n_views", int(n_views), merge=True)
        OmegaConf.update(cfg, "da3_finetune.n_views", int(n_views), merge=True)


def normalize_da3_image_batch(encoder, images: torch.Tensor) -> torch.Tensor:
    mean = encoder.encoder_mean.float()
    std = encoder.encoder_std.float()
    return (images.float() - mean) / std


def prepare_da3_finetune_batch(encoder, batch, device):
    all_view_images = batch["all_view_images"].to(device)
    batch_size, timesteps, n_views, _, height, width = all_view_images.shape
    all_views = all_view_images.reshape(batch_size, timesteps * n_views, 3, height, width)
    all_views_norm = normalize_da3_image_batch(encoder, all_views)
    target_view_images = batch.get("all_view_target_images")
    teacher_views_norm = None
    teacher_depth_valid_mask = None
    view_valid_mask = None
    if target_view_images is not None:
        target_view_images = target_view_images.to(device)
        teacher_views = target_view_images.reshape(batch_size, timesteps * n_views, 3, height, width)
        teacher_views_norm = normalize_da3_image_batch(encoder, teacher_views)
    if "all_view_target_mask" in batch:
        teacher_depth_valid_mask = batch["all_view_target_mask"].to(device=device, dtype=torch.bool)
    if "view_valid_mask" in batch:
        view_valid_mask = batch["view_valid_mask"].to(device=device, dtype=torch.bool)
    return (
        all_views_norm,
        batch["actions"].to(device),
        batch["proprioception"].to(device),
        timesteps,
        n_views,
        teacher_views_norm,
        teacher_depth_valid_mask,
        view_valid_mask,
    )


def zero_invalid_context_proprio(
    proprio: torch.Tensor,
    context_valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Keep boundary-padding proprio from becoming normalized outliers."""
    if context_valid_mask is None or proprio.ndim < 3:
        return proprio
    mask = context_valid_mask.to(device=proprio.device, dtype=torch.bool)
    while mask.ndim < proprio.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, proprio, torch.zeros_like(proprio))
