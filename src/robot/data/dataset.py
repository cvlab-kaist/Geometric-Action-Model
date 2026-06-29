"""Robot datasets for DA3 action training."""

from __future__ import annotations

from dataclasses import replace
import io
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    import h5py  # Optional: only needed for LIBERO HDF5 demos.
except ModuleNotFoundError:
    h5py = None  # type: ignore[assignment]
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ACTION_STATS_KEY = "__default__"
DEFAULT_ACTION_NORM_MASK = np.array([True, True, True, True, True, True, False], dtype=bool)

WRIST_CAMERA_TOKENS = (
    "wrist",
    "hand",
    "gripper",
    "eye_in_hand",
    "image_2",
    "cam_low",
)
_TORCHVISION_JPEG_AVAILABLE: Optional[bool] = None


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        return {str(k): v for k, v in items()}
    return {}


def _side_scale_from_area(area: Any, default: float) -> float:
    try:
        area_f = float(area)
    except (TypeError, ValueError):
        area_f = float(default)
    area_f = max(0.25, min(1.0, area_f))
    return math.sqrt(area_f)


def resolve_image_augmentation_config(dataset_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve legacy flat image keys plus the optional nested aug profile.

    Existing configs without ``dataset.image_augmentation`` keep their legacy
    behavior. The Cosmos profile uses area-style crop values, converted to the
    side-scale crop API already used by this loader.
    """
    aug_cfg = _as_plain_mapping(dataset_cfg.get("image_augmentation", {}))
    train_cfg = _as_plain_mapping(aug_cfg.get("train", {}))
    eval_cfg = _as_plain_mapping(aug_cfg.get("eval", {}))
    color_cfg = _as_plain_mapping(train_cfg.get("color_jitter", {}))
    train_jpeg_cfg = _as_plain_mapping(train_cfg.get("jpeg", {}))
    eval_jpeg_cfg = _as_plain_mapping(eval_cfg.get("jpeg", {}))
    profile = str(aug_cfg.get("profile", "") or "").strip().lower()

    resolved = {
        "image_aug_profile": profile,
        "train_crop_min_scale": float(dataset_cfg.get("train_crop_min_scale", 0.9)),
        "eval_crop_scale": float(dataset_cfg.get("eval_crop_scale", 0.9)),
        "color_jitter_brightness": float(dataset_cfg.get("color_jitter_brightness", 0.0)),
        "color_jitter_contrast": float(dataset_cfg.get("color_jitter_contrast", 0.0)),
        "color_jitter_saturation": float(dataset_cfg.get("color_jitter_saturation", 0.0)),
        "color_jitter_hue": float(dataset_cfg.get("color_jitter_hue", 0.0)),
        "openpi_libero_augment": bool(dataset_cfg.get("openpi_libero_augment", False)),
        "openpi_base_crop_scale": float(dataset_cfg.get("openpi_base_crop_scale", 0.95)),
        "openpi_base_rotate_degrees": float(dataset_cfg.get("openpi_base_rotate_degrees", 5.0)),
        "image_jpeg_train_enabled": bool(
            dataset_cfg.get("image_jpeg_train_enabled", dataset_cfg.get("image_jpeg_enabled", False))
        ),
        "image_jpeg_eval_enabled": bool(
            dataset_cfg.get("image_jpeg_eval_enabled", dataset_cfg.get("image_jpeg_enabled", False))
        ),
        "image_jpeg_train_quality": int(
            dataset_cfg.get("image_jpeg_train_quality", dataset_cfg.get("image_jpeg_quality", 95))
        ),
        "image_jpeg_eval_quality": int(
            dataset_cfg.get("image_jpeg_eval_quality", dataset_cfg.get("image_jpeg_quality", 95))
        ),
    }

    if profile in {"cosmos_policy_strong", "cosmos_policy_strong_openpi"}:
        train_side = _side_scale_from_area(
            train_cfg.get("random_resized_crop_area", aug_cfg.get("random_resized_crop_area", 0.9)),
            0.9,
        )
        eval_side = _side_scale_from_area(
            eval_cfg.get("center_crop_area", aug_cfg.get("center_crop_area", 0.9)),
            0.9,
        )
        resolved.update(
            {
                "train_crop_min_scale": float(train_cfg.get("train_crop_min_scale", train_side)),
                "eval_crop_scale": float(eval_cfg.get("eval_crop_scale", eval_side)),
                "openpi_libero_augment": bool(dataset_cfg.get("openpi_libero_augment", True)),
                "openpi_base_crop_scale": float(
                    train_cfg.get("base_crop_scale", dataset_cfg.get("openpi_base_crop_scale", train_side))
                ),
                "openpi_base_rotate_degrees": float(
                    train_cfg.get(
                        "base_only_rotation_degrees",
                        dataset_cfg.get("openpi_base_rotate_degrees", 5.0),
                    )
                ),
                "color_jitter_brightness": float(color_cfg.get("brightness", 0.3)),
                "color_jitter_contrast": float(color_cfg.get("contrast", 0.4)),
                "color_jitter_saturation": float(color_cfg.get("saturation", 0.5)),
                "color_jitter_hue": float(color_cfg.get("hue", 0.05)),
                "image_jpeg_train_enabled": bool(train_jpeg_cfg.get("enabled", True)),
                "image_jpeg_eval_enabled": bool(eval_jpeg_cfg.get("enabled", True)),
                "image_jpeg_train_quality": int(train_jpeg_cfg.get("quality", 95)),
                "image_jpeg_eval_quality": int(eval_jpeg_cfg.get("quality", 95)),
            }
        )

    resolved["image_jpeg_train_quality"] = max(1, min(100, int(resolved["image_jpeg_train_quality"])))
    resolved["image_jpeg_eval_quality"] = max(1, min(100, int(resolved["image_jpeg_eval_quality"])))
    return resolved


def _is_wrist_camera_key(camera_key: Optional[str]) -> bool:
    if camera_key is None:
        return False
    key = str(camera_key).lower()
    return any(token in key for token in WRIST_CAMERA_TOKENS)


def _full_crop_params(height: int, width: int) -> Tuple[float, int, int, int, int]:
    return 1.0, 0, 0, int(height), int(width)


def _sample_fixed_crop_params(
    height: int,
    width: int,
    crop_scale: float,
) -> Tuple[float, int, int, int, int]:
    crop_scale = max(0.5, min(1.0, float(crop_scale)))
    if crop_scale >= 0.999:
        return _full_crop_params(height, width)
    crop_h = max(1, int(round(height * crop_scale)))
    crop_w = max(1, int(round(width * crop_scale)))
    top = random.randint(0, max(0, height - crop_h))
    left = random.randint(0, max(0, width - crop_w))
    return crop_scale, top, left, crop_h, crop_w


def _sample_crop_params(
    height: int,
    width: int,
    image_size: Tuple[int, int],
    is_eval: bool,
    train_crop_min_scale: float,
    eval_crop_scale: float,
) -> Tuple[float, int, int]:
    """Sample crop parameters once, to be reused across all timesteps in a sample."""
    crop_scale = float(eval_crop_scale if is_eval else random.uniform(train_crop_min_scale, 1.0))
    crop_scale = max(0.5, min(1.0, crop_scale))
    if crop_scale < 0.999:
        crop_h = max(1, int(round(height * crop_scale)))
        crop_w = max(1, int(round(width * crop_scale)))
        if is_eval:
            top = max(0, (height - crop_h) // 2)
            left = max(0, (width - crop_w) // 2)
        else:
            top = random.randint(0, max(0, height - crop_h))
            left = random.randint(0, max(0, width - crop_w))
    else:
        crop_h, crop_w = height, width
        top, left = 0, 0
    return crop_scale, top, left, crop_h, crop_w


def _crop_and_resize_image(
    img: torch.Tensor,
    image_size: Tuple[int, int],
    is_eval: bool,
    train_crop_min_scale: float,
    eval_crop_scale: float,
    crop_params: Optional[Tuple] = None,
) -> torch.Tensor:
    height, width = img.shape[-2:]
    if crop_params is not None:
        crop_scale, top, left, crop_h, crop_w = crop_params
    else:
        crop_scale, top, left, crop_h, crop_w = _sample_crop_params(
            height, width, image_size, is_eval, train_crop_min_scale, eval_crop_scale,
        )

    if crop_scale < 0.999:
        img = img[:, top : top + crop_h, left : left + crop_w]

    if img.shape[-2:] != image_size:
        img = F.interpolate(
            img.unsqueeze(0),
            size=image_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    return img.clamp(0, 1)


def _maybe_rotate_image_180(frame: Any, enabled: bool) -> Any:
    """Rotate a single image frame by 180 degrees before crop/resize.

    This is for raw dataset orientation fixes. It must happen before crop
    sampling is applied so RGB/depth crops stay aligned in the corrected frame.
    """
    if not enabled:
        return frame
    if isinstance(frame, torch.Tensor):
        if frame.ndim == 2:
            return torch.flip(frame, dims=[-2, -1])
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3):
                return torch.flip(frame, dims=[-2, -1])
            return torch.flip(frame, dims=[0, 1])
        raise ValueError(f"Expected image tensor with ndim 2 or 3, got shape {tuple(frame.shape)}")

    arr = np.asarray(frame)
    if arr.ndim < 2:
        raise ValueError(f"Expected image array with ndim >= 2, got shape {arr.shape}")
    return np.rot90(arr, 2).copy()


def _maybe_hflip_image(frame: Any, enabled: bool) -> Any:
    """Flip a single image frame left-right before crop/resize when enabled."""
    if not enabled:
        return frame
    if isinstance(frame, torch.Tensor):
        if frame.ndim == 2:
            return torch.flip(frame, dims=[-1])
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3):
                return torch.flip(frame, dims=[-1])
            return torch.flip(frame, dims=[1])
        raise ValueError(f"Expected image tensor with ndim 2 or 3, got shape {tuple(frame.shape)}")

    arr = np.asarray(frame)
    if arr.ndim < 2:
        raise ValueError(f"Expected image array with ndim >= 2, got shape {arr.shape}")
    return np.flip(arr, axis=1).copy()


def _maybe_vflip_image(frame: Any, enabled: bool) -> Any:
    """Flip a single image frame top-bottom before crop/resize when enabled."""
    if not enabled:
        return frame
    if isinstance(frame, torch.Tensor):
        if frame.ndim == 2:
            return torch.flip(frame, dims=[-2])
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3):
                return torch.flip(frame, dims=[-2])
            return torch.flip(frame, dims=[0])
        raise ValueError(f"Expected image tensor with ndim 2 or 3, got shape {tuple(frame.shape)}")

    arr = np.asarray(frame)
    if arr.ndim < 2:
        raise ValueError(f"Expected image array with ndim >= 2, got shape {arr.shape}")
    return np.flip(arr, axis=0).copy()


def infer_libero_hdf5_hflip(
    raw_value: Any,
    dataset_name: Any,
    hdf5_root: Any,
) -> bool:
    """Infer the horizontal-flip default for LIBERO HDF5-family datasets.

    Raw LIBERO HDF5 keeps its historical rotate180 contract. Replayed
    ``libero_noop`` exports intentionally preserve the vertical flip at capture
    time, so an omitted horizontal-flip key defaults to ``true`` there to
    recover the same effective orientation as raw-HDF5 rotate180 runs.
    """
    if raw_value is not None:
        return bool(raw_value)
    dataset_name_text = str(dataset_name or "").lower()
    hdf5_root_text = str(hdf5_root or "").lower()
    return "libero_noop" in dataset_name_text or "libero_noop" in hdf5_root_text


def normalize_action_frame(value: Any) -> str:
    """Normalize model action-frame names used by dataset and rollout code."""
    text = str(value or "base").strip().lower().replace("-", "_")
    aliases = {
        "": "base",
        "world": "base",
        "base_world": "base",
        "world_base": "base",
        "osc_pose": "base",
        "libero_osc_pose": "base",
        "world_delta": "base_delta",
        "base_world_delta": "base_delta",
        "world_base_delta": "base_delta",
        "base_frame_delta": "base_delta",
        "world_frame_delta": "base_delta",
        "controller_delta": "base_delta",
        "eef": "eef_relative",
        "eef_rel": "eef_relative",
        "ee_relative": "eef_relative",
        "endeffector_relative": "eef_relative",
        "end_effector_relative": "eef_relative",
        "eef_relative_trajectory": "eef_relative",
        "relative_trajectory": "eef_relative",
        "chunk_start_relative": "eef_relative",
        "umi": "eef_relative",
        "umi_relative": "eef_relative",
        "pretraining": "eef_relative",
        "pretraining_eef": "eef_relative",
        "pretraining_eef_relative": "eef_relative",
        "eef_delta": "eef_delta",
        "eef_local_delta": "eef_delta",
        "local_delta": "eef_delta",
        "moving_eef_delta": "eef_delta",
        "per_step_eef_delta": "eef_delta",
        "old_eef_relative": "eef_delta",
    }
    text = aliases.get(text, text)
    if text not in {"base", "base_delta", "eef_delta", "eef_relative"}:
        raise ValueError(
            f"Unsupported action_frame={value!r}; expected 'base', 'base_delta', "
            "'eef_delta', or 'eef_relative'."
        )
    return text


def normalize_proprio_orientation_mode(value: Any) -> str:
    """Normalize the 3D orientation representation carried in canonical proprio."""
    text = str(value or "rpy").strip().lower().replace("-", "_")
    aliases = {
        "": "rpy",
        "auto": "rpy",
        "euler": "rpy",
        "xyz_euler": "rpy",
        "axisangle": "axis_angle",
        "axis_angle": "axis_angle",
        "rotvec": "axis_angle",
        "rotation_vector": "axis_angle",
    }
    text = aliases.get(text, text)
    if text not in {"rpy", "axis_angle"}:
        raise ValueError(f"Unsupported proprio_orientation={value!r}; expected 'rpy' or 'axis_angle'.")
    return text


def infer_libero_hdf5_proprio_orientation(
    raw_value: Any,
    dataset_name: Any,
    hdf5_root: Any,
) -> str:
    """Infer LIBERO HDF5 proprio orientation when the config omits it."""
    if raw_value is not None:
        return normalize_proprio_orientation_mode(raw_value)
    dataset_name_text = str(dataset_name or "").lower()
    hdf5_root_text = str(hdf5_root or "").lower()
    # Current regenerated no-op HDF5s store obs/ee_states as robosuite
    # axis-angle values. Raw LIBERO HDF5 keeps the historical rpy contract.
    if "libero_noop" in dataset_name_text or "libero_noop" in hdf5_root_text:
        return "axis_angle"
    return "rpy"


def _libero_compute_keep_indices(actions: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Return original frame indices kept after OpenVLA-style noop filtering.

    A frame `t` is dropped iff its action is a no-op:
      (1) ``np.linalg.norm(actions[t, :-1]) < threshold`` (pose delta near 0)
      (2) AND (for ``t > 0``) ``actions[t, -1] == actions[t-1, -1]``
          (gripper command unchanged from previous frame).

    The first frame's gripper criterion is bypassed (no prev_action). This is
    a direct port of OpenVLA's `experiments/robot/libero/regenerate_libero_dataset.py:is_noop`.

    Args:
        actions: (T, A) numpy array. Last column is gripper, rest are pose deltas.
        threshold: L2-norm threshold on pose-delta magnitude.

    Returns:
        kept_indices: 1-D int64 numpy array, sorted ascending, of length K <= T.
    """
    if actions.ndim != 2 or actions.shape[0] == 0:
        return np.arange(0, actions.shape[0], dtype=np.int64)
    pose_norm = np.linalg.norm(actions[:, :-1], axis=-1)         # (T,)
    is_low_pose = pose_norm < float(threshold)                    # (T,)
    grip_eq_prev = np.zeros(actions.shape[0], dtype=bool)         # (T,) : first frame is False
    grip_eq_prev[1:] = actions[1:, -1] == actions[:-1, -1]
    is_noop = np.empty(actions.shape[0], dtype=bool)
    is_noop[0] = bool(is_low_pose[0])                             # first frame: criterion-1 only
    is_noop[1:] = is_low_pose[1:] & grip_eq_prev[1:]
    keep = np.where(~is_noop)[0].astype(np.int64)
    return keep


def _libero_depth_sidecar_path(depth_root: Path, episode_index: int, hdf5_path: str, demo_key: str) -> Path:
    stem = f"episode_{int(episode_index):06d}__{Path(hdf5_path).stem}__{demo_key}"
    return depth_root / f"{stem}.npz"


def _libero_depth_memmap_paths(
    depth_root: Path, episode_index: int, hdf5_path: str, demo_key: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (depth_memmap_npy, geometry_npz) paths in the sibling
    `<gt_depth_root>_memmap/` directory, with (None, None) for an absent
    sibling directory.
    """
    sibling = depth_root.with_name(depth_root.name + "_memmap")
    if not sibling.exists():
        return None, None
    stem = f"episode_{int(episode_index):06d}__{Path(hdf5_path).stem}__{demo_key}"
    return sibling / f"{stem}.depth.npy", sibling / f"{stem}.geometry.npz"


def _libero_depth_camera_name(camera_key: str) -> str:
    key = str(camera_key)
    if "." in key:
        key = key.split(".")[-1]
    lower = key.lower()
    if "eye_in_hand" in lower or "wrist" in lower or "hand" in lower or "image2" in lower:
        return "robot0_eye_in_hand"
    for suffix in ("_rgb", "_image"):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
    return lower


def _libero_hdf5_obs_camera_prefix(camera_key: str) -> str:
    """Map a LIBERO RGB camera key to the regenerated HDF5 obs prefix.

    The replayed no-op HDF5s store embedded GT depth / geometry under
    `agentview_*` and `eye_in_hand_*` keys, while the sidecar NPZ schema uses
    `agentview` / `robot0_eye_in_hand`. Keep this helper separate from
    `_libero_depth_camera_name` so the two storage formats can evolve
    independently without ambiguous string fallbacks.
    """
    key = str(camera_key)
    if "." in key:
        key = key.split(".")[-1]
    lower = key.lower()
    if "eye_in_hand" in lower or "wrist" in lower or "hand" in lower or "image2" in lower:
        return "eye_in_hand"
    for suffix in ("_rgb", "_image"):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
    return lower


def _crop_resize_depth_and_mask(
    depth: np.ndarray,
    image_size: Tuple[int, int],
    crop_params: Tuple[float, int, int, int, int],
    min_depth_m: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RGB crop/resize geometry to dense metric depth and validity mask."""
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got {arr.shape}")
    crop_scale, top, left, crop_h, crop_w = crop_params
    valid = np.isfinite(arr) & (arr > float(min_depth_m))
    arr = np.where(valid, arr, 0.0).astype(np.float32)
    if crop_scale < 0.999:
        arr = arr[top : top + crop_h, left : left + crop_w]
        valid = valid[top : top + crop_h, left : left + crop_w]

    depth_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    valid_t = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if tuple(arr.shape) != tuple(image_size):
        depth_num = F.interpolate(
            depth_t * valid_t,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        valid_weight = F.interpolate(
            valid_t,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        depth_t = depth_num / valid_weight.clamp_min(1e-6)
        valid_t = valid_weight
    mask = valid_t.squeeze(0).squeeze(0) > 0.5
    depth_out = depth_t.squeeze(0).squeeze(0)
    depth_out = torch.where(mask, depth_out, torch.zeros_like(depth_out))
    return depth_out.float(), mask


def _rotate_depth_and_mask_train(
    depth: torch.Tensor,
    mask: torch.Tensor,
    angle_degrees: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate dense metric depth with validity-weighted bilinear resampling."""
    angle_degrees = float(angle_degrees)
    if abs(angle_degrees) <= 1.0e-6:
        return depth, mask
    from torchvision.transforms import functional as TF
    try:
        from torchvision.transforms import InterpolationMode
        interpolation = InterpolationMode.BILINEAR
    except Exception:
        interpolation = 2

    mask_f = mask.to(dtype=depth.dtype)
    depth_chw = (depth * mask_f).unsqueeze(0)
    mask_chw = mask_f.unsqueeze(0)
    try:
        depth_num = TF.rotate(depth_chw, angle_degrees, interpolation=interpolation, fill=[0.0])
        mask_weight = TF.rotate(mask_chw, angle_degrees, interpolation=interpolation, fill=[0.0])
    except TypeError:
        depth_num = TF.rotate(depth_chw, angle_degrees, interpolation=interpolation, fill=0.0)
        mask_weight = TF.rotate(mask_chw, angle_degrees, interpolation=interpolation, fill=0.0)
    mask_rot = mask_weight.squeeze(0) > 0.5
    depth_rot = (depth_num / mask_weight.clamp_min(1.0e-6)).squeeze(0)
    depth_rot = torch.where(mask_rot, depth_rot, torch.zeros_like(depth_rot))
    return depth_rot.float(), mask_rot


def _image_rotation_affine(
    image_size: Tuple[int, int],
    angle_degrees: float,
) -> np.ndarray:
    """Pixel-coordinate affine for the same post-resize in-plane rotation as RGB."""
    angle_degrees = float(angle_degrees)
    out_h, out_w = int(image_size[0]), int(image_size[1])
    if abs(angle_degrees) <= 1.0e-6:
        return np.eye(3, dtype=np.float32)
    theta = math.radians(angle_degrees)
    alpha = math.cos(theta)
    beta = math.sin(theta)
    cx = float(out_w - 1) * 0.5
    cy = float(out_h - 1) * 0.5
    affine = np.asarray(
        [
            [alpha, beta, (1.0 - alpha) * cx - beta * cy],
            [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return affine


def _scale_crop_params_to_source(
    crop_params: Tuple[float, int, int, int, int],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> Tuple[float, int, int, int, int]:
    """Map a crop sampled in RGB pixel coordinates to another aligned source size."""
    crop_scale, top, left, crop_h, crop_w = crop_params
    from_h, from_w = int(from_size[0]), int(from_size[1])
    to_h, to_w = int(to_size[0]), int(to_size[1])
    if from_h <= 0 or from_w <= 0 or to_h <= 0 or to_w <= 0:
        raise ValueError(f"Invalid crop source sizes: from={from_size} to={to_size}")
    if (from_h, from_w) == (to_h, to_w):
        return crop_params

    sy = float(to_h) / float(from_h)
    sx = float(to_w) / float(from_w)
    top_s = int(round(float(top) * sy))
    left_s = int(round(float(left) * sx))
    crop_h_s = max(1, int(round(float(crop_h) * sy)))
    crop_w_s = max(1, int(round(float(crop_w) * sx)))
    top_s = min(max(0, top_s), max(0, to_h - 1))
    left_s = min(max(0, left_s), max(0, to_w - 1))
    crop_h_s = min(crop_h_s, to_h - top_s)
    crop_w_s = min(crop_w_s, to_w - left_s)
    return crop_scale, top_s, left_s, crop_h_s, crop_w_s


def _transform_intrinsics_for_crop_resize(
    intrinsics: np.ndarray,
    crop_params: Tuple[float, int, int, int, int],
    image_size: Tuple[int, int],
    source_size: Tuple[int, int],
    rotate180: bool,
    hflip: bool,
    vflip: bool,
    post_resize_rotation_degrees: float = 0.0,
) -> np.ndarray:
    """Transform a camera matrix through the same geometry path as RGB/depth."""
    _, top, left, crop_h, crop_w = crop_params
    out_h, out_w = int(image_size[0]), int(image_size[1])
    src_h, src_w = int(source_size[0]), int(source_size[1])
    k = np.asarray(intrinsics, dtype=np.float32).copy()
    if rotate180:
        k[0, 0] *= -1.0
        k[1, 1] *= -1.0
        k[0, 2] = float(src_w - 1) - k[0, 2]
        k[1, 2] = float(src_h - 1) - k[1, 2]
    if hflip:
        k[0, 0] *= -1.0
        k[0, 2] = float(src_w - 1) - k[0, 2]
    if vflip:
        k[1, 1] *= -1.0
        k[1, 2] = float(src_h - 1) - k[1, 2]
    k[0, 2] -= float(left)
    k[1, 2] -= float(top)
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    if abs(float(post_resize_rotation_degrees)) > 1.0e-6:
        k = _image_rotation_affine(image_size, post_resize_rotation_degrees) @ k
    return k.astype(np.float32)


def _scene_scale_from_pointmaps(
    depths: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: np.ndarray,
    extrinsics_c2w: np.ndarray,
    min_scale: float = 1e-3,
) -> float:
    """DA3-style common scale: mean L2 norm of valid point maps in ref-camera coordinates.

    Robosuite's ``get_camera_extrinsic_matrix`` rebases MuJoCo's native
    ``(x right, y up, z toward viewer)`` to the OpenCV convention
    ``(x right, y down, z forward)`` via ``diag(1, -1, -1, 1)``. K from
    ``get_camera_intrinsic_matrix`` projects camera-frame ``(x, y, z)`` as
    ``u = fx·x/z + cx``, ``v = fy·y/z + cy`` with image y also pointing down,
    so the y ray component is left unsigned (matches
    ``_unproject_depth_to_pointmap`` / ``_compute_gt_ray_map``). Pixel centres
    use the ``+0.5`` offset for the same reason.
    """
    depth_np = depths.detach().cpu().numpy().astype(np.float32)
    mask_np = masks.detach().cpu().numpy().astype(bool)
    k_np = np.asarray(intrinsics, dtype=np.float32)
    c2w_np = np.asarray(extrinsics_c2w, dtype=np.float32)
    if depth_np.ndim != 4:
        raise ValueError(f"Expected depths as (T,V,H,W), got {depth_np.shape}")
    if k_np.shape[:2] != depth_np.shape[:2] or c2w_np.shape[:2] != depth_np.shape[:2]:
        raise ValueError(
            "Camera geometry shape must match depth time/view axes: "
            f"depth={depth_np.shape[:2]} K={k_np.shape[:2]} c2w={c2w_np.shape[:2]}"
        )

    ref_w2c = np.linalg.inv(c2w_np[0, 0].astype(np.float64))
    values: list[np.ndarray] = []
    t_count, v_count, height, width = depth_np.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    ones = np.ones_like(xx, dtype=np.float32)
    for t in range(t_count):
        for v in range(v_count):
            valid = mask_np[t, v] & np.isfinite(depth_np[t, v]) & (depth_np[t, v] > 0)
            if not valid.any():
                continue
            k = k_np[t, v]
            z = depth_np[t, v]
            x = (xx - k[0, 2]) / k[0, 0] * z
            y = (yy - k[1, 2]) / k[1, 1] * z
            pts_cam = np.stack([x, y, z, ones], axis=-1).reshape(-1, 4)
            pts_world = pts_cam @ c2w_np[t, v].astype(np.float64).T
            pts_ref = pts_world @ ref_w2c.T
            norms = np.linalg.norm(pts_ref[:, :3], axis=-1).reshape(height, width)
            norms = norms[valid & np.isfinite(norms)]
            if norms.size:
                values.append(norms.astype(np.float64))
    if not values:
        return float(min_scale)
    scale = float(np.concatenate(values, axis=0).mean())
    if not math.isfinite(scale):
        return float(min_scale)
    return max(scale, float(min_scale))


def _color_jitter_train(
    img: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> torch.Tensor:
    """Apply color jitter on a CHW float [0,1] tensor. Each call samples fresh.

    Matches VGA-style augmentation (Tab. 6 in arxiv 2604.12908v1):
      brightness=0.2, contrast=[0.8,1.2], saturation=[0.8,1.2], hue=0.05.
    Symmetric scalars are interpreted as ranges around 1.0 (multiplicative)
    or 0.0 (additive for hue), matching torchvision.transforms.ColorJitter.
    Skipped entirely when all factors are 0.0.
    """
    if max(brightness, contrast, saturation, hue) <= 0.0:
        return img
    from torchvision.transforms import functional as TF
    if brightness > 0.0:
        f = float(torch.empty(1).uniform_(max(0.0, 1.0 - brightness), 1.0 + brightness).item())
        img = TF.adjust_brightness(img, f)
    if contrast > 0.0:
        f = float(torch.empty(1).uniform_(max(0.0, 1.0 - contrast), 1.0 + contrast).item())
        img = TF.adjust_contrast(img, f)
    if saturation > 0.0:
        f = float(torch.empty(1).uniform_(max(0.0, 1.0 - saturation), 1.0 + saturation).item())
        img = TF.adjust_saturation(img, f)
    if hue > 0.0:
        f = float(torch.empty(1).uniform_(-hue, hue).item())
        img = TF.adjust_hue(img.clamp(0.0, 1.0), f)
    return img.clamp(0.0, 1.0)


def _jpeg_compress_chw(img: torch.Tensor, quality: int) -> torch.Tensor:
    global _TORCHVISION_JPEG_AVAILABLE
    quality = max(1, min(100, int(quality)))
    if quality >= 100:
        return img.clamp(0.0, 1.0)
    img_u8 = (
        img.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    if _TORCHVISION_JPEG_AVAILABLE is not False:
        try:
            from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

            encoded = encode_jpeg(img_u8.cpu(), quality=quality)
            out_t = decode_jpeg(encoded, mode=ImageReadMode.RGB)
            _TORCHVISION_JPEG_AVAILABLE = True
            return out_t.float().div(255.0)
        except Exception:
            _TORCHVISION_JPEG_AVAILABLE = False
    arr = (
        img_u8
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"), copy=True)
    return torch.from_numpy(out).permute(2, 0, 1).float().div(255.0)


def _sample_rotation_angle(max_degrees: float) -> float:
    max_degrees = float(max_degrees)
    if max_degrees <= 0.0:
        return 0.0
    return float(torch.empty(1).uniform_(-max_degrees, max_degrees).item())


def _rotate_chw_train(img: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    angle_degrees = float(angle_degrees)
    if abs(angle_degrees) <= 1.0e-6:
        return img
    from torchvision.transforms import functional as TF
    try:
        from torchvision.transforms import InterpolationMode
        interpolation = InterpolationMode.BILINEAR
    except Exception:
        interpolation = 2
    try:
        fill = [0.0] * int(img.shape[0])
        return TF.rotate(img, angle_degrees, interpolation=interpolation, fill=fill).clamp(0.0, 1.0)
    except TypeError:
        return TF.rotate(img, angle_degrees, interpolation=interpolation, fill=0.0).clamp(0.0, 1.0)


def _rotation_valid_mask_train(
    image_size: Tuple[int, int],
    angle_degrees: float,
) -> torch.Tensor:
    """Mask pixels that are still backed by real image content after rotation."""
    angle_degrees = float(angle_degrees)
    height, width = int(image_size[0]), int(image_size[1])
    if abs(angle_degrees) <= 1.0e-6:
        return torch.ones(height, width, dtype=torch.bool)
    from torchvision.transforms import functional as TF
    try:
        from torchvision.transforms import InterpolationMode
        interpolation = InterpolationMode.BILINEAR
    except Exception:
        interpolation = 2
    mask = torch.ones(1, height, width, dtype=torch.float32)
    try:
        rotated = TF.rotate(mask, angle_degrees, interpolation=interpolation, fill=[0.0])
    except TypeError:
        rotated = TF.rotate(mask, angle_degrees, interpolation=interpolation, fill=0.0)
    return rotated.squeeze(0) > 0.999


def _normalize_image_tensor(
    image: Any,
    image_size: Tuple[int, int],
    *,
    is_eval: bool,
    train_crop_min_scale: float,
    eval_crop_scale: float,
    crop_params: Optional[Tuple] = None,
    color_jitter_brightness: float = 0.0,
    color_jitter_contrast: float = 0.0,
    color_jitter_saturation: float = 0.0,
    color_jitter_hue: float = 0.0,
    camera_key: Optional[str] = None,
    openpi_libero_augment: bool = False,
    openpi_base_rotate_degrees: float = 5.0,
    openpi_base_rotation_angle: Optional[float] = None,
    jpeg_enabled: bool = False,
    jpeg_quality: int = 95,
) -> torch.Tensor:
    """Convert image to float CHW in [0, 1] and apply crop+resize.

    Color jitter (brightness/contrast/saturation/hue) is applied ONLY when
    is_eval=False AND any factor > 0. Eval pipeline always skips jitter.
    """
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if img.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims, got shape={tuple(img.shape)}")
        if img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)
        img = img.float()
    else:
        img = torch.as_tensor(np.asarray(image))
        if img.ndim != 3:
            raise ValueError(f"Expected image array with 3 dims, got shape={tuple(img.shape)}")
        img = img.permute(2, 0, 1).float()

    if img.max().item() > 1.0:
        img = img / 255.0

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels after conversion, got shape={tuple(img.shape)}")

    if bool(jpeg_enabled):
        img = _jpeg_compress_chw(img, int(jpeg_quality))

    if bool(openpi_libero_augment) and not is_eval:
        img = _crop_and_resize_image(
            img,
            image_size=image_size,
            is_eval=is_eval,
            train_crop_min_scale=train_crop_min_scale,
            eval_crop_scale=eval_crop_scale,
            crop_params=crop_params,
        )
        if not _is_wrist_camera_key(camera_key):
            angle = (
                _sample_rotation_angle(openpi_base_rotate_degrees)
                if openpi_base_rotation_angle is None
                else float(openpi_base_rotation_angle)
            )
            img = _rotate_chw_train(img, angle)
        return _color_jitter_train(
            img,
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )

    # Color jitter strictly TRAIN-ONLY. Eval bypass even when factors set in cfg.
    if not is_eval:
        img = _color_jitter_train(
            img,
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )
    return _crop_and_resize_image(
        img,
        image_size=image_size,
        is_eval=is_eval,
        train_crop_min_scale=train_crop_min_scale,
        eval_crop_scale=eval_crop_scale,
        crop_params=crop_params,
    )


def _rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    """(N, 3) rotvec → (N, 4) quaternion [w, x, y, z]."""
    angle = rv.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (N, 1)
    axis = rv / angle
    half = angle * 0.5
    w = half.cos()
    xyz = axis * half.sin()
    return torch.cat([w, xyz], dim=-1)  # (N, 4)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (N, 4) quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternion [w, x, y, z] → (N, 3) rotvec."""
    # Ensure w >= 0 for consistent angle
    q = torch.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    half_angle = torch.atan2(sin_half, q[..., :1])
    angle = 2.0 * half_angle
    axis = xyz / sin_half
    return axis * angle




def _quat_xyzw_to_rpy(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) quaternion in (qx, qy, qz, qw) order → (N, 3) extrinsic XYZ Euler (rpy).

    Matches scipy `Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)`.
    Handles gimbal lock at pitch = ±π/2 by clamping asin argument to [-1, +1].

    Args:
        q_xyzw: (..., 4) quaternion, (qx, qy, qz, qw) order (scipy/Tavish9 convention).
    Returns:
        (..., 3) extrinsic XYZ Euler angles (roll, pitch, yaw) in radians.
    """
    # Normalize to guard against numerical drift
    q = q_xyzw / q_xyzw.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    qx, qy, qz, qw = q.unbind(-1)
    # scipy extrinsic XYZ formula:
    #   roll  = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx² + qy²))
    #   pitch = asin(clamp(2*(qw*qy - qz*qx), -1, 1))
    #   yaw   = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch_arg = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)
    pitch = torch.asin(pitch_arg)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _rpy_xyz_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert extrinsic XYZ Euler angles to active rotation matrices."""
    rpy = torch.as_tensor(rpy, dtype=torch.float32)
    orig_shape = rpy.shape[:-1]
    roll, pitch, yaw = rpy.reshape(-1, 3).unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    row0 = torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1)
    row1 = torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1)
    row2 = torch.stack([-sp, cp * sr, cp * cr], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2).reshape(*orig_shape, 3, 3)


def _rotmat_to_rpy_xyz(rotmat: torch.Tensor) -> torch.Tensor:
    """Convert active rotation matrices to extrinsic XYZ Euler angles."""
    rotmat = torch.as_tensor(rotmat, dtype=torch.float32)
    if rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {tuple(rotmat.shape)}")
    orig_shape = rotmat.shape[:-2]
    m = rotmat.reshape(-1, 3, 3)
    pitch_arg = (-m[:, 2, 0]).clamp(-1.0, 1.0)
    pitch = torch.asin(pitch_arg)
    roll = torch.atan2(m[:, 2, 1], m[:, 2, 2])
    yaw = torch.atan2(m[:, 1, 0], m[:, 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1).reshape(*orig_shape, 3)


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert SO(3) rotation vectors to active rotation matrices."""
    rotvec = torch.as_tensor(rotvec, dtype=torch.float32)
    orig_shape = rotvec.shape[:-1]
    q = _rotvec_to_quat(rotvec.reshape(-1, 3))
    w, x, y, z = q.unbind(-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2).reshape(*orig_shape, 3, 3)


def _rotmat_to_rotvec(rotmat: torch.Tensor) -> torch.Tensor:
    """SO(3) log map using a quaternion branch for near-pi stability."""
    rotmat = torch.as_tensor(rotmat, dtype=torch.float32)
    if rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {tuple(rotmat.shape)}")

    orig_shape = rotmat.shape[:-2]
    m = rotmat.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))
    candidates = torch.stack([
        torch.stack([q_abs[:, 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[:, 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[:, 2] ** 2, m21 + m12], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[:, 3] ** 2], dim=-1),
    ], dim=1)
    candidates = candidates / (2.0 * q_abs.clamp(min=1e-6).unsqueeze(-1))

    best = q_abs.argmax(dim=-1)
    q = candidates[torch.arange(m.shape[0], device=m.device), best]
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return _quat_to_rotvec(q).reshape(*orig_shape, 3)


def _proprio_orientation_to_matrix(
    orientation: torch.Tensor,
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    mode = normalize_proprio_orientation_mode(proprio_orientation)
    if mode == "axis_angle":
        return _rotvec_to_matrix(orientation)
    return _rpy_xyz_to_matrix(orientation)


def _world_delta_to_eef_relative_action(
    action: torch.Tensor,
    current_proprio: Optional[torch.Tensor],
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Legacy moving-frame EEF-local delta.

    Despite the historical name, this is *not* a chunk-start relative
    trajectory. It rotates each base/world action delta into that action
    timestep's own EEF frame. New code should call
    ``_world_delta_to_eef_delta_action`` for this semantics, and reserve
    ``action_frame='eef_relative'`` for chunk-start relative trajectories.
    """
    action = _ensure_2d_tensor(action).to(dtype=torch.float32)
    if current_proprio is None or action.shape[-1] < 6:
        return action
    current = _ensure_2d_tensor(current_proprio).to(dtype=action.dtype)
    if current.shape[-1] < 6:
        return action
    if current.shape[0] < action.shape[0]:
        if current.shape[0] == 1:
            current = current.expand(action.shape[0], -1)
        else:
            return action
    current = current[: action.shape[0]].to(device=action.device)

    r_current = _proprio_orientation_to_matrix(current[..., 3:6], proprio_orientation).to(
        dtype=action.dtype,
        device=action.device,
    )
    r_current_t = r_current.transpose(-1, -2)
    pos_local = torch.matmul(r_current_t, action[..., :3].unsqueeze(-1)).squeeze(-1)

    r_delta_world = _rotvec_to_matrix(action[..., 3:6]).to(dtype=action.dtype, device=action.device)
    r_delta_local = r_current_t @ r_delta_world @ r_current
    rot_local = _rotmat_to_rotvec(r_delta_local).to(dtype=action.dtype, device=action.device)
    return torch.cat([pos_local, rot_local, action[..., 6:7]], dim=-1)


def _world_delta_to_eef_delta_action(
    action: torch.Tensor,
    current_proprio: Optional[torch.Tensor],
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Express each base/world delta in its own current EEF frame."""
    return _world_delta_to_eef_relative_action(
        action,
        current_proprio,
        proprio_orientation=proprio_orientation,
    )


def _eef_relative_to_world_delta_action(
    action: torch.Tensor,
    current_proprio: Optional[torch.Tensor],
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Express an EEF-relative 7D delta back in the base/world frame."""
    action = _ensure_2d_tensor(action).to(dtype=torch.float32)
    if current_proprio is None or action.shape[-1] < 6:
        return action
    current = _ensure_2d_tensor(current_proprio).to(dtype=action.dtype)
    if current.shape[-1] < 6:
        return action
    if current.shape[0] < action.shape[0]:
        if current.shape[0] == 1:
            current = current.expand(action.shape[0], -1)
        else:
            return action
    current = current[: action.shape[0]].to(device=action.device)

    r_current = _proprio_orientation_to_matrix(current[..., 3:6], proprio_orientation).to(
        dtype=action.dtype,
        device=action.device,
    )
    pos_world = torch.matmul(r_current, action[..., :3].unsqueeze(-1)).squeeze(-1)

    r_delta_local = _rotvec_to_matrix(action[..., 3:6]).to(dtype=action.dtype, device=action.device)
    r_delta_world = r_current @ r_delta_local @ r_current.transpose(-1, -2)
    rot_world = _rotmat_to_rotvec(r_delta_world).to(dtype=action.dtype, device=action.device)
    return torch.cat([pos_world, rot_world, action[..., 6:7]], dim=-1)


def _eef_relative_targets_from_proprio(
    target_proprio: torch.Tensor,
    anchor_proprio: torch.Tensor,
    gripper: torch.Tensor,
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Express future EEF target poses relative to one chunk-start anchor."""
    target = _ensure_2d_tensor(target_proprio).to(dtype=torch.float32)
    anchor = _ensure_2d_tensor(anchor_proprio).to(dtype=target.dtype, device=target.device)
    if target.shape[-1] < 6 or anchor.shape[-1] < 6:
        raise ValueError(
            "EEF-relative trajectory requires canonical proprio with at least "
            f"6 pose dims, got target={tuple(target.shape)} anchor={tuple(anchor.shape)}."
        )
    anchor = anchor[:1].expand(target.shape[0], -1)
    r_anchor = _proprio_orientation_to_matrix(anchor[..., 3:6], proprio_orientation).to(
        dtype=target.dtype,
        device=target.device,
    )
    r_anchor_t = r_anchor.transpose(-1, -2)
    pos_rel = torch.matmul(
        r_anchor_t,
        (target[..., :3] - anchor[..., :3]).unsqueeze(-1),
    ).squeeze(-1)
    r_target = _proprio_orientation_to_matrix(target[..., 3:6], proprio_orientation).to(
        dtype=target.dtype,
        device=target.device,
    )
    rot_rel = _rotmat_to_rotvec(r_anchor_t @ r_target).to(dtype=target.dtype, device=target.device)
    grip = _ensure_2d_tensor(gripper).to(dtype=target.dtype, device=target.device)
    if grip.shape[0] < target.shape[0]:
        grip = grip[:1].expand(target.shape[0], -1)
    return torch.cat([pos_rel, rot_rel, grip[: target.shape[0], -1:]], dim=-1)


def _integrate_world_delta_targets(
    action: torch.Tensor,
    anchor_proprio: torch.Tensor,
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Build commanded world target poses by integrating base/world deltas."""
    action = _ensure_2d_tensor(action).to(dtype=torch.float32)
    anchor = _ensure_2d_tensor(anchor_proprio).to(dtype=action.dtype, device=action.device)
    if action.shape[-1] < 7 or anchor.shape[-1] < 6:
        return action
    anchor = anchor[:1]
    pos = anchor[..., :3].clone()
    rot = _proprio_orientation_to_matrix(anchor[..., 3:6], proprio_orientation).to(
        dtype=action.dtype,
        device=action.device,
    )
    targets: list[torch.Tensor] = []
    for step in range(action.shape[0]):
        pos = pos + action[step : step + 1, :3]
        delta_rot = _rotvec_to_matrix(action[step : step + 1, 3:6]).to(
            dtype=action.dtype,
            device=action.device,
        )
        rot = delta_rot @ rot
        target_rpy = _rotmat_to_rpy_xyz(rot).to(dtype=action.dtype, device=action.device)
        targets.append(torch.cat([pos, target_rpy, action[step : step + 1, 6:7]], dim=-1))
    return torch.cat(targets, dim=0)


def _world_delta_sequence_to_eef_relative_trajectory(
    action: torch.Tensor,
    anchor_proprio: Optional[torch.Tensor],
    target_proprio: Optional[torch.Tensor] = None,
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Convert a delta sequence to chunk-start EEF-relative target trajectory.

    The output slot ``j`` is a future target pose expressed in the first
    anchor frame: ``T_anchor^-1 T_target_j``. When target proprio is available,
    it is used directly. Otherwise we integrate the commanded base/world deltas
    from the anchor pose and express those commanded targets relative to the
    same anchor.
    """
    action = _ensure_2d_tensor(action).to(dtype=torch.float32)
    if anchor_proprio is None or action.shape[-1] < 7:
        return action
    anchor = _ensure_2d_tensor(anchor_proprio).to(dtype=action.dtype, device=action.device)
    if anchor.shape[-1] < 6:
        return action
    if target_proprio is None:
        target = _integrate_world_delta_targets(action, anchor, proprio_orientation=proprio_orientation)
    else:
        target = _ensure_2d_tensor(target_proprio).to(dtype=action.dtype, device=action.device)
        if target.shape[0] < action.shape[0]:
            if target.shape[0] == 1:
                target = target.expand(action.shape[0], -1)
            else:
                target = _integrate_world_delta_targets(action, anchor, proprio_orientation=proprio_orientation)
    return _eef_relative_targets_from_proprio(
        target[: action.shape[0]],
        anchor[:1],
        action[..., 6:7],
        proprio_orientation=proprio_orientation,
    )


def _eef_relative_trajectory_to_world_delta_action(
    action: torch.Tensor,
    anchor_proprio: Optional[torch.Tensor],
    current_proprio: Optional[torch.Tensor],
    proprio_orientation: Any = "rpy",
) -> torch.Tensor:
    """Convert a chunk-start relative target pose to an executable world delta."""
    action = _ensure_2d_tensor(action).to(dtype=torch.float32)
    if anchor_proprio is None or current_proprio is None or action.shape[-1] < 7:
        return action
    anchor = _ensure_2d_tensor(anchor_proprio).to(dtype=action.dtype, device=action.device)
    current = _ensure_2d_tensor(current_proprio).to(dtype=action.dtype, device=action.device)
    if anchor.shape[-1] < 6 or current.shape[-1] < 6:
        return action
    if current.shape[0] < action.shape[0]:
        current = current[:1].expand(action.shape[0], -1)
    if anchor.shape[0] < action.shape[0]:
        anchor = anchor[:1].expand(action.shape[0], -1)
    anchor = anchor[: action.shape[0]]
    current = current[: action.shape[0]]

    r_anchor = _proprio_orientation_to_matrix(anchor[..., 3:6], proprio_orientation).to(
        dtype=action.dtype,
        device=action.device,
    )
    r_current = _proprio_orientation_to_matrix(current[..., 3:6], proprio_orientation).to(
        dtype=action.dtype,
        device=action.device,
    )
    target_pos_world = anchor[..., :3] + torch.matmul(
        r_anchor,
        action[..., :3].unsqueeze(-1),
    ).squeeze(-1)
    r_rel = _rotvec_to_matrix(action[..., 3:6]).to(dtype=action.dtype, device=action.device)
    r_target = r_anchor @ r_rel
    delta_pos_world = target_pos_world - current[..., :3]
    delta_rot_world = _rotmat_to_rotvec(r_target @ r_current.transpose(-1, -2)).to(
        dtype=action.dtype,
        device=action.device,
    )
    return torch.cat([delta_pos_world, delta_rot_world, action[..., 6:7]], dim=-1)


def _select_action_anchor_proprio(
    current_proprio: Optional[torch.Tensor],
    n_actions: int,
    stride: int,
) -> Optional[torch.Tensor]:
    """Pick the EEF pose at the start of each aggregated action label."""
    if current_proprio is None or n_actions <= 0:
        return None
    current = _ensure_2d_tensor(current_proprio).to(dtype=torch.float32)
    if current.shape[0] == 0:
        return None
    offsets = torch.arange(int(n_actions), dtype=torch.long, device=current.device) * max(1, int(stride))
    offsets = offsets.clamp(max=max(0, current.shape[0] - 1))
    return current.index_select(0, offsets)


def _compose_rotvec_batch(rotvecs: torch.Tensor) -> torch.Tensor:
    """Compose left-multiplied base-frame rotation vectors along dim=1.

    Args:
        rotvecs: (N, S, 3) : N groups of S sequential rotation vectors.
    Returns:
        (N, 3) : composed rotation vector per group, equivalent to applying
        step 0, then step 1, etc. with `R_next = Exp(delta) @ R_now`.
    """
    N, S, _ = rotvecs.shape
    # Identity quaternion
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotvecs.device, dtype=rotvecs.dtype).expand(N, 4).contiguous()
    for s in range(S):
        q_step = _rotvec_to_quat(rotvecs[:, s, :])  # (N, 4)
        q = _quat_mul(q_step, q)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return _quat_to_rotvec(q)


def _ensure_2d_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=dtype)


def _pad_or_truncate_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[-1] > target_dim:
        return tensor[..., :target_dim]
    if tensor.shape[-1] < target_dim:
        pad = torch.zeros(*tensor.shape[:-1], target_dim - tensor.shape[-1], dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=-1)
    return tensor


def _sample_episode_start(min_idx: int, max_idx: int, idx: int, is_eval: bool) -> int:
    if max_idx <= min_idx:
        return min_idx
    if is_eval:
        span = max_idx - min_idx + 1
        return min_idx + (idx % span)
    return random.randint(min_idx, max_idx)


class LiberoHDF5SequenceDataset(Dataset):
    """Plain LIBERO HDF5 sequence loader with the OpenX sample contract.

    It reads the original LIBERO HDF5 demos directly so frame indices can be
    joined with simulator-state GT-depth exports without a LeRobot mapping step.

    Pretraining branch note: this loader now emits EEF-relative action labels,
    matching MimicGen and convertible OxE rows. A separate server is currently
    training LIBERO with the older unmodified world/base OSC_POSE labels; mix
    those action stats or checkpoints with this branch only after an explicit
    action-frame conversion.
    """

    def __init__(
        self,
        hdf5_root: str,
        hdf5_paths: Optional[Sequence[str]] = None,
        pattern: str = "*.hdf5",
        suites: Optional[Sequence[str]] = None,
        tasks: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (224, 224),
        future_steps: int = 6,
        chunk_size: int = 1,
        include_current_action: bool = False,
        n_views: int = 2,
        proprio_dim: int = 7,
        eval_ratio: float = 0.0,
        is_eval: bool = False,
        source_fps: float = 20.0,
        target_hz: Optional[float] = None,
        random_stride: bool = False,
        camera_keys: Optional[Sequence[str]] = None,
        train_crop_min_scale: float = 0.9,
        eval_crop_scale: float = 0.9,
        color_jitter_brightness: float = 0.0,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.0,
        color_jitter_hue: float = 0.0,
        openpi_libero_augment: bool = False,
        openpi_base_crop_scale: float = 0.95,
        openpi_base_rotate_degrees: float = 5.0,
        image_aug_profile: str = "",
        image_jpeg_train_enabled: bool = False,
        image_jpeg_eval_enabled: bool = False,
        image_jpeg_train_quality: int = 95,
        image_jpeg_eval_quality: int = 95,
        dataset_name: str = "libero_hdf5",
        da3_input_rotate180: bool = True,
        da3_input_hflip: bool = False,
        da3_input_vflip: bool = False,
        gt_depth_root: Optional[str] = None,
        gt_depth_key: str = "depth_meters",
        gt_depth_rotate180: bool = True,
        gt_depth_hflip: bool = False,
        gt_depth_vflip: bool = False,
        gt_depth_require_geometry: bool = False,
        gt_depth_scale_mode: str = "pointmap",
        gt_depth_min_meters: float = 1e-3,
        gt_depth_require_sidecar_file: bool = False,
        filter_noops: bool = False,
        noop_threshold: float = 1.0e-4,
        uniform_action_sampling: bool = False,
        action_sampling_mode: str = "valid_window",
        virtual_terminal_action_loss: bool = False,
        virtual_front_padding: bool = False,
        hdf5_mask_key: Optional[str] = None,
        hdf5_mask_required: bool = False,
        hdf5_mask_limit: Optional[int] = None,
        action_frame: str = "base",
        proprio_orientation: str = "rpy",
        repeat_missing_views: bool = True,
        view_max_views: Optional[int] = None,
    ):
        super().__init__()
        if h5py is None:
            raise ModuleNotFoundError(
                "LiberoHDF5SequenceDataset requires h5py; install h5py in this environment."
            )

        self.dataset_name = dataset_name
        self.action_stats_key = dataset_name
        self.action_frame = normalize_action_frame(action_frame)
        self.proprio_orientation = normalize_proprio_orientation_mode(proprio_orientation)
        self.filter_noops = bool(filter_noops)
        self.noop_threshold = float(noop_threshold)
        action_sampling_mode = str(action_sampling_mode or "valid_window").lower()
        action_sampling_mode = {
            "valid": "valid_window",
            "default": "valid_window",
            "virtual": "virtual_mask",
            "virtual_action_mask": "virtual_mask",
        }.get(action_sampling_mode, action_sampling_mode)
        if action_sampling_mode not in {"valid_window", "virtual_mask"}:
            raise ValueError(
                "Unsupported LIBERO action_sampling_mode="
                f"{action_sampling_mode!r}; expected 'valid_window' or 'virtual_mask'."
            )
        legacy_virtual = action_sampling_mode == "virtual_mask"
        self.uniform_action_sampling = bool(uniform_action_sampling) or legacy_virtual
        if bool(virtual_terminal_action_loss) or bool(virtual_front_padding):
            self.uniform_action_sampling = True
        self.action_sampling_mode = "uniform" if self.uniform_action_sampling else "valid_window"
        # Keyed by (resolved hdf5_path str, demo_key). Each value is the
        # ascending int64 array of *original* HDF5 frame indices that survive
        # the OpenVLA is_noop filter. When filter_noops=False this dict stays
        # empty and __getitem__ falls back to the identity map.
        self._keep_indices_by_demo: dict[tuple[str, str], np.ndarray] = {}
        self._noop_total_orig: int = 0
        self._noop_total_kept: int = 0
        self.has_real_actions = True
        self.hdf5_root = Path(hdf5_root).expanduser()
        self.pattern = pattern
        self.hdf5_mask_key = str(hdf5_mask_key).strip() if hdf5_mask_key else None
        self.hdf5_mask_required = bool(hdf5_mask_required)
        self.hdf5_mask_limit = int(hdf5_mask_limit) if hdf5_mask_limit not in (None, "") else None
        if self.hdf5_mask_limit is not None and self.hdf5_mask_limit <= 0:
            self.hdf5_mask_limit = None
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.future_steps = int(future_steps)
        self.chunk_size = int(chunk_size)
        self.include_current_action = bool(include_current_action)
        self.action_steps = self.future_steps + 1 if self.include_current_action else self.future_steps
        self.n_views = int(n_views)
        self.proprio_dim = int(proprio_dim)
        self.is_eval = bool(is_eval)
        self.train_crop_min_scale = float(train_crop_min_scale)
        self.eval_crop_scale = float(eval_crop_scale)
        self.color_jitter_brightness = float(color_jitter_brightness)
        self.color_jitter_contrast = float(color_jitter_contrast)
        self.color_jitter_saturation = float(color_jitter_saturation)
        self.color_jitter_hue = float(color_jitter_hue)
        self.openpi_libero_augment = bool(openpi_libero_augment)
        self.openpi_base_crop_scale = float(openpi_base_crop_scale)
        self.openpi_base_rotate_degrees = float(openpi_base_rotate_degrees)
        self.image_aug_profile = str(image_aug_profile or "")
        self.image_jpeg_train_enabled = bool(image_jpeg_train_enabled)
        self.image_jpeg_eval_enabled = bool(image_jpeg_eval_enabled)
        self.image_jpeg_train_quality = max(1, min(100, int(image_jpeg_train_quality)))
        self.image_jpeg_eval_quality = max(1, min(100, int(image_jpeg_eval_quality)))
        self.da3_input_rotate180 = bool(da3_input_rotate180)
        self.da3_input_hflip = bool(da3_input_hflip)
        self.da3_input_vflip = bool(da3_input_vflip)
        self.gt_depth_root = Path(gt_depth_root).expanduser() if gt_depth_root else None
        self.gt_depth_key = str(gt_depth_key)
        self.gt_depth_rotate180 = bool(gt_depth_rotate180)
        self.gt_depth_hflip = bool(gt_depth_hflip)
        self.gt_depth_vflip = bool(gt_depth_vflip)
        self.gt_depth_require_geometry = bool(gt_depth_require_geometry)
        self.gt_depth_scale_mode = str(gt_depth_scale_mode)
        self.gt_depth_min_meters = float(gt_depth_min_meters)
        self.gt_depth_require_sidecar_file = bool(gt_depth_require_sidecar_file)
        self.fps = float(source_fps)
        self.requested_target_hz = float(target_hz) if target_hz is not None and float(target_hz) > 0 else None
        self.target_hz = self.requested_target_hz
        self.temporal_stride = 1
        if self.target_hz is not None:
            if self.target_hz > self.fps:
                self.target_hz = self.fps
            self.temporal_stride = max(1, int(round(self.fps / self.target_hz)))
        self.effective_fps = self.fps / float(self.temporal_stride)
        self.max_stride = self.temporal_stride
        self.random_stride = bool(random_stride) and not self.is_eval
        if self.random_stride and self.max_stride > 1:
            self.temporal_stride = 1
            self.effective_fps = self.fps
        elif self.random_stride and self.max_stride <= 1:
            self.random_stride = False
        self._uniform_action_sampling = self.uniform_action_sampling and not self.is_eval
        if self._uniform_action_sampling and (self.random_stride or self.temporal_stride != 1):
            raise ValueError(
                "LIBERO uniform_action_sampling currently requires stride=1 "
                "(no target_hz downsampling and no random_stride)."
            )

        if camera_keys is None:
            camera_keys = ("agentview_rgb", "eye_in_hand_rgb")
        keys = [str(key) for key in camera_keys if str(key)]
        if not keys:
            raise ValueError("LiberoHDF5SequenceDataset requires at least one camera key.")
        if bool(repeat_missing_views) and len(keys) < self.n_views:
            keys = keys + [keys[-1]] * (self.n_views - len(keys))
        self.camera_keys = keys[: self.n_views]
        self.view_max_views = int(view_max_views) if view_max_views is not None else None

        suite_filter = {str(suite) for suite in suites} if suites else None
        task_filter = {self._normalize_task_name(task) for task in tasks} if tasks else None
        # IMPORTANT: pass `None` for suite_filter to _resolve_hdf5_paths so the
        # path list covers every LIBERO suite. This keeps `global_episode_index`
        # in `_build_samples` aligned with the indexing used when the GT depth
        # sidecar files were written (sidecar filenames embed the
        # all-suite global index : see _libero_depth_sidecar_path). The suite
        # filter is then enforced at sample-build time so we still only train
        # on the requested suite.
        self._suite_filter = suite_filter
        self.hdf5_paths = self._resolve_hdf5_paths(hdf5_paths, None, task_filter)
        self.samples: list[tuple[str, str, int, str, int, int]] = []
        self._stat_samples: list[tuple[str, str, int, str, int, int]] = []
        self._fixed_starts: list[Optional[int]] = []
        self._hdf5_cache: dict[str, Any] = {}
        self._build_samples(float(eval_ratio), task_filter)
        total_before_filter = len(self.samples)
        dropped_missing_sidecar = 0
        if self.gt_depth_require_sidecar_file and self.gt_depth_root is not None:
            kept: list[tuple[str, str, int, str, int, int]] = []
            kept_fixed_starts: list[Optional[int]] = []
            for sample, fixed_start in zip(self.samples, self._fixed_starts):
                hdf5_path, demo_key, _, _, episode_index, _ = sample
                sidecar = _libero_depth_sidecar_path(
                    self.gt_depth_root, int(episode_index), hdf5_path, demo_key,
                )
                if sidecar.exists():
                    kept.append(sample)
                    kept_fixed_starts.append(fixed_start)
                else:
                    dropped_missing_sidecar += 1
            self.samples = kept
            self._fixed_starts = kept_fixed_starts
            if not self.samples:
                raise FileNotFoundError(
                    "gt_depth_require_sidecar_file=true but no demo has a sidecar "
                    f"NPZ under {self.gt_depth_root}. Checked {total_before_filter} demos."
                )

        self._has_embedded_gt_depth = False
        if self.gt_depth_root is None and self.samples:
            sample_hdf5_path, sample_demo_key, *_ = self.samples[0]
            sample_demo = self._get_hdf5(sample_hdf5_path)["data"][sample_demo_key]
            self._has_embedded_gt_depth = self._demo_has_embedded_gt_depth(sample_demo)

        mode = "eval" if self.is_eval else "train"
        depth_tag = "off"
        if self.gt_depth_root is not None:
            depth_tag = "on"
            if self.gt_depth_require_sidecar_file:
                depth_tag = f"on+filtered(-{dropped_missing_sidecar}/{total_before_filter})"
        elif self._has_embedded_gt_depth:
            depth_tag = "embedded_hdf5"
        if self.filter_noops:
            dropped = self._noop_total_orig - self._noop_total_kept
            ratio = (dropped / max(1, self._noop_total_orig)) * 100.0
            noop_tag = (
                f"on(kept={self._noop_total_kept}/{self._noop_total_orig} "
                f"drop={ratio:.1f}% threshold={self.noop_threshold:g})"
            )
        else:
            noop_tag = "off"
        print(
            f"LiberoHDF5SequenceDataset [{mode}]: {len(self.samples)} samples from "
            f"{len(self.hdf5_paths)} files fps={self.fps}->{self.effective_fps:.2f}Hz "
            f"chunk_size={self.chunk_size} random_stride={self.random_stride} cameras={self.camera_keys} "
            f"uniform_action_sampling={self.uniform_action_sampling} "
            f"hdf5_mask={self.hdf5_mask_key or 'off'} "
            f"da3_input_rotate180={self.da3_input_rotate180} "
            f"da3_input_hflip={self.da3_input_hflip} da3_input_vflip={self.da3_input_vflip} "
            f"gt_depth_rotate180={self.gt_depth_rotate180} "
            f"gt_depth_hflip={self.gt_depth_hflip} gt_depth_vflip={self.gt_depth_vflip} "
            f"gt_depth={depth_tag} noop_filter={noop_tag} "
            f"action_frame={self.action_frame} proprio_orientation={self.proprio_orientation} "
            f"openpi_libero_augment={self.openpi_libero_augment} "
            f"image_aug_profile={self.image_aug_profile or 'legacy'} "
            f"jpeg_train={self.image_jpeg_train_enabled}@q{self.image_jpeg_train_quality} "
            f"jpeg_eval={self.image_jpeg_eval_enabled}@q{self.image_jpeg_eval_quality}"
        )

    @staticmethod
    def _sort_demo_key(key: str) -> tuple[str, int]:
        match = re.search(r"(\d+)$", key)
        if match is None:
            return key, -1
        return key[: match.start()], int(match.group(1))

    @staticmethod
    def _normalize_task_name(value: Any) -> str:
        text = str(value)
        text = re.sub(r"_demo$", "", text)
        text = re.sub(r"^(?:[A-Z]+_)+SCENE\d+_", "", text)
        text = text.replace("_", " ")
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
        return " ".join(text.lower().split())

    def _task_text_from_path(self, path: Path) -> str:
        return self._normalize_task_name(path.stem)

    @staticmethod
    def _decode_hdf5_attr_text(value: Any) -> str:
        if hasattr(value, "shape") and getattr(value, "shape", None) == ():
            value = value.item()
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
        return str(value)

    def _task_text_from_hdf5_data(self, data_group: Any, path: Path) -> str:
        raw_problem_info = data_group.attrs.get("problem_info")
        if raw_problem_info is not None:
            try:
                problem_info = json.loads(self._decode_hdf5_attr_text(raw_problem_info))
            except (TypeError, ValueError, json.JSONDecodeError):
                problem_info = None
            if isinstance(problem_info, dict):
                language_instruction = problem_info.get("language_instruction")
                if language_instruction:
                    task_text = self._normalize_task_name(language_instruction)
                    if task_text:
                        return task_text
        return self._task_text_from_path(path)

    def _task_text_for_demo(
        self,
        data_group: Any,
        path: Path,
        demo_key: str,
        demo: Any,
        file_task_text: str,
    ) -> str:
        return file_task_text

    def _resolve_hdf5_paths(
        self,
        explicit_paths: Optional[Sequence[str]],
        suite_filter: Optional[set[str]],
        task_filter: Optional[set[str]],
    ) -> list[Path]:
        paths: list[Path] = []
        if explicit_paths:
            paths.extend(Path(path).expanduser() for path in explicit_paths)
        if self.hdf5_root:
            paths.extend(self.hdf5_root.rglob(self.pattern))

        unique = sorted({str(path.resolve()): path.resolve() for path in paths}.values(), key=str)
        filtered: list[Path] = []
        for path in unique:
            if not path.exists():
                raise FileNotFoundError(f"Missing LIBERO HDF5 file: {path}")
            if suite_filter is not None and path.parent.name not in suite_filter:
                continue
            # Task text is canonicalized from HDF5 metadata in _build_samples.
            # Filename filtering here can reject valid multi-token scene names.
            filtered.append(path)
        if not filtered:
            raise FileNotFoundError(
                f"No LIBERO HDF5 files matched root={self.hdf5_root} pattern={self.pattern!r}."
            )
        return filtered

    def _select_hdf5_demos_from_mask(self, h5_file: Any, hdf5_path: Path, demos: Sequence[str]) -> list[str]:
        if self.hdf5_mask_key is None:
            return list(demos)
        if "mask" not in h5_file or self.hdf5_mask_key not in h5_file["mask"]:
            if self.hdf5_mask_required:
                raise KeyError(f"{hdf5_path} is missing required HDF5 mask/{self.hdf5_mask_key}.")
            return list(demos)

        values = np.asarray(h5_file["mask"][self.hdf5_mask_key])
        mask_keys = [self._decode_hdf5_attr_text(item) for item in values.tolist()]
        available = set(demos)
        selected = [key for key in mask_keys if key in available]
        if self.hdf5_mask_limit is not None:
            selected = selected[: self.hdf5_mask_limit]
        if self.hdf5_mask_required and not selected:
            raise ValueError(f"{hdf5_path} mask/{self.hdf5_mask_key} selected no existing demos.")
        return sorted(selected, key=self._sort_demo_key)

    def _actions_for_noop_filter(self, actions: np.ndarray) -> np.ndarray:
        return actions

    def _visual_anchor_stride(self, action_stride: int) -> int:
        return max(1, int(self.chunk_size)) * max(1, int(action_stride))

    def _max_start_for_length(self, n_frames: int, action_stride: int) -> int:
        visual_stride = self._visual_anchor_stride(action_stride)
        last_image_offset = self.future_steps * visual_stride
        n_raw_actions = self.action_steps * max(1, int(self.chunk_size)) * max(1, int(action_stride))
        last_action_offset = max(0, n_raw_actions - 1)
        return int(n_frames) - 1 - max(last_image_offset, last_action_offset)

    def _build_samples(self, eval_ratio: float, task_filter: Optional[set[str]]) -> None:
        global_episode_index = 0
        split_stride = self.max_stride if self.random_stride else self.temporal_stride
        for hdf5_path in self.hdf5_paths:
            # Suite-level filter: skip emitting samples for non-matching suites
            # while advancing global_episode_index so it stays aligned with
            # the GT depth sidecar filenames (which were written from the
            # all-suite global ordering).
            suite_skipped = (
                self._suite_filter is not None
                and hdf5_path.parent.name not in self._suite_filter
            )
            with h5py.File(hdf5_path, "r") as f:
                data_group = f["data"]
                task_text = self._task_text_from_hdf5_data(data_group, hdf5_path)
                task_skipped = task_filter is not None and task_text not in task_filter
                if suite_skipped or task_skipped:
                    global_episode_index += len(f["data"].keys())
                    continue
                all_demos = sorted(data_group.keys(), key=self._sort_demo_key)
                demos = self._select_hdf5_demos_from_mask(f, hdf5_path, all_demos)
                n_total = len(demos)
                n_eval = max(1, int(n_total * eval_ratio)) if eval_ratio > 0 and n_total > 0 else 0
                if self.is_eval:
                    selected = set(demos[n_total - n_eval :]) if n_eval > 0 else set()
                else:
                    selected = set(demos[: n_total - n_eval]) if n_eval > 0 else set(demos)

                for demo_key in all_demos:
                    demo = data_group[demo_key]
                    if demo_key in selected:
                        sample_task_text = self._task_text_for_demo(
                            data_group,
                            hdf5_path,
                            demo_key,
                            demo,
                            task_text,
                        )
                        n_frames = int(demo["actions"].shape[0])
                        n_kept = n_frames
                        if self.filter_noops:
                            actions_np = np.asarray(demo["actions"], dtype=np.float32)
                            noop_actions_np = self._actions_for_noop_filter(actions_np)
                            keep_arr = _libero_compute_keep_indices(noop_actions_np, self.noop_threshold)
                            n_kept = int(keep_arr.shape[0])
                            self._noop_total_orig += n_frames
                            self._noop_total_kept += n_kept
                            self._keep_indices_by_demo[(str(hdf5_path), demo_key)] = keep_arr
                        max_start = self._max_start_for_length(n_kept, split_stride)
                        if max_start >= 0:
                            obs = demo.get("obs")
                            if obs is None:
                                raise KeyError(f"{hdf5_path}:{demo_key} has no obs group.")
                            missing = [key for key in self.camera_keys if key not in obs]
                            if missing:
                                raise KeyError(f"{hdf5_path}:{demo_key} missing camera key(s): {missing}")
                            sample = (
                                str(hdf5_path),
                                demo_key,
                                max_start,
                                sample_task_text,
                                global_episode_index,
                                n_kept,
                            )
                            self._stat_samples.append(sample)
                            if self._uniform_action_sampling:
                                supervised_raw_steps = self.future_steps * max(1, int(self.chunk_size))
                                # Uniform action coverage: every real raw action
                                # row is eligible for the same number of
                                # action-loss terms over the max supervised
                                # action span. Boundary observations/actions
                                # missing boundary observations/actions are
                                # zeroed and masked in __getitem__ instead of
                                # being represented by
                                # repeated first/last frames.
                                min_virtual_start = -(supervised_raw_steps - 1)
                                max_virtual_start = n_kept - 1
                                if max_virtual_start >= min_virtual_start:
                                    for fixed_start in range(min_virtual_start, max_virtual_start + 1):
                                        self.samples.append(sample)
                                        self._fixed_starts.append(int(fixed_start))
                            else:
                                self.samples.append(sample)
                                self._fixed_starts.append(None)
                    global_episode_index += 1

    def _get_hdf5(self, path: str) -> Any:
        if path not in self._hdf5_cache:
            self._hdf5_cache[path] = h5py.File(path, "r")
        return self._hdf5_cache[path]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hdf5_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_stride(self, rng: Optional[np.random.Generator] = None) -> int:
        if self.random_stride and self.max_stride > 1:
            if rng is not None:
                return int(rng.integers(1, self.max_stride + 1))
            return random.randint(1, self.max_stride)
        return int(self.temporal_stride)

    def _canonicalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action = _ensure_2d_tensor(action).to(dtype=torch.float32)
        if action.shape[-1] < 7:
            raise ValueError(f"LIBERO HDF5 action dim {action.shape[-1]} is unsupported.")
        if action.shape[-1] > 7:
            action = action[..., :7]
        action = action.clone()
        action[..., 6] = (action[..., 6] + 1.0) * 0.5
        return action

    def _maybe_convert_action_frame(
        self,
        action: torch.Tensor,
        current_proprio: Optional[torch.Tensor],
        target_proprio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_frame in {"base", "base_delta"}:
            return action
        if self.action_frame == "eef_delta":
            return _world_delta_to_eef_delta_action(
                action,
                current_proprio,
                proprio_orientation=self.proprio_orientation,
            )
        if self.action_frame == "eef_relative":
            anchor = None
            if current_proprio is not None:
                anchor = current_proprio.reshape(-1, current_proprio.shape[-1])[:1]
            return _world_delta_sequence_to_eef_relative_trajectory(
                action,
                anchor,
                target_proprio=target_proprio,
                proprio_orientation=self.proprio_orientation,
            )
        raise AssertionError(f"Unhandled action_frame={self.action_frame!r}")

    def _resample_with_stride(self, action: torch.Tensor, stride: int) -> torch.Tensor:
        action = _ensure_2d_tensor(action).to(dtype=torch.float32)
        total_target = self.action_steps * max(1, int(self.chunk_size))
        stride = max(1, int(stride))
        if stride <= 1:
            return action[:total_target]

        needed = total_target * stride
        if action.shape[0] < needed:
            pad = action[-1:].expand(needed - action.shape[0], -1)
            action = torch.cat([action, pad], dim=0)
        grouped = action[:needed].reshape(total_target, stride, action.shape[-1])
        aggregated = torch.zeros(total_target, action.shape[-1], dtype=action.dtype, device=action.device)
        aggregated[:, :3] = grouped[:, :, :3].sum(dim=1)
        aggregated[:, 3:6] = _compose_rotvec_batch(grouped[:, :, 3:6])
        aggregated[:, 6] = grouped[:, -1, 6]
        return aggregated

    def _read_proprio(self, demo: Any, frame_indices: Sequence[int]) -> torch.Tensor:
        obs = demo["obs"]
        indices = list(int(i) for i in frame_indices)
        def _read_indexed(ds: Any) -> np.ndarray:
            if len(indices) <= 1 or all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)):
                return np.asarray(ds[indices])
            return np.asarray([np.asarray(ds[int(i)]) for i in indices])

        if "ee_states" in obs:
            pos_rpy = torch.as_tensor(_read_indexed(obs["ee_states"]), dtype=torch.float32)
        elif "ee_pos" in obs and "ee_ori" in obs:
            pos = torch.as_tensor(_read_indexed(obs["ee_pos"]), dtype=torch.float32)
            rpy = torch.as_tensor(_read_indexed(obs["ee_ori"]), dtype=torch.float32)
            pos_rpy = torch.cat([pos, rpy], dim=-1)
        else:
            pos_rpy = torch.zeros(len(indices), 6, dtype=torch.float32)

        if "gripper_states" in obs:
            gripper = torch.as_tensor(_read_indexed(obs["gripper_states"]), dtype=torch.float32)
            if gripper.shape[-1] >= 2:
                grip_width = gripper[..., 0:1] - gripper[..., 1:2]
            else:
                grip_width = _pad_or_truncate_last_dim(gripper, 1)
        else:
            grip_width = torch.zeros(len(indices), 1, dtype=torch.float32)
        proprio = torch.cat([pos_rpy[..., :6], grip_width], dim=-1)
        if proprio.shape[-1] > self.proprio_dim:
            proprio = proprio[..., : self.proprio_dim]
        elif proprio.shape[-1] < self.proprio_dim:
            pad = torch.zeros(proprio.shape[0], self.proprio_dim - proprio.shape[-1], dtype=proprio.dtype)
            proprio = torch.cat([proprio, pad], dim=-1)
        return proprio

    def _kept_positions_to_hdf5_indices(
        self,
        hdf5_path: str,
        demo_key: str,
        kept_positions: Sequence[int],
        n_kept: int,
    ) -> list[int]:
        if n_kept <= 0:
            raise ValueError(f"{hdf5_path}:{demo_key} has no kept frames/actions.")
        clipped = np.clip(np.asarray(kept_positions, dtype=np.int64), 0, int(n_kept) - 1)
        keep_arr = self._keep_indices_by_demo.get((hdf5_path, demo_key))
        if keep_arr is not None:
            return [int(keep_arr[int(pos)]) for pos in clipped.tolist()]
        return [int(pos) for pos in clipped.tolist()]

    def _read_target_proprio_for_kept_positions(
        self,
        demo: Any,
        hdf5_path: str,
        demo_key: str,
        target_positions: Sequence[int],
        n_kept: int,
    ) -> torch.Tensor:
        target_indices = self._kept_positions_to_hdf5_indices(
            hdf5_path,
            demo_key,
            target_positions,
            n_kept,
        )
        return self._read_proprio(demo, target_indices)

    def _read_actions_for_window(
        self,
        demo: Any,
        hdf5_path: str,
        demo_key: str,
        start_t: int,
        n_kept: int,
        stride: int,
        use_uniform_mask: bool,
        valid_raw_count: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_raw_actions = self.action_steps * max(1, int(self.chunk_size)) * max(1, int(stride))
        if not use_uniform_mask:
            keep_arr = self._keep_indices_by_demo.get((hdf5_path, demo_key))
            if keep_arr is not None:
                action_orig_idx = keep_arr[start_t : start_t + n_raw_actions]
                action_indices = action_orig_idx.tolist()
                raw_action_np = np.asarray(demo["actions"][action_indices])
            else:
                action_orig_idx = list(range(start_t, start_t + n_raw_actions))
                raw_action_np = np.asarray(demo["actions"][action_orig_idx])
            raw_action = torch.as_tensor(raw_action_np, dtype=torch.float32)
            action_proprio = self._read_proprio(demo, action_orig_idx)
            actions = self._resample_with_stride(self._canonicalize_action(raw_action), stride)
            target_positions = int(start_t) + (np.arange(actions.shape[0], dtype=np.int64) + 1) * max(1, int(stride))
            target_proprio = self._read_target_proprio_for_kept_positions(
                demo,
                hdf5_path,
                demo_key,
                target_positions.tolist(),
                n_kept,
            )
            actions = self._maybe_convert_action_frame(
                actions,
                _select_action_anchor_proprio(action_proprio, actions.shape[0], stride),
                target_proprio,
            )
            action_loss_mask = None
        else:
            if stride != 1:
                raise ValueError("Uniform LIBERO action sampling only supports stride=1.")
            flat_offsets = np.arange(n_raw_actions, dtype=np.int64)
            kept_positions = int(start_t) + flat_offsets
            valid_raw_count = 0 if valid_raw_count is None else int(valid_raw_count)
            valid_raw_count = max(0, min(valid_raw_count, n_raw_actions))
            valid_flat = flat_offsets < valid_raw_count
            action_indices = self._kept_positions_to_hdf5_indices(
                hdf5_path, demo_key, kept_positions.tolist(), n_kept
            )
            raw_action_np = np.asarray(
                [np.asarray(demo["actions"][int(action_idx)]) for action_idx in action_indices],
                dtype=np.float32,
            )
            raw_action = torch.as_tensor(raw_action_np, dtype=torch.float32)
            action_proprio = self._read_proprio(demo, action_indices)
            target_proprio = self._read_target_proprio_for_kept_positions(
                demo,
                hdf5_path,
                demo_key,
                (kept_positions + 1).tolist(),
                n_kept,
            )
            actions = self._maybe_convert_action_frame(
                self._canonicalize_action(raw_action),
                action_proprio,
                target_proprio,
            )
            if not bool(valid_flat.all()):
                actions = actions.clone()
                actions[torch.from_numpy(~valid_flat)] = 0.0
            action_loss_mask = torch.from_numpy(valid_flat.reshape(self.action_steps, max(1, int(self.chunk_size))))

        if self.chunk_size > 1:
            actions = actions.reshape(self.action_steps, self.chunk_size, -1)
        else:
            actions = actions.reshape(self.action_steps, -1)
        return actions, action_loss_mask

    def _read_past_action_history_for_window(
        self,
        demo: Any,
        hdf5_path: str,
        demo_key: str,
        start_t: int,
        n_kept: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stride != 1:
            raise ValueError("LIBERO past_action_history currently supports stride=1.")
        chunk_span = max(1, int(self.chunk_size))
        n_raw_actions = self.action_steps * chunk_span
        flat_offsets = np.arange(n_raw_actions, dtype=np.int64)
        step_offsets = flat_offsets // chunk_span
        chunk_offsets = flat_offsets % chunk_span
        kept_positions = int(start_t) + step_offsets * chunk_span - chunk_span + chunk_offsets
        valid_flat = (kept_positions >= 0) & (kept_positions <= int(n_kept) - 1)
        action_indices = self._kept_positions_to_hdf5_indices(
            hdf5_path, demo_key, kept_positions.tolist(), n_kept
        )
        raw_action_np = np.asarray(
            [np.asarray(demo["actions"][int(action_idx)]) for action_idx in action_indices],
            dtype=np.float32,
        )
        raw_action = torch.as_tensor(raw_action_np, dtype=torch.float32)
        action_proprio = self._read_proprio(demo, action_indices)
        target_proprio = self._read_target_proprio_for_kept_positions(
            demo,
            hdf5_path,
            demo_key,
            (kept_positions + 1).tolist(),
            n_kept,
        )
        actions = self._maybe_convert_action_frame(
            self._canonicalize_action(raw_action),
            action_proprio,
            target_proprio,
        )
        if not bool(valid_flat.all()):
            actions = actions.clone()
            actions[torch.from_numpy(~valid_flat)] = 0.0
        action_mask = torch.from_numpy(valid_flat.reshape(self.action_steps, chunk_span))
        if self.chunk_size > 1:
            actions = actions.reshape(self.action_steps, self.chunk_size, -1)
        else:
            actions = actions.reshape(self.action_steps, -1)
        return actions, action_mask

    def _uniform_transition_loss_mask(
        self,
        start_t: int,
        n_kept: int,
        visual_stride: int,
        valid_raw_count: int,
    ) -> torch.Tensor:
        mask = np.zeros(self.action_steps, dtype=bool)
        chunk_span = max(1, int(self.chunk_size))
        valid_raw_count = max(0, int(valid_raw_count))
        for step in range(self.action_steps):
            target_pos = int(start_t) + step * int(visual_stride)
            if not (0 <= target_pos < int(n_kept)):
                continue
            if step == 0:
                mask[step] = True
                continue
            if step > self.future_steps:
                continue
            # A future anchor at step j is valid only if all low-level actions
            # leading from the previous anchor to that observation exist.
            mask[step] = step * chunk_span <= valid_raw_count
        return torch.from_numpy(mask)

    def _demo_has_embedded_gt_depth(self, demo: Any) -> bool:
        """Return True when a LIBERO HDF5 demo carries embedded metric depth.

        Regenerated no-op LIBERO HDF5s can store `obs/agentview_depth` and
        `obs/eye_in_hand_depth` directly inside the demo instead of using the
        external sidecar NPZ format. Geometry is optional unless the current
        depth-loss mode requires it.
        """
        obs = demo.get("obs")
        if obs is None:
            return False
        for camera_key in self.camera_keys:
            prefix = _libero_hdf5_obs_camera_prefix(camera_key)
            if f"{prefix}_depth" not in obs:
                return False
        return True

    def _load_gt_depth_targets(
        self,
        hdf5_path: str,
        demo_key: str,
        episode_index: int,
        frame_indices: Sequence[int],
        cam_crop_params: Mapping[str, Tuple[float, int, int, int, int]],
        cam_source_sizes: Mapping[str, Tuple[int, int]],
        cam_rotation_angles: Optional[Mapping[str, float]] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        demo: Any | None = None,
    ) -> Dict[str, torch.Tensor]:
        if demo is None:
            demo = self._get_hdf5(hdf5_path)["data"][demo_key]
        obs = demo.get("obs")

        if self.gt_depth_root is None:
            if obs is None or not self._demo_has_embedded_gt_depth(demo):
                return {}

            has_geometry = True
            for camera_key in self.camera_keys:
                prefix = _libero_hdf5_obs_camera_prefix(camera_key)
                if (
                    f"{prefix}_intrinsic" not in obs
                    or f"{prefix}_extrinsic_c2w" not in obs
                ):
                    has_geometry = False
                    break

            if self.gt_depth_scale_mode == "pointmap" and not has_geometry:
                raise KeyError(
                    f"{hdf5_path}:{demo_key} is missing embedded camera geometry. "
                    "Regenerate the no-op HDF5 with intrinsic/extrinsic capture enabled."
                )
            if self.gt_depth_require_geometry and not has_geometry:
                raise KeyError(
                    f"{hdf5_path}:{demo_key} has no required embedded camera geometry."
                )

            depth_frames: list[torch.Tensor] = []
            mask_frames: list[torch.Tensor] = []
            k_frames: list[np.ndarray] = []
            c2w_frames: list[np.ndarray] = []

            for frame_idx in frame_indices:
                timestep_depths: list[torch.Tensor] = []
                timestep_masks: list[torch.Tensor] = []
                timestep_k: list[np.ndarray] = []
                timestep_c2w: list[np.ndarray] = []
                for camera_key in self.camera_keys:
                    prefix = _libero_hdf5_obs_camera_prefix(camera_key)
                    crop_params = cam_crop_params[camera_key]
                    rotation_angle = float((cam_rotation_angles or {}).get(camera_key, 0.0))
                    depth_slice = np.asarray(obs[f"{prefix}_depth"][int(frame_idx)], dtype=np.float32)
                    depth_source_size = (
                        int(depth_slice.shape[0]),
                        int(depth_slice.shape[1]),
                    )
                    crop_params_depth = _scale_crop_params_to_source(
                        crop_params,
                        cam_source_sizes[camera_key],
                        depth_source_size,
                    )
                    depth_source = _maybe_vflip_image(
                        _maybe_hflip_image(
                            _maybe_rotate_image_180(
                                depth_slice,
                                self.gt_depth_rotate180,
                            ),
                            self.gt_depth_hflip,
                        ),
                        self.gt_depth_vflip,
                    )
                    depth_t, mask_t = _crop_resize_depth_and_mask(
                        np.asarray(depth_source, dtype=np.float32),
                        self.image_size,
                        crop_params_depth,
                        self.gt_depth_min_meters,
                    )
                    if abs(rotation_angle) > 1.0e-6:
                        depth_t, mask_t = _rotate_depth_and_mask_train(
                            depth_t,
                            mask_t,
                            rotation_angle,
                        )
                    timestep_depths.append(depth_t)
                    timestep_masks.append(mask_t)

                    if has_geometry:
                        intrinsic = np.asarray(
                            obs[f"{prefix}_intrinsic"][int(frame_idx)],
                            dtype=np.float32,
                        )
                        extrinsic = np.asarray(
                            obs[f"{prefix}_extrinsic_c2w"][int(frame_idx)],
                            dtype=np.float32,
                        )
                        timestep_k.append(
                            _transform_intrinsics_for_crop_resize(
                                intrinsic,
                                crop_params_depth,
                                self.image_size,
                                depth_source_size,
                                self.gt_depth_rotate180,
                                self.gt_depth_hflip,
                                self.gt_depth_vflip,
                                rotation_angle,
                            )
                        )
                        timestep_c2w.append(extrinsic)
                depth_frames.append(torch.stack(timestep_depths, dim=0))
                mask_frames.append(torch.stack(timestep_masks, dim=0))
                if has_geometry:
                    k_frames.append(np.stack(timestep_k, axis=0))
                    c2w_frames.append(np.stack(timestep_c2w, axis=0))

            gt_depth_m = torch.stack(depth_frames, dim=0).float()
            gt_depth_mask = torch.stack(mask_frames, dim=0).bool()
            if frame_valid_mask is not None:
                valid = frame_valid_mask.to(dtype=torch.bool).view(-1, 1, 1, 1)
                gt_depth_mask = gt_depth_mask & valid
                gt_depth_m = torch.where(valid, gt_depth_m, torch.zeros_like(gt_depth_m))
            if self.gt_depth_scale_mode == "pointmap":
                k_np = np.stack(k_frames, axis=0).astype(np.float32)
                c2w_np = np.stack(c2w_frames, axis=0).astype(np.float32)
                scene_scale = _scene_scale_from_pointmaps(
                    gt_depth_m,
                    gt_depth_mask,
                    k_np,
                    c2w_np,
                )
            elif self.gt_depth_scale_mode == "median_depth":
                valid_depth = gt_depth_m[gt_depth_mask]
                scene_scale = float(torch.median(valid_depth).item()) if valid_depth.numel() else 1.0
                scene_scale = max(scene_scale, 1e-3)
            elif self.gt_depth_scale_mode in {"none", "raw"}:
                scene_scale = 1.0
            else:
                raise ValueError(f"Unsupported gt_depth_scale_mode={self.gt_depth_scale_mode!r}")

            gt_depth_da3 = gt_depth_m / float(scene_scale)
            out = {
                "gt_depth_meters": gt_depth_m,
                "gt_depth_da3": gt_depth_da3.float(),
                "gt_depth_mask": gt_depth_mask,
                "gt_depth_scene_scale": torch.tensor(float(scene_scale), dtype=torch.float32),
            }
            if k_frames and c2w_frames:
                out["gt_camera_intrinsics"] = torch.from_numpy(
                    np.stack(k_frames, axis=0).astype(np.float32)
                )
                out["gt_camera_extrinsics_c2w"] = torch.from_numpy(
                    np.stack(c2w_frames, axis=0).astype(np.float32)
                )
            return out
        # Prefer the fast memmap layout when a sibling
        # {gt_depth_root}_memmap/ directory has both {stem}.depth.npy and
        # {stem}.geometry.npz. See scripts/convert_libero_sidecar_to_memmap.py
        # for the format. Reading is ~300x faster because we memory-map the
        # fp16 depth and slice only the time window we need instead of
        # decompressing the full demo's depth_meters array from zlib NPZ.
        mmap_depth_path, mmap_geom_path = _libero_depth_memmap_paths(
            self.gt_depth_root, int(episode_index), hdf5_path, demo_key,
        )
        use_memmap = mmap_depth_path is not None and mmap_depth_path.exists() and mmap_geom_path.exists()

        if use_memmap:
            depth_meters_mm = np.load(mmap_depth_path, mmap_mode="r")
            with np.load(mmap_geom_path, allow_pickle=True) as geom:
                sidecar_indices = np.asarray(geom["frame_indices"], dtype=np.int64)
                sidecar_cameras = [str(x) for x in geom["camera_names"].tolist()]
                has_geometry = "camera_intrinsics" in geom and "camera_extrinsics_c2w" in geom
                camera_intrinsics = (
                    np.asarray(geom["camera_intrinsics"], dtype=np.float32)
                    if "camera_intrinsics" in geom else None
                )
                camera_extrinsics_c2w = (
                    np.asarray(geom["camera_extrinsics_c2w"], dtype=np.float32)
                    if "camera_extrinsics_c2w" in geom else None
                )
            # Keep `depth_meters` as the memmap; we'll fp32-cast per slice below.
            depth_meters = depth_meters_mm
            sidecar = mmap_depth_path  # for error messages
        else:
            sidecar = _libero_depth_sidecar_path(
                self.gt_depth_root,
                int(episode_index),
                hdf5_path,
                demo_key,
            )
            if not sidecar.exists():
                raise FileNotFoundError(f"Missing LIBERO GT depth sidecar: {sidecar}")

            with np.load(sidecar, allow_pickle=True) as data:
                if self.gt_depth_key not in data:
                    raise KeyError(f"{sidecar} has no {self.gt_depth_key!r} array")
                depth_meters = np.asarray(data[self.gt_depth_key], dtype=np.float32)
                sidecar_indices = np.asarray(data["frame_indices"], dtype=np.int64)
                sidecar_cameras = [str(x) for x in data["camera_names"].tolist()]
                has_geometry = "camera_intrinsics" in data and "camera_extrinsics_c2w" in data
                camera_intrinsics = (
                    np.asarray(data["camera_intrinsics"], dtype=np.float32)
                    if "camera_intrinsics" in data else None
                )
                camera_extrinsics_c2w = (
                    np.asarray(data["camera_extrinsics_c2w"], dtype=np.float32)
                    if "camera_extrinsics_c2w" in data else None
                )

        if self.gt_depth_scale_mode == "pointmap" and not has_geometry:
            raise KeyError(
                f"{sidecar} has no camera geometry. Re-export with "
                "scripts/export_libero_aligned_gt_depth.py without --no-camera-geometry."
            )
        if self.gt_depth_require_geometry and not has_geometry:
            raise KeyError(f"{sidecar} has no required camera geometry.")

        frame_lookup = {int(frame): i for i, frame in enumerate(sidecar_indices.tolist())}
        camera_lookup = {name: i for i, name in enumerate(sidecar_cameras)}
        depth_frames: list[torch.Tensor] = []
        mask_frames: list[torch.Tensor] = []
        k_frames: list[np.ndarray] = []
        c2w_frames: list[np.ndarray] = []

        for frame_idx in frame_indices:
            src_t = frame_lookup.get(int(frame_idx))
            if src_t is None:
                raise KeyError(f"Frame {frame_idx} missing in {sidecar}")
            timestep_depths: list[torch.Tensor] = []
            timestep_masks: list[torch.Tensor] = []
            timestep_k: list[np.ndarray] = []
            timestep_c2w: list[np.ndarray] = []
            for camera_key in self.camera_keys:
                cam_name = _libero_depth_camera_name(camera_key)
                src_v = camera_lookup.get(cam_name)
                if src_v is None:
                    raise KeyError(
                        f"Camera {camera_key!r} normalized to {cam_name!r} missing in {sidecar}; "
                        f"available={sidecar_cameras}"
                    )
                crop_params = cam_crop_params[camera_key]
                rotation_angle = float((cam_rotation_angles or {}).get(camera_key, 0.0))
                # Pull the single (H, W) slice into memory; cast to float32
                # so downstream crop/resize math is dtype-safe whether the
                # source was a fp32 NPZ array or a fp16 memmap.
                depth_slice = np.asarray(depth_meters[src_t, src_v], dtype=np.float32)
                depth_source_size = (
                    int(depth_slice.shape[0]),
                    int(depth_slice.shape[1]),
                )
                crop_params_depth = _scale_crop_params_to_source(
                    crop_params,
                    cam_source_sizes[camera_key],
                    depth_source_size,
                )
                depth_source = _maybe_vflip_image(
                    _maybe_hflip_image(
                        _maybe_rotate_image_180(
                            depth_slice,
                            self.gt_depth_rotate180,
                        ),
                        self.gt_depth_hflip,
                    ),
                    self.gt_depth_vflip,
                )
                depth_t, mask_t = _crop_resize_depth_and_mask(
                    np.asarray(depth_source, dtype=np.float32),
                    self.image_size,
                    crop_params_depth,
                    self.gt_depth_min_meters,
                )
                if abs(rotation_angle) > 1.0e-6:
                    depth_t, mask_t = _rotate_depth_and_mask_train(
                        depth_t,
                        mask_t,
                        rotation_angle,
                    )
                timestep_depths.append(depth_t)
                timestep_masks.append(mask_t)

                if has_geometry:
                    assert camera_intrinsics is not None
                    assert camera_extrinsics_c2w is not None
                    timestep_k.append(
                        _transform_intrinsics_for_crop_resize(
                            camera_intrinsics[src_t, src_v],
                            crop_params_depth,
                            self.image_size,
                            depth_source_size,
                            self.gt_depth_rotate180,
                            self.gt_depth_hflip,
                            self.gt_depth_vflip,
                            rotation_angle,
                        )
                    )
                    timestep_c2w.append(camera_extrinsics_c2w[src_t, src_v].astype(np.float32))
            depth_frames.append(torch.stack(timestep_depths, dim=0))
            mask_frames.append(torch.stack(timestep_masks, dim=0))
            if has_geometry:
                k_frames.append(np.stack(timestep_k, axis=0))
                c2w_frames.append(np.stack(timestep_c2w, axis=0))

        gt_depth_m = torch.stack(depth_frames, dim=0).float()
        gt_depth_mask = torch.stack(mask_frames, dim=0).bool()
        if frame_valid_mask is not None:
            valid = frame_valid_mask.to(dtype=torch.bool).view(-1, 1, 1, 1)
            gt_depth_mask = gt_depth_mask & valid
            gt_depth_m = torch.where(valid, gt_depth_m, torch.zeros_like(gt_depth_m))
        if self.gt_depth_scale_mode == "pointmap":
            k_np = np.stack(k_frames, axis=0).astype(np.float32)
            c2w_np = np.stack(c2w_frames, axis=0).astype(np.float32)
            scene_scale = _scene_scale_from_pointmaps(
                gt_depth_m,
                gt_depth_mask,
                k_np,
                c2w_np,
            )
        elif self.gt_depth_scale_mode == "median_depth":
            valid_depth = gt_depth_m[gt_depth_mask]
            scene_scale = float(torch.median(valid_depth).item()) if valid_depth.numel() else 1.0
            scene_scale = max(scene_scale, 1e-3)
        elif self.gt_depth_scale_mode in {"none", "raw"}:
            scene_scale = 1.0
        else:
            raise ValueError(f"Unsupported gt_depth_scale_mode={self.gt_depth_scale_mode!r}")

        gt_depth_da3 = gt_depth_m / float(scene_scale)
        out = {
            "gt_depth_meters": gt_depth_m,
            "gt_depth_da3": gt_depth_da3.float(),
            "gt_depth_mask": gt_depth_mask,
            "gt_depth_scene_scale": torch.tensor(float(scene_scale), dtype=torch.float32),
        }
        if k_frames and c2w_frames:
            out["gt_camera_intrinsics"] = torch.from_numpy(
                np.stack(k_frames, axis=0).astype(np.float32)
            )
            out["gt_camera_extrinsics_c2w"] = torch.from_numpy(
                np.stack(c2w_frames, axis=0).astype(np.float32)
            )
        return out

    def _resolve_obs_image_array(
        self,
        demo: Any,
        camera_key: str,
        hdf5_path: str,
        demo_key: str,
    ) -> Any:
        """Return an indexable [T, H, W, C] image array for (demo, camera_key).

        Default: read from HDF5 obs. Subclasses may override to redirect to a
        per-camera memmap or other backing store.
        """
        return demo["obs"][camera_key]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        hdf5_path, demo_key, max_start, task_text, episode_index, n_kept = self.samples[idx]
        demo = self._get_hdf5(hdf5_path)["data"][demo_key]
        stride = self._sample_stride()
        visual_stride = self._visual_anchor_stride(stride)
        fixed_start = self._fixed_starts[idx] if idx < len(self._fixed_starts) else None
        use_uniform_mask = fixed_start is not None
        virtual_t = int(fixed_start) if use_uniform_mask else _sample_episode_start(0, max_start, idx, self.is_eval)
        t = max(0, virtual_t) if use_uniform_mask else virtual_t
        valid_raw_count: Optional[int] = None
        if use_uniform_mask:
            supervised_raw_steps = self.future_steps * max(1, int(self.chunk_size))
            front_missing = max(0, -int(virtual_t))
            remaining_window = max(0, supervised_raw_steps - front_missing)
            remaining_episode = max(0, int(n_kept) - int(t))
            valid_raw_count = min(remaining_window, remaining_episode)
        kept_visual_positions = [t + dt * visual_stride for dt in range(self.future_steps + 1)]
        if use_uniform_mask:
            valid_raw_count_int = int(valid_raw_count or 0)
            context_valid_mask = torch.as_tensor(
                [
                    (0 <= int(pos) < int(n_kept))
                    and (dt * int(visual_stride) <= valid_raw_count_int)
                    for dt, pos in enumerate(kept_visual_positions)
                ],
                dtype=torch.bool,
            )
        else:
            context_valid_mask = torch.as_tensor(
                [0 <= int(pos) < int(n_kept) for pos in kept_visual_positions],
                dtype=torch.bool,
            )

        # Map kept-space positions → original HDF5 indices when noop filter is on.
        # When off, identity map keeps behavior bit-identical to the legacy path.
        frame_indices = self._kept_positions_to_hdf5_indices(
            hdf5_path, demo_key, kept_visual_positions, n_kept
        )

        cam_crop_params = {}
        cam_rotation_angles = {}
        cam_source_sizes = {}
        for camera_key in self.camera_keys:
            first_frame = self._resolve_obs_image_array(demo, camera_key, hdf5_path, demo_key)[frame_indices[0]]
            img_t = torch.as_tensor(np.asarray(first_frame))
            if img_t.ndim == 3 and img_t.shape[0] not in (1, 3):
                img_t = img_t.permute(2, 0, 1)
            h, w = img_t.shape[-2:]
            cam_source_sizes[camera_key] = (int(h), int(w))
            if self.openpi_libero_augment and not self.is_eval:
                if _is_wrist_camera_key(camera_key):
                    cam_crop_params[camera_key] = _full_crop_params(h, w)
                else:
                    cam_crop_params[camera_key] = _sample_fixed_crop_params(
                        h,
                        w,
                        self.openpi_base_crop_scale,
                    )
                    cam_rotation_angles[camera_key] = _sample_rotation_angle(
                        self.openpi_base_rotate_degrees
                    )
            else:
                cam_crop_params[camera_key] = _sample_crop_params(
                    h,
                    w,
                    self.image_size,
                    self.is_eval,
                    self.train_crop_min_scale,
                    self.eval_crop_scale,
                )

        all_view_images = []
        all_view_target_images = []
        all_view_target_masks = []
        build_target_images = self.openpi_libero_augment and not self.is_eval
        zero_image = torch.zeros(3, int(self.image_size[0]), int(self.image_size[1]), dtype=torch.float32)
        zero_mask = torch.zeros(int(self.image_size[0]), int(self.image_size[1]), dtype=torch.bool)
        for frame_idx, frame_valid in zip(frame_indices, context_valid_mask.tolist()):
            timestep_images = []
            timestep_target_images = []
            timestep_target_masks = []
            for camera_key in self.camera_keys:
                if not frame_valid:
                    timestep_images.append(zero_image.clone())
                    if build_target_images:
                        timestep_target_images.append(zero_image.clone())
                        timestep_target_masks.append(zero_mask.clone())
                else:
                    raw_image = _maybe_vflip_image(
                        _maybe_hflip_image(
                            _maybe_rotate_image_180(
                                self._resolve_obs_image_array(demo, camera_key, hdf5_path, demo_key)[frame_idx],
                                self.da3_input_rotate180,
                            ),
                            self.da3_input_hflip,
                        ),
                        self.da3_input_vflip,
                    )
                    timestep_images.append(
                        _normalize_image_tensor(
                            raw_image,
                            self.image_size,
                            is_eval=self.is_eval,
                            train_crop_min_scale=self.train_crop_min_scale,
                            eval_crop_scale=self.eval_crop_scale,
                            crop_params=cam_crop_params[camera_key],
                            color_jitter_brightness=self.color_jitter_brightness,
                            color_jitter_contrast=self.color_jitter_contrast,
                            color_jitter_saturation=self.color_jitter_saturation,
                            color_jitter_hue=self.color_jitter_hue,
                            camera_key=camera_key,
                            openpi_libero_augment=self.openpi_libero_augment,
                            openpi_base_rotate_degrees=self.openpi_base_rotate_degrees,
                            openpi_base_rotation_angle=cam_rotation_angles.get(camera_key),
                            jpeg_enabled=(
                                self.image_jpeg_eval_enabled if self.is_eval else self.image_jpeg_train_enabled
                            ),
                            jpeg_quality=(
                                self.image_jpeg_eval_quality if self.is_eval else self.image_jpeg_train_quality
                            ),
                        )
                    )
                    if build_target_images:
                        angle = 0.0 if _is_wrist_camera_key(camera_key) else float(cam_rotation_angles.get(camera_key, 0.0))
                        timestep_target_images.append(
                            _normalize_image_tensor(
                                raw_image,
                                self.image_size,
                                is_eval=self.is_eval,
                                train_crop_min_scale=self.train_crop_min_scale,
                                eval_crop_scale=self.eval_crop_scale,
                                crop_params=cam_crop_params[camera_key],
                                color_jitter_brightness=0.0,
                                color_jitter_contrast=0.0,
                                color_jitter_saturation=0.0,
                                color_jitter_hue=0.0,
                                camera_key=camera_key,
                                openpi_libero_augment=self.openpi_libero_augment,
                                openpi_base_rotate_degrees=self.openpi_base_rotate_degrees,
                                openpi_base_rotation_angle=cam_rotation_angles.get(camera_key),
                                jpeg_enabled=False,
                            )
                        )
                        timestep_target_masks.append(
                            _rotation_valid_mask_train(self.image_size, angle)
                        )
            all_view_images.append(torch.stack(timestep_images, dim=0))
            if build_target_images:
                all_view_target_images.append(torch.stack(timestep_target_images, dim=0))
                all_view_target_masks.append(torch.stack(timestep_target_masks, dim=0))
        all_view_images = torch.stack(all_view_images, dim=0)
        all_view_target_images_t = None
        all_view_target_mask_t = None
        if build_target_images:
            all_view_target_images_t = torch.stack(all_view_target_images, dim=0)
            all_view_target_mask_t = torch.stack(all_view_target_masks, dim=0)
        actions, action_loss_mask = self._read_actions_for_window(
            demo=demo,
            hdf5_path=hdf5_path,
            demo_key=demo_key,
            start_t=t,
            n_kept=n_kept,
            stride=stride,
            use_uniform_mask=use_uniform_mask,
            valid_raw_count=valid_raw_count,
        )
        past_action_history = None
        past_action_history_mask = None
        if use_uniform_mask:
            past_action_history, past_action_history_mask = self._read_past_action_history_for_window(
                demo=demo,
                hdf5_path=hdf5_path,
                demo_key=demo_key,
                start_t=t,
                n_kept=n_kept,
                stride=stride,
            )

        proprioception = self._read_proprio(demo, frame_indices)
        if use_uniform_mask and not bool(context_valid_mask.all()):
            proprioception = proprioception * context_valid_mask.to(dtype=proprioception.dtype).unsqueeze(-1)
        sample = {
            "current_images": all_view_images[0],
            "future_images": all_view_images[1:],
            "all_view_images": all_view_images,
            "actions": actions,
            "proprioception": proprioception,
            "task_description": task_text,
            "start_t": torch.tensor(virtual_t, dtype=torch.long),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "camera_keys": list(self.camera_keys),
            "view_max_views": self.view_max_views,
            "dataset_name": self.dataset_name,
            "action_stats_key": self.action_stats_key,
            "action_output_frame": self.action_frame,
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "episode_id": f"{Path(hdf5_path).stem}:{demo_key}",
            "hdf5_path": hdf5_path,
            "demo_key": demo_key,
            "has_action": True,
        }
        if all_view_target_images_t is not None:
            sample["all_view_target_images"] = all_view_target_images_t
        if all_view_target_mask_t is not None:
            sample["all_view_target_mask"] = all_view_target_mask_t
        if use_uniform_mask:
            sample["action_loss_mask"] = action_loss_mask
            sample["past_action_history"] = past_action_history
            sample["past_action_history_mask"] = past_action_history_mask
            sample["transition_loss_mask"] = self._uniform_transition_loss_mask(
                start_t=t,
                n_kept=n_kept,
                visual_stride=visual_stride,
                valid_raw_count=int(valid_raw_count or 0),
            )
            sample["context_valid_mask"] = context_valid_mask
        sample.update(
            self._load_gt_depth_targets(
                hdf5_path=hdf5_path,
                demo_key=demo_key,
                episode_index=int(episode_index),
                frame_indices=frame_indices,
                frame_valid_mask=context_valid_mask,
                cam_crop_params=cam_crop_params,
                cam_rotation_angles=cam_rotation_angles,
                cam_source_sizes=cam_source_sizes,
                demo=demo,
            )
        )
        return sample

    def _iter_stat_starts(self, max_samples: Optional[int]):
        stat_samples = self._stat_samples if self._stat_samples else self.samples
        spans = np.asarray([sample[2] + 1 for sample in stat_samples], dtype=np.int64)
        total = int(spans.sum())
        if total <= 0:
            return
        rng = np.random.default_rng(42)
        if max_samples is None or max_samples <= 0 or max_samples >= total:
            for sample_idx, span in enumerate(spans.tolist()):
                for start in range(int(span)):
                    yield sample_idx, start, self._sample_stride(rng)
            return
        cum_spans = np.cumsum(spans)
        offsets = rng.integers(0, total, size=int(max_samples))
        for offset in offsets:
            sample_idx = int(np.searchsorted(cum_spans, int(offset), side="right"))
            prev = int(cum_spans[sample_idx - 1]) if sample_idx > 0 else 0
            yield sample_idx, int(offset) - prev, self._sample_stride(rng)

    def compute_action_statistics(self, max_samples: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        rows: list[torch.Tensor] = []
        stat_samples = self._stat_samples if self._stat_samples else self.samples
        if self._uniform_action_sampling:
            # Uniform sampler supervision gives every kept raw action row the
            # same loss count. Statistics must therefore be computed from the
            # kept rows themselves rather than legacy full-window starts that
            # heavily under-sample episode boundaries.
            for hdf5_path, demo_key, _, _, _, n_kept in stat_samples:
                demo = self._get_hdf5(hdf5_path)["data"][demo_key]
                keep_arr = self._keep_indices_by_demo.get((hdf5_path, demo_key))
                if keep_arr is not None:
                    raw_np = np.asarray(demo["actions"][keep_arr[: int(n_kept)].tolist()])
                else:
                    raw_np = np.asarray(demo["actions"][: int(n_kept)])
                if raw_np.size == 0:
                    continue
                action_indices = (
                    keep_arr[: int(n_kept)].tolist()
                    if keep_arr is not None
                    else list(range(int(n_kept)))
                )
                action = self._maybe_convert_action_frame(
                    self._canonicalize_action(torch.as_tensor(raw_np, dtype=torch.float32)),
                    self._read_proprio(demo, action_indices),
                    self._read_target_proprio_for_kept_positions(
                        demo,
                        hdf5_path,
                        demo_key,
                        (np.arange(int(n_kept), dtype=np.int64) + 1).tolist(),
                        int(n_kept),
                    ),
                )
                rows.append(action.reshape(-1, action.shape[-1]))
        else:
            for sample_idx, start, stride in self._iter_stat_starts(max_samples) or []:
                hdf5_path, demo_key, _, _, _, n_kept = stat_samples[sample_idx]
                demo = self._get_hdf5(hdf5_path)["data"][demo_key]
                n_raw = self.action_steps * max(1, int(self.chunk_size)) * max(1, int(stride))
                keep_arr = self._keep_indices_by_demo.get((hdf5_path, demo_key))
                if keep_arr is not None:
                    action_orig_idx = keep_arr[start : start + n_raw]
                    action_indices = action_orig_idx.tolist()
                    raw_np = np.asarray(demo["actions"][action_orig_idx.tolist()])
                else:
                    action_indices = list(range(start, start + n_raw))
                    raw_np = np.asarray(demo["actions"][action_indices])
                raw = torch.as_tensor(raw_np, dtype=torch.float32)
                action_proprio = self._read_proprio(demo, action_indices)
                action = self._resample_with_stride(self._canonicalize_action(raw), stride)
                target_positions = int(start) + (np.arange(action.shape[0], dtype=np.int64) + 1) * max(1, int(stride))
                target_proprio = self._read_target_proprio_for_kept_positions(
                    demo,
                    hdf5_path,
                    demo_key,
                    target_positions.tolist(),
                    int(n_kept),
                )
                action = self._maybe_convert_action_frame(
                    action,
                    _select_action_anchor_proprio(action_proprio, action.shape[0], stride),
                    target_proprio,
                )
                rows.append(action.reshape(-1, action.shape[-1]))
        if not rows:
            raise ValueError("No valid LIBERO HDF5 action-stat windows found.")
        actions = torch.cat(rows, dim=0).cpu().numpy()
        return {
            self.action_stats_key: {
                "mean": actions.mean(axis=0),
                "std": actions.std(axis=0),
                "min": actions.min(axis=0),
                "max": actions.max(axis=0),
                "q01": np.percentile(actions, 1, axis=0),
                "q99": np.percentile(actions, 99, axis=0),
                "mask": DEFAULT_ACTION_NORM_MASK.copy(),
            }
        }

    def compute_proprio_statistics(self, max_samples: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        rows: list[torch.Tensor] = []
        stat_samples = self._stat_samples if self._stat_samples else self.samples
        if self._uniform_action_sampling:
            for hdf5_path, demo_key, _, _, _, n_kept in stat_samples:
                demo = self._get_hdf5(hdf5_path)["data"][demo_key]
                keep_arr = self._keep_indices_by_demo.get((hdf5_path, demo_key))
                if keep_arr is not None:
                    frame_indices = [int(i) for i in keep_arr[: int(n_kept)].tolist()]
                else:
                    frame_indices = list(range(int(n_kept)))
                if not frame_indices:
                    continue
                rows.append(self._read_proprio(demo, frame_indices))
        else:
            for sample_idx, start, _ in self._iter_stat_starts(max_samples) or []:
                hdf5_path, demo_key, _, _, _, _ = stat_samples[sample_idx]
                demo = self._get_hdf5(hdf5_path)["data"][demo_key]
                keep_arr = self._keep_indices_by_demo.get((hdf5_path, demo_key))
                orig_idx = int(keep_arr[start]) if keep_arr is not None else int(start)
                rows.append(self._read_proprio(demo, [orig_idx])[0])
        if not rows:
            raise ValueError("No valid LIBERO HDF5 proprio-stat windows found.")
        proprio = torch.cat([row.reshape(-1, row.shape[-1]) for row in rows], dim=0).cpu().numpy()
        return {
            self.action_stats_key: {
                "mean": proprio.mean(axis=0),
                "std": proprio.std(axis=0),
                "min": proprio.min(axis=0),
                "max": proprio.max(axis=0),
                "q01": np.percentile(proprio, 1, axis=0),
                "q99": np.percentile(proprio, 99, axis=0),
                "mask": np.array([True] * proprio.shape[-1], dtype=bool),
            }
        }

    def __del__(self):
        for f in getattr(self, "_hdf5_cache", {}).values():
            try:
                f.close()
            except Exception:
                pass


def _is_auto_view_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "infer", "from_camera_keys"}


def _pad_views_to_max_enabled(dataset_cfg: Mapping[str, Any]) -> bool:
    mode = str(dataset_cfg.get("view_mode", dataset_cfg.get("view_policy", "")) or "").strip().lower()
    mode_enabled = mode in {"pad_to_max", "padded", "pad", "variable", "variable_pad"}
    return bool(dataset_cfg.get("pad_views_to_max", mode_enabled))


def _configured_view_max(dataset_cfg: Mapping[str, Any]) -> Optional[int]:
    raw = dataset_cfg.get("max_views", dataset_cfg.get("view_max_views"))
    if raw is None or _is_auto_view_value(raw):
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"dataset.max_views must be positive, got {raw!r}.")
    return value


def _selection_n_views(dataset_cfg: Mapping[str, Any], default: int) -> int:
    view_max = _configured_view_max(dataset_cfg)
    if _pad_views_to_max_enabled(dataset_cfg) and view_max is not None:
        return view_max
    raw = dataset_cfg.get("n_views", default)
    if _is_auto_view_value(raw):
        return view_max if view_max is not None else int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"dataset.n_views must be positive, got {raw!r}.")
    return value


def _repeat_missing_views(dataset_cfg: Mapping[str, Any]) -> bool:
    if "repeat_missing_views" in dataset_cfg:
        return bool(dataset_cfg["repeat_missing_views"])
    return not _pad_views_to_max_enabled(dataset_cfg)


def build_robot_dataset(dataset_cfg: Dict[str, Any], is_eval: bool = False) -> Dataset:
    dataset_type = str(dataset_cfg.get("type", "mimicgen")).lower()
    pad_views_to_max = _pad_views_to_max_enabled(dataset_cfg)
    view_max_views = _configured_view_max(dataset_cfg) if pad_views_to_max else None
    repeat_missing_views = _repeat_missing_views(dataset_cfg)
    if dataset_type == "mimicgen":
        raise NotImplementedError(
            "dataset.type='mimicgen' is outside this public LIBERO release."
        )

    if dataset_type in {"ssv2", "something_something_v2", "video_manifest"}:
        raise NotImplementedError(
            "dataset.type='video' is outside this public LIBERO release."
        )

    if dataset_type == "mixer":
        raise NotImplementedError(
            "dataset.type='mixer' is outside this public LIBERO release."
        )

    if dataset_type == "robocasa":
        raise NotImplementedError(
            "dataset.type='robocasa' is outside this public LIBERO release."
        )

    if dataset_type in {"robocasa_cosmos_hdf5", "robocasa_hdf5_cosmos"}:
        raise NotImplementedError(
            "dataset.type='robocasa_cosmos' is outside this public LIBERO release."
        )

    if dataset_type in {"rlbench", "rlbench_peract", "rlbench_peract18"}:
        raise NotImplementedError(
            "dataset.type='rlbench' is outside this public LIBERO release."
        )

    if dataset_type in {"libero_hdf5", "hdf5_libero"}:
        default_hdf5_root = os.environ.get(
            "DA3_LIBERO_HDF5_ROOT", "data/libero_hdf5/LIBERO-datasets"
        )
        hdf5_root = dataset_cfg.get("hdf5_root", default_hdf5_root)
        dataset_name = dataset_cfg.get("dataset_name", "libero_hdf5")
        image_aug = resolve_image_augmentation_config(dataset_cfg)
        action_frame = normalize_action_frame(
            dataset_cfg.get("action_frame", dataset_cfg.get("action_output_frame", "base"))
        )
        proprio_orientation = infer_libero_hdf5_proprio_orientation(
            dataset_cfg.get("proprio_orientation", dataset_cfg.get("action_frame_proprio_orientation")),
            dataset_name,
            hdf5_root,
        )
        return LiberoHDF5SequenceDataset(
            hdf5_root=hdf5_root,
            hdf5_paths=dataset_cfg.get("hdf5_paths"),
            pattern=dataset_cfg.get("pattern", "*.hdf5"),
            suites=dataset_cfg.get("suites"),
            tasks=dataset_cfg.get("tasks"),
            image_size=tuple(dataset_cfg.get("image_size", [224, 224])),
            future_steps=int(dataset_cfg.get("future_steps", 6)),
            chunk_size=int(dataset_cfg.get("chunk_size", 1)),
            include_current_action=bool(dataset_cfg.get("include_current_action", False)),
            n_views=_selection_n_views(dataset_cfg, 2),
            proprio_dim=int(dataset_cfg.get("proprio_dim", 7)),
            eval_ratio=float(dataset_cfg.get("eval_ratio", 0.05)),
            is_eval=is_eval,
            source_fps=float(dataset_cfg.get("source_fps", 20.0)),
            target_hz=dataset_cfg.get("target_hz"),
            random_stride=bool(dataset_cfg.get("random_stride", False)),
            camera_keys=dataset_cfg.get("camera_keys"),
            train_crop_min_scale=float(image_aug["train_crop_min_scale"]),
            eval_crop_scale=float(image_aug["eval_crop_scale"]),
            color_jitter_brightness=float(image_aug["color_jitter_brightness"]),
            color_jitter_contrast=float(image_aug["color_jitter_contrast"]),
            color_jitter_saturation=float(image_aug["color_jitter_saturation"]),
            color_jitter_hue=float(image_aug["color_jitter_hue"]),
            openpi_libero_augment=bool(image_aug["openpi_libero_augment"]),
            openpi_base_crop_scale=float(image_aug["openpi_base_crop_scale"]),
            openpi_base_rotate_degrees=float(image_aug["openpi_base_rotate_degrees"]),
            image_aug_profile=str(image_aug["image_aug_profile"]),
            image_jpeg_train_enabled=bool(image_aug["image_jpeg_train_enabled"]),
            image_jpeg_eval_enabled=bool(image_aug["image_jpeg_eval_enabled"]),
            image_jpeg_train_quality=int(image_aug["image_jpeg_train_quality"]),
            image_jpeg_eval_quality=int(image_aug["image_jpeg_eval_quality"]),
            dataset_name=dataset_name,
            hdf5_mask_key=dataset_cfg.get("hdf5_mask_key", dataset_cfg.get("demo_mask_key")),
            hdf5_mask_required=bool(dataset_cfg.get("hdf5_mask_required", False)),
            hdf5_mask_limit=dataset_cfg.get("hdf5_mask_limit", dataset_cfg.get("demo_mask_limit")),
            da3_input_rotate180=bool(dataset_cfg.get("da3_input_rotate180", True)),
            da3_input_hflip=infer_libero_hdf5_hflip(
                dataset_cfg.get("da3_input_hflip"),
                dataset_name,
                hdf5_root,
            ),
            da3_input_vflip=bool(dataset_cfg.get("da3_input_vflip", False)),
            gt_depth_root=dataset_cfg.get("gt_depth_root"),
            gt_depth_key=str(dataset_cfg.get("gt_depth_key", "depth_meters")),
            gt_depth_rotate180=bool(dataset_cfg.get("gt_depth_rotate180", True)),
            gt_depth_hflip=infer_libero_hdf5_hflip(
                dataset_cfg.get("gt_depth_hflip"),
                dataset_name,
                hdf5_root,
            ),
            gt_depth_vflip=bool(dataset_cfg.get("gt_depth_vflip", False)),
            gt_depth_require_geometry=bool(dataset_cfg.get("gt_depth_require_geometry", False)),
            gt_depth_scale_mode=str(dataset_cfg.get("gt_depth_scale_mode", "pointmap")),
            gt_depth_min_meters=float(dataset_cfg.get("gt_depth_min_meters", 1e-3)),
            gt_depth_require_sidecar_file=bool(dataset_cfg.get("gt_depth_require_sidecar_file", False)),
            filter_noops=bool(dataset_cfg.get("filter_noops", False)),
            noop_threshold=float(dataset_cfg.get("noop_threshold", 1.0e-4)),
            uniform_action_sampling=bool(
                dataset_cfg.get(
                    "uniform_action_sampling",
                    str(dataset_cfg.get("action_sampling_mode", "valid_window")).lower()
                    in {"virtual", "virtual_mask", "virtual_action_mask"},
                )
            ),
            action_frame=action_frame,
            proprio_orientation=proprio_orientation,
            repeat_missing_views=repeat_missing_views,
            view_max_views=view_max_views,
        )

    if dataset_type in {"openx", "oxe", "lerobot_openx", "libero"}:
        raise NotImplementedError(
            "dataset.type='openx'/'libero' (offline LeRobot) is outside "
            "this public release; train with type='libero_hdf5'. LIBERO-Plus is "
            "an eval-only benchmark (use eval_libero_unified.py --plus, simulator "
            "rollout)."
        )

    raise ValueError(f"Unsupported dataset.type={dataset_type}")


def compute_action_statistics(dataset: Dataset, max_samples: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
    if hasattr(dataset, "compute_action_statistics"):
        return dataset.compute_action_statistics(max_samples=max_samples)
    raise TypeError(f"Dataset {type(dataset)!r} lacks compute_action_statistics().")


def compute_proprio_statistics(dataset: Dataset, max_samples: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
    if hasattr(dataset, "compute_proprio_statistics"):
        return dataset.compute_proprio_statistics(max_samples=max_samples)
    raise TypeError(f"Dataset {type(dataset)!r} lacks compute_proprio_statistics().")


def summarize_action_statistics(stats_by_key: Dict[str, Dict[str, np.ndarray]]) -> str:
    parts = []
    for key in sorted(stats_by_key):
        stats = stats_by_key[key]
        parts.append(
            f"{key}: mean={np.round(stats['mean'], 4).tolist()} std={np.round(stats['std'], 4).tolist()}"
        )
    return " | ".join(parts)


class ActionNormalizer:
    """Per-dataset action normalization. Two modes:

    - ``q01_q99`` (default): ``2*(x - q01)/(q99 - q01) - 1``, unclamped.
    - ``mean_std``: ``(x - mean) / (std + eps)``, no clamp.

    Both modes can be backed by the same cached stats files since
    ``compute_action_statistics`` always emits ``q01``/``q99``/``mean``/``std``.
    Constant dims (degenerate range/std) are masked out and pass through
    unchanged, so a frozen normalizer never injects NaNs on flat trajectories.
    """

    CONSTANT_DIM_THRESHOLD = 1e-6  # dims with q99-q01 (or std) below this are treated as constant
    SUPPORTED_NORM_MODES = ("q01_q99", "mean_std")
    FULL_7D_REQUIRED_KEYS = {"robocasa_cosmos24_7d"}

    @staticmethod
    def _timing_signature_matches(actual: Optional[str], expected: Optional[str]) -> bool:
        if expected is None or actual is None:
            return True
        if actual == expected:
            return True
        # Several RoboCasa365 mini-repos share one action_stats_key.  The
        # offline stats job stores their combined fixed-stride policy-action
        # distribution as grouped_openx, which is compatible with fixed leaf
        # datasets; random-stride timing requires matching stats.
        if actual == "grouped_openx" and str(expected).startswith("fixed:"):
            return True
        return False

    def __init__(
        self,
        stats_by_key: Dict[str, Dict[str, np.ndarray | torch.Tensor]],
        eps: float = 1e-8,
        default_key: str = DEFAULT_ACTION_STATS_KEY,
        norm_mode: str = "q01_q99",
    ):
        self.eps = eps
        norm_mode = str(norm_mode).strip().lower()
        if norm_mode not in self.SUPPORTED_NORM_MODES:
            raise ValueError(
                f"ActionNormalizer.norm_mode={norm_mode!r} not in {self.SUPPORTED_NORM_MODES}."
            )
        self.norm_mode = norm_mode
        self.stats_by_key: dict[str, dict[str, torch.Tensor]] = {}
        for key, stats in stats_by_key.items():
            mask = stats.get("mask", DEFAULT_ACTION_NORM_MASK)
            q01 = torch.as_tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.as_tensor(stats["q99"], dtype=torch.float32)
            mask_t = torch.as_tensor(mask, dtype=torch.bool)
            mean_t: Optional[torch.Tensor] = None
            std_t: Optional[torch.Tensor] = None
            if "mean" in stats and "std" in stats:
                mean_t = torch.as_tensor(stats["mean"], dtype=torch.float32)
                std_t = torch.as_tensor(stats["std"], dtype=torch.float32)
            if norm_mode == "mean_std":
                if mean_t is None or std_t is None:
                    raise KeyError(
                        f"ActionNormalizer(norm_mode='mean_std') requires 'mean'/'std' "
                        f"in stats for key={key!r}, but they are missing. Re-run with "
                        "--refresh-action-stats so compute_action_statistics fills them in."
                    )
                # Disable normalization on constant dims (std ≈ 0).
                constant_dims = std_t.abs() < self.CONSTANT_DIM_THRESHOLD
            else:
                # q01_q99 path : disable on degenerate range.
                constant_dims = (q99 - q01).abs() < self.CONSTANT_DIM_THRESHOLD
            if str(key) in self.FULL_7D_REQUIRED_KEYS:
                disabled_dims = ~mask_t
                invalid_dims = constant_dims | disabled_dims
                if bool(invalid_dims.any()):
                    bad = torch.nonzero(invalid_dims, as_tuple=False).flatten().tolist()
                    disabled = torch.nonzero(disabled_dims, as_tuple=False).flatten().tolist()
                    degenerate = torch.nonzero(constant_dims, as_tuple=False).flatten().tolist()
                    details = []
                    if disabled:
                        details.append(f"disabled mask dims={disabled}")
                    if degenerate:
                        details.append(f"degenerate dims={degenerate}")
                    detail_text = "; ".join(details)
                    if detail_text:
                        detail_text = f" ({detail_text})"
                    raise ValueError(
                        f"Action stats for {key!r} leave gaps in the full 7D Cosmos action{detail_text}. "
                        "This stats file/checkpoint normalizer is incompatible with the official "
                        "RoboCasa Cosmos 7D action protocol. Recompute stats with "
                        "--refresh-action-stats so compute_action_statistics writes a full 7D mask."
                    )
            mask_t = mask_t & ~constant_dims
            entry: dict[str, torch.Tensor] = {
                "q01": q01,
                "q99": q99,
                "scale": q99 - q01 + eps,
                "mask": mask_t,
            }
            if mean_t is not None:
                entry["mean"] = mean_t
                entry["std"] = std_t
            self.stats_by_key[key] = entry
        self.default_key = default_key if default_key in self.stats_by_key else next(iter(self.stats_by_key))

    def _normalize_single(self, actions: torch.Tensor, stats_key: str) -> torch.Tensor:
        stats = self.stats_by_key[stats_key]
        mask = stats["mask"].to(actions.device)
        if self.norm_mode == "mean_std":
            mean = stats["mean"].to(actions.device, actions.dtype)
            std = stats["std"].to(actions.device, actions.dtype)
            normalized = (actions - mean) / (std + self.eps)
            return torch.where(mask.view(*([1] * (actions.ndim - 1)), -1), normalized, actions)
        q01 = stats["q01"].to(actions.device, actions.dtype)
        scale = stats["scale"].to(actions.device, actions.dtype)
        normalized = 2.0 * (actions - q01) / scale - 1.0
        return torch.where(mask.view(*([1] * (actions.ndim - 1)), -1), normalized, actions)

    def _denormalize_single(self, actions: torch.Tensor, stats_key: str) -> torch.Tensor:
        stats = self.stats_by_key[stats_key]
        mask = stats["mask"].to(actions.device)
        if self.norm_mode == "mean_std":
            mean = stats["mean"].to(actions.device, actions.dtype)
            std = stats["std"].to(actions.device, actions.dtype)
            denorm = actions * (std + self.eps) + mean
            return torch.where(mask.view(*([1] * (actions.ndim - 1)), -1), denorm, actions)
        q01 = stats["q01"].to(actions.device, actions.dtype)
        scale = stats["scale"].to(actions.device, actions.dtype)
        denorm = (actions + 1.0) / 2.0 * scale + q01
        return torch.where(mask.view(*([1] * (actions.ndim - 1)), -1), denorm, actions)

    def normalize(
        self,
        actions: torch.Tensor,
        stats_key: Optional[str] = None,
        stats_keys: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if stats_keys is None:
            return self._normalize_single(actions, stats_key or self.default_key)
        if actions.shape[0] != len(stats_keys):
            raise ValueError("Batch size and stats_keys length must match.")
        output = actions.clone()
        for key in sorted(set(stats_keys)):
            batch_idx = [i for i, current in enumerate(stats_keys) if current == key]
            output[batch_idx] = self._normalize_single(actions[batch_idx], key)
        return output

    def denormalize(
        self,
        actions: torch.Tensor,
        stats_key: Optional[str] = None,
        stats_keys: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if stats_keys is None:
            return self._denormalize_single(actions, stats_key or self.default_key)
        if actions.shape[0] != len(stats_keys):
            raise ValueError("Batch size and stats_keys length must match.")
        output = actions.clone()
        for key in sorted(set(stats_keys)):
            batch_idx = [i for i, current in enumerate(stats_keys) if current == key]
            output[batch_idx] = self._denormalize_single(actions[batch_idx], key)
        return output

    def state_dict(self):
        per_key: dict[str, dict] = {}
        for key, stats in self.stats_by_key.items():
            entry = {
                "q01": stats["q01"],
                "q99": stats["q99"],
                "mask": stats["mask"],
            }
            if "mean" in stats:
                entry["mean"] = stats["mean"]
                entry["std"] = stats["std"]
            per_key[key] = entry
        return {
            "format": "per_dataset_action_norm_v2",
            "norm_mode": self.norm_mode,
            "eps": self.eps,
            "default_key": self.default_key,
            "stats_by_key": per_key,
        }

    @classmethod
    def from_state_dict(cls, state, override_norm_mode: Optional[str] = None):
        ckpt_mode = state.get("norm_mode", "q01_q99")
        runtime_mode = override_norm_mode or ckpt_mode
        if runtime_mode != ckpt_mode:
            raise ValueError(
                f"ActionNormalizer state_dict was saved with norm_mode={ckpt_mode!r} "
                f"but runtime config requests norm_mode={runtime_mode!r}. "
                "Resume with --refresh-action-stats to recompute under the new mode "
                "instead of mixing them."
            )
        if "stats_by_key" in state:
            return cls(
                stats_by_key=state["stats_by_key"],
                eps=state.get("eps", 1e-8),
                default_key=state.get("default_key", DEFAULT_ACTION_STATS_KEY),
                norm_mode=runtime_mode,
            )
        if "q01" in state and "q99" in state:
            return cls(
                stats_by_key={
                    DEFAULT_ACTION_STATS_KEY: {
                        "q01": state["q01"],
                        "q99": state["q99"],
                        "mask": DEFAULT_ACTION_NORM_MASK,
                    }
                },
                eps=state.get("eps", 1e-8),
                default_key=DEFAULT_ACTION_STATS_KEY,
                norm_mode=runtime_mode,
            )
        raise ValueError("Unsupported action normalizer checkpoint format.")

    @classmethod
    def from_dataset(cls, dataset: Dataset, max_samples: int = -1, norm_mode: str = "q01_q99"):
        stats = compute_action_statistics(dataset, max_samples=max_samples)
        default_key = next(iter(stats))
        return cls(stats_by_key=stats, default_key=default_key, norm_mode=norm_mode)

    @classmethod
    def from_stats_dir(
        cls,
        stats_dir: str,
        dataset_names: Sequence[str],
        target_hz: Optional[float] = None,
        target_hz_map: Optional[Dict[str, float]] = None,
        expected_timing_signatures: Optional[Mapping[str, str]] = None,
        require_timing_signature: bool = False,
        norm_mode: str = "q01_q99",
    ):
        """Load pre-computed stats from JSON files.

        Per-dataset hz resolution:
          1. if `target_hz_map[name]` is set → use it
          2. elif `target_hz` (global) is set → use it
          3. else → load `{name}.json` (native/fixed-stride default)

        Lookup order per name: `{name}_hz{hz}.json` → `{name}.json` fallback.
        Legacy files without timing metadata are accepted for fixed-stride
        stats, but random-stride training must use stats explicitly saved for
        the same sampled-stride distribution.
        """
        import json as _json
        stats_path = Path(stats_dir)
        stats_by_key: dict[str, dict] = {}
        rejected: list[str] = []
        for name in dataset_names:
            hz = None
            if target_hz_map is not None and name in target_hz_map and target_hz_map[name]:
                hz = target_hz_map[name]
            elif target_hz is not None:
                hz = target_hz
            candidates: list[Path] = []
            if hz is not None:
                candidates.append(stats_path / f"{name}_hz{int(hz)}.json")
            candidates.append(stats_path / f"{name}.json")
            expected_sig = (
                expected_timing_signatures.get(name)
                if expected_timing_signatures is not None else None
            )
            for fpath in candidates:
                if not fpath.exists():
                    continue
                d = _json.loads(fpath.read_text())
                actual_sig = d.get("timing_signature")
                if expected_sig is not None:
                    if actual_sig is None and (require_timing_signature or expected_sig.startswith("random")):
                        rejected.append(f"{name}: {fpath.name} has no timing_signature")
                        continue
                    if actual_sig is not None and not cls._timing_signature_matches(actual_sig, expected_sig):
                        rejected.append(
                            f"{name}: {fpath.name} timing_signature={actual_sig!r} "
                            f"expected={expected_sig!r}"
                        )
                        continue
                stats_by_key[name] = d
                break
        if expected_timing_signatures is not None:
            missing = [name for name in dataset_names if name not in stats_by_key]
            if missing:
                detail = f"; rejected: {'; '.join(rejected[:8])}" if rejected else ""
                raise ValueError(
                    f"No compatible stats files found in {stats_dir} for {missing}{detail}"
                )
        if not stats_by_key:
            detail = f"; rejected: {'; '.join(rejected[:8])}" if rejected else ""
            raise ValueError(f"No compatible stats files found in {stats_dir} for {dataset_names}{detail}")
        return cls(
            stats_by_key=stats_by_key,
            default_key=next(iter(stats_by_key)),
            norm_mode=norm_mode,
        )

    def save_to_stats_dir(
        self,
        stats_dir: str,
        target_hz: Optional[float] = None,
        timing_signatures: Optional[Mapping[str, str]] = None,
    ) -> None:
        stats_path = Path(stats_dir)
        stats_path.mkdir(parents=True, exist_ok=True)
        suffix = f"_hz{int(target_hz)}" if target_hz is not None else ""
        for key, stats in self.state_dict()["stats_by_key"].items():
            payload = {
                "q01": torch.as_tensor(stats["q01"]).tolist(),
                "q99": torch.as_tensor(stats["q99"]).tolist(),
                "mask": torch.as_tensor(stats["mask"]).tolist(),
                "normalizer_unit": "policy_action",
            }
            if "mean" in stats and "std" in stats:
                payload["mean"] = torch.as_tensor(stats["mean"]).tolist()
                payload["std"] = torch.as_tensor(stats["std"]).tolist()
            if timing_signatures is not None and key in timing_signatures:
                payload["timing_signature"] = timing_signatures[key]
            # Save hz-specific AND default (no suffix)
            (stats_path / f"{key}{suffix}.json").write_text(json.dumps(payload))
            if suffix:
                (stats_path / f"{key}.json").write_text(json.dumps(payload))


class StateNormalizer:
    """Per-dataset proprio/state normalization. Two modes (mirrors ActionNormalizer):

    - ``q01_q99`` (default): ``2*(x - q01)/(q99 - q01) - 1``, unclamped.
    - ``mean_std``: ``(x - mean) / (std + eps)``, no clamp.
    """

    SUPPORTED_NORM_MODES = ("q01_q99", "mean_std")
    CONSTANT_DIM_THRESHOLD = 1e-6

    def __init__(
        self,
        stats_by_key: Dict[str, Dict[str, np.ndarray | torch.Tensor]],
        eps: float = 1e-8,
        default_key: str = DEFAULT_ACTION_STATS_KEY,
        norm_mode: str = "q01_q99",
    ):
        self.eps = eps
        norm_mode = str(norm_mode).strip().lower()
        if norm_mode not in self.SUPPORTED_NORM_MODES:
            raise ValueError(
                f"StateNormalizer.norm_mode={norm_mode!r} not in {self.SUPPORTED_NORM_MODES}."
            )
        self.norm_mode = norm_mode
        self.stats_by_key: dict[str, dict[str, torch.Tensor]] = {}
        for key, stats in stats_by_key.items():
            q01 = torch.as_tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.as_tensor(stats["q99"], dtype=torch.float32)
            raw_mask = stats.get("mask", np.array([True] * int(q01.shape[-1]), dtype=bool))
            mask_t = torch.as_tensor(raw_mask, dtype=torch.bool)
            mean_t: Optional[torch.Tensor] = None
            std_t: Optional[torch.Tensor] = None
            if "mean" in stats and "std" in stats:
                mean_t = torch.as_tensor(stats["mean"], dtype=torch.float32)
                std_t = torch.as_tensor(stats["std"], dtype=torch.float32)
            if norm_mode == "mean_std":
                if mean_t is None or std_t is None:
                    raise KeyError(
                        f"StateNormalizer(norm_mode='mean_std') requires 'mean'/'std' "
                        f"in stats for key={key!r}, but they are missing. Re-run with "
                        "--refresh-action-stats so compute_proprio_statistics fills them in."
                    )
                constant_dims = std_t.abs() < self.CONSTANT_DIM_THRESHOLD
            else:
                constant_dims = (q99 - q01).abs() < self.CONSTANT_DIM_THRESHOLD
            mask_t = mask_t & ~constant_dims
            entry: dict[str, torch.Tensor] = {
                "q01": q01,
                "q99": q99,
                "scale": q99 - q01 + eps,
                "mask": mask_t,
            }
            if mean_t is not None:
                entry["mean"] = mean_t
                entry["std"] = std_t
            self.stats_by_key[key] = entry
        self.default_key = default_key if default_key in self.stats_by_key else next(iter(self.stats_by_key))

    def _normalize_single(self, proprio: torch.Tensor, stats_key: str) -> torch.Tensor:
        stats = self.stats_by_key[stats_key]
        mask = stats["mask"].to(proprio.device)
        if self.norm_mode == "mean_std":
            mean = stats["mean"].to(proprio.device, proprio.dtype)
            std = stats["std"].to(proprio.device, proprio.dtype)
            normalized = (proprio - mean) / (std + self.eps)
            return torch.where(mask.view(*([1] * (proprio.ndim - 1)), -1), normalized, proprio)
        q01 = stats["q01"].to(proprio.device, proprio.dtype)
        scale = stats["scale"].to(proprio.device, proprio.dtype)
        normalized = 2.0 * (proprio - q01) / scale - 1.0
        return torch.where(mask.view(*([1] * (proprio.ndim - 1)), -1), normalized, proprio)

    def normalize(
        self,
        proprio: torch.Tensor,
        stats_key: Optional[str] = None,
        stats_keys: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if stats_keys is None:
            return self._normalize_single(proprio, stats_key or self.default_key)
        if proprio.shape[0] != len(stats_keys):
            raise ValueError("Batch size and stats_keys length must match.")
        output = proprio.clone()
        for key in sorted(set(stats_keys)):
            batch_idx = [i for i, current in enumerate(stats_keys) if current == key]
            output[batch_idx] = self._normalize_single(proprio[batch_idx], key)
        return output

    def state_dict(self):
        per_key: dict[str, dict] = {}
        for key, stats in self.stats_by_key.items():
            entry = {
                "q01": stats["q01"],
                "q99": stats["q99"],
                "mask": stats["mask"],
            }
            if "mean" in stats:
                entry["mean"] = stats["mean"]
                entry["std"] = stats["std"]
            per_key[key] = entry
        return {
            "format": "per_dataset_state_norm_v2",
            "norm_mode": self.norm_mode,
            "eps": self.eps,
            "default_key": self.default_key,
            "stats_by_key": per_key,
        }

    @classmethod
    def from_state_dict(cls, state, override_norm_mode: Optional[str] = None):
        if "stats_by_key" not in state:
            raise ValueError("Unsupported proprio/state normalizer checkpoint format.")
        ckpt_mode = state.get("norm_mode", "q01_q99")
        runtime_mode = override_norm_mode or ckpt_mode
        if runtime_mode != ckpt_mode:
            raise ValueError(
                f"StateNormalizer state_dict was saved with norm_mode={ckpt_mode!r} "
                f"but runtime config requests norm_mode={runtime_mode!r}. "
                "Resume with --refresh-action-stats to recompute under the new mode."
            )
        return cls(
            stats_by_key=state["stats_by_key"],
            eps=state.get("eps", 1e-8),
            default_key=state.get("default_key", DEFAULT_ACTION_STATS_KEY),
            norm_mode=runtime_mode,
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset, max_samples: int = -1, norm_mode: str = "q01_q99"):
        stats = compute_proprio_statistics(dataset, max_samples=max_samples)
        default_key = next(iter(stats))
        return cls(stats_by_key=stats, default_key=default_key, norm_mode=norm_mode)

    @classmethod
    def from_stats_dir(cls, stats_dir: str, dataset_names: Sequence[str], norm_mode: str = "q01_q99"):
        """Load pre-computed proprio stats from JSON files (named {dataset}_proprio.json)."""
        import json as _json
        stats_path = Path(stats_dir)
        stats_by_key: dict[str, dict] = {}
        for name in dataset_names:
            fpath = stats_path / f"{name}_proprio.json"
            if not fpath.exists():
                continue
            stats_by_key[name] = _json.loads(fpath.read_text())
        if not stats_by_key:
            raise ValueError(f"No proprio stats files found in {stats_dir} for {dataset_names}")
        return cls(
            stats_by_key=stats_by_key,
            default_key=next(iter(stats_by_key)),
            norm_mode=norm_mode,
        )

    def save_to_stats_dir(self, stats_dir: str) -> None:
        stats_path = Path(stats_dir)
        stats_path.mkdir(parents=True, exist_ok=True)
        for key, stats in self.state_dict()["stats_by_key"].items():
            payload = {
                "q01": torch.as_tensor(stats["q01"]).tolist(),
                "q99": torch.as_tensor(stats["q99"]).tolist(),
                "mask": torch.as_tensor(stats["mask"]).tolist(),
            }
            if "mean" in stats and "std" in stats:
                payload["mean"] = torch.as_tensor(stats["mean"]).tolist()
                payload["std"] = torch.as_tensor(stats["std"]).tolist()
            (stats_path / f"{key}_proprio.json").write_text(json.dumps(payload))
