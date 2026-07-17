"""GAM paper pretraining datasets.

This module contains the data path used by the paper release: 23
Open X-Embodiment LeRobot datasets, MimicGen core, and the manipulation subset
of RoboCasa365. Every source returns 224px RGB observations, canonical 7D
proprioception, canonical 7D action chunks, task language, and depth targets
when simulator depth is available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import (
    _crop_resize_depth_and_mask,
    _full_crop_params,
    _is_wrist_camera_key,
    _normalize_image_tensor,
    _quat_xyzw_to_rpy,
    _rotate_depth_and_mask_train,
    _rotation_valid_mask_train,
    _rotmat_to_rotvec,
    _rotvec_to_matrix,
    _rpy_xyz_to_matrix,
    _scene_scale_from_pointmaps,
    _sample_fixed_crop_params,
    _sample_rotation_angle,
    _scale_crop_params_to_source,
    _transform_intrinsics_for_crop_resize,
)


CANONICAL_ACTION = (
    "dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper_close",
)
PAPER_SOURCE_RATIOS = {"open_x_embodiment": 0.72, "mimicgen": 0.18, "robocasa365": 0.10}
MIMICGEN_ACTION_SCALE = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
FULL_ACTION_MASK = (True,) * 7


@dataclass(frozen=True)
class OXESpec:
    name: str
    repo_id: str
    weight: int
    cameras: tuple[str, ...]
    action_transform: str
    state_transform: str
    action_key: str = "action"
    state_keys: tuple[str, ...] = ("observation.state",)
    action_mask: tuple[bool, ...] = FULL_ACTION_MASK
    bgr_cameras: tuple[str, ...] = ()


def _spec(
    name: str,
    repo_id: str,
    weight: int,
    cameras: Sequence[str],
    action_transform: str,
    state_transform: str,
    **kwargs: Any,
) -> OXESpec:
    fields = {
        key: tuple(value) if key in {"state_keys", "action_mask", "bgr_cameras"} else value
        for key, value in kwargs.items()
    }
    return OXESpec(
        name, repo_id, weight, tuple(cameras), action_transform, state_transform, **fields
    )


# OXE weights follow the constituent-dataset proportions used by the paper run.
OXE_SPECS = (
    _spec("bridge", "BrunoM42/bridge_orig_lerobot", 224,
          ("observation.images.image_0", "observation.images.image_1"),
          "state_euler_residual", "pos_euler_8d"),
    _spec("droid", "lerobot/droid_1.0.1", 303,
          ("observation.images.exterior_2_left", "observation.images.wrist_left",
           "observation.images.exterior_1_left"),
          "droid_target", "droid_state", action_key="action.cartesian_position",
          state_keys=("observation.state.cartesian_position",
                      "observation.state.gripper_position")),
    _spec("taco_play", "lerobot/taco_play", 60,
          ("observation.images.rgb_static", "observation.images.rgb_gripper"),
          "taco_world", "pos_euler_7d"),
    _spec("utaustin_mutex", "lerobot/utaustin_mutex", 39,
          ("observation.images.image", "observation.images.wrist_image"),
          "open_to_close", "pos_euler_8d",
          bgr_cameras=("observation.images.image", "observation.images.wrist_image")),
    _spec("stanford_hydra_dataset", "lerobot/stanford_hydra_dataset", 24,
          ("observation.images.image", "observation.images.wrist_image"),
          "euler_xyz_open", "pos_euler_8d",
          bgr_cameras=("observation.images.image", "observation.images.wrist_image")),
    _spec("berkeley_autolab_ur5", "lerobot/berkeley_autolab_ur5", 32,
          ("observation.images.image", "observation.images.hand_image"),
          "base_rpy_open", "pos_quat",
          bgr_cameras=("observation.images.hand_image",)),
    _spec("austin_sailor_dataset", "lerobot/austin_sailor_dataset", 15,
          ("observation.images.image", "observation.images.wrist_image"),
          "open_to_close", "pos_quat",
          action_mask=(True, True, True, False, False, True, True)),
    _spec("austin_sirius_dataset", "lerobot/austin_sirius_dataset", 24,
          ("observation.images.image", "observation.images.wrist_image"),
          "open_to_close", "pos_euler_8d",
          action_mask=(True, True, True, False, False, True, True)),
    _spec("berkeley_fanuc_manipulation", "lerobot/berkeley_fanuc_manipulation", 20,
          ("observation.images.image", "observation.images.wrist_image"),
          "euler_xyz_no_grip", "pos_euler_8d",
          action_mask=(True, True, True, True, True, True, False),
          bgr_cameras=("observation.images.image", "observation.images.wrist_image")),
    _spec("jaco_play", "lerobot/jaco_play", 33,
          ("observation.images.image", "observation.images.image_wrist"),
          "open_to_close", "pos_quat",
          action_mask=(True, True, True, False, False, False, True)),
    _spec("fmb_dataset", "lerobot/fmb", 42,
          ("observation.images.image_side_1", "observation.images.image_wrist_1"),
          "base_rpy_close", "pos_quat",
          bgr_cameras=("observation.images.image_side_1", "observation.images.image_side_2",
                       "observation.images.image_wrist_1", "observation.images.image_wrist_2")),
    _spec("kuka", "lerobot/stanford_kuka_multimodal_dataset", 50,
          ("observation.images.image",), "xyz_only", "pos_quat_no_grip",
          action_mask=(True, True, True, False, False, False, False)),
    _spec("fractal20220817_data", "BrunoM42/fractal20220817_data_lerobot", 271,
          ("observation.images.image",), "base_rpy_open", "pos_quat"),
    _spec("berkeley_cable_routing", "lerobot/berkeley_cable_routing", 8,
          ("observation.images.image", "observation.images.wrist45_image"),
          "velocity_no_grip", "pos_quat",
          action_mask=(True, True, True, False, False, True, False)),
    _spec("roboturk", "lerobot/roboturk", 20,
          ("observation.images.front_rgb",), "open_to_close", "zero"),
    _spec("dlr_edan_shared_control", "lerobot/dlr_edan_shared_control", 1,
          ("observation.images.image",), "euler_zxy_open", "pos_euler_7d"),
    _spec("austin_buds_dataset", "lerobot/austin_buds_dataset", 7,
          ("observation.images.image", "observation.images.wrist_image"),
          "signed_to_close", "austin_buds",
          action_mask=(True, True, True, False, False, False, True)),
    _spec("nyu_franka_play_dataset", "lerobot/nyu_franka_play_dataset", 10,
          ("observation.images.image", "observation.images.image_additional_view"),
          "nyu_franka", "nyu_franka"),
    _spec("nyu_door_opening_surprising_effectiveness",
          "lerobot/nyu_door_opening_surprising_effectiveness", 10,
          ("observation.images.image",), "velocity_open", "zero"),
    _spec("cmu_stretch", "lerobot/cmu_stretch", 5,
          ("observation.images.image",), "drop_last", "cmu_stretch",
          action_mask=(True, False, True, False, False, False, False)),
    _spec("furniture_bench_dataset", "tailong-wu/furniture_bench_dataset_lerobot_v30", 71,
          ("observation.images.image", "observation.images.wrist_image"),
          "furniture_bench", "pos_quat"),
    _spec("bc_z", "tailong-wu/bc_z_lerobot_v30", 208,
          ("observation.images.image",), "axis_angle_residual", "pos_euler_8d"),
    _spec("language_table", "tailong-wu/language_table_lerobot_v30", 100,
          ("observation.images.rgb",), "xy_only", "language_table",
          action_mask=(True, True, False, False, False, False, False)),
)


ROBOCASA_SPEC = _spec(
    "robocasa365", "__local__", 1,
    ("observation.images.robot0_agentview_left",
     "observation.images.robot0_eye_in_hand"),
    "robocasa", "robocasa",
)


def _as_2d(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


def _time_column(value: Any, length: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1 and tensor.numel() == length:
        return tensor.reshape(length, 1)
    return _as_2d(tensor)


def _pad7(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] >= 7:
        return value[..., :7]
    return torch.cat([value, value.new_zeros(*value.shape[:-1], 7 - value.shape[-1])], dim=-1)


def _euler_zxy_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    z, x, y = euler.unbind(-1)
    cz, sz = torch.cos(z), torch.sin(z)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    zeros, ones = torch.zeros_like(z), torch.ones_like(z)
    rz = torch.stack((cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones), -1).reshape(*z.shape, 3, 3)
    rx = torch.stack((ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx), -1).reshape(*z.shape, 3, 3)
    ry = torch.stack((cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy), -1).reshape(*z.shape, 3, 3)
    return ry @ rx @ rz


def canonicalize_state(raw: Any, transform: str) -> torch.Tensor:
    state = _as_2d(raw)
    n = state.shape[0]
    zero = state.new_zeros(n, 1)
    if transform == "pos_euler_8d":
        return torch.cat((state[..., :6], state[..., 7:8]), -1)
    if transform == "pos_euler_7d":
        return _pad7(state)
    if transform == "pos_quat":
        return torch.cat((state[..., :3], _quat_xyzw_to_rpy(state[..., 3:7]), state[..., 7:8]), -1)
    if transform == "pos_quat_no_grip":
        return torch.cat((state[..., :3], _quat_xyzw_to_rpy(state[..., 3:7]), zero), -1)
    if transform == "droid_state":
        return torch.cat((state[..., :6], state[..., 6:7]), -1)
    if transform == "zero":
        return state.new_zeros(n, 7)
    if transform == "austin_buds":
        return torch.cat((state[..., :6], state[..., 7:8]), -1)
    if transform == "nyu_franka":
        return torch.cat((state[..., -6:], zero), -1)
    if transform == "cmu_stretch":
        return torch.cat((state[..., :3], state.new_zeros(n, 2), state[..., 3:4], zero), -1)
    if transform == "language_table":
        return torch.cat((state[..., :2], state.new_zeros(n, 5)), -1)
    if transform == "robocasa":
        pos = state[..., 7:10]
        rpy = _quat_xyzw_to_rpy(state[..., 10:14])
        grip = state[..., 14:15] - state[..., 15:16]
        return torch.cat((pos, rpy, grip), -1)
    raise ValueError(f"Unknown state transform: {transform}")


def canonicalize_action(
    raw: Any,
    spec: OXESpec,
    *,
    state: Optional[torch.Tensor] = None,
    item: Optional[Mapping[str, Any]] = None,
    fps: float = 10.0,
) -> torch.Tensor:
    action = _as_2d(raw)
    transform = spec.action_transform
    state = None if state is None else _as_2d(state)[: action.shape[0]]

    if transform == "droid_target":
        if item is None:
            raise ValueError("DROID conversion requires the current pose and gripper columns.")
        current = _as_2d(item["observation.state.cartesian_position"])[: action.shape[0]]
        grip = _time_column(item["action.gripper_position"], action.shape[0])[:, :1]
        dpos = action[..., :3] - current[..., :3]
        drot = _rotmat_to_rotvec(
            _rpy_xyz_to_matrix(action[..., 3:6])
            @ _rpy_xyz_to_matrix(current[..., 3:6]).transpose(-1, -2)
        )
        action = torch.cat((dpos, drot, grip), -1)
    elif transform == "state_euler_residual":
        if state is None:
            raise ValueError("Bridge conversion requires current state.")
        target_rpy = state[..., 3:6] + action[..., 3:6]
        drot = _rotmat_to_rotvec(
            _rpy_xyz_to_matrix(target_rpy)
            @ _rpy_xyz_to_matrix(state[..., 3:6]).transpose(-1, -2)
        )
        action = torch.cat((action[..., :3], drot, 1.0 - action[..., 6:7]), -1)
    elif transform == "axis_angle_residual":
        if state is None:
            raise ValueError("BC-Z conversion requires current state.")
        current = _rotvec_to_matrix(state[..., 3:6])
        target = _rotvec_to_matrix(state[..., 3:6] + action[..., 3:6])
        action = torch.cat(
            (action[..., :3], _rotmat_to_rotvec(target @ current.transpose(-1, -2)),
             1.0 - action[..., 6:7]), -1
        )
    elif transform == "furniture_bench":
        if state is None:
            raise ValueError("FurnitureBench conversion requires current state.")
        current = _rpy_xyz_to_matrix(state[..., 3:6])
        local = _rpy_xyz_to_matrix(action[..., 3:6])
        action = torch.cat(
            (action[..., :3], _rotmat_to_rotvec(current @ local @ current.transpose(-1, -2)),
             1.0 - action[..., 6:7]), -1
        )
    elif transform in {"base_rpy_open", "base_rpy_close"}:
        grip = action[..., 6:7] if transform.endswith("close") else 1.0 - action[..., 6:7]
        action = torch.cat(
            (action[..., :3], _rotmat_to_rotvec(_rpy_xyz_to_matrix(action[..., 3:6])), grip), -1
        )
    elif transform == "taco_world":
        action = torch.cat(
            (action[..., :3] / 50.0,
             _rotmat_to_rotvec(_rpy_xyz_to_matrix(action[..., 3:6] / 20.0)),
             1.0 - action[..., 6:7]), -1
        )
    elif transform in {"velocity_open", "velocity_no_grip"}:
        grip = (1.0 - action[..., 6:7]) if transform.endswith("open") else action.new_zeros(action.shape[0], 1)
        action = torch.cat((action[..., :6] / float(fps), grip), -1)
    elif transform in {"euler_xyz_open", "euler_xyz_no_grip", "euler_zxy_open"}:
        matrix = (
            _euler_zxy_to_matrix(action[..., 3:6])
            if "zxy" in transform else _rpy_xyz_to_matrix(action[..., 3:6])
        )
        grip = (
            1.0 - action[..., 6:7]
            if transform.endswith("open") else action.new_zeros(action.shape[0], 1)
        )
        action = torch.cat((action[..., :3], _rotmat_to_rotvec(matrix), grip), -1)
    elif transform == "open_to_close":
        action = _pad7(action).clone()
        action[..., 6] = 1.0 - action[..., 6]
    elif transform == "signed_to_close":
        action = _pad7(action).clone()
        action[..., 6] = (action[..., 6] + 1.0) * 0.5
    elif transform == "xyz_only":
        action = torch.cat((action[..., :3], action.new_zeros(action.shape[0], 4)), -1)
    elif transform == "nyu_franka":
        eef = action[..., -8:-2]
        action = torch.cat(
            (eef[..., :3], _rotmat_to_rotvec(_rpy_xyz_to_matrix(eef[..., 3:6])),
             1.0 - action[..., -2:-1]), -1
        )
    elif transform == "drop_last":
        action = _pad7(action[..., :-1])
    elif transform == "xy_only":
        action = torch.cat((action[..., :2], action.new_zeros(action.shape[0], 5)), -1)
    elif transform == "robocasa":
        if action.shape[-1] >= 12:
            action = torch.cat((action[..., 5:11], action[..., 11:12]), -1)
        else:
            action = _pad7(action)
        action = action.clone()
        action[..., 6] = ((action[..., 6] + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown action transform: {transform}")

    action = _pad7(action)
    mask = torch.tensor(spec.action_mask, dtype=action.dtype, device=action.device)
    return action * mask


def _camera_name(key: str) -> str:
    name = str(key).split(".")[-1].lower()
    for suffix in ("_rgb", "_image"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if "wrist" in name or "hand" in name or "eye_in_hand" in name:
        return "robot0_eye_in_hand"
    return name


def _select_cameras(
    available: Sequence[str],
    preferred: Sequence[str],
    n_views: Optional[int],
) -> list[str]:
    available = list(dict.fromkeys(str(key) for key in available))
    selected = [key for key in preferred if key in available]
    selected.extend(key for key in available if key not in selected)
    if n_views is None:
        return selected
    if n_views <= 0:
        raise ValueError("n_views must be positive or None.")
    return selected[:n_views]


def _swap_bgr(frame: Any) -> Any:
    tensor = torch.as_tensor(np.asarray(frame) if not torch.is_tensor(frame) else frame)
    if tensor.ndim != 3:
        return frame
    return tensor.flip(0) if tensor.shape[0] == 3 else tensor.flip(-1)


def _image_sequence(
    frames: Mapping[str, Sequence[Any]],
    *,
    camera_keys: Sequence[str],
    bgr_cameras: Sequence[str],
    image_size: tuple[int, int],
    is_eval: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, tuple], dict[str, float], dict[str, tuple[int, int]]]:
    images, targets, masks = [], [], []
    crop_by_camera: dict[str, tuple] = {}
    angle_by_camera: dict[str, float] = {}
    source_size: dict[str, tuple[int, int]] = {}
    bgr = set(bgr_cameras)
    for camera in camera_keys:
        sequence = list(frames[camera])
        first = torch.as_tensor(sequence[0])
        if first.ndim != 3:
            raise ValueError(f"Camera {camera} returned shape {tuple(first.shape)}")
        height, width = (
            (int(first.shape[-2]), int(first.shape[-1]))
            if first.shape[0] in (1, 3) else (int(first.shape[0]), int(first.shape[1]))
        )
        source_size[camera] = (height, width)
        if is_eval or _is_wrist_camera_key(camera):
            crop = _full_crop_params(height, width)
        else:
            crop = _sample_fixed_crop_params(height, width, math.sqrt(0.9))
        angle = 0.0 if is_eval or _is_wrist_camera_key(camera) else _sample_rotation_angle(5.0)
        crop_by_camera[camera] = crop
        angle_by_camera[camera] = angle
        camera_images, camera_targets, camera_masks = [], [], []
        for raw_frame in sequence:
            frame = _swap_bgr(raw_frame) if camera in bgr else raw_frame
            common = dict(
                image_size=image_size, is_eval=is_eval, train_crop_min_scale=math.sqrt(0.9),
                eval_crop_scale=1.0, crop_params=crop, camera_key=camera,
                openpi_libero_augment=not is_eval, openpi_base_rotate_degrees=5.0,
                openpi_base_rotation_angle=angle,
            )
            camera_images.append(_normalize_image_tensor(
                frame, **common, color_jitter_brightness=0.3,
                color_jitter_contrast=0.4, color_jitter_saturation=0.5,
                color_jitter_hue=0.05, jpeg_enabled=not is_eval, jpeg_quality=95,
            ))
            camera_targets.append(_normalize_image_tensor(
                frame, **common, color_jitter_brightness=0.0,
                color_jitter_contrast=0.0, color_jitter_saturation=0.0,
                color_jitter_hue=0.0, jpeg_enabled=False,
            ))
            camera_masks.append(_rotation_valid_mask_train(image_size, angle))
        images.append(torch.stack(camera_images))
        targets.append(torch.stack(camera_targets))
        masks.append(torch.stack(camera_masks))
    return (
        torch.stack(images, dim=1), torch.stack(targets, dim=1), torch.stack(masks, dim=1),
        crop_by_camera, angle_by_camera, source_size,
    )


def _depth_sequence(
    path: Path,
    frame_indices: Sequence[int],
    camera_keys: Sequence[str],
    crop_by_camera: Mapping[str, tuple],
    angle_by_camera: Mapping[str, float],
    rgb_sizes: Mapping[str, tuple[int, int]],
    image_size: tuple[int, int],
    scale_mode: str,
) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        source_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
        positions = {int(frame): index for index, frame in enumerate(source_indices)}
        missing = [int(frame) for frame in frame_indices if int(frame) not in positions]
        if missing:
            raise KeyError(f"Depth sidecar {path} misses frames {missing[:4]}")
        source_cameras = [str(value) for value in payload["camera_names"].tolist()]
        camera_positions = {_camera_name(name): index for index, name in enumerate(source_cameras)}
        depth_np = np.asarray(payload["depth_meters"], dtype=np.float32)
        intrinsics_np = np.asarray(payload["camera_intrinsics"], dtype=np.float32) if "camera_intrinsics" in payload else None
        extrinsics_np = np.asarray(payload["camera_extrinsics_c2w"], dtype=np.float32) if "camera_extrinsics_c2w" in payload else None

    depth_frames, mask_frames, k_frames, e_frames = [], [], [], []
    for frame in frame_indices:
        timestep_depth, timestep_mask, timestep_k, timestep_e = [], [], [], []
        source_t = positions[int(frame)]
        for camera in camera_keys:
            name = _camera_name(camera)
            if name not in camera_positions:
                raise KeyError(f"Depth sidecar {path} has no camera matching {camera}")
            source_v = camera_positions[name]
            depth_raw = depth_np[source_t, source_v]
            crop = _scale_crop_params_to_source(
                crop_by_camera[camera], rgb_sizes[camera], tuple(depth_raw.shape)
            )
            depth, mask = _crop_resize_depth_and_mask(depth_raw, image_size, crop, 1.0e-3)
            depth, mask = _rotate_depth_and_mask_train(depth, mask, angle_by_camera[camera])
            timestep_depth.append(depth)
            timestep_mask.append(mask)
            if intrinsics_np is not None:
                timestep_k.append(_transform_intrinsics_for_crop_resize(
                    intrinsics_np[source_t, source_v], crop, image_size, tuple(depth_raw.shape),
                    False, False, False, angle_by_camera[camera],
                ))
            if extrinsics_np is not None:
                timestep_e.append(extrinsics_np[source_t, source_v])
        depth_frames.append(torch.stack(timestep_depth))
        mask_frames.append(torch.stack(timestep_mask))
        if timestep_k:
            k_frames.append(np.stack(timestep_k))
        if timestep_e:
            e_frames.append(np.stack(timestep_e))

    depth_m = torch.stack(depth_frames)
    depth_mask = torch.stack(mask_frames).bool()
    k_np = np.stack(k_frames).astype(np.float32) if k_frames else None
    e_np = np.stack(e_frames).astype(np.float32) if e_frames else None
    if scale_mode == "pointmap":
        if k_np is None or e_np is None:
            raise ValueError(f"Point-map depth scaling requires camera geometry in {path}")
        scale = _scene_scale_from_pointmaps(depth_m, depth_mask, k_np, e_np)
    elif scale_mode == "median_depth":
        valid = depth_m[depth_mask]
        scale = float(valid.median()) if valid.numel() else 1.0
    else:
        raise ValueError(f"Unknown depth scale mode: {scale_mode}")
    scale = max(float(scale), 1.0e-6)
    result = {
        "gt_depth_meters": depth_m,
        "gt_depth_da3": depth_m / scale,
        "gt_depth_mask": depth_mask,
        "gt_depth_scene_scale": torch.tensor(scale, dtype=torch.float32),
    }
    if k_np is not None and e_np is not None:
        result["gt_camera_intrinsics"] = torch.from_numpy(k_np)
        result["gt_camera_extrinsics_c2w"] = torch.from_numpy(e_np)
    return result


def _sample_contract(
    images: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor,
    *,
    camera_keys: Sequence[str],
    task: str,
    dataset_name: str,
    episode: Any,
    start: int,
    action_mask: Sequence[bool],
    view_max_views: Optional[int],
) -> dict[str, Any]:
    loss_mask = torch.tensor(action_mask, dtype=torch.bool).view(1, 1, 7).expand_as(actions)
    return {
        "current_images": images[0],
        "future_images": images[1:],
        "all_view_images": images,
        "all_view_target_images": targets,
        "all_view_target_mask": target_mask,
        "view_valid_mask": torch.ones(images.shape[:2], dtype=torch.bool),
        "actions": actions,
        "action_loss_mask": loss_mask,
        "proprioception": proprio,
        "task_description": str(task),
        "camera_keys": list(camera_keys),
        "view_max_views": int(view_max_views or images.shape[1]),
        "dataset_name": dataset_name,
        "action_stats_key": dataset_name,
        "action_output_frame": "base_delta",
        "episode_id": str(episode),
        "start_t": torch.tensor(int(start), dtype=torch.long),
        "has_action": True,
    }


class _StatisticsMixin:
    def _statistics(self, field: str, max_samples: Optional[int]) -> dict[str, dict[str, np.ndarray]]:
        total = len(self) if max_samples is None or max_samples < 0 else min(len(self), max_samples)
        if total <= 0:
            raise ValueError("Cannot compute statistics from an empty dataset.")
        indices = np.linspace(0, len(self) - 1, total, dtype=np.int64)
        rows: dict[str, list[np.ndarray]] = {}
        masks: dict[str, np.ndarray] = {}
        for index in indices:
            sample = self[int(index)]
            key = str(sample["action_stats_key"])
            value = sample[field].detach().cpu().numpy().reshape(-1, 7)
            rows.setdefault(key, []).append(value)
            if field == "actions":
                mask = sample.get("action_loss_mask")
                masks[key] = (
                    mask.detach().cpu().numpy().reshape(-1, 7).any(axis=0)
                    if mask is not None else np.ones(7, dtype=bool)
                )
        result = {}
        for key, chunks in rows.items():
            value = np.concatenate(chunks, axis=0).astype(np.float32)
            result[key] = {
                "q01": np.percentile(value, 1, axis=0).astype(np.float32),
                "q99": np.percentile(value, 99, axis=0).astype(np.float32),
                "mean": value.mean(axis=0).astype(np.float32),
                "std": value.std(axis=0).astype(np.float32),
                "mask": masks.get(key, np.ones(7, dtype=bool)),
            }
        return result

    def compute_action_statistics(self, max_samples: Optional[int] = None):
        return self._statistics("actions", max_samples)

    def compute_proprio_statistics(self, max_samples: Optional[int] = None):
        return self._statistics("proprioception", max_samples)


@dataclass(frozen=True)
class _HDF5Demo:
    path: Path
    demo: str
    split: str
    task_stem: str
    max_start: int


class MimicGenSequenceDataset(_StatisticsMixin, Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        depth_root: str | Path,
        task_descriptions_path: str | Path | None = None,
        image_size: tuple[int, int] = (224, 224),
        future_steps: int = 1,
        chunk_size: int = 8,
        include_current_action: bool = True,
        n_views: Optional[int] = 2,
        eval_ratio: float = 0.001,
        is_eval: bool = False,
        require_depth: bool = True,
    ):
        self.root = Path(root).expanduser()
        self.depth_root = Path(depth_root).expanduser()
        self.image_size = tuple(image_size)
        self.future_steps = int(future_steps)
        self.chunk_size = int(chunk_size)
        self.action_steps = self.future_steps + 1 if include_current_action else self.future_steps
        self.n_views = n_views
        self.is_eval = bool(is_eval)
        self.dataset_name = self.action_stats_key = "mimicgen"
        descriptions = Path(task_descriptions_path) if task_descriptions_path else self.root / "task_descriptions.json"
        self.task_descriptions = json.loads(descriptions.read_text()) if descriptions.exists() else {}
        self._files: dict[Path, h5py.File] = {}
        self.samples: list[_HDF5Demo] = []
        for path in sorted((self.root / "core").glob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                demos = sorted(handle["data"].keys())
                n_eval = max(1, int(len(demos) * eval_ratio)) if eval_ratio > 0 else 0
                demos = demos[-n_eval:] if is_eval and n_eval else (demos[:-n_eval] if n_eval else demos)
                for demo_name in demos:
                    n_actions = int(handle["data"][demo_name]["actions"].shape[0])
                    max_start = n_actions - self.action_steps * self.chunk_size
                    sidecar = self.depth_root / "core" / f"{path.stem}__{demo_name}.npz"
                    if max_start >= 0 and (sidecar.exists() or not require_depth):
                        self.samples.append(_HDF5Demo(path, demo_name, "core", path.stem, max_start))
        if not self.samples:
            raise FileNotFoundError(f"No usable MimicGen core demos under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self, path: Path) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    @staticmethod
    def _task_name(stem: str) -> str:
        stem = re.sub(r"_(iiwa|panda|sawyer|ur5e)$", "", stem)
        return re.sub(r"_[do]\d+$", "", stem)

    def _task_text(self, task_name: str) -> str:
        camel_case = "".join(part.capitalize() for part in task_name.split("_"))
        return self.task_descriptions.get(
            camel_case,
            self.task_descriptions.get(task_name, task_name.replace("_", " ")),
        )

    @staticmethod
    def _proprio(obs: Mapping[str, Any], indices: Sequence[int]) -> torch.Tensor:
        pos = torch.as_tensor(np.asarray(obs["robot0_eef_pos"][list(indices)]), dtype=torch.float32)
        quat = torch.as_tensor(np.asarray(obs["robot0_eef_quat"][list(indices)]), dtype=torch.float32)
        qpos = torch.as_tensor(np.asarray(obs["robot0_gripper_qpos"][list(indices)]), dtype=torch.float32)
        if qpos.shape[-1] == 1:
            grip = qpos
        elif qpos.shape[-1] == 2:
            grip = qpos[..., :1] - qpos[..., 1:2]
        else:
            grip = qpos.abs().mean(dim=-1, keepdim=True)
        return torch.cat((pos, _quat_xyzw_to_rpy(quat), grip), -1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        demo = self._file(sample.path)["data"][sample.demo]
        start = 0 if self.is_eval else random.randint(0, sample.max_start)
        visual_indices = [start + step * self.chunk_size for step in range(self.future_steps + 1)]
        action_indices = list(range(start, start + self.action_steps * self.chunk_size))
        available = [key for key in demo["obs"].keys() if key.endswith("_image")]
        preferred = ("agentview_image", "robot0_eye_in_hand_image")
        cameras = _select_cameras(available, preferred, self.n_views)
        frames = {camera: [demo["obs"][camera][t] for t in visual_indices] for camera in cameras}
        images, targets, target_mask, crops, angles, sizes = _image_sequence(
            frames, camera_keys=cameras, bgr_cameras=(), image_size=self.image_size, is_eval=self.is_eval
        )
        raw_actions = torch.as_tensor(np.asarray(demo["actions"][action_indices]), dtype=torch.float32)
        scale = raw_actions.new_tensor(MIMICGEN_ACTION_SCALE)
        actions = torch.cat(
            (raw_actions[..., :6] * scale, (raw_actions[..., 6:7] + 1.0) * 0.5), -1
        ).reshape(self.action_steps, self.chunk_size, 7)
        proprio = self._proprio(demo["obs"], visual_indices)
        task_name = self._task_name(sample.task_stem)
        result = _sample_contract(
            images, targets, target_mask, actions, proprio, camera_keys=cameras,
            task=self._task_text(task_name),
            dataset_name=self.dataset_name, episode=sample.demo, start=start,
            action_mask=FULL_ACTION_MASK, view_max_views=self.n_views,
        )
        sidecar = self.depth_root / sample.split / f"{sample.task_stem}__{sample.demo}.npz"
        if sidecar.exists():
            result.update(_depth_sequence(
                sidecar, visual_indices, cameras, crops, angles, sizes, self.image_size,
                "pointmap",
            ))
        return result


class LeRobotSequenceDataset(_StatisticsMixin, Dataset):
    def __init__(
        self,
        spec: OXESpec,
        root: str | Path,
        *,
        image_size: tuple[int, int] = (224, 224),
        future_steps: int = 1,
        chunk_size: int = 8,
        include_current_action: bool = True,
        n_views: Optional[int] = 2,
        eval_ratio: float = 0.001,
        is_eval: bool = False,
        depth_entries: Optional[Mapping[tuple[str, int], Path]] = None,
        max_episodes: Optional[int] = None,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        except ImportError as exc:
            raise ImportError("Open X-Embodiment pretraining requires lerobot==0.4.4.") from exc
        self.spec = spec
        self.root = Path(root).expanduser()
        self.image_size = tuple(image_size)
        self.future_steps = int(future_steps)
        self.chunk_size = int(chunk_size)
        self.action_steps = self.future_steps + 1 if include_current_action else self.future_steps
        self.n_views = n_views
        self.is_eval = bool(is_eval)
        self.dataset_name = self.action_stats_key = spec.name
        self.depth_entries = dict(depth_entries or {})
        metadata = LeRobotDatasetMetadata(spec.repo_id, root=self.root)
        self.fps = float(metadata.fps)
        camera_keys = _select_cameras(metadata.camera_keys, spec.cameras, n_views)
        self.camera_keys = camera_keys
        action_steps = self.action_steps * self.chunk_size
        delta_timestamps = {
            spec.action_key: [step / self.fps for step in range(action_steps)],
            **{camera: [step * self.chunk_size / self.fps for step in range(self.future_steps + 1)]
               for camera in camera_keys},
        }
        for state_key in spec.state_keys:
            delta_timestamps[state_key] = [step / self.fps for step in range(action_steps)]
        if spec.action_transform == "droid_target":
            delta_timestamps["action.gripper_position"] = delta_timestamps[spec.action_key]
        self.samples: list[tuple[int, int, int]] = []
        episodes = list(range(int(metadata.total_episodes)))
        n_eval = max(1, int(len(episodes) * eval_ratio)) if eval_ratio > 0 else 0
        episodes = episodes[-n_eval:] if is_eval and n_eval else (episodes[:-n_eval] if n_eval else episodes)
        if max_episodes is not None:
            episodes = episodes[: max(0, int(max_episodes))]
        for episode_index in episodes:
            episode = metadata.episodes[episode_index]
            first = int(episode["dataset_from_index"])
            last = int(episode["dataset_to_index"])
            max_start = last - action_steps - 1
            if max_start >= first:
                self.samples.append((episode_index, first, max_start))
        if not self.samples:
            raise FileNotFoundError(f"No usable episodes in {self.root} for {spec.name}")
        self.dataset = LeRobotDataset(
            spec.repo_id, root=self.root, episodes=episodes,
            delta_timestamps=delta_timestamps, download_videos=True,
            video_backend="pyav", tolerance_s=0.04,
        )
        self.absolute_to_relative = getattr(self.dataset, "_absolute_to_relative_idx", None)

    def __len__(self) -> int:
        return len(self.samples)

    def _state(self, item: Mapping[str, Any]) -> torch.Tensor:
        raw_parts = [torch.as_tensor(item[key], dtype=torch.float32) for key in self.spec.state_keys]
        length = max((part.shape[0] for part in raw_parts if part.ndim > 1), default=1)
        parts = [_time_column(part, length) for part in raw_parts]
        return canonicalize_state(torch.cat(parts, -1), self.spec.state_transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, first, max_start = self.samples[index]
        start = first if self.is_eval else random.randint(first, max_start)
        dataset_index = (
            start if self.absolute_to_relative is None else self.absolute_to_relative[start]
        )
        item = self.dataset[dataset_index]
        visual_offsets = [step * self.chunk_size for step in range(self.future_steps + 1)]
        frames = {}
        for camera in self.camera_keys:
            sequence = item[camera]
            if torch.is_tensor(sequence) and sequence.ndim == 3:
                sequence = sequence.unsqueeze(0)
            frames[camera] = list(sequence)
        images, targets, target_mask, crops, angles, sizes = _image_sequence(
            frames, camera_keys=self.camera_keys, bgr_cameras=self.spec.bgr_cameras,
            image_size=self.image_size, is_eval=self.is_eval,
        )
        raw_state = self._state(item)
        raw_action = item[self.spec.action_key]
        actions = canonicalize_action(
            raw_action, self.spec, state=raw_state, item=item, fps=self.fps
        ).reshape(self.action_steps, self.chunk_size, 7)
        proprio = raw_state[visual_offsets]
        task = item.get("task") or self.spec.name.replace("_", " ")
        if isinstance(task, (list, tuple, np.ndarray)):
            task = task[0]
        result = _sample_contract(
            images, targets, target_mask, actions, proprio,
            camera_keys=self.camera_keys, task=str(task), dataset_name=self.dataset_name,
            episode=episode, start=start - first, action_mask=self.spec.action_mask,
            view_max_views=self.n_views,
        )
        result["episode_index"] = torch.tensor(episode, dtype=torch.long)
        depth_path = self.depth_entries.get((str(self.root.resolve()), episode))
        if depth_path is not None:
            frame_indices = [start - first + offset for offset in visual_offsets]
            result.update(_depth_sequence(
                depth_path, frame_indices, self.camera_keys, crops, angles, sizes,
                self.image_size, "median_depth",
            ))
        return result


class DatasetMixture(_StatisticsMixin, Dataset):
    def __init__(
        self,
        sources: Sequence[tuple[str, Dataset, float]],
        *,
        epoch_size: int,
        seed: int = 42,
    ):
        if not sources or any(len(dataset) == 0 for _, dataset, _ in sources):
            raise ValueError("All mixture sources must be non-empty.")
        self.names = [name for name, _, _ in sources]
        self.datasets = {name: dataset for name, dataset, _ in sources}
        weights = np.asarray([weight for _, _, weight in sources], dtype=np.float64)
        weights /= weights.sum()
        counts = np.maximum(1, np.rint(weights * 10_000).astype(np.int64))
        schedule = np.repeat(np.arange(len(sources), dtype=np.int64), counts)
        rng = np.random.default_rng(seed)
        rng.shuffle(schedule)
        self.schedule = schedule
        self.counts = np.bincount(schedule, minlength=len(sources))
        seen = np.zeros(len(sources), dtype=np.int64)
        self.ordinals = np.empty(len(schedule), dtype=np.int64)
        for slot, source_index in enumerate(schedule):
            self.ordinals[slot] = seen[source_index]
            seen[source_index] += 1
        self.source_list = [dataset for _, dataset, _ in sources]
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        slot = int(index) % len(self.schedule)
        cycle = int(index) // len(self.schedule)
        source_index = int(self.schedule[slot])
        ordinal = cycle * int(self.counts[source_index]) + int(self.ordinals[slot])
        dataset = self.source_list[source_index]
        child_index = (ordinal * 1_000_003 + self.seed * (source_index + 1)) % len(dataset)
        sample = dict(dataset[child_index])
        sample["mixture_source"] = self.names[source_index]
        return sample


class DatasetConcat(_StatisticsMixin, Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = {str(index): dataset for index, dataset in enumerate(datasets)}
        self.children = list(datasets)
        self.cumulative_sizes = np.cumsum([len(dataset) for dataset in self.children])

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        child = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        offset = 0 if child == 0 else int(self.cumulative_sizes[child - 1])
        return self.children[child][index - offset]


def _robocasa_depth_entries(
    index_path: str | Path,
    robocasa_root: Path,
) -> dict[tuple[str, int], Path]:
    path = Path(index_path).expanduser()
    payload = json.loads(path.read_text())
    entries = payload.get("episodes", payload.get("entries", []))
    result = {}
    for entry in entries:
        dataset_root = (robocasa_root / entry["dataset_root"]).resolve()
        depth_path = (path.parent / entry.get("depth_path", entry.get("depth_npz_path"))).resolve()
        result[(str(dataset_root), int(entry["episode_index"]))] = depth_path
    return result


def _build_robocasa(
    root: str | Path,
    depth_index_path: str | Path,
    **common: Any,
) -> Dataset:
    root = Path(root).expanduser().resolve()
    depth_entries = _robocasa_depth_entries(depth_index_path, root)
    repo_roots = sorted({Path(repo) for repo, _ in depth_entries})
    datasets = []
    for repo_root in repo_roots:
        if not (repo_root / "meta" / "info.json").exists():
            continue
        local_spec = replace(ROBOCASA_SPEC, repo_id=repo_root.name)
        datasets.append(LeRobotSequenceDataset(
            local_spec, repo_root, depth_entries=depth_entries, **common
        ))
    if not datasets:
        raise FileNotFoundError(f"No indexed RoboCasa365 datasets under {root}")
    return DatasetConcat(datasets)


def build_pretraining_dataset(config: Mapping[str, Any], is_eval: bool = False) -> DatasetMixture:
    """Build the paper 72/18/10 pretraining mixture."""
    image_size = tuple(config.get("image_size", (224, 224)))
    future_steps = int(config.get("future_steps", 1))
    chunk_size = int(config.get("chunk_size", 8))
    include_current_action = bool(config.get("include_current_action", True))
    raw_views = config.get("n_views", 2)
    n_views = None if raw_views is None or str(raw_views).lower() == "all" else int(raw_views)
    eval_ratio = float(config.get("eval_ratio", 0.001))
    max_episodes = config.get("max_episodes")
    common = dict(
        image_size=image_size, future_steps=future_steps, chunk_size=chunk_size,
        include_current_action=include_current_action, n_views=n_views,
        eval_ratio=eval_ratio, is_eval=is_eval,
    )
    enabled = set(config.get("sources", PAPER_SOURCE_RATIOS))
    unknown = enabled.difference(PAPER_SOURCE_RATIOS)
    if unknown:
        raise ValueError(f"Unknown pretraining sources: {sorted(unknown)}")

    sources = []
    if "open_x_embodiment" in enabled:
        openx_root = Path(config["openx_root"]).expanduser()
        oxe_sources = [(spec.name, LeRobotSequenceDataset(
            spec, openx_root / spec.repo_id, max_episodes=max_episodes, **common,
        ), float(spec.weight)) for spec in OXE_SPECS]
        oxe = DatasetMixture(oxe_sources, epoch_size=int(config.get("oxe_epoch_size", 784_000)), seed=int(config.get("seed", 42)))
        sources.append(("open_x_embodiment", oxe, PAPER_SOURCE_RATIOS["open_x_embodiment"]))
    if "mimicgen" in enabled:
        mimicgen = MimicGenSequenceDataset(
            config["mimicgen_root"], depth_root=config["mimicgen_depth_root"],
            task_descriptions_path=config.get("task_descriptions_path"), **common,
        )
        sources.append(("mimicgen", mimicgen, PAPER_SOURCE_RATIOS["mimicgen"]))
    if "robocasa365" in enabled:
        robocasa = _build_robocasa(
            config["robocasa_root"], config["robocasa_depth_index_path"],
            max_episodes=max_episodes, **common,
        )
        sources.append(("robocasa365", robocasa, PAPER_SOURCE_RATIOS["robocasa365"]))
    return DatasetMixture(
        sources, epoch_size=int(config.get("epoch_size", 784_000)), seed=int(config.get("seed", 42))
    )


__all__ = [
    "CANONICAL_ACTION",
    "DatasetMixture",
    "LeRobotSequenceDataset",
    "MimicGenSequenceDataset",
    "OXE_SPECS",
    "PAPER_SOURCE_RATIOS",
    "build_pretraining_dataset",
    "canonicalize_action",
    "canonicalize_state",
]
