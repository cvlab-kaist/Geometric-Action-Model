"""Rollout video / panel rendering helpers.

Split out of ``eval_libero_unified.py`` (behavior-preserving extraction). This
module must remain free of any import of ``eval_libero_unified`` to avoid a
circular import; ``eval_libero_unified`` re-imports these names so the public
API is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


LIBERO_CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")


def get_bicubic_resample() -> int:
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def depth_from_obs(obs: dict[str, Any], key: str) -> torch.Tensor | None:
    depth = obs.get(key)
    if depth is None:
        return None
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    return torch.from_numpy(arr.copy()).float()


def video_frame_to_rgb_uint8(frame: Any, *, camera_size: int | None = None) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Video frame must be HWC/CHW/gray, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"Video frame must have 1, 3, or 4 channels, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)
        if not bool(finite.all()):
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.nanmax(arr)) <= 1.0 and float(np.nanmin(arr)) >= -0.05:
            arr = arr * 255.0
        elif float(np.nanmin(arr)) < -0.05:
            arr = (arr + 1.0) * 127.5
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr[..., :3]

    if camera_size is not None and (arr.shape[0] != int(camera_size) or arr.shape[1] != int(camera_size)):
        # Match the historical env.render(height,width,...) behavior.
        from PIL import Image as _PIL_Image

        arr = np.array(_PIL_Image.fromarray(arr).resize((int(camera_size), int(camera_size))))
    return np.ascontiguousarray(arr)


def render_rollout_frame(
    env: Any,
    camera_size: int,
    obs: dict[str, Any] | None = None,
    camera_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Build a horizontal RGB strip from per-camera frames.

    Prefers obs[<cam>_image] when present (already rendered by env.step()) to
    avoid an extra env.render() per call. The extra render call leaks MuJoCo
    EGL context state on Daint GH200 nodes and SIGABRTs after a few hundred
    invocations. Falls back to env.render() only when obs is missing the keys
    (e.g. during synthesis from cached state).
    """
    frames = []
    for cam in tuple(camera_names or LIBERO_CAMERA_NAMES):
        key = f"{cam}_image"
        if obs is not None and key in obs:
            frame = video_frame_to_rgb_uint8(obs[key], camera_size=camera_size)
        else:
            frame = env.render(mode="rgb_array", height=camera_size, width=camera_size, camera_name=cam)
            frame = video_frame_to_rgb_uint8(frame, camera_size=camera_size)
        frames.append(frame)
    return np.ascontiguousarray(np.concatenate(frames, axis=1))


def rotate_policy_frame_to_raw(image: np.ndarray) -> np.ndarray:
    return np.rot90(image, 2).copy()


def chw_to_rgb_uint8(
    tensor: torch.Tensor,
    size: tuple[int, int] = (160, 160),
    rotate_display: bool = False,
) -> np.ndarray:
    arr = tensor.detach().cpu().float()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    arr_np = arr.numpy()
    if arr_np.ndim == 2:
        arr_np = np.stack([arr_np] * 3, axis=-1)
    if arr_np.shape[-1] == 1:
        arr_np = np.repeat(arr_np, 3, axis=-1)
    if arr_np.dtype != np.uint8:
        if np.nanmax(arr_np) <= 1.0 and np.nanmin(arr_np) >= -0.05:
            arr_np = arr_np * 255.0
        elif np.nanmin(arr_np) < -0.05:
            arr_np = (arr_np + 1.0) * 127.5
        arr_np = np.clip(arr_np, 0.0, 255.0).astype(np.uint8)
    if rotate_display:
        arr_np = rotate_policy_frame_to_raw(arr_np)
    pil = Image.fromarray(arr_np).resize(size, get_bicubic_resample())
    return np.array(pil)


def depth_to_rgb_uint8(
    depth: torch.Tensor,
    size: tuple[int, int] = (160, 160),
    rotate_display: bool = False,
) -> np.ndarray:
    arr = depth.detach().cpu().float().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Match the training W&B depth monitor rendering so rollout diagnostics are comparable.
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    try:
        import matplotlib.cm as cm

        rgb = (cm.turbo(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
    if rotate_display:
        rgb = rotate_policy_frame_to_raw(rgb)
    pil = Image.fromarray(rgb).resize(size, get_bicubic_resample())
    return np.array(pil)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    pil = Image.fromarray(image)
    label_h = 24
    out = Image.new("RGB", (pil.width, pil.height + label_h), (0, 0, 0))
    out.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((5, 5), label[:120], fill=(255, 255, 255))
    return np.array(out)


def placeholder_panel(label: str, size: tuple[int, int] = (160, 160)) -> np.ndarray:
    img = Image.new("RGB", size, (25, 25, 25))
    draw = ImageDraw.Draw(img)
    draw.text((10, size[1] // 2 - 8), label[:32], fill=(220, 220, 220))
    return np.array(img)


def pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    pad = np.zeros((image.shape[0], width - image.shape[1], 3), dtype=np.uint8)
    return np.concatenate([image, pad], axis=1)


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


def tile_panels(panels: list[np.ndarray], cols: int) -> np.ndarray:
    if not panels:
        return placeholder_panel("empty")
    cols = max(1, cols)
    rows = []
    for start in range(0, len(panels), cols):
        row = panels[start:start + cols]
        row_h = max(panel.shape[0] for panel in row)
        if len(row) < cols:
            filler = np.zeros((row_h, row[0].shape[1], 3), dtype=np.uint8)
            row = row + [filler for _ in range(cols - len(row))]
        row = [pad_to_height(panel, row_h) for panel in row]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def view_role(view_idx: int) -> str:
    return "ext" if view_idx % 2 == 0 else "wrist"


def is_gam_predicted_sequence_debug(debug: dict[str, Any]) -> bool:
    return str(debug.get("rollout_visual_contract", "")) in {
        "gam_predicted_sequence",
        "gam_predicted_sequence_autoregressive",
        "gam_predicted_sequence_trainlike",
    }


def view_time_label(view_idx: int, cond_num: int) -> str:
    step_idx = view_idx // 2
    cond_steps = max(1, cond_num // 2)
    if step_idx < cond_steps:
        return f"t{step_idx}"
    return f"t+{step_idx - cond_steps + 1}"


def view_label(prefix: str, view_idx: int, cond_num: int) -> str:
    return f"{prefix} {view_time_label(view_idx, cond_num)} {view_role(view_idx)} v{view_idx}"


def predicted_sequence_view_label(prefix: str, view_idx: int, start_timestep: int) -> str:
    obs_t = int(start_timestep) + int(view_idx // 2)
    return f"{prefix} obs_t={obs_t} {view_role(view_idx)} v{view_idx}"


def tile_ext_wrist_timeline(
    panels_by_view: dict[int, np.ndarray],
    total_view: int,
    cond_num: int,
    cell_size: tuple[int, int],
    missing_prefix: str,
) -> np.ndarray:
    if not panels_by_view:
        return add_label(placeholder_panel(missing_prefix, cell_size), f"{missing_prefix} unavailable")
    cond_steps = max(1, cond_num // 2)
    max_step = max(cond_steps, (total_view + 1) // 2)
    rows = []
    for parity, role in ((0, "ext"), (1, "wrist")):
        row = []
        for step_idx in range(cond_steps, max_step):
            view_idx = step_idx * 2 + parity
            if view_idx >= total_view:
                continue
            label = f"{view_time_label(view_idx, cond_num)} {role} v{view_idx}"
            row.append(
                panels_by_view.get(
                    view_idx,
                    add_label(placeholder_panel(f"missing {role}", cell_size), label),
                )
            )
        if row:
            row_h = max(panel.shape[0] for panel in row)
            rows.append(np.concatenate([pad_to_height(panel, row_h) for panel in row], axis=1))
    if not rows:
        return add_label(placeholder_panel(missing_prefix, cell_size), f"{missing_prefix} unavailable")
    width = max(row.shape[1] for row in rows)
    return np.concatenate([pad_to_width(row, width) for row in rows], axis=0)


def tile_predicted_sequence_timeline(
    panels_by_view: dict[int, np.ndarray],
    total_view: int,
    start_timestep: int,
    display_total_view: int,
    cell_size: tuple[int, int],
    missing_prefix: str,
) -> np.ndarray:
    target_total_view = max(int(total_view), int(display_total_view))
    if target_total_view <= 0:
        return add_label(placeholder_panel(missing_prefix, cell_size), f"{missing_prefix} unavailable")
    rows = []
    max_step = max(1, (target_total_view + 1) // 2)
    for parity, role in ((0, "ext"), (1, "wrist")):
        row = []
        for step_idx in range(max_step):
            view_idx = step_idx * 2 + parity
            if view_idx >= target_total_view:
                continue
            label = predicted_sequence_view_label(missing_prefix, view_idx, start_timestep)
            row.append(
                panels_by_view.get(
                    view_idx,
                    add_label(placeholder_panel(f"missing {role}", cell_size), label),
                )
            )
        if row:
            row_h = max(panel.shape[0] for panel in row)
            rows.append(np.concatenate([pad_to_height(panel, row_h) for panel in row], axis=1))
    if not rows:
        return add_label(placeholder_panel(missing_prefix, cell_size), f"{missing_prefix} unavailable")
    width = max(row.shape[1] for row in rows)
    return np.concatenate([pad_to_width(row, width) for row in rows], axis=0)


def tile_ext_wrist_range(
    panels_by_view: dict[int, np.ndarray],
    start_step: int,
    stop_step: int,
    label_cond_num: int,
    cell_size: tuple[int, int],
    missing_prefix: str,
) -> np.ndarray:
    if stop_step <= start_step:
        return add_label(placeholder_panel(missing_prefix, cell_size), f"{missing_prefix} unavailable")
    rows = []
    for parity, role in ((0, "ext"), (1, "wrist")):
        row = []
        for step_idx in range(start_step, stop_step):
            view_idx = step_idx * 2 + parity
            label = f"{view_time_label(view_idx, label_cond_num)} {role} v{view_idx}"
            row.append(
                panels_by_view.get(
                    view_idx,
                    add_label(placeholder_panel(f"missing {role}", cell_size), label),
                )
            )
        row_h = max(panel.shape[0] for panel in row)
        rows.append(np.concatenate([pad_to_height(panel, row_h) for panel in row], axis=1))
    width = max(row.shape[1] for row in rows)
    return np.concatenate([pad_to_width(row, width) for row in rows], axis=0)


def extract_chw_panels_by_view(
    images: Any,
    target_count: int,
    label_cond_num: int,
    cell_size: tuple[int, int],
    prefix: str,
    rotate_display: bool = False,
) -> dict[int, np.ndarray]:
    if not isinstance(images, torch.Tensor):
        return {}
    x = images.detach().cpu()
    if x.ndim == 5:
        x = x[0]
    if x.ndim != 4:
        return {}
    panels = {}
    for view_idx in range(max(0, target_count)):
        if view_idx < x.shape[0]:
            panel = chw_to_rgb_uint8(x[view_idx], size=cell_size, rotate_display=rotate_display)
        else:
            panel = placeholder_panel(f"missing {prefix}", cell_size)
        panels[view_idx] = add_label(panel, view_label(prefix, view_idx, label_cond_num))
    return panels


def extract_predicted_sequence_depth_panels_by_view(
    debug: dict[str, Any],
    cell_size: tuple[int, int],
    prefix: str = "pred depth",
    rotate_display: bool = True,
) -> dict[int, np.ndarray]:
    depth = debug.get("depth")
    if depth is None:
        return {}
    total_view = int(debug.get("total_view", 0))
    start_timestep = int(debug.get("predicted_sequence_start_timestep", 0))
    d = depth.detach().cpu()
    if d.ndim == 4:
        d = d[0]
    elif d.ndim == 3 and total_view > 0 and d.shape[0] % total_view == 0:
        d = d.reshape(-1, total_view, *d.shape[1:])[0]
    if d.ndim != 3:
        return {}
    panels = {}
    for view_idx in range(min(total_view, d.shape[0])):
        panels[view_idx] = add_label(
            depth_to_rgb_uint8(d[view_idx], size=cell_size, rotate_display=rotate_display),
            predicted_sequence_view_label(prefix, view_idx, start_timestep),
        )
    return panels


def extract_depth_panels_by_view(
    debug: dict[str, Any],
    cell_size: tuple[int, int],
    view_start: int | None = None,
    view_stop: int | None = None,
    prefix: str = "pred depth",
    rotate_display: bool = True,
) -> dict[int, np.ndarray]:
    depth = debug.get("depth")
    if depth is None:
        return {}
    total_view = int(debug.get("total_view", 0))
    cond_num = int(debug.get("cond_num", 0))
    d = depth.detach().cpu()
    if d.ndim == 4:
        d = d[0]
    elif d.ndim == 3 and total_view > 0 and d.shape[0] == total_view:
        pass
    elif d.ndim == 3 and total_view > 0 and d.shape[0] % total_view == 0:
        d = d.reshape(-1, total_view, *d.shape[1:])[0]
    else:
        return {}
    panels = {}
    start = cond_num if view_start is None else int(view_start)
    stop = total_view if view_stop is None else int(view_stop)
    for view_idx in range(max(0, start), min(total_view, d.shape[0], stop)):
        panels[view_idx] = add_label(
            depth_to_rgb_uint8(d[view_idx], size=cell_size, rotate_display=rotate_display),
            view_label(prefix, view_idx, cond_num),
        )
    return panels


def extract_predicted_sequence_rgb_panels_by_view(
    debug: dict[str, Any],
    cell_size: tuple[int, int],
    rotate_display: bool = True,
) -> dict[int, np.ndarray]:
    rgb = debug.get("rgb")
    if rgb is None:
        return {}
    prefix = str(debug.get("rgb_prefix", "pred rgb"))
    total_view = int(debug.get("total_view", 0))
    start_timestep = int(debug.get("predicted_sequence_start_timestep", 0))
    r = rgb.detach().cpu()
    if r.ndim == 5:
        r = r[0]
    elif r.ndim == 4 and total_view > 0 and r.shape[0] % total_view == 0:
        r = r.reshape(-1, total_view, *r.shape[1:])[0]
    if r.ndim != 4:
        return {}
    panels = {}
    for view_idx in range(min(total_view, r.shape[0])):
        panels[view_idx] = add_label(
            chw_to_rgb_uint8(r[view_idx], size=cell_size, rotate_display=rotate_display),
            predicted_sequence_view_label(prefix, view_idx, start_timestep),
        )
    return panels


def extract_rgb_panels_by_view(
    debug: dict[str, Any],
    cell_size: tuple[int, int],
    rotate_display: bool = True,
) -> dict[int, np.ndarray]:
    rgb = debug.get("rgb")
    if rgb is None:
        return {}
    prefix = str(debug.get("rgb_prefix", "pred rgb"))
    total_view = int(debug.get("total_view", 0))
    cond_num = int(debug.get("cond_num", 0))
    r = rgb.detach().cpu()
    if r.ndim == 5:
        r = r[0]
    elif r.ndim == 4 and total_view > 0 and r.shape[0] == total_view:
        pass
    elif r.ndim == 4 and total_view > 0 and r.shape[0] % total_view == 0:
        r = r.reshape(-1, total_view, *r.shape[1:])[0]
    else:
        return {}
    panels = {}
    for view_idx in range(cond_num, min(total_view, r.shape[0])):
        panels[view_idx] = add_label(
            chw_to_rgb_uint8(r[view_idx], size=cell_size, rotate_display=rotate_display),
            view_label(prefix, view_idx, cond_num),
        )
    return panels


def extract_obs_depth_panels(
    debug: dict[str, Any],
    cell_size: tuple[int, int],
    target_count: int | None = None,
    label_cond_num: int | None = None,
) -> list[np.ndarray]:
    depths = debug.get("obs_depths_raw")
    cond_num = int(debug.get("cond_num", 0))
    label_cond_num = cond_num if label_cond_num is None else int(label_cond_num)
    target_count = cond_num if target_count is None else int(target_count)
    if not isinstance(depths, torch.Tensor):
        return [
            add_label(
                placeholder_panel("missing depth", cell_size),
                view_label("live env depth", view_idx, label_cond_num),
            )
            for view_idx in range(max(0, target_count))
        ]
    d = depths.detach().cpu()
    if d.ndim == 4:
        d = d[0]
    if d.ndim != 3:
        return [
            add_label(
                placeholder_panel("missing depth", cell_size),
                view_label("live env depth", view_idx, label_cond_num),
            )
            for view_idx in range(max(0, target_count))
        ]
    panels = []
    actual_count = min(target_count, d.shape[0])
    for view_idx in range(max(0, target_count)):
        if view_idx < actual_count:
            panel = depth_to_rgb_uint8(d[view_idx], size=cell_size, rotate_display=False)
        else:
            panel = placeholder_panel("missing depth", cell_size)
        panels.append(
            add_label(
                panel,
                view_label("live env depth", view_idx, label_cond_num),
            )
        )
    return panels


def action_text_panel(
    debug: dict[str, Any],
    action_idx: int,
    repeat_idx: int,
    policy_call_idx: int,
    policy_obs_step: int,
    executed_steps: int,
    action_horizon: int,
    action_repeat: int,
    action_repeat_mode: str,
    env_action: np.ndarray,
    success: bool,
    size: tuple[int, int],
) -> np.ndarray:
    img = Image.new("RGB", size, (12, 12, 12))
    draw = ImageDraw.Draw(img)
    actions = debug.get("actions")
    task_desc = str(debug.get("task_desc", ""))
    text_audit = debug.get("text_prompt_audit") or {}
    actions_available = int(actions.shape[0]) if isinstance(actions, torch.Tensor) and actions.ndim >= 2 else 0
    lines = [
        (
            f"policy_call={policy_call_idx} policy_obs_step={policy_obs_step} "
            f"env_step_after={executed_steps} action_idx={action_idx} "
            f"repeat_idx={repeat_idx}/{max(action_repeat, 1)}"
        ),
    ]
    env_actions_per_model_step = int(debug.get("env_actions_per_model_step", 1))
    lines += [
        (
            f"action_horizon={action_horizon} action_repeat={action_repeat} "
            f"repeat_mode={action_repeat_mode} "
            f"policy_reobserve_every_env_steps~="
            f"{max(action_horizon, 1) * max(action_repeat, 1) * max(env_actions_per_model_step, 1)}"
        ),
        (
            f"history_H={debug.get('effective_history_horizon', 'n/a')}/"
            f"{debug.get('history_horizon', 'n/a')} "
            f"predicted_steps={debug.get('predicted_steps', 'n/a')} "
            f"exec_steps={debug.get('executed_model_steps', 'n/a')} "
            f"history_commit={debug.get('history_commit_stride_actions', 'n/a')} model_steps"
        ),
        (
            f"decode_horizon={debug.get('rollout_decode_horizon', 'n/a')} "
            f"mode={debug.get('rollout_decode_horizon_mode', 'n/a')} "
            f"requested={debug.get('rollout_decode_horizon_requested', 'n/a')}"
        ),
        (
            f"preprocess crop={debug.get('eval_crop_scale', 'n/a')} "
            f"dataset_rotate180={debug.get('dataset_da3_input_rotate180', 'n/a')} "
            f"hflip={debug.get('dataset_da3_input_hflip', 'n/a')} "
            f"vflip={debug.get('da3_input_vflip', 'n/a')}"
        ),
        f"success={success}",
        f"task={task_desc[:100]}",
        (
            f"text_sha1={text_audit.get('prompt_sha1', 'n/a')} "
            f"tokens={text_audit.get('token_count', 'n/a')} "
            f"encoder={text_audit.get('text_encoder_type', 'n/a')} "
            f"text_norm={text_audit.get('text_norm', 'n/a')}"
        ),
        f"env action raw={np.array2string(env_action, precision=3, suppress_small=True)}",
    ]
    if isinstance(actions, torch.Tensor) and actions.numel() > 0:
        act_np = actions.detach().cpu().float().numpy()
        lines.append(f"canonical a0={np.array2string(act_np[min(action_idx, len(act_np)-1)], precision=3, suppress_small=True)}")
        lines.append(f"canonical chunk shape={act_np.shape} actions_available={actions_available}")
        lines.append(
            f"norm clamp={debug.get('actions_norm_clamped', 'n/a')} "
            f"raw_max_abs={debug.get('actions_norm_max_abs_raw', 'n/a')}"
        )
    proprio = debug.get("proprio_raw")
    if isinstance(proprio, torch.Tensor):
        lines.append(f"proprio7={np.array2string(proprio.numpy(), precision=3, suppress_small=True)}")
    for i, line in enumerate(lines):
        draw.text((8, 8 + i * 18), line, fill=(240, 240, 240))
    return np.array(img)


def compact_action_text_panel(
    debug: dict[str, Any],
    action_idx: int,
    repeat_idx: int,
    policy_call_idx: int,
    policy_obs_step: int,
    executed_steps: int,
    action_horizon: int,
    action_repeat: int,
    action_repeat_mode: str,
    env_action: np.ndarray,
    success: bool,
    size: tuple[int, int],
) -> np.ndarray:
    img = Image.new("RGB", size, (12, 12, 12))
    draw = ImageDraw.Draw(img)
    actions = debug.get("actions")
    text_audit = debug.get("text_prompt_audit") or {}
    task_desc = str(debug.get("task_desc", ""))
    env_actions_per_model_step = int(debug.get("env_actions_per_model_step", 1))
    lines = [
        f"call={policy_call_idx} obs={policy_obs_step} after={executed_steps} a={action_idx} r={repeat_idx}/{max(action_repeat, 1)}",
        f"horizon={action_horizon} repeat={action_repeat} mode={action_repeat_mode}",
        f"reobserve~{max(action_horizon, 1) * max(action_repeat, 1) * max(env_actions_per_model_step, 1)} env steps",
        (
            f"history_H={debug.get('effective_history_horizon', 'n/a')}/"
            f"{debug.get('history_horizon', 'n/a')} predicted={debug.get('predicted_steps', 'n/a')} "
            f"exec={debug.get('executed_model_steps', 'n/a')}"
        ),
        (
            f"decode_H={debug.get('rollout_decode_horizon', 'n/a')} "
            f"mode={debug.get('rollout_decode_horizon_mode', 'n/a')}"
        ),
        (
            f"hist_commit={debug.get('history_commit_stride_actions', 'n/a')} model_steps "
            f"crop={debug.get('eval_crop_scale', 'n/a')} "
            f"rot180={debug.get('dataset_da3_input_rotate180', 'n/a')} "
            f"hflip={debug.get('dataset_da3_input_hflip', 'n/a')} "
            f"vflip={debug.get('da3_input_vflip', 'n/a')}"
        ),
        f"success={success}",
        f"task={task_desc[:54]}",
        f"sha={text_audit.get('prompt_sha1', 'n/a')}",
        f"env_a={np.array2string(env_action, precision=2, suppress_small=True)}",
    ]
    if isinstance(actions, torch.Tensor) and actions.numel() > 0:
        act_np = actions.detach().cpu().float().numpy()
        lines.append(f"canon={np.array2string(act_np[min(action_idx, len(act_np)-1)], precision=2, suppress_small=True)}")
        lines.append(f"chunk={act_np.shape}")
        raw_abs = debug.get("actions_norm_max_abs_raw", "n/a")
        raw_abs_text = f"{float(raw_abs):.2f}" if isinstance(raw_abs, (int, float)) else str(raw_abs)
        lines.append(
            f"norm_clamp={debug.get('actions_norm_clamped', 'n/a')} "
            f"raw_abs={raw_abs_text}"
        )
    proprio = debug.get("proprio_raw")
    if isinstance(proprio, torch.Tensor):
        lines.append(f"prop={np.array2string(proprio.numpy(), precision=2, suppress_small=True)}")
    for i, line in enumerate(lines):
        draw.text((8, 8 + i * 18), line, fill=(240, 240, 240))
    return np.array(img)


def labeled_thumbnail(image: np.ndarray, size: tuple[int, int], label: str) -> np.ndarray:
    pil = Image.fromarray(image).resize(size, get_bicubic_resample())
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, size[0], 22), fill=(0, 0, 0))
    draw.text((5, 5), label[:80], fill=(255, 255, 255))
    return np.array(pil)


def render_compact_policy_frame(
    debug: dict[str, Any],
    action_idx: int,
    repeat_idx: int,
    policy_call_idx: int,
    policy_obs_step: int,
    executed_steps: int,
    action_horizon: int,
    action_repeat: int,
    action_repeat_mode: str,
    env_action: np.ndarray,
    success: bool,
    live_frame: np.ndarray | None = None,
) -> np.ndarray:
    cell_size = (96, 96)
    cond_num = int(debug.get("cond_num", 0))
    total_view = int(debug.get("total_view", 0))
    observed_view_count = max(cond_num, int(debug.get("observed_view_count", cond_num)))
    history_steps = max(1, observed_view_count // 2)
    rotate_prediction_display = bool(debug.get("rotate_policy_input"))
    visual_debug_label = str(debug.get("visual_debug_label", "policy"))
    is_predicted_sequence = is_gam_predicted_sequence_debug(debug)

    obs_grid = tile_ext_wrist_range(
        extract_chw_panels_by_view(
            debug.get("obs_images_raw"),
            target_count=observed_view_count,
            label_cond_num=observed_view_count,
            cell_size=cell_size,
            prefix="obs rgb",
            rotate_display=False,
        ),
        start_step=0,
        stop_step=max(1, observed_view_count // 2),
        label_cond_num=observed_view_count,
        cell_size=cell_size,
        missing_prefix="obs rgb",
    )

    if is_predicted_sequence:
        pred_rgb_grid = tile_predicted_sequence_timeline(
            extract_predicted_sequence_rgb_panels_by_view(
                debug,
                cell_size,
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            start_timestep=int(debug.get("predicted_sequence_start_timestep", 0)),
            display_total_view=total_view,
            cell_size=cell_size,
            missing_prefix="pred rgb",
        )
        pred_depth_grid = tile_predicted_sequence_timeline(
            extract_predicted_sequence_depth_panels_by_view(
                debug,
                cell_size,
                prefix="pred depth",
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            start_timestep=int(debug.get("predicted_sequence_start_timestep", 0)),
            display_total_view=total_view,
            cell_size=cell_size,
            missing_prefix="pred depth",
        )
        diag_h = max(obs_grid.shape[0], pred_rgb_grid.shape[0], pred_depth_grid.shape[0])
        diagnostic_strip = np.concatenate(
            [
                pad_to_height(add_label(obs_grid, "observed RGB history"), diag_h + 24),
                pad_to_height(
                    add_label(pred_rgb_grid, f"{visual_debug_label} predicted RGB sequence"),
                    diag_h + 24,
                ),
                pad_to_height(
                    add_label(pred_depth_grid, f"{visual_debug_label} predicted depth sequence"),
                    diag_h + 24,
                ),
            ],
            axis=1,
        )
    else:
        current_depth_grid = tile_ext_wrist_range(
            extract_depth_panels_by_view(
                debug,
                cell_size,
                view_start=0,
                view_stop=cond_num,
                prefix="DA3 depth",
                rotate_display=rotate_prediction_display,
            ),
            start_step=0,
            stop_step=history_steps,
            label_cond_num=cond_num,
            cell_size=cell_size,
            missing_prefix="DA3 depth",
        )

        future_depth_grid = tile_ext_wrist_timeline(
            extract_depth_panels_by_view(
                debug,
                cell_size,
                prefix="future depth",
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            cond_num=cond_num,
            cell_size=cell_size,
            missing_prefix="future depth",
        )

        diag_h = max(obs_grid.shape[0], current_depth_grid.shape[0], future_depth_grid.shape[0])
        diagnostic_strip = np.concatenate(
            [
                pad_to_height(add_label(obs_grid, "obs RGB history"), diag_h + 24),
                pad_to_height(add_label(current_depth_grid, f"{visual_debug_label} conditioning depth"), diag_h + 24),
                pad_to_height(add_label(future_depth_grid, f"{visual_debug_label} future depth"), diag_h + 24),
            ],
            axis=1,
        )

    # H warms up from 1 to the target history, so the number of decoded future
    # slots shrinks during the first few policy calls. Keep the compact canvas
    # at the widest warm-up layout so mp4 frame sizes remain stable.
    width = max(diagnostic_strip.shape[1], 1152)
    live_w = min(640, max(560, width - 320))
    live_h = 480
    if live_frame is not None:
        live_panel = labeled_thumbnail(live_frame, (live_w, live_h), "rollout env after action")
    else:
        live_panel = add_label(placeholder_panel("no live render", (live_w, live_h - 24)), "rollout env unavailable")
    text = compact_action_text_panel(
        debug,
        action_idx,
        repeat_idx,
        policy_call_idx,
        policy_obs_step,
        executed_steps,
        action_horizon,
        action_repeat,
        action_repeat_mode,
        env_action,
        success,
        size=(width - live_w, live_h),
    )
    top_row = np.concatenate([live_panel, text], axis=1)

    rows = [
        pad_to_width(top_row, width),
        pad_to_width(diagnostic_strip, width),
    ]
    return np.concatenate(rows, axis=0)


SPLIT_VIDEO_KEYS = ("rollout_rgb", "pred_future_depth", "gt_depth")
SPLIT_VIDEO_PANEL_SIZE = (256, 256)


def new_split_video_frames() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in SPLIT_VIDEO_KEYS}


def black_video_panel(size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE) -> np.ndarray:
    return np.zeros((int(size[1]), int(size[0]), 3), dtype=np.uint8)


def resize_video_panel(image: np.ndarray, size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE) -> np.ndarray:
    arr = video_frame_to_rgb_uint8(image)
    if arr.shape[1] == int(size[0]) and arr.shape[0] == int(size[1]):
        return np.ascontiguousarray(arr)
    pil = Image.fromarray(arr).resize(size, get_bicubic_resample())
    return np.ascontiguousarray(np.array(pil))


def resize_two_view_strip(image: np.ndarray, size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE) -> np.ndarray:
    arr = video_frame_to_rgb_uint8(image)
    target_size = (int(size[0]) * 2, int(size[1]))
    if arr.shape[1] == target_size[0] and arr.shape[0] == target_size[1]:
        return np.ascontiguousarray(arr)
    pil = Image.fromarray(arr).resize(target_size, get_bicubic_resample())
    return np.ascontiguousarray(np.array(pil))


def clean_two_view_strip(
    panels: Sequence[np.ndarray | None],
    size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE,
) -> np.ndarray:
    normalized = []
    for idx in range(2):
        panel = panels[idx] if idx < len(panels) else None
        if panel is None:
            normalized.append(black_video_panel(size))
        else:
            normalized.append(resize_video_panel(panel, size))
    return np.ascontiguousarray(np.concatenate(normalized, axis=1))


def depth_view_stack_from_debug(debug: dict[str, Any]) -> torch.Tensor | None:
    depth = debug.get("depth")
    if not isinstance(depth, torch.Tensor):
        return None
    total_view = int(debug.get("total_view", 0))
    d = depth.detach().cpu()
    if d.ndim == 4:
        if d.shape[0] == 1:
            d = d[0]
        elif total_view > 0 and d.shape[0] == total_view:
            pass
        else:
            return None
    elif d.ndim == 3 and total_view > 0 and d.shape[0] % total_view == 0:
        d = d.reshape(-1, total_view, *d.shape[1:])[0]
    if d.ndim != 3:
        return None
    return d


def predicted_depth_step_index(debug: dict[str, Any], n_views: int, num_steps: int) -> int:
    if num_steps <= 0:
        return 0
    if is_gam_predicted_sequence_debug(debug):
        step_idx = int(debug.get("executed_sequence_start") or 0)
    else:
        step_idx = int(debug.get("cond_num", 0)) // max(1, n_views)
    return max(0, min(int(step_idx), int(num_steps) - 1))


def clean_predicted_depth_strip(
    debug: dict[str, Any],
    size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE,
) -> np.ndarray:
    d = depth_view_stack_from_debug(debug)
    if d is None:
        return clean_two_view_strip([], size)
    n_views = max(1, int(debug.get("n_views", 2)))
    num_steps = max(1, int(math.ceil(float(d.shape[0]) / float(n_views))))
    step_idx = predicted_depth_step_index(debug, n_views=n_views, num_steps=num_steps)
    rotate_prediction_display = bool(debug.get("rotate_policy_input"))
    panels: list[np.ndarray | None] = []
    for camera_idx in range(2):
        view_idx = step_idx * n_views + camera_idx
        if camera_idx < n_views and view_idx < d.shape[0]:
            panels.append(
                depth_to_rgb_uint8(
                    d[view_idx],
                    size=size,
                    rotate_display=rotate_prediction_display,
                )
            )
        else:
            panels.append(None)
    return clean_two_view_strip(panels, size)


def clean_gt_depth_strip_from_obs(
    obs: dict[str, Any] | None,
    size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE,
) -> np.ndarray | None:
    if obs is None:
        return None
    panels: list[np.ndarray | None] = []
    for camera_name in LIBERO_CAMERA_NAMES[:2]:
        depth = depth_from_obs(obs, f"{camera_name}_depth")
        if depth is None:
            panels.append(None)
        else:
            panels.append(depth_to_rgb_uint8(depth, size=size, rotate_display=False))
    if all(panel is None for panel in panels):
        return None
    return clean_two_view_strip(panels, size)


def clean_gt_depth_strip_from_debug(
    debug: dict[str, Any],
    size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE,
) -> np.ndarray:
    depths = debug.get("obs_depths_raw")
    if not isinstance(depths, torch.Tensor):
        return clean_two_view_strip([], size)
    d = depths.detach().cpu()
    if d.ndim == 4:
        d = d[0]
    if d.ndim != 3 or d.shape[0] <= 0:
        return clean_two_view_strip([], size)
    n_views = max(1, int(debug.get("n_views", 2)))
    step_idx = max(0, int(math.ceil(float(d.shape[0]) / float(n_views))) - 1)
    panels: list[np.ndarray | None] = []
    for camera_idx in range(2):
        view_idx = step_idx * n_views + camera_idx
        if camera_idx < n_views and view_idx < d.shape[0]:
            panels.append(depth_to_rgb_uint8(d[view_idx], size=size, rotate_display=False))
        else:
            panels.append(None)
    return clean_two_view_strip(panels, size)


def render_predicted_depth_split_frame(
    debug: dict[str, Any],
    cell_size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE,
) -> np.ndarray:
    return clean_predicted_depth_strip(debug, cell_size) if debug else clean_two_view_strip([], cell_size)


def render_gt_depth_split_frame(
    debug: dict[str, Any],
    obs: dict[str, Any] | None = None,
    cell_size: tuple[int, int] = SPLIT_VIDEO_PANEL_SIZE,
) -> np.ndarray:
    live_depth = clean_gt_depth_strip_from_obs(obs, cell_size)
    if live_depth is not None:
        return live_depth
    return clean_gt_depth_strip_from_debug(debug, cell_size) if debug else clean_two_view_strip([], cell_size)


def append_split_video_frame(
    split_frames: dict[str, list[np.ndarray]] | None,
    debug: dict[str, Any] | None,
    live_frame: np.ndarray,
    obs: dict[str, Any] | None = None,
) -> None:
    if split_frames is None:
        return
    debug = debug if isinstance(debug, dict) else {}
    split_frames["rollout_rgb"].append(resize_two_view_strip(live_frame))
    split_frames["pred_future_depth"].append(render_predicted_depth_split_frame(debug))
    split_frames["gt_depth"].append(render_gt_depth_split_frame(debug, obs=obs))


def render_detailed_policy_frame(
    debug: dict[str, Any],
    action_idx: int,
    repeat_idx: int,
    policy_call_idx: int,
    policy_obs_step: int,
    executed_steps: int,
    action_horizon: int,
    action_repeat: int,
    action_repeat_mode: str,
    env_action: np.ndarray,
    success: bool,
    live_frame: np.ndarray | None = None,
) -> np.ndarray:
    if bool(debug.get("compact_detailed_video", False)):
        return render_compact_policy_frame(
            debug,
            action_idx,
            repeat_idx,
            policy_call_idx,
            policy_obs_step,
            executed_steps,
            action_horizon,
            action_repeat,
            action_repeat_mode,
            env_action,
            success,
            live_frame=live_frame,
        )
    cell_size = (160, 160)
    obs_raw = debug.get("obs_images_raw")
    obs_policy = debug.get("obs_images_policy", debug.get("obs_images"))
    total_view = int(debug.get("total_view", 0))
    cond_num = int(debug.get("cond_num", 0))
    observed_view_count = max(cond_num, int(debug.get("observed_view_count", cond_num)))
    rotate_prediction_display = bool(debug.get("rotate_policy_input"))
    visual_debug_label = str(debug.get("visual_debug_label", "Stage 2"))
    is_predicted_sequence = is_gam_predicted_sequence_debug(debug)

    raw_obs_count = max(2, observed_view_count)
    raw_obs_by_view = extract_chw_panels_by_view(
        obs_raw,
        target_count=raw_obs_count,
        label_cond_num=max(raw_obs_count, observed_view_count),
        cell_size=cell_size,
        prefix="raw obs rgb",
        rotate_display=False,
    )
    raw_obs_row = tile_panels(
        [raw_obs_by_view[i] for i in range(raw_obs_count)] or [placeholder_panel("no raw obs")],
        cols=2,
    )
    env_depth_row = tile_panels(
        extract_obs_depth_panels(
            debug,
            cell_size,
            target_count=raw_obs_count,
            label_cond_num=max(raw_obs_count, observed_view_count),
        )
        or [placeholder_panel("no env depth", cell_size)],
        cols=2,
    )

    policy_rotation_label = (
        "OpenPI 180-degree frame" if debug.get("rotate_policy_input") else "raw env frame"
    )
    policy_obs_by_view = extract_chw_panels_by_view(
        obs_policy,
        target_count=raw_obs_count,
        label_cond_num=max(raw_obs_count, observed_view_count),
        cell_size=cell_size,
        prefix="policy input rgb",
        rotate_display=False,
    )
    policy_obs_row = tile_panels(
        [policy_obs_by_view[i] for i in range(raw_obs_count)] or [placeholder_panel("no policy obs")],
        cols=2,
    )

    current_depth_row = None
    if is_predicted_sequence:
        rgb_grid = tile_predicted_sequence_timeline(
            extract_predicted_sequence_rgb_panels_by_view(
                debug,
                cell_size,
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            start_timestep=int(debug.get("predicted_sequence_start_timestep", 0)),
            display_total_view=total_view,
            cell_size=cell_size,
            missing_prefix="pred rgb",
        )
        depth_grid = tile_predicted_sequence_timeline(
            extract_predicted_sequence_depth_panels_by_view(
                debug,
                cell_size,
                prefix="pred depth",
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            start_timestep=int(debug.get("predicted_sequence_start_timestep", 0)),
            display_total_view=total_view,
            cell_size=cell_size,
            missing_prefix="pred depth",
        )
    else:
        current_depth_by_view = extract_depth_panels_by_view(
            debug,
            cell_size,
            view_start=0,
            view_stop=cond_num,
            prefix="decoded current depth",
            rotate_display=rotate_prediction_display,
        )
        current_depth_panels = [
            current_depth_by_view.get(
                i,
                add_label(
                    placeholder_panel("missing current depth", cell_size),
                    view_label("decoded current depth", i, max(raw_obs_count, cond_num)),
                ),
            )
            for i in range(raw_obs_count)
        ]
        current_depth_row = tile_panels(
            current_depth_panels or [placeholder_panel("no current depth", cell_size)],
            cols=2,
        )

        rgb_grid = tile_ext_wrist_timeline(
            extract_rgb_panels_by_view(
                debug,
                cell_size,
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            cond_num=cond_num,
            cell_size=cell_size,
            missing_prefix="pred rgb",
        )

        depth_grid = tile_ext_wrist_timeline(
            extract_depth_panels_by_view(
                debug,
                cell_size,
                prefix="pred future depth",
                rotate_display=rotate_prediction_display,
            ),
            total_view=total_view,
            cond_num=cond_num,
            cell_size=cell_size,
            missing_prefix="pred depth",
        )

    width = max(
        raw_obs_row.shape[1],
        env_depth_row.shape[1],
        policy_obs_row.shape[1],
        0 if current_depth_row is None else current_depth_row.shape[1],
        rgb_grid.shape[1],
        depth_grid.shape[1],
        0 if live_frame is None else live_frame.shape[1],
        960,
    )
    text = action_text_panel(
        debug,
        action_idx,
        repeat_idx,
        policy_call_idx,
        policy_obs_step,
        executed_steps,
        action_horizon,
        action_repeat,
        action_repeat_mode,
        env_action,
        success,
        size=(width, 180),
    )
    rows = [
        *(
            [pad_to_width(add_label(live_frame, "live env render after this env.step"), width)]
            if live_frame is not None else []
        ),
        pad_to_width(add_label(raw_obs_row, "current live observation RGB (raw env frame)"), width),
        pad_to_width(add_label(env_depth_row, "current live observation depth buffer"), width),
        pad_to_width(add_label(policy_obs_row, f"current policy input RGB ({policy_rotation_label})"), width),
        *(
            []
            if current_depth_row is None
            else [pad_to_width(add_label(current_depth_row, f"{visual_debug_label} decoded current depth (raw-frame display)"), width)]
        ),
        pad_to_width(
            add_label(
                rgb_grid,
                (
                    f"{visual_debug_label} predicted RGB sequence (raw-frame display)"
                    if is_predicted_sequence
                    else f"{visual_debug_label} {debug.get('rgb_source', 'decoded future RGB')} raw-frame display"
                ),
            ),
            width,
        ),
        pad_to_width(
            add_label(
                depth_grid,
                (
                    f"{visual_debug_label} predicted depth sequence (DA3 deep blocks, raw-frame display)"
                    if is_predicted_sequence
                    else f"{visual_debug_label} decoded future depth (raw-frame display)"
                ),
            ),
            width,
        ),
        text,
    ]
    return np.concatenate(rows, axis=0)


def pad_frames_to_common_size(frames: list[np.ndarray], multiple: int = 16) -> list[np.ndarray]:
    if not frames:
        return []
    coerced = [video_frame_to_rgb_uint8(frame) for frame in frames]
    max_height = max(frame.shape[0] for frame in coerced)
    max_width = max(frame.shape[1] for frame in coerced)
    target_height = max_height + ((-max_height) % multiple)
    target_width = max_width + ((-max_width) % multiple)
    normalized = []
    for frame in coerced:
        pad_h = target_height - frame.shape[0]
        pad_w = target_width - frame.shape[1]
        normalized.append(
            np.ascontiguousarray(
                np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            )
        )
    return normalized


def save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    imageio.mimsave(path, pad_frames_to_common_size(frames), fps=fps)
