"""Unified LIBERO closed-loop rollout evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_REPO_ROOT = Path(__file__).resolve().parents[1]

from robot.modeling.action_head_v2 import ActionHeadV2
from robot.modeling.backbone_factory import create_stage1_backbone, stage1_backbone_type
from robot.modeling.conditioning import ProprioConditioner, TextConditioner
from robot.data.dataset import (
    ActionNormalizer,
    StateNormalizer,
    _eef_relative_to_world_delta_action,
    infer_libero_hdf5_hflip,
    normalize_action_frame,
    resolve_image_augmentation_config,
    _maybe_rotate_image_180,
    _normalize_image_tensor,
    _quat_xyzw_to_rpy,
)
from gam.evaluation.registry import append_eval_record, local_now_minute
from robot.modeling.future_predictor import build_future_predictor
from robot.evaluation.rollout_env import (
    create_rollout_env_libero,
    create_rollout_env_libero_isolated,
    list_libero_task_metadata,
)
from robot.losses.unified_loss import (
    _slots_to_da3_propagation_inputs,
    extract_level0_slots,
)
from train_robot import DA3FineTuneModel

# Behavior-preserving split: these modules hold function groups extracted from
# this file. Keep imports one-way into this entrypoint to avoid circular imports.
# Re-importing here keeps both internal call sites and the external
# ``eval_libero_unified.<name>`` API stable.
from gam.evaluation.libero_plus import (
    LIBERO_PLUS_OFFICIAL_CATEGORY_ALIASES,
    LIBERO_PLUS_OFFICIAL_CATEGORY_SLUGS,
    LIBERO_PLUS_PERTURBATION_ALIASES,
    _row_int,
    aggregate_plus_category_results,
    aggregate_plus_perturbation_results,
    annotate_libero_plus_official_categories,
    classify_libero_plus_perturbation,
    filter_libero_plus_official_category_metadata,
    filter_libero_plus_task_metadata,
    load_libero_plus_task_classification,
    parse_csv_list,
    parse_int_list,
    select_eval_task_entries,
    select_libero_plus_task_subset,
)
from gam.evaluation.video import (
    LIBERO_CAMERA_NAMES,
    SPLIT_VIDEO_KEYS,
    SPLIT_VIDEO_PANEL_SIZE,
    action_text_panel,
    add_label,
    append_split_video_frame,
    black_video_panel,
    chw_to_rgb_uint8,
    clean_gt_depth_strip_from_debug,
    clean_gt_depth_strip_from_obs,
    clean_predicted_depth_strip,
    clean_two_view_strip,
    compact_action_text_panel,
    depth_from_obs,
    depth_to_rgb_uint8,
    depth_view_stack_from_debug,
    extract_chw_panels_by_view,
    extract_depth_panels_by_view,
    extract_obs_depth_panels,
    extract_predicted_sequence_depth_panels_by_view,
    extract_predicted_sequence_rgb_panels_by_view,
    extract_rgb_panels_by_view,
    get_bicubic_resample,
    is_gam_predicted_sequence_debug,
    labeled_thumbnail,
    new_split_video_frames,
    pad_frames_to_common_size,
    pad_to_height,
    pad_to_width,
    placeholder_panel,
    predicted_depth_step_index,
    predicted_sequence_view_label,
    render_compact_policy_frame,
    render_detailed_policy_frame,
    render_gt_depth_split_frame,
    render_predicted_depth_split_frame,
    render_rollout_frame,
    resize_two_view_strip,
    resize_video_panel,
    rotate_policy_frame_to_raw,
    save_video,
    tile_ext_wrist_range,
    tile_ext_wrist_timeline,
    tile_panels,
    tile_predicted_sequence_timeline,
    video_frame_to_rgb_uint8,
    view_label,
    view_role,
    view_time_label,
)


# ---------------------------------------------------------------------------
# Optional CUDA-graph inference compile.
#
# Gated entirely by env vars; the DEFAULT (DA3_COMPILE_INFERENCE unset/none)
# path skips torch.compile and cudagraph_mark_step_begin.
#
#   DA3_COMPILE_INFERENCE      : "none" (default) | "all" | comma subset of
#                                {predictor, shallow, propagate, action_head}
#   DA3_COMPILE_INFERENCE_MODE : reduce-overhead (default) | max-autotune |
#                                max-autotune-no-cudagraphs | default
#   DA3_CUDAGRAPH_CLONE        : "1" (default) | "0"; wrap compiled outputs in
#                                the clone module for CUDA-graph modes
# ---------------------------------------------------------------------------

# CUDA-graph compile modes write outputs into static graph buffers that the
# next graph replay overwrites. GAM keeps shallow tokens in its history and
# threads module outputs across separate compiled graphs, so a raw
# graph-buffer output raises "accessing tensor output of CUDAGraphs that has
# been overwritten by a subsequent run". _CloneOutputModule clones each
# compiled module's outputs so callers hold independent memory. One extra
# memcpy is negligible next to the kernel-launch savings.
_CUDAGRAPH_COMPILE_MODES = {"reduce-overhead", "max-autotune"}
_VALID_INFERENCE_COMPILE_MODES = (
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
    "default",
)
_INFERENCE_COMPILE_TARGETS = {"predictor", "shallow", "propagate", "action_head"}

# Module-level flag: True once at least one inference target was compiled with
# a CUDA-graph mode, so call_policy knows to emit cudagraph_mark_step_begin().
# Stays False on the default (none) path, keeping the rollout a pure no-op.
_INFERENCE_CUDAGRAPH_ACTIVE = False

try:
    import torch.utils._pytree as _clone_pytree
except Exception:  # noqa: BLE001
    _clone_pytree = None


def _clone_one(x: Any) -> Any:
    return x.clone() if isinstance(x, torch.Tensor) else x


class _CloneOutputModule(torch.nn.Module):
    """Wrap a compiled callable so its outputs are cloned out of the CUDA-graph
    static buffers before returning. Implemented as an nn.Module so it can be
    assigned to nn.Module submodule attributes (e.g. model.action_head) as well
    as bound-method slots."""

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.inner(*args, **kwargs)
        if _clone_pytree is not None:
            return _clone_pytree.tree_map(_clone_one, out)
        if isinstance(out, torch.Tensor):
            return out.clone()
        if isinstance(out, (tuple, list)):
            return type(out)(_clone_one(o) for o in out)
        if isinstance(out, dict):
            return {k: _clone_one(v) for k, v in out.items()}
        return out

    def __getattr__(self, name: str) -> Any:
        # nn.Module.__getattr__ handles params/buffers/submodules (including
        # our 'inner'); fall through to the wrapped module's own attributes
        # (e.g. predictor.language_len, .rope) so callers that read attributes
        # off the compiled module keep working.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            return getattr(inner, name)


def _resolve_inference_compile_targets(raw: str) -> set[str]:
    """Parse DA3_COMPILE_INFERENCE into a set of target labels."""
    raw = str(raw).strip().lower()
    if raw in {"", "0", "false", "off", "none", "no"}:
        return set()
    if raw in {"1", "true", "on", "yes", "all"}:
        return set(_INFERENCE_COMPILE_TARGETS)
    aliases = {
        "future_predictor": "predictor",
        "fp": "predictor",
        "shallow_encoder": "shallow",
        "encode_shallow": "shallow",
        "deep": "propagate",
        "deep_propagate": "propagate",
        "propagation": "propagate",
        "head": "action_head",
        "actionhead": "action_head",
    }
    out: set[str] = set()
    for part in raw.replace("+", ",").replace(";", ",").split(","):
        token = aliases.get(part.strip(), part.strip())
        if not token:
            continue
        if token not in _INFERENCE_COMPILE_TARGETS:
            raise ValueError(
                "DA3_COMPILE_INFERENCE must be none, all, or a comma-separated "
                "subset of predictor, shallow, propagate, action_head; got "
                f"{raw!r}."
            )
        out.add(token)
    return out


def _resolve_inference_compile_mode() -> str:
    mode = str(os.environ.get("DA3_COMPILE_INFERENCE_MODE", "reduce-overhead")).strip().lower()
    if mode not in _VALID_INFERENCE_COMPILE_MODES:
        raise ValueError(
            "DA3_COMPILE_INFERENCE_MODE must be one of "
            f"{_VALID_INFERENCE_COMPILE_MODES}; got {mode!r}."
        )
    return mode


def _compile_for_inference(label: str, target: Any, mode: str, clone: bool) -> Any:
    """torch.compile a target and (for CUDA-graph modes) clone its outputs.

    Marks the module-level CUDA-graph-active flag when a CUDA-graph mode is
    selected, so call_policy emits cudagraph_mark_step_begin() at rollout time.
    """
    global _INFERENCE_CUDAGRAPH_ACTIVE
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "DA3_COMPILE_INFERENCE requested, but this PyTorch build has no torch.compile."
        )
    compiled = torch.compile(target, mode=mode, dynamic=False)
    if mode in _CUDAGRAPH_COMPILE_MODES:
        _INFERENCE_CUDAGRAPH_ACTIVE = True
        if clone:
            compiled = _CloneOutputModule(compiled)
    print(f"[compile-inference] {label}: mode={mode} clone={clone and mode in _CUDAGRAPH_COMPILE_MODES}")
    return compiled


LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_SUITE_ORDER = {
    "libero_spatial": 0,
    "libero_object": 1,
    "libero_goal": 2,
    "libero_10": 3,
    "libero_90": 4,
}
LIBERO_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PER_TASK_FIELDNAMES = [
    "suite",
    "task_id",
    "eval_task_index",
    "task_name",
    "plus_perturbation",
    "plus_official_task_id",
    "plus_official_category",
    "plus_official_category_slug",
    "plus_official_difficulty_level",
    "raw_task_language",
    "policy_language",
    "bddl_file",
    "init_states_file",
    "num_trials",
    "num_success",
    "success_rate",
    "avg_steps",
]
DEFAULT_CONFIG = None
# OpenX default removed because dataset.target_hz=5 leaked into LIBERO
# HDF5 evals (ckpt config has no target_hz key, so OmegaConf.merge kept the
# OpenX default value), forcing action_repeat=4 and ~4× slower / wrong-timing
# rollouts. With None, --config is optional: when omitted, the architecture
# config is taken entirely from `ckpt['config']`.


def normalize_action_frame(value: Any) -> str:
    """Normalize action-frame names without requiring a freshly imported dataset module."""
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


def eef_delta_to_world_delta_action_for_rollout(
    action: torch.Tensor,
    current_proprio: torch.Tensor,
    *,
    proprio_orientation: str = "rpy",
) -> torch.Tensor:
    """Lazy bridge to dataset's legacy moving-frame EEF-delta converter.

    In-training eval can import this file after `robot.data.dataset` was already
    loaded at job start. Keeping the helper lookup lazy lets base-frame
    rollouts continue in those long-lived processes while still failing clearly.
    """
    from robot import dataset as _dataset

    helper = getattr(_dataset, "_eef_relative_to_world_delta_action", None)
    if helper is None:
        raise RuntimeError(
            "action_frame='eef_delta' requires robot.data.dataset._eef_relative_to_world_delta_action. "
            "Restart the job with the committed action-frame code before running EEF-delta rollout."
        )
    return helper(action, current_proprio, proprio_orientation=proprio_orientation)


def eef_relative_trajectory_to_world_delta_action_for_rollout(
    action: torch.Tensor,
    anchor_proprio: torch.Tensor,
    current_proprio: torch.Tensor,
    *,
    proprio_orientation: str = "rpy",
) -> torch.Tensor:
    """Lazy bridge to dataset's chunk-start relative target converter."""
    from robot import dataset as _dataset

    helper = getattr(_dataset, "_eef_relative_trajectory_to_world_delta_action", None)
    if helper is None:
        raise RuntimeError(
            "action_frame='eef_relative' requires "
            "robot.data.dataset._eef_relative_trajectory_to_world_delta_action. "
            "Restart the job with the committed action-frame code before rollout."
        )
    return helper(
        action,
        anchor_proprio,
        current_proprio,
        proprio_orientation=proprio_orientation,
    )


@dataclass(frozen=True)
class ProtocolPreset:
    name: str
    num_trials_per_task: int
    seed: int
    env_seed: int
    num_steps_wait: int
    max_steps_by_suite: dict[str, int]
    action_horizon: int = 1


OPENVLA_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

PRESETS = {
    "openvla_50": ProtocolPreset(
        name="openvla_50",
        num_trials_per_task=50,
        seed=7,
        env_seed=0,
        num_steps_wait=10,
        max_steps_by_suite=OPENVLA_STEPS,
        action_horizon=1,
    ),
    "openpi_50": ProtocolPreset(
        name="openpi_50",
        num_trials_per_task=50,
        seed=7,
        env_seed=7,
        num_steps_wait=10,
        max_steps_by_suite=OPENVLA_STEPS,
        action_horizon=5,
    ),
    "libero_original_20": ProtocolPreset(
        name="libero_original_20",
        num_trials_per_task=20,
        seed=42,
        env_seed=42,
        num_steps_wait=0,
        max_steps_by_suite={
            "libero_spatial": 600,
            "libero_object": 600,
            "libero_goal": 600,
            "libero_10": 600,
            "libero_90": 600,
        },
        action_horizon=1,
    ),
    "smoke_5": ProtocolPreset(
        name="smoke_5",
        num_trials_per_task=5,
        seed=7,
        env_seed=0,
        num_steps_wait=10,
        max_steps_by_suite=OPENVLA_STEPS,
        action_horizon=1,
    ),
}


@dataclass(frozen=True)
class EvalShard:
    index: int
    count: int
    source: str

    @property
    def enabled(self) -> bool:
        return self.count > 1

    def owns(self, work_index: int) -> bool:
        return (int(work_index) % self.count) == self.index


def _env_int(names: tuple[str, ...]) -> tuple[int | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(value), name
        except ValueError:
            continue
    return None, None


def resolve_eval_shard(index_arg: str | int | None, count_arg: str | int | None) -> EvalShard:
    count_text = str(count_arg if count_arg is not None else "auto").strip().lower()
    index_text = str(index_arg if index_arg is not None else "auto").strip().lower()
    source = "cli"

    if count_text in {"", "auto"}:
        count, count_source = _env_int(("SLURM_NTASKS", "WORLD_SIZE"))
        count = count if count and count > 0 else 1
        source = count_source or "default"
    else:
        count = int(count_text)

    if index_text in {"", "auto"}:
        index, index_source = _env_int(("SLURM_PROCID", "RANK"))
        index = index if index is not None else 0
        if index_source is not None:
            source = index_source
    else:
        index = int(index_text)

    if count < 1:
        raise ValueError(f"--shard-count must be >= 1, got {count}")
    if index < 0 or index >= count:
        raise ValueError(f"--shard-index must satisfy 0 <= index < count, got {index}/{count}")
    return EvalShard(index=index, count=count, source=source)


def default_eval_run_name(ckpt: str, preset: str, shard: EvalShard) -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if shard.enabled and job_id:
        suffix = f"job{job_id}"
    else:
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{Path(ckpt).stem}-{preset}-{suffix}"


def assigned_episode_indices(global_start: int, num_trials: int, shard: EvalShard) -> list[int]:
    return [episode_idx for episode_idx in range(num_trials) if shard.owns(global_start + episode_idx)]


def _row_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def merge_per_task_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("suite", "")),
            _row_int(row, "task_id"),
            _row_int(row, "eval_task_index"),
        )
        trials = _row_int(row, "num_trials")
        successes = _row_int(row, "num_success")
        avg_steps = _row_float(row, "avg_steps")
        if key not in merged:
            bucket = {field: row.get(field, "") for field in PER_TASK_FIELDNAMES}
            bucket["num_trials"] = trials
            bucket["num_success"] = successes
            bucket["_weighted_steps"] = avg_steps * trials
            merged[key] = bucket
        else:
            bucket = merged[key]
            bucket["num_trials"] = _row_int(bucket, "num_trials") + trials
            bucket["num_success"] = _row_int(bucket, "num_success") + successes
            bucket["_weighted_steps"] = _row_float(bucket, "_weighted_steps") + avg_steps * trials

    output: list[dict[str, Any]] = []
    for bucket in merged.values():
        trials = _row_int(bucket, "num_trials")
        successes = _row_int(bucket, "num_success")
        bucket["success_rate"] = float(successes / trials) if trials else 0.0
        bucket["avg_steps"] = float(_row_float(bucket, "_weighted_steps") / trials) if trials else 0.0
        bucket.pop("_weighted_steps", None)
        output.append(bucket)

    return sorted(
        output,
        key=lambda row: (
            LIBERO_SUITE_ORDER.get(str(row.get("suite", "")), 999),
            _row_int(row, "eval_task_index"),
            _row_int(row, "task_id"),
        ),
    )


def suite_results_from_rows(rows: list[dict[str, Any]], suites: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for suite in suites:
        suite_rows = [row for row in rows if str(row.get("suite", "")) == suite]
        trials = sum(_row_int(row, "num_trials") for row in suite_rows)
        successes = sum(_row_int(row, "num_success") for row in suite_rows)
        results[suite] = {
            "success_rate": float(successes / trials) if trials else 0.0,
            "num_trials": int(trials),
            "num_success": int(successes),
        }
    return results


def _compact_sr_results(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    compact: dict[str, dict[str, Any]] = {}
    for key, result in results.items():
        item = {
            "success_rate": float(result.get("success_rate", 0.0) or 0.0),
            "num_success": int(result.get("num_success", 0) or 0),
            "num_trials": int(result.get("num_trials", 0) or 0),
        }
        if "num_tasks" in result:
            item["num_tasks"] = int(result.get("num_tasks", 0) or 0)
        if "category" in result:
            item["category"] = str(result.get("category", ""))
        compact[str(key)] = item
    return compact


def _format_sr_result(label: str, result: dict[str, Any]) -> str:
    trials = int(result.get("num_trials", 0) or 0)
    successes = int(result.get("num_success", 0) or 0)
    rate = float(result.get("success_rate", 0.0) or 0.0)
    return f"{label}={rate:.1%}({successes}/{trials})"


def make_episode_progress_row(
    *,
    suite: str,
    task_id: int,
    eval_task_index: int,
    task_name: str,
    task_desc: str,
    entry: dict[str, Any],
    plus_perturbation: str,
    plus_official_task_id: Any,
    plus_official_category: str,
    plus_official_category_slug: str,
    plus_official_difficulty_level: Any,
    success: bool,
    steps: int,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "task_id": int(task_id),
        "eval_task_index": int(eval_task_index),
        "task_name": task_name,
        "plus_perturbation": plus_perturbation,
        "plus_official_task_id": plus_official_task_id,
        "plus_official_category": plus_official_category,
        "plus_official_category_slug": plus_official_category_slug,
        "plus_official_difficulty_level": plus_official_difficulty_level,
        "raw_task_language": str(entry.get("language", "")),
        "policy_language": task_desc,
        "bddl_file": str(entry.get("bddl_file", "")),
        "init_states_file": str(entry.get("init_states_file", "")),
        "num_trials": 1,
        "num_success": int(bool(success)),
        "success_rate": float(bool(success)),
        "avg_steps": float(steps),
    }


def emit_rollout_progress(
    *,
    progress_rows: list[dict[str, Any]],
    suites: list[str],
    plus: bool,
    run_name: str,
    eval_shard: EvalShard,
    progress_log_path: Path,
    suite: str,
    task_id: int,
    eval_task_index: int,
    episode_idx: int,
) -> None:
    suite_results = suite_results_from_rows(progress_rows, suites)
    total_trials = sum(int(result.get("num_trials", 0) or 0) for result in suite_results.values())
    total_successes = sum(int(result.get("num_success", 0) or 0) for result in suite_results.values())
    overall_success = float(total_successes / total_trials) if total_trials else 0.0
    plus_category_results = aggregate_plus_category_results(progress_rows) if plus else {}
    plus_perturbation_results = aggregate_plus_perturbation_results(progress_rows) if plus else {}

    payload = {
        "event": "libero_rollout_progress",
        "run_name": run_name,
        "scope": "shard" if eval_shard.enabled else "run",
        "shard": {
            "enabled": bool(eval_shard.enabled),
            "index": int(eval_shard.index),
            "count": int(eval_shard.count),
            "source": eval_shard.source,
        },
        "last_episode": {
            "suite": suite,
            "task_id": int(task_id),
            "eval_task_index": int(eval_task_index),
            "episode_idx": int(episode_idx),
        },
        "total_successes": int(total_successes),
        "total_episodes": int(total_trials),
        "overall_success_rate": float(overall_success),
        "suite_results": _compact_sr_results(suite_results),
        "plus_official_category_results": _compact_sr_results(plus_category_results),
        "plus_perturbation_results": _compact_sr_results(plus_perturbation_results),
    }

    suite_text = " ".join(_format_sr_result(key, value) for key, value in suite_results.items())
    category_text = " ".join(
        _format_sr_result(key, value) for key, value in plus_category_results.items()
    )
    perturbation_text = " ".join(
        _format_sr_result(key, value) for key, value in plus_perturbation_results.items()
    )
    scope = f"shard={eval_shard.index}/{eval_shard.count}" if eval_shard.enabled else "run"
    message = (
        f"    progress: {scope} episodes={total_trials} "
        f"overall={overall_success:.1%}({total_successes}/{total_trials}) "
        f"suite[{suite_text or 'n/a'}]"
    )
    if plus:
        message += f" official[{category_text or 'n/a'}] perturbation[{perturbation_text or 'n/a'}]"
    print(message, flush=True)
    json_line = json.dumps(payload, sort_keys=True)
    print("PROGRESS_LIBERO_RESULT_JSON=" + json_line, flush=True)
    with progress_log_path.open("a") as f:
        f.write(json_line + "\n")


def write_per_task_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TASK_FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in PER_TASK_FIELDNAMES} for row in rows])


def read_per_task_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate_shard_outputs(
    root_run_dir: Path,
    shard_count: int,
    timeout_sec: float,
    base_summary: dict[str, Any],
    suites: list[str],
    plus: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shard_root = root_run_dir / "shards"
    deadline = time.time() + float(timeout_sec)
    expected = [shard_root / f"shard{idx:04d}" for idx in range(shard_count)]
    while True:
        missing = [
            str(path)
            for path in expected
            if not (path / "summary.json").exists() or not (path / "per_task.csv").exists()
        ]
        if not missing:
            break
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for shard outputs: {missing}")
        time.sleep(10.0)

    shard_summaries = []
    all_rows: list[dict[str, Any]] = []
    for path in expected:
        with (path / "summary.json").open() as f:
            shard_summaries.append(json.load(f))
        all_rows.extend(read_per_task_csv(path / "per_task.csv"))

    merged_rows = merge_per_task_rows(all_rows)
    suite_results = suite_results_from_rows(merged_rows, suites)
    if plus:
        plus_by_suite = {
            suite: aggregate_plus_category_results([row for row in merged_rows if row.get("suite") == suite])
            for suite in suites
        }
        for suite, category_results in plus_by_suite.items():
            if suite in suite_results:
                suite_results[suite]["plus_official_category_results"] = category_results
        plus_results = aggregate_plus_category_results(merged_rows)
    else:
        plus_by_suite = {}
        plus_results = {}

    total_successes = int(sum(int(item["num_success"]) for item in suite_results.values()))
    total_episodes = int(sum(int(item["num_trials"]) for item in suite_results.values()))
    overall_success = float(total_successes / total_episodes) if total_episodes else 0.0
    average_success = float(np.mean([item["success_rate"] for item in suite_results.values()])) if suite_results else 0.0

    summary = dict(base_summary)
    summary.update(
        {
            "suite_results": suite_results,
            "average_success_rate": average_success,
            "overall_success_rate": overall_success,
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "plus_official_category_results": plus_results,
            "plus_official_category_results_by_suite": plus_by_suite,
            "action_timing": [
                item
                for shard_summary in shard_summaries
                for item in shard_summary.get("action_timing", [])
            ],
            "prompt_audit": [
                item
                for shard_summary in shard_summaries
                for item in shard_summary.get("prompt_audit", [])
            ],
            "elapsed_sec": max(float(base_summary.get("elapsed_sec", 0.0) or 0.0), time.time() - float(base_summary.get("_aggregate_start", time.time()))),
            "shard": {
                "enabled": True,
                "count": int(shard_count),
                "aggregated": True,
                "summary_paths": [str(path / "summary.json") for path in expected],
            },
        }
    )
    summary.pop("_aggregate_start", None)
    return summary, merged_rows


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env(env: Any, seed: int) -> None:
    for candidate in (env, getattr(env, "env", None), getattr(env, "base_env", None)):
        if candidate is None:
            continue
        seed_fn = getattr(candidate, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
            return


def image_from_obs(
    obs: dict[str, Any],
    key: str,
    image_size: tuple[int, int],
    rotate_for_policy: bool = False,
    train_crop_min_scale: float = 0.9,
    eval_crop_scale: float = 0.9,
    da3_input_vflip: bool = False,
    dataset_preprocess: bool = True,
    image_jpeg_eval_enabled: bool = False,
    image_jpeg_eval_quality: int = 95,
) -> torch.Tensor:
    img = obs.get(key)
    if img is None and key.endswith("_image"):
        img = obs.get(key[:-6])
    if img is None:
            raise KeyError(f"Observation image key '{key}' missing. Keys: {sorted(obs.keys())}")

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image for {key}, got shape {arr.shape}")

    if rotate_for_policy:
        # Diagnostic override: rotate both agentview and wrist inputs by 180 degrees.
        arr = np.asarray(_maybe_rotate_image_180(arr, enabled=True))
    elif any(stride < 0 for stride in arr.strides):
        arr = arr.copy()
    if dataset_preprocess:
        tensor = _normalize_image_tensor(
            arr,
            image_size=image_size,
            is_eval=True,
            train_crop_min_scale=float(train_crop_min_scale),
            eval_crop_scale=float(eval_crop_scale),
            jpeg_enabled=bool(image_jpeg_eval_enabled),
            jpeg_quality=int(image_jpeg_eval_quality),
        )
        if da3_input_vflip:
            tensor = torch.flip(tensor, dims=[-2])
        return tensor

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    # Non-policy display/debug path. Policy inputs must use
    # dataset_preprocess=True so crop+resize exactly matches training.
    pil = Image.fromarray(arr).resize(image_size, get_bicubic_resample())
    tensor = torch.from_numpy(np.array(pil, copy=True)).permute(2, 0, 1).float()
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    return tensor


def clamp_normalized_action_for_rollout(actions_norm_raw: torch.Tensor, normalizer: Any) -> torch.Tensor:
    """Return normalized rollout actions unchanged before denormalization."""
    return actions_norm_raw


def _normalize_rollout_camera_key(value: Any) -> str:
    key = str(value).strip()
    key = key.removeprefix("observation.images.")
    lower = key.lower()
    # LIBERO HDF5 training configs use dataset RGB keys, while live robosuite
    # rollout observations expose rendered image keys.
    libero_live_aliases = {
        "agentview": "agentview_image",
        "agentview_rgb": "agentview_image",
        "agentview_image": "agentview_image",
        "eye_in_hand": "robot0_eye_in_hand_image",
        "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        "eye_in_hand_image": "robot0_eye_in_hand_image",
        "robot0_eye_in_hand": "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_rgb": "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_image": "robot0_eye_in_hand_image",
    }
    if lower in libero_live_aliases:
        return libero_live_aliases[lower]
    if not key.endswith("_image"):
        key = f"{key}_image"
    return key


def rollout_camera_keys_from_dataset_cfg(dataset_cfg: dict[str, Any], n_views: int) -> tuple[str, ...]:
    raw_keys = (
        dataset_cfg.get("rollout_camera_keys")
        or dataset_cfg.get("rollout_camera_names")
        or dataset_cfg.get("camera_keys")
    )
    if raw_keys is None:
        keys = list(LIBERO_CAMERA_KEYS)
    elif isinstance(raw_keys, str):
        keys = [_normalize_rollout_camera_key(item) for item in raw_keys.split(",") if item.strip()]
    else:
        keys = [_normalize_rollout_camera_key(item) for item in raw_keys]
    if len(keys) != int(n_views):
        raise ValueError(f"Rollout expects n_views={n_views}, but camera keys are {keys}")
    return tuple(keys)


def resolve_policy_action_frame(value: Any = "auto", dataset_cfg: dict[str, Any] | None = None) -> str:
    """Resolve the model action frame used by a checkpoint/config."""
    raw = str(value or "auto").strip().lower().replace("-", "_")
    dataset_cfg = dataset_cfg or {}
    if raw in {"auto", ""}:
        raw = dataset_cfg.get(
            "rollout_action_frame",
            dataset_cfg.get("action_frame", dataset_cfg.get("action_output_frame", "base")),
        )
    return normalize_action_frame(raw)


def resolve_rollout_action_frame(value: Any = "auto", policy_info: dict[str, Any] | None = None) -> str:
    """Resolve a CLI/profile rollout action-frame override against policy metadata."""
    raw = str(value or "auto").strip().lower().replace("-", "_")
    if raw in {"auto", ""}:
        raw = (policy_info or {}).get("action_frame", "base")
    return normalize_action_frame(raw)


def images_from_obs(
    obs: dict[str, Any],
    image_size: tuple[int, int],
    camera_keys: Sequence[str] = LIBERO_CAMERA_KEYS,
    rotate_for_policy: bool = False,
    train_crop_min_scale: float = 0.9,
    eval_crop_scale: float = 0.9,
    da3_input_vflip: bool = False,
    dataset_preprocess: bool = True,
    image_jpeg_eval_enabled: bool = False,
    image_jpeg_eval_quality: int = 95,
) -> torch.Tensor:
    images = [
        image_from_obs(
            obs,
            key,
            image_size,
            rotate_for_policy=rotate_for_policy,
            train_crop_min_scale=train_crop_min_scale,
            eval_crop_scale=eval_crop_scale,
            da3_input_vflip=da3_input_vflip,
            dataset_preprocess=dataset_preprocess,
            image_jpeg_eval_enabled=image_jpeg_eval_enabled,
            image_jpeg_eval_quality=image_jpeg_eval_quality,
        )
        for key in camera_keys
    ]
    return torch.stack(images, dim=0)


def normalize_libero_prompt_for_dataset(value: Any) -> str:
    """Match LiberoHDF5SequenceDataset task text normalization."""
    text = str(value or "").strip()
    text = re.sub(r"_demo$", "", text)
    text = re.sub(r"^(?:[A-Z]+_)+SCENE\d+_", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return " ".join(text.lower().split())


def normalize_text_prompt_mode(mode: Any) -> str:
    value = str(mode or "libero_hdf5_task_text").strip().lower().replace("-", "_")
    aliases = {
        "libero": "libero_hdf5_task_text",
        "libero_hdf5": "libero_hdf5_task_text",
        "libero_hdf5_task": "libero_hdf5_task_text",
        "raw": "raw_task_text",
        "raw_task": "raw_task_text",
        "none": "raw_task_text",
    }
    value = aliases.get(value, value)
    if value not in {"libero_hdf5_task_text", "raw_task_text"}:
        raise ValueError(
            f"Unknown text prompt normalization mode {mode!r}; expected "
            "'libero_hdf5_task_text' or 'raw_task_text'."
        )
    return value


def normalize_text_prompt_for_policy(value: Any, mode: str) -> str:
    mode = normalize_text_prompt_mode(mode)
    if mode == "raw_task_text":
        return str(value or "").strip()
    # LIBERO HDF5 trains on lower-case, punctuation-stripped task strings.
    return normalize_libero_prompt_for_dataset(value)


def depths_from_obs(obs: dict[str, Any]) -> torch.Tensor | None:
    depths = []
    for camera_name in LIBERO_CAMERA_NAMES:
        depth = depth_from_obs(obs, f"{camera_name}_depth")
        if depth is None:
            return None
        depths.append(depth)
    return torch.stack(depths, dim=0)


def _unwrap_rollout_sim(env: Any) -> Any | None:
    """Locate the underlying robosuite/MuJoCo sim handle from a rollout env wrapper."""
    candidates = [env, getattr(env, "env", None), getattr(env, "base_env", None)]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.append(getattr(candidate, "env", None))
    for candidate in candidates:
        sim = getattr(candidate, "sim", None)
        if sim is not None:
            return sim
    return None


def rollout_depth_to_meters(env: Any, depth: np.ndarray) -> np.ndarray | None:
    """Convert raw MuJoCo OpenGL z-buffer depth (the value `RolloutEnv.render_depth`
    returns, in [0,1]) into metric meters via robosuite's `get_real_depth_map`.

    `depth` may be `(H, W)` or `(V, H, W)`. Returns a float32 array of the same shape,
    or None if the converter / sim handle is unavailable (caller then keeps raw depth).
    Mirrors `scripts/export_libero_aligned_gt_depth.py:convert_metric_depth`.
    """
    try:
        from robosuite.utils.camera_utils import get_real_depth_map
    except Exception:
        return None
    sim = _unwrap_rollout_sim(env)
    if sim is None:
        return None
    arr = np.asarray(depth, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    out = np.empty_like(flat)
    try:
        for i in range(flat.shape[0]):
            out[i] = np.asarray(get_real_depth_map(sim, flat[i]), dtype=np.float32).reshape(
                arr.shape[-2], arr.shape[-1]
            )
    except Exception:
        return None
    return out.reshape(arr.shape)


def as_batched_view_depth(depth: torch.Tensor | None, total_view: int) -> torch.Tensor | None:
    """Normalize DA3 depth output to `(1, total_view, H, W)` for debug panels."""
    if depth is None or total_view <= 0:
        return None
    d = depth.detach().cpu()
    if d.ndim == 4:
        if d.shape[0] == 1 and d.shape[1] == total_view:
            return d
        if d.shape[0] == total_view:
            return d.unsqueeze(0)
        if d.shape[1] == 1 and d.shape[0] == total_view:
            return d[:, 0].unsqueeze(0)
        if d.shape[0] * d.shape[1] == total_view:
            return d.reshape(1, total_view, *d.shape[-2:])
    if d.ndim == 3:
        if d.shape[0] == total_view:
            return d.unsqueeze(0)
        if d.shape[0] > total_view and d.shape[0] % total_view == 0:
            return d.reshape(-1, total_view, *d.shape[-2:])[:1]
    return None


def splice_direct_conditioning_depth(
    proxy_depth: torch.Tensor | None,
    direct_depth: torch.Tensor | None,
    total_view: int,
    cond_view_count: int,
) -> tuple[torch.Tensor | None, str]:
    """Use direct DA3 depth for observed slots and proxy depth for future slots."""
    proxy_bsv = as_batched_view_depth(proxy_depth, total_view)
    if proxy_bsv is None:
        direct_bsv = as_batched_view_depth(direct_depth, cond_view_count)
        return direct_bsv, "direct_conditioning_only" if direct_bsv is not None else "unavailable"
    direct_bsv = as_batched_view_depth(direct_depth, cond_view_count)
    if direct_bsv is None:
        return proxy_bsv, "proxy_all_slots"
    if proxy_bsv.shape[-2:] != direct_bsv.shape[-2:]:
        return proxy_bsv, "proxy_all_slots_shape_mismatch"
    n = min(int(cond_view_count), proxy_bsv.shape[1], direct_bsv.shape[1])
    if n <= 0:
        return proxy_bsv, "proxy_all_slots"
    merged = proxy_bsv.clone()
    merged[:, :n] = direct_bsv[:, :n].to(dtype=merged.dtype)
    return merged, "direct_conditioning_plus_proxy_future"


def normalize_proprio_orientation_mode(mode: str | None) -> str:
    value = str(mode or "rpy").strip().lower().replace("-", "_")
    aliases = {
        "euler": "rpy",
        "axisangle": "axis_angle",
        "axis_angle": "axis_angle",
        "rotvec": "axis_angle",
        "rotation_vector": "axis_angle",
    }
    value = aliases.get(value, value)
    if value not in {"rpy", "axis_angle"}:
        raise ValueError(f"Unknown proprio orientation mode {mode!r}; expected 'rpy' or 'axis_angle'.")
    return value


def resolve_proprio_orientation_mode(mode: str | None, dataset_cfg: dict[str, Any] | None = None) -> str:
    value = str(mode or "auto").strip().lower().replace("-", "_")
    if value not in {"auto", ""}:
        return normalize_proprio_orientation_mode(value)
    dataset_cfg = dataset_cfg or {}
    cfg_mode = dataset_cfg.get("rollout_proprio_orientation", dataset_cfg.get("proprio_orientation"))
    if cfg_mode is not None:
        return normalize_proprio_orientation_mode(str(cfg_mode))
    dataset_name = str(dataset_cfg.get("dataset_name", "")).lower()
    hdf5_root = str(dataset_cfg.get("hdf5_root", "")).lower()
    # Current regenerated LIBERO no-op HDF5s store obs/ee_states as
    # robosuite quat2axisangle, so live rollout must use the same convention.
    if "libero_noop" in dataset_name or "libero_noop" in hdf5_root or dataset_name.startswith("noop"):
        return "axis_angle"
    return "rpy"


def _quat_xyzw_to_robosuite_axis_angle(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Match robosuite `quat2axisangle` for xyzw quaternions.

    Preserve `qw` sign because the no-op HDF5 regeneration script stored
    robosuite's axis-angle convention exactly, including angles in [0, 2*pi].
    """
    q = q_xyzw / q_xyzw.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    xyz = q[..., :3]
    qw = q[..., 3:4].clamp(-1.0, 1.0)
    den = torch.sqrt((1.0 - qw * qw).clamp(min=0.0))
    angle = 2.0 * torch.acos(qw)
    axis_angle = xyz * angle / den.clamp(min=1e-12)
    return torch.where(den > 1e-8, axis_angle, torch.zeros_like(axis_angle))


def obs_to_canonical_7d_proprio(obs: dict[str, Any], orientation_mode: str = "rpy") -> torch.Tensor:
    """Convert live LIBERO robosuite obs to [pos(3), orient(3), grip_width(1)]."""
    missing = [key for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos") if key not in obs]
    if missing:
        raise KeyError(f"Missing proprio keys in LIBERO obs: {missing}. Keys: {sorted(obs.keys())}")

    orientation_mode = normalize_proprio_orientation_mode(orientation_mode)
    pos = torch.as_tensor(np.asarray(obs["robot0_eef_pos"], dtype=np.float32)).flatten()[:3]
    quat = torch.as_tensor(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)).flatten()[:4]
    if orientation_mode == "axis_angle":
        orient = _quat_xyzw_to_robosuite_axis_angle(quat.view(1, 4)).view(3)
    else:
        orient = _quat_xyzw_to_rpy(quat.view(1, 4)).view(3)
    qpos = torch.as_tensor(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)).flatten()
    if qpos.numel() < 2:
        raise ValueError(f"Expected two LIBERO gripper qpos values, got shape {tuple(qpos.shape)}")
    grip_width = (qpos[0:1] - qpos[1:2]).to(dtype=torch.float32)
    return torch.cat([pos.to(torch.float32), orient.to(torch.float32), grip_width], dim=0)


def select_live_proprio_converter(dataset_cfg: dict[str, Any]) -> Callable[[dict[str, Any], str], torch.Tensor]:
    del dataset_cfg
    return obs_to_canonical_7d_proprio


def canonical_to_libero_action(
    action: np.ndarray | torch.Tensor,
    binarize_gripper: bool = True,
    action_frame: str = "base",
    current_proprio: torch.Tensor | np.ndarray | None = None,
    anchor_proprio: torch.Tensor | np.ndarray | None = None,
    proprio_orientation: str = "rpy",
) -> np.ndarray:
    """Invert training transform before `env.step()`.

    `action_frame=base` / `base_delta` preserves the historical LIBERO
    base/world delta convention.
    `eef_delta` is the old moving-frame local delta. `eef_relative` is a
    chunk-start relative target trajectory; we first reconstruct the world
    target from the policy-call anchor, then compute the executable delta from
    the actual current pose.
    """
    if isinstance(action, torch.Tensor):
        arr = action.detach().cpu().float().numpy()
    else:
        arr = np.asarray(action, dtype=np.float32).copy()
    arr = arr.astype(np.float32, copy=True)
    if arr.shape[-1] != 7:
        raise ValueError(f"Expected 7D action, got shape {arr.shape}")
    frame = normalize_action_frame(action_frame)
    if frame == "eef_delta":
        if current_proprio is None:
            raise ValueError("action_frame='eef_delta' rollout requires current_proprio.")
        orig_shape = arr.shape
        world_action = eef_delta_to_world_delta_action_for_rollout(
            torch.as_tensor(arr, dtype=torch.float32),
            torch.as_tensor(current_proprio, dtype=torch.float32),
            proprio_orientation=proprio_orientation,
        )
        arr = world_action.detach().cpu().numpy().reshape(orig_shape).astype(np.float32, copy=False)
    elif frame == "eef_relative":
        if current_proprio is None or anchor_proprio is None:
            raise ValueError(
                "action_frame='eef_relative' rollout requires both anchor_proprio and current_proprio."
            )
        orig_shape = arr.shape
        world_action = eef_relative_trajectory_to_world_delta_action_for_rollout(
            torch.as_tensor(arr, dtype=torch.float32),
            torch.as_tensor(anchor_proprio, dtype=torch.float32),
            torch.as_tensor(current_proprio, dtype=torch.float32),
            proprio_orientation=proprio_orientation,
        )
        arr = world_action.detach().cpu().numpy().reshape(orig_shape).astype(np.float32, copy=False)
    close = arr[..., -1]
    if binarize_gripper:
        close = (close > 0.5).astype(np.float32)
    else:
        close = np.clip(close, 0.0, 1.0)
    arr[..., -1] = 2.0 * close - 1.0
    return arr


def canonical_action_for_history(
    action: np.ndarray | torch.Tensor,
    binarize_gripper: bool = True,
) -> torch.Tensor:
    """Canonical action actually executed, with gripper post-processing applied."""
    action_t = torch.as_tensor(action, dtype=torch.float32).detach().clone()
    close = action_t[..., -1]
    if binarize_gripper:
        close = (close > 0.5).to(dtype=action_t.dtype)
    else:
        close = close.clamp(0.0, 1.0)
    action_t[..., -1] = close
    return action_t


def _trace_json_value(value: Any, *, max_elements: int = 2048) -> Any:
    """Convert small numeric rollout diagnostics to JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        if arr.numel() > max_elements:
            flat = arr.float().reshape(-1)
            return {
                "shape": list(arr.shape),
                "numel": int(arr.numel()),
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
                "min": float(flat.min().item()),
                "max": float(flat.max().item()),
            }
        return arr.tolist()
    if isinstance(value, np.ndarray):
        if value.size > max_elements:
            flat = value.astype(np.float32, copy=False).reshape(-1)
            return {
                "shape": list(value.shape),
                "numel": int(value.size),
                "mean": float(flat.mean()),
                "std": float(flat.std()),
                "min": float(flat.min()),
                "max": float(flat.max()),
            }
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _trace_json_value(v, max_elements=max_elements) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_json_value(v, max_elements=max_elements) for v in value]
    return str(value)


def append_rollout_debug_event(
    path: str | Path | None,
    context: dict[str, Any] | None,
    record: dict[str, Any],
) -> None:
    """Append one flushed lifecycle debug event for native-abort diagnosis."""
    if path is None:
        return
    debug_path = Path(path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": os.environ.get("HOSTNAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "slurm_localid": os.environ.get("SLURM_LOCALID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        **(context or {}),
        **record,
    }
    with debug_path.open("a", buffering=1) as f:
        f.write(json.dumps(_trace_json_value(payload), ensure_ascii=False) + "\n")
        f.flush()


def _small_numeric_obs_summary(obs: dict[str, Any]) -> dict[str, Any]:
    """Capture non-image robosuite obs fields that help diagnose grasp failures."""
    summary: dict[str, Any] = {}
    for key, value in obs.items():
        key_l = str(key).lower()
        if "image" in key_l or "depth" in key_l or "seg" in key_l:
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if not np.issubdtype(arr.dtype, np.number) or arr.size > 32:
            continue
        summary[str(key)] = _trace_json_value(arr.astype(np.float32, copy=False))
    if "robot0_gripper_qpos" in obs:
        qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if qpos.size >= 2:
            summary["robot0_gripper_width"] = float(qpos[0] - qpos[1])
    return summary


def _policy_debug_trace_snapshot(debug: Any) -> dict[str, Any]:
    """Keep action/proprio/prompt rollout internals, excluding heavy image/depth tensors."""
    if not isinstance(debug, dict):
        return {}
    keep_keys = (
        "proprio_raw",
        "proprio_history_raw",
        "actions",
        "action_chunks",
        "actions_norm",
        "actions_norm_raw",
        "actions_full",
        "action_chunks_full",
        "actions_norm_full",
        "actions_norm_raw_full",
        "actions_norm_clamped",
        "actions_norm_max_abs_raw",
        "history_horizon",
        "effective_history_horizon",
        "predicted_steps",
        "executed_model_steps",
        "executed_sequence_start",
        "rollout_decode_horizon",
        "rollout_decode_horizon_mode",
        "rollout_decode_horizon_requested",
        "history_committed_entries",
        "history_commit_stride_actions",
        "history_commit_stride_env_actions",
        "env_actions_per_model_step",
        "rollout_visual_contract",
        "predicted_sequence_start_timestep",
        "observed_view_count",
        "total_view",
        "cond_num",
        "rotate_policy_input",
        "dataset_da3_input_rotate180",
        "dataset_da3_input_hflip",
        "da3_input_vflip",
        "policy_image_preprocess",
        "eval_crop_scale",
        "text_prompt_audit",
        "temporal_ensemble",
        "temporal_ensemble_candidates",
        "temporal_ensemble_decay",
        "temporal_ensemble_forecast_horizon",
        "temporal_ensemble_model_steps",
        "temporal_ensemble_chunk_size",
        "temporal_ensemble_unit",
    )
    return {key: _trace_json_value(debug.get(key)) for key in keep_keys if key in debug}


def _tensor_lifecycle_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value).detach().cpu().float()
    except Exception:
        return None
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "numel": 0}
    flat = tensor.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
    }


def _policy_debug_lifecycle_summary(debug: Any) -> dict[str, Any]:
    """Small policy-forward summary for lifecycle logs; action traces keep details."""
    if not isinstance(debug, dict):
        return {}
    keep_keys = (
        "history_horizon",
        "effective_history_horizon",
        "predicted_steps",
        "executed_model_steps",
        "executed_sequence_start",
        "rollout_decode_horizon",
        "rollout_decode_horizon_mode",
        "history_committed_entries",
        "history_commit_stride_actions",
        "history_commit_stride_env_actions",
        "env_actions_per_model_step",
        "rollout_visual_contract",
        "observed_view_count",
        "total_view",
        "cond_num",
        "actions_norm_clamped",
        "actions_norm_max_abs_raw",
        "eval_crop_scale",
        "text_prompt_audit",
        "temporal_ensemble",
        "temporal_ensemble_candidates",
    )
    out = {key: _trace_json_value(debug.get(key)) for key in keep_keys if key in debug}
    for key in (
        "actions",
        "action_chunks",
        "actions_norm",
        "actions_norm_raw",
        "action_chunks_full",
        "actions_norm_full",
        "actions_norm_raw_full",
        "proprio_raw",
    ):
        summary = _tensor_lifecycle_summary(debug.get(key))
        if summary is not None:
            out[f"{key}_summary"] = summary
    return out


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}


def resolve_existing_path(path_value: str | None, base_dir: Path | None = None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return str(path)
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return str(candidate)
    return str(path)


def _portable_checkpoint_roots(base_dir: Path | None = None) -> list[Path]:
    roots: list[Path] = []

    def _append(root: Path | None) -> None:
        if root is None:
            return
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            resolved = root.expanduser()
        if resolved not in roots:
            roots.append(resolved)

    if base_dir is not None:
        _append(base_dir)
        _append(base_dir.parent)
    _append(_REPO_ROOT)
    _append(_REPO_ROOT.parent)
    _append(Path.cwd())
    _append(Path.cwd().parent)

    for env_key in ("DA3_CODE_ROOT", "DA3_PROJECT_ROOT", "DA3_CHECKPOINT_DIR"):
        raw = os.environ.get(env_key)
        if raw:
            _append(Path(raw))

    expanded: list[Path] = []
    for root in roots:
        if root not in expanded:
            expanded.append(root)
        checkpoints_dir = root / "checkpoints"
        if checkpoints_dir not in expanded:
            expanded.append(checkpoints_dir)
    return expanded


def resolve_portable_checkpoint_path(
    path_value: str | None,
    *,
    base_dir: Path | None = None,
    label: str = "checkpoint",
) -> str | None:
    resolved = resolve_existing_path(path_value, base_dir)
    if resolved is None:
        return None
    resolved_path = Path(resolved)
    if resolved_path.exists():
        return str(resolved_path)

    basename = Path(path_value).name if path_value else ""
    if basename:
        for root in _portable_checkpoint_roots(base_dir):
            candidate = root / basename
            if candidate.exists():
                print(
                    f"[eval_libero_unified] INFO: {label} path {path_value!r} missing; "
                    f"using local candidate {candidate}"
                )
                return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve {label} path {path_value!r}. "
        f"Tried direct/base-dir resolution plus basename search under "
        f"{[str(p) for p in _portable_checkpoint_roots(base_dir)]}."
    )


def detect_stage_from_checkpoint(ckpt: dict[str, Any]) -> str:
    if "student_da3" in ckpt and "action_head" in ckpt:
        return "1"
    return "unknown"


def action_stats_key_candidates_from_config(dataset_cfg: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        key = str(value).strip()
        if key and key not in candidates:
            candidates.append(key)

    add(dataset_cfg.get("action_stats_key"))
    add(dataset_cfg.get("dataset_name"))
    add(dataset_cfg.get("name"))

    dataset_type = str(dataset_cfg.get("type", "")).lower()
    preset = str(dataset_cfg.get("preset", "")).lower()
    openx_root = str(dataset_cfg.get("openx_root", "")).lower()
    hdf5_root = str(dataset_cfg.get("hdf5_root", "")).lower()
    if dataset_type in {"libero_hdf5", "hdf5_libero"} or "libero_hdf5" in hdf5_root:
        add("libero_hdf5")
    if dataset_type == "libero" or preset == "libero" or "libero" in openx_root:
        add("libero")
    if bool(dataset_cfg.get("plus", False)) or "libero_plus" in preset or "libero_plus" in openx_root:
        add("libero_plus")
    return candidates


def choose_action_stats_key(
    normalizer: ActionNormalizer,
    requested: str | None,
    preferred_keys: list[str] | tuple[str, ...] | None = None,
) -> str:
    if requested:
        if requested not in normalizer.stats_by_key:
            raise KeyError(f"Requested action stats key '{requested}' not in {sorted(normalizer.stats_by_key)}")
        return requested
    for key in preferred_keys or ():
        if key in normalizer.stats_by_key:
            return key
    if normalizer.default_key in normalizer.stats_by_key:
        return normalizer.default_key
    for key in ("libero_hdf5", "libero", "libero_plus"):
        if key in normalizer.stats_by_key:
            return key
    return normalizer.default_key


def build_action_normalizer(
    ckpt: dict[str, Any],
    action_stats_json: str | None = None,
) -> ActionNormalizer:
    if "action_normalizer" in ckpt:
        return ActionNormalizer.from_state_dict(ckpt["action_normalizer"])

    if action_stats_json:
        with open(action_stats_json, "r") as f:
            state = json.load(f)
        try:
            return ActionNormalizer.from_state_dict(state)
        except ValueError:
            return ActionNormalizer(state)

    raise KeyError(
        "Checkpoint has no action_normalizer. Pass --action-stats-json with a "
        "compatible ActionNormalizer state, or use a newer Stage 1 checkpoint."
    )


def infer_policy_target_hz(dataset_cfg: dict[str, Any]) -> float | None:
    """Infer policy action Hz for LIBERO rollout timing.

    `dataset.target_hz` is optional in native-Hz training configs. The
    `yifengzhu-hf` LIBERO HDF5 demos and the live robosuite LIBERO env
    both run at `control_freq=20` (verified directly from `data.attrs.env_args`
    on 2026-04-24). Older lerobot HF parquet conversions of LIBERO are
    sometimes downsampled to 10Hz; treat `dataset.type == "libero"` (the
    LeRobot path) as 10Hz and `libero_hdf5` as native 20Hz.
    """
    raw_target_hz = dataset_cfg.get("target_hz")
    if raw_target_hz is not None:
        hz = float(raw_target_hz)
        return hz if math.isfinite(hz) and hz > 0 else None
    dataset_type = str(dataset_cfg.get("type", "")).lower()
    preset = str(dataset_cfg.get("preset", "")).lower()
    openx_root = str(dataset_cfg.get("openx_root", "")).lower()
    if dataset_type in {"libero_hdf5", "hdf5_libero"}:
        return 20.0
    if dataset_type == "libero" or preset == "libero" or "libero" in openx_root:
        return 10.0
    return None


def infer_dataset_da3_input_rotate180(dataset_cfg: dict[str, Any]) -> bool:
    raw = dataset_cfg.get("da3_input_rotate180")
    if raw is not None:
        return bool(raw)
    dataset_type = str(dataset_cfg.get("type", "")).lower()
    if dataset_type in {"libero_hdf5", "hdf5_libero"}:
        return True
    return False


def infer_dataset_da3_input_hflip(dataset_cfg: dict[str, Any]) -> bool:
    return infer_libero_hdf5_hflip(
        dataset_cfg.get("da3_input_hflip"),
        dataset_cfg.get("dataset_name"),
        dataset_cfg.get("hdf5_root"),
    )


def infer_libero_hdf5_env_rotate180(
    dataset_cfg: dict[str, Any],
    dataset_da3_input_rotate180: bool,
    dataset_da3_input_hflip: bool,
    rotate_policy_input: bool,
) -> bool:
    """Backward-compatible alias for the live-env horizontal-flip contract."""
    return infer_libero_hdf5_env_hflip(
        dataset_cfg,
        dataset_da3_input_rotate180,
        dataset_da3_input_hflip,
        rotate_policy_input,
    )


def infer_libero_hdf5_env_hflip(
    dataset_cfg: dict[str, Any],
    dataset_da3_input_rotate180: bool,
    dataset_da3_input_hflip: bool,
    rotate_policy_input: bool,
) -> bool:
    """Return the canonical extra live-env horizontal flip for HDF5 policies.

    The rollout env wrapper always applies the OpenGL bottom-up vertical flip.
    Raw LIBERO HDF5 then needs one extra horizontal flip to reproduce the
    historical rotate180 train-frame contract. Replayed ``libero_noop`` HDF5s
    can also request the same extra horizontal flip explicitly via
    ``dataset.da3_input_hflip``.
    """
    dataset_type = str(dataset_cfg.get("type", "")).lower()
    hdf5_root = str(dataset_cfg.get("hdf5_root", "")).lower()
    is_hdf5 = dataset_type in {"libero_hdf5", "hdf5_libero"} or "libero_hdf5" in hdf5_root
    needs_hflip = bool(dataset_da3_input_rotate180 or dataset_da3_input_hflip)
    return bool(is_hdf5 and needs_hflip and not rotate_policy_input)


def _log_incompatible_keys(component: str, incompatible: Any) -> tuple[list[str], list[str]]:
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        print(
            f"[eval_libero_unified] WARN: {component} load_state_dict strict=False "
            f"missing={missing} unexpected={unexpected}"
        )
    return missing, unexpected


def _stage1_ema_metadata(ckpt: dict[str, Any]) -> dict[str, Any]:
    meta = ckpt.get("ema")
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if k != "shadow"}


def _require_stage1_ema_state(ckpt: dict[str, Any], key: str) -> dict[str, Any]:
    state = ckpt.get(key)
    if state is None:
        available = sorted(k for k in ckpt.keys() if k.endswith("_ema") or k == "ema")
        raise RuntimeError(
            f"--use-ema requested, but checkpoint has no `{key}` state. "
            f"Available EMA keys: {available}. Re-run training with training.ema.enabled=true."
        )
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint `{key}` to be a state_dict, got {type(state).__name__}.")
    return state


def load_stage1_policy(
    cfg: Any,
    ckpt: dict[str, Any],
    ckpt_path: str,
    device: torch.device,
    stats_key: str | None,
    action_stats_json: str | None,
    decode_visuals: bool = False,
    history_horizon: str | int | None = "auto",
    rollout_decode_horizon: str | int | None = "full",
    rotate_policy_input: bool = False,
    proprio_orientation: str = "auto",
    text_prompt_normalization: str = "libero_hdf5_task_text",
    config_source: str = "cli",
    preloaded_modules: dict[str, Any] | None = None,
    use_ema: bool = False,
) -> tuple[Callable[[dict[str, Any]], torch.Tensor], dict[str, Any]]:
    """Build the LIBERO rollout policy closure.

    When `preloaded_modules` is provided (during in-training closed-loop eval),
    the existing in-memory modules are reused instead of loading from disk.
    Keys consumed, all optional and built from ckpt if absent:
        teacher, student, action_head, proprio_conditioner, future_predictor,
        text_conditioner, action_normalizer, proprio_normalizer.
    The ckpt dict is still consulted for config + fallback weights for any
    unprovided module. When `preloaded_modules` is None this function's
    behaviour is unchanged.
    """
    repo_root = _REPO_ROOT
    if preloaded_modules is None:
        preloaded_modules = {}
    text_prompt_normalization = normalize_text_prompt_mode(text_prompt_normalization)

    def _cfg_section(name: str) -> dict[str, Any]:
        section = cfg.get(name, {}) if isinstance(cfg, dict) else cfg.get(name, {})
        if OmegaConf.is_config(section):
            section = OmegaConf.to_container(section, resolve=True)
        if section is None:
            return {}
        if not isinstance(section, dict):
            raise TypeError(f"Expected cfg['{name}'] to be a dict-like object, got {type(section).__name__}.")
        return dict(section)

    stage1_cfg = _cfg_section("stage_1")
    da3_ft_cfg = _cfg_section("da3_finetune")
    action_head_cfg = _cfg_section("action_head")
    proprio_cfg = _cfg_section("proprioception")
    predictor_cfg = _cfg_section("predictor")
    dataset_cfg = _cfg_section("dataset")
    training_cfg = _cfg_section("training")

    # GAM AR rollout action source: "refine" (default) reads the action
    # from the DA3 deep-refined action token; "direct" reads it straight from
    # the predictor's per-step action token (matches a training run with
    # lambda_action_refine=0 / lambda_action_direct>0, where the deep-refine
    # action head is never supervised). Captured by both _decode_gam_*
    # closures below.
    rollout_action_source = str(
        predictor_cfg.get("rollout_action_source", "refine")
    ).lower()
    if rollout_action_source not in {"refine", "direct"}:
        raise ValueError(
            f"Unknown predictor.rollout_action_source={rollout_action_source!r}. "
            "Expected 'refine' or 'direct'."
        )

    n_views = int(da3_ft_cfg.get("n_views", 2))
    rollout_camera_keys = rollout_camera_keys_from_dataset_cfg(dataset_cfg, n_views)

    future_steps = int(dataset_cfg.get("future_steps", 6))
    include_current = bool(dataset_cfg.get("include_current_action", True))
    action_steps = future_steps + 1 if include_current else future_steps
    chunk_size = int(action_head_cfg.get("chunk_size", 1))
    use_temporal = bool(da3_ft_cfg.get("use_temporal_embed", False))
    use_bf16 = bool(training_cfg.get("bf16", True)) and torch.cuda.is_available()
    image_size = tuple(int(x) for x in dataset_cfg.get("image_size", [224, 224]))
    image_aug_cfg = resolve_image_augmentation_config(dataset_cfg)
    train_crop_min_scale = float(image_aug_cfg["train_crop_min_scale"])
    eval_crop_scale = float(image_aug_cfg["eval_crop_scale"])
    image_jpeg_eval_enabled = bool(image_aug_cfg["image_jpeg_eval_enabled"])
    image_jpeg_eval_quality = int(image_aug_cfg["image_jpeg_eval_quality"])
    dataset_da3_input_rotate180 = infer_dataset_da3_input_rotate180(dataset_cfg)
    dataset_da3_input_hflip = infer_dataset_da3_input_hflip(dataset_cfg)
    da3_input_vflip = bool(dataset_cfg.get("da3_input_vflip", False))
    proprio_orientation = resolve_proprio_orientation_mode(proprio_orientation, dataset_cfg)
    action_frame = resolve_policy_action_frame("auto", dataset_cfg)
    live_proprio_converter = select_live_proprio_converter(dataset_cfg)
    da3_ckpt_path = resolve_portable_checkpoint_path(
        stage1_cfg.get("ckpt_path"),
        base_dir=repo_root,
        label="stage1.da3_ckpt",
    )
    stage1_cfg_for_backbone = dict(stage1_cfg)
    stage1_cfg_for_backbone["ckpt_path"] = da3_ckpt_path
    backbone_type = stage1_backbone_type(stage1_cfg_for_backbone)
    if backbone_type != "da3" and not bool(predictor_cfg.get("enabled", False)):
        raise ValueError(
            "Non-DA3 Stage 1 backbones require predictor.enabled=true in LIBERO unified eval."
        )

    teacher = preloaded_modules.get("teacher")
    if teacher is None:
        teacher = create_stage1_backbone(
            stage1_cfg_for_backbone,
            freeze_backbone=True,
        ).to(device).eval()

    student = preloaded_modules.get("student")
    if student is None:
        student = create_stage1_backbone(
            stage1_cfg_for_backbone,
            freeze_backbone=False,
            n_action_steps=action_steps,
            views_per_timestep=n_views,
            action_steps_per_token=chunk_size,
            use_temporal_embed=use_temporal,
            action_input_rate=float(da3_ft_cfg.get("action_input_rate", 0.4)),
        ).to(device)

    proprio_cond = preloaded_modules.get("proprio_conditioner")
    if (
        proprio_cond is None
        and bool(proprio_cfg.get("enabled", True))
        and not bool(predictor_cfg.get("enabled", False))
    ):
        proprio_cond = ProprioConditioner(
            proprio_dim=int(proprio_cfg.get("proprio_dim", 7)),
            hidden_dim=int(proprio_cfg.get("hidden_dim", 256)),
            out_dim=student.embed_dim,
        ).to(device)

    action_head = preloaded_modules.get("action_head")
    if action_head is None:
        raw_action_head_input_dim = action_head_cfg.get("input_dim", student.embed_dim)
        if str(raw_action_head_input_dim).lower() in {"auto", "da3", "backbone", "embed_dim"}:
            action_head_input_dim = int(student.embed_dim)
        else:
            action_head_input_dim = int(raw_action_head_input_dim)
        if action_head_input_dim != int(student.embed_dim):
            if action_head_input_dim == 1536:
                action_head_input_dim = int(student.embed_dim)
            else:
                raise ValueError(
                    f"action_head.input_dim={action_head_input_dim} must match "
                    f"selected Stage 1 embed_dim={int(student.embed_dim)}."
                )
        action_head = ActionHeadV2(
            input_dim=action_head_input_dim,
            n_views=n_views,
            hidden_dim=int(action_head_cfg.get("hidden_dim", student.embed_dim)),
            n_dims=int(action_head_cfg.get("n_dims", 7)),
            chunk_size=chunk_size,
            num_blocks=int(action_head_cfg.get("num_blocks", 2)),
            pool_mode=str(action_head_cfg.get("pool_mode", "mean")),
            chunk_position_encoding=str(action_head_cfg.get("chunk_position_encoding", "none")),
        ).to(device)

    student_da3_missing_keys: list[str] = []
    student_da3_unexpected_keys: list[str] = []
    proprio_conditioner_missing_keys: list[str] = []
    proprio_conditioner_unexpected_keys: list[str] = []
    text_conditioner_proj_missing_keys: list[str] = []
    text_conditioner_proj_unexpected_keys: list[str] = []
    ema_loaded_keys: list[str] = []
    ema_meta = _stage1_ema_metadata(ckpt)
    ema_available = any(
        ckpt.get(key) is not None
        for key in (
            "student_da3_ema",
            "action_head_ema",
            "future_predictor_ema",
            "text_conditioner_proj_ema",
        )
    )
    if use_ema and preloaded_modules:
        raise RuntimeError(
            "--use-ema is supported when loading Stage 1 modules from checkpoint. "
            "In-training closed-loop eval reuses live modules; EMA overlays are checkpoint-only."
        )
    model = DA3FineTuneModel(student, action_head, proprio_cond).to(device)
    # When preloaded_modules supplies weights, skip the ckpt-driven load.
    _use_preloaded_core = bool(preloaded_modules.get("student") is not None
                                and preloaded_modules.get("action_head") is not None)
    if not _use_preloaded_core:
        if "student_da3" not in ckpt or "action_head" not in ckpt:
            raise KeyError(f"{ckpt_path} is outside the Stage 1 checkpoint format.")
        s1_load = model.student_da3.load_state_dict(strip_compile_prefix(ckpt["student_da3"]), strict=False)
        student_da3_missing_keys, student_da3_unexpected_keys = _log_incompatible_keys(
            "stage1.student_da3",
            s1_load,
        )
        model.action_head.load_state_dict(strip_compile_prefix(ckpt["action_head"]))
        if use_ema:
            s1_ema_load = model.student_da3.load_state_dict(
                strip_compile_prefix(_require_stage1_ema_state(ckpt, "student_da3_ema")),
                strict=False,
            )
            _log_incompatible_keys("stage1.student_da3_ema", s1_ema_load)
            action_ema_load = model.action_head.load_state_dict(
                strip_compile_prefix(_require_stage1_ema_state(ckpt, "action_head_ema")),
                strict=False,
            )
            _log_incompatible_keys("stage1.action_head_ema", action_ema_load)
            ema_loaded_keys.extend(["student_da3_ema", "action_head_ema"])
        if ckpt.get("proprio_conditioner") is not None and model.proprio_conditioner is not None:
            proprio_load = model.proprio_conditioner.load_state_dict(
                strip_compile_prefix(ckpt["proprio_conditioner"]),
                strict=False,
            )
            proprio_conditioner_missing_keys, proprio_conditioner_unexpected_keys = _log_incompatible_keys(
                "stage1.proprio_conditioner",
                proprio_load,
            )
    model.eval()
    model.requires_grad_(False)
    teacher.requires_grad_(False)

    normalizer = preloaded_modules.get("action_normalizer")
    if normalizer is None:
        normalizer = build_action_normalizer(ckpt, action_stats_json=action_stats_json)
    preferred_stats_keys = action_stats_key_candidates_from_config(dataset_cfg)
    chosen_stats_key = choose_action_stats_key(normalizer, stats_key, preferred_stats_keys)
    print(
        "[eval_libero_unified] action_stats_key=%s requested=%s preferred=%s default=%s available=%s"
        % (
            chosen_stats_key,
            stats_key,
            preferred_stats_keys,
            normalizer.default_key,
            sorted(normalizer.stats_by_key.keys()),
        )
    )
    encoder_mean = teacher.encoder_mean.float().to(device)
    encoder_std = teacher.encoder_std.float().to(device)
    proprio_normalizer = preloaded_modules.get("proprio_normalizer")
    if proprio_normalizer is None and ckpt.get("proprio_normalizer") is not None:
        proprio_normalizer = StateNormalizer.from_state_dict(ckpt["proprio_normalizer"])

    predictor_H_choices = [int(x) for x in predictor_cfg.get("H_choices", []) if int(x) > 0]
    # In gam training, H is capped at n_action_steps - 1 because the
    # future visual/proprio target is obs_{t+1:t+H}. The current no-op run has
    # action_steps=9 with H_choices=[8], so "full" rollout decode means 8.
    stage1_native_train_horizon = max(
        1,
        min(
            max(1, action_steps - 1),
            max(predictor_H_choices) if predictor_H_choices else max(1, action_steps - 1),
        ),
    )

    def resolve_history_horizon(value: str | int | None) -> int:
        raw = "auto" if value is None else str(value).strip().lower()
        if raw in {"auto", ""}:
            _has_fp = bool(predictor_cfg.get("enabled", False)) and (
                ckpt.get("future_predictor") is not None
                or preloaded_modules.get("future_predictor") is not None
            )
            if _has_fp:
                choices = predictor_H_choices or [stage1_native_train_horizon]
                return max(1, min(stage1_native_train_horizon, max(choices)))
            return 1
        resolved = int(raw)
        if resolved < 1:
            raise ValueError(f"--history-horizon must be >=1 or auto, got {value!r}")
        return max(1, min(stage1_native_train_horizon, resolved))

    stage1_history_horizon = resolve_history_horizon(history_horizon)

    def resolve_rollout_decode_horizon(value: str | int | None) -> tuple[int | None, str]:
        raw = "full" if value is None else str(value).strip().lower()
        if raw in {"", "auto", "full", "train", "train_like"}:
            return stage1_native_train_horizon, "full"
        if raw in {"exec", "execute", "executed", "action_horizon", "h_exec"}:
            return None, "exec"
        resolved = int(raw)
        if resolved < 1:
            raise ValueError(f"--rollout-decode-horizon must be >=1, full, or exec; got {value!r}")
        return max(1, min(stage1_native_train_horizon, resolved)), raw

    stage1_rollout_decode_horizon, stage1_rollout_decode_horizon_mode = resolve_rollout_decode_horizon(
        rollout_decode_horizon
    )

    _preloaded_fp = preloaded_modules.get("future_predictor")
    predictor_enabled = bool(predictor_cfg.get("enabled", False)) and (
        ckpt.get("future_predictor") is not None or _preloaded_fp is not None
    )
    predictor_type = str(predictor_cfg.get("type", predictor_cfg.get("architecture", "gam"))).lower()
    future_predictor = None
    text_conditioner = None
    text_cache: dict[str, dict[str, torch.Tensor]] = {}
    future_predictor_missing_keys: list[str] = []
    future_predictor_unexpected_keys: list[str] = []
    predictor_num_register_tokens = int(
        predictor_cfg.get("num_register_tokens", student.num_register_tokens)
    )
    if predictor_enabled:
        if _preloaded_fp is not None:
            future_predictor = _preloaded_fp
        else:
            predictor_build_cfg = dict(predictor_cfg)
            for legacy_key in ("action_seed_mode", "use_action_history", "predict_future_proprio", "deep_context_steps", "deep_context_full_prob", "attention_fp32", "max_timesteps", "max_views"):
                predictor_build_cfg.pop(legacy_key, None)
            predictor_build_cfg.update(
                {
                    "d_da3": student.embed_dim,
                    "proprio_dim": int(proprio_cfg.get("proprio_dim", 7)),
                    "action_dim": int(action_head_cfg.get("n_dims", 7)),
                    "action_chunk_size": chunk_size,
                    "num_patches_per_view": int(
                        predictor_build_cfg.get(
                            "num_patches_per_view",
                            int(getattr(student, "num_patches", 0))
                            or (stage1_cfg.get("encoder_input_size", 224) // int(getattr(student, "patch_size", 14))) ** 2,
                        )
                    ),
                    "num_register_tokens": predictor_num_register_tokens,
                }
            )
            future_predictor = build_future_predictor(predictor_build_cfg).to(device).eval()
            fp_load = future_predictor.load_state_dict(strip_compile_prefix(ckpt["future_predictor"]), strict=False)
            future_predictor_missing_keys, future_predictor_unexpected_keys = _log_incompatible_keys(
                "stage1.future_predictor",
                fp_load,
            )
            critical_missing = [
                key for key in future_predictor_missing_keys
                if not key.startswith("sigreg")
            ]
            if critical_missing:
                raise RuntimeError(f"FuturePredictor checkpoint is missing critical key(s): {critical_missing}")
            if use_ema:
                fp_ema_load = future_predictor.load_state_dict(
                    strip_compile_prefix(_require_stage1_ema_state(ckpt, "future_predictor_ema")),
                    strict=False,
                )
                _log_incompatible_keys("stage1.future_predictor_ema", fp_ema_load)
                ema_loaded_keys.append("future_predictor_ema")
        future_predictor.requires_grad_(False)
        _preloaded_text = preloaded_modules.get("text_conditioner")
        if _preloaded_text is not None:
            text_conditioner = _preloaded_text
            text_conditioner.requires_grad_(False)
        elif bool(predictor_cfg.get("use_language", True)):
            encoder_type = str(predictor_cfg.get("language_encoder_type", "clip"))
            text_conditioner = TextConditioner(
                encoder_type=encoder_type,
                clip_model=str(predictor_cfg.get("clip_model", "openai/clip-vit-large-patch14")),
                t5_model=str(predictor_cfg.get("t5_model", "google-t5/t5-base")),
                proj_dim=int(predictor_cfg.get("language_dim", 768)),
            ).to(device).eval()
            cfg_lang_dim = int(predictor_cfg.get("language_dim", 768))
            if cfg_lang_dim != text_conditioner.hidden_size:
                raise ValueError(
                    f"predictor.language_dim={cfg_lang_dim} mismatches "
                    f"TextConditioner(encoder_type={encoder_type!r}).hidden_size="
                    f"{text_conditioner.hidden_size}."
                )
            text_proj_state = ckpt.get("text_conditioner_proj")
            if text_proj_state is None:
                raise RuntimeError(
                    "Checkpoint enables future-predictor language conditioning but has no "
                    "`text_conditioner_proj`. Refusing to evaluate with a randomly initialized "
                    "language projection. Use a checkpoint saved with text_conditioner_proj or "
                    "set predictor.use_language=false only for an intentional ablation."
                )
            text_conditioner.proj.load_state_dict(strip_compile_prefix(text_proj_state), strict=True)
            if use_ema:
                text_ema_load = text_conditioner.proj.load_state_dict(
                    strip_compile_prefix(_require_stage1_ema_state(ckpt, "text_conditioner_proj_ema")),
                    strict=False,
                )
                _log_incompatible_keys("stage1.text_conditioner_proj_ema", text_ema_load)
                ema_loaded_keys.append("text_conditioner_proj_ema")
            text_conditioner.requires_grad_(False)

    # --- Optional CUDA-graph inference compile + DA3_MAX_OPTIMIZE fused path ---
    #
    # Two env-gated tiers, both no-ops by default:
    #   * DA3_COMPILE_INFERENCE (handled by _resolve_inference_compile_targets):
    #     per-submodule torch.compile of {predictor, shallow, propagate,
    #     action_head}. Existing simplified port.
    #   * DA3_MAX_OPTIMIZE=1: in addition, prebakes the predictor / DA3 RoPE
    #     builders into graph-stable static buffers, casts weights to bf16, and
    #     FUSES predictor -> DA3 deep propagation -> action head (optionally with
    #     the shallow encoder folded in) into a single compiled callable so the
    #     whole h=1 forward replays as one CUDA graph (~6.9 ms model-only).
    #
    # When BOTH DA3_MAX_OPTIMIZE is unset AND DA3_COMPILE_INFERENCE is
    # unset/"none", `_inference_compile_targets` is empty and `max_optimize_active`
    # is False: every branch below is skipped, no monkey-patch / bf16 cast /
    # compile / fused callable runs, and behaviour is byte-identical to the
    # uncompiled path. This is the top correctness bar.
    #
    # This block lives here (after the predictor/text build, before the AR
    # generators + policy closures) so the fused callables it builds are visible
    # to `_generate_gam_chunks` and `policy` as free variables. The compiled
    # submodule slots (`future_predictor`, `model.student_da3.*`,
    # `model.action_head`) are read lazily at rollout time, so reassigning them
    # here is observed by those closures.
    _inference_compile_targets = _resolve_inference_compile_targets(
        os.environ.get("DA3_COMPILE_INFERENCE", "none")
    )
    max_optimize_active = os.environ.get("DA3_MAX_OPTIMIZE") == "1"
    _inference_compile_info: dict[str, Any] = {}
    # Resolve the compile mode only for active compile paths. The default
    # no-op path skips DA3_COMPILE_INFERENCE_MODE validation.
    _inference_compile_mode = (
        _resolve_inference_compile_mode()
        if (_inference_compile_targets or max_optimize_active)
        else "reduce-overhead"
    )
    _cudagraph_clone = os.environ.get("DA3_CUDAGRAPH_CLONE", "1") == "1"
    # FUSE only under max-optimize and only when a predictor exists; the fused
    # callables replace the separate predictor/propagate/action_head graphs.
    _fuse_h1 = bool(max_optimize_active) and future_predictor is not None
    fused_h1_inference = None
    fused_h1_inference_with_shallow = None

    if max_optimize_active and future_predictor is not None:
        _max_optimize_meta: dict[str, Any] = {"rope_positions_cache": False, "bf16_cast": False}
        # 1) Graph-STABLE, graph-INTERNAL RoPE positions for the predictor.
        # build_positions does a torch.empty dynamic alloc that breaks the CUDA
        # graph. For h=1 inference the position tensor is fully deterministic, so
        # PREBAKE it once (eagerly, before compile) into a static buffer and patch
        # build_positions to return it. It then runs INSIDE the graph with zero
        # allocation, so the predictor fuses into a single CUDA graph. The cos/sin
        # data_ptr cache is bypassed so cos/sin is recomputed in-graph from the
        # static positions (identical across depth blocks).
        try:
            rope_module = getattr(future_predictor, "rope", None)
            if rope_module is not None and hasattr(rope_module, "build_positions"):
                _orig_build_positions = rope_module.build_positions
                _V_h1 = len(rollout_camera_keys) if rollout_camera_keys else 2
                _np_pv = int(getattr(future_predictor, "num_patches_per_view", 256))
                _npv = int(getattr(future_predictor, "num_prefix_visual", 1))
                _frozen_positions = None
                try:
                    _frozen_positions = _orig_build_positions(1, _V_h1, _np_pv, _npv, device)
                except Exception:  # noqa: BLE001
                    _frozen_positions = None

                def _h1_build_positions(H, V, num_patches, num_prefix_visual, device, *,
                                        __frozen=_frozen_positions, __orig=_orig_build_positions,
                                        __V=_V_h1):
                    # Static buffer read for the h=1 inference geometry. Runs
                    # inside the CUDA graph. int(H)/int(V) are
                    # Python ints (callers pass H_obs as an int), so the branch is
                    # a compile-time constant.
                    if __frozen is not None and int(H) == 1 and int(V) == __V:
                        return __frozen
                    return __orig(H, V, num_patches, num_prefix_visual, device)

                rope_module.build_positions = _h1_build_positions  # type: ignore[assignment]
                if hasattr(rope_module, "_disable_cache"):
                    rope_module._disable_cache = True
                else:
                    setattr(rope_module, "_disable_cache", True)
                _max_optimize_meta["rope_positions_prebaked_in_graph"] = bool(_frozen_positions is not None)
                _max_optimize_meta["rope_cos_sin_cache_bypassed_in_graph"] = True
                _max_optimize_meta["rope_positions_cache"] = True
        except Exception as exc:  # noqa: BLE001
            _max_optimize_meta["rope_positions_cache_error"] = str(exc)[:200]

        # 1b) FULL inference prebake of the predictor's concat-mode dynamic
        # builders. Captured graphs receive prebaked tensors:
        #   - Force flex_attention OFF (return None) -> the dense additive mask
        #     path is taken; flex BlockMask is a non-replayable object that
        #     fragments the graph. For h=1 the dense mask is tiny (~595x595).
        #   - Prebake the dense block-causal / concat mask once (eager) and patch
        #     the builder to return that static buffer.
        #   - Prebake the concat language RoPE positions once and patch.
        # These depend only on (H=1, V, lang_len), so they are deterministic and
        # safe to freeze for h=1 inference.
        try:
            if future_predictor is not None:
                _V_h1 = len(rollout_camera_keys) if rollout_camera_keys else 2
                _lang_len = int(getattr(future_predictor, "language_len", 0) or 0)
                _use_lang = bool(getattr(future_predictor, "use_language", False))
                _cmode = str(getattr(future_predictor, "condition_mode", ""))
                _prepended = _lang_len if (_use_lang and _cmode == "concat") else 0
                _pf: dict[str, Any] = {}

                if hasattr(future_predictor, "_get_flex_block_mask"):
                    future_predictor._get_flex_block_mask = (  # type: ignore[assignment]
                        lambda H, V, device, lang_len=0: None
                    )
                    _pf["flex_off"] = True

                try:
                    if _prepended > 0 and hasattr(future_predictor, "_build_concat_dense_mask"):
                        _orig_cdm = future_predictor._build_concat_dense_mask
                        _frozen_dense = _orig_cdm(1, _V_h1, _prepended, device)

                        def _frozen_concat_dense_mask(H, V, lang_len, device, *,
                                                      __f=_frozen_dense, __o=_orig_cdm, __V=_V_h1, __L=_prepended):
                            if int(H) == 1 and int(V) == __V and int(lang_len) == __L:
                                return __f
                            return __o(H, V, lang_len, device)
                        future_predictor._build_concat_dense_mask = _frozen_concat_dense_mask  # type: ignore[assignment]
                        _pf["concat_dense_mask"] = True
                    if hasattr(future_predictor, "_build_dense_block_causal_mask"):
                        _orig_dbcm = future_predictor._build_dense_block_causal_mask
                        _frozen_bcm = _orig_dbcm(1, _V_h1, device)

                        def _frozen_dense_block_causal_mask(H, V, device, *,
                                                            __f=_frozen_bcm, __o=_orig_dbcm, __V=_V_h1):
                            if int(H) == 1 and int(V) == __V:
                                return __f
                            return __o(H, V, device)
                        future_predictor._build_dense_block_causal_mask = _frozen_dense_block_causal_mask  # type: ignore[assignment]
                        _pf["dense_block_causal_mask"] = True
                except Exception as exc:  # noqa: BLE001
                    _pf["dense_mask_error"] = str(exc)[:160]

                try:
                    if _prepended > 0 and hasattr(future_predictor, "_build_concat_lang_positions"):
                        _orig_clp = future_predictor._build_concat_lang_positions
                        _frozen_lang_pos = _orig_clp(lang_len=_prepended, V=_V_h1, device=device)

                        def _frozen_concat_lang_positions(lang_len, V, device, *,
                                                          __f=_frozen_lang_pos, __o=_orig_clp, __V=_V_h1, __L=_prepended):
                            if int(lang_len) == __L and int(V) == __V:
                                return __f
                            return __o(lang_len=lang_len, V=V, device=device)
                        future_predictor._build_concat_lang_positions = _frozen_concat_lang_positions  # type: ignore[assignment]
                        _pf["concat_lang_positions"] = True
                except Exception as exc:  # noqa: BLE001
                    _pf["lang_positions_error"] = str(exc)[:160]

                _max_optimize_meta["predictor_inference_prebake"] = _pf
        except Exception as exc:  # noqa: BLE001
            _max_optimize_meta["predictor_prebake_error"] = str(exc)[:200]

        # 2) bf16 weight cast on the predictor + DA3 backbone Linear/Conv/Embedding
        # weights. LayerNorm / RoPE inv_freq / DPT depth-head stay fp32. The DA3
        # patch_embed Conv2d is kept fp32 (it runs on the fp32 image tensor under
        # autocast; a bf16 weight there raises "Input type float / bias bf16"
        # inside torch.compile). Every deeper Linear already runs under bf16
        # autocast, so casting its weights is numerically equivalent and halves
        # resident weight memory. DA3_MAX_OPTIMIZE_NO_BF16=1 disables this block.
        _bf16_disabled = os.environ.get("DA3_MAX_OPTIMIZE_NO_BF16") == "1"
        if _bf16_disabled:
            _max_optimize_meta["bf16_cast"] = False
            _max_optimize_meta["bf16_cast_scope"] = "disabled_by_no_bf16_flag"
        else:
            try:
                _cast_count, _skip_names = 0, []

                def _bf16_eligible(module: Any, name: str) -> bool:
                    low = name.lower()
                    if any(tok in low for tok in ("dpt", "depth_head", "shallowrope", "rope", "layernorm")):
                        return False
                    return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding))

                for mod_name, mod in future_predictor.named_modules():
                    if _bf16_eligible(mod, mod_name):
                        mod.to(torch.bfloat16)
                        _cast_count += 1
                    else:
                        _skip_names.append(mod_name)

                _bb_cast = 0

                def _bf16_eligible_backbone(module: Any, name: str) -> bool:
                    low = name.lower()
                    if any(
                        tok in low
                        for tok in ("dpt", "depth_head", "shallowrope", "rope", "layernorm", "patch_embed", "norm")
                    ):
                        return False
                    return isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding))

                _student_backbone = getattr(model, "student_da3", None)
                if _student_backbone is not None:
                    for mod_name, mod in _student_backbone.named_modules():
                        if _bf16_eligible_backbone(mod, mod_name):
                            mod.to(torch.bfloat16)
                            _bb_cast += 1
                _max_optimize_meta["bf16_cast"] = True
                _max_optimize_meta["bf16_cast_scope"] = "predictor+backbone"
                _max_optimize_meta["bf16_cast_modules"] = int(_cast_count)
                _max_optimize_meta["bf16_cast_backbone_modules"] = int(_bb_cast)
                _max_optimize_meta["bf16_skip_examples"] = _skip_names[:8]
            except Exception as exc:  # noqa: BLE001
                _max_optimize_meta["bf16_cast_error"] = str(exc)[:200]

        # 3) DA3 backbone (dinov2) RoPE is also CUDA-graph-hostile in the deep
        # stack: RotaryPositionEmbedding2D.forward does int(positions.max()) (a
        # host sync) and caches cos/sin in a Python dict that lands in graph
        # buffers. For h=1 the patch grid is tiny, so pin the table to a fixed
        # large max_position (256 >> any patch index) and recompute cos/sin
        # in-graph (no dict cache). RoPE outputs stay identical (embedding indexes
        # by positions). Class-level, applied once, gated by max-optimize.
        try:
            import depth_anything_3.model.dinov2.layers.rope as _da3rope  # type: ignore
            _R = getattr(_da3rope, "RotaryPositionEmbedding2D", None)
            if _R is not None and not getattr(_R, "_max_opt_patched", False):
                def _compute_freq_nocache(self, dim, seq_len, device, dtype):
                    exponents = torch.arange(0, dim, 2, device=device).float() / dim
                    inv_freq = 1.0 / (self.base_frequency ** exponents)
                    pos = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                    angles = torch.einsum("i,j->ij", pos, inv_freq).to(dtype)
                    angles = torch.cat((angles, angles), dim=-1)
                    return angles.cos().to(dtype), angles.sin().to(dtype)

                def _forward_fixed_maxpos(self, tokens, positions, *, __MP=256):
                    feature_dim = tokens.size(-1) // 2
                    cos_comp, sin_comp = self._compute_frequency_components(
                        feature_dim, __MP, tokens.device, tokens.dtype
                    )
                    vf, hf = tokens.chunk(2, dim=-1)
                    vf = self._apply_1d_rope(vf, positions[..., 0], cos_comp, sin_comp)
                    hf = self._apply_1d_rope(hf, positions[..., 1], cos_comp, sin_comp)
                    return torch.cat((vf, hf), dim=-1)

                _R._compute_frequency_components = _compute_freq_nocache
                _R.forward = _forward_fixed_maxpos
                _R._max_opt_patched = True
                _max_optimize_meta["da3_rope_fixed_maxpos"] = True
        except Exception as exc:  # noqa: BLE001
            _max_optimize_meta["da3_rope_patch_error"] = str(exc)[:200]

        _inference_compile_info["max_optimize_meta"] = _max_optimize_meta

    # Per-submodule torch.compile. Under max-optimize the predictor / propagate /
    # action_head are FUSED below into a single graph, so skip their separate
    # compile here for the unfused path. Compiling them individually
    # plus the clone wrappers between them fragments the graph and gives no
    # speedup. The shallow encoder may still be compiled on its own if requested.
    if _inference_compile_targets:
        _compiled_labels: list[str] = []
        if "predictor" in _inference_compile_targets and not _fuse_h1:
            if future_predictor is not None:
                future_predictor = _compile_for_inference(
                    "future_predictor", future_predictor, _inference_compile_mode, _cudagraph_clone
                )
                _compiled_labels.append("predictor")
            else:
                print("[compile-inference] predictor: skipped (future_predictor not loaded)")
        if "shallow" in _inference_compile_targets:
            if hasattr(model.student_da3, "encode_shallow_visual_slots"):
                model.student_da3.encode_shallow_visual_slots = _compile_for_inference(
                    "student_da3.encode_shallow_visual_slots",
                    model.student_da3.encode_shallow_visual_slots,
                    _inference_compile_mode,
                    _cudagraph_clone,
                )
                _compiled_labels.append("shallow")
            else:
                print("[compile-inference] shallow: skipped (encode_shallow_visual_slots missing)")
        if "propagate" in _inference_compile_targets and not _fuse_h1:
            if hasattr(model.student_da3, "propagate_shallow_with_actions"):
                model.student_da3.propagate_shallow_with_actions = _compile_for_inference(
                    "student_da3.propagate_shallow_with_actions",
                    model.student_da3.propagate_shallow_with_actions,
                    _inference_compile_mode,
                    _cudagraph_clone,
                )
                _compiled_labels.append("propagate")
            else:
                print("[compile-inference] propagate: skipped (propagate_shallow_with_actions missing)")
        if "action_head" in _inference_compile_targets and not _fuse_h1:
            if getattr(model, "action_head", None) is not None:
                model.action_head = _compile_for_inference(
                    "action_head", model.action_head, _inference_compile_mode, _cudagraph_clone
                )
                _compiled_labels.append("action_head")
            else:
                print("[compile-inference] action_head: skipped (action_head missing)")
        _inference_compile_info["targets"] = sorted(_inference_compile_targets)
        _inference_compile_info["mode"] = _inference_compile_mode
        _inference_compile_info["clone"] = bool(_cudagraph_clone)
        _inference_compile_info["compiled"] = _compiled_labels

    # Build the fused single-graph h=1 inference callable(s). One compiled
    # callable runs predictor -> DA3 deep propagation -> action head (and,
    # optionally, the frozen DA3 shallow encoder as its first stage) so CUDA
    # graphs capture it as a single replay with no module-boundary breaks or
    # clone wrappers. The .cpu()/normalizer steps stay outside (the caller does
    # them). Returns raw (unnormalized) actions as a (1, 1, V, 7) float tensor,
    # matching _action_head_chunks4d's output contract. These closures read
    # `future_predictor` / `model.*` as free variables so they pick up the
    # bf16-cast / prebaked modules above.
    if _fuse_h1:
        global _INFERENCE_CUDAGRAPH_ACTIVE
        # The fused callables are bare torch.compile, outside
        # _compile_for_inference, so flip the module-level CUDA-graph flag here
        # too. call_policy emits cudagraph_mark_step_begin() per step when a
        # CUDA-graph compile mode (reduce-overhead / max-autotune) is active.
        if _inference_compile_mode in _CUDAGRAPH_COMPILE_MODES:
            _INFERENCE_CUDAGRAPH_ACTIVE = True
        _fuse_nv = len(rollout_camera_keys) if rollout_camera_keys else 2

        def _fused_h1_core(visual_history, proprio_last, proprio_history,
                           prev_action, lang_feats_in, lang_mask_in):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = future_predictor(
                    past_visual_tokens=visual_history,
                    proprio=proprio_last,
                    proprio_history=proprio_history,
                    past_action_history=prev_action,
                    lang_feats=lang_feats_in,
                    lang_padding_mask=lang_mask_in,
                )
                pv = pred["predicted_next_visual_tokens"]
                pa = pred["predicted_action_tokens"]
                deep = model.student_da3.propagate_shallow_with_actions(
                    pv, pa, decode_visuals=False,
                )
                at = deep["action_tokens"].reshape(1, 1, _fuse_nv, -1)
                raw = model.action_head(at).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(2)
            return raw

        # Shallow-folded core. `images_norm_in` is the encoder-normalized image
        # tensor with static shape (1, V, 3, H, W) for h=1 inference. The shallow
        # encode runs DA3 blocks 0-12 (frozen, local attention) and returns
        # tokens shaped (B, T=1, V, tokens, D), exactly the predictor's
        # `past_visual_tokens` layout. The dinov2 RoPE + PositionGetter hazards
        # are covered by the class-level RoPE monkeypatch above plus the
        # position_getter prewarm below.
        def _fused_h1_core_with_shallow(images_norm_in, proprio_last, proprio_history,
                                        prev_action, lang_feats_in, lang_mask_in):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                shallow = model.student_da3.encode_shallow_visual_slots(
                    images_norm_in,
                    T=1,
                    V=_fuse_nv,
                )
                visual_history = shallow["visual_tokens"]
                pred = future_predictor(
                    past_visual_tokens=visual_history,
                    proprio=proprio_last,
                    proprio_history=proprio_history,
                    past_action_history=prev_action,
                    lang_feats=lang_feats_in,
                    lang_padding_mask=lang_mask_in,
                )
                pv = pred["predicted_next_visual_tokens"]
                pa = pred["predicted_action_tokens"]
                deep = model.student_da3.propagate_shallow_with_actions(
                    pv, pa, decode_visuals=False,
                )
                at = deep["action_tokens"].reshape(1, 1, _fuse_nv, -1)
                raw = model.action_head(at).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(2)
            return raw

        _mode = _inference_compile_mode

        # Prewarm the dinov2 PositionGetter cache for the exact h=1 shallow
        # geometry so the allocating cartesian_prod miss branch never runs inside
        # the captured graph. patch grid = (image_size / patch_size).
        try:
            _trans = model.student_da3.backbone.pretrained
            _pg = getattr(_trans, "position_getter", None)
            if _pg is not None:
                _ps = int(getattr(_trans, "patch_size", 14))
                _h_px, _w_px = int(image_size[0]), int(image_size[1])
                _ = _pg(1 * _fuse_nv, _h_px // _ps, _w_px // _ps, device=device)
                _inference_compile_info.setdefault("max_optimize_meta", {})[
                    "shallow_position_getter_prewarmed"
                ] = True
        except Exception as exc:  # noqa: BLE001
            _inference_compile_info.setdefault("max_optimize_meta", {})[
                "shallow_position_getter_prewarm_error"
            ] = str(exc)[:160]

        # CUDA-graph output aliasing: both fused callables are bare torch.compile
        # (NO output-clone wrapper) so the CUDA-graph tree treats the call as a
        # single replayable unit. The single (1,1,V,7) raw-action output is
        # copied out with .cpu() IMMEDIATELY at the call site (in
        # _generate_gam_chunks), before any subsequent graph replay can overwrite
        # the static buffer, so aliasing is safe. DA3_CUDAGRAPH_CLONE=1 opts the
        # shallow-folded graph into the clone wrapper as a fail-safe; it defaults
        # OFF for the fused path to preserve the single-graph capture.
        _wrap_graph_out = (
            _mode in _CUDAGRAPH_COMPILE_MODES
            and os.environ.get("DA3_CUDAGRAPH_CLONE", "0") == "1"
        )

        try:
            fused_h1_inference = torch.compile(_fused_h1_core, mode=_mode, dynamic=False)
            _inference_compile_info["fused_h1"] = True
        except Exception as exc:  # noqa: BLE001
            _inference_compile_info["fused_h1_error"] = str(exc)[:200]
            fused_h1_inference = None

        # DA3_FUSE_SHALLOW=1 (default) folds the shallow encoder into the graph;
        # =0 keeps the predictor-only graph plus a separately-encoded shallow
        # input (previous behavior) to isolate the fold's effect / drift.
        _fuse_shallow_enabled = os.environ.get("DA3_FUSE_SHALLOW", "1") == "1"
        if _fuse_shallow_enabled and hasattr(model.student_da3, "encode_shallow_visual_slots"):
            try:
                fused_h1_inference_with_shallow = torch.compile(
                    _fused_h1_core_with_shallow, mode=_mode, dynamic=False
                )
                if _wrap_graph_out:
                    fused_h1_inference_with_shallow = _CloneOutputModule(
                        fused_h1_inference_with_shallow
                    )
                _inference_compile_info["fused_h1_with_shallow"] = True
            except Exception as exc:  # noqa: BLE001
                _inference_compile_info["fused_h1_with_shallow_error"] = str(exc)[:200]
                fused_h1_inference_with_shallow = None
        else:
            _inference_compile_info["fused_h1_with_shallow"] = False
            if not _fuse_shallow_enabled:
                _inference_compile_info["fused_h1_with_shallow_skip"] = "DA3_FUSE_SHALLOW=0"

    def policy_text_prompt(task_desc: str) -> str:
        return normalize_text_prompt_for_policy(task_desc, text_prompt_normalization) or "perform the task"

    def predictor_text_tokens(task_desc: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if text_conditioner is None or future_predictor is None:
            return None, None
        cache_key = policy_text_prompt(task_desc)
        if cache_key not in text_cache:
            with torch.no_grad():
                tok_out = text_conditioner.encode_tokens([cache_key], pad_to=future_predictor.language_len)
            text_cache[cache_key] = {
                "last_hidden_state": tok_out["last_hidden_state"].detach(),
                "attention_mask": tok_out["attention_mask"].detach(),
            }
        cached = text_cache[cache_key]
        return cached["last_hidden_state"].to(device), cached["attention_mask"].to(device)

    def describe_text_prompt(task_desc: str) -> dict[str, Any]:
        prompt = policy_text_prompt(task_desc)
        encoder_type = (
            str(getattr(text_conditioner, "encoder_type", "text"))
            if text_conditioner is not None and future_predictor is not None
            else "none"
        )
        audit: dict[str, Any] = {
            "prompt_raw": task_desc,
            "prompt": prompt,
            "prompt_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
            "normalization": text_prompt_normalization,
            "empty_or_dummy": prompt in {"", "dummy", "perform the task"},
            "text_encoder_type": encoder_type,
        }
        if text_conditioner is not None and future_predictor is not None:
            with torch.no_grad():
                lang_feats, lang_mask = predictor_text_tokens(prompt)
            if lang_mask is not None:
                audit["token_count"] = int(lang_mask.detach().cpu().sum().item())
            if lang_feats is not None:
                audit["text_norm"] = float(lang_feats.detach().float().norm(dim=-1).mean().cpu().item())
        return audit

    image_history: list[torch.Tensor] = []
    raw_image_history: list[torch.Tensor] = []
    depth_history: list[torch.Tensor | None] = []
    proprio_history: list[torch.Tensor] = []
    action_chunk_history: list[torch.Tensor] = []
    pending_history: dict[str, Any] = {}

    def _action_head_chunks4d(action_tokens_4d: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred = model.action_head(action_tokens_4d).float()
        if pred.ndim == 3:
            pred = pred.unsqueeze(2)
        if pred.ndim != 4:
            raise RuntimeError(f"Action head returned unexpected shape {tuple(pred.shape)}.")
        return pred

    def _cat_gam_sequence(parts: torch.Tensor | list[torch.Tensor], name: str) -> torch.Tensor:
        if isinstance(parts, torch.Tensor):
            return parts
        if not parts:
            raise RuntimeError(f"Cannot decode an empty gam {name} sequence.")
        return torch.cat(parts, dim=1)

    def _decode_gam_sequence(
        visual_tokens: torch.Tensor | list[torch.Tensor],
        action_tokens: torch.Tensor | list[torch.Tensor],
        *,
        decode_visuals: bool,
    ) -> dict[str, Any]:
        deep_visual = _cat_gam_sequence(visual_tokens, "visual")
        deep_actions = _cat_gam_sequence(action_tokens, "action")
        if deep_visual.shape[:3] != deep_actions.shape[:3]:
            raise RuntimeError(
                "gam deep visual/action sequence mismatch: "
                f"{tuple(deep_visual.shape)} vs {tuple(deep_actions.shape)}"
            )
        # Defensive NaN/Inf guards immediately before DA3 deep propagation.
        # Background: jobs 3415714 / 3424641 / 3425039 / 3425048 all sporadically
        # hang INSIDE SDPA inside the DA3 deep attention. The hang is consistent
        # with one of q/k/v containing non-finite values that drive
        # F.scaled_dot_product_attention's flash/mem-efficient kernel into a
        # divergent loop (a known PyTorch SDPA + non-finite-input failure mode
        # on aarch64/H100/GH200). Log predictor output and shallow visual tokens,
        # then sanitize the deep stack input with zero replacement. A guard gives
        # a deterministic source line for CUDA hang diagnosis.
        for _name, _t in (("deep_visual", deep_visual), ("deep_actions", deep_actions)):
            _bad = ~torch.isfinite(_t)
            if bool(_bad.any()):
                _n_bad = int(_bad.sum().item())
                _n_tot = int(_t.numel())
                print(
                    f"[gam deep-input WARN] {_name} non-finite "
                    f"({_n_bad}/{_n_tot}) shape={tuple(_t.shape)}; replacing with zeros",
                    flush=True,
                )
        deep_visual = torch.nan_to_num(deep_visual, nan=0.0, posinf=0.0, neginf=0.0)
        deep_actions = torch.nan_to_num(deep_actions, nan=0.0, posinf=0.0, neginf=0.0)
        # Bypass the SDPA flash/mem-efficient backends and force the math
        # (vanilla PyTorch) backend for the deep DA3 attention during rollout.
        # The flash-attention CUDA kernel on aarch64+GH200 has been observed to
        # hang inside the DA3 deep stack when called with small batch / small
        # seq-len + bf16 inputs (rollout-time shape). The math backend is the
        # stable rollout backend for those conditions.
        _sdpa_ctx = None
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
            _sdpa_ctx = sdpa_kernel([SDPBackend.MATH])
        except Exception:  # noqa: BLE001
            try:
                _sdpa_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                )
            except Exception:  # noqa: BLE001
                _sdpa_ctx = None
        if _sdpa_ctx is not None:
            with _sdpa_ctx:
                decoded_future = model.student_da3.propagate_shallow_with_actions(
                    deep_visual,
                    deep_actions,
                    decode_visuals=bool(decode_visuals),
                )
        else:
            decoded_future = model.student_da3.propagate_shallow_with_actions(
                deep_visual,
                deep_actions,
                decode_visuals=bool(decode_visuals),
            )
        if "action_tokens" not in decoded_future:
            raise RuntimeError("DA3 shallow propagation did not return action tokens.")
        # rollout_action_source: "direct" reads the action straight from the
        # predictor's per-step action token (matches lambda_action_refine=0
        # training); "refine" (default) reads the DA3 deep-refined action token.
        if rollout_action_source == "direct":
            action_src_tokens = deep_actions.reshape(
                1, deep_visual.shape[1], n_views, -1
            )
        else:
            action_src_tokens = decoded_future["action_tokens"].reshape(
                1, deep_visual.shape[1], n_views, -1
            )
        action_chunks_norm_raw = _action_head_chunks4d(action_src_tokens)[0].detach().cpu()
        action_chunks_norm = clamp_normalized_action_for_rollout(action_chunks_norm_raw, normalizer)
        action_chunks = normalizer.denormalize(
            action_chunks_norm.reshape(-1, action_chunks_norm.shape[-1]),
            stats_key=chosen_stats_key,
        ).reshape(action_chunks_norm.shape)

        debug_depth = None
        debug_rgb = None
        depth_source = "unavailable"
        if decode_visuals and "depth" in decoded_future:
            debug_depth = decoded_future["depth"].reshape(
                1,
                deep_visual.shape[1] * n_views,
                *decoded_future["depth"].shape[-2:],
            ).detach().cpu()
            depth_source = "gam_autoregressive_full_sequence"
        if decode_visuals and "rgb" in decoded_future:
            debug_rgb = decoded_future["rgb"].detach().cpu()
        return {
            "action_chunks": action_chunks.detach().cpu(),
            "action_chunks_norm": action_chunks_norm.detach().cpu(),
            "action_chunks_norm_raw": action_chunks_norm_raw.detach().cpu(),
            "generated_steps": int(action_chunks_norm.shape[0]),
            "depth": debug_depth,
            "depth_source": depth_source,
            "rgb": debug_rgb,
        }

    def _generate_gam_chunks_kv(
        *,
        observed_visual_tokens: torch.Tensor,
        observed_proprio_tokens: torch.Tensor,
        observed_prev_action_chunks: torch.Tensor,
        lang_feats: torch.Tensor | None,
        lang_mask: torch.Tensor | None,
        decode_steps: int,
        execute_steps: int,
        decode_visuals: bool,
        warm_past_kvs: list | None = None,
        warm_past_length: int = 0,
    ) -> dict[str, Any]:
        """Compatibility wrapper for the old KV-cached gam AR rollout.

        Correct DA3 deep refinement now needs all observed-prefix predictor
        heads, not just attention K/Vs. Until the cache persists those heads too,
        this wrapper delegates to the full-forward path for train/eval semantic
        equivalence.
        """
        del warm_past_kvs, warm_past_length
        return _generate_gam_chunks(
            observed_visual_tokens=observed_visual_tokens,
            observed_proprio_tokens=observed_proprio_tokens,
            observed_prev_action_chunks=observed_prev_action_chunks,
            lang_feats=lang_feats,
            lang_mask=lang_mask,
            decode_steps=decode_steps,
            execute_steps=execute_steps,
            decode_visuals=decode_visuals,
        )

    def _generate_gam_chunks(
        *,
        observed_visual_tokens: torch.Tensor,
        observed_proprio_tokens: torch.Tensor,
        observed_prev_action_chunks: torch.Tensor,
        lang_feats: torch.Tensor | None,
        lang_mask: torch.Tensor | None,
        decode_steps: int,
        execute_steps: int,
        decode_visuals: bool,
        observed_images: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Autoregressive gam rollout for a single policy call.

        Terminology:
            N_deep = decode_steps         : DA3 deep-refine sequence length
            K_chunk = chunk_size          : low-level actions per iteration
            H_hist = stage1_history_horizon: predictor input window length

        Training feeds DA3 deep the full predictor output sequence for the
        observed window: [o_{t-H+2}, a_{t-H+1}], ..., [o_{t+1}, a_t].
        Rollout must preserve that observed-prefix sequence and execute slot
        H_obs - 1, rather than dropping the prefix and decoding only o_{t+1}.
        """
        if future_predictor is None:
            raise RuntimeError("gam AR rollout requested without future_predictor.")
        H_obs = int(observed_visual_tokens.shape[1])
        if H_obs < 1:
            raise ValueError("gam AR rollout requires at least one observed timestep.")
        execute_start = H_obs - 1
        min_decode_steps = execute_start + max(1, int(execute_steps))
        decode_steps = max(H_obs, int(decode_steps), min_decode_steps)
        decode_steps = min(int(stage1_native_train_horizon), decode_steps)

        visual_history_tokens = observed_visual_tokens.detach()
        proprio_history_tokens = observed_proprio_tokens.detach()
        prev_action_chunk_history = observed_prev_action_chunks.detach().to(
            device=device, dtype=torch.float32
        )

        # Fused single-CUDA-graph fast path (DA3_MAX_OPTIMIZE, decode_steps==1).
        # When the shallow-folded graph is available AND the caller threaded the
        # raw encoder-normalized images through (`observed_images`), run the WHOLE
        # model as ONE compiled callable: DA3 shallow encode (blocks 0-12) ->
        # predictor -> DA3 deep propagation (blocks 13-39) -> action head.
        # Otherwise fall back to the predictor-only fused graph that consumes the
        # already-encoded shallow `observed_visual_tokens`. Either way the
        # cpu()/normalizer steps stay outside the graph (done here). The fused
        # callables are None unless DA3_MAX_OPTIMIZE=1, so this branch is a no-op
        # on the default path.
        _use_shallow_fold = (
            fused_h1_inference_with_shallow is not None
            and observed_images is not None
            and int(decode_steps) == 1
        )
        if _use_shallow_fold:
            _raw4d = fused_h1_inference_with_shallow(
                observed_images,
                proprio_history_tokens[:, -1, :],
                proprio_history_tokens,
                prev_action_chunk_history,
                lang_feats,
                lang_mask,
            )
        elif fused_h1_inference is not None and int(decode_steps) == 1:
            _raw4d = fused_h1_inference(
                visual_history_tokens,
                proprio_history_tokens[:, -1, :],
                proprio_history_tokens,
                prev_action_chunk_history,
                lang_feats,
                lang_mask,
            )
        if _use_shallow_fold or (fused_h1_inference is not None and int(decode_steps) == 1):
            # Copy the raw action out of the CUDA-graph static buffer IMMEDIATELY
            # (.cpu()), before any subsequent replay can overwrite it. Then the
            # normalizer/denorm steps run on CPU outside the graph, producing the
            # exact ar_result dict shape the eager AR path returns for a
            # single-decode-step h=1 call (generated_steps==1, execute_start==0).
            _acnr = _raw4d[0].detach().cpu()
            _acn = clamp_normalized_action_for_rollout(_acnr, normalizer)
            _ac = normalizer.denormalize(
                _acn.reshape(-1, _acn.shape[-1]), stats_key=chosen_stats_key,
            ).reshape(_acn.shape)
            _es = slice(execute_start, execute_start + 1)
            return {
                "action_chunks": _ac[_es].detach().cpu(),
                "action_chunks_norm": _acn[_es].detach().cpu(),
                "action_chunks_norm_raw": _acnr[_es].detach().cpu(),
                "action_chunks_full": _ac.detach().cpu(),
                "action_chunks_norm_full": _acn.detach().cpu(),
                "action_chunks_norm_raw_full": _acnr.detach().cpu(),
                "first_action_chunk_norm": _acn[execute_start:execute_start + 1].detach().cpu(),
                "generated_steps": 1,
                "execute_steps": 1,
                "execute_start": int(execute_start),
                "depth": None,
                "depth_source": "unavailable",
                "rgb": None,
            }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred_out = future_predictor(
                past_visual_tokens=visual_history_tokens,
                proprio=proprio_history_tokens[:, -1, :],
                proprio_history=proprio_history_tokens,
                past_action_history=prev_action_chunk_history,
                lang_feats=lang_feats,
                lang_padding_mask=lang_mask,
            )

        sequence_visual = pred_out.get("predicted_next_visual_tokens", None)
        sequence_proprio = pred_out.get("predicted_next_proprio", None)
        sequence_actions = pred_out.get("predicted_action_tokens", None)
        if sequence_visual is None or sequence_proprio is None or sequence_actions is None:
            raise RuntimeError("gam AR rollout requires predicted visual/proprio/action outputs.")
        sequence_visual = sequence_visual.to(observed_visual_tokens.dtype)
        sequence_proprio = sequence_proprio.to(observed_proprio_tokens.dtype)
        sequence_actions = sequence_actions.to(observed_visual_tokens.dtype)

        while int(sequence_visual.shape[1]) < int(decode_steps):
            prefix_decode = _decode_gam_sequence(
                sequence_visual,
                sequence_actions,
                decode_visuals=False,
            )
            next_prev_action = (
                prefix_decode["action_chunks_norm"][-1:]
                .unsqueeze(0)
                .to(device=device, dtype=prev_action_chunk_history.dtype)
            )
            visual_history_tokens = torch.cat([visual_history_tokens, sequence_visual[:, -1:].detach()], dim=1)
            proprio_history_tokens = torch.cat([proprio_history_tokens, sequence_proprio[:, -1:].detach()], dim=1)
            prev_action_chunk_history = torch.cat([prev_action_chunk_history, next_prev_action.detach()], dim=1)

            window_visual = visual_history_tokens[:, -stage1_history_horizon:]
            window_proprio = proprio_history_tokens[:, -stage1_history_horizon:]
            window_prev_actions = prev_action_chunk_history[:, -stage1_history_horizon:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred_out = future_predictor(
                    past_visual_tokens=window_visual,
                    proprio=window_proprio[:, -1, :],
                    proprio_history=window_proprio,
                    past_action_history=window_prev_actions,
                    lang_feats=lang_feats,
                    lang_padding_mask=lang_mask,
                )
            next_visual = pred_out.get("predicted_next_visual_tokens", None)
            next_proprio = pred_out.get("predicted_next_proprio", None)
            next_actions = pred_out.get("predicted_action_tokens", None)
            if next_visual is None or next_proprio is None or next_actions is None:
                raise RuntimeError("gam AR rollout requires predicted visual/proprio/action outputs.")
            sequence_visual = torch.cat(
                [sequence_visual, next_visual[:, -1:].to(sequence_visual.dtype)],
                dim=1,
            )
            sequence_proprio = torch.cat(
                [sequence_proprio, next_proprio[:, -1:].to(sequence_proprio.dtype)],
                dim=1,
            )
            sequence_actions = torch.cat(
                [sequence_actions, next_actions[:, -1:].to(sequence_actions.dtype)],
                dim=1,
            )

        final_decode = _decode_gam_sequence(
            sequence_visual,
            sequence_actions,
            decode_visuals=decode_visuals,
        )
        generated_steps = int(final_decode["generated_steps"])
        execute_start = min(execute_start, max(0, generated_steps - 1))
        execute_steps_resolved = max(1, min(int(execute_steps), generated_steps - execute_start))
        execute_slice = slice(execute_start, execute_start + execute_steps_resolved)

        return {
            "action_chunks": final_decode["action_chunks"][execute_slice].detach().cpu(),
            "action_chunks_norm": final_decode["action_chunks_norm"][execute_slice].detach().cpu(),
            "action_chunks_norm_raw": final_decode["action_chunks_norm_raw"][execute_slice].detach().cpu(),
            "action_chunks_full": final_decode["action_chunks"].detach().cpu(),
            "action_chunks_norm_full": final_decode["action_chunks_norm"].detach().cpu(),
            "action_chunks_norm_raw_full": final_decode["action_chunks_norm_raw"].detach().cpu(),
            "first_action_chunk_norm": final_decode["action_chunks_norm"][execute_start : execute_start + 1].detach().cpu(),
            "generated_steps": generated_steps,
            "execute_steps": execute_steps_resolved,
            "execute_start": int(execute_start),
            "depth": final_decode["depth"],
            "depth_source": final_decode["depth_source"],
            "rgb": final_decode["rgb"],
        }

    def policy(obs: dict[str, Any], task_desc: str = "") -> torch.Tensor:
        # Per-stage inference profiling, gated by DA3_PROFILE_INFERENCE=1.
        # Off by default (no overhead). Records timestamps at key checkpoints
        # and prints a single summary line per call. Used for offline
        # closed-loop policy profiling (see scripts/profile_inference_*.sh).
        _prof = os.environ.get("DA3_PROFILE_INFERENCE") == "1"
        _prof_t = {}
        def _mark(label):
            if _prof:
                torch.cuda.synchronize()
                _prof_t[label] = time.time()
        _mark("t0_enter")
        current_raw_images = images_from_obs(
            obs,
            image_size=image_size,
            camera_keys=rollout_camera_keys,
            rotate_for_policy=False,
            train_crop_min_scale=train_crop_min_scale,
            eval_crop_scale=eval_crop_scale,
            da3_input_vflip=False,
            dataset_preprocess=False,
        )
        current_images = images_from_obs(
            obs,
            image_size=image_size,
            camera_keys=rollout_camera_keys,
            rotate_for_policy=bool(rotate_policy_input),
            train_crop_min_scale=train_crop_min_scale,
            eval_crop_scale=eval_crop_scale,
            da3_input_vflip=da3_input_vflip,
            dataset_preprocess=True,
            image_jpeg_eval_enabled=image_jpeg_eval_enabled,
            image_jpeg_eval_quality=image_jpeg_eval_quality,
        )
        obs_depths = depths_from_obs(obs)
        prev_count = min(
            max(0, stage1_history_horizon - 1),
            len(image_history),
            len(proprio_history),
            len(action_chunk_history),
        )
        hist_images = image_history[-prev_count:] if prev_count > 0 else []
        hist_raw_images = raw_image_history[-prev_count:] if prev_count > 0 else []
        hist_depths = depth_history[-prev_count:] if prev_count > 0 else []
        hist_proprio = proprio_history[-prev_count:] if prev_count > 0 else []
        images_cpu = torch.cat([*hist_images, current_images], dim=0)
        raw_images_cpu = torch.cat([*hist_raw_images, current_raw_images], dim=0)
        images = images_cpu.unsqueeze(0).to(device)
        images_norm = (images.float() - encoder_mean) / encoder_std
        current_proprio_cpu = live_proprio_converter(
            obs,
            orientation_mode=proprio_orientation,
        ).view(1, 7).detach().cpu()
        proprio_cpu = torch.cat([*hist_proprio, current_proprio_cpu], dim=0)
        proprio = proprio_cpu.unsqueeze(0).to(device)
        debug_depth = None
        debug_rgb = None
        depth_source = "unavailable"
        H_eff = prev_count + 1
        observed_view_count = H_eff * n_views
        rollout_visual_contract = "conditioning_plus_future"
        predicted_sequence_start_timestep = None
        total_debug_view = H_eff * n_views
        cond_debug_num = H_eff * n_views
        action_input = None
        actions_full = None
        actions_norm_full = None
        actions_norm_raw_full = None
        rollout_decode_steps = None
        rollout_execute_steps = None
        rollout_execute_start = None
        per_slot_ae_mask = torch.zeros(H_eff, dtype=torch.bool, device=device)
        prev_chunks = action_chunk_history[-prev_count:] if prev_count > 0 else []
        if prev_count > 0:
            action_input = torch.zeros(1, H_eff, chunk_size, 7, dtype=torch.float32, device=device)
            action_input[:, :prev_count] = torch.stack(prev_chunks, dim=0).unsqueeze(0).to(device)
            per_slot_ae_mask[:prev_count] = True

        def _build_observed_prev_action_history() -> torch.Tensor:
            zero_chunk = torch.zeros(max(1, chunk_size), 7, dtype=torch.float32)
            chunks: list[torch.Tensor] = []
            selected_start = len(action_chunk_history) - prev_count
            for local_idx in range(prev_count):
                prev_idx = selected_start + local_idx - 1
                if 0 <= prev_idx < len(action_chunk_history):
                    chunks.append(action_chunk_history[prev_idx])
                else:
                    chunks.append(zero_chunk)
            chunks.append(action_chunk_history[-1] if action_chunk_history else zero_chunk)
            return torch.stack(chunks, dim=0).unsqueeze(0).to(device=device, dtype=torch.float32)

        _mark("t1_after_preprocess")
        with torch.no_grad():
            # `encode_with_actions` is the full DA3 (blocks 0-39) forward and
            # is used downstream ONLY for: (a) direct-observation depth decode
            # when decode_visuals=True, (b) the legacy non-AR predictor branch
            # (extract_level0_slots from raw_levels), and (c) the
            # non-gam branch (last_action_tokens for action_head). The
            # gam AR path uses NONE of those for action selection.
            _shallow_ar_active = (
                future_predictor is not None
                and predictor_type in {"gam"}
            )
            _skip_full_encode = (
                _shallow_ar_active
                and (
                    os.environ.get("DA3_SKIP_FULL_ENCODE") == "1"
                    or not decode_visuals
                )
            )
            if _skip_full_encode:
                features_per_level = None
                action_tokens = None
                raw_levels = None
                last_action_tokens = None
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    features_per_level, action_tokens, raw_levels = model.student_da3.encode_with_actions(
                        images_norm,
                        action_input=action_input,
                        override_n_steps=H_eff,
                        per_slot_ae_mask=per_slot_ae_mask,
                    )
                    action_tokens_4d = action_tokens.reshape(1, H_eff, n_views, -1)
                    last_action_tokens = action_tokens_4d[:, -1:, :, :]
            _mark("t2_after_full_da3_encode")
            decoded_depth = None
            if decode_visuals and features_per_level is not None:
                decoded_depth = model.student_da3.decode_depth(features_per_level)
            _mark("t3_after_obs_decode_depth")
            rollout_action_tokens = None
            if future_predictor is not None:
                proprio_cond = proprio
                if proprio_normalizer is not None:
                    proprio_cond = proprio_normalizer.normalize(proprio_cond, stats_key=chosen_stats_key)
                model_task_desc = policy_text_prompt(task_desc)
                # Language embed cache: task_desc is constant within an episode
                # (and usually across all episodes of the same task). Skip the
                # CLIP text pass when it matches the last call.
                cached_desc = getattr(policy, "_lang_task_desc", None)
                if cached_desc == model_task_desc:
                    lang_feats = policy._lang_feats  # type: ignore[attr-defined]
                    lang_mask = policy._lang_mask    # type: ignore[attr-defined]
                else:
                    lang_feats, lang_mask = predictor_text_tokens(model_task_desc)
                    policy._lang_task_desc = model_task_desc  # type: ignore[attr-defined]
                    policy._lang_feats = lang_feats          # type: ignore[attr-defined]
                    policy._lang_mask = lang_mask            # type: ignore[attr-defined]
                if predictor_type in {"gam"}:
                    # --- Shallow encode cache (persistent across policy calls) ---
                    # Blocks 0-12 are pre-global local-attention only, so each
                    # (timestep, view) frame is independently encodable. We cache
                    # shallow tokens per historical observation and only encode
                    # the NEW current observation each call.
                    # Train-equivalent DA3 deep refinement requires predictor
                    # head outputs for every observed-prefix slot. The old
                    # incremental path cached attention K/Vs only. Keep it
                    # disabled until prefix head outputs are cached too.
                    _use_kv = False
                    _persist_ok = _use_kv and predictor_type in {"gam"}
                    _cache_shallow = getattr(policy, "_obs_shallow_cache", None)
                    _cache_kv = getattr(policy, "_obs_kv_cache", None)
                    _cache_len = int(getattr(policy, "_obs_cache_length", 0))
                    # Cache is valid iff it covers exactly the previous observations
                    # (prev_count); otherwise bootstrap from scratch.
                    _cache_valid = _persist_ok and _cache_shallow is not None and _cache_len == prev_count and prev_count > 0
                    _dbg = os.environ.get("DA3_DEBUG_CACHE") == "1"

                    past_action_history = _build_observed_prev_action_history()
                    # N_decode is the total DA3 deep-refine sequence length.
                    # The first H_eff slots are the predictor outputs for the
                    # real observed window, exactly as in training. The current
                    # action is therefore slot H_eff - 1; any slots after that
                    # are rollout-generated AR suffix.
                    #
                    # decode_steps is resolved BEFORE the shallow encode so the
                    # shallow-fold decision (_will_fold) can SKIP the redundant
                    # eager encode: the fused-with-shallow graph recomputes shallow
                    # tokens from images_norm internally.
                    _h_exec = int(getattr(policy, "active_action_horizon", action_steps) or action_steps)
                    execute_steps = max(1, min(_h_exec, stage1_native_train_horizon))
                    min_decode_steps = (H_eff - 1) + execute_steps
                    if stage1_rollout_decode_horizon is None:
                        decode_steps = min_decode_steps
                    else:
                        decode_steps = max(
                            min_decode_steps,
                            min(int(stage1_rollout_decode_horizon), stage1_native_train_horizon),
                        )
                    decode_steps = max(H_eff, min(int(stage1_native_train_horizon), int(decode_steps)))

                    # Shallow-fold decision. When the shallow encoder is folded
                    # into the single CUDA graph (h=1, decode_steps==1), the eager
                    # encode below is REDUNDANT. The graph recomputes shallow
                    # tokens from `images_norm` internally. Skipping the eager
                    # encode here is the entire point of the fold: the ~11ms eager
                    # shallow encode collapses into the captured graph. We still
                    # pass a tiny placeholder so `_generate_gam_chunks` can read
                    # H_obs (==H_eff==1) without an extra forward; the placeholder
                    # is only consumed for its shape on the fold path. Folding is
                    # never active unless DA3_MAX_OPTIMIZE built the fused graph,
                    # so the default path always takes the eager-encode branches.
                    _will_fold = (
                        not _use_kv
                        and fused_h1_inference_with_shallow is not None
                        and int(decode_steps) == 1
                        and int(H_eff) == 1
                    )
                    _shallow_t0 = time.time()
                    if _will_fold:
                        full_visual_tokens = images_norm.new_zeros((1, H_eff, 1, 1, 1))
                        warm_kvs = None
                        warm_len = 0
                        if _dbg:
                            print(
                                f"[SHALLOW FOLD] H_eff={H_eff} decode_steps={decode_steps} "
                                f"-> eager shallow encode skipped (folded into graph)",
                                flush=True,
                            )
                    elif _cache_valid:
                        # Encode only the current observation (last n_views in images_norm).
                        current_imgs_only = images_norm[:, -n_views:]
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            new_shallow = model.student_da3.encode_shallow_visual_slots(
                                current_imgs_only,
                                T=1,
                                V=n_views,
                            )
                        full_visual_tokens = torch.cat(
                            [_cache_shallow, new_shallow["visual_tokens"]], dim=1
                        )
                        warm_kvs = _cache_kv
                        warm_len = _cache_len
                        if _dbg:
                            torch.cuda.synchronize()
                            _sd = (time.time() - _shallow_t0) * 1000
                            print(f"[CACHE HIT ] H_eff={H_eff} prev_count={prev_count} shallow={_sd:.1f}ms (1 ts) warm_len={warm_len}", flush=True)
                    else:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            shallow = model.student_da3.encode_shallow_visual_slots(
                                images_norm,
                                T=H_eff,
                                V=n_views,
                            )
                        full_visual_tokens = shallow["visual_tokens"]
                        warm_kvs = None
                        warm_len = 0
                        if _dbg:
                            torch.cuda.synchronize()
                            _sd = (time.time() - _shallow_t0) * 1000
                            print(f"[CACHE MISS] H_eff={H_eff} prev_count={prev_count} shallow={_sd:.1f}ms ({H_eff} ts) cache_len={_cache_len} cache_is_none={_cache_shallow is None}", flush=True)

                    # KV cache is currently disabled above for train-equivalent
                    # prefix-head semantics.
                    _ar_fn = _generate_gam_chunks_kv if _use_kv else _generate_gam_chunks
                    _t_pf0 = time.time()
                    if _use_kv:
                        ar_result = _ar_fn(
                            observed_visual_tokens=full_visual_tokens,
                            observed_proprio_tokens=proprio_cond,
                            observed_prev_action_chunks=past_action_history,
                            lang_feats=lang_feats,
                            lang_mask=lang_mask,
                            decode_steps=decode_steps,
                            execute_steps=execute_steps,
                            decode_visuals=decode_visuals,
                            warm_past_kvs=warm_kvs,
                            warm_past_length=warm_len,
                        )
                    else:
                        ar_result = _ar_fn(
                            observed_visual_tokens=full_visual_tokens,
                            observed_proprio_tokens=proprio_cond,
                            observed_prev_action_chunks=past_action_history,
                            lang_feats=lang_feats,
                            lang_mask=lang_mask,
                            decode_steps=decode_steps,
                            execute_steps=execute_steps,
                            decode_visuals=decode_visuals,
                            # Thread the encoder-normalized images so the AR
                            # generator can run the shallow-folded single graph
                            # (image -> shallow -> predictor -> deep -> action).
                            # Only passed when the fold will apply; otherwise None
                            # keeps the existing eager-tokens fast path.
                            observed_images=images_norm if _will_fold else None,
                        )
                    # Publish timing for outer instrumentation.
                    if _dbg:
                        torch.cuda.synchronize()
                    policy._last_ar_forward_ms = (time.time() - _t_pf0) * 1000.0
                    policy._last_ar_backend = "kv_cache" if _use_kv else "no_cache"
                    if _dbg:
                        print(f"[ARFWD    ] H_eff={H_eff} warm_len={warm_len} ar_forward_total={policy._last_ar_forward_ms:.1f}ms", flush=True)

                    # --- Persist shallow + predictor KV cache for next call ---
                    if _persist_ok and "obs_kv_snapshot" in ar_result:
                        new_cache_shallow = full_visual_tokens.detach()
                        new_cache_kv = ar_result["obs_kv_snapshot"]
                        # Slide when exceeding H_hist: drop the oldest timestep
                        # from both caches. tokens_per_step must match
                        # future_predictor's internal layout.
                        if new_cache_shallow.shape[1] > int(stage1_history_horizon):
                            excess = int(new_cache_shallow.shape[1] - stage1_history_horizon)
                            new_cache_shallow = new_cache_shallow[:, excess:]
                            tps = int(n_views) * int(future_predictor.visual_tokens_per_view) + 2
                            drop = excess * tps
                            new_cache_kv = [
                                (k[:, :, drop:, :].contiguous(), v[:, :, drop:, :].contiguous())
                                for (k, v) in new_cache_kv
                            ]
                        policy._obs_shallow_cache = new_cache_shallow  # type: ignore[attr-defined]
                        policy._obs_kv_cache = new_cache_kv           # type: ignore[attr-defined]
                        policy._obs_cache_length = int(new_cache_shallow.shape[1])  # type: ignore[attr-defined]
                    rollout_action_tokens = None
                    predicted_action_chunks = ar_result["action_chunks"]
                    actions_norm_raw = ar_result["action_chunks_norm_raw"]
                    actions_norm = ar_result["action_chunks_norm"]
                    actions = predicted_action_chunks
                    actions_full = ar_result.get("action_chunks_full", predicted_action_chunks)
                    actions_norm_full = ar_result.get("action_chunks_norm_full", actions_norm)
                    actions_norm_raw_full = ar_result.get("action_chunks_norm_raw_full", actions_norm_raw)
                    rollout_decode_steps = int(ar_result.get("generated_steps", decode_steps))
                    rollout_execute_steps = int(ar_result.get("execute_steps", execute_steps))
                    rollout_execute_start = int(ar_result.get("execute_start", max(0, H_eff - 1)))
                    total_debug_view = int(ar_result["generated_steps"]) * n_views
                    cond_debug_num = 0
                    rollout_visual_contract = "gam_predicted_sequence_trainlike"
                    predicted_sequence_start_timestep = 2 - H_eff
                    debug_depth = ar_result["depth"]
                    depth_source = ar_result["depth_source"]
                    debug_rgb = ar_result["rgb"]
                else:
                    past_slots = extract_level0_slots(
                        raw_level0=raw_levels[0],
                        current_norm_action=action_tokens,
                        T=H_eff,
                        V=n_views,
                        embed_dim=student.embed_dim,
                        num_register_tokens=predictor_num_register_tokens,
                    )
                    future_steps = max(0, action_steps - H_eff)
                    if future_steps > 0:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            pred_out = future_predictor(
                                past_tokens=past_slots,
                                proprio=proprio_cond[:, -1, :],
                                F_=future_steps,
                                lang_feats=lang_feats,
                                lang_padding_mask=lang_mask,
                            )
                        future_slots = pred_out["z_future"]
                    else:
                        future_slots = past_slots.new_empty(past_slots.shape[0], 0, *past_slots.shape[2:])
                    if future_steps > 0:
                        propagate_slots = torch.cat([past_slots.detach(), future_slots], dim=1)
                        patches_for_prop, cls_for_prop, action_for_prop = _slots_to_da3_propagation_inputs(
                            propagate_slots,
                            student.embed_dim,
                        )
                        decoded_future = model.student_da3.propagate_and_predict(
                            patches_for_prop,
                            cls_for_prop,
                            action_for_prop,
                            total_view=(H_eff + future_steps) * n_views,
                            action_head=None,
                            cond_num=H_eff * n_views,
                            decode_visuals=decode_visuals,
                        )
                        if "action_tokens" not in decoded_future:
                            raise RuntimeError("DA3 propagation did not return future action tokens.")
                        propagated_action_tokens = decoded_future["action_tokens"].reshape(
                            1, H_eff + future_steps, n_views, -1
                        )
                        rollout_action_tokens = propagated_action_tokens[:, H_eff:, :, :].contiguous()
                        if decode_visuals and "depth" in decoded_future:
                            total_debug_view = (H_eff + future_steps) * n_views
                            cond_debug_num = H_eff * n_views
                            debug_depth, depth_source = splice_direct_conditioning_depth(
                                decoded_future["depth"].detach().cpu(),
                                decoded_depth.detach().cpu() if decoded_depth is not None else None,
                                total_view=total_debug_view,
                                cond_view_count=cond_debug_num,
                            )
                        if decode_visuals and "rgb" in decoded_future:
                            debug_rgb = decoded_future["rgb"].detach().cpu()
            if predictor_type not in {"gam"}:
                action_tokens_for_head = rollout_action_tokens if rollout_action_tokens is not None else last_action_tokens
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    pred = model.action_head(action_tokens_for_head)

        if predictor_type not in {"gam"}:
            if pred.ndim == 4:
                pred = pred.reshape(pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3])
            actions_norm_raw = pred[0].float().cpu()
            actions_norm = clamp_normalized_action_for_rollout(actions_norm_raw, normalizer)
            actions = normalizer.denormalize(actions_norm, stats_key=chosen_stats_key)
            actions_full = actions
            actions_norm_full = actions_norm
            actions_norm_raw_full = actions_norm_raw
            rollout_decode_steps = int(actions.shape[0]) if actions.ndim >= 2 else 1
            rollout_execute_steps = rollout_decode_steps
        if debug_depth is None and decoded_depth is not None:
            debug_depth = decoded_depth.reshape(1, H_eff * n_views, *decoded_depth.shape[-2:]).detach().cpu()
            depth_source = "direct_conditioning_only"
        debug_obs_depths = None
        if obs_depths is not None and all(depth is not None for depth in hist_depths):
            debug_obs_depths = torch.cat([*(depth for depth in hist_depths if depth is not None), obs_depths], dim=0)
        policy.last_debug = {
            "obs_images": images_cpu.detach().cpu(),
            "obs_images_policy": images_cpu.detach().cpu(),
            "obs_images_raw": raw_images_cpu.detach().cpu(),
            "obs_depths_raw": debug_obs_depths.detach().cpu() if debug_obs_depths is not None else None,
            "proprio_raw": proprio[:, -1].reshape(-1).detach().cpu(),
            "proprio_history_raw": proprio_cpu.detach().cpu(),
            "actions": actions.reshape(-1, actions.shape[-1]).detach().cpu(),
            "action_chunks": actions.detach().cpu() if actions.ndim == 3 else actions.unsqueeze(1).detach().cpu(),
            "actions_norm": actions_norm.reshape(-1, actions_norm.shape[-1]).detach().cpu(),
            "actions_norm_raw": actions_norm_raw.reshape(-1, actions_norm_raw.shape[-1]).detach().cpu(),
            "actions_full": (
                actions_full.reshape(-1, actions_full.shape[-1]).detach().cpu()
                if isinstance(actions_full, torch.Tensor) else None
            ),
            "action_chunks_full": (
                actions_full.detach().cpu()
                if isinstance(actions_full, torch.Tensor) and actions_full.ndim == 3
                else (actions_full.unsqueeze(1).detach().cpu() if isinstance(actions_full, torch.Tensor) else None)
            ),
            "actions_norm_full": (
                actions_norm_full.reshape(-1, actions_norm_full.shape[-1]).detach().cpu()
                if isinstance(actions_norm_full, torch.Tensor) else None
            ),
            "actions_norm_raw_full": (
                actions_norm_raw_full.reshape(-1, actions_norm_raw_full.shape[-1]).detach().cpu()
                if isinstance(actions_norm_raw_full, torch.Tensor) else None
            ),
            "actions_norm_clamped": bool(torch.any(actions_norm_raw != actions_norm).item()),
            "actions_norm_max_abs_raw": (
                float(actions_norm_raw_full.abs().max().item())
                if isinstance(actions_norm_raw_full, torch.Tensor) and actions_norm_raw_full.numel()
                else (float(actions_norm_raw.abs().max().item()) if actions_norm_raw.numel() else 0.0)
            ),
            "depth": debug_depth,
            "depth_source": depth_source,
            "future_depth_seed": (
                "gam_block13_input"
                if predictor_type in {"gam"}
                else "current_stream_1536"
            ),
            "rgb": debug_rgb,
            "rgb_source": "decoded future RGB" if debug_rgb is not None else "decoded future RGB unavailable",
            "rgb_prefix": "pred rgb",
            "rotate_policy_input": bool(rotate_policy_input),
            "total_view": total_debug_view,
            "cond_num": cond_debug_num,
            "observed_view_count": observed_view_count,
            "rollout_visual_contract": rollout_visual_contract,
            "predicted_sequence_start_timestep": predicted_sequence_start_timestep,
            "n_views": n_views,
            "task_desc": task_desc,
            "visual_debug_label": "Stage 1 DA3",
            "compact_detailed_video": True,
            "history_horizon": stage1_history_horizon,
            "effective_history_horizon": H_eff,
            "predicted_steps": (
                int(rollout_decode_steps)
                if rollout_decode_steps is not None
                else (
                    action_steps if predictor_type in {"gam"}
                    else max(0, action_steps - H_eff)
                )
            ),
            "executed_model_steps": int(rollout_execute_steps) if rollout_execute_steps is not None else None,
            "executed_sequence_start": int(rollout_execute_start) if rollout_execute_start is not None else None,
            "rollout_decode_horizon": (
                int(rollout_decode_steps)
                if rollout_decode_steps is not None
                else stage1_rollout_decode_horizon_mode
            ),
            "rollout_decode_horizon_mode": stage1_rollout_decode_horizon_mode,
            "rollout_decode_horizon_requested": str(rollout_decode_horizon),
            "history_commit_stride_actions": 1 if predictor_type in {"gam"} else max(1, chunk_size),
            "history_commit_stride_env_actions": max(1, chunk_size),
            "env_actions_per_model_step": max(1, chunk_size),
            "history_committed_entries": len(image_history),
            "eval_crop_scale": eval_crop_scale,
            "image_aug_profile": image_aug_cfg.get("image_aug_profile") or "legacy",
            "image_jpeg_eval_enabled": image_jpeg_eval_enabled,
            "image_jpeg_eval_quality": image_jpeg_eval_quality,
            "dataset_da3_input_rotate180": dataset_da3_input_rotate180,
            "dataset_da3_input_hflip": dataset_da3_input_hflip,
            "da3_input_vflip": da3_input_vflip,
            "policy_image_preprocess": "dataset_eval_bicubic_crop_resize",
            "text_prompt_audit": describe_text_prompt(task_desc),
        }
        pending_history.clear()
        pending_history.update(
            {
                "images": current_images.detach().cpu(),
                "raw_images": current_raw_images.detach().cpu(),
                "depths": obs_depths.detach().cpu() if obs_depths is not None else None,
                "proprio": current_proprio_cpu,
                "action_chunk": (
                    actions_norm[: max(1, chunk_size)].detach().cpu()
                    if actions_norm.ndim == 2
                    else actions_norm[0].detach().cpu()
                ),
            }
        )
        _mark("t9_exit")
        if _prof and _prof_t:
            # Compute deltas in ms; AR-forward time was already captured into
            # policy._last_ar_forward_ms by the predictor block above.
            _ord = ["t0_enter", "t1_after_preprocess", "t2_after_full_da3_encode",
                    "t3_after_obs_decode_depth", "t9_exit"]
            _present = [k for k in _ord if k in _prof_t]
            _deltas = []
            for a, b in zip(_present[:-1], _present[1:]):
                _deltas.append(f"{b.split('_', 1)[1]}={(_prof_t[b] - _prof_t[a]) * 1000:.1f}ms")
            _ar_ms = float(getattr(policy, "_last_ar_forward_ms", 0.0))
            _ar_backend = getattr(policy, "_last_ar_backend", "?")
            _total_ms = (_prof_t["t9_exit"] - _prof_t["t0_enter"]) * 1000.0
            _skip_lbl = "skip" if locals().get("_skip_full_encode") else "on"
            print(
                f"[INFER PROFILE] H_eff={H_eff} dec_vis={int(bool(decode_visuals))} "
                f"full_enc={_skip_lbl} "
                f"total={_total_ms:.1f}ms ar({_ar_backend})={_ar_ms:.1f}ms "
                + " ".join(_deltas),
                flush=True,
            )
        return actions

    # Batched rollout support. This mirrors the vla-evaluation-harness model
    # server contract: multiple episode sessions share one GPU model, while
    # each session keeps independent rollout history.
    batch_session_states: dict[str, dict[str, Any]] = {}

    def _new_batch_session_state() -> dict[str, Any]:
        return {
            "image_history": [],
            "raw_image_history": [],
            "depth_history": [],
            "proprio_history": [],
            "action_chunk_history": [],
            "pending_history": {},
            "last_debug": {},
        }

    def _batch_session_state(session_id: str) -> dict[str, Any]:
        key = str(session_id)
        state = batch_session_states.get(key)
        if state is None:
            state = _new_batch_session_state()
            batch_session_states[key] = state
        return state

    def _reset_batch_session(session_id: str) -> None:
        batch_session_states[str(session_id)] = _new_batch_session_state()

    def _decode_gam_sequence_batched(
        visual_tokens: torch.Tensor | list[torch.Tensor],
        action_tokens: torch.Tensor | list[torch.Tensor],
        *,
        decode_visuals: bool,
    ) -> dict[str, Any]:
        deep_visual = _cat_gam_sequence(visual_tokens, "visual")
        deep_actions = _cat_gam_sequence(action_tokens, "action")
        if deep_visual.shape[:3] != deep_actions.shape[:3]:
            raise RuntimeError(
                "gam batched deep visual/action sequence mismatch: "
                f"{tuple(deep_visual.shape)} vs {tuple(deep_actions.shape)}"
            )
        for _name, _t in (("deep_visual", deep_visual), ("deep_actions", deep_actions)):
            _bad = ~torch.isfinite(_t)
            if bool(_bad.any()):
                _n_bad = int(_bad.sum().item())
                _n_tot = int(_t.numel())
                print(
                    f"[gam batch deep-input WARN] {_name} non-finite "
                    f"({_n_bad}/{_n_tot}) shape={tuple(_t.shape)}; replacing with zeros",
                    flush=True,
                )
        deep_visual = torch.nan_to_num(deep_visual, nan=0.0, posinf=0.0, neginf=0.0)
        deep_actions = torch.nan_to_num(deep_actions, nan=0.0, posinf=0.0, neginf=0.0)
        _sdpa_ctx = None
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
            _sdpa_ctx = sdpa_kernel([SDPBackend.MATH])
        except Exception:  # noqa: BLE001
            try:
                _sdpa_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                )
            except Exception:  # noqa: BLE001
                _sdpa_ctx = None
        if _sdpa_ctx is not None:
            with _sdpa_ctx:
                decoded_future = model.student_da3.propagate_shallow_with_actions(
                    deep_visual,
                    deep_actions,
                    decode_visuals=bool(decode_visuals),
                )
        else:
            decoded_future = model.student_da3.propagate_shallow_with_actions(
                deep_visual,
                deep_actions,
                decode_visuals=bool(decode_visuals),
            )
        if "action_tokens" not in decoded_future:
            raise RuntimeError("DA3 shallow propagation did not return action tokens.")
        bsz = int(deep_visual.shape[0])
        # rollout_action_source: "direct" reads the action straight from the
        # predictor's per-step action token (matches lambda_action_refine=0
        # training); "refine" (default) reads the DA3 deep-refined action token.
        if rollout_action_source == "direct":
            action_tokens_4d = deep_actions.reshape(
                bsz, deep_visual.shape[1], n_views, -1
            )
        else:
            action_tokens_4d = decoded_future["action_tokens"].reshape(
                bsz, deep_visual.shape[1], n_views, -1
            )
        action_chunks_norm_raw = _action_head_chunks4d(action_tokens_4d).detach().cpu()
        action_chunks_norm = clamp_normalized_action_for_rollout(action_chunks_norm_raw, normalizer)
        action_chunks = normalizer.denormalize(
            action_chunks_norm.reshape(-1, action_chunks_norm.shape[-1]),
            stats_key=chosen_stats_key,
        ).reshape(action_chunks_norm.shape)

        debug_depth = None
        debug_rgb = None
        depth_source = "unavailable"
        if decode_visuals and "depth" in decoded_future:
            debug_depth = decoded_future["depth"].reshape(
                bsz,
                deep_visual.shape[1] * n_views,
                *decoded_future["depth"].shape[-2:],
            ).detach().cpu()
            depth_source = "gam_autoregressive_full_sequence"
        if decode_visuals and "rgb" in decoded_future:
            debug_rgb = decoded_future["rgb"].detach().cpu()
        return {
            "action_chunks": action_chunks.detach().cpu(),
            "action_chunks_norm": action_chunks_norm.detach().cpu(),
            "action_chunks_norm_raw": action_chunks_norm_raw.detach().cpu(),
            "generated_steps": int(action_chunks_norm.shape[1]),
            "depth": debug_depth,
            "depth_source": depth_source,
            "rgb": debug_rgb,
        }

    def _generate_gam_chunks_batched(
        *,
        observed_visual_tokens: torch.Tensor,
        observed_proprio_tokens: torch.Tensor,
        observed_prev_action_chunks: torch.Tensor,
        lang_feats: torch.Tensor | None,
        lang_mask: torch.Tensor | None,
        decode_steps: int,
        execute_steps: int,
        decode_visuals: bool,
    ) -> dict[str, Any]:
        if future_predictor is None:
            raise RuntimeError("gam AR batched rollout requested without future_predictor.")
        H_obs = int(observed_visual_tokens.shape[1])
        if H_obs < 1:
            raise ValueError("gam AR batched rollout requires at least one observed timestep.")
        execute_start = H_obs - 1
        min_decode_steps = execute_start + max(1, int(execute_steps))
        decode_steps = max(H_obs, int(decode_steps), min_decode_steps)
        decode_steps = min(int(stage1_native_train_horizon), decode_steps)

        visual_history_tokens = observed_visual_tokens.detach()
        proprio_history_tokens = observed_proprio_tokens.detach()
        prev_action_chunk_history = observed_prev_action_chunks.detach().to(
            device=device, dtype=torch.float32
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred_out = future_predictor(
                past_visual_tokens=visual_history_tokens,
                proprio=proprio_history_tokens[:, -1, :],
                proprio_history=proprio_history_tokens,
                past_action_history=prev_action_chunk_history,
                lang_feats=lang_feats,
                lang_padding_mask=lang_mask,
            )

        sequence_visual = pred_out.get("predicted_next_visual_tokens", None)
        sequence_proprio = pred_out.get("predicted_next_proprio", None)
        sequence_actions = pred_out.get("predicted_action_tokens", None)
        if sequence_visual is None or sequence_proprio is None or sequence_actions is None:
            raise RuntimeError("gam AR batched rollout requires predicted visual/proprio/action outputs.")
        sequence_visual = sequence_visual.to(observed_visual_tokens.dtype)
        sequence_proprio = sequence_proprio.to(observed_proprio_tokens.dtype)
        sequence_actions = sequence_actions.to(observed_visual_tokens.dtype)

        while int(sequence_visual.shape[1]) < int(decode_steps):
            prefix_decode = _decode_gam_sequence_batched(
                sequence_visual,
                sequence_actions,
                decode_visuals=False,
            )
            next_prev_action = (
                prefix_decode["action_chunks_norm"][:, -1:]
                .to(device=device, dtype=prev_action_chunk_history.dtype)
            )
            visual_history_tokens = torch.cat([visual_history_tokens, sequence_visual[:, -1:].detach()], dim=1)
            proprio_history_tokens = torch.cat([proprio_history_tokens, sequence_proprio[:, -1:].detach()], dim=1)
            prev_action_chunk_history = torch.cat([prev_action_chunk_history, next_prev_action.detach()], dim=1)

            window_visual = visual_history_tokens[:, -stage1_history_horizon:]
            window_proprio = proprio_history_tokens[:, -stage1_history_horizon:]
            window_prev_actions = prev_action_chunk_history[:, -stage1_history_horizon:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred_out = future_predictor(
                    past_visual_tokens=window_visual,
                    proprio=window_proprio[:, -1, :],
                    proprio_history=window_proprio,
                    past_action_history=window_prev_actions,
                    lang_feats=lang_feats,
                    lang_padding_mask=lang_mask,
                )
            next_visual = pred_out.get("predicted_next_visual_tokens", None)
            next_proprio = pred_out.get("predicted_next_proprio", None)
            next_actions = pred_out.get("predicted_action_tokens", None)
            if next_visual is None or next_proprio is None or next_actions is None:
                raise RuntimeError("gam AR batched rollout requires predicted visual/proprio/action outputs.")
            sequence_visual = torch.cat(
                [sequence_visual, next_visual[:, -1:].to(sequence_visual.dtype)],
                dim=1,
            )
            sequence_proprio = torch.cat(
                [sequence_proprio, next_proprio[:, -1:].to(sequence_proprio.dtype)],
                dim=1,
            )
            sequence_actions = torch.cat(
                [sequence_actions, next_actions[:, -1:].to(sequence_actions.dtype)],
                dim=1,
            )

        final_decode = _decode_gam_sequence_batched(
            sequence_visual,
            sequence_actions,
            decode_visuals=decode_visuals,
        )
        generated_steps = int(final_decode["generated_steps"])
        execute_start = min(execute_start, max(0, generated_steps - 1))
        execute_steps_resolved = max(1, min(int(execute_steps), generated_steps - execute_start))
        execute_slice = slice(execute_start, execute_start + execute_steps_resolved)
        return {
            "action_chunks": final_decode["action_chunks"][:, execute_slice].detach().cpu(),
            "action_chunks_norm": final_decode["action_chunks_norm"][:, execute_slice].detach().cpu(),
            "action_chunks_norm_raw": final_decode["action_chunks_norm_raw"][:, execute_slice].detach().cpu(),
            "action_chunks_full": final_decode["action_chunks"].detach().cpu(),
            "action_chunks_norm_full": final_decode["action_chunks_norm"].detach().cpu(),
            "action_chunks_norm_raw_full": final_decode["action_chunks_norm_raw"].detach().cpu(),
            "first_action_chunk_norm": final_decode["action_chunks_norm"][
                :, execute_start : execute_start + 1
            ].detach().cpu(),
            "generated_steps": generated_steps,
            "execute_steps": execute_steps_resolved,
            "execute_start": int(execute_start),
            "depth": final_decode["depth"],
            "depth_source": final_decode["depth_source"],
            "rgb": final_decode["rgb"],
        }

    def _build_batched_prev_action_history(
        states: list[dict[str, Any]],
        prev_count: int,
    ) -> torch.Tensor:
        zero_chunk = torch.zeros(max(1, chunk_size), 7, dtype=torch.float32)
        rows: list[torch.Tensor] = []
        for state in states:
            action_history = state["action_chunk_history"]
            selected_start = len(action_history) - prev_count
            chunks: list[torch.Tensor] = []
            for local_idx in range(prev_count):
                prev_idx = selected_start + local_idx - 1
                if 0 <= prev_idx < len(action_history):
                    chunks.append(action_history[prev_idx])
                else:
                    chunks.append(zero_chunk)
            chunks.append(action_history[-1] if action_history else zero_chunk)
            rows.append(torch.stack(chunks, dim=0))
        return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)

    def predict_batch(requests: list[dict[str, Any]]) -> list[torch.Tensor]:
        if not requests:
            return []
        if future_predictor is None or predictor_type not in {"gam"}:
            raise RuntimeError(
                "Batched LIBERO eval currently supports Stage 1 gam policies only. "
                "Use src/eval_libero_unified.py for non-predictor or Stage 2 checkpoints."
            )
        if bool(decode_visuals):
            raise RuntimeError("Batched LIBERO eval requires decode_visuals=false for the fast path.")

        results: list[torch.Tensor | None] = [None] * len(requests)
        groups: dict[tuple[int, int, int], list[int]] = {}
        prepared: list[dict[str, Any]] = []
        for req_idx, req in enumerate(requests):
            session_id = str(req.get("session_id", req_idx))
            state = _batch_session_state(session_id)
            obs = req["obs"]
            task_desc = str(req.get("task_desc", ""))
            current_raw_images = images_from_obs(
                obs,
                image_size=image_size,
                camera_keys=rollout_camera_keys,
                rotate_for_policy=False,
                train_crop_min_scale=train_crop_min_scale,
                eval_crop_scale=eval_crop_scale,
                da3_input_vflip=False,
                dataset_preprocess=False,
            )
            current_images = images_from_obs(
                obs,
                image_size=image_size,
                camera_keys=rollout_camera_keys,
                rotate_for_policy=bool(rotate_policy_input),
                train_crop_min_scale=train_crop_min_scale,
                eval_crop_scale=eval_crop_scale,
                da3_input_vflip=da3_input_vflip,
                dataset_preprocess=True,
                image_jpeg_eval_enabled=image_jpeg_eval_enabled,
                image_jpeg_eval_quality=image_jpeg_eval_quality,
            )
            obs_depths = depths_from_obs(obs)
            prev_count = min(
                max(0, stage1_history_horizon - 1),
                len(state["image_history"]),
                len(state["proprio_history"]),
                len(state["action_chunk_history"]),
            )
            H_eff = prev_count + 1
            _h_exec = int(req.get("active_action_horizon", getattr(policy, "active_action_horizon", action_steps)) or action_steps)
            execute_steps = max(1, min(_h_exec, stage1_native_train_horizon))
            min_decode_steps = (H_eff - 1) + execute_steps
            if stage1_rollout_decode_horizon is None:
                decode_steps = min_decode_steps
            else:
                decode_steps = max(
                    min_decode_steps,
                    min(int(stage1_rollout_decode_horizon), stage1_native_train_horizon),
                )
            decode_steps = max(H_eff, min(int(stage1_native_train_horizon), int(decode_steps)))
            current_proprio_cpu = live_proprio_converter(
                obs,
                orientation_mode=proprio_orientation,
            ).view(1, 7).detach().cpu()
            prepared.append(
                {
                    "session_id": session_id,
                    "state": state,
                    "obs": obs,
                    "task_desc": task_desc,
                    "current_images": current_images,
                    "current_raw_images": current_raw_images,
                    "obs_depths": obs_depths,
                    "current_proprio_cpu": current_proprio_cpu,
                    "prev_count": prev_count,
                    "H_eff": H_eff,
                    "execute_steps": execute_steps,
                    "decode_steps": decode_steps,
                }
            )
            groups.setdefault((H_eff, execute_steps, decode_steps), []).append(req_idx)

        with torch.no_grad():
            for (H_eff, execute_steps, decode_steps), req_indices in groups.items():
                group_prepared = [prepared[i] for i in req_indices]
                images_cpu = torch.stack(
                    [
                        torch.cat(
                            [
                                *(
                                    item["state"]["image_history"][-item["prev_count"] :]
                                    if item["prev_count"] > 0
                                    else []
                                ),
                                item["current_images"],
                            ],
                            dim=0,
                        )
                        for item in group_prepared
                    ],
                    dim=0,
                )
                images = images_cpu.to(device)
                images_norm = (images.float() - encoder_mean) / encoder_std
                proprio_cpu = torch.stack(
                    [
                        torch.cat(
                            [
                                *(
                                    item["state"]["proprio_history"][-item["prev_count"] :]
                                    if item["prev_count"] > 0
                                    else []
                                ),
                                item["current_proprio_cpu"],
                            ],
                            dim=0,
                        )
                        for item in group_prepared
                    ],
                    dim=0,
                )
                proprio = proprio_cpu.to(device)
                proprio_cond = proprio
                if proprio_normalizer is not None:
                    proprio_cond = proprio_normalizer.normalize(proprio_cond, stats_key=chosen_stats_key)
                lang_feat_rows: list[torch.Tensor] = []
                lang_mask_rows: list[torch.Tensor] = []
                for item in group_prepared:
                    model_task_desc = policy_text_prompt(item["task_desc"])
                    lang_feats, lang_mask = predictor_text_tokens(model_task_desc)
                    if lang_feats is not None and lang_mask is not None:
                        lang_feat_rows.append(lang_feats.detach().cpu())
                        lang_mask_rows.append(lang_mask.detach().cpu())
                lang_feats_batch = None
                lang_mask_batch = None
                if lang_feat_rows:
                    if len(lang_feat_rows) != len(group_prepared):
                        raise RuntimeError("Mixed language/no-language requests in batched predictor group.")
                    lang_feats_batch = torch.cat(lang_feat_rows, dim=0).to(device)
                    lang_mask_batch = torch.cat(lang_mask_rows, dim=0).to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    shallow = model.student_da3.encode_shallow_visual_slots(
                        images_norm,
                        T=int(H_eff),
                        V=n_views,
                    )
                full_visual_tokens = shallow["visual_tokens"]
                past_action_history = _build_batched_prev_action_history(
                    [item["state"] for item in group_prepared],
                    prev_count=int(H_eff) - 1,
                )
                ar_result = _generate_gam_chunks_batched(
                    observed_visual_tokens=full_visual_tokens,
                    observed_proprio_tokens=proprio_cond,
                    observed_prev_action_chunks=past_action_history,
                    lang_feats=lang_feats_batch,
                    lang_mask=lang_mask_batch,
                    decode_steps=int(decode_steps),
                    execute_steps=int(execute_steps),
                    decode_visuals=False,
                )
                actions = ar_result["action_chunks"]
                actions_norm = ar_result["action_chunks_norm"]
                for local_idx, req_idx in enumerate(req_indices):
                    item = prepared[req_idx]
                    state = item["state"]
                    action_item = actions[local_idx].detach().cpu()
                    action_norm_item = actions_norm[local_idx].detach().cpu()
                    full_item = ar_result["action_chunks_full"][local_idx].detach().cpu()
                    full_norm_item = ar_result["action_chunks_norm_full"][local_idx].detach().cpu()
                    state["last_debug"] = {
                        "actions": action_item,
                        "actions_norm": action_norm_item,
                        "action_chunks_full": full_item,
                        "action_chunks_norm_full": full_norm_item,
                        "env_actions_per_model_step": max(1, chunk_size),
                        "history_horizon": stage1_history_horizon,
                        "effective_history_horizon": int(H_eff),
                        "predicted_steps": int(ar_result.get("generated_steps", decode_steps)),
                        "executed_model_steps": int(ar_result.get("execute_steps", execute_steps)),
                        "executed_sequence_start": int(ar_result.get("execute_start", max(0, H_eff - 1))),
                        "rollout_decode_horizon": int(ar_result.get("generated_steps", decode_steps)),
                        "rollout_decode_horizon_mode": stage1_rollout_decode_horizon_mode,
                        "history_commit_stride_actions": 1,
                        "history_commit_stride_env_actions": max(1, chunk_size),
                        "text_prompt_audit": describe_text_prompt(item["task_desc"]),
                        "batched_policy": True,
                        "batch_size": len(req_indices),
                    }
                    state["pending_history"].clear()
                    state["pending_history"].update(
                        {
                            "images": item["current_images"].detach().cpu(),
                            "raw_images": item["current_raw_images"].detach().cpu(),
                            "depths": (
                                item["obs_depths"].detach().cpu()
                                if item["obs_depths"] is not None
                                else None
                            ),
                            "proprio": item["current_proprio_cpu"],
                            "action_chunk": action_norm_item[0].detach().cpu(),
                        }
                    )
                    results[req_idx] = action_item

        final: list[torch.Tensor] = []
        for idx, item in enumerate(results):
            if item is None:
                raise RuntimeError(f"Batched policy request {idx} did not produce a result.")
            final.append(item)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
        return final

    def _commit_batch_session_observation(session_id: str, executed_policy_actions: int) -> None:
        state = _batch_session_state(session_id)
        pending = state["pending_history"]
        if not pending:
            return
        expected_commit_steps = 1 if predictor_type in {"gam"} else max(1, chunk_size)
        if int(executed_policy_actions) == expected_commit_steps:
            state["image_history"].append(pending["images"])
            state["raw_image_history"].append(pending["raw_images"])
            state["depth_history"].append(pending["depths"])
            state["proprio_history"].append(pending["proprio"])
            state["action_chunk_history"].append(pending["action_chunk"])
            trim_to = max(1, stage1_history_horizon)
            del state["action_chunk_history"][:-trim_to]
            del state["image_history"][:-trim_to]
            del state["raw_image_history"][:-trim_to]
            del state["depth_history"][:-trim_to]
            del state["proprio_history"][:-trim_to]
        else:
            state["image_history"].clear()
            state["raw_image_history"].clear()
            state["depth_history"].clear()
            state["proprio_history"].clear()
            state["action_chunk_history"].clear()
        pending.clear()

    def _override_batch_session_pending_action_chunk(session_id: str, raw_action_chunk: torch.Tensor) -> None:
        state = _batch_session_state(session_id)
        pending = state["pending_history"]
        if not pending:
            return
        chunk = torch.as_tensor(raw_action_chunk, dtype=torch.float32)
        if chunk.ndim == 1:
            chunk = chunk.view(1, -1)
        if chunk.ndim != 2 or chunk.shape[-1] != int(action_head_cfg.get("n_dims", 7)):
            raise ValueError(
                "override_session_pending_action_chunk expected (K, action_dim), "
                f"got {tuple(chunk.shape)}."
            )
        norm_chunk = normalizer.normalize(chunk, stats_key=chosen_stats_key).detach().cpu()
        pending["action_chunk"] = norm_chunk[: max(1, chunk_size)]

    def _get_batch_session_debug(session_id: str) -> dict[str, Any]:
        debug = _batch_session_state(session_id).get("last_debug", {})
        return dict(debug) if isinstance(debug, dict) else {}

    def commit_observation(executed_policy_actions: int) -> None:
        """Commit the latest observation as history only at the training anchor stride."""
        if not pending_history:
            return
        expected_commit_steps = 1 if predictor_type in {"gam"} else max(1, chunk_size)
        if int(executed_policy_actions) == expected_commit_steps:
            image_history.append(pending_history["images"])
            raw_image_history.append(pending_history["raw_images"])
            depth_history.append(pending_history["depths"])
            proprio_history.append(pending_history["proprio"])
            action_chunk_history.append(pending_history["action_chunk"])
            trim_to = max(1, stage1_history_horizon)
            del action_chunk_history[:-trim_to]
            del image_history[:-trim_to]
            del raw_image_history[:-trim_to]
            del depth_history[:-trim_to]
            del proprio_history[:-trim_to]
        else:
            image_history.clear()
            raw_image_history.clear()
            depth_history.clear()
            proprio_history.clear()
            action_chunk_history.clear()
        pending_history.clear()

    def override_pending_action_chunk(raw_action_chunk: torch.Tensor) -> None:
        """Replace pending previous-action history with the action actually executed."""
        if not pending_history:
            return
        chunk = torch.as_tensor(raw_action_chunk, dtype=torch.float32)
        if chunk.ndim == 1:
            chunk = chunk.view(1, -1)
        if chunk.ndim != 2 or chunk.shape[-1] != int(action_head_cfg.get("n_dims", 7)):
            raise ValueError(
                "override_pending_action_chunk expected (K, action_dim), "
                f"got {tuple(chunk.shape)}."
            )
        norm_chunk = normalizer.normalize(chunk, stats_key=chosen_stats_key).detach().cpu()
        pending_history["action_chunk"] = norm_chunk[: max(1, chunk_size)]

    def snapshot_pending_history() -> dict[str, Any]:
        """Capture the current policy-call history anchor for deferred commit."""
        return dict(pending_history)

    def restore_pending_history(snapshot: dict[str, Any]) -> None:
        """Restore a policy-call history anchor before committing it."""
        pending_history.clear()
        pending_history.update(snapshot)

    def reset_episode() -> None:
        image_history.clear()
        raw_image_history.clear()
        depth_history.clear()
        proprio_history.clear()
        action_chunk_history.clear()
        pending_history.clear()
        # Clear persistent shallow + predictor KV caches between episodes.
        policy._obs_shallow_cache = None      # type: ignore[attr-defined]
        policy._obs_kv_cache = None           # type: ignore[attr-defined]
        policy._obs_cache_length = 0          # type: ignore[attr-defined]

    policy.last_debug = {}
    policy.accepts_task_desc = True
    policy.reset_episode = reset_episode
    policy.commit_observation = commit_observation
    policy.override_pending_action_chunk = override_pending_action_chunk
    policy.snapshot_pending_history = snapshot_pending_history
    policy.restore_pending_history = restore_pending_history
    policy.describe_text_prompt = describe_text_prompt
    policy.predict_batch = predict_batch
    policy.reset_session = _reset_batch_session
    policy.commit_session_observation = _commit_batch_session_observation
    policy.override_session_pending_action_chunk = _override_batch_session_pending_action_chunk
    policy.get_session_debug = _get_batch_session_debug

    steady_action_timesteps = (
        stage1_native_train_horizon if predictor_type in {"gam"}
        else max(1, action_steps - stage1_history_horizon + 1)
    )
    max_action_horizon = steady_action_timesteps if future_predictor is not None else max(1, chunk_size)
    future_predictor_bn_eval = None
    if future_predictor is not None:
        bn_modules = [m for m in future_predictor.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        future_predictor_bn_eval = all(not m.training for m in bn_modules)
    info = {
        "stage": "1",
        "train_steps": int(ckpt.get("train_steps", 0)),
        "use_bf16_autocast": use_bf16,
        "image_size": list(image_size),
        "rollout_camera_keys": list(rollout_camera_keys),
        "rollout_action_source": rollout_action_source,
        "action_stats_key": chosen_stats_key,
        "normalizer_keys": sorted(normalizer.stats_by_key.keys()),
        "action_stats_key_candidates": preferred_stats_keys,
        "action_stats_default_key": normalizer.default_key,
        "chunk_size": chunk_size,
        "action_chunk_size": chunk_size,
        "max_low_level_actions_per_call": steady_action_timesteps * max(1, chunk_size),
        "action_horizon_unit": (
            "model_step"
            if future_predictor is not None and predictor_type in {"gam"}
            else "env_action"
        ),
        "chunk_position_encoding": str(action_head_cfg.get("chunk_position_encoding", "none")),
        "max_action_horizon": max_action_horizon,
        "stage1_rollout_mode": (
            "gam_autoregressive_chunks"
            if future_predictor is not None and predictor_type in {"gam"}
            else ("future_predictor_action_sequence" if future_predictor is not None else "current_action_token_chunk")
        ),
        "dataset_target_hz": infer_policy_target_hz(dataset_cfg),
        "decode_visuals": bool(decode_visuals),
        "rotate_policy_input": bool(rotate_policy_input),
        "future_predictor_debug": bool(future_predictor is not None),
        "future_predictor_type": predictor_type if future_predictor is not None else None,
        "stage1_history_horizon": stage1_history_horizon,
        "stage1_history_horizon_requested": str(history_horizon),
        "stage1_H_choices": predictor_H_choices,
        "stage1_native_train_horizon": stage1_native_train_horizon,
        "stage1_rollout_decode_horizon": (
            int(stage1_rollout_decode_horizon)
            if stage1_rollout_decode_horizon is not None
            else "exec"
        ),
        "stage1_rollout_decode_horizon_requested": str(rollout_decode_horizon),
        "stage1_rollout_decode_horizon_mode": stage1_rollout_decode_horizon_mode,
        "history_commit_stride_actions": (
            1 if predictor_type in {"gam"} else max(1, chunk_size)
        ),
        "history_commit_stride_env_actions": max(1, chunk_size),
        "eval_crop_scale": eval_crop_scale,
        "train_crop_min_scale": train_crop_min_scale,
        "image_aug_profile": image_aug_cfg.get("image_aug_profile") or "legacy",
        "image_jpeg_eval_enabled": image_jpeg_eval_enabled,
        "image_jpeg_eval_quality": image_jpeg_eval_quality,
        "dataset_da3_input_rotate180": dataset_da3_input_rotate180,
        "dataset_da3_input_hflip": dataset_da3_input_hflip,
        "da3_input_vflip": da3_input_vflip,
        "libero_hdf5_env_hflip": infer_libero_hdf5_env_hflip(
            dataset_cfg,
            dataset_da3_input_rotate180,
            dataset_da3_input_hflip,
            bool(rotate_policy_input),
        ),
        "libero_hdf5_env_rotate180": infer_libero_hdf5_env_rotate180(
            dataset_cfg,
            dataset_da3_input_rotate180,
            dataset_da3_input_hflip,
            bool(rotate_policy_input),
        ),
        "policy_image_preprocess": "dataset_eval_bicubic_crop_resize",
        "proprio_orientation": proprio_orientation,
        "action_frame": action_frame,
        "live_proprio_converter": getattr(live_proprio_converter, "__name__", str(live_proprio_converter)),
        "da3_ckpt_path": da3_ckpt_path,
        "text_prompt_normalization": text_prompt_normalization,
        "text_encoder_type": (
            str(getattr(text_conditioner, "encoder_type", "none"))
            if text_conditioner is not None else "none"
        ),
        "student_da3_missing_keys": student_da3_missing_keys,
        "student_da3_unexpected_keys": student_da3_unexpected_keys,
        "proprio_conditioner_missing_keys": proprio_conditioner_missing_keys,
        "proprio_conditioner_unexpected_keys": proprio_conditioner_unexpected_keys,
        "text_conditioner_proj_missing_keys": text_conditioner_proj_missing_keys,
        "text_conditioner_proj_unexpected_keys": text_conditioner_proj_unexpected_keys,
        "future_predictor_bn_eval": future_predictor_bn_eval,
        "future_predictor_missing_keys": future_predictor_missing_keys,
        "future_predictor_unexpected_keys": future_predictor_unexpected_keys,
        "use_ema": bool(use_ema),
        "ema_available": bool(ema_available),
        "ema_loaded_keys": list(ema_loaded_keys),
        "ema": ema_meta,
        "config_source": config_source,
    }

    # --- Inference-compile status (env-gated, default no-op) ----------------
    # The actual torch.compile / DA3_MAX_OPTIMIZE fused-path wiring is done
    # earlier (right after the predictor/text build) so the fused callables are
    # visible to the AR generator + policy closures. Here we only surface the
    # status that block accumulated in `_inference_compile_info` onto `info`.
    # When DA3_MAX_OPTIMIZE is unset AND DA3_COMPILE_INFERENCE is unset/"none",
    # `_inference_compile_info` is empty and the behavior matches the
    # uncompiled path.
    if _inference_compile_targets:
        info["inference_compile_targets"] = _inference_compile_info.get(
            "targets", sorted(_inference_compile_targets)
        )
        info["inference_compile_mode"] = _inference_compile_info.get(
            "mode", _inference_compile_mode
        )
        info["inference_compile_clone"] = _inference_compile_info.get(
            "clone", bool(_cudagraph_clone)
        )
        info["inference_compile_compiled"] = _inference_compile_info.get("compiled", [])
        info["inference_cudagraph_active"] = bool(_INFERENCE_CUDAGRAPH_ACTIVE)
    if max_optimize_active:
        info["max_optimize"] = True
        info["inference_compile_mode"] = _inference_compile_info.get(
            "mode", info.get("inference_compile_mode", _inference_compile_mode)
        )
        info["inference_cudagraph_active"] = bool(_INFERENCE_CUDAGRAPH_ACTIVE)
        if "max_optimize_meta" in _inference_compile_info:
            info["max_optimize_meta"] = _inference_compile_info["max_optimize_meta"]
        for _k in ("fused_h1", "fused_h1_error", "fused_h1_with_shallow",
                   "fused_h1_with_shallow_error", "fused_h1_with_shallow_skip"):
            if _k in _inference_compile_info:
                info[_k] = _inference_compile_info[_k]

    return policy, info


def load_policy(args: argparse.Namespace, cfg: Any, ckpt: dict[str, Any], device: torch.device):
    detected = detect_stage_from_checkpoint(ckpt)
    stage = str(args.stage)
    if stage == "auto":
        stage = detected
    if stage == "unknown":
        raise ValueError(f"Could not infer checkpoint stage from keys: {sorted(ckpt.keys())[:20]}")
    if stage != "1":
        raise ValueError(f"Unsupported --stage {args.stage!r}; expected auto or 1.")
    stage1_cfg_source = "cli"
    ckpt_cfg = ckpt.get("config")
    if ckpt_cfg is not None:
        if OmegaConf.is_config(ckpt_cfg):
            ckpt_cfg = OmegaConf.to_container(ckpt_cfg, resolve=True)
        if isinstance(ckpt_cfg, dict):
            ckpt_predictor = ckpt_cfg.get("predictor")
            if isinstance(ckpt_predictor, dict):
                ckpt_predictor["type"] = "gam"
            cfg = OmegaConf.merge(cfg, OmegaConf.create(ckpt_cfg))
            if getattr(cfg, "predictor", None) is not None:
                cfg.predictor.type = "gam"
            stage1_cfg_source = "checkpoint+cli_fallback"
    return load_stage1_policy(
        cfg=cfg,
        ckpt=ckpt,
        ckpt_path=args.ckpt,
        device=device,
        stats_key=args.action_stats_key,
        action_stats_json=args.action_stats_json,
        decode_visuals=bool(args.decode_visuals or args.detailed_video),
        history_horizon=args.history_horizon,
        rollout_decode_horizon=args.rollout_decode_horizon,
        rotate_policy_input=bool(args.rotate_policy_input),
        proprio_orientation=args.proprio_orientation,
        text_prompt_normalization=args.text_prompt_normalization,
        config_source=stage1_cfg_source,
        use_ema=bool(args.use_ema),
    )


def apply_initial_state(env: Any, init_state: np.ndarray) -> dict[str, Any]:
    set_init_state = getattr(env, "set_init_state", None)
    if callable(set_init_state):
        return set_init_state(init_state)

    env.reset()
    env.base_env.sim.set_state_from_flattened(init_state)
    env.base_env.sim.forward()
    return env.get_observation()


def run_wait_steps(
    env: Any,
    obs: dict[str, Any],
    num_steps_wait: int,
    wait_action_mode: str = "open_gripper",
) -> tuple[dict[str, Any], bool]:
    done = False
    wait_action_mode = str(wait_action_mode or "open_gripper").strip().lower()
    if wait_action_mode not in {"open_gripper", "zero"}:
        raise ValueError(f"wait_action_mode must be 'open_gripper' or 'zero', got {wait_action_mode!r}.")
    dummy = np.zeros(env.action_dimension, dtype=np.float32)
    if wait_action_mode == "open_gripper" and dummy.shape[0] >= 7:
        dummy[-1] = -1.0
    for _ in range(max(0, num_steps_wait)):
        obs, _, done, _ = env.step(dummy)
        if done:
            break
    return obs, bool(done)


def iter_env_candidates(env: Any):
    """Yield wrapper and nested env objects without assuming a specific LIBERO wrapper."""
    seen: set[int] = set()
    stack = [env]
    while stack:
        candidate = stack.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for attr in ("env", "base_env"):
            child = getattr(candidate, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def get_env_control_hz(env: Any) -> float | None:
    for candidate in iter_env_candidates(env):
        value = getattr(candidate, "control_freq", None)
        if value is None:
            continue
        try:
            hz = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(hz) and hz > 0:
            return hz
    return None


def _policy_hz_from_info(policy_info: dict[str, Any], override: float | None) -> float | None:
    if override is not None and override > 0:
        return float(override)
    value = policy_info.get("dataset_target_hz")
    if value is None:
        return None
    try:
        hz = float(value)
    except (TypeError, ValueError):
        return None
    return hz if math.isfinite(hz) and hz > 0 else None


def resolve_action_repeat(
    requested: str,
    *,
    policy_info: dict[str, Any],
    env: Any,
    policy_hz_override: float | None,
) -> tuple[int, float | None, float | None]:
    env_hz = get_env_control_hz(env)
    policy_hz = _policy_hz_from_info(policy_info, policy_hz_override)
    if str(requested).lower() != "auto":
        repeat = int(requested)
        if repeat < 1:
            raise ValueError(f"--action-repeat must be >=1 or auto, got {requested!r}")
        return repeat, policy_hz, env_hz
    if env_hz is None or policy_hz is None:
        return 1, policy_hz, env_hz
    return max(1, int(round(env_hz / policy_hz))), policy_hz, env_hz


def resolve_action_horizon(
    requested: str | int | None,
    *,
    preset: ProtocolPreset,
    policy_info: dict[str, Any],
) -> tuple[int, str]:
    value = str(preset.action_horizon if requested is None else requested).strip().lower()
    max_actions = int(policy_info.get("max_action_horizon") or 0)
    if value in {"all", "full", "chunk"}:
        if max_actions < 1:
            raise ValueError("--action-horizon all requires policy max_action_horizon metadata")
        return max_actions, value
    horizon = int(value)
    if horizon < 1:
        raise ValueError(f"--action-horizon must be >=1 or 'all', got {requested!r}")
    if max_actions > 0:
        horizon = min(horizon, max_actions)
    return horizon, value


def apply_action_repeat_mode(env_action: np.ndarray, action_repeat: int, mode: str) -> np.ndarray:
    arr = np.asarray(env_action, dtype=np.float32).copy()
    if mode == "hold":
        return arr
    if mode == "split_delta":
        repeat = max(1, int(action_repeat))
        arr[..., :6] = arr[..., :6] / float(repeat)
        return arr
    raise ValueError(f"Unknown action repeat mode: {mode}")


def call_policy(policy: Callable[..., torch.Tensor], obs: dict[str, Any], task_desc: str) -> torch.Tensor:
    # CUDA-graph step boundary. When inference modules are compiled with a
    # CUDA-graph mode (reduce-overhead / max-autotune), the graphs reuse static
    # output buffers across replays. Without an explicit step marker, chaining
    # several compiled modules within one policy call (encode_shallow ->
    # predictor -> propagate -> action_head) raises "accessing tensor output of
    # CUDAGraphs that has been overwritten by a subsequent run".
    # cudagraph_mark_step_begin() tells the CUDA-graph trees that a new
    # inference step is starting so prior-step output buffers are safe to
    # reuse. Gated on the module-level flag, so this is a pure no-op on the
    # default (DA3_COMPILE_INFERENCE unset/none) path.
    if _INFERENCE_CUDAGRAPH_ACTIVE and hasattr(torch, "compiler") and hasattr(
        torch.compiler, "cudagraph_mark_step_begin"
    ):
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:  # noqa: BLE001
            pass
    if bool(getattr(policy, "accepts_task_desc", False)):
        out = policy(obs, task_desc=task_desc)
    else:
        out = policy(obs)
    # NaN/Inf guard for closed_loop rollout. A non-finite action is fed into
    # env.step which corrupts mujoco's qpos/qvel; the next policy call then
    # receives non-finite proprio observations and the DA3 attention CUDA
    # kernel diverges into an infinite loop (observed on the OXE-pretrain
    # 0164k -> spatial-eef-relative path, jobs 3415714 / 3424641). Replace
    # with zero action and log so the rollout fails fast instead of stalling.
    if isinstance(out, torch.Tensor):
        bad = ~torch.isfinite(out)
        if bool(bad.any()):
            n_bad = int(bad.sum().item())
            n_total = int(out.numel())
            try:
                rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else -1
            except Exception:  # noqa: BLE001
                rank = -1
            print(
                f"[call_policy WARNING rank={rank}] non-finite policy output "
                f"({n_bad}/{n_total}); replacing with zeros to avoid mujoco corruption",
                flush=True,
            )
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # CUDA-EGL same-process driver-internal race mitigation. Policy CUDA
    # forward and the subsequent robosuite EGL mjr_readPixels share the same
    # NVIDIA driver per-process state on CSCS GH200. With work still pending
    # on the CUDA stream, the next EGL render call can stochastically hang
    # one rank's CUDA softmax kernel inside DA3 attention while other ranks
    # progress fine (observed in 3415714 / 3424641 / 3425039: stuck stack
    # frame consistently at da3_giant_encoder.propagate_shallow_with_actions).
    # Drain the stream explicitly before returning so the env.step + render
    # path that follows runs against a quiescent CUDA queue.
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
    return out


def _as_model_step_chunks(actions: torch.Tensor, forecast_horizon: int) -> torch.Tensor:
    """Return actions as `(model_steps, low_level_actions, action_dim)`."""
    if actions.ndim == 1:
        actions = actions.view(1, 1, -1)
    elif actions.ndim == 2:
        actions = actions.unsqueeze(1)
    elif actions.ndim != 3:
        raise ValueError(f"Expected policy actions with 1, 2, or 3 dims, got {tuple(actions.shape)}.")
    horizon = max(1, min(int(forecast_horizon), int(actions.shape[0])))
    return actions[:horizon].detach().float().cpu()


def _weighted_temporal_ensemble_action(
    candidates: list[tuple[int, torch.Tensor]],
    current_action_step: int,
    decay: float,
) -> tuple[torch.Tensor, int]:
    """Average candidate low-level actions for one target step using ACT-style age decay."""
    if not candidates:
        raise ValueError("temporal ensemble needs at least one candidate action")
    actions = torch.stack([action.float() for _, action in candidates], dim=0)
    ages = torch.tensor(
        [max(0, int(current_action_step) - int(pred_step)) for pred_step, _ in candidates],
        dtype=torch.float32,
    )
    if float(decay) <= 0.0:
        weights = torch.ones_like(ages)
    else:
        # Newer predictions have age 0 and receive the largest weight.
        weights = torch.exp(-float(decay) * ages)
    weights = weights / weights.sum().clamp_min(1.0e-8)
    ensembled = (actions * weights.view(-1, 1)).sum(dim=0)
    return ensembled, int(actions.shape[0])


def _debug_action_chunks_full(debug: Any) -> torch.Tensor | None:
    """Return full decoded model-step chunks from policy debug, if available."""
    if not isinstance(debug, dict):
        return None
    chunks = debug.get("action_chunks_full")
    if chunks is None:
        flat = debug.get("actions_full")
        if flat is None:
            return None
        chunk_size = int(debug.get("env_actions_per_model_step", 0) or 0)
        if chunk_size <= 0:
            return None
        flat_t = torch.as_tensor(flat, dtype=torch.float32)
        if flat_t.ndim != 2 or flat_t.shape[0] < chunk_size:
            return None
        usable = (flat_t.shape[0] // chunk_size) * chunk_size
        chunks_t = flat_t[:usable].view(-1, chunk_size, flat_t.shape[-1])
    else:
        chunks_t = torch.as_tensor(chunks, dtype=torch.float32)
        if chunks_t.ndim == 2:
            chunk_size = int(debug.get("env_actions_per_model_step", 0) or 0)
            if chunk_size <= 0:
                return None
            usable = (chunks_t.shape[0] // chunk_size) * chunk_size
            chunks_t = chunks_t[:usable].view(-1, chunk_size, chunks_t.shape[-1])
    if chunks_t.ndim != 3 or chunks_t.shape[-1] < 1:
        return None
    return chunks_t.detach().float().cpu()


def _select_full_plan_close_prefix(
    debug: Any,
    *,
    close_threshold: float,
    through_end: bool = False,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Select current-slot through first close chunk or end from a full decoded plan."""
    meta: dict[str, Any] = {
        "strategy_applied": False,
        "strategy_reason": "no_full_plan",
    }
    chunks = _debug_action_chunks_full(debug)
    if chunks is None:
        return None, meta
    start_slot = 0
    if isinstance(debug, dict):
        start_slot = int(debug.get("executed_sequence_start", 0) or 0)
    start_slot = max(0, min(start_slot, int(chunks.shape[0]) - 1))
    chunk_size = int(chunks.shape[1])
    flat = chunks.reshape(-1, chunks.shape[-1])
    start_flat = start_slot * chunk_size
    grip = flat[start_flat:, -1]
    close_rel = torch.nonzero(grip > float(close_threshold), as_tuple=False)
    meta.update(
        {
            "full_plan_start_slot": int(start_slot),
            "full_plan_chunk_size": int(chunk_size),
            "full_plan_steps": int(chunks.shape[0]),
            "full_plan_max_gripper": float(flat[:, -1].max().item()),
        }
    )
    if close_rel.numel() == 0:
        meta["strategy_reason"] = "no_close_in_full_plan"
        return None, meta
    first_flat = int(start_flat + close_rel[0].item())
    close_slot = first_flat // chunk_size
    end_slot = int(chunks.shape[0] - 1) if through_end else close_slot
    selected = chunks[start_slot : end_slot + 1].contiguous()
    meta.update(
        {
            "strategy_applied": True,
            "strategy_reason": "selected_close_to_end" if through_end else "selected_close_prefix",
            "selected_start_slot": int(start_slot),
            "selected_end_slot": int(end_slot),
            "selected_model_steps": int(selected.shape[0]),
            "selected_first_close_flat_idx": int(first_flat),
            "selected_first_close_slot": int(close_slot),
            "selected_first_close_chunk_idx": int(first_flat % chunk_size),
        }
    )
    return selected, meta


def _select_full_plan_all(debug: Any) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Select the current-slot through end of a full decoded debug plan."""
    meta: dict[str, Any] = {
        "strategy_applied": False,
        "strategy_reason": "no_full_plan",
    }
    chunks = _debug_action_chunks_full(debug)
    if chunks is None:
        return None, meta
    start_slot = 0
    if isinstance(debug, dict):
        start_slot = int(debug.get("executed_sequence_start", 0) or 0)
    start_slot = max(0, min(start_slot, int(chunks.shape[0]) - 1))
    selected = chunks[start_slot:].contiguous()
    flat = chunks.reshape(-1, chunks.shape[-1])
    meta.update(
        {
            "strategy_applied": True,
            "strategy_reason": "selected_full_plan_all",
            "full_plan_start_slot": int(start_slot),
            "full_plan_chunk_size": int(chunks.shape[1]),
            "full_plan_steps": int(chunks.shape[0]),
            "full_plan_max_gripper": float(flat[:, -1].max().item()),
            "selected_start_slot": int(start_slot),
            "selected_end_slot": int(chunks.shape[0] - 1),
            "selected_model_steps": int(selected.shape[0]),
        }
    )
    return selected, meta


def _infer_libero_object_prefix(task_desc: str, obs: dict[str, Any]) -> str | None:
    """Infer the target object prefix used in LIBERO object observations."""
    text = str(task_desc).lower().replace("-", " ").replace("_", " ")
    name_map = {
        "alphabet soup": "alphabet_soup_1",
        "cream cheese": "cream_cheese_1",
        "salad dressing": "salad_dressing_1",
        "bbq sauce": "bbq_sauce_1",
        "barbecue sauce": "bbq_sauce_1",
        "tomato sauce": "tomato_sauce_1",
        "ketchup": "ketchup_1",
        "butter": "butter_1",
        "milk": "milk_1",
        "chocolate pudding": "chocolate_pudding_1",
        "orange juice": "orange_juice_1",
    }
    for phrase, prefix in name_map.items():
        if phrase in text and (
            f"{prefix}_to_robot0_eef_pos" in obs or f"{prefix}_pos" in obs
        ):
            return prefix
    for key in obs.keys():
        key_s = str(key)
        if not key_s.endswith("_to_robot0_eef_pos"):
            continue
        prefix = key_s[: -len("_to_robot0_eef_pos")]
        if prefix.startswith("basket"):
            continue
        return prefix
    return None


def _object_eef_distance(obs: dict[str, Any], prefix: str | None) -> float | None:
    if not prefix:
        return None
    rel_key = f"{prefix}_to_robot0_eef_pos"
    if rel_key in obs:
        rel = np.asarray(obs[rel_key], dtype=np.float32).reshape(-1)
        if rel.size >= 3:
            return float(np.linalg.norm(rel[:3]))
    pos_key = f"{prefix}_pos"
    if pos_key in obs and "robot0_eef_pos" in obs:
        obj = np.asarray(obs[pos_key], dtype=np.float32).reshape(-1)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        if obj.size >= 3 and eef.size >= 3:
            return float(np.linalg.norm(obj[:3] - eef[:3]))
    return None


def _object_basket_xy_distance(obs: dict[str, Any], prefix: str | None) -> float | None:
    if not prefix:
        return None
    obj_key = f"{prefix}_pos"
    basket_key = "basket_1_pos"
    if obj_key not in obs or basket_key not in obs:
        return None
    obj = np.asarray(obs[obj_key], dtype=np.float32).reshape(-1)
    basket = np.asarray(obs[basket_key], dtype=np.float32).reshape(-1)
    if obj.size < 2 or basket.size < 2:
        return None
    return float(np.linalg.norm(obj[:2] - basket[:2]))


def rollout_episode(
    env: Any,
    init_state: np.ndarray,
    policy: Callable[[dict[str, Any]], torch.Tensor],
    max_steps: int,
    action_horizon: int,
    action_repeat: int,
    action_repeat_mode: str,
    num_steps_wait: int,
    camera_size: int,
    record_video: bool,
    detailed_video: bool,
    binarize_gripper: bool,
    task_desc: str,
    split_video: bool = False,
    action_frame: str = "base",
    proprio_orientation: str = "rpy",
    temporal_ensemble: bool = False,
    temporal_ensemble_decay: float = 0.01,
    execution_strategy: str = "default",
    full_plan_close_threshold: float = 0.5,
    gripper_override: str = "default",
    near_object_close_threshold: float = 0.04,
    near_object_close_hold_steps: int = 80,
    basket_release_threshold: float = 0.06,
    basket_release_hold_steps: int = 80,
    action_trace_path: str | Path | None = None,
    action_trace_context: dict[str, Any] | None = None,
    rollout_debug_log_path: str | Path | None = None,
    rollout_debug_context: dict[str, Any] | None = None,
    rollout_debug_heartbeat_steps: int = 25,
    execute_chunk_prefix: int | None = None,
    partial_chunk_history: str = "default",
    warmup_full_chunk_once: bool = False,
    rollout_wall_timeout_sec: float = 0.0,
    video_camera_names: Sequence[str] | None = None,
    wait_action_mode: str = "open_gripper",
    collect_depth_forecast: bool = False,
    collect_depth_forecast_rgb: bool = False,
) -> dict[str, Any]:
    episode_start_time = time.time()
    action_frame = normalize_action_frame(action_frame)
    proprio_orientation = normalize_proprio_orientation_mode(proprio_orientation)
    if temporal_ensemble and action_frame == "eef_relative":
        raise ValueError(
            "temporal_ensemble with action_frame='eef_relative' is unsupported because "
            "ensemble candidates can come from different policy-call anchors. Use base_delta/eef_delta "
            "or disable temporal ensembling."
        )
    trace_f = None
    if action_trace_path is not None:
        trace_path = Path(action_trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_f = trace_path.open("w", buffering=1)
    debug_f = None
    if rollout_debug_log_path is not None:
        debug_path = Path(rollout_debug_log_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_f = debug_path.open("w", buffering=1)

    def write_action_trace(record: dict[str, Any]) -> None:
        if trace_f is None:
            return
        payload = {
            "schema_version": 1,
            **(action_trace_context or {}),
            **record,
        }
        trace_f.write(json.dumps(_trace_json_value(payload), ensure_ascii=False) + "\n")

    def write_rollout_debug_event(record: dict[str, Any]) -> None:
        if debug_f is None:
            return
        payload = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - episode_start_time, 4),
            "hostname": os.environ.get("HOSTNAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_procid": os.environ.get("SLURM_PROCID"),
            "slurm_localid": os.environ.get("SLURM_LOCALID"),
            "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
            **(rollout_debug_context or {}),
            **record,
        }
        debug_f.write(json.dumps(_trace_json_value(payload), ensure_ascii=False) + "\n")
        debug_f.flush()

    def env_step_trace_info(info: Any) -> dict[str, Any]:
        """Keep native-env action metadata when wrappers expose it."""
        if not isinstance(info, dict):
            return {}
        out: dict[str, Any] = {}
        if "raw_action" in info:
            out["native_env_action"] = info["raw_action"]
        if "raw_action_dimension" in info:
            out["native_env_action_dim"] = int(info["raw_action_dimension"])
        if "action_layout" in info:
            out["native_env_action_layout"] = str(info["action_layout"])
        return out

    def current_action_frame_proprio(obs_for_action: dict[str, Any]) -> torch.Tensor | None:
        if action_frame in {"base", "base_delta"}:
            return None
        return obs_to_canonical_7d_proprio(obs_for_action, orientation_mode=proprio_orientation)

    def finalize_result(result: dict[str, Any]) -> dict[str, Any]:
        if trace_f is not None:
            trace_f.close()
        if debug_f is not None:
            debug_f.close()
        if depth_forecast_records is not None and "depth_forecast_records" not in result:
            result["depth_forecast_records"] = depth_forecast_records
        return result

    execute_chunk_prefix = int(execute_chunk_prefix or 0)
    if execute_chunk_prefix < 0:
        raise ValueError(f"execute_chunk_prefix must be >= 0, got {execute_chunk_prefix}.")
    partial_chunk_history = str(partial_chunk_history or "default").strip().lower()
    if partial_chunk_history in {"", "auto"}:
        partial_chunk_history = "default"
    if execute_chunk_prefix > 0 and partial_chunk_history == "default":
        partial_chunk_history = "rolling_last_k"
    if partial_chunk_history not in {"default", "rolling_last_k"}:
        raise ValueError(
            "partial_chunk_history must be one of {'default', 'rolling_last_k'}, "
            f"got {partial_chunk_history!r}."
        )
    prefix_replan_enabled = execute_chunk_prefix > 0
    if prefix_replan_enabled and temporal_ensemble:
        raise ValueError("execute_chunk_prefix cannot be combined with --temporal-ensemble.")
    if prefix_replan_enabled and str(execution_strategy) != "default":
        raise ValueError("execute_chunk_prefix currently supports only --execution-strategy default.")
    rolling_history_actions: list[torch.Tensor] = []
    rolling_history_chunk_size: int | None = None
    warmup_full_chunk_pending = bool(prefix_replan_enabled and warmup_full_chunk_once)

    heartbeat_steps = max(1, int(rollout_debug_heartbeat_steps))
    write_rollout_debug_event(
        {
            "event": "episode_begin",
            "task_desc": task_desc,
            "max_steps": int(max_steps),
            "num_steps_wait": int(num_steps_wait),
            "wait_action_mode": str(wait_action_mode),
            "action_horizon": int(action_horizon),
            "action_repeat": int(action_repeat),
            "action_repeat_mode": str(action_repeat_mode),
            "split_video": bool(split_video),
            "temporal_ensemble": bool(temporal_ensemble),
            "execution_strategy": str(execution_strategy),
            "execute_chunk_prefix": int(execute_chunk_prefix),
            "partial_chunk_history": partial_chunk_history,
        }
    )
    write_rollout_debug_event({"event": "apply_initial_state_start"})
    obs = apply_initial_state(env, init_state)
    get_live_task_desc = getattr(env, "get_task_description", None)
    if callable(get_live_task_desc):
        try:
            live_task_desc = get_live_task_desc()
        except Exception:
            live_task_desc = None
        if live_task_desc:
            task_desc = str(live_task_desc)
    write_rollout_debug_event(
        {
            "event": "apply_initial_state_end",
            "task_desc": task_desc,
            "obs_summary": _small_numeric_obs_summary(obs),
        }
    )
    reset_policy = getattr(policy, "reset_episode", None)
    if callable(reset_policy):
        write_rollout_debug_event({"event": "policy_reset_start"})
        reset_policy()
        write_rollout_debug_event({"event": "policy_reset_end"})
    write_rollout_debug_event(
        {
            "event": "wait_steps_start",
            "num_steps_wait": int(num_steps_wait),
            "wait_action_mode": str(wait_action_mode),
        }
    )
    obs, done = run_wait_steps(env, obs, num_steps_wait, wait_action_mode=wait_action_mode)
    write_rollout_debug_event(
        {
            "event": "wait_steps_end",
            "initial_success_after_wait": bool(done),
            "wait_action_mode": str(wait_action_mode),
            "obs_summary": _small_numeric_obs_summary(obs),
        }
    )
    write_action_trace(
        {
            "record_type": "episode_start",
            "task_desc": task_desc,
            "num_steps_wait": int(num_steps_wait),
            "wait_action_mode": str(wait_action_mode),
            "max_steps": int(max_steps),
            "execution_strategy": str(execution_strategy),
            "gripper_override": str(gripper_override),
            "initial_success_after_wait": bool(done),
            "obs_after_wait": _small_numeric_obs_summary(obs),
        }
    )
    frames: list[np.ndarray] = []
    split_frames = new_split_video_frames() if split_video else None
    depth_forecast_records: list[dict[str, Any]] | None = [] if collect_depth_forecast else None

    def capture_depth_forecast_record() -> None:
        """Record (predicted next-obs depth, realized GT depth in meters) for the obs that
        was just fed to the policy. One record per policy call; 1-step-ahead pairing
        (record[k].pred vs record[k+1].gt) is done offline by the eval script.
        """
        if depth_forecast_records is None:
            return
        debug = getattr(policy, "last_debug", None)
        debug = debug if isinstance(debug, dict) else {}
        n_views = max(1, int(debug.get("n_views", 2)))
        pred_next = None
        stack = depth_view_stack_from_debug(debug)
        if stack is not None and stack.numel() > 0:
            num_steps = max(1, int(math.ceil(float(stack.shape[0]) / float(n_views))))
            step_idx = predicted_depth_step_index(debug, n_views=n_views, num_steps=num_steps)
            cams: list[torch.Tensor] = []
            for v in range(n_views):
                vi = step_idx * n_views + v
                if vi < stack.shape[0]:
                    cams.append(stack[vi].detach().to(torch.float32).cpu())
            if len(cams) == n_views:
                pred_next = torch.stack(cams, dim=0)
        gt_raw = depths_from_obs(obs)
        gt_depth = None
        gt_units = "unavailable"
        if gt_raw is not None:
            converted = rollout_depth_to_meters(env, gt_raw.detach().cpu().numpy())
            if converted is not None:
                gt_depth = torch.from_numpy(converted).float()
                gt_units = "meters"
            else:
                gt_depth = gt_raw.detach().float()
                gt_units = "zbuffer_raw"
        rgb = None
        if collect_depth_forecast_rgb:
            cams: list[np.ndarray] = []
            for cam in LIBERO_CAMERA_NAMES:
                im = obs.get(f"{cam}_image")
                if im is None:
                    cams = []
                    break
                cams.append(np.ascontiguousarray(np.asarray(im)[..., :3].astype(np.uint8)))
            if cams:
                rgb = np.stack(cams, axis=0)  # (V, H, W, 3) uint8, env-oriented
        depth_forecast_records.append(
            {
                "policy_call_idx": int(policy_call_idx),
                "policy_obs_step": int(policy_obs_step),
                "n_views": int(n_views),
                "rotate_policy_input": bool(debug.get("rotate_policy_input", False)),
                "executed_sequence_start": debug.get("executed_sequence_start"),
                "predicted_steps": debug.get("predicted_steps"),
                "eval_crop_scale": float(debug.get("eval_crop_scale", 1.0) or 1.0),
                "pred_next_depth": pred_next,
                "gt_depth": gt_depth,
                "gt_depth_units": gt_units,
                "rgb": rgb,
            }
        )

    success = bool(done)
    executed_steps = 0
    policy_call_idx = 0
    target_object_prefix = _infer_libero_object_prefix(task_desc, obs)
    near_object_close_remaining = 0
    basket_release_remaining = 0
    rollout_wall_timeout_sec = float(rollout_wall_timeout_sec or 0.0)

    def rollout_timed_out(where: str) -> bool:
        if rollout_wall_timeout_sec <= 0:
            return False
        elapsed = time.time() - episode_start_time
        if elapsed < rollout_wall_timeout_sec:
            return False
        write_rollout_debug_event(
            {
                "event": "episode_wall_timeout",
                "where": where,
                "elapsed_sec": round(elapsed, 4),
                "timeout_sec": float(rollout_wall_timeout_sec),
                "executed_steps": int(executed_steps),
            }
        )
        write_action_trace(
            {
                "record_type": "episode_wall_timeout",
                "where": where,
                "elapsed_sec": round(elapsed, 4),
                "timeout_sec": float(rollout_wall_timeout_sec),
                "executed_steps": int(executed_steps),
            }
        )
        return True

    def timeout_result(where: str) -> dict[str, Any]:
        del where
        return finalize_result(
            {
                "success": False,
                "steps": int(executed_steps),
                "frames": frames,
                "split_frames": split_frames,
                "timeout": True,
            }
        )

    def apply_gripper_override(
        *,
        obs_before_step: dict[str, Any],
        base_env_action: np.ndarray,
        env_action: np.ndarray,
        history_action: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, dict[str, Any]]:
        nonlocal near_object_close_remaining, basket_release_remaining
        meta: dict[str, Any] = {
            "gripper_override": str(gripper_override),
            "gripper_override_applied": False,
        }
        override_mode = str(gripper_override)
        if override_mode == "default":
            return base_env_action, env_action, history_action, meta
        dist = _object_eef_distance(obs_before_step, target_object_prefix)
        basket_xy = _object_basket_xy_distance(obs_before_step, target_object_prefix)
        meta.update(
            {
                "target_object_prefix": target_object_prefix,
                "target_object_eef_distance": dist,
                "target_object_basket_xy_distance": basket_xy,
                "near_object_close_threshold": float(near_object_close_threshold),
                "near_object_close_remaining_before": int(near_object_close_remaining),
                "basket_release_threshold": float(basket_release_threshold),
                "basket_release_remaining_before": int(basket_release_remaining),
            }
        )
        if override_mode in {"near_object_close", "near_object_close_basket_release"} and dist is not None and dist <= float(near_object_close_threshold):
            near_object_close_remaining = max(
                near_object_close_remaining,
                max(1, int(near_object_close_hold_steps)),
            )
        if override_mode in {"basket_release_open", "near_object_close_basket_release"} and basket_xy is not None and basket_xy <= float(basket_release_threshold):
            basket_release_remaining = max(
                basket_release_remaining,
                max(1, int(basket_release_hold_steps)),
            )
        if near_object_close_remaining <= 0 and basket_release_remaining <= 0:
            return base_env_action, env_action, history_action, meta
        base_env_action = np.asarray(base_env_action, dtype=np.float32).copy()
        env_action = np.asarray(env_action, dtype=np.float32).copy()
        history_action = history_action.clone()
        if basket_release_remaining > 0:
            base_env_action[-1] = -1.0
            env_action[-1] = -1.0
            history_action[..., -1] = 0.0
            basket_release_remaining -= 1
            action = "open"
        else:
            base_env_action[-1] = 1.0
            env_action[-1] = 1.0
            history_action[..., -1] = 1.0
            near_object_close_remaining -= 1
            action = "close"
        meta.update(
            {
                "gripper_override_applied": True,
                "gripper_override_action": action,
                "near_object_close_remaining_after": int(near_object_close_remaining),
                "basket_release_remaining_after": int(basket_release_remaining),
            }
        )
        return base_env_action, env_action, history_action, meta

    if temporal_ensemble:
        # ACT-style temporal ensembling at the low-level action level. Each
        # policy call predicts `action_horizon` model-step chunks, each usually
        # containing 8 low-level actions. We execute only the ensembled current
        # low-level action, then re-observe and predict another forecast. History
        # is still committed after one full action-head chunk of executed
        # actions, preserving the training anchor stride used by gam.
        action_step_idx = 0
        action_buffer: dict[int, list[tuple[int, torch.Tensor]]] = {}
        anchor_history_snapshot: dict[str, Any] | None = None
        anchor_executed_actions: list[torch.Tensor] = []
        chunk_horizon = 1
        forecast_horizon = 1
        ensemble_model_steps = max(1, int(action_horizon))
        while executed_steps < max_steps and not success:
            if rollout_timed_out("before_policy_call"):
                return timeout_result("before_policy_call")
            policy_obs_step = executed_steps
            policy_call_idx += 1
            write_rollout_debug_event(
                {
                    "event": "policy_call_start",
                    "mode": "temporal_ensemble",
                    "policy_call_idx": int(policy_call_idx),
                    "policy_obs_step": int(policy_obs_step),
                    "executed_steps": int(executed_steps),
                }
            )
            canonical_actions = call_policy(policy, obs, task_desc)
            write_rollout_debug_event(
                {
                    "event": "policy_call_end",
                    "mode": "temporal_ensemble",
                    "policy_call_idx": int(policy_call_idx),
                    "policy_obs_step": int(policy_obs_step),
                    "executed_steps": int(executed_steps),
                    "canonical_action_shape": list(canonical_actions.shape),
                    "policy_debug": _policy_debug_lifecycle_summary(getattr(policy, "last_debug", None)),
                }
            )
            chunks = _as_model_step_chunks(canonical_actions, ensemble_model_steps)
            chunk_horizon = int(chunks.shape[1])
            predicted_forecast = chunks.reshape(-1, chunks.shape[-1])
            forecast_horizon = int(predicted_forecast.shape[0])
            if anchor_history_snapshot is None and hasattr(policy, "snapshot_pending_history"):
                anchor_history_snapshot = policy.snapshot_pending_history()  # type: ignore[attr-defined]
            for offset in range(forecast_horizon):
                target_step = action_step_idx + offset
                action_buffer.setdefault(target_step, []).append((action_step_idx, predicted_forecast[offset]))

            candidates = action_buffer.get(action_step_idx)
            if not candidates:
                candidates = [(action_step_idx, predicted_forecast[0])]
            step_action, num_ensemble_candidates = _weighted_temporal_ensemble_action(
                candidates,
                current_action_step=action_step_idx,
                decay=float(temporal_ensemble_decay),
            )
            if isinstance(getattr(policy, "last_debug", None), dict):
                policy.last_debug["temporal_ensemble"] = True
                policy.last_debug["temporal_ensemble_candidates"] = num_ensemble_candidates
                policy.last_debug["temporal_ensemble_decay"] = float(temporal_ensemble_decay)
                policy.last_debug["temporal_ensemble_forecast_horizon"] = forecast_horizon
                policy.last_debug["temporal_ensemble_model_steps"] = int(chunks.shape[0])
                policy.last_debug["temporal_ensemble_chunk_size"] = chunk_horizon
                policy.last_debug["temporal_ensemble_unit"] = "low_level_action"

            action_started = False
            base_env_action = canonical_to_libero_action(
                step_action,
                binarize_gripper=binarize_gripper,
                action_frame=action_frame,
                current_proprio=current_action_frame_proprio(obs),
                proprio_orientation=proprio_orientation,
            )
            history_action = canonical_action_for_history(
                step_action,
                binarize_gripper=binarize_gripper,
            ).cpu()
            env_action = apply_action_repeat_mode(
                base_env_action,
                action_repeat=max(1, int(action_repeat)),
                mode=action_repeat_mode,
            )
            base_env_action, env_action, history_action, gripper_override_meta = apply_gripper_override(
                obs_before_step=obs,
                base_env_action=base_env_action,
                env_action=env_action,
                history_action=history_action,
            )
            repeats_this_action = min(max(1, int(action_repeat)), max_steps - executed_steps)
            for repeat_idx in range(repeats_this_action):
                if rollout_timed_out("before_env_step"):
                    return timeout_result("before_env_step")
                obs, _, done, env_info = env.step(env_action)
                env_info_trace = env_step_trace_info(env_info)
                executed_steps += 1
                action_started = True
                if executed_steps % heartbeat_steps == 0 or bool(done) or executed_steps >= max_steps:
                    write_rollout_debug_event(
                        {
                            "event": "env_step_heartbeat",
                            "mode": "temporal_ensemble",
                            "policy_call_idx": int(policy_call_idx),
                            "policy_obs_step": int(policy_obs_step),
                            "executed_step": int(executed_steps),
                            "action_step_idx": int(action_step_idx),
                            "repeat_idx": int(repeat_idx),
                            "success_after_step": bool(done),
                            "env_action": env_action,
                            **env_info_trace,
                            "obs_summary": _small_numeric_obs_summary(obs),
                        }
                    )
                write_action_trace(
                    {
                        "record_type": "env_step",
                        "mode": "temporal_ensemble",
                        "policy_call_idx": int(policy_call_idx),
                        "policy_obs_step": int(policy_obs_step),
                        "executed_step": int(executed_steps),
                        "action_step_idx": int(action_step_idx),
                        "repeat_idx": int(repeat_idx),
                        "action_repeat": int(action_repeat),
                        "action_repeat_mode": action_repeat_mode,
                        "canonical_action_model": step_action,
                        "canonical_action_history": history_action,
                        "canonical_gripper_model": float(step_action.detach().cpu().reshape(-1)[-1].item()),
                        "canonical_gripper_history": float(history_action.reshape(-1)[-1].item()),
                        "action_frame": action_frame,
                        "proprio_orientation": proprio_orientation,
                        "gripper_close_threshold": 0.5 if binarize_gripper else None,
                        "base_env_action": base_env_action,
                        "env_action": env_action,
                        "env_gripper": float(np.asarray(env_action).reshape(-1)[-1]),
                        **env_info_trace,
                        "temporal_ensemble_candidates": int(num_ensemble_candidates),
                        **gripper_override_meta,
                        "success_after_step": bool(done),
                        "obs_after_step": _small_numeric_obs_summary(obs),
                        "policy_debug": _policy_debug_trace_snapshot(getattr(policy, "last_debug", None)),
                    }
                )

                if detailed_video and getattr(policy, "last_debug", None):
                    live_frame = render_rollout_frame(
                        env,
                        camera_size=camera_size,
                        obs=obs,
                        camera_names=video_camera_names,
                    )
                    append_split_video_frame(split_frames, getattr(policy, "last_debug", None), live_frame, obs=obs)
                    frames.append(
                        render_detailed_policy_frame(
                            policy.last_debug,
                            action_idx=action_step_idx % max(1, chunk_horizon),
                            repeat_idx=repeat_idx + 1,
                            policy_call_idx=policy_call_idx,
                            policy_obs_step=policy_obs_step,
                            executed_steps=executed_steps,
                            action_horizon=ensemble_model_steps,
                            action_repeat=max(1, int(action_repeat)),
                            action_repeat_mode=action_repeat_mode,
                            env_action=env_action,
                            success=bool(done),
                            live_frame=live_frame,
                        )
                    )
                elif record_video or split_frames is not None:
                    live_frame = render_rollout_frame(
                        env,
                        camera_size=camera_size,
                        obs=obs,
                        camera_names=video_camera_names,
                    )
                    append_split_video_frame(split_frames, getattr(policy, "last_debug", None), live_frame, obs=obs)
                    if record_video:
                        frames.append(live_frame)

                if done:
                    success = True
                    break
                if rollout_timed_out("after_env_step"):
                    return timeout_result("after_env_step")
                if executed_steps >= max_steps:
                    break

            if action_started:
                anchor_executed_actions.append(history_action)
                action_step_idx += 1
                for key in list(action_buffer.keys()):
                    if key < action_step_idx:
                        del action_buffer[key]
                if len(anchor_executed_actions) >= chunk_horizon:
                    commit_policy_observation = getattr(policy, "commit_observation", None)
                    if (
                        callable(commit_policy_observation)
                        and anchor_history_snapshot is not None
                        and hasattr(policy, "restore_pending_history")
                    ):
                        policy.restore_pending_history(anchor_history_snapshot)  # type: ignore[attr-defined]
                        if hasattr(policy, "override_pending_action_chunk"):
                            policy.override_pending_action_chunk(  # type: ignore[attr-defined]
                                torch.stack(anchor_executed_actions, dim=0)
                            )
                        commit_policy_observation(1)
                    anchor_history_snapshot = None
                    anchor_executed_actions.clear()

        write_rollout_debug_event(
            {
                "event": "episode_end",
                "success": bool(success),
                "steps": int(executed_steps),
            }
        )
        return finalize_result(
            {
                "success": bool(success),
                "steps": int(executed_steps),
                "frames": frames,
                "split_frames": split_frames,
            }
        )

    while executed_steps < max_steps and not success:
        if rollout_timed_out("before_policy_call"):
            return timeout_result("before_policy_call")
        policy_obs_step = executed_steps
        policy_call_idx += 1
        write_rollout_debug_event(
            {
                "event": "policy_call_start",
                "mode": "standard",
                "policy_call_idx": int(policy_call_idx),
                "policy_obs_step": int(policy_obs_step),
                "executed_steps": int(executed_steps),
            }
        )
        policy_call_anchor_proprio = current_action_frame_proprio(obs)
        canonical_actions = call_policy(policy, obs, task_desc)
        capture_depth_forecast_record()
        write_rollout_debug_event(
            {
                "event": "policy_call_end",
                "mode": "standard",
                "policy_call_idx": int(policy_call_idx),
                "policy_obs_step": int(policy_obs_step),
                "executed_steps": int(executed_steps),
                "canonical_action_shape": list(canonical_actions.shape),
                "policy_debug": _policy_debug_lifecycle_summary(getattr(policy, "last_debug", None)),
            }
        )
        strategy_meta: dict[str, Any] = {
            "execution_strategy": str(execution_strategy),
            "strategy_applied": False,
        }
        if str(execution_strategy) == "full_plan_all":
            selected_actions, strategy_meta = _select_full_plan_all(
                getattr(policy, "last_debug", None),
            )
            strategy_meta["execution_strategy"] = str(execution_strategy)
            if selected_actions is not None:
                canonical_actions = selected_actions
        elif str(execution_strategy) in {"full_plan_close_prefix", "full_plan_close_to_end"}:
            selected_actions, strategy_meta = _select_full_plan_close_prefix(
                getattr(policy, "last_debug", None),
                close_threshold=float(full_plan_close_threshold),
                through_end=str(execution_strategy) == "full_plan_close_to_end",
            )
            strategy_meta["execution_strategy"] = str(execution_strategy)
            if selected_actions is not None:
                canonical_actions = selected_actions
        if canonical_actions.ndim == 1:
            canonical_actions = canonical_actions.view(1, -1)
        chunked_actions = canonical_actions.ndim == 3
        horizon = min(int(action_horizon), canonical_actions.shape[0])
        if strategy_meta.get("strategy_applied"):
            horizon = int(canonical_actions.shape[0])
        if isinstance(getattr(policy, "last_debug", None), dict):
            policy.last_debug["execute_chunk_prefix"] = int(execute_chunk_prefix)
            policy.last_debug["partial_chunk_history"] = partial_chunk_history
            policy.last_debug["warmup_full_chunk_once"] = bool(warmup_full_chunk_once)
        policy_actions_executed = 0
        executed_history_chunks: list[torch.Tensor] = []

        for i in range(horizon):
            action_started = False
            step_actions = canonical_actions[i] if chunked_actions else canonical_actions[i : i + 1]
            if step_actions.ndim == 1:
                step_actions = step_actions.view(1, -1)
            chunk_len = int(step_actions.shape[0])
            if rolling_history_chunk_size is None:
                rolling_history_chunk_size = max(1, chunk_len)
            execute_limit = chunk_len
            prefix_warmup_this_chunk = False
            if prefix_replan_enabled:
                if warmup_full_chunk_pending:
                    prefix_warmup_this_chunk = True
                else:
                    execute_limit = min(max(1, int(execute_chunk_prefix)), chunk_len)
                    if (
                        partial_chunk_history == "rolling_last_k"
                        and execute_limit < chunk_len
                        and len(rolling_history_actions) < chunk_len
                    ):
                        raise ValueError(
                            "rolling_last_k partial chunk history needs one full action chunk before "
                            "prefix execution. Use --warmup-full-chunk-once or set "
                            "--execute-chunk-prefix >= action chunk size."
                        )
            if isinstance(getattr(policy, "last_debug", None), dict):
                policy.last_debug["execute_chunk_prefix_effective"] = int(execute_limit)
                policy.last_debug["warmup_full_chunk_pending"] = bool(warmup_full_chunk_pending)
                policy.last_debug["warmup_full_chunk_this_call"] = bool(prefix_warmup_this_chunk)
            executed_step_history_actions: list[torch.Tensor] = []
            for chunk_idx in range(execute_limit):
                base_env_action = canonical_to_libero_action(
                    step_actions[chunk_idx],
                    binarize_gripper=binarize_gripper,
                    action_frame=action_frame,
                    current_proprio=current_action_frame_proprio(obs),
                    anchor_proprio=policy_call_anchor_proprio,
                    proprio_orientation=proprio_orientation,
                )
                history_action = canonical_action_for_history(
                    step_actions[chunk_idx],
                    binarize_gripper=binarize_gripper,
                ).cpu()
                env_action = apply_action_repeat_mode(
                    base_env_action,
                    action_repeat=max(1, int(action_repeat)),
                    mode=action_repeat_mode,
                )
                base_env_action, env_action, history_action, gripper_override_meta = apply_gripper_override(
                    obs_before_step=obs,
                    base_env_action=base_env_action,
                    env_action=env_action,
                    history_action=history_action,
                )
                repeats_this_action = min(max(1, int(action_repeat)), max_steps - executed_steps)
                chunk_action_started = False
                # Final-stage env_action safety net. If anything upstream
                # (eef-relative inverse transform with degenerate proprio,
                # cascaded NaN proprio in mujoco state, etc.) produced a
                # non-finite env_action, replace with a zero open-gripper
                # action so mujoco physics stays stable into the next step.
                if not np.isfinite(env_action).all():
                    print(
                        f"[rollout WARNING] non-finite env_action replaced with zero "
                        f"(step={executed_steps} chunk_idx={chunk_idx})",
                        flush=True,
                    )
                    env_action = np.zeros_like(env_action, dtype=np.float32)
                    if env_action.shape[0] >= 7:
                        env_action[-1] = -1.0
                for repeat_idx in range(repeats_this_action):
                    if rollout_timed_out("before_env_step"):
                        return timeout_result("before_env_step")
                    obs, _, done, env_info = env.step(env_action)
                    env_info_trace = env_step_trace_info(env_info)
                    # Post-env.step proprio sanity check. OXE-pretrain ckpt +
                    # eef_relative + spatial (OOD combo) produces actions that
                    # are individually finite/small while accumulating over ~200
                    # steps into NaN/Inf qpos. Subsequent env.step or LIBERO
                    # predicate evaluation then enters an infinite native loop
                    # (stuck stacks observed at binding_utils.step / predicates
                    # eval_predicate_fn). Detect divergence the moment it
                    # crosses into non-finite proprio and abort the rollout
                    # cleanly so the eval scores it as a crash instead of the
                    # whole DDP job dying.
                    try:
                        _post_ps = obs_to_canonical_7d_proprio(
                            obs, orientation_mode=proprio_orientation
                        )
                        if isinstance(_post_ps, torch.Tensor) and not bool(torch.isfinite(_post_ps).all()):
                            print(
                                f"[rollout WARNING] non-finite proprio after env.step "
                                f"(step={executed_steps} chunk_idx={chunk_idx}); aborting rollout as crash",
                                flush=True,
                            )
                            return finalize_result(
                                {
                                    "success": False,
                                    "steps": int(executed_steps),
                                    "frames": frames,
                                    "split_frames": split_frames,
                                    "timeout": False,
                                    "crash": True,
                                }
                            )
                    except Exception:  # noqa: BLE001
                        pass
                    executed_steps += 1
                    action_started = True
                    chunk_action_started = True
                    if executed_steps % heartbeat_steps == 0 or bool(done) or executed_steps >= max_steps:
                        write_rollout_debug_event(
                            {
                                "event": "env_step_heartbeat",
                                "mode": "standard",
                                "policy_call_idx": int(policy_call_idx),
                                "policy_obs_step": int(policy_obs_step),
                                "executed_step": int(executed_steps),
                                "model_step_idx": int(i),
                                "chunk_idx": int(chunk_idx),
                                "repeat_idx": int(repeat_idx),
                                "success_after_step": bool(done),
                                "env_action": env_action,
                                **env_info_trace,
                                "obs_summary": _small_numeric_obs_summary(obs),
                            }
                        )
                    write_action_trace(
                        {
                            "record_type": "env_step",
                            "mode": "standard",
                            "policy_call_idx": int(policy_call_idx),
                            "policy_obs_step": int(policy_obs_step),
                            "executed_step": int(executed_steps),
                            "model_step_idx": int(i),
                            "chunk_idx": int(chunk_idx),
                            "flat_action_idx": int(i * max(1, step_actions.shape[0]) + chunk_idx),
                            "repeat_idx": int(repeat_idx),
                            "action_repeat": int(action_repeat),
                            "action_repeat_mode": action_repeat_mode,
                            "execute_chunk_prefix": int(execute_chunk_prefix),
                            "execute_chunk_prefix_effective": int(execute_limit),
                            "partial_chunk_history": partial_chunk_history,
                            "warmup_full_chunk_once": bool(warmup_full_chunk_once),
                            "warmup_full_chunk_this_call": bool(prefix_warmup_this_chunk),
                            **strategy_meta,
                            **gripper_override_meta,
                            "canonical_action_model": step_actions[chunk_idx],
                            "canonical_action_history": history_action,
                            "canonical_gripper_model": float(step_actions[chunk_idx].detach().cpu().reshape(-1)[-1].item()),
                            "canonical_gripper_history": float(history_action.reshape(-1)[-1].item()),
                            "action_frame": action_frame,
                            "proprio_orientation": proprio_orientation,
                            "gripper_close_threshold": 0.5 if binarize_gripper else None,
                            "base_env_action": base_env_action,
                            "env_action": env_action,
                            "env_gripper": float(np.asarray(env_action).reshape(-1)[-1]),
                            **env_info_trace,
                            "success_after_step": bool(done),
                            "obs_after_step": _small_numeric_obs_summary(obs),
                            "policy_debug": _policy_debug_trace_snapshot(getattr(policy, "last_debug", None)),
                        }
                    )

                    if detailed_video and getattr(policy, "last_debug", None):
                        live_frame = render_rollout_frame(
                            env,
                            camera_size=camera_size,
                            obs=obs,
                            camera_names=video_camera_names,
                        )
                        append_split_video_frame(split_frames, getattr(policy, "last_debug", None), live_frame, obs=obs)
                        flat_action_idx = i * max(1, step_actions.shape[0]) + chunk_idx
                        frames.append(
                            render_detailed_policy_frame(
                                policy.last_debug,
                                action_idx=flat_action_idx,
                                repeat_idx=repeat_idx + 1,
                                policy_call_idx=policy_call_idx,
                                policy_obs_step=policy_obs_step,
                                executed_steps=executed_steps,
                                action_horizon=int(horizon),
                                action_repeat=max(1, int(action_repeat)),
                                action_repeat_mode=action_repeat_mode,
                                env_action=env_action,
                                success=bool(done),
                                live_frame=live_frame,
                            )
                        )
                    elif record_video or split_frames is not None:
                        live_frame = render_rollout_frame(
                            env,
                            camera_size=camera_size,
                            obs=obs,
                            camera_names=video_camera_names,
                        )
                        append_split_video_frame(split_frames, getattr(policy, "last_debug", None), live_frame, obs=obs)
                        if record_video:
                            frames.append(live_frame)

                    if done:
                        success = True
                        break
                    if rollout_timed_out("after_env_step"):
                        return timeout_result("after_env_step")
                    if executed_steps >= max_steps:
                        break
                if chunk_action_started:
                    executed_step_history_actions.append(history_action.detach().cpu())
                    if prefix_replan_enabled:
                        rolling_history_actions.append(history_action.detach().cpu())
                        if rolling_history_chunk_size is not None:
                            del rolling_history_actions[:-max(1, rolling_history_chunk_size)]
                if success or executed_steps >= max_steps:
                    break
            if action_started:
                policy_actions_executed += 1
                if prefix_replan_enabled and executed_step_history_actions:
                    executed_history_chunks.append(torch.stack(executed_step_history_actions, dim=0))
                    if prefix_warmup_this_chunk:
                        warmup_full_chunk_pending = False
                else:
                    executed_history_chunks.append(
                        canonical_action_for_history(step_actions, binarize_gripper=binarize_gripper).cpu()
                    )
            if success or executed_steps >= max_steps:
                break
        commit_policy_observation = getattr(policy, "commit_observation", None)
        if callable(commit_policy_observation):
            if (
                policy_actions_executed == 1
                and executed_history_chunks
                and hasattr(policy, "override_pending_action_chunk")
            ):
                override_chunk = executed_history_chunks[0]
                if (
                    prefix_replan_enabled
                    and partial_chunk_history == "rolling_last_k"
                    and rolling_history_chunk_size is not None
                    and len(rolling_history_actions) >= rolling_history_chunk_size
                ):
                    override_chunk = torch.stack(
                        rolling_history_actions[-rolling_history_chunk_size:],
                        dim=0,
                    )
                policy.override_pending_action_chunk(  # type: ignore[attr-defined]
                    override_chunk
                )
            commit_policy_observation(policy_actions_executed)

    write_rollout_debug_event(
        {
            "event": "episode_end",
            "success": bool(success),
            "steps": int(executed_steps),
        }
    )
    return finalize_result(
        {
            "success": bool(success),
            "steps": int(executed_steps),
            "frames": frames,
            "split_frames": split_frames,
        }
    )


def append_eval_registry(args: argparse.Namespace, summary: dict[str, Any], policy_info: dict[str, Any]) -> str:
    metrics = {
        "average_success_rate": round(float(summary["average_success_rate"]), 4),
        "overall_success_rate": round(float(summary.get("overall_success_rate", summary["average_success_rate"])), 4),
        "total_successes": int(summary.get("total_successes", 0)),
        "total_trials": int(summary.get("total_episodes", 0)),
        "elapsed_sec": round(float(summary.get("elapsed_sec", 0.0) or 0.0), 3),
        **{
            f"{suite}_success_rate": round(float(result["success_rate"]), 4)
            for suite, result in summary["suite_results"].items()
        },
    }
    for suite, result in summary["suite_results"].items():
        metrics[f"{suite}_num_success"] = int(result.get("num_success", 0))
        metrics[f"{suite}_num_trials"] = int(result.get("num_trials", 0))
        if result.get("avg_steps") is not None:
            metrics[f"{suite}_avg_steps"] = round(float(result["avg_steps"]), 3)
    if summary.get("plus"):
        for slug, result in summary.get("plus_official_category_results", {}).items():
            metrics[f"plus_category_{slug}_success_rate"] = round(float(result["success_rate"]), 4)
            metrics[f"plus_category_{slug}_num_trials"] = int(result["num_trials"])

    run_name = summary.get("run_name")
    root_run_dir = Path(args.output_dir) / str(run_name) if run_name else None
    artifacts = {
        "result_dir": str(root_run_dir) if root_run_dir else None,
        "summary_json": str(root_run_dir / "summary.json") if root_run_dir else None,
        "per_task_csv": str(root_run_dir / "per_task.csv") if root_run_dir else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    artifacts = {key: value for key, value in artifacts.items() if value}

    entry = {
        "date": local_now_minute(),
        "ckpt_path": args.ckpt,
        "ckpt_step": policy_info.get("train_steps", 0),
        "config": summary.get("config", args.config),
        "eval_data": "libero_plus_closed_loop" if summary.get("plus") else "libero_closed_loop",
        "protocol": args.preset,
        "suites": summary["suites"],
        "plus": bool(summary.get("plus", False)),
        "plus_perturbation": summary.get("plus_perturbation"),
        "plus_official_category": summary.get("plus_official_category"),
        "n_episodes": summary["total_episodes"],
        "metrics": metrics,
        "suite_results": summary.get("suite_results", {}),
        "plus_official_classification": summary.get("plus_official_classification"),
        "run_name": run_name,
        "task_ids": args.task_ids,
        "registry_tier": None if args.registry_tier == "auto" else args.registry_tier,
        "policy": {
            "stage": policy_info.get("stage"),
            "use_ema": bool(policy_info.get("use_ema", False)),
            "ema_loaded_keys": policy_info.get("ema_loaded_keys", []),
            "action_stats_key": policy_info.get("action_stats_key"),
            "action_frame": policy_info.get("action_frame"),
            "rollout_action_frame": policy_info.get("rollout_action_frame"),
            "proprio_orientation": policy_info.get("proprio_orientation"),
            "rollout_decode_horizon": policy_info.get("stage1_rollout_decode_horizon"),
            "rollout_decode_horizon_mode": policy_info.get("stage1_rollout_decode_horizon_mode"),
            "dataset_target_hz": policy_info.get("dataset_target_hz"),
            "action_repeat_requested": summary.get("action_repeat_requested"),
            "action_repeat_mode": summary.get("action_repeat_mode"),
            "env_control_hz_override": summary.get("env_control_hz_override"),
            "temporal_ensemble": summary.get("temporal_ensemble"),
            "temporal_ensemble_decay": summary.get("temporal_ensemble_decay"),
        },
        "artifacts": artifacts,
        "shard": summary.get("shard"),
        "note": args.note,
    }
    return append_eval_record(
        args.registry,
        entry,
        repo_root=_REPO_ROOT,
        entrypoint="src/eval_libero_unified.py",
        record_type="closed_loop_rollout",
        tier=None if args.registry_tier == "auto" else args.registry_tier,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified LIBERO closed-loop rollout eval")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Stage 1 architecture config")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path (required).",
    )
    parser.add_argument("--stage", choices=["auto", "1"], default="auto")
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help=(
            "For Stage 1 checkpoints, load EMA overlays for the saved gam "
            "policy subset (future_predictor/action_head/text_proj/DA3 deep blocks)."
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="openvla_50")
    parser.add_argument("--suites", type=str, default=",".join(LIBERO_SUITES))
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated task ids within each suite")
    parser.add_argument("--num-trials-per-task", "--n-episodes", dest="num_trials_per_task", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Override preset horizon for every suite")
    parser.add_argument("--num-steps-wait", type=int, default=None, help="Override preset dummy wait steps")
    parser.add_argument(
        "--action-horizon",
        type=str,
        default=None,
        help=(
            "H_exec: AR chunks executed per policy call (= re-observe cadence). "
            "For gam, one chunk = K_chunk low-level actions × R_env env steps. "
            "H_exec=1 -> re-observe after every chunk (real-obs only); "
            "H_exec='all' -> commit to full predicted AR sequence (open-loop). "
            "The generated/decode prefix length is controlled separately by "
            "--rollout-decode-horizon."
        ),
    )
    parser.add_argument(
        "--action-repeat",
        type=str,
        default="auto",
        help=(
            "Env steps to execute for each predicted action. 'auto' aligns "
            "checkpoint dataset.target_hz to the LIBERO env control_freq. "
            "Use 1 to reproduce the old one-env-step behavior."
        ),
    )
    parser.add_argument(
        "--action-repeat-mode",
        choices=["hold", "split_delta"],
        default="split_delta",
        help=(
            "How to execute repeated actions: hold repeats the raw env action; "
            "split_delta divides xyz/rpy deltas across repeats and keeps gripper unchanged. "
            "The default matches target_hz action aggregation in the robot dataset loader."
        ),
    )
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=None,
        help="Override checkpoint dataset.target_hz for --action-repeat auto.",
    )
    parser.add_argument(
        "--history-horizon",
        type=str,
        default="auto",
        help=(
            "Stage 1 observed-slot history horizon. 'auto' uses max predictor.H_choices "
            "for unified FuturePredictor checkpoints and 1 otherwise."
        ),
    )
    parser.add_argument(
        "--rollout-decode-horizon",
        type=str,
        default="full",
        help=(
            "For gam Stage 1 rollout, AR-generate this many model-step "
            "slots before DA3 deep refine/action decode, independent of "
            "--action-horizon execution. 'full' (default) uses the checkpoint's "
            "native train H from predictor.H_choices, matching train-time full-H deep refine length; "
            "'exec' restores the old short-prefix behavior."
        ),
    )
    parser.add_argument(
        "--env-control-hz",
        type=float,
        default=20.0,
        help=(
            "Live LIBERO robosuite control_freq. Default 20Hz matches LIBERO HDF5 "
            "demo native rate (env_args.control_freq=20 in the HDF5 metadata; "
            "each demo action is a 0.05s OSC_POSE delta). Running eval at any "
            "other env_hz distorts per-action execution time; the same action "
            "delta then covers a different physical duration than training."
        ),
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=256,
        help=(
            "Robosuite camera render size (H=W). Default 256 matches the current "
            "train-time LIBERO closed-loop profiles and CSCS eval launchers. "
            "Override explicitly when probing a render-resolution ablation."
        ),
    )
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=None,
        help=(
            "robosuite render_gpu_device_id override. Under MUJOCO_GL=egl, the default "
            "resolves the rank-local CUDA_VISIBLE_DEVICES token required by robosuite."
        ),
    )
    parser.add_argument(
        "--env-process-isolation",
        action="store_true",
        help=(
            "Own each LIBERO/MuJoCo env in a spawned child process. Native EGL aborts "
            "then count as crashed episodes instead of killing the eval rank."
        ),
    )
    parser.add_argument(
        "--env-worker-timeout-sec",
        type=float,
        default=300.0,
        help="Per-command timeout for --env-process-isolation worker IPC.",
    )
    parser.add_argument("--output-dir", type=str, default="results/eval_libero")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--registry", type=str, default="docs/experiments/eval_registry.json")
    parser.add_argument("--no-registry", action="store_true", help="Skip docs/experiments/eval_registry.json append")
    parser.add_argument(
        "--registry-tier",
        choices=["auto", "canonical", "candidate", "smoke", "debug", "subset", "sweep", "diagnostic", "legacy"],
        default="auto",
        help="Classification for eval_registry.json; default infers from protocol, task filter, and Plus coverage.",
    )
    parser.add_argument(
        "--shard-index",
        type=str,
        default="auto",
        help="Eval worker shard index. 'auto' reads SLURM_PROCID/RANK.",
    )
    parser.add_argument(
        "--shard-count",
        type=str,
        default="auto",
        help="Total eval shards. 'auto' reads SLURM_NTASKS/WORLD_SIZE.",
    )
    parser.add_argument(
        "--shard-wait-timeout-sec",
        type=float,
        default=86400.0,
        help="Shard-0 timeout while waiting for distributed shard outputs.",
    )
    parser.add_argument("--render", "--record-video", dest="record_video", action="store_true")
    parser.add_argument(
        "--video-every",
        type=int,
        default=0,
        help=(
            "Force record_video + trace_actions + raw obs dump for every Nth "
            "episode in the global ordering (shard-aware). 0 disables. Use "
            "--video-every 100 to sample one rollout per 100 episodes for "
            "qualitative debugging without saving all 10k MP4s."
        ),
    )
    parser.add_argument("--detailed-video", action="store_true", help="Save RGB/depth/action diagnostic rollout video")
    parser.add_argument(
        "--trace-actions",
        action="store_true",
        help="Write per-env-step action/proprio/policy diagnostics as JSONL without requiring video.",
    )
    parser.add_argument(
        "--rollout-debug-log",
        action="store_true",
        help=(
            "Write flushed per-episode lifecycle JSONL logs. Useful for diagnosing "
            "native aborts because the last event shows whether rollout was in "
            "init-state reset, wait steps, policy forward, or env stepping."
        ),
    )
    parser.add_argument(
        "--rollout-debug-heartbeat-steps",
        type=int,
        default=25,
        help="Env-step interval for --rollout-debug-log heartbeat records.",
    )
    parser.add_argument(
        "--decode-visuals",
        action="store_true",
        help="Decode Stage 2 depth/RGB even without --detailed-video; slower but useful for diagnostics",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help=(
            "ACT-style low-level temporal ensemble. The policy predicts one "
            "--action-horizon model-step forecast each env action, averages "
            "overlapping predictions for the current action, and commits "
            "history only after a full action-head chunk."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.01,
        help="Exponential age decay for --temporal-ensemble; 0 means uniform averaging.",
    )
    parser.add_argument(
        "--execution-strategy",
        choices=["default", "full_plan_close_prefix", "full_plan_close_to_end", "full_plan_all"],
        default="default",
        help=(
            "Rollout execution strategy. default executes the returned policy action chunks. "
            "full_plan_close_prefix is a diagnostic gam strategy: keep the policy "
            "forward horizon unchanged, but execute the full decoded plan prefix through "
            "the first gripper-close action when such a close appears in policy debug. "
            "full_plan_close_to_end waits for such a close-bearing debug plan, then executes "
            "from current slot through the end of that plan. "
            "full_plan_all keeps the same forward horizon but executes the current-slot "
            "through end of that decoded full plan."
        ),
    )
    parser.add_argument(
        "--execute-chunk-prefix",
        type=int,
        default=0,
        help=(
            "Diagnostic subchunk replanning mode. When >0, execute only this "
            "many low-level actions from each predicted action-head chunk before "
            "re-observing."
        ),
    )
    parser.add_argument(
        "--partial-chunk-history",
        choices=["default", "rolling_last_k"],
        default="default",
        help=(
            "Previous-action history policy for --execute-chunk-prefix. "
            "rolling_last_k stores the most recent actually executed K actions "
            "as the next previous-action chunk."
        ),
    )
    parser.add_argument(
        "--warmup-full-chunk-once",
        action="store_true",
        help=(
            "With --execute-chunk-prefix, execute one full action chunk at the "
            "start of each episode so rolling_last_k history has a full buffer."
        ),
    )
    parser.add_argument(
        "--full-plan-close-threshold",
        type=float,
        default=0.5,
        help="Canonical gripper threshold used by --execution-strategy full_plan_close_prefix.",
    )
    parser.add_argument(
        "--gripper-override",
        choices=["default", "near_object_close", "basket_release_open", "near_object_close_basket_release"],
        default="default",
        help=(
            "Privileged diagnostic gripper override. default leaves policy actions unchanged. "
            "near_object_close forces a close latch when the target object is within "
            "--near-object-close-threshold of the EEF. basket_release_open forces an open "
            "latch once the target object is close to the basket in XY. Treat these "
            "as diagnostic runs outside default policy success."
        ),
    )
    parser.add_argument(
        "--near-object-close-threshold",
        type=float,
        default=0.04,
        help="EEF-to-target-object distance in meters that triggers near_object_close.",
    )
    parser.add_argument(
        "--near-object-close-hold-steps",
        type=int,
        default=80,
        help="Number of low-level env steps to keep the gripper closed after near_object_close triggers.",
    )
    parser.add_argument(
        "--basket-release-threshold",
        type=float,
        default=0.06,
        help="Target-object to basket XY distance in meters that triggers basket_release_open.",
    )
    parser.add_argument(
        "--basket-release-hold-steps",
        type=int,
        default=80,
        help="Number of low-level env steps to keep the gripper open after basket_release_open triggers.",
    )
    parser.add_argument(
        "--rotate-policy-input",
        action="store_true",
        help="Rotate live RGB by 180 degrees before policy inference. Default is raw env orientation.",
    )
    parser.add_argument(
        "--proprio-orientation",
        type=str,
        default="auto",
        help=(
            "Live proprio orientation representation: 'auto' picks axis_angle for current "
            "libero_noop HDF5 checkpoints and rpy otherwise; override with 'rpy' or 'axis_angle'."
        ),
    )
    parser.add_argument(
        "--text-prompt-normalization",
        type=str,
        default="libero_hdf5_task_text",
        choices=[
            "libero_hdf5_task_text",
            "raw_task_text",
            "libero",
            "raw",
        ],
        help=(
            "Language prompt normalization before text encoding. LIBERO eval defaults to "
            "libero_hdf5_task_text; use raw_task_text to pass the raw training "
            "task_description strings unchanged."
        ),
    )
    parser.add_argument(
        "--action-frame",
        type=str,
        default="auto",
        help=(
            "Model action frame for rollout. 'auto' uses dataset.action_frame from the checkpoint/config. "
            "Use 'base_delta' for base/world-frame OSC_POSE deltas, "
            "'eef_delta' for the old moving-frame EEF-local delta, and 'eef_relative' for "
            "chunk-start relative target trajectories. 'base' remains a compatibility alias."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--num-ode-steps", type=int, default=10, help="Stage 2 ODE sampling steps")
    parser.add_argument("--sampling-method", type=str, default="euler", help="Stage 2 ODE sampler method")
    parser.add_argument(
        "--split-inference-mode",
        type=str,
        default="joint",
        choices=["joint", "policy", "dynamics", "video_only"],
        help="Stage 2 split-schedule inference mode; live rollout supports joint/video_only only",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="da3-action-finetune")
    parser.add_argument("--action-stats-key", type=str, default=None)
    parser.add_argument("--action-stats-json", type=str, default=None)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--plus", "--pro", dest="plus", action="store_true")
    parser.add_argument("--plus-root", "--pro-root", dest="plus_root", type=str, default=os.environ.get("DA3_LIBERO_PLUS_DIR"))
    parser.add_argument("--plus-perturbation", "--pro-perturbation", dest="plus_perturbation", type=str, default="all")
    parser.add_argument(
        "--libero-plus-robot-init-qpos-mode",
        choices=["preserve", "original"],
        default="preserve",
        help=(
            "LIBERO-Plus robot initial-state handling. 'preserve' keeps DA3's "
            "post-set_init_state robot qpos restore; 'original' matches upstream "
            "LIBERO-Plus by leaving the flattened base init state untouched."
        ),
    )
    parser.add_argument(
        "--plus-official-category",
        "--pro-official-category",
        dest="plus_official_category",
        type=str,
        default="all",
        help=(
            "Filter LIBERO-Plus tasks by official task_classification category. "
            "Accepts all, camera, robot, language, light, background, noise, layout, "
            "or comma-separated combinations. Use this instead of --plus-perturbation "
            "when reproducing paper category rows."
        ),
    )
    parser.add_argument(
        "--plus-sample-group-by",
        choices=["none", "official_category", "perturbation", "suite_category"],
        default="none",
        help="Deterministically sample LIBERO-Plus tasks by group after Plus filters.",
    )
    parser.add_argument(
        "--plus-samples-per-group",
        type=int,
        default=0,
        help="Number of LIBERO-Plus tasks to keep per sampled group; <=0 disables sampling.",
    )
    parser.add_argument(
        "--plus-sample-seed",
        type=int,
        default=0,
        help="Stable seed for --plus-sample-group-by deterministic task sampling.",
    )
    parser.add_argument("--note", type=str, default="")
    return parser


def _parse_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw).split(",", 1)[0])
    except ValueError:
        return None


def _is_fatal_rollout_exception(exc: Exception, args: argparse.Namespace) -> bool:
    """Identify infrastructure errors that make the whole eval invalid.

    Task failures should be counted as failures, but server/env dependency
    wiring errors should stop the run instead of silently producing many 0-SR
    episodes.
    """
    text = f"{type(exc).__name__}: {exc}"
    if bool(getattr(args, "plus", False)):
        dependency_markers = (
            "MagickWand shared library missing",
            "LIBERO-Plus motion-blur perturbations require",
        )
        if any(marker in text for marker in dependency_markers):
            return True
    return False


def _cuda_visible_device_tokens() -> list[str]:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not visible:
        return []
    return [token.strip() for token in visible.split(",") if token.strip()]


def _rank_local_visible_cuda_token() -> str | None:
    tokens = _cuda_visible_device_tokens()
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0]
    if torch.cuda.is_available():
        visible_idx = int(torch.cuda.current_device())
    else:
        local_rank = _parse_int_env("LOCAL_RANK")
        if local_rank is None:
            local_rank = _parse_int_env("SLURM_LOCALID")
        visible_idx = int(local_rank or 0)
    return tokens[visible_idx % len(tokens)]


def _resolve_eval_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    local_rank = _parse_int_env("LOCAL_RANK")
    if local_rank is None:
        local_rank = _parse_int_env("SLURM_LOCALID")
    if visible and "," not in visible:
        device_idx = 0
    elif local_rank is not None:
        device_idx = local_rank % max(1, torch.cuda.device_count())
    else:
        device_idx = 0
    torch.cuda.set_device(device_idx)
    return torch.device(f"cuda:{device_idx}")


def _resolve_render_gpu_device_id(cli_value: int | None) -> int | None:
    mujoco_gl = str(os.environ.get("MUJOCO_GL") or os.environ.get("DA3_MUJOCO_GL") or "").lower()
    if mujoco_gl != "egl":
        return None if cli_value is None else int(cli_value)

    visible_token = _rank_local_visible_cuda_token()
    if visible_token is not None:
        if not visible_token.isdigit():
            raise ValueError(
                "robosuite EGL requires numeric CUDA_VISIBLE_DEVICES tokens; "
                f"got CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        visible_device_id = int(visible_token)
        visible_tokens = _cuda_visible_device_tokens()
        if cli_value is not None and str(int(cli_value)) in visible_tokens:
            device_id = int(cli_value)
        else:
            if cli_value is not None:
                print(
                    "[eval_libero_unified] WARN: --render-gpu-device-id=%s is not in "
                    "CUDA_VISIBLE_DEVICES=%s; using rank-local robosuite EGL device %s"
                    % (int(cli_value), os.environ.get("CUDA_VISIBLE_DEVICES", ""), visible_device_id)
                )
            device_id = visible_device_id
    elif cli_value is not None:
        device_id = int(cli_value)
    else:
        local_rank = _parse_int_env("LOCAL_RANK")
        if local_rank is None:
            local_rank = _parse_int_env("SLURM_LOCALID")
        env_device = _parse_int_env("MUJOCO_EGL_DEVICE_ID")
        device_id = int(local_rank if local_rank is not None else (env_device if env_device is not None else 0))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    return device_id


def main() -> None:
    args = build_arg_parser().parse_args()
    if int(args.execute_chunk_prefix) < 0:
        raise ValueError("--execute-chunk-prefix must be >= 0.")
    if int(args.execute_chunk_prefix) > 0 and args.partial_chunk_history == "default":
        args.partial_chunk_history = "rolling_last_k"
    raw_proprio_orientation = str(args.proprio_orientation or "auto").strip().lower().replace("-", "_")
    args.proprio_orientation = (
        "auto" if raw_proprio_orientation in {"", "auto"} else normalize_proprio_orientation_mode(raw_proprio_orientation)
    )
    args.text_prompt_normalization = normalize_text_prompt_mode(args.text_prompt_normalization)
    plus_root = args.plus_root if args.plus else None
    if args.plus and not plus_root:
        raise ValueError("LIBERO-Plus eval requires --plus-root or DA3_LIBERO_PLUS_DIR.")

    preset = PRESETS[args.preset]
    seed = preset.seed if args.seed is None else int(args.seed)
    env_seed = preset.env_seed if args.env_seed is None else int(args.env_seed)
    num_steps_wait = preset.num_steps_wait if args.num_steps_wait is None else int(args.num_steps_wait)
    if args.num_trials_per_task is not None:
        num_trials = int(args.num_trials_per_task)
        num_trials_source = "cli"
    elif args.plus:
        # LIBERO-Plus expands perturbations as separate tasks. One trial per
        # synthetic task matches the train-time Plus monitor profiles and keeps
        # full sweeps from silently multiplying thousands of episodes by the
        # plain-LIBERO preset's trial count.
        num_trials = 1
        num_trials_source = "libero_plus_default"
    else:
        num_trials = int(preset.num_trials_per_task)
        num_trials_source = "preset"
    suites = parse_csv_list(args.suites)
    task_ids = parse_int_list(args.task_ids)
    eval_shard = resolve_eval_shard(args.shard_index, args.shard_count)
    if not suites:
        raise ValueError("At least one suite is required.")

    set_global_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = _resolve_eval_cuda_device()
    resolved_render_gpu_device_id = _resolve_render_gpu_device_id(args.render_gpu_device_id)
    # Install the Mesa GLContext shim BEFORE any robosuite/MjRenderContext
    # construction when the native NVIDIA EGL device-display path is
    # unavailable. On CSCS nodes the native path must stay active so
    # `render_gpu_device_id` and `MUJOCO_EGL_DEVICE_ID` select the rank-local GPU.
    try:
        from robot.evaluation.closed_loop_libero_eval import _install_mujoco_glcontext_patch
        _install_mujoco_glcontext_patch(resolved_render_gpu_device_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval_libero_unified] WARN: GLContext patch skipped: {exc}")
    print(
        "[eval_libero_unified] host=%s rank=%s local_rank=%s CUDA_VISIBLE_DEVICES=%s "
        "device=%s render_gpu_device_id=%s MUJOCO_EGL_DEVICE_ID=%s glcontext=%s"
        % (
            os.uname().nodename,
            os.environ.get("RANK", os.environ.get("SLURM_PROCID", "<unset>")),
            os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "<unset>")),
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            str(device),
            "default" if resolved_render_gpu_device_id is None else str(int(resolved_render_gpu_device_id)),
            os.environ.get("MUJOCO_EGL_DEVICE_ID", "<unset>"),
            os.environ.get("DA3_LIBERO_GLCONTEXT_MODE", "default"),
        )
    )

    # Empty base when --config omitted; ckpt['config'] then provides everything.
    # Avoids the previous bug where the OpenX default leaked target_hz=5 into
    # LIBERO HDF5 evals.
    cfg = OmegaConf.create({}) if args.config is None else OmegaConf.load(args.config)
    if not args.ckpt:
        raise SystemExit("--ckpt is required.")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    policy, policy_info = load_policy(args, cfg, ckpt, device)
    rollout_action_frame = resolve_rollout_action_frame(args.action_frame, policy_info)
    policy_info["rollout_action_frame"] = rollout_action_frame
    action_horizon, action_horizon_requested = resolve_action_horizon(
        args.action_horizon,
        preset=preset,
        policy_info=policy_info,
    )
    # Publish H_exec to the policy closure so the AR rollout only runs
    # min(H_exec, N_action_steps) internal iterations (no wasted compute for
    # discarded chunks). Policy reads policy.active_action_horizon at forward.
    policy.active_action_horizon = int(action_horizon)
    config_source = str(policy_info.get("config_source", "cli"))
    effective_config = args.config if config_source == "cli" else f"{args.ckpt}:config"
    plus_classification = load_libero_plus_task_classification(plus_root) if args.plus else {
        "loaded": False,
        "path": None,
        "suite_task_counts": {},
        "category_task_counts": {},
    }

    run_name = args.run_name or default_eval_run_name(args.ckpt, args.preset, eval_shard)
    root_run_dir = Path(args.output_dir) / run_name
    run_dir = root_run_dir if not eval_shard.enabled else root_run_dir / "shards" / f"shard{eval_shard.index:04d}"
    rollout_dir = run_dir / "rollouts"
    run_dir.mkdir(parents=True, exist_ok=True)
    te_model_steps = int(action_horizon) if args.temporal_ensemble else None
    try:
        te_chunk_size = int(policy_info.get("action_chunk_size", 1) or 1)
    except Exception:
        te_chunk_size = 1
    te_forecast_horizon = (te_model_steps or 0) * te_chunk_size if args.temporal_ensemble else None

    wandb_run = None
    if args.wandb and eval_shard.enabled:
        print("[eval_libero_unified] WARN: wandb logging is disabled for sharded eval; aggregate files are written locally.")
    if args.wandb and not eval_shard.enabled:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or run_name,
            config={
                "eval_type": "libero_closed_loop_unified",
                "ckpt": args.ckpt,
                "config": effective_config,
                "cli_config": args.config,
                "config_source": config_source,
                "preset": args.preset,
                "seed": seed,
                "env_seed": env_seed,
                "num_trials_per_task": num_trials,
                "num_trials_per_task_source": num_trials_source,
                "num_steps_wait": num_steps_wait,
                "camera_size": int(args.camera_size),
                "render_gpu_device_id": resolved_render_gpu_device_id,
                "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                "glcontext_mode": os.environ.get("DA3_LIBERO_GLCONTEXT_MODE"),
                "env_process_isolation": bool(args.env_process_isolation),
                "env_worker_timeout_sec": float(args.env_worker_timeout_sec),
                "action_horizon": action_horizon,
                "action_horizon_requested": action_horizon_requested,
                "max_action_horizon": policy_info.get("max_action_horizon"),
                "rollout_decode_horizon": policy_info.get("stage1_rollout_decode_horizon"),
                "rollout_decode_horizon_requested": args.rollout_decode_horizon,
                "rollout_decode_horizon_mode": policy_info.get("stage1_rollout_decode_horizon_mode"),
                "temporal_ensemble": bool(args.temporal_ensemble),
                "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
                "temporal_ensemble_unit": "low_level_action" if args.temporal_ensemble else None,
                "temporal_ensemble_forecast_horizon": te_forecast_horizon,
                "temporal_ensemble_model_steps": te_model_steps,
                "temporal_ensemble_chunk_size": te_chunk_size if args.temporal_ensemble else None,
                "execution_strategy": args.execution_strategy,
                "execute_chunk_prefix": int(args.execute_chunk_prefix),
                "partial_chunk_history": args.partial_chunk_history,
                "warmup_full_chunk_once": bool(args.warmup_full_chunk_once),
                "full_plan_close_threshold": float(args.full_plan_close_threshold),
                "gripper_override": args.gripper_override,
                "near_object_close_threshold": float(args.near_object_close_threshold),
                "near_object_close_hold_steps": int(args.near_object_close_hold_steps),
                "basket_release_threshold": float(args.basket_release_threshold),
                "basket_release_hold_steps": int(args.basket_release_hold_steps),
                "action_repeat": args.action_repeat,
                "action_repeat_mode": args.action_repeat_mode,
                "history_horizon": args.history_horizon,
                "proprio_orientation": args.proprio_orientation,
                "action_frame": policy_info.get("action_frame"),
                "rollout_action_frame": rollout_action_frame,
                "policy_hz": args.policy_hz,
                "env_control_hz_override": args.env_control_hz,
                "detailed_video": args.detailed_video,
                "trace_actions": bool(args.trace_actions),
                "rollout_debug_log": bool(args.rollout_debug_log),
                "rollout_debug_heartbeat_steps": int(args.rollout_debug_heartbeat_steps),
                "num_ode_steps": args.num_ode_steps,
                "sampling_method": args.sampling_method,
                "split_inference_mode": args.split_inference_mode,
                "plus": bool(args.plus),
                "plus_root": plus_root,
                "plus_perturbation": args.plus_perturbation if args.plus else None,
                "plus_official_category": args.plus_official_category if args.plus else None,
                "plus_sample_group_by": args.plus_sample_group_by if args.plus else None,
                "plus_samples_per_group": int(args.plus_samples_per_group) if args.plus else 0,
                "plus_sample_seed": int(args.plus_sample_seed) if args.plus else 0,
                "plus_classification_loaded": bool(plus_classification.get("loaded", False)),
                "plus_classification_path": plus_classification.get("path"),
                "suites": suites,
                "task_ids": task_ids,
                "shard_index": eval_shard.index,
                "shard_count": eval_shard.count,
                **policy_info,
            },
        )

    per_task_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    suite_results: dict[str, dict[str, Any]] = {}
    action_timing_rows: list[dict[str, Any]] = []
    prompt_audit_rows: list[dict[str, Any]] = []
    plus_subset_manifests: dict[str, Any] = {}
    progress_log_path = run_dir / "progress.jsonl"
    progress_log_path.write_text("")
    total_start = time.time()

    eval_label = "Unified LIBERO-Plus eval" if args.plus else "Unified LIBERO eval"
    print(f"{eval_label}: ckpt={args.ckpt}")
    print(f"  stage={policy_info['stage']} step={policy_info.get('train_steps', 0)} device={device}")
    print(
        f"  preset={args.preset} seed={seed} env_seed={env_seed} "
        f"trials/task={num_trials} trials_source={num_trials_source} "
        f"camera_size={int(args.camera_size)}"
    )
    print(
        f"  render_gpu_device_id="
        f"{'default' if resolved_render_gpu_device_id is None else int(resolved_render_gpu_device_id)} "
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID', '<unset>')} "
        f"glcontext={os.environ.get('DA3_LIBERO_GLCONTEXT_MODE', 'default')}"
    )
    print(
        f"  env_process_isolation={bool(args.env_process_isolation)} "
        f"env_worker_timeout_sec={float(args.env_worker_timeout_sec):.1f}"
    )
    print(f"  config={effective_config} (source={config_source})")
    if eval_shard.enabled:
        print(
            f"  shard={eval_shard.index}/{eval_shard.count} "
            f"source={eval_shard.source} root_output={root_run_dir} worker_output={run_dir}"
        )
    if args.plus:
        print(
            f"  plus_root={plus_root} perturbation={args.plus_perturbation} "
            f"official_category={args.plus_official_category}"
        )
        if args.plus_sample_group_by != "none" and int(args.plus_samples_per_group) > 0:
            print(
                f"  plus subset: group_by={args.plus_sample_group_by} "
                f"samples_per_group={int(args.plus_samples_per_group)} seed={int(args.plus_sample_seed)}"
            )
        print(
            "  plus official classification: loaded=%s path=%s"
            % (bool(plus_classification.get("loaded", False)), plus_classification.get("path"))
        )
    # Terminology (applied codebase-wide):
    #   H_exec  = action_horizon           : AR chunks executed per policy call
    #                                        (= re-observe cadence in chunks)
    #   H_decode = rollout_decode_horizon  : AR slots generated before DA3
    #                                        deep refine/action decode
    #   N_action_steps                     : model's native prediction horizon
    #   K_chunk = chunk_size               : low-level actions per AR iter
    #   R_env   = action_repeat            : env sub-steps per low-level action
    #   T_call  = H_exec * K_chunk * R_env : env steps between re-observes
    #   H_hist  = history_horizon          : predictor input window length
    print(
        "  action timing: H_exec=%s %s (requested=%s, max=%s %ss, max_low_level=%s) "
        "H_decode=%s mode=%s R_env=%s mode=%s policy_hz=%s env_control_hz=%s H_hist=%s"
        % (
            action_horizon,
            policy_info.get("action_horizon_unit", "actions"),
            action_horizon_requested,
            policy_info.get("max_action_horizon"),
            policy_info.get("action_horizon_unit", "action"),
            policy_info.get("max_low_level_actions_per_call"),
            policy_info.get("stage1_rollout_decode_horizon", "n/a"),
            policy_info.get("stage1_rollout_decode_horizon_mode", "n/a"),
            args.action_repeat,
            args.action_repeat_mode,
            args.policy_hz or policy_info.get("dataset_target_hz"),
            args.env_control_hz,
            policy_info.get("stage1_history_horizon", "n/a"),
        )
    )
    if int(args.execute_chunk_prefix) > 0:
        print(
            "  subchunk replan: execute_chunk_prefix=%s partial_chunk_history=%s "
            "warmup_full_chunk_once=%s"
            % (
                int(args.execute_chunk_prefix),
                args.partial_chunk_history,
                bool(args.warmup_full_chunk_once),
            )
        )
    if args.temporal_ensemble:
        print(
            "  temporal ensemble: enabled unit=low_level_action forecast_horizon=%s actions "
            "policy_model_steps=%s decay=%s"
            % (
                te_forecast_horizon,
                te_model_steps,
                args.temporal_ensemble_decay,
            )
        )
    if args.execution_strategy != "default":
        print(
            "  execution strategy: %s close_threshold=%s"
            % (args.execution_strategy, float(args.full_plan_close_threshold))
        )
    if args.gripper_override != "default":
        print(
            "  gripper override: %s near_threshold=%.4f near_hold=%d basket_threshold=%.4f basket_hold=%d"
            % (
                args.gripper_override,
                float(args.near_object_close_threshold),
                int(args.near_object_close_hold_steps),
                float(args.basket_release_threshold),
                int(args.basket_release_hold_steps),
            )
        )
    print(
        "  preprocess=%s crop=%s env_hflip=%s stats_key=%s text_encoder=%s proprio_orientation=%s action_frame=%s rollout_action_frame=%s"
        % (
            policy_info.get("policy_image_preprocess"),
            policy_info.get("eval_crop_scale"),
            policy_info.get(
                "libero_hdf5_env_hflip",
                policy_info.get("libero_hdf5_env_rotate180", policy_info.get("libero_hdf5_env_hflip_fix")),
            ),
            policy_info.get("action_stats_key"),
            policy_info.get("text_encoder_type", "n/a"),
            policy_info.get("proprio_orientation", args.proprio_orientation),
            policy_info.get("action_frame", "base"),
            rollout_action_frame,
        )
    )
    print(f"  output={run_dir}")
    print(f"  progress_log={progress_log_path}")
    if args.trace_actions:
        print(f"  action traces={run_dir / 'action_traces'}")
    if args.rollout_debug_log:
        print(
            f"  rollout debug events={run_dir / 'debug_events'} "
            f"(heartbeat_steps={int(args.rollout_debug_heartbeat_steps)})"
        )

    global_episode_index = 0
    for suite in suites:
        if suite not in OPENVLA_STEPS:
            raise ValueError(f"Unknown LIBERO suite '{suite}'. Expected one of {sorted(OPENVLA_STEPS)}")

        max_steps = int(args.max_steps or preset.max_steps_by_suite[suite])
        task_metadata = [dict(item) for item in list_libero_task_metadata(suite, plus_root=plus_root)]
        if args.plus:
            for item in task_metadata:
                item["plus_perturbation"] = classify_libero_plus_perturbation(item)
            annotate_libero_plus_official_categories(task_metadata, suite, plus_classification)
            task_metadata = filter_libero_plus_task_metadata(task_metadata, args.plus_perturbation)
            task_metadata = filter_libero_plus_official_category_metadata(
                task_metadata,
                args.plus_official_category,
            )
            task_metadata, subset_manifest = select_libero_plus_task_subset(
                task_metadata,
                group_by=args.plus_sample_group_by,
                samples_per_group=int(args.plus_samples_per_group),
                sample_seed=int(args.plus_sample_seed),
                suite=suite,
            )
            if subset_manifest.get("enabled"):
                plus_subset_manifests[suite] = subset_manifest
        eval_entries = select_eval_task_entries(task_metadata, task_ids)
        suite_successes: list[bool] = []

        selected_display = [f"{entry['eval_task_index']}->{entry['task_id']}" for entry in eval_entries[:20]]
        if len(eval_entries) > 20:
            selected_display.append(f"...(+{len(eval_entries) - 20} more)")
        print(f"\nSuite {suite}: max_steps={max_steps}, tasks={selected_display}")
        if args.plus:
            categories = sorted({str(item.get("plus_perturbation")) for item in task_metadata})
            print(f"  LIBERO-Plus filtered tasks={len(task_metadata)} categories={categories}")
            official_categories = sorted(
                {
                    str(item.get("plus_official_category") or item.get("plus_official_category_slug") or "")
                    for item in task_metadata
                    if item.get("plus_official_category") or item.get("plus_official_category_slug")
                }
            )
            if official_categories:
                print(f"  LIBERO-Plus official categories={official_categories}")
            if suite in plus_subset_manifests:
                manifest = plus_subset_manifests[suite]
                print(
                    f"  LIBERO-Plus sampled tasks={manifest['total_selected']}/"
                    f"{manifest['total_available']} by {manifest['group_by']}"
                )
        for entry in eval_entries:
            eval_task_index = int(entry["eval_task_index"])
            task_id = int(entry["task_id"])
            shard_episode_indices = assigned_episode_indices(global_episode_index, num_trials, eval_shard)
            global_episode_index += num_trials
            if not shard_episode_indices:
                if eval_shard.enabled:
                    print(
                        f"  Task {task_id}: skipped on shard {eval_shard.index}/{eval_shard.count} "
                        f"(eval_index={eval_task_index})"
                    )
                continue
            # Our DA3 policies use the BASE BDDL's parsed language
            # (policy_language) which strips Plus's per-variant filename markers.
            task_desc = str(entry.get("policy_language") or entry["language"])
            plus_perturbation = str(entry.get("plus_perturbation", ""))
            plus_official_task_id = entry.get("plus_official_task_id", "")
            plus_official_category = str(entry.get("plus_official_category", "") or "")
            plus_official_category_slug = str(entry.get("plus_official_category_slug", "") or "")
            plus_official_difficulty_level = entry.get("plus_official_difficulty_level", "")
            env_kwargs = {
                "suite_name": suite,
                "task_id": task_id,
                "camera_names": LIBERO_CAMERA_NAMES,
                "camera_size": args.camera_size,
                "render_gpu_device_id": resolved_render_gpu_device_id,
                "control_freq": args.env_control_hz,
                "horizon": max_steps + max(0, int(num_steps_wait)),
                "plus_root": plus_root,
                "preserve_libero_plus_robot_init_qpos": args.libero_plus_robot_init_qpos_mode == "preserve",
                # Fixed False to match in-training closed_loop_libero_eval. Env-rendered
                # depth via MUJOCO EGL leaks GL context across env.step() calls and SIGABRTs
                # in robosuite/binding_utils.py:read_pixels after 1-2 trials on Daint
                # GH200 nodes. Detailed_video uses model-predicted depth (policy.last_debug),
                # so video content stays model-predicted.
                "camera_depths": False,
                "env_image_hflip": bool(
                    policy_info.get(
                        "libero_hdf5_env_hflip",
                        policy_info.get("libero_hdf5_env_rotate180", policy_info.get("libero_hdf5_env_hflip_fix", False)),
                    )
                ),
                "env_image_rotate180": bool(
                    policy_info.get(
                        "libero_hdf5_env_rotate180",
                        policy_info.get("libero_hdf5_env_hflip_fix", False),
                    )
                ),
            }
            if args.env_process_isolation:
                env, task_name, init_states = create_rollout_env_libero_isolated(
                    **env_kwargs,
                    worker_timeout_sec=float(args.env_worker_timeout_sec),
                    worker_rank=eval_shard.index if eval_shard.enabled else _parse_int_env("SLURM_PROCID"),
                )
            else:
                env, task_name, init_states = create_rollout_env_libero(**env_kwargs)
            action_repeat, policy_hz, env_control_hz = resolve_action_repeat(
                args.action_repeat,
                policy_info=policy_info,
                env=env,
                policy_hz_override=args.policy_hz,
            )
            if task_desc != task_name:
                print(
                    f"    [prompt-warning] policy prompt and env task differ: "
                    f"policy={task_desc!r} env={task_name!r}"
                )
            prompt_audit: dict[str, Any] = {
                "suite": suite,
                "task_id": task_id,
                "eval_task_index": eval_task_index,
                "prompt": task_desc,
                "prompt_sha1": hashlib.sha1(task_desc.encode("utf-8")).hexdigest()[:12],
                "source": "policy_language" if task_desc != task_name else "libero_benchmark_task.language",
                "env_task_name": task_name,
                "plus_perturbation": plus_perturbation,
                "plus_official_task_id": plus_official_task_id,
                "plus_official_category": plus_official_category,
                "plus_official_category_slug": plus_official_category_slug,
                "plus_official_difficulty_level": plus_official_difficulty_level,
            }
            if hasattr(policy, "describe_text_prompt"):
                model_prompt_audit = policy.describe_text_prompt(task_desc)  # type: ignore[attr-defined]
                prompt_audit.update(model_prompt_audit)
                prompt_audit["source"] = (
                    f"{prompt_audit['source']} -> "
                    f"{str(model_prompt_audit.get('text_encoder_type', 'text')).upper()}"
                )
            prompt_audit_rows.append(prompt_audit)
            action_timing_rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "eval_task_index": eval_task_index,
                    "plus_perturbation": plus_perturbation,
                    "plus_official_task_id": plus_official_task_id,
                    "plus_official_category": plus_official_category,
                    "plus_official_category_slug": plus_official_category_slug,
                    "plus_official_difficulty_level": plus_official_difficulty_level,
                    "action_repeat": action_repeat,
                    "action_horizon": action_horizon,
                    "action_horizon_requested": action_horizon_requested,
                    "rollout_decode_horizon": policy_info.get("stage1_rollout_decode_horizon"),
                    "rollout_decode_horizon_requested": args.rollout_decode_horizon,
                    "rollout_decode_horizon_mode": policy_info.get("stage1_rollout_decode_horizon_mode"),
                    "temporal_ensemble": bool(args.temporal_ensemble),
                    "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
                    "temporal_ensemble_unit": "low_level_action" if args.temporal_ensemble else None,
                    "temporal_ensemble_forecast_horizon": te_forecast_horizon,
                    "temporal_ensemble_model_steps": te_model_steps,
                    "temporal_ensemble_chunk_size": te_chunk_size if args.temporal_ensemble else None,
                    "temporal_ensemble_execute_horizon": action_horizon,
                    "execution_strategy": args.execution_strategy,
                    "execute_chunk_prefix": int(args.execute_chunk_prefix),
                    "partial_chunk_history": args.partial_chunk_history,
                    "warmup_full_chunk_once": bool(args.warmup_full_chunk_once),
                    "full_plan_close_threshold": float(args.full_plan_close_threshold),
                    "gripper_override": args.gripper_override,
                    "near_object_close_threshold": float(args.near_object_close_threshold),
                    "near_object_close_hold_steps": int(args.near_object_close_hold_steps),
                    "basket_release_threshold": float(args.basket_release_threshold),
                    "basket_release_hold_steps": int(args.basket_release_hold_steps),
                    "policy_hz": policy_hz,
                    "env_control_hz": env_control_hz,
                    "action_repeat_mode": args.action_repeat_mode,
                    "env_process_isolation": bool(args.env_process_isolation),
                    "env_worker_timeout_sec": float(args.env_worker_timeout_sec),
                    "history_horizon": policy_info.get("stage1_history_horizon"),
                    "history_commit_stride_actions": policy_info.get("history_commit_stride_actions"),
                    "eval_crop_scale": policy_info.get("eval_crop_scale"),
                    "dataset_da3_input_rotate180": policy_info.get("dataset_da3_input_rotate180"),
                    "dataset_da3_input_hflip": policy_info.get("dataset_da3_input_hflip"),
                    "da3_input_vflip": policy_info.get("da3_input_vflip"),
                    "libero_hdf5_env_hflip": policy_info.get("libero_hdf5_env_hflip"),
                    "libero_hdf5_env_rotate180": policy_info.get("libero_hdf5_env_rotate180"),
                    "policy_image_preprocess": policy_info.get("policy_image_preprocess"),
                    "text_prompt_normalization": policy_info.get("text_prompt_normalization"),
                    "text_encoder_type": policy_info.get("text_encoder_type"),
                    "action_frame": policy_info.get("action_frame"),
                    "rollout_action_frame": rollout_action_frame,
                }
            )
            seed_env(env, env_seed)
            task_successes: list[bool] = []
            step_counts: list[int] = []

            plus_suffix = (
                f", plus={plus_perturbation}, official_category={plus_official_category or 'n/a'}, "
                f"official_id={plus_official_task_id or 'n/a'}, eval_index={eval_task_index}"
                if args.plus
                else ""
            )
            print(
                f"  Task {task_id}: {task_desc} ({len(init_states)} init states{plus_suffix}, "
                f"action_repeat={action_repeat}, policy_hz={policy_hz}, env_hz={env_control_hz})"
            )
            if eval_shard.enabled:
                print(f"    shard episodes: {shard_episode_indices}")
            print(
                "    text prompt audit: source=%s sha1=%s tokens=%s text_norm=%s prompt=%r"
                % (
                    prompt_audit.get("source"),
                    prompt_audit.get("prompt_sha1"),
                    prompt_audit.get("token_count", "n/a"),
                    prompt_audit.get("text_norm", "n/a"),
                    task_desc,
                )
            )
            try:
                for episode_idx in shard_episode_indices:
                    init_idx = episode_idx % len(init_states)
                    t0 = time.time()
                    # --video-every: opt-in sampled debug capture. Records the
                    # MP4 rollout and the action trace JSONL for every Nth
                    # global task index so we can verify the baseline is acting
                    # on real obs. Use eval_task_index (suite-level task
                    # counter) rather than episode_idx, which is always 0
                    # when --num-trials-per-task=1 (single trial per Plus task).
                    video_every = int(getattr(args, "video_every", 0) or 0)
                    is_video_episode = video_every > 0 and (int(eval_task_index) % video_every == 0)
                    record_video_this_ep = bool(args.record_video) or is_video_episode
                    trace_actions_this_ep = bool(args.trace_actions) or is_video_episode
                    trace_path = None
                    if trace_actions_this_ep:
                        trace_path = run_dir / "action_traces" / suite / f"task{task_id}_ep{episode_idx}.jsonl"
                    obs_dump_dir = None
                    if is_video_episode:
                        obs_dump_dir = run_dir / "obs_dumps" / suite / f"task{task_id}_ep{episode_idx}"
                        obs_dump_dir.mkdir(parents=True, exist_ok=True)
                    rollout_debug_path = None
                    rollout_debug_context = None
                    if args.rollout_debug_log:
                        rollout_debug_path = run_dir / "debug_events" / suite / f"task{task_id}_ep{episode_idx}.jsonl"
                        rollout_debug_context = {
                            "run_name": run_name,
                            "suite": suite,
                            "task_id": int(task_id),
                            "eval_task_index": int(eval_task_index),
                            "episode_idx": int(episode_idx),
                            "init_idx": int(init_idx),
                            "task_name": task_desc,
                            "env_task_name": task_name,
                            "plus_perturbation": plus_perturbation,
                            "plus_official_task_id": plus_official_task_id,
                            "plus_official_category": plus_official_category,
                            "plus_official_category_slug": plus_official_category_slug,
                            "ckpt": args.ckpt,
                            "train_step": int(policy_info.get("train_steps", 0)),
                            "history_horizon": policy_info.get("stage1_history_horizon"),
                            "action_horizon": int(action_horizon),
                            "rollout_decode_horizon": policy_info.get("stage1_rollout_decode_horizon"),
                            "action_chunk_size": policy_info.get("action_chunk_size"),
                            "action_frame": policy_info.get("action_frame"),
                            "rollout_action_frame": rollout_action_frame,
                            "execution_strategy": args.execution_strategy,
                            "execute_chunk_prefix": int(args.execute_chunk_prefix),
                            "partial_chunk_history": args.partial_chunk_history,
                            "warmup_full_chunk_once": bool(args.warmup_full_chunk_once),
                            "gripper_override": args.gripper_override,
                            "env_process_isolation": bool(args.env_process_isolation),
                        }
                    try:
                        result = rollout_episode(
                            env=env,
                            init_state=init_states[init_idx],
                            policy=policy,
                            max_steps=max_steps,
                            action_horizon=action_horizon,
                            action_repeat=action_repeat,
                            action_repeat_mode=args.action_repeat_mode,
                            num_steps_wait=num_steps_wait,
                            camera_size=args.camera_size,
                            record_video=record_video_this_ep,
                            detailed_video=args.detailed_video,
                            binarize_gripper=not args.no_binarize_gripper,
                            task_desc=task_desc,
                            action_frame=rollout_action_frame,
                            proprio_orientation=policy_info.get("proprio_orientation", args.proprio_orientation),
                            temporal_ensemble=bool(args.temporal_ensemble),
                            temporal_ensemble_decay=float(args.temporal_ensemble_decay),
                            execution_strategy=args.execution_strategy,
                            full_plan_close_threshold=float(args.full_plan_close_threshold),
                            gripper_override=args.gripper_override,
                            near_object_close_threshold=float(args.near_object_close_threshold),
                            near_object_close_hold_steps=int(args.near_object_close_hold_steps),
                            basket_release_threshold=float(args.basket_release_threshold),
                            basket_release_hold_steps=int(args.basket_release_hold_steps),
                            execute_chunk_prefix=int(args.execute_chunk_prefix),
                            partial_chunk_history=args.partial_chunk_history,
                            warmup_full_chunk_once=bool(args.warmup_full_chunk_once),
                            action_trace_path=trace_path,
                            action_trace_context={
                                "run_name": run_name,
                                "suite": suite,
                                "task_id": int(task_id),
                                "eval_task_index": int(eval_task_index),
                                "episode_idx": int(episode_idx),
                                "init_idx": int(init_idx),
                                "task_name": task_desc,
                                "env_task_name": task_name,
                                "plus_perturbation": plus_perturbation,
                                "plus_official_task_id": plus_official_task_id,
                                "plus_official_category": plus_official_category,
                                "plus_official_category_slug": plus_official_category_slug,
                                "ckpt": args.ckpt,
                                "train_step": int(policy_info.get("train_steps", 0)),
                                "history_horizon": policy_info.get("stage1_history_horizon"),
                                "action_horizon": int(action_horizon),
                                "rollout_decode_horizon": policy_info.get("stage1_rollout_decode_horizon"),
                                "action_chunk_size": policy_info.get("action_chunk_size"),
                                "binarize_gripper": not args.no_binarize_gripper,
                                "action_frame": policy_info.get("action_frame"),
                                "rollout_action_frame": rollout_action_frame,
                                "execution_strategy": args.execution_strategy,
                                "execute_chunk_prefix": int(args.execute_chunk_prefix),
                                "partial_chunk_history": args.partial_chunk_history,
                                "warmup_full_chunk_once": bool(args.warmup_full_chunk_once),
                                "gripper_override": args.gripper_override,
                                "env_process_isolation": bool(args.env_process_isolation),
                                "video_every": int(video_every),
                                "is_video_episode": bool(is_video_episode),
                            } if trace_actions_this_ep else None,
                            rollout_debug_log_path=rollout_debug_path,
                            rollout_debug_context=rollout_debug_context,
                            rollout_debug_heartbeat_steps=int(args.rollout_debug_heartbeat_steps),
                        )
                    except Exception as exc:
                        elapsed = time.time() - t0
                        append_rollout_debug_event(
                            rollout_debug_path,
                            rollout_debug_context,
                            {
                                "event": "episode_exception",
                                "elapsed_sec": round(elapsed, 4),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        if _is_fatal_rollout_exception(exc, args):
                            print(
                                f"    ep {episode_idx:03d} init {init_idx:03d}: FATAL "
                                f"steps={max_steps} time={elapsed:.1f}s error={exc}"
                            )
                            raise
                        task_successes.append(False)
                        step_counts.append(int(max_steps))
                        progress_rows.append(
                            make_episode_progress_row(
                                suite=suite,
                                task_id=task_id,
                                eval_task_index=eval_task_index,
                                task_name=task_name,
                                task_desc=task_desc,
                                entry=entry,
                                plus_perturbation=plus_perturbation,
                                plus_official_task_id=plus_official_task_id,
                                plus_official_category=plus_official_category,
                                plus_official_category_slug=plus_official_category_slug,
                                plus_official_difficulty_level=plus_official_difficulty_level,
                                success=False,
                                steps=int(max_steps),
                            )
                        )
                        print(
                            f"    ep {episode_idx:03d} init {init_idx:03d}: ERROR "
                            f"steps={max_steps} time={elapsed:.1f}s error={exc}"
                        )
                        emit_rollout_progress(
                            progress_rows=progress_rows,
                            suites=suites,
                            plus=bool(args.plus),
                            run_name=run_name,
                            eval_shard=eval_shard,
                            progress_log_path=progress_log_path,
                            suite=suite,
                            task_id=task_id,
                            eval_task_index=eval_task_index,
                            episode_idx=episode_idx,
                        )
                        if wandb_run:
                            suite_offset = LIBERO_SUITE_ORDER.get(suite, 0) * 100000
                            wb_step = suite_offset + task_id * 1000 + episode_idx
                            wandb_run.log(
                                {
                                    f"libero/{suite}/episode_success": 0.0,
                                    f"libero/{suite}/episode_steps": int(max_steps),
                                    "libero/episode_success": 0.0,
                                },
                                step=wb_step,
                            )
                        continue

                    elapsed = time.time() - t0
                    task_successes.append(bool(result["success"]))
                    step_counts.append(int(result["steps"]))
                    progress_rows.append(
                        make_episode_progress_row(
                            suite=suite,
                            task_id=task_id,
                            eval_task_index=eval_task_index,
                            task_name=task_name,
                            task_desc=task_desc,
                            entry=entry,
                            plus_perturbation=plus_perturbation,
                            plus_official_task_id=plus_official_task_id,
                            plus_official_category=plus_official_category,
                            plus_official_category_slug=plus_official_category_slug,
                            plus_official_difficulty_level=plus_official_difficulty_level,
                            success=bool(result["success"]),
                            steps=int(result["steps"]),
                        )
                    )

                    if (record_video_this_ep or args.detailed_video) and result["frames"]:
                        video_path = rollout_dir / suite / f"task{task_id}_ep{episode_idx}.mp4"
                        save_video(result["frames"], video_path, fps=args.video_fps)
                    if obs_dump_dir is not None and result.get("frames"):
                        # Dump the raw first/last rollout frames as PNG so the
                        # baseline's actual input view can be eyeballed without
                        # MP4 decoding. PNGs are friendly for thumbnails / git diffs.
                        try:
                            from PIL import Image as _PILImage

                            for idx_label, frame_idx in (("first", 0), ("last", len(result["frames"]) - 1)):
                                if not (0 <= frame_idx < len(result["frames"])):
                                    continue
                                frame = result["frames"][frame_idx]
                                arr = np.asarray(frame)
                                if arr.ndim == 3 and arr.shape[-1] == 3:
                                    _PILImage.fromarray(arr.astype(np.uint8)).save(
                                        obs_dump_dir / f"{idx_label}_frame.png"
                                    )
                            summary_path = obs_dump_dir / "summary.json"
                            summary_path.write_text(
                                json.dumps(
                                    {
                                        "suite": suite,
                                        "task_id": int(task_id),
                                        "episode_idx": int(episode_idx),
                                        "task_desc": task_desc,
                                        "env_task_name": task_name,
                                        "success": bool(result["success"]),
                                        "steps": int(result["steps"]),
                                        "plus_perturbation": plus_perturbation,
                                        "plus_official_category": plus_official_category,
                                        "video_path": str(rollout_dir / suite / f"task{task_id}_ep{episode_idx}.mp4"),
                                        "trace_path": str(trace_path) if trace_path else None,
                                        "n_frames": len(result["frames"]),
                                    },
                                    indent=2,
                                )
                            )
                        except Exception as _dump_exc:  # noqa: BLE001
                            print(f"    [obs-dump WARN] failed to write obs dumps: {_dump_exc}")

                    status = "SUCCESS" if result["success"] else "FAIL"
                    print(f"    ep {episode_idx:03d} init {init_idx:03d}: {status} steps={result['steps']} time={elapsed:.1f}s")
                    emit_rollout_progress(
                        progress_rows=progress_rows,
                        suites=suites,
                        plus=bool(args.plus),
                        run_name=run_name,
                        eval_shard=eval_shard,
                        progress_log_path=progress_log_path,
                        suite=suite,
                        task_id=task_id,
                        eval_task_index=eval_task_index,
                        episode_idx=episode_idx,
                    )

                    if wandb_run:
                        suite_offset = LIBERO_SUITE_ORDER.get(suite, 0) * 100000
                        wb_step = suite_offset + task_id * 1000 + episode_idx
                        wandb_run.log(
                            {
                                f"libero/{suite}/episode_success": float(result["success"]),
                                f"libero/{suite}/episode_steps": int(result["steps"]),
                                "libero/episode_success": float(result["success"]),
                            },
                            step=wb_step,
                        )
            finally:
                env.close()

            success_rate = float(np.mean(task_successes)) if task_successes else 0.0
            suite_successes.extend(task_successes)
            row = {
                "suite": suite,
                "task_id": task_id,
                "eval_task_index": eval_task_index,
                "task_name": task_name,
                "plus_perturbation": plus_perturbation,
                "plus_official_task_id": plus_official_task_id,
                "plus_official_category": plus_official_category,
                "plus_official_category_slug": plus_official_category_slug,
                "plus_official_difficulty_level": plus_official_difficulty_level,
                "raw_task_language": str(entry.get("language", "")),
                "policy_language": task_desc,
                "bddl_file": str(entry.get("bddl_file", "")),
                "init_states_file": str(entry.get("init_states_file", "")),
                "num_trials": len(task_successes),
                "num_success": int(sum(task_successes)),
                "success_rate": success_rate,
                "avg_steps": float(np.mean(step_counts)) if step_counts else 0.0,
            }
            per_task_rows.append(row)
            print(f"    task success: {success_rate:.1%} ({row['num_success']}/{row['num_trials']})")

            if wandb_run:
                wandb_run.log({f"libero/{suite}/task_{task_id}_success_rate": success_rate})

        suite_rate = float(np.mean(suite_successes)) if suite_successes else 0.0
        suite_results[suite] = {
            "success_rate": suite_rate,
            "num_trials": len(suite_successes),
            "num_success": int(sum(suite_successes)),
        }
        print(f"Suite {suite}: {suite_rate:.1%} ({suite_results[suite]['num_success']}/{len(suite_successes)})")

    if args.plus:
        plus_official_category_results_by_suite = {
            suite: aggregate_plus_category_results([row for row in per_task_rows if row.get("suite") == suite])
            for suite in suites
        }
        for suite, category_results in plus_official_category_results_by_suite.items():
            if suite in suite_results:
                suite_results[suite]["plus_official_category_results"] = category_results
        plus_official_category_results = aggregate_plus_category_results(per_task_rows)
    else:
        plus_official_category_results_by_suite = {}
        plus_official_category_results = {}

    total_successes = int(sum(int(item["num_success"]) for item in suite_results.values()))
    total_episodes = int(sum(int(item["num_trials"]) for item in suite_results.values()))
    overall_success = float(total_successes / total_episodes) if total_episodes else 0.0
    average_success = float(np.mean([item["success_rate"] for item in suite_results.values()])) if suite_results else 0.0
    summary = {
        "run_name": run_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": args.ckpt,
        "config": effective_config,
        "cli_config": args.config,
        "config_source": config_source,
        "preset": args.preset,
        "seed": seed,
        "env_seed": env_seed,
        "num_trials_per_task": num_trials,
        "num_trials_per_task_source": num_trials_source,
        "num_steps_wait": num_steps_wait,
        "camera_size": int(args.camera_size),
        "render_gpu_device_id": resolved_render_gpu_device_id,
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "glcontext_mode": os.environ.get("DA3_LIBERO_GLCONTEXT_MODE"),
        "env_process_isolation": bool(args.env_process_isolation),
        "env_worker_timeout_sec": float(args.env_worker_timeout_sec),
        "action_horizon": action_horizon,
        "action_horizon_requested": action_horizon_requested,
        "max_action_horizon": policy_info.get("max_action_horizon"),
        "rollout_decode_horizon": policy_info.get("stage1_rollout_decode_horizon"),
        "rollout_decode_horizon_requested": args.rollout_decode_horizon,
        "rollout_decode_horizon_mode": policy_info.get("stage1_rollout_decode_horizon_mode"),
        "temporal_ensemble": bool(args.temporal_ensemble),
        "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
        "temporal_ensemble_unit": "low_level_action" if args.temporal_ensemble else None,
        "temporal_ensemble_forecast_horizon": te_forecast_horizon,
        "temporal_ensemble_model_steps": te_model_steps,
        "temporal_ensemble_chunk_size": te_chunk_size if args.temporal_ensemble else None,
        "temporal_ensemble_execute_horizon": action_horizon,
        "execution_strategy": args.execution_strategy,
        "execute_chunk_prefix": int(args.execute_chunk_prefix),
        "partial_chunk_history": args.partial_chunk_history,
        "warmup_full_chunk_once": bool(args.warmup_full_chunk_once),
        "full_plan_close_threshold": float(args.full_plan_close_threshold),
        "gripper_override": args.gripper_override,
        "near_object_close_threshold": float(args.near_object_close_threshold),
        "near_object_close_hold_steps": int(args.near_object_close_hold_steps),
        "basket_release_threshold": float(args.basket_release_threshold),
        "basket_release_hold_steps": int(args.basket_release_hold_steps),
        "action_repeat_requested": args.action_repeat,
        "action_repeat_mode": args.action_repeat_mode,
        "action_frame": policy_info.get("action_frame"),
        "rollout_action_frame": rollout_action_frame,
        "history_horizon_requested": args.history_horizon,
        "env_control_hz_override": args.env_control_hz,
        "action_timing": action_timing_rows,
        "prompt_audit": prompt_audit_rows,
        "detailed_video": args.detailed_video,
        "trace_actions": bool(args.trace_actions),
        "rollout_debug_log": bool(args.rollout_debug_log),
        "rollout_debug_heartbeat_steps": int(args.rollout_debug_heartbeat_steps),
        "num_ode_steps": args.num_ode_steps,
        "sampling_method": args.sampling_method,
        "split_inference_mode": args.split_inference_mode,
        "plus": bool(args.plus),
        "plus_root": plus_root,
        "plus_perturbation": args.plus_perturbation if args.plus else None,
        "plus_official_category": args.plus_official_category if args.plus else None,
        "libero_plus_robot_init_qpos_mode": args.libero_plus_robot_init_qpos_mode if args.plus else None,
        "plus_sample_group_by": args.plus_sample_group_by if args.plus else None,
        "plus_samples_per_group": int(args.plus_samples_per_group) if args.plus else 0,
        "plus_sample_seed": int(args.plus_sample_seed) if args.plus else 0,
        "plus_subset_manifests": plus_subset_manifests if args.plus else {},
        "plus_official_classification": {
            "loaded": bool(plus_classification.get("loaded", False)),
            "path": plus_classification.get("path"),
            "suite_task_counts": plus_classification.get("suite_task_counts", {}),
            "category_task_counts": plus_classification.get("category_task_counts", {}),
        } if args.plus else None,
        "plus_official_category_results": plus_official_category_results,
        "plus_official_category_results_by_suite": plus_official_category_results_by_suite,
        "suites": suites,
        "suite_results": suite_results,
        "average_success_rate": average_success,
        "overall_success_rate": overall_success,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "elapsed_sec": time.time() - total_start,
        "policy": policy_info,
        "shard": {
            "enabled": bool(eval_shard.enabled),
            "index": int(eval_shard.index),
            "count": int(eval_shard.count),
            "source": eval_shard.source,
            "aggregated": False,
            "root_run_dir": str(root_run_dir),
            "worker_run_dir": str(run_dir),
        },
        "_aggregate_start": total_start,
    }

    summary_to_write = dict(summary)
    summary_to_write.pop("_aggregate_start", None)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary_to_write, f, indent=2)
    write_per_task_csv(run_dir / "per_task.csv", per_task_rows)

    final_summary = summary_to_write
    final_per_task_rows = per_task_rows
    eval_id = None
    if eval_shard.enabled:
        (run_dir / "DONE").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        if eval_shard.index == 0:
            final_summary, final_per_task_rows = aggregate_shard_outputs(
                root_run_dir=root_run_dir,
                shard_count=eval_shard.count,
                timeout_sec=float(args.shard_wait_timeout_sec),
                base_summary=summary,
                suites=suites,
                plus=bool(args.plus),
            )
            with (root_run_dir / "summary.json").open("w") as f:
                json.dump(final_summary, f, indent=2)
            write_per_task_csv(root_run_dir / "per_task.csv", final_per_task_rows)
            if not args.no_registry:
                eval_id = append_eval_registry(args, final_summary, policy_info)
    elif not args.no_registry:
        eval_id = append_eval_registry(args, final_summary, policy_info)

    if wandb_run:
        final_suite_results = final_summary["suite_results"]
        final_plus_category_results = final_summary.get("plus_official_category_results", {})
        final_plus_category_results_by_suite = final_summary.get("plus_official_category_results_by_suite", {})
        flat = {
            "libero/average_success_rate": float(final_summary["average_success_rate"]),
            "libero/overall_success_rate": float(final_summary["overall_success_rate"]),
            "libero/total_successes": int(final_summary["total_successes"]),
            "libero/total_episodes": int(final_summary["total_episodes"]),
        }
        for suite, result in final_suite_results.items():
            flat[f"libero/{suite}/success_rate"] = result["success_rate"]
        if args.plus:
            flat["libero_plus/official_success_rate"] = float(final_summary["overall_success_rate"])
            for slug, result in final_plus_category_results.items():
                flat[f"libero_plus/category/{slug}/success_rate"] = result["success_rate"]
                flat[f"libero_plus/category/{slug}/num_trials"] = result["num_trials"]
            for suite, category_results in final_plus_category_results_by_suite.items():
                for slug, result in category_results.items():
                    flat[f"libero_plus/{suite}/category/{slug}/success_rate"] = result["success_rate"]
        wandb_run.log(flat)
        wandb_run.finish()

    final_suite_results = final_summary["suite_results"]
    final_plus_category_results = final_summary.get("plus_official_category_results", {}) if args.plus else {}
    final_total_successes = int(final_summary["total_successes"])
    final_total_episodes = int(final_summary["total_episodes"])
    final_overall_success = float(final_summary["overall_success_rate"])
    final_average_success = float(final_summary["average_success_rate"])
    final_run_dir = root_run_dir if bool(final_summary.get("shard", {}).get("aggregated", False)) else run_dir

    print("\nLIBERO summary")
    for suite, result in final_suite_results.items():
        print(f"  {suite}: {result['success_rate']:.1%} ({result['num_success']}/{result['num_trials']})")
    print(f"  average: {final_average_success:.1%}")
    print(f"  overall: {final_overall_success:.1%} ({final_total_successes}/{final_total_episodes})")
    if args.plus and final_plus_category_results:
        print("  official Plus categories:")
        for slug, result in final_plus_category_results.items():
            print(
                "    %s: %.1f%% (%d/%d, tasks=%d)"
                % (
                    result["category"],
                    100.0 * float(result["success_rate"]),
                    int(result["num_success"]),
                    int(result["num_trials"]),
                    int(result["num_tasks"]),
                )
            )
    print(f"  summary: {final_run_dir / 'summary.json'}")
    print(f"  per-task: {final_run_dir / 'per_task.csv'}")
    if eval_id:
        print(f"  eval_registry: {eval_id}")
    final_log = {
        "run_name": run_name,
        "ckpt": args.ckpt,
        "suites": suites,
        "plus": bool(args.plus),
        "plus_perturbation": args.plus_perturbation if args.plus else None,
        "total_successes": final_total_successes,
        "total_episodes": final_total_episodes,
        "overall_success_rate": final_overall_success,
        "average_success_rate": final_average_success,
        "suite_results": final_suite_results,
        "plus_official_category_results": final_plus_category_results,
        "summary_path": str(final_run_dir / "summary.json"),
        "per_task_path": str(final_run_dir / "per_task.csv"),
    }
    print("FINAL_LIBERO_RESULT_JSON=" + json.dumps(final_log, sort_keys=True))


if __name__ == "__main__":
    main()
