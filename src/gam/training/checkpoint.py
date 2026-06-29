"""Checkpoint loading, stats resolution, and action-timing validation helpers.

Extracted from ``train_robot.py``. Self-contained: depends only on
stdlib/torch.
"""

from __future__ import annotations

import os

import torch


def load_state_dict_forgiving(
    module,
    state_dict,
    logger=None,
    module_name="module",
    allow_action_head_chunk_resize: bool = False,
):
    """Load checkpoint weights while skipping missing or shape-mismatched keys."""
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    current = module.state_dict()
    filtered = {}
    skipped = []
    partial = []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append((key, "missing"))
            continue
        if current[key].shape != value.shape:
            can_partial_load = (
                allow_action_head_chunk_resize
                and key in {"output.weight", "output.bias", "chunk_pos_embed"}
                and value.ndim == current[key].ndim
                and all(value.shape[i] == current[key].shape[i] for i in range(1, value.ndim))
            )
            if can_partial_load:
                merged = current[key].detach().clone()
                rows = min(int(value.shape[0]), int(current[key].shape[0]))
                merged[:rows].copy_(value[:rows].to(dtype=merged.dtype))
                filtered[key] = merged
                partial.append((key, f"shape {tuple(value.shape)} -> {tuple(current[key].shape)}, rows={rows}"))
                continue
            skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(current[key].shape)}"))
            continue
        filtered[key] = value
    missing_keys, unexpected_keys = module.load_state_dict(filtered, strict=False)
    if logger is not None and partial:
        logger.info(
            "Forgiving load for %s partially loaded resized tensors: %s",
            module_name,
            ", ".join(f"{k} ({reason})" for k, reason in partial),
        )
    if logger is not None and skipped:
        logger.info(
            "Forgiving load for %s skipped: %s",
            module_name,
            ", ".join(f"{k} ({reason})" for k, reason in skipped),
        )
    if logger is not None and (missing_keys or unexpected_keys):
        logger.info(
            "Forgiving load for %s missing=%s unexpected=%s",
            module_name,
            list(missing_keys),
            list(unexpected_keys),
        )
    return {
        "skipped": skipped,
        "partial": partial,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def resolve_stats_dir(dataset_cfg):
    stats_dir = dataset_cfg.get("stats_dir") or dataset_cfg.get("action_stats_dir")
    if stats_dir is not None:
        return stats_dir
    if dataset_cfg.get("openx_root"):
        return os.path.join(dataset_cfg["openx_root"], "_stats")
    if dataset_cfg.get("mimicgen_root"):
        return os.path.join(dataset_cfg["mimicgen_root"], "_stats")
    return None


def state_normalizer_dim_mismatches(normalizer, dataset_names, expected_dim: int) -> list[str]:
    mismatches: list[str] = []
    for name in dataset_names:
        stats = normalizer.stats_by_key.get(name)
        if stats is None:
            continue
        q01_dim = int(torch.as_tensor(stats["q01"]).shape[-1])
        mask_dim = int(torch.as_tensor(stats["mask"]).shape[-1])
        if q01_dim != expected_dim or mask_dim != expected_dim:
            mismatches.append(f"{name}: q01={q01_dim}, mask={mask_dim}, expected={expected_dim}")
    return mismatches


def _flatten_leaf_datasets(ds) -> list:
    """Return leaf datasets with dataset_name under mixers / weighted OpenX mixes."""
    if not hasattr(ds, "datasets"):
        return [ds]
    leaves: list = []
    for child in ds.datasets.values():
        leaves.extend(_flatten_leaf_datasets(child))
    return leaves


def _action_timing_signature_for_dataset(ds) -> tuple[str, int]:
    max_stride = int(getattr(ds, "max_stride", getattr(ds, "temporal_stride", 1)) or 1)
    if bool(getattr(ds, "random_stride", False)) and max_stride > 1:
        return ("random", max_stride)
    return ("fixed", int(getattr(ds, "temporal_stride", 1) or 1))


def _action_timing_signature_from_cfg(ds, dataset_cfg: dict) -> tuple[str, int]:
    fps = float(getattr(ds, "fps", 0) or 0)
    raw_target_hz = dataset_cfg.get("target_hz")
    if raw_target_hz is not None and float(raw_target_hz) > 0 and fps > 0:
        target_hz = min(float(raw_target_hz), fps)
        stride = max(1, int(round(fps / target_hz)))
    else:
        stride = 1
    if bool(dataset_cfg.get("random_stride", False)) and stride > 1:
        return ("random", stride)
    return ("fixed", stride)


def _format_action_timing_signature(sig: tuple[str, int]) -> str:
    mode, stride = sig
    if mode == "random":
        return f"random[1..{stride}]"
    return f"fixed:{stride}"


def _action_timing_signatures_by_dataset(dataset) -> dict[str, str]:
    ds_list = _flatten_leaf_datasets(dataset)
    return {
        getattr(ds, "dataset_name", type(ds).__name__): _format_action_timing_signature(
            _action_timing_signature_for_dataset(ds)
        )
        for ds in ds_list
    }


def _validate_checkpoint_action_timing(
    *,
    ckpt: dict,
    dataset,
    current_dataset_cfg: dict,
    logger,
    rank: int,
) -> None:
    """Fail fast when checkpoint action stats were built for another action rate.

    Normalizer stats are per policy action after temporal aggregation. Changing
    `chunk_size` alone is fine because it only groups multiple policy actions
    under a visual anchor. Changing `target_hz` or random-stride mode changes
    the policy-action distribution and requires refreshed stats.
    """
    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    ckpt_dataset_cfg = ckpt_cfg.get("dataset", {}) if isinstance(ckpt_cfg, dict) else {}
    if not isinstance(ckpt_dataset_cfg, dict) or not ckpt_dataset_cfg:
        return

    ds_list = _flatten_leaf_datasets(dataset)
    mismatches: list[str] = []
    for ds in ds_list:
        current_sig = _action_timing_signature_for_dataset(ds)
        ckpt_sig = _action_timing_signature_from_cfg(ds, ckpt_dataset_cfg)
        if current_sig != ckpt_sig:
            mismatches.append(
                f"{getattr(ds, 'dataset_name', type(ds).__name__)}: "
                f"checkpoint={_format_action_timing_signature(ckpt_sig)} "
                f"current={_format_action_timing_signature(current_sig)}"
            )
    if mismatches:
        message = (
            "Checkpoint action_normalizer timing mismatches the current dataset. "
            "Pass --refresh-action-stats so q01/q99 are recomputed or loaded for the "
            "current target_hz/random_stride policy-action distribution. "
            + "; ".join(mismatches[:8])
        )
        if rank == 0 and logger is not None:
            logger.error(message)
        raise ValueError(message)


def load_feature_channel_stats(path, device, expected_dim: int, eps: float = 1e-6, logger=None):
    if path is None or str(path).strip() == "":
        return None
    stats_path = os.path.expanduser(str(path))
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Feature channel stats file missing: {stats_path}")
    obj = torch.load(stats_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Feature channel stats must be a dict, got {type(obj).__name__}.")
    if "mean" not in obj:
        raise ValueError(f"Feature channel stats missing 'mean': {stats_path}")
    mean = obj["mean"].float().reshape(-1)
    if "std" in obj:
        std = obj["std"].float().reshape(-1)
    elif "var" in obj:
        std = obj["var"].float().reshape(-1).clamp_min(0).sqrt()
    else:
        raise ValueError(f"Feature channel stats missing 'std' or 'var': {stats_path}")
    if mean.numel() != int(expected_dim) or std.numel() != int(expected_dim):
        raise ValueError(
            f"Feature channel stats dim mismatch for {stats_path}: "
            f"mean={mean.numel()} std={std.numel()} expected={int(expected_dim)}"
        )
    std = std.clamp_min(float(eps))
    if logger is not None:
        logger.info(
            "Loaded feature channel stats from %s: dim=%d count=%s mean_abs=%.4f std_mean=%.4f target_mode=%s",
            stats_path,
            mean.numel(),
            obj.get("count", "<unknown>"),
            mean.abs().mean().item(),
            std.mean().item(),
            str(obj.get("feature_target_mode", "future")).lower(),
        )
    return {
        "mean": mean.to(device=device),
        "std": std.to(device=device),
        "feature_target_mode": str(obj.get("feature_target_mode", "future")).lower(),
        "stats_scope": obj.get("stats_scope", None),
    }


def _contains_glob_metacharacters(path: str | None) -> bool:
    return path is not None and any(ch in str(path) for ch in ("*", "?", "["))


def _patch_deepspeed_checkpoint_glob(model_engine, logger=None, rank: int = 0) -> None:
    """Escape literal checkpoint paths in DeepSpeed's model-state glob search."""
    if model_engine is None or getattr(model_engine, "_da3_ckpt_glob_patch", False):
        return

    import glob
    import types

    def _get_all_ckpt_names_glob_escaped(self, checkpoints_path, tag):
        tag_for_glob = None if tag is None else glob.escape(str(tag))
        ckpt_file_pattern = self._get_ckpt_name(
            glob.escape(str(checkpoints_path)),
            tag_for_glob,
            mp_placeholder="*",
            pp_placeholder="0" if self.load_universal_checkpoint() else None,
        )
        ckpt_files = glob.glob(ckpt_file_pattern)
        ckpt_files.sort()
        return ckpt_files

    model_engine._get_all_ckpt_names = types.MethodType(_get_all_ckpt_names_glob_escaped, model_engine)
    model_engine._da3_ckpt_glob_patch = True
    if logger is not None and rank == 0:
        logger.info("Patched DeepSpeed checkpoint glob search for literal metacharacters in resume paths.")
