"""Dataset / DataLoader / collate helpers for Stage 1 training.

Extracted from ``train_robot.py``. Self-contained: depends only on
stdlib/torch + ``robot.data.dataset.build_robot_dataset``.
"""

from __future__ import annotations

import logging
from collections import Counter
from time import time
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from robot.data.dataset import build_robot_dataset


def _expand_bool_mask_to(mask, target, *, fill_value: bool):
    if mask is None:
        return torch.full_like(target, fill_value=fill_value, dtype=torch.bool)
    mask = mask.to(dtype=torch.bool)
    if tuple(mask.shape) == tuple(target.shape):
        return mask
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(target)


def _sample_view_dim(sample: dict) -> Optional[int]:
    if "all_view_images" in sample:
        return int(sample["all_view_images"].shape[1])
    if "current_images" in sample:
        return int(sample["current_images"].shape[0])
    return None


def _collate_view_count(batch: list[dict]) -> Optional[int]:
    configured = []
    actual = []
    for sample in batch:
        raw_max = sample.get("view_max_views")
        if raw_max not in (None, ""):
            configured.append(int(raw_max))
        dim = _sample_view_dim(sample)
        if dim is not None:
            actual.append(dim)
    if not actual:
        return None
    actual_max = max(actual)
    if configured:
        configured_max = max(configured)
        if actual_max > configured_max:
            raise ValueError(
                f"Sample has {actual_max} views, exceeding configured max_views={configured_max}."
            )
        return configured_max
    return actual_max


def _pad_tensor_dim(tensor: torch.Tensor, dim: int, target: Optional[int], *, fill_value=0):
    if target is None:
        return tensor
    current = int(tensor.shape[dim])
    if current == int(target):
        return tensor
    if current > int(target):
        raise ValueError(f"Cannot pad dim {dim} from {current} down to {target}.")
    pad_shape = list(tensor.shape)
    pad_shape[dim] = int(target) - current
    if tensor.dtype == torch.bool:
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    elif fill_value == 0:
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    else:
        pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


def _pad_view_tensor(tensor: torch.Tensor, max_views: Optional[int], *, view_dim: int, mask: bool = False):
    return _pad_tensor_dim(tensor, view_dim, max_views, fill_value=False if mask else 0)


def _compact_counter(values, *, limit: int = 8) -> str:
    counter = Counter(str(value) for value in values if str(value) != "")
    if not counter:
        return "-"
    items = counter.most_common(max(1, int(limit)))
    suffix = ""
    if len(counter) > len(items):
        suffix = f",...(+{len(counter) - len(items)})"
    return ",".join(f"{key}:{count}" for key, count in items) + suffix


def _batch_source_wait_summary(batch: dict, *, limit: int = 8) -> dict:
    """Return compact CPU-only metadata for diagnosing dataloader stalls."""
    summary = {
        "datasets": _compact_counter(batch.get("dataset_name", []), limit=limit),
        "stats": _compact_counter(batch.get("action_stats_key", []), limit=limit),
        "camera_sets": "-",
        "view_counts": "-",
        "view_valid_ratio": 1.0,
        "max_real_views": 0,
        "episodes": "-",
        "start_t": "-",
        "frames": "-",
        "unique_datasets": 0,
    }
    dataset_names = [str(value) for value in batch.get("dataset_name", []) if str(value) != ""]
    summary["unique_datasets"] = len(set(dataset_names))

    camera_keys = batch.get("camera_keys")
    if camera_keys:
        camera_sets = ["+".join(str(key) for key in keys if str(key) != "") for keys in camera_keys]
        summary["camera_sets"] = _compact_counter(camera_sets, limit=limit)

    view_mask = batch.get("view_valid_mask")
    if torch.is_tensor(view_mask):
        valid = view_mask.detach().to(dtype=torch.bool, device="cpu")
        if valid.numel() > 0:
            per_sample = valid.any(dim=1).sum(dim=1).tolist() if valid.ndim >= 3 else valid.sum(dim=-1).tolist()
            summary["view_counts"] = _compact_counter([int(v) for v in per_sample], limit=limit)
            summary["view_valid_ratio"] = float(valid.float().mean().item())
            summary["max_real_views"] = int(max(per_sample)) if per_sample else 0

    if "episode_index" in batch and torch.is_tensor(batch["episode_index"]):
        episode = batch["episode_index"].detach().cpu().flatten()
        if episode.numel() > 0:
            summary["episodes"] = f"{int(episode.min())}-{int(episode.max())}"
    elif "episode_id" in batch:
        summary["episodes"] = _compact_counter(batch.get("episode_id", []), limit=limit)

    if "start_t" in batch and torch.is_tensor(batch["start_t"]):
        start_t = batch["start_t"].detach().cpu().flatten()
        if start_t.numel() > 0:
            summary["start_t"] = f"{int(start_t.min())}-{int(start_t.max())}"
    if "frame_indices" in batch and torch.is_tensor(batch["frame_indices"]):
        frames = batch["frame_indices"].detach().cpu().flatten()
        if frames.numel() > 0:
            summary["frames"] = f"{int(frames.min())}-{int(frames.max())}"
    return summary


def collate_fn(batch):
    max_views = _collate_view_count(batch)
    result = {
        "current_images": torch.stack([
            _pad_view_tensor(b["current_images"], max_views, view_dim=0)
            for b in batch
        ]),
        "future_images": torch.stack([
            _pad_view_tensor(b["future_images"], max_views, view_dim=1)
            for b in batch
        ]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "proprioception": torch.stack([b["proprioception"] for b in batch]),
        "task_description": [b["task_description"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
        "action_stats_key": [b["action_stats_key"] for b in batch],
    }
    if "all_view_images" in batch[0]:
        result["all_view_images"] = torch.stack([
            _pad_view_tensor(b["all_view_images"], max_views, view_dim=1)
            for b in batch
        ])
        view_masks = []
        for b in batch:
            if "view_valid_mask" in b:
                mask = b["view_valid_mask"].to(dtype=torch.bool)
            else:
                shape = b["all_view_images"].shape[:2]
                mask = torch.ones(shape, dtype=torch.bool)
            view_masks.append(_pad_view_tensor(mask, max_views, view_dim=1, mask=True))
        result["view_valid_mask"] = torch.stack(view_masks)
    if any("all_view_target_images" in b for b in batch):
        result["all_view_target_images"] = torch.stack([
            _pad_view_tensor(
                b.get("all_view_target_images", b["all_view_images"]),
                max_views,
                view_dim=1,
            )
            for b in batch
        ])
    if any("all_view_target_mask" in b for b in batch):
        target_masks = []
        for b in batch:
            mask = b.get(
                "all_view_target_mask",
                torch.ones(
                    b["all_view_images"].shape[0],
                    b["all_view_images"].shape[1],
                    b["all_view_images"].shape[-2],
                    b["all_view_images"].shape[-1],
                    dtype=torch.bool,
                ),
            ).to(dtype=torch.bool)
            target_masks.append(_pad_view_tensor(mask, max_views, view_dim=1, mask=True))
        result["all_view_target_mask"] = torch.stack(target_masks)
        if "view_valid_mask" in result:
            valid = result["view_valid_mask"]
            while valid.ndim < result["all_view_target_mask"].ndim:
                valid = valid.unsqueeze(-1)
            result["all_view_target_mask"] = result["all_view_target_mask"] & valid
    if any("past_action_history" in b for b in batch):
        template = next(b["past_action_history"] for b in batch if "past_action_history" in b)
        fill = torch.zeros_like(template)
        result["past_action_history"] = torch.stack([
            b["past_action_history"] if "past_action_history" in b else fill.clone()
            for b in batch
        ])
    if any("past_action_history_mask" in b for b in batch):
        template_history = next(
            b["past_action_history"] for b in batch if "past_action_history" in b
        )
        result["past_action_history_mask"] = torch.stack([
            _expand_bool_mask_to(
                b.get("past_action_history_mask"),
                b.get("past_action_history", template_history),
                fill_value=False,
            )
            for b in batch
        ])
    if "start_t" in batch[0]:
        result["start_t"] = torch.stack([b["start_t"] for b in batch])
    if "frame_indices" in batch[0]:
        result["frame_indices"] = torch.stack([b["frame_indices"] for b in batch])
    if any("action_loss_mask" in b for b in batch):
        result["action_loss_mask"] = torch.stack([
            _expand_bool_mask_to(b.get("action_loss_mask"), b["actions"], fill_value=True)
            for b in batch
        ])
    for key in ("transition_loss_mask", "context_valid_mask"):
        if any(key in b for b in batch):
            template = next(b[key] for b in batch if key in b)
            fill = torch.ones_like(template, dtype=torch.bool)
            result[key] = torch.stack([
                b[key].to(dtype=torch.bool) if key in b else fill.clone()
                for b in batch
            ])
    # Mixed batches may have only some samples carrying episode_index (OxE) or
    # episode_id (MimicGen).  Stack episode_index if every sample has it;
    # otherwise skip : it's an optional metadata field.
    if all("episode_index" in b for b in batch):
        result["episode_index"] = torch.stack([b["episode_index"] for b in batch])
    if all("episode_id" in b for b in batch):
        result["episode_id"] = [b["episode_id"] for b in batch]
    if all("camera_keys" in b for b in batch):
        camera_keys = []
        for b in batch:
            keys = list(b["camera_keys"])
            if max_views is not None and len(keys) < max_views:
                keys = keys + [""] * (max_views - len(keys))
            camera_keys.append(keys)
        result["camera_keys"] = camera_keys
    for key in ("action_taxonomy", "action_transform", "state_transform", "raw_action_key", "raw_state_key"):
        if any(key in b for b in batch):
            result[key] = [b.get(key, "") for b in batch]
    if any("action_dim_mask_metadata" in b for b in batch):
        result["action_dim_mask_metadata"] = torch.stack([
            b.get("action_dim_mask_metadata", torch.ones(7, dtype=torch.bool))
            for b in batch
        ])
    # GT depth keys may be present on only a subset of the batch (mixer scenario
    # where MimicGen samples carry sidecars while OxE samples carry empty entries). We stack
    # if ANY sample has the key, filling missing samples with zeros (mask=False
    # for gt_depth_mask).  The unified depth loss can then fill mask-empty
    # sample/time/view slots with teacher DA3 pseudo-depth when fallback is on.
    gt_keys = (
        "gt_depth_meters",
        "gt_depth_da3",
        "gt_depth_mask",
        "gt_depth_scene_scale",
        "gt_camera_intrinsics",
        "gt_camera_extrinsics_c2w",
    )
    for key in gt_keys:
        any_have = any(key in b for b in batch)
        if not any_have:
            continue
        # Find a template tensor to learn shape/dtype.
        template = None
        for b in batch:
            if key in b:
                template = b[key]
                break
        stacked = []
        for b in batch:
            if key in b:
                value = b[key]
                if key != "gt_depth_scene_scale":
                    value = _pad_view_tensor(
                        value.to(dtype=torch.bool) if key == "gt_depth_mask" else value,
                        max_views,
                        view_dim=1,
                        mask=(key == "gt_depth_mask"),
                    )
                stacked.append(value)
            else:
                template_padded = template
                if key != "gt_depth_scene_scale":
                    template_padded = _pad_view_tensor(
                        template.to(dtype=torch.bool) if key == "gt_depth_mask" else template,
                        max_views,
                        view_dim=1,
                        mask=(key == "gt_depth_mask"),
                    )
                fill = (
                    torch.zeros_like(template_padded)
                    if key != "gt_depth_mask"
                    else torch.zeros_like(template_padded, dtype=torch.bool)
                )
                stacked.append(fill)
        result[key] = torch.stack(stacked)
        if key == "gt_depth_mask" and "view_valid_mask" in result:
            valid = result["view_valid_mask"]
            while valid.ndim < result[key].ndim:
                valid = valid.unsqueeze(-1)
            result[key] = result[key] & valid
    return result


def _dataloader_worker_init(worker_id: int):
    """Pin each DataLoader worker to single-thread tensor ops.

    Each python process (main trainer + every DataLoader worker) defaults to
    `torch.get_num_threads() == cpu_count`. On CVLAB1 (B200 × 8-core host)
    with 4 ranks × num_workers=8 = 36 python processes, the resulting
    1000+ OMP/Torch threads saturate the load average (~50+) and stall all
    of the small tensor ops inside the dataset's crop/resize/normalize
    chain : a torch.stack that should take 0.1 ms ends up at 40 ms.
    Forcing 1 thread per worker removes the contention; the main trainer
    still gets its default multi-threaded math for model forwards.
    """
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


class RestartableDistributedSampler(DistributedSampler):
    """DistributedSampler variant that can resume inside the current epoch.

    Closed-loop simulator eval may intentionally shut down persistent
    DataLoader workers mid-epoch. Recreating a regular DistributedSampler
    iterator would replay the same epoch prefix, so this sampler keeps the
    normal deterministic ordering but skips the rank-local batches that were
    already consumed.
    """

    def __init__(
        self,
        dataset,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self._resume_sample_index = 0
        self._resume_batch_size = max(1, int(batch_size))

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._resume_sample_index = 0

    def set_resume_batch(self, batch_idx: int, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self._resume_batch_size = max(1, int(batch_size))
        self._resume_sample_index = max(0, int(batch_idx)) * self._resume_batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        start = min(self._resume_sample_index, len(indices))
        indices = indices[start:]
        if hasattr(self.dataset, "make_epoch_index"):
            epoch = int(getattr(self, "epoch", 0))
            indices = [self.dataset.make_epoch_index(idx, epoch) for idx in indices]
        return iter(indices)

    def __len__(self) -> int:
        start = min(self._resume_sample_index, int(self.num_samples))
        return max(0, int(self.num_samples) - start)


class VirtualEpochDataset(Dataset):
    """Repeat a small map-style dataset to provide enough batches per epoch.

    This preserves the underlying samples and normalization/stat methods while
    making DistributedSampler/DataLoader see a longer epoch. Repeated
    indices are acceptable for datasets that randomize starts/augmentations in
    __getitem__, and they let DataLoader workers prefetch multiple batches
    instead of rebuilding a one-batch epoch every optimizer step.
    """

    def __init__(self, dataset: Dataset, virtual_length: int):
        self.dataset = dataset
        self.physical_length = len(dataset)
        if self.physical_length <= 0:
            raise ValueError("Cannot virtualize an empty dataset.")
        self.virtual_length = max(int(virtual_length), self.physical_length)

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, idx: int):
        idx = int(idx)
        epoch = idx // self.virtual_length
        virtual_idx = idx % self.virtual_length
        physical_idx = virtual_idx % self.physical_length
        if hasattr(self.dataset, "make_epoch_index"):
            return self.dataset[self.dataset.make_epoch_index(physical_idx, epoch)]
        return self.dataset[physical_idx]

    def make_epoch_index(self, idx: int, epoch: int) -> int:
        return int(epoch) * int(self.virtual_length) + (int(idx) % int(self.virtual_length))

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def _maybe_virtualize_train_epoch(dataset, training_cfg, world_size: int, rank: int):
    min_batches = int(training_cfg.get("min_train_batches_per_epoch", 0) or 0)
    if min_batches <= 0:
        return dataset
    batch_size = int(training_cfg.get("micro_batch_size", 4))
    target_len = batch_size * max(int(world_size), 1) * min_batches
    if len(dataset) >= target_len:
        return dataset
    wrapped = VirtualEpochDataset(dataset, target_len)
    if rank == 0:
        print(
            "Dataset virtual epoch [train] "
            f"physical_size={wrapped.physical_length} virtual_size={len(wrapped)} "
            f"min_batches_per_rank={min_batches}",
            flush=True,
        )
    return wrapped


def _set_train_sampler_position(
    sampler,
    *,
    epoch: int,
    start_batch_idx: int,
    batch_size: int,
    rank: int,
    logger: Optional[logging.Logger] = None,
    reason: str = "",
) -> None:
    """Set deterministic train sampler epoch and optional mid-epoch cursor."""
    if sampler is None:
        return
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))
    if int(start_batch_idx) <= 0:
        return
    if hasattr(sampler, "set_resume_batch"):
        sampler.set_resume_batch(int(start_batch_idx), batch_size=batch_size)
        if rank == 0 and logger is not None:
            suffix = f" ({reason})" if reason else ""
            logger.info(
                "Resuming train DataLoader at epoch=%d batch=%d%s",
                int(epoch),
                int(start_batch_idx),
                suffix,
            )
    elif rank == 0 and logger is not None:
        logger.warning(
            "Train sampler lacks mid-epoch resume support; recreated iterator "
            "may replay epoch prefix at epoch=%d batch=%d",
            int(epoch),
            int(start_batch_idx),
        )


def create_dataset_and_loader(dataset_cfg, training_cfg, use_deepspeed, world_size, rank, is_eval=False):
    dataset_cfg_for_build = dataset_cfg
    if is_eval and isinstance(dataset_cfg, dict) and dataset_cfg.get("eval_max_episodes") is not None:
        dataset_cfg_for_build = dict(dataset_cfg)
        eval_max_episodes = dataset_cfg_for_build.get("eval_max_episodes")
        dataset_cfg_for_build["max_episodes"] = None if int(eval_max_episodes) <= 0 else int(eval_max_episodes)

    train_batch_size = int(training_cfg.get("micro_batch_size", 4))
    batch_size = train_batch_size
    if is_eval:
        eval_batch_override = training_cfg.get(
            "eval_micro_batch_size",
            training_cfg.get("eval_batch_size", None),
        )
        if eval_batch_override is not None:
            batch_size = max(1, int(eval_batch_override))

    dataset_t0 = time()
    dataset = build_robot_dataset(dataset_cfg_for_build, is_eval=is_eval)
    dataset_build_seconds = time() - dataset_t0
    if rank == 0:
        mode = "eval" if is_eval else "train"
        max_episodes = dataset_cfg_for_build.get("max_episodes") if isinstance(dataset_cfg_for_build, dict) else None
        eval_max_episodes = dataset_cfg.get("eval_max_episodes") if isinstance(dataset_cfg, dict) else None
        print(
            f"Dataset build [{mode}] size={len(dataset)} elapsed={dataset_build_seconds:.1f}s "
            f"max_episodes={max_episodes} eval_max_episodes={eval_max_episodes} "
            f"batch_size={batch_size}",
            flush=True,
        )
    if not is_eval:
        dataset = _maybe_virtualize_train_epoch(dataset, training_cfg, world_size, rank)
    if is_eval:
        # Distributed eval: shard across all ranks, no shuffle
        if use_deepspeed or world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        else:
            sampler = None
        shuffle = False
    elif use_deepspeed or world_size > 1:
        sampler = RestartableDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            batch_size=batch_size,
        )
        shuffle = False
    else:
        sampler = RestartableDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            batch_size=batch_size,
        )
        shuffle = False
    n_workers = int(training_cfg.get("num_workers", 4))
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=not is_eval,
        collate_fn=collate_fn,
        worker_init_fn=_dataloader_worker_init,
    )
    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
        if training_cfg.get("persistent_workers", False):
            loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, sampler, loader
