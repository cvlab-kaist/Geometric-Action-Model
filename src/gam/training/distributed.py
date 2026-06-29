"""Distributed setup, Slurm preemption, signal handling, and git helpers.

Extracted from ``train_robot.py``. Self-contained: depends only on
stdlib/torch + ``omegaconf``. Module-level globals that back the Slurm /
preemption helpers live here too because nothing outside this group reads
them.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
from time import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def _plain_config_container(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_distributed_timeout_minutes(args, cfg=None) -> float:
    """Resolve process-group timeout for long in-training eval sections."""
    candidates = [
        getattr(args, "ddp_timeout_minutes", None),
        os.environ.get("DA3_DISTRIBUTED_TIMEOUT_MINUTES"),
        os.environ.get("DA3_DDP_TIMEOUT_MINUTES"),
    ]
    if cfg is not None:
        training_cfg = _plain_config_container(cfg.get("training", {})) or {}
        if isinstance(training_cfg, dict):
            candidates.extend(
                [
                    training_cfg.get("distributed_timeout_minutes"),
                    training_cfg.get("ddp_timeout_minutes"),
                    training_cfg.get("process_group_timeout_minutes"),
                ]
            )

    for value in candidates:
        if value in (None, ""):
            continue
        timeout_minutes = float(value)
        if timeout_minutes <= 0:
            raise ValueError(f"Distributed timeout must be positive minutes, got {value!r}.")
        return timeout_minutes
    return 120.0


_TRAINING_IMPORT_TIME_EPOCH = time()
_TRAINING_TERMINATION_REQUESTED = False
_TRAINING_TERMINATION_SIGNAL = None
_SLURM_SCONTROL_TIME_LIMIT_SECONDS = None
_SLURM_SCONTROL_TIME_LIMIT_CHECKED = False


def _training_signal_handler(signum, frame):  # noqa: ARG001
    """Record preemption without doing unsafe work inside the signal handler."""
    global _TRAINING_TERMINATION_REQUESTED, _TRAINING_TERMINATION_SIGNAL
    _TRAINING_TERMINATION_REQUESTED = True
    _TRAINING_TERMINATION_SIGNAL = int(signum)


def _install_training_signal_handlers(logger=None, rank: int = 0) -> None:
    installed = []
    for sig_name in ("SIGUSR1", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _training_signal_handler)
            installed.append(sig_name)
        except (OSError, RuntimeError, ValueError):
            continue
    if logger is not None and rank == 0 and installed:
        logger.info("Installed training preemption signal handlers: %s", ", ".join(installed))


def _parse_slurm_time_limit_seconds(value) -> int | None:
    """Parse Slurm time limits from env (`minutes`, `D-HH:MM:SS`, or `HH:MM:SS`)."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"UNLIMITED", "NOT_SET", "N/A", "UNKNOWN"}:
        return None
    if text.isdigit():
        # SLURM_TIMELIMIT is commonly exported as minutes.
        return int(text) * 60
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        try:
            days = int(day_text)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(p) for p in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(p) for p in parts)
        elif len(parts) == 1:
            hours = 0
            minutes = int(parts[0])
            seconds = 0
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _slurm_time_limit_seconds_from_scontrol() -> int | None:
    """Resolve Slurm TimeLimit through scontrol when env propagation is incomplete."""
    global _SLURM_SCONTROL_TIME_LIMIT_SECONDS, _SLURM_SCONTROL_TIME_LIMIT_CHECKED
    if _SLURM_SCONTROL_TIME_LIMIT_CHECKED:
        return _SLURM_SCONTROL_TIME_LIMIT_SECONDS
    _SLURM_SCONTROL_TIME_LIMIT_CHECKED = True

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        proc = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    for token in proc.stdout.replace("\n", " ").split():
        if token.startswith("TimeLimit="):
            _SLURM_SCONTROL_TIME_LIMIT_SECONDS = _parse_slurm_time_limit_seconds(
                token.split("=", 1)[1]
            )
            break
    return _SLURM_SCONTROL_TIME_LIMIT_SECONDS


def _slurm_remaining_seconds() -> float | None:
    limit = os.environ.get("DA3_JOB_TIME_LIMIT_SEC")
    try:
        limit_seconds = float(limit) if limit not in (None, "") else None
    except ValueError:
        limit_seconds = None
    if limit_seconds is None:
        limit_seconds = _parse_slurm_time_limit_seconds(os.environ.get("SLURM_TIMELIMIT"))
    if limit_seconds is None:
        limit_seconds = _slurm_time_limit_seconds_from_scontrol()
    if limit_seconds is None or limit_seconds <= 0:
        return None

    start = os.environ.get("DA3_JOB_START_TIME_EPOCH")
    try:
        start_epoch = float(start) if start not in (None, "") else _TRAINING_IMPORT_TIME_EPOCH
    except ValueError:
        start_epoch = _TRAINING_IMPORT_TIME_EPOCH
    return max(0.0, float(limit_seconds) - (time() - start_epoch))


def _resolve_seconds_setting(training_cfg: dict, key: str, env_key: str, default: float) -> float:
    env_val = os.environ.get(env_key)
    cfg_val = training_cfg.get(key, None) if isinstance(training_cfg, dict) else None
    value = env_val if env_val not in (None, "") else cfg_val
    if value in (None, ""):
        return float(default)
    seconds = float(value)
    if seconds < 0:
        raise ValueError(f"{key} must be non-negative seconds, got {value!r}.")
    return seconds


def _termination_requested_across_ranks(device) -> tuple[bool, int | None]:
    local_requested = 1 if _TRAINING_TERMINATION_REQUESTED else 0
    signal_value = int(_TRAINING_TERMINATION_SIGNAL or 0)
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([local_requested, signal_value], device=device, dtype=torch.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        local_requested = int(tensor[0].item())
        signal_value = int(tensor[1].item())
    return bool(local_requested), signal_value or None


def _get_git_info():
    """Get git commit hash, branch, and dirty flag."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def setup_distributed(args, cfg=None):
    use_deepspeed = getattr(args, "deepspeed_config", None) is not None and not args.single_gpu
    timeout_minutes = _resolve_distributed_timeout_minutes(args, cfg)
    setattr(args, "_distributed_timeout_minutes", timeout_minutes)
    timeout_delta = datetime.timedelta(minutes=timeout_minutes)
    if use_deepspeed:
        import deepspeed

        deepspeed.init_distributed(timeout=timeout_delta)
        # Increase NCCL collective timeout (default 600s is too short for
        # dataset loading and long train-time closed-loop rollout eval).
        if dist.is_initialized():
            pg = dist.distributed_c10d._get_default_group()
            pg._get_backend(torch.device("cuda")).options._timeout = timeout_delta
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_idx = rank % torch.cuda.device_count()
    elif not args.single_gpu and "RANK" in os.environ:
        # torchrun DDP
        dist.init_process_group(backend="nccl", timeout=timeout_delta)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_idx = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        device_idx = 0

    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)
    if os.environ.get("MUJOCO_GL", "").strip().lower() == "egl":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(device_idx))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Hopper: enable the cuDNN-fused-flash-attention SDPA backend (PyTorch
    # 2.5+; reported up to ~75% over FA2 on H100-class GPUs in cuDNN 9 blog).
    # Off by default in PyTorch. Soft-fail on older builds.
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
    except (AttributeError, RuntimeError):
        pass
    return use_deepspeed, rank, world_size, device


def validate_deepspeed_batch_config(args, training_cfg, world_size, logger=None):
    ds_config_path = getattr(args, "deepspeed_config", None)
    if ds_config_path is None:
        return

    with open(ds_config_path, "r", encoding="utf-8") as f:
        ds_cfg = json.load(f)

    cfg_micro = int(training_cfg.get("micro_batch_size", 4))
    cfg_accum = int(training_cfg.get("grad_accum_steps", 1))
    ds_micro = ds_cfg.get("train_micro_batch_size_per_gpu")
    ds_accum = ds_cfg.get("gradient_accumulation_steps")

    if ds_micro is not None and int(ds_micro) != cfg_micro:
        raise ValueError(
            f"DeepSpeed micro batch mismatch: deepspeed={ds_micro}, training.micro_batch_size={cfg_micro}. "
            f"Update either {ds_config_path} or the training config."
        )
    if ds_accum is not None and int(ds_accum) != cfg_accum:
        raise ValueError(
            f"DeepSpeed grad accumulation mismatch: deepspeed={ds_accum}, training.grad_accum_steps={cfg_accum}. "
            f"Update either {ds_config_path} or the training config."
        )

    global_batch_size = training_cfg.get("global_batch_size")
    if global_batch_size is not None:
        expected_global = cfg_micro * cfg_accum * world_size
        if int(global_batch_size) != expected_global:
            raise ValueError(
                f"Global batch mismatch: training.global_batch_size={int(global_batch_size)} "
                f"but micro_batch_size({cfg_micro}) * grad_accum_steps({cfg_accum}) * world_size({world_size}) "
                f"= {expected_global}."
            )

    if logger is not None:
        logger.info(
            "DeepSpeed batch config OK: micro_batch=%d, grad_accum=%d, world_size=%d",
            cfg_micro, cfg_accum, world_size,
        )
