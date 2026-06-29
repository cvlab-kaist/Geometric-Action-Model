"""NaN / non-finite debugging helpers for Stage 1 training.

Extracted from ``train_robot.py``. These are gated by ``DA3_NAN_DEBUG`` in the
training loop because they synchronize CUDA tensors and can be expensive.
Self-contained: depend only on stdlib/torch.
"""

from __future__ import annotations

import logging
import os
import re

import torch
import torch.nn as nn


def _parse_debug_name_filters() -> tuple[str, ...]:
    raw = os.environ.get("DA3_NAN_DEBUG_FILTER", "").strip()
    if not raw:
        return ()
    # sbatch --export itself is comma-separated, so support ':' / ';' /
    # whitespace delimiters for multi-name filters passed through Slurm.
    return tuple(part.strip() for part in re.split(r"[,;:\s]+", raw) if part.strip())


def _name_matches_debug_filter(name: str, filters: tuple[str, ...]) -> bool:
    return not filters or any(part in name for part in filters)


def _summarize_nonfinite_named_tensors(
    named_tensors,
    *,
    filters: tuple[str, ...],
    max_items: int = 12,
):
    """Return compact diagnostics for named tensors containing NaN/Inf.

    This is intentionally gated by DA3_NAN_DEBUG in the training loop because
    it synchronizes CUDA tensors and can be expensive on large models.
    """
    bad = []
    for name, tensor in named_tensors:
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not torch.is_floating_point(tensor):
            continue
        if not _name_matches_debug_filter(name, filters):
            continue
        with torch.no_grad():
            finite = torch.isfinite(tensor)
            if bool(finite.all().item()):
                continue
            finite_count = int(finite.sum().item())
            total = tensor.numel()
            finite_vals = tensor.detach()[finite]
            if finite_vals.numel() > 0:
                finite_abs_max = float(finite_vals.float().abs().max().item())
            else:
                finite_abs_max = None
            bad.append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "nonfinite": int(total - finite_count),
                    "total": int(total),
                    "finite_abs_max": finite_abs_max,
                }
            )
            if len(bad) >= max_items:
                break
    return bad


def _summarize_named_tensor_stats(
    named_tensors,
    *,
    filters: tuple[str, ...],
    max_items: int = 8,
):
    summaries = []
    for name, tensor in named_tensors:
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not torch.is_floating_point(tensor):
            continue
        if not _name_matches_debug_filter(name, filters):
            continue
        with torch.no_grad():
            detached = tensor.detach()
            finite = torch.isfinite(detached)
            finite_count = int(finite.sum().item())
            total = int(detached.numel())
            finite_vals = detached[finite].float()
            summaries.append(
                {
                    "name": name,
                    "shape": tuple(detached.shape),
                    "dtype": str(detached.dtype),
                    "finite": finite_count,
                    "total": total,
                    "abs_max": None
                    if finite_vals.numel() == 0
                    else float(finite_vals.abs().max().item()),
                    "norm": None
                    if finite_vals.numel() == 0
                    else float(finite_vals.norm().item()),
                    "mean": None
                    if finite_vals.numel() == 0
                    else float(finite_vals.mean().item()),
                }
            )
            if len(summaries) >= max_items:
                break
    return summaries


def _iter_named_floating_tensors(name: str, value):
    if value is None:
        return
    if torch.is_tensor(value):
        if torch.is_floating_point(value):
            yield name, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_named_floating_tensors(f"{name}.{key}", item)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_named_floating_tensors(f"{name}.{idx}", item)


def _debug_log_nonfinite_forward_state(
    *,
    model: nn.Module,
    out: dict | None,
    rank: int,
    step: int,
    logger: logging.Logger,
    filters: tuple[str, ...],
):
    """Log compact state when the loss is already non-finite before backward."""
    output_named = []
    if out is not None:
        interesting = (
            "action_pred",
            "target_actions",
            "predicted_last_visual_tokens",
            "target_last_visual_tokens",
            "last_predicted_action_tokens",
            "past_slots",
            "student_raw",
            "teacher_past_raw",
        )
        for key in interesting:
            if key in out:
                output_named.extend(_iter_named_floating_tensors(f"out.{key}", out[key]))
    bad_outputs = _summarize_nonfinite_named_tensors(
        output_named,
        filters=(),
        max_items=20,
    )
    if bad_outputs:
        logger.error(
            "DA3_NAN_DEBUG non-finite forward outputs at step=%d rank=%d: %s",
            step,
            rank,
            bad_outputs,
        )

    # On failure, scan all trainable params once. This expensive path runs only
    # on the failing step and catches parameters outside the active step-by-step
    # debug filter.
    named_params = [
        (name, p.data)
        for name, p in model.named_parameters()
        if p.requires_grad and torch.is_floating_point(p.data)
    ]
    bad_params = _summarize_nonfinite_named_tensors(
        named_params,
        filters=(),
        max_items=20,
    )
    if bad_params:
        logger.error(
            "DA3_NAN_DEBUG non-finite trainable parameters at step=%d rank=%d: %s",
            step,
            rank,
            bad_params,
        )
    elif filters:
        stats = _summarize_named_tensor_stats(named_params, filters=filters, max_items=12)
        if stats and rank == 0:
            logger.info(
                "DA3_NAN_DEBUG filtered parameter stats on loss failure at step=%d: %s",
                step,
                stats,
            )


def _debug_log_finetune_tensor_stats(
    model: nn.Module,
    *,
    rank: int,
    step: int,
    where: str,
    logger: logging.Logger,
    filters: tuple[str, ...],
    check_grads: bool,
):
    if rank != 0:
        return
    named = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        named.append((f"{name}.grad" if check_grads else name, p.grad if check_grads else p.data))
    summaries = _summarize_named_tensor_stats(named, filters=filters)
    if summaries:
        logger.info(
            "DA3_NAN_DEBUG stats %s at step=%d where=%s: %s",
            "gradients" if check_grads else "parameters",
            step,
            where,
            summaries,
        )


def _debug_check_finetune_tensors(
    model: nn.Module,
    *,
    rank: int,
    step: int,
    where: str,
    logger: logging.Logger,
    filters: tuple[str, ...],
    check_grads: bool,
):
    named = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        named.append((f"{name}.grad" if check_grads else name, p.grad if check_grads else p.data))
    bad = _summarize_nonfinite_named_tensors(named, filters=filters)
    if bad:
        logger.error(
            "DA3_NAN_DEBUG non-finite %s at step=%d rank=%d where=%s: %s",
            "gradients" if check_grads else "parameters",
            step,
            rank,
            where,
            bad,
        )
        raise FloatingPointError(
            f"DA3_NAN_DEBUG found non-finite {'gradients' if check_grads else 'parameters'} "
            f"at step {step} ({where})"
        )


def _register_nan_debug_grad_hooks(
    model: nn.Module,
    *,
    rank: int,
    logger: logging.Logger,
    filters: tuple[str, ...],
    step_context: dict,
    start_step: int,
    log_stats: bool,
) -> list:
    handles = []

    for name, p in model.named_parameters():
        if not p.requires_grad or not _name_matches_debug_filter(name, filters):
            continue

        def _hook(grad, param_name=name):
            step = int(step_context.get("step", 0))
            if step < start_step:
                return grad
            if log_stats and rank == 0:
                stats = _summarize_named_tensor_stats(
                    [(f"{param_name}.grad_hook", grad)],
                    filters=(),
                    max_items=1,
                )
                if stats:
                    logger.info(
                        "DA3_NAN_DEBUG stats raw_gradient at step=%d: %s",
                        step,
                        stats,
                    )
            bad = _summarize_nonfinite_named_tensors(
                [(f"{param_name}.grad_hook", grad)],
                filters=(),
                max_items=1,
            )
            if bad:
                logger.error(
                    "DA3_NAN_DEBUG non-finite raw gradient at step=%d rank=%d: %s",
                    step,
                    rank,
                    bad,
                )
                raise FloatingPointError(
                    f"DA3_NAN_DEBUG found non-finite raw gradient at step {step}"
                )
            return grad

        handles.append(p.register_hook(_hook))

    return handles
