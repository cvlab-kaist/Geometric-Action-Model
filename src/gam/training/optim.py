"""Optimizer and scheduler helpers for GAM training."""

from __future__ import annotations

import math

import torch


def build_finetune_optimizer(finetune_model, training_cfg):
    base_lr = float(training_cfg.get("base_lr", 3e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    head_lr_mult = float(training_cfg.get("head_lr_mult", 10.0))
    predictor_lr_mult = float(training_cfg.get("predictor_lr_mult", head_lr_mult))
    adam_eps = float(training_cfg.get("adam_eps", 1e-8))
    adam_beta1 = float(training_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(training_cfg.get("adam_beta2", 0.999))
    backbone_params, head_params, predictor_params = [], [], []
    raw_model = finetune_model.module if hasattr(finetune_model, "module") else finetune_model
    predictor_keywords = (
        "future_predictor.",
        "text_conditioner.proj.",
    )
    head_keywords = (
        "action_head.",
        "action_token",
        "action_timestep_embed",
        "action_input_proj",
    )
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in predictor_keywords):
            predictor_params.append(p)
        elif any(kw in name for kw in head_keywords):
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = [{"params": backbone_params, "lr": base_lr}]
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr * head_lr_mult})
    if predictor_params:
        param_groups.append({"params": predictor_params, "lr": base_lr * predictor_lr_mult})
    opt = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        eps=adam_eps,
        betas=(adam_beta1, adam_beta2),
        fused=True,
    )
    return opt, base_lr, head_lr_mult, predictor_lr_mult, adam_eps


def build_finetune_scheduler(opt, training_cfg, max_steps, logger=None):
    warmup_steps = int(training_cfg.get("warmup_steps", 500))
    min_lr_ratio = float(training_cfg.get("min_lr_ratio", 0.01))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    if logger is not None:
        logger.info(
            "Schedule: cosine decay, warmup=%d steps, max=%d steps, min_lr_ratio=%.3f",
            warmup_steps,
            max_steps,
            min_lr_ratio,
        )
    return scheduler
