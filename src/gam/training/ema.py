"""EMA helpers for GAM Stage 1 training."""

from __future__ import annotations

import re

import torch

from robot.modeling.backbone_factory import backbone_block_count, default_freeze_blocks_before

from .model import strip_compile_prefix_state_dict, unwrap_train_model


def stage1_subset_ema_config(training_cfg):
    raw_cfg = training_cfg.get("ema", {}) or {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {"enabled": bool(raw_cfg)}
    enabled = bool(raw_cfg.get("enabled", training_cfg.get("ema_enabled", False)))
    include = raw_cfg.get(
        "include",
        ["future_predictor", "action_head", "text_conditioner_proj", "student_da3_blocks"],
    )
    if isinstance(include, str):
        include = [include]
    blocks_start = raw_cfg.get("student_da3_blocks_start", None)
    blocks_end = raw_cfg.get("student_da3_blocks_end", None)
    return {
        "enabled": enabled,
        "decay": float(raw_cfg.get("decay", training_cfg.get("ema_decay", 0.999))),
        "device": str(raw_cfg.get("device", "cpu")).lower(),
        "dtype": str(raw_cfg.get("dtype", "float32")).lower(),
        "start_step": int(raw_cfg.get("start_step", 0)),
        "update_every": max(1, int(raw_cfg.get("update_every", 1))),
        "include": tuple(str(x) for x in include),
        "student_da3_blocks_start": None if blocks_start is None else int(blocks_start),
        "student_da3_blocks_end": None if blocks_end is None else int(blocks_end),
    }


def finalize_stage1_subset_ema_config(ema_cfg, student_da3):
    cfg = dict(ema_cfg)
    if cfg["student_da3_blocks_start"] is None:
        cfg["student_da3_blocks_start"] = default_freeze_blocks_before(student_da3)
    if cfg["student_da3_blocks_end"] is None:
        cfg["student_da3_blocks_end"] = max(0, backbone_block_count(student_da3) - 1)
    return cfg


class Stage1SubsetEMA:
    """EMA tracker for the trainable GAM policy subgraph."""

    _PREFIX_TO_CKPT_KEY = {
        "student_da3": "student_da3_ema",
        "action_head": "action_head_ema",
        "future_predictor": "future_predictor_ema",
        "text_conditioner.proj": "text_conditioner_proj_ema",
    }

    def __init__(self, model, cfg, logger=None):
        self.decay = float(cfg["decay"])
        if not (0.0 <= self.decay < 1.0):
            raise ValueError(f"training.ema.decay must be in [0, 1), got {self.decay}")
        self.device_spec = str(cfg["device"]).lower()
        self.dtype = self._resolve_dtype(str(cfg["dtype"]).lower())
        self.start_step = int(cfg["start_step"])
        self.update_every = max(1, int(cfg["update_every"]))
        self.include = tuple(cfg["include"])
        self.blocks_start = int(cfg["student_da3_blocks_start"])
        self.blocks_end = int(cfg["student_da3_blocks_end"])
        if self.blocks_start > self.blocks_end:
            raise ValueError(
                "training.ema.student_da3_blocks_start must be <= "
                "student_da3_blocks_end"
            )

        self.names = []
        self.shadow = {}
        raw_model = unwrap_train_model(model)
        for name, param in raw_model.named_parameters():
            if not param.requires_grad or not self._tracks_name(name):
                continue
            self.names.append(name)
            self.shadow[name] = self._copy_param(param)

        self.num_params = sum(int(self.shadow[name].numel()) for name in self.names)
        self.num_tensors = len(self.names)
        if logger is not None:
            logger.info(
                "Stage1 subset EMA enabled: decay=%.6f update_every=%d start_step=%d "
                "device=%s dtype=%s tensors=%d params=%.1fM scope=%s blocks=%d-%d",
                self.decay,
                self.update_every,
                self.start_step,
                self.device_spec,
                str(self.dtype).replace("torch.", ""),
                self.num_tensors,
                self.num_params / 1e6,
                ",".join(self.include),
                self.blocks_start,
                self.blocks_end,
            )
            if self.num_tensors == 0:
                logger.warning("Stage1 subset EMA is enabled but no trainable parameters matched the scope.")

    @staticmethod
    def _resolve_dtype(name):
        aliases = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if name not in aliases:
            raise ValueError(f"Unsupported training.ema.dtype={name!r}; expected float32, bfloat16, or float16")
        return aliases[name]

    def _target_device(self, param):
        if self.device_spec in {"model", "param", "same"}:
            return param.device
        return torch.device(self.device_spec)

    def _copy_param(self, param):
        return param.detach().to(device=self._target_device(param), dtype=self.dtype).clone()

    def _tracks_name(self, name):
        include = set(self.include)
        if "future_predictor" in include and name.startswith("future_predictor."):
            return True
        if "action_head" in include and name.startswith("action_head."):
            return True
        if "text_conditioner_proj" in include and name.startswith("text_conditioner.proj."):
            return True
        if "student_da3_blocks" in include and name.startswith("student_da3."):
            match = re.search(r"\.blocks\.(\d+)\.", name)
            if match is not None:
                block_idx = int(match.group(1))
                return self.blocks_start <= block_idx <= self.blocks_end
        return False

    @torch.no_grad()
    def update(self, model, step):
        if self.num_tensors == 0:
            return False
        if step < self.start_step or step % self.update_every != 0:
            return False
        raw_model = unwrap_train_model(model)
        params = dict(raw_model.named_parameters())
        decay = self.decay
        one_minus = 1.0 - decay
        for name in self.names:
            param = params.get(name)
            if param is None:
                continue
            src = param.detach().to(device=self.shadow[name].device, dtype=self.shadow[name].dtype)
            self.shadow[name].mul_(decay).add_(src, alpha=one_minus)
        return True

    @torch.no_grad()
    def load_from_checkpoint(self, ckpt, logger=None):
        loaded = 0
        skipped = []
        flat_shadow = ckpt.get("ema", {}).get("shadow") if isinstance(ckpt.get("ema"), dict) else None
        if isinstance(flat_shadow, dict):
            for name, tensor in strip_compile_prefix_state_dict(flat_shadow).items():
                loaded += self._copy_into_shadow(name, tensor, skipped)

        for prefix, ckpt_key in self._PREFIX_TO_CKPT_KEY.items():
            state = ckpt.get(ckpt_key)
            if state is None:
                continue
            for key, tensor in strip_compile_prefix_state_dict(state).items():
                loaded += self._copy_into_shadow(f"{prefix}.{key}", tensor, skipped)

        if logger is not None:
            if loaded:
                logger.info("Restored Stage1 subset EMA tensors from checkpoint: %d", loaded)
            else:
                logger.info("No Stage1 subset EMA tensors found in checkpoint; initialized EMA from live weights.")
            if skipped:
                logger.warning("Skipped %d incompatible Stage1 EMA tensors during load.", len(skipped))
        return loaded

    def _copy_into_shadow(self, name, tensor, skipped):
        name = str(name).replace("_orig_mod.", "")
        if name not in self.shadow:
            return 0
        if tuple(tensor.shape) != tuple(self.shadow[name].shape):
            skipped.append((name, tuple(tensor.shape), tuple(self.shadow[name].shape)))
            return 0
        self.shadow[name].copy_(tensor.detach().to(device=self.shadow[name].device, dtype=self.shadow[name].dtype))
        return 1

    def _module_state(self, prefix):
        prefix_dot = f"{prefix}."
        state = {}
        for name in self.names:
            if not name.startswith(prefix_dot):
                continue
            state[name[len(prefix_dot):]] = self.shadow[name].detach().cpu().clone()
        return state

    def checkpoint_state(self, step):
        meta = {
            "enabled": True,
            "kind": "stage1_subset",
            "decay": self.decay,
            "device": self.device_spec,
            "dtype": str(self.dtype).replace("torch.", ""),
            "start_step": self.start_step,
            "update_every": self.update_every,
            "include": list(self.include),
            "student_da3_blocks_start": self.blocks_start,
            "student_da3_blocks_end": self.blocks_end,
            "num_tensors": self.num_tensors,
            "num_params": self.num_params,
            "step": int(step),
            "checkpoint_keys": list(self._PREFIX_TO_CKPT_KEY.values()),
        }
        state = {"ema": meta}
        for prefix, ckpt_key in self._PREFIX_TO_CKPT_KEY.items():
            module_state = self._module_state(prefix)
            if module_state:
                state[ckpt_key] = module_state
        return state

    @torch.no_grad()
    def store_and_swap(self, model):
        """Snapshot live params and swap EMA shadow in-place."""
        if self.num_tensors == 0:
            return {}
        raw_model = unwrap_train_model(model)
        params = dict(raw_model.named_parameters())
        backup = {}
        for name in self.names:
            param = params.get(name)
            if param is None:
                continue
            backup[name] = param.detach().clone()
            shadow = self.shadow[name].to(device=param.device, dtype=param.dtype)
            param.data.copy_(shadow)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        if not backup:
            return
        raw_model = unwrap_train_model(model)
        params = dict(raw_model.named_parameters())
        for name, tensor in backup.items():
            param = params.get(name)
            if param is None:
                continue
            param.data.copy_(tensor)
