"""Stage 1 backbone factory.

The historical Stage 1 implementation directly instantiated
``DA3GiantEncoder``. New geometry backbones should be selected here so the
training/eval code can keep using the same gam contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .da3_giant_encoder import DA3GiantEncoder


def stage1_backbone_type(stage1_cfg: Dict[str, Any]) -> str:
    return str(stage1_cfg.get("backbone_type", stage1_cfg.get("type", "da3"))).lower()


def create_stage1_backbone(
    stage1_cfg: Dict[str, Any],
    *,
    freeze_backbone: bool,
    n_action_steps: int = 0,
    views_per_timestep: int = 2,
    action_steps_per_token: int = 1,
    use_temporal_embed: bool = False,
    action_input_rate: float = 0.4,
    action_only_frame_attn: bool = False,
    logger: Optional[Any] = None,
):
    """Instantiate the configured Stage 1 backbone.

    This public release supports the DA3-Giant backbone only.
    """
    backbone_type = stage1_backbone_type(stage1_cfg)
    if backbone_type in {"da3", "depth_anything3", "depth-anything3"}:
        model_name = str(stage1_cfg.get("model_name", "da3-giant"))
        return DA3GiantEncoder(
            ckpt_path=stage1_cfg.get("ckpt_path"),
            model_name=model_name,
            encoder_input_size=stage1_cfg.get("encoder_input_size", 224),
            normalization_stat_path=stage1_cfg.get("normalization_stat_path", None),
            freeze_backbone=freeze_backbone,
            n_action_steps=n_action_steps,
            views_per_timestep=views_per_timestep,
            action_steps_per_token=action_steps_per_token,
            use_temporal_embed=use_temporal_embed,
            action_input_rate=action_input_rate,
            action_only_frame_attn=action_only_frame_attn,
        )

    raise ValueError(
        f"Unknown stage_1.backbone_type={backbone_type!r}. "
        "This public release supports the DA3-Giant backbone only (da3)."
    )


def default_freeze_blocks_before(backbone) -> int:
    if hasattr(backbone, "default_freeze_blocks_before"):
        return int(getattr(backbone, "default_freeze_blocks_before"))
    trans = getattr(getattr(backbone, "backbone", None), "pretrained", None)
    return int(getattr(trans, "alt_start", 13))


def backbone_block_count(backbone) -> int:
    if hasattr(backbone, "block_count"):
        return int(getattr(backbone, "block_count"))
    trans = getattr(getattr(backbone, "backbone", None), "pretrained", None)
    blocks = getattr(trans, "blocks", None)
    return len(blocks) if blocks is not None else 0
