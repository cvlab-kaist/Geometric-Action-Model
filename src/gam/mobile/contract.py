"""Release schema; intentionally independent of training paths and packages."""
from __future__ import annotations
import re

FORMAT = "gam-mobile-eef-v1"
AGIBOT_KEY = "agibot_eef_relative_joint16"
OXE_SUFFIX = "_eef_relative_c16_right16_v1"
CAMERAS = ("head", "hand_left", "hand_right")


def release_config(config: dict, step: int) -> dict:
    """Whitelist model fields instead of publishing the training configuration."""
    ds, pred = config["dataset"], config["predictor"]
    head, state, ft = config["action_head"], config["proprioception"], config["da3_finetune"]
    required = (
        ds.get("action_schema") == "agibot_oxe_eef_relative_right16_v1",
        ds.get("state_schema") == "left_joint7_gripper_right_joint7_gripper_v1",
        ds.get("action_frame") == "eef_relative",
        pred.get("enabled") is True,
        pred.get("type") in ("shallow12_ar", "gam"),
        pred.get("shared_padded_proprio") is True,
        pred.get("single_arm_proprio_dim", 0) == 0,
        head.get("type") == "mlp_resnet",
        head.get("n_dims") == state.get("proprio_dim") == 16,
        head.get("chunk_size") == ds.get("chunk_size") == 16,
        ft.get("n_views") == 3,
        ft.get("n_action_steps") == 5,
        not ft.get("use_temporal_embed", False),
        pred.get("H_choices") == [1, 2, 3, 4],
        pred.get("deep_temporal_causal_mask") is True,
        pred.get("language_encoder_type") == "t5",
        pred.get("t5_model") == "google-t5/t5-base",
        pred.get("use_language") is True,
        pred.get("language_dim") == 768,
        pred.get("language_len") == 77,
        not pred.get("use_proprio_head", False),
        pred.get("boundary_mode", "future_predictor") == "future_predictor",
        config["stage_1"].get("model_name", "da3-giant") == "da3-giant",
        list(ds.get("image_size", [])) == [224, 224],
        head.get("pool_mode") == "mean",
    )
    if not all(required):
        raise ValueError("Expected the shared 16D AgiBot/OXE H1..4/C16/V3 EEF checkpoint schema")
    predictor_keys = ("d_model", "depth", "num_heads", "ffn_ratio", "dropout",
                      "num_patches_per_view", "num_register_tokens", "use_language",
                      "language_dim", "language_len", "condition_mode", "input_proj_norm")
    head_keys = ("input_dim", "hidden_dim", "n_dims", "chunk_size", "num_blocks",
                 "pool_mode", "chunk_position_encoding")
    return {
        "format": FORMAT, "training_step": int(step),
        "max_history": 4, "chunk_size": 16, "image_size": [224, 224],
        "camera_order": list(CAMERAS), "agibot_action_hz": 30,
        "agibot_history_stride_frames": 16,
        "state_schema": ds["state_schema"], "action_schema": ds["action_schema"],
        "action_frame": "fixed_chunk_anchor_eef", "quaternion_order": "xyzw",
        "gripper_polarity": "0=open,1=close",
        "text_model": "google-t5/t5-base", "text_length": int(pred["language_len"]),
        "backbone": {"model_name": "da3-giant", "encoder_input_size": 224,
                     "n_action_steps": 5, "views_per_timestep": 3,
                     "action_steps_per_token": 16,
                     "use_temporal_embed": False,
                     "action_only_frame_attn": bool(ft.get("action_only_frame_attn", False))},
        "predictor": {**{k: pred[k] for k in predictor_keys}, "type": "gam",
                      "d_da3": 1536, "proprio_dim": 16, "action_dim": 16,
                      "action_chunk_size": 16, "gradient_checkpointing": False},
        "action_head": {k: head[k] for k in head_keys},
    }


def validate_config(config):
    expected = {
        "format": FORMAT,
        "max_history": 4,
        "chunk_size": 16,
        "image_size": [224, 224],
        "camera_order": list(CAMERAS),
        "agibot_action_hz": 30,
        "agibot_history_stride_frames": 16,
        "state_schema": "left_joint7_gripper_right_joint7_gripper_v1",
        "action_schema": "agibot_oxe_eef_relative_right16_v1",
        "action_frame": "fixed_chunk_anchor_eef",
        "quaternion_order": "xyzw",
        "gripper_polarity": "0=open,1=close",
        "text_model": "google-t5/t5-base",
        "text_length": 77,
    }
    sections = (
        (config, expected),
        (config.get("backbone", {}), {
            "model_name": "da3-giant", "encoder_input_size": 224,
            "n_action_steps": 5, "views_per_timestep": 3,
            "action_steps_per_token": 16,
            "use_temporal_embed": False,
        }),
        (config.get("predictor", {}), {
            "type": "gam", "d_da3": 1536, "proprio_dim": 16,
            "action_dim": 16, "action_chunk_size": 16,
            "num_patches_per_view": 256, "num_register_tokens": 0,
            "use_language": True, "language_dim": 768, "language_len": 77,
        }),
        (config.get("action_head", {}), {
            "input_dim": 1536, "n_dims": 16, "chunk_size": 16,
            "pool_mode": "mean",
        }),
    )
    for section, values in sections:
        if any(section.get(key) != value for key, value in values.items()):
            raise ValueError("Unsupported mobile GAM bundle contract")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("text_revision", "")):
        raise ValueError("The bundle must pin a 40-character T5 Hub revision")


def embodiment_for_key(key):
    if key == AGIBOT_KEY:
        return "agibot"
    if key.endswith(OXE_SUFFIX):
        return "oxe"
    raise ValueError(f"Unsupported statistics key: {key!r}")
