"""GLD-Robot training entrypoint."""

import argparse
import json
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
import datetime
from time import perf_counter, time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from robot.modeling.action_head_v2 import ActionHeadV2
from robot.modeling.action_head_oft import OFTL1RegressionHead
from robot.modeling.conditioning import ProprioConditioner, TextConditioner
from robot.modeling.backbone_factory import (
    create_stage1_backbone,
    default_freeze_blocks_before,
    stage1_backbone_type,
)
from robot.data.dataset import (
    ActionNormalizer,
    StateNormalizer,
    DEFAULT_ACTION_NORM_MASK,
)
from robot.evaluation.closed_loop_libero_eval import (
    all_reduce_counts as closed_loop_all_reduce_counts,
    evaluate_closed_loop_libero_from_training,
    format_wandb_log as closed_loop_format_wandb_log,
    get_cached_policy_info as closed_loop_get_cached_policy_info,
    validate_libero_env as closed_loop_validate_libero_env,
    write_eval_artifacts as closed_loop_write_eval_artifacts,
)
from robot.modeling.future_predictor import build_future_predictor
from robot.losses.reg_loss import FeatureRegularizer
from robot.modeling.sigreg import SIGReg
from robot.viz.visualization import (
    log_action_trajectory,
    log_camera_visualization,
    log_da3_visualizations,
    log_robot_debug_batch,
    log_gam_future_visualizations,
    log_training_input_images,
    log_unified_future_visualizations,
)

# Behavior-preserving helper groups extracted from this entrypoint. The main
# training loop stays here; reusable model, EMA, optimizer, eval, data, metrics,
# distributed, checkpoint, and debug helpers live under src/gam/training.
from gam.training.metrics import (
    ACTION_DIM_NAMES,
    _masked_mean,
    add_indexed_metrics,
    add_named_metrics,
    add_per_dataset_metrics,
    compute_action_detail_metrics,
    compute_action_metrics,
    compute_per_dataset_action_metrics,
    compute_raw_action_metrics,
    maybe_log_action_stats,
    reshape_action_mask,
    reshape_action_sequence,
)
from gam.training.debug import (
    _debug_check_finetune_tensors,
    _debug_log_finetune_tensor_stats,
    _debug_log_nonfinite_forward_state,
    _iter_named_floating_tensors,
    _name_matches_debug_filter,
    _parse_debug_name_filters,
    _register_nan_debug_grad_hooks,
    _summarize_named_tensor_stats,
    _summarize_nonfinite_named_tensors,
)
from gam.training.distributed import (
    _get_git_info,
    _install_training_signal_handlers,
    _plain_config_container,
    _resolve_distributed_timeout_minutes,
    _resolve_seconds_setting,
    _slurm_remaining_seconds,
    _termination_requested_across_ranks,
    setup_distributed,
    validate_deepspeed_batch_config,
)
from gam.training.data import (
    RestartableDistributedSampler,
    VirtualEpochDataset,
    _batch_source_wait_summary,
    _compact_counter,
    _expand_bool_mask_to,
    _maybe_virtualize_train_epoch,
    _pad_tensor_dim,
    _pad_view_tensor,
    _sample_view_dim,
    _set_train_sampler_position,
    collate_fn,
    create_dataset_and_loader,
)
from gam.training.checkpoint import (
    _action_timing_signature_for_dataset,
    _action_timing_signature_from_cfg,
    _action_timing_signatures_by_dataset,
    _contains_glob_metacharacters,
    _flatten_leaf_datasets,
    _format_action_timing_signature,
    _patch_deepspeed_checkpoint_glob,
    _validate_checkpoint_action_timing,
    load_feature_channel_stats,
    load_state_dict_forgiving,
    resolve_stats_dir,
    state_normalizer_dim_mismatches,
)
from gam.training.ema import (
    Stage1SubsetEMA,
    finalize_stage1_subset_ema_config as _finalize_stage1_subset_ema_config,
    stage1_subset_ema_config as _stage1_subset_ema_config,
)
from gam.training.eval_runtime import (
    closed_loop_video_dir as _closed_loop_video_dir,
    gather_eval_tensors as _gather_eval_tensors,
    gather_variable_eval_tensor as _gather_variable_eval_tensor,
    normalize_closed_loop_eval_profiles as _normalize_closed_loop_eval_profiles,
    prepare_rollout_video_frames as _prepare_rollout_video_frames,
    shutdown_train_loader_workers_for_closed_loop_eval as _shutdown_train_loader_workers_for_closed_loop_eval,
)
from gam.training.model import (
    DA3FineTuneModel,
    prepare_da3_finetune_batch,
    resolve_backbone_action_input_dim as _resolve_backbone_action_input_dim,
    resolve_da3_n_views as _resolve_da3_n_views,
    sync_da3_view_count_to_cfg as _sync_da3_view_count_to_cfg,
    unwrap_train_model as _unwrap_train_model,
    zero_invalid_context_proprio as _zero_invalid_context_proprio,
)
from gam.training.optim import build_finetune_optimizer, build_finetune_scheduler


def setup_logging(args, cfg, rank):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") if rank == 0 else ""
    if dist.is_available() and dist.is_initialized():
        timestamp_payload = [timestamp]
        dist.broadcast_object_list(timestamp_payload, src=0)
        timestamp = str(timestamp_payload[0])
    experiment_dir = os.path.join(args.results_dir, f"robot-{timestamp}")
    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)

    logger = logging.getLogger("robot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if rank == 0:
        file_handler = logging.FileHandler(os.path.join(experiment_dir, "log.txt"))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(f"Config: {args.config}")
        logger.info(f"Experiment dir: {experiment_dir}")
        logger.info(
            "Distributed process-group timeout: %.1f minutes",
            float(getattr(args, "_distributed_timeout_minutes", 120.0)),
        )

    wandb_run = None
    if rank == 0 and args.wandb:
        import wandb

        run_name = args.wandb_name if args.wandb_name else f"robot-{timestamp}"
        # Resume wandb run if checkpoint contains run id (unless --wandb-new-run)
        wandb_resume_id = None
        if args.ckpt is not None and not getattr(args, "wandb_new_run", False):
            ckpt_meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            wandb_resume_id = ckpt_meta.get("wandb_run_id", None)
            del ckpt_meta
        wandb_cfg = cfg.get("wandb", {}) or {}
        predictor_enabled = bool(cfg.get("predictor", {}).get("enabled", False))
        default_project = "robot-gld" if predictor_enabled else "da3-action-finetune"
        wandb_project = (
            getattr(args, "wandb_project", None)
            or os.environ.get("WANDB_PROJECT")
            or wandb_cfg.get("project", None)
            or default_project
        )
        wandb_kwargs = dict(
            project=wandb_project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        forced_wandb_run_id = os.environ.get("DA3_WANDB_RESUME_ID") or os.environ.get("ROBOT_WANDB_RESUME_ID")
        forced_wandb_resume_mode = os.environ.get("DA3_WANDB_RESUME_MODE", "must")
        wandb_resume_from = getattr(args, "wandb_resume_from", None) or os.environ.get("WANDB_RESUME_FROM")
        if wandb_resume_from:
            wandb_kwargs["resume_from"] = wandb_resume_from
        elif forced_wandb_run_id and not getattr(args, "wandb_new_run", False):
            wandb_kwargs["id"] = forced_wandb_run_id
            wandb_kwargs["resume"] = forced_wandb_resume_mode
        elif wandb_resume_id:
            wandb_kwargs["id"] = wandb_resume_id
            wandb_kwargs["resume"] = "must"
        wandb_run = wandb.init(**wandb_kwargs)

        git_info = _get_git_info()
        # allow_val_change=True: on resume, these fields already exist in wandb
        # config from the first launch and would otherwise raise ConfigError.
        wandb.config.update({
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "git_dirty": git_info["dirty"],
            "intent": getattr(args, "intent", ""),
        }, allow_val_change=True)

    # Save config snapshot for reproducibility
    if rank == 0:
        OmegaConf.save(cfg, os.path.join(experiment_dir, "config.yaml"))

    return experiment_dir, logger, wandb_run


def run_da3_finetune_training(args, cfg):
    use_deepspeed, rank, world_size, device = setup_distributed(args, cfg)

    stage1_cfg = OmegaConf.to_container(cfg.get("stage_1", {}), resolve=True)
    training_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True)
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    da3_ft_cfg = OmegaConf.to_container(cfg.get("da3_finetune", {}), resolve=True)
    action_head_cfg = OmegaConf.to_container(cfg.get("action_head", {}), resolve=True)
    reg_cfg = OmegaConf.to_container(cfg.get("regularization", {}), resolve=True)
    proprio_cfg = OmegaConf.to_container(cfg.get("proprioception", {}), resolve=True)
    if "predictor" in cfg:
        predictor_cfg = OmegaConf.to_container(cfg.predictor, resolve=True)
    else:
        predictor_cfg = {}
    predictor_enabled = bool(predictor_cfg.get("enabled", False))

    if bool(cfg.get("use_dit", False)):
        raise ValueError("DA3 fine-tune mode currently supports action-only training (`use_dit: false`).")

    seed = int(training_cfg.get("global_seed", 42))
    torch.manual_seed(seed * world_size + rank)
    np.random.seed(seed * world_size + rank)
    random.seed(seed * world_size + rank)

    num_epochs = int(training_cfg.get("epochs", 500000))
    log_every = int(training_cfg.get("log_every", 50))
    ckpt_every = int(training_cfg.get("ckpt_every", 2000))
    scheduled_checkpointing = ckpt_every > 0
    vis_every = int(training_cfg.get("vis_every", ckpt_every))
    visualization_cfg = training_cfg.get("visualization", {})
    if not isinstance(visualization_cfg, dict):
        visualization_cfg = {}
    log_unified_depth_visualization = bool(
        visualization_cfg.get(
            "log_unified_depth",
            training_cfg.get("log_unified_depth", True),
        )
    )
    unified_depth_vis_every = int(
        visualization_cfg.get(
            "unified_depth_every",
            training_cfg.get("unified_depth_vis_every", vis_every),
        )
    )
    base_lr = float(training_cfg.get("base_lr", 3e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    lambda_action = float(training_cfg.get("lambda_action", 1.0))
    lambda_action_direct = float(training_cfg.get("lambda_action_direct", 1.0))
    lambda_action_refine = float(training_cfg.get("lambda_action_refine", 1.0))
    ema_cfg = _stage1_subset_ema_config(training_cfg)

    include_current_action = bool(dataset_cfg.get("include_current_action", True))
    future_steps = int(dataset_cfg.get("future_steps", 6))
    action_steps = future_steps + 1 if include_current_action else future_steps
    dataset_chunk_size = int(dataset_cfg.get("chunk_size", 1))
    aggregate_chunk_actions = bool(dataset_cfg.get("aggregate_chunk_actions", False))
    chunk_size = int(action_head_cfg.get("chunk_size", dataset_chunk_size))
    expected_head_chunk = 1 if aggregate_chunk_actions else dataset_chunk_size
    if chunk_size != expected_head_chunk:
        raise ValueError(
            f"Chunk size mismatch: dataset.chunk_size={dataset_chunk_size}, "
            f"dataset.aggregate_chunk_actions={aggregate_chunk_actions}, "
            f"expected action_head.chunk_size={expected_head_chunk}, got {chunk_size}"
        )
    n_views = _resolve_da3_n_views(dataset_cfg, da3_ft_cfg)
    _sync_da3_view_count_to_cfg(cfg, dataset_cfg, da3_ft_cfg, n_views)
    total_view = action_steps * n_views
    n_action_steps = int(da3_ft_cfg.get("n_action_steps", action_steps))
    if n_action_steps != action_steps:
        raise ValueError(
            f"Config mismatch: n_action_steps={n_action_steps}, expected {action_steps} from dataset."
        )

    use_bf16 = bool(training_cfg.get("bf16", True))
    use_compile = bool(training_cfg.get("compile", False))
    nan_debug = bool(int(os.environ.get("DA3_NAN_DEBUG", "0"))) or bool(
        training_cfg.get("nan_debug", False)
    )
    nan_debug_start_step = int(
        os.environ.get("DA3_NAN_DEBUG_START_STEP", training_cfg.get("nan_debug_start_step", 0))
    )
    nan_debug_filters = _parse_debug_name_filters()
    nan_debug_log_stats = bool(int(os.environ.get("DA3_NAN_DEBUG_LOG_STATS", "0"))) or bool(
        training_cfg.get("nan_debug_log_stats", False)
    )
    nan_debug_grad_hooks = bool(int(os.environ.get("DA3_NAN_DEBUG_GRAD_HOOKS", "0"))) or bool(
        training_cfg.get("nan_debug_grad_hooks", False)
    )

    experiment_dir, logger, wandb_run = setup_logging(args, cfg, rank)
    if rank == 0 and not scheduled_checkpointing:
        logger.info("Scheduled checkpointing disabled: training.ckpt_every=%d", ckpt_every)
    _install_training_signal_handlers(logger=logger, rank=rank)
    preemption_min_remaining_sec = _resolve_seconds_setting(
        training_cfg,
        "slurm_min_remaining_sec",
        "DA3_MIN_REMAINING_SEC",
        600.0,
    )
    closed_loop_min_remaining_sec = _resolve_seconds_setting(
        training_cfg,
        "closed_loop_eval_min_remaining_sec",
        "DA3_CLOSED_LOOP_EVAL_MIN_REMAINING_SEC",
        1800.0,
    )
    termination_check_every = int(
        os.environ.get(
            "DA3_PREEMPTION_CHECK_EVERY_STEPS",
            training_cfg.get("preemption_check_every_steps", 0),
        )
    )
    if termination_check_every < 0:
        raise ValueError(
            "training.preemption_check_every_steps must be >= 0; "
            "use 0 to disable periodic checks"
        )
    if rank == 0:
        _remaining = _slurm_remaining_seconds()
        if _remaining is None:
            logger.info(
                "Slurm walltime guard: no job time limit detected "
                "(preemption_guard=%.0fs closed_loop_guard=%.0fs)",
                preemption_min_remaining_sec,
                closed_loop_min_remaining_sec,
            )
        else:
            logger.info(
                "Slurm walltime guard: remaining=%.0fs preemption_guard=%.0fs "
                "closed_loop_guard=%.0fs",
                _remaining,
                preemption_min_remaining_sec,
                closed_loop_min_remaining_sec,
            )
        if termination_check_every > 0:
            logger.info(
                "Distributed preemption signal check interval: every %d step(s)",
                termination_check_every,
            )
        else:
            logger.info(
                "Distributed periodic preemption signal check disabled; "
                "checkpoint/eval boundary checks remain enabled"
            )
    if use_deepspeed:
        validate_deepspeed_batch_config(args, training_cfg, world_size, logger=logger if rank == 0 else None)

    if rank == 0:
        logger.info("bf16=%s, compile=%s, world_size=%d", use_bf16, use_compile, world_size)
        logger.info(
            "DA3 fine-tune views: n_views=%d camera_keys=%s rollout_camera_keys=%s",
            n_views,
            dataset_cfg.get("camera_keys"),
            dataset_cfg.get("rollout_camera_keys"),
        )
        if nan_debug:
            logger.info(
                "DA3_NAN_DEBUG enabled: start_step=%d filters=%s log_stats=%s grad_hooks=%s",
                nan_debug_start_step,
                nan_debug_filters or ("<all trainable params>",),
                nan_debug_log_stats,
                nan_debug_grad_hooks,
            )

    backbone_type = stage1_backbone_type(stage1_cfg)
    if backbone_type != "da3" and not predictor_enabled:
        raise ValueError(
            "Non-DA3 Stage 1 backbones currently require predictor.enabled=true "
            "because the legacy encode_with_actions path is DA3-specific."
        )
    teacher_stage1_cfg = deepcopy(stage1_cfg)
    teacher_override_cfg = teacher_stage1_cfg.get("teacher", {}) or {}
    if not isinstance(teacher_override_cfg, dict):
        raise TypeError("stage_1.teacher must be a mapping when provided.")
    teacher_ckpt_path = teacher_override_cfg.get(
        "ckpt_path",
        teacher_stage1_cfg.get("teacher_ckpt_path"),
    )
    if teacher_ckpt_path:
        teacher_stage1_cfg["ckpt_path"] = teacher_ckpt_path
    if rank == 0:
        logger.info(
            "Stage 1 teacher checkpoint: %s (student/bootstrap checkpoint: %s)",
            teacher_stage1_cfg.get("ckpt_path"),
            stage1_cfg.get("ckpt_path"),
        )
        if (
            teacher_stage1_cfg.get("ckpt_path")
            and stage1_cfg.get("ckpt_path")
            and os.path.abspath(str(teacher_stage1_cfg.get("ckpt_path")))
            == os.path.abspath(str(stage1_cfg.get("ckpt_path")))
        ):
            logger.warning(
                "Teacher and student/bootstrap Stage 1 checkpoints are identical; "
                "set stage_1.teacher_ckpt_path to keep the depth teacher pristine."
            )
    teacher_da3 = create_stage1_backbone(
        teacher_stage1_cfg,
        freeze_backbone=True,
        logger=logger if rank == 0 else None,
    ).to(device)
    teacher_da3.eval()
    for p in teacher_da3.parameters():
        p.requires_grad = False
    if use_bf16:
        teacher_da3 = teacher_da3.to(torch.bfloat16)

    use_temporal_embed = bool(da3_ft_cfg.get("use_temporal_embed", False))
    action_only_frame_attn = bool(da3_ft_cfg.get("action_only_frame_attn", False))
    student_da3 = create_stage1_backbone(
        stage1_cfg,
        freeze_backbone=False,
        n_action_steps=n_action_steps,
        views_per_timestep=n_views,
        action_steps_per_token=chunk_size,
        use_temporal_embed=use_temporal_embed,
        action_input_rate=float(da3_ft_cfg.get("action_input_rate", 0.4)),
        action_only_frame_attn=action_only_frame_attn,
        logger=logger if rank == 0 else None,
    ).to(device)
    default_freeze_before = default_freeze_blocks_before(student_da3)
    student_da3.freeze_blocks_before(int(da3_ft_cfg.get("freeze_blocks_before", default_freeze_before)))
    ema_cfg = _finalize_stage1_subset_ema_config(ema_cfg, student_da3)
    if rank == 0:
        logger.info(
            "Stage 1 backbone: type=%s embed_dim=%d patches/view=%d registers=%d freeze_before=%d",
            backbone_type,
            int(student_da3.embed_dim),
            int(getattr(student_da3, "num_patches", -1)),
            int(getattr(student_da3, "num_register_tokens", 0)),
            int(da3_ft_cfg.get("freeze_blocks_before", default_freeze_before)),
        )
    if rank == 0 and action_only_frame_attn:
        logger.info(
            "VGA action_only_frame_attn=True: action tokens use within-modality "
            "local attention across all views; global attention unchanged."
        )

    proprio_conditioner = None
    if bool(proprio_cfg.get("enabled", True)) and not predictor_enabled:
        proprio_conditioner = ProprioConditioner(
            proprio_dim=int(proprio_cfg.get("proprio_dim", 9)),
            hidden_dim=int(proprio_cfg.get("hidden_dim", 256)),
            out_dim=student_da3.embed_dim,
        ).to(device)

    head_type = str(action_head_cfg.get("type", "mlp_resnet")).lower()
    action_head_input_dim = _resolve_backbone_action_input_dim(
        action_head_cfg,
        student_da3.embed_dim,
        logger=logger if rank == 0 else None,
    )
    head_kwargs = dict(
        input_dim=action_head_input_dim,
        n_views=n_views,
        hidden_dim=int(action_head_cfg.get("hidden_dim", student_da3.embed_dim)),
        n_dims=int(action_head_cfg.get("n_dims", 7)),
        chunk_size=chunk_size,
        num_blocks=int(action_head_cfg.get("num_blocks", 2)),
        pool_mode=str(action_head_cfg.get("pool_mode", "mean")),
    )
    chunk_position_encoding = str(action_head_cfg.get("chunk_position_encoding", "none"))
    if head_type in ("mlp_resnet", "mlp-resnet", "v2", "action_head_v2"):
        action_head = ActionHeadV2(
            **head_kwargs,
            chunk_position_encoding=chunk_position_encoding,
        ).to(device)
        if rank == 0:
            logger.info(
                "ActionHeadV2: chunk_size=%d chunk_position_encoding=%s",
                chunk_size,
                chunk_position_encoding,
            )
    elif head_type in ("oft", "openvla_oft", "openvla-oft", "l1_regression"):
        if chunk_position_encoding.lower() != "none":
            raise ValueError("action_head.chunk_position_encoding is only supported by ActionHeadV2.")
        action_head = OFTL1RegressionHead(**head_kwargs).to(device)
        if rank == 0:
            logger.info(
                "VGA action_head=OFTL1RegressionHead (openvla-oft port): "
                "MLPResNet depth=%d, hidden=%d, pool=%s",
                head_kwargs["num_blocks"], head_kwargs["hidden_dim"], head_kwargs["pool_mode"],
            )
    else:
        raise ValueError(
            f"Unknown action_head.type={head_type!r}. Use 'mlp_resnet' or 'oft'."
        )

    # -------- Optional: FuturePredictor (unified Stage 1 + 2) --------
    future_predictor = None
    text_conditioner = None
    proprio_head = None
    sigreg_module = None
    lambda_feat_future = float(predictor_cfg.get("lambda_feat_future", 1.0))
    lambda_feat_current = float(predictor_cfg.get("lambda_feat_current", 0.0))
    lambda_proprio_future = float(predictor_cfg.get("lambda_proprio_future", 0.0))
    lambda_sigreg = float(predictor_cfg.get("lambda_sigreg", 0.0))
    unified_H_choices = list(predictor_cfg.get("H_choices", [3]))
    unified_H_weights = predictor_cfg.get("H_weights", None)
    unified_num_register_tokens = int(
        predictor_cfg.get("num_register_tokens", student_da3.num_register_tokens)
    )
    gam_compat_gradient_checkpointing = bool(
        predictor_cfg.get(
            "deep_gradient_checkpointing",
            predictor_cfg.get("gradient_checkpointing", False),
        )
    )
    gam_predictor_gradient_checkpointing = bool(
        predictor_cfg.get(
            "predictor_gradient_checkpointing",
            gam_compat_gradient_checkpointing,
        )
    )
    gam_backbone_deep_gradient_checkpointing = bool(
        predictor_cfg.get(
            "backbone_deep_gradient_checkpointing",
            gam_compat_gradient_checkpointing,
        )
    )
    # Strict temporal-block-causal mask on the DA3 deep stack's global
    # attention. Default OFF so existing checkpoints remain bit-identical.
    # Enable explicitly for a strictly causal policy that prevents the deep stack
    # from mixing predicted future obs back into earlier action tokens. See
    # _propagate_shallow_with_actions_impl.
    gam_deep_temporal_causal_mask = bool(
        predictor_cfg.get("deep_temporal_causal_mask", False)
    )
    gam_use_proprio_head = bool(predictor_cfg.get("use_proprio_head", lambda_proprio_future > 0.0))
    feature_loss_norm = str(predictor_cfg.get("feature_loss_norm", "none")).lower()
    feature_loss_type = str(predictor_cfg.get("feature_loss_type", "l2")).lower()
    feature_channel_stats_path = predictor_cfg.get("feature_channel_stats_path", None)
    feature_loss_norm_eps = float(predictor_cfg.get("feature_loss_norm_eps", 1e-6))
    feature_target_mode = str(predictor_cfg.get("feature_target_mode", "future")).lower()
    feature_channel_stats = None
    predictor_type = str(predictor_cfg.get("type", predictor_cfg.get("architecture", "gam"))).lower()
    use_language = bool(predictor_cfg.get("use_language", True))
    if predictor_enabled:
        if feature_loss_norm in {"channel", "channel_stats", "stats"}:
            feature_channel_stats = load_feature_channel_stats(
                feature_channel_stats_path,
                device=device,
                expected_dim=student_da3.embed_dim,
                eps=feature_loss_norm_eps,
                logger=logger if rank == 0 else None,
            )
            loaded_target_mode = str(feature_channel_stats.get("feature_target_mode", "future")).lower()
            if loaded_target_mode != feature_target_mode:
                raise ValueError(
                    "predictor.feature_target_mode="
                    f"{feature_target_mode!r} but feature stats were computed for {loaded_target_mode!r}."
                )
        elif feature_loss_norm not in {"", "none", "raw"}:
            raise ValueError(f"Unknown predictor.feature_loss_norm={feature_loss_norm!r}.")
        if feature_loss_type not in {"l2", "mse", "l1", "mae"}:
            raise ValueError(f"Unknown predictor.feature_loss_type={feature_loss_type!r}.")
        if feature_target_mode not in {"future", "delta"}:
            raise ValueError(f"Unknown predictor.feature_target_mode={feature_target_mode!r}.")
        if rank == 0:
            logger.info(
                "Unified mode ON: FuturePredictor type=%s, H_choices=%s, "
                "lambda_feat_future=%.4f, lambda_feat_current=%.4f, "
                "lambda_proprio_future=%.4f, lambda_sigreg=%.4f, "
                "lambda_action_direct=%.4f, lambda_action_refine=%.4f, "
                "use_language=%s, deep_gradient_checkpointing=%s, "
                "predictor_gradient_checkpointing=%s, "
                "backbone_deep_gradient_checkpointing=%s, "
                "deep_temporal_causal_mask=%s, "
                "use_proprio_head=%s, feature_loss_norm=%s, feature_loss_type=%s, feature_target_mode=%s",
                predictor_type, unified_H_choices, lambda_feat_future,
                lambda_feat_current, lambda_proprio_future, lambda_sigreg,
                lambda_action_direct, lambda_action_refine, use_language,
                gam_compat_gradient_checkpointing,
                gam_predictor_gradient_checkpointing,
                gam_backbone_deep_gradient_checkpointing,
                gam_deep_temporal_causal_mask,
                gam_use_proprio_head,
                feature_loss_norm,
                feature_loss_type,
                feature_target_mode,
            )
        # SIGReg is optional. With a frozen DA3 future-feature target, the main
        # anti-collapse pressure is the supervised feature L2 itself.
        sigreg_proj_dim = int(predictor_cfg.get("sigreg_proj_dim", 256))
        if lambda_sigreg > 0.0:
            sigreg_module = SIGReg(
                d_model=sigreg_proj_dim,
                n_projections=int(predictor_cfg.get("sigreg_n_projections", 1024)),
                knots=int(predictor_cfg.get("sigreg_knots", 17)),
                max_knot=float(predictor_cfg.get("sigreg_max_knot", 3.0)),
                redraw=bool(predictor_cfg.get("sigreg_redraw", True)),
            )
        future_predictor = build_future_predictor(
            cfg={
                "type": predictor_type,
                "d_da3": student_da3.embed_dim,
                "d_model": int(predictor_cfg.get("d_model", 1024)),
                "depth": int(predictor_cfg.get("depth", 12)),
                "num_heads": int(predictor_cfg.get("num_heads", 16)),
                "ffn_ratio": float(predictor_cfg.get("ffn_ratio", 4.0)),
                "dropout": float(predictor_cfg.get("dropout", 0.0)),
                "num_patches_per_view": int(predictor_cfg.get(
                    "num_patches_per_view",
                    int(getattr(student_da3, "num_patches", 0))
                    or (stage1_cfg.get("encoder_input_size", 224) // int(getattr(student_da3, "patch_size", 14))) ** 2,
                )),
                "use_language": use_language,
                "language_dim": int(predictor_cfg.get("language_dim", 768)),
                "language_len": int(predictor_cfg.get("language_len", 77)),
                "proprio_dim": int(proprio_cfg.get("proprio_dim", 7)),
                "action_dim": int(action_head_cfg.get("n_dims", 7)),
                "action_chunk_size": chunk_size,
                "num_register_tokens": unified_num_register_tokens,
                "sigreg_proj_dim": sigreg_proj_dim,
                "sigreg_pool_mode": str(predictor_cfg.get("sigreg_pool_mode", "cls")),
                "condition_mode": str(predictor_cfg.get("condition_mode", "cross_attn")),
                "input_proj_norm": str(predictor_cfg.get("input_proj_norm", "ln")),
                "gradient_checkpointing": gam_predictor_gradient_checkpointing,
            },
            sigreg=sigreg_module,
        ).to(device)
        if gam_use_proprio_head:
            proprio_head = ActionHeadV2(
                input_dim=action_head_input_dim,
                n_views=n_views,
                hidden_dim=int(action_head_cfg.get("hidden_dim", student_da3.embed_dim)),
                n_dims=int(proprio_cfg.get("proprio_dim", 7)),
                chunk_size=1,
                num_blocks=int(action_head_cfg.get("num_blocks", 2)),
                pool_mode=str(action_head_cfg.get("pool_mode", "mean")),
                chunk_position_encoding="none",
            ).to(device)
        if use_language:
            encoder_type = str(predictor_cfg.get("language_encoder_type", "clip"))
            text_conditioner = TextConditioner(
                encoder_type=encoder_type,
                clip_model=str(predictor_cfg.get("clip_model", "openai/clip-vit-large-patch14")),
                t5_model=str(predictor_cfg.get("t5_model", "google-t5/t5-base")),
                proj_dim=int(predictor_cfg.get("language_dim", 768)),
                cache_token_embeddings=bool(
                    predictor_cfg.get(
                        "cache_token_embeddings",
                        predictor_cfg.get("language_cache", False),
                    )
                ),
                cache_max_entries=int(
                    predictor_cfg.get(
                        "language_cache_max_entries",
                        predictor_cfg.get("cache_max_entries", 0),
                    )
                ),
                cache_device=str(predictor_cfg.get("language_cache_device", "cpu")),
            ).to(device)
            cfg_lang_dim = int(predictor_cfg.get("language_dim", 768))
            if cfg_lang_dim != text_conditioner.hidden_size:
                raise ValueError(
                    f"predictor.language_dim={cfg_lang_dim} mismatches "
                    f"TextConditioner({encoder_type=}).hidden_size="
                    f"{text_conditioner.hidden_size}. Set predictor.language_dim "
                    "to the encoder's native hidden size (CLIP-L: 768, "
                    "T5-base: 768, T5-large: 1024, T5-XXL: 4096)."
                )
            if rank == 0 and text_conditioner.cache_token_embeddings:
                logger.info(
                    "Language token cache enabled: max_entries=%d device=%s encoder=%s",
                    text_conditioner.cache_max_entries,
                    text_conditioner.cache_device,
                    encoder_type,
                )
        if rank == 0:
            n_pred_params = sum(p.numel() for p in future_predictor.parameters() if p.requires_grad)
            logger.info("FuturePredictor trainable params: %.1fM", n_pred_params / 1e6)

    finetune_model = DA3FineTuneModel(
        student_da3=student_da3,
        action_head=action_head,
        proprio_head=proprio_head,
        proprio_conditioner=proprio_conditioner,
        future_predictor=future_predictor,
        text_conditioner=text_conditioner,
    )
    regularizer = FeatureRegularizer(
        lambda_feat=float(reg_cfg.get("lambda_feat", 0.01)),
        layer_weight_min=float(reg_cfg.get("layer_weight_min", 0.25)),
        lambda_depth=float(reg_cfg.get("lambda_depth", 0.0)),
        lambda_camera=float(reg_cfg.get("lambda_camera", 0.0)),
        depth_loss_type=str(reg_cfg.get("depth_loss_type", "da3_style")),
        lambda_depth_conf=float(reg_cfg.get("lambda_depth_conf", 0.2)),
        lambda_ray=float(reg_cfg.get("lambda_ray", 0.0)),
        lambda_point=float(reg_cfg.get("lambda_point", 0.0)),
        lambda_camera_pose=float(reg_cfg.get("lambda_camera_pose", 0.0)),
    )
    depth_grad_weight = float(reg_cfg.get("depth_grad_weight", 1.0))
    depth_decode_chunk_size = int(reg_cfg.get("depth_decode_chunk_size", 1))
    depth_future_steps = int(reg_cfg.get("depth_future_steps", 0))
    depth_decode_context = str(reg_cfg.get("depth_decode_context", "full_sequence"))

    total_trainable = sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info("Student DA3 + head + predictor trainable params: %.1fM", total_trainable / 1e6)
        if predictor_enabled and regularizer.lambda_depth > 0:
            logger.info(
                "Unified DA3 GT depth loss enabled: type=%s lambda_depth=%.4f "
                "depth_grad_weight=%.4f lambda_depth_conf=%.4f lambda_ray=%.4f "
                "lambda_point=%.4f lambda_camera_pose=%.4f",
                regularizer.depth_loss_type,
                regularizer.lambda_depth,
                depth_grad_weight,
                regularizer.lambda_depth_conf,
                regularizer.lambda_ray,
                regularizer.lambda_point,
                regularizer.lambda_camera_pose,
            )
            logger.info("Unified GT depth decode chunk size: %d", depth_decode_chunk_size)
            logger.info(
                "Unified GT depth decode context: %s future_steps=%d",
                depth_decode_context,
                depth_future_steps,
            )

    # --- torch.compile ---
    # DA3_TRAIN_COMPILE_MODE selects the inductor mode (default unchanged).
    # "reduce-overhead" / "max-autotune" enable CUDA graphs (fwd+bwd) for
    # training; the RoPE/graph-stability work that makes this safe is shared
    # with the inference fast path. Default is the historical mode=default.
    if use_compile:
        _train_compile_mode = os.environ.get("DA3_TRAIN_COMPILE_MODE", "default").strip().lower()
        if rank == 0:
            logger.info("Compiling model with torch.compile (mode=%s)...", _train_compile_mode)
        if _train_compile_mode in ("", "default"):
            finetune_model = torch.compile(finetune_model)
        else:
            finetune_model = torch.compile(finetune_model, mode=_train_compile_mode)

    max_steps = int(training_cfg.get("max_steps", 100000))

    if use_deepspeed:
        import deepspeed

        opt, base_lr, head_lr_mult, predictor_lr_mult, adam_eps = build_finetune_optimizer(
            finetune_model, training_cfg
        )
        scheduler = build_finetune_scheduler(
            opt,
            training_cfg,
            max_steps,
            logger=logger if rank == 0 else None,
        )
        model_engine, opt, _, _ = deepspeed.initialize(
            model=finetune_model,
            optimizer=opt,
            lr_scheduler=scheduler,
            config=args.deepspeed_config,
            model_parameters=[p for p in finetune_model.parameters() if p.requires_grad],
        )
        device = model_engine.device
        if rank == 0:
            _b1 = float(training_cfg.get("adam_beta1", 0.9))
            _b2 = float(training_cfg.get("adam_beta2", 0.999))
            logger.info(
                "DeepSpeed optimizer: backbone_lr=%.1e, head_lr=%.1e (%.0fx), "
                "predictor_lr=%.1e (%.2gx), wd=%.4f, adam_eps=%.1e, betas=(%.4f, %.4f)",
                base_lr, base_lr * head_lr_mult, head_lr_mult,
                base_lr * predictor_lr_mult, predictor_lr_mult, weight_decay, adam_eps, _b1, _b2,
            )
    else:
        finetune_model = finetune_model.to(device)
        # --- DDP for multi-GPU ---
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            finetune_model = DDP(
                finetune_model, device_ids=[device.index],
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            if rank == 0:
                logger.info("Wrapped model in DDP (world_size=%d)", world_size)

        # --- Param groups: backbone vs head ---
        opt, base_lr, head_lr_mult, predictor_lr_mult, adam_eps = build_finetune_optimizer(
            finetune_model, training_cfg
        )
        if rank == 0:
            _b1 = float(training_cfg.get("adam_beta1", 0.9))
            _b2 = float(training_cfg.get("adam_beta2", 0.999))
            logger.info(
                "Optimizer: backbone_lr=%.1e, head_lr=%.1e (%.0fx), "
                "predictor_lr=%.1e (%.2gx), wd=%.4f, adam_eps=%.1e, betas=(%.4f, %.4f)",
                base_lr, base_lr * head_lr_mult, head_lr_mult,
                base_lr * predictor_lr_mult, predictor_lr_mult, weight_decay, adam_eps, _b1, _b2,
            )

        # --- Cosine schedule with linear warmup ---
        scheduler = build_finetune_scheduler(
            opt,
            training_cfg,
            max_steps,
            logger=logger if rank == 0 else None,
        )

        model_engine = None

    nan_debug_step_context = {"step": 0}
    nan_debug_hook_handles = []
    if nan_debug and nan_debug_grad_hooks:
        hook_model = model_engine.module if model_engine is not None else finetune_model
        hook_model = hook_model.module if hasattr(hook_model, "module") else hook_model
        if hasattr(hook_model, "_orig_mod"):
            hook_model = hook_model._orig_mod
        nan_debug_hook_handles = _register_nan_debug_grad_hooks(
            hook_model,
            rank=rank,
            logger=logger,
            filters=nan_debug_filters,
            step_context=nan_debug_step_context,
            start_step=nan_debug_start_step,
            log_stats=nan_debug_log_stats,
        )
        if rank == 0:
            logger.info("DA3_NAN_DEBUG registered %d raw-gradient hooks", len(nan_debug_hook_handles))

    # --- GradScaler: fp16 path only; bf16 skips it ---
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # --- Resume from checkpoint ---
    start_epoch = 0
    train_steps = 0
    # Gradient accumulation counter for the DDP path. Counts micro-batches
    # processed since the last optimizer.step. Resets to 0 after each
    # optimizer.step so that `(micro_idx_in_accum + 1) % grad_accum_steps == 0`
    # cleanly identifies the accum boundary. The DeepSpeed path ignores this
    # because deepspeed handles accum internally.
    micro_idx_in_accum = 0
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    ddp_accum_no_sync = bool(
        (not use_deepspeed)
        and world_size > 1
        and grad_accum_steps > 1
    )
    if rank == 0:
        logger.info(
            "DDP gradient accumulation: grad_accum_steps=%d no_sync_non_boundary=%s",
            grad_accum_steps,
            ddp_accum_no_sync,
        )
    resume_ckpt_for_ema = None
    if args.ckpt is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        resume_ckpt_for_ema = ckpt
        raw_model = finetune_model.module if hasattr(finetune_model, "module") else finetune_model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        load_infos = []
        load_info = load_state_dict_forgiving(
            raw_model.student_da3, ckpt["student_da3"], logger=logger, module_name="student_da3"
        )
        load_infos.append(load_info)
        _strip = lambda sd: {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        # Use forgiving load: ActionHeadV2 ↔ OFTL1RegressionHead have different
        # parameter names, so a cross-type resume should skip mismatched keys
        # and warn rather than crash. Same-type resume still loads strictly.
        load_infos.append(load_state_dict_forgiving(
            raw_model.action_head, _strip(ckpt["action_head"]),
            logger=logger, module_name="action_head",
            allow_action_head_chunk_resize=True,
        ))
        if ckpt.get("proprio_head") is not None and raw_model.proprio_head is not None:
            load_state_dict_forgiving(
                raw_model.proprio_head,
                _strip(ckpt["proprio_head"]),
                logger=logger,
                module_name="proprio_head",
            )
        elif ckpt.get("proprio_head") is not None and raw_model.proprio_head is None:
            if rank == 0:
                logger.info("Skipping ckpt proprio_head: use_proprio_head=false in current config.")
        skipped_proprio_conditioner = False
        if ckpt.get("proprio_conditioner") is not None:
            if raw_model.proprio_conditioner is None:
                skipped_proprio_conditioner = True
                if rank == 0:
                    logger.info(
                        "Skipping checkpoint proprio_conditioner: DA3 fine-tune no longer uses proprio conditioning."
                    )
            else:
                try:
                    raw_model.proprio_conditioner.load_state_dict(_strip(ckpt["proprio_conditioner"]))
                except RuntimeError as exc:
                    skipped_proprio_conditioner = True
                    if rank == 0:
                        logger.info(
                            "Skipping incompatible unused proprio_conditioner state during resume: %s",
                            exc,
                        )
        # Unified future-predictor state (optional; legacy ckpts don't carry these).
        if ckpt.get("future_predictor") is not None and raw_model.future_predictor is not None:
            load_infos.append(load_state_dict_forgiving(
                raw_model.future_predictor,
                _strip(ckpt["future_predictor"]),
                logger=logger,
                module_name="future_predictor",
            ))
        elif ckpt.get("future_predictor") is not None and raw_model.future_predictor is None:
            if rank == 0:
                logger.info("Skipping ckpt future_predictor: predictor.enabled=false in current config.")
        if ckpt.get("text_conditioner_proj") is not None and raw_model.text_conditioner is not None:
            try:
                raw_model.text_conditioner.proj.load_state_dict(_strip(ckpt["text_conditioner_proj"]))
            except RuntimeError as exc:
                raise RuntimeError(
                    "Checkpoint text_conditioner_proj is incompatible with the current "
                    "language-conditioned predictor config."
                ) from exc
        elif (
            raw_model.text_conditioner is not None
            and raw_model.future_predictor is not None
            and ckpt.get("future_predictor") is not None
        ):
            raise RuntimeError(
                "Checkpoint has future_predictor weights but no text_conditioner_proj. "
                "Refusing to resume with a randomly initialized language projection."
            )
        has_model_change = (
            any("shape " in reason or reason == "missing" for info in load_infos for _, reason in info["skipped"])
            or any(len(info.get("partial", [])) > 0 for info in load_infos)
            or any(len(info.get("missing_keys", [])) > 0 for info in load_infos)
            or any(len(info.get("unexpected_keys", [])) > 0 for info in load_infos)
        )
        reset_schedule = getattr(args, "reset_schedule", False)
        reset_optimizer_state = getattr(args, "reset_optimizer_state", False)
        if reset_schedule:
            if rank == 0:
                logger.info("--reset-schedule: skipping optimizer/scheduler/scaler state load (LR warm restart)")
        elif reset_optimizer_state:
            if rank == 0:
                logger.info(
                    "--reset-optimizer-state: skipping optimizer/scheduler/scaler state load "
                    "while preserving checkpoint train_steps/epoch"
                )
        else:
            # DeepSpeed ZeRO partitions optimizer state across ranks: each rank
            # owns its shard, so the rank-0-only .pt save is incomplete for
            # opt.load_state_dict on other ranks (KeyError: <rank>).
            # Prefer ds load_checkpoint off the sharded directory written by
            # model_engine.save_checkpoint. Fall back to fresh optimizer when
            # the directory is missing (pre-fix checkpoints).
            ds_ckpt_loaded = False
            if use_deepspeed and model_engine is not None:
                ds_ckpt_dir = os.path.dirname(args.ckpt)
                ds_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
                ds_tag_dir = os.path.join(ds_ckpt_dir, ds_tag)
                if os.path.isdir(ds_tag_dir) and not has_model_change:
                    try:
                        if _contains_glob_metacharacters(ds_ckpt_dir) or _contains_glob_metacharacters(ds_tag):
                            _patch_deepspeed_checkpoint_glob(model_engine, logger=logger, rank=rank)
                        load_path, _ = model_engine.load_checkpoint(
                            ds_ckpt_dir,
                            tag=ds_tag,
                            load_module_strict=False,
                        )
                        if load_path is None:
                            raise RuntimeError(f"DeepSpeed returned no checkpoint path for tag={ds_tag}")
                        ds_ckpt_loaded = True
                        if rank == 0:
                            logger.info("Restored DeepSpeed optimizer state from %s (tag=%s)", ds_ckpt_dir, ds_tag)
                    except Exception as e:
                        if rank == 0:
                            logger.exception("DeepSpeed load_checkpoint failed (%r) : optimizer will start fresh", e)
                        if os.environ.get("DA3_STRICT_DEEPSPEED_RESUME", "0") == "1":
                            raise
                elif rank == 0:
                    logger.warning("No DeepSpeed checkpoint dir at %s : optimizer will start fresh", ds_tag_dir)

            if not ds_ckpt_loaded and not use_deepspeed:
                if "optimizer" in ckpt and ckpt["optimizer"] is not None and not has_model_change:
                    try:
                        opt.load_state_dict(ckpt["optimizer"])
                        if skipped_proprio_conditioner and raw_model.proprio_conditioner is not None:
                            cleared = 0
                            for param in raw_model.proprio_conditioner.parameters():
                                if param in opt.state:
                                    opt.state.pop(param, None)
                                    cleared += 1
                            if rank == 0 and cleared:
                                logger.info(
                                    "Cleared optimizer state for %d unused proprio_conditioner parameters.",
                                    cleared,
                                )
                    except (ValueError, RuntimeError) as e:
                        if rank == 0:
                            logger.info("Skipping optimizer state load: %s", e)
                        has_model_change = True
                elif "optimizer" in ckpt and rank == 0 and has_model_change:
                    logger.info("Skipping optimizer state load due to model change during resume")
            # ckpt may record `"scheduler": None` when saved under DeepSpeed (the
            # engine owns the schedule there); loading None into a native
            # LambdaLR crashes inside torch.optim.lr_scheduler. Guard on value
            # in addition to key presence.
            if (
                ckpt.get("scheduler") is not None
                and scheduler is not None
                and not has_model_change
            ):
                scheduler.load_state_dict(ckpt["scheduler"])
            elif "scheduler" in ckpt and ckpt.get("scheduler") is None and rank == 0:
                logger.info(
                    "Checkpoint scheduler state is None (DeepSpeed-owned or unsaved); "
                    "keeping freshly-built scheduler."
                )
            if ckpt.get("scaler") is not None and not has_model_change:
                scaler.load_state_dict(ckpt["scaler"])
        train_steps = ckpt.get("train_steps", 0)
        start_epoch = ckpt.get("epoch", 0)
        if reset_schedule:
            # A fresh LR schedule is only meaningful if the step counter also
            # restarts at 0; otherwise the loop runs for `max_steps -
            # resumed_step` iterations while the scheduler thinks it's at step
            # 0..(max_steps-resumed_step), producing a nearly-flat LR at the
            # top of the cosine. Treat --reset-schedule as "warm restart from
            # weights only": optimizer + scheduler + step counter + epoch
            # all reset; only the model parameters come from the checkpoint.
            if rank == 0:
                logger.info(
                    "--reset-schedule: also resetting train_steps and start_epoch to 0 "
                    "(was step=%d epoch=%d in ckpt)",
                    train_steps, start_epoch,
                )
            train_steps = 0
            start_epoch = 0
        if rank == 0:
            logger.info("Resumed at step=%d, epoch=%d", train_steps, start_epoch)

    ema_tracker = None
    # DDP: every rank tracks its own (identical) shadow so closed_loop_eval
    # can swap EMA across all ranks during distributed rollout.  DeepSpeed
    # ZeRO partitions params, so we keep rank-0-only there : closed_loop_eval
    # use_ema is unsupported under DeepSpeed.
    if ema_cfg["enabled"] and (not use_deepspeed or rank == 0):
        if use_deepspeed and rank == 0:
            logger.warning(
                "Stage1 subset EMA is rank-0 only under DeepSpeed; "
                "closed_loop_eval.use_ema is unsupported in this mode."
            )
        ema_model_ref = model_engine.module if model_engine is not None else finetune_model
        ema_tracker = Stage1SubsetEMA(ema_model_ref, ema_cfg, logger=logger)
        if resume_ckpt_for_ema is not None:
            ema_tracker.load_from_checkpoint(resume_ckpt_for_ema, logger=logger)

    eval_every = int(training_cfg.get("eval_every", ckpt_every))
    eval_ae = bool(training_cfg.get("eval_ae", True))
    eval_noact = bool(training_cfg.get("eval_noact", True))
    eval_only = bool(getattr(args, "eval_only", False))
    lazy_eval_dataset = bool(training_cfg.get("lazy_eval_dataset", False))

    if eval_only:
        # Eval-only uses the eval split for stats-key discovery and forward
        # evaluation. Building the train OpenX mixer can dominate wall time
        # because large Arrow tables are opened on every rank.
        eval_dataset, _, eval_loader = create_dataset_and_loader(
            dataset_cfg, training_cfg, use_deepspeed, world_size, rank, is_eval=True
        )
        dataset, sampler, loader = eval_dataset, None, eval_loader
    else:
        dataset, sampler, loader = create_dataset_and_loader(
            dataset_cfg, training_cfg, use_deepspeed, world_size, rank, is_eval=False
        )
        if eval_every < 999999 and not lazy_eval_dataset:
            eval_dataset, _, eval_loader = create_dataset_and_loader(
                dataset_cfg, training_cfg, use_deepspeed, world_size, rank, is_eval=True
            )
        else:
            eval_dataset, eval_loader = None, None
            if rank == 0:
                if eval_every >= 999999:
                    logger.info("Skipping eval dataset build (eval_every=%d)", eval_every)
                else:
                    logger.info(
                        "Deferring eval dataset build until first eval "
                        "(eval_every=%d, lazy_eval_dataset=true)",
                        eval_every,
                    )

    # Closed-loop rollout eval config (gam + live simulator datasets).
    # See docs/closed-loop-eval-env.md for per-server env setup.
    closed_loop_profiles, _closed_loop_legacy_mode = _normalize_closed_loop_eval_profiles(training_cfg)
    dataset_type_lower = str(dataset_cfg.get("type", "")).lower()
    for _profile in closed_loop_profiles:
        if not _profile.get("benchmark"):
            _profile["benchmark"] = "libero"
    closed_loop_enabled = (
        bool(closed_loop_profiles)
        and predictor_enabled
        and (
            dataset_type_lower in {"libero", "libero_hdf5"}
            or any(
                str(_profile.get("benchmark", "libero") or "libero").lower().replace("-", "_")
                in {"libero", "plain_libero", "libero_plus", "plus"}
                for _profile in closed_loop_profiles
            )
        )
    )
    if closed_loop_enabled:
        for _profile in closed_loop_profiles:
            _benchmark = str(_profile.get("benchmark", "libero") or "libero").lower().replace("-", "_")
            if _benchmark in {"libero_plus", "plus"} and not (
                _profile.get("plus_root") or os.environ.get("DA3_LIBERO_PLUS_DIR")
            ):
                raise ValueError(
                    f"closed_loop_evals profile {_profile.get('name')} uses benchmark=libero_plus "
                    "but neither plus_root nor DA3_LIBERO_PLUS_DIR is set."
                )
            if _benchmark not in {"libero", "plain_libero", "libero_plus", "plus"}:
                raise ValueError(
                    f"closed_loop_evals profile {_profile.get('name')} has unsupported "
                    f"benchmark={_benchmark!r}; expected libero or libero_plus."
                )
        if rank == 0:
            logger.info(
                "Validating closed_loop_eval env on all %d rank(s); profiles=%s",
                world_size,
                [(p.get("name"), p.get("benchmark", "libero")) for p in closed_loop_profiles],
            )
        validation_error: Exception | None = None
        try:
            _benchmarks_to_validate = {
                str(_p.get("benchmark", "libero") or "libero").lower().replace("-", "_")
                for _p in closed_loop_profiles
            }
            if _benchmarks_to_validate & {"libero", "plain_libero", "libero_plus", "plus"}:
                closed_loop_validate_libero_env()
        except Exception as e:  # noqa: BLE001
            validation_error = e
            logger.error(
                "closed_loop_eval env validation failed on rank %d/%d: %s",
                rank,
                world_size,
                e,
                exc_info=True,
            )

        failed_ranks = 1 if validation_error is not None else 0
        if dist.is_available() and dist.is_initialized():
            failed_tensor = torch.tensor([failed_ranks], device=device, dtype=torch.int32)
            dist.all_reduce(failed_tensor, op=dist.ReduceOp.SUM)
            failed_ranks = int(failed_tensor.item())

        if validation_error is not None:
            raise RuntimeError(
                f"closed_loop_eval env validation failed on local rank {rank}"
            ) from validation_error
        if failed_ranks:
            raise RuntimeError(
                f"closed_loop_eval env validation failed on {failed_ranks} rank(s)"
            )
        if rank == 0:
            logger.info(
                "closed_loop_eval env validation passed on all %d rank(s)",
                world_size,
            )

    normalize_proprio = bool(dataset_cfg.get("normalize_proprio", True))
    action_stats_samples = int(dataset_cfg.get("action_stats_samples", -1))
    proprio_stats_samples = int(dataset_cfg.get("proprio_stats_samples", -1))

    stats_dir = resolve_stats_dir(dataset_cfg)

    # --- Normalization mode (q01_q99 default; mean_std opt-in) ---
    _norm_mode_default = str(dataset_cfg.get("norm_mode", "q01_q99")).strip().lower()
    action_norm_mode = str(dataset_cfg.get("action_norm_mode", _norm_mode_default)).strip().lower()
    proprio_norm_mode = str(dataset_cfg.get("proprio_norm_mode", _norm_mode_default)).strip().lower()
    if rank == 0 and (action_norm_mode != "q01_q99" or proprio_norm_mode != "q01_q99"):
        logger.info(
            "Normalization mode: action=%s proprio=%s",
            action_norm_mode, proprio_norm_mode,
        )
    _norm_stats_distributed = dist.is_available() and dist.is_initialized() and world_size > 1

    def _sync_norm_stats(stage: str) -> None:
        if _norm_stats_distributed:
            if rank == 0:
                logger.info("Syncing %s normalizer stats across %d ranks", stage, world_size)
            dist.barrier()

    # --- Action normalizer (q01/q99 default; mean/std opt-in) ---
    _skip_ckpt_norm = getattr(args, "refresh_action_stats", False)
    if _skip_ckpt_norm and rank == 0:
        logger.info("--refresh-action-stats: ignoring action_normalizer from checkpoint; "
                    "will load from stats_dir instead")
    if args.ckpt is not None and "action_normalizer" in ckpt and not _skip_ckpt_norm:
        action_normalizer = ActionNormalizer.from_state_dict(
            ckpt["action_normalizer"], override_norm_mode=action_norm_mode,
        )
        _validate_checkpoint_action_timing(
            ckpt=ckpt,
            dataset=dataset,
            current_dataset_cfg=dataset_cfg,
            logger=logger,
            rank=rank,
        )
        if rank == 0:
            logger.info("Loaded action normalizer from checkpoint")
        # Supplement with stats_dir for datasets outside the checkpoint.
        # Same dedup as the from-stats-dir path so checkpoint-baked normalizers
        # don't trigger a spurious "missing" complaint when a single shared key
        # covers multiple specs.
        ds_names = sorted({
            getattr(ds, "action_stats_key", None) or ds.dataset_name
            for ds in _flatten_leaf_datasets(dataset)
        })
        missing = [n for n in ds_names if n not in action_normalizer.stats_by_key]
        if missing:
            timing_signatures = _action_timing_signatures_by_dataset(dataset)
            _sup_dir = dataset_cfg.get("stats_dir") or dataset_cfg.get("action_stats_dir")
            if _sup_dir is None and dataset_cfg.get("openx_root"):
                _sup_dir = os.path.join(dataset_cfg["openx_root"], "_stats")
            if _sup_dir and os.path.isdir(_sup_dir):
                import json as _json
                eps = action_normalizer.eps
                supplemented: list[str] = []
                for name in missing:
                    fpath = os.path.join(_sup_dir, f"{name}.json")
                    if os.path.exists(fpath):
                        d = _json.loads(open(fpath).read())
                        expected_sig = timing_signatures.get(name)
                        actual_sig = d.get("timing_signature")
                        if expected_sig is not None:
                            if actual_sig is None and expected_sig.startswith("random"):
                                if rank == 0:
                                    logger.warning(
                                        "Skipping legacy action stats for %s: random-stride timing metadata is missing",
                                        name,
                                    )
                                continue
                            if actual_sig is not None and not ActionNormalizer._timing_signature_matches(
                                actual_sig, expected_sig
                            ):
                                if rank == 0:
                                    logger.warning(
                                        "Skipping action stats for %s: timing_signature=%s expected=%s",
                                        name,
                                        actual_sig,
                                        expected_sig,
                                    )
                                continue
                        q01 = torch.tensor(d["q01"], dtype=torch.float32)
                        q99 = torch.tensor(d["q99"], dtype=torch.float32)
                        mask = torch.tensor(d.get("mask", DEFAULT_ACTION_NORM_MASK), dtype=torch.bool)
                        mean = std = None
                        if "mean" in d and "std" in d:
                            mean = torch.tensor(d["mean"], dtype=torch.float32)
                            std = torch.tensor(d["std"], dtype=torch.float32)
                        if action_normalizer.norm_mode == "mean_std":
                            if mean is None or std is None:
                                raise ValueError(
                                    f"Action stats for {name!r} in {_sup_dir} are missing mean/std "
                                    "but runtime action_norm_mode='mean_std'. Re-run with "
                                    "--refresh-action-stats."
                                )
                            constant_dims = std.abs() < ActionNormalizer.CONSTANT_DIM_THRESHOLD
                        else:
                            constant_dims = (q99 - q01).abs() < ActionNormalizer.CONSTANT_DIM_THRESHOLD
                        mask = mask & ~constant_dims
                        entry = {
                            "q01": q01, "q99": q99,
                            "scale": q99 - q01 + eps, "mask": mask,
                        }
                        if mean is not None and std is not None:
                            entry["mean"] = mean
                            entry["std"] = std
                        action_normalizer.stats_by_key[name] = entry
                        supplemented.append(name)
                if rank == 0:
                    logger.info(
                        "Supplemented action normalizer with %d/%d missing datasets from %s: %s",
                        len(supplemented),
                        len(missing),
                        _sup_dir,
                        supplemented,
                    )
            still_missing = [n for n in missing if n not in action_normalizer.stats_by_key]
            if still_missing:
                raise ValueError(
                    "Checkpoint action_normalizer is missing compatible stats for "
                    f"{still_missing}. Pass --refresh-action-stats to recompute stats for "
                    "the current target_hz/random_stride policy-action distribution."
                )
    else:
        _target_hz_raw = dataset_cfg.get("target_hz")
        target_hz = float(_target_hz_raw) if _target_hz_raw else None
        ds_list = _flatten_leaf_datasets(dataset)
        # Use `action_stats_key` (= spec.name unless overridden) so several
        # specs that share a normalizer (e.g. all four franka_* tasks share
        # `franka_realrobot`) collapse to one stats sidecar lookup.
        ds_names = sorted({
            getattr(ds, "action_stats_key", None) or ds.dataset_name
            for ds in ds_list
        })
        timing_signatures = _action_timing_signatures_by_dataset(dataset)
        target_hz_map = None
        if target_hz is None:
            target_hz_map = {
                (getattr(ds, "action_stats_key", None) or ds.dataset_name): float(
                    getattr(ds, "fps", 0) or 0
                )
                for ds in ds_list
                if getattr(ds, "fps", None)
            }
        if stats_dir:
            if rank == 0:
                os.makedirs(stats_dir, exist_ok=True)
            _sync_norm_stats("action-stats-dir-create")
        if stats_dir and os.path.isdir(stats_dir):
            if rank == 0:
                logger.info("Loading action normalizer from %s for %d datasets (target_hz=%s, per_ds_hz=%s)",
                            stats_dir, len(ds_names), target_hz, target_hz_map)
            try:
                action_normalizer = ActionNormalizer.from_stats_dir(
                    stats_dir, ds_names,
                    target_hz=target_hz, target_hz_map=target_hz_map,
                    expected_timing_signatures=timing_signatures,
                    require_timing_signature=bool(dataset_cfg.get("require_stats_timing_signature", False)),
                    norm_mode=action_norm_mode,
                )
            except (ValueError, KeyError) as exc:
                if rank == 0:
                    logger.warning("Could not load compatible action stats; recomputing: %s", exc)
                if rank == 0:
                    logger.info("Computing action normalizer (max_samples=%s)", action_stats_samples)
                    action_normalizer = ActionNormalizer.from_dataset(
                        dataset, max_samples=action_stats_samples, norm_mode=action_norm_mode,
                    )
                    action_normalizer.save_to_stats_dir(
                        stats_dir,
                        target_hz=target_hz,
                        timing_signatures=timing_signatures,
                    )
                    logger.info("Saved action normalizer stats to %s", stats_dir)
                _sync_norm_stats("action")
                if rank != 0:
                    action_normalizer = ActionNormalizer.from_stats_dir(
                        stats_dir, ds_names,
                        target_hz=target_hz, target_hz_map=target_hz_map,
                        expected_timing_signatures=timing_signatures,
                        require_timing_signature=bool(dataset_cfg.get("require_stats_timing_signature", False)),
                        norm_mode=action_norm_mode,
                    )
        else:
            if rank == 0:
                logger.info("Computing action normalizer (max_samples=%s)", action_stats_samples)
            action_normalizer = ActionNormalizer.from_dataset(
                dataset, max_samples=action_stats_samples, norm_mode=action_norm_mode,
            )
            if stats_dir and rank == 0:
                action_normalizer.save_to_stats_dir(
                    stats_dir,
                    target_hz=target_hz,
                    timing_signatures=_action_timing_signatures_by_dataset(dataset),
                )
                logger.info("Saved action normalizer stats to %s", stats_dir)
            _sync_norm_stats("action")
        if rank == 0:
            logger.info("Action normalizer stats keys: %s", sorted(action_normalizer.stats_by_key))

    proprio_normalizer = None
    if normalize_proprio:
        # Mirror the action normalizer's shared-key collapse: when several
        # specs declare the same `action_stats_key`, they share a single
        # proprio sidecar too.
        ds_names = sorted({
            getattr(ds, "action_stats_key", None) or ds.dataset_name
            for ds in _flatten_leaf_datasets(dataset)
        })
        expected_proprio_dim = int(dataset_cfg.get("proprio_dim", 7))
        _skip_ckpt_proprio_norm = getattr(args, "refresh_action_stats", False)
        if _skip_ckpt_proprio_norm and rank == 0:
            logger.info(
                "--refresh-action-stats: ignoring proprio_normalizer from checkpoint; "
                "will load from stats_dir instead"
            )
        if (
            args.ckpt is not None
            and "proprio_normalizer" in ckpt
            and ckpt["proprio_normalizer"] is not None
            and not _skip_ckpt_proprio_norm
        ):
            proprio_normalizer = StateNormalizer.from_state_dict(
                ckpt["proprio_normalizer"], override_norm_mode=proprio_norm_mode,
            )
            if rank == 0:
                logger.info("Loaded proprio normalizer from checkpoint")
            mismatches = state_normalizer_dim_mismatches(
                proprio_normalizer,
                ds_names,
                expected_dim=expected_proprio_dim,
            )
            if mismatches:
                if rank == 0:
                    preview = "; ".join(mismatches[:5])
                    if len(mismatches) > 5:
                        preview += f"; ... (+{len(mismatches) - 5} more)"
                    logger.info(
                        "Ignoring checkpoint proprio normalizer due to dim mismatch: %s",
                        preview,
                    )
                proprio_normalizer = None
            # Supplement with stats_dir for datasets outside the checkpoint.
            missing = [] if proprio_normalizer is None else [n for n in ds_names if n not in proprio_normalizer.stats_by_key]
            if proprio_normalizer is not None and missing:
                _stats_dir = dataset_cfg.get("stats_dir") or dataset_cfg.get("action_stats_dir")
                if _stats_dir is None and dataset_cfg.get("openx_root"):
                    _stats_dir = os.path.join(dataset_cfg["openx_root"], "_stats")
                if _stats_dir and os.path.isdir(_stats_dir):
                    import json as _json
                    eps = proprio_normalizer.eps
                    for name in missing:
                        fpath = os.path.join(_stats_dir, f"{name}_proprio.json")
                        if os.path.exists(fpath):
                            d = _json.loads(open(fpath).read())
                            q01 = torch.tensor(d["q01"], dtype=torch.float32)
                            q99 = torch.tensor(d["q99"], dtype=torch.float32)
                            mask = torch.tensor(d.get("mask", [True] * int(q01.shape[-1])), dtype=torch.bool)
                            mean = std = None
                            if "mean" in d and "std" in d:
                                mean = torch.tensor(d["mean"], dtype=torch.float32)
                                std = torch.tensor(d["std"], dtype=torch.float32)
                            if proprio_normalizer.norm_mode == "mean_std":
                                if mean is None or std is None:
                                    raise ValueError(
                                        f"Proprio stats for {name!r} in {_stats_dir} are missing mean/std "
                                        "but runtime proprio_norm_mode='mean_std'. Re-run with "
                                        "--refresh-action-stats."
                                    )
                                constant_dims = std.abs() < StateNormalizer.CONSTANT_DIM_THRESHOLD
                            else:
                                constant_dims = (q99 - q01).abs() < StateNormalizer.CONSTANT_DIM_THRESHOLD
                            mask = mask & ~constant_dims
                            entry = {
                                "q01": q01, "q99": q99,
                                "scale": q99 - q01 + eps, "mask": mask,
                            }
                            if mean is not None and std is not None:
                                entry["mean"] = mean
                                entry["std"] = std
                            proprio_normalizer.stats_by_key[name] = entry
                    if rank == 0:
                        logger.info("Supplemented proprio normalizer with %d missing datasets: %s", len(missing), missing)
        if proprio_normalizer is None:
            if stats_dir:
                if rank == 0:
                    os.makedirs(stats_dir, exist_ok=True)
                _sync_norm_stats("proprio-stats-dir-create")
            if stats_dir and os.path.isdir(stats_dir):
                try:
                    if rank == 0:
                        logger.info("Loading proprio normalizer from %s for %d datasets", stats_dir, len(ds_names))
                    proprio_normalizer = StateNormalizer.from_stats_dir(
                        stats_dir, ds_names, norm_mode=proprio_norm_mode,
                    )
                except (ValueError, KeyError) as exc:
                    # Falls through when proprio stats files are missing from an
                    # otherwise-populated stats_dir (e.g. action stats exist while
                    # proprio stats are absent on this server).
                    if rank == 0:
                        logger.info(
                            "from_stats_dir failed (%s); computing proprio normalizer from dataset "
                            "(max_samples=%s)", exc, proprio_stats_samples,
                        )
                    if rank == 0:
                        proprio_normalizer = StateNormalizer.from_dataset(
                            dataset, max_samples=proprio_stats_samples, norm_mode=proprio_norm_mode,
                        )
                        proprio_normalizer.save_to_stats_dir(stats_dir)
                        logger.info("Saved proprio normalizer stats to %s", stats_dir)
                    _sync_norm_stats("proprio")
                    if rank != 0:
                        proprio_normalizer = StateNormalizer.from_stats_dir(
                            stats_dir, ds_names, norm_mode=proprio_norm_mode,
                        )
            else:
                if rank == 0:
                    logger.info("Computing proprio normalizer (max_samples=%s)", proprio_stats_samples)
                proprio_normalizer = StateNormalizer.from_dataset(
                    dataset, max_samples=proprio_stats_samples, norm_mode=proprio_norm_mode,
                )
                if stats_dir and rank == 0:
                    proprio_normalizer.save_to_stats_dir(stats_dir)
                    logger.info("Saved proprio normalizer stats to %s", stats_dir)
                _sync_norm_stats("proprio")
        if rank == 0:
            logger.info("Proprio normalizer stats keys: %s", sorted(proprio_normalizer.stats_by_key))

    maybe_log_action_stats(dataset, dataset_cfg, logger, wandb_run)

    running_total = 0.0
    running_action = 0.0
    running_feat = 0.0
    # Separate running means for the DA3 paper depth-loss breakdown (L_D, L_M, L_P).
    running_depth = 0.0
    running_ray = 0.0
    running_point = 0.0
    log_steps = 0
    start_time = time()

    # Per-dataset metric accumulator for wandb.Table logging
    from collections import defaultdict
    per_ds_accum = defaultdict(lambda: {"ae_l1": [], "mse": [], "count": 0})

    data_wait_cfg = _plain_config_container(training_cfg.get("dataloader_wait_log", {})) or {}
    data_wait_log_enabled = bool(int(os.environ.get("DA3_DATALOADER_WAIT_LOG", "0"))) or bool(
        data_wait_cfg.get("enabled", False)
    )
    data_wait_log_threshold_s = float(
        os.environ.get("DA3_DATALOADER_WAIT_LOG_THRESHOLD_S", data_wait_cfg.get("threshold_s", 20.0))
    )
    data_wait_log_every = int(
        os.environ.get("DA3_DATALOADER_WAIT_LOG_EVERY", data_wait_cfg.get("every", 0))
    )
    data_wait_log_all_ranks = str(
        os.environ.get("DA3_DATALOADER_WAIT_LOG_RANKS", data_wait_cfg.get("ranks", "all"))
    ).lower() not in {"0", "rank0", "rank_0"}
    data_wait_running_s = 0.0
    data_wait_max_s = 0.0
    data_wait_slow_batches = 0
    data_wait_last_summary: dict = {}

    def _save_training_checkpoint(model_ref, step: int, epoch_idx: int, *, reason: str = "") -> None:
        step_int = int(step)
        reason_suffix = f" ({reason})" if reason else ""
        if use_deepspeed and model_engine is not None:
            # DeepSpeed ZeRO partitions optimizer state across ranks. Use ds
            # save_checkpoint so every rank dumps its shard; the directory can
            # be restored by ds load_checkpoint.
            ds_ckpt_dir = os.path.join(experiment_dir, "checkpoints")
            ds_tag = f"{step_int:07d}"
            model_engine.save_checkpoint(ds_ckpt_dir, tag=ds_tag)
            if rank == 0:
                logger.info("Saved DeepSpeed checkpoint%s: %s/tag=%s", reason_suffix, ds_ckpt_dir, ds_tag)
        if rank == 0:
            # Lightweight model-only ckpt for eval / cross-server portability.
            # When use_deepspeed, optimizer state lives in the ds_tag dir, so
            # skip it here to avoid writing an unloadable single-rank shard.
            raw_model = model_ref
            if hasattr(raw_model, "module"):
                raw_model = raw_model.module
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            ckpt = {
                "student_da3": raw_model.student_da3.state_dict(),
                "action_head": raw_model.action_head.state_dict(),
                "proprio_head": None
                if raw_model.proprio_head is None
                else raw_model.proprio_head.state_dict(),
                "proprio_conditioner": None
                if raw_model.proprio_conditioner is None
                else raw_model.proprio_conditioner.state_dict(),
                "future_predictor": None
                if raw_model.future_predictor is None
                else raw_model.future_predictor.state_dict(),
                "text_conditioner_proj": None
                if raw_model.text_conditioner is None
                else raw_model.text_conditioner.proj.state_dict(),
                "optimizer": opt.state_dict() if not use_deepspeed else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict(),
                "action_normalizer": action_normalizer.state_dict(),
                "proprio_normalizer": None if proprio_normalizer is None else proprio_normalizer.state_dict(),
                "wandb_run_id": wandb_run.id if wandb_run is not None else None,
                "train_steps": step_int,
                "epoch": epoch_idx,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if ema_tracker is not None:
                ckpt.update(ema_tracker.checkpoint_state(step_int))
            ckpt_path = os.path.join(experiment_dir, "checkpoints", f"{step_int:07d}.pt")
            torch.save(ckpt, ckpt_path)
            logger.info("Saved checkpoint%s: %s", reason_suffix, ckpt_path)

    def _ensure_eval_loader(eval_step: int) -> bool:
        nonlocal eval_dataset, eval_loader
        if eval_loader is not None:
            return len(eval_loader) > 0
        if eval_every <= 0 or eval_every >= 999999:
            return False
        if rank == 0:
            logger.info("[step=%07d] building eval dataset lazily", int(eval_step))
        eval_dataset, _, eval_loader = create_dataset_and_loader(
            dataset_cfg, training_cfg, use_deepspeed, world_size, rank, is_eval=True
        )
        return eval_loader is not None and len(eval_loader) > 0

    def _run_unified_predictor_eval(eval_step: int, *, max_batches: int = 0) -> None:
        """Run open-loop eval for the unified/predictor path using the real forward loss."""
        if not predictor_enabled:
            return
        if not _ensure_eval_loader(int(eval_step)):
            if rank == 0:
                logger.info("[step=%07d] unified eval skipped: empty eval loader", int(eval_step))
            return
        if predictor_type not in {"gam"}:
            raise RuntimeError(
                f"predictor.type={predictor_type!r} is unsupported by unified eval; "
                "use gam."
            )

        from robot.losses.unified_loss import compute_gam_forward_loss

        eval_h_cfg = training_cfg.get("eval_H", predictor_cfg.get("eval_H", None))
        if eval_h_cfg is None or str(eval_h_cfg).lower() in {"", "max", "full"}:
            eval_H = max(int(h) for h in unified_H_choices)
        elif str(eval_h_cfg).lower() == "min":
            eval_H = min(int(h) for h in unified_H_choices)
        else:
            eval_H = int(eval_h_cfg)
        eval_H = max(1, min(int(eval_H), n_action_steps - 1))

        if max_batches <= 0:
            max_batches = int(training_cfg.get("eval_max_batches", 0))
        max_batches = int(max_batches)

        model_ref_eval = model_engine.module if model_engine is not None else finetune_model
        raw_model_unwrapped = model_ref_eval.module if hasattr(model_ref_eval, "module") else model_ref_eval
        if hasattr(raw_model_unwrapped, "_orig_mod"):
            raw_model_unwrapped = raw_model_unwrapped._orig_mod

        prior_train_mode = finetune_model.training
        finetune_model.eval()
        teacher_da3.eval()

        local_preds = []
        local_gts = []
        local_masks = []
        local_keys = []
        first_on_rank0 = {"pred": None, "actions": None, "mask": None, "keys": None}
        local_scalar_sums: dict[str, float] = {}
        local_weight = 0
        local_batches = 0

        if rank == 0:
            cap_desc = "full" if max_batches <= 0 else str(max_batches)
            logger.info(
                "[step=%07d] unified eval starting H=%d batches=%s eval_len=%d",
                int(eval_step), eval_H, cap_desc, len(eval_loader),
            )

        def _add_scalar(name: str, value, weight: int) -> None:
            if value is None:
                return
            if not torch.is_tensor(value):
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    return
            else:
                if value.numel() != 1:
                    return
                scalar = float(value.detach().float().item())
            if math.isfinite(scalar):
                local_scalar_sums[name] = local_scalar_sums.get(name, 0.0) + scalar * float(weight)

        with torch.no_grad():
            for eval_batch in eval_loader:
                if max_batches > 0 and local_batches >= max_batches:
                    break
                (
                    ev_views,
                    ev_actions_raw,
                    ev_proprio,
                    _,
                    _,
                    ev_teacher_views,
                    ev_teacher_depth_valid_mask,
                    ev_view_valid_mask,
                ) = prepare_da3_finetune_batch(teacher_da3, eval_batch, device)
                ev_stats_keys = list(eval_batch["action_stats_key"])
                ev_action_loss_mask = eval_batch.get("action_loss_mask")
                if ev_action_loss_mask is not None:
                    ev_action_loss_mask = ev_action_loss_mask.to(device=device, dtype=torch.bool)
                ev_transition_loss_mask = eval_batch.get("transition_loss_mask")
                if ev_transition_loss_mask is not None:
                    ev_transition_loss_mask = ev_transition_loss_mask.to(device=device, dtype=torch.bool)
                ev_context_valid_mask = eval_batch.get("context_valid_mask")
                if ev_context_valid_mask is not None:
                    ev_context_valid_mask = ev_context_valid_mask.to(device=device, dtype=torch.bool)

                ev_actions = action_normalizer.normalize(ev_actions_raw, stats_keys=ev_stats_keys)
                if ev_action_loss_mask is not None:
                    mask_for_actions = ev_action_loss_mask
                    while mask_for_actions.ndim < ev_actions.ndim:
                        mask_for_actions = mask_for_actions.unsqueeze(-1)
                    ev_actions = torch.where(mask_for_actions, ev_actions, torch.zeros_like(ev_actions))

                ev_past_action_history = None
                ev_past_action_history_raw = eval_batch.get("past_action_history")
                if ev_past_action_history_raw is not None:
                    ev_past_action_history = action_normalizer.normalize(
                        ev_past_action_history_raw.to(device=device),
                        stats_keys=ev_stats_keys,
                    )
                    ev_past_action_history_mask = eval_batch.get("past_action_history_mask")
                    if ev_past_action_history_mask is not None:
                        input_mask = ev_past_action_history_mask.to(device=device, dtype=torch.bool)
                        while input_mask.ndim < ev_past_action_history.ndim:
                            input_mask = input_mask.unsqueeze(-1)
                        ev_past_action_history = torch.where(
                            input_mask,
                            ev_past_action_history,
                            torch.zeros_like(ev_past_action_history),
                        )

                if proprio_normalizer is not None:
                    ev_proprio = proprio_normalizer.normalize(ev_proprio, stats_keys=ev_stats_keys)
                    ev_proprio = _zero_invalid_context_proprio(ev_proprio, ev_context_valid_mask)

                ev_gt_depth_da3 = eval_batch.get("gt_depth_da3")
                ev_gt_depth_mask = eval_batch.get("gt_depth_mask")
                ev_gt_camera_intrinsics = eval_batch.get("gt_camera_intrinsics")
                ev_gt_camera_extrinsics_c2w = eval_batch.get("gt_camera_extrinsics_c2w")
                ev_gt_depth_scene_scale = eval_batch.get("gt_depth_scene_scale")
                ev_language_texts = eval_batch.get("task_description", None) if use_language else None

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    ev_out = compute_gam_forward_loss(
                        student_da3=raw_model_unwrapped.student_da3,
                        teacher_da3=teacher_da3,
                        future_predictor=raw_model_unwrapped.future_predictor,
                        text_conditioner=raw_model_unwrapped.text_conditioner,
                        action_head=raw_model_unwrapped.action_head,
                        proprio_head=raw_model_unwrapped.proprio_head,
                        regularizer=regularizer,
                        all_views_norm=ev_views,
                        teacher_views_norm=ev_teacher_views,
                        teacher_depth_valid_mask=ev_teacher_depth_valid_mask,
                        gt_actions=ev_actions,
                        proprio=ev_proprio,
                        language_texts=ev_language_texts,
                        T=n_action_steps,
                        V=n_views,
                        H=eval_H,
                        lambda_action=lambda_action,
                        lambda_feat_future=lambda_feat_future,
                        lambda_proprio_future=lambda_proprio_future,
                        lambda_sigreg=lambda_sigreg,
                        use_bf16=use_bf16,
                        lambda_action_direct=lambda_action_direct,
                        lambda_action_refine=lambda_action_refine,
                        lambda_depth=regularizer.lambda_depth,
                        gt_depth_da3=None if ev_gt_depth_da3 is None else ev_gt_depth_da3.to(device),
                        gt_depth_mask=None if ev_gt_depth_mask is None else ev_gt_depth_mask.to(device),
                        depth_grad_weight=depth_grad_weight,
                        depth_decode_chunk_size=depth_decode_chunk_size,
                        teacher_depth_fallback=bool(reg_cfg.get("teacher_depth_fallback", False)),
                        skip_depth_if_no_gt=bool(reg_cfg.get("skip_depth_if_no_gt", False)),
                        lambda_path_b_deep_feat_reg=float(reg_cfg.get("lambda_path_b_deep_feat_reg", 0.0)),
                        deep_feat_reg_layer_weight_min=float(reg_cfg.get("deep_feat_reg_layer_weight_min", 0.5)),
                        lambda_feat_current=lambda_feat_current,
                        embed_dim=raw_model_unwrapped.student_da3.embed_dim,
                        deep_gradient_checkpointing=False,
                        deep_temporal_causal_mask=gam_deep_temporal_causal_mask,
                        feature_channel_stats=feature_channel_stats,
                        feature_loss_norm=feature_loss_norm,
                        feature_loss_type=feature_loss_type,
                        feature_loss_norm_eps=feature_loss_norm_eps,
                        feature_target_mode=feature_target_mode,
                        depth_loss_type=regularizer.depth_loss_type,
                        lambda_depth_conf=regularizer.lambda_depth_conf,
                        lambda_ray=regularizer.lambda_ray,
                        lambda_point=regularizer.lambda_point,
                        gt_camera_intrinsics=None if ev_gt_camera_intrinsics is None else ev_gt_camera_intrinsics.to(device),
                        gt_camera_extrinsics_c2w=None if ev_gt_camera_extrinsics_c2w is None else ev_gt_camera_extrinsics_c2w.to(device),
                        gt_depth_scene_scale=None if ev_gt_depth_scene_scale is None else ev_gt_depth_scene_scale.to(device),
                        prev_action_mask_rate=0.0,
                        action_loss_mask=ev_action_loss_mask,
                        transition_loss_mask=ev_transition_loss_mask,
                        context_valid_mask=ev_context_valid_mask,
                        view_valid_mask=ev_view_valid_mask,
                        past_action_history=ev_past_action_history,
                        forward_profile=None,
                    )

                ev_pred = ev_out["action_pred"].detach().float()
                ev_target = ev_out["target_actions"].detach().float()
                ev_mask = ev_out.get("action_loss_mask_used")
                if ev_mask is None:
                    ev_mask = torch.ones(ev_pred.shape[:-1], device=device, dtype=torch.bool)
                else:
                    ev_mask = ev_mask.detach().to(device=device, dtype=torch.bool)
                local_preds.append(ev_pred)
                local_gts.append(ev_target)
                local_masks.append(ev_mask.to(dtype=torch.uint8))
                local_keys.extend(ev_stats_keys)
                if first_on_rank0["pred"] is None and rank == 0:
                    first_on_rank0["pred"] = ev_pred.detach()
                    first_on_rank0["actions"] = ev_target.detach()
                    first_on_rank0["mask"] = ev_mask.detach()
                    first_on_rank0["keys"] = ev_stats_keys

                batch_weight = int(ev_pred.shape[0])
                for key in (
                    "loss_total",
                    "loss_action",
                    "loss_action_direct",
                    "loss_action_refine",
                    "loss_feat_current",
                    "loss_feat_future",
                    "loss_proprio_future",
                    "loss_sigreg",
                    "loss_depth",
                    "loss_ray",
                    "loss_point",
                    "loss_camera",
                    "loss_deep_feat_reg",
                ):
                    _add_scalar(key, ev_out.get(key), batch_weight)
                for key, value in (ev_out.get("depth_metrics") or {}).items():
                    _add_scalar(f"depth/{key}", value, batch_weight)
                local_weight += batch_weight
                local_batches += 1

                del ev_views, ev_actions_raw, ev_actions, ev_proprio, ev_out
                torch.cuda.empty_cache()

        pred_all, gt_all, keys_all, count = _gather_eval_tensors(
            local_preds, local_gts, local_keys, world_size, rank, device
        )
        mask_all_u8, mask_count = _gather_variable_eval_tensor(
            local_masks, world_size, rank, device, dtype=torch.uint8
        )
        mask_all = mask_all_u8.bool()

        summary = {"weight": local_weight, "batches": local_batches, "sums": local_scalar_sums}
        if world_size > 1:
            import torch.distributed as _dist
            summaries = [None for _ in range(world_size)]
            _dist.all_gather_object(summaries, summary)
        else:
            summaries = [summary]

        if rank == 0 and count > 0:
            if mask_count != count:
                logger.warning(
                    "[step=%07d] unified eval mask count mismatch: pred=%d mask=%d; "
                    "falling back to unmasked metrics",
                    int(eval_step), count, mask_count,
                )
                mask_all = None
            total_weight = sum(int(s.get("weight", 0)) for s in summaries if s)
            total_batches = sum(int(s.get("batches", 0)) for s in summaries if s)
            scalar_totals: dict[str, float] = {}
            for summary_item in summaries:
                if not summary_item:
                    continue
                for key, value in summary_item.get("sums", {}).items():
                    scalar_totals[key] = scalar_totals.get(key, 0.0) + float(value)
            denom = float(max(total_weight, 1))

            norm_metrics = compute_action_detail_metrics(pred_all, gt_all, action_loss_mask=mask_all)
            raw_metrics = compute_raw_action_metrics(
                pred_all, gt_all, action_normalizer, stats_keys=keys_all, action_loss_mask=mask_all
            )
            per_dataset = compute_per_dataset_action_metrics(
                pred_all, gt_all, action_normalizer, stats_keys=keys_all, action_loss_mask=mask_all
            )
            logger.info(
                "[step=%07d] EVAL-UNIFIED H=%d l1_norm=%.4f mse_norm=%.5f "
                "l1_raw=%.4f mse_raw=%.5f r2_norm=%.3f rel@5/10=%.3f/%.3f "
                "(%d samples, %d batches)",
                int(eval_step), eval_H,
                norm_metrics["l1"], norm_metrics["mse"],
                raw_metrics["l1"], raw_metrics["mse"], norm_metrics["r2"],
                norm_metrics["rel_acc_5pct"], norm_metrics["rel_acc_10pct"],
                count, total_batches,
            )

            if wandb_run is not None:
                eval_log_step = int(eval_step)
                try:
                    current_wandb_step = int(getattr(wandb_run, "step", 0) or 0)
                    if current_wandb_step > eval_log_step:
                        eval_log_step = current_wandb_step
                except Exception:
                    current_wandb_step = 0
                if eval_log_step != int(eval_step):
                    logger.warning(
                        "[step=%07d] W&B current step is %d; logging unified eval at step %d",
                        int(eval_step), int(current_wandb_step), int(eval_log_step),
                    )
                wandb_log: dict = {
                    "eval_unified/H": int(eval_H),
                    "eval_unified/count": int(count),
                    "eval_unified/batches": int(total_batches),
                    "eval_unified/l1_norm": norm_metrics["l1"],
                    "eval_unified/mse_norm": norm_metrics["mse"],
                    "eval_unified/r2_norm": norm_metrics["r2"],
                    "eval_unified/l1_raw": raw_metrics["l1"],
                    "eval_unified/mse_raw": raw_metrics["mse"],
                    "eval_unified/r2_raw": raw_metrics["r2"],
                    "eval_unified/rel_acc_1pct": norm_metrics["rel_acc_1pct"],
                    "eval_unified/rel_acc_5pct": norm_metrics["rel_acc_5pct"],
                    "eval_unified/rel_acc_10pct": norm_metrics["rel_acc_10pct"],
                    "eval_unified/seq_rel_acc_5pct": norm_metrics["seq_acc_5pct"],
                    "eval_unified/seq_rel_acc_10pct": norm_metrics["seq_acc_10pct"],
                    "eval_unified/translation_l1_norm": norm_metrics["translation_mae"],
                    "eval_unified/rotation_l1_norm": norm_metrics["rotation_mae"],
                    "eval_unified/gripper_l1_norm": norm_metrics["gripper_mae"],
                    "eval_unified/translation_l1_raw": raw_metrics["translation_mae"],
                    "eval_unified/rotation_l1_raw": raw_metrics["rotation_mae"],
                    "eval_unified/gripper_l1_raw": raw_metrics["gripper_mae"],
                    "eval_unified/trans_vec_l1_raw": raw_metrics["trans_vec_mae"],
                    "eval_unified/rot_vec_l1_raw": raw_metrics["rot_vec_mae"],
                }
                add_named_metrics(
                    wandb_log, "eval_unified/per_dim_l1_norm",
                    ACTION_DIM_NAMES, norm_metrics["per_dim_mae"],
                )
                add_named_metrics(
                    wandb_log, "eval_unified/per_dim_l1_raw",
                    ACTION_DIM_NAMES, raw_metrics["per_dim_mae"],
                )
                add_indexed_metrics(wandb_log, "eval_unified/timestep_l1_norm", norm_metrics["timestep_mae"])
                add_indexed_metrics(wandb_log, "eval_unified/timestep_l1_raw", raw_metrics["timestep_mae"])
                for key, value in scalar_totals.items():
                    wandb_log[f"eval_unified/{key}"] = float(value) / denom
                add_per_dataset_metrics(wandb_log, "eval_unified_by_dataset", per_dataset)
                try:
                    wandb_run.log(wandb_log, step=int(eval_log_step))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[step=%07d] wandb unified eval log failed: %s", int(eval_step), exc)

                if first_on_rank0["pred"] is not None:
                    try:
                        log_action_trajectory(
                            action_normalizer.denormalize(
                                first_on_rank0["actions"], stats_keys=first_on_rank0["keys"]
                            ),
                            action_normalizer.denormalize(
                                first_on_rank0["pred"], stats_keys=first_on_rank0["keys"]
                            ),
                            int(eval_log_step),
                            wandb_run=wandb_run,
                            batch_idx=0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[step=%07d] unified eval trajectory vis failed: %s", int(eval_step), exc)

        if prior_train_mode:
            finetune_model.train()
        else:
            finetune_model.eval()
        torch.cuda.empty_cache()

    if rank == 0:
        # Prominent Hz / stride / normalization summary
        _ds_list = dataset.datasets.values() if hasattr(dataset, "datasets") else [dataset]
        logger.info("=" * 70)
        logger.info("TRAINING CONFIGURATION")
        logger.info("=" * 70)
        _target_hz_raw = dataset_cfg.get("target_hz", None)
        _target_hz = float(_target_hz_raw) if _target_hz_raw is not None else None
        logger.info("  target_hz: %s", _target_hz)
        for _ds in _ds_list:
            if hasattr(_ds, "temporal_stride"):
                chunk = int(getattr(_ds, "chunk_size", dataset_chunk_size))
                max_stride = int(getattr(_ds, "max_stride", getattr(_ds, "temporal_stride", 1)) or 1)
                timing_sig = _action_timing_signature_for_dataset(_ds)
                if timing_sig[0] == "random":
                    action_stride_desc = f"random[1..{timing_sig[1]}]"
                    visual_stride_desc = f"random[{chunk}..{chunk * timing_sig[1]}]"
                else:
                    action_stride_desc = str(timing_sig[1])
                    visual_stride_desc = str(chunk * timing_sig[1])
                logger.info(
                    "  %-30s  native=%dfps  action_stride=%s  visual_anchor_stride=%s  "
                    "chunk=%d  labels/sample=%d  effective=%.1fHz  max_stride=%d",
                    _ds.dataset_name,
                    _ds.fps,
                    action_stride_desc,
                    visual_stride_desc,
                    chunk,
                    action_steps * chunk,
                    _ds.effective_fps,
                    max_stride,
                )
        logger.info("  action normalizer: %d datasets, stats_keys=%s",
                     len(action_normalizer.stats_by_key), sorted(action_normalizer.stats_by_key))
        logger.info(
            "  action normalizer unit: per policy action after action_stride aggregation; "
            "chunk_size groups multiple normalized 7D policy actions under one visual anchor."
        )
        logger.info("  action_head chunk_position_encoding: %s", chunk_position_encoding)
        logger.info("  samples=%d  batch=%d  epochs=%d  batches/epoch=%d",
                     len(dataset), loader.batch_size, num_epochs, len(loader))
        logger.info("=" * 70)

    stop_training = False
    if eval_only:
        _run_unified_predictor_eval(
            int(train_steps),
            max_batches=int(getattr(args, "eval_max_batches", 0) or 0),
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if rank == 0 and wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb finish after eval-only failed: %s", exc)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, num_epochs):
        if stop_training:
            break
        train_loader_batch_size = int(getattr(loader, "batch_size", 0) or int(training_cfg.get("micro_batch_size", 4)))
        _set_train_sampler_position(
            sampler,
            epoch=epoch,
            start_batch_idx=0,
            batch_size=train_loader_batch_size,
            rank=rank,
            logger=logger,
            reason="epoch start",
        )

        finetune_model.train()

        # Optional per-step timing breakdown (loader wait / host prep /
        # forward / backward / optimizer). Enable with env DA3_STEP_PROFILE=1
        # or training.step_profile.enabled=true.
        _step_profile_cfg = _plain_config_container(training_cfg.get("step_profile", {})) or {}
        step_profile_enabled = bool(int(os.environ.get("DA3_STEP_PROFILE", "0"))) or bool(
            _step_profile_cfg.get("enabled", False)
        )
        step_profile_every = int(os.environ.get("DA3_STEP_PROFILE_EVERY", _step_profile_cfg.get("every", 10)))
        _prof_t_last_step_end = None
        import torch as _torch_for_prof

        _data_iter = iter(loader)
        _epoch_batches_consumed = 0
        while True:
            if stop_training:
                break
            if _data_iter is None:
                _set_train_sampler_position(
                    sampler,
                    epoch=epoch,
                    start_batch_idx=_epoch_batches_consumed,
                    batch_size=train_loader_batch_size,
                    rank=rank,
                    logger=logger,
                    reason="after closed-loop eval",
                )
                _data_iter = iter(loader)
            _prof_host_data_start = perf_counter() if (step_profile_enabled or data_wait_log_enabled) else None
            try:
                batch = next(_data_iter)
            except StopIteration:
                break
            _epoch_batches_consumed += 1
            _data_wait_s = 0.0
            _prof_host_batch_ready = perf_counter() if step_profile_enabled else None
            if data_wait_log_enabled:
                _data_wait_s = perf_counter() - float(_prof_host_data_start or perf_counter())
            if step_profile_enabled:
                _prof_t_batch_received = _torch_for_prof.cuda.Event(enable_timing=True)
                _prof_t_fwd_start = _torch_for_prof.cuda.Event(enable_timing=True)
                _prof_t_loss_done = _torch_for_prof.cuda.Event(enable_timing=True)
                _prof_t_bwd_done = _torch_for_prof.cuda.Event(enable_timing=True)
                _prof_t_opt_done = _torch_for_prof.cuda.Event(enable_timing=True)
                _prof_wall = {"t0": _prof_host_data_start}
                _prof_t_batch_received.record()
            next_step = train_steps + 1
            if data_wait_log_enabled:
                data_wait_running_s += _data_wait_s
                data_wait_max_s = max(data_wait_max_s, _data_wait_s)
                is_slow_data_wait = _data_wait_s >= data_wait_log_threshold_s
                log_periodic_data_wait = data_wait_log_every > 0 and next_step % data_wait_log_every == 0
                if is_slow_data_wait:
                    data_wait_slow_batches += 1
                if (is_slow_data_wait or log_periodic_data_wait) and (data_wait_log_all_ranks or rank == 0):
                    data_wait_last_summary = _batch_source_wait_summary(batch)
                    logger.info(
                        "[data-wait step=%07d rank=%d] wait=%.2fs threshold=%.2fs "
                        "epoch_batch=%d datasets=%s stats=%s views=%s valid=%.3f "
                        "cameras=%s episodes=%s start_t=%s frames=%s",
                        next_step,
                        rank,
                        _data_wait_s,
                        data_wait_log_threshold_s,
                        _epoch_batches_consumed,
                        data_wait_last_summary.get("datasets", "-"),
                        data_wait_last_summary.get("stats", "-"),
                        data_wait_last_summary.get("view_counts", "-"),
                        float(data_wait_last_summary.get("view_valid_ratio", 1.0)),
                        data_wait_last_summary.get("camera_sets", "-"),
                        data_wait_last_summary.get("episodes", "-"),
                        data_wait_last_summary.get("start_t", "-"),
                        data_wait_last_summary.get("frames", "-"),
                    )
            nan_debug_step_context["step"] = next_step
            (
                all_views,
                gt_actions_raw,
                proprio,
                _,
                _,
                teacher_views,
                teacher_depth_valid_mask,
                view_valid_mask,
            ) = prepare_da3_finetune_batch(teacher_da3, batch, device)
            stats_keys = batch["action_stats_key"]
            action_loss_mask = batch.get("action_loss_mask")
            if action_loss_mask is not None:
                action_loss_mask = action_loss_mask.to(device=device, dtype=torch.bool)
            transition_loss_mask = batch.get("transition_loss_mask")
            if transition_loss_mask is not None:
                transition_loss_mask = transition_loss_mask.to(device=device, dtype=torch.bool)
            context_valid_mask = batch.get("context_valid_mask")
            if context_valid_mask is not None:
                context_valid_mask = context_valid_mask.to(device=device, dtype=torch.bool)
            gt_actions = action_normalizer.normalize(gt_actions_raw, stats_keys=stats_keys)
            if action_loss_mask is not None:
                mask = action_loss_mask
                while mask.ndim < gt_actions.ndim:
                    mask = mask.unsqueeze(-1)
                gt_actions = torch.where(mask, gt_actions, torch.zeros_like(gt_actions))
            past_action_history = None
            past_action_history_raw = batch.get("past_action_history")
            if past_action_history_raw is not None:
                past_action_history = action_normalizer.normalize(
                    past_action_history_raw.to(device=device),
                    stats_keys=stats_keys,
                )
                past_action_history_mask = batch.get("past_action_history_mask")
                if past_action_history_mask is not None:
                    input_mask = past_action_history_mask.to(device=device, dtype=torch.bool)
                    while input_mask.ndim < past_action_history.ndim:
                        input_mask = input_mask.unsqueeze(-1)
                    past_action_history = torch.where(
                        input_mask,
                        past_action_history,
                        torch.zeros_like(past_action_history),
                    )
            if proprio_normalizer is not None:
                proprio = proprio_normalizer.normalize(proprio, stats_keys=stats_keys)
                proprio = _zero_invalid_context_proprio(proprio, context_valid_mask)
            model_ref = model_engine.module if model_engine is not None else finetune_model
            debug_model = model_ref.module if hasattr(model_ref, "module") else model_ref
            if hasattr(debug_model, "_orig_mod"):
                debug_model = debug_model._orig_mod

            action_for_input = gt_actions
            out = None

            if step_profile_enabled:
                _prof_t_fwd_start.record()

            # Reshape all_views from (B, T*V, 3, H, W) to (B, T*V, 3, H, W) for both paths;
            # unified path also uses T and V separately.

            if predictor_enabled:
                # -------- Unified future-predictor path --------
                from robot.losses.unified_loss import (
                    compute_gam_forward_loss,
                    sample_H,
                )
                raw_model_unwrapped = model_ref.module if hasattr(model_ref, "module") else model_ref
                if hasattr(raw_model_unwrapped, "_orig_mod"):
                    raw_model_unwrapped = raw_model_unwrapped._orig_mod
                H = sample_H(unified_H_choices, unified_H_weights)
                if predictor_type in {"gam"}:
                    H = max(1, min(int(H), n_action_steps - 1))
                language_texts = batch.get("task_description", None) if use_language else None
                gt_depth_da3 = batch.get("gt_depth_da3")
                gt_depth_mask = batch.get("gt_depth_mask")
                gt_camera_intrinsics = batch.get("gt_camera_intrinsics")
                gt_camera_extrinsics_c2w = batch.get("gt_camera_extrinsics_c2w")
                gt_depth_scene_scale = batch.get("gt_depth_scene_scale")
                forward_profile = {} if step_profile_enabled else None
                if predictor_type in {"gam"}:
                    out = compute_gam_forward_loss(
                        student_da3=raw_model_unwrapped.student_da3,
                        teacher_da3=teacher_da3,
                        future_predictor=raw_model_unwrapped.future_predictor,
                        text_conditioner=raw_model_unwrapped.text_conditioner,
                        action_head=raw_model_unwrapped.action_head,
                        proprio_head=raw_model_unwrapped.proprio_head,
                        regularizer=regularizer,
                        all_views_norm=all_views,
                        teacher_views_norm=teacher_views,
                        teacher_depth_valid_mask=teacher_depth_valid_mask,
                        gt_actions=gt_actions,
                        proprio=proprio,
                        language_texts=language_texts,
                        T=n_action_steps,
                        V=n_views,
                        H=H,
                        lambda_action=lambda_action,
                        lambda_feat_future=lambda_feat_future,
                        lambda_proprio_future=lambda_proprio_future,
                        lambda_sigreg=lambda_sigreg,
                        use_bf16=use_bf16,
                        lambda_action_direct=lambda_action_direct,
                        lambda_action_refine=lambda_action_refine,
                        lambda_depth=regularizer.lambda_depth,
                        gt_depth_da3=None if gt_depth_da3 is None else gt_depth_da3.to(device),
                        gt_depth_mask=None if gt_depth_mask is None else gt_depth_mask.to(device),
                        depth_grad_weight=depth_grad_weight,
                        depth_decode_chunk_size=depth_decode_chunk_size,
                        teacher_depth_fallback=bool(reg_cfg.get("teacher_depth_fallback", False)),
                        skip_depth_if_no_gt=bool(reg_cfg.get("skip_depth_if_no_gt", False)),
                        lambda_path_b_deep_feat_reg=float(reg_cfg.get("lambda_path_b_deep_feat_reg", 0.0)),
                        deep_feat_reg_layer_weight_min=float(reg_cfg.get("deep_feat_reg_layer_weight_min", 0.5)),
                        lambda_feat_current=lambda_feat_current,
                        embed_dim=raw_model_unwrapped.student_da3.embed_dim,
                        deep_gradient_checkpointing=gam_backbone_deep_gradient_checkpointing,
                        deep_temporal_causal_mask=gam_deep_temporal_causal_mask,
                        feature_channel_stats=feature_channel_stats,
                        feature_loss_norm=feature_loss_norm,
                        feature_loss_type=feature_loss_type,
                        feature_loss_norm_eps=feature_loss_norm_eps,
                        feature_target_mode=feature_target_mode,
                        depth_loss_type=regularizer.depth_loss_type,
                        lambda_depth_conf=regularizer.lambda_depth_conf,
                        lambda_ray=regularizer.lambda_ray,
                        lambda_point=regularizer.lambda_point,
                        gt_camera_intrinsics=None if gt_camera_intrinsics is None else gt_camera_intrinsics.to(device),
                        gt_camera_extrinsics_c2w=None if gt_camera_extrinsics_c2w is None else gt_camera_extrinsics_c2w.to(device),
                        gt_depth_scene_scale=None if gt_depth_scene_scale is None else gt_depth_scene_scale.to(device),
                        prev_action_mask_rate=float(predictor_cfg.get("prev_action_mask_rate", 0.0)),
                        prev_action_mask_include_t0=bool(predictor_cfg.get("prev_action_mask_include_t0", False)),
                        action_loss_mask=action_loss_mask,
                        transition_loss_mask=transition_loss_mask,
                        context_valid_mask=context_valid_mask,
                        view_valid_mask=view_valid_mask,
                        past_action_history=past_action_history,
                        forward_profile=forward_profile,
                    )
                else:
                    # Legacy FuturePredictor (v1 / v2 / level0) was removed in
                    # the 2026-04-21 arch-modernize refactor. The factory would
                    # have already raised; keep this defensive guard.
                    raise RuntimeError(
                        f"predictor.type={predictor_type!r} is no longer supported. "
                        "Use predictor.type: gam (the factory rejects legacy types)."
                    )
                total_loss = out["loss_total"]
                loss_action = out["loss_action"]
                loss_feat_future = out["loss_feat_future"]
                loss_feat_current = out.get("loss_feat_current", None)
                if predictor_type in {"gam"} and loss_feat_current is not None:
                    loss_feat = loss_feat_current + loss_feat_future
                else:
                    loss_feat = out["loss_feat_past"]
                loss_sigreg = out["loss_sigreg"]
                loss_depth = out["loss_depth"]
                loss_camera = out["loss_camera"]
                # Legacy-compatible variables for downstream logging.
                action_pred = out["action_pred"]
                gt_actions = out["target_actions"]
                student_raw = out["student_raw"]
                student_feats = out.get("student_feats", None)
                teacher_raw = out["teacher_past_raw"]
                (
                    loss_action_m,
                    action_mse,
                    acc_1pct,
                    acc_5pct,
                    acc_10pct,
                ) = compute_action_metrics(action_pred, gt_actions, action_loss_mask=out.get("action_loss_mask_used"))
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    action_pred, student_raw, student_feats = model_ref(all_views, proprio=proprio, action_input=action_for_input)
                    with torch.no_grad():
                        teacher_raw = teacher_da3.encode_all_levels_raw(all_views)

                    loss_action, action_mse, acc_1pct, acc_5pct, acc_10pct = compute_action_metrics(
                        action_pred,
                        gt_actions,
                        action_loss_mask=action_loss_mask,
                    )
                    loss_feat = regularizer.feature_reg_loss(student_raw, teacher_raw)
                    total_loss = lambda_action * loss_action + regularizer.lambda_feat * loss_feat

                    # Depth and camera output regularization
                    loss_depth = student_feats[0][0].new_tensor(0.0)
                    loss_camera = student_feats[0][0].new_tensor(0.0)
                    if regularizer.lambda_depth > 0 or regularizer.lambda_camera > 0:
                        with torch.no_grad():
                            teacher_feats = teacher_da3.encode_all_levels(all_views)
                            teacher_feats_list = [teacher_feats[i] for i in range(len(teacher_feats))]
                        raw_model = model_ref.module if hasattr(model_ref, "module") else model_ref
                        if regularizer.lambda_depth > 0:
                            student_depth = raw_model.student_da3.decode_depth(student_feats)
                            # GT depth path (when the batch carries an aligned sidecar,
                            # e.g. libero_hdf5 / mimicgen with gt_depth_root set).
                            # For mixed batches (mixer OxE+MimicGen), per-sample routing:
                            # samples whose mask has any valid pixel → GT loss;
                            # samples without GT (or with fully-False mask) → teacher loss.
                            # OxE / MimicGen-only runs without GT keys remain bit-identical.
                            gt_depth_da3_batch = batch.get("gt_depth_da3") if isinstance(batch, dict) else None
                            gt_depth_mask_batch = batch.get("gt_depth_mask") if isinstance(batch, dict) else None
                            B_ = all_views.shape[0]
                            TV = all_views.shape[1]
                            V_ = n_views
                            T_ = TV // V_
                            H_ = student_depth.shape[-2]
                            W_ = student_depth.shape[-1]
                            pred_depth = student_depth.reshape(B_, T_, V_, H_, W_)
                            if gt_depth_da3_batch is not None and gt_depth_mask_batch is not None:
                                from robot.losses.unified_loss import da3_style_depth_loss
                                target_depth = gt_depth_da3_batch.to(device=device, dtype=torch.float32)
                                target_mask = gt_depth_mask_batch.to(device=device).bool()
                                # Per-sample flag: does this sample have any valid GT pixel?
                                has_gt_per_sample = target_mask.reshape(B_, -1).any(dim=1)
                                any_gt = bool(has_gt_per_sample.any())
                                any_teacher = bool((~has_gt_per_sample).any())

                                gt_loss = student_depth.new_zeros(())
                                teacher_loss = student_depth.new_zeros(())
                                if any_gt:
                                    gt_mask_routed = target_mask & has_gt_per_sample.view(B_, 1, 1, 1, 1)
                                    gt_loss, _depth_metrics = da3_style_depth_loss(
                                        pred_depth=pred_depth,
                                        target_depth=target_depth,
                                        target_mask=gt_mask_routed,
                                        grad_weight=depth_grad_weight,
                                    )
                                if any_teacher:
                                    with torch.no_grad():
                                        teacher_depth = teacher_da3.decode_depth(teacher_feats_list)
                                    teacher_pred = pred_depth
                                    teacher_tgt = teacher_depth.reshape(B_, T_, V_, H_, W_).detach()
                                    teacher_sample_mask = (~has_gt_per_sample).view(B_, 1, 1, 1, 1).float()
                                    diff_sq = (teacher_pred - teacher_tgt).pow(2) * teacher_sample_mask
                                    denom = teacher_sample_mask.sum() * T_ * V_ * H_ * W_ + 1e-8
                                    teacher_loss = diff_sq.sum() / denom
                                loss_depth = gt_loss + teacher_loss
                            else:
                                with torch.no_grad():
                                    teacher_depth = teacher_da3.decode_depth(teacher_feats_list)
                                loss_depth = regularizer.depth_reg_loss(student_depth, teacher_depth)
                            total_loss = total_loss + regularizer.lambda_depth * loss_depth
                        if regularizer.lambda_camera > 0:
                            student_pose = raw_model.student_da3.decode_camera(student_feats)
                            if student_pose is not None:
                                with torch.no_grad():
                                    teacher_pose = teacher_da3.decode_camera(teacher_feats_list)
                                loss_camera = regularizer.camera_reg_loss(student_pose, teacher_pose)
                                total_loss = total_loss + regularizer.lambda_camera * loss_camera
                loss_feat_future = None
                loss_sigreg = None

            local_loss_finite = torch.isfinite(total_loss.detach())
            global_loss_finite = local_loss_finite
            if use_deepspeed and dist.is_initialized():
                finite_flag = local_loss_finite.to(device=device, dtype=torch.int32)
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                global_loss_finite = finite_flag.bool()
            if not bool(global_loss_finite.item()):
                components = {
                    "total": total_loss,
                    "action": loss_action,
                    "feat_past": loss_feat,
                    "feat_current": loss_feat_current if predictor_enabled else None,
                    "feat_future": loss_feat_future,
                    "sigreg": loss_sigreg,
                    "depth": loss_depth,
                    "camera": loss_camera,
                    "action_mse": action_mse,
                }
                component_values = {}
                for name, value in components.items():
                    if value is None:
                        component_values[name] = None
                    else:
                        component_values[name] = float(value.detach().float().item())
                logger.error(
                    "Non-finite loss before backward at next_step=%d epoch=%d rank=%d "
                    "local_finite=%s H=%s components=%s",
                    next_step,
                    epoch,
                    rank,
                    bool(local_loss_finite.item()),
                    int(out["H"]) if predictor_enabled and isinstance(out, dict) and "H" in out else None,
                    component_values,
                )
                if nan_debug and next_step >= nan_debug_start_step:
                    _debug_log_nonfinite_forward_state(
                        model=debug_model,
                        out=out if isinstance(out, dict) else None,
                        rank=rank,
                        step=next_step,
                        logger=logger,
                        filters=nan_debug_filters,
                    )
                raise FloatingPointError(
                    f"Non-finite training loss before backward at step {next_step}"
                )

            debug_this_step = nan_debug and next_step >= nan_debug_start_step

            if step_profile_enabled:
                _prof_t_loss_done.record()

            # Whether this micro-batch closes a gradient-accumulation cycle.
            # DeepSpeed handles accum internally, so we treat every iteration
            # as a boundary on that path. DDP path uses an explicit counter.
            is_accum_boundary = (
                True
                if model_engine is not None
                else ((micro_idx_in_accum + 1) % grad_accum_steps == 0)
            )

            if model_engine is not None:
                model_engine.backward(total_loss)
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(debug_model.parameters(), clip_grad)
                if debug_this_step:
                    if nan_debug_log_stats:
                        _debug_log_finetune_tensor_stats(
                            debug_model,
                            rank=rank,
                            step=next_step,
                            where="after_backward_before_deepspeed_step",
                            logger=logger,
                            filters=nan_debug_filters,
                            check_grads=True,
                        )
                    _debug_check_finetune_tensors(
                        debug_model,
                        rank=rank,
                        step=next_step,
                        where="after_backward_before_deepspeed_step",
                        logger=logger,
                        filters=nan_debug_filters,
                        check_grads=True,
                    )
                if step_profile_enabled:
                    _prof_t_bwd_done.record()
                model_engine.step()
                if step_profile_enabled:
                    _prof_t_opt_done.record()
            else:
                # DDP path with manual gradient accumulation. Backward
                # accumulates gradients into .grad on every micro-batch
                # (scaled by 1/N so the summed grad equals one big-batch
                # grad). Optimizer.step + zero_grad fire only on the accum
                # boundary, after which the accum counter resets to 0.
                loss_for_backward = (
                    total_loss / float(grad_accum_steps)
                    if grad_accum_steps > 1
                    else total_loss
                )
                sync_context = (
                    finetune_model.no_sync()
                    if ddp_accum_no_sync and not is_accum_boundary and hasattr(finetune_model, "no_sync")
                    else nullcontext()
                )
                with sync_context:
                    scaler.scale(loss_for_backward).backward()
                if not is_accum_boundary:
                    micro_idx_in_accum += 1
                    if step_profile_enabled:
                        _prof_t_bwd_done.record()
                        _prof_t_opt_done.record()
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), clip_grad)
                if debug_this_step:
                    if nan_debug_log_stats:
                        _debug_log_finetune_tensor_stats(
                            debug_model,
                            rank=rank,
                            step=next_step,
                            where="after_backward_before_optimizer_step",
                            logger=logger,
                            filters=nan_debug_filters,
                            check_grads=True,
                        )
                    _debug_check_finetune_tensors(
                        debug_model,
                        rank=rank,
                        step=next_step,
                        where="after_backward_before_optimizer_step",
                        logger=logger,
                        filters=nan_debug_filters,
                        check_grads=True,
                    )
                if step_profile_enabled:
                    _prof_t_bwd_done.record()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if step_profile_enabled:
                    _prof_t_opt_done.record()
                micro_idx_in_accum = 0
            # Host-time checkpoints to break down what python_residual is
            # actually made of (CUDA Events only cover GPU work; the residual
            # is whatever the host does after the optimizer step + sync waits
            # on the next iteration). Recorded only when profiling is enabled.
            if step_profile_enabled:
                _prof_host_post_opt = __import__("time").perf_counter()
            if ema_tracker is not None:
                ema_tracker.update(model_ref, next_step)
            if step_profile_enabled:
                _prof_host_post_ema = __import__("time").perf_counter()
            if debug_this_step:
                if nan_debug_log_stats:
                    _debug_log_finetune_tensor_stats(
                        debug_model,
                        rank=rank,
                        step=next_step,
                        where="after_optimizer_step",
                        logger=logger,
                        filters=nan_debug_filters,
                        check_grads=False,
                    )
                _debug_check_finetune_tensors(
                    debug_model,
                    rank=rank,
                    step=next_step,
                    where="after_optimizer_step",
                    logger=logger,
                    filters=nan_debug_filters,
                    check_grads=False,
                )

            train_steps += 1
            # Defragment CUDA caching allocator every N steps. Progressive
            # slowdown of backward+optimizer on long DeepSpeed ZeRO-2 runs is
            # often memory-allocator fragmentation: cuMalloc/cuFree latency
            # grows with the free-block count, which dominates per-step time
            # by step ~1000. Explicit empty_cache returns all free blocks to
            # the driver, briefly pauses, then flattens subsequent cost. Only
            # fires when the training loop is otherwise idle-ish (after
            # optimizer step, before next data fetch).
            cuda_defrag_every = int(os.environ.get("DA3_CUDA_DEFRAG_EVERY", "0"))
            if cuda_defrag_every > 0 and next_step % cuda_defrag_every == 0:
                torch.cuda.empty_cache()
            if step_profile_enabled and next_step % step_profile_every == 0:
                # Sync all the CUDA events and print the per-phase breakdown for
                # this step. rank 0 only to avoid spam.
                try:
                    _prof_t_opt_done.synchronize()
                    dt_loader_wait = (_prof_host_batch_ready - _prof_host_data_start) * 1000.0
                    dt_prep = _prof_t_batch_received.elapsed_time(_prof_t_fwd_start)
                    dt_fwd  = _prof_t_fwd_start.elapsed_time(_prof_t_loss_done)
                    dt_bwd  = _prof_t_loss_done.elapsed_time(_prof_t_bwd_done)
                    dt_opt  = _prof_t_bwd_done.elapsed_time(_prof_t_opt_done)
                    dt_wall_ms = (perf_counter() - _prof_wall["t0"]) * 1000
                    dt_python_overhead = dt_wall_ms - (dt_loader_wait + dt_prep + dt_fwd + dt_bwd + dt_opt)
                    # Host-time chunks within the residual.
                    dt_ema_host = (_prof_host_post_ema - _prof_host_post_opt) * 1000.0
                    _prof_host_post_loglog = __import__("time").perf_counter()
                    dt_post_opt_to_logblock = (_prof_host_post_loglog - _prof_host_post_opt) * 1000.0
                    if rank == 0:
                        logger.info(
                            "[prof step=%07d] wall=%.1fms  loader_wait=%.1fms  prep_to_fwd=%.1fms  fwd+loss=%.1fms  "
                            "backward=%.1fms  optimizer=%.1fms  python_residual=%.1fms  "
                            "(ema_host=%.1fms post_opt_to_logblock=%.1fms)",
                            next_step, dt_wall_ms, dt_loader_wait, dt_prep, dt_fwd, dt_bwd, dt_opt, dt_python_overhead,
                            dt_ema_host, dt_post_opt_to_logblock,
                        )
                        if predictor_enabled and isinstance(out, dict):
                            fprof = out.get("forward_profile") or {}
                            if fprof:
                                logger.info(
                                    "[prof-fwd step=%07d] H=%s total=%.1fms "
                                    "student_shallow=%.1f teacher_shallow=%.1f text=%.1f "
                                    "cond=%.1f predictor=%.1f action_direct=%.1f "
                                    "deep_total=%.1f deep_prep=%.1f deep_blocks=%.1f "
                                    "deep_dpt=%.1f deep_final=%.1f action_refine=%.1f "
                                    "proprio=%.1f feature=%.1f teacher_depth=%.1f "
                                    "depth_loss=%.1f deep_feat_reg=%.1f total_loss=%.1f",
                                    next_step,
                                    int(out.get("H", -1)),
                                    float(fprof.get("fwd_profile_total_ms", 0.0)),
                                    float(fprof.get("student_shallow_ms", 0.0)),
                                    float(fprof.get("teacher_shallow_ms", 0.0)),
                                    float(fprof.get("text_encode_ms", 0.0)),
                                    float(fprof.get("condition_prep_ms", 0.0)),
                                    float(fprof.get("predictor_ms", 0.0)),
                                    float(fprof.get("action_direct_ms", 0.0)),
                                    float(fprof.get("deep_propagate_total_ms", 0.0)),
                                    float(fprof.get("deep_prep_ms", 0.0)),
                                    float(fprof.get("deep_blocks_ms", 0.0)),
                                    float(fprof.get("deep_dpt_ms", 0.0)),
                                    float(fprof.get("deep_final_ms", 0.0)),
                                    float(fprof.get("action_refine_loss_ms", 0.0)),
                                    float(fprof.get("proprio_loss_ms", 0.0)),
                                    float(fprof.get("feature_loss_ms", 0.0)),
                                    float(fprof.get("teacher_depth_decode_ms", 0.0)),
                                    float(fprof.get("depth_target_loss_ms", 0.0)),
                                    float(fprof.get("deep_feat_reg_ms", 0.0)),
                                    float(fprof.get("loss_total_ms", 0.0)),
                                )
                except Exception as _prof_err:
                    if rank == 0:
                        logger.warning("step_profile breakdown failed: %s", _prof_err)
            if step_profile_enabled:
                _prof_host_pre_items = __import__("time").perf_counter()
            running_total += float(total_loss.detach().item())
            running_action += float(loss_action.detach().item())
            running_feat += float(loss_feat.detach().item())
            if step_profile_enabled:
                _prof_host_post_items = __import__("time").perf_counter()
                if next_step % step_profile_every == 0 and rank == 0:
                    dt_items_host = (_prof_host_post_items - _prof_host_pre_items) * 1000.0
                    logger.info(
                        "[prof step=%07d] items_host (3 .item() syncs)=%.1fms",
                        next_step, dt_items_host,
                    )
            # Per-term depth breakdown. In the gam path `out` is a dict
            # with L_D / L_M / L_P components; in the legacy path `loss_depth`
            # exists as a local tensor. Prefer the dict when present, otherwise
            # fall back to the local tensor.
            if predictor_enabled and isinstance(out, dict):
                od = out.get("loss_depth")
                orr = out.get("loss_ray")
                op = out.get("loss_point")
                if od is not None: running_depth += float(od.detach().item())
                if orr is not None: running_ray += float(orr.detach().item())
                if op is not None: running_point += float(op.detach().item())
            else:
                if loss_depth is not None:
                    running_depth += float(loss_depth.detach().item())
            if not hasattr(regularizer, '_running_depth'):
                regularizer._running_depth = 0.0
                regularizer._running_camera = 0.0
            regularizer._running_depth += float(loss_depth.detach().item())
            regularizer._running_camera += float(loss_camera.detach().item())
            log_steps += 1

            # Per-dataset metric accumulation (near-zero overhead)
            if step_profile_enabled:
                _prof_host_pre_per_ds = __import__("time").perf_counter()
            if rank == 0 and stats_keys is not None:
                with torch.no_grad():
                    per_sample_err = F.l1_loss(
                        action_pred.detach(), gt_actions.detach(), reduction='none'
                    )
                    metric_action_loss_mask = (
                        out.get("action_loss_mask_used")
                        if isinstance(out, dict) and out.get("action_loss_mask_used", None) is not None
                        else action_loss_mask
                    )
                    if metric_action_loss_mask is not None:
                        per_sample_mask = metric_action_loss_mask
                        while per_sample_mask.ndim < per_sample_err.ndim:
                            per_sample_mask = per_sample_mask.unsqueeze(-1)
                        per_sample_mask = per_sample_mask.to(device=per_sample_err.device, dtype=per_sample_err.dtype)
                        per_sample_num = (per_sample_err * per_sample_mask).sum(dim=tuple(range(1, per_sample_err.ndim)))
                        per_sample_den = per_sample_mask.expand_as(per_sample_err).sum(
                            dim=tuple(range(1, per_sample_err.ndim))
                        ).clamp_min(1.0)
                        per_sample_l1 = per_sample_num / per_sample_den
                    else:
                        per_sample_l1 = per_sample_err.mean(dim=tuple(range(1, action_pred.ndim)))
                for i, key in enumerate(stats_keys):
                    if i < len(per_sample_l1):
                        per_ds_accum[key]["ae_l1"].append(per_sample_l1[i].item())
                        per_ds_accum[key]["count"] += 1
            if step_profile_enabled:
                _prof_host_post_per_ds = __import__("time").perf_counter()
                _prof_host_pre_log_block = _prof_host_post_per_ds

            if train_steps % log_every == 0 and rank == 0:
                avg_total = running_total / log_steps
                avg_action = running_action / log_steps
                avg_feat = running_feat / log_steps
                elapsed = time() - start_time

                cur_lr_bb = opt.param_groups[0]["lr"]
                cur_lr_head = opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else cur_lr_bb
                logger.info(
                    "[step=%07d] total=%.4f action_l1=%.4f feat=%.4f "
                    "depth=%.4f ray=%.4f point=%.4f "
                    "rel@1/5/10=%.3f/%.3f/%.3f mse=%.5f s/step=%.1f lr=%.1e/%.1e",
                    train_steps,
                    avg_total,
                    avg_action,
                    avg_feat,
                    running_depth / log_steps,
                    running_ray / log_steps,
                    running_point / log_steps,
                    acc_1pct.item(),
                    acc_5pct.item(),
                    acc_10pct.item(),
                    action_mse.item(),
                    elapsed / log_steps,
                    cur_lr_bb,
                    cur_lr_head,
                )

                if wandb_run is not None:
                    import wandb

                    logged_action_mask = out.get("action_loss_mask_used") if isinstance(out, dict) else action_loss_mask
                    norm_metrics = compute_action_detail_metrics(
                        action_pred.detach(), gt_actions.detach(), action_loss_mask=logged_action_mask
                    )
                    raw_metrics = compute_raw_action_metrics(
                        action_pred.detach(),
                        gt_actions.detach(),
                        action_normalizer,
                        stats_keys=stats_keys,
                        action_loss_mask=logged_action_mask,
                    )
                    wandb_log = {
                        "train/loss_total": avg_total,
                        "train/l1_norm": norm_metrics["l1"],
                        "train/mse_norm": norm_metrics["mse"],
                        "train/r2_norm": norm_metrics["r2"],
                        "train/l1_raw": raw_metrics["l1"],
                        "train/mse_raw": raw_metrics["mse"],
                        "train/r2_raw": raw_metrics["r2"],
                        "train/rel_acc_1pct": norm_metrics["rel_acc_1pct"],
                        "train/rel_acc_5pct": norm_metrics["rel_acc_5pct"],
                        "train/rel_acc_10pct": norm_metrics["rel_acc_10pct"],
                        "train/seq_rel_acc_5pct": norm_metrics["seq_acc_5pct"],
                        "train/seq_rel_acc_10pct": norm_metrics["seq_acc_10pct"],
                        "train/translation_l1_norm": norm_metrics["translation_mae"],
                        "train/rotation_l1_norm": norm_metrics["rotation_mae"],
                        "train/gripper_l1_norm": norm_metrics["gripper_mae"],
                        "train/translation_l1_raw": raw_metrics["translation_mae"],
                        "train/rotation_l1_raw": raw_metrics["rotation_mae"],
                        "train/gripper_l1_raw": raw_metrics["gripper_mae"],
                    }
                    if data_wait_log_enabled:
                        wandb_log["debug/data_wait_s_mean"] = data_wait_running_s / max(log_steps, 1)
                        wandb_log["debug/data_wait_s_max"] = data_wait_max_s
                        wandb_log["debug/data_wait_slow_batches"] = int(data_wait_slow_batches)
                        if data_wait_last_summary:
                            wandb_log["debug/data_wait_unique_datasets"] = int(
                                data_wait_last_summary.get("unique_datasets", 0)
                            )
                            wandb_log["debug/data_wait_view_valid_ratio"] = float(
                                data_wait_last_summary.get("view_valid_ratio", 1.0)
                            )
                            wandb_log["debug/data_wait_max_real_views"] = int(
                                data_wait_last_summary.get("max_real_views", 0)
                            )
                    # `avg_feat` and `regularizer._running_depth` are both
                    # populated under Path A AND Path B; they refer to
                    # *different* losses depending on the path:
                    #   Path A: avg_feat = FeatureRegularizer.feature_reg_loss
                    #     (multi-level student vs teacher distillation)
                    #     _running_depth = regularizer.depth_reg_loss (or
                    #     da3_style_depth_loss when sidecar present)
                    #   Path B: avg_feat = loss_feat_future (predictor's
                    #     H+1 future visual token vs teacher's H+1 token)
                    #     _running_depth = da3_style_depth_loss
                    # Old wandb keys reused the same names which was
                    # misleading; split them so each chart is labelled with
                    # the actual loss it tracks.
                    if predictor_enabled:
                        wandb_log["reg/path_b_future_feat_loss"] = avg_feat
                        wandb_log["reg/path_b_lambda_feat_future"] = float(lambda_feat_future)
                        wandb_log["reg/path_b_depth_loss"] = regularizer._running_depth / log_steps
                        wandb_log["reg/lambda_depth"] = regularizer.lambda_depth
                        # Path B deep multi-level feature distillation (off
                        # unless `regularization.lambda_path_b_deep_feat_reg > 0`).
                        _deep_fr_lambda = float(reg_cfg.get("lambda_path_b_deep_feat_reg", 0.0))
                        if _deep_fr_lambda > 0.0 and out is not None and out.get("loss_deep_feat_reg") is not None:
                            wandb_log["reg/path_b_deep_feat_reg_loss"] = float(
                                out["loss_deep_feat_reg"].detach().float().item()
                            )
                            wandb_log["reg/path_b_lambda_deep_feat_reg"] = _deep_fr_lambda
                    else:
                        wandb_log["reg/path_a_feature_reg_loss"] = avg_feat
                        wandb_log["reg/path_a_lambda_feat"] = regularizer.lambda_feat
                        wandb_log["reg/path_a_depth_reg_loss"] = regularizer._running_depth / log_steps
                        wandb_log["reg/path_a_camera_reg_loss"] = regularizer._running_camera / log_steps
                        wandb_log["reg/lambda_depth"] = regularizer.lambda_depth
                        wandb_log["reg/lambda_camera"] = regularizer.lambda_camera
                    if predictor_enabled:
                        # Log unified-mode specific losses.
                        wandb_log["unified/feat_future_l2"] = (
                            float(loss_feat_future.detach().float().item()) if loss_feat_future is not None else 0.0
                        )
                        if out.get("loss_feat_current", None) is not None:
                            wandb_log["unified/feat_current_l2"] = float(out["loss_feat_current"].detach().float().item())
                        for _metric_key, _wandb_key in (
                            ("loss_feat_future_raw", "unified/feat_future_l2_raw"),
                            ("loss_feat_current_raw", "unified/feat_current_l2_raw"),
                            ("loss_feat_future_norm", "unified/feat_future_l2_norm"),
                            ("loss_feat_current_norm", "unified/feat_current_l2_norm"),
                            ("loss_feat_future_copy_raw", "unified/feat_future_copy_current_l2_raw"),
                            ("loss_feat_future_copy_norm", "unified/feat_future_copy_current_l2_norm"),
                        ):
                            if out.get(_metric_key, None) is not None:
                                wandb_log[_wandb_key] = float(out[_metric_key].detach().float().item())
                        if out.get("loss_feat_future_copy_raw", None) is not None:
                            _copy_raw = out["loss_feat_future_copy_raw"].detach().float().clamp_min(1e-12)
                            _future_raw = out.get("loss_feat_future_raw", loss_feat_future).detach().float()
                            wandb_log["unified/feat_future_vs_copy_ratio_raw"] = float((_future_raw / _copy_raw).item())
                        if out.get("loss_feat_future_copy_norm", None) is not None:
                            _copy_norm = out["loss_feat_future_copy_norm"].detach().float().clamp_min(1e-12)
                            _future_norm = out.get("loss_feat_future_norm", loss_feat_future).detach().float()
                            wandb_log["unified/feat_future_vs_copy_ratio_norm"] = float((_future_norm / _copy_norm).item())
                        if "feature_loss_norm" in out:
                            wandb_log["unified/feature_loss_uses_channel_stats"] = (
                                1 if str(out["feature_loss_norm"]).lower() in {"channel", "channel_stats", "stats"} else 0
                            )
                        if "feature_loss_type" in out:
                            wandb_log["unified/feature_loss_is_l1"] = (
                                1 if str(out["feature_loss_type"]).lower() in {"l1", "mae"} else 0
                            )
                        if "feature_target_mode" in out:
                            wandb_log["unified/feature_target_is_delta"] = (
                                1 if str(out["feature_target_mode"]).lower() == "delta" else 0
                            )
                        if out.get("loss_action_direct", None) is not None:
                            wandb_log["unified/action_direct_l1"] = float(
                                out["loss_action_direct"].detach().float().item()
                            )
                        if out.get("loss_action_refine", None) is not None:
                            wandb_log["unified/action_refine_l1"] = float(
                                out["loss_action_refine"].detach().float().item()
                            )
                        wandb_log["unified/action_direct_weight"] = float(lambda_action_direct)
                        wandb_log["unified/action_refine_weight"] = float(lambda_action_refine)
                        if out.get("loss_proprio_future_direct", None) is not None:
                            wandb_log["unified/proprio_direct_l1"] = float(
                                out["loss_proprio_future_direct"].detach().float().item()
                            )
                        if out.get("loss_proprio_future_head", None) is not None:
                            wandb_log["unified/proprio_refine_l1"] = float(
                                out["loss_proprio_future_head"].detach().float().item()
                            )
                        if out.get("predicted_next_visual_tokens", None) is not None:
                            wandb_log["unified/ar_dense_steps"] = int(out["predicted_next_visual_tokens"].shape[1])
                        wandb_log["unified/sigreg"] = (
                            float(loss_sigreg.detach().float().item()) if loss_sigreg is not None else 0.0
                        )
                        wandb_log["unified/lambda_feat_future"] = lambda_feat_future
                        wandb_log["unified/lambda_feat_current"] = lambda_feat_current
                        wandb_log["unified/lambda_sigreg"] = lambda_sigreg
                        wandb_log["unified/H"] = int(out["H"])
                        if out.get("predicted_sequence_steps", None) is not None:
                            wandb_log["unified/predicted_steps"] = int(out["predicted_sequence_steps"])
                        if out.get("action_loss_mask_used", None) is not None:
                            wandb_log["unified/action_loss_valid_ratio"] = float(
                                out["action_loss_mask_used"].detach().float().mean().item()
                            )
                        if out.get("transition_loss_mask_used", None) is not None:
                            wandb_log["unified/transition_loss_valid_ratio"] = float(
                                out["transition_loss_mask_used"].detach().float().mean().item()
                            )
                        if out.get("context_valid_mask_used", None) is not None:
                            wandb_log["unified/context_valid_ratio"] = float(
                                out["context_valid_mask_used"].detach().float().mean().item()
                            )
                        wandb_log["unified/monitor_timesteps"] = int(out["T"])
                        if "predicted_sequence_steps" in out:
                            wandb_log["unified/deep_sequence_steps"] = int(out["predicted_sequence_steps"])
                            wandb_log["unified/deep_gradient_checkpointing"] = int(
                                bool(out.get("deep_gradient_checkpointing", False))
                            )
                            wandb_log["unified/deep_temporal_causal_mask"] = int(
                                bool(out.get("deep_temporal_causal_mask", False))
                            )
                        for depth_key, depth_value in out.get("depth_metrics", {}).items():
                            wandb_log[f"unified/{depth_key}"] = float(depth_value.item())
                    add_named_metrics(
                        wandb_log, "train/per_dim_l1_norm", ACTION_DIM_NAMES, norm_metrics["per_dim_mae"]
                    )
                    add_named_metrics(
                        wandb_log, "train/per_dim_mse_norm", ACTION_DIM_NAMES, norm_metrics["per_dim_mse"]
                    )
                    add_named_metrics(
                        wandb_log, "train/per_dim_r2_norm", ACTION_DIM_NAMES, norm_metrics["per_dim_r2"]
                    )
                    add_named_metrics(
                        wandb_log, "train/per_dim_l1_raw", ACTION_DIM_NAMES, raw_metrics["per_dim_mae"]
                    )
                    add_named_metrics(
                        wandb_log, "train/per_dim_mse_raw", ACTION_DIM_NAMES, raw_metrics["per_dim_mse"]
                    )
                    add_named_metrics(
                        wandb_log, "train/per_dim_r2_raw", ACTION_DIM_NAMES, raw_metrics["per_dim_r2"]
                    )
                    for level_idx, (student_level, teacher_level) in enumerate(zip(student_raw, teacher_raw)):
                        with torch.no_grad():
                            student_cmp = torch.cat([student_level[:, :, :1], student_level[:, :, 2:]], dim=2)
                            cos_sim = F.cosine_similarity(
                                student_cmp.flatten(2), teacher_level.flatten(2), dim=-1
                            ).mean()
                        wandb_log[f"reg/cosine_sim_level{level_idx}"] = cos_sim.item()
                    wandb_log["lr/backbone"] = cur_lr_bb
                    wandb_log["lr/head"] = cur_lr_head
                    # Log embedding norms
                    raw_m = model_ref.module if hasattr(model_ref, "module") else model_ref
                    if hasattr(raw_m, "_orig_mod"):
                        raw_m = raw_m._orig_mod
                    sd = raw_m.student_da3
                    if sd.action_token is not None:
                        wandb_log["embed/action_token_norm"] = sd.action_token.data.norm().item()
                    if sd.action_timestep_embed is not None:
                        wandb_log["embed/timestep_embed_norm"] = sd.action_timestep_embed.data[:, :n_action_steps].norm().item()
                    if sd.action_view_embed is not None:
                        wandb_log["embed/view_embed_norm"] = sd.action_view_embed.data[:, :n_views].norm().item()
                    if sd.temporal_embed is not None:
                        wandb_log["embed/temporal_embed_norm"] = sd.temporal_embed.data[:, :n_action_steps].norm().item()
                    if (vis_every > 0 and train_steps % vis_every == 0) or train_steps == 1:
                        if predictor_enabled:
                            # Always log the exact post-augmentation RGB
                            # going into the encoder so input pipeline issues
                            # (wrong camera order, flipped frames, broken
                            # crop/color, wrong task prompt, wrong stats key)
                            # surface in the first vis step.
                            try:
                                log_training_input_images(
                                    batch,
                                    train_steps,
                                    log_dict=wandb_log,
                                    prefix="inputs",
                                    num_samples=2,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Training input image visualization failed: %s", exc
                                )
                            try:
                                log_depth_media = (
                                    log_unified_depth_visualization
                                    and unified_depth_vis_every > 0
                                    and (train_steps % unified_depth_vis_every == 0 or train_steps == 1)
                                )
                                past_slots = out.get("past_slots", None)
                                predicted_next_visual_tokens = out.get("predicted_next_visual_tokens", None)
                                if (
                                    log_depth_media
                                    and predictor_type in {"gam"}
                                ):
                                    log_gam_future_visualizations(
                                        sd,
                                        batch,
                                        device,
                                        train_steps,
                                        wandb_run=wandb_run,
                                        visual_tokens=out.get("deep_visual_tokens", None),
                                        action_tokens=out.get("deep_action_tokens", None),
                                        target_depth=out.get("depth_target", None),
                                        target_mask=out.get("depth_target_mask", None),
                                        target_label=out.get("depth_target_label", None),
                                        start_timestep=int(out.get("predicted_sequence_start_timestep", 0)),
                                        current_timestep=out.get("deep_current_timestep", int(out.get("H", 1)) - 1),
                                        prefix="unified",
                                        log_dict=wandb_log,
                                    )
                                elif (
                                    log_depth_media
                                    and past_slots is not None
                                    and predicted_next_visual_tokens is not None
                                ):
                                    pred_slots = torch.cat(
                                        [past_slots.detach(), predicted_next_visual_tokens.detach()],
                                        dim=1,
                                    )
                                    log_unified_future_visualizations(
                                        sd,
                                        batch,
                                        device,
                                        train_steps,
                                        wandb_run=wandb_run,
                                        pred_slots=pred_slots,
                                        direct_current_features=out.get("student_feats", None),
                                        H=int(out["H"]),
                                        start_timestep=0,
                                        target_depth=out.get("depth_target", None),
                                        target_mask=out.get("depth_target_mask", None),
                                        target_label=out.get("depth_target_label", None),
                                        prefix="unified",
                                        log_dict=wandb_log,
                                    )
                                    del pred_slots

                                monitor_h = int(out["H"])
                                wandb_log["unified_traj/observed_steps"] = monitor_h
                                wandb_log["unified_traj/predicted_action_tokens"] = int(action_pred.shape[1])
                                if action_pred.ndim >= 4:
                                    wandb_log["unified_traj/predicted_flat_actions"] = int(
                                        action_pred.shape[1] * action_pred.shape[2]
                                    )
                                else:
                                    wandb_log["unified_traj/predicted_flat_actions"] = int(action_pred.shape[1])
                                gt_actions_window_norm = gt_actions.detach()
                                gt_actions_denorm = action_normalizer.denormalize(
                                    gt_actions_window_norm, stats_keys=stats_keys
                                )
                                pred_actions_denorm = action_normalizer.denormalize(
                                    action_pred.detach(), stats_keys=stats_keys
                                )
                                log_action_trajectory(
                                    gt_actions_denorm,
                                    pred_actions_denorm,
                                    train_steps,
                                    wandb_run=wandb_run,
                                    batch_idx=0,
                                    prefix="unified_traj",
                                    caption_note="Loss window only: current and next action tokens.",
                                    log_dict=wandb_log,
                                )
                                del gt_actions_denorm, pred_actions_denorm
                            except Exception:
                                logger.exception("Unified monitor visualization failed")
                        else:
                            # --- Monitor noact (learnable token only) on same batch ---
                            with torch.no_grad():
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                                    noact_pred, _, _ = model_ref(all_views, proprio=proprio, action_input=None)
                                na_l1, na_mse, na_a1, na_a5, na_a10 = compute_action_metrics(
                                    noact_pred,
                                    gt_actions,
                                    action_loss_mask=logged_action_mask,
                                )
                                wandb_log["monitor/noact_l1"] = na_l1.item()
                                wandb_log["monitor/noact_mse"] = na_mse.item()
                                wandb_log["monitor/noact_rel5"] = na_a5.item()
                                wandb_log["monitor/noact_rel10"] = na_a10.item()
                                # Also log AE (with GT action) for comparison
                                wandb_log["monitor/ae_l1"] = norm_metrics["l1"]
                                wandb_log["monitor/ae_mse"] = norm_metrics["mse"]
                                wandb_log["monitor/ae_rel5"] = norm_metrics["rel_acc_5pct"]
                                wandb_log["monitor/ae_rel10"] = norm_metrics["rel_acc_10pct"]
                                del noact_pred

                        add_indexed_metrics(wandb_log, "train/timestep_l1_norm", norm_metrics["timestep_mae"])
                        add_indexed_metrics(
                            wandb_log, "train/timestep_translation_l1_norm", norm_metrics["timestep_translation_mae"]
                        )
                        add_indexed_metrics(
                            wandb_log, "train/timestep_rotation_l1_norm", norm_metrics["timestep_rotation_mae"]
                        )
                        add_indexed_metrics(
                            wandb_log, "train/timestep_gripper_l1_norm", norm_metrics["timestep_gripper_mae"]
                        )
                        add_indexed_metrics(
                            wandb_log, "train/timestep_l1_raw", raw_metrics["timestep_mae"]
                        )
                        add_indexed_metrics(
                            wandb_log, "train/timestep_translation_l1_raw", raw_metrics["timestep_translation_mae"]
                        )
                        add_indexed_metrics(
                            wandb_log, "train/timestep_rotation_l1_raw", raw_metrics["timestep_rotation_mae"]
                        )
                        add_indexed_metrics(
                            wandb_log, "train/timestep_gripper_l1_raw", raw_metrics["timestep_gripper_mae"]
                        )

                        # Per-dataset breakdown as wandb.Table
                        if per_ds_accum:
                            ds_table = wandb.Table(
                                columns=["dataset", "n_samples", "ae_l1_mean", "ae_l1_std"]
                            )
                            for ds_name in sorted(per_ds_accum):
                                ds_vals = per_ds_accum[ds_name]
                                if ds_vals["ae_l1"]:
                                    ds_table.add_data(
                                        ds_name,
                                        ds_vals["count"],
                                        float(np.mean(ds_vals["ae_l1"])),
                                        float(np.std(ds_vals["ae_l1"])),
                                    )
                            wandb_log["per_dataset/train_breakdown"] = ds_table
                            per_ds_accum.clear()

                    wandb.log(wandb_log, step=train_steps)

                running_total = running_action = running_feat = 0.0
                running_depth = running_ray = running_point = 0.0
                data_wait_running_s = 0.0
                data_wait_max_s = 0.0
                data_wait_slow_batches = 0
                data_wait_last_summary = {}
                regularizer._running_depth = 0.0
                regularizer._running_camera = 0.0
                log_steps = 0
                start_time = time()
            if step_profile_enabled:
                _prof_host_post_log_block = __import__("time").perf_counter()
                if next_step % step_profile_every == 0 and rank == 0:
                    logger.info(
                        "[prof step=%07d] host_tail per_dataset=%.1fms log_block=%.1fms post_opt_to_log_end=%.1fms",
                        next_step,
                        (_prof_host_post_per_ds - _prof_host_pre_per_ds) * 1000.0,
                        (_prof_host_post_log_block - _prof_host_pre_log_block) * 1000.0,
                        (_prof_host_post_log_block - _prof_host_post_opt) * 1000.0,
                    )

            vis_due = (vis_every > 0 and train_steps % vis_every == 0) or train_steps == 1
            if vis_due and rank == 0 and wandb_run is not None and not predictor_enabled:
                with torch.no_grad():
                    try:
                        student_level0 = torch.cat([student_raw[0][:, :, :1], student_raw[0][:, :, 2:]], dim=2)
                        student_level0 = student_level0.reshape(
                            student_level0.shape[0] * student_level0.shape[1],
                            student_level0.shape[2],
                            student_level0.shape[3],
                        ).float()
                        # Cast teacher to float32 for visualization
                        teacher_da3.float()
                        log_da3_visualizations(
                            teacher_da3,
                            batch,
                            device,
                            train_steps,
                            wandb_run,
                            pred_features=student_level0[:, 1:],
                            pred_cls=student_level0[:, 0],
                        )
                        if use_bf16:
                            teacher_da3.to(torch.bfloat16)
                    except Exception as exc:
                        logger.warning("Fine-tune visualization failed: %s", exc)
                        if use_bf16:
                            teacher_da3.to(torch.bfloat16)

                    try:
                        gt_actions_denorm = action_normalizer.denormalize(gt_actions.detach(), stats_keys=stats_keys)
                        # Dedicated AE forward pass: force GT action injection and bypass stochastic
                        # sampling. Training pred_action uses stochastic GT injection
                        # (action_input_rate=0.2), so it is mostly a noact visualization mix.
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            ae_pred_vis, _, _ = model_ref(
                                all_views, proprio=proprio,
                                action_input=gt_actions, force_action_input=True,
                            )
                        ae_denorm = action_normalizer.denormalize(ae_pred_vis.detach(), stats_keys=stats_keys)
                        log_robot_debug_batch(
                            batch,
                            gt_actions_denorm,
                            ae_denorm,
                            train_steps,
                            wandb_run=wandb_run,
                            batch_idx=0,
                            prefix="debug/train",
                        )
                        # AE trajectory (with GT action injection)
                        log_action_trajectory(
                            gt_actions_denorm,
                            ae_denorm,
                            train_steps,
                            wandb_run=wandb_run, batch_idx=0,
                            prefix="ae",
                        )
                        del ae_pred_vis
                        # noact trajectory (learnable token only)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            noact_pred_vis, _, _ = model_ref(all_views, proprio=proprio, action_input=None)
                        noact_denorm = action_normalizer.denormalize(noact_pred_vis.detach(), stats_keys=stats_keys)
                        log_action_trajectory(
                            gt_actions_denorm,
                            noact_denorm,
                            train_steps,
                            wandb_run=wandb_run, batch_idx=0,
                            prefix="noact",
                        )
                        del noact_pred_vis, noact_denorm, ae_denorm
                    except Exception as exc:
                        logger.warning("Action trajectory visualization failed: %s", exc)

                    # Camera pose visualization (teacher vs student). Let errors surface.
                    if regularizer.lambda_camera > 0 or True:  # always log if cam_dec exists
                        with torch.no_grad():
                            if not isinstance(student_feats, list) or len(student_feats) == 0:
                                raise RuntimeError("student_feats missing for camera vis")
                            _raw = model_ref.module if hasattr(model_ref, "module") else model_ref
                            s_pose = _raw.student_da3.decode_camera(student_feats)
                            if s_pose is not None:
                                _teacher_was_bf16 = next(teacher_da3.parameters()).dtype == torch.bfloat16
                                if _teacher_was_bf16:
                                    teacher_da3.float()
                                teacher_feats_vis = teacher_da3.encode_all_levels(all_views.float())
                                teacher_feats_list = [teacher_feats_vis[i] for i in range(len(teacher_feats_vis))]
                                t_pose = teacher_da3.decode_camera(teacher_feats_list)
                                if _teacher_was_bf16:
                                    teacher_da3.to(torch.bfloat16)
                                log_camera_visualization(
                                    t_pose, s_pose, train_steps,
                                    wandb_run=wandb_run,
                                    n_views=int(cfg.get("da3_finetune", {}).get("n_views", 2)),
                                )

                torch.cuda.empty_cache()

            closed_loop_due_profiles = []
            if closed_loop_enabled:
                for _profile in closed_loop_profiles:
                    _profile_eval_every = int(_profile.get("eval_every", eval_every))
                    _first_eval_step = _profile.get("first_eval_step")
                    _first_due = False
                    if _first_eval_step is not None:
                        _first_eval_step = int(_first_eval_step)
                        _first_due = _first_eval_step > 0 and int(train_steps) == _first_eval_step
                    _periodic_due = (
                        _profile_eval_every > 0
                        and train_steps % _profile_eval_every == 0
                        and train_steps > 0
                    )
                    if _first_due or _periodic_due:
                        closed_loop_due_profiles.append(_profile)
            closed_loop_due = bool(closed_loop_due_profiles)
            checkpoint_due = scheduled_checkpointing and train_steps % ckpt_every == 0
            checkpoint_saved_this_step = False
            periodic_eval_due = eval_every > 0 and train_steps % eval_every == 0
            predictor_eval_due = bool(predictor_enabled and periodic_eval_due)
            legacy_eval_due = bool(
                (not predictor_enabled)
                and periodic_eval_due
                and (eval_ae or eval_noact)
            )
            any_eval_due = bool(closed_loop_due or predictor_eval_due or legacy_eval_due)
            stop_reason = None
            remaining_seconds = _slurm_remaining_seconds()
            check_termination_signal = (
                checkpoint_due
                or any_eval_due
                or (
                    termination_check_every > 0
                    and train_steps % termination_check_every == 0
                )
            )
            if check_termination_signal:
                termination_requested, termination_signal = _termination_requested_across_ranks(device)
            else:
                termination_requested, termination_signal = False, None
            if termination_requested:
                stop_reason = (
                    f"preemption signal {termination_signal}"
                    if termination_signal is not None else "preemption signal"
                )
            elif (
                remaining_seconds is not None
                and preemption_min_remaining_sec > 0
                and remaining_seconds <= preemption_min_remaining_sec
            ):
                stop_reason = (
                    f"walltime guard remaining={remaining_seconds:.0f}s "
                    f"<= {preemption_min_remaining_sec:.0f}s"
                )
            elif (
                closed_loop_due
                and remaining_seconds is not None
                and closed_loop_min_remaining_sec > 0
                and remaining_seconds <= closed_loop_min_remaining_sec
            ):
                stop_reason = (
                    f"closed-loop eval walltime guard remaining={remaining_seconds:.0f}s "
                    f"<= {closed_loop_min_remaining_sec:.0f}s"
                )

            if stop_reason is not None:
                if rank == 0:
                    logger.warning(
                        "[step=%07d] %s; saving checkpoint and exiting cleanly before eval/next step",
                        train_steps,
                        stop_reason,
                    )
                _save_training_checkpoint(
                    model_ref,
                    train_steps,
                    epoch,
                    reason=stop_reason,
                )
                checkpoint_saved_this_step = True
                closed_loop_due_profiles = []
                closed_loop_due = False
                predictor_eval_due = False
                legacy_eval_due = False
                any_eval_due = False
                stop_training = True
                if dist.is_initialized():
                    dist.barrier()

            if checkpoint_due and any_eval_due and not checkpoint_saved_this_step:
                _save_training_checkpoint(
                    model_ref,
                    train_steps,
                    epoch,
                    reason="before eval",
                )
                checkpoint_saved_this_step = True
                if dist.is_initialized():
                    dist.barrier()

            if any_eval_due:
                # Keep eval memory below train-step peak: the just-finished
                # train batch and loss outputs are otherwise still referenced
                # until the end of this loop iteration.
                all_views = gt_actions_raw = gt_actions = proprio = None
                action_pred = student_raw = total_loss = loss_action = loss_feat = None
                teacher_raw = None
                torch.cuda.empty_cache()
                if dist.is_initialized():
                    dist.barrier()

            # Unified (gam) closed-loop LIBERO rollout eval. Gated by
            # training.closed_loop_eval.enabled + predictor_enabled + dataset
            # type in {libero, libero_hdf5}. Rollouts distributed across ranks.
            if closed_loop_due:
                _data_iter = _shutdown_train_loader_workers_for_closed_loop_eval(
                    loader,
                    _data_iter,
                    logger=logger,
                    rank=rank,
                    step=int(train_steps),
                )
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    logger.info(
                        "[step=%07d] closed-loop LIBERO eval starting profiles=%s",
                        train_steps,
                        [p.get("name") for p in closed_loop_due_profiles],
                    )
                _raw_ft = finetune_model.module if hasattr(finetune_model, "module") else finetune_model
                # Snapshot train/eval mode and per-parameter requires_grad so
                # we restore cleanly. eval_libero_unified.py:748+ calls
                # `.requires_grad_(False)` on the passed modules (including
                # teacher_da3). `.train()` restores the training-mode flag while
                # this snapshot restores per-parameter gradients for the next
                # backward pass.
                _prior_mode = _raw_ft.training
                _raw_grad_snapshot = [
                    (p, p.requires_grad) for p in _raw_ft.parameters()
                ]
                _teacher_grad_snapshot = (
                    [(p, p.requires_grad) for p in teacher_da3.parameters()]
                    if teacher_da3 is not None else []
                )
                _raw_ft.eval()
                # Optional EMA swap: rollout the EMA shadow instead of the
                # live training params. Restored unconditionally in `finally`.
                _ema_use = any(bool(p.get("use_ema", False)) for p in closed_loop_due_profiles) and ema_tracker is not None
                _ema_backup = ema_tracker.store_and_swap(_raw_ft) if _ema_use else None
                if _ema_use and rank == 0:
                    logger.info("[step=%07d] closed-loop using EMA weights", train_steps)
                try:
                    if dist.is_initialized():
                        dist.barrier()

                    for _profile_cfg in closed_loop_due_profiles:
                        _profile_name = str(_profile_cfg.get("name", "rollout"))
                        _profile_prefix = str(_profile_cfg.get("_wandb_prefix", f"rollout/{_profile_name}"))
                        _profile_benchmark = str(_profile_cfg.get("benchmark", "libero") or "libero").lower().replace("-", "_")
                        if rank == 0:
                            logger.info(
                                "[step=%07d] closed-loop profile=%s benchmark=%s prefix=%s starting",
                                train_steps, _profile_name, _profile_benchmark, _profile_prefix,
                            )
                        _local_counts = {}
                        _local_videos = {}
                        _local_eval_error = None
                        try:
                            _eval_fn = evaluate_closed_loop_libero_from_training
                            _eval_profile_cfg = dict(_profile_cfg)
                            _local_counts, _local_videos = _eval_fn(
                                cfg=cfg,
                                closed_loop_cfg=_eval_profile_cfg,
                                device=device,
                                rank=rank,
                                world_size=world_size,
                                teacher_da3=teacher_da3,
                                student_da3=_raw_ft.student_da3,
                                action_head=_raw_ft.action_head,
                                future_predictor=getattr(_raw_ft, "future_predictor", None),
                                text_conditioner=getattr(_raw_ft, "text_conditioner", None),
                                proprio_conditioner=getattr(_raw_ft, "proprio_conditioner", None),
                                action_normalizer=action_normalizer,
                                proprio_normalizer=proprio_normalizer,
                                train_steps=int(train_steps),
                            )
                        except Exception as _e:  # noqa: BLE001
                            _local_eval_error = _e
                            logger.exception(
                                "[step=%07d] closed-loop profile=%s benchmark=%s local eval errored on rank %d: %s",
                                train_steps, _profile_name, _profile_benchmark, rank, _e,
                            )

                        _failed_eval_ranks = 1 if _local_eval_error is not None else 0
                        if dist.is_initialized():
                            _failed_tensor = torch.tensor(
                                [_failed_eval_ranks],
                                device=device,
                                dtype=torch.int32,
                            )
                            dist.all_reduce(_failed_tensor, op=dist.ReduceOp.SUM)
                            _failed_eval_ranks = int(_failed_tensor.item())

                        if _failed_eval_ranks:
                            if rank == 0:
                                logger.warning(
                                    "[step=%07d] closed-loop profile=%s failed on %d/%d rank(s); "
                                    "skipping distributed rollout aggregation and continuing training",
                                    train_steps, _profile_name, _failed_eval_ranks, world_size,
                                )
                        else:
                            _global_counts = closed_loop_all_reduce_counts(
                                _local_counts, world_size=world_size, device=device
                            )
                            if rank == 0:
                                _rollout_log = closed_loop_format_wandb_log(_global_counts, prefix=_profile_prefix)
                                try:
                                    _artifact_paths = closed_loop_write_eval_artifacts(
                                        global_counts=_global_counts,
                                        closed_loop_cfg=_profile_cfg,
                                        experiment_dir=experiment_dir,
                                        profile_name=_profile_name,
                                        train_steps=int(train_steps),
                                        prefix=_profile_prefix,
                                        policy_info=closed_loop_get_cached_policy_info(),
                                        world_size=world_size,
                                    )
                                    if _artifact_paths:
                                        logger.info(
                                            "[step=%07d] closed-loop profile=%s artifacts summary=%s per_task=%s",
                                            train_steps,
                                            _profile_name,
                                            _artifact_paths.get("summary_path"),
                                            _artifact_paths.get("per_task_path"),
                                        )
                                        _rollout_log[f"{_profile_prefix}/artifact_written"] = 1.0
                                except Exception as _artifact_e:  # noqa: BLE001
                                    logger.warning(
                                        "[step=%07d] closed-loop profile=%s artifact write failed: %s",
                                        train_steps,
                                        _profile_name,
                                        _artifact_e,
                                    )
                                _all_sr = _rollout_log.get(f"{_profile_prefix}/all/mean_sr", float("nan"))
                                _n_trials = _rollout_log.get(f"{_profile_prefix}/all/num_trials", 0)
                                _n_crashes = _rollout_log.get(f"{_profile_prefix}/all/num_crashes", 0)
                                logger.info(
                                    "[step=%07d] closed-loop profile=%s SR=%.3f "
                                    "(%d trials, %d crashes across %d suite×task)",
                                    train_steps, _profile_name,
                                    float(_all_sr) if _all_sr == _all_sr else 0.0,
                                    int(_n_trials), int(_n_crashes), len(_global_counts),
                                )
                                if _local_videos:
                                    try:
                                        import wandb as _wandb_mod
                                        import imageio.v2 as _imageio_mod
                                        from pathlib import Path as _Path
                                        _vid_dir = _Path(
                                            _eval_profile_cfg.get(
                                                "_video_dir",
                                                _closed_loop_video_dir(
                                                    profile_cfg=_eval_profile_cfg,
                                                    experiment_dir=experiment_dir,
                                                    profile_name=_profile_name,
                                                    train_steps=int(train_steps),
                                                    benchmark=_profile_benchmark,
                                                ),
                                            )
                                        ) / "wandb_rank0_pack"
                                        _vid_dir.mkdir(parents=True, exist_ok=True)
                                        for (_v_suite, _v_task), _frames in _local_videos.items():
                                            if not _frames:
                                                continue
                                            _vid_path = _vid_dir / f"{_v_suite}_task{int(_v_task)}.mp4"
                                            try:
                                                _video_frames = _prepare_rollout_video_frames(_frames)
                                                if not _video_frames:
                                                    continue
                                                _imageio_mod.mimsave(
                                                    str(_vid_path), _video_frames, fps=20, codec="libx264"
                                                )
                                                _rollout_log[
                                                    f"{_profile_prefix}/{_v_suite}/task{int(_v_task)}/video"
                                                ] = _wandb_mod.Video(str(_vid_path), fps=20, format="mp4")
                                            except Exception as _enc_e:  # noqa: BLE001
                                                logger.warning(
                                                    "[step=%07d] video encode failed for profile=%s %s task%d: %s",
                                                    train_steps, _profile_name, _v_suite, int(_v_task), _enc_e,
                                                )
                                    except Exception as _ve:  # noqa: BLE001
                                        logger.warning(
                                            "[step=%07d] wandb video pack failed for profile=%s: %s",
                                            train_steps, _profile_name, _ve,
                                        )
                                if wandb_run is not None:
                                    try:
                                        wandb_run.log(_rollout_log, step=int(train_steps))
                                    except Exception as _e:  # noqa: BLE001
                                        logger.warning(
                                            "[step=%07d] wandb rollout log failed for profile=%s: %s",
                                            train_steps, _profile_name, _e,
                                        )
                        if dist.is_initialized():
                            dist.barrier()
                except Exception as _e:  # noqa: BLE001
                    logger.exception(
                        "[step=%07d] closed-loop eval errored: %s : continuing training",
                        train_steps, _e,
                    )
                finally:
                    # Restore live params before mode/grad toggles so the
                    # next training step backward sees the optimizer-tracked
                    # live tensors instead of EMA shadow values.
                    if _ema_use and _ema_backup:
                        ema_tracker.restore(_raw_ft, _ema_backup)
                    if _prior_mode:
                        _raw_ft.train()
                    else:
                        _raw_ft.eval()
                    # Restore per-parameter requires_grad after closed_loop's
                    # load_stage1_policy() force-freezes the modules.
                    for _p, _rg in _raw_grad_snapshot:
                        if _p.requires_grad != _rg:
                            _p.requires_grad_(_rg)
                    for _p, _rg in _teacher_grad_snapshot:
                        if _p.requires_grad != _rg:
                            _p.requires_grad_(_rg)
                    # The rollout path is wrapped in @torch.no_grad() and may
                    # also nudge the global grad-enabled flag : force on.
                    torch.set_grad_enabled(True)
                    torch.cuda.empty_cache()
            elif predictor_enabled:
                if predictor_eval_due:
                    _run_unified_predictor_eval(
                        int(train_steps),
                        max_batches=int(training_cfg.get("eval_max_batches", 0) or 0),
                    )
            elif (
                legacy_eval_due
                and _ensure_eval_loader(int(train_steps))
            ):
                # Distributed eval: every rank processes its shard of the reserved split,
                # then rank 0 aggregates predictions and computes metrics. Both AE (GT-injection)
                # and NOACT (learnable-token only) modes are controlled by training.eval_ae /
                # training.eval_noact flags.
                wandb_log: dict = {}
                last_ev_pred = last_ev_actions = last_ev_keys = None

                def _eval_pass(action_input_mode: str):
                    """Run one full distributed eval pass. Returns (cat_pred, cat_gt, keys, count)
                    on rank 0, empty on others. `action_input_mode`: "gt" injects GT actions,
                    "none" uses learnable tokens only."""
                    local_preds, local_gts, local_keys = [], [], []
                    first_on_rank0 = {"pred": None, "actions": None, "keys": None}
                    with torch.no_grad():
                        for eval_batch in eval_loader:
                            ev_views, ev_actions_raw, ev_proprio, _, _, _, _, _ = prepare_da3_finetune_batch(
                                teacher_da3, eval_batch, device
                            )
                            ev_stats_keys = eval_batch["action_stats_key"]
                            ev_actions = action_normalizer.normalize(ev_actions_raw, stats_keys=ev_stats_keys)
                            if proprio_normalizer is not None:
                                ev_proprio = proprio_normalizer.normalize(ev_proprio, stats_keys=ev_stats_keys)
                                ev_context_valid_mask = eval_batch.get("context_valid_mask")
                                if ev_context_valid_mask is not None:
                                    ev_proprio = _zero_invalid_context_proprio(
                                        ev_proprio,
                                        ev_context_valid_mask.to(device=device, dtype=torch.bool),
                                    )
                            act_in = ev_actions if action_input_mode == "gt" else None
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                                ev_pred, _, _ = model_ref(
                                    ev_views, proprio=ev_proprio, action_input=act_in
                                )
                            local_preds.append(ev_pred.detach())
                            local_gts.append(ev_actions.detach())
                            local_keys.extend(ev_stats_keys)
                            if first_on_rank0["pred"] is None and rank == 0:
                                first_on_rank0["pred"] = ev_pred.detach()
                                first_on_rank0["actions"] = ev_actions.detach()
                                first_on_rank0["keys"] = ev_stats_keys
                            del ev_views, ev_actions_raw, ev_proprio
                    return _gather_eval_tensors(local_preds, local_gts, local_keys,
                                                 world_size, rank, device) + (first_on_rank0,)

                def _log_mode_metrics(prefix: str, preds, gts, keys, count, first_batch, log_last=False):
                    """Fill wandb_log with eval metrics for one mode. Returns norm/raw metrics on rank 0."""
                    if rank != 0 or count <= 0:
                        return None, None
                    norm_metrics = compute_action_detail_metrics(preds, gts)
                    raw_metrics = compute_raw_action_metrics(preds, gts, action_normalizer, stats_keys=keys)
                    per_dataset = compute_per_dataset_action_metrics(preds, gts, action_normalizer, stats_keys=keys)
                    logger.info(
                        "[step=%07d] EVAL-%s l1_norm=%.4f mse_norm=%.5f l1_raw=%.4f mse_raw=%.5f "
                        "r2_norm=%.3f rel@5/10=%.3f/%.3f (%d samples)",
                        train_steps, prefix.upper(),
                        norm_metrics["l1"], norm_metrics["mse"],
                        raw_metrics["l1"], raw_metrics["mse"], norm_metrics["r2"],
                        norm_metrics["rel_acc_5pct"], norm_metrics["rel_acc_10pct"], count,
                    )
                    if wandb_run is None:
                        return norm_metrics, raw_metrics
                    p = f"eval_{prefix}" if prefix != "ae" else "eval"
                    wandb_log[f"{p}/l1_norm"] = norm_metrics["l1"]
                    wandb_log[f"{p}/mse_norm"] = norm_metrics["mse"]
                    wandb_log[f"{p}/r2_norm"] = norm_metrics["r2"]
                    wandb_log[f"{p}/l1_raw"] = raw_metrics["l1"]
                    wandb_log[f"{p}/mse_raw"] = raw_metrics["mse"]
                    wandb_log[f"{p}/r2_raw"] = raw_metrics["r2"]
                    wandb_log[f"{p}/rel_acc_1pct"] = norm_metrics["rel_acc_1pct"]
                    wandb_log[f"{p}/rel_acc_5pct"] = norm_metrics["rel_acc_5pct"]
                    wandb_log[f"{p}/rel_acc_10pct"] = norm_metrics["rel_acc_10pct"]
                    wandb_log[f"{p}/translation_l1_norm"] = norm_metrics["translation_mae"]
                    wandb_log[f"{p}/rotation_l1_norm"] = norm_metrics["rotation_mae"]
                    wandb_log[f"{p}/gripper_l1_norm"] = norm_metrics["gripper_mae"]
                    wandb_log[f"{p}/translation_l1_raw"] = raw_metrics["translation_mae"]
                    wandb_log[f"{p}/rotation_l1_raw"] = raw_metrics["rotation_mae"]
                    wandb_log[f"{p}/gripper_l1_raw"] = raw_metrics["gripper_mae"]
                    if prefix == "ae":
                        wandb_log[f"{p}/seq_rel_acc_5pct"] = norm_metrics["seq_acc_5pct"]
                        wandb_log[f"{p}/seq_rel_acc_10pct"] = norm_metrics["seq_acc_10pct"]
                        add_named_metrics(wandb_log, f"{p}/per_dim_l1_norm", ACTION_DIM_NAMES, norm_metrics["per_dim_mae"])
                        add_named_metrics(wandb_log, f"{p}/per_dim_mse_norm", ACTION_DIM_NAMES, norm_metrics["per_dim_mse"])
                        add_named_metrics(wandb_log, f"{p}/per_dim_r2_norm", ACTION_DIM_NAMES, norm_metrics["per_dim_r2"])
                        add_named_metrics(wandb_log, f"{p}/per_dim_l1_raw", ACTION_DIM_NAMES, raw_metrics["per_dim_mae"])
                        add_named_metrics(wandb_log, f"{p}/per_dim_mse_raw", ACTION_DIM_NAMES, raw_metrics["per_dim_mse"])
                        add_named_metrics(wandb_log, f"{p}/per_dim_r2_raw", ACTION_DIM_NAMES, raw_metrics["per_dim_r2"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_l1_norm", norm_metrics["timestep_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_translation_l1_norm", norm_metrics["timestep_translation_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_rotation_l1_norm", norm_metrics["timestep_rotation_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_gripper_l1_norm", norm_metrics["timestep_gripper_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_l1_raw", raw_metrics["timestep_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_translation_l1_raw", raw_metrics["timestep_translation_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_rotation_l1_raw", raw_metrics["timestep_rotation_mae"])
                        add_indexed_metrics(wandb_log, f"{p}/timestep_gripper_l1_raw", raw_metrics["timestep_gripper_mae"])
                        add_per_dataset_metrics(wandb_log, "eval_by_dataset", per_dataset)
                    if log_last and first_batch["pred"] is not None:
                        last_norm = compute_action_detail_metrics(first_batch["pred"], first_batch["actions"])
                        last_raw = compute_raw_action_metrics(
                            first_batch["pred"], first_batch["actions"], action_normalizer,
                            stats_keys=first_batch["keys"],
                        )
                        wandb_log[f"{p}/last_batch_l1_norm"] = last_norm["l1"]
                        wandb_log[f"{p}/last_batch_mse_norm"] = last_norm["mse"]
                        wandb_log[f"{p}/last_batch_l1_raw"] = last_raw["l1"]
                        wandb_log[f"{p}/last_batch_mse_raw"] = last_raw["mse"]
                    return norm_metrics, raw_metrics

                finetune_model.eval()

                if eval_ae:
                    ae_pred, ae_gt, ae_keys, ae_count, ae_first = _eval_pass("gt")
                    _log_mode_metrics("ae", ae_pred, ae_gt, ae_keys, ae_count, ae_first, log_last=True)
                    if rank == 0 and ae_first["pred"] is not None:
                        last_ev_pred = ae_first["pred"]
                        last_ev_actions = ae_first["actions"]
                        last_ev_keys = ae_first["keys"]
                    del ae_pred, ae_gt, ae_keys

                if eval_noact:
                    na_pred, na_gt, na_keys, na_count, na_first = _eval_pass("none")
                    _log_mode_metrics("noact", na_pred, na_gt, na_keys, na_count, na_first, log_last=False)
                    if last_ev_pred is None and rank == 0 and na_first["pred"] is not None:
                        last_ev_pred = na_first["pred"]
                        last_ev_actions = na_first["actions"]
                        last_ev_keys = na_first["keys"]
                    del na_pred, na_gt, na_keys

                finetune_model.train()

                if rank == 0 and wandb_run is not None:
                    if wandb_log:
                        import wandb
                        wandb.log(wandb_log, step=train_steps)
                    if last_ev_pred is not None:
                        try:
                            log_action_trajectory(
                                action_normalizer.denormalize(last_ev_actions, stats_keys=last_ev_keys),
                                action_normalizer.denormalize(last_ev_pred, stats_keys=last_ev_keys),
                                train_steps,
                                wandb_run=wandb_run, batch_idx=0,
                            )
                        except Exception as exc:
                            logger.warning("Eval action trajectory vis failed: %s", exc)

                del last_ev_pred, last_ev_actions, last_ev_keys, wandb_log
                torch.cuda.empty_cache()

            if checkpoint_due and not checkpoint_saved_this_step:
                _save_training_checkpoint(model_ref, train_steps, epoch)

            # Free large tensors to prevent memory accumulation
            del all_views, gt_actions_raw, gt_actions, proprio
            del action_pred, student_raw, total_loss, loss_action, loss_feat
            # teacher_raw is path-dependent; guard cleanup.
            try:
                del teacher_raw
            except NameError:
                pass

            step_barrier_every = int(os.environ.get("DA3_TRAIN_STEP_BARRIER_EVERY", "0"))
            if use_deepspeed and step_barrier_every > 0 and train_steps % step_barrier_every == 0:
                dist.barrier()

            if stop_training:
                break

            if max_steps > 0 and train_steps >= max_steps:
                break

        if stop_training or (max_steps > 0 and train_steps >= max_steps):
            break

    if rank == 0:
        logger.info("Training complete%s.", " (clean preemption exit)" if stop_training else "")
    if use_deepspeed:
        dist.destroy_process_group()


def main(args):
    torch.set_float32_matmul_precision("high")
    cfg = OmegaConf.load(args.config)
    if getattr(args, "set", None):
        for override in args.set:
            if "=" not in override:
                raise ValueError(f"--set override must be key=value, got {override!r}")
            key, value_text = override.split("=", 1)
            parsed_value = OmegaConf.from_dotlist([f"value={value_text}"]).get("value")
            OmegaConf.update(cfg, key, parsed_value, merge=True)
    if bool(cfg.get("da3_finetune", {}).get("enabled", False)):
        run_da3_finetune_training(args, cfg)
    else:
        raise ValueError(
            "This public release supports the gam DA3-Giant path only; "
            "set da3_finetune.enabled=true."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLD-Robot Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/robot")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-new-run", action="store_true",
                        help="Force a new wandb run instead of resuming from checkpoint's run ID")
    parser.add_argument("--wandb-resume-from", type=str, default=None,
                        help="W&B rewind point, for example '<run_id>?_step=72000'.")
    parser.add_argument("--reset-schedule", action="store_true",
                        help="Reset LR scheduler and optimizer state on resume (for LR warm restart)")
    parser.add_argument("--reset-optimizer-state", action="store_true",
                        help="Skip optimizer/scheduler/scaler state load on resume while preserving "
                             "checkpoint train_steps and epoch. Use when changing LR or batch size "
                             "without restarting the step counter.")
    parser.add_argument("--refresh-action-stats", action="store_true",
                        help="Skip action_normalizer and proprio_normalizer from checkpoint; force "
                             "reload from stats_dir. Use this when stats (q01/q99) have been "
                             "recomputed after a data pipeline fix so the resumed run picks up the "
                             "new normalization.")
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="Load checkpoint, run the configured eval split once, log to W&B, and exit.")
    parser.add_argument("--eval-max-batches", type=int, default=0,
                        help="Optional cap for --eval-only / unified eval batches per rank; 0 means full split.")
    parser.add_argument("--intent", type=str, default="", help="Short description of experiment intent")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=float,
        default=None,
        help=(
            "Distributed process-group timeout in minutes. Defaults to 120; "
            "can also be set with training.distributed_timeout_minutes or "
            "DA3_DISTRIBUTED_TIMEOUT_MINUTES."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with OmegaConf dotlist entries.",
    )
    try:
        import deepspeed

        parser = deepspeed.add_config_arguments(parser)
    except ImportError:
        pass
    main(parser.parse_args())
