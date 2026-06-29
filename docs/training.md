# GAM Training

This document collects the training reference details for the public GAM
release. The main README keeps only the quick-start commands.

## Quick Start

Single-GPU smoke run:

```bash
PYTHONPATH=src:$PYTHONPATH python src/train_robot.py \
  --config configs/training/libero_unified/smoke/gam_chunk2.yaml \
  --single-gpu \
  --set training.max_steps=1
```

Multi-GPU GAM fine-tuning with DeepSpeed ZeRO-2:

```bash
PYTHONPATH=src:$PYTHONPATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --wandb \
  --wandb-name gam_libero
```

## Config Reference

`src/train_robot.py` reads an OmegaConf YAML and accepts `--set key=value`
overrides. The main GAM config is:

```text
configs/training/libero_unified/gam/chunk8_150k_2node.yaml
```

| YAML key | Meaning |
|----------|---------|
| `stage_1.ckpt_path` | DA3-Giant base checkpoint loaded before robot fine-tuning |
| `da3_finetune.enabled` | Enables the DA3-Giant GAM fine-tuning path |
| `da3_finetune.freeze_blocks_before` | Freezes DA3 blocks before this index, with GAM using blocks 0-12 as the geometric encoder |
| `da3_finetune.n_action_steps` | Low-level actions represented by one GAM action token sequence |
| `da3_finetune.n_views` | Camera views per timestep |
| `action_head.chunk_size` | Low-level actions predicted per action-head token |
| `action_head.n_dims` | Action dimensionality, 7 for LIBERO delta actions |
| `predictor.enabled` | Enables `GAMFuturePredictor` |
| `predictor.type` | `gam` for this release |
| `predictor.H_choices` | Observed history lengths sampled during training |
| `predictor.H_weights` | Sampling weights for `H_choices` |
| `predictor.lambda_feat_future` | Future latent feature distillation weight |
| `predictor.lambda_sigreg` | SIGReg regularization weight |
| `regularization.lambda_depth` | DA3 depth decode loss weight |
| `training.global_batch_size` | Target global batch size across data-parallel ranks |
| `training.micro_batch_size` | Per-GPU batch before gradient accumulation |
| `training.grad_accum_steps` | Gradient accumulation factor |
| `training.base_lr` | DA3 backbone base learning rate |
| `training.head_lr_mult` | Action-head learning-rate multiplier |
| `training.predictor_lr_mult` | GAM predictor learning-rate multiplier |
| `training.max_steps` | Total optimizer steps |
| `training.ckpt_every` | Checkpoint save interval in steps |
| `training.vis_every` | Visualization interval in steps |
| `training.bf16` | bf16 autocast training |
| `training.compile` | Calls `torch.compile` around the training model |
| `dataset.hdf5_root` | LIBERO HDF5 root |
| `dataset.stats_dir` | Action and proprioception stats root |
| `dataset.future_steps` | Future action and observation horizon in dataset samples |
| `dataset.chunk_size` | Low-level action chunk length from the dataset |
| `dataset.camera_keys` | Camera keys read from HDF5 |
| `dataset.da3_input_rotate180` | Train-time DA3 image rotation convention |
| `dataset.gt_depth_root` | Depth source. `null` reads embedded HDF5 depth keys |

## CLI Flags

| Flag | Meaning |
|------|---------|
| `--config` | YAML config path |
| `--results-dir` | Output root for checkpoints, logs, and visualizations |
| `--ckpt` | Resume checkpoint |
| `--single-gpu` | Run one local GPU process |
| `--wandb` | Enable W&B logging |
| `--wandb-name` | W&B run display name |
| `--wandb-project` | W&B project override |
| `--wandb-new-run` | Start a fresh W&B run during resume |
| `--wandb-resume-from` | Rewind W&B resume point, for example `<run_id>?_step=72000` |
| `--reset-schedule` | Reset optimizer and LR scheduler on resume |
| `--reset-optimizer-state` | Load model weights while starting optimizer, scheduler, and scaler fresh |
| `--refresh-action-stats` | Reload normalizers from `dataset.stats_dir` during resume |
| `--eval-only` | Load checkpoint, run configured eval split, then exit |
| `--eval-max-batches` | Cap eval batches per rank for `--eval-only` |
| `--ddp-timeout-minutes` | Distributed process group timeout |
| `--set key=value` | Override YAML keys with OmegaConf dotlist syntax |
| `--deepspeed_config` | DeepSpeed config path, added by DeepSpeed |

## Resume

Resume with checkpoint optimizer state:

```bash
PYTHONPATH=src:$PYTHONPATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --ckpt /path/to/checkpoint.pt \
  --wandb \
  --wandb-name gam_resume
```

Resume model weights with a fresh optimizer and scheduler:

```bash
PYTHONPATH=src:$PYTHONPATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --include localhost:0,1,2,3 src/train_robot.py \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --deepspeed_config configs/training/libero_unified/deepspeed/micro2.json \
  --ckpt /path/to/checkpoint.pt \
  --reset-optimizer-state \
  --wandb \
  --wandb-name gam_resume_fresh_optimizer
```

Use `--reset-optimizer-state` for a deliberate LR and optimizer restart. Leave
the flag out for continuation training from the checkpoint state.

## DeepSpeed ZeRO-2

DeepSpeed ZeRO-2 is selected by the JSON passed to `--deepspeed_config`.
`configs/training/libero_unified/deepspeed/micro2.json` sets:

| JSON key | Meaning |
|----------|---------|
| `train_micro_batch_size_per_gpu` | Per-GPU micro batch seen by DeepSpeed |
| `gradient_accumulation_steps` | DeepSpeed accumulation factor |
| `zero_optimization.stage` | ZeRO stage, `2` for optimizer-state sharding |
| `bf16.enabled` | bf16 training |
| `optimizer.type` | AdamW |
| `optimizer.params.lr` | Base optimizer LR, overridden by train param groups |
| `gradient_clipping` | Global grad clipping value |

## Headless Container Runtime

Training only requires CUDA compute and utility devices. On some Slurm,
Pyxis/enroot, or NVIDIA Container Toolkit setups,
`NVIDIA_DRIVER_CAPABILITIES=all` can make the NVIDIA hook request graphics
device files such as `/dev/nvidia-modeset` before Python starts. For training
jobs, set:

```bash
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

For LIBERO rollout eval with EGL rendering, use a runtime that exposes the
NVIDIA EGL/OpenGL device stack and verify the node provides the graphics device
files required by that runtime.

## Compile Controls

| Setting | Meaning |
|---------|---------|
| `training.compile=true` | Calls `torch.compile` around the training model |
| `DA3_TRAIN_COMPILE_MODE=default` | Standard training compile mode |
| `training.compile=false` | Eager training path used by the public configs |

## In-Training Closed-Loop Eval

Closed-loop eval during training is configured in YAML with
`training.closed_loop_evals`. Each entry is a rollout profile consumed by
`src/robot/evaluation/closed_loop_libero_eval.py`.

```yaml
training:
  closed_loop_evals:
    - name: plus_spatial_smoke
      benchmark: libero_plus
      suites: [libero_spatial]
      num_trials_per_task: 1
      max_tasks_per_suite: 4
      every_steps: 2000
      plus_official_category: camera
      libero_plus_robot_init_qpos_mode: original
      action_horizon: 1
      rollout_decode_horizon: 1
      action_repeat: 1
      action_repeat_mode: split_delta
      camera_size: 256
      env_process_isolation: true
```

| Profile key | Meaning |
|-------------|---------|
| `name` | Label used in logs and W&B metrics |
| `benchmark` | `libero` or `libero_plus` |
| `suites` | Suite list such as `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `num_trials_per_task` | Rollout trials per task |
| `max_tasks_per_suite` | Optional task cap for smoke profiles |
| `every_steps` | Training step interval |
| `plus_official_category` | LIBERO-Plus category filter |
| `libero_plus_robot_init_qpos_mode` | Use `original` for official LIBERO-Plus qpos |
| `action_horizon` | GAM model-step chunks executed per policy call |
| `rollout_decode_horizon` | GAM AR model steps decoded before action selection |
| `action_repeat` | Env steps per predicted action |
| `action_repeat_mode` | `split_delta` divides motion deltas across repeats |
| `camera_size` | Simulator render resolution |
| `env_process_isolation` | Runs each env inside a child process |
