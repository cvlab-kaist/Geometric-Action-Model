# GAM Evaluation

This document collects the rollout and latency reference details for the public
GAM release. The main README keeps only the quick-start commands.

## LIBERO-Plus Assets

Install LIBERO-Plus source checkouts and assets before rollout:

```bash
bash scripts/setup_sources.sh
bash scripts/setup_libero_plus.sh --download-assets
```

`scripts/setup_libero_plus.sh --download-assets` downloads
`Sylvest/LIBERO-plus` `assets.zip`, strips the nested archive prefix, and
installs assets under:

```text
$DA3_LIBERO_PLUS_DIR/libero/libero/assets
```

Set `DA3_LIBERO_PLUS_ASSETS_DIR` to use an existing asset directory.

## Standalone LIBERO-Plus Evaluation

The standalone script runs one process per GPU, shards the task list, and writes
suite-level `summary.json` and `per_task.csv` files.

```bash
GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh spatial
GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh object
GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh goal
GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh long
GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh all
```

Default protocol:

| Argument | Value |
|----------|-------|
| `--plus` | enabled |
| `--plus-perturbation` | `all` |
| `--plus-official-category` | `all` |
| `--num-trials-per-task` | `1` |
| `--libero-plus-robot-init-qpos-mode` | `original` |
| `--history-horizon` | `1` |
| `--rollout-decode-horizon` | `1` |
| `--action-horizon` | `1` |
| `--action-repeat` | `1` |
| `--action-repeat-mode` | `split_delta` |
| `--camera-size` | `256` |
| `--parallel-envs` | `16` |
| `--max-batch-size` | `16` |
| `--env-process-isolation` | enabled |

`original` robot qpos follows the official LIBERO-Plus robot initialization.

Full LIBERO-Plus contains 10,030 one-trial episodes:

| Suite | Episodes |
|-------|---------:|
| `libero_spatial` | 2,402 |
| `libero_object` | 2,518 |
| `libero_goal` | 2,591 |
| `libero_10` | 2,519 |

## Parallelism

| Variable | Default | Meaning |
|----------|---------|---------|
| `GAM_EVAL_GPUS` | `CUDA_VISIBLE_DEVICES` or `0` | One eval process and one task shard per listed GPU |
| `PARALLEL_ENVS_PER_GPU` | `16` | Simulator workers per GPU process |
| `MAX_BATCH_SIZE` | `16` | Maximum observations per policy forward |
| `MAX_WAIT_TIME` | `0.5` | Batch wait time in seconds |
| `ENV_CACHE_SIZE` | `1` | Cached simulator instances per worker |
| `GAM_PLUS_PERTURBATION` | `all` | LIBERO-Plus perturbation filter |
| `GAM_PLUS_OFFICIAL_CATEGORY` | `all` | Official category filter |

The launcher sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
`EGL_PLATFORM=device`. On a bare GPU machine, install the GL, EGL, OSMesa, GLFW,
ffmpeg, and ImageMagick packages listed in the README installation section.
For EGL rollout eval inside a headless container, the runtime must expose the
NVIDIA EGL/OpenGL device stack. If startup fails while requesting
`/dev/nvidia-modeset`, verify that the selected node provides that graphics
device file before enabling graphics capabilities.

## LIBERO Evaluation

```bash
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials-per-task 5
```

## Evaluation CLI Flags

Standalone rollout eval uses `src/eval_libero_unified.py`.

| Flag | Meaning |
|------|---------|
| `--ckpt` | Stage 1 GAM checkpoint |
| `--config` | Training YAML used to rebuild model architecture |
| `--use-ema` | Load EMA weights saved in the checkpoint |
| `--suites` | Comma-separated suite list |
| `--task-ids` | Comma-separated task ids within each suite |
| `--num-trials-per-task` | Trials per task |
| `--preset` | Preset horizon and wait-step bundle |
| `--max-steps` | Rollout horizon override |
| `--num-steps-wait` | Initial dummy wait steps |
| `--history-horizon` | Observed history length given to GAM |
| `--rollout-decode-horizon` | GAM AR decode length before action selection |
| `--action-horizon` | Model-step chunks executed per policy call |
| `--action-repeat` | Env steps per predicted action |
| `--action-repeat-mode` | `hold` or `split_delta` |
| `--policy-hz` | Policy frequency used by `action-repeat=auto` |
| `--env-control-hz` | LIBERO robosuite control frequency |
| `--camera-size` | Simulator camera resolution |
| `--render-gpu-device-id` | robosuite EGL render GPU override |
| `--env-process-isolation` | Spawn child env workers |
| `--output-dir` | Eval output root |
| `--run-name` | Eval run folder name |
| `--shard-index` | Shard id for distributed eval |
| `--shard-count` | Total shard count |
| `--video-every` | Save one diagnostic video every N global episodes |
| `--detailed-video` | Save RGB, depth, and action diagnostic video |
| `--trace-actions` | Write per-step action and proprio diagnostics |
| `--decode-visuals` | Decode depth and RGB for diagnostics |
| `--temporal-ensemble` | ACT-style low-level action ensemble |
| `--execution-strategy` | Diagnostic execution strategy |
| `--execute-chunk-prefix` | Execute a prefix of each chunk before re-observing |
| `--partial-chunk-history` | Previous-action history policy for prefix execution |
| `--rotate-policy-input` | Rotate live RGB by 180 degrees |
| `--proprio-orientation` | `auto`, `rpy`, or `axis_angle` live proprio convention |
| `--text-prompt-normalization` | Text normalization before encoding |
| `--action-frame` | Model action frame override |
| `--wandb` | Enable W&B logging |
| `--action-stats-key` | Normalizer stats key override |
| `--plus` | Enable LIBERO-Plus |
| `--plus-root` | LIBERO-Plus source checkout |
| `--plus-perturbation` | LIBERO-Plus perturbation filter |
| `--plus-official-category` | Official category filter such as `camera` or `noise` |
| `--libero-plus-robot-init-qpos-mode` | Use `original` for official Plus qpos |
| `--plus-sample-group-by` | Deterministic Plus task sampling group |
| `--plus-samples-per-group` | Tasks per sampled group |

## CUDA Graph Latency Mode

The paper latency path is an eval-time CUDA graph path inside
`src/eval_libero_unified.py`. It fuses GAM h=1 inference into one compiled
callable:

```text
DA3 shallow encode -> GAMFuturePredictor -> DA3 deep propagation -> ActionHeadV2
```

Use this mode for model-forward latency measurement:

```bash
DA3_MAX_OPTIMIZE=1 \
DA3_COMPILE_INFERENCE_MODE=reduce-overhead \
DA3_FUSE_SHALLOW=1 \
DA3_SKIP_FULL_ENCODE=1 \
DA3_PROFILE_INFERENCE=1 \
PYTHONPATH=src:$PYTHONPATH python src/eval_libero_unified.py \
  --ckpt /path/to/checkpoint.pt \
  --config configs/training/libero_unified/gam/chunk8_150k_2node.yaml \
  --suites libero_spatial \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --history-horizon 1 \
  --rollout-decode-horizon 1 \
  --action-horizon 1 \
  --action-repeat 1 \
  --action-repeat-mode split_delta \
  --camera-size 256 \
  --env-process-isolation
```

The log line has this form:

```text
[INFER PROFILE] H_eff=1 dec_vis=0 full_enc=skip total=...ms ar(no_cache)=...ms ...
```

For the paper latency number, read the `ar(...)` model-forward field after
warmup. `total` includes preprocessing, CPU copies, normalization, and logging
guards.

| Variable | Value | Meaning |
|----------|-------|---------|
| `DA3_MAX_OPTIMIZE` | `1` | Enables the fused h=1 path |
| `DA3_COMPILE_INFERENCE_MODE` | `reduce-overhead` | Uses PyTorch CUDA graph replay mode |
| `DA3_FUSE_SHALLOW` | `1` | Folds DA3 blocks 0-12 into the fused graph |
| `DA3_SKIP_FULL_ENCODE` | `1` | Skips the separate full DA3 encode in GAM action selection |
| `DA3_PROFILE_INFERENCE` | `1` | Prints `[INFER PROFILE]` timing lines |
| `DA3_CUDAGRAPH_CLONE` | `0` | Keeps fused graph output clone-free for the single graph path |
| `DA3_MAX_OPTIMIZE_NO_BF16` | `1` | Leaves inference weights in fp32 for ablation |

General per-submodule compile is also available:

| Variable | Values | Meaning |
|----------|--------|---------|
| `DA3_COMPILE_INFERENCE` | `all`, `predictor`, `shallow`, `propagate`, `action_head` | Compiles selected eval modules |
| `DA3_COMPILE_INFERENCE_MODE` | `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`, `default` | PyTorch compile mode |
| `DA3_CUDAGRAPH_CLONE` | `1` or `0` | Clones outputs from CUDA graph buffers for separate compiled modules |
