# Pretraining Data

## Mixture

| Source | Ratio | Depth |
|---|---:|---|
| Open X-Embodiment | 72% | DA3 teacher pseudo-depth |
| MimicGen | 18% | Simulator depth |
| RoboCasa365 | 10% | Simulator depth |

Base: 224x224 images, external + wrist views, 8-step action chunks. Views are
configurable.

## Loader

```bash
pip install lerobot==0.4.4 jsonlines
```

```yaml
dataset:
  type: gam_pretraining
  openx_root: /path/to/openx_lerobot
  mimicgen_root: /path/to/mimicgen
  mimicgen_depth_root: /path/to/mimicgen_depth
  task_descriptions_path: /path/to/mimicgen/task_descriptions.json
  robocasa_root: /path/to/robocasa/v1.0
  robocasa_depth_index_path: /path/to/robocasa365_depth/index.json
  stats_dir: /path/to/stats
  image_size: [224, 224]
  future_steps: 1
  chunk_size: 8
  include_current_action: true
  n_views: 2
```

Set `n_views` to the number of camera views used by the model. Use `all` to keep every available view.

## Datasets

| Source | Dataset key | Action conversion | Download |
|---|---|---|---|
| Open X-Embodiment | `bridge` | Reached Euler residual -> relative rotvec; open -> close | [BrunoM42/bridge_orig_lerobot](https://huggingface.co/datasets/BrunoM42/bridge_orig_lerobot) |
| Open X-Embodiment | `droid` | Target-current pose; `R_target R_current^T` -> rotvec; gripper retained | [lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) |
| Open X-Embodiment | `taco_play` | Position `/ 50`; Euler `/ 20` -> rotvec; open -> close | [lerobot/taco_play](https://huggingface.co/datasets/lerobot/taco_play) |
| Open X-Embodiment | `utaustin_mutex` | Stored delta; open -> close | [lerobot/utaustin_mutex](https://huggingface.co/datasets/lerobot/utaustin_mutex) |
| Open X-Embodiment | `stanford_hydra_dataset` | Euler XYZ delta -> rotvec; open -> close | [lerobot/stanford_hydra_dataset](https://huggingface.co/datasets/lerobot/stanford_hydra_dataset) |
| Open X-Embodiment | `berkeley_autolab_ur5` | Base RPY delta -> rotvec; open -> close | [lerobot/berkeley_autolab_ur5](https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5) |
| Open X-Embodiment | `austin_sailor_dataset` | Stored xyz/yaw delta; open -> close; roll/pitch masked | [lerobot/austin_sailor_dataset](https://huggingface.co/datasets/lerobot/austin_sailor_dataset) |
| Open X-Embodiment | `austin_sirius_dataset` | Stored xyz/yaw delta; open -> close; roll/pitch masked | [lerobot/austin_sirius_dataset](https://huggingface.co/datasets/lerobot/austin_sirius_dataset) |
| Open X-Embodiment | `berkeley_fanuc_manipulation` | Euler XYZ delta -> rotvec; gripper masked | [lerobot/berkeley_fanuc_manipulation](https://huggingface.co/datasets/lerobot/berkeley_fanuc_manipulation) |
| Open X-Embodiment | `jaco_play` | Stored xyz delta; rotation masked; open -> close | [lerobot/jaco_play](https://huggingface.co/datasets/lerobot/jaco_play) |
| Open X-Embodiment | `fmb_dataset` | Base RPY delta -> rotvec; close retained | [lerobot/fmb](https://huggingface.co/datasets/lerobot/fmb) |
| Open X-Embodiment | `kuka` | Stored xyz delta; rotation and gripper masked | [lerobot/stanford_kuka_multimodal_dataset](https://huggingface.co/datasets/lerobot/stanford_kuka_multimodal_dataset) |
| Open X-Embodiment | `fractal20220817_data` | Base RPY delta -> rotvec; open -> close | [BrunoM42/fractal20220817_data_lerobot](https://huggingface.co/datasets/BrunoM42/fractal20220817_data_lerobot) |
| Open X-Embodiment | `berkeley_cable_routing` | Cartesian velocity `/ fps`; roll, pitch, gripper masked | [lerobot/berkeley_cable_routing](https://huggingface.co/datasets/lerobot/berkeley_cable_routing) |
| Open X-Embodiment | `roboturk` | Stored delta; open -> close | [lerobot/roboturk](https://huggingface.co/datasets/lerobot/roboturk) |
| Open X-Embodiment | `dlr_edan_shared_control` | Euler ZXY delta -> rotvec; open -> close | [lerobot/dlr_edan_shared_control](https://huggingface.co/datasets/lerobot/dlr_edan_shared_control) |
| Open X-Embodiment | `austin_buds_dataset` | Stored xyz; signed gripper `[-1, 1]` -> `[0, 1]`; rotation masked | [lerobot/austin_buds_dataset](https://huggingface.co/datasets/lerobot/austin_buds_dataset) |
| Open X-Embodiment | `nyu_franka_play_dataset` | Slice raw EEF xyz/RPY; RPY -> rotvec; open -> close | [lerobot/nyu_franka_play_dataset](https://huggingface.co/datasets/lerobot/nyu_franka_play_dataset) |
| Open X-Embodiment | `nyu_door_opening_surprising_effectiveness` | Cartesian velocity `/ fps`; open -> close | [lerobot/nyu_door_opening_surprising_effectiveness](https://huggingface.co/datasets/lerobot/nyu_door_opening_surprising_effectiveness) |
| Open X-Embodiment | `cmu_stretch` | Drop final slot; supervise x/z only | [lerobot/cmu_stretch](https://huggingface.co/datasets/lerobot/cmu_stretch) |
| Open X-Embodiment | `furniture_bench_dataset` | Base xyz; body-local RPY -> base rotvec; open -> close | [tailong-wu/furniture_bench_dataset_lerobot_v30](https://huggingface.co/datasets/tailong-wu/furniture_bench_dataset_lerobot_v30) |
| Open X-Embodiment | `bc_z` | Axis-angle residual plus state -> relative rotvec; open -> close | [tailong-wu/bc_z_lerobot_v30](https://huggingface.co/datasets/tailong-wu/bc_z_lerobot_v30) |
| Open X-Embodiment | `language_table` | Keep xy only; other dimensions masked | [tailong-wu/language_table_lerobot_v30](https://huggingface.co/datasets/tailong-wu/language_table_lerobot_v30) |
| MimicGen | `core` | OSC position x `0.05`; rotation x `0.5`; signed gripper -> close | [amandlek/mimicgen_datasets](https://huggingface.co/datasets/amandlek/mimicgen_datasets) |
| RoboCasa365 | `pretrain / human / atomic + composite` | Select 6D arm delta; signed gripper -> close | [robocasa/robocasa](https://github.com/robocasa/robocasa) |

## Paths

```bash
export GAM_ROOT="$(git rev-parse --show-toplevel)"
export DATA_ROOT="${DATA_ROOT:-$HOME/gam-data}"
mkdir -p "$DATA_ROOT"
```

## Download

### Open X-Embodiment

```bash
python "$GAM_ROOT/scripts/pretraining/download_openx.py" \
  --output-root "$DATA_ROOT/openx_lerobot"

python "$GAM_ROOT/scripts/pretraining/compute_droid_nonidle_ranges.py" \
  --openx-root "$DATA_ROOT/openx_lerobot"
```

### MimicGen

```bash
hf download amandlek/mimicgen_datasets \
  --repo-type dataset \
  --include 'core/*.hdf5' \
  --local-dir "$DATA_ROOT/mimicgen"
```

### RoboCasa365

```bash
python "$GAM_ROOT/scripts/pretraining/download_robocasa.py" \
  --output-root "$DATA_ROOT/robocasa"
```

The downloader converts LeRobot v2.1 data to v3 and keeps replay states in `extras`.

## Depth

Use the upstream MimicGen and RoboCasa simulator environments for depth export.

### MimicGen

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python "$GAM_ROOT/scripts/pretraining/export_mimicgen_depth.py" \
  --mimicgen-root "$DATA_ROOT/mimicgen" \
  --output-root "$DATA_ROOT/mimicgen_depth" \
  --splits core \
  --cameras agentview,robot0_eye_in_hand
```

Output: metric depth from HDF5 state replay and MuJoCo `get_real_depth_map`.

### RoboCasa365

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python "$GAM_ROOT/scripts/pretraining/export_robocasa_depth.py" \
  --robocasa-root /path/to/robocasa/v1.0 \
  --output-root "$DATA_ROOT/robocasa365_depth" \
  --sources atomic,composite \
  --target-demos-per-task 100 \
  --cameras robot0_agentview_left,robot0_eye_in_hand
```

Output: metric depth and `index.json` from successful base-static episodes after
no-op filtering.

## Actions

```text
[delta_position(3), delta_rotation_axis_angle(3), gripper_close(1)]
```

Each row is converted before q01/q99 normalization. Masked dimensions are not
included in the action loss.

```bash
python "$GAM_ROOT/scripts/pretraining/compute_action_stats.py" \
  "$DATA_ROOT/mimicgen/core" \
  --key actions \
  --layout mimicgen-osc \
  --output "$DATA_ROOT/stats/mimicgen.json"
```

Use `--layout canonical7` for converted actions and
`--layout robocasa365-12d` for raw RoboCasa365 actions.

```text
a_norm = 2 * (a - q01) / (q99 - q01 + eps) - 1
```

`train_robot.py` computes per-dataset q01/q99 files through the loader when
`stats_dir` is empty.

Upstream licenses and citation requirements apply.

## Samples

```bash
python "$GAM_ROOT/scripts/pretraining/visualize_pretraining_samples.py" \
  --config /path/to/training.yaml \
  --output-root "$DATA_ROOT/pretraining_samples"
```
