# Mobile bimanual inference (AgiBot + OXE)

This branch prepares inference for the **shared 16D AgiBot/OXE continued-pretraining
checkpoint family**. Its weights are separate from the existing 7D LIBERO release.
No mobile checkpoint has been published by this code change. Obtain an exported
mobile bundle from the model owner; the existing `pretrained-gam.pt` cannot be
substituted. Pretraining inference is not a measured downstream robot success rate.

## Setup

Use the repository's Python environment. This inference entrypoint does not import
LIBERO, MuJoCo, LeRobot, DeepSpeed or dataset loaders. It uses PyTorch, NumPy,
SciPy, Pillow, Transformers, OmegaConf/einops and the DA3 source dependency.
Clone DA3 without running the simulator/package setup script:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3
git -C Depth-Anything-3 checkout 2c21ea849ceec7b469a3e62ea0c0e270afc3281a
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

The bundle contains `model.pt`, `config.json`, `normalization.json` and
`manifest.json`. The loader verifies SHA256s, loads tensor-only weights with
`weights_only=True`, and strictly checks every model component. Separate DA3
base weights, training data, optimizer shards and server paths are unnecessary.
Frozen T5 is loaded from the exact Hub revision recorded in the bundle.
Use `text_model_path=` for an offline local copy of that same T5 snapshot.

## Python API

```python
import numpy as np
from gam.mobile import MobileGAMPolicy
from gam.mobile.geometry import joint_state

policy = MobileGAMPolicy.from_pretrained("mobile-bundle", device="cuda")
# RGB uint8 arrays; chronological camera order: head, left wrist, right wrist.
images = np.stack([head_rgb, left_wrist_rgb, right_wrist_rgb])[None]
state = joint_state(left_joint_angles, right_joint_angles,
                    left_measured_closure, right_measured_closure)[None]
# Measured flange-to-body poses at THIS policy call: xyz(m), xyzw quaternion.
anchors = np.stack([left_flange_pose, right_flange_pose])
result = policy.predict(images=images, states=state,
                        instruction="Pick up the object with both hands.",
                        anchor_poses=anchors)
relative_actions = result["actions"]       # (16,16), unnormalized
flange_targets = result["flange_poses"]    # (16,2,7), xyz+xyzw
closures = result["gripper_closure"]       # (16,2), canonical closure
base_velocities = result["base_velocity"] # (16,2), forward m/s and yaw rad/s
```

The omitted previous chunk above is valid **only at episode start**. Every later
call must provide `previous_actions` and `previous_action_mask`, including H1
calls. If no reliable previous command is available, explicitly pass zeros and
an all-false mask. No action is sent to hardware by this library.

## Input and output contract

| Field | Shape | Meaning |
|---|---|---|
| `images` | H,3,height,width,3 | RGB uint8; H=1..4 real observed anchors |
| `states` | H,16 | Left measured joint7, left closure, right measured joint7, right closure |
| `previous_actions` | H,16,16 | Previous executed command chunk for each observed anchor, in that previous chunk's own frame |
| `previous_action_mask` | H,16,16 | Boolean availability; unavailable entries become zero **after** normalization |
| `view_mask` | H,3 | Boolean camera availability; at least one camera per anchor |
| `anchor_poses` | 2,7 | Optional current measured left/right flange xyz+xyzw in body coordinates |
| `actions` | 16,16 | Left EEF7, right EEF7, body-forward velocity, yaw velocity |

Joint positions are radians. Both state grippers are measured closure in [0,1],
with 0=open and 1=closed. AgiBot Beta's raw measured effector positions use
`clip((position-35)/85,0,1)`; `agibot_measured_closure` implements that calibration.
Other robots must supply their own measured-closure calibration.

Each EEF7 is `[translation3, rotation_vector3, absolute_closure]`:

```text
p_relative[j] = R_anchor.T @ (p_command[j] - p_anchor)
R_relative[j] = R_anchor.T @ R_command[j]
p_command[j]  = p_anchor + R_anchor @ p_relative[j]
R_command[j]  = R_anchor @ Exp(rotation_vector[j])
```

**Hold the measured chunk-start anchor fixed for all 16 outputs.** These are
absolute targets relative to one anchor, not consecutive increments. Do not sum
them or substitute the changing live EEF pose. Poses refer to the flange, not a
calibrated gripper-tip TCP. BASE retains body-forward m/s and yaw rad/s;
the code does not turn these into world-frame motion or joint commands.
Outputs are unnormalized model predictions without implicit clipping; grippers,
velocity limits, TCP transforms and IK belong to the robot integration.

AgiBot command samples are 30 Hz. History anchors are 16 source frames apart
(about 0.5333 s); H4 is four observed policy-call anchors, not four consecutive RGB
frames. Keep the complete observed prefix for both the predictor and causal DA3
deep refinement, and return the last/current slot. Clear histories on episode,
instruction or embodiment changes. For the next policy call, use the previous
**executed** chunk in its original anchor frame. If a controller modified the
commands, re-encode those actual targets with `encode_targets`.

Images resize to 224x224 using Pillow bilinear and DA3 encoder normalization.
Inference does not apply random crops/color jitter, flips, or additional JPEG
encoding. Keep camera order fixed; never duplicate a missing camera. Pass its
slot as zero RGB with `view_mask=False`.

## OXE single-arm inputs

Select the matching checkpoint statistics key ending in
`_eef_relative_c16_right16_v1`; unknown keys fail instead of falling back.
`pad_oxe_state` packs canonical `[xyz, RPY, closure]` into 0:7 and pads 7:16.
`pad_oxe_actions` packs an **already chunk-anchor-relative** action7 into 7:14;
left-arm and BASE entries must be zero and their previous-action mask false.
OXE camera slots use the source's trained order (first external view, then wrist
when available), padded to three; they are not reassigned by arm side.
The returned single-arm action is `result['actions'][:,7:14]`; absent outputs
are explicitly zeroed. The bimanual `anchor_poses` recovery helper is AgiBot-only.
OXE pretraining uses measured next-pose targets, which are not reconstructed
controller command labels. Source-specific frame/TCP conventions still apply.

## File-based inference

Save the fields in the input contract as an NPZ with `numpy.savez`; no object
arrays/pickle are accepted. Text is supplied separately:

```bash
python scripts/infer_mobile.py --bundle mobile-bundle \
  --input observed-history.npz --instruction "Pick up the object." \
  --output prediction.npz
```

## Maintainer export and verification

Export only an owned/trusted training checkpoint. The export reader uses pickle;
the resulting public loader accepts only tensor dictionaries. Use a fresh output
directory. Export preserves complete component weights and embedded statistics,
while excluding optimizer, scheduler, W&B, training paths and dataset metadata.
The unused pooled T5 projection is omitted: the predictor consumes frozen T5
per-token features through its own trained projection.

```bash
python scripts/export_mobile_checkpoint.py \
  --checkpoint /path/to/owned-training-checkpoint.pt \
  --output mobile-bundle \
  --text-revision EXACT_T5_COMMIT_USED_IN_TRAINING \
  --trust-training-checkpoint
PYTHONPATH=src python -m unittest discover -s tests -p 'test_mobile_inference.py' -v
```

Export and tests do not upload anything. Share the bundle separately after the
intended checkpoint is selected. Keep code, weight-release status and actual
robot validation results distinct.

## Verification status

Static checks passed for the Python sources and both CLI help entrypoints. The
current training configuration passes the release schema, while twelve deliberately
incompatible configuration variants are rejected. The whitelist check found no
training paths or run metadata in the exported configuration.

Ten numerical/contract/pipeline unit tests passed in the compute environment.
The first real checkpoint check identified an omitted backbone
`action_steps_per_token=16` export setting; the export now preserves that trained
projection shape. Real checkpoint H1/H4 inference and comparison with the
training implementation are being rechecked. H1 inference passed; H4 exposed a
Q/K versus V dtype mismatch in the public deep causal-attention path. The training
implementation's post-normalization/RoPE dtype correction is now ported, with an
additional regression test. This is a development branch, not a verified robot
deployment release.
