# Robot Package Layout

`src/robot` contains domain code used by the GAM training and rollout
entrypoints.

| Package | Contents |
|---------|----------|
| `modeling/` | DA3 backbone wrapper, GAM future predictor, action heads, conditioning |
| `data/` | LIBERO HDF5 dataset, action and proprioception normalizers |
| `evaluation/` | LIBERO and LIBERO-Plus rollout environments and in-training eval |
| `losses/` | GAM forward losses, depth losses, feature regularization |
| `viz/` | Training and rollout diagnostic visualizations |

Legacy imports such as `robot.dataset` and `robot.future_predictor` are aliased
from `robot.__init__` for existing scripts.
