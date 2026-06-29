"""DA3-Giant gam action model : LIBERO / LIBERO-Plus."""

import importlib
import sys

from .modeling import (
    ActionHeadV2,
    ActionTokenizer,
    DA3GiantEncoder,
    GAMFuturePredictor,
    ProprioConditioner,
    TextConditioner,
    build_future_predictor,
)

_LEGACY_SUBMODULES = {
    "action_head_oft": "robot.modeling.action_head_oft",
    "action_head_v2": "robot.modeling.action_head_v2",
    "action_tokenizer": "robot.modeling.action_tokenizer",
    "backbone_factory": "robot.modeling.backbone_factory",
    "conditioning": "robot.modeling.conditioning",
    "da3_giant_encoder": "robot.modeling.da3_giant_encoder",
    "future_predictor": "robot.modeling.future_predictor",
    "sigreg": "robot.modeling.sigreg",
    "dataset": "robot.data.dataset",
    "closed_loop_libero_eval": "robot.evaluation.closed_loop_libero_eval",
    "rollout_env": "robot.evaluation.rollout_env",
    "reg_loss": "robot.losses.reg_loss",
    "unified_loss": "robot.losses.unified_loss",
    "visualization": "robot.viz.visualization",
}

for _old_name, _new_name in _LEGACY_SUBMODULES.items():
    sys.modules.setdefault(f"{__name__}.{_old_name}", importlib.import_module(_new_name))

__all__ = [
    "ActionHeadV2",
    "DA3GiantEncoder",
    "GAMFuturePredictor",
    "ActionTokenizer",
    "TextConditioner",
    "ProprioConditioner",
    "build_future_predictor",
]
