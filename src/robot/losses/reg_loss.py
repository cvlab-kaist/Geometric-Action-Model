"""Feature regularization for DA3 fine-tuning."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRegularizer:
    """Preserve pretrained DA3 features during action fine-tuning.

    Later layers (closer to output) get stronger regularization since they
    diverge more during fine-tuning. Weights scale linearly from
    layer_weight_min (level 0 / earliest) to 1.0 (level N / latest).

    Args:
        lambda_feat: Base regularization strength.
        layer_weight_min: Weight for the earliest level (default 0.25).
            Latest level always gets weight 1.0.
            E.g. with 4 levels and min=0.25: [0.25, 0.5, 0.75, 1.0]
        lambda_depth: Weight for depth output regularization loss.
        lambda_camera: Weight for camera output regularization loss.
    """

    def __init__(
        self,
        lambda_feat: float = 0.01,
        layer_weight_min: float = 0.25,
        lambda_depth: float = 0.0,
        lambda_camera: float = 0.0,
        depth_loss_type: str = "da3_style",
        lambda_depth_conf: float = 0.2,
        lambda_ray: float = 0.0,
        lambda_point: float = 0.0,
        lambda_camera_pose: float = 0.0,
    ):
        self.lambda_feat = float(lambda_feat)
        self.layer_weight_min = float(layer_weight_min)
        self.lambda_depth = float(lambda_depth)
        self.lambda_camera = float(lambda_camera)
        # DA3 paper full depth-loss knobs (arXiv 2511.10647 Section 3.3).
        # depth_loss_type="da3_style" keeps the subset loss (masked L1 + grad L1)
        # for backward compatibility; "da3_full" adds confidence-weighted L_D,
        # ray L_M, point-map L_P, and optional camera-pose L_C.
        self.depth_loss_type = str(depth_loss_type).lower()
        self.lambda_depth_conf = float(lambda_depth_conf)
        self.lambda_ray = float(lambda_ray)
        self.lambda_point = float(lambda_point)
        self.lambda_camera_pose = float(lambda_camera_pose)
        if self.depth_loss_type not in {"da3_style", "da3_full"}:
            raise ValueError(
                f"Unknown depth_loss_type={self.depth_loss_type!r}. "
                "Must be 'da3_style' or 'da3_full'."
            )

    def feature_reg_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(student_features) != len(teacher_features):
            raise ValueError("Student/teacher feature levels mismatch.")

        n_levels = len(student_features)
        loss = student_features[0].new_tensor(0.0)
        total_weight = 0.0

        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # Skip action token (index 1) in student
            student_cmp = torch.cat([student_feat[:, :, :1], student_feat[:, :, 2:]], dim=2)
            # Linear weight: min at level 0, 1.0 at last level
            if n_levels > 1:
                w = self.layer_weight_min + (1.0 - self.layer_weight_min) * i / (n_levels - 1)
            else:
                w = 1.0
            loss = loss + w * F.mse_loss(student_cmp, teacher_feat.detach())
            total_weight += w

        return loss / max(total_weight, 1e-8)

    @staticmethod
    def depth_reg_loss(
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between student and teacher depth predictions."""
        return F.mse_loss(student_depth, teacher_depth.detach())

    @staticmethod
    def camera_reg_loss(
        student_pose_enc: torch.Tensor,
        teacher_pose_enc: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between student and teacher camera pose encodings."""
        return F.mse_loss(student_pose_enc, teacher_pose_enc.detach())
