"""Fixed flange-to-body chunk anchors; no simulator or dataset dependencies."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def _poses(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.ndim < 1 or pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError("Poses must contain finite xyz + xyzw")
    norm = np.linalg.norm(pose[..., 3:], axis=-1)
    if not np.allclose(norm, 1., atol=1e-3, rtol=0):
        raise ValueError("Expected unit xyzw quaternions")
    return pose, Rotation.from_quat(pose[..., 3:].reshape(-1, 4)).as_matrix().reshape(*pose.shape[:-1], 3, 3)


def joint_state(left_joints, right_joints, left_closure, right_closure):
    """Interleave measured joint7 + canonical closure for each arm."""
    left, right = np.asarray(left_joints), np.asarray(right_joints)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("Each arm requires seven measured joint angles in radians")
    grip = np.asarray([left_closure, right_closure])
    if not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
        raise ValueError("Measured closure must be in [0,1]")
    out = np.concatenate((left, grip[:1], right, grip[1:])).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite joint state")
    return out


def agibot_measured_closure(raw_position):
    """AgiBot Beta calibration only: 35=open, 120=closed."""
    value = np.asarray(raw_position, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite measured gripper position")
    return np.clip((value - 35.) / 85., 0., 1.).astype(np.float32)


def encode_targets(target_poses, anchor_poses, grippers, base_velocity):
    """Encode one C-step bimanual chunk with the SAME measured anchor for all C."""
    target, rt = _poses(target_poses)
    anchor, ra = _poses(anchor_poses)
    if target.ndim != 3 or target.shape[1:] != (2, 7) or anchor.shape != (2, 7):
        raise ValueError("Expected targets(C,2,7), anchors(2,7)")
    c = len(target)
    grip, base = np.asarray(grippers), np.asarray(base_velocity)
    if grip.shape != (c, 2) or base.shape != (c, 2) or not np.isfinite(grip).all() or not np.isfinite(base).all():
        raise ValueError("Expected finite grippers(C,2), base velocity(C,2)")
    inverse = ra.swapaxes(-1, -2)[None]
    pos = (inverse @ (target[..., :3] - anchor[None, :, :3])[..., None])[..., 0]
    rot = Rotation.from_matrix((inverse @ rt).reshape(-1, 3, 3)).as_rotvec().reshape(c, 2, 3)
    return np.concatenate((np.concatenate((pos, rot, grip[..., None]), -1).reshape(c, 14), base), -1).astype(np.float32)


def recover_targets(actions, anchor_poses):
    """Return body-frame flange poses(C,2,7), closure(C,2), body vx/yaw(C,2).

    Outputs are absolute targets relative to a FIXED measured chunk anchor.
    Do not sum the predictions or replace the anchor while executing a chunk.
    """
    a = np.asarray(actions, dtype=np.float64)
    anchor, rotation = _poses(anchor_poses)
    if a.ndim != 2 or a.shape[-1] != 16 or anchor.shape != (2, 7) or not np.isfinite(a).all():
        raise ValueError("Expected finite actions(C,16), anchors(2,7)")
    arms = a[:, :14].reshape(-1, 2, 7)
    pos = (rotation[None] @ arms[..., :3, None])[..., 0] + anchor[None, :, :3]
    delta = Rotation.from_rotvec(arms[..., 3:6].reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3)
    quat = Rotation.from_matrix((rotation[None] @ delta).reshape(-1, 3, 3)).as_quat().reshape(-1, 2, 4)
    return {"flange_poses": np.concatenate((pos, quat), -1).astype(np.float32),
            "gripper_closure": arms[..., 6].astype(np.float32),
            "base_velocity": a[:, 14:16].astype(np.float32)}


def pad_oxe_state(state):
    """OXE canonical [xyz,rpy,closure] occupies slots 0:7; joint-state slots differ."""
    x = np.asarray(state, dtype=np.float32)
    if x.ndim < 1 or x.shape[-1] != 7 or not np.isfinite(x).all():
        raise ValueError("Expected finite OXE state(...,7)")
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 9)])


def pad_oxe_actions(actions):
    """Already chunk-anchor-relative OXE7 goes in right-arm slots 7:14."""
    x = np.asarray(actions, dtype=np.float32)
    if x.ndim < 1 or x.shape[-1] != 7 or not np.isfinite(x).all():
        raise ValueError("Expected finite EEF-relative OXE actions(...,7)")
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(7, 2)])
