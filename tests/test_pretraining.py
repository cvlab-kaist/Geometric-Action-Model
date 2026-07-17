from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch import nn

from gam.training.data import collate_fn
from robot.data.pretraining import (
    DatasetMixture,
    MimicGenSequenceDataset,
    OXE_SPECS,
    PAPER_SOURCE_RATIOS,
    canonicalize_action,
    canonicalize_state,
)


class _NamedDataset(torch.utils.data.Dataset):
    def __init__(self, name: str, length: int = 11):
        self.name = name
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"value": index, "mixture_source": self.name}


class PretrainingRegistryTest(unittest.TestCase):
    def test_registry_and_paper_ratios(self):
        self.assertEqual(len(OXE_SPECS), 23)
        self.assertEqual(len({spec.name for spec in OXE_SPECS}), 23)
        self.assertAlmostEqual(sum(PAPER_SOURCE_RATIOS.values()), 1.0)

    def test_source_schedule(self):
        sources = [
            ("open_x_embodiment", _NamedDataset("oxe"), 0.72),
            ("mimicgen", _NamedDataset("mimicgen"), 0.18),
            ("robocasa365", _NamedDataset("robocasa"), 0.10),
        ]
        mixture = DatasetMixture(sources, epoch_size=10_000)
        self.assertEqual(mixture.counts.tolist(), [7200, 1800, 1000])

    def test_canonical_action_shapes(self):
        state = torch.zeros(4, 8)
        state[:, 6] = 1.0
        for spec in OXE_SPECS:
            raw_dim = {
                "droid_target": 6,
                "nyu_franka": 15,
                "drop_last": 8,
                "xy_only": 2,
            }.get(spec.action_transform, 7)
            raw = torch.zeros(4, raw_dim)
            item = {
                "observation.state.cartesian_position": torch.zeros(4, 6),
                "action.gripper_position": torch.zeros(4, 1),
            }
            output = canonicalize_action(raw, spec, state=state, item=item, fps=10.0)
            self.assertEqual(tuple(output.shape), (4, 7), spec.name)
            self.assertTrue(torch.isfinite(output).all(), spec.name)

    def test_canonical_state_shapes(self):
        cases = {
            "pos_euler_8d": torch.zeros(2, 8),
            "pos_euler_7d": torch.zeros(2, 7),
            "pos_quat": torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0]]).repeat(2, 1),
            "pos_quat_no_grip": torch.tensor([[0, 0, 0, 0, 0, 0, 1]]).repeat(2, 1),
            "droid_state": torch.zeros(2, 7),
            "zero": torch.zeros(2, 8),
            "austin_buds": torch.zeros(2, 24),
            "nyu_franka": torch.zeros(2, 13),
            "cmu_stretch": torch.zeros(2, 4),
            "language_table": torch.zeros(2, 8),
            "robocasa": torch.tensor(
                [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
            ).repeat(2, 1),
        }
        for transform, raw in cases.items():
            output = canonicalize_state(raw, transform)
            self.assertEqual(tuple(output.shape), (2, 7), transform)
            self.assertTrue(torch.isfinite(output).all(), transform)


class MimicGenLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data_root = root / "mimicgen"
        self.depth_root = root / "depth"
        (self.data_root / "core").mkdir(parents=True)
        (self.depth_root / "core").mkdir(parents=True)
        (self.data_root / "task_descriptions.json").write_text(
            json.dumps({"Coffee": "Prepare coffee."}), encoding="utf-8"
        )
        self._write_hdf5(self.data_root / "core" / "coffee_d0.hdf5")
        self._write_depth(self.depth_root / "core" / "coffee_d0__demo_0.npz")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_hdf5(path: Path):
        rng = np.random.default_rng(0)
        n_actions = 24
        with h5py.File(path, "w") as handle:
            demo = handle.create_group("data/demo_0")
            actions = np.zeros((n_actions, 7), dtype=np.float32)
            actions[:, :6] = 0.5
            actions[:, 6] = np.linspace(-1.0, 1.0, n_actions)
            demo.create_dataset("actions", data=actions)
            obs = demo.create_group("obs")
            frames = rng.integers(0, 256, (n_actions + 1, 24, 32, 3), dtype=np.uint8)
            obs.create_dataset("agentview_image", data=frames)
            obs.create_dataset("robot0_eye_in_hand_image", data=frames[:, :, ::-1])
            obs.create_dataset("robot0_eef_pos", data=np.zeros((n_actions + 1, 3), np.float32))
            quat = np.zeros((n_actions + 1, 4), np.float32)
            quat[:, 3] = 1.0
            obs.create_dataset("robot0_eef_quat", data=quat)
            obs.create_dataset("robot0_gripper_qpos", data=np.zeros((n_actions + 1, 2), np.float32))

    @staticmethod
    def _write_depth(path: Path):
        n_frames = 25
        intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_frames, 2, 1, 1))
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 2, 1, 1))
        np.savez_compressed(
            path,
            depth_meters=np.ones((n_frames, 2, 12, 16), dtype=np.float32),
            frame_indices=np.arange(n_frames, dtype=np.int64),
            camera_names=np.asarray(["agentview", "robot0_eye_in_hand"]),
            camera_intrinsics=intrinsics,
            camera_extrinsics_c2w=extrinsics,
        )

    def test_loader_collate_and_backward(self):
        dataset = MimicGenSequenceDataset(
            self.data_root,
            depth_root=self.depth_root,
            image_size=(32, 32),
            future_steps=1,
            chunk_size=8,
            include_current_action=True,
            n_views=2,
            eval_ratio=0.0,
            is_eval=True,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["all_view_images"].shape), (2, 2, 3, 32, 32))
        self.assertEqual(tuple(sample["actions"].shape), (2, 8, 7))
        self.assertEqual(tuple(sample["proprioception"].shape), (2, 7))
        self.assertEqual(tuple(sample["gt_depth_meters"].shape), (2, 2, 32, 32))
        self.assertEqual(sample["task_description"], "Prepare coffee.")

        batch = collate_fn([sample, sample])
        self.assertEqual(tuple(batch["all_view_images"].shape), (2, 2, 2, 3, 32, 32))
        model = nn.Sequential(nn.Flatten(), nn.Linear(2 * 2 * 3 * 32 * 32, 2 * 8 * 7))
        prediction = model(batch["all_view_images"])
        loss = (prediction - batch["actions"].flatten(1)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
