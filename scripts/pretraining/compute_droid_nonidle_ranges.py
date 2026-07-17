#!/usr/bin/env python3
"""Build DROID non-idle and missing-file filters from a local LeRobot repo."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_REPO_ID = "lerobot/droid_1.0.1"
DEFAULT_CAMERAS = (
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _truthy(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "none", "null", "nan", "not provided"}


def _keep_ranges(
    joint_velocity: np.ndarray,
    *,
    min_idle_len: int,
    min_nonidle_len: int,
    trim_tail: int,
    threshold: float,
) -> list[list[int]]:
    velocity = np.asarray(joint_velocity, dtype=np.float32)
    if velocity.ndim != 2 or not len(velocity):
        return []
    idle = np.concatenate((np.array([False]), np.all(np.abs(np.diff(velocity, axis=0)) < threshold, axis=1)))
    padded = np.concatenate((np.array([False]), idle, np.array([False])))
    changes = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    keep = np.ones(len(velocity), dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        if end - start >= min_idle_len:
            keep[start:end] = False
    padded = np.concatenate((np.array([False]), keep, np.array([False])))
    changes = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [[int(start), int(end - trim_tail)] for start, end in zip(starts, ends, strict=True)
            if end - start >= min_nonidle_len and end - trim_tail > start]


def _list_array(table: pq.Table, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    lengths = np.asarray(pc.list_value_length(array).to_numpy(zero_copy_only=False), dtype=np.int64)
    if not len(lengths):
        return np.empty((0, 0), dtype=np.float32)
    if not np.all(lengths == lengths[0]):
        raise ValueError(f"{key} has variable-length rows.")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(len(lengths), int(lengths[0]))


def _local_episodes(metadata: Any, root: Path, cameras: Iterable[str]) -> tuple[list[int], list[int]]:
    available, missing = [], []
    for episode_index in range(int(metadata.total_episodes)):
        data_path = root / metadata.get_data_file_path(episode_index)
        video_paths = [root / metadata.get_video_file_path(episode_index, key) for key in cameras]
        if data_path.exists() and all(path.exists() for path in video_paths):
            available.append(episode_index)
        else:
            missing.append(episode_index)
    return available, missing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openx-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--blacklist-output", type=Path, default=None)
    parser.add_argument("--max-parquet-files", type=int, default=None)
    parser.add_argument("--min-idle-len", type=int, default=7)
    parser.add_argument("--min-nonidle-len", type=int, default=16)
    parser.add_argument("--trim-tail", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = args.openx_root.expanduser().resolve()
    dataset_root = root / args.repo_id
    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    available, missing = _local_episodes(metadata, dataset_root, DEFAULT_CAMERAS)
    data_files = sorted({dataset_root / metadata.get_data_file_path(index) for index in available})
    if args.max_parquet_files is not None:
        data_files = data_files[: max(0, int(args.max_parquet_files))]
    if not data_files:
        raise FileNotFoundError(f"No local DROID parquet files under {dataset_root}")

    ranges: dict[str, list[list[int]]] = {}
    processed = set()
    columns = [
        "episode_index", "action.joint_velocity", "language_instruction",
        "language_instruction_2", "language_instruction_3", "is_episode_successful", "reward",
    ]
    for path in data_files:
        table = pq.read_table(path, columns=columns)
        episode_index = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        velocity = _list_array(table, "action.joint_velocity")
        language_columns = [table[key].combine_chunks().to_pylist() for key in columns[2:5]]
        successful = np.asarray(table["is_episode_successful"].to_numpy(zero_copy_only=False), dtype=bool)
        reward = np.asarray(table["reward"].to_numpy(zero_copy_only=False), dtype=np.float32)
        boundaries = np.flatnonzero(episode_index[1:] != episode_index[:-1]) + 1
        for start, end in zip(np.r_[0, boundaries], np.r_[boundaries, len(episode_index)], strict=True):
            index = int(episode_index[start])
            processed.add(index)
            has_language = any(_truthy(values[position]) for values in language_columns for position in range(start, end))
            has_success = bool(successful[start:end].any() or np.abs(reward[start:end]).any())
            if not has_language or not has_success:
                continue
            keep = _keep_ranges(
                velocity[start:end], min_idle_len=args.min_idle_len,
                min_nonidle_len=args.min_nonidle_len, trim_tail=args.trim_tail,
                threshold=args.threshold,
            )
            if keep:
                ranges[str(index)] = keep

    blacklist = sorted(set(missing) | (set(available) - processed))
    output = args.output or root / "_stats" / "droid_openpi_nonidle_ranges.json"
    blacklist_output = args.blacklist_output or root / "_stats" / "droid_blacklist_eps.json"
    _write_json(output, ranges)
    _write_json(blacklist_output, blacklist)
    print(json.dumps({
        "available_episodes": len(available), "blacklisted_episodes": len(blacklist),
        "processed_episodes": len(processed), "episodes_with_nonidle_ranges": len(ranges),
        "parquet_files": len(data_files), "ranges": str(output), "blacklist": str(blacklist_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
