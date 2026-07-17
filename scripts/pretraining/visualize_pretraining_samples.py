#!/usr/bin/env python3
"""Render one actual sample from every GAM pretraining dataset leaf."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from robot.data.pretraining import DatasetConcat, DatasetMixture, build_pretraining_dataset


def _tensor_image(value: torch.Tensor) -> Image.Image:
    image = value.detach().cpu().float().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _depth_image(value: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    depth = value.detach().cpu().float().numpy()
    valid = mask.detach().cpu().numpy().astype(bool)
    output = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        output[valid] = np.clip((depth[valid] - lo) * 255.0 / max(hi - lo, 1.0e-6), 0, 255)
    return Image.fromarray(output, mode="L").convert("RGB")


def _labeled(image: Image.Image, text: str) -> Image.Image:
    label = Image.new("RGB", (image.width, 18), "white")
    ImageDraw.Draw(label).text((3, 3), text, fill="black")
    output = Image.new("RGB", (image.width, image.height + label.height), "white")
    output.paste(label, (0, 0))
    output.paste(image, (0, label.height))
    return output


def _tile(images: list[Image.Image], columns: int) -> Image.Image:
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), "black")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    return canvas


def _render_sample(sample: dict) -> Image.Image:
    views = sample["all_view_images"]
    camera_keys = list(sample["camera_keys"])
    panels = [
        _labeled(_tensor_image(views[time, view]), f"rgb t={time} {camera_keys[view]}")
        for time in range(views.shape[0]) for view in range(views.shape[1])
    ]
    if "gt_depth_meters" in sample:
        depth = sample["gt_depth_meters"]
        mask = sample["gt_depth_mask"]
        panels.extend(
            _labeled(_depth_image(depth[time, view], mask[time, view]), f"depth t={time} {camera_keys[view]}")
            for time in range(depth.shape[0]) for view in range(depth.shape[1])
        )
    image = _tile(panels, int(views.shape[1]))
    header = Image.new("RGB", (image.width, 26), "white")
    text = f"{sample['dataset_name']}  episode={sample['episode_id']}  views={views.shape[1]}  action={tuple(sample['actions'].shape)}"
    ImageDraw.Draw(header).text((5, 6), text, fill="black")
    output = Image.new("RGB", (image.width, image.height + header.height), "white")
    output.paste(header, (0, 0))
    output.paste(image, (0, header.height))
    return output


def _leaves(dataset) -> Iterable:
    if isinstance(dataset, DatasetMixture):
        for child in dataset.source_list:
            yield from _leaves(child)
    elif isinstance(dataset, DatasetConcat):
        for child in dataset.children:
            yield from _leaves(child)
    else:
        yield dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text())
    config = payload.get("dataset", payload)
    random.seed(0)
    torch.manual_seed(0)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = build_pretraining_dataset(config, is_eval=False)
    rendered = []
    for leaf in _leaves(dataset):
        sample = leaf[0]
        name = str(sample["dataset_name"])
        path = output_root / f"{name}.png"
        _render_sample(sample).save(path)
        rendered.append({"dataset": name, "path": str(path), "episode": sample["episode_id"]})
        print(f"wrote {path}")
    (output_root / "index.json").write_text(json.dumps(rendered, indent=2) + "\n")


if __name__ == "__main__":
    main()
