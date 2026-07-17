#!/usr/bin/env python3
"""Download the RoboCasa365 human pretraining datasets to a chosen root."""

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--split", nargs="+", default=["pretrain"], choices=["pretrain", "target"])
    parser.add_argument("--source", nargs="+", default=["human"], choices=["human", "mimicgen"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-v30-conversion", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    return parser.parse_args()


def convert_v21_dataset(dataset_root: Path) -> None:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        return
    try:
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    except ImportError as exc:
        raise RuntimeError("RoboCasa v2.1 conversion requires lerobot==0.4.4 and jsonlines.") from exc
    convert_dataset(
        repo_id=dataset_root.name,
        root=dataset_root.parent,
        push_to_hub=False,
    )
    legacy_root = dataset_root.parent / f"{dataset_root.name}_old"
    legacy_extras = legacy_root / "extras"
    if legacy_extras.is_dir() and not (dataset_root / "extras").exists():
        (dataset_root / "extras").symlink_to(os.path.relpath(legacy_extras, dataset_root))


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.convert_only:
        import robocasa.macros as macros

        macros.DATASET_BASE_PATH = str(output_root)
        from robocasa.scripts.download_datasets import download_datasets

        download_datasets(
            split=args.split,
            tasks=args.tasks,
            source=args.source,
            overwrite=args.overwrite,
            dryrun=args.dry_run,
        )
    if not args.dry_run and not args.skip_v30_conversion:
        for info_path in sorted((output_root / "v1.0").rglob("meta/info.json")):
            convert_v21_dataset(info_path.parent.parent)
    print(f"RoboCasa365 root: {output_root / 'v1.0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
