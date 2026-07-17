#!/usr/bin/env python3
"""Download the RoboCasa365 human pretraining datasets to a chosen root."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--split", nargs="+", default=["pretrain"], choices=["pretrain", "target"])
    parser.add_argument("--source", nargs="+", default=["human"], choices=["human", "mimicgen"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
    print(f"RoboCasa365 root: {output_root / 'v1.0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
