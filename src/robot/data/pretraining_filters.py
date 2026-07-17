"""Episode filters shared by the GAM pretraining loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def resolve_filter_path(path: str | Path | None, root: Path) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def load_episode_blacklist(path: str | Path | None) -> set[int]:
    if path is None or not Path(path).exists():
        return set()
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, Mapping):
        payload = payload.get("episodes", [])
    return {int(value) for value in payload}


def load_nonidle_ranges(path: str | Path | None) -> Optional[dict[int, list[tuple[int, int]]]]:
    if path is None or not Path(path).exists():
        return None
    payload = json.loads(Path(path).read_text())
    ranges: dict[int, list[tuple[int, int]]] = {}
    for raw_episode, raw_ranges in payload.items():
        if isinstance(raw_ranges, Mapping):
            raw_ranges = [raw_ranges]
        if not isinstance(raw_ranges, list):
            continue
        parsed = []
        for raw_range in raw_ranges:
            if isinstance(raw_range, Mapping):
                start, end = raw_range.get("start"), raw_range.get("end")
            elif isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
                start, end = raw_range[:2]
            else:
                continue
            try:
                start_i, end_i = int(start), int(end)
            except (TypeError, ValueError):
                continue
            if end_i > start_i:
                parsed.append((start_i, end_i))
        if parsed:
            ranges[int(raw_episode)] = parsed
    return ranges

