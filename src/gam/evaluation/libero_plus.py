"""LIBERO-Plus metadata utilities.

Pure data helpers split out of ``eval_libero_unified.py`` (behavior-preserving
extraction). This module must remain free of any import of
``eval_libero_unified`` to avoid a circular import; ``eval_libero_unified``
re-imports these names so the public API is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any


LIBERO_PLUS_PERTURBATION_ALIASES = {
    "language": "language",
    "lang": "language",
    "view": "view",
    "camera": "view",
    "table": "table",
    "tabletop": "table",
    "tb": "table",
    "light": "light",
    "lighting": "light",
    "object": "object",
    "newobj": "object",
    "add": "object",
    "level": "object",
    "base": "base",
    "clean": "base",
    "original": "base",
}
LIBERO_PLUS_OFFICIAL_CATEGORY_SLUGS = {
    "Background Textures": "background",
    "Camera Viewpoints": "camera",
    "Language Instructions": "language",
    "Light Conditions": "light",
    "Objects Layout": "layout",
    "Robot Initial States": "robot",
    "Sensor Noise": "noise",
}
LIBERO_PLUS_OFFICIAL_CATEGORY_ALIASES = {
    "background": "background",
    "background_texture": "background",
    "background_textures": "background",
    "camera": "camera",
    "camera_viewpoint": "camera",
    "camera_viewpoints": "camera",
    "language": "language",
    "language_instruction": "language",
    "language_instructions": "language",
    "light": "light",
    "light_condition": "light",
    "light_conditions": "light",
    "layout": "layout",
    "object_layout": "layout",
    "objects_layout": "layout",
    "robot": "robot",
    "robot_initial_state": "robot",
    "robot_initial_states": "robot",
    "initstate": "robot",
    "initial_state": "robot",
    "initial_states": "robot",
    "noise": "noise",
    "sensor_noise": "noise",
}


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _row_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0



def classify_libero_plus_perturbation(task_meta: dict[str, Any]) -> str:
    text = " ".join(
        str(task_meta.get(key, ""))
        for key in ("name", "bddl_file", "init_states_file")
    ).lower()
    if "_language_" in text:
        return "language"
    if "_view_" in text:
        return "view"
    if "_table_" in text or "_tb_" in text:
        return "table"
    if "_light_" in text:
        return "light"
    if "_add_" in text or "_level" in text:
        return "object"
    return "base"


def filter_libero_plus_task_metadata(
    task_metadata: list[dict[str, Any]],
    perturbation: str,
) -> list[dict[str, Any]]:
    tokens = [token.lower() for token in parse_csv_list(perturbation)]
    if not tokens or "all" in tokens:
        return task_metadata

    wanted: set[str] = set()
    unknown: list[str] = []
    for token in tokens:
        canonical = LIBERO_PLUS_PERTURBATION_ALIASES.get(token)
        if canonical is None:
            unknown.append(token)
        else:
            wanted.add(canonical)
    if unknown:
        choices = sorted(set(LIBERO_PLUS_PERTURBATION_ALIASES) | {"all"})
        raise ValueError(f"Unknown --plus-perturbation value(s) {unknown}; expected one of {choices}")

    filtered = [item for item in task_metadata if str(item.get("plus_perturbation")) in wanted]
    if not filtered:
        raise ValueError(
            f"No LIBERO-Plus tasks matched --plus-perturbation={perturbation!r}. "
            f"Available categories: {sorted({item.get('plus_perturbation') for item in task_metadata})}"
        )
    return filtered


def filter_libero_plus_official_category_metadata(
    task_metadata: list[dict[str, Any]],
    category: str | None,
) -> list[dict[str, Any]]:
    tokens = [_category_slug(token) for token in parse_csv_list(category)]
    if not tokens or "all" in tokens:
        return task_metadata

    wanted: set[str] = set()
    unknown: list[str] = []
    for token in tokens:
        canonical = LIBERO_PLUS_OFFICIAL_CATEGORY_ALIASES.get(token)
        if canonical is None:
            unknown.append(token)
        else:
            wanted.add(canonical)
    if unknown:
        choices = sorted(set(LIBERO_PLUS_OFFICIAL_CATEGORY_ALIASES) | {"all"})
        raise ValueError(f"Unknown --plus-official-category value(s) {unknown}; expected one of {choices}")

    filtered = [
        item
        for item in task_metadata
        if str(item.get("plus_official_category_slug") or "") in wanted
    ]
    if not filtered:
        available = sorted(
            {
                str(item.get("plus_official_category_slug") or item.get("plus_official_category") or "")
                for item in task_metadata
                if item.get("plus_official_category_slug") or item.get("plus_official_category")
            }
        )
        raise ValueError(
            f"No LIBERO-Plus tasks matched --plus-official-category={category!r}. "
            f"Available official categories: {available}"
        )
    return filtered


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _category_slug(category: Any) -> str:
    text = str(category or "").strip()
    if not text:
        return ""
    if text in LIBERO_PLUS_OFFICIAL_CATEGORY_SLUGS:
        return LIBERO_PLUS_OFFICIAL_CATEGORY_SLUGS[text]
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug


def _task_name_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).stem.lower()


def _classification_row_from_mapping(key: Any, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        row = dict(value)
    else:
        row = {"category": value}
    if "id" not in row and "task_id" not in row:
        maybe_id = _safe_int(key)
        if maybe_id is not None:
            row["id"] = maybe_id
        else:
            row["name"] = str(key)
    return row


def _iter_classification_rows(rows: Any) -> list[tuple[int, dict[str, Any]]]:
    if isinstance(rows, list):
        result: list[tuple[int, dict[str, Any]]] = []
        for idx, value in enumerate(rows):
            if isinstance(value, dict):
                result.append((idx, dict(value)))
            else:
                result.append((idx, {"category": value}))
        return result
    if isinstance(rows, dict):
        return [
            (idx, _classification_row_from_mapping(key, value))
            for idx, (key, value) in enumerate(rows.items())
        ]
    return []


def load_libero_plus_task_classification(
    plus_root: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Load official LIBERO-Plus task categories.

    The current upstream JSON stores 1-based official ids in task order. The
    evaluator's internal ``task_id`` is 0-based, so we index both by
    ``official_id - 1`` and by task name for compatibility with future JSON
    layouts.
    """
    info: dict[str, Any] = {
        "loaded": False,
        "path": None,
        "by_suite": {},
        "suite_task_counts": {},
        "category_task_counts": {},
    }
    if plus_root is None:
        return info

    path = Path(plus_root).expanduser().resolve() / "libero" / "libero" / "benchmark" / "task_classification.json"
    info["path"] = str(path)
    if not path.exists():
        print(f"  [plus-warning] official task_classification.json missing at {path}")
        return info

    with path.open("r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        print(f"  [plus-warning] unexpected task_classification.json root type: {type(raw).__name__}")
        return info

    by_suite: dict[str, dict[str, Any]] = {}
    suite_counts: dict[str, int] = {}
    category_counts: dict[str, dict[str, int]] = {}
    for suite, rows in raw.items():
        by_internal_id: dict[int, dict[str, Any]] = {}
        by_name: dict[str, dict[str, Any]] = {}
        for fallback_idx, row in _iter_classification_rows(rows):
            official_id = _safe_int(row.get("id", row.get("task_id")))
            internal_id = official_id - 1 if official_id is not None and official_id > 0 else fallback_idx
            category = str(row.get("category", "") or "")
            slug = _category_slug(category)
            entry = {
                "official_task_id": official_id if official_id is not None else fallback_idx + 1,
                "category": category,
                "category_slug": slug,
                "difficulty_level": row.get("difficulty_level", row.get("difficulty")),
                "name": str(row.get("name", "") or ""),
            }
            by_internal_id[int(internal_id)] = entry
            for name_key in (
                _task_name_key(row.get("name")),
                _task_name_key(row.get("bddl_file")),
                _task_name_key(row.get("task")),
            ):
                if name_key:
                    by_name[name_key] = entry
            category_counts.setdefault(str(suite), {})
            category_counts[str(suite)][slug or "unknown"] = category_counts[str(suite)].get(slug or "unknown", 0) + 1
        by_suite[str(suite)] = {"by_internal_id": by_internal_id, "by_name": by_name}
        suite_counts[str(suite)] = len(by_internal_id)

    info.update(
        {
            "loaded": True,
            "by_suite": by_suite,
            "suite_task_counts": suite_counts,
            "category_task_counts": category_counts,
        }
    )
    return info


def annotate_libero_plus_official_categories(
    task_metadata: list[dict[str, Any]],
    suite: str,
    classification: dict[str, Any],
) -> None:
    suite_info = classification.get("by_suite", {}).get(suite, {})
    by_internal_id = suite_info.get("by_internal_id", {})
    by_name = suite_info.get("by_name", {})
    for item in task_metadata:
        entry = by_internal_id.get(int(item.get("task_id", -1)))
        if entry is None:
            for key in (
                _task_name_key(item.get("name")),
                _task_name_key(item.get("bddl_file")),
                _task_name_key(item.get("init_states_file")),
                _task_name_key(item.get("language")),
            ):
                entry = by_name.get(key)
                if entry is not None:
                    break
        if entry is None:
            item["plus_official_task_id"] = ""
            item["plus_official_category"] = ""
            item["plus_official_category_slug"] = ""
            item["plus_official_difficulty_level"] = ""
            continue
        item["plus_official_task_id"] = entry.get("official_task_id", "")
        item["plus_official_category"] = entry.get("category", "")
        item["plus_official_category_slug"] = entry.get("category_slug", "")
        item["plus_official_difficulty_level"] = entry.get("difficulty_level", "")


def aggregate_plus_category_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = str(row.get("plus_official_category_slug", "") or "")
        if not slug:
            continue
        category = str(row.get("plus_official_category", "") or slug)
        task_key = (
            str(row.get("suite", "")),
            _row_int(row, "eval_task_index"),
            _row_int(row, "task_id"),
        )
        bucket = aggregates.setdefault(
            slug,
            {
                "category": category,
                "success_rate": 0.0,
                "num_trials": 0,
                "num_success": 0,
                "num_tasks": 0,
                "_task_keys": set(),
            },
        )
        bucket["num_trials"] += int(row.get("num_trials", 0) or 0)
        bucket["num_success"] += int(row.get("num_success", 0) or 0)
        bucket["_task_keys"].add(task_key)
    for bucket in aggregates.values():
        trials = int(bucket["num_trials"])
        bucket["success_rate"] = float(bucket["num_success"] / trials) if trials else 0.0
        task_keys = bucket.pop("_task_keys", set())
        bucket["num_tasks"] = len(task_keys)
    return dict(sorted(aggregates.items()))


def aggregate_plus_perturbation_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = str(row.get("plus_perturbation", "") or "")
        if not slug:
            continue
        task_key = (
            str(row.get("suite", "")),
            _row_int(row, "eval_task_index"),
            _row_int(row, "task_id"),
        )
        bucket = aggregates.setdefault(
            slug,
            {
                "success_rate": 0.0,
                "num_trials": 0,
                "num_success": 0,
                "num_tasks": 0,
                "_task_keys": set(),
            },
        )
        bucket["num_trials"] += int(row.get("num_trials", 0) or 0)
        bucket["num_success"] += int(row.get("num_success", 0) or 0)
        bucket["_task_keys"].add(task_key)
    for bucket in aggregates.values():
        trials = int(bucket["num_trials"])
        bucket["success_rate"] = float(bucket["num_success"] / trials) if trials else 0.0
        task_keys = bucket.pop("_task_keys", set())
        bucket["num_tasks"] = len(task_keys)
    return dict(sorted(aggregates.items()))


def select_eval_task_entries(
    task_metadata: list[dict[str, Any]],
    task_ids: list[int] | None,
) -> list[dict[str, Any]]:
    if task_ids is None:
        selected = list(enumerate(task_metadata))
    else:
        selected = []
        for index in task_ids:
            if index < 0 or index >= len(task_metadata):
                raise IndexError(f"Task index {index} is out of range for filtered task list of size {len(task_metadata)}")
            selected.append((index, task_metadata[index]))
    entries: list[dict[str, Any]] = []
    for eval_index, item in selected:
        entry = dict(item)
        entry["eval_task_index"] = int(eval_index)
        entries.append(entry)
    return entries


def _plus_subset_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("suite", "")),
        str(item.get("plus_official_task_id", "")),
        int(item.get("task_id", 0)),
        str(item.get("bddl_file", "")),
        str(item.get("language", "")),
    )


def _plus_subset_group_key(item: dict[str, Any], group_by: str, suite: str) -> str:
    group_by = str(group_by or "none").strip().lower()
    if group_by == "official_category":
        return str(
            item.get("plus_official_category_slug")
            or item.get("plus_official_category")
            or "unknown"
        )
    if group_by == "perturbation":
        return str(item.get("plus_perturbation") or "unknown")
    if group_by == "suite_category":
        category = str(
            item.get("plus_official_category_slug")
            or item.get("plus_official_category")
            or "unknown"
        )
        return f"{suite}:{category}"
    raise ValueError(
        f"Unsupported LIBERO-Plus subset group_by={group_by!r}; expected "
        "none, official_category, perturbation, or suite_category."
    )


def select_libero_plus_task_subset(
    task_metadata: list[dict[str, Any]],
    *,
    group_by: str = "none",
    samples_per_group: int = 0,
    sample_seed: int = 0,
    suite: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically sample LIBERO-Plus tasks from annotated metadata."""
    group_by = str(group_by or "none").strip().lower()
    samples_per_group = int(samples_per_group or 0)
    if group_by in {"", "none"} or samples_per_group <= 0:
        return list(task_metadata), {
            "enabled": False,
            "group_by": group_by or "none",
            "samples_per_group": samples_per_group,
            "sample_seed": int(sample_seed),
            "suite": suite,
            "total_available": len(task_metadata),
            "total_selected": len(task_metadata),
            "groups": [],
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in task_metadata:
        entry = dict(item)
        entry.setdefault("suite", suite)
        key = _plus_subset_group_key(entry, group_by, suite)
        groups.setdefault(key, []).append(entry)

    selected: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []
    for group, candidates in sorted(groups.items()):
        ordered = sorted(candidates, key=_plus_subset_sort_key)
        seed_material = f"{int(sample_seed)}:{suite}:{group}"
        seed_int = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        shuffled = list(ordered)
        random.Random(seed_int).shuffle(shuffled)
        chosen = sorted(shuffled[:samples_per_group], key=_plus_subset_sort_key)
        selected.extend(chosen)
        manifest_groups.append(
            {
                "group": group,
                "seed": seed_int,
                "available": len(ordered),
                "selected": len(chosen),
                "task_ids": [int(x.get("task_id", -1)) for x in chosen],
                "official_task_ids": [str(x.get("plus_official_task_id", "")) for x in chosen],
                "bddl_files": [str(x.get("bddl_file", "")) for x in chosen],
            }
        )

    selected = sorted(selected, key=_plus_subset_sort_key)
    return selected, {
        "enabled": True,
        "group_by": group_by,
        "samples_per_group": samples_per_group,
        "sample_seed": int(sample_seed),
        "suite": suite,
        "total_available": len(task_metadata),
        "total_selected": len(selected),
        "groups": manifest_groups,
    }
