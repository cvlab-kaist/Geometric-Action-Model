"""Shared helpers for the experiment evaluation registry.

The registry intentionally keeps historical flat fields for compatibility, but
all new records also get normalized v2 sections for reliable comparison.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_now_minute() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _run_git(args: list[str], repo_root: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root) if repo_root else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def current_git_state(repo_root: Path | None = None) -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    branch = _run_git(["branch", "--show-current"], repo_root)
    dirty = None
    try:
        subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=str(repo_root) if repo_root else None,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        dirty = False
    except subprocess.CalledProcessError:
        dirty = True
    except Exception:
        dirty = None
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _safe_int(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ckpt_source_run(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "checkpoints" not in parts:
        return None
    idxs = [i for i, part in enumerate(parts) if part == "checkpoints"]
    for idx in idxs:
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if not candidate.endswith(".pt"):
                return candidate
    for part in reversed(parts):
        if part.startswith("robot-"):
            return part
    return None


def _ckpt_exists(path: str | None) -> bool | None:
    if not path or "{" in path or "}" in path:
        return None
    try:
        return Path(path).exists()
    except OSError:
        return None


def _metric(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _infer_benchmark(entry: dict[str, Any]) -> tuple[str, str | None]:
    eval_data = str(entry.get("eval_data") or "").lower()
    suites = entry.get("suites") or []
    plus = bool(entry.get("plus")) or "libero_plus" in eval_data
    if plus:
        return "libero", "plus"
    if "libero" in eval_data or any(str(suite).startswith("libero") for suite in suites):
        return "libero", "original"
    if "robocasa" in eval_data or str(entry.get("benchmark") or "").lower() == "robocasa":
        return "robocasa", None
    if "mimicgen" in eval_data:
        return "mimicgen", None
    if "openx" in eval_data or "oxe" in eval_data:
        return "openx", None
    return "unknown", None


def _infer_record_type(entry: dict[str, Any]) -> str:
    schema = entry.get("schema") if isinstance(entry.get("schema"), dict) else {}
    classification = entry.get("classification") if isinstance(entry.get("classification"), dict) else {}
    if schema.get("record_type"):
        return str(schema["record_type"])
    if classification.get("record_type"):
        return str(classification["record_type"])
    eval_data = str(entry.get("eval_data") or "").lower()
    ckpt_step = str(entry.get("ckpt_step") or "")
    ckpt_path = str(entry.get("ckpt_path") or "")
    if "sweep" in ckpt_step or "{" in ckpt_path:
        return "checkpoint_sweep"
    if "closed_loop" in eval_data or entry.get("suites"):
        return "closed_loop_rollout"
    metrics = entry.get("metrics") or {}
    if any(key.startswith("ae_") or key.startswith("noact_") for key in metrics):
        return "open_loop_action"
    return "unknown"


def _infer_tier(entry: dict[str, Any], record_type: str) -> str:
    schema = entry.get("schema") if isinstance(entry.get("schema"), dict) else {}
    classification = entry.get("classification") if isinstance(entry.get("classification"), dict) else {}
    explicit = entry.get("registry_tier") or entry.get("tier") or schema.get("tier") or classification.get("tier")
    if explicit:
        return str(explicit)
    if record_type == "open_loop_action":
        return "diagnostic"
    if record_type == "checkpoint_sweep":
        return "sweep"

    protocol = str(entry.get("protocol") or "").lower()
    note = str(entry.get("note") or entry.get("notes") or "").lower()
    n_episodes = _safe_int(entry.get("n_episodes"))
    metrics = entry.get("metrics") or {}
    plus = bool(entry.get("plus"))
    task_ids = entry.get("task_ids") or entry.get("task_filter")

    if "debug" in protocol or "debug" in note:
        return "debug"
    if "smoke" in protocol or "smoke" in note:
        return "smoke"
    if task_ids:
        return "subset"
    if plus and metrics.get("plus_category_camera_num_trials") is not None:
        return "canonical"
    if isinstance(n_episodes, int) and n_episodes < 100:
        return "smoke"
    if _safe_float(_metric(metrics, "overall_success_rate", "average_success_rate")) is not None:
        return "candidate"
    return "legacy"


def _infer_entrypoint(record_type: str) -> str | None:
    if record_type == "open_loop_action":
        return "src/eval_stage1.py"
    if record_type in {"closed_loop_rollout", "checkpoint_sweep"}:
        return "src/eval_libero_unified.py"
    return None


def _suite_scores(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = entry.get("metrics") or {}
    suite_results = entry.get("suite_results") or {}
    suites = entry.get("suites") or []
    scores: dict[str, dict[str, Any]] = {}
    for suite, result in suite_results.items():
        scores[str(suite)] = {
            "success_rate": _safe_float(result.get("success_rate")),
            "successes": _safe_int(result.get("num_success")),
            "trials": _safe_int(result.get("num_trials")),
        }
    for suite in suites:
        key = f"{suite}_success_rate"
        if key in metrics and str(suite) not in scores:
            scores[str(suite)] = {"success_rate": _safe_float(metrics.get(key))}
    return scores


def _plus_category_scores(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = entry.get("metrics") or {}
    category_scores: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^plus_category_(.+)_success_rate$")
    for key, value in metrics.items():
        match = pattern.match(key)
        if not match:
            continue
        slug = match.group(1)
        category_scores[slug] = {
            "success_rate": _safe_float(value),
            "trials": _safe_int(metrics.get(f"plus_category_{slug}_num_trials")),
        }
    return category_scores


def _primary_score(entry: dict[str, Any], record_type: str) -> dict[str, Any] | None:
    metrics = entry.get("metrics") or {}
    if record_type in {"closed_loop_rollout", "checkpoint_sweep"}:
        value = _safe_float(_metric(metrics, "overall_success_rate", "average_success_rate", "best_success_rate"))
        if value is None:
            return None
        return {"name": "overall_success_rate", "value": value, "higher_is_better": True}
    value = _safe_float(_metric(metrics, "noact_l1", "ae_l1"))
    if value is None:
        return None
    name = "noact_l1" if metrics.get("noact_l1") is not None else "ae_l1"
    return {"name": name, "value": value, "higher_is_better": False}


def normalize_eval_entry(
    entry: dict[str, Any],
    *,
    code_state: dict[str, Any] | None = None,
    entrypoint: str | None = None,
    record_type: str | None = None,
    tier: str | None = None,
    fill_runtime_env: bool = False,
) -> dict[str, Any]:
    """Return a v2-compatible entry while preserving historical flat fields."""

    original = copy.deepcopy(entry)
    normalized = copy.deepcopy(entry)
    existing_schema = normalized.get("schema") if isinstance(normalized.get("schema"), dict) else {}
    existing_code = normalized.get("code") if isinstance(normalized.get("code"), dict) else {}
    already_v2 = normalized.get("schema_version") == SCHEMA_VERSION or existing_schema.get("version") == SCHEMA_VERSION

    if not normalized.get("id"):
        normalized["id"] = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not normalized.get("date"):
        normalized["date"] = local_now_minute()

    record_type = record_type or _infer_record_type(normalized)
    tier = tier or _infer_tier(normalized, record_type)
    benchmark_name, benchmark_variant = _infer_benchmark(normalized)
    metrics = normalized.get("metrics") if isinstance(normalized.get("metrics"), dict) else {}
    policy = normalized.get("policy") if isinstance(normalized.get("policy"), dict) else {}
    artifacts = normalized.get("artifacts") if isinstance(normalized.get("artifacts"), dict) else {}
    existing_execution = normalized.get("execution") if isinstance(normalized.get("execution"), dict) else {}
    checkpoint_path = normalized.get("ckpt_path") or normalized.get("checkpoint")
    checkpoint_step = normalized.get("ckpt_step") or normalized.get("checkpoint_step")
    code_state = code_state or {}
    slurm_job_id = (
        existing_execution.get("slurm_job_id")
        if "slurm_job_id" in existing_execution
        else artifacts.get("slurm_job_id")
        or normalized.get("slurm_job_id")
        or (os.environ.get("SLURM_JOB_ID") if fill_runtime_env else None)
    )

    normalized["schema_version"] = SCHEMA_VERSION
    normalized["schema"] = {
        "version": SCHEMA_VERSION,
        "record_type": record_type,
        "tier": tier,
    }
    if not already_v2 or not normalized.get("updated_at_utc"):
        normalized["updated_at_utc"] = utc_now_iso()
    normalized["classification"] = {
        "tier": tier,
        "record_type": record_type,
        "benchmark": benchmark_name,
        "benchmark_variant": benchmark_variant,
        "is_canonical": tier == "canonical",
        "is_smoke": tier == "smoke",
        "is_debug": tier == "debug",
        "is_libero_plus": benchmark_name == "libero" and benchmark_variant == "plus",
    }
    normalized["benchmark"] = {
        "name": benchmark_name,
        "variant": benchmark_variant,
        "suites": normalized.get("suites") or [],
        "task_sets": normalized.get("task_sets") or [],
        "tasks": normalized.get("tasks") or [],
        "split": normalized.get("split"),
        "plus": bool(normalized.get("plus", False)),
        "plus_perturbation": normalized.get("plus_perturbation"),
        "plus_official_category": normalized.get("plus_official_category"),
        "eval_data": normalized.get("eval_data"),
    }
    normalized["checkpoint"] = {
        "path": checkpoint_path,
        "step": _safe_int(checkpoint_step),
        "source_run": _ckpt_source_run(str(checkpoint_path)) if checkpoint_path else None,
        "exists_at_record_time": _ckpt_exists(str(checkpoint_path)) if checkpoint_path else None,
    }
    normalized["code"] = {
        "git_commit": normalized.get("git_commit") or existing_code.get("git_commit") or code_state.get("commit"),
        "git_branch": normalized.get("git_branch") or existing_code.get("git_branch") or code_state.get("branch"),
        "git_dirty": normalized.get("git_dirty", existing_code.get("git_dirty", code_state.get("dirty"))),
        "entrypoint": entrypoint or normalized.get("entrypoint") or existing_code.get("entrypoint") or _infer_entrypoint(record_type),
    }
    normalized["execution"] = {
        "status": normalized.get("status", "completed"),
        "config": normalized.get("config"),
        "protocol": normalized.get("protocol"),
        "n_episodes": _safe_int(normalized.get("n_episodes")),
        "policy_stage": policy.get("stage"),
        "slurm_job_id": slurm_job_id,
        "shard": normalized.get("shard"),
    }
    normalized["scores"] = {
        "primary": _primary_score(normalized, record_type),
        "success_rate": _safe_float(_metric(metrics, "overall_success_rate", "average_success_rate", "best_success_rate")),
        "successes": _safe_int(_metric(metrics, "total_successes", "best_successes")),
        "trials": _safe_int(_metric(metrics, "total_trials")) or _safe_int(normalized.get("n_episodes")),
        "per_suite": _suite_scores(normalized),
        "per_task": metrics.get("per_task"),
        "plus_categories": _plus_category_scores(normalized),
        "raw_metrics": metrics,
    }

    if not already_v2:
        normalized["legacy"] = original
    else:
        normalized.setdefault("legacy", original.get("legacy"))
    return normalized


def registry_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {
        "by_tier": {},
        "by_record_type": {},
        "by_benchmark": {},
    }
    for entry in evaluations:
        classification = entry.get("classification") or {}
        for key, source in (
            ("by_tier", classification.get("tier")),
            ("by_record_type", classification.get("record_type")),
            ("by_benchmark", classification.get("benchmark_variant") or classification.get("benchmark")),
        ):
            value = str(source or "unknown")
            counts[key][value] = counts[key].get(value, 0) + 1
    return {"total": len(evaluations), **counts}


def normalize_registry(registry: dict[str, Any], *, code_state: dict[str, Any] | None = None) -> dict[str, Any]:
    evaluations = [
        normalize_eval_entry(entry, code_state=code_state)
        for entry in registry.get("evaluations", [])
    ]
    normalized = copy.deepcopy(registry)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["schema"] = {
        "name": "3da_eval_registry",
        "version": SCHEMA_VERSION,
        "compatible_top_level": "evaluations",
    }
    normalized["updated_at_utc"] = utc_now_iso()
    normalized["summary"] = registry_summary(evaluations)
    normalized["evaluations"] = evaluations
    return normalized


def append_eval_record(
    registry_path: str | Path,
    entry: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    entrypoint: str | None = None,
    record_type: str | None = None,
    tier: str | None = None,
) -> str:
    path = Path(registry_path)
    if path.exists():
        with path.open("r") as f:
            registry = json.load(f)
    else:
        registry = {"evaluations": []}

    repo = Path(repo_root) if repo_root is not None else None
    code_state = current_git_state(repo)
    record = normalize_eval_entry(
        entry,
        code_state=code_state,
        entrypoint=entrypoint,
        record_type=record_type,
        tier=tier,
        fill_runtime_env=True,
    )
    registry["evaluations"] = [
        normalize_eval_entry(existing)
        for existing in registry.get("evaluations", [])
    ] + [record]
    registry = normalize_registry(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return str(record["id"])
