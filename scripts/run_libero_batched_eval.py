#!/usr/bin/env python3
"""Harness-style batched LIBERO eval.

This runner mirrors the two key ideas from allenai/vla-evaluation-harness:
flat episode sharding for environment parallelism and a shared batched policy
dispatcher for GPU inference. It intentionally avoids WandB and writes local
summary/per-task/progress artifacts only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Keep these imports lazy. Isolated env workers are started with
# multiprocessing spawn, which imports this script as the child main module.
# Importing torch/eval_libero_unified there can fan out TorchInductor compile
# workers before the env worker has even initialized.
torch: Any | None = None
OmegaConf: Any | None = None
elu: Any | None = None
create_rollout_env_libero: Any | None = None
create_rollout_env_libero_isolated: Any | None = None
list_libero_task_metadata: Any | None = None


def _load_heavy_modules() -> None:
    global OmegaConf
    global create_rollout_env_libero
    global create_rollout_env_libero_isolated
    global elu
    global list_libero_task_metadata
    global torch

    if elu is not None:
        return

    import torch as _torch
    from omegaconf import OmegaConf as _OmegaConf

    import eval_libero_unified as _elu
    from robot.evaluation.rollout_env import (
        create_rollout_env_libero as _create_rollout_env_libero,
        create_rollout_env_libero_isolated as _create_rollout_env_libero_isolated,
        list_libero_task_metadata as _list_libero_task_metadata,
    )

    torch = _torch
    OmegaConf = _OmegaConf
    elu = _elu
    create_rollout_env_libero = _create_rollout_env_libero
    create_rollout_env_libero_isolated = _create_rollout_env_libero_isolated
    list_libero_task_metadata = _list_libero_task_metadata


@dataclass
class _PolicyRequest:
    session_id: str
    obs: dict[str, Any]
    task_desc: str
    active_action_horizon: int
    event: threading.Event
    result: torch.Tensor | None = None
    error: BaseException | None = None


class BatchedPolicyDispatcher:
    """Queue observations and dispatch batched policy forwards."""

    def __init__(self, policy: Any, *, max_batch_size: int, max_wait_time: float) -> None:
        self.policy = policy
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_time = max(0.0, float(max_wait_time))
        self._queue: queue.Queue[_PolicyRequest | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="libero-batched-policy", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30.0)

    def predict(
        self,
        *,
        session_id: str,
        obs: dict[str, Any],
        task_desc: str,
        active_action_horizon: int,
    ) -> torch.Tensor:
        req = _PolicyRequest(
            session_id=str(session_id),
            obs=obs,
            task_desc=str(task_desc or ""),
            active_action_horizon=int(active_action_horizon),
            event=threading.Event(),
        )
        self._queue.put(req)
        req.event.wait()
        if req.error is not None:
            raise req.error
        if req.result is None:
            raise RuntimeError("Batched policy dispatcher returned no result.")
        return req.result

    def reset_session(self, session_id: str) -> None:
        reset = getattr(self.policy, "reset_session", None)
        if callable(reset):
            reset(str(session_id))

    def commit_session_observation(self, session_id: str, executed_policy_actions: int) -> None:
        commit = getattr(self.policy, "commit_session_observation", None)
        if callable(commit):
            commit(str(session_id), int(executed_policy_actions))

    def override_session_pending_action_chunk(self, session_id: str, raw_action_chunk: torch.Tensor) -> None:
        override = getattr(self.policy, "override_session_pending_action_chunk", None)
        if callable(override):
            override(str(session_id), raw_action_chunk)

    def get_session_debug(self, session_id: str) -> dict[str, Any]:
        getter = getattr(self.policy, "get_session_debug", None)
        if callable(getter):
            return getter(str(session_id))
        return {}

    def _loop(self) -> None:
        predict_batch = getattr(self.policy, "predict_batch", None)
        if not callable(predict_batch):
            raise RuntimeError("Policy lacks predict_batch().")
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.max_wait_time
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self._queue.put(None)
                    break
                batch.append(item)
            payload = [
                {
                    "session_id": req.session_id,
                    "obs": req.obs,
                    "task_desc": req.task_desc,
                    "active_action_horizon": req.active_action_horizon,
                }
                for req in batch
            ]
            try:
                results = predict_batch(payload)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"predict_batch returned {len(results)} results for {len(batch)} requests."
                    )
                for req, result in zip(batch, results):
                    req.result = result
            except BaseException as exc:  # noqa: BLE001
                for req in batch:
                    req.error = exc
            finally:
                for req in batch:
                    req.event.set()


class SessionPolicyProxy:
    """Per-worker policy facade used by the existing rollout loop."""

    accepts_task_desc = True

    def __init__(
        self,
        dispatcher: BatchedPolicyDispatcher,
        *,
        session_id: str,
        active_action_horizon: int,
    ) -> None:
        self.dispatcher = dispatcher
        self.session_id = str(session_id)
        self.active_action_horizon = int(active_action_horizon)
        self.last_debug: dict[str, Any] = {}

    def __call__(self, obs: dict[str, Any], task_desc: str = "") -> torch.Tensor:
        out = self.dispatcher.predict(
            session_id=self.session_id,
            obs=obs,
            task_desc=task_desc,
            active_action_horizon=self.active_action_horizon,
        )
        self.last_debug = self.dispatcher.get_session_debug(self.session_id)
        return out

    def reset_episode(self) -> None:
        self.dispatcher.reset_session(self.session_id)
        self.last_debug = {}

    def commit_observation(self, executed_policy_actions: int) -> None:
        self.dispatcher.commit_session_observation(self.session_id, int(executed_policy_actions))

    def override_pending_action_chunk(self, raw_action_chunk: torch.Tensor) -> None:
        self.dispatcher.override_session_pending_action_chunk(self.session_id, raw_action_chunk)


class ExternalVLASessionPolicyProxy:
    """Per-worker facade for a vla-evaluation-harness WebSocket model server."""

    accepts_task_desc = True

    def __init__(
        self,
        *,
        server_url: str,
        timeout_sec: float,
        session_id: str,
        benchmark_name: str,
    ) -> None:
        self.server_url = str(server_url)
        self.timeout_sec = float(timeout_sec)
        self.session_id = str(session_id)
        self.benchmark_name = str(benchmark_name or "libero_batched")
        self.last_debug: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._conn: Any | None = None
        self._episode_started = False
        self._episode_context: dict[str, Any] = {}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._run(self._conn.close())
            finally:
                self._conn = None
        self._loop.close()

    def set_episode_context(self, context: dict[str, Any]) -> None:
        self._episode_context = _serializable_task_context(context)

    def start_episode(self) -> None:
        self._ensure_connected()
        if self._episode_started:
            return
        assert self._conn is not None
        self._run(self._conn.start_episode({"task": self._episode_context}))
        self._episode_started = True

    def end_episode(self, result: dict[str, Any]) -> None:
        if self._conn is None or not self._episode_started:
            return
        try:
            self._run(self._conn.end_episode(_serializable_task_context(result)))
        finally:
            self._episode_started = False

    def reset_episode(self) -> None:
        self.last_debug = {}
        if not self._episode_started:
            self.start_episode()

    def __call__(self, obs: dict[str, Any], task_desc: str = "") -> torch.Tensor:
        self._ensure_connected()
        if not self._episode_started:
            self.start_episode()
        assert self._conn is not None
        payload = dict(obs)
        if task_desc:
            payload.setdefault("task_description", task_desc)
            payload.setdefault("language", task_desc)
        action = self._run(self._conn.act(payload))
        raw = action.get("actions", action.get("action")) if isinstance(action, dict) else action
        if raw is None:
            raise RuntimeError(f"External VLA server returned no action: {action!r}")
        out = torch.as_tensor(raw, dtype=torch.float32)
        if out.ndim == 1:
            out = out.view(1, -1)
        if out.ndim != 2 or int(out.shape[-1]) != 7:
            raise RuntimeError(f"External VLA server returned invalid action shape: {tuple(out.shape)}")
        self.last_debug = {
            "external_vla_server_url": self.server_url,
            "external_vla_session_id": self.session_id,
            "external_vla_action_shape": list(out.shape),
        }
        return out

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    def _ensure_connected(self) -> None:
        if self._conn is not None:
            return
        from vla_eval.connection import Connection

        self._conn = Connection(self.server_url, timeout=self.timeout_sec)
        self._run(self._conn.connect(benchmark=self.benchmark_name))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _serializable_task_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _serializable_task_context(v)
            for k, v in value.items()
            if isinstance(v, (str, int, float, bool, list, tuple, dict)) or v is None
        }
    if isinstance(value, (list, tuple)):
        return [_serializable_task_context(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _external_vla_policy_info(args: Any) -> dict[str, Any]:
    return {
        "stage": "baseline",
        "baseline": "external_vla_server",
        "config_source": "external_vla_server",
        "model_server_url": str(args.external_vla_server_url),
        "policy_image_preprocess": "external_vla_server",
        "text_encoder_type": "external_vla_server",
        "decode_visuals": False,
        "use_bf16_autocast": False,
        "action_frame": "base",
        "rollout_action_frame": "base",
        "proprio_orientation": "rpy",
        "action_chunk_size": 1,
        "max_action_horizon": 1,
        "max_low_level_actions_per_call": 1,
        "libero_hdf5_env_hflip": False,
        "libero_hdf5_env_vflip": False,
        "libero_hdf5_env_rotate180": False,
        "benchmark_name": str(args.external_vla_benchmark_name),
    }


def _build_work_items(args: Any, suites: list[str], num_trials: int, plus_root: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert elu is not None
    assert list_libero_task_metadata is not None
    task_ids = elu.parse_int_list(args.task_ids)
    manifest_path = str(getattr(args, "task_manifest_path", "") or "").strip()
    if manifest_path:
        with Path(manifest_path).open() as f:
            manifest = json.load(f)
        manifest_tasks = [dict(item) for item in manifest.get("tasks", [])]
        requested_suites = set(suites)
        selected = [
            item
            for item in manifest_tasks
            if not requested_suites or str(item.get("suite", "")) in requested_suites
        ]
        if task_ids:
            task_id_set = {int(x) for x in task_ids}
            selected = [item for item in selected if int(item.get("task_id", -1)) in task_id_set]
        work: list[dict[str, Any]] = []
        global_index = 0
        preset = elu.PRESETS[args.preset]
        for entry in selected:
            suite = str(entry["suite"])
            if suite not in elu.OPENVLA_STEPS:
                raise ValueError(f"Unknown LIBERO suite in task manifest: {suite!r}.")
            max_steps = int(args.max_steps or entry.get("max_steps") or preset.max_steps_by_suite[suite])
            task_id = int(entry["task_id"])
            eval_task_index = int(entry.get("eval_task_index", task_id))
            normalized_entry = dict(entry)
            normalized_entry.setdefault("language", entry.get("task_description", entry.get("name", "")))
            normalized_entry.setdefault("policy_language", normalized_entry.get("language", ""))
            normalized_entry.setdefault("eval_task_index", eval_task_index)
            normalized_entry.setdefault("task_id", task_id)
            for episode_idx in range(num_trials):
                work.append(
                    {
                        "global_index": global_index,
                        "suite": suite,
                        "task_id": task_id,
                        "eval_task_index": eval_task_index,
                        "episode_idx": int(episode_idx),
                        "max_steps": int(max_steps),
                        "entry": normalized_entry,
                    }
                )
                global_index += 1
        return work, {
            "plus_classification": manifest.get("plus_classification")
            or {"loaded": False, "path": None, "suite_task_counts": {}, "category_task_counts": {}},
            "plus_subset_manifests": {},
            "task_manifest_path": str(Path(manifest_path).resolve()),
            "task_manifest_total_tasks": len(manifest_tasks),
            "task_manifest_selected_tasks": len(selected),
        }
    plus_classification = (
        elu.load_libero_plus_task_classification(plus_root)
        if args.plus
        else {"loaded": False, "path": None, "suite_task_counts": {}, "category_task_counts": {}}
    )
    plus_subset_manifests: dict[str, Any] = {}
    work: list[dict[str, Any]] = []
    global_index = 0
    preset = elu.PRESETS[args.preset]
    for suite in suites:
        if suite not in elu.OPENVLA_STEPS:
            raise ValueError(f"Unknown LIBERO suite {suite!r}.")
        max_steps = int(args.max_steps or preset.max_steps_by_suite[suite])
        metadata = [dict(item) for item in list_libero_task_metadata(suite, plus_root=plus_root)]
        if args.plus:
            for item in metadata:
                item["plus_perturbation"] = elu.classify_libero_plus_perturbation(item)
            elu.annotate_libero_plus_official_categories(metadata, suite, plus_classification)
            metadata = elu.filter_libero_plus_task_metadata(metadata, args.plus_perturbation)
            metadata = elu.filter_libero_plus_official_category_metadata(
                metadata,
                args.plus_official_category,
            )
            metadata, subset_manifest = elu.select_libero_plus_task_subset(
                metadata,
                group_by=args.plus_sample_group_by,
                samples_per_group=int(args.plus_samples_per_group),
                sample_seed=int(args.plus_sample_seed),
                suite=suite,
            )
            if subset_manifest.get("enabled"):
                plus_subset_manifests[suite] = subset_manifest
        for entry in elu.select_eval_task_entries(metadata, task_ids):
            task_id = int(entry["task_id"])
            eval_task_index = int(entry["eval_task_index"])
            for episode_idx in range(num_trials):
                work.append(
                    {
                        "global_index": global_index,
                        "suite": suite,
                        "task_id": task_id,
                        "eval_task_index": eval_task_index,
                        "episode_idx": int(episode_idx),
                        "max_steps": int(max_steps),
                        "entry": dict(entry),
                    }
                )
                global_index += 1
    return work, {
        "plus_classification": plus_classification,
        "plus_subset_manifests": plus_subset_manifests,
    }


def _make_env(
    *,
    args: Any,
    worker_idx: int,
    item: dict[str, Any],
    policy_info: dict[str, Any],
    render_gpu_device_id: int | None,
    plus_root: str | None,
    env_horizon: int,
) -> tuple[Any, str, np.ndarray]:
    assert elu is not None
    assert create_rollout_env_libero is not None
    assert create_rollout_env_libero_isolated is not None
    creator = create_rollout_env_libero_isolated if args.env_process_isolation else create_rollout_env_libero
    kwargs = {
        "suite_name": item["suite"],
        "task_id": int(item["task_id"]),
        "camera_names": elu.LIBERO_CAMERA_NAMES,
        "camera_size": int(args.camera_size),
        "render_gpu_device_id": render_gpu_device_id,
        "control_freq": float(args.env_control_hz) if args.env_control_hz is not None else None,
        "horizon": int(env_horizon),
        "plus_root": plus_root,
        "preserve_libero_plus_robot_init_qpos": args.libero_plus_robot_init_qpos_mode == "preserve",
        "camera_depths": False,
        "env_image_hflip": bool(
            policy_info.get(
                "libero_hdf5_env_hflip",
                policy_info.get("libero_hdf5_env_rotate180", False),
            )
        ),
        "env_image_rotate180": bool(policy_info.get("libero_hdf5_env_rotate180", False)),
    }
    if "libero_hdf5_env_vflip" in policy_info:
        kwargs["env_image_vflip"] = bool(policy_info["libero_hdf5_env_vflip"])
    if args.env_process_isolation:
        kwargs["worker_timeout_sec"] = float(args.env_worker_timeout_sec)
        kwargs["worker_rank"] = int(worker_idx)
    return creator(**kwargs)


def _worker_run(
    *,
    worker_idx: int,
    items: list[dict[str, Any]],
    args: Any,
    dispatcher: BatchedPolicyDispatcher | None,
    action_horizon: int,
    action_frame: str,
    policy_info: dict[str, Any],
    render_gpu_device_id: int | None,
    plus_root: str | None,
    env_seed: int,
    num_steps_wait: int,
    progress_lock: threading.Lock,
    progress_rows: list[dict[str, Any]],
    progress_log_path: Path,
    suites: list[str],
    run_name: str,
    env_cache_size: int,
) -> list[dict[str, Any]]:
    assert elu is not None
    external_server_url = str(getattr(args, "external_vla_server_url", "") or "").strip()
    if external_server_url:
        proxy: SessionPolicyProxy | ExternalVLASessionPolicyProxy = ExternalVLASessionPolicyProxy(
            server_url=external_server_url,
            timeout_sec=float(getattr(args, "external_vla_server_timeout_sec", 900.0)),
            session_id=f"{run_name}:worker{worker_idx}",
            benchmark_name=str(getattr(args, "external_vla_benchmark_name", "libero_batched")),
        )
    else:
        if dispatcher is None:
            raise RuntimeError("Internal batched policy mode requires a dispatcher.")
        proxy = SessionPolicyProxy(
            dispatcher,
            session_id=f"worker{worker_idx}",
            active_action_horizon=int(action_horizon),
        )
    rows: list[dict[str, Any]] = []
    env_cache: OrderedDict[tuple[str, int, int], tuple[Any, str, np.ndarray, int]] = OrderedDict()
    try:
        for local_idx, item in enumerate(items):
            suite = str(item["suite"])
            task_id = int(item["task_id"])
            eval_task_index = int(item["eval_task_index"])
            episode_idx = int(item["episode_idx"])
            max_steps = int(item["max_steps"])
            env_horizon = int(max_steps + max(0, num_steps_wait))
            env_key = (suite, task_id, env_horizon)
            entry = dict(item["entry"])
            task_desc = str(entry.get("policy_language") or entry.get("language") or "")
            task_name = task_desc
            plus_perturbation = str(entry.get("plus_perturbation", ""))
            plus_official_task_id = entry.get("plus_official_task_id", "")
            plus_official_category = str(entry.get("plus_official_category", ""))
            plus_official_category_slug = str(entry.get("plus_official_category_slug", ""))
            plus_official_difficulty_level = entry.get("plus_official_difficulty_level", "")
            t0 = time.time()
            env = None
            close_after_episode = False
            result: dict[str, Any] = {}
            success = False
            steps = int(max_steps)
            error = ""
            try:
                cached = None if env_cache_size == 0 else env_cache.pop(env_key, None)
                if cached is None:
                    env, task_name, init_states = _make_env(
                        args=args,
                        worker_idx=worker_idx,
                        item=item,
                        policy_info=policy_info,
                        render_gpu_device_id=render_gpu_device_id,
                        plus_root=plus_root,
                        env_horizon=env_horizon,
                    )
                    action_repeat, _policy_hz, _env_hz = elu.resolve_action_repeat(
                        args.action_repeat,
                        policy_info=policy_info,
                        env=env,
                        policy_hz_override=args.policy_hz,
                    )
                    cached = (env, str(task_name), np.asarray(init_states), int(action_repeat))
                    close_after_episode = env_cache_size == 0
                if env_cache_size != 0:
                    env_cache[env_key] = cached
                    while env_cache_size > 0 and len(env_cache) > env_cache_size:
                        _old_key, old_cached = env_cache.popitem(last=False)
                        try:
                            old_cached[0].close()
                        except Exception:
                            pass
                env, task_name, init_states, action_repeat = cached
                init_idx = episode_idx % max(1, len(init_states))
                elu.seed_env(env, int(env_seed) + episode_idx)
                if isinstance(proxy, ExternalVLASessionPolicyProxy):
                    proxy.set_episode_context(
                        {
                            "suite": suite,
                            "task_id": task_id,
                            "eval_task_index": eval_task_index,
                            "episode_idx": episode_idx,
                            "task_description": task_desc or str(task_name),
                            "name": task_name,
                            "entry": entry,
                            "worker_idx": worker_idx,
                            "run_name": run_name,
                        }
                    )
                    proxy.start_episode()
                result = elu.rollout_episode(
                    env=env,
                    init_state=init_states[init_idx],
                    policy=proxy,
                    max_steps=max_steps,
                    action_horizon=int(action_horizon),
                    action_repeat=int(action_repeat),
                    action_repeat_mode=args.action_repeat_mode,
                    num_steps_wait=int(num_steps_wait),
                    camera_size=int(args.camera_size),
                    record_video=False,
                    detailed_video=False,
                    binarize_gripper=not args.no_binarize_gripper,
                    task_desc=task_desc or str(task_name),
                    action_frame=action_frame,
                    proprio_orientation=policy_info.get("proprio_orientation", args.proprio_orientation),
                    temporal_ensemble=False,
                    execution_strategy="default",
                    execute_chunk_prefix=int(args.execute_chunk_prefix),
                    partial_chunk_history=args.partial_chunk_history,
                    warmup_full_chunk_once=bool(args.warmup_full_chunk_once),
                    rollout_wall_timeout_sec=float(args.rollout_wall_timeout_sec),
                    wait_action_mode="open_gripper",
                )
                success = bool(result.get("success", False))
                steps = int(result.get("steps", max_steps))
                error = ""
            except Exception as exc:  # noqa: BLE001
                success = False
                steps = int(max_steps)
                error = f"{type(exc).__name__}: {exc}"
                if env_cache_size != 0:
                    cached = env_cache.pop(env_key, None)
                    env_to_close = cached[0] if cached is not None else env
                else:
                    env_to_close = env
                if env_to_close is not None:
                    try:
                        env_to_close.close()
                    except Exception:
                        pass
                    close_after_episode = False
            finally:
                if isinstance(proxy, ExternalVLASessionPolicyProxy):
                    proxy.end_episode(
                        {
                            "metrics": {"success": bool(success)},
                            "success": bool(success),
                            "steps": int(steps),
                            "error": str(error),
                            "timeout": bool(result.get("timeout", False)) if isinstance(result, dict) else False,
                        }
                    )
                if close_after_episode and env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            elapsed = time.time() - t0
            row = elu.make_episode_progress_row(
                suite=suite,
                task_id=task_id,
                eval_task_index=eval_task_index,
                task_name=str(task_name),
                task_desc=task_desc,
                entry=entry,
                plus_perturbation=plus_perturbation,
                plus_official_task_id=plus_official_task_id,
                plus_official_category=plus_official_category,
                plus_official_category_slug=plus_official_category_slug,
                plus_official_difficulty_level=plus_official_difficulty_level,
                success=success,
                steps=steps,
            )
            row.update(
                {
                    "episode_idx": episode_idx,
                    "worker_idx": int(worker_idx),
                    "elapsed_sec": float(elapsed),
                    "error": error,
                }
            )
            rows.append(row)
            with progress_lock:
                progress_rows.append(row)
                progress_log_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_log_path.open("a") as f:
                    f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
                suite_results = elu.suite_results_from_rows(progress_rows, suites)
                total_trials = sum(int(x.get("num_trials", 0) or 0) for x in suite_results.values())
                total_success = sum(int(x.get("num_success", 0) or 0) for x in suite_results.values())
                print(
                    f"[worker {worker_idx}] {local_idx + 1}/{len(items)} "
                    f"{suite}:task{task_id}:ep{episode_idx} "
                    f"{'SUCCESS' if success else 'FAIL'} steps={steps} elapsed={elapsed:.1f}s "
                    f"overall={total_success}/{total_trials}",
                    flush=True,
                )
    finally:
        for env, _task_name, _init_states, _repeat in env_cache.values():
            try:
                env.close()
            except Exception:
                pass
        if isinstance(proxy, ExternalVLASessionPolicyProxy):
            proxy.close()
    return rows


def _per_task_rows_from_progress(progress_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert elu is not None
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in progress_rows:
        key = (str(row["suite"]), int(row["eval_task_index"]), int(row["task_id"]))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        first = dict(rows[0])
        trials = len(rows)
        successes = sum(int(bool(r.get("num_success", 0))) for r in rows)
        steps = [float(r.get("avg_steps", 0.0) or 0.0) for r in rows]
        out.append(
            {
                field: first.get(field, "")
                for field in elu.PER_TASK_FIELDNAMES
            }
        )
        out[-1].update(
            {
                "num_trials": int(trials),
                "num_success": int(successes),
                "success_rate": float(successes / trials) if trials else 0.0,
                "avg_steps": float(np.mean(steps)) if steps else 0.0,
            }
        )
    return out


def main() -> None:
    _load_heavy_modules()
    assert torch is not None
    assert OmegaConf is not None
    assert elu is not None

    parser = elu.build_arg_parser()
    parser.description = "Harness-style batched LIBERO eval"
    parser.add_argument("--parallel-envs", type=int, default=8, help="Number of concurrent episode workers.")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum observations per policy forward.")
    parser.add_argument("--max-wait-time", type=float, default=0.02, help="Seconds to wait for a partial batch.")
    parser.add_argument(
        "--env-cache-size",
        type=int,
        default=1,
        help=(
            "Maximum cached LIBERO envs per worker. Default 1 bounds MuJoCo child "
            "processes while preserving repeated-task reuse; 0 closes after each "
            "episode; negative restores unbounded legacy caching."
        ),
    )
    parser.add_argument(
        "--rollout-wall-timeout-sec",
        type=float,
        default=0.0,
        help="Per-episode wall timeout passed through to rollout_episode; 0 disables.",
    )
    parser.add_argument(
        "--task-manifest-path",
        type=str,
        default="",
        help="Optional JSON manifest with a tasks list; bypasses suite metadata discovery for continuation evals.",
    )
    parser.add_argument(
        "--external-vla-server-url",
        type=str,
        default="",
        help="Use a vla-evaluation-harness WebSocket model server instead of loading a local checkpoint.",
    )
    parser.add_argument(
        "--external-vla-server-timeout-sec",
        type=float,
        default=900.0,
        help="Per-request timeout for --external-vla-server-url.",
    )
    parser.add_argument(
        "--external-vla-benchmark-name",
        type=str,
        default="libero_batched",
        help="HELLO benchmark name sent to the external VLA server.",
    )
    args = parser.parse_args()

    if args.wandb:
        raise ValueError("Batched eval is intentionally W&B-free; omit --wandb.")
    if args.record_video or args.detailed_video or args.decode_visuals:
        raise ValueError("Batched eval is a throughput path; video/decode diagnostics are unsupported.")
    if args.temporal_ensemble:
        raise ValueError("Batched eval currently lacks --temporal-ensemble support.")
    if args.execution_strategy != "default":
        raise ValueError("Batched eval currently supports --execution-strategy default only.")

    args.proprio_orientation = (
        "auto"
        if str(args.proprio_orientation or "auto").strip().lower() in {"", "auto"}
        else elu.normalize_proprio_orientation_mode(args.proprio_orientation)
    )
    args.text_prompt_normalization = elu.normalize_text_prompt_mode(args.text_prompt_normalization)
    suites = elu.parse_csv_list(args.suites)
    if not suites:
        raise ValueError("At least one suite is required.")
    plus_root = args.plus_root if args.plus else None
    if args.plus and not plus_root:
        raise ValueError("LIBERO-Plus eval requires --plus-root or DA3_LIBERO_PLUS_DIR.")

    preset = elu.PRESETS[args.preset]
    seed = preset.seed if args.seed is None else int(args.seed)
    env_seed = preset.env_seed if args.env_seed is None else int(args.env_seed)
    num_steps_wait = preset.num_steps_wait if args.num_steps_wait is None else int(args.num_steps_wait)
    if args.num_trials_per_task is not None:
        num_trials = int(args.num_trials_per_task)
        num_trials_source = "cli"
    elif args.plus:
        num_trials = 1
        num_trials_source = "libero_plus_default"
    else:
        num_trials = int(preset.num_trials_per_task)
        num_trials_source = "preset"
    external_vla_server_url = str(args.external_vla_server_url or "").strip()
    if external_vla_server_url:
        if args.ckpt is None:
            args.ckpt = "external-vla-server"
        if args.action_horizon is None:
            args.action_horizon = "1"

    elu.set_global_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = elu._resolve_eval_cuda_device()
    render_gpu_device_id = elu._resolve_render_gpu_device_id(args.render_gpu_device_id)
    try:
        from robot.evaluation.closed_loop_libero_eval import _install_mujoco_glcontext_patch

        _install_mujoco_glcontext_patch(render_gpu_device_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[batched_eval] WARN: GLContext patch skipped: {exc}", flush=True)

    if external_vla_server_url:
        cfg = OmegaConf.create({})
        policy = None
        policy_info = _external_vla_policy_info(args)
    else:
        cfg = OmegaConf.create({}) if args.config is None else OmegaConf.load(args.config)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        policy, policy_info = elu.load_policy(args, cfg, ckpt, device)
        if not callable(getattr(policy, "predict_batch", None)):
            raise RuntimeError("Loaded policy lacks predict_batch().")
    rollout_action_frame = elu.resolve_rollout_action_frame(args.action_frame, policy_info)
    policy_info["rollout_action_frame"] = rollout_action_frame
    action_horizon, action_horizon_requested = elu.resolve_action_horizon(
        args.action_horizon,
        preset=preset,
        policy_info=policy_info,
    )
    if policy is not None:
        policy.active_action_horizon = int(action_horizon)

    work, build_meta = _build_work_items(args, suites, num_trials, plus_root)
    eval_shard = elu.resolve_eval_shard(args.shard_index, args.shard_count)
    if eval_shard.enabled:
        work = [item for item in work if eval_shard.owns(int(item["global_index"]))]
    if not work:
        raise ValueError("No LIBERO episodes selected for batched eval.")
    parallel_envs = max(1, int(args.parallel_envs))
    max_batch_size = max(1, int(args.max_batch_size))
    env_cache_size = int(args.env_cache_size)
    worker_items = [work[i::parallel_envs] for i in range(parallel_envs)]
    worker_items = [items for items in worker_items if items]

    run_name = args.run_name or f"{Path(args.ckpt).stem}-{args.preset}-batched-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if eval_shard.enabled and args.run_name is None:
        run_name += f"-shard{eval_shard.index}of{eval_shard.count}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = run_dir / "progress.jsonl"
    progress_log_path.write_text("")

    print(
        "[batched_eval] ckpt=%s external_vla=%s suites=%s episodes=%d parallel_envs=%d max_batch_size=%d max_wait_time=%.3f env_cache_size=%d shard=%s output=%s"
        % (
            args.ckpt,
            external_vla_server_url or "none",
            ",".join(suites),
            len(work),
            len(worker_items),
            max_batch_size,
            float(args.max_wait_time),
            env_cache_size,
            f"{eval_shard.index}/{eval_shard.count}" if eval_shard.enabled else "disabled",
            run_dir,
        ),
        flush=True,
    )

    dispatcher = (
        None
        if external_vla_server_url
        else BatchedPolicyDispatcher(
            policy,
            max_batch_size=max_batch_size,
            max_wait_time=float(args.max_wait_time),
        )
    )
    total_start = time.time()
    progress_rows: list[dict[str, Any]] = []
    progress_lock = threading.Lock()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_items)) as executor:
            futures = [
                executor.submit(
                    _worker_run,
                    worker_idx=worker_idx,
                    items=items,
                    args=args,
                    dispatcher=dispatcher,
                    action_horizon=int(action_horizon),
                    action_frame=rollout_action_frame,
                    policy_info=policy_info,
                    render_gpu_device_id=render_gpu_device_id,
                    plus_root=plus_root,
                    env_seed=int(env_seed),
                    num_steps_wait=int(num_steps_wait),
                    progress_lock=progress_lock,
                    progress_rows=progress_rows,
                    progress_log_path=progress_log_path,
                    suites=suites,
                    run_name=run_name,
                    env_cache_size=env_cache_size,
                )
                for worker_idx, items in enumerate(worker_items)
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    finally:
        if dispatcher is not None:
            dispatcher.close()

    progress_rows = sorted(progress_rows, key=lambda row: (str(row.get("suite", "")), int(row.get("eval_task_index", 0)), int(row.get("episode_idx", 0)), int(row.get("worker_idx", 0))))
    per_task_rows = _per_task_rows_from_progress(progress_rows)
    suite_results = elu.suite_results_from_rows(per_task_rows, suites)
    plus_category_results = elu.aggregate_plus_category_results(per_task_rows) if args.plus else {}
    total_successes = int(sum(int(item["num_success"]) for item in suite_results.values()))
    total_episodes = int(sum(int(item["num_trials"]) for item in suite_results.values()))
    overall_success = float(total_successes / total_episodes) if total_episodes else 0.0
    average_success = float(np.mean([item["success_rate"] for item in suite_results.values()])) if suite_results else 0.0
    elapsed_sec = time.time() - total_start
    summary = {
        "run_name": run_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_type": "libero_batched_harness_style",
        "ckpt": args.ckpt,
        "config": (
            "external_vla_server"
            if external_vla_server_url
            else args.config if args.config is not None else f"{args.ckpt}:config"
        ),
        "preset": args.preset,
        "seed": seed,
        "env_seed": env_seed,
        "num_trials_per_task": num_trials,
        "num_trials_per_task_source": num_trials_source,
        "num_steps_wait": num_steps_wait,
        "camera_size": int(args.camera_size),
        "parallel_envs": len(worker_items),
        "max_batch_size": max_batch_size,
        "max_wait_time": float(args.max_wait_time),
        "env_cache_size": env_cache_size,
        "task_manifest_path": build_meta.get("task_manifest_path"),
        "task_manifest_total_tasks": build_meta.get("task_manifest_total_tasks"),
        "task_manifest_selected_tasks": build_meta.get("task_manifest_selected_tasks"),
        "external_vla_server_url": external_vla_server_url or None,
        "external_vla_server_timeout_sec": (
            float(args.external_vla_server_timeout_sec) if external_vla_server_url else None
        ),
        "external_vla_benchmark_name": (
            str(args.external_vla_benchmark_name) if external_vla_server_url else None
        ),
        "shard": {
            "enabled": bool(eval_shard.enabled),
            "index": int(eval_shard.index),
            "count": int(eval_shard.count),
        },
        "work_assignment": "flat_round_robin",
        "env_process_isolation": bool(args.env_process_isolation),
        "env_worker_timeout_sec": float(args.env_worker_timeout_sec),
        "action_horizon": int(action_horizon),
        "action_horizon_requested": action_horizon_requested,
        "action_repeat_requested": args.action_repeat,
        "action_repeat_mode": args.action_repeat_mode,
        "history_horizon_requested": args.history_horizon,
        "rollout_decode_horizon_requested": args.rollout_decode_horizon,
        "execute_chunk_prefix": int(args.execute_chunk_prefix),
        "partial_chunk_history": args.partial_chunk_history,
        "warmup_full_chunk_once": bool(args.warmup_full_chunk_once),
        "rollout_wall_timeout_sec": float(args.rollout_wall_timeout_sec),
        "plus": bool(args.plus),
        "plus_root": plus_root,
        "plus_perturbation": args.plus_perturbation if args.plus else None,
        "plus_official_category": args.plus_official_category if args.plus else None,
        "libero_plus_robot_init_qpos_mode": args.libero_plus_robot_init_qpos_mode if args.plus else None,
        "plus_subset_manifests": build_meta["plus_subset_manifests"] if args.plus else {},
        "plus_official_classification": build_meta["plus_classification"] if args.plus else None,
        "plus_official_category_results": plus_category_results,
        "suites": suites,
        "suite_results": suite_results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": overall_success,
        "average_success_rate": average_success,
        "elapsed_sec": elapsed_sec,
        "throughput_episodes_per_sec": float(total_episodes / elapsed_sec) if elapsed_sec > 0 else 0.0,
        "policy": policy_info,
        "artifacts": {
            "summary_path": str(run_dir / "summary.json"),
            "per_task_path": str(run_dir / "per_task.csv"),
            "progress_log": str(progress_log_path),
        },
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    elu.write_per_task_csv(run_dir / "per_task.csv", per_task_rows)
    print(
        "[batched_eval] done overall=%.1f%%(%d/%d) elapsed=%.1fs throughput=%.3f eps summary=%s"
        % (
            overall_success * 100.0,
            total_successes,
            total_episodes,
            elapsed_sec,
            summary["throughput_episodes_per_sec"],
            run_dir / "summary.json",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
