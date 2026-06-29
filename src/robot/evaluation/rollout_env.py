"""Utilities for robosuite-based rollout evaluation without robomimic."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import faulthandler
import io
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import traceback
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

logger = logging.getLogger(__name__)
_ROBOSUITE_EGL_CONTEXT_PATCHED = False


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def iter_env_candidates(env: Any):
    """Yield wrapper and nested env objects without assuming a specific wrapper."""
    seen: set[int] = set()
    stack = [env]
    while stack:
        candidate = stack.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for attr in ("env", "base_env"):
            child = getattr(candidate, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def patch_robosuite_egl_context() -> None:
    """Make robosuite offscreen EGL readback robust to stale current context."""
    global _ROBOSUITE_EGL_CONTEXT_PATCHED
    mujoco_gl = (
        str(os.environ.get("MUJOCO_GL") or os.environ.get("DA3_MUJOCO_GL") or "")
        .strip()
        .lower()
    )
    if mujoco_gl and mujoco_gl != "egl":
        return
    try:
        from robosuite.utils import binding_utils  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not patch robosuite EGL context before rollout: %s", exc)
        return

    cls = getattr(binding_utils, "MjRenderContext", None)
    if cls is None:
        logger.warning("Could not patch robosuite EGL context: MjRenderContext missing")
        return
    patch_version = int(getattr(cls, "_da3_egl_make_current_patch_version", 0) or 0)
    if patch_version >= 2:
        _ROBOSUITE_EGL_CONTEXT_PATCHED = True
        return

    orig_render = getattr(cls, "_da3_orig_render", None) or getattr(cls, "render", None)
    orig_read_pixels = getattr(cls, "_da3_orig_read_pixels", None) or getattr(cls, "read_pixels", None)
    if not callable(orig_render) or not callable(orig_read_pixels):
        logger.warning("Could not patch robosuite EGL context: render/read_pixels missing")
        return

    def _make_current(self: Any) -> None:
        try:
            gl_ctx = getattr(self, "gl_ctx", None)
            make_current = getattr(gl_ctx, "make_current", None)
            if callable(make_current):
                make_current()
        except Exception:  # noqa: BLE001
            pass

    def _load_gl_finish() -> Any | None:
        for lib_name in (
            None,
            ctypes.util.find_library("GL"),
            ctypes.util.find_library("OpenGL"),
            "libGL.so.1",
        ):
            try:
                lib = ctypes.CDLL(lib_name) if lib_name is not None else ctypes.CDLL(None)
                func = getattr(lib, "glFinish")
                func.argtypes = []
                func.restype = None
                return func
            except Exception:  # noqa: BLE001
                continue
        return None

    use_gl_finish = _env_flag("DA3_ROBOSUITE_EGL_GLFINISH", True)
    gl_finish = _load_gl_finish() if use_gl_finish else None

    def _gl_finish() -> None:
        if gl_finish is None:
            return
        try:
            gl_finish()
        except Exception:  # noqa: BLE001
            pass

    def render_with_current(self: Any, *args: Any, **kwargs: Any):
        _make_current(self)
        return orig_render(self, *args, **kwargs)

    def read_pixels_with_current(self: Any, *args: Any, **kwargs: Any):
        _make_current(self)
        _gl_finish()
        return orig_read_pixels(self, *args, **kwargs)

    cls._da3_orig_render = orig_render
    cls._da3_orig_read_pixels = orig_read_pixels
    cls.render = render_with_current
    cls.read_pixels = read_pixels_with_current
    cls._da3_egl_make_current_patch = True
    cls._da3_egl_make_current_patch_version = 2
    _ROBOSUITE_EGL_CONTEXT_PATCHED = True
    logger.info(
        "Patched robosuite MjRenderContext to make EGL context current before render/readback "
        "and glFinish before readback (glFinish=%s)",
        "enabled" if use_gl_finish else "disabled",
    )


class RobosuiteRolloutEnv:
    """Small compatibility wrapper exposing the robomimic env methods used in eval."""

    def __init__(
        self,
        env,
        camera_names: Iterable[str],
        camera_height: int,
        camera_width: int,
        enable_render: bool,
        camera_depths: bool = False,
        env_image_hflip: bool | None = None,
        env_image_vflip: bool | None = None,
        env_image_rotate180: bool | None = None,
        hdf5_hflip_fix: bool | None = None,
        preserve_libero_plus_robot_init_qpos: bool = False,
        action_dimension_override: int | None = None,
    ) -> None:
        self.env = env
        self.base_env = env
        self.camera_names = tuple(camera_names)
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.enable_render = bool(enable_render)
        self.camera_depths = bool(camera_depths)
        if env_image_hflip is None:
            if env_image_rotate180 is not None:
                env_image_hflip = bool(env_image_rotate180)
            elif hdf5_hflip_fix is not None:
                env_image_hflip = bool(hdf5_hflip_fix)
            else:
                env_image_hflip = False
        self.env_image_hflip = bool(env_image_hflip)
        self.env_image_vflip = True if env_image_vflip is None else bool(env_image_vflip)
        # Backward-compatible alias: historical callers only distinguished
        # between "vflip-only" and "rotate180". For OpenGL bottom-up renders,
        # rotate180 is equivalent to applying an extra horizontal flip after the
        # always-on vertical flip.
        self.env_image_rotate180 = bool(env_image_rotate180) if env_image_rotate180 is not None else self.env_image_hflip
        self.preserve_libero_plus_robot_init_qpos = bool(preserve_libero_plus_robot_init_qpos)
        inner_env = getattr(env, "env", None)
        action_dim = None if action_dimension_override is None else int(action_dimension_override)
        if action_dim is None:
            action_dim = getattr(env, "action_dim", None)
            if action_dim is None and inner_env is not None:
                action_dim = getattr(inner_env, "action_dim", None)
            if action_dim is None:
                action_spec = getattr(env, "action_spec", None)
                if action_spec is None and inner_env is not None:
                    action_spec = getattr(inner_env, "action_spec", None)
                if isinstance(action_spec, (tuple, list)) and action_spec:
                    action_dim = int(np.asarray(action_spec[0]).reshape(-1).shape[0])
        if action_dim is None:
            raise AttributeError(f"Could not infer action_dim from {type(env).__name__}")
        self.action_dimension = int(action_dim)

    def reset(self):
        return self.get_observation(self.env.reset())

    def set_init_state(self, init_state):
        """Reset the simulator to a LIBERO flattened initial state."""
        set_init_state = getattr(self.env, "set_init_state", None)
        if callable(set_init_state):
            self.env.reset()
            obs = set_init_state(init_state)
            if self.preserve_libero_plus_robot_init_qpos and self._restore_libero_plus_robot_init_qpos():
                return self.get_observation()
            return self.get_observation(obs)

        self.env.reset()
        self.env.sim.set_state_from_flattened(np.asarray(init_state))
        self.env.sim.forward()
        return self.get_observation()

    @staticmethod
    def _robot_joint_indexes(inner_env, robot, *, position: bool) -> np.ndarray | None:
        if position:
            attr_names = ("_ref_joint_pos_indexes", "joint_pos_indexes", "_joint_pos_indexes")
            model_addr = "jnt_qposadr"
        else:
            attr_names = ("_ref_joint_vel_indexes", "joint_vel_indexes", "_joint_vel_indexes")
            model_addr = "jnt_dofadr"

        for attr in attr_names:
            indexes = getattr(robot, attr, None)
            if indexes is not None:
                return np.asarray(indexes, dtype=int).reshape(-1)

        robot_model = getattr(robot, "robot_model", None)
        joint_names = tuple(getattr(robot_model, "joints", ()) or ())
        sim_model = getattr(getattr(inner_env, "sim", None), "model", None)
        if not joint_names or sim_model is None or not hasattr(sim_model, model_addr):
            return None

        indexes: list[int] = []
        address_array = getattr(sim_model, model_addr)
        for joint_name in joint_names:
            try:
                joint_id = sim_model.joint_name2id(joint_name)
            except Exception:
                continue
            indexes.append(int(address_array[joint_id]))
        return np.asarray(indexes, dtype=int).reshape(-1) if indexes else None

    def _restore_libero_plus_robot_init_qpos(self) -> bool:
        """Preserve LIBERO-Plus synthetic robot initial-state perturbations.

        LIBERO-Plus encodes robot perturbations by selecting robot classes such
        as OnTheGroundPanda11. The shared base init-state file is still loaded
        for these synthetic tasks, so restore the selected robot class qpos
        after the flattened MuJoCo state has been applied.
        """
        raw_env = self.env
        inner_env = getattr(raw_env, "env", raw_env)
        try:
            init_state_id = int(getattr(inner_env, "init_state", 0))
        except (TypeError, ValueError):
            return False
        if init_state_id == 0:
            return False

        robots = getattr(inner_env, "robots", None) or []
        if not robots:
            return False
        robot = robots[0]
        robot_model = getattr(robot, "robot_model", None)
        init_qpos = getattr(robot_model, "init_qpos", None)
        sim = getattr(inner_env, "sim", None)
        if init_qpos is None or sim is None:
            return False

        init_qpos_array = np.asarray(init_qpos, dtype=float).reshape(-1)
        pos_indexes = self._robot_joint_indexes(inner_env, robot, position=True)
        if pos_indexes is None or len(pos_indexes) == 0:
            return False
        count = min(len(pos_indexes), len(init_qpos_array))
        sim.data.qpos[pos_indexes[:count]] = init_qpos_array[:count]

        vel_indexes = self._robot_joint_indexes(inner_env, robot, position=False)
        if vel_indexes is not None and len(vel_indexes) > 0:
            sim.data.qvel[vel_indexes[: min(len(vel_indexes), count)]] = 0.0

        sim.forward()
        post_process = getattr(raw_env, "_post_process", None)
        if callable(post_process):
            post_process()
        update_observables = getattr(raw_env, "_update_observables", None)
        if callable(update_observables):
            update_observables(force=True)
        return True

    def reset_to(self, state: Mapping[str, object], raw_observation: bool = False):
        """Restore XML + simulator state captured in the dataset."""
        should_return_obs = False
        edit_env = self.env
        if not hasattr(edit_env, "edit_model_xml") and hasattr(edit_env, "env"):
            edit_env = self.env.env
        model_xml = state.get("model", state.get("model_file"))
        ep_meta_raw = state.get("ep_meta")
        if ep_meta_raw is not None:
            if isinstance(ep_meta_raw, bytes):
                ep_meta_raw = ep_meta_raw.decode("utf-8")
            if isinstance(ep_meta_raw, str):
                ep_meta = json.loads(ep_meta_raw) if ep_meta_raw.strip() else {}
            elif isinstance(ep_meta_raw, dict):
                ep_meta = dict(ep_meta_raw)
            else:
                ep_meta = {}
            for candidate in iter_env_candidates(self.env):
                for setter_name in ("set_attrs_from_ep_meta", "set_ep_meta"):
                    setter = getattr(candidate, setter_name, None)
                    if callable(setter):
                        setter(ep_meta)
                        break
        if model_xml is not None:
            if isinstance(model_xml, bytes):
                model_xml = model_xml.decode("utf-8")
            self.env.reset()
            xml = edit_env.edit_model_xml(model_xml)
            edit_env.reset_from_xml_string(xml)
            edit_env.sim.reset()
        if "states" in state:
            state_array = np.asarray(state["states"])
            if model_xml is None:
                regenerate_obs = getattr(self.env, "regenerate_obs_from_state", None)
                if callable(regenerate_obs):
                    obs_dict = regenerate_obs(state_array)
                    return obs_dict if raw_observation else self.get_observation(obs_dict)
            edit_env.sim.set_state_from_flattened(state_array)
            edit_env.sim.forward()
            should_return_obs = True
        if should_return_obs:
            return self.get_observation()
        return None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.get_observation(obs), reward, done, info

    def render(self, mode="human", height=None, width=None, camera_name=None):
        camera_name = camera_name or self.camera_names[0]
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        if mode == "rgb_array":
            height = int(height or self.camera_height)
            width = int(width or self.camera_width)
            return np.ascontiguousarray(
                self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
            )
        if mode == "depth_array":
            return self.render_depth(height=height, width=width, camera_name=camera_name)
        raise NotImplementedError(f"Unsupported render mode: {mode}")

    def render_depth(self, height=None, width=None, camera_name=None):
        camera_name = camera_name or self.camera_names[0]
        height = int(height or self.camera_height)
        width = int(width or self.camera_width)
        rendered = self.env.sim.render(height=height, width=width, camera_name=camera_name, depth=True)
        if isinstance(rendered, tuple):
            _, depth = rendered
        else:
            depth = rendered
        return np.ascontiguousarray(np.asarray(depth)[::-1])

    def get_observation(self, obs_dict=None):
        if obs_dict is None:
            obs_source = self.env
            if not hasattr(obs_source, "_get_observations") and hasattr(self.env, "env"):
                obs_source = self.env.env
            if hasattr(obs_source, "_get_observations"):
                obs_dict = obs_source._get_observations(force_update=True)
            else:
                obs_dict = obs_source._get_observation()

        # Default: vflip-only on OpenGL raw (natural orientation).
        # Some LIBERO HDF5-family datasets intentionally add an extra
        # horizontal flip on top of that natural orientation. On raw HDF5 this
        # reproduces the historical rotate180 contract; on replayed libero_noop
        # it upgrades the stored vflip-only frames to the same effective view.
        ret = {}
        for key, value in obs_dict.items():
            if key.endswith("_image"):
                arr = np.asarray(value)
                if self.env_image_vflip:
                    arr = arr[::-1]
                ret[key] = np.ascontiguousarray(arr[:, ::-1] if self.env_image_hflip else arr)
            elif key.endswith("_depth"):
                arr = np.asarray(value)
                if self.env_image_vflip:
                    arr = arr[::-1]
                ret[key] = np.ascontiguousarray(arr[:, ::-1] if self.env_image_hflip else arr)
            else:
                ret[key] = np.asarray(value)

        if self.enable_render:
            for camera_name in self.camera_names:
                image_key = f"{camera_name}_image"
                if image_key not in ret:
                    rendered = self.render(
                        mode="rgb_array",
                        height=self.camera_height,
                        width=self.camera_width,
                        camera_name=camera_name,
                    )
                    ret[image_key] = np.ascontiguousarray(
                        np.asarray(rendered)[:, ::-1] if self.env_image_hflip else rendered
                    )
                depth_key = f"{camera_name}_depth"
                if self.camera_depths and depth_key not in ret:
                    rendered_depth = self.render_depth(
                        height=self.camera_height,
                        width=self.camera_width,
                        camera_name=camera_name,
                    )
                    ret[depth_key] = (
                        np.ascontiguousarray(np.asarray(rendered_depth)[:, ::-1])
                        if self.env_image_hflip
                        else np.ascontiguousarray(rendered_depth)
                    )
        return ret

    def is_success(self):
        if hasattr(self.env, "_check_success"):
            return {"task": bool(self.env._check_success())}
        if hasattr(self.env, "env") and hasattr(self.env.env, "_check_success"):
            return {"task": bool(self.env.env._check_success())}
        return {"task": False}

    def close(self):
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()


def _ipc_safe_value(value: Any) -> Any:
    """Return values that are cheap and safe to pickle across an env-worker pipe."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _ipc_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ipc_safe_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _signal_name_from_returncode(returncode: int | None) -> str:
    if returncode is None:
        return "running"
    if returncode >= 0:
        return f"exit={returncode}"
    signum = -int(returncode)
    try:
        return f"signal={signal.Signals(signum).name}({signum})"
    except Exception:  # noqa: BLE001
        return f"signal={signum}"


def _seed_worker_env(env: Any, seed: int) -> bool:
    for candidate in (env, getattr(env, "env", None), getattr(env, "base_env", None)):
        if candidate is None:
            continue
        seed_fn = getattr(candidate, "seed", None)
        if callable(seed_fn):
            seed_fn(int(seed))
            return True
    return False


def _worker_control_freq(env: Any) -> float | None:
    for candidate in (env, getattr(env, "env", None), getattr(env, "base_env", None)):
        value = getattr(candidate, "control_freq", None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _safe_process_name_part(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _redirect_libero_worker_stderr(env_kwargs: Mapping[str, Any]) -> None:
    """Move isolated worker native crash logs out of the parent shard log."""
    log_dir = str(os.environ.get("DA3_LIBERO_ENV_WORKER_STDERR_DIR", "")).strip()
    if not log_dir:
        return
    try:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "r"
        suite = _safe_process_name_part(env_kwargs.get("suite_name", "suite"))
        task = _safe_process_name_part(env_kwargs.get("task_id", "task"))
        stderr_path = path / f"libero_env_worker_r{rank}_t{task}_{suite}_pid{os.getpid()}.log"
        fd = os.open(str(stderr_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, sys.stderr.fileno())
        os.close(fd)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "a", buffering=1, closefd=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not redirect LIBERO env worker stderr: %s", exc)


def _libero_env_worker_ready_payload(env: Any, task_name: str, init_states: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "type": "ready",
        "task_name": str(task_name),
        "init_states": _ipc_safe_value(np.asarray(init_states)),
        "action_dimension": int(getattr(env, "action_dimension", 7)),
        "control_freq": _worker_control_freq(env),
        "camera_names": list(getattr(env, "camera_names", ())),
    }


def _libero_env_worker_main(conn: Any, env_kwargs: dict[str, Any]) -> None:
    """Own one LIBERO/MuJoCo env in a child process.

    MuJoCo/EGL failures can call abort(3), bypassing Python exception handling.
    Keeping the simulator in this child lets the parent eval rank count the
    episode as crashed instead of losing the whole Slurm task.
    """
    _redirect_libero_worker_stderr(env_kwargs)
    faulthandler.enable(file=sys.stderr)
    env = None
    task_name = ""
    init_states = None
    try:
        env, task_name, init_states = create_rollout_env_libero(**env_kwargs)
        conn.send(_libero_env_worker_ready_payload(env, task_name, init_states))
        while True:
            msg = conn.recv()
            cmd = str(msg.get("cmd", ""))
            try:
                if cmd == "seed":
                    seeded = _seed_worker_env(env, int(msg["seed"]))
                    conn.send({"ok": True, "seeded": bool(seeded)})
                elif cmd == "set_init_state":
                    obs = env.set_init_state(msg.get("init_state"))
                    conn.send({"ok": True, "obs": _ipc_safe_value(obs)})
                elif cmd == "step":
                    action = np.asarray(msg["action"], dtype=np.float32)
                    obs, reward, done, info = env.step(action)
                    conn.send(
                        {
                            "ok": True,
                            "obs": _ipc_safe_value(obs),
                            "reward": float(reward),
                            "done": bool(done),
                            "info": _ipc_safe_value(info or {}),
                        }
                    )
                elif cmd == "render":
                    frame = env.render(
                        mode=msg.get("mode", "rgb_array"),
                        height=msg.get("height"),
                        width=msg.get("width"),
                        camera_name=msg.get("camera_name"),
                    )
                    conn.send({"ok": True, "frame": _ipc_safe_value(frame)})
                elif cmd == "render_depth":
                    frame = env.render_depth(
                        height=msg.get("height"),
                        width=msg.get("width"),
                        camera_name=msg.get("camera_name"),
                    )
                    conn.send({"ok": True, "frame": _ipc_safe_value(frame)})
                elif cmd == "get_task_description":
                    conn.send({"ok": True, "task_desc": str(task_name)})
                elif cmd == "is_success":
                    conn.send({"ok": True, "success": bool(env.is_success().get("task", False))})
                elif cmd == "recreate_env":
                    next_env_kwargs = dict(msg.get("env_kwargs") or {})
                    if env is not None:
                        close_fn = getattr(env, "close", None)
                        if callable(close_fn):
                            close_fn()
                    env, task_name, init_states = create_rollout_env_libero(**next_env_kwargs)
                    env_kwargs = next_env_kwargs
                    conn.send(_libero_env_worker_ready_payload(env, task_name, init_states))
                elif cmd == "close":
                    # Acknowledge before closing. If MuJoCo aborts during
                    # teardown, the parent has already detached from the child.
                    conn.send({"ok": True, "closing": True})
                    conn.close()
                    try:
                        env.close()
                    finally:
                        return
                else:
                    raise ValueError(f"Unknown LIBERO env-worker command: {cmd!r}")
            except Exception as exc:  # noqa: BLE001
                conn.send(
                    {
                        "ok": False,
                        "cmd": cmd,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send(
                {
                    "ok": False,
                    "cmd": "worker_init",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


class LiberoEnvWorkerProxy:
    """Small proxy that makes a child-process LIBERO env look like a rollout env."""

    def __init__(
        self,
        *,
        env_kwargs: dict[str, Any],
        timeout_sec: float,
        rank: int | None = None,
        task_id: int | None = None,
    ) -> None:
        self._timeout_sec = max(1.0, float(timeout_sec))
        self._rank = rank
        self._task_id = task_id
        self._closed = False
        self._ctx = mp.get_context("spawn")
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._conn = parent_conn
        process_name = "libero-env"
        if rank is not None:
            process_name += f"-r{rank}"
        if task_id is not None:
            process_name += f"-t{task_id}"
        process_name += f"-{_safe_process_name_part(env_kwargs.get('suite_name', 'suite'))}"
        self._proc = self._ctx.Process(
            target=_libero_env_worker_main,
            args=(child_conn, dict(env_kwargs)),
            name=process_name,
        )
        self._proc.daemon = True
        self._proc.start()
        child_conn.close()
        ready = self._recv_response("worker_init")
        self.task_name = str(ready.get("task_name", ""))
        self.init_states = np.asarray(ready.get("init_states"))
        self.action_dimension = int(ready.get("action_dimension", 7))
        self.control_freq = ready.get("control_freq")
        self.camera_names = tuple(ready.get("camera_names") or env_kwargs.get("camera_names") or ())
        self.env = None
        self.base_env = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def _context(self) -> str:
        return (
            f"rank={self._rank}, task_id={self._task_id}, "
            f"pid={self.pid}, return={_signal_name_from_returncode(self._proc.exitcode)}"
        )

    def _recv_response(self, cmd: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_sec
        while True:
            if self._conn.poll(0.2):
                try:
                    msg = self._conn.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        f"LIBERO env worker exited before replying to {cmd} ({self._context()})."
                    ) from exc
                if not bool(msg.get("ok", False)):
                    raise RuntimeError(
                        "LIBERO env worker command failed "
                        f"(cmd={cmd}, {self._context()}): "
                        f"{msg.get('error')}\n{msg.get('traceback', '')}"
                    )
                return msg
            if self._proc.exitcode is not None:
                raise RuntimeError(
                    f"LIBERO env worker died while waiting for {cmd} ({self._context()})."
                )
            if time.monotonic() >= deadline:
                self._terminate()
                raise TimeoutError(
                    f"LIBERO env worker timed out waiting for {cmd} after "
                    f"{self._timeout_sec:.1f}s ({self._context()})."
                )

    def _request(self, cmd: str, **payload: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("LIBERO env worker proxy is closed.")
        self._conn.send({"cmd": cmd, **payload})
        return self._recv_response(cmd)

    def recreate(
        self,
        *,
        env_kwargs: dict[str, Any],
        rank: int | None = None,
        task_id: int | None = None,
    ) -> None:
        """Replace the child-owned MuJoCo env without respawning Python.

        The child still constructs a fresh robosuite/LIBERO env for the new
        task, so task semantics and init-state handling are unchanged. If the
        child dies while closing/reopening native MuJoCo resources, the caller
        can fall back to creating a new worker process.
        """
        self._rank = rank
        self._task_id = task_id
        ready = self._request("recreate_env", env_kwargs=dict(env_kwargs))
        self.task_name = str(ready.get("task_name", ""))
        self.init_states = np.asarray(ready.get("init_states"))
        self.action_dimension = int(ready.get("action_dimension", 7))
        self.control_freq = ready.get("control_freq")
        self.camera_names = tuple(ready.get("camera_names") or env_kwargs.get("camera_names") or ())

    def seed(self, seed: int) -> None:
        self._request("seed", seed=int(seed))

    def set_init_state(self, init_state: Any) -> dict[str, Any]:
        return self._request("set_init_state", init_state=init_state)["obs"]

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        msg = self._request("step", action=np.asarray(action, dtype=np.float32))
        return msg["obs"], float(msg["reward"]), bool(msg["done"]), dict(msg.get("info") or {})

    def render(
        self,
        mode: str = "rgb_array",
        height: int | None = None,
        width: int | None = None,
        camera_name: str | None = None,
    ):
        return self._request("render", mode=mode, height=height, width=width, camera_name=camera_name)["frame"]

    def render_depth(
        self,
        height: int | None = None,
        width: int | None = None,
        camera_name: str | None = None,
    ):
        return self._request("render_depth", height=height, width=width, camera_name=camera_name)["frame"]

    def get_task_description(self) -> str | None:
        value = self._request("get_task_description").get("task_desc")
        return None if value is None else str(value)

    def is_success(self) -> dict[str, bool]:
        return {"task": bool(self._request("is_success").get("success", False))}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.is_alive():
                try:
                    self._conn.send({"cmd": "close"})
                    if self._conn.poll(2.0):
                        try:
                            self._conn.recv()
                        except EOFError:
                            pass
                except Exception:  # noqa: BLE001
                    pass
                self._proc.join(timeout=5.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=5.0)
                if self._proc.is_alive():
                    self._proc.kill()
                    self._proc.join(timeout=2.0)
        finally:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _terminate(self) -> None:
        try:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# ── LIBERO env creation ──────────────────────────────────────────────

_UNSET = object()
_ACTIVE_LIBERO_PLUS_ROOT: str | None = None
_PREV_LIBERO_CONFIG_PATH: str | None | object = _UNSET


def _delete_libero_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]


def _validate_libero_plus_dependencies() -> None:
    """Validate LIBERO-Plus import-time dependencies from the active env."""
    missing: list[str] = []
    for module_name, package_hint in (
        ("gym", "gym"),
        ("wand.image", "wand + ImageMagick"),
        ("skimage.filters", "scikit-image"),
    ):
        try:
            __import__(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{package_hint} ({exc})")
    if missing:
        raise RuntimeError(
            "LIBERO-Plus perturbation dependencies are missing from the active "
            "Docker/conda environment: " + "; ".join(missing)
        )


def _libero_assets_marker(path: Path) -> bool:
    return (path / "scenes" / "libero_tabletop_base_style.xml").exists()


def _read_assets_from_libero_config() -> Path | None:
    config_file = Path.home() / ".libero" / "config.yaml"
    if not config_file.exists():
        return None
    for line in config_file.read_text().splitlines():
        if line.strip().startswith("assets:"):
            return Path(line.split(":", 1)[1].strip()).expanduser()
    return None


def _find_libero_assets_dir(plus_root: Path) -> Path | None:
    candidates: list[Path] = []
    for env_name in ("DA3_LIBERO_PLUS_ASSETS_DIR", "DA3_LIBERO_ASSETS_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            plus_root / "libero" / "libero" / "assets",
            Path.home() / ".cache" / "libero" / "assets",
        ]
    )
    config_assets = _read_assets_from_libero_config()
    if config_assets is not None:
        candidates.append(config_assets)

    for candidate in candidates:
        if _libero_assets_marker(candidate):
            return candidate.resolve()
    return None


def _patch_libero_plus_asset_roots(asset_dir: Path) -> None:
    package_root = asset_dir.parent
    bddl_base_domain = sys.modules.get("libero.libero.envs.bddl_base_domain")
    if bddl_base_domain is not None:
        bddl_base_domain.DIR_PATH = str(package_root / "envs")
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("libero.libero.envs.objects.") and hasattr(module, "absolute_path"):
            module.absolute_path = package_root


def _write_libero_plus_config(plus_root: Path) -> Path:
    config_dir = plus_root / ".libero_config_da3"
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = plus_root / "libero" / "libero"
    assets_dir = _find_libero_assets_dir(plus_root) or (benchmark_root / "assets")
    config = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": plus_root / "libero" / "datasets",
        "assets": assets_dir,
    }
    body = "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n"
    config_file = config_dir / "config.yaml"
    if not config_file.exists() or config_file.read_text() != body:
        config_file.write_text(body)
    return config_dir


def _activate_libero_plus_source(plus_root: str | os.PathLike[str]) -> Path:
    global _ACTIVE_LIBERO_PLUS_ROOT, _PREV_LIBERO_CONFIG_PATH

    root = Path(plus_root).expanduser().resolve()
    namespace_root = root / "libero"
    inner_package = namespace_root / "libero" / "__init__.py"
    if not inner_package.exists():
        raise FileNotFoundError(
            f"LIBERO-Plus source checkout missing at {root}. "
            "Run scripts/setup_libero_plus.sh or set DA3_LIBERO_PLUS_DIR / --plus-root."
        )

    root_str = str(root)
    if _ACTIVE_LIBERO_PLUS_ROOT == root_str and "libero.libero" in sys.modules:
        return root

    _validate_libero_plus_dependencies()
    config_dir = _write_libero_plus_config(root)
    if _ACTIVE_LIBERO_PLUS_ROOT is None:
        _PREV_LIBERO_CONFIG_PATH = os.environ.get("LIBERO_CONFIG_PATH")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    _delete_libero_modules()
    top_module = types.ModuleType("libero")
    top_module.__path__ = [str(namespace_root)]
    top_module.__package__ = "libero"
    spec = ModuleSpec("libero", loader=None, is_package=True)
    spec.submodule_search_locations = [str(namespace_root)]
    top_module.__spec__ = spec
    sys.modules["libero"] = top_module
    _ACTIVE_LIBERO_PLUS_ROOT = root_str
    return root


def _deactivate_libero_plus_source() -> None:
    global _ACTIVE_LIBERO_PLUS_ROOT, _PREV_LIBERO_CONFIG_PATH

    if _ACTIVE_LIBERO_PLUS_ROOT is None:
        return
    _delete_libero_modules()
    if _PREV_LIBERO_CONFIG_PATH is None:
        os.environ.pop("LIBERO_CONFIG_PATH", None)
    elif isinstance(_PREV_LIBERO_CONFIG_PATH, str):
        os.environ["LIBERO_CONFIG_PATH"] = _PREV_LIBERO_CONFIG_PATH
    _ACTIVE_LIBERO_PLUS_ROOT = None
    _PREV_LIBERO_CONFIG_PATH = _UNSET


def _import_libero_api(plus_root: str | os.PathLike[str] | None = None):
    if plus_root is None:
        _deactivate_libero_plus_source()
    else:
        _activate_libero_plus_source(plus_root)

    try:
        import gym  # noqa: F401
    except ModuleNotFoundError:
        import gymnasium as gym

        sys.modules.setdefault("gym", gym)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if plus_root is not None:
        asset_dir = _find_libero_assets_dir(Path(plus_root).expanduser().resolve())
        if asset_dir is not None:
            _patch_libero_plus_asset_roots(asset_dir)

    return benchmark, get_libero_path, OffScreenRenderEnv


def _call_with_torch_load_compat(fn, *args, **kwargs):
    """Call legacy LIBERO helpers under PyTorch 2.6 torch.load defaults."""
    import torch

    original_load = torch.load

    def load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load_compat
    try:
        return fn(*args, **kwargs)
    finally:
        torch.load = original_load


def _get_libero_benchmark(suite_name: str, plus_root: str | os.PathLike[str] | None = None):
    """Return a LIBERO benchmark object for the given suite."""
    benchmark, _, _ = _import_libero_api(plus_root)
    bm_dict = benchmark.get_benchmark_dict()
    if suite_name not in bm_dict:
        raise ValueError(f"Unknown LIBERO suite: {suite_name}. Available: {sorted(bm_dict)}")
    if plus_root is None:
        return bm_dict[suite_name](0)  # task_order_index=0
    with contextlib.redirect_stdout(io.StringIO()):
        return bm_dict[suite_name](0)  # LIBERO-Plus prints thousands of task ids.


def _parse_bddl_language(bddl_path: str) -> str | None:
    try:
        from libero.libero.envs import bddl_utils as BDDLUtils

        problem_info = BDDLUtils.get_problem_info(bddl_path)
    except Exception:
        return None
    language = problem_info.get("language_instruction")
    return str(language) if language else None


def _base_bddl_for_language(
    bddl_base: str,
    problem_folder: str,
    bddl_file: str,
) -> str:
    if "_view_" in bddl_file and "_initstate_" in bddl_file:
        bddl_file = bddl_file.split("_view_")[0] + ".bddl"
    return os.path.join(bddl_base, problem_folder, bddl_file)


def _task_metadata(task, task_id: int, get_libero_path=None) -> dict[str, object]:
    language = str(getattr(task, "language", getattr(task, "name", "")))
    policy_language = language
    if get_libero_path is not None:
        bddl_base = get_libero_path("bddl_files")
        problem_folder = str(getattr(task, "problem_folder", ""))
        bddl_file = str(getattr(task, "bddl_file", ""))
        parsed_language = _parse_bddl_language(_base_bddl_for_language(bddl_base, problem_folder, bddl_file))
        if parsed_language:
            policy_language = parsed_language

    return {
        "task_id": int(task_id),
        "name": str(getattr(task, "name", "")),
        "language": language,
        "policy_language": policy_language,
        "problem_folder": str(getattr(task, "problem_folder", "")),
        "bddl_file": str(getattr(task, "bddl_file", "")),
        "init_states_file": str(getattr(task, "init_states_file", "")),
    }


def _libero_plus_init_state_path(task, get_libero_path) -> str:
    init_states_file = str(task.init_states_file)
    problem_folder = str(task.problem_folder)
    init_root = get_libero_path("init_states")
    suffix = init_states_file.split(".")[-1]

    if "_language_" in init_states_file:
        filename = init_states_file.split("_language_")[0] + "." + suffix
        return os.path.join(init_root, problem_folder, filename)
    if "_view_" in init_states_file:
        filename = init_states_file.split("_view_")[0] + "." + suffix
        return os.path.join(init_root, problem_folder, filename)
    if re.search(r"_table_\d+", init_states_file):
        return os.path.join(init_root, problem_folder, re.sub(r"_table_\d+", "", init_states_file))
    if re.search(r"_tb_\d+", init_states_file):
        return os.path.join(init_root, problem_folder, re.sub(r"_tb_\d+", "", init_states_file))
    if "_light_" in init_states_file:
        filename = init_states_file.split("_light_")[0] + "." + suffix
        return os.path.join(init_root, problem_folder, filename)
    if "_add_" in init_states_file or "_level" in init_states_file:
        return os.path.join(init_root, "libero_newobj", problem_folder, init_states_file)
    return os.path.join(init_root, problem_folder, init_states_file)


def _load_libero_plus_init_states(task, get_libero_path) -> np.ndarray:
    import torch

    init_states_path = _libero_plus_init_state_path(task, get_libero_path)
    if not os.path.exists(init_states_path):
        raise FileNotFoundError(f"LIBERO-Plus init states missing: {init_states_path}")

    try:
        init_states = torch.load(init_states_path, weights_only=False)
    except TypeError:
        init_states = torch.load(init_states_path)

    init_states_file = str(task.init_states_file)
    if "_add_" in init_states_file or "_level" in init_states_file:
        init_states = init_states.reshape(1, -1)
    if hasattr(init_states, "detach"):
        init_states = init_states.detach().cpu().numpy()
    return np.asarray(init_states)


def _resolve_bddl_file(
    bddl_base: str,
    problem_folder: str,
    bddl_file: str,
    plus_root: str | os.PathLike[str] | None = None,
) -> str:
    bddl_path = os.path.join(bddl_base, problem_folder, bddl_file)
    if os.path.exists(bddl_path):
        return bddl_path

    if plus_root is not None and "_view_" in bddl_file and "_initstate_" in bddl_file:
        candidate_name = bddl_file.split("_view_")[0] + ".bddl"
        candidate_path = os.path.join(bddl_base, problem_folder, candidate_name)
        if os.path.exists(candidate_path):
            # LIBERO-Plus encodes camera, robot-initial-state, and noise
            # perturbations in synthetic BDDL filenames. Its OffScreenRenderEnv
            # parses the suffix before mapping back to the base BDDL, so
            # preserve the full path here.
            return bddl_path

    raise FileNotFoundError(f"BDDL file missing: {bddl_path}")


def create_rollout_env_libero(
    suite_name: str,
    task_id: int,
    camera_names=("agentview", "robot0_eye_in_hand"),
    camera_size: int = 256,
    render_gpu_device_id: int | None = None,
    control_freq: float | None = None,
    horizon: int | None = None,
    plus_root: str | os.PathLike[str] | None = None,
    camera_depths: bool = False,
    env_image_hflip: bool | None = None,
    env_image_vflip: bool | None = None,
    env_image_rotate180: bool | None = None,
    hdf5_hflip_fix: bool | None = None,
    preserve_libero_plus_robot_init_qpos: bool | None = None,
):
    """Create a LIBERO rollout env for a specific suite + task.

    Uses base ``libero`` by default. When ``plus_root`` is provided, it loads
    the LIBERO-Plus source checkout in-process without replacing the installed
    base package on disk.

    Returns:
        env: RobosuiteRolloutEnv
        task_name: str (human-readable task description)
        init_states: np.ndarray (N, state_dim) : initial states for rollout episodes
    """
    _, get_libero_path, OffScreenRenderEnv = _import_libero_api(plus_root)
    patch_robosuite_egl_context()
    bm = _get_libero_benchmark(suite_name, plus_root=plus_root)
    task = bm.get_task(task_id)
    task_name = task.language if hasattr(task, "language") else task.name

    bddl_base = get_libero_path("bddl_files")
    problem_folder = getattr(task, "problem_folder", suite_name)
    bddl_file = _resolve_bddl_file(bddl_base, problem_folder, task.bddl_file, plus_root=plus_root)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": int(camera_size),
        "camera_widths": int(camera_size),
        "camera_names": list(camera_names),
        "camera_depths": bool(camera_depths),
    }
    if render_gpu_device_id is not None:
        env_args["render_gpu_device_id"] = int(render_gpu_device_id)
    if control_freq is not None:
        if float(control_freq) <= 0:
            raise ValueError(f"control_freq must be positive, got {control_freq}")
        env_args["control_freq"] = int(control_freq) if float(control_freq).is_integer() else float(control_freq)
    if horizon is not None:
        if int(horizon) <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        env_args["horizon"] = int(horizon)
    if plus_root is not None:
        asset_dir = _find_libero_assets_dir(Path(plus_root).expanduser().resolve())
        if asset_dir is None:
            raise FileNotFoundError(
                "LIBERO-Plus assets are missing. Download assets.zip from "
                "https://huggingface.co/datasets/Sylvest/LIBERO-plus and unzip it to "
                "<LIBERO-plus>/libero/libero/assets, or set DA3_LIBERO_PLUS_ASSETS_DIR."
            )
        _patch_libero_plus_asset_roots(asset_dir)
    try:
        raw_env = OffScreenRenderEnv(**env_args)
    except FileNotFoundError as exc:
        if plus_root is not None and "assets" in str(exc):
            raise FileNotFoundError(
                "LIBERO-Plus assets are incomplete for this perturbation. Download assets.zip from "
                "https://huggingface.co/datasets/Sylvest/LIBERO-plus and unzip it to "
                "<LIBERO-plus>/libero/libero/assets, or run scripts/setup_libero_plus.sh --download-assets."
            ) from exc
        raise

    if plus_root is None:
        init_states = _call_with_torch_load_compat(bm.get_task_init_states, task_id)
    else:
        init_states = _load_libero_plus_init_states(task, get_libero_path)

    if preserve_libero_plus_robot_init_qpos is None:
        preserve_libero_plus_robot_init_qpos = plus_root is not None

    env = RobosuiteRolloutEnv(
        env=raw_env,
        camera_names=camera_names,
        camera_height=int(camera_size),
        camera_width=int(camera_size),
        enable_render=True,
        camera_depths=bool(camera_depths),
        env_image_hflip=env_image_hflip,
        env_image_vflip=env_image_vflip,
        env_image_rotate180=env_image_rotate180,
        hdf5_hflip_fix=hdf5_hflip_fix,
        preserve_libero_plus_robot_init_qpos=bool(preserve_libero_plus_robot_init_qpos),
    )
    return env, task_name, init_states


def create_rollout_env_libero_isolated(
    *,
    worker_timeout_sec: float = 300.0,
    worker_rank: int | None = None,
    **env_kwargs: Any,
):
    """Create a LIBERO rollout env owned by a spawned child process."""
    proxy = LiberoEnvWorkerProxy(
        env_kwargs=dict(env_kwargs),
        timeout_sec=float(worker_timeout_sec),
        rank=worker_rank,
        task_id=int(env_kwargs["task_id"]) if "task_id" in env_kwargs else None,
    )
    return proxy, proxy.task_name, proxy.init_states


def list_libero_task_metadata(suite_name: str, plus_root: str | os.PathLike[str] | None = None):
    """List task metadata in a LIBERO or LIBERO-Plus suite."""
    bm = _get_libero_benchmark(suite_name, plus_root=plus_root)
    get_libero_path = None
    if plus_root is not None:
        _, get_libero_path, _ = _import_libero_api(plus_root)
    return [_task_metadata(bm.get_task(i), i, get_libero_path=get_libero_path) for i in range(bm.n_tasks)]


def list_libero_tasks(suite_name: str, plus_root: str | os.PathLike[str] | None = None):
    """List all task names in a LIBERO suite."""
    return [str(item["language"]) for item in list_libero_task_metadata(suite_name, plus_root=plus_root)]
