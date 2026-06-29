"""In-training closed-loop LIBERO rollout eval.

Invoked by `train_robot.py` at `eval_every` cadence when
`closed_loop_eval.enabled=true` in the training config. Each training rank runs
a disjoint slice of the (suite, task, trial) grid; the caller all-reduces the
per-(suite, task) success/total counts across ranks and logs to wandb.

The policy is built by calling the same `load_stage1_policy()` used by the
stand-alone eval script, but with `preloaded_modules=` pointing at the live
training modules so we avoid re-loading 1.1B DA3 Giant weights from disk on
every eval call. Teacher (frozen) + LIBERO envs are cached across calls in the
`_EvalState` handle returned to the caller.

Env setup prerequisites (libero / robosuite / mujoco import, per-server
MUJOCO_GL backend) are validated by `validate_libero_env()` : call once at
training start.

See `docs/closed-loop-eval-env.md` for per-server env configuration.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import csv
import faulthandler
import gc
import hashlib
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _coerce_mapping(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    return dict(value)


def _flush_logger_handlers() -> None:
    """Best-effort flush so native aborts still leave the last rollout marker."""
    handlers = list(logger.handlers)
    if logger.propagate:
        handlers.extend(logging.getLogger().handlers)
    for handler in handlers:
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _cuda_visible_device_tokens() -> list[str]:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not visible:
        return []
    return [token.strip() for token in visible.split(",") if token.strip()]


def _rank_local_visible_cuda_token(rank: int | None = None) -> str | None:
    tokens = _cuda_visible_device_tokens()
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0]
    try:
        visible_idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else int(rank or 0)
    except Exception:  # noqa: BLE001
        visible_idx = int(rank or 0)
    return tokens[visible_idx % len(tokens)]


def _resolve_render_gpu_device_id(rank: int | None = None) -> int | None:
    """Return the robosuite EGL device id for in-training rollout envs."""
    if os.environ.get("MUJOCO_GL", "").strip().lower() != "egl":
        return None
    visible_token = _rank_local_visible_cuda_token(rank)
    if visible_token is not None:
        if not visible_token.isdigit():
            raise RuntimeError(
                "robosuite EGL requires numeric CUDA_VISIBLE_DEVICES tokens; "
                f"got CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        device_id = int(visible_token)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
        return device_id
    try:
        if torch.cuda.is_available():
            device_id = int(torch.cuda.current_device())
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
            return device_id
    except Exception:  # noqa: BLE001
        pass
    for name in ("LOCAL_RANK", "MUJOCO_EGL_DEVICE_ID"):
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            device_id = int(raw)
        except ValueError:
            continue
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
        return device_id
    if rank is not None:
        try:
            device_id = int(rank)
        except (TypeError, ValueError):
            return None
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
        return device_id
    return None


def _release_native_env_resources() -> None:
    """Best-effort release of native MuJoCo/EGL resources after env close."""
    try:
        from OpenGL import EGL

        EGL.eglReleaseThread()
    except Exception:  # noqa: BLE001
        pass
    try:
        gc.collect()
    except Exception:  # noqa: BLE001
        pass


def _safe_close_env(env: Any, *, rank: int, host: str, suite: str, task_id: int | None, reason: str) -> None:
    if env is None:
        return
    # Explicit ordered teardown of robosuite MjRenderContext BEFORE env.close().
    # Background: rank=2 of obj-H2 jobs (3414942 / 3414943 / 3415490) consistently
    # hangs in mujoco.mjr_readPixels (binding_utils.py:171) during the second
    # rollout after ENV_RECREATE. faulthandler-captured native stack confirms
    # the stuck thread sits inside readPixels.
    # Hypothesis (M1): robosuite's MjRenderContext relies on Python GC to call
    # MjrContext.free() and GLContext.free() via __del__. When env.close() runs
    # only `del self._render_context_offscreen` (binding_utils.py:1173), the
    # actual native EGL/MjR resources can persist until the next reference cycle
    # collection. If the new env builds its NVIDIA EGL device context before
    # that, the driver-internal state of the previous context is still live and
    # the subsequent mjr_readPixels deadlocks inside the driver. Forcing
    # con.free() + gl_ctx.free() then dropping the attribute reference makes
    # the teardown ordering deterministic and process-local before the next
    # GLContext/MjrContext pair is constructed.
    try:
        candidate_chains = (
            ("env",),
            ("base_env",),
            ("env", "env"),
            ("base_env", "env"),
        )
        for chain in candidate_chains:
            obj: Any = env
            for attr_name in chain:
                obj = getattr(obj, attr_name, None)
                if obj is None:
                    break
            if obj is None:
                continue
            sim = getattr(obj, "sim", None)
            if sim is None:
                continue
            ctx = getattr(sim, "_render_context_offscreen", None)
            if ctx is None:
                continue
            for free_attr in ("con", "gl_ctx"):
                inner = getattr(ctx, free_attr, None)
                if inner is None:
                    continue
                free_fn = getattr(inner, "free", None)
                if callable(free_fn):
                    try:
                        free_fn()
                    except Exception:  # noqa: BLE001
                        pass
            try:
                setattr(sim, "_render_context_offscreen", None)
            except Exception:  # noqa: BLE001
                pass
            break  # one chain is enough; stop after the first hit.
    except Exception:  # noqa: BLE001
        pass
    try:
        env.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[closed_loop_eval rank=%d host=%s] env.close failed suite=%s task=%s reason=%s: %s",
            rank,
            host,
            suite,
            "n/a" if task_id is None else int(task_id),
            reason,
            exc,
        )
    finally:
        _release_native_env_resources()


def _lazy_eval_libero_unified():
    """Lazy import to avoid circular dep.

    `eval_libero_unified.py` imports `DA3FineTuneModel` from `train_robot.py`
    which in turn imports this module. Resolve at call time.
    """
    import eval_libero_unified as _elu  # type: ignore
    return _elu


_MUJOCO_GLCONTEXT_PATCH_INSTALLED = False
_MJ_RENDER_PATCH_INSTALLED = False


def _install_robosuite_render_make_current_patch() -> None:
    """Monkey-patch robosuite MjRenderContext.{render, read_pixels} to force
    self.gl_ctx.make_current() + glFinish() before each call.

    Root cause for libero_object SIGABRT on CSCS GH200 + multi-rank EGL device
    path (jobs 3414942 / 3414943 / 3415490 / 3415548 all hang on rank-local
    mjr_readPixels at binding_utils.py:171 during the 2nd rollout after
    ENV_RECREATE; explicit MjrContext.free()/GLContext.free() refuted the
    cleanup-ordering hypothesis (M1):

    robosuite's MjRenderContext only calls gl_ctx.make_current() inside
    __init__ (binding_utils.py:79). Subsequent render() / read_pixels() rely
    on the EGL driver keeping the right context current on the calling
    thread. After env.close() + new env build, the NVIDIA EGL device path
    leaves the per-thread current context in an inconsistent state : the new
    MjRenderContext.gl_ctx is the active context object in Python while the
    driver-side current binding can still reference the destroyed previous
    context. The blocking glReadPixels then deadlocks waiting on the stale
    context's framebuffer.

    Fix: at every render() and read_pixels() call we re-bind this MjRender
    Context's own GLContext with make_current() and drain pending GPU work
    with glFinish() before the blocking host transfer. Cost is one extra
    EGL call + one full sync per render; negligible vs the rollout step
    cost. Patch is process-local and idempotent.
    """
    global _MJ_RENDER_PATCH_INSTALLED
    if _MJ_RENDER_PATCH_INSTALLED:
        return
    try:
        from robosuite.utils.binding_utils import MjRenderContext  # type: ignore
    except Exception:  # noqa: BLE001
        return

    _orig_render = MjRenderContext.render
    _orig_read_pixels = MjRenderContext.read_pixels

    def _safe_make_current(self) -> None:
        try:
            gl_ctx = getattr(self, "gl_ctx", None)
            if gl_ctx is not None and hasattr(gl_ctx, "make_current"):
                gl_ctx.make_current()
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

    gl_finish = _load_gl_finish()

    def _safe_gl_finish() -> None:
        if gl_finish is None:
            return
        try:
            gl_finish()
        except Exception:  # noqa: BLE001
            pass

    def _patched_render(self, *args, **kwargs):
        _safe_make_current(self)
        return _orig_render(self, *args, **kwargs)

    def _patched_read_pixels(self, *args, **kwargs):
        _safe_make_current(self)
        _safe_gl_finish()
        return _orig_read_pixels(self, *args, **kwargs)

    MjRenderContext.render = _patched_render
    MjRenderContext.read_pixels = _patched_read_pixels
    _MJ_RENDER_PATCH_INSTALLED = True
    try:
        logger.info(
            "[closed_loop_eval] robosuite MjRenderContext make_current+glFinish patch INSTALLED (pid=%d)",
            os.getpid(),
        )
        _flush_logger_handlers()
    except Exception:  # noqa: BLE001
        pass


def _restore_mujoco_gl_env(value: str | None) -> None:
    if value is None:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ["MUJOCO_GL"] = value


def _egl_device_display_available(preferred_device_id: int | None = None) -> bool:
    """Return True when robosuite's native EGL device path can initialize."""
    try:
        from mujoco.egl import egl_ext as EGL
    except Exception:  # noqa: BLE001
        return False
    try:
        all_devices = EGL.eglQueryDevicesEXT()
        if not all_devices:
            return False
        if preferred_device_id is not None and int(preferred_device_id) >= 0:
            device_idx = int(preferred_device_id)
        else:
            selected = os.environ.get("MUJOCO_EGL_DEVICE_ID") or _rank_local_visible_cuda_token()
            if selected is None:
                device_idx = 0
            elif str(selected).isdigit():
                device_idx = int(selected)
            else:
                device_idx = int(str(selected).split(",", 1)[0])
        if not 0 <= device_idx < len(all_devices):
            return False
        display = EGL.eglGetPlatformDisplayEXT(
            EGL.EGL_PLATFORM_DEVICE_EXT,
            all_devices[device_idx],
            None,
        )
        if display == EGL.EGL_NO_DISPLAY or EGL.eglGetError() != EGL.EGL_SUCCESS:
            return False
        try:
            initialized = EGL.eglInitialize(display, None, None)
        except Exception:  # noqa: BLE001
            return False
        ok = bool(initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS)
        if ok:
            try:
                EGL.eglTerminate(display)
                EGL.eglReleaseThread()
            except Exception:  # noqa: BLE001
                pass
        return ok
    except Exception:  # noqa: BLE001
        return False


def _install_mujoco_glcontext_patch(preferred_device_id: int | None = None) -> None:
    """Replace robosuite GLContext with a Mesa-safe EGL adapter when needed.

    CVLAB1 exposes `EGL_MESA_platform_surfaceless` while the NVIDIA
    `EGL_PLATFORM_DEVICE_EXT` path required by MuJoCo's stock `mujoco.egl`
    loader is absent. Importing `mujoco` directly under `MUJOCO_GL=egl`
    therefore dies before robosuite even gets a chance to install its own
    context.

    Fix: import the core MuJoCo bindings under a neutral backend first. If the
    native EGL device-display path is available, keep robosuite's own NVIDIA
    EGL context so `render_gpu_device_id` is honored. Only hosts without that
    path get the Mesa surfaceless fallback. Patch must run BEFORE any
    `MjRenderContext*` construction.
    """
    global _MUJOCO_GLCONTEXT_PATCH_INSTALLED
    if _MUJOCO_GLCONTEXT_PATCH_INSTALLED:
        return
    requested_backend = str(os.environ.get("MUJOCO_GL", "") or "egl").strip().lower()
    if not requested_backend:
        requested_backend = "egl"

    # Import core MuJoCo bindings without triggering the fragile stock
    # `mujoco.egl` device-display path on Mesa-backed EGL hosts.
    restore_backend = os.environ.get("MUJOCO_GL")
    if requested_backend == "egl":
        os.environ["MUJOCO_GL"] = ""
    elif not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = requested_backend

    import mujoco

    if requested_backend == "egl":
        if _egl_device_display_available(preferred_device_id):
            _restore_mujoco_gl_env(restore_backend)
            os.environ["DA3_LIBERO_GLCONTEXT_MODE"] = "native_egl_device"
            # Critical for CSCS GH200 + multi-rank EGL device path. Without this
            # rollout's 2nd ENV_RECREATE deadlocks inside mjr_readPixels because
            # the NVIDIA driver retains a per-thread current binding to the
            # destroyed previous GLContext. See _install_robosuite_render_make_current_patch
            # docstring for full root cause.
            _install_robosuite_render_make_current_patch()
            _MUJOCO_GLCONTEXT_PATCH_INSTALLED = True
            return

        import atexit
        import ctypes
        import types
        from OpenGL import EGL

        # Mesa advertises this client extension even when the NVIDIA
        # PLATFORM_DEVICE path is absent.
        EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
        EGL_ATTRIBUTES = (
            EGL.EGL_RED_SIZE, 8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE, 8,
            EGL.EGL_ALPHA_SIZE, 8,
            EGL.EGL_DEPTH_SIZE, 24,
            EGL.EGL_STENCIL_SIZE, 8,
            EGL.EGL_COLOR_BUFFER_TYPE, EGL.EGL_RGB_BUFFER,
            EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
            EGL.EGL_NONE,
        )

        egl_display = EGL.eglGetPlatformDisplayEXT(
            EGL_PLATFORM_SURFACELESS_MESA,
            EGL.EGL_DEFAULT_DISPLAY,
            None,
        )
        if egl_display == EGL.EGL_NO_DISPLAY or EGL.eglInitialize(egl_display, None, None) != EGL.EGL_TRUE:
            raise RuntimeError(
                "Failed to initialize Mesa surfaceless EGL display for LIBERO rollout. "
                "CVLAB1 should provide EGL_MESA_platform_surfaceless."
            )
        atexit.register(EGL.eglTerminate, egl_display)

        class _MesaSurfacelessEGLContext:
            def __init__(self, max_width=640, max_height=480, device_id=-1):
                del max_width, max_height, device_id
                num_configs = ctypes.c_long()
                config = EGL.EGLConfig()
                EGL.eglReleaseThread()
                EGL.eglChooseConfig(
                    egl_display,
                    EGL_ATTRIBUTES,
                    ctypes.byref(config),
                    1,
                    num_configs,
                )
                if num_configs.value < 1:
                    raise RuntimeError("Mesa surfaceless EGL failed to choose a framebuffer config.")
                EGL.eglBindAPI(EGL.EGL_OPENGL_API)
                self._context = EGL.eglCreateContext(
                    egl_display,
                    config,
                    EGL.EGL_NO_CONTEXT,
                    None,
                )
                if not self._context:
                    raise RuntimeError("Mesa surfaceless EGL failed to create an OpenGL context.")

            def make_current(self):
                ok = EGL.eglMakeCurrent(
                    egl_display,
                    EGL.EGL_NO_SURFACE,
                    EGL.EGL_NO_SURFACE,
                    self._context,
                )
                if not ok:
                    raise RuntimeError("Mesa surfaceless EGL failed to make the context current.")

            def free(self):
                if not self._context:
                    return
                current_context = EGL.eglGetCurrentContext()
                if current_context and self._context.address == current_context.address:
                    EGL.eglMakeCurrent(
                        egl_display,
                        EGL.EGL_NO_SURFACE,
                        EGL.EGL_NO_SURFACE,
                        EGL.EGL_NO_CONTEXT,
                    )
                EGL.eglDestroyContext(egl_display, self._context)
                EGL.eglReleaseThread()
                self._context = None

        injected_egl_context = types.ModuleType("robosuite.renderers.context.egl_context")
        injected_egl_context.EGLGLContext = _MesaSurfacelessEGLContext
        sys.modules["robosuite.renderers.context.egl_context"] = injected_egl_context

    _restore_mujoco_gl_env(restore_backend)
    import robosuite.utils.binding_utils as bu

    if requested_backend == "egl":
        class _MujocoGLContextAdapter(_MesaSurfacelessEGLContext):
            pass
        os.environ["DA3_LIBERO_GLCONTEXT_MODE"] = "mesa_surfaceless_fallback"
    else:
        class _MujocoGLContextAdapter:
            def __init__(self, max_width=640, max_height=480, device_id=-1):
                del device_id
                self._ctx = mujoco.GLContext(max_width, max_height)

            def make_current(self):
                self._ctx.make_current()

            def free(self):
                self._ctx.free()

    bu.GLContext = _MujocoGLContextAdapter
    _MUJOCO_GLCONTEXT_PATCH_INSTALLED = True


# --------------------------------------------------------------------------
# Env validation (call once per training rank before long training)
# --------------------------------------------------------------------------


def validate_libero_env() -> None:
    """Sanity-check LIBERO/robosuite/mujoco import + 1-frame render.

    Raises RuntimeError with a pointer to `docs/closed-loop-eval-env.md` if
    anything is missing.

    Fail-fast philosophy: surface env issues immediately at training start
    rather than hours later at the first `eval_every` tick.
    """
    try:
        import libero  # noqa: F401
        import robosuite  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"closed_loop_eval requires libero/robosuite/mujoco packages. "
            f"Import failed: {e}. "
            "See docs/closed-loop-eval-env.md for per-server install guide."
        ) from e

    mujoco_gl = os.environ.get("MUJOCO_GL", "")
    if not mujoco_gl:
        raise RuntimeError(
            "MUJOCO_GL environment variable is not set. LIBERO requires one of "
            "{glx, egl, osmesa}. See docs/closed-loop-eval-env.md for per-server "
            "defaults (CVLAB1=egl, CSCS=egl, Elice=osmesa)."
        )

    render_gpu_device_id = _resolve_render_gpu_device_id(
        int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)) or 0)
    )
    # Apply the Mesa GLContext adapter only on hosts where robosuite's own EGL
    # device path can't initialize. Idempotent : second call is a no-op.
    _install_mujoco_glcontext_patch(render_gpu_device_id)

    _elu = _lazy_eval_libero_unified()
    # Smoke-test: construct a tiny env via the same factory used at rollout
    # time. Skip env.reset() here because it
    # requires a prior `set_init_state(...)` for some LIBERO suites and errors
    # with "Current sensor for observable ... is invalid" when called cold.
    # The real rollout always calls apply_initial_state first so skipping
    # reset in the validation path is safe.
    try:
        env, _task_name, init_states = _elu.create_rollout_env_libero(
            suite_name="libero_spatial",
            task_id=0,
            camera_names=_elu.LIBERO_CAMERA_NAMES,
            camera_size=128,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=20,
            plus_root=None,
            camera_depths=False,
        )
        if len(init_states) == 0:
            raise RuntimeError("LIBERO env produced zero init states : benchmark metadata broken")
        _safe_close_env(
            env,
            rank=int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)) or 0),
            host=socket.gethostname(),
            suite="libero_spatial",
            task_id=0,
            reason="validation",
        )
        del env, init_states
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"LIBERO env smoke-test failed with MUJOCO_GL={mujoco_gl!r}: "
            f"{type(e).__name__}: {e}. See docs/closed-loop-eval-env.md."
        ) from e

    logger.info(
        "[closed_loop_eval] validated LIBERO env OK (MUJOCO_GL=%s)",
        mujoco_gl,
    )


# --------------------------------------------------------------------------
# Eval state (persistent across eval calls)
# --------------------------------------------------------------------------


@dataclass
class _EvalState:
    policy: Any | None = None
    policy_info: dict[str, Any] = field(default_factory=dict)
    envs: dict[str, Any] = field(default_factory=dict)  # {suite: env}
    # Hash of module pointers. If the policy is ever rebuilt because modules
    # changed during training, reset envs too as a defensive guard.
    config_hash: str | None = None


_GLOBAL_EVAL_STATE = _EvalState()


def _hash_modules(*modules: Any) -> str:
    h = hashlib.sha1()
    for m in modules:
        h.update(str(id(m)).encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------
# Work partitioning
# --------------------------------------------------------------------------


def _build_work_tuples(
    suites: list[str],
    num_tasks_per_suite: int | str,
    num_trials_per_task: int,
    plus_root: str | None = None,
    plus_perturbation: str = "all",
    plus_official_category: str = "all",
    plus_subset: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate per-rollout work items with metadata attached."""
    _elu = _lazy_eval_libero_unified()
    tuples: list[dict[str, Any]] = []
    plus_classification = (
        _elu.load_libero_plus_task_classification(plus_root)
        if plus_root
        else {"loaded": False}
    )
    plus_subset = _coerce_mapping(plus_subset)
    for suite in suites:
        task_meta = [dict(item) for item in _elu.list_libero_task_metadata(suite, plus_root=plus_root)]
        if plus_root:
            for item in task_meta:
                item["plus_perturbation"] = _elu.classify_libero_plus_perturbation(item)
            _elu.annotate_libero_plus_official_categories(task_meta, suite, plus_classification)
            task_meta = _elu.filter_libero_plus_task_metadata(task_meta, plus_perturbation)
            task_meta = _elu.filter_libero_plus_official_category_metadata(task_meta, plus_official_category)
            task_meta, _manifest = _elu.select_libero_plus_task_subset(
                task_meta,
                group_by=str(plus_subset.get("group_by", "none")),
                samples_per_group=int(plus_subset.get("samples_per_group", 0) or 0),
                sample_seed=int(plus_subset.get("sample_seed", 0) or 0),
                suite=suite,
            )
        n_tasks = len(task_meta)
        if isinstance(num_tasks_per_suite, str):
            if num_tasks_per_suite.lower() == "all":
                sel = list(range(n_tasks))
            else:
                raise ValueError(
                    f"num_tasks_per_suite={num_tasks_per_suite!r} must be int or 'all'"
                )
        else:
            sel = list(range(min(int(num_tasks_per_suite), n_tasks)))
        for eval_idx in sel:
            entry = dict(task_meta[eval_idx])
            task_id = int(entry.get("task_id", eval_idx))
            count_task_id = int(eval_idx) if plus_root else task_id
            for trial in range(int(num_trials_per_task)):
                tuples.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "count_task_id": count_task_id,
                        "eval_task_index": int(eval_idx),
                        "trial_idx": int(trial),
                        "task_name": str(entry.get("task_name") or entry.get("name") or ""),
                        "plus_perturbation": str(entry.get("plus_perturbation") or ""),
                        "plus_official_task_id": entry.get("plus_official_task_id", ""),
                        "plus_official_category": str(entry.get("plus_official_category") or ""),
                        "plus_official_category_slug": str(entry.get("plus_official_category_slug") or ""),
                        "plus_official_difficulty_level": str(
                            entry.get("plus_official_difficulty_level") or ""
                        ),
                        "raw_task_language": str(entry.get("language") or ""),
                        "policy_language": str(entry.get("policy_language") or entry.get("language") or ""),
                        "bddl_file": str(entry.get("bddl_file") or ""),
                        "init_states_file": str(entry.get("init_states_file") or ""),
                        "task_desc": str(entry.get("policy_language") or entry.get("language") or ""),
                    }
                )
    return tuples


def _assign_work_tuples(
    work: list[dict[str, Any]],
    *,
    active_rank_index: int,
    num_active_ranks: int,
    assignment: str,
) -> list[dict[str, Any]]:
    """Assign rollout work to active ranks.

    ``round_robin`` is the historical vla-eval-like work-item sharding:
    adjacent trials of the same task are spread across ranks. That is well
    balanced, but it recreates LIBERO envs frequently in short in-training
    monitors. ``task_round_robin`` keeps all trials for a task on the same rank,
    usually making many-eval sweeps faster by amortizing env construction.
    """
    n = max(1, int(num_active_ranks))
    idx = int(active_rank_index)
    mode = str(assignment or "round_robin").strip().lower().replace("-", "_")
    if mode in {"round_robin", "episode_round_robin", "work_round_robin"}:
        return [item for i, item in enumerate(work) if (i % n) == idx]
    if mode in {"task_round_robin", "task", "by_task"}:
        task_order: dict[tuple[str, int, int], int] = {}
        assigned: list[dict[str, Any]] = []
        for item in work:
            key = (
                str(item.get("suite", "")),
                int(item.get("eval_task_index", item.get("task_id", 0))),
                int(item.get("count_task_id", item.get("task_id", 0))),
            )
            if key not in task_order:
                task_order[key] = len(task_order)
            if (task_order[key] % n) == idx:
                assigned.append(item)
        return assigned
    raise ValueError(
        "closed_loop_eval.work_assignment must be round_robin or task_round_robin, "
        f"got {assignment!r}."
    )


def _isolated_env_cache_key(
    *,
    benchmark: str,
    suite: str,
    task_id: int,
    env_horizon: int,
    plus_root: str | None,
) -> str:
    """Stable key for per-task isolated LIBERO env reuse within one eval profile."""
    plus_sig = hashlib.sha1(str(plus_root or "").encode("utf-8")).hexdigest()[:8]
    return (
        f"isolated|benchmark={benchmark}|suite={suite}|task={int(task_id)}|"
        f"horizon={int(env_horizon)}|plus={plus_sig}"
    )


# --------------------------------------------------------------------------
# Main eval entry
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate_closed_loop_libero_from_training(
    *,
    cfg: Any,
    closed_loop_cfg: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
    # Preloaded training modules (live, will be set to eval mode here; caller
    # restores train() after return).
    teacher_da3: Any,
    student_da3: Any,
    action_head: Any,
    future_predictor: Any | None,
    text_conditioner: Any | None,
    proprio_conditioner: Any | None,
    action_normalizer: Any,
    proprio_normalizer: Any | None,
    train_steps: int,
) -> tuple[dict[tuple[str, int], dict[str, int]], dict[tuple[str, int], list]]:
    """Run closed-loop LIBERO rollouts and return per-(suite, task_id) counts.

    Returns: (local_counts, local_videos)
        local_counts: {(suite, task_id): {"success": int, "total": int, "steps": int}}
            counts for ONLY this rank's share. Caller all-reduces across ranks.
        local_videos: {(suite, task_id): list[np.ndarray]} : first-trial RGB
            frame buffers, populated only when `log_video=true` and only on
            rank 0 (other ranks return empty dict). Caller is expected to
            log to wandb directly without all-gathering.
    """
    if not closed_loop_cfg.get("enabled", False):
        return {}, {}

    active_eval_ranks_raw = closed_loop_cfg.get("eval_num_active_ranks", None)
    if active_eval_ranks_raw is None:
        active_eval_ranks = int(world_size)
    else:
        active_eval_ranks = int(active_eval_ranks_raw)
        if active_eval_ranks <= 0:
            raise ValueError(
                f"closed_loop_eval.eval_num_active_ranks must be positive, got {active_eval_ranks_raw!r}."
            )
        active_eval_ranks = min(active_eval_ranks, int(world_size))
    if rank >= active_eval_ranks:
        logger.info(
            "[closed_loop_eval rank=%d host=%s] inactive for this profile "
            "(eval_num_active_ranks=%d, world_size=%d); contributing zero counts",
            rank, socket.gethostname(), active_eval_ranks, world_size,
        )
        _flush_logger_handlers()
        return {}, {}

    # Diagnostic: capture native abort (SIGABRT/SIGSEGV/SIGFPE/SIGBUS/SIGILL)
    # stack traces from C-level (mujoco/EGL/CUDA) inside this rank's eval.
    # This preserves the native crash site before NCCL-watchdog noise dominates.
    try:
        import signal as _signal_diag
        if not faulthandler.is_enabled():
            faulthandler.enable(file=sys.stderr, all_threads=True)
        # SIGUSR2: external diagnostic dump (kill -USR2 <pid>).
        try:
            faulthandler.register(
                _signal_diag.SIGUSR2,
                file=sys.stderr,
                all_threads=True,
                chain=False,
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    render_gpu_device_id = _resolve_render_gpu_device_id(rank)
    # All ranks need the GLContext selection since each rank constructs its own
    # rollout envs. validate_libero_env() only runs on rank 0.
    _install_mujoco_glcontext_patch(render_gpu_device_id)
    logger.info(
        "[closed_loop_eval rank=%d host=%s] render_gpu_device_id=%s MUJOCO_EGL_DEVICE_ID=%s glcontext=%s",
        rank,
        socket.gethostname(),
        "default" if render_gpu_device_id is None else int(render_gpu_device_id),
        os.environ.get("MUJOCO_EGL_DEVICE_ID", "<unset>"),
        os.environ.get("DA3_LIBERO_GLCONTEXT_MODE", "default"),
    )
    _flush_logger_handlers()

    benchmark = str(closed_loop_cfg.get("benchmark", "libero") or "libero").strip().lower().replace("-", "_")
    plus_root = None
    if benchmark in {"libero_plus", "plus"}:
        plus_root = str(closed_loop_cfg.get("plus_root") or os.environ.get("DA3_LIBERO_PLUS_DIR") or "")
        if not plus_root:
            raise ValueError("closed_loop_eval benchmark=libero_plus requires plus_root or DA3_LIBERO_PLUS_DIR.")
    elif benchmark not in {"libero", "plain_libero"}:
        raise ValueError(
            f"Unsupported closed_loop_eval benchmark={benchmark!r}; expected libero or libero_plus."
        )
    plus_perturbation = str(closed_loop_cfg.get("plus_perturbation", "all"))
    plus_official_category = str(closed_loop_cfg.get("plus_official_category", "all"))
    plus_subset = _coerce_mapping(closed_loop_cfg.get("plus_subset", {}) or {})
    suites = list(closed_loop_cfg.get("suites", ["libero_spatial"]))
    num_tasks_per_suite = closed_loop_cfg.get("num_tasks_per_suite", "all")
    num_trials_per_task = int(closed_loop_cfg.get("num_trials_per_task", 2))
    num_steps_wait = int(closed_loop_cfg.get("num_steps_wait", 10))
    action_horizon_req = str(closed_loop_cfg.get("action_horizon", 1))
    action_repeat_req = str(closed_loop_cfg.get("action_repeat", 1))
    log_video = bool(closed_loop_cfg.get("log_video", False))
    video_max_per_eval = int(closed_loop_cfg.get("video_max_per_eval", 8))
    # When detailed_video=true, each frame is the policy debug grid
    # (raw obs / policy obs / predicted depth / predicted RGB / action chunks
    # + live env render). Requires the policy to populate `last_debug` :
    # which `load_stage1_policy` does for gam Path B. Heavier than
    # plain RGB rollout video; defaults to off for backward compatibility.
    detailed_video = bool(closed_loop_cfg.get("detailed_video", False))
    action_repeat_mode = str(closed_loop_cfg.get("action_repeat_mode", "split_delta"))
    execute_chunk_prefix = int(closed_loop_cfg.get("execute_chunk_prefix", 0) or 0)
    partial_chunk_history = str(closed_loop_cfg.get("partial_chunk_history", "default"))
    warmup_full_chunk_once = bool(closed_loop_cfg.get("warmup_full_chunk_once", False))
    if execute_chunk_prefix < 0:
        raise ValueError(f"closed_loop_eval.execute_chunk_prefix must be >= 0, got {execute_chunk_prefix}.")
    if execute_chunk_prefix > 0 and partial_chunk_history == "default":
        partial_chunk_history = "rolling_last_k"
    env_control_hz = float(closed_loop_cfg.get("env_control_hz", 20.0))
    policy_hz_override = float(closed_loop_cfg.get("policy_hz", 20.0))
    camera_size = int(closed_loop_cfg.get("camera_size", 256))
    seed = int(closed_loop_cfg.get("seed", 7))
    env_seed = int(closed_loop_cfg.get("env_seed", 0))
    traceback_timeout_sec = float(
        closed_loop_cfg.get(
            "traceback_timeout_sec",
            closed_loop_cfg.get("rollout_traceback_timeout_sec", 300.0),
        )
    )
    rollout_wall_timeout_sec = float(
        closed_loop_cfg.get(
            "rollout_wall_timeout_sec",
            closed_loop_cfg.get("rollout_timeout_sec", 240.0),
        )
    )
    cleanup_env_cache_after_eval = bool(closed_loop_cfg.get("cleanup_env_cache_after_eval", True))
    env_process_isolation = bool(closed_loop_cfg.get("env_process_isolation", True))
    cache_isolated_envs = bool(closed_loop_cfg.get("cache_isolated_envs", False))
    env_worker_timeout_sec = float(closed_loop_cfg.get("env_worker_timeout_sec", 300.0))
    # history_horizon controls the predictor's observed-history window size
    # at rollout. Default 3 is empirically optimal: H ablation on goal 50k ckpt
    # (2026-04-24) gave H=3 → 20% / H=5 → 18% / H=7 → 14% (stronger than the
    # default H=auto=max(H_choices)=7). H=3 is also ~30% faster wall-clock than
    # H=7 because the predictor's KV cache + per-step attention scales with H.
    # Override to "auto" or any int in predictor.H_choices to A/B test.
    history_horizon = closed_loop_cfg.get("history_horizon", 3)
    rollout_decode_horizon = closed_loop_cfg.get("rollout_decode_horizon", "full")
    _elu = _lazy_eval_libero_unified()
    max_steps_by_suite = dict(closed_loop_cfg.get("max_steps_by_suite", _elu.OPENVLA_STEPS))

    # (1) Build / reuse policy closure.
    state = _GLOBAL_EVAL_STATE
    module_hash = _hash_modules(
        student_da3, action_head, future_predictor, proprio_conditioner,
        text_conditioner, action_normalizer, proprio_normalizer,
    )
    # Include rollout-shape knobs so changing them across eval calls forces a
    # rebuild; the policy closure captures both at construction.
    cache_key = f"{module_hash}|H={history_horizon}|D={rollout_decode_horizon}"
    if state.policy is None or state.config_hash != cache_key:
        # Ensure modules are in eval mode for rollout.
        student_da3.eval()
        action_head.eval()
        if future_predictor is not None:
            future_predictor.eval()
        if text_conditioner is not None:
            text_conditioner.eval()
        if proprio_conditioner is not None:
            proprio_conditioner.eval()

        preloaded = {
            "teacher": teacher_da3,
            "student": student_da3,
            "action_head": action_head,
            "proprio_conditioner": proprio_conditioner,
            "future_predictor": future_predictor,
            "text_conditioner": text_conditioner,
            "action_normalizer": action_normalizer,
            "proprio_normalizer": proprio_normalizer,
        }
        # Minimal dummy ckpt for metadata surfaces that peek at top-level keys.
        # Actual weights are already in the passed modules.
        dummy_ckpt: dict[str, Any] = {
            "config": cfg,
            "train_steps": int(train_steps),
            # Mark optional module presence so load_stage1_policy preloaded checks pass.
            "future_predictor": (
                {"__preloaded__": True} if future_predictor is not None else None
            ),
            "text_conditioner_proj": (
                {"__preloaded__": True} if text_conditioner is not None else None
            ),
            "proprio_conditioner": (
                {"__preloaded__": True} if proprio_conditioner is not None else None
            ),
            # student_da3 / action_head keys are required by an internal
            # assertion; content is unused because preloaded modules skip load.
            "student_da3": {"__preloaded__": True},
            "action_head": {"__preloaded__": True},
            # action_normalizer/proprio_normalizer are supplied by preloaded modules.
            "action_normalizer": None,
            "proprio_normalizer": None,
        }
        policy, info = _elu.load_stage1_policy(
            cfg=cfg,
            ckpt=dummy_ckpt,
            ckpt_path="<in-training-closed-loop>",
            device=device,
            stats_key=None,
            action_stats_json=None,
            decode_visuals=False,
            history_horizon=history_horizon,
            rollout_decode_horizon=rollout_decode_horizon,
            rotate_policy_input=False,
            config_source="training",
            preloaded_modules=preloaded,
        )
        state.policy = policy
        state.policy_info = info
        state.config_hash = cache_key

    policy = state.policy
    policy_info = dict(state.policy_info)
    rollout_action_frame = _elu.resolve_rollout_action_frame(
        closed_loop_cfg.get("rollout_action_frame", closed_loop_cfg.get("action_frame", "auto")),
        policy_info,
    )
    policy_info["rollout_action_frame"] = rollout_action_frame
    state.policy_info = policy_info
    env_image_hflip = bool(
        policy_info.get(
            "libero_hdf5_env_hflip",
            policy_info.get("libero_hdf5_env_rotate180", policy_info.get("libero_hdf5_env_hflip_fix", False)),
        )
    )
    env_image_rotate180 = bool(
        policy_info.get(
            "libero_hdf5_env_rotate180",
            policy_info.get("libero_hdf5_env_hflip_fix", False),
        )
    )
    if rank == 0:
        logger.info(
            "[closed_loop_eval] policy_preprocess=%s eval_crop=%s env_image_hflip=%s camera_size=%s stats_key=%s text_encoder=%s proprio_orientation=%s action_frame=%s rollout_action_frame=%s rollout_decode_horizon=%s mode=%s",
            policy_info.get("policy_image_preprocess"),
            policy_info.get("eval_crop_scale"),
            env_image_hflip,
            camera_size,
            policy_info.get("action_stats_key"),
            policy_info.get("text_encoder_type", "n/a"),
            policy_info.get("proprio_orientation", "n/a"),
            policy_info.get("action_frame", "base"),
            rollout_action_frame,
            policy_info.get("stage1_rollout_decode_horizon", "n/a"),
            policy_info.get("stage1_rollout_decode_horizon_mode", "n/a"),
        )

    # Set the per-call action_horizon on the policy closure (same contract as
    # the standalone eval script).
    resolved_h, _h_req = _elu.resolve_action_horizon(
        action_horizon_req, preset=_make_dummy_preset(), policy_info=policy_info,
    )
    policy.active_action_horizon = int(resolved_h)

    # (2) Enumerate + partition work.
    work = _build_work_tuples(
        suites=suites,
        num_tasks_per_suite=num_tasks_per_suite,
        num_trials_per_task=num_trials_per_task,
        plus_root=plus_root,
        plus_perturbation=plus_perturbation,
        plus_official_category=plus_official_category,
        plus_subset=plus_subset,
    )
    # Deterministic partition over active eval ranks. Training may still use
    # more ranks/nodes; inactive ranks skip simulator work and later
    # participate in the caller's distributed zero-count aggregation.
    default_work_assignment = (
        "task_round_robin"
        if (env_process_isolation and cache_isolated_envs)
        else "round_robin"
    )
    work_assignment = str(closed_loop_cfg.get("work_assignment", default_work_assignment))
    my_work = _assign_work_tuples(
        work,
        active_rank_index=int(rank),
        num_active_ranks=int(active_eval_ranks),
        assignment=work_assignment,
    )
    host = socket.gethostname()
    touched_env_keys: set[str] = set()
    assignment = ", ".join(
        f"{item['suite']}:task{int(item['task_id'])}:eval{int(item['eval_task_index'])}:trial{int(item['trial_idx'])}"
        for item in my_work
    )
    logger.info(
        "[closed_loop_eval rank=%d host=%s] assigned %d/%d rollouts across %d active eval ranks "
        "work_assignment=%s: %s",
        rank, host, len(my_work), len(work), active_eval_ranks, work_assignment, assignment or "<none>",
    )
    logger.info(
        "[closed_loop_eval rank=%d host=%s] env_process_isolation=%s env_worker_timeout_sec=%.1f "
        "cache_isolated_envs=%s cleanup_env_cache_after_eval=%s",
        rank,
        host,
        int(env_process_isolation),
        env_worker_timeout_sec,
        int(cache_isolated_envs),
        int(cleanup_env_cache_after_eval),
    )
    _flush_logger_handlers()

    # (3) For each local item, run a rollout.
    local_counts: dict[tuple[str, int], dict[str, int]] = {}
    # Video buffers populated only on rank 0 when log_video is on. Each entry
    # holds the first-trial RGB frame list for that (suite, task_id). Caller
    # logs to wandb directly without all-gathering.
    local_videos: dict[tuple[str, int], list] = {}
    t_eval_start = time.time()
    for i_local, item in enumerate(my_work):
        suite = str(item["suite"])
        task_id = int(item["task_id"])
        count_task_id = int(item.get("count_task_id", task_id))
        _eval_idx = int(item["eval_task_index"])
        trial_idx = int(item["trial_idx"])
        task_desc = str(item.get("task_desc") or "")
        max_steps = int(max_steps_by_suite.get(suite, 220))
        env_horizon = int(max_steps + max(0, num_steps_wait))
        logger.info(
            "[closed_loop_eval rank=%d host=%s] BEGIN %d/%d suite=%s task=%d eval_idx=%d trial=%d",
            rank, host, i_local + 1, len(my_work), suite, task_id, _eval_idx, trial_idx,
        )
        _flush_logger_handlers()
        env_kwargs = {
            "suite_name": suite,
            "task_id": task_id,
            "camera_names": _elu.LIBERO_CAMERA_NAMES,
            "camera_size": camera_size,
            "render_gpu_device_id": render_gpu_device_id,
            "control_freq": env_control_hz,
            "horizon": env_horizon,
            "plus_root": plus_root,
            "camera_depths": False,
            "env_image_hflip": env_image_hflip,
            "env_image_rotate180": env_image_rotate180,
        }
        env = None
        env_owned_by_rollout = False
        env_cache_key: str | None = None
        watchdog_enabled = False
        # Cache env per benchmark/suite only in non-isolated mode so plain
        # LIBERO and LIBERO-Plus profiles never share a simulator instance.
        env_key = f"{benchmark}|{suite}"
        try:
            if env_process_isolation:
                env_cache_key = _isolated_env_cache_key(
                    benchmark=benchmark,
                    suite=suite,
                    task_id=task_id,
                    env_horizon=env_horizon,
                    plus_root=plus_root,
                )
                env_entry = state.envs.get(env_cache_key) if cache_isolated_envs else None
                if env_entry is not None:
                    env = env_entry["env"]
                    init_states_by_task = env_entry["init_states_by_task"]
                    touched_env_keys.add(env_cache_key)
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_WORKER_REUSE suite=%s task=%d "
                        "pid=%s horizon=%d",
                        rank,
                        host,
                        suite,
                        task_id,
                        getattr(env, "pid", "n/a"),
                        env_horizon,
                    )
                    _flush_logger_handlers()
                else:
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_WORKER_CREATE_START suite=%s task=%d horizon=%d",
                        rank, host, suite, task_id, env_horizon,
                    )
                    _flush_logger_handlers()
                    env, _task_name, init_states = _elu.create_rollout_env_libero_isolated(
                        **env_kwargs,
                        worker_timeout_sec=env_worker_timeout_sec,
                        worker_rank=rank,
                    )
                    init_states_by_task = {task_id: init_states}
                    if cache_isolated_envs:
                        state.envs[env_cache_key] = {
                            "env": env,
                            "init_states_by_task": init_states_by_task,
                            "horizon": env_horizon,
                            "suite": suite,
                            "task_id": task_id,
                            "isolated": True,
                        }
                        touched_env_keys.add(env_cache_key)
                        env_owned_by_rollout = False
                    else:
                        env_owned_by_rollout = True
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_WORKER_CREATE_DONE suite=%s task=%d "
                        "pid=%s init_states=%d horizon=%d cached=%s",
                        rank,
                        host,
                        suite,
                        task_id,
                        getattr(env, "pid", "n/a"),
                        len(init_states),
                        env_horizon,
                        int(cache_isolated_envs),
                    )
                    _flush_logger_handlers()
            else:
                needs_new_env = env_key not in state.envs
                if not needs_new_env:
                    cached_horizon = int(state.envs[env_key].get("horizon", -1))
                    needs_new_env = cached_horizon != env_horizon
                if needs_new_env:
                    if env_key in state.envs:
                        old_entry = state.envs[env_key]
                        _safe_close_env(
                            old_entry.get("env"),
                            rank=rank,
                            host=host,
                            suite=str(old_entry.get("suite", suite)),
                            task_id=old_entry.get("task_id"),
                            reason="replace_env",
                        )
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_CREATE_START suite=%s task=%d horizon=%d",
                        rank, host, suite, task_id, env_horizon,
                    )
                    _flush_logger_handlers()
                    env, _task_name, init_states = _elu.create_rollout_env_libero(**env_kwargs)
                    state.envs[env_key] = {
                        "env": env,
                        "init_states_by_task": {task_id: init_states},
                        "horizon": env_horizon,
                        "suite": suite,
                        "task_id": task_id,
                    }
                    touched_env_keys.add(env_key)
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_CREATE_DONE suite=%s task=%d "
                        "init_states=%d horizon=%d",
                        rank, host, suite, task_id, len(init_states), env_horizon,
                    )
                    _flush_logger_handlers()
                env_entry = state.envs[env_key]
                touched_env_keys.add(env_key)
                env = env_entry["env"]
                init_states_by_task = env_entry["init_states_by_task"]
                if task_id not in init_states_by_task:
                    # Re-create env for the new task.
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_RECREATE_START suite=%s task=%d",
                        rank, host, suite, task_id,
                    )
                    _flush_logger_handlers()
                    _safe_close_env(
                        env,
                        rank=rank,
                        host=host,
                        suite=suite,
                        task_id=env_entry.get("task_id"),
                        reason="recreate_task",
                    )
                    env, _task_name, init_states = _elu.create_rollout_env_libero(**env_kwargs)
                    env_entry["env"] = env
                    env_entry["init_states_by_task"] = {task_id: init_states}
                    env_entry["horizon"] = env_horizon
                    env_entry["suite"] = suite
                    env_entry["task_id"] = task_id
                    touched_env_keys.add(env_key)
                    init_states_by_task = env_entry["init_states_by_task"]
                    logger.info(
                        "[closed_loop_eval rank=%d host=%s] ENV_RECREATE_DONE suite=%s task=%d "
                        "init_states=%d horizon=%d",
                        rank, host, suite, task_id, len(init_states), env_horizon,
                    )
                    _flush_logger_handlers()

            init_states = init_states_by_task[task_id]

            # Resolve action_repeat from env's control_freq + policy_hz.
            action_repeat, _policy_hz, _env_hz = _elu.resolve_action_repeat(
                action_repeat_req,
                policy_info=policy_info,
                env=env,
                policy_hz_override=policy_hz_override,
            )
            init_idx = trial_idx % max(1, len(init_states))
            _elu.seed_env(env, env_seed + trial_idx)

            # Pull task description from the already filtered metadata entry.
            if not task_desc:
                task_meta = _elu.list_libero_task_metadata(suite, plus_root=plus_root)
                task_desc = str(task_meta[_eval_idx].get("language", ""))

            # Record video only on rank 0, only for the first trial of each
            # (suite, task), and only until video_max_per_eval is reached. This
            # keeps render overhead negligible (one render per recorded rollout)
            # and gives the user a small representative reel per eval call.
            # Other ranks always pass record_video=False.
            record_this_video = bool(
                log_video
                and rank == 0
                and trial_idx == 0
                and (suite, int(task_id)) not in local_videos
                and len(local_videos) < video_max_per_eval
            )
            # detailed_video=True produces the policy debug grid (raw obs +
            # policy obs + predicted depth + predicted RGB + action chunks +
            # live env render). When both flags are True, rollout_episode picks
            # detailed over plain RGB. We only set them when this rollout is the
            # chosen video target on rank 0.
            logger.info(
                "[closed_loop_eval rank=%d host=%s] ROLLOUT_START suite=%s task=%d eval_idx=%d "
                "trial=%d init_idx=%d max_steps=%d action_horizon=%d action_repeat=%d "
                "execute_chunk_prefix=%d partial_chunk_history=%s warmup_full_chunk_once=%s "
                "record_video=%s detailed_video=%s",
                rank, host, suite, task_id, _eval_idx, trial_idx, init_idx, max_steps,
                int(resolved_h), int(action_repeat), int(execute_chunk_prefix),
                partial_chunk_history, bool(warmup_full_chunk_once), record_this_video,
                bool(detailed_video and record_this_video),
            )
            _flush_logger_handlers()
            watchdog_enabled = traceback_timeout_sec > 0
            t_rollout_start = time.time()
            if watchdog_enabled:
                faulthandler.dump_traceback_later(
                    max(1, int(traceback_timeout_sec)),
                    repeat=True,
                    file=sys.stderr,
                )
            result = _elu.rollout_episode(
                env=env,
                init_state=init_states[init_idx],
                policy=policy,
                max_steps=max_steps,
                action_horizon=int(resolved_h),
                action_repeat=int(action_repeat),
                action_repeat_mode=action_repeat_mode,
                num_steps_wait=num_steps_wait,
                camera_size=camera_size,
                record_video=record_this_video,
                detailed_video=bool(detailed_video and record_this_video),
                binarize_gripper=True,
                task_desc=task_desc,
                action_frame=rollout_action_frame,
                proprio_orientation=str(policy_info.get("proprio_orientation", "rpy")),
                execute_chunk_prefix=int(execute_chunk_prefix),
                partial_chunk_history=partial_chunk_history,
                warmup_full_chunk_once=bool(warmup_full_chunk_once),
                rollout_wall_timeout_sec=rollout_wall_timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[closed_loop_eval rank=%d] rollout/setup crashed on suite=%s task=%d trial=%d "
                "exc_type=%s exc_repr=%r",
                rank, suite, task_id, trial_idx, type(exc).__name__, exc,
            )
            if env_process_isolation and cache_isolated_envs and env_cache_key is not None:
                cached_entry = state.envs.pop(env_cache_key, None)
                cached_env = cached_entry.get("env") if cached_entry is not None else env
                if cached_env is not None:
                    _safe_close_env(
                        cached_env,
                        rank=rank,
                        host=host,
                        suite=suite,
                        task_id=task_id,
                        reason="isolated_cached_rollout_crash",
                    )
                env = None
            key = (suite, int(count_task_id))
            counts = local_counts.setdefault(
                key,
                {"success": 0, "total": 0, "steps": 0, "crashes": 0},
            )
            counts["total"] += 1
            counts["steps"] += int(max_steps)
            counts["crashes"] += 1
            _flush_logger_handlers()
            continue
        finally:
            if watchdog_enabled:
                faulthandler.cancel_dump_traceback_later()
            if env_owned_by_rollout and env is not None:
                _safe_close_env(
                    env,
                    rank=rank,
                    host=host,
                    suite=suite,
                    task_id=task_id,
                    reason="isolated_rollout",
                )
                env = None

        key = (suite, int(count_task_id))
        counts = local_counts.setdefault(key, {"success": 0, "total": 0, "steps": 0, "crashes": 0})
        counts["success"] += int(bool(result["success"]))
        counts["total"] += 1
        counts["steps"] += int(result["steps"])
        counts["crashes"] += int(bool(result.get("timeout", False)))
        logger.info(
            "[closed_loop_eval rank=%d host=%s] ROLLOUT_DONE suite=%s task=%d eval_idx=%d "
            "trial=%d success=%d steps=%d timeout=%s elapsed=%.1fs",
            rank, host, suite, task_id, _eval_idx, trial_idx,
            int(bool(result["success"])), int(result["steps"]),
            bool(result.get("timeout", False)), time.time() - t_rollout_start,
        )
        _flush_logger_handlers()

        if record_this_video:
            frames = result.get("frames") or []
            if frames:
                local_videos[key] = frames

    t_eval = time.time() - t_eval_start
    logger.info(
        "[closed_loop_eval rank=%d] ran %d rollouts in %.1fs (videos=%d)",
        rank, len(my_work), t_eval, len(local_videos),
    )
    if cleanup_env_cache_after_eval and touched_env_keys:
        for key in sorted(touched_env_keys):
            entry = state.envs.pop(key, None)
            if entry is None:
                continue
            _safe_close_env(
                entry.get("env"),
                rank=rank,
                host=host,
                suite=str(entry.get("suite", key)),
                task_id=entry.get("task_id"),
                reason="profile_cleanup",
            )
        logger.info(
            "[closed_loop_eval rank=%d] cleaned %d env cache entr%s after profile",
            rank,
            len(touched_env_keys),
            "y" if len(touched_env_keys) == 1 else "ies",
        )
        _flush_logger_handlers()
    return local_counts, local_videos


# Small throwaway preset object used only for action_horizon resolve. The
# resolver treats it as a carrier for `preset.action_horizon` and
# `preset.num_trials_per_task`, neither of which we use here. Lazy-built so the
# import of eval_libero_unified is deferred past module import time.
def _make_dummy_preset():
    _elu = _lazy_eval_libero_unified()
    return _elu.ProtocolPreset(
        name="in_training",
        num_trials_per_task=1,
        seed=0,
        env_seed=0,
        num_steps_wait=10,
        max_steps_by_suite=_elu.OPENVLA_STEPS,
        action_horizon=1,
    )


# --------------------------------------------------------------------------
# All-reduce helper (called by training loop on rank 0 caller)
# --------------------------------------------------------------------------


def all_reduce_counts(
    local_counts: dict[tuple[str, int], dict[str, int]],
    world_size: int,
    device: torch.device,
) -> dict[tuple[str, int], dict[str, int]]:
    """Sum local per-(suite, task_id) counts across all ranks.

    Each rank passes in its own local dict. Keys must be consistent across
    ranks (derived from the shared work list). Returns a dict with global
    counts (every rank receives the same dict after this call).

    Missing keys on a rank are implicitly zero : we union all keys first.
    """
    if not dist.is_available() or not dist.is_initialized() or world_size == 1:
        return {k: dict(v) for k, v in local_counts.items()}

    # Union keys across ranks via object_list all_gather. Keys are small.
    # (Using all_gather_object keeps the dict-of-tuples structure intact.)
    key_set = {k for k in local_counts.keys()}
    try:
        rank = dist.get_rank()
    except Exception:  # noqa: BLE001
        rank = -1
    logger.info(
        "[closed_loop_eval rank=%d] AGGREGATE_START local_keys=%s local_totals=%s",
        rank,
        sorted(key_set),
        {k: int(v.get("total", 0)) for k, v in sorted(local_counts.items())},
    )
    _flush_logger_handlers()
    all_keys: list[set] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(all_keys, key_set)
    logger.info("[closed_loop_eval rank=%d] AGGREGATE_KEYS_DONE", rank)
    _flush_logger_handlers()
    global_keys: set[tuple[str, int]] = set()
    for ks in all_keys:
        global_keys |= ks

    # Build flat tensor of [success, total, steps, crashes] per key (in sorted order so
    # all ranks agree on indexing).
    sorted_keys = sorted(global_keys)
    flat = torch.zeros(len(sorted_keys) * 4, dtype=torch.long, device=device)
    for i, k in enumerate(sorted_keys):
        c = local_counts.get(k)
        if c is not None:
            flat[i * 4 + 0] = int(c.get("success", 0))
            flat[i * 4 + 1] = int(c.get("total", 0))
            flat[i * 4 + 2] = int(c.get("steps", 0))
            flat[i * 4 + 3] = int(c.get("crashes", 0))
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    logger.info("[closed_loop_eval rank=%d] AGGREGATE_REDUCE_DONE global_keys=%s", rank, sorted(global_keys))
    _flush_logger_handlers()
    out: dict[tuple[str, int], dict[str, int]] = {}
    for i, k in enumerate(sorted_keys):
        s = int(flat[i * 4 + 0].item())
        t = int(flat[i * 4 + 1].item())
        st = int(flat[i * 4 + 2].item())
        cr = int(flat[i * 4 + 3].item())
        if t > 0:
            out[k] = {"success": s, "total": t, "steps": st, "crashes": cr}
    return out


def format_wandb_log(
    global_counts: dict[tuple[str, int], dict[str, int]],
    prefix: str = "rollout",
) -> dict[str, float]:
    """Flatten global per-(suite, task_id) counts into a wandb-ready dict."""
    log: dict[str, float] = {}
    suite_agg: dict[str, dict[str, int]] = {}
    for (suite, task_id), c in global_counts.items():
        sr = c["success"] / max(1, c["total"])
        mean_steps = c["steps"] / max(1, c["total"])
        log[f"{prefix}/{suite}/task{task_id}/success_rate"] = sr
        log[f"{prefix}/{suite}/task{task_id}/mean_steps"] = mean_steps
        log[f"{prefix}/{suite}/task{task_id}/num_trials"] = c["total"]
        log[f"{prefix}/{suite}/task{task_id}/num_crashes"] = c.get("crashes", 0)
        agg = suite_agg.setdefault(suite, {"success": 0, "total": 0, "crashes": 0})
        agg["success"] += c["success"]
        agg["total"] += c["total"]
        agg["crashes"] += c.get("crashes", 0)
    total_sr_num = 0
    total_sr_den = 0
    for suite, agg in suite_agg.items():
        if agg["total"] > 0:
            log[f"{prefix}/{suite}/success_rate"] = agg["success"] / agg["total"]
            log[f"{prefix}/{suite}/num_trials"] = agg["total"]
            log[f"{prefix}/{suite}/num_crashes"] = agg.get("crashes", 0)
            total_sr_num += agg["success"]
            total_sr_den += agg["total"]
    total_crashes = sum(agg.get("crashes", 0) for agg in suite_agg.values())
    if total_sr_den > 0:
        log[f"{prefix}/all/mean_sr"] = total_sr_num / total_sr_den
        log[f"{prefix}/all/num_trials"] = total_sr_den
        log[f"{prefix}/all/num_crashes"] = total_crashes
    return log


def get_cached_policy_info() -> dict[str, Any]:
    """Return a JSON-friendly copy of the currently cached rollout policy info."""
    return dict(_GLOBAL_EVAL_STATE.policy_info or {})


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as _np

        if isinstance(value, _np.generic):
            return value.item()
        if isinstance(value, _np.ndarray):
            return value.tolist()
    except Exception:  # noqa: BLE001
        pass
    try:
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def write_eval_artifacts(
    *,
    global_counts: dict[tuple[str, int], dict[str, int]],
    closed_loop_cfg: dict[str, Any],
    experiment_dir: str | os.PathLike[str],
    profile_name: str,
    train_steps: int,
    prefix: str,
    policy_info: dict[str, Any] | None = None,
    world_size: int | None = None,
) -> dict[str, str]:
    """Persist rank-0 train-time LIBERO rollout artifacts for standalone diffing.

    The training loop still owns distributed aggregation. This helper only
    rebuilds the deterministic work manifest and joins it with the already
    reduced success counts.
    """
    if not global_counts:
        return {}
    if bool(closed_loop_cfg.get("write_artifacts", True)) is False:
        return {}

    benchmark = str(closed_loop_cfg.get("benchmark", "libero") or "libero").strip().lower().replace("-", "_")
    plus_root = None
    if benchmark in {"libero_plus", "plus"}:
        plus_root = str(closed_loop_cfg.get("plus_root") or os.environ.get("DA3_LIBERO_PLUS_DIR") or "")
    plus_perturbation = str(closed_loop_cfg.get("plus_perturbation", "all"))
    plus_official_category = str(closed_loop_cfg.get("plus_official_category", "all"))
    plus_subset = _coerce_mapping(closed_loop_cfg.get("plus_subset", {}) or {})
    suites = list(closed_loop_cfg.get("suites", ["libero_spatial"]))
    num_tasks_per_suite = closed_loop_cfg.get("num_tasks_per_suite", "all")
    num_trials_per_task = int(closed_loop_cfg.get("num_trials_per_task", 2))

    work = _build_work_tuples(
        suites=suites,
        num_tasks_per_suite=num_tasks_per_suite,
        num_trials_per_task=num_trials_per_task,
        plus_root=plus_root,
        plus_perturbation=plus_perturbation,
        plus_official_category=plus_official_category,
        plus_subset=plus_subset,
    )
    metadata_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in work:
        key = (str(item["suite"]), int(item.get("count_task_id", item["task_id"])))
        metadata_by_key.setdefault(key, item)

    out_dir = (
        Path(experiment_dir)
        / "closed_loop_eval_results"
        / str(profile_name)
        / f"step_{int(train_steps):07d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "suite",
        "task_id",
        "eval_task_index",
        "task_name",
        "plus_perturbation",
        "plus_official_task_id",
        "plus_official_category",
        "plus_official_category_slug",
        "plus_official_difficulty_level",
        "raw_task_language",
        "policy_language",
        "bddl_file",
        "init_states_file",
        "num_trials",
        "num_success",
        "num_crashes",
        "success_rate",
        "avg_steps",
    ]
    rows: list[dict[str, Any]] = []
    suite_results: dict[str, dict[str, Any]] = {}
    category_results: dict[str, dict[str, Any]] = {}
    for key, counts in sorted(global_counts.items()):
        suite, count_task_id = key
        meta = metadata_by_key.get(key, {})
        total = int(counts.get("total", 0))
        success = int(counts.get("success", 0))
        steps = int(counts.get("steps", 0))
        crashes = int(counts.get("crashes", 0))
        sr = success / max(1, total)
        avg_steps = steps / max(1, total)
        row = {
            "suite": suite,
            "task_id": int(meta.get("task_id", count_task_id)),
            "eval_task_index": int(meta.get("eval_task_index", count_task_id)),
            "task_name": meta.get("task_name", ""),
            "plus_perturbation": meta.get("plus_perturbation", ""),
            "plus_official_task_id": meta.get("plus_official_task_id", ""),
            "plus_official_category": meta.get("plus_official_category", ""),
            "plus_official_category_slug": meta.get("plus_official_category_slug", ""),
            "plus_official_difficulty_level": meta.get("plus_official_difficulty_level", ""),
            "raw_task_language": meta.get("raw_task_language", ""),
            "policy_language": meta.get("policy_language", meta.get("task_desc", "")),
            "bddl_file": meta.get("bddl_file", ""),
            "init_states_file": meta.get("init_states_file", ""),
            "num_trials": total,
            "num_success": success,
            "num_crashes": crashes,
            "success_rate": sr,
            "avg_steps": avg_steps,
        }
        rows.append(row)

        suite_agg = suite_results.setdefault(
            suite,
            {
                "num_success": 0,
                "num_trials": 0,
                "num_crashes": 0,
                "num_tasks": 0,
                "success_rate": 0.0,
            },
        )
        suite_agg["num_success"] += success
        suite_agg["num_trials"] += total
        suite_agg["num_crashes"] += crashes
        suite_agg["num_tasks"] += 1

        category_slug = str(row.get("plus_official_category_slug") or "")
        if category_slug:
            category_name = str(row.get("plus_official_category") or category_slug)
            cat_agg = category_results.setdefault(
                category_slug,
                {
                    "category": category_name,
                    "num_success": 0,
                    "num_trials": 0,
                    "num_crashes": 0,
                    "num_tasks": 0,
                    "success_rate": 0.0,
                },
            )
            cat_agg["num_success"] += success
            cat_agg["num_trials"] += total
            cat_agg["num_crashes"] += crashes
            cat_agg["num_tasks"] += 1

    for agg in suite_results.values():
        agg["success_rate"] = agg["num_success"] / max(1, agg["num_trials"])
    for agg in category_results.values():
        agg["success_rate"] = agg["num_success"] / max(1, agg["num_trials"])

    total_success = sum(int(c.get("success", 0)) for c in global_counts.values())
    total_episodes = sum(int(c.get("total", 0)) for c in global_counts.values())
    total_crashes = sum(int(c.get("crashes", 0)) for c in global_counts.values())

    per_task_path = out_dir / "per_task.csv"
    with per_task_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k, "")) for k in fieldnames})

    manifest_by_category: dict[str, list[int]] = {}
    for row in rows:
        cat = str(row.get("plus_official_category_slug") or "")
        if cat:
            manifest_by_category.setdefault(cat, []).append(int(row["task_id"]))

    policy = _json_safe(policy_info if policy_info is not None else get_cached_policy_info())
    summary = {
        "run_name": f"{profile_name}-step{int(train_steps):07d}",
        "source": "in_training_closed_loop",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_name": str(profile_name),
        "prefix": str(prefix),
        "train_steps": int(train_steps),
        "benchmark": benchmark,
        "suites": suites,
        "plus": bool(plus_root),
        "plus_root": plus_root,
        "plus_perturbation": plus_perturbation,
        "plus_official_category": plus_official_category,
        "plus_sample_group_by": str(plus_subset.get("group_by", "none")),
        "plus_samples_per_group": int(plus_subset.get("samples_per_group", 0) or 0),
        "plus_sample_seed": int(plus_subset.get("sample_seed", 0) or 0),
        "plus_subset_manifest": manifest_by_category,
        "camera_size": int(closed_loop_cfg.get("camera_size", 256)),
        "seed": int(closed_loop_cfg.get("seed", 7)),
        "env_seed": int(closed_loop_cfg.get("env_seed", 0)),
        "num_tasks_per_suite": num_tasks_per_suite,
        "num_trials_per_task": num_trials_per_task,
        "num_work_items": len(work),
        "num_steps_wait": int(closed_loop_cfg.get("num_steps_wait", 10)),
        "max_steps_by_suite": _json_safe(
            _coerce_mapping(closed_loop_cfg.get("max_steps_by_suite", {}) or {})
        ),
        "action_horizon": closed_loop_cfg.get("action_horizon", 1),
        "action_repeat": closed_loop_cfg.get("action_repeat", 1),
        "action_repeat_mode": str(closed_loop_cfg.get("action_repeat_mode", "split_delta")),
        "env_control_hz": float(closed_loop_cfg.get("env_control_hz", 20.0)),
        "policy_hz": float(closed_loop_cfg.get("policy_hz", 20.0)),
        "history_horizon": closed_loop_cfg.get("history_horizon", 3),
        "rollout_decode_horizon": closed_loop_cfg.get("rollout_decode_horizon", "full"),
        "execute_chunk_prefix": int(closed_loop_cfg.get("execute_chunk_prefix", 0) or 0),
        "partial_chunk_history": str(closed_loop_cfg.get("partial_chunk_history", "default")),
        "warmup_full_chunk_once": bool(closed_loop_cfg.get("warmup_full_chunk_once", False)),
        "eval_num_active_ranks": int(closed_loop_cfg.get("eval_num_active_ranks", world_size or 1)),
        "env_process_isolation": bool(closed_loop_cfg.get("env_process_isolation", True)),
        "env_worker_timeout_sec": float(closed_loop_cfg.get("env_worker_timeout_sec", 300.0)),
        "world_size": int(world_size or 1),
        "policy": policy,
        "total_successes": total_success,
        "total_episodes": total_episodes,
        "total_crashes": total_crashes,
        "overall_success_rate": total_success / max(1, total_episodes),
        "suite_results": suite_results,
        "plus_official_category_results": category_results,
        "artifacts": {
            "summary_path": str(out_dir / "summary.json"),
            "per_task_path": str(per_task_path),
        },
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
        f.write("\n")
    return {"summary_path": str(summary_path), "per_task_path": str(per_task_path)}
