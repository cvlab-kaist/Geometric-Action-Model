"""Opt-in runtime compatibility hooks for project-owned launchers."""

from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() not in {"", "0", "false", "no", "off"}


if _truthy(os.environ.get("VLA_EVAL_START_EPISODE_BEFORE_RESET")):
    import asyncio
    import itertools
    import logging
    from contextlib import suppress
    from typing import Any

    try:
        from vla_eval.runners.sync_runner import SyncEpisodeRunner
    except Exception:
        SyncEpisodeRunner = None  # type: ignore[assignment]

    if SyncEpisodeRunner is not None:
        _LOG = logging.getLogger("da3.vla_eval")

        def _int_env(name: str, default: int = 0) -> int:
            try:
                return max(0, int(str(os.environ.get(name, default)).strip()))
            except (TypeError, ValueError):
                return default

        def _float_env(name: str, default: float = 0.0) -> float:
            try:
                return max(0.0, float(str(os.environ.get(name, default)).strip()))
            except (TypeError, ValueError):
                return default

        def _is_env_infra_exception(exc: BaseException) -> bool:
            text = f"{type(exc).__name__}: {exc}"
            markers = (
                "LIBERO env worker",
                "worker died",
                "worker exited",
                "worker timed out",
                "read_pixels",
                "mjr_readPixels",
                "EGLError",
                "EGL",
                "SIGABRT",
                "Aborted",
                "BrokenPipeError",
                "EOFError",
            )
            return any(marker in text for marker in markers)

        async def _early_start_run_episode(
            self: SyncEpisodeRunner,
            benchmark: Any,
            task: dict[str, Any],
            conn: Any,
            *,
            max_steps: int | None = None,
        ) -> dict[str, Any]:
            """Route/reset the model episode before blocking MuJoCo reset/render."""
            del self
            task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
            infra_retries = _int_env("VLA_EVAL_INFRA_RETRY_ATTEMPTS", 0)
            retry_backoff = _float_env("VLA_EVAL_INFRA_RETRY_BACKOFF_SEC", 1.0)
            abort_on_infra_error = _truthy(os.environ.get("VLA_EVAL_ABORT_ON_INFRA_ERROR"))
            env_phases = {"benchmark_start", "benchmark_observe", "benchmark_apply", "benchmark_done", "benchmark_result"}

            for attempt in range(infra_retries + 1):
                episode_started = False
                phase = "episode_start"
                try:
                    await conn.start_episode({"task": task_info, "infra_attempt": attempt})
                    episode_started = True

                    phase = "benchmark_start"
                    await benchmark.start_episode(task)
                    phase = "benchmark_observe"
                    obs_dict = await benchmark.get_observation()

                    steps = range(max_steps) if max_steps is not None else itertools.count()
                    for step in steps:
                        phase = "model_act"
                        action = await conn.act(obs_dict)
                        phase = "benchmark_apply"
                        await benchmark.apply_action(action)
                        phase = "benchmark_done"
                        if await benchmark.is_done():
                            break
                        phase = "benchmark_observe"
                        obs_dict = await benchmark.get_observation()

                    phase = "benchmark_result"
                    elapsed = await benchmark.get_time()
                    metrics = await benchmark.get_result()
                    episode_result: dict[str, Any] = {
                        "metrics": metrics,
                        "steps": step + 1,
                        "elapsed_sec": round(elapsed, 3),
                    }
                    await conn.end_episode(episode_result)
                    episode_started = False
                    return episode_result
                except BaseException as exc:
                    is_env_infra = phase in env_phases and _is_env_infra_exception(exc)
                    should_retry = attempt < infra_retries and is_env_infra
                    abort_as_invalid = abort_on_infra_error and is_env_infra and not should_retry
                    if episode_started:
                        with suppress(Exception):
                            await conn.end_episode(
                                {
                                    "metrics": {"success": False},
                                    "failure_reason": "env_infrastructure_failure" if is_env_infra else "runner_exception",
                                    "infra_attempt": attempt,
                                }
                            )
                    if should_retry:
                        _LOG.warning(
                            "Retrying VLA episode after env infrastructure failure "
                            "(attempt %d/%d, phase=%s, task=%s): %s",
                            attempt + 1,
                            infra_retries,
                            phase,
                            task_info.get("name") or task_info.get("task_description") or task_info,
                            exc,
                        )
                        with suppress(Exception):
                            benchmark.cleanup()
                        if retry_backoff > 0:
                            await asyncio.sleep(retry_backoff)
                        continue
                    if abort_as_invalid:
                        _LOG.error(
                            "Aborting VLA shard after unrecovered env infrastructure failure "
                            "(attempts=%d, phase=%s, task=%s): %s",
                            infra_retries + 1,
                            phase,
                            task_info.get("name") or task_info.get("task_description") or task_info,
                            exc,
                        )
                        os._exit(86)
                    raise
            raise RuntimeError("unreachable VLA infra retry state")

        if not getattr(SyncEpisodeRunner.run_episode, "_da3_early_start_patch", False):
            _early_start_run_episode._da3_early_start_patch = True  # type: ignore[attr-defined]
            SyncEpisodeRunner.run_episode = _early_start_run_episode  # type: ignore[method-assign]


if _truthy(os.environ.get("DA3_ROBOCASA_IMPORT_COMPAT")):
    import importlib.abc
    import importlib.machinery
    import re
    import sys
    from types import ModuleType

    _ROBOCASA_MUJOCO_ASSERT_RE = re.compile(
        r"assert \(\n"
        r"\s+mujoco\.__version__ == \"(?P<version>[^\"]+)\"\n"
        r"\), \"MuJoCo version must be [^\"]+\"",
        re.MULTILINE,
    )
    _ROBOCASA_NUMPY_ASSERT_RE = re.compile(
        r"assert numpy\.__version__ in \[\n"
        r"(?P<body>.*?)"
        r"\], \"numpy version must be [^\"]+\"",
        re.DOTALL,
    )


    def _patch_robocasa_init_source(source: str) -> str:
        def replace_mujoco(match: re.Match[str]) -> str:
            expected = match.group("version")
            return (
                f"if mujoco.__version__ != \"{expected}\":\n"
                "    import warnings\n"
                "    warnings.warn(\n"
                f"        \"RoboCasa requested MuJoCo {expected}; running with \"\n"
                "        f\"{mujoco.__version__} under DA3_ROBOCASA_IMPORT_COMPAT.\",\n"
                "        RuntimeWarning,\n"
                "    )"
            )

        def replace_numpy(match: re.Match[str]) -> str:
            expected_versions = re.findall(r'"([^"]+)"', match.group("body"))
            expected_list = ", ".join(repr(version) for version in expected_versions)
            return (
                f"if numpy.__version__ not in [{expected_list}]:\n"
                "    import warnings\n"
                "    warnings.warn(\n"
                "        \"RoboCasa requested numpy \"\n"
                f"        + \",\".join([{expected_list}])\n"
                "        + f\"; running with {numpy.__version__} under DA3_ROBOCASA_IMPORT_COMPAT.\",\n"
                "        RuntimeWarning,\n"
                "    )"
            )

        patched, mujoco_count = _ROBOCASA_MUJOCO_ASSERT_RE.subn(replace_mujoco, source)
        patched, numpy_count = _ROBOCASA_NUMPY_ASSERT_RE.subn(replace_numpy, patched)
        if mujoco_count != 1 or numpy_count != 1:
            raise RuntimeError(
                "DA3_ROBOCASA_IMPORT_COMPAT could not patch RoboCasa version checks "
                f"(mujoco={mujoco_count}, numpy={numpy_count})"
            )
        return patched

    class _RoboCasaCompatLoader(importlib.abc.Loader):
        def __init__(self, loader: importlib.abc.Loader) -> None:
            self._loader = loader

        def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
            create = getattr(self._loader, "create_module", None)
            if create is None:
                return None
            return create(spec)

        def exec_module(self, module: ModuleType) -> None:
            origin = getattr(module.__spec__, "origin", None)
            get_data = getattr(self._loader, "get_data", None)
            if origin and get_data is not None:
                source = get_data(origin).decode("utf-8")
                patched = _patch_robocasa_init_source(source)
                code = compile(patched, origin, "exec", dont_inherit=True)
                exec(code, module.__dict__)
                return
            exec_module = getattr(self._loader, "exec_module")
            exec_module(module)

    class _RoboCasaCompatFinder(importlib.abc.MetaPathFinder):
        _finding = False

        def find_spec(
            self,
            fullname: str,
            path: list[str] | None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname != "robocasa" or self._finding:
                return None
            self._finding = True
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            finally:
                self._finding = False
            if spec is None or spec.loader is None or isinstance(spec.loader, _RoboCasaCompatLoader):
                return spec
            spec.loader = _RoboCasaCompatLoader(spec.loader)
            return spec

    if not any(isinstance(finder, _RoboCasaCompatFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RoboCasaCompatFinder())
