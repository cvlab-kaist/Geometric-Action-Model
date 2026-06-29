#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export DA3_CODE_ROOT="${DA3_CODE_ROOT:-$REPO_ROOT}"
export DA3_ROOT="${DA3_ROOT:-$REPO_ROOT}"
export DA3_PYTHON="${DA3_PYTHON:-python}"
export DA3_LIBERO_SOURCE_DIR="${DA3_LIBERO_SOURCE_DIR:-$REPO_ROOT/LIBERO}"
export DA3_LIBERO_PLUS_DIR="${DA3_LIBERO_PLUS_DIR:-$REPO_ROOT/LIBERO-plus}"
export DA3_MUJOCO_GL="${DA3_MUJOCO_GL:-${MUJOCO_GL:-egl}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DA3_SMOKE_GPU:-0}}"

if [[ -d "$DA3_LIBERO_SOURCE_DIR/libero" ]]; then
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$DA3_LIBERO_SOURCE_DIR/.libero_config_da3}"
fi

case "$DA3_MUJOCO_GL" in
    egl|osmesa)
        if [[ -n "${PYOPENGL_PLATFORM:-}" && "$PYOPENGL_PLATFORM" != "$DA3_MUJOCO_GL" ]]; then
            echo "ERROR: PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM conflicts with DA3_MUJOCO_GL=$DA3_MUJOCO_GL" >&2
            exit 1
        fi
        export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$DA3_MUJOCO_GL}"
        ;;
    glx)
        if [[ -n "${PYOPENGL_PLATFORM:-}" ]]; then
            echo "ERROR: PYOPENGL_PLATFORM must be unset for DA3_MUJOCO_GL=glx (got $PYOPENGL_PLATFORM)" >&2
            exit 1
        fi
        unset PYOPENGL_PLATFORM
        ;;
    *)
        echo "ERROR: unsupported DA3_MUJOCO_GL=$DA3_MUJOCO_GL" >&2
        exit 1
        ;;
esac
if [[ "$DA3_MUJOCO_GL" == "egl" && -n "${CUDA_VISIBLE_DEVICES:-${DA3_SMOKE_GPU:-}}" ]]; then
    _visible_for_egl="$CUDA_VISIBLE_DEVICES"
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${_visible_for_egl%%,*}}"
    unset _visible_for_egl
fi
export MUJOCO_GL="$DA3_MUJOCO_GL"
export PYTHONPATH="$DA3_CODE_ROOT/src:$DA3_LIBERO_PLUS_DIR:$DA3_LIBERO_SOURCE_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "[libero-eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_GL=$MUJOCO_GL"
exec "$DA3_PYTHON" -u "$DA3_CODE_ROOT/src/eval_libero_unified.py" "$@"
