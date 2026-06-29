# =============================================================================
# GAM for LIBERO / LIBERO-Plus training and closed-loop inference
#
# Self-contained, reproducible build: installs the full pinned Python stack,
# the LIBERO benchmark, LIBERO-Plus perturbation dependencies, and the DA3
# backbone from source. Datasets and base weights are mounted at run time under
# $DA3_ROOT (see README).
#
#   docker build -t gam-libero .
#   docker run --gpus all -it --rm \
#     -e DA3_ROOT=/data -v /host/data_root:/data \
#     -e WANDB_API_KEY=$WANDB_API_KEY gam-libero
#
# The base image fixes torch==2.5.1 (+cu124) and Python 3.11. The pinned
# package set in requirements.txt targets CUDA GPU machines. torch >= 2.5 is
# required for the predictor's flex_attention / BlockMask path.
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# System libs for headless MuJoCo / OpenGL, LIBERO-Plus motion blur, and build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ca-certificates \
      libgl1 libglib2.0-0 libglvnd0 libegl1 libgles2 libosmesa6 libglfw3 \
      libx11-6 libxext6 libxrender1 ffmpeg imagemagick libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gam-libero

# ---- Python deps (pinned). torch (2.5.1) already in the base image satisfies
#      `torch>=2.5`, so pip keeps that build. ------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # robosuite pulls non-headless opencv-python (needs libGL); both opencv pkgs
    # share the cv2/ dir, so force a clean headless-only cv2 for robust headless use.
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir --no-deps opencv-python-headless==4.11.0.86

# ---- DA3 backbone --------------------------------------------------------------
# da3_giant_encoder.py adds  $DA3_ROOT/Depth-Anything-3/src  to sys.path, so the
# backbone only needs to be present on disk. Installing the package would pull
# heavy NVS-only deps such as open3d/pycolmap/xformers that the gam encoder skips.
ARG DA3_BACKBONE_COMMIT=2c21ea849ceec7b469a3e62ea0c0e270afc3281a
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git Depth-Anything-3 \
    && git -C Depth-Anything-3 checkout ${DA3_BACKBONE_COMMIT}

# ---- LIBERO benchmark ----------------------------------------------------------
# Used via PYTHONPATH: put the LIBERO repo root on PYTHONPATH
# so `import libero.libero.*` resolves (LIBERO/libero is a namespace package, so
# only the repo root goes on
# the path). LIBERO's own requirements.txt is ignored (it pins old, conflicting
# numpy/transformers/gym); the runtime deps its env classes need (bddl, easydict,
# future) are pinned in requirements.txt.
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO

# ---- LIBERO-Plus for `--plus` perturbed-task eval ------------------------------
ARG LIBERO_PLUS_COMMIT=4976dc30028e805ff8094b55501d532c48fec182
RUN git clone https://github.com/sylvestf/LIBERO-plus.git /opt/LIBERO-plus \
    && git -C /opt/LIBERO-plus checkout ${LIBERO_PLUS_COMMIT}

# ---- Project source ------------------------------------------------------------
COPY . .

# Headless rendering + runtime defaults. PYTHONPATH includes the project src and
# LIBERO source; DA3 backbone is found via DA3_ROOT.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    DA3_ROOT=/workspace/gam-libero \
    DA3_LIBERO_SOURCE_DIR=/opt/LIBERO \
    DA3_LIBERO_PLUS_DIR=/opt/LIBERO-plus \
    PYTHONPATH=/workspace/gam-libero/src:/opt/LIBERO \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Datasets + base weights are mounted or downloaded under $DA3_ROOT:
#   $DA3_ROOT/checkpoints/track4world_da3.pth       (DA3-Giant base weights)
#   $DA3_ROOT/data/libero_noop/<suite>/*.hdf5       (LIBERO HDF5 demos, embedded depth)
#   $DA3_ROOT/data/libero_noop/_stats               (action/proprio normalizer stats)
# LIBERO benchmark assets are downloaded by the LIBERO package on first use.
CMD ["bash"]
