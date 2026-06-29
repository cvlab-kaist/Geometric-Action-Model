#!/usr/bin/env bash
set -euo pipefail

# Standalone local-GPU LIBERO-Plus rollout eval for
# SeonghuJeon/3da-libero-gam.
#
# Launches one Python process per GPU, passes explicit
# --shard-index/--shard-count values, then aggregates the shard outputs into a
# suite-level summary.json and per_task.csv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_hf_gam_libero_plus_eval.sh spatial|object|goal|long|all

Environment:
  GAM_EVAL_GPUS              Comma-separated physical GPU ids. Default: CUDA_VISIBLE_DEVICES or 0.
  HF_ROOT                    Downloaded HF checkpoint root.
                             Default: checkpoints_hf/3da-libero-gam
  DA3_BASE_CKPT              DA3-Giant base checkpoint.
                             Default: $DA3_ROOT/checkpoints/track4world_da3.pth
  OUT_ROOT                   Eval output root.
                             Default: results/eval_libero_batched/hf_gam_plus_local
  DA3_LIBERO_SOURCE_DIR      LIBERO checkout root. Default: ./LIBERO
  DA3_LIBERO_PLUS_DIR        LIBERO-Plus checkout root. Default: ./LIBERO-plus
  DA3_PYTHON                 Python executable. Default: python
  PARALLEL_ENVS_PER_GPU      Env workers per GPU process. Default: 16
  MAX_BATCH_SIZE             Max observations per policy forward. Default: 16
  MAX_WAIT_TIME              Batch wait time in seconds. Default: 0.5
  ENV_CACHE_SIZE             Cached envs per worker. Default: 1
  GAM_PLUS_PERTURBATION      Plus perturbation filter. Default: all
  GAM_PLUS_OFFICIAL_CATEGORY Plus official category filter. Default: all

Example:
  GAM_EVAL_GPUS=0,1,2,3 bash scripts/run_hf_gam_libero_plus_eval.sh spatial
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SUITE_REQUEST="${1:-all}"
case "$SUITE_REQUEST" in
  spatial|object|goal|long|all) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

export DA3_CODE_ROOT="${DA3_CODE_ROOT:-$REPO_ROOT}"
export DA3_ROOT="${DA3_ROOT:-$REPO_ROOT}"
export DA3_PYTHON="${DA3_PYTHON:-python}"
export DA3_BASE_CKPT="${DA3_BASE_CKPT:-$DA3_ROOT/checkpoints/track4world_da3.pth}"
export DA3_LIBERO_SOURCE_DIR="${DA3_LIBERO_SOURCE_DIR:-$REPO_ROOT/LIBERO}"
export DA3_LIBERO_PLUS_DIR="${DA3_LIBERO_PLUS_DIR:-$REPO_ROOT/LIBERO-plus}"
export HF_ROOT="${HF_ROOT:-$REPO_ROOT/checkpoints_hf/3da-libero-gam}"
export OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/results/eval_libero_batched/hf_gam_plus_local}"
export PARALLEL_ENVS_PER_GPU="${PARALLEL_ENVS_PER_GPU:-16}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
export MAX_WAIT_TIME="${MAX_WAIT_TIME:-0.5}"
export ENV_CACHE_SIZE="${ENV_CACHE_SIZE:-1}"
export GAM_PLUS_PERTURBATION="${GAM_PLUS_PERTURBATION:-all}"
export GAM_PLUS_OFFICIAL_CATEGORY="${GAM_PLUS_OFFICIAL_CATEGORY:-all}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-device}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -d "$DA3_LIBERO_SOURCE_DIR/libero" ]]; then
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$DA3_LIBERO_SOURCE_DIR/.libero_config_da3}"
fi

export PYTHONPATH="$REPO_ROOT/src:$DA3_LIBERO_PLUS_DIR:$DA3_LIBERO_SOURCE_DIR:${PYTHONPATH:-}"

GPU_CSV="${GAM_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -ra GPUS <<< "$GPU_CSV"
if (( ${#GPUS[@]} < 1 )); then
  echo "ERROR: no GPUs selected. Set GAM_EVAL_GPUS=0,1,2,3 or CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

if [[ ! -s "$DA3_BASE_CKPT" ]]; then
  echo "ERROR: missing DA3 base checkpoint: $DA3_BASE_CKPT" >&2
  echo "Download with: hf download SeonghuJeon/3da-libero-training-assets --repo-type dataset --local-dir $DA3_ROOT" >&2
  exit 2
fi
if [[ ! -d "$DA3_LIBERO_PLUS_DIR/libero" ]]; then
  echo "ERROR: missing LIBERO-Plus checkout: $DA3_LIBERO_PLUS_DIR" >&2
  exit 2
fi

aggregate_suite() {
  local run_root="$1"
  local suite_name="$2"
  local run_name="$3"
  local ckpt="$4"
  local config="$5"
  local shard_count="$6"

  "$DA3_PYTHON" - "$run_root" "$suite_name" "$run_name" "$ckpt" "$config" "$shard_count" <<'PY'
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

run_root = Path(sys.argv[1])
suite_name = sys.argv[2]
run_name = sys.argv[3]
ckpt = sys.argv[4]
config = sys.argv[5]
shard_count = int(sys.argv[6])

shard_root = run_root / "shards"
expected = [shard_root / f"shard{idx:04d}" for idx in range(shard_count)]
missing = [str(path) for path in expected if not (path / "summary.json").is_file() or not (path / "per_task.csv").is_file()]
if missing:
    raise SystemExit("missing shard outputs: " + ", ".join(missing))

def as_int(value):
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0

def as_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0

rows = []
summaries = []
fieldnames = []
for shard_dir in expected:
    summaries.append(json.loads((shard_dir / "summary.json").read_text()))
    with (shard_dir / "per_task.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
        rows.extend(dict(row) for row in reader)

merged = {}
for row in rows:
    key = (str(row.get("suite") or ""), as_int(row.get("task_id")), as_int(row.get("eval_task_index")))
    trials = as_int(row.get("num_trials"))
    success = as_int(row.get("num_success"))
    avg_steps = as_float(row.get("avg_steps"))
    if key not in merged:
        bucket = {field: row.get(field, "") for field in fieldnames}
        bucket["num_trials"] = trials
        bucket["num_success"] = success
        bucket["_weighted_steps"] = avg_steps * trials
        merged[key] = bucket
    else:
        bucket = merged[key]
        bucket["num_trials"] = as_int(bucket.get("num_trials")) + trials
        bucket["num_success"] = as_int(bucket.get("num_success")) + success
        bucket["_weighted_steps"] = as_float(bucket.get("_weighted_steps")) + avg_steps * trials

merged_rows = []
for bucket in merged.values():
    trials = as_int(bucket.get("num_trials"))
    success = as_int(bucket.get("num_success"))
    bucket["success_rate"] = float(success / trials) if trials else 0.0
    bucket["avg_steps"] = float(as_float(bucket.get("_weighted_steps")) / trials) if trials else 0.0
    bucket.pop("_weighted_steps", None)
    merged_rows.append(bucket)
merged_rows.sort(key=lambda row: (str(row.get("suite") or ""), as_int(row.get("eval_task_index")), as_int(row.get("task_id"))))

def compact_category(items):
    out = {}
    task_keys = defaultdict(set)
    for row in items:
        slug = str(row.get("plus_official_category_slug") or "")
        if not slug:
            continue
        bucket = out.setdefault(
            slug,
            {
                "category": str(row.get("plus_official_category") or slug),
                "success_rate": 0.0,
                "num_trials": 0,
                "num_success": 0,
                "num_tasks": 0,
            },
        )
        trials = as_int(row.get("num_trials"))
        success = as_int(row.get("num_success"))
        bucket["num_trials"] += trials
        bucket["num_success"] += success
        task_keys[slug].add((str(row.get("suite") or ""), as_int(row.get("eval_task_index")), as_int(row.get("task_id"))))
    for slug, bucket in out.items():
        trials = int(bucket["num_trials"])
        bucket["success_rate"] = float(bucket["num_success"] / trials) if trials else 0.0
        bucket["num_tasks"] = len(task_keys[slug])
    return dict(sorted(out.items()))

total_trials = sum(as_int(row.get("num_trials")) for row in merged_rows)
total_success = sum(as_int(row.get("num_success")) for row in merged_rows)
summary = dict(summaries[0]) if summaries else {}
summary.update(
    {
        "run_name": run_name,
        "scope": "suite",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "eval_type": "libero_batched_harness_style_standalone",
        "ckpt": ckpt,
        "config": config,
        "suites": [suite_name],
        "suite_results": {
            suite_name: {
                "success_rate": float(total_success / total_trials) if total_trials else 0.0,
                "num_trials": int(total_trials),
                "num_success": int(total_success),
                "plus_official_category_results": compact_category(merged_rows),
            }
        },
        "plus_official_category_results": compact_category(merged_rows),
        "total_successes": int(total_success),
        "total_episodes": int(total_trials),
        "overall_success_rate": float(total_success / total_trials) if total_trials else 0.0,
        "average_success_rate": float(total_success / total_trials) if total_trials else 0.0,
        "shard": {
            "enabled": True,
            "count": shard_count,
            "aggregated": True,
            "summary_paths": [str(path / "summary.json") for path in expected],
        },
        "artifacts": {
            "summary_path": str(run_root / "summary.json"),
            "per_task_path": str(run_root / "per_task.csv"),
            "progress_logs": [str(path / "progress.jsonl") for path in expected],
        },
    }
)

(run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (run_root / "per_task.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in merged_rows])
print(f"[aggregate] {suite_name} {total_success}/{total_trials} sr={summary['overall_success_rate']:.4f} root={run_root}", flush=True)
PY
}

prepare_gam_config() {
  local src_config="$1"
  local dst_config="$2"
  "$DA3_PYTHON" - "$src_config" "$dst_config" <<'PY'
import os
import sys
from omegaconf import OmegaConf

src, dst = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(src)
if not hasattr(cfg, "predictor") or cfg.predictor is None:
    cfg.predictor = {}
cfg.predictor.type = "gam"
if not hasattr(cfg, "stage_1") or cfg.stage_1 is None:
    cfg.stage_1 = {}
cfg.stage_1.ckpt_path = os.environ.get(
    "DA3_BASE_CKPT",
    "${oc.env:DA3_ROOT,.}/checkpoints/track4world_da3.pth",
)
OmegaConf.save(config=cfg, f=dst)
PY
}

run_suite() {
  local suite_key="$1"
  local suite_name="$2"
  local ckpt_rel="$3"
  local config_rel="$4"
  local step_name="$5"
  local ckpt="$HF_ROOT/$ckpt_rel"
  local source_config="$HF_ROOT/$config_rel"
  local shard_count="${#GPUS[@]}"
  local filter_name="full"
  if [[ "$GAM_PLUS_PERTURBATION" != "all" || "$GAM_PLUS_OFFICIAL_CATEGORY" != "all" ]]; then
    filter_name="${GAM_PLUS_PERTURBATION}_${GAM_PLUS_OFFICIAL_CATEGORY}"
    filter_name="${filter_name//[^A-Za-z0-9_]/_}"
  fi
  local run_name="${suite_key}_hf_gam_${step_name}_plus_${filter_name}_qpos_original_$(date +%Y%m%d_%H%M%S)"
  local run_root="$OUT_ROOT/$suite_key/$run_name"
  local config="$run_root/config.gam.yaml"

  for path in "$ckpt" "$source_config"; do
    if [[ ! -s "$path" ]]; then
      echo "ERROR: missing required HF file: $path" >&2
      echo "Download with: hf download SeonghuJeon/3da-libero-gam --local-dir $HF_ROOT" >&2
      exit 2
    fi
  done

  mkdir -p "$run_root/shards"
  prepare_gam_config "$source_config" "$config"
  cat > "$run_root/run_config.txt" <<EOF
suite_key=$suite_key
suite_name=$suite_name
checkpoint=$ckpt
config=$config
source_config=$source_config
run_name=$run_name
run_root=$run_root
gpu_csv=$GPU_CSV
plus=true
plus_perturbation=$GAM_PLUS_PERTURBATION
plus_official_category=$GAM_PLUS_OFFICIAL_CATEGORY
num_trials_per_task=1
libero_plus_robot_init_qpos_mode=original
shard_count=$shard_count
parallel_envs_per_gpu=$PARALLEL_ENVS_PER_GPU
max_batch_size=$MAX_BATCH_SIZE
max_wait_time=$MAX_WAIT_TIME
env_cache_size=$ENV_CACHE_SIZE
EOF

  echo "[hf-plus] suite=$suite_key suite_name=$suite_name shards=$shard_count run_root=$run_root"
  local pids=()
  local shard_idx=0
  for gpu in "${GPUS[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -z "$gpu" ]] && continue
    local shard_name
    shard_name="$(printf "shard%04d" "$shard_idx")"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export MUJOCO_EGL_DEVICE_ID="$gpu"
      "$DA3_PYTHON" -u "$REPO_ROOT/scripts/run_libero_batched_eval.py" \
        --ckpt "$ckpt" \
        --config "$config" \
        --preset openvla_50 \
        --suites "$suite_name" \
        --plus \
        --plus-root "$DA3_LIBERO_PLUS_DIR" \
        --plus-perturbation "$GAM_PLUS_PERTURBATION" \
        --plus-official-category "$GAM_PLUS_OFFICIAL_CATEGORY" \
        --num-trials-per-task 1 \
        --libero-plus-robot-init-qpos-mode original \
        --history-horizon 1 \
        --rollout-decode-horizon 1 \
        --action-horizon 1 \
        --action-repeat 1 \
        --action-repeat-mode split_delta \
        --policy-hz 20 \
        --env-control-hz 20 \
        --camera-size 256 \
        --parallel-envs "$PARALLEL_ENVS_PER_GPU" \
        --max-batch-size "$MAX_BATCH_SIZE" \
        --max-wait-time "$MAX_WAIT_TIME" \
        --env-cache-size "$ENV_CACHE_SIZE" \
        --env-process-isolation \
        --env-worker-timeout-sec 900 \
        --rollout-wall-timeout-sec 900 \
        --shard-index "$shard_idx" \
        --shard-count "$shard_count" \
        --run-name "$shard_name" \
        --output-dir "$run_root/shards" \
        --no-registry
    ) &
    pids+=("$!")
    shard_idx=$((shard_idx + 1))
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "ERROR: at least one shard failed for suite=$suite_key; leaving partial outputs in $run_root" >&2
    exit 1
  fi

  aggregate_suite "$run_root" "$suite_name" "$run_name" "$ckpt" "$config" "$shard_count"
  echo "[hf-plus] completed suite=$suite_key summary=$run_root/summary.json"
}

case "$SUITE_REQUEST" in
  spatial)
    run_suite spatial libero_spatial spatial/gam.pt spatial/config.yaml gam
    ;;
  object)
    run_suite object libero_object object/gam.pt object/config.yaml gam
    ;;
  goal)
    run_suite goal libero_goal goal/gam.pt goal/config.yaml gam
    ;;
  long)
    run_suite long libero_10 long/gam.pt long/config.yaml gam
    ;;
  all)
    run_suite spatial libero_spatial spatial/gam.pt spatial/config.yaml gam
    run_suite object libero_object object/gam.pt object/config.yaml gam
    run_suite goal libero_goal goal/gam.pt goal/config.yaml gam
    run_suite long libero_10 long/gam.pt long/config.yaml gam
    ;;
esac
