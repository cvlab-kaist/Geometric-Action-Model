#!/usr/bin/env bash
set -euo pipefail

ROOT="${DA3_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PLUS_DIR="${DA3_LIBERO_PLUS_DIR:-$ROOT/LIBERO-plus}"
DOWNLOAD_DIR="${DA3_LIBERO_PLUS_DOWNLOAD_DIR:-$ROOT/downloads/libero_plus_assets}"
ASSETS_DIR="${DA3_LIBERO_PLUS_ASSETS_DIR:-$PLUS_DIR/libero/libero/assets}"
LIBERO_PLUS_COMMIT="${LIBERO_PLUS_COMMIT:-4976dc30028e805ff8094b55501d532c48fec182}"
DA3_PYTHON="${DA3_PYTHON:-python3}"

DOWNLOAD_ASSETS=0
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_libero_plus.sh [--download-assets] [--force]

Environment:
  DA3_ROOT                       Repository root. Default: this script's parent.
  DA3_LIBERO_PLUS_DIR            LIBERO-Plus checkout. Default: $DA3_ROOT/LIBERO-plus.
  DA3_LIBERO_PLUS_DOWNLOAD_DIR   assets.zip cache. Default: $DA3_ROOT/downloads/libero_plus_assets.
  DA3_LIBERO_PLUS_ASSETS_DIR     Extracted assets dir. Default: $DA3_LIBERO_PLUS_DIR/libero/libero/assets.
  DA3_PYTHON                     Python executable. Default: python3.
  LIBERO_PLUS_COMMIT             LIBERO-Plus source commit.

Examples:
  bash scripts/setup_libero_plus.sh --download-assets
  DA3_LIBERO_PLUS_DIR=/data/LIBERO-plus bash scripts/setup_libero_plus.sh --download-assets
EOF
}

while (($#)); do
  case "$1" in
    --download-assets)
      DOWNLOAD_ASSETS=1
      ;;
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

echo "[setup_libero_plus] root: $ROOT"
echo "[setup_libero_plus] plus dir: $PLUS_DIR"

if [[ ! -d "$PLUS_DIR/libero" ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git "$PLUS_DIR"
  git -C "$PLUS_DIR" checkout "$LIBERO_PLUS_COMMIT"
else
  echo "[setup_libero_plus] LIBERO-Plus source already present."
fi

if (( DOWNLOAD_ASSETS == 0 )); then
  cat <<EOF
[setup_libero_plus] source ready.
To install perturbation assets, run:
  bash scripts/setup_libero_plus.sh --download-assets
EOF
  exit 0
fi

mkdir -p "$DOWNLOAD_DIR" "$ASSETS_DIR"

"$DA3_PYTHON" - "$DOWNLOAD_DIR" "$ASSETS_DIR" "$FORCE" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

download_dir = Path(sys.argv[1]).expanduser().resolve()
assets_dir = Path(sys.argv[2]).expanduser().resolve()
force = bool(int(sys.argv[3]))

marker = assets_dir / "scenes" / "libero_tabletop_base_style.xml"
object_marker = assets_dir / "stable_scanned_objects" / "akita_black_bowl" / "akita_black_bowl.xml"
complete_marker = assets_dir / ".da3_assets_unzip_complete"

if complete_marker.exists() and marker.exists() and object_marker.exists() and not force:
    print(f"[setup_libero_plus] assets already installed: {assets_dir}")
    raise SystemExit(0)

archive = download_dir / "assets.zip"
if not archive.exists() or archive.stat().st_size == 0:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit(
            "huggingface_hub is required to download LIBERO-Plus assets. "
            "Install requirements.txt first, or place assets.zip under "
            f"{download_dir}."
        ) from exc
    path = hf_hub_download(
        repo_id="Sylvest/LIBERO-plus",
        repo_type="dataset",
        filename="assets.zip",
        local_dir=download_dir,
    )
    archive = Path(path)

print(f"[setup_libero_plus] archive: {archive} ({archive.stat().st_size} bytes)")
assets_dir.mkdir(parents=True, exist_ok=True)
if force and assets_dir.exists():
    for child in assets_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

with zipfile.ZipFile(archive) as zf:
    names = zf.namelist()
    marker_name = next(
        (
            name
            for name in names
            if name.endswith("assets/scenes/libero_tabletop_base_style.xml")
        ),
        None,
    )
    if marker_name is None:
        raise FileNotFoundError("assets.zip does not contain assets/scenes/libero_tabletop_base_style.xml")
    prefix = marker_name[: -len("assets/scenes/libero_tabletop_base_style.xml")]
    asset_prefix = prefix + "assets/"
    print(f"[setup_libero_plus] zip asset prefix: {asset_prefix}")

    extracted = 0
    for member in zf.infolist():
        if not member.filename.startswith(asset_prefix):
            continue
        rel = member.filename[len(asset_prefix):]
        if not rel:
            continue
        target = (assets_dir / rel).resolve()
        if not str(target).startswith(str(assets_dir.resolve())):
            raise RuntimeError(f"unsafe zip member: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, target.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
        extracted += 1
        if extracted % 10000 == 0:
            print(f"[setup_libero_plus] extracted files: {extracted}", flush=True)

if not marker.exists():
    raise FileNotFoundError(f"missing marker after extraction: {marker}")
if not object_marker.exists():
    raise FileNotFoundError(f"missing object asset after extraction: {object_marker}")
complete_marker.write_text("ok\n")
print(f"[setup_libero_plus] assets ready: {assets_dir}")
PY

cat <<EOF
[setup_libero_plus] done.
Set this if you use a non-default asset directory:
  export DA3_LIBERO_PLUS_ASSETS_DIR="$ASSETS_DIR"
EOF
