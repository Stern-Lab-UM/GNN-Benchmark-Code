#!/usr/bin/env bash
# Create an isolated Lighthouse/shared-HPC Python environment for this repo.
# The environment is outside the git clone by default, keeping installed
# packages, caches, checkpoints, and outputs away from publication source code.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_lh_env.sh [options]

Options:
  --component all|mpnn|ppgn   Which dependency/check set to install [default: all]
  --env-dir PATH              Environment directory [default: $SCRATCH/dcg_gnn_envs/gnn_benchmark_code_py310]
  --python PYTHON             Python executable used to create the venv [default: auto-detect >=3.10]
  --skip-install              Create/activate env and run checker without pip installing requirements
  -h, --help                  Show this help

Environment variables:
  DCG_GNN_ENV_DIR             Alternative default for --env-dir
  DCG_GNN_PYTHON              Alternative default for --python

Examples:
  bash scripts/setup_lh_env.sh --component all
  bash scripts/setup_lh_env.sh --component mpnn --env-dir "$SCRATCH/dcg_envs/mpnn_pub"
  bash scripts/setup_lh_env.sh --component ppgn --skip-install
EOF
}

component="all"
python_bin="${DCG_GNN_PYTHON:-}"
default_base="${SCRATCH:-$HOME}"
env_dir="${DCG_GNN_ENV_DIR:-$default_base/dcg_gnn_envs/gnn_benchmark_code_py310}"
do_install=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --component)
      component="${2:-}"
      shift 2
      ;;
    --env-dir)
      env_dir="${2:-}"
      shift 2
      ;;
    --python)
      python_bin="${2:-}"
      shift 2
      ;;
    --skip-install)
      do_install=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$component" in
  all|mpnn|ppgn) ;;
  *)
    echo "Invalid --component '$component'; expected all, mpnn, or ppgn." >&2
    exit 2
    ;;
esac

if [[ -z "$env_dir" ]]; then
  echo "Environment directory is empty." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

if [[ "$component" == "ppgn" ]]; then
  req_file="$repo_dir/requirements/ppgn.txt"
else
  req_file="$repo_dir/requirements/mpnn.txt"
fi

if [[ -z "$python_bin" ]]; then
  for candidate in python3.10 python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi

if [[ -z "$python_bin" ]] || ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Could not find a usable Python >= 3.10 executable." >&2
  echo "Try loading a Python module, for example: module load python/3.11.5" >&2
  echo "Or pass --python /path/to/python3.10-or-newer" >&2
  exit 1
fi

if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  echo "Python executable '$python_bin' is too old; Python >= 3.10 is required." >&2
  exit 1
fi

echo "[setup] repo:         $repo_dir"
echo "[setup] component:    $component"
echo "[setup] python:       $(command -v "$python_bin")"
echo "[setup] env:          $env_dir"
echo "[setup] requirements: $req_file"

if [[ ! -x "$env_dir/bin/python" ]]; then
  echo "[setup] creating virtual environment"
  mkdir -p "$(dirname "$env_dir")"
  "$python_bin" -m venv "$env_dir"
else
  echo "[setup] reusing existing virtual environment"
fi

# shellcheck disable=SC1091
source "$env_dir/bin/activate"

echo "[setup] active python: $(which python)"
python -m pip --version

if [[ "$do_install" -eq 1 ]]; then
  echo "[setup] upgrading pip"
  python -m pip install --upgrade pip
  echo "[setup] installing requirements"
  python -m pip install -r "$req_file"
else
  echo "[setup] skipping pip install"
fi

echo "[setup] running install checker"
python "$repo_dir/scripts/check_install.py" --component "$component"

cat <<EOF

[setup] Done.
To reuse this environment later, run:
  source "$env_dir/bin/activate"
  cd "$repo_dir"
  python scripts/check_install.py --component $component
EOF
