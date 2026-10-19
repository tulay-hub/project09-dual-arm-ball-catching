#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${ISAAC_PYTHON:-python3}"
mode="${1:-help}"
if (( $# )); then shift; fi
common=(env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1)
case "$mode" in
  train) exec "${common[@]}" HEADLESS=1 "$python_bin" scripts/train_ball_catch.py --visualizer none "$@" ;;
  play-policy) exec "${common[@]}" HEADLESS=0 "$python_bin" scripts/train_ball_catch.py --play --visualizer kit "$@" ;;
  play-stream) exec "${common[@]}" HEADLESS=1 "$python_bin" scripts/train_ball_catch.py --play --live_view --visualizer none "$@" ;;
  play-mujoco) exec "${common[@]}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" "$python_bin" scripts/play_mujoco_policy.py "$@" ;;
  prepare) exec "${common[@]}" "$python_bin" scripts/prepare_scene.py "$@" ;;
  verify) exec "${common[@]}" "$python_bin" scripts/inspect_model.py "$@" ;;
  smoke|drop|release|catch) exec "${common[@]}" HEADLESS=1 "$python_bin" scripts/run_scene.py --mode "$mode" --visualizer none "$@" ;;
  scene) exec "${common[@]}" HEADLESS=0 "$python_bin" scripts/run_scene.py --mode drop --seconds 30 --visualizer kit "$@" ;;
  play-grasp) exec "${common[@]}" HEADLESS=0 "$python_bin" scripts/run_scene.py --mode catch --secure_strategy geometry_grasp --seconds 7 --visualizer kit "$@" ;;
  *) echo 'Usage: bash scripts/run.sh {train|play-policy|play-stream|play-mujoco|prepare|verify|smoke|drop|release|catch|scene|play-grasp} [arguments].' ;;
esac
