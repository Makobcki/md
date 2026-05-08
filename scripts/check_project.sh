#!/usr/bin/env bash
set -euo pipefail

run_checked() {
  local seconds="$1"
  shift
  timeout --kill-after=5s "${seconds}s" "$@"
}

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
run_checked 30 python -m scripts.lint
python -m pytest -q
run_checked 30 python -m train.cli --dry-run --set training.preset=single_gpu_debug
run_checked 30 python -m train.cli --dry-run --set model.preset=mmdit_1024 --set training.batch_size=1
run_checked 30 python -m config.cli --target train --set training.preset=single_gpu_debug >/dev/null
run_checked 30 python -m sample.cli --help >/dev/null
run_checked 30 python -m train.eval_cli --help >/dev/null
run_checked 30 python -m scripts.prepare_training_cache --help >/dev/null
run_checked 30 python -m scripts.validate_cache --help >/dev/null

grep -RIn -E "unet|U-Net|DDPM|DDIM|DPM|BPE|v_prediction|min_snr" . \
  --exclude='AGENTS.md' \
  --exclude='README.md' \
  --exclude='check_project.sh' \
  --exclude-dir='.git' \
  --exclude-dir='.venv' \
  --exclude-dir='.pytest_cache' \
  --exclude-dir='.cache' \
  --exclude-dir='dist' \
  --exclude-dir='md-dev' \
  --exclude-dir='md_diffusion.egg-info' \
  --exclude-dir='node_modules' \
  --exclude-dir='runs' \
  --exclude-dir='__pycache__' \
  --exclude='*.pyc' && exit 1 || true
exit 0
