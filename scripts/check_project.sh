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
run_checked 30 python -m train.cli --profile smoke --dry-run
run_checked 30 python -m train.cli --profile overfit --dry-run
run_checked 30 python -m train.cli --profile dev --dry-run
run_checked 30 python -m train.cli --profile base --dry-run
run_checked 30 python -m train.cli --profile milestone_a --dry-run
run_checked 30 python -m train.cli --profile milestone_b --dry-run
run_checked 30 python -m train.cli --profile milestone_c --dry-run
run_checked 30 python -m train.cli --profile distributed_smoke --dry-run
run_checked 30 python -m train.cli --profile fsdp_template --dry-run
run_checked 30 python -m sample.cli --help >/dev/null
run_checked 30 python -m train.eval_cli --help >/dev/null
run_checked 30 python -m scripts.prepare_training_cache --help >/dev/null
run_checked 30 python -m scripts.validate_cache --help >/dev/null

grep -RIn -E "unet|U-Net|legacy|DDPM|DDIM|DPM|BPE|v_prediction|min_snr" . \
  --exclude='check_project.sh' \
  --exclude-dir='.git' \
  --exclude-dir='.pytest_cache' \
  --exclude-dir='.cache' \
  --exclude-dir='runs' \
  --exclude-dir='__pycache__' \
  --exclude='*.pyc' && exit 1 || true
exit 0
