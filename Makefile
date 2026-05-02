PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
BIN := $(if $(wildcard .venv/bin/md-train),.venv/bin/,)

.PHONY: test test-mmdit prepare-mmdit-text prepare-mmdit-text-cpu prepare-mmdit-latents smoke-mmdit smoke-mmdit-synthetic train-mmdit-smoke train-mmdit-smoke-resume check-mmdit-smoke-resume sample-mmdit-smoke prepare-mmdit-overfit-text prepare-mmdit-overfit-text-cpu prepare-mmdit-overfit-latents train-mmdit-overfit train-mmdit-overfit-resume check-mmdit-overfit sample-mmdit-overfit

test:
	CI=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q \
		$$(find tests -maxdepth 1 -name 'test_*.py' ! -name 'test_webui_endpoints.py' | sort)
	CI=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q \
		tests/test_webui_endpoints.py

test-mmdit:
	CI=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q \
		tests/test_mmdit_shapes.py \
		tests/test_patchify_roundtrip.py \
		tests/test_flow_objective.py \
		tests/test_mmdit_cfg.py \
		tests/test_training_overfit.py \
		tests/test_text_cache.py \
		tests/test_checkpoint_compat_mmdit.py \
		tests/test_train_profiles.py \
		tests/test_smoke_mmdit.py

prepare-mmdit-text:
	$(BIN)md-prepare-text-cache \
		--config config/train_mmdit_rf_smoke.yaml \
		--device cuda

prepare-mmdit-text-cpu:
	$(BIN)md-prepare-text-cache \
		--config config/train_mmdit_rf_smoke.yaml \
		--device cpu \
		--batch-size 1

prepare-mmdit-latents:
	$(BIN)md-prepare-latents \
		--config config/train_mmdit_rf_smoke.yaml

smoke-mmdit:
	$(BIN)md-smoke-mmdit-rf \
		--config config/train_mmdit_rf_smoke.yaml

smoke-mmdit-synthetic:
	$(BIN)md-smoke-mmdit-rf \
		--config config/train_mmdit_rf_smoke.yaml \
		--synthetic

train-mmdit-smoke:
	$(BIN)md-train \
		--config config/train_mmdit_rf_smoke.yaml

train-mmdit-smoke-resume:
	$(BIN)md-train \
		--config config/train_mmdit_rf_smoke_resume.yaml

check-mmdit-smoke-resume:
	$(PYTHON) scripts/check_checkpoint_step.py \
		--ckpt ./runs/mmdit_smoke/ckpt_final.pt \
		--step 15

sample-mmdit-smoke:
	$(BIN)md-sample \
		--ckpt ./runs/mmdit_smoke/ckpt_final.pt \
		--prompt "1girl, simple background" \
		--n 1 \
		--sampler flow_heun \
		--steps 2 \
		--cfg 1 \
		--seed 42 \
		--out ./samples/mmdit_smoke.png

prepare-mmdit-overfit-text:
	$(BIN)md-prepare-text-cache \
		--config config/train_mmdit_rf_overfit.yaml \
		--device cuda

prepare-mmdit-overfit-text-cpu:
	$(BIN)md-prepare-text-cache \
		--config config/train_mmdit_rf_overfit.yaml \
		--device cpu \
		--batch-size 1

prepare-mmdit-overfit-latents:
	$(BIN)md-prepare-latents \
		--config config/train_mmdit_rf_overfit.yaml

train-mmdit-overfit:
	$(BIN)md-train \
		--config config/train_mmdit_rf_overfit.yaml

train-mmdit-overfit-resume:
	$(BIN)md-train \
		--config config/train_mmdit_rf_overfit_resume.yaml

check-mmdit-overfit:
	$(PYTHON) scripts/check_checkpoint_step.py \
		--ckpt ./runs/mmdit_overfit/ckpt_final.pt \
		--step 2100

sample-mmdit-overfit:
	$(BIN)md-sample \
		--ckpt ./runs/mmdit_overfit/ckpt_final.pt \
		--prompt "1girl, simple background" \
		--sampler flow_heun \
		--steps 16 \
		--cfg 3 \
		--seed 42 \
		--out ./samples/mmdit_overfit.png
