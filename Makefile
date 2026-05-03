.PHONY: test test-mmdit prepare-mmdit-text prepare-mmdit-text-cpu prepare-mmdit-latents smoke-mmdit smoke-mmdit-synthetic train-mmdit-smoke train-mmdit-smoke-resume check-mmdit-smoke-resume sample-mmdit-smoke prepare-mmdit-overfit-text prepare-mmdit-overfit-text-cpu prepare-mmdit-overfit-latents train-mmdit-overfit train-mmdit-overfit-resume check-mmdit-overfit sample-mmdit-overfit

test:
	CI=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
		$$(find tests -maxdepth 1 -name 'test_*.py' ! -name 'test_webui_endpoints.py' | sort)
	CI=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
		tests/test_webui_endpoints.py

test-mmdit:
	CI=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
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
	md-prepare-text-cache \
		--config config/train_mmdit_rf_smoke.yaml \
		--device cuda

prepare-mmdit-text-cpu:
	md-prepare-text-cache \
		--config config/train_mmdit_rf_smoke.yaml \
		--device cpu \
		--batch-size 1

prepare-mmdit-latents:
	md-prepare-latents \
		--config config/train_mmdit_rf_smoke.yaml

smoke-mmdit:
	md-smoke-mmdit-rf \
		--config config/train_mmdit_rf_smoke.yaml

smoke-mmdit-synthetic:
	md-smoke-mmdit-rf \
		--config config/train_mmdit_rf_smoke.yaml \
		--synthetic

train-mmdit-smoke:
	md-train \
		--config config/train_mmdit_rf_smoke.yaml

train-mmdit-smoke-resume:
	md-train \
		--config config/train_mmdit_rf_smoke_resume.yaml

check-mmdit-smoke-resume:
	python scripts/check_checkpoint_step.py \
		--ckpt ./runs/mmdit_smoke/ckpt_final.pt \
		--step 15

sample-mmdit-smoke:
	md-sample \
		--ckpt ./runs/mmdit_smoke/ckpt_final.pt \
		--prompt "1girl, simple background" \
		--sampler flow_heun \
		--steps 2 \
		--cfg 1 \
		--seed 42 \
		--out ./samples/mmdit_smoke.png

prepare-mmdit-overfit-text:
	md-prepare-text-cache \
		--config config/train_mmdit_rf_overfit.yaml \
		--device cuda

prepare-mmdit-overfit-text-cpu:
	md-prepare-text-cache \
		--config config/train_mmdit_rf_overfit.yaml \
		--device cpu \
		--batch-size 1

prepare-mmdit-overfit-latents:
	md-prepare-latents \
		--config config/train_mmdit_rf_overfit.yaml

train-mmdit-overfit:
	md-train \
		--config config/train_mmdit_rf_overfit.yaml

train-mmdit-overfit-resume:
	md-train \
		--config config/train_mmdit_rf_overfit_resume.yaml

check-mmdit-overfit:
	python scripts/check_checkpoint_step.py \
		--ckpt ./runs/mmdit_overfit/ckpt_final.pt \
		--step 2100

sample-mmdit-overfit:
	md-sample \
		--ckpt ./runs/mmdit_overfit/ckpt_final.pt \
		--prompt "1girl, simple background" \
		--sampler flow_heun \
		--steps 16 \
		--cfg 3 \
		--seed 42 \
		--out ./samples/mmdit_overfit.png
