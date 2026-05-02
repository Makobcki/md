.PHONY: test test-mmdit

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

test-mmdit:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
		tests/test_mmdit_shapes.py \
		tests/test_patchify_roundtrip.py \
		tests/test_flow_objective.py \
		tests/test_mmdit_cfg.py \
		tests/test_training_overfit.py \
		tests/test_text_cache.py \
		tests/test_checkpoint_compat_mmdit.py \
		tests/test_train_profiles.py
