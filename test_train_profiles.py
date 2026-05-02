from __future__ import annotations

from pathlib import Path

from config.train import TrainConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_profile(name: str) -> TrainConfig:
    return TrainConfig.from_yaml(str(ROOT / "config" / name))


def test_image_only_profile_is_unconditional_latent_baseline() -> None:
    cfg = _load_profile("train_image_only.yaml")

    assert cfg.mode == "latent"
    assert cfg.latent_cache is True
    assert cfg.latent_cache_sharded is True
    assert cfg.images_only is True
    assert cfg.use_text_conditioning is False
    assert cfg.eval_prompts_file == ""
    assert cfg.eval_cfg == 1.0
    assert cfg.noise_schedule == "cosine"
    assert cfg.prediction_type == "v"
    assert cfg.self_conditioning is True


def test_text_to_image_profile_enables_text_conditioning() -> None:
    cfg = _load_profile("train_text_to_image.yaml")

    assert cfg.mode == "latent"
    assert cfg.latent_cache is True
    assert cfg.latent_cache_sharded is True
    assert cfg.images_only is False
    assert cfg.use_text_conditioning is True
    assert cfg.meta_dir == "meta"
    assert cfg.tags_dir == "tags"
    assert cfg.min_tag_count >= 1
    assert cfg.eval_prompts_file
    assert cfg.eval_cfg > 1.0
    assert 0.10 <= cfg.cond_drop_prob <= 0.20


def test_profiles_map_to_distinct_unet_conditioning_modes() -> None:
    image_only = _load_profile("train_image_only.yaml")
    text_to_image = _load_profile("train_text_to_image.yaml")

    image_only_model = image_only.to_model_config()
    text_to_image_model = text_to_image.to_model_config()

    assert image_only_model.use_text_conditioning is False
    assert text_to_image_model.use_text_conditioning is True
    assert image_only_model.self_conditioning is True
    assert text_to_image_model.self_conditioning is True


def test_mmdit_smoke_resume_profile_extends_smoke_run() -> None:
    smoke = _load_profile("train_mmdit_rf_smoke.yaml")
    resume = _load_profile("train_mmdit_rf_smoke_resume.yaml")

    assert smoke.architecture == "mmdit_rf"
    assert resume.architecture == "mmdit_rf"
    assert resume.out_dir == smoke.out_dir
    assert resume.resume_ckpt.endswith("ckpt_latest.pt")
    assert resume.max_steps > smoke.max_steps
    assert resume.eval_every == 0


def test_mmdit_overfit_profile_is_next_control_stage() -> None:
    cfg = _load_profile("train_mmdit_rf_overfit.yaml")

    assert cfg.architecture == "mmdit_rf"
    assert cfg.objective == "rectified_flow"
    assert cfg.out_dir.endswith("mmdit_overfit")
    assert cfg.dataset_limit == 8
    assert cfg.max_steps == cfg.sanity_overfit_steps
    assert cfg.eval_every == 0
    assert cfg.sampling_sampler == "flow_heun"
