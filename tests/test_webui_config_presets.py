from __future__ import annotations

from pathlib import Path

from webui.backend.services import config_service

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_list_config_presets_groups_real_kdl_presets() -> None:
    payload = config_service.list_config_presets(ROOT_DIR)

    model_names = {item["name"] for item in payload["groups"]["model"]}
    assert {"mmdit_576", "mmdit_1024"}.issubset(model_names)

    mmdit = next(item for item in payload["groups"]["model"] if item["name"] == "mmdit_576")
    assert mmdit["kind"] == "model"
    assert mmdit["path"].endswith("configs/presets/model/mmdit_576.kdl")
    assert "family=mmdit" in mmdit["summary"]
    assert 'preset kind="model" name="mmdit_576"' in mmdit["content"]


def test_active_config_presets_are_extracted_from_train_config() -> None:
    active = config_service.extract_used_presets(
        """
        config target="train" version=2 {
          model { use "mmdit_576" }
          training {
            use "bf16_adamw"
            use "single_gpu_debug"
          }
          data { use "latent_cache_576" }
        }
        """
    )

    assert active == {
        "model": ["mmdit_576"],
        "training": ["bf16_adamw", "single_gpu_debug"],
        "data": ["latent_cache_576"],
    }
