from __future__ import annotations

from model.registry import MODEL_REGISTRY, model_family


def test_model_registry_uses_semantic_family() -> None:
    assert "mmdit" in MODEL_REGISTRY
    assert "flux_like" in MODEL_REGISTRY
    assert model_family({"model": {"family": "mmdit", "variant": "576"}}) == "mmdit"
