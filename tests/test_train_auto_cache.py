from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
import train.runner as runner


def test_train_auto_prepares_missing_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = TrainConfig(
        data_root=str(tmp_path),
        text_cache_dir=".cache/text",
        latent_cache_dir=".cache/latents",
        latent_cache=True,
        latent_cache_sharded=True,
        vae_pretrained="./vae_sd_mse",
        cache_auto_prepare=True,
    )
    entries = [{"md5": "abc123"}]
    calls: list[str] = []

    class _MissingPath:
        def __init__(self, exists: bool) -> None:
            self._exists = exists

        def exists(self) -> bool:
            return self._exists

    class _FakeTextCache:
        created = 0

        def __init__(self, root: str | Path, *, shard_cache_size: int = 2) -> None:
            type(self).created += 1
            ready = type(self).created > 1
            self.root = Path(root)
            self.index_path = _MissingPath(ready)
            self.metadata_path = _MissingPath(ready)
            self.entries = {"abc123": object()} if ready else {}
            self.metadata = {"text_dim": cfg.text_dim, "pooled_dim": cfg.pooled_dim} if ready else {}

    def fake_prepare_text_cache(**kwargs) -> None:
        assert kwargs["cfg"] is cfg
        calls.append("text")

    state_calls = {"n": 0}

    def fake_latent_cache_state(_cfg: TrainConfig, _entries: list[dict]) -> tuple[str, str]:
        state_calls["n"] += 1
        if state_calls["n"] == 1:
            return "missing", "missing sharded latent index"
        return "ready", ""

    def fake_prepare_latent_cache_for_config(_cfg: TrainConfig, *, overwrite: bool | None = None) -> None:
        assert _cfg is cfg
        assert overwrite is False
        calls.append("latent")

    import scripts.prepare_text_cache as prepare_text_module
    import scripts.prepare_latents as prepare_latents_module

    monkeypatch.setattr(runner, "TextCache", _FakeTextCache)
    monkeypatch.setattr(runner, "_validate_text_cache_for_mmdit", lambda *args, **kwargs: None)
    monkeypatch.setattr(prepare_text_module, "prepare_text_cache", fake_prepare_text_cache)
    monkeypatch.setattr(runner, "_latent_cache_state", fake_latent_cache_state)
    monkeypatch.setattr(prepare_latents_module, "prepare_latent_cache_for_config", fake_prepare_latent_cache_for_config)

    runner._ensure_mmdit_caches_ready(cfg, entries, torch.device("cpu"))

    assert calls == ["text", "latent"]
