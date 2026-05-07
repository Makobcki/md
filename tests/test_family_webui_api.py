from __future__ import annotations

import pytest

from sample.api import SampleOptions, SampleValidationError
from webui.backend.app import validate_family_workflow


def test_sample_options_reject_var_diffusion_sampler() -> None:
    options = SampleOptions(
        ckpt="model.pt",
        out="tokens.pt",
        sampler="flow_euler",
        family="var",
        latent_only=True,
    )

    with pytest.raises(SampleValidationError, match="autoregressive"):
        options.validate()


def test_backend_family_workflow_gating() -> None:
    validate_family_workflow("mmdit", "inpaint", "flow_heun")
    with pytest.raises(ValueError, match="img2img"):
        validate_family_workflow("pixart_sigma", "img2img", "flow_heun")
    with pytest.raises(ValueError, match="diffusion"):
        validate_family_workflow("var", "txt2img", "flow_heun")
