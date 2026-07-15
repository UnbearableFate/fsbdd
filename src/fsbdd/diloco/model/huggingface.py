from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from fsbdd.diloco.model.learner_assets import FrozenLearnerProfile


class HuggingFaceModelError(RuntimeError):
    """A frozen Hugging Face model or optimizer contract was violated."""


def load_frozen_causal_lm(
    profile: FrozenLearnerProfile,
    inventory: Mapping[str, Any],
    device: Any,
) -> Any:
    """Load the profile-pinned causal LM without permitting network fallback."""

    import torch
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    model_root = inventory.get("model_root")
    if not isinstance(model_root, str) or not model_root:
        raise HuggingFaceModelError("verified model inventory has no model_root")
    transformers_logging.disable_progress_bar()
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        attn_implementation=str(profile.model["attention_implementation"]),
        dtype=torch.float32,
    )
    model.config.use_cache = bool(profile.model["use_cache"])
    model.loss_type = "ForCausalLM"
    model.to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    expected = int(profile.model["expected_parameter_count"])
    if count != expected:
        raise HuggingFaceModelError(
            f"model parameter count mismatch: expected {expected}, observed {count}"
        )
    return model


def build_frozen_adamw(profile: FrozenLearnerProfile, model: Any) -> Any:
    """Construct exactly the AdamW optimizer frozen in a validated profile."""

    import torch

    spec = profile.training["optimizer"]
    betas = spec.get("betas")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise HuggingFaceModelError("validated optimizer betas must contain two values")
    beta_pair = (float(betas[0]), float(betas[1]))
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["lr"]),
        betas=beta_pair,
        eps=float(spec["eps"]),
        weight_decay=float(spec["weight_decay"]),
    )


def model_parameter_digest(model: Any) -> str:
    """Hash named model parameters with stable metadata and fp32 CPU bytes."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        value = parameter.detach().float().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()
