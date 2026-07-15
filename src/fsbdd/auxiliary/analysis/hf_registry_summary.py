from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fsbdd.diloco.common.identity import canonical_digest
from fsbdd.diloco.model.model_registry import (
    RegistryError,
    build_logical_layer_registry,
)


def summarize(model_repository: str, revision: str, output: Path) -> dict[str, Any]:
    # Heavy framework imports stay behind the explicit compute-node command.
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    if output.exists():
        raise RegistryError(f"refusing to overwrite HF registry summary: {output}")
    config = AutoConfig.from_pretrained(model_repository, revision=revision)
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise RegistryError("encoder-decoder configs are outside the supported scope")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    registry = build_logical_layer_registry(model)
    # Hugging Face config dicts may contain integer label-map keys. Its JSON
    # representation is the portable identity surface and normalizes them.
    config_body = json.loads(config.to_json_string(use_diff=False))
    summary = {
        "schema_version": 1,
        "model": {
            "repository": model_repository,
            "requested_revision": revision,
            "resolved_revision": getattr(config, "_commit_hash", None),
            "model_type": config.model_type,
            "config_sha256": canonical_digest(config_body),
            "parameter_count": sum(record.numel for record in registry.parameters),
        },
        "registry": registry.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fsbdd.hf_registry_summary")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = summarize(args.model, args.revision, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "model": summary["model"],
                "registry_digest": summary["registry"]["digest"],
                "logical_layers": len(summary["registry"]["layers"]),
                "unique_parameters": summary["registry"]["coverage"][
                    "unique_trainable"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
