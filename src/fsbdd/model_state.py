from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .fragment_map import FragmentMap, validate_fragment_map
from .global_state import BootstrapFragment, FragmentStateDescriptor
from .identity import canonical_bytes, canonical_digest
from .model_registry import LogicalLayerRegistry


class ModelStateError(ValueError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenModelState:
    config_identity: str
    model_identity: str
    fragments: tuple[BootstrapFragment, ...]


def freeze_hf_model_fragments(
    model: Any,
    registry: LogicalLayerRegistry,
    fragment_map: FragmentMap,
) -> FrozenModelState:
    """Serialize every uniquely owned trainable parameter into one fragment blob.

    Imports of torch-backed safetensors are intentionally lazy so filesystem-only
    roles do not initialize a GPU framework.
    """

    validate_fragment_map(fragment_map, registry)
    try:
        from safetensors.torch import save
    except ImportError as error:  # pragma: no cover - dependency contract
        raise ModelStateError(
            "safetensors is required to freeze model fragments"
        ) from error
    config = getattr(model, "config", None)
    if config is None or not callable(getattr(config, "to_json_string", None)):
        raise ModelStateError("Hugging Face model must expose config.to_json_string()")
    try:
        config_value = json.loads(config.to_json_string(use_diff=False))
        config_identity = canonical_digest(config_value)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelStateError("model config is not canonically serializable") from error
    try:
        named_parameters = dict(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    except (AttributeError, TypeError) as error:
        raise ModelStateError(
            "model must expose named_parameters(remove_duplicate=False)"
        ) from error
    records = {record.identity: record for record in registry.parameters}
    fragments: list[BootstrapFragment] = []
    fragment_hashes: list[str] = []
    serialized_names: set[str] = set()
    for fragment in fragment_map.fragments:
        tensors: dict[str, Any] = {}
        for parameter_identity in fragment.parameter_identities:
            try:
                record = records[parameter_identity]
                parameter = named_parameters[record.owner_name]
            except KeyError as error:
                raise ModelStateError(
                    f"frozen parameter is missing: {parameter_identity}"
                ) from error
            if record.owner_name in serialized_names:
                raise ModelStateError(
                    f"parameter owner was serialized twice: {record.owner_name}"
                )
            try:
                tensor = parameter.detach().cpu().contiguous()
            except AttributeError as error:
                raise ModelStateError(
                    f"parameter is not tensor-like: {record.owner_name}"
                ) from error
            if (
                tuple(int(item) for item in tensor.shape) != record.shape
                or str(tensor.dtype) != record.dtype
            ):
                raise ModelStateError(
                    f"parameter metadata changed after registry freeze: {record.owner_name}"
                )
            tensors[record.owner_name] = tensor
            serialized_names.add(record.owner_name)
        try:
            parameters = save(tensors)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ModelStateError(
                f"failed to serialize fragment {fragment.index}"
            ) from error
        outer_state = canonical_bytes(
            {
                "schema_version": 1,
                "outer_update_count": 0,
                "parameter_identities": fragment.parameter_identities,
            }
        )
        descriptor = FragmentStateDescriptor(
            index=fragment.index,
            identity=canonical_digest(
                {
                    "fragment_map_identity": fragment_map.digest,
                    "fragment": fragment.to_dict(),
                }
            ),
            dtype="safetensors",
            shape=(fragment.parameter_count,),
            parameter_identities=fragment.parameter_identities,
        )
        fragments.append(BootstrapFragment(descriptor, parameters, outer_state))
        fragment_hashes.append(hashlib.sha256(parameters).hexdigest())
    if len(serialized_names) != len(registry.parameters):
        raise ModelStateError(
            "frozen fragments do not cover every uniquely owned trainable parameter"
        )
    model_identity = canonical_digest(
        {
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config_identity": config_identity,
            "registry_identity": registry.digest,
            "fragment_map_identity": fragment_map.digest,
            "fragment_parameter_sha256": fragment_hashes,
        }
    )
    return FrozenModelState(
        config_identity=config_identity,
        model_identity=model_identity,
        fragments=tuple(fragments),
    )
