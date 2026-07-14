from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .identity import canonical_digest


class RegistryError(ValueError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class MiscAssignment:
    module_path: str
    left_layer: int
    right_layer: int


@dataclasses.dataclass(frozen=True, slots=True)
class ExplicitMapping:
    family: str
    embedding_path: str
    block_paths: tuple[str, ...]
    head_path: str
    misc: tuple[MiscAssignment, ...] = ()
    reconstructable_buffers: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterRecord:
    identity: str
    owner_layer: str
    owner_name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    numel: int
    sync_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class BufferRecord:
    identity: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    numel: int
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LogicalLayer:
    index: int
    name: str
    kind: str
    module_path: str
    parameter_identities: tuple[str, ...]
    parameter_count: int
    sync_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Coverage:
    unique_trainable: int
    owned: int
    duplicate_owners: int
    unowned: int
    total_buffers: int
    reconstructable_buffers: int
    unclassified_buffers: int

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LogicalLayerRegistry:
    schema_version: int
    family: str
    sync_dtype_bytes: int
    layers: tuple[LogicalLayer, ...]
    parameters: tuple[ParameterRecord, ...]
    buffers: tuple[BufferRecord, ...]
    tied_identities: tuple[str, ...]
    misc_assignments: tuple[dict[str, Any], ...]
    coverage: Coverage
    digest: str

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "sync_dtype_bytes": self.sync_dtype_bytes,
            "layers": [layer.to_dict() for layer in self.layers],
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "buffers": [buffer.to_dict() for buffer in self.buffers],
            "tied_identities": list(self.tied_identities),
            "misc_assignments": list(self.misc_assignments),
            "coverage": self.coverage.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}


def _module(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as error:
                raise RegistryError(f"module path does not exist: {path}") from error
        else:
            if not hasattr(current, part):
                raise RegistryError(f"module path does not exist: {path}")
            current = getattr(current, part)
    return current


def _children_paths(root: Any, path: str) -> tuple[str, ...]:
    module = _module(root, path)
    try:
        children = list(module.named_children())
    except AttributeError as error:
        raise RegistryError(f"block container is not a module: {path}") from error
    if not children:
        raise RegistryError(f"block container is empty: {path}")
    return tuple(f"{path}.{name}" for name, _ in children)


def _builtin_mapping(model: Any) -> ExplicitMapping:
    config = getattr(model, "config", None)
    if config is None:
        raise RegistryError("model lacks config.model_type; provide explicit mapping")
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise RegistryError("encoder-decoder models are outside the supported scope")
    model_type = getattr(config, "model_type", None)
    if model_type == "gpt_neox":
        blocks = _children_paths(model, "gpt_neox.layers")
        return ExplicitMapping(
            family="gpt_neox",
            embedding_path="gpt_neox.embed_in",
            block_paths=blocks,
            head_path="embed_out",
            misc=(MiscAssignment("gpt_neox.final_layer_norm", len(blocks), len(blocks) + 1),),
            reconstructable_buffers=(
                "gpt_neox.rotary_emb.inv_freq",
                "gpt_neox.rotary_emb.original_inv_freq",
            ),
        )
    if model_type == "llama":
        blocks = _children_paths(model, "model.layers")
        return ExplicitMapping(
            family="llama",
            embedding_path="model.embed_tokens",
            block_paths=blocks,
            head_path="lm_head",
            misc=(MiscAssignment("model.norm", len(blocks), len(blocks) + 1),),
            reconstructable_buffers=(
                "model.rotary_emb.inv_freq",
                "model.rotary_emb.original_inv_freq",
            ),
        )
    raise RegistryError(f"unsupported or ambiguous model_type {model_type!r}; provide explicit mapping")


def _global_aliases(model: Any) -> tuple[dict[int, Any], dict[int, tuple[str, ...]]]:
    parameters: dict[int, Any] = {}
    aliases: dict[int, list[str]] = defaultdict(list)
    try:
        named = model.named_parameters(recurse=True, remove_duplicate=False)
    except (AttributeError, TypeError) as error:
        raise RegistryError("model must support named_parameters(remove_duplicate=False)") from error
    for name, parameter in named:
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        key = id(parameter)
        parameters[key] = parameter
        if name not in aliases[key]:
            aliases[key].append(name)
    return parameters, {key: tuple(names) for key, names in aliases.items()}


def _global_buffers(model: Any) -> tuple[dict[int, Any], dict[int, tuple[str, ...]]]:
    buffers: dict[int, Any] = {}
    aliases: dict[int, list[str]] = defaultdict(list)
    try:
        named = model.named_buffers(recurse=True, remove_duplicate=False)
    except (AttributeError, TypeError) as error:
        raise RegistryError("model must support named_buffers(remove_duplicate=False)") from error
    for name, buffer in named:
        key = id(buffer)
        buffers[key] = buffer
        if name not in aliases[key]:
            aliases[key].append(name)
    return buffers, {key: tuple(names) for key, names in aliases.items()}


def _under(name: str, module_path: str) -> bool:
    return name == module_path or name.startswith(module_path + ".")


def _parameter_bytes(parameter: Any, sync_dtype_bytes: int) -> int:
    try:
        return int(parameter.numel()) * sync_dtype_bytes
    except (AttributeError, TypeError, ValueError) as error:
        raise RegistryError("trainable parameter lacks a valid numel") from error


def _parameter_shape(parameter: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in parameter.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise RegistryError("trainable parameter lacks a concrete shape") from error


def validate_parameter_ownership(owners: list[tuple[str, str]]) -> None:
    counts = Counter(identity for _, identity in owners)
    duplicates = sorted(identity for identity, count in counts.items() if count != 1)
    if duplicates:
        raise RegistryError(f"duplicate parameter ownership: {duplicates}")


def build_logical_layer_registry(
    model: Any,
    *,
    explicit: ExplicitMapping | None = None,
    sync_dtype_bytes: int = 4,
) -> LogicalLayerRegistry:
    if not isinstance(sync_dtype_bytes, int) or isinstance(sync_dtype_bytes, bool) or sync_dtype_bytes <= 0:
        raise RegistryError("sync_dtype_bytes must be a positive integer")
    config = getattr(model, "config", None)
    if config is None:
        raise RegistryError("model lacks config; supported scope cannot be established")
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise RegistryError("encoder-decoder models are outside the supported scope")
    mapping = explicit or _builtin_mapping(model)
    if not mapping.family or not mapping.block_paths:
        raise RegistryError("logical mapping requires a family and at least one complete block")
    logical_paths = (mapping.embedding_path, *mapping.block_paths, mapping.head_path)
    if len(set(logical_paths)) != len(logical_paths):
        raise RegistryError("logical module paths must be unique")
    for path in logical_paths:
        _module(model, path)
    for assignment in mapping.misc:
        _module(model, assignment.module_path)
        if assignment.left_layer < 0 or assignment.right_layer >= len(logical_paths):
            raise RegistryError("misc adjacency index outside logical-layer range")
        if assignment.right_layer != assignment.left_layer + 1:
            raise RegistryError("misc assignment must name adjacent logical layers")

    parameters, aliases = _global_aliases(model)
    buffers, buffer_aliases = _global_buffers(model)
    allowed_buffers = set(mapping.reconstructable_buffers)
    if len(allowed_buffers) != len(mapping.reconstructable_buffers):
        raise RegistryError("reconstructable buffer declarations must be unique")
    unclassified_buffers = sorted(
        name for names in buffer_aliases.values() for name in names if name not in allowed_buffers
    )
    if unclassified_buffers:
        raise RegistryError(
            "unclassified buffers require an explicit reconstructable/static or mutable-state policy: "
            f"{unclassified_buffers}"
        )
    buffer_records = tuple(
        BufferRecord(
            identity=canonical_digest(
                {
                    "aliases": names,
                    "shape": _parameter_shape(buffers[key]),
                    "dtype": str(getattr(buffers[key], "dtype", "unknown")),
                    "classification": "reconstructable_from_config",
                }
            ),
            aliases=names,
            shape=_parameter_shape(buffers[key]),
            dtype=str(getattr(buffers[key], "dtype", "unknown")),
            numel=int(buffers[key].numel()),
            classification="reconstructable_from_config",
        )
        for key, names in sorted(buffer_aliases.items(), key=lambda item: item[1][0])
    )
    owner_index: dict[int, int] = {}
    base_candidates: dict[int, set[int]] = defaultdict(set)
    for key, names in aliases.items():
        for index, path in enumerate(logical_paths):
            if any(_under(name, path) for name in names):
                base_candidates[key].add(index)
        candidates = base_candidates[key]
        if candidates:
            if 0 in candidates:
                # MODEL-04: embedding is the single owner for tied input/head storage.
                if candidates - {0, len(logical_paths) - 1}:
                    raise RegistryError(f"parameter shared across unsupported logical layers: {names}")
                owner_index[key] = 0
            elif len(candidates) == 1:
                owner_index[key] = next(iter(candidates))
            else:
                raise RegistryError(f"parameter shared across unsupported logical layers: {names}")

    layer_bytes = [0 for _ in logical_paths]
    for key, index in owner_index.items():
        layer_bytes[index] += _parameter_bytes(parameters[key], sync_dtype_bytes)

    misc_report: list[dict[str, Any]] = []
    for assignment in mapping.misc:
        misc_keys = [
            key for key, names in aliases.items() if any(_under(name, assignment.module_path) for name in names)
        ]
        for key in misc_keys:
            if key in owner_index:
                raise RegistryError(f"misc module overlaps an existing logical owner: {aliases[key]}")
        left_bytes = layer_bytes[assignment.left_layer]
        right_bytes = layer_bytes[assignment.right_layer]
        selected = assignment.left_layer if left_bytes <= right_bytes else assignment.right_layer
        for key in misc_keys:
            owner_index[key] = selected
            layer_bytes[selected] += _parameter_bytes(parameters[key], sync_dtype_bytes)
        misc_report.append(
            {
                "module_path": assignment.module_path,
                "left_layer": assignment.left_layer,
                "right_layer": assignment.right_layer,
                "left_bytes_before": left_bytes,
                "right_bytes_before": right_bytes,
                "selected_layer": selected,
            }
        )

    unowned = sorted(aliases[key][0] for key in parameters if key not in owner_index)
    if unowned:
        raise RegistryError(f"unowned trainable parameters require explicit mapping: {unowned}")

    layer_names = ("embedding", *(f"block-{index}" for index in range(len(mapping.block_paths))), "lm_head")
    kinds = ("embedding", *("block" for _ in mapping.block_paths), "lm_head")
    records: list[ParameterRecord] = []
    identities_by_layer: dict[int, list[str]] = defaultdict(list)
    tied: list[str] = []
    for key in sorted(parameters, key=lambda item: aliases[item][0]):
        parameter = parameters[key]
        names = aliases[key]
        owner = owner_index[key]
        shape = _parameter_shape(parameter)
        dtype = str(getattr(parameter, "dtype", "unknown"))
        identity = canonical_digest({"aliases": names, "shape": shape, "dtype": dtype})
        record = ParameterRecord(
            identity=identity,
            owner_layer=layer_names[owner],
            owner_name=names[0],
            aliases=names,
            shape=shape,
            dtype=dtype,
            numel=int(parameter.numel()),
            sync_bytes=_parameter_bytes(parameter, sync_dtype_bytes),
        )
        records.append(record)
        identities_by_layer[owner].append(identity)
        if len(names) > 1:
            tied.append(identity)
    validate_parameter_ownership([(record.owner_layer, record.identity) for record in records])

    layers = tuple(
        LogicalLayer(
            index=index,
            name=layer_names[index],
            kind=kinds[index],
            module_path=logical_paths[index],
            parameter_identities=tuple(identities_by_layer[index]),
            parameter_count=sum(record.numel for record in records if record.owner_layer == layer_names[index]),
            sync_bytes=sum(record.sync_bytes for record in records if record.owner_layer == layer_names[index]),
        )
        for index in range(len(logical_paths))
    )
    coverage = Coverage(
        unique_trainable=len(parameters),
        owned=len(records),
        duplicate_owners=0,
        unowned=0,
        total_buffers=len(buffers),
        reconstructable_buffers=len(buffer_records),
        unclassified_buffers=0,
    )
    body = {
        "schema_version": 1,
        "family": mapping.family,
        "sync_dtype_bytes": sync_dtype_bytes,
        "layers": [layer.to_dict() for layer in layers],
        "parameters": [record.to_dict() for record in records],
        "buffers": [buffer.to_dict() for buffer in buffer_records],
        "tied_identities": sorted(tied),
        "misc_assignments": misc_report,
        "coverage": coverage.to_dict(),
    }
    return LogicalLayerRegistry(
        schema_version=1,
        family=mapping.family,
        sync_dtype_bytes=sync_dtype_bytes,
        layers=layers,
        parameters=tuple(records),
        buffers=buffer_records,
        tied_identities=tuple(sorted(tied)),
        misc_assignments=tuple(misc_report),
        coverage=coverage,
        digest=canonical_digest(body),
    )


def write_registry_report(registry: LogicalLayerRegistry, destination: Path) -> str:
    if destination.exists():
        raise RegistryError(f"refusing to overwrite registry report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(registry.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return registry.digest
