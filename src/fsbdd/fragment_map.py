from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .identity import canonical_digest
from .model_registry import LogicalLayerRegistry


class FragmentMapError(ValueError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentLayerSummary:
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
class MiscOwnershipSummary:
    module_path: str
    left_layer: int
    right_layer: int
    left_bytes_before: int
    right_bytes_before: int
    selected_layer: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Fragment:
    index: int
    start_layer: int
    end_layer_exclusive: int
    layer_indices: tuple[int, ...]
    layer_names: tuple[str, ...]
    parameter_identities: tuple[str, ...]
    parameter_count: int
    sync_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentObjective:
    maximum_sync_bytes: int
    total_deviation_scaled: int
    cut_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class BalanceSummary:
    total_sync_bytes: int
    minimum_sync_bytes: int
    maximum_sync_bytes: int
    average_sync_bytes: float
    maximum_to_average_ratio: float
    average_exact_numerator: int
    average_exact_denominator: int
    ratio_exact_numerator: int
    ratio_exact_denominator: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentMap:
    schema_version: int
    registry_digest: str
    fragment_count: int
    production_mode: bool
    sync_dtype_bytes: int
    layer_count: int
    layers: tuple[FragmentLayerSummary, ...]
    fragments: tuple[Fragment, ...]
    tied_identities: tuple[str, ...]
    misc_assignments: tuple[MiscOwnershipSummary, ...]
    objective: FragmentObjective
    balance: BalanceSummary
    digest: str

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_digest": self.registry_digest,
            "fragment_count": self.fragment_count,
            "production_mode": self.production_mode,
            "sync_dtype_bytes": self.sync_dtype_bytes,
            "layer_count": self.layer_count,
            "layers": [layer.to_dict() for layer in self.layers],
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "tied_identities": list(self.tied_identities),
            "misc_assignments": [assignment.to_dict() for assignment in self.misc_assignments],
            "objective": self.objective.to_dict(),
            "balance": self.balance.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}


def _validate_fragment_count(fragment_count: int, layer_count: int, production: bool) -> None:
    if not isinstance(fragment_count, int) or isinstance(fragment_count, bool):
        raise FragmentMapError("fragment_count must be an integer")
    if not isinstance(production, bool):
        raise FragmentMapError("production must be a boolean")
    if layer_count < 2:
        raise FragmentMapError("fragment mapping requires at least two logical layers")
    if production:
        if not 1 < fragment_count < layer_count:
            raise FragmentMapError("production fragment_count must satisfy 1 < F < L")
    elif fragment_count != 1:
        raise FragmentMapError("nonproduction oracle mode permits only F = 1")


def _minimum_maximum(weights: tuple[int, ...], fragment_count: int) -> int:
    layer_count = len(weights)
    infinity = sum(weights) + 1
    previous = [infinity] * (layer_count + 1)
    previous[0] = 0
    for fragments_used in range(1, fragment_count + 1):
        current = [infinity] * (layer_count + 1)
        minimum_end = fragments_used
        maximum_end = layer_count - (fragment_count - fragments_used)
        for end in range(minimum_end, maximum_end + 1):
            segment_bytes = 0
            for start in range(end - 1, fragments_used - 2, -1):
                segment_bytes += weights[start]
                if previous[start] == infinity:
                    continue
                candidate = max(previous[start], segment_bytes)
                if candidate < current[end]:
                    current[end] = candidate
        previous = current
    result = previous[layer_count]
    if result == infinity:
        raise FragmentMapError("no feasible nonempty contiguous partition")
    return result


def partition_layer_bytes(weights: tuple[int, ...], fragment_count: int) -> FragmentObjective:
    if not weights or any(not isinstance(weight, int) or isinstance(weight, bool) or weight < 0 for weight in weights):
        raise FragmentMapError("layer synchronization bytes must be nonnegative integers")
    if not isinstance(fragment_count, int) or isinstance(fragment_count, bool):
        raise FragmentMapError("fragment_count must be an integer")
    if not 1 <= fragment_count <= len(weights):
        raise FragmentMapError("partition requires 1 <= F <= L")
    total_bytes = sum(weights)
    if total_bytes <= 0:
        raise FragmentMapError("fragment mapping requires positive total synchronization bytes")
    maximum = _minimum_maximum(weights, fragment_count)

    # With the minimax cap frozen, the remaining objective is additive. Each
    # state stores the exact scaled deviation and lexicographically earliest
    # exclusive cuts, avoiding floats in optimization and tie-breaking.
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    layer_count = len(weights)
    for fragments_used in range(1, fragment_count + 1):
        minimum_end = fragments_used
        maximum_end = layer_count - (fragment_count - fragments_used)
        for end in range(minimum_end, maximum_end + 1):
            best: tuple[int, tuple[int, ...]] | None = None
            for start in range(fragments_used - 1, end):
                previous = states.get((fragments_used - 1, start))
                if previous is None:
                    continue
                segment_bytes = prefix[end] - prefix[start]
                if segment_bytes > maximum:
                    continue
                deviation = previous[0] + abs(fragment_count * segment_bytes - total_bytes)
                cuts = previous[1] + ((end,) if fragments_used < fragment_count else ())
                candidate = (deviation, cuts)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                states[(fragments_used, end)] = best
    try:
        deviation, cuts = states[(fragment_count, layer_count)]
    except KeyError as error:
        raise FragmentMapError("no partition satisfies the computed minimax bound") from error
    return FragmentObjective(maximum, deviation, cuts)


def _validate_registry(registry: LogicalLayerRegistry) -> None:
    if registry.digest != canonical_digest(registry._body()):
        raise FragmentMapError("logical-layer registry digest mismatch")
    if (
        not isinstance(registry.sync_dtype_bytes, int)
        or isinstance(registry.sync_dtype_bytes, bool)
        or registry.sync_dtype_bytes <= 0
    ):
        raise FragmentMapError("registry sync_dtype_bytes must be a positive integer")
    layers = registry.layers
    if tuple(layer.index for layer in layers) != tuple(range(len(layers))):
        raise FragmentMapError("logical-layer indices must be contiguous and ordered")
    parameter_records = {record.identity: record for record in registry.parameters}
    if len(parameter_records) != len(registry.parameters):
        raise FragmentMapError("registry parameter identities must be unique")
    flattened = [identity for layer in layers for identity in layer.parameter_identities]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(parameter_records):
        raise FragmentMapError("logical layers must cover every parameter identity exactly once")
    if (
        registry.coverage.owned != len(parameter_records)
        or registry.coverage.unique_trainable != len(parameter_records)
        or registry.coverage.duplicate_owners != 0
        or registry.coverage.unowned != 0
    ):
        raise FragmentMapError("registry trainable-parameter coverage is incomplete")
    for record in registry.parameters:
        if record.numel < 0 or record.sync_bytes != record.numel * registry.sync_dtype_bytes:
            raise FragmentMapError(f"registry parameter synchronization bytes mismatch: {record.owner_name}")
    for layer in layers:
        records = [parameter_records[identity] for identity in layer.parameter_identities]
        if layer.parameter_count != sum(record.numel for record in records):
            raise FragmentMapError(f"logical-layer parameter count mismatch: {layer.name}")
        if layer.sync_bytes != sum(record.sync_bytes for record in records):
            raise FragmentMapError(f"logical-layer synchronization bytes mismatch: {layer.name}")


def _misc_summaries(registry: LogicalLayerRegistry) -> tuple[MiscOwnershipSummary, ...]:
    expected = {
        "module_path",
        "left_layer",
        "right_layer",
        "left_bytes_before",
        "right_bytes_before",
        "selected_layer",
    }
    summaries: list[MiscOwnershipSummary] = []
    for assignment in registry.misc_assignments:
        if set(assignment) != expected:
            raise FragmentMapError("registry misc assignment has an unsupported schema")
        try:
            summaries.append(MiscOwnershipSummary(**assignment))
        except TypeError as error:
            raise FragmentMapError("registry misc assignment is malformed") from error
    return tuple(summaries)


def build_fragment_map(
    registry: LogicalLayerRegistry,
    fragment_count: int,
    *,
    production: bool = True,
) -> FragmentMap:
    _validate_registry(registry)
    _validate_fragment_count(fragment_count, len(registry.layers), production)
    weights = tuple(layer.sync_bytes for layer in registry.layers)
    objective = partition_layer_bytes(weights, fragment_count)
    boundaries = (0, *objective.cut_indices, len(registry.layers))
    layer_summaries = tuple(
        FragmentLayerSummary(
            index=layer.index,
            name=layer.name,
            kind=layer.kind,
            module_path=layer.module_path,
            parameter_identities=layer.parameter_identities,
            parameter_count=layer.parameter_count,
            sync_bytes=layer.sync_bytes,
        )
        for layer in registry.layers
    )
    fragments: list[Fragment] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        selected = layer_summaries[start:end]
        fragments.append(
            Fragment(
                index=index,
                start_layer=start,
                end_layer_exclusive=end,
                layer_indices=tuple(layer.index for layer in selected),
                layer_names=tuple(layer.name for layer in selected),
                parameter_identities=tuple(
                    identity for layer in selected for identity in layer.parameter_identities
                ),
                parameter_count=sum(layer.parameter_count for layer in selected),
                sync_bytes=sum(layer.sync_bytes for layer in selected),
            )
        )
    fragment_bytes = tuple(fragment.sync_bytes for fragment in fragments)
    total_bytes = sum(fragment_bytes)
    balance = BalanceSummary(
        total_sync_bytes=total_bytes,
        minimum_sync_bytes=min(fragment_bytes),
        maximum_sync_bytes=max(fragment_bytes),
        average_sync_bytes=total_bytes / fragment_count,
        maximum_to_average_ratio=max(fragment_bytes) * fragment_count / total_bytes,
        average_exact_numerator=total_bytes,
        average_exact_denominator=fragment_count,
        ratio_exact_numerator=max(fragment_bytes) * fragment_count,
        ratio_exact_denominator=total_bytes,
    )
    body = {
        "schema_version": 1,
        "registry_digest": registry.digest,
        "fragment_count": fragment_count,
        "production_mode": production,
        "sync_dtype_bytes": registry.sync_dtype_bytes,
        "layer_count": len(layer_summaries),
        "layers": [layer.to_dict() for layer in layer_summaries],
        "fragments": [fragment.to_dict() for fragment in fragments],
        "tied_identities": list(registry.tied_identities),
        "misc_assignments": [assignment.to_dict() for assignment in _misc_summaries(registry)],
        "objective": objective.to_dict(),
        "balance": balance.to_dict(),
    }
    result = FragmentMap(
        schema_version=1,
        registry_digest=registry.digest,
        fragment_count=fragment_count,
        production_mode=production,
        sync_dtype_bytes=registry.sync_dtype_bytes,
        layer_count=len(layer_summaries),
        layers=layer_summaries,
        fragments=tuple(fragments),
        tied_identities=registry.tied_identities,
        misc_assignments=_misc_summaries(registry),
        objective=objective,
        balance=balance,
        digest=canonical_digest(body),
    )
    validate_fragment_map(result, registry)
    return result


def validate_fragment_map(fragment_map: FragmentMap, registry: LogicalLayerRegistry) -> None:
    _validate_registry(registry)
    if fragment_map.schema_version != 1:
        raise FragmentMapError("unsupported fragment map schema_version")
    if fragment_map.digest != canonical_digest(fragment_map._body()):
        raise FragmentMapError("fragment map digest mismatch")
    if fragment_map.registry_digest != registry.digest:
        raise FragmentMapError("fragment map references a different logical-layer registry")
    if fragment_map.sync_dtype_bytes != registry.sync_dtype_bytes:
        raise FragmentMapError("fragment map sync_dtype_bytes differs from its registry")
    if fragment_map.layer_count != len(registry.layers):
        raise FragmentMapError("fragment map layer count mismatch")
    expected_layers = tuple(
        FragmentLayerSummary(
            index=layer.index,
            name=layer.name,
            kind=layer.kind,
            module_path=layer.module_path,
            parameter_identities=layer.parameter_identities,
            parameter_count=layer.parameter_count,
            sync_bytes=layer.sync_bytes,
        )
        for layer in registry.layers
    )
    if fragment_map.layers != expected_layers:
        raise FragmentMapError("fragment map layer summaries differ from its registry")
    if fragment_map.tied_identities != registry.tied_identities:
        raise FragmentMapError("fragment map tied ownership differs from its registry")
    if fragment_map.misc_assignments != _misc_summaries(registry):
        raise FragmentMapError("fragment map misc ownership differs from its registry")
    if fragment_map.fragment_count != len(fragment_map.fragments):
        raise FragmentMapError("fragment map fragment count mismatch")
    _validate_fragment_count(fragment_map.fragment_count, fragment_map.layer_count, fragment_map.production_mode)
    expected_start = 0
    parameter_identities: list[str] = []
    for index, fragment in enumerate(fragment_map.fragments):
        if fragment.index != index or fragment.start_layer != expected_start:
            raise FragmentMapError("fragment intervals are not ordered and contiguous")
        if fragment.end_layer_exclusive <= fragment.start_layer:
            raise FragmentMapError("fragment intervals must be nonempty")
        selected = fragment_map.layers[fragment.start_layer : fragment.end_layer_exclusive]
        if fragment.layer_indices != tuple(layer.index for layer in selected):
            raise FragmentMapError("fragment layer indices do not match its interval")
        if fragment.layer_names != tuple(layer.name for layer in selected):
            raise FragmentMapError("fragment layer names do not match its interval")
        expected_parameters = tuple(identity for layer in selected for identity in layer.parameter_identities)
        if fragment.parameter_identities != expected_parameters:
            raise FragmentMapError("fragment parameter identities do not match its complete layers")
        if fragment.parameter_count != sum(layer.parameter_count for layer in selected):
            raise FragmentMapError("fragment parameter count mismatch")
        if fragment.sync_bytes != sum(layer.sync_bytes for layer in selected):
            raise FragmentMapError("fragment synchronization byte count mismatch")
        parameter_identities.extend(fragment.parameter_identities)
        expected_start = fragment.end_layer_exclusive
    if expected_start != fragment_map.layer_count:
        raise FragmentMapError("fragment intervals do not cover every logical layer")
    registry_identities = [record.identity for record in registry.parameters]
    if len(parameter_identities) != len(set(parameter_identities)) or set(parameter_identities) != set(
        registry_identities
    ):
        raise FragmentMapError("every registry parameter must belong to exactly one fragment")
    fragment_bytes = tuple(fragment.sync_bytes for fragment in fragment_map.fragments)
    expected_objective = partition_layer_bytes(
        tuple(layer.sync_bytes for layer in fragment_map.layers), fragment_map.fragment_count
    )
    actual_cuts = tuple(fragment.end_layer_exclusive for fragment in fragment_map.fragments[:-1])
    actual_objective = FragmentObjective(
        maximum_sync_bytes=max(fragment_bytes),
        total_deviation_scaled=sum(
            abs(fragment_map.fragment_count * size - sum(fragment_bytes)) for size in fragment_bytes
        ),
        cut_indices=actual_cuts,
    )
    if fragment_map.objective != actual_objective:
        raise FragmentMapError("fragment map objective does not describe its actual cuts and bytes")
    if actual_objective != expected_objective:
        raise FragmentMapError("fragment map actual cuts do not satisfy the frozen optimal objective")
    if max(fragment_bytes) != fragment_map.balance.maximum_sync_bytes:
        raise FragmentMapError("fragment balance maximum mismatch")
    if min(fragment_bytes) != fragment_map.balance.minimum_sync_bytes:
        raise FragmentMapError("fragment balance minimum mismatch")
    if sum(fragment_bytes) != fragment_map.balance.total_sync_bytes:
        raise FragmentMapError("fragment balance total mismatch")
    total_bytes = sum(fragment_bytes)
    maximum_bytes = max(fragment_bytes)
    expected_balance = BalanceSummary(
        total_sync_bytes=total_bytes,
        minimum_sync_bytes=min(fragment_bytes),
        maximum_sync_bytes=maximum_bytes,
        average_sync_bytes=total_bytes / fragment_map.fragment_count,
        maximum_to_average_ratio=maximum_bytes * fragment_map.fragment_count / total_bytes,
        average_exact_numerator=total_bytes,
        average_exact_denominator=fragment_map.fragment_count,
        ratio_exact_numerator=maximum_bytes * fragment_map.fragment_count,
        ratio_exact_denominator=total_bytes,
    )
    if fragment_map.balance != expected_balance:
        raise FragmentMapError("fragment balance summary is not canonical")


def write_fragment_map_report(fragment_map: FragmentMap, destination: Path) -> str:
    if destination.exists():
        raise FragmentMapError(f"refusing to overwrite fragment map report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(fragment_map.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return fragment_map.digest
