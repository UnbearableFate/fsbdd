from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

from .identity import canonical_digest


class ConfigError(ValueError):
    pass


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _strict(raw: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{name} must be an object")
    missing = fields - raw.keys()
    unknown = raw.keys() - fields
    if missing:
        raise ConfigError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{name} has unknown fields: {sorted(unknown)}")
    return raw


def _positive_int(value: object, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        raise ConfigError(f"{field} must be {'non-negative' if allow_zero else 'positive'} integer")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    seed: int


@dataclasses.dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    S_max: int
    lambda_s: float
    Q: int
    Q_fresh: int
    grace_seconds: float


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleConfig:
    H: int
    offsets: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class AssetConfig:
    repository: str
    revision: str


@dataclasses.dataclass(frozen=True, slots=True)
class DataConfig:
    repository: str
    revision: str
    split: str


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentMapConfig:
    identity: str
    count: int


@dataclasses.dataclass(frozen=True, slots=True)
class StorageConfig:
    backend: str
    root: Path


@dataclasses.dataclass(frozen=True, slots=True)
class StopConfig:
    target_global_cycles: int
    max_walltime_seconds: int


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedConfig:
    schema_version: int
    run: RunConfig
    algorithm: AlgorithmConfig
    schedule: ScheduleConfig
    model: AssetConfig
    data: DataConfig
    fragment_map: FragmentMapConfig
    storage: StorageConfig
    stop: StopConfig
    digest: str
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.raw)


@dataclasses.dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    learner_local_steps: dict[str, int]
    learner_tokens: dict[str, int]
    fragment_outer_updates: dict[str, int]
    global_cycle: int
    fragment_accepted_tokens: dict[str, int]
    fresh_accepted: int
    stale_accepted: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_TOP = {"schema_version", "run", "algorithm", "schedule", "model", "data", "fragment_map", "storage", "stop"}


def load_config(path: Path, *, environ: dict[str, str] | None = None) -> ResolvedConfig:
    env = os.environ if environ is None else environ
    conflicts = sorted(key for key in env if key.startswith("FSBDD_CONFIG_"))
    if conflicts:
        raise ConfigError(f"environment overrides are forbidden for frozen config: {conflicts}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load config: {error}") from error
    top = _strict(raw, "config", _TOP)
    if top["schema_version"] != 1:
        raise ConfigError("unsupported config schema_version")

    run = _strict(top["run"], "run", {"name", "seed"})
    algorithm = _strict(top["algorithm"], "algorithm", {"S_max", "lambda_s", "Q", "Q_fresh", "grace_seconds"})
    schedule = _strict(top["schedule"], "schedule", {"H", "offsets"})
    model = _strict(top["model"], "model", {"repository", "revision"})
    data = _strict(top["data"], "data", {"repository", "revision", "split"})
    fragment = _strict(top["fragment_map"], "fragment_map", {"identity", "count"})
    storage = _strict(top["storage"], "storage", {"backend", "root"})
    stop = _strict(top["stop"], "stop", {"target_global_cycles", "max_walltime_seconds"})

    S_max = _positive_int(algorithm["S_max"], "algorithm.S_max", allow_zero=True)
    Q = _positive_int(algorithm["Q"], "algorithm.Q")
    Q_fresh = _positive_int(algorithm["Q_fresh"], "algorithm.Q_fresh", allow_zero=True)
    if Q_fresh > Q:
        raise ConfigError("algorithm.Q_fresh cannot exceed Q")
    lambda_s = algorithm["lambda_s"]
    grace = algorithm["grace_seconds"]
    if not isinstance(lambda_s, (int, float)) or isinstance(lambda_s, bool) or lambda_s < 0:
        raise ConfigError("algorithm.lambda_s must be non-negative")
    if not isinstance(grace, (int, float)) or isinstance(grace, bool) or grace < 0:
        raise ConfigError("algorithm.grace_seconds must be non-negative")
    H = _positive_int(schedule["H"], "schedule.H")
    offsets_raw = schedule["offsets"]
    if not isinstance(offsets_raw, list) or not offsets_raw:
        raise ConfigError("schedule.offsets must be a non-empty list")
    offsets = tuple(_positive_int(value, "schedule.offset", allow_zero=True) for value in offsets_raw)
    if any(offset >= H for offset in offsets):
        raise ConfigError("schedule offsets must be less than H")
    count = _positive_int(fragment["count"], "fragment_map.count")
    if len(offsets) != count:
        raise ConfigError("schedule.offsets length must equal fragment_map.count")
    identity = _nonempty(fragment["identity"], "fragment_map.identity")
    if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ConfigError("fragment_map.identity must be a lowercase SHA-256")

    frozen_raw = json.loads(json.dumps(raw, sort_keys=True))
    return ResolvedConfig(
        schema_version=1,
        run=RunConfig(_nonempty(run["name"], "run.name"), _positive_int(run["seed"], "run.seed", allow_zero=True)),
        algorithm=AlgorithmConfig(S_max, float(lambda_s), Q, Q_fresh, float(grace)),
        schedule=ScheduleConfig(H, offsets),
        model=AssetConfig(_nonempty(model["repository"], "model.repository"), _nonempty(model["revision"], "model.revision")),
        data=DataConfig(_nonempty(data["repository"], "data.repository"), _nonempty(data["revision"], "data.revision"), _nonempty(data["split"], "data.split")),
        fragment_map=FragmentMapConfig(identity, count),
        storage=StorageConfig(_nonempty(storage["backend"], "storage.backend"), Path(_nonempty(storage["root"], "storage.root"))),
        stop=StopConfig(_positive_int(stop["target_global_cycles"], "stop.target_global_cycles"), _positive_int(stop["max_walltime_seconds"], "stop.max_walltime_seconds")),
        digest=canonical_digest(frozen_raw),
        raw=_deep_freeze(frozen_raw),
    )


def freeze_config(config: ResolvedConfig, destination: Path) -> str:
    if destination.exists():
        raise ConfigError(f"refusing to overwrite frozen config: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return config.digest
