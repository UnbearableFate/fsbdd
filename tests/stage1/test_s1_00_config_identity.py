from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsbdd.auxiliary.contracts.config import ConfigError, ProgressSnapshot, load_config
from fsbdd.diloco.common.identity import canonical_digest, file_digest


def valid_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": {"name": "unit", "seed": 7},
        "algorithm": {
            "S_max": 0,
            "lambda_s": 1.0,
            "Q": 8,
            "Q_fresh": 8,
            "grace_seconds": 0.0,
        },
        "schedule": {"H": 50, "offsets": [0, 12, 25, 37]},
        "model": {"repository": "fixture/model", "revision": "abc"},
        "data": {"repository": "fixture/data", "revision": "def", "split": "train"},
        "fragment_map": {"identity": "f" * 64, "count": 4},
        "storage": {"backend": "posix", "root": "/tmp/fsbdd"},
        "stop": {"target_global_cycles": 10, "max_walltime_seconds": 120},
    }


def test_required_fields_unknown_fields_and_frozen_digest(tmp_path: Path) -> None:
    raw = valid_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    resolved = load_config(path)
    assert resolved.schedule.H == 50
    assert resolved.digest == canonical_digest(raw)

    for section, field in [
        ("algorithm", "S_max"),
        ("algorithm", "Q"),
        ("algorithm", "Q_fresh"),
        ("schedule", "H"),
        ("schedule", "offsets"),
        ("model", "repository"),
        ("data", "repository"),
        ("storage", "root"),
        ("stop", "target_global_cycles"),
        ("run", "seed"),
    ]:
        broken = json.loads(json.dumps(raw))
        del broken[section][field]
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    broken = json.loads(json.dumps(raw))
    broken["algorithm"]["surprise"] = 1
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown"):
        load_config(path)


def test_identity_mutations_and_canonical_order() -> None:
    base = valid_config()
    reordered = dict(reversed(list(base.items())))
    assert canonical_digest(base) == canonical_digest(reordered)
    for section, field, value in [
        ("model", "revision", "changed"),
        ("fragment_map", "identity", "e" * 64),
        ("algorithm", "Q", 7),
        ("data", "split", "validation"),
        ("storage", "root", "/other"),
        ("stop", "target_global_cycles", 11),
        ("run", "seed", 8),
    ]:
        changed = json.loads(json.dumps(base))
        changed[section][field] = value
        assert canonical_digest(base) != canonical_digest(changed)


def test_progress_dimensions_are_not_collapsed() -> None:
    snapshot = ProgressSnapshot(
        learner_local_steps={"learner-0": 10},
        learner_tokens={"learner-0": 2048},
        fragment_outer_updates={"fragment-0": 2},
        global_cycle=2,
        fragment_accepted_tokens={"fragment-0": 1024},
        fresh_accepted=1,
        stale_accepted=0,
    )
    assert "global_step" not in snapshot.to_dict()
    assert snapshot.to_dict()["global_cycle"] == 2


def test_resolved_config_is_deeply_frozen_and_freeze_digest_matches_bytes(
    tmp_path: Path,
) -> None:
    from fsbdd.auxiliary.contracts.config import freeze_config

    source = tmp_path / "source.json"
    source.write_text(json.dumps(valid_config()), encoding="utf-8")
    resolved = load_config(source)
    with pytest.raises(TypeError):
        resolved.raw["algorithm"]["S_max"] = 1
    destination = tmp_path / "resolved.json"
    returned = freeze_config(resolved, destination)
    assert returned == canonical_digest(
        json.loads(destination.read_text(encoding="utf-8"))
    )
    assert (
        file_digest(destination) != returned
    )  # semantic digest is canonical JSON, not presentation bytes
