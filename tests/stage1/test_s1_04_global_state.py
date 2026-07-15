from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from fsbdd.cli import main
from fsbdd.global_state import (
    BootstrapFragment,
    BootstrapInterrupted,
    CountingStorageBackend,
    FragmentStateDescriptor,
    GlobalStateError,
    GlobalStateIdentities,
    GlobalStateStore,
    decode_global_state,
    encode_global_state,
)
from fsbdd.storage import PosixStorageBackend, PublicationError, PublicationNotFound


def identities(**changes) -> GlobalStateIdentities:
    value = GlobalStateIdentities(
        run_identity="run-s1-04",
        config_identity="c" * 64,
        model_identity="d" * 64,
        fragment_map_identity="a" * 64,
    )
    return dataclasses.replace(value, **changes)


def fragments(
    count: int = 4, *, payload_bytes: int = 32
) -> tuple[BootstrapFragment, ...]:
    return tuple(
        BootstrapFragment(
            descriptor=FragmentStateDescriptor(
                index=index,
                identity=f"fragment-{index}",
                dtype="uint8",
                shape=(payload_bytes,),
                parameter_identities=(f"parameter-{index}",),
            ),
            parameters=bytes([index + 1]) * payload_bytes,
            outer_state=json.dumps({"momentum": index}, sort_keys=True).encode(),
        )
        for index in range(count)
    )


def store_at(
    root: Path,
    *,
    initial: tuple[BootstrapFragment, ...] | None = None,
    frozen_identities: GlobalStateIdentities | None = None,
    s_max: int = 0,
    counting: bool = False,
):
    initial = initial or fragments()
    backend = PosixStorageBackend(root)
    wrapped = CountingStorageBackend(backend) if counting else backend
    store = GlobalStateStore(
        wrapped,
        identities=frozen_identities or identities(),
        descriptors=tuple(item.descriptor for item in initial),
        s_max=s_max,
    )
    return store, wrapped


def test_bootstrap_publishes_one_compound_current_record_per_fragment(
    tmp_path: Path,
) -> None:
    initial = fragments()
    store, _ = store_at(tmp_path, initial=initial)
    result = store.bootstrap(initial)
    assert result.published_indices == (0, 1, 2, 3)
    assert result.existing_indices == ()
    assert result.snapshot.version_vector == (0, 0, 0, 0)
    assert tuple(state.parameters for state in result.snapshot.states) == tuple(
        item.parameters for item in initial
    )
    assert tuple(state.outer_state for state in result.snapshot.states) == tuple(
        item.outer_state for item in initial
    )
    assert all(
        state.outer_update_count == 0 and len(state.base_history) == 1
        for state in result.snapshot.states
    )
    assert sorted(path.name for path in (tmp_path / "visibility").iterdir()) == [
        f"global-current-{index:06d}.json" for index in range(4)
    ]
    assert not (tmp_path / "visibility" / "global-head.json").exists()


def test_partial_bootstrap_fails_as_a_snapshot_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    initial = fragments()
    store, _ = store_at(tmp_path, initial=initial)
    with pytest.raises(BootstrapInterrupted, match="after 2"):
        store.bootstrap(initial, interrupt_after_fragments=2)
    with pytest.raises(PublicationNotFound):
        store.load_snapshot()
    resumed = store.bootstrap(initial)
    assert resumed.existing_indices == (0, 1)
    assert resumed.published_indices == (2, 3)
    payloads = {path.name for path in (tmp_path / "payloads").iterdir()}
    repeated = store.bootstrap(initial)
    assert repeated.existing_indices == (0, 1, 2, 3)
    assert repeated.published_indices == ()
    assert {path.name for path in (tmp_path / "payloads").iterdir()} == payloads


def test_conflicting_bootstrap_rejects_before_any_overwrite_or_new_payload(
    tmp_path: Path,
) -> None:
    initial = fragments()
    store, _ = store_at(tmp_path, initial=initial)
    original = store.bootstrap(initial).snapshot
    payloads = {path.name for path in (tmp_path / "payloads").iterdir()}
    changed = list(initial)
    changed[2] = dataclasses.replace(changed[2], parameters=b"wrong" * 8)
    with pytest.raises(GlobalStateError, match="conflicting bootstrap"):
        store.bootstrap(tuple(changed))
    assert store.load_snapshot() == original
    assert {path.name for path in (tmp_path / "payloads").iterdir()} == payloads


def test_restart_reads_exactly_f_records_independent_of_10000_history_objects(
    tmp_path: Path, monkeypatch
) -> None:
    initial = fragments()
    store, counting = store_at(tmp_path, initial=initial, counting=True)
    store.bootstrap(initial)
    counting.reset_counts()
    assert store.load_snapshot().version_vector == (0, 0, 0, 0)
    baseline = counting.read_calls
    assert baseline == len(initial)
    for index in range(10_000):
        (tmp_path / "payloads" / f"historical-{index:05d}.bin").write_bytes(b"old")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal startup must not scan a directory")

    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    counting.reset_counts()
    assert store.load_snapshot().version_vector == (0, 0, 0, 0)
    assert counting.read_calls == baseline


def test_fragment_load_is_independent_and_offline_live_set_is_bounded(
    tmp_path: Path,
) -> None:
    initial = fragments()
    store, counting = store_at(tmp_path, initial=initial, counting=True)
    store.bootstrap(initial)
    counting.reset_counts()
    assert store.load_fragment(3).parameters == initial[3].parameters
    assert counting.read_calls == 1
    report = store.inspect_live_set()
    assert report.current_records == report.referenced_payloads == 4
    assert report.retained_base_entries == report.maximum_base_entries == 4
    assert report.physical_payloads == 4
    assert report.orphan_payload_candidates == 0
    assert report.shared_global_head_present is False


def test_successor_derives_exact_plus_one_and_rolls_bounded_base_window(
    tmp_path: Path,
) -> None:
    initial = fragments(2)
    store, _ = store_at(tmp_path, initial=initial, s_max=1)
    initial_content_identity = (
        store.bootstrap(initial).snapshot.states[0].content_identity
    )
    first = store.publish_successor(0, parameters=b"v1", outer_state=b"outer-1")
    second = store.publish_successor(0, parameters=b"v2", outer_state=b"outer-2")
    assert first.version == first.outer_update_count == 1
    assert first.base_content_identity == initial_content_identity
    assert second.base_content_identity == first.content_identity
    assert tuple(item.version for item in first.base_history) == (0, 1)
    assert second.version == second.outer_update_count == 2
    assert tuple(item.version for item in second.base_history) == (1, 2)
    assert tuple(item.parameters for item in second.base_history) == (b"v1", b"v2")
    assert store.load_fragment(1).version == 0


def test_wrong_frozen_identity_descriptor_and_compound_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    initial = fragments(2)
    store, _ = store_at(tmp_path, initial=initial)
    store.bootstrap(initial)
    wrong_config, _ = store_at(
        tmp_path,
        initial=initial,
        frozen_identities=identities(config_identity="e" * 64),
    )
    with pytest.raises(GlobalStateError, match="frozen identity"):
        wrong_config.load_fragment(0)
    changed_descriptor = dataclasses.replace(initial[0].descriptor, shape=(999,))
    wrong_shape = GlobalStateStore(
        PosixStorageBackend(tmp_path),
        identities=identities(),
        descriptors=(changed_descriptor, initial[1].descriptor),
        s_max=0,
    )
    with pytest.raises(PublicationError, match="shape"):
        wrong_shape.load_fragment(0)
    visibility_path = tmp_path / "visibility" / "global-current-000000.json"
    original_visibility = visibility_path.read_bytes()
    visibility = json.loads(original_visibility)
    visibility["base_content_identity"] = "f" * 64
    visibility_path.write_text(json.dumps(visibility), encoding="utf-8")
    with pytest.raises(GlobalStateError, match="base identity"):
        store.load_fragment(0)
    visibility_path.write_bytes(original_visibility)
    payload_path = tmp_path / visibility["payload_relative_path"]
    payload = bytearray(payload_path.read_bytes())
    payload[-1] ^= 1
    payload_path.write_bytes(payload)
    with pytest.raises(PublicationError, match="complete and readable"):
        store.load_fragment(0)


def test_binary_compound_codec_rejects_trailing_bytes_and_is_frozen(
    tmp_path: Path,
) -> None:
    initial = fragments(1)
    store, _ = store_at(tmp_path, initial=initial)
    state = store.bootstrap(initial).snapshot.states[0]
    assert decode_global_state(encode_global_state(state)) == state
    with pytest.raises(GlobalStateError, match="trailing"):
        decode_global_state(encode_global_state(state) + b"junk")
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.version = 2  # type: ignore[misc]


def test_real_bootstrap_cli_verifies_plan_inputs_and_is_idempotent(
    tmp_path: Path, capsys
) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    plan_fragments = []
    for item in fragments(2):
        parameter_path = input_root / f"parameters-{item.descriptor.index}.bin"
        outer_path = input_root / f"outer-{item.descriptor.index}.bin"
        parameter_path.write_bytes(item.parameters)
        outer_path.write_bytes(item.outer_state)
        plan_fragments.append(
            {
                **item.descriptor.to_dict(),
                "parameters_path": str(parameter_path.relative_to(tmp_path)),
                "parameters_sha256": hashlib.sha256(item.parameters).hexdigest(),
                "outer_state_path": str(outer_path.relative_to(tmp_path)),
                "outer_state_sha256": hashlib.sha256(item.outer_state).hexdigest(),
            }
        )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identities": identities().to_dict(),
                "s_max": 0,
                "fragments": plan_fragments,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    storage = tmp_path / "storage"
    assert main(["bootstrap", "--plan", str(plan), "--storage-root", str(storage)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["published_indices"] == [0, 1]
    assert main(["bootstrap", "--plan", str(plan), "--storage-root", str(storage)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["published_indices"] == []
    assert second["existing_indices"] == [0, 1]
    parameter_path.write_bytes(b"tampered")
    with pytest.raises(GlobalStateError, match="checksum"):
        main(["bootstrap", "--plan", str(plan), "--storage-root", str(storage)])


def test_tiny_real_hugging_face_model_round_trips_exact_fragment_tensors(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from fsbdd.fragment_map import build_fragment_map
    from fsbdd.model_registry import build_logical_layer_registry
    from fsbdd.model_state import freeze_hf_model_fragments

    torch.manual_seed(1704)
    config = transformers.GPTNeoXConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    model = transformers.GPTNeoXForCausalLM(config)
    registry = build_logical_layer_registry(model, sync_dtype_bytes=4)
    fragment_map = build_fragment_map(registry, 2)
    frozen = freeze_hf_model_fragments(model, registry, fragment_map)
    frozen_identities = GlobalStateIdentities(
        run_identity="tiny-neox-seed-1704",
        config_identity=frozen.config_identity,
        model_identity=frozen.model_identity,
        fragment_map_identity=fragment_map.digest,
    )
    store, _ = store_at(
        tmp_path,
        initial=frozen.fragments,
        frozen_identities=frozen_identities,
    )
    snapshot = store.bootstrap(frozen.fragments).snapshot
    loaded = {}
    for state in snapshot.states:
        loaded.update(safetensors_torch.load(state.parameters))
    records = {record.owner_name for record in registry.parameters}
    assert set(loaded) == records
    named = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    assert all(
        torch.equal(loaded[name], named[name].detach().cpu()) for name in records
    )


def test_pbs_contract_is_two_node_and_bootstrap_focused() -> None:
    from fsbdd.global_state_stress import _fragments, derive_stress_identities

    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "pbs" / "stage1_s1_04_global_state.pbs"
    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text(encoding="utf-8")
    assert "#PBS -l select=2" in content
    assert "--map-by ppr:1:node" in content
    assert "fsbdd.global_state_stress role" in content
    assert "HISTORY_OBJECTS=10000" in content
    assert "fsbdd.cli evidence finalize" in content
    assert "--config-identity" in content
    assert "--model-identity" in content
    assert "--fragment-map-identity" in content
    assert '--run-id "$RUN_ID"' in content
    assert derive_stress_identities(
        "dd1dc08de331759cc49de1959bc1f2d9ba305ff985b07c2efb9084746fda3c50",
        _fragments(4, 65536),
    ) == (
        "d91a92be805f856514a9f13ac049e5c91ba8489bc415e2bf12c4bbadc3082100",
        "d1fb096403b6c0043570e3784cbaffce8c625bc48a98a5fdb37e06ef1d03b2ab",
    )
    assert 'date -u -d "$QTIME_RAW JST"' in content
    assert "torchrun" not in content
    assert "nccl" not in content.lower()
