from __future__ import annotations

from pathlib import Path

from fsbdd.auxiliary.stage1.capacity import _run_numpy_update_trial


def test_capacity_preflight_measures_real_numpy_merge_and_atomic_commit(
    tmp_path: Path,
) -> None:
    result = _run_numpy_update_trial(
        root=tmp_path / "global",
        run_id="s1-13-capacity-unit",
        parameter_bytes=256,
        target_global_payload_bytes=4096,
        parameter_identity_count=2,
        repetition=0,
    )

    assert result["fragment_parameter_bytes"] == 256
    assert result["merge_seconds"] > 0
    assert result["commit_seconds"] > 0
    assert result["merge_commit_seconds"] == (
        result["merge_seconds"] + result["commit_seconds"]
    )
    assert result["successor_version"] == 1
    assert result["committed_global_payload_bytes"] > 256
    assert result["verification_mode"] == "source_sha256_complete_write"
