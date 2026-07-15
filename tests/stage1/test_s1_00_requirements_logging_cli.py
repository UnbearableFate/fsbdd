from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from fsbdd.cli import main
from fsbdd.diloco.common.logging import StructuredLogger
from fsbdd.auxiliary.contracts.requirements import (
    MatrixError,
    authority_ids,
    validate_matrix,
)


def test_requirement_ids_are_derived_from_authority(tmp_path: Path) -> None:
    card = tmp_path / "loop.md"
    card.write_text(
        "| 规范条款 | DISC-03, ORACLE-05, PROP-01, GLOBAL-06, SYNC-02 |\n"
        "| Acceptance | A-X-01 |\n",
        encoding="utf-8",
    )
    expected = authority_ids(card)
    assert expected == {
        "DISC-03",
        "ORACLE-05",
        "PROP-01",
        "GLOBAL-06",
        "SYNC-02",
        "A-X-01",
    }
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "id,assertion,counterexample,target_api,authoritative_observation,aggregation,evidence_path,close_condition\nDISC-03,a,c,t,o,g,e,x\nORACLE-05,a,c,t,o,g,e,x\nPROP-01,a,c,t,o,g,e,x\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="missing"):
        validate_matrix(card, matrix)


def test_structured_logging_and_dry_run_roles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = StructuredLogger(log_path, role="checker", run_id="r")
    logger.emit("ready", count=1)
    event = json.loads(log_path.read_text(encoding="utf-8"))
    assert event["timestamp_utc"].endswith("Z")
    assert isinstance(event["monotonic_ns"], int)
    for role in ("bootstrap", "learner", "syncer", "checker"):
        assert main([role, "--dry-run"]) == 0
        assert json.loads(capsys.readouterr().out)["role"] == role

    with pytest.raises(ValueError, match="reserved"):
        logger.emit("spoof", role="syncer", timestamp_utc="invalid")


def test_structured_logger_serializes_concurrent_role_threads(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    logger = StructuredLogger(path, role="learner", run_id="concurrent")

    def emit(worker: int) -> None:
        for sequence in range(25):
            logger.emit("thread_event", worker=worker, sequence=sequence)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(emit, range(4)))
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 100
    assert {(item["worker"], item["sequence"]) for item in records} == {
        (worker, sequence) for worker in range(4) for sequence in range(25)
    }


def test_structured_logger_batches_flush_with_fsync_interval(tmp_path: Path) -> None:
    path = tmp_path / "batched.jsonl"
    logger = StructuredLogger(
        path,
        role="learner",
        run_id="batched",
        fsync_every=2,
    )
    logger.emit("first")
    assert path.read_text(encoding="utf-8") == ""
    logger.emit("second")
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == [
        "first",
        "second",
    ]
    logger.close()


def test_evidence_init_cli_is_fail_if_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "package"
    assert main(["evidence", "init", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "building"
    with pytest.raises(Exception, match="exists"):
        main(["evidence", "init", str(root)])
